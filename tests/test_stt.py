"""Tests for the production Whisper and experimental Parakeet STT backends.

All real-engine behavior is mocked: tests never download weights or require a
GPU. They verify one-time loading, off-loop blocking work, confidence mapping,
trace propagation, safe stub fallback, and actionable CUDA OOM handling.
"""
from __future__ import annotations

import asyncio
import io
import logging
import math
import threading
import wave
from typing import Any

import pytest

from core.config_loader import ModuleConfig
from core.event_bus import Event, EventBus
from core.gpu_lock import GPULock
import modules.stt as stt_mod
from modules.stt import STTModule


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def gpu_lock() -> GPULock:
    return GPULock()


def _config(device: str = "cpu") -> ModuleConfig:
    return ModuleConfig(
        device=device,
        model="base",
        params={
            "language": "ru",
            "download_root": "models/openai-whisper",
            "fp16": device == "cuda",
            "initial_prompt": "Калькулятор, Пейнт, Дискорд.",
            "temperature": 0.0,
            "beam_size": 3,
            "patience": 1.0,
            "condition_on_previous_text": False,
        },
    )


class _FakeModel:
    def __init__(self) -> None:
        self.transcribe_calls: list[dict[str, Any]] = []
        self.transcribe_thread_ids: list[int] = []

    def transcribe(self, audio, **kwargs):
        self.transcribe_thread_ids.append(threading.get_ident())
        self.transcribe_calls.append({"audio": audio, **kwargs})
        return {
            "text": " hello world ",
            "segments": [
                {"avg_logprob": -0.1, "no_speech_prob": 0.1},
                {"avg_logprob": -0.2, "no_speech_prob": 0.0},
            ],
        }


class _FakeWhisperPackage:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.load_calls: list[dict[str, Any]] = []
        self.load_thread_ids: list[int] = []

    def load_model(self, name, *, device, download_root):
        self.load_thread_ids.append(threading.get_ident())
        self.load_calls.append(
            {"name": name, "device": device, "download_root": download_root}
        )
        return self.model


@pytest.fixture
def fake_whisper(monkeypatch):
    package = _FakeWhisperPackage()
    monkeypatch.setattr(stt_mod, "_WHISPER", package)
    return package


async def _run_audio(mod: STTModule, bus: EventBus, trace_id: str = "stt-tr"):
    output: list[Event] = []
    ready = asyncio.Event()

    async def record(event: Event) -> None:
        output.append(event)
        ready.set()

    bus.subscribe("transcription_ready", record)
    run_task = asyncio.create_task(bus.run())
    bus.publish("audio_captured", {"audio": b"x"}, trace_id=trace_id)
    await asyncio.wait_for(ready.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()
    return output


async def test_missing_package_fails_closed_with_trace_and_confidence(
    bus, gpu_lock, monkeypatch, caplog
):
    monkeypatch.setattr(stt_mod, "_WHISPER", None)
    mod = STTModule(_config(), gpu_lock)
    with caplog.at_level(logging.WARNING, logger="jarvis.module.stt"):
        await mod.start(bus)
    output = await _run_audio(mod, bus)

    assert output[0].trace_id == "stt-tr"
    assert output[0].payload == {
        "text": "",
        "confidence": 0.0,
    }
    assert "openai-whisper not installed" in caplog.text


async def test_gpu_lock_is_acquired_on_stub_path(bus, gpu_lock, monkeypatch, caplog):
    monkeypatch.setattr(stt_mod, "_WHISPER", None)
    mod = STTModule(_config(), gpu_lock)
    await mod.start(bus)
    with caplog.at_level(logging.INFO, logger="jarvis.gpu"):
        await _run_audio(mod, bus)
    assert "GPU_ACQUIRE label=stt" in caplog.text
    assert "GPU_RELEASE label=stt" in caplog.text


async def test_model_loads_once_off_event_loop(bus, gpu_lock, fake_whisper):
    event_loop_thread = threading.get_ident()
    mod = STTModule(_config(), gpu_lock)
    await mod.start(bus)
    assert mod._model is fake_whisper.model
    assert fake_whisper.load_calls == [
        {
            "name": "base",
            "device": "cpu",
            "download_root": "models/openai-whisper",
        }
    ]
    assert all(tid != event_loop_thread for tid in fake_whisper.load_thread_ids)
    await mod.stop()


async def test_decodable_audio_uses_real_result_and_runs_off_loop(
    bus, gpu_lock, fake_whisper, monkeypatch
):
    event_loop_thread = threading.get_ident()
    mod = STTModule(_config(), gpu_lock)
    fake_audio = object()
    monkeypatch.setattr(mod, "_decode_audio", lambda payload: fake_audio)
    await mod.start(bus)
    output = await _run_audio(mod, bus, trace_id="real-tr")

    expected = (math.exp(-0.1) * 0.9 + math.exp(-0.2)) / 2
    assert output[0].trace_id == "real-tr"
    assert output[0].payload["text"] == "hello world"
    assert output[0].payload["confidence"] == pytest.approx(expected)
    call = fake_whisper.model.transcribe_calls[0]
    assert call["audio"] is fake_audio
    assert call["language"] == "ru"
    assert call["task"] == "transcribe"
    assert call["fp16"] is False
    assert call["verbose"] is None
    assert call["initial_prompt"] == "Калькулятор, Пейнт, Дискорд."
    assert call["temperature"] == 0.0
    assert call["beam_size"] == 3
    assert call["patience"] == 1.0
    assert call["condition_on_previous_text"] is False
    assert all(
        tid != event_loop_thread for tid in fake_whisper.model.transcribe_thread_ids
    )


async def test_non_decodable_audio_does_not_call_real_model(
    bus, gpu_lock, fake_whisper, caplog
):
    mod = STTModule(_config(), gpu_lock)
    await mod.start(bus)
    with caplog.at_level(logging.WARNING, logger="jarvis.module.stt"):
        output = await _run_audio(mod, bus)
    assert output[0].payload["text"] == ""
    assert not fake_whisper.model.transcribe_calls
    assert "non-decodable audio payload" in caplog.text


async def test_cuda_oom_is_reraised_with_actionable_log(
    bus, gpu_lock, fake_whisper, monkeypatch, caplog
):
    def raise_oom(audio, **kwargs):
        raise RuntimeError("CUDA error: out of memory")

    fake_whisper.model.transcribe = raise_oom
    mod = STTModule(_config(device="cuda"), gpu_lock)
    monkeypatch.setattr(mod, "_decode_audio", lambda payload: object())
    await mod.start(bus)
    with caplog.at_level(logging.ERROR, logger="jarvis.module.stt"):
        with pytest.raises(RuntimeError, match="out of memory"):
            await mod._transcribe({"audio": b"x"})
    assert "CUDA out of memory while running OpenAI Whisper" in caplog.text


def test_project_has_no_faster_whisper_dependency():
    requirements = (
        stt_mod.__file__ and __import__("pathlib").Path("requirements.txt").read_text(
            encoding="utf-8"
        )
    )
    assert "faster-whisper" not in requirements.lower()
    assert "huggingface" not in requirements.lower()


def test_beam_search_can_be_disabled(gpu_lock):
    config = _config()
    config.params["beam_size"] = 0

    mod = STTModule(config, gpu_lock)

    assert mod._beam_size is None


@pytest.mark.parametrize(
    ("cuda_available", "expected_device", "expected_fp16"),
    [(True, "cuda", True), (False, "cpu", False)],
)
def test_auto_device_selects_cuda_or_cpu(
    gpu_lock, monkeypatch, cuda_available, expected_device, expected_fp16
):
    monkeypatch.setattr(stt_mod, "_cuda_is_available", lambda: cuda_available)
    config = _config(device="auto")
    config.params.pop("fp16")

    mod = STTModule(config, gpu_lock)

    assert mod._device == expected_device
    assert mod._fp16 is expected_fp16


def _parakeet_config(*, approved: bool = True) -> ModuleConfig:
    return ModuleConfig(
        device="cpu",
        model="base",
        params={
            "engine": "parakeet",
            "experimental_production": approved,
            "parakeet_model_dir": ".local/test-model",
            "parakeet_python": "venv/Scripts/python.exe",
            "parakeet_timeout_seconds": 2,
        },
    )


def test_raw_16khz_pcm_is_wrapped_for_parakeet():
    pcm = b"\x01\x00" * 160
    payload = stt_mod._payload_to_pcm_wav(
        {
            "audio": pcm,
            "sample_rate": 16_000,
            "channels": 1,
            "sample_width": 2,
        }
    )

    assert payload is not None
    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        assert wav_file.getparams()[:3] == (1, 2, 16_000)
        assert wav_file.readframes(160) == pcm


@pytest.mark.parametrize(
    "payload",
    (
        {"audio": b"\x00\x00", "sample_rate": 8_000},
        {"audio": b"\x00\x00", "sample_rate": 16_000, "channels": 2},
        {"audio": b"\x00", "sample_rate": 16_000},
        {"audio": b"<stub-pcm-chunks>", "sample_rate": 16_000},
    ),
)
def test_parakeet_rejects_non_production_audio_shapes(payload):
    assert stt_mod._payload_to_pcm_wav(payload) is None


async def test_parakeet_requires_explicit_production_gate(bus, gpu_lock):
    mod = STTModule(_parakeet_config(approved=False), gpu_lock)

    with pytest.raises(RuntimeError, match="experimental_production"):
        await mod.start(bus)


async def test_parakeet_worker_is_warm_reused_and_closed_off_loop(
    bus, gpu_lock, monkeypatch
):
    event_loop_thread = threading.get_ident()
    instances = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.thread_ids = []
            self.closed = False
            instances.append(self)

        def start(self):
            self.thread_ids.append(threading.get_ident())
            return {
                "event": "ready",
                "model_load_ms": 10,
                "warm_up_ms": 2,
                "health": {"model_id": "parakeet", "model_revision": "rev"},
            }

        def decode(self, wav_bytes):
            self.thread_ids.append(threading.get_ident())
            assert wav_bytes.startswith(b"RIFF")
            return {"event": "result", "text": " открой браузер ", "decode_ms": 4}

        def close(self):
            self.thread_ids.append(threading.get_ident())
            self.closed = True

    monkeypatch.setattr(stt_mod, "PersistentParakeetClient", FakeClient)
    mod = STTModule(_parakeet_config(), gpu_lock)
    await mod.start(bus)
    pcm_payload = {
        "audio": b"\x00\x00" * 160,
        "sample_rate": 16_000,
        "channels": 1,
        "sample_width": 2,
    }

    assert await mod._transcribe(pcm_payload) == ("открой браузер", 0.0)
    assert await mod._transcribe(pcm_payload) == ("открой браузер", 0.0)
    await mod.stop()

    assert len(instances) == 1
    assert instances[0].kwargs["provider"] == "cpu"
    assert instances[0].closed is True
    assert all(thread_id != event_loop_thread for thread_id in instances[0].thread_ids)
