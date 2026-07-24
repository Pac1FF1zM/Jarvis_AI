"""Tests for the official OpenAI Whisper STT backend.

All real-engine behavior is mocked: tests never download weights or require a
GPU. They verify one-time loading, off-loop blocking work, confidence mapping,
trace propagation, safe stub fallback, and actionable CUDA OOM handling.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
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

    async def record(event: Event) -> None:
        output.append(event)

    bus.subscribe("transcription_ready", record)
    run_task = asyncio.create_task(bus.run())
    bus.publish("audio_captured", {"audio": b"x"}, trace_id=trace_id)
    await asyncio.sleep(0.2)
    await bus.stop()
    await run_task
    await mod.stop()
    return output


async def test_missing_package_falls_back_with_trace_and_confidence(
    bus, gpu_lock, monkeypatch, caplog
):
    monkeypatch.setattr(stt_mod, "_WHISPER", None)
    mod = STTModule(_config(), gpu_lock)
    with caplog.at_level(logging.WARNING, logger="jarvis.module.stt"):
        await mod.start(bus)
    output = await _run_audio(mod, bus)

    assert output[0].trace_id == "stt-tr"
    assert output[0].payload == {
        "text": stt_mod.STUB_TEXT,
        "confidence": stt_mod.STUB_CONFIDENCE,
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
    assert output[0].payload["text"] == stt_mod.STUB_TEXT
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
