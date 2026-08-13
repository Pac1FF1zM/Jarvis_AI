"""Tests for the production Parakeet STT event and worker lifecycle."""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
import threading
import wave

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


def _config(device: str = "cpu", *, timeout: float = 2.0) -> ModuleConfig:
    return ModuleConfig(
        device=device,
        params={
            "model_dir": ".local/test-model",
            "python": "venv/Scripts/python.exe",
            "timeout_seconds": timeout,
        },
    )


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.thread_ids: list[int] = []
        self.decode_calls: list[bytes] = []
        self.closed = False
        self.__class__.instances.append(self)

    def start(self):
        self.thread_ids.append(threading.get_ident())
        return {
            "event": "ready",
            "model_load_ms": 10.0,
            "warm_up_ms": 2.0,
            "health": {
                "model_id": "nvidia/parakeet-tdt-0.6b-v3",
                "model_revision": "pinned-revision",
            },
        }

    def decode(self, wav_bytes: bytes):
        self.thread_ids.append(threading.get_ident())
        self.decode_calls.append(wav_bytes)
        return {
            "event": "result",
            "text": " открой браузер ",
            "decode_ms": 4.0,
        }

    def close(self):
        self.thread_ids.append(threading.get_ident())
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.instances.clear()
    monkeypatch.setattr(stt_mod, "PersistentParakeetClient", _FakeClient)
    return _FakeClient


async def _run_audio(
    module: STTModule,
    bus: EventBus,
    payload: dict | None = None,
    *,
    trace_id: str = "stt-trace",
) -> list[Event]:
    output: list[Event] = []
    ready = asyncio.Event()

    async def record(event: Event) -> None:
        output.append(event)
        ready.set()

    bus.subscribe("transcription_ready", record)
    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "audio_captured",
        payload
        or {
            "audio": b"\x00\x00" * 160,
            "sample_rate": 16_000,
            "channels": 1,
            "sample_width": 2,
        },
        trace_id=trace_id,
    )
    await asyncio.wait_for(ready.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    return output


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


async def test_worker_loads_once_and_closes_off_loop(bus, gpu_lock, fake_client):
    event_loop_thread = threading.get_ident()
    module = STTModule(_config(), gpu_lock)
    await module.start(bus)
    assert len(fake_client.instances) == 1
    client = fake_client.instances[0]
    assert client.kwargs["provider"] == "cpu"
    assert client.kwargs["model_dir"] == ".local/test-model"

    assert await module._transcribe(
        {"audio": b"\x00\x00" * 160, "sample_rate": 16_000}
    ) == ("открой браузер", 0.0)
    assert await module._transcribe(
        {"audio": b"\x00\x00" * 160, "sample_rate": 16_000}
    ) == ("открой браузер", 0.0)
    await module.stop()

    assert len(client.decode_calls) == 2
    assert client.closed is True
    assert all(thread_id != event_loop_thread for thread_id in client.thread_ids)


async def test_event_path_preserves_trace_source_and_gpu_lock(
    bus, gpu_lock, fake_client, caplog
):
    module = STTModule(_config(), gpu_lock)
    await module.start(bus)
    with caplog.at_level(logging.INFO):
        output = await _run_audio(module, bus, trace_id="real-parakeet")
    await module.stop()

    assert output[0].trace_id == "real-parakeet"
    assert dict(output[0].payload) == {
        "text": "открой браузер",
        "confidence": 0.0,
        "source": "parakeet",
    }
    assert "GPU_ACQUIRE label=stt" in caplog.text
    assert "GPU_RELEASE label=stt" in caplog.text


async def test_invalid_audio_fails_closed_without_calling_worker(
    bus, gpu_lock, fake_client, caplog
):
    module = STTModule(_config(), gpu_lock)
    await module.start(bus)
    with caplog.at_level(logging.WARNING, logger="jarvis.module.stt"):
        output = await _run_audio(
            module,
            bus,
            {"audio": b"<stub-pcm-chunks>", "sample_rate": 16_000},
        )
    await module.stop()

    assert output[0].payload["text"] == ""
    assert output[0].payload["source"] == "parakeet"
    assert not fake_client.instances[0].decode_calls
    assert "non-decodable audio payload" in caplog.text


async def test_worker_error_fails_closed_at_event_boundary(
    bus, gpu_lock, fake_client, caplog
):
    module = STTModule(_config(), gpu_lock)
    await module.start(bus)

    def fail(_wav_bytes):
        raise TimeoutError("worker deadline")

    fake_client.instances[0].decode = fail
    with caplog.at_level(logging.ERROR, logger="jarvis.module.stt"):
        output = await _run_audio(module, bus)
    await module.stop()

    assert output[0].payload["text"] == ""
    assert "worker deadline" in caplog.text


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    ((True, "cuda"), (False, "cpu")),
)
def test_auto_device_selects_cuda_or_cpu(
    gpu_lock, monkeypatch, cuda_available, expected_device
):
    monkeypatch.setattr(stt_mod, "_cuda_is_available", lambda: cuda_available)
    module = STTModule(_config(device="auto"), gpu_lock)
    assert module._device == expected_device


def test_non_positive_timeout_is_rejected(gpu_lock):
    with pytest.raises(ValueError, match="timeout_seconds"):
        STTModule(_config(timeout=0), gpu_lock)


def test_production_runtime_has_no_whisper_dependency_or_code_path():
    requirements = Path("requirements.txt").read_text(encoding="utf-8").casefold()
    source = Path(stt_mod.__file__).read_text(encoding="utf-8").casefold()

    assert "openai-whisper" not in requirements
    assert "import whisper" not in source
    assert "whisper" not in source
