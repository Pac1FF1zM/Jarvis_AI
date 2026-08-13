"""Production speech-to-text via the isolated local Parakeet TDT worker.

The module subscribes to ``audio_captured``, normalizes Jarvis microphone PCM
to a strict 16 kHz mono WAV, decodes it under the shared GPU lock, and publishes
``transcription_ready`` on the original trace. The pinned model is loaded and
warmed once in a child process, then reused for the whole Jarvis session.

Failures are fail-closed: malformed audio, a worker error, or a timeout can
never fabricate a command. The Transformers adapter does not expose calibrated
utterance confidence, so the payload carries ``0.0`` while command acceptance
continues to use the independent NLU intent confidence.

Manual real-file test::

    python -m modules.stt --test --wav path/to/16khz-mono.wav
"""
from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
import wave
from typing import Any

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import TranscriptionReadyPayload
from core.gpu_lock import GPULock
from modules.parakeet_client import DEFAULT_MODEL_DIR, PersistentParakeetClient

logger = logging.getLogger("jarvis.module.stt")

STUB_TEXT = "сколько сейчас времени"
STUB_CONFIDENCE = 0.0
_STUB_AUDIO_MARKERS = (b"<stub-pcm-chunks>", b"<fake>", b"<fake-pcm-chunks>")
_SAMPLE_RATE = 16_000


def _cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, OSError):
        return False


def _resolve_device(requested: Any) -> str:
    device = str(requested or "cpu").strip().casefold()
    if device == "auto":
        return "cuda" if _cuda_is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("STT device must be 'auto', 'cuda', or 'cpu'")
    return device


class STTModule(BaseModule):
    """Transcribe captured audio with one persistent Parakeet worker."""

    name = "stt"
    enabled = True

    def __init__(self, config: Any, gpu_lock: GPULock) -> None:
        super().__init__(config)
        self.gpu_lock = gpu_lock
        params = getattr(config, "params", {}) or {}
        self._device = _resolve_device(config.device)
        self._model_dir = str(params.get("model_dir", DEFAULT_MODEL_DIR))
        self._python = str(params.get("python", "venv/Scripts/python.exe"))
        self._timeout = float(params.get("timeout_seconds", 45.0))
        if self._timeout <= 0:
            raise ValueError("stt.params.timeout_seconds must be > 0")
        self._client: PersistentParakeetClient | None = None

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("audio_captured", self._on_audio)
        self._client = PersistentParakeetClient(
            repository_root=Path.cwd(),
            model_dir=self._model_dir,
            python_executable=self._python,
            provider=self._device,
            timeout_seconds=self._timeout,
        )
        startup = await asyncio.to_thread(self._client.start)
        health = startup.get("health", {})
        logger.info(
            "STT_ACTIVE engine=parakeet provider=%s model=%s revision=%s "
            "load_ms=%.2f warm_up_ms=%.2f",
            self._device,
            health.get("model_id", "unknown"),
            health.get("model_revision", "unknown"),
            float(startup.get("model_load_ms") or 0.0),
            float(startup.get("warm_up_ms") or 0.0),
        )

    async def stop(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await asyncio.to_thread(client.close)
        logger.info("STTModule stopped")

    async def _on_audio(self, event: Event) -> None:
        assert self.bus is not None
        if self.bus.is_trace_cancelled(event.trace_id):
            return
        try:
            text, confidence = await self._transcribe(event.payload)
        except Exception as exc:
            logger.error("STT failed trace=%s: %s", event.trace_id, exc)
            text, confidence = "", 0.0
        if self.bus.is_trace_cancelled(event.trace_id):
            return
        output = event.child(
            "transcription_ready",
            TranscriptionReadyPayload(
                text=text,
                confidence=confidence,
                source="parakeet",
            ),
        )
        self.bus.publish_event(output)
        logger.info(
            "TRANSCRIPTION_READY trace=%s engine=parakeet text=%r conf=%.2f",
            output.trace_id,
            text,
            confidence,
        )

    async def _transcribe(self, audio_payload: dict[str, Any]) -> tuple[str, float]:
        client = self._client
        if client is None:
            return "", 0.0
        wav_bytes = _payload_to_pcm_wav(audio_payload)
        if wav_bytes is None:
            logger.warning("non-decodable audio payload — returning empty transcription")
            return "", 0.0
        async with self.gpu_lock.section("stt"):
            result = await asyncio.to_thread(client.decode, wav_bytes)
        text = str(result.get("text", "")).strip()
        logger.info(
            "PARAKEET_DECODE decode_ms=%.2f text_chars=%d",
            float(result.get("decode_ms") or 0.0),
            len(text),
        )
        return text, 0.0


def _payload_to_pcm_wav(audio_payload: dict[str, Any]) -> bytes | None:
    """Normalize Jarvis raw PCM (or an already valid WAV) for Parakeet."""
    raw = audio_payload.get("audio")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    raw_bytes = bytes(raw)
    if not raw_bytes or raw_bytes in _STUB_AUDIO_MARKERS:
        return None
    sample_rate = int(audio_payload.get("sample_rate", _SAMPLE_RATE))
    channels = int(audio_payload.get("channels", 1))
    sample_width = int(audio_payload.get("sample_width", 2))
    if raw_bytes.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
                valid = (
                    wav_file.getframerate() == _SAMPLE_RATE
                    and wav_file.getnchannels() == 1
                    and wav_file.getsampwidth() == 2
                    and wav_file.getnframes() > 0
                )
            return raw_bytes if valid else None
        except (wave.Error, EOFError):
            return None
    if (
        sample_rate != _SAMPLE_RATE
        or channels != 1
        or sample_width != 2
        or len(raw_bytes) % 2
    ):
        return None
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SAMPLE_RATE)
        wav_file.writeframes(raw_bytes)
    return output.getvalue()


async def _standalone_test(wav_path: str) -> None:
    from core.config_loader import ModuleConfig

    module = STTModule(
        ModuleConfig(
            device="auto",
            params={
                "model_dir": str(DEFAULT_MODEL_DIR),
                "python": "venv/Scripts/python.exe",
                "timeout_seconds": 45.0,
            },
        ),
        GPULock(),
    )
    bus = EventBus()
    results: list[Event] = []

    async def record(event: Event) -> None:
        results.append(event)

    bus.subscribe("transcription_ready", record)
    await module.start(bus)
    run_task = asyncio.create_task(bus.run())
    try:
        audio = Path(wav_path).read_bytes()
        bus.publish(
            "audio_captured",
            {"audio": audio, "sample_rate": _SAMPLE_RATE},
            trace_id="stt-wav",
        )
        for _ in range(450):
            if results:
                break
            await asyncio.sleep(0.1)
    finally:
        await bus.stop()
        await run_task
        await module.stop()
    if not results:
        raise TimeoutError("no transcription_ready emitted")
    print(dict(results[0].payload))


if __name__ == "__main__":
    import sys

    if "--test" not in sys.argv or "--wav" not in sys.argv:
        print("usage: python -m modules.stt --test --wav path/to/file.wav")
        raise SystemExit(2)
    index = sys.argv.index("--wav")
    if index + 1 >= len(sys.argv):
        raise SystemExit("--wav requires a path")
    asyncio.run(_standalone_test(sys.argv[index + 1]))
