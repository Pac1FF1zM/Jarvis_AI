"""STT module — speech-to-text via OpenAI's official Whisper package.

Subscribes to ``audio_captured``, runs transcription under the shared GPU lock,
and publishes ``transcription_ready`` as a child event so the trace_id carries
through to the LLM.

OpenAI Whisper is wired behind a lazy, guarded import so the project stays
fully runnable without it installed:

- If ``whisper`` is importable AND ``self.config.enabled`` is true, the
  model is loaded **once** in :meth:`STTModule.start` (loading is expensive —
  never per-call) and used for every transcribe.
- If ``whisper`` is not importable, the module logs an actionable
  message and stays in stub-only mode: it still handles events, still emits
  ``transcription_ready``, just with stub text.
- If the incoming ``audio_captured`` payload is not decodable audio (e.g. the
  literal stub marker bytes that ``modules/wake_word.py`` still publishes while
  no real microphone exists), the module logs a clear warning and returns the
  existing stub text/confidence so the demo round-trip keeps working.

CUDA OOM is detected and re-raised with an actionable message. An explicit
``cpu`` or ``cuda`` setting is respected; ``auto`` selects CUDA when PyTorch
can use it and otherwise selects CPU.

Model names are resolved by the official package from OpenAI's Azure storage;
an explicit local checkpoint path is also accepted. No Hugging Face client,
model, cache, or endpoint is used by this module.

Manual real-file testing (not part of default pytest)::

    python -m modules.stt --test --wav path/to/file.wav

prints the real transcription. Without ``--wav`` the standalone test runs the
stub/demo path.

``device`` and ``model`` are read from ``self.config``. ``params.language``
defaults to Russian and ``params.download_root`` controls the official model
cache directory.
"""
from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from core.base_module import BaseModule
from core.event_bus import EventBus, Event
from core.event_payloads import TranscriptionReadyPayload
from core.gpu_lock import GPULock

logger = logging.getLogger("jarvis.module.stt")

# Stub fallback values — kept identical to the original stub so the demo
# round-trip's observable output is unchanged when real audio isn't available.
# The voice demo is configured for Russian, so its synthetic transcript must
# look like a plausible Whisper result rather than contain a technical prefix
# that the NLU model can never hear from a microphone.
STUB_TEXT = "сколько сейчас времени"
STUB_CONFIDENCE = 0.92

# Literal sentinel that wake_word.py currently publishes; treated as
# "not real audio" so the demo path falls back gracefully. Anything that
# fails decode also falls back, so this is just the known-cheap fast path.
_STUB_AUDIO_MARKERS = (b"<stub-pcm-chunks>", b"<fake>", b"<fake-pcm-chunks>")

# Whisper expects 16 kHz mono float32.
_WHISPER_SAMPLE_RATE = 16000

# Lazily-populated module reference (None when openai-whisper isn't installed).
_WHISPER: Any = None
try:  # pragma: no cover - exercised only when the package is installed
    import whisper  # type: ignore  # noqa: F401
    _WHISPER = whisper
except ImportError:  # pragma: no cover
    _WHISPER = None


def _is_cuda_oom(exc: BaseException) -> bool:
    """Best-effort detection of a CUDA out-of-memory error."""
    # torch.cuda.OutOfMemoryError is the canonical type; some CUDA runtimes
    # raise a plain RuntimeError whose message contains "out of memory".
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _cuda_is_available() -> bool:
    """Return whether the installed PyTorch runtime can use CUDA."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except (ImportError, OSError):
        return False


def _resolve_device(requested: Any) -> str:
    """Resolve ``auto`` once while preserving explicit device choices."""
    device = str(requested or "cpu").strip().lower()
    if device == "auto":
        return "cuda" if _cuda_is_available() else "cpu"
    return device


class STTModule(BaseModule):
    """Transcribes captured audio into text."""

    name = "stt"
    enabled = True

    def __init__(self, config: Any, gpu_lock: GPULock) -> None:
        super().__init__(config)
        self.gpu_lock = gpu_lock
        # Real model handle, or None when running in stub-only mode.
        self._model: Any = None
        # numpy is only needed on the real path; import lazily so the module
        # stays importable when only stub mode is used.
        self._np: Any = None
        params = getattr(config, "params", {}) or {}
        self._language = str(params.get("language", "ru"))
        self._download_root = str(
            params.get("download_root", "models/openai-whisper")
        )
        self._device = _resolve_device(config.device)
        self._fp16 = bool(params.get("fp16", self._device == "cuda"))
        # Each microphone capture is one independent command.  Deterministic
        # beam search improves short Russian commands, while disabling text
        # carry-over prevents a previous command from biasing the next one.
        self._temperature = float(params.get("temperature", 0.0))
        beam_size = int(params.get("beam_size", 3))
        self._beam_size = beam_size if beam_size > 0 else None
        self._patience = float(params.get("patience", 1.0))
        self._condition_on_previous_text = bool(
            params.get("condition_on_previous_text", False)
        )
        self._initial_prompt = str(
            params.get(
                "initial_prompt",
                (
                    "Команды голосового ассистента Джарвис. Приложения: "
                    "калькулятор, блокнот, проводник, Пейнт, Дискорд, "
                    "диспетчер задач, браузер."
                ),
            )
        ).strip()

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("audio_captured", self._on_audio)

        if _WHISPER is None:
            logger.warning(
                "openai-whisper not installed — pip install openai-whisper; "
                "STT will run in stub-only mode"
            )
        elif self.config.enabled:
            # Load the model once. Loading is expensive; never do it per-call.
            try:
                self._model = await self._load_model()
            except Exception:
                # CUDA OOM at load time: actionable message + re-raise. Config
                # drives behavior; do not silently switch device.
                if _is_cuda_oom(_exc_cause()):
                    logger.error(
                        "CUDA out of memory while loading OpenAI Whisper model "
                        "'%s' (device=%s). Set STT device: cpu or choose a "
                        "smaller model in config.yaml.",
                        self.config.model,
                        self._device,
                    )
                raise

        logger.info(
            "STTModule started (mode=%s) device=%s model=%s language=%s "
            "temperature=%.1f beam_size=%s previous_text=%s",
            "real" if self._model is not None else "stub",
            self._device,
            self.config.model,
            self._language,
            self._temperature,
            self._beam_size,
            self._condition_on_previous_text,
        )

    async def stop(self) -> None:
        logger.info("STTModule stopped")

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    async def _load_model(self) -> Any:
        """Load one official Whisper model outside the event-loop thread."""
        assert _WHISPER is not None
        return await asyncio.to_thread(
            _WHISPER.load_model,
            self.config.model,
            device=self._device,
            download_root=self._download_root,
        )

    # ------------------------------------------------------------------ #
    # Event handler
    # ------------------------------------------------------------------ #
    async def _on_audio(self, event: Event) -> None:
        assert self.bus is not None
        text, confidence = await self._transcribe(event.payload)
        out = event.child(
            "transcription_ready",
            TranscriptionReadyPayload(text=text, confidence=confidence),
        )
        self.bus.publish_event(out)
        logger.info(
            "TRANSCRIPTION_READY trace=%s text=%r conf=%.2f",
            out.trace_id,
            text,
            confidence,
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    async def _transcribe(self, audio_payload: dict[str, Any]) -> tuple[str, float]:
        """Run transcription under the shared GPU lock.

        Decision flow:
          - If a real model is loaded AND the audio payload is decodable ->
            real OpenAI Whisper path (via a worker thread).
          - Otherwise -> graceful fallback to stub values with a clear warning.
        """
        if self._model is None:
            # Stub-only mode (OpenAI Whisper missing). Keep the lock acquired
            # for shape parity with the real path.
            async with self.gpu_lock.section("stt"):
                await asyncio.sleep(0.05)
            return STUB_TEXT, STUB_CONFIDENCE

        # Real model loaded — try to decode the payload. If it isn't real
        # audio (stub marker bytes, or decode raises), fall back to stub.
        audio_array = self._decode_audio(audio_payload)
        if audio_array is None:
            logger.warning(
                "non-decodable audio payload — falling back to stub transcription"
            )
            async with self.gpu_lock.section("stt"):
                await asyncio.sleep(0.05)
            return STUB_TEXT, STUB_CONFIDENCE

        return await self._transcribe_real(audio_array)

    def _decode_audio(self, audio_payload: dict[str, Any]) -> Any:
        """Decode payload bytes into a float32 numpy array at 16 kHz.

        Returns ``None`` if the payload is the known stub marker or cannot be
        decoded — the caller falls back to stub transcription in that case.
        """
        raw: Any = audio_payload.get("audio")
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)) and bytes(raw) in _STUB_AUDIO_MARKERS:
            return None
        try:
            np = self._get_numpy()
        except ImportError:
            logger.warning(
                "numpy not installed — cannot decode audio; falling back to stub"
            )
            return None

        sample_rate = int(audio_payload.get("sample_rate", _WHISPER_SAMPLE_RATE))

        # Preferred shape: already a float32 ndarray.
        if isinstance(raw, np.ndarray):  # type: ignore[union-attr]
            arr = raw.astype(np.float32)  # type: ignore[union-attr]
        else:
            # Try WAV bytes first (stdlib `wave`), then raw PCM int16 fallback.
            arr = _bytes_to_float32(np, bytes(raw), sample_rate)  # type: ignore[union-attr]
            if arr is None:
                return None

        # Resample to 16 kHz if needed (linear, naive — fine for this pass;
        # a real resampler like librosa/soxr is a future upgrade).
        if sample_rate != _WHISPER_SAMPLE_RATE:
            arr = _resample_linear(
                np, arr, sample_rate, _WHISPER_SAMPLE_RATE  # type: ignore[union-attr]
            )
        return arr

    def _get_numpy(self) -> Any:
        if self._np is None:
            import numpy as np  # type: ignore

            self._np = np
        return self._np

    async def _transcribe_real(self, audio_array: Any) -> tuple[str, float]:
        """Run official Whisper's blocking ``transcribe`` off the event loop."""

        def _sync_transcribe():
            options: dict[str, Any] = {
                "language": self._language,
                "task": "transcribe",
                "fp16": self._fp16,
                # In OpenAI Whisper ``False`` still renders a tqdm progress
                # bar. ``None`` disables both per-segment text and the bar.
                "verbose": None,
                "initial_prompt": self._initial_prompt or None,
                "temperature": self._temperature,
                "condition_on_previous_text": self._condition_on_previous_text,
            }
            if self._beam_size is not None:
                options["beam_size"] = self._beam_size
                options["patience"] = self._patience
            return self._model.transcribe(audio_array, **options)

        try:
            async with self.gpu_lock.section("stt"):
                result = await asyncio.to_thread(_sync_transcribe)
        except Exception:
            if _is_cuda_oom(_exc_cause()):
                logger.error(
                    "CUDA out of memory while running OpenAI Whisper "
                    "(device=%s). Set STT device: cpu or choose a smaller "
                    "model in config.yaml.",
                    self._device,
                )
            raise

        text = str(result.get("text", "")).strip()
        segments = list(result.get("segments") or [])
        confidences: list[float] = []
        for segment in segments:
            avg_logprob = float(segment.get("avg_logprob", -10.0))
            no_speech = float(segment.get("no_speech_prob", 0.0))
            confidences.append(math.exp(min(0.0, avg_logprob)) * (1.0 - no_speech))
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        confidence = max(0.0, min(1.0, confidence))
        return text, confidence


# ---------------------------------------------------------------------------- #
# Helpers (module-level so they're easy to monkeypatch in tests)
# ---------------------------------------------------------------------------- #
def _exc_cause() -> BaseException:
    """Return the currently-handled exception (the one in `except`)."""
    import sys

    return sys.exc_info()[1] or RuntimeError("unknown")


def _bytes_to_float32(np_module: Any, raw: bytes, sample_rate: int) -> Any:
    """Decode raw bytes into a float32 numpy array.

    Tries WAV (16-bit PCM) first via stdlib `wave`; if that fails, assumes
    raw little-endian int16 mono PCM. Returns ``None`` if neither path works.
    """
    import io
    import wave

    # WAV path
    try:
        with wave.open(io.BytesIO(raw), "rb") as wf:
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(n_frames)
        if sampwidth == 2:
            arr = np_module.frombuffer(frames, dtype=np_module.int16).astype(
                np_module.float32
            )
        elif sampwidth == 4:
            arr = np_module.frombuffer(frames, dtype=np_module.int32).astype(
                np_module.float32
            )
        else:
            return None
        # Downmix to mono by averaging channels.
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        arr /= 32768.0
        return arr
    except Exception:
        pass

    # Raw int16 PCM fallback.
    try:
        if len(raw) % 2 == 0 and len(raw) > 0:
            arr = np_module.frombuffer(raw, dtype=np_module.int16).astype(
                np_module.float32
            )
            arr /= 32768.0
            return arr
    except Exception:
        pass
    return None


def _resample_linear(
    np_module: Any, arr: Any, from_rate: int, to_rate: int
) -> Any:
    """Naive linear resample — fine for this pass."""
    if from_rate == to_rate or len(arr) == 0:
        return arr
    n_out = int(round(len(arr) * to_rate / from_rate))
    if n_out <= 1:
        return arr[:1].astype(np_module.float32)
    idx = np_module.linspace(0, len(arr) - 1, n_out)
    return np_module.interp(idx, np_module.arange(len(arr)), arr).astype(
        np_module.float32
    )


# ---------------------------------------------------------------------------- #
# Standalone test entry
#
#   python -m modules.stt --test                 # stub/demo path
#   python -m modules.stt --test --wav FILE.wav  # real OpenAI Whisper path
# ---------------------------------------------------------------------------- #
async def _standalone_test(wav_path: str | None = None) -> None:
    from core.config_loader import ModuleConfig

    mod = STTModule(
        config=ModuleConfig(device="cpu", model="base", compute_type="int8"),
        gpu_lock=GPULock(),
    )
    bus = EventBus()
    results: list[Event] = []

    async def record(event: Event) -> None:
        results.append(event)

    bus.subscribe("transcription_ready", record)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    if wav_path:
        with open(wav_path, "rb") as fh:
            audio_bytes = fh.read()
        bus.publish(
            "audio_captured",
            {"audio": audio_bytes, "sample_rate": _WHISPER_SAMPLE_RATE},
            trace_id="stt-wav",
        )
    else:
        bus.publish("audio_captured", {"audio": b"<fake>"}, trace_id="stt-only")
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    await mod.stop()

    assert results, "no transcription_ready emitted"
    r = results[0]
    assert r.trace_id == ("stt-wav" if wav_path else "stt-only"), (
        f"trace_id dropped: {r.trace_id}"
    )
    assert "confidence" in r.payload and "text" in r.payload
    print(f"trace_id={r.trace_id}")
    print(f"payload={r.payload}")
    print("OK stt standalone")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        wav: str | None = None
        if "--wav" in sys.argv:
            i = sys.argv.index("--wav")
            if i + 1 >= len(sys.argv):
                print("usage: python -m modules.stt --test --wav path/to/file.wav")
                sys.exit(2)
            wav = sys.argv[i + 1]
        asyncio.run(_standalone_test(wav_path=wav))
    else:
        print("usage: python -m modules.stt --test [--wav path/to/file.wav]")
