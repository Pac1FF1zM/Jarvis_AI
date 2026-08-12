"""STT module with production Whisper and experimental Parakeet engines.

Subscribes to ``audio_captured``, runs transcription under the shared GPU lock,
and publishes ``transcription_ready`` as a child event so the trace_id carries
through to the LLM.

OpenAI Whisper is wired behind a lazy, guarded import so the project stays
fully runnable without it installed:

- If ``whisper`` is importable AND ``self.config.enabled`` is true, the
  model is loaded **once** in :meth:`STTModule.start` (loading is expensive —
  never per-call) and used for every transcribe.
- If ``whisper`` is not importable, the module logs an actionable message and
  emits an empty final transcript. It never fabricates a command.
- If the incoming ``audio_captured`` payload is not decodable audio (e.g. the
  literal stub marker bytes that ``modules/wake_word.py`` still publishes while
  no real microphone exists), the module logs a clear warning and emits an
  empty transcript with zero confidence.

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
import io
import logging
import math
from pathlib import Path
import wave
from typing import Any

from core.base_module import BaseModule
from core.event_bus import EventBus, Event
from core.event_payloads import TranscriptionReadyPayload
from core.gpu_lock import GPULock
from modules.parakeet_client import DEFAULT_MODEL_DIR, PersistentParakeetClient

logger = logging.getLogger("jarvis.module.stt")

# Historical demo string. It is deliberately never emitted by production STT:
# malformed audio must not fabricate a command.
STUB_TEXT = "сколько сейчас времени"
STUB_CONFIDENCE = 0.0

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
        # Real model handle, or None when the optional runtime is unavailable.
        self._model: Any = None
        # numpy is only needed on the real path; import lazily so the module
        # stays importable when only stub mode is used.
        self._np: Any = None
        params = getattr(config, "params", {}) or {}
        self._engine = str(params.get("engine", "whisper")).strip().casefold()
        if self._engine not in {"whisper", "parakeet"}:
            raise ValueError("stt.params.engine must be 'whisper' or 'parakeet'")
        self._experimental_production = bool(
            params.get("experimental_production", False)
        )
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
        self._parakeet: PersistentParakeetClient | None = None
        self._parakeet_model_dir = str(
            params.get("parakeet_model_dir", DEFAULT_MODEL_DIR)
        )
        self._parakeet_python = str(
            params.get("parakeet_python", "venv/Scripts/python.exe")
        )
        self._parakeet_timeout = float(params.get("parakeet_timeout_seconds", 45.0))
        if self._parakeet_timeout <= 0:
            raise ValueError("stt.params.parakeet_timeout_seconds must be > 0")

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("audio_captured", self._on_audio)

        if self._engine == "parakeet":
            if not self._experimental_production:
                raise RuntimeError(
                    "Parakeet production wiring requires "
                    "stt.params.experimental_production: true"
                )
            self._parakeet = PersistentParakeetClient(
                repository_root=Path.cwd(),
                model_dir=self._parakeet_model_dir,
                python_executable=self._parakeet_python,
                provider=self._device,
                timeout_seconds=self._parakeet_timeout,
            )
            startup = await asyncio.to_thread(self._parakeet.start)
            logger.warning(
                "EXPERIMENTAL_STT_ACTIVE engine=parakeet provider=%s "
                "model=%s revision=%s load_ms=%.2f warm_up_ms=%.2f",
                self._device,
                startup.get("health", {}).get("model_id", "unknown"),
                startup.get("health", {}).get("model_revision", "unknown"),
                float(startup.get("model_load_ms") or 0.0),
                float(startup.get("warm_up_ms") or 0.0),
            )
        elif _WHISPER is None:
            logger.warning(
                "openai-whisper not installed — pip install openai-whisper; "
                "STT will return empty transcripts"
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
            "STTModule started (mode=%s) engine=%s device=%s model=%s language=%s "
            "temperature=%.1f beam_size=%s previous_text=%s",
            (
                "real"
                if self._model is not None or self._parakeet is not None
                else "unavailable"
            ),
            self._engine,
            self._device,
            self.config.model,
            self._language,
            self._temperature,
            self._beam_size,
            self._condition_on_previous_text,
        )

    async def stop(self) -> None:
        if self._parakeet is not None:
            parakeet, self._parakeet = self._parakeet, None
            await asyncio.to_thread(parakeet.close)
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
        if self.bus.is_trace_cancelled(event.trace_id):
            return
        try:
            text, confidence = await self._transcribe(event.payload)
        except Exception as exc:
            logger.error("STT failed trace=%s: %s", event.trace_id, exc)
            text, confidence = "", 0.0
        if self.bus.is_trace_cancelled(event.trace_id):
            return
        out = event.child(
            "transcription_ready",
            TranscriptionReadyPayload(
                text=text,
                confidence=confidence,
                source=self._engine if self._engine == "parakeet" else None,
            ),
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
          - Otherwise -> an empty, fail-closed final transcript.
        """
        if self._engine == "parakeet":
            return await self._transcribe_parakeet(audio_payload)

        if self._model is None:
            # Missing STT is an explicit empty result, never a guessed command.
            async with self.gpu_lock.section("stt"):
                await asyncio.sleep(0.05)
            return "", 0.0

        # Real model loaded — invalid audio fails closed, never as a command.
        audio_array = self._decode_audio(audio_payload)
        if audio_array is None:
            logger.warning("non-decodable audio payload — returning empty transcription")
            async with self.gpu_lock.section("stt"):
                await asyncio.sleep(0.05)
            return "", 0.0

        return await self._transcribe_real(audio_array)

    async def _transcribe_parakeet(
        self, audio_payload: dict[str, Any]
    ) -> tuple[str, float]:
        """Send a strict 16 kHz mono PCM WAV to the isolated warm worker."""
        if self._parakeet is None:
            return "", 0.0
        wav_bytes = _payload_to_pcm_wav(audio_payload)
        if wav_bytes is None:
            logger.warning(
                "non-decodable Parakeet audio payload — returning empty transcription"
            )
            return "", 0.0
        async with self.gpu_lock.section("stt"):
            result = await asyncio.to_thread(self._parakeet.decode, wav_bytes)
        text = str(result.get("text", "")).strip()
        logger.info(
            "PARAKEET_DECODE decode_ms=%.2f text_chars=%d",
            float(result.get("decode_ms") or 0.0),
            len(text),
        )
        # The Transformers TDT adapter does not expose calibrated utterance
        # confidence. Zero explicitly means "unavailable", not "rejected";
        # command acceptance is controlled by NLU intent confidence.
        return text, 0.0

    def _decode_audio(self, audio_payload: dict[str, Any]) -> Any:
        """Decode payload bytes into a float32 numpy array at 16 kHz.

        Returns ``None`` if the payload is the known stub marker or cannot be
        decoded — the caller emits an empty transcript in that case.
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
                "numpy not installed — cannot decode audio; returning empty transcript"
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


def _payload_to_pcm_wav(audio_payload: dict[str, Any]) -> bytes | None:
    """Normalize Jarvis raw PCM (or an already valid WAV) for Parakeet."""
    raw = audio_payload.get("audio")
    if not isinstance(raw, (bytes, bytearray)):
        return None
    raw_bytes = bytes(raw)
    if not raw_bytes or raw_bytes in _STUB_AUDIO_MARKERS:
        return None
    sample_rate = int(audio_payload.get("sample_rate", _WHISPER_SAMPLE_RATE))
    channels = int(audio_payload.get("channels", 1))
    sample_width = int(audio_payload.get("sample_width", 2))
    if raw_bytes.startswith(b"RIFF"):
        try:
            with wave.open(io.BytesIO(raw_bytes), "rb") as wav_file:
                valid = (
                    wav_file.getframerate() == _WHISPER_SAMPLE_RATE
                    and wav_file.getnchannels() == 1
                    and wav_file.getsampwidth() == 2
                    and wav_file.getnframes() > 0
                )
            return raw_bytes if valid else None
        except (wave.Error, EOFError):
            return None
    if (
        sample_rate != _WHISPER_SAMPLE_RATE
        or channels != 1
        or sample_width != 2
        or len(raw_bytes) % 2
    ):
        return None
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_WHISPER_SAMPLE_RATE)
        wav_file.writeframes(raw_bytes)
    return output.getvalue()


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
