"""Local-only Parakeet TDT adapter for the no-action diagnostic."""
from __future__ import annotations

import io
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
SAMPLE_RATE = 16_000


class ParakeetConfigurationError(RuntimeError):
    """The local model/runtime is absent or incompatible."""


class AudioFormatError(ValueError):
    """Captured audio is not complete 16-bit PCM WAV."""


@dataclass(frozen=True)
class BackendHealth:
    state: str
    provider: str
    model_dir: str
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION


def _decode_wav(audio: bytes) -> list[float]:
    if not audio:
        raise AudioFormatError("audio must be non-empty WAV bytes")
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioFormatError("expected a valid PCM WAV capture") from exc
    if channels != 1 or width != 2 or sample_rate != SAMPLE_RATE or len(frames) % 2:
        raise AudioFormatError("Parakeet input must be mono 16-bit PCM WAV at 16 kHz")
    import array

    values = array.array("h")
    values.frombytes(frames)
    if not values:
        raise AudioFormatError("audio contains no samples")
    return [value / 32768.0 for value in values]


def _decoded_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return str(value[0] if value else "").strip()
    return str(value).strip()


class ParakeetBackend:
    """Loads one pinned local snapshot and never performs network access."""

    def __init__(self, *, model_dir: str | Path, provider: str = "cuda") -> None:
        self.model_dir = Path(model_dir).resolve()
        self.provider = provider.casefold()
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None
        self.load_ms: float | None = None
        self.warm_up_ms: float | None = None
        self._closed = False

    def start(self) -> None:
        required = ("config.json", "processor_config.json", "model.safetensors")
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise ParakeetConfigurationError(
                f"approved Parakeet snapshot is incomplete at {self.model_dir}; missing {missing}"
            )
        try:
            import torch
            from transformers import AutoModelForTDT, AutoProcessor
        except ImportError as exc:
            raise ParakeetConfigurationError(
                "Parakeet runtime is missing; run SETUP_PARAKEET.cmd --runtime"
            ) from exc
        if self.provider not in {"cuda", "cpu"}:
            raise ParakeetConfigurationError("provider must be 'cuda' or 'cpu'")
        if self.provider == "cuda" and not torch.cuda.is_available():
            raise ParakeetConfigurationError(
                "CUDA was requested but PyTorch cannot see the NVIDIA GPU; no CPU fallback was used"
            )

        started = time.perf_counter()
        dtype = torch.float16 if self.provider == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForTDT.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
        )
        model.to(self.provider).eval()
        self._torch, self._processor, self._model = torch, processor, model
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def warm_up(self) -> None:
        started = time.perf_counter()
        self.transcribe(_silence_wav(0.25))
        self.warm_up_ms = (time.perf_counter() - started) * 1000.0

    def transcribe(self, wav_bytes: bytes) -> str:
        if self._closed or self._model is None:
            raise ParakeetConfigurationError("Parakeet backend is not started")
        import numpy as np

        samples = np.asarray(_decode_wav(wav_bytes), dtype=np.float32)
        torch, processor, model = self._torch, self._processor, self._model
        inputs = processor([samples], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(device=model.device, dtype=model.dtype)
        with torch.inference_mode():
            # Jarvis captures short commands.  Bound decoder work so a noisy
            # or adversarial clip cannot consume the worker indefinitely.
            # Match the model/runtime's previously effective 40-token command
            # bound explicitly. This removes the generic generation warning;
            # longer speech belongs in a dictation backend, not this command UI.
            output = model.generate(
                **inputs,
                return_dict_in_generate=True,
                max_new_tokens=40,
            )
        return _decoded_text(processor.decode(output.sequences, skip_special_tokens=True))

    def close(self) -> None:
        model, torch = self._model, self._torch
        self._closed = True
        self._model = self._processor = self._torch = None
        del model
        if torch is not None and self.provider == "cuda":
            torch.cuda.empty_cache()

    def health(self) -> dict[str, object]:
        state = "closed" if self._closed else ("ready" if self._model is not None else "uninitialized")
        return asdict(BackendHealth(state, self.provider, str(self.model_dir)))


def _silence_wav(seconds: float) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\x00\x00" * int(SAMPLE_RATE * seconds))
    return output.getvalue()
