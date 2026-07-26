"""Microphone calibration without retaining biometric recordings."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from core.profile_manager import ProfileManager, device_fingerprint

SAMPLE_RATE = 16_000
BLOCK_SIZE = 512


class CalibrationQualityError(RuntimeError):
    """The captured signal cannot produce a trustworthy calibration."""


@dataclass(frozen=True)
class SignalMetrics:
    rms_dbfs_p50: float
    rms_dbfs_p95: float
    peak_dbfs: float
    clipping_ratio: float
    vad_p50: float
    vad_p95: float
    vad_speech_ratio: float = 0.0


def analyze_signal(samples: Any, vad_probabilities: Iterable[float]) -> SignalMetrics:
    """Summarize PCM16 audio in frame-level dBFS and VAD percentiles."""
    import numpy as np

    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
    if pcm.size == 0:
        raise CalibrationQualityError("запись пуста")
    normalized = pcm.astype(np.float64) / 32768.0
    frame_levels: list[float] = []
    for offset in range(0, len(normalized), BLOCK_SIZE):
        frame = normalized[offset : offset + BLOCK_SIZE]
        if len(frame):
            rms = math.sqrt(float(np.mean(frame * frame)))
            frame_levels.append(20.0 * math.log10(max(rms, 1e-7)))
    probabilities = list(float(value) for value in vad_probabilities)
    if not probabilities:
        probabilities = [0.0]
    peak = float(np.max(np.abs(normalized)))
    return SignalMetrics(
        rms_dbfs_p50=round(float(np.percentile(frame_levels, 50)), 2),
        rms_dbfs_p95=round(float(np.percentile(frame_levels, 95)), 2),
        peak_dbfs=round(20.0 * math.log10(max(peak, 1e-7)), 2),
        clipping_ratio=round(float(np.mean(np.abs(pcm.astype(np.int32)) >= 32760)), 6),
        vad_p50=round(float(np.percentile(probabilities, 50)), 4),
        vad_p95=round(float(np.percentile(probabilities, 95)), 4),
        vad_speech_ratio=round(
            sum(value >= 0.5 for value in probabilities) / len(probabilities), 4
        ),
    )


def derive_calibration(
    device: dict[str, Any], silence: SignalMetrics, speech: SignalMetrics
) -> dict[str, Any]:
    """Derive conservative capture parameters and reject poor measurements."""
    snr_db = speech.rms_dbfs_p95 - silence.rms_dbfs_p95
    vad_separation = speech.vad_p95 - silence.vad_p95
    if snr_db < 8.0:
        raise CalibrationQualityError(
            f"голос недостаточно отделён от шума (SNR {snr_db:.1f} дБ; нужно ≥ 8 дБ)"
        )
    if vad_separation < 0.15:
        raise CalibrationQualityError(
            "VAD недостаточно уверенно отличает речь от фона; подойдите ближе к микрофону"
        )
    if speech.clipping_ratio > 0.01:
        raise CalibrationQualityError(
            f"микрофон перегружен ({speech.clipping_ratio:.2%} отсчётов с клиппингом)"
        )
    if speech.vad_speech_ratio < 0.10:
        raise CalibrationQualityError(
            "в речевых упражнениях слишком мало речи; произнесите фразы полностью"
        )
    start_threshold = min(0.75, max(0.35, (silence.vad_p95 + speech.vad_p95) / 2))
    end_threshold = max(0.05, start_threshold - 0.15)
    gain_db = min(12.0, max(-6.0, -20.0 - speech.rms_dbfs_p50))
    return {
        "schema_version": 1,
        "device_fingerprint": device_fingerprint(device),
        "device": {
            "name": str(device.get("name", "unknown")),
            "max_input_channels": int(device.get("max_input_channels", 0) or 0),
            "default_samplerate": round(float(device.get("default_samplerate", 0) or 0)),
        },
        "sample_rate": SAMPLE_RATE,
        "vad_start_threshold": round(start_threshold, 3),
        "vad_end_threshold": round(end_threshold, 3),
        "end_silence_ms": 650,
        "min_speech_ms": 250,
        "pre_roll_ms": 320,
        "pcm_gain_db": round(gain_db, 2),
        "quality": {
            "noise_dbfs_p95": silence.rms_dbfs_p95,
            "speech_dbfs_p50": speech.rms_dbfs_p50,
            "speech_dbfs_p95": speech.rms_dbfs_p95,
            "snr_db": round(snr_db, 2),
            "vad_separation": round(vad_separation, 3),
            "clipping_ratio": speech.clipping_ratio,
        },
    }


def apply_pcm_gain(pcm: bytes, gain_db: float) -> bytes:
    """Apply bounded gain with saturation, preserving signed PCM16 format."""
    import numpy as np

    if not pcm or abs(gain_db) < 0.01:
        return pcm
    gain_db = min(12.0, max(-6.0, float(gain_db)))
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    samples *= 10.0 ** (gain_db / 20.0)
    return np.clip(np.rint(samples), -32768, 32767).astype(np.int16).tobytes()


def run_interactive_calibration(
    manager: ProfileManager,
    profile_id: str,
    *,
    profile_name: str | None = None,
    input_device: Any = None,
) -> dict[str, Any]:
    """Record fixed exercises, validate them, and atomically save aggregates."""
    import numpy as np
    import sounddevice as sd
    from silero_vad import load_silero_vad

    manager.ensure_profile(profile_id, profile_name)
    device = dict(sd.query_devices(input_device, kind="input"))
    if int(device.get("max_input_channels", 0) or 0) < 1:
        raise CalibrationQualityError("выбранный микрофон не имеет входных каналов")
    print(f"Профиль: {profile_id}; микрофон: {device.get('name', 'unknown')}")
    print("Сырые записи не сохраняются. Сначала 4 секунды тишины.")
    model = load_silero_vad(onnx=True)

    silence_pcm = _record(sd, 4.0, input_device)
    silence = analyze_signal(silence_pcm, _vad_scores(model, silence_pcm))
    speech_recordings: list[Any] = []
    prompts = (
        "Обычным голосом: Джарвис, открой браузер",
        "Тише обычного: Джарвис, который сейчас час",
        "Громче обычного: Джарвис, поставь напоминание через десять минут",
    )
    for prompt in prompts:
        input(f"\n{prompt}. Нажмите Enter и говорите 5 секунд...")
        speech_recordings.append(_record(sd, 5.0, input_device))
    speech_pcm = np.concatenate(speech_recordings)
    speech = analyze_signal(speech_pcm, _vad_scores(model, speech_pcm))
    calibration = derive_calibration(device, silence, speech)
    input("\nПроверка: Джарвис, открой калькулятор. Нажмите Enter и говорите...")
    validation_pcm = _record(sd, 5.0, input_device)
    validation = analyze_signal(validation_pcm, _vad_scores(model, validation_pcm))
    if (
        validation.vad_speech_ratio < 0.10
        or validation.vad_p95 < calibration["vad_start_threshold"]
    ):
        raise CalibrationQualityError(
            "проверочная фраза не прошла выбранный VAD-порог; калибровка не сохранена"
        )
    if validation.clipping_ratio > 0.01:
        raise CalibrationQualityError("проверочная фраза перегружает микрофон")
    calibration["quality"]["validation_vad_speech_ratio"] = (
        validation.vad_speech_ratio
    )
    manager.save_calibration(profile_id, calibration)
    manager.set_active(profile_id)
    print(
        "Калибровка сохранена: "
        f"SNR={calibration['quality']['snr_db']:.1f} дБ, "
        f"VAD={calibration['vad_start_threshold']:.2f}/"
        f"{calibration['vad_end_threshold']:.2f}, "
        f"gain={calibration['pcm_gain_db']:+.1f} дБ."
    )
    return calibration


def _record(sounddevice: Any, seconds: float, input_device: Any) -> Any:
    print(f"Запись {seconds:.0f} с...")
    return sounddevice.rec(
        round(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=input_device,
        blocking=True,
    ).reshape(-1)


def _vad_scores(model: Any, samples: Any) -> list[float]:
    import numpy as np
    import torch

    reset = getattr(model, "reset_states", None)
    if callable(reset):
        reset()
    pcm = np.asarray(samples, dtype=np.int16).reshape(-1)
    scores: list[float] = []
    for offset in range(0, len(pcm) - BLOCK_SIZE + 1, BLOCK_SIZE):
        block = pcm[offset : offset + BLOCK_SIZE].copy()
        result = model(torch.from_numpy(block).float().div_(32768.0), SAMPLE_RATE)
        scores.append(float(result.item() if hasattr(result, "item") else result))
    return scores
