"""User-facing no-action diagnostic for Parakeet STT and production NLU."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

MICROPHONE_SETTLE_SECONDS = 0.20
MICROPHONE_POST_ROLL_SECONDS = 0.45

from core.config_loader import load_config
from experiments.parakeet.shadow_pipeline import (
    PersistentParakeetDecoder,
    ShadowNLU,
    decode_in_child,
)


def _pcm_to_wav(pcm: bytes, *, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _device_value(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def _format_output(*, stt: dict[str, object] | None, nlu: dict[str, object], nlu_ms: float) -> dict[str, object]:
    return {
        "mode": "parakeet_shadow_nlu_diagnostic",
        "no_action": True,
        "network_allowed": False,
        "stt": stt,
        "nlu": {**nlu, "inference_ms": nlu_ms},
        "execution": "blocked",
        "warning": "Predicted actions are diagnostics only and were not executed.",
    }


def _run_microphone(nlu: ShadowNLU, *, device: int | str | None, timeout: float) -> None:
    print("Загрузка и прогрев Parakeet на GPU. Это происходит один раз...")
    with PersistentParakeetDecoder(
        repository_root=REPOSITORY_ROOT, timeout_seconds=timeout
    ) as decoder:
        assert decoder.startup is not None
        print(
            "Готово. Модель загружена за "
            f"{float(decoder.startup.get('model_load_ms') or 0) / 1000:.2f} с."
        )
        print("Enter — начать запись; Enter ещё раз — закончить; Q + Enter — выход.")
        while True:
            try:
                choice = input("\nНажми Enter для записи (или Q для выхода): ").strip().casefold()
            except EOFError:
                print("\nВвод закрыт. Никакие команды не выполнялись.")
                return
            if choice in {"q", "quit", "exit", "й"}:
                break
            # Do not require or initialise PortAudio until the user actually
            # starts a capture. Closed/non-interactive stdin must exit cleanly
            # in diagnostics and environment-isolated tests.
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise SystemExit(
                    "sounddevice is missing from venv; install project requirements"
                ) from exc
            chunks: list[bytes] = []

            def callback(indata, _frames, _time_info, status) -> None:
                if status:
                    print(f"\nПредупреждение микрофона: {status}", file=sys.stderr)
                chunks.append(indata.copy().tobytes())

            with sd.InputStream(
                samplerate=16_000,
                channels=1,
                dtype="int16",
                device=device,
                callback=callback,
            ):
                # Do not invite speech until PortAudio has actually opened the
                # device. Keep a small quiet lead-in and tail so the Enter key
                # cannot clip the first or final phoneme of a short command.
                time.sleep(MICROPHONE_SETTLE_SECONDS)
                recorded_at = time.perf_counter()
                print("● Говори сейчас. Нажми Enter, когда закончишь.")
                try:
                    input()
                except EOFError:
                    print("\nВвод закрыт. Запись отброшена; команды не выполнялись.")
                    return
                print("Завершаю фразу...")
                time.sleep(MICROPHONE_POST_ROLL_SECONDS)
            capture_ms = (time.perf_counter() - recorded_at) * 1000.0
            pcm = b"".join(chunks)
            if len(pcm) < 1600:
                print("Запись слишком короткая; попробуй ещё раз.")
                continue
            try:
                stt = decoder.decode(_pcm_to_wav(pcm))
            except TimeoutError as exc:
                print(json.dumps({
                    "mode": "parakeet_shadow_nlu_diagnostic",
                    "no_action": True,
                    "stt": {"event": "timeout", "error": str(exc)},
                    "nlu": None,
                    "execution": "blocked",
                    "warning": "Capture discarded. The worker will restart for the next phrase.",
                }, ensure_ascii=False, indent=2))
                continue
            stt["capture_ms"] = capture_ms
            stt["audio_duration_ms"] = len(pcm) / 2 / 16_000 * 1000.0
            stt["post_roll_ms"] = MICROPHONE_POST_ROLL_SECONDS * 1000.0
            started = time.perf_counter()
            prediction = nlu.predict(str(stt.get("text", "")))
            nlu_ms = (time.perf_counter() - started) * 1000.0
            print(json.dumps(_format_output(stt=stt, nlu=prediction, nlu_ms=nlu_ms), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-action Parakeet -> Jarvis NLU diagnostic (never executes tools)"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="local 16 kHz PCM WAV to transcribe with Parakeet")
    source.add_argument("--text", help="skip STT and test production NLU with this text")
    source.add_argument("--mic", action="store_true", help="live push-to-talk microphone loop")
    source.add_argument("--list-devices", action="store_true", help="show microphone devices and exit")
    parser.add_argument("--device", help="microphone device index or name for --mic")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise SystemExit("sounddevice is missing from the Jarvis venv") from exc
        print(sd.query_devices())
        return

    config = load_config(str((REPOSITORY_ROOT / args.config).resolve()))
    nlu_config = config.module("nlu")
    checkpoint = (REPOSITORY_ROOT / nlu_config.model).resolve()
    threshold = float(nlu_config.params.get("confidence_threshold", 0.55))
    if args.mic:
        print("Загрузка production NLU...")
    nlu_predictor = ShadowNLU(checkpoint, threshold=threshold)

    if args.mic:
        try:
            _run_microphone(nlu_predictor, device=_device_value(args.device), timeout=args.timeout)
        except KeyboardInterrupt:
            print("\nОстановлено. Никакие команды не выполнялись.")
        return

    stt: dict[str, object] | None = None
    text = args.text
    if args.wav:
        stt = decode_in_child(
            Path(args.wav), repository_root=REPOSITORY_ROOT, timeout_seconds=args.timeout
        )
        text = str(stt["text"])

    started = time.perf_counter()
    nlu = nlu_predictor.predict(str(text or ""))
    nlu_ms = (time.perf_counter() - started) * 1000.0
    output = _format_output(stt=stt, nlu=nlu, nlu_ms=nlu_ms)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
