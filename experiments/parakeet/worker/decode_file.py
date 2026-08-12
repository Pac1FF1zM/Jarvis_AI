"""Decode one local WAV with Parakeet and print diagnostic JSON only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.parakeet.worker.backend import MODEL_ID, MODEL_REVISION, ParakeetBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="No-action Parakeet WAV diagnostic")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    wav = Path(args.wav).resolve()
    if not wav.is_file():
        raise FileNotFoundError(f"WAV not found: {wav}")
    backend = ParakeetBackend(model_dir=args.model_dir, provider=args.provider)
    backend.start()
    try:
        backend.warm_up()
        started = time.perf_counter()
        text = backend.transcribe(wav.read_bytes())
        result = {
            "schema_version": 1,
            "backend": "parakeet_tdt",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "provider": args.provider,
            "text": text,
            "confidence": None,
            "model_load_ms": backend.load_ms,
            "warm_up_ms": backend.warm_up_ms,
            "decode_ms": (time.perf_counter() - started) * 1000.0,
            "execution": "blocked",
        }
        print(json.dumps(result, ensure_ascii=False))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
