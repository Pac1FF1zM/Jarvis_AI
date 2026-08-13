"""Persistent local Parakeet decoder used by production and shadow clients."""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

from experiments.parakeet.worker.backend import ParakeetBackend


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _serve(model_dir: Path, provider: str) -> None:
    backend = ParakeetBackend(model_dir=model_dir, provider=provider)
    backend.start()
    try:
        backend.warm_up()
        _emit({
            "event": "ready",
            "model_load_ms": backend.load_ms,
            "warm_up_ms": backend.warm_up_ms,
            "health": backend.health(),
        })
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("op") == "close":
                    _emit({"event": "closed"})
                    return
                if request.get("op") != "decode":
                    raise ValueError("unsupported worker operation")
                audio = base64.b64decode(request["audio_b64"], validate=True)
                started = time.perf_counter()
                text = backend.transcribe(audio)
                _emit({
                    "event": "result",
                    "text": text,
                    "confidence": None,
                    "decode_ms": (time.perf_counter() - started) * 1000.0,
                })
            except Exception as exc:
                _emit({
                    "event": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    _serve(Path(args.model_dir).resolve(), args.provider)


if __name__ == "__main__":
    main()
