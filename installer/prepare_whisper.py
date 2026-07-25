"""Download and verify the configured Whisper model during installation."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--download-root", required=True)
    args = parser.parse_args()

    import whisper

    root = Path(args.download_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model, device="cpu", download_root=str(root))
    del model
    gc.collect()
    expected = root / f"{args.model}.pt"
    if not expected.is_file() or expected.stat().st_size == 0:
        raise RuntimeError(f"Whisper download did not produce {expected}")
    print(f"WHISPER_READY path={expected} bytes={expected.stat().st_size}")


if __name__ == "__main__":
    main()
