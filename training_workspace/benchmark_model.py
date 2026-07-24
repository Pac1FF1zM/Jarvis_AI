"""Measure Jarvis NLU accuracy and single-command latency on this computer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.nlu.custom_data import load_jsonl
from ml.nlu.data import build_examples
from training_workspace.run import benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-data", default="training_workspace/data/validation.jsonl")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args()
    examples = build_examples("validation") + load_jsonl(args.validation_data)
    result = benchmark(
        Path(args.checkpoint), examples, device=args.device,
        warmup=args.warmup, repetitions=args.repetitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
