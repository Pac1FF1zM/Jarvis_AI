"""Create a checked JSONL manifest from the official IPN MP4+Annotations download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.gesture.data import import_ipn_segments, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import the official IPN Hand MP4 dataset.")
    parser.add_argument("--videos", required=True, type=Path, help="Folder containing extracted .mp4 files")
    parser.add_argument("--annotations", required=True, type=Path, help="Extracted official Annotations folder")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_workspace/gesture_data/ipn_manifest.jsonl"),
        help="Destination JSONL manifest",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = import_ipn_segments(args.videos.resolve(), args.annotations.resolve())
    report = write_manifest(records, args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
