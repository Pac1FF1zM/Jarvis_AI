"""Create a checked JSONL manifest from the official IPN MP4+Annotations download."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.gesture.data import import_ipn_segments, write_manifest


class GestureImportInputError(ValueError):
    """The requested IPN source folders are absent or incomplete."""


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


def build_manifest(videos: Path, annotations: Path, output: Path) -> dict:
    videos = videos.resolve()
    annotations = annotations.resolve()
    if not videos.is_dir():
        raise GestureImportInputError(
            f"IPN videos directory was not found: {videos}\n"
            "Replace <folder-with-AVI> with the real folder containing extracted .avi files."
        )
    if not annotations.is_dir():
        raise GestureImportInputError(
            f"IPN annotations directory was not found: {annotations}\n"
            "Replace <annotations-folder> with the real folder containing classIdx.txt "
            "and Annot_TrainList.txt."
        )
    try:
        records = import_ipn_segments(videos, annotations)
    except FileNotFoundError as error:
        raise GestureImportInputError(str(error)) from error
    return write_manifest(records, output.resolve())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = build_manifest(args.videos, args.annotations, args.output)
    except GestureImportInputError as error:
        parser.exit(2, f"GESTURE_IMPORT_INPUT_ERROR\n{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
