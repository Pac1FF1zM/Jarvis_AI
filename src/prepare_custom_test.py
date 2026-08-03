"""Prepare user-recorded isolated gestures for repeatable evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class GestureSpec:
    source_file: str
    output_file: str
    label: str
    class_name: str
    start_frame: int = 1
    end_frame: int | None = None


# Zoom filenames are intentionally crossed: visual inspection confirmed that the
# motion in zoom_out.mp4 is G10 (fingers spread) and zoom_in.mp4 is G11 (pinch).
GESTURES = (
    GestureSpec("pointing_with_one_finger.mp4", "B0A_01.mp4", "B0A", "Pointing with one finger"),
    GestureSpec("pointing_with_two_fingers.mp4", "B0B_01.mp4", "B0B", "Pointing with two fingers"),
    GestureSpec("click_one_finger.mp4", "G01_01.mp4", "G01", "Click with one finger"),
    GestureSpec("click_two_fingers.mp4", "G02_01.mp4", "G02", "Click with two fingers"),
    GestureSpec("drop_up.mp4", "G03_01.mp4", "G03", "Throw up", end_frame=100),
    GestureSpec("drop_down.mp4", "G04_01.mp4", "G04", "Throw down"),
    GestureSpec("drop_left.mp4", "G05_01.mp4", "G05", "Throw left"),
    GestureSpec("drop_right.mp4", "G06_01.mp4", "G06", "Throw right"),
    GestureSpec("open_twice.mp4", "G07_01.mp4", "G07", "Open twice"),
    GestureSpec("double_click_one_finger.mp4", "G08_01.mp4", "G08", "Double click with one finger"),
    GestureSpec("double_click_two_fingers.mp4", "G09_01.mp4", "G09", "Double click with two fingers"),
    GestureSpec("zoom_out.mp4", "G10_01.mp4", "G10", "Zoom in"),
    GestureSpec("zoom_in.mp4", "G11_01.mp4", "G11", "Zoom out"),
)


def probe_video(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frames < 1 or fps <= 0 or width < 1 or height < 1:
        raise ValueError(f"Invalid video metadata: {path}")
    return {"frames": frames, "fps": fps, "width": width, "height": height}


def write_frame_range(source: Path, destination: Path, start_frame: int, end_frame: int) -> int:
    metadata = probe_video(source)
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(metadata["fps"]),
        (int(metadata["width"]), int(metadata["height"])),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create video: {destination}")

    written = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        for _ in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Decode failed while trimming {source}")
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        capture.release()
    return written


def prepare(source_dir: Path, output_dir: Path, labels_path: Path, *, force: bool = False) -> dict:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    labels_path = labels_path.resolve()
    missing = [spec.source_file for spec in GESTURES if not (source_dir / spec.source_file).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source videos: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for spec in GESTURES:
        source = source_dir / spec.source_file
        destination = output_dir / spec.output_file
        if destination.exists() and not force:
            raise FileExistsError(f"Prepared file already exists: {destination}")

        source_metadata = probe_video(source)
        end_frame = spec.end_frame or int(source_metadata["frames"])
        if spec.start_frame < 1 or end_frame > int(source_metadata["frames"]) or end_frame < spec.start_frame:
            raise ValueError(f"Invalid range {spec.start_frame}..{end_frame} for {source}")

        if spec.start_frame == 1 and end_frame == int(source_metadata["frames"]):
            shutil.copy2(source, destination)
        else:
            write_frame_range(source, destination, spec.start_frame, end_frame)

        prepared_metadata = probe_video(destination)
        expected_frames = end_frame - spec.start_frame + 1
        if int(prepared_metadata["frames"]) != expected_frames:
            raise ValueError(
                f"Prepared frame count mismatch for {destination}: "
                f"expected {expected_frames}, got {prepared_metadata['frames']}"
            )

        records.append(
            {
                "file": spec.output_file,
                "label": spec.label,
                "class_name": spec.class_name,
                "source_file": spec.source_file,
                "source_start_frame": spec.start_frame,
                "source_end_frame": end_frame,
                "frames": int(prepared_metadata["frames"]),
                "fps": round(float(prepared_metadata["fps"]), 3),
                "duration_s": round(int(prepared_metadata["frames"]) / float(prepared_metadata["fps"]), 3),
                "width": int(prepared_metadata["width"]),
                "height": int(prepared_metadata["height"]),
            }
        )

    fieldnames = list(records[0])
    with labels_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    report = {
        "status": "completed",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "labels_path": str(labels_path),
        "gesture_count": len(records),
        "gestures": records,
        "specification": [asdict(spec) for spec in GESTURES],
    }
    report_path = labels_path.with_name("preparation_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/custom_test/videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/custom_test/prepared"))
    parser.add_argument("--labels", type=Path, default=Path("data/custom_test/labels.csv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = prepare(args.source_dir, args.output_dir, args.labels, force=args.force)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
