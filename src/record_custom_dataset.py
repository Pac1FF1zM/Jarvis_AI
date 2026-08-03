"""Interactive webcam recorder for a labelled custom IPN Hand dataset."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2


@dataclass(frozen=True)
class GestureClass:
    label: str
    name: str
    instruction: str


@dataclass(frozen=True)
class CaptureItem:
    label: str
    class_name: str
    instruction: str
    take: int
    takes_for_class: int


GESTURE_CLASSES = (
    GestureClass("B0A", "Pointing with one finger", "Point with one finger and keep the full motion in frame."),
    GestureClass("B0B", "Pointing with two fingers", "Point with two fingers and keep the full motion in frame."),
    GestureClass("G01", "Click with one finger", "Perform one click with the index finger."),
    GestureClass("G02", "Click with two fingers", "Perform one click with two fingers together."),
    GestureClass("G03", "Throw up", "Perform one clear upward throw motion."),
    GestureClass("G04", "Throw down", "Perform one clear downward throw motion."),
    GestureClass("G05", "Throw left", "Perform one clear throw to the left as shown in preview."),
    GestureClass("G06", "Throw right", "Perform one clear throw to the right as shown in preview."),
    GestureClass("G07", "Open twice", "Open the hand twice; both openings must be recorded."),
    GestureClass("G08", "Double click with one finger", "Perform two clicks with the index finger."),
    GestureClass("G09", "Double click with two fingers", "Perform two clicks with two fingers together."),
    GestureClass("G10", "Zoom in", "Start pinched, then spread the fingers."),
    GestureClass("G11", "Zoom out", "Start spread, then pinch the fingers."),
)

NON_GESTURE = GestureClass(
    "D0X",
    "Non-gesture",
    "Move your hands naturally without performing any listed gesture.",
)

PRESETS = {
    "train": {"gesture_repetitions": 5, "non_gesture_repetitions": 15},
    "val": {"gesture_repetitions": 2, "non_gesture_repetitions": 5},
    "test": {"gesture_repetitions": 2, "non_gesture_repetitions": 5},
}

MANIFEST_FIELDS = (
    "path",
    "split",
    "label",
    "class_name",
    "session_id",
    "take",
    "frames",
    "fps",
    "duration_s",
    "width",
    "height",
    "recorded_at",
)


def build_capture_plan(
    split: str,
    gesture_repetitions: int | None = None,
    non_gesture_repetitions: int | None = None,
) -> list[CaptureItem]:
    if split not in PRESETS:
        raise ValueError(f"Unknown split: {split}")
    preset = PRESETS[split]
    gesture_repetitions = (
        preset["gesture_repetitions"] if gesture_repetitions is None else gesture_repetitions
    )
    non_gesture_repetitions = (
        preset["non_gesture_repetitions"]
        if non_gesture_repetitions is None
        else non_gesture_repetitions
    )
    if gesture_repetitions < 1:
        raise ValueError("gesture_repetitions must be at least 1")
    if non_gesture_repetitions < 1:
        raise ValueError("non_gesture_repetitions must be at least 1")

    plan = [
        CaptureItem(
            gesture.label,
            gesture.name,
            gesture.instruction,
            take,
            gesture_repetitions,
        )
        for gesture in GESTURE_CLASSES
        for take in range(1, gesture_repetitions + 1)
    ]
    plan.extend(
        CaptureItem(
            NON_GESTURE.label,
            NON_GESTURE.name,
            NON_GESTURE.instruction,
            take,
            non_gesture_repetitions,
        )
        for take in range(1, non_gesture_repetitions + 1)
    )
    return plan


def validate_session_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("session_id may contain only letters, digits, underscores, and hyphens")
    return value


def append_manifest(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row[field] for field in MANIFEST_FIELDS})


def draw_text(frame: Any, text: str, origin: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_overlay(
    frame: Any,
    *,
    split: str,
    item: CaptureItem,
    item_index: int,
    item_count: int,
    state: str,
    state_progress: float,
    seconds_remaining: float | None = None,
) -> Any:
    display = frame.copy()
    height, width = display.shape[:2]
    top_height = min(170, max(125, height // 4))
    bottom_height = min(70, max(48, height // 10))
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (width, top_height), (12, 18, 30), -1)
    cv2.rectangle(overlay, (0, height - bottom_height), (width, height), (12, 18, 30), -1)
    cv2.addWeighted(overlay, 0.82, display, 0.18, 0, display)

    draw_text(display, f"{split.upper()}  |  clip {item_index + 1}/{item_count}", (22, 30), 0.7, (190, 205, 225), 1)
    draw_text(display, f"{item.label}  {item.class_name}", (22, 70), 1.0, (90, 210, 255), 2)
    draw_text(display, f"Take {item.take}/{item.takes_for_class}", (22, 101), 0.65, (225, 235, 245), 1)
    draw_text(display, item.instruction, (22, 132), 0.53, (220, 225, 235), 1)
    draw_text(display, "Raw preview is saved exactly as shown (no horizontal flip).", (22, 157), 0.45, (155, 175, 205), 1)

    state_text = state.upper()
    state_color = (110, 220, 120)
    if state == "recording":
        state_text = "RECORDING"
        state_color = (70, 80, 255)
        cv2.circle(display, (width - 35, 32), 11, state_color, -1)
    elif state == "countdown" and seconds_remaining is not None:
        state_text = f"START IN {max(1, int(seconds_remaining + 0.999))}"
        state_color = (80, 210, 255)
    elif state == "review":
        state_text = "SAVED PREVIEW  |  R = redo  ENTER = accept"
        state_color = (110, 220, 120)
    elif state == "prompt":
        state_text = "GET READY"
        state_color = (80, 210, 255)
    elif state == "waiting":
        state_text = "SPACE = start session  |  ESC = exit"

    draw_text(display, state_text, (22, height - 24), 0.62, state_color, 2)
    bar_x0, bar_x1 = max(330, width // 2), width - 22
    bar_y0, bar_y1 = height - 38, height - 20
    cv2.rectangle(display, (bar_x0, bar_y0), (bar_x1, bar_y1), (75, 88, 108), 1)
    fill_x = bar_x0 + int((bar_x1 - bar_x0) * min(max(state_progress, 0.0), 1.0))
    cv2.rectangle(display, (bar_x0, bar_y0), (fill_x, bar_y1), state_color, -1)
    return display


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4 file: {path}")
    return writer


def record_session(args: argparse.Namespace, plan: list[CaptureItem]) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    session_dir = output_root / args.split / args.session_id
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(f"Session directory is not empty: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"

    session_plan = {
        "split": args.split,
        "session_id": args.session_id,
        "camera": args.camera,
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "record_seconds": args.record_seconds,
        "items": [asdict(item) for item in plan],
    }
    (session_dir / "session_plan.json").write_text(
        json.dumps(session_plan, indent=2) + "\n", encoding="utf-8"
    )

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(args.fps)
    writer_fps = min(max(actual_fps, 1.0), 120.0)
    target_frames = max(1, round(args.record_seconds * writer_fps))

    window = "Jarvis gesture dataset recorder"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(args.width, 1280), min(args.height, 720))

    state = "waiting"
    state_started = time.monotonic()
    item_index = 0
    writer: cv2.VideoWriter | None = None
    temp_path: Path | None = None
    final_path: Path | None = None
    last_recorded_frame = None
    frames_written = 0
    accepted = 0
    aborted = False

    def begin_state(value: str) -> None:
        nonlocal state, state_started
        state = value
        state_started = time.monotonic()

    def discard_temporary() -> None:
        nonlocal writer, temp_path
        if writer is not None:
            writer.release()
            writer = None
        if temp_path is not None and temp_path.is_file():
            temp_path.unlink()

    def accept_current() -> None:
        nonlocal accepted, item_index, temp_path, final_path
        if temp_path is None or final_path is None or not temp_path.is_file():
            raise RuntimeError("No completed temporary recording to accept")
        os.replace(temp_path, final_path)
        item = plan[item_index]
        relative_path = final_path.relative_to(output_root).as_posix()
        append_manifest(
            manifest_path,
            {
                "path": relative_path,
                "split": args.split,
                "label": item.label,
                "class_name": item.class_name,
                "session_id": args.session_id,
                "take": item.take,
                "frames": frames_written,
                "fps": round(writer_fps, 3),
                "duration_s": round(frames_written / writer_fps, 3),
                "width": int(last_recorded_frame.shape[1]),
                "height": int(last_recorded_frame.shape[0]),
                "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        )
        accepted += 1
        item_index += 1
        temp_path = None
        final_path = None
        if item_index < len(plan):
            begin_state("prompt")

    try:
        while item_index < len(plan):
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame capture failed")
            now = time.monotonic()
            elapsed = now - state_started
            item = plan[item_index]

            if state == "prompt" and elapsed >= args.prompt_seconds:
                begin_state("countdown")
                elapsed = 0.0
            elif state == "countdown" and elapsed >= args.countdown_seconds:
                final_path = session_dir / f"{item.label}_{item.take:02d}.mp4"
                temp_path = session_dir / f"{item.label}_{item.take:02d}.partial.mp4"
                writer = open_writer(temp_path, writer_fps, (camera_frame.shape[1], camera_frame.shape[0]))
                frames_written = 0
                begin_state("recording")
                elapsed = 0.0

            if state == "recording":
                if writer is None:
                    raise RuntimeError("Writer is unavailable during recording")
                writer.write(camera_frame)
                frames_written += 1
                last_recorded_frame = camera_frame.copy()
                if frames_written >= target_frames:
                    writer.release()
                    writer = None
                    begin_state("review")
                    elapsed = 0.0
            elif state == "review" and elapsed >= args.review_seconds:
                accept_current()
                if item_index >= len(plan):
                    break
                item = plan[item_index]
                elapsed = 0.0

            if state == "recording":
                progress = frames_written / target_frames
                remaining = max(0.0, (target_frames - frames_written) / writer_fps)
            elif state == "prompt":
                progress = elapsed / max(args.prompt_seconds, 0.001)
                remaining = max(0.0, args.prompt_seconds - elapsed)
            elif state == "countdown":
                progress = elapsed / max(args.countdown_seconds, 0.001)
                remaining = max(0.0, args.countdown_seconds - elapsed)
            elif state == "review":
                progress = elapsed / max(args.review_seconds, 0.001)
                remaining = max(0.0, args.review_seconds - elapsed)
            else:
                progress = 0.0
                remaining = None

            base_frame = last_recorded_frame if state == "review" and last_recorded_frame is not None else camera_frame
            display = render_overlay(
                base_frame,
                split=args.split,
                item=item,
                item_index=item_index,
                item_count=len(plan),
                state=state,
                state_progress=progress,
                seconds_remaining=remaining,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                aborted = True
                break
            if state == "waiting" and key == 32:
                begin_state("prompt")
            elif state == "review" and key in (10, 13):
                accept_current()
            elif state == "review" and key in (ord("r"), ord("R")):
                discard_temporary()
                begin_state("prompt")
    finally:
        if writer is not None:
            writer.release()
        capture.release()
        cv2.destroyAllWindows()
        if temp_path is not None and temp_path.is_file():
            temp_path.unlink()

    result = {
        "status": "aborted" if aborted else "completed",
        "split": args.split,
        "session_id": args.session_id,
        "session_dir": str(session_dir),
        "manifest": str(manifest_path),
        "planned_clips": len(plan),
        "accepted_clips": accepted,
        "remaining_clips": len(plan) - accepted,
        "camera_fps": writer_fps,
    }
    (session_dir / "session_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(PRESETS), required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("data/custom_capture"))
    parser.add_argument("--session-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--gesture-repetitions", type=int)
    parser.add_argument("--non-gesture-repetitions", type=int)
    parser.add_argument("--record-seconds", type=float, default=3.0)
    parser.add_argument("--prompt-seconds", type=float, default=2.0)
    parser.add_argument("--countdown-seconds", type=float, default=3.0)
    parser.add_argument("--review-seconds", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_session_id(args.session_id)
    for name in ("record_seconds", "prompt_seconds", "countdown_seconds", "review_seconds", "fps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.width < 160 or args.height < 120:
        parser.error("Camera resolution is too small")
    return args


def main() -> None:
    args = parse_args()
    plan = build_capture_plan(
        args.split,
        gesture_repetitions=args.gesture_repetitions,
        non_gesture_repetitions=args.non_gesture_repetitions,
    )
    if args.dry_run:
        counts = Counter(item.label for item in plan)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "split": args.split,
                    "session_id": args.session_id,
                    "output_root": str(args.output_root.resolve()),
                    "clip_count": len(plan),
                    "class_counts": dict(counts),
                    "estimated_minutes": round(
                        len(plan)
                        * (
                            args.prompt_seconds
                            + args.countdown_seconds
                            + args.record_seconds
                            + args.review_seconds
                        )
                        / 60,
                        1,
                    ),
                },
                indent=2,
            )
        )
        return
    result = record_session(args, plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
