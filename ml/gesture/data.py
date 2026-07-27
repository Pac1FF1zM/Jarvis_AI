"""Leak-resistant MP4 clip loading and IPN annotation import."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import IPN_LABELS, validate_label


@dataclass(frozen=True)
class GestureSegment:
    """One labelled temporal segment inside an RGB MP4 video.

    Frame numbers are one-based and inclusive, matching the original IPN
    annotations.  The manifest stores resolved paths deliberately: raw videos
    may live outside the Git repository and are never copied into it.
    """

    video: str
    label: str
    start_frame: int
    end_frame: int
    split: str
    video_id: str

    def __post_init__(self) -> None:
        validate_label(self.label)
        if self.start_frame < 1 or self.end_frame < self.start_frame:
            raise ValueError(f"Invalid inclusive frame range: {self.start_frame}..{self.end_frame}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split {self.split!r}")


def _split_name(value: str) -> str:
    value = value.strip().lower()
    aliases = {
        "training": "train",
        "train": "train",
        "validation": "validation",
        "valid": "validation",
        "val": "validation",
        "testing": "test",
        "test": "test",
    }
    if value not in aliases:
        raise ValueError(f"Unknown annotation subset {value!r}")
    return aliases[value]


def _find_annotation_file(annotations_dir: Path) -> Path:
    candidates = sorted(annotations_dir.rglob("*.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and {"labels", "database"} <= set(payload):
            return candidate
    raise FileNotFoundError(
        "IPN annotation JSON was not found. Extract the official Annotations archive "
        "and point --annotations-dir to its extracted folder."
    )


def _resolve_video(video_root: Path, annotation_key: str) -> Path:
    """Locate a downloaded MP4 without relying on a particular archive layout."""
    name = annotation_key.split("^")[0].replace("\\", "/")
    direct_candidates = [video_root / name, video_root / f"{name}.mp4"]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    basename = Path(name).name
    if not basename.lower().endswith(".mp4"):
        basename += ".mp4"
    matches = list(video_root.rglob(basename))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"MP4 for annotation {annotation_key!r} was not found under {video_root}")
    raise ValueError(f"Several MP4 files match {annotation_key!r}: {matches[:3]}")


def import_ipn_segments(video_root: Path, annotations_dir: Path) -> list[GestureSegment]:
    """Read the official IPN ActivityNet-style JSON without importing its code."""
    annotation_file = _find_annotation_file(annotations_dir)
    payload = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    labels = tuple(payload["labels"])
    unexpected = set(labels) - set(IPN_LABELS)
    if unexpected:
        raise ValueError(f"Annotation contains unknown labels: {sorted(unexpected)}")
    records: list[GestureSegment] = []
    for key, value in payload["database"].items():
        annotation = value.get("annotations", {})
        label = annotation.get("label")
        if label is None:
            continue
        video = _resolve_video(video_root, key)
        records.append(
            GestureSegment(
                video=str(video.resolve()),
                label=validate_label(label),
                start_frame=int(annotation["start_frame"]),
                end_frame=int(annotation["end_frame"]),
                split=_split_name(str(value["subset"])),
                video_id=key.split("^")[0],
            )
        )
    if not records:
        raise ValueError(f"No labelled segments were found in {annotation_file}")
    return records


def write_manifest(records: Iterable[GestureSegment], path: Path) -> dict[str, Any]:
    """Persist a portable JSONL manifest and return its audit summary."""
    records = list(records)
    if not records:
        raise ValueError("Cannot write an empty gesture manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    relative_records = []
    for item in records:
        data = asdict(item)
        data["video"] = str(Path(item.video).resolve())
        relative_records.append(data)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in relative_records)
        + "\n",
        encoding="utf-8",
    )
    by_split: dict[str, dict[str, int]] = {}
    videos_by_split: dict[str, set[str]] = {}
    for item in records:
        by_split.setdefault(item.split, {}).setdefault(item.label, 0)
        by_split[item.split][item.label] += 1
        videos_by_split.setdefault(item.split, set()).add(item.video_id)
    overlap = set.intersection(*videos_by_split.values()) if len(videos_by_split) > 1 else set()
    if overlap:
        raise ValueError(f"Split leakage: videos occur in more than one split: {sorted(overlap)[:5]}")
    return {
        "manifest": str(path.resolve()),
        "segments": len(records),
        "labels": list(IPN_LABELS),
        "split_counts": {key: sum(value.values()) for key, value in by_split.items()},
        "class_counts": by_split,
        "videos_per_split": {key: len(value) for key, value in videos_by_split.items()},
    }


def load_manifest(path: Path) -> list[GestureSegment]:
    records = [
        GestureSegment(**json.loads(line))
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Gesture manifest is empty: {path}")
    return records


def audit_segments(records: Iterable[GestureSegment]) -> dict[str, Any]:
    records = list(records)
    videos: dict[str, set[str]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for item in records:
        if not Path(item.video).is_file():
            raise FileNotFoundError(f"Manifest references a missing video: {item.video}")
        videos.setdefault(item.split, set()).add(item.video_id)
        class_counts.setdefault(item.split, {}).setdefault(item.label, 0)
        class_counts[item.split][item.label] += 1
    for left, left_videos in videos.items():
        for right, right_videos in videos.items():
            if left >= right:
                continue
            shared = left_videos & right_videos
            if shared:
                raise ValueError(f"Video leakage between {left} and {right}: {sorted(shared)[:5]}")
    return {
        "segments": len(records),
        "videos": {name: len(value) for name, value in videos.items()},
        "class_counts": class_counts,
    }


def _sample_indices(start: int, end: int, frames: int, *, training: bool) -> list[int]:
    span = end - start + 1
    if span <= 0:
        raise ValueError("segment must contain at least one frame")
    if training and span > frames:
        max_offset = span - frames
        offset = random.randint(0, max_offset)
        return list(range(start + offset, start + offset + frames))
    positions = np.linspace(start, end, num=frames)
    return [int(round(value)) for value in positions]


class VideoGestureDataset(Dataset[tuple[torch.Tensor, int]]):
    """Decode a short RGB clip at read time; no 800k-frame JPEG cache is created."""

    def __init__(
        self,
        records: list[GestureSegment],
        *,
        frames: int = 32,
        image_size: int = 112,
        training: bool = False,
    ) -> None:
        if frames < 4 or image_size < 32:
            raise ValueError("frames must be >= 4 and image_size must be >= 32")
        self.records = records
        self.frames = frames
        self.image_size = image_size
        self.training = training
        self.label_to_index = {label: index for index, label in enumerate(IPN_LABELS)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        try:
            import cv2
        except ModuleNotFoundError as error:  # pragma: no cover - depends on local setup
            raise RuntimeError("Install opencv-python-headless in the training environment") from error
        item = self.records[index]
        capture = cv2.VideoCapture(item.video)
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video {item.video}")
        frames: list[np.ndarray] = []
        try:
            for frame_index in _sample_indices(
                item.start_frame, item.end_frame, self.frames, training=self.training
            ):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_index} from {item.video}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                if self.training and random.random() < 0.5:
                    frame = np.ascontiguousarray(frame[:, ::-1])
                frames.append(frame)
        finally:
            capture.release()
        clip = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div_(255.0)
        return clip, self.label_to_index[item.label]
