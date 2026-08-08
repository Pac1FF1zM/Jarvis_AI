"""Jester JPEG clip dataset with clip-consistent transforms."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import ClipTransform, ClipTransformConfig
from .labels import JESTER_LABELS


@dataclass(frozen=True)
class ManifestRecord:
    clip_id: str
    label: str
    class_id: int
    split: str
    frame_dir: str
    num_frames: int

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ManifestRecord":
        record = cls(**{name: value[name] for name in cls.__dataclass_fields__})  # type: ignore[arg-type]
        if record.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split {record.split!r}")
        if record.class_id < 0 or record.class_id >= len(JESTER_LABELS):
            raise ValueError(f"invalid class_id for clip {record.clip_id}")
        if JESTER_LABELS[record.class_id] != record.label:
            raise ValueError(f"class_id/label mismatch for clip {record.clip_id}")
        if record.num_frames < 2:
            raise ValueError(f"clip {record.clip_id} has fewer than two frames")
        return record


def load_manifest(path: Path, split: str | None = None) -> list[ManifestRecord]:
    records = [
        ManifestRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if split is not None:
        records = [record for record in records if record.split == split]
    if not records:
        raise ValueError(f"no Jester records found for split={split!r}")
    return records


def temporal_indices(frame_count: int, clip_len: int, *, training: bool) -> list[int]:
    """Return one-based indices, with stratified jitter only during training."""
    if frame_count < 2 or clip_len < 1:
        raise ValueError("invalid frame_count/clip_len")
    edges = np.linspace(0, frame_count, clip_len + 1)
    indices: list[int] = []
    for index in range(clip_len):
        low = min(frame_count - 1, int(np.floor(edges[index])))
        high = min(frame_count - 1, max(low, int(np.ceil(edges[index + 1])) - 1))
        selected = random.randint(low, high) if training else (low + high) // 2
        indices.append(selected + 1)
    return indices


class JesterDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        records: list[ManifestRecord],
        *,
        frames_root: Path,
        clip_len: int,
        frame_size: int,
        resize_size: int,
        training: bool,
    ) -> None:
        self.records = records
        self.frames_root = frames_root.resolve()
        self.clip_len = clip_len
        self.training = training
        self.transform = ClipTransform(
            ClipTransformConfig(frame_size=frame_size, resize_size=resize_size),
            training=training,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        record = self.records[item]
        directory = (self.frames_root / record.frame_dir).resolve()
        if self.frames_root != directory and self.frames_root not in directory.parents:
            raise ValueError(f"unsafe frame directory for {record.clip_id}")
        frames: list[np.ndarray] = []
        for frame_index in temporal_indices(record.num_frames, self.clip_len, training=self.training):
            path = directory / f"{frame_index:05d}.jpg"
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"cannot decode Jester frame {path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.transform(frames), record.class_id
