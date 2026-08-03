"""Fault-aware isolated-gesture Dataset backed by audited JSONL manifests."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .audit import EXPECTED_LABELS
from .transforms import ClipTransform, ClipTransformConfig, resize_clip_for_cache


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClipRecord:
    video: str
    video_id: str
    subject_id: str
    label: str
    class_id: int
    start_frame: int
    end_frame: int
    split: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClipRecord":
        record = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        if record.label not in EXPECTED_LABELS:
            raise ValueError(f"Unknown label {record.label!r}")
        if EXPECTED_LABELS[record.class_id] != record.label:
            raise ValueError(f"class_id/label mismatch for {record.video_id}")
        if record.start_frame < 1 or record.end_frame < record.start_frame:
            raise ValueError(f"Invalid frame range for {record.video_id}")
        if record.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split {record.split!r}")
        return record


def load_manifest(path: Path, *, split: str | None = None) -> list[ClipRecord]:
    rows = [
        ClipRecord.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if split is not None:
        rows = [row for row in rows if row.split == split]
    if not rows:
        raise ValueError(f"No manifest records found in {path} for split={split!r}")
    return rows


def uniform_frame_indices(start_frame: int, end_frame: int, clip_len: int) -> list[int]:
    """Sample one-based inclusive indices over the entire annotated interval."""
    if start_frame < 1 or end_frame < start_frame:
        raise ValueError("Invalid inclusive frame range")
    if clip_len < 1:
        raise ValueError("clip_len must be positive")
    return np.rint(np.linspace(start_frame, end_frame, num=clip_len)).astype(int).tolist()


class ClipDecodeError(RuntimeError):
    """A clip remains unreadable after independent decoder attempts."""


def _decode_with_opencv(path: Path, indices: list[int], attempts: int) -> list[np.ndarray]:
    last_error = "unknown decoder error"
    for _attempt in range(attempts):
        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            if not capture.isOpened():
                last_error = "VideoCapture could not open the container"
                continue
            failed = None
            for one_based_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, one_based_index - 1)
                ok, frame = capture.read()
                if not ok:
                    failed = one_based_index
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if failed is None:
                return frames
            last_error = f"could not seek/decode one-based frame {failed}"
        finally:
            capture.release()
    raise ClipDecodeError(f"OpenCV failed after {attempts} attempt(s): {last_error}")


def _decode_with_pyav(path: Path, indices: list[int]) -> list[np.ndarray]:
    try:
        import av
    except ModuleNotFoundError as error:  # pragma: no cover - pinned dependency
        raise ClipDecodeError("PyAV fallback is unavailable") from error
    wanted = set(indices)
    decoded: dict[int, np.ndarray] = {}
    try:
        with av.open(str(path), mode="r") as container:
            for frame_number, frame in enumerate(container.decode(video=0), start=1):
                if frame_number in wanted:
                    decoded[frame_number] = frame.to_ndarray(format="rgb24")
                if frame_number >= max(indices):
                    break
    except Exception as error:  # noqa: BLE001 - codec exceptions vary by backend
        raise ClipDecodeError(f"PyAV failed for {path}: {error}") from error
    missing = sorted(wanted - set(decoded))
    if missing:
        raise ClipDecodeError(f"PyAV could not decode requested frame(s) {missing[:5]} from {path}")
    return [decoded[index] for index in indices]


def decode_clip(path: Path, indices: list[int], *, decode_retries: int) -> list[np.ndarray]:
    if decode_retries < 0:
        raise ValueError("decode_retries cannot be negative")
    try:
        return _decode_with_opencv(path, indices, decode_retries + 1)
    except ClipDecodeError as opencv_error:
        try:
            return _decode_with_pyav(path, indices)
        except ClipDecodeError as pyav_error:
            raise ClipDecodeError(
                f"Could not decode {path}; {opencv_error}; fallback: {pyav_error}"
            ) from pyav_error


class VideoGestureDataset(Dataset[tuple[torch.Tensor, int] | None]):
    """Return `(T,C,H,W), label`; unreadable clips are logged and skipped."""

    def __init__(
        self,
        records: list[ClipRecord],
        *,
        data_root: Path,
        clip_len: int = 16,
        frame_size: int = 112,
        training: bool = False,
        decode_retries: int = 2,
        max_decode_error_rate: float = 0.01,
        cache_dir: Path | None = None,
        cache_resize_size: int | None = None,
    ) -> None:
        if not 0 <= max_decode_error_rate <= 1:
            raise ValueError("max_decode_error_rate must be in [0,1]")
        self.records = records
        self.data_root = data_root.resolve()
        self.clip_len = clip_len
        self.decode_retries = decode_retries
        self.max_decode_error_rate = max_decode_error_rate
        self.cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self.cache_resize_size = cache_resize_size or max(frame_size, round(frame_size * 8 / 7))
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.transform = ClipTransform(
            ClipTransformConfig(frame_size=frame_size, resize_size=self.cache_resize_size),
            training=training,
        )
        self.decode_attempts = 0
        self.decode_failures = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | None:
        record = self.records[index]
        path = (self.data_root / record.video).resolve()
        indices = uniform_frame_indices(record.start_frame, record.end_frame, self.clip_len)
        self.decode_attempts += 1
        try:
            frames = self._load_or_decode(path, indices, record)
        except ClipDecodeError as error:
            self.decode_failures += 1
            LOGGER.error("Skipping unreadable IPN sample video=%s error=%s", path, error)
            return None
        return self.transform(frames), record.class_id

    def _cache_path(self, record: ClipRecord) -> Path | None:
        if self.cache_dir is None:
            return None
        identity = (
            f"ipn-tsn-cache-v1|{record.video}|{record.start_frame}|{record.end_frame}|"
            f"{self.clip_len}|{self.cache_resize_size}"
        )
        return self.cache_dir / f"{hashlib.sha256(identity.encode()).hexdigest()}.npy"

    def _load_or_decode(
        self,
        path: Path,
        indices: list[int],
        record: ClipRecord,
    ) -> list[np.ndarray]:
        cache_path = self._cache_path(record)
        if cache_path is not None and cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=False)
            expected_shape = (self.clip_len, self.cache_resize_size)
            if cached.dtype != np.uint8 or cached.shape[0] != expected_shape[0] or min(cached.shape[1:3]) != expected_shape[1]:
                raise ClipDecodeError(f"Invalid cached clip {cache_path}: {cached.shape} {cached.dtype}")
            return list(cached)
        frames = decode_clip(path, indices, decode_retries=self.decode_retries)
        resized = resize_clip_for_cache(frames, self.cache_resize_size)
        if cache_path is not None:
            temporary = cache_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, resized, allow_pickle=False)
            temporary.replace(cache_path)
        return list(resized)

    @property
    def decode_error_rate(self) -> float:
        return self.decode_failures / max(self.decode_attempts, 1)

    def assert_decode_health(self) -> None:
        if self.decode_error_rate > self.max_decode_error_rate:
            raise RuntimeError(
                f"Decode failure rate {self.decode_error_rate:.2%} exceeds "
                f"limit {self.max_decode_error_rate:.2%} "
                f"({self.decode_failures}/{self.decode_attempts})"
            )


def collate_skip_bad(
    batch: list[tuple[torch.Tensor, int] | None],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    clips, labels = zip(*valid, strict=True)
    return torch.stack(clips), torch.tensor(labels, dtype=torch.long)
