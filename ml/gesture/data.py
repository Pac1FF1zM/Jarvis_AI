"""Leak-resistant MP4 clip loading and IPN annotation import."""
from __future__ import annotations

import hashlib
import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

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


def _find_annotation_file(annotations_dir: Path) -> Path | None:
    candidates = sorted(annotations_dir.rglob("*.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and {"labels", "database"} <= set(payload):
            return candidate
    return None


def _find_named_file(root: Path, name: str) -> Path | None:
    expected = name.casefold()
    return next(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.casefold() == expected
        ),
        None,
    )


def _ipn_subject_id(annotation_key: str) -> str:
    """Return the official subject identity encoded in an IPN video name."""
    basename = Path(annotation_key.replace("\\", "/").split("^")[0]).stem
    parts = basename.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot derive an IPN subject from {annotation_key!r}")
    return "_".join(parts[:2])


def _internal_validation_subjects(keys: Iterable[str]) -> set[str]:
    """Reserve 20% of official training subjects without touching official test."""
    subjects = sorted({_ipn_subject_id(key) for key in keys})
    if len(subjects) < 2:
        raise ValueError("Official IPN training annotations need at least two subjects")
    count = max(1, min(len(subjects) - 1, round(len(subjects) * 0.20)))
    ranked = sorted(
        subjects,
        key=lambda subject: hashlib.sha256(
            f"jarvis-ipn-validation-v1:{subject}".encode("utf-8")
        ).digest(),
    )
    return set(ranked[:count])


def _read_class_index(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = (
            [part.strip() for part in line.split(",")]
            if "," in line
            else line.split()
        )
        if [part.casefold() for part in parts] == ["id", "label"]:
            continue
        if len(parts) != 2:
            raise ValueError(f"Invalid class index at {path}:{line_number}: {line!r}")
        index, label = int(parts[0]), validate_label(parts[1])
        if index in labels:
            raise ValueError(f"Duplicate class index {index} in {path}")
        labels[index] = label
    if set(labels.values()) != set(IPN_LABELS):
        raise ValueError(
            f"IPN class index labels differ from expected labels: {sorted(labels.values())}"
        )
    return labels


def _read_official_rows(
    path: Path, labels: dict[int, str]
) -> list[tuple[str, str, int, int]]:
    rows: list[tuple[str, str, int, int]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = (
            [part.strip() for part in line.split(",")]
            if "," in line
            else line.split()
        )
        if parts[0].casefold() in {"video", "video_id", "video_name"}:
            continue
        if len(parts) == 6:
            key, label, _segment_id, start_frame, end_frame, _length = parts
            label = validate_label(label)
        elif len(parts) == 4:
            key, class_index, start_frame, end_frame = parts
            index = int(class_index)
            if index not in labels:
                raise ValueError(f"Unknown IPN class index {index} in {key!r}")
            label = labels[index]
        else:
            raise ValueError(f"Invalid IPN annotation at {path}:{line_number}: {line!r}")
        rows.append((key, label, int(start_frame), int(end_frame)))
    if not rows:
        raise ValueError(f"No IPN annotations found in {path}")
    return rows


def _import_official_text_annotations(
    video_root: Path,
    class_index_path: Path,
    train_path: Path,
    test_path: Path,
) -> list[GestureSegment]:
    labels = _read_class_index(class_index_path)
    train_rows = _read_official_rows(train_path, labels)
    test_rows = _read_official_rows(test_path, labels)
    validation_subjects = _internal_validation_subjects(row[0] for row in train_rows)
    records: list[GestureSegment] = []
    resolved_videos: dict[str, Path] = {}
    for source_split, rows in (("training", train_rows), ("test", test_rows)):
        for key, label, start_frame, end_frame in rows:
            split = "test"
            if source_split == "training":
                split = (
                    "validation"
                    if _ipn_subject_id(key) in validation_subjects
                    else "train"
                )
            video_id = key.split("^")[0]
            video = resolved_videos.get(video_id)
            if video is None:
                video = _resolve_video(video_root, video_id)
                resolved_videos[video_id] = video
            records.append(
                GestureSegment(
                    video=str(video.resolve()),
                    label=label,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    split=split,
                    video_id=video_id,
                )
            )
    return records


def _resolve_video(video_root: Path, annotation_key: str) -> Path:
    """Locate a downloaded AVI/MP4 without relying on an archive layout."""
    name = annotation_key.split("^")[0].replace("\\", "/")
    supported_extensions = (".avi", ".mp4")
    direct_candidates = [video_root / name]
    if not Path(name).suffix:
        direct_candidates.extend(video_root / f"{name}{ext}" for ext in supported_extensions)
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    source = Path(name)
    basenames = (
        (source.name,)
        if source.suffix
        else tuple(f"{source.name}{ext}" for ext in supported_extensions)
    )
    matches = [match for basename in basenames for match in video_root.rglob(basename)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"AVI/MP4 for annotation {annotation_key!r} was not found under {video_root}"
        )
    raise ValueError(f"Several video files match {annotation_key!r}: {matches[:3]}")


def import_ipn_segments(video_root: Path, annotations_dir: Path) -> list[GestureSegment]:
    """Read the official Google Drive text files or ActivityNet-style JSON."""
    class_index = _find_named_file(annotations_dir, "classIdx.txt")
    train_list = _find_named_file(annotations_dir, "Annot_TrainList.txt")
    test_list = _find_named_file(annotations_dir, "Annot_TestList.txt")
    if class_index and train_list and test_list:
        return _import_official_text_annotations(
            video_root, class_index, train_list, test_list
        )

    annotation_file = _find_annotation_file(annotations_dir)
    if annotation_file is None:
        raise FileNotFoundError(
            "Official IPN annotations were not found. Expected classIdx.txt plus "
            "Annot_TrainList.txt and Annot_TestList.txt, or an ActivityNet-style JSON."
        )
    payload = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    labels = tuple(payload["labels"])
    unexpected = set(labels) - set(IPN_LABELS)
    if unexpected:
        raise ValueError(f"Annotation contains unknown labels: {sorted(unexpected)}")
    records: list[GestureSegment] = []
    resolved_videos: dict[str, Path] = {}
    for key, value in payload["database"].items():
        annotation = value.get("annotations", {})
        label = annotation.get("label")
        if label is None:
            continue
        video_id = key.split("^")[0]
        video = resolved_videos.get(video_id)
        if video is None:
            video = _resolve_video(video_root, video_id)
            resolved_videos[video_id] = video
        records.append(
            GestureSegment(
                video=str(video.resolve()),
                label=validate_label(label),
                start_frame=int(annotation["start_frame"]),
                end_frame=int(annotation["end_frame"]),
                split=_split_name(str(value["subset"])),
                video_id=video_id,
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
    relative_records = []
    for item in records:
        data = asdict(item)
        data["video"] = str(Path(item.video).resolve())
        relative_records.append(data)
    by_split: dict[str, dict[str, int]] = {}
    videos_by_split: dict[str, set[str]] = {}
    for item in records:
        by_split.setdefault(item.split, {}).setdefault(item.label, 0)
        by_split[item.split][item.label] += 1
        videos_by_split.setdefault(item.split, set()).add(item.video_id)
    for left, left_videos in videos_by_split.items():
        for right, right_videos in videos_by_split.items():
            if left >= right:
                continue
            overlap = left_videos & right_videos
            if overlap:
                raise ValueError(
                    f"Split leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in relative_records)
        + "\n",
        encoding="utf-8",
    )
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


def _read_video_frame(
    capture: Any,
    frame_index: int,
    *,
    position_property: int,
    max_backtrack: int = 4,
) -> tuple[np.ndarray, int]:
    """Decode an annotated frame, tolerating only a small AVI tail mismatch."""
    lowest_candidate = max(1, frame_index - max_backtrack)
    for candidate in range(frame_index, lowest_candidate - 1, -1):
        capture.set(position_property, candidate - 1)
        ok, frame = capture.read()
        if ok:
            return frame, candidate
    raise RuntimeError(
        f"Could not decode frame {frame_index} or the previous "
        f"{frame_index - lowest_candidate} frame(s)"
    )


def _decode_video_frames_pyav(
    video: str,
    frame_indices: list[int],
    *,
    max_backtrack: int = 4,
) -> list[tuple[np.ndarray, int]]:
    """Decode frames sequentially with an independent FFmpeg binding.

    This is deliberately a fallback rather than the normal path. Some official
    IPN AVI seek indexes intermittently fail through OpenCV even though their
    underlying video stream remains decodable from the beginning.
    """
    try:
        import av
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV exhausted all AVI retries and PyAV is not installed; "
            "run `.venv-training\\Scripts\\python.exe -m pip install av`"
        ) from error

    wanted = {
        candidate
        for requested in frame_indices
        for candidate in range(requested, max(1, requested - max_backtrack) - 1, -1)
    }
    decoded: dict[int, np.ndarray] = {}
    last_requested = max(frame_indices)
    try:
        with av.open(video, mode="r") as container:
            for frame_number, frame in enumerate(container.decode(video=0), start=1):
                if frame_number in wanted:
                    decoded[frame_number] = frame.to_ndarray(format="bgr24")
                if frame_number >= last_requested:
                    break
    except Exception as error:  # noqa: BLE001 - external codec errors vary by format
        raise RuntimeError(f"PyAV could not decode {video}: {error}") from error

    result: list[tuple[np.ndarray, int]] = []
    for requested in frame_indices:
        lowest_candidate = max(1, requested - max_backtrack)
        decoded_index = next(
            (
                candidate
                for candidate in range(requested, lowest_candidate - 1, -1)
                if candidate in decoded
            ),
            None,
        )
        if decoded_index is None:
            raise RuntimeError(
                f"PyAV could not decode frame {requested} or the previous "
                f"{requested - lowest_candidate} frame(s) from {video}"
            )
        result.append((decoded[decoded_index], decoded_index))
    return result


def _decode_video_frames(
    video: str,
    frame_indices: list[int],
    *,
    capture_factory: Callable[[str], Any],
    position_property: int,
    max_attempts: int = 3,
) -> tuple[list[tuple[np.ndarray, int]], int]:
    """Decode a clip, reopening the container after a transient codec failure.

    OpenCV's AVI backend can occasionally fail a seek when several Windows
    DataLoader processes are decoding concurrently.  A failed attempt is
    discarded in full: mixing frames from a broken decoder with a new capture
    would make a silently corrupted training sample.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")

    last_error: RuntimeError | None = None
    for attempt in range(1, max_attempts + 1):
        capture = capture_factory(video)
        decoded: list[tuple[np.ndarray, int]] = []
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the video container")
            for frame_index in frame_indices:
                try:
                    decoded.append(
                        _read_video_frame(
                            capture,
                            frame_index,
                            position_property=position_property,
                        )
                    )
                except RuntimeError as error:
                    raise RuntimeError(f"frame {frame_index}: {error}") from error
            return decoded, attempt
        except RuntimeError as error:
            last_error = error
        finally:
            capture.release()

    try:
        return _decode_video_frames_pyav(video, frame_indices), 0
    except RuntimeError as fallback_error:
        raise RuntimeError(
            f"Could not decode {video} after {max_attempts} fresh OpenCV "
            f"capture attempt(s), then PyAV fallback failed: {fallback_error}; "
            f"last OpenCV error: {last_error}"
        ) from fallback_error


class VideoGestureDataset(Dataset[tuple[torch.Tensor, int]]):
    """Decode a short RGB clip at read time; no 800k-frame JPEG cache is created."""

    def __init__(
        self,
        records: list[GestureSegment],
        *,
        frames: int = 32,
        image_size: int = 112,
        training: bool = False,
        decode_retries: int = 2,
    ) -> None:
        if frames < 4 or image_size < 32:
            raise ValueError("frames must be >= 4 and image_size must be >= 32")
        if decode_retries < 0:
            raise ValueError("decode_retries cannot be negative")
        self.records = records
        self.frames = frames
        self.image_size = image_size
        self.training = training
        self.decode_retries = decode_retries
        self.label_to_index = {label: index for index, label in enumerate(IPN_LABELS)}
        self._warned_retry_videos: set[str] = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        try:
            import cv2
        except ModuleNotFoundError as error:  # pragma: no cover - depends on local setup
            raise RuntimeError("Install opencv-python-headless in the training environment") from error
        item = self.records[index]
        requested_indices = _sample_indices(
            item.start_frame, item.end_frame, self.frames, training=self.training
        )
        try:
            decoded_frames, attempts_used = _decode_video_frames(
                item.video,
                requested_indices,
                capture_factory=cv2.VideoCapture,
                position_property=cv2.CAP_PROP_POS_FRAMES,
                max_attempts=self.decode_retries + 1,
            )
        except RuntimeError as error:
            raise RuntimeError(f"Could not decode clip from {item.video}: {error}") from error
        if attempts_used != 1 and item.video not in self._warned_retry_videos:
            if attempts_used == 0:
                recovery = f"AVI decoder recovered with PyAV fallback for {item.video}"
            else:
                recovery = (
                    f"AVI decoder recovered after reopening {item.video} "
                    f"(attempt {attempts_used}/{self.decode_retries + 1})"
                )
            warnings.warn(recovery, RuntimeWarning, stacklevel=2)
            self._warned_retry_videos.add(item.video)

        frames: list[np.ndarray] = []
        for frame_index, (frame, decoded_index) in zip(
            requested_indices, decoded_frames, strict=True
        ):
            if decoded_index != frame_index:
                warnings.warn(
                    f"AVI boundary fallback for {item.video}: requested frame "
                    f"{frame_index}, decoded {decoded_index}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            if self.training and random.random() < 0.5:
                frame = np.ascontiguousarray(frame[:, ::-1])
            frames.append(frame)
        clip = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float().div_(255.0)
        return clip, self.label_to_index[item.label]
