"""Build audited, subject-disjoint IPN Hand manifests from official files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_LABELS = (
    "D0X",
    "B0A",
    "B0B",
    "G01",
    "G02",
    "G03",
    "G04",
    "G05",
    "G06",
    "G07",
    "G08",
    "G09",
    "G10",
    "G11",
)
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
VALIDATION_FRACTION = 0.20
VALIDATION_SALT = "jarvis-ipn-validation-v1"


def subject_id(video_id: str) -> str:
    """Return the subject token used by IPN's four-videos-per-subject split."""
    parts = Path(video_id).stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot derive subject from IPN video id {video_id!r}")
    return "_".join(parts[:2])


def _validation_subjects(train_subjects: Iterable[str]) -> set[str]:
    subjects = sorted(set(train_subjects))
    if len(subjects) < 2:
        raise ValueError("At least two official training subjects are required")
    count = max(1, min(len(subjects) - 1, round(len(subjects) * VALIDATION_FRACTION)))
    ranked = sorted(
        subjects,
        key=lambda value: hashlib.sha256(f"{VALIDATION_SALT}:{value}".encode()).digest(),
    )
    return set(ranked[:count])


def _read_classes(path: Path) -> dict[int, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    classes = {int(row["id"]): row["label"] for row in rows}
    if tuple(classes[index] for index in sorted(classes)) != EXPECTED_LABELS:
        raise ValueError(f"Unexpected class mapping in {path}: {classes}")
    return classes


def _read_segments(path: Path, classes: dict[int, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 6:
                raise ValueError(f"{path}:{line_number}: expected 6 CSV fields, got {len(row)}")
            video_id, label, raw_class_id, raw_start, raw_end, raw_frames = row
            class_id = int(raw_class_id)
            start = int(raw_start)
            end = int(raw_end)
            frames = int(raw_frames)
            if classes.get(class_id) != label:
                raise ValueError(f"{path}:{line_number}: class id {class_id} != label {label}")
            if frames != end - start + 1:
                raise ValueError(f"{path}:{line_number}: inconsistent inclusive frame span")
            records.append(
                {
                    "video_id": video_id,
                    "subject_id": subject_id(video_id),
                    "label": label,
                    "class_id": class_id - 1,
                    "start_frame": start,
                    "end_frame": end,
                    "source_file": path.name,
                    "source_line": line_number,
                }
            )
    return records


def _video_index(video_root: Path) -> tuple[dict[str, Path], Counter[str]]:
    paths = sorted(
        path for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    index: dict[str, Path] = {}
    for path in paths:
        if path.stem in index:
            raise ValueError(f"Duplicate video stem {path.stem}: {index[path.stem]} and {path}")
        index[path.stem] = path.resolve()
    return index, Counter(path.suffix.lower() for path in paths)


def _load_corrections(audit_report: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(audit_report.read_text(encoding="utf-8"))
    rows = payload["video_validation"]["boundary_seek"]["safe_bound_overrides"]
    corrections: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row["video"])
        if video_id in corrections:
            raise ValueError(f"Duplicate frame-bound correction for {video_id}")
        corrections[video_id] = {
            "video_id": video_id,
            "old_end_frame": int(row["original_end"]),
            "new_end_frame": int(row["safe_end"]),
            "reason": (
                "Official t_end exceeds the last frame decodable by both OpenCV and "
                "sequential PyAV decoding"
            ),
        }
    return corrections


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def build_manifests(data_root: Path, output_dir: Path, audit_report: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    annotation_root = data_root / "annotations"
    video_root = data_root / "videos"
    classes = _read_classes(annotation_root / "classIdx.txt")
    official_train = _read_segments(annotation_root / "Annot_TrainList.txt", classes)
    official_test = _read_segments(annotation_root / "Annot_TestList.txt", classes)
    video_index, extensions = _video_index(video_root)
    corrections = _load_corrections(audit_report.resolve())

    official_train_subjects = {row["subject_id"] for row in official_train}
    official_test_subjects = {row["subject_id"] for row in official_test}
    if official_train_subjects & official_test_subjects:
        raise ValueError("Official train/test split is not subject-disjoint")
    validation_subjects = _validation_subjects(official_train_subjects)

    used_corrections: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for source_split, source_rows in (("train", official_train), ("test", official_test)):
        for source in source_rows:
            row = dict(source)
            video_id = row["video_id"]
            if video_id not in video_index:
                raise FileNotFoundError(f"No downloaded video for annotation {video_id}")
            row["video"] = video_index[video_id].relative_to(data_root).as_posix()
            row["source_split"] = source_split
            row["split"] = (
                "test"
                if source_split == "test"
                else "val" if row["subject_id"] in validation_subjects else "train"
            )
            correction = corrections.get(video_id)
            if correction is not None and row["end_frame"] == correction["old_end_frame"]:
                old_end = row["end_frame"]
                row["original_end_frame"] = old_end
                row["end_frame"] = correction["new_end_frame"]
                row["correction"] = correction["reason"]
                used_corrections.append(
                    {
                        "video": video_index[video_id].name,
                        "video_id": video_id,
                        "label": row["label"],
                        "source_file": row["source_file"],
                        "source_line": row["source_line"],
                        "old_end_frame": old_end,
                        "new_end_frame": row["end_frame"],
                        "reason": correction["reason"],
                    }
                )
            if row["end_frame"] < row["start_frame"]:
                raise ValueError(f"Correction invalidated frame span for {video_id}")
            records.append(row)

    if set(corrections) != {row["video_id"] for row in used_corrections}:
        raise ValueError("Not every audited frame-bound correction matched a manifest row")

    split_rows = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "val", "test")
    }
    split_subjects = {
        split: sorted({row["subject_id"] for row in rows})
        for split, rows in split_rows.items()
    }
    for left in split_subjects:
        for right in split_subjects:
            if left < right and set(split_subjects[left]) & set(split_subjects[right]):
                raise ValueError(f"Subject leakage between {left} and {right}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest_path = output_dir / "manifest.jsonl"
    _write_jsonl(manifest_path, records)

    report = {
        "schema_version": 1,
        "seed": 42,
        "validation_policy": {
            "kind": "strict_subject_disjoint",
            "source": "official training subjects only",
            "fraction": VALIDATION_FRACTION,
            "stable_hash_salt": VALIDATION_SALT,
        },
        "data_root": str(data_root),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "video_count": len(video_index),
        "video_extensions": dict(sorted(extensions.items())),
        "instance_count": len(records),
        "class_count": len(classes),
        "split_instance_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_video_counts": {
            split: len({row["video_id"] for row in rows}) for split, rows in split_rows.items()
        },
        "split_subject_counts": {split: len(values) for split, values in split_subjects.items()},
        "split_subjects": split_subjects,
        "subject_intersections": {
            "train_val": sorted(set(split_subjects["train"]) & set(split_subjects["val"])),
            "train_test": sorted(set(split_subjects["train"]) & set(split_subjects["test"])),
            "val_test": sorted(set(split_subjects["val"]) & set(split_subjects["test"])),
        },
        "class_counts": {
            split: dict(sorted(Counter(row["label"] for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "frame_bound_corrections": used_corrections,
        "matches_expectation": len(records) == 5649 and len(classes) == 14 and len(video_index) == 200,
    }
    report_path = output_dir / "split_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/ipn"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--audit-report", type=Path, default=Path("data/audit_report.json"))
    args = parser.parse_args()
    report = build_manifests(args.data_root, args.output_dir, args.audit_report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
