"""Build and audit a leak-resistant manifest from official Jester metadata."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .labels import JESTER_LABELS, LABEL_TO_INDEX


@dataclass(frozen=True)
class JesterRecord:
    clip_id: str
    label: str
    class_id: int
    split: str
    frame_dir: str
    num_frames: int


def read_labels(path: Path) -> tuple[str, ...]:
    labels = tuple(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if labels != JESTER_LABELS:
        raise ValueError("official labels.csv does not match the canonical Jester class order")
    return labels


def read_labeled_split(path: Path, split: str) -> list[JesterRecord]:
    rows: list[JesterRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(";", 1)
        if len(parts) != 2 or parts[1] not in LABEL_TO_INDEX:
            raise ValueError(f"invalid Jester metadata at {path}:{line_number}")
        clip_id, label = parts
        if clip_id in seen:
            raise ValueError(f"duplicate clip id {clip_id} in {path}")
        seen.add(clip_id)
        rows.append(JesterRecord(clip_id, label, LABEL_TO_INDEX[label], split, clip_id, 0))
    return rows


def build_manifest(metadata: Path, frames_root: Path, output: Path, *, verify_frames: bool = True) -> dict[str, object]:
    read_labels(metadata / "labels.csv")
    records = [
        *read_labeled_split(metadata / "train.csv", "train"),
        *read_labeled_split(metadata / "validation.csv", "val"),
        *read_labeled_split(metadata / "test-answers.csv", "test"),
    ]
    ids = [record.clip_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("clip IDs overlap between official Jester splits")
    missing: list[str] = []
    if verify_frames:
        verified: list[JesterRecord] = []
        for record in records:
            directory = frames_root / record.frame_dir
            count = sum(1 for _ in directory.glob("*.jpg")) if directory.is_dir() else 0
            if count < 2:
                missing.append(record.clip_id)
                if len(missing) >= 20:
                    break
            verified.append(replace(record, num_frames=count))
        if missing:
            raise FileNotFoundError(f"missing extracted frame directories, first IDs: {missing}")
        records = verified
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False, separators=(",", ":")) + "\n")
    split_counts = Counter(record.split for record in records)
    class_counts = {
        split: dict(Counter(record.label for record in records if record.split == split))
        for split in ("train", "val", "test")
    }
    report: dict[str, object] = {
        "status": "ready" if verify_frames else "metadata_ready",
        "records": len(records),
        "splits": dict(split_counts),
        "classes": len(JESTER_LABELS),
        "class_counts": class_counts,
        "manifest": str(output.resolve()),
        "frames_verified": verify_frames,
    }
    report_path = output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/raw/jester/metadata/jester_labels"))
    parser.add_argument("--frames-root", type=Path, default=Path("data/raw/jester/frames/20bn-jester-v1"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/jester/manifest.jsonl"))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.metadata, args.frames_root, args.output, verify_frames=not args.metadata_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
