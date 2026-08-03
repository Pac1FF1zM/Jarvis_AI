from __future__ import annotations

import csv
from collections import Counter

import pytest

from src.record_custom_dataset import MANIFEST_FIELDS, append_manifest, build_capture_plan, validate_session_id


@pytest.mark.parametrize(
    ("split", "expected_total", "gesture_repetitions", "d0x_repetitions"),
    (("train", 80, 5, 15), ("val", 31, 2, 5), ("test", 31, 2, 5)),
)
def test_default_capture_plans(split, expected_total, gesture_repetitions, d0x_repetitions):
    plan = build_capture_plan(split)
    counts = Counter(item.label for item in plan)
    assert len(plan) == expected_total
    assert counts["D0X"] == d0x_repetitions
    assert all(counts[label] == gesture_repetitions for label in counts if label != "D0X")
    assert len(counts) == 14


def test_capture_plan_override():
    plan = build_capture_plan("train", gesture_repetitions=1, non_gesture_repetitions=2)
    counts = Counter(item.label for item in plan)
    assert len(plan) == 15
    assert counts["D0X"] == 2
    assert all(counts[label] == 1 for label in counts if label != "D0X")


def test_session_id_rejects_paths():
    assert validate_session_id("session_20260804-01") == "session_20260804-01"
    with pytest.raises(ValueError):
        validate_session_id("../outside")


def test_append_manifest_writes_header_once(tmp_path):
    manifest = tmp_path / "manifest.csv"
    row = {
        "path": "train/session/B0A_01.mp4",
        "split": "train",
        "label": "B0A",
        "class_name": "Pointing with one finger",
        "session_id": "session",
        "take": 1,
        "frames": 90,
        "fps": 30.0,
        "duration_s": 3.0,
        "width": 1280,
        "height": 720,
        "recorded_at": "2026-08-04T00:00:00+05:00",
    }
    append_manifest(manifest, row)
    append_manifest(manifest, {**row, "take": 2, "path": "train/session/B0A_02.mp4"})

    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == MANIFEST_FIELDS
    assert len(rows) == 2
    assert rows[1]["take"] == "2"
