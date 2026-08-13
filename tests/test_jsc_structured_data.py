"""Contracts for train-only Structured JSC family augmentation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from ml.jsc.data import load_jsc_jsonl
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.structured_features import serialize_structured_source
from training_workspace.build_jsc_structured_dataset import (
    build,
    build_augmentations,
)


def test_structured_augmentations_target_known_program_bottlenecks():
    rows = build_augmentations()
    steps = Counter(len(row.target.steps) for row in rows)
    categories = Counter(row.category for row in rows)

    assert len(rows) >= 1_300
    assert all(steps[count] >= 180 for count in range(2, 6))
    assert categories["multi_turn"] >= 100
    assert categories["hard_negative"] >= 150
    assert len({row.input_signature for row in rows}) == len(rows)


def test_structured_dataset_keeps_locked_splits_closed(tmp_path):
    source = Path("training_workspace/jsc_data")
    manifest = build(source, tmp_path)
    registry = build_project_schema_registry()
    train = load_jsc_jsonl(tmp_path / "train.jsonl", registry, expected_split="train")

    assert manifest["structured_augmentation"]["migration_suite_opened"] is False
    assert manifest["structured_augmentation"]["test_opened"] is False
    assert manifest["structured_augmentation"]["evaluation_holdout_opened"] is False
    assert not (tmp_path / "test.jsonl").exists()
    assert not (tmp_path / "evaluation_holdout.jsonl").exists()
    assert (tmp_path / "validation.jsonl").read_bytes() == (
        source / "validation.jsonl"
    ).read_bytes()
    assert max(len(serialize_structured_source(row)) + 2 for row in train) <= 416
