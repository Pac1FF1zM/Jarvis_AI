"""Protocol tests for the JSC data-first scaling curve."""
from __future__ import annotations

from pathlib import Path

from ml.jsc.data import load_jsc_jsonl
from ml.jsc.project_registry import build_project_schema_registry
from training_workspace.run_jsc_data_scaling import (
    _diagnose,
    nested_family_indices,
    prepare_scaling_datasets,
)


def test_family_subsets_are_nested_and_never_split_a_family():
    records = [
        {"category": "single", "family_id": "single.a"},
        {"category": "single", "family_id": "single.a"},
        {"category": "single", "family_id": "single.b"},
        {"category": "compound", "family_id": "compound.a"},
        {"category": "compound", "family_id": "compound.b"},
        {"category": "compound", "family_id": "compound.b"},
    ]

    subsets = nested_family_indices(records, (0.25, 0.5, 0.75, 1.0))

    assert set(subsets[0.25]) <= set(subsets[0.5]) <= set(subsets[0.75]) <= set(
        subsets[1.0]
    )
    assert set(subsets[1.0]) == set(range(len(records)))
    for indices in subsets.values():
        chosen = set(indices)
        for family in {row["family_id"] for row in records}:
            family_indices = {
                index for index, row in enumerate(records) if row["family_id"] == family
            }
            assert not chosen & family_indices or family_indices <= chosen


def test_prepared_scaling_data_keeps_validation_and_locked_splits_closed(tmp_path):
    source = Path("training_workspace/jsc_data")
    datasets = prepare_scaling_datasets(source, tmp_path, (0.25, 0.5, 1.0))
    registry = build_project_schema_registry()
    previous_families: set[str] = set()

    for fraction in (0.25, 0.5, 1.0):
        directory = Path(datasets[fraction]["directory"])
        train = load_jsc_jsonl(directory / "train.jsonl", registry, expected_split="train")
        families = {row.family_id for row in train}
        assert previous_families <= families
        assert not (directory / "test.jsonl").exists()
        assert not (directory / "evaluation_holdout.jsonl").exists()
        assert (directory / "validation.jsonl").read_bytes() == (
            source / "validation.jsonl"
        ).read_bytes()
        previous_families = families


def test_scaling_diagnosis_distinguishes_data_headroom_from_plateau():
    headroom = [
        {"migration_structured_exact_jal": {"mean": value, "std": 0.01}}
        for value in (0.10, 0.14, 0.18, 0.21)
    ]
    plateau = [
        {"migration_structured_exact_jal": {"mean": value, "std": 0.01}}
        for value in (0.10, 0.15, 0.181, 0.185)
    ]

    assert _diagnose(headroom)["verdict"] == "data_limited_with_remaining_headroom"
    assert _diagnose(plateau)["verdict"] == "plateau_architecture_or_objective_limited"
