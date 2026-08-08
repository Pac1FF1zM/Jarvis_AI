"""Regression tests for the leak-resistant JSC Dataset v5 factory."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ml.jsc.data import SPLITS, load_jsc_jsonl, validate_jsc_splits
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, ToolSchemaRegistry, dumps
from tools.registry import ToolRegistry
from ml.jsc.project_registry import build_project_schema_registry
from training_workspace.build_jsc_dataset import (
    DATA_DIR,
    TARGETS,
    Candidate,
    _to_record,
    generate,
)


@pytest.fixture(scope="module")
def schemas() -> ToolSchemaRegistry:
    return build_project_schema_registry()


def test_committed_corpus_is_reproducible_and_matches_manifest(schemas):
    generated, fingerprint = generate()
    repeated, repeated_fingerprint = generate()

    assert repeated == generated
    assert repeated_fingerprint == fingerprint == schemas.schema_fingerprint

    manifest = json.loads((DATA_DIR / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 5
    assert manifest["external_sources"] is False
    assert manifest["synthetic_holdout"] is True
    assert manifest["split_policy"] == "no structural-family or exact-model-input overlap"
    assert manifest["tool_schema_sha256"] == fingerprint

    for split in SPLITS:
        path = DATA_DIR / f"{split}.jsonl"
        committed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert committed == generated[split]
        assert manifest["splits"][split]["examples"] == sum(TARGETS[split].values())
        assert manifest["splits"][split]["categories"] == TARGETS[split]
        assert manifest["splits"][split]["sha256"] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_all_splits_pass_jal_schema_and_leakage_audit(schemas):
    loaded = {
        split: load_jsc_jsonl(
            DATA_DIR / f"{split}.jsonl",
            schemas,
            expected_split=split,
        )
        for split in SPLITS
    }

    report = validate_jsc_splits(loaded)

    assert {split: report[split]["examples"] for split in SPLITS} == {
        split: sum(TARGETS[split].values()) for split in SPLITS
    }
    assert all(set(report[split]["categories"]) == set(TARGETS[split]) for split in SPLITS)
    assert report["train"]["acts"]["ask"] >= 30
    assert report["train"]["acts"]["confirm"] >= 8
    assert report["train"]["acts"]["cancel"] >= 20
    assert report["validation"]["acts"]["ask"] >= 8
    assert report["test"]["acts"]["ask"] >= 8
    for tool in (
        "browser_control",
        "file_control",
        "system_control",
        "window_control",
        "gesture_mode",
        "workspace_control",
    ):
        assert report["train"]["tools"][tool] >= 20


def test_structural_scenarios_represent_the_intended_learning_problems(schemas):
    examples = [
        example
        for split in SPLITS
        for example in load_jsc_jsonl(DATA_DIR / f"{split}.jsonl", schemas)
    ]
    by_category = {
        category: [example for example in examples if example.category == category]
        for category in TARGETS["train"]
    }

    assert all(
        example.target.act == DialogueAct.EXECUTE and len(example.target.steps) >= 2
        for example in by_category["compound"]
    )
    assert any(example.target.act == DialogueAct.ASK for example in by_category["multi_turn"])
    assert any(example.target.act == DialogueAct.CONFIRM for example in by_category["multi_turn"])
    assert any(example.history and example.state for example in by_category["multi_turn"])
    assert all(example.history and example.state for example in by_category["correction"])
    assert all(
        example.target.act == DialogueAct.DIALOGUE
        for example in by_category["hard_negative"]
    )
    assert all(example.target.act == DialogueAct.REJECT for example in by_category["ood"])
    assert all(
        example.target.act == DialogueAct.EXECUTE
        and example.text != example.metadata["clean_text"]
        for example in by_category["asr_noise"]
    )
    assert any(
        len(example.target.steps) == 4
        and tuple(step.tool for step in example.target.steps)
        == ("open_application", "gesture_mode", "set_reminder", "window_control")
        and "," not in example.text
        for example in by_category["compound"]
    )


def test_written_numbers_are_normalized_to_integer_slots_in_every_split(schemas):
    for split in SPLITS:
        examples = load_jsc_jsonl(DATA_DIR / f"{split}.jsonl", schemas)
        written = [
            example
            for example in examples
            if example.metadata.get("number_surface") == "words"
        ]
        assert len(written) >= 25
        assert any("четырнадцать" in example.text for example in written) or split != "evaluation_holdout"
        for example in written:
            numeric_values = [
                value
                for step in example.target.steps
                for value in step.arguments.values()
                if isinstance(value, int) and not isinstance(value, bool)
            ]
            assert numeric_values


def test_loader_rejects_noncanonical_jal_and_duplicate_model_input(tmp_path, schemas):
    target = dumps(JALPlan(DialogueAct.EXECUTE, steps=(ToolCall("get_current_time"),)))
    record = {
        "schema_version": 1,
        "scenario_id": "test.single.00001",
        "split": "test",
        "family_id": "test.single.time",
        "category": "single",
        "history": [],
        "text": "который час",
        "state_jal": None,
        "target_jal": target,
        "metadata": {"synthetic": True},
    }
    path = tmp_path / "invalid.jsonl"
    noncanonical = dict(record)
    noncanonical["target_jal"] = json.dumps(json.loads(target), ensure_ascii=False)
    path.write_text(json.dumps(noncanonical, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="not canonical"):
        load_jsc_jsonl(path, schemas)

    duplicate = dict(record, scenario_id="test.single.00002")
    path.write_text(
        "\n".join(json.dumps(value, ensure_ascii=False) for value in (record, duplicate)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate model input"):
        load_jsc_jsonl(path, schemas)


def test_loader_rejects_duplicate_json_keys_and_invalid_category_contract(tmp_path, schemas):
    target = dumps(JALPlan(DialogueAct.EXECUTE, steps=(ToolCall("get_current_time"),)))
    raw = (
        '{"schema_version":1,"schema_version":1,"scenario_id":"test.bad.1",'
        '"split":"test","family_id":"test.bad","category":"single",'
        '"history":[],"text":"время","state_jal":null,"target_jal":'
        + json.dumps(target, ensure_ascii=False)
        + ',"metadata":{}}'
    )
    path = tmp_path / "invalid.jsonl"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_jsc_jsonl(path, schemas)

    invalid_contract = {
        "schema_version": 1,
        "scenario_id": "test.bad.2",
        "split": "test",
        "family_id": "test.bad.contract",
        "category": "hard_negative",
        "history": [],
        "text": "расскажи про часы",
        "state_jal": None,
        "target_jal": target,
        "metadata": {},
    }
    path.write_text(json.dumps(invalid_contract, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must not execute"):
        load_jsc_jsonl(path, schemas)


def test_cross_split_input_leak_is_rejected(schemas):
    train = load_jsc_jsonl(DATA_DIR / "train.jsonl", schemas)[0]
    leaked = replace(
        train,
        scenario_id="validation.leak.00001",
        split="validation",
        family_id="validation.different.family",
    )

    with pytest.raises(ValueError, match="model input leakage"):
        validate_jsc_splits({"train": [train], "validation": [leaked]})


def test_family_id_masks_slot_values_instead_of_trusting_manual_labels():
    calculator = Candidate(
        "single",
        "manual.family.one",
        "открой калькулятор",
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("open_application", {"application": "calculator"}),),
        ),
    )
    notepad = Candidate(
        "single",
        "unrelated.manual.label",
        "открой системный блокнот",
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("open_application", {"application": "notepad"}),),
        ),
    )

    assert _to_record("train", calculator, 1)["family_id"] == _to_record(
        "validation", notepad, 1
    )["family_id"]
