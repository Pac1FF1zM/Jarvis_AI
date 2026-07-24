"""Regression tests for the portable NLU fine-tuning workspace."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from ml.nlu.custom_data import load_jsonl, validate_splits
from ml.nlu.models import build_model
from ml.nlu.tokenizer import WordTokenizer
from ml.nlu.finetune import _restore_with_expanded_vocabulary
from ml.nlu.manager_train import _route_logits, _route_targets
from training_workspace.build_dataset import APPLICATIONS, TARGETS, build
from training_workspace.run import run


def test_custom_jsonl_builds_slot_spans(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "открой калькулятор",
                "intent": "open_application",
                "slots": {"application": "калькулятор"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    example = load_jsonl(path, allow_empty=False)[0]
    assert example.text[example.spans[0].start:example.spans[0].end] == "калькулятор"
    assert example.spans[0].label == "application"


def test_custom_jsonl_rejects_unknown_intent(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"x","intent":"delete_system","slots":{}}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown intent"):
        load_jsonl(path)


def test_custom_train_validation_overlap_is_rejected(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"text":"привет","intent":"general_chat","slots":{}}', encoding="utf-8"
    )
    examples = load_jsonl(path)
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(examples, examples)


def test_tokenizer_extension_preserves_existing_ids():
    tokenizer = WordTokenizer.fit(["старое слово"], max_length=8)
    before = dict(tokenizer.stoi)
    added = tokenizer.extend(["старое новое слово"])
    assert added == 1
    assert all(tokenizer.stoi[token] == token_id for token, token_id in before.items())
    assert "новое" in tokenizer.stoi


def test_expanded_embedding_restores_old_rows_exactly():
    old = build_model(
        "word_bigru", vocab_size=4, num_intents=7, num_slots=7,
        pad_id=0, embedding_dim=4, hidden_dim=4,
    )
    with torch.no_grad():
        old.embedding.weight.copy_(torch.arange(16).reshape(4, 4))
    expanded = build_model(
        "word_bigru", vocab_size=6, num_intents=7, num_slots=7,
        pad_id=0, embedding_dim=4, hidden_dim=4,
    )
    _restore_with_expanded_vocabulary(expanded, old.state_dict())
    assert torch.equal(expanded.embedding.weight[:4], old.embedding.weight)
    assert expanded.embedding.weight.shape == (6, 4)


def test_workspace_default_config_passes_check_only():
    config = Path(__file__).resolve().parents[1] / "training_workspace" / "config.yaml"
    report = run(config, check_only=True)
    assert report["status"] == "configuration_ok"
    assert report["data"]["train_examples"] == 840
    assert report["data"]["validation_examples"] == 210
    assert set(report["data"]["train_intents"].values()) == {120}
    assert set(report["data"]["validation_intents"].values()) == {30}
    assert report["data"]["evaluation_holdout_examples"] == 105
    assert report["data"]["final_holdout_examples"] == 49


def test_generated_dataset_is_current_balanced_and_fully_disjoint():
    data_dir = Path(__file__).resolve().parents[1] / "training_workspace" / "data"
    filenames = {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "evaluation_holdout": "evaluation_holdout.jsonl",
    }
    generated = build()
    text_sets: dict[str, set[str]] = {}

    for split, filename in filenames.items():
        records = [
            json.loads(line)
            for line in (data_dir / filename).read_text(encoding="utf-8").splitlines()
        ]
        assert records == generated[split]
        assert len(records) == TARGETS[split] * 7
        assert set(Counter(record["intent"] for record in records).values()) == {
            TARGETS[split]
        }
        text_sets[split] = {record["text"].casefold() for record in records}

    assert text_sets["train"].isdisjoint(text_sets["validation"])
    assert text_sets["train"].isdisjoint(text_sets["evaluation_holdout"])
    assert text_sets["validation"].isdisjoint(text_sets["evaluation_holdout"])


def test_train_open_application_examples_cover_allowlist_evenly():
    data_path = (
        Path(__file__).resolve().parents[1]
        / "training_workspace"
        / "data"
        / "train.jsonl"
    )
    records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines()]
    alias_to_app = {
        alias: application
        for application, aliases in APPLICATIONS.items()
        for alias in aliases
    }
    counts = Counter(
        alias_to_app[record["slots"]["application"]]
        for record in records
        if record["intent"] == "open_application"
    )

    assert set(counts) == set(APPLICATIONS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_dataset_manifest_hashes_match_canonical_files():
    data_dir = Path(__file__).resolve().parents[1] / "training_workspace" / "data"
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))

    assert manifest["external_sources"] is False
    for metadata in manifest["splits"].values():
        content = (data_dir / metadata["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]


def test_manager_route_loss_groups_intents_and_backpropagates():
    intent_logits = torch.randn(7, 7, requires_grad=True)
    intent_targets = torch.arange(7)

    route_logits = _route_logits(intent_logits)
    route_targets = _route_targets(intent_targets)
    loss = torch.nn.functional.cross_entropy(route_logits, route_targets)
    loss.backward()

    assert route_logits.shape == (7, 4)
    assert route_targets.tolist() == [0, 0, 0, 0, 1, 2, 3]
    assert intent_logits.grad is not None
    assert torch.isfinite(intent_logits.grad).all()


def test_workspace_uses_char_manager_experiments_and_strict_export_gates():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "training_workspace" / "config.yaml").read_text(encoding="utf-8")
    )
    experiments = config["experiments"]
    selection = config["selection"]

    assert {experiment["method"] for experiment in experiments} == {
        "standard", "augmented", "curriculum"
    }
    assert all(experiment["trainer"] == "manager" for experiment in experiments)
    assert all(experiment["architecture"] == "char_cnn" for experiment in experiments)
    assert selection["max_legacy_macro_f1_drop"] == 0.0
    assert selection["min_evaluation_holdout_macro_f1_improvement"] > 0.0
    assert selection["min_holdout_worst_recall"] >= 0.6


def test_copy_script_requires_approved_hash():
    script = (
        Path(__file__).resolve().parents[1]
        / "training_workspace"
        / "COPY_BEST_TO_MODELS.ps1"
    ).read_text(encoding="utf-8")

    assert "approved.json" in script
    assert "Get-FileHash" in script
    assert "Refusing to copy stale or modified weights" in script
