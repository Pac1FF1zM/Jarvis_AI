"""Regression tests for the portable NLU fine-tuning workspace."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.nlu.custom_data import load_jsonl, validate_splits
from ml.nlu.models import build_model
from ml.nlu.tokenizer import WordTokenizer
from ml.nlu.finetune import _restore_with_expanded_vocabulary
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
    assert report["data"]["train_examples"] == 14
    assert report["data"]["validation_examples"] == 7
