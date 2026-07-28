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
from ml.nlu.data import Example, Span
from ml.nlu.inference import _normalise_slots
from ml.nlu.models import build_model
from ml.nlu.metrics import semantic_frame_metrics
from ml.nlu.schema import INTENTS, NLUResult, SLOT_LABELS
from ml.nlu.tokenizer import WordTokenizer
from ml.nlu.finetune import _restore_with_expanded_vocabulary
from ml.nlu.manager_train import _route_logits, _route_targets, _slot_consistency_loss
from training_workspace.build_dataset import APPLICATIONS, TARGETS, build
from training_workspace.nlu_search import confirmation_experiments, generate_phase_one
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


def test_slot_consistency_loss_penalises_intent_illegal_slots():
    targets = torch.zeros((2, 3), dtype=torch.long)
    intents = torch.tensor(
        [INTENTS.index("general_chat"), INTENTS.index("open_application")]
    )
    legal = torch.full((2, 3, len(SLOT_LABELS)), -4.0, requires_grad=True)
    illegal = torch.full((2, 3, len(SLOT_LABELS)), -4.0, requires_grad=True)
    with torch.no_grad():
        legal[0, :, SLOT_LABELS.index("O")] = 4.0
        legal[1, :, SLOT_LABELS.index("B-application")] = 4.0
        illegal[0, :, SLOT_LABELS.index("B-application")] = 4.0
        illegal[1, :, SLOT_LABELS.index("B-reminder_text")] = 4.0

    legal_loss = _slot_consistency_loss(legal, intents, targets)
    illegal_loss = _slot_consistency_loss(illegal, intents, targets)
    illegal_loss.backward()

    assert legal_loss < illegal_loss
    assert illegal.grad is not None
    assert torch.isfinite(illegal.grad).all()


def test_semantic_metrics_detect_exact_frames_and_slot_hallucination():
    reminder = "через 12 минут напомни позвонить"
    examples = [
        Example("открой paint", "open_application", (Span(7, 12, "application"),)),
        Example(
            reminder,
            "set_reminder",
            (
                Span(reminder.index("12"), reminder.index("12") + 2, "duration"),
                Span(
                    reminder.index("позвонить"),
                    reminder.index("позвонить") + len("позвонить"),
                    "reminder_text",
                ),
            ),
        ),
        Example("привет", "general_chat"),
    ]
    predictions = [
        NLUResult("open_application", 0.9, {"application": "Paint"}),
        NLUResult("set_reminder", 0.8, {"minutes": "12", "reminder_text": "написать"}),
        NLUResult("open_application", 0.7, {"application": "привет"}),
    ]

    metrics = semantic_frame_metrics(examples, predictions)

    assert metrics["semantic_frame_exact_match"] == pytest.approx(1 / 3)
    assert metrics["slot_hallucination_rate"] == 1.0
    assert metrics["per_slot"]["application"]["f1"] == pytest.approx(2 / 3)
    assert metrics["per_slot"]["minutes"]["f1"] == 1.0
    assert metrics["per_slot"]["reminder_text"]["f1"] == 0.0


def test_runtime_drops_slots_that_are_illegal_for_predicted_intent():
    assert _normalise_slots(
        "привет", "general_chat", {"application": "paint", "reminder_text": "привет"}
    ) == {}
    assert _normalise_slots(
        "открой paint",
        "open_application",
        {"reminder_text": "лишнее", "duration": "12"},
    ) == {"application": "paint"}


def test_two_stage_search_is_reproducible_and_seed_fair():
    search = {
        "trials": 6,
        "search_seed": 123,
        "phase_one_seed": 17,
        "confirmation_seeds": [17, 43, 101],
        "space": {"method": ["standard", "augmented", "curriculum"]},
    }

    first = generate_phase_one(search)
    second = generate_phase_one(search)
    confirmation = confirmation_experiments(first[:2], search)

    assert first == second
    assert {item["seed"] for item in first} == {17}
    assert {item["method"] for item in first} == {
        "standard", "augmented", "curriculum"
    }
    assert len(confirmation) == 6
    assert {item["seed"] for item in confirmation} == {17, 43, 101}


def test_workspace_uses_char_manager_experiments_and_strict_export_gates():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "training_workspace" / "config.yaml").read_text(encoding="utf-8")
    )
    experiments = config["experiments"]
    selection = config["selection"]
    search = config["search"]

    assert {experiment["method"] for experiment in experiments} == {
        "standard", "augmented", "curriculum"
    }
    assert all(experiment["trainer"] == "manager" for experiment in experiments)
    assert all(experiment["architecture"] == "char_cnn" for experiment in experiments)
    assert search["enabled"] is True
    assert search["trials"] >= 20
    assert len(search["confirmation_seeds"]) >= 5
    assert len({experiment["seed"] for experiment in experiments}) == 1
    assert selection["max_legacy_macro_f1_drop"] == 0.0
    assert selection["min_evaluation_holdout_macro_f1_improvement"] > 0.0
    assert selection["min_holdout_worst_recall"] >= 0.6
    assert selection["min_slot_entity_f1"] >= 0.78
    assert selection["max_slot_hallucination_rate"] <= 0.02
    assert selection["min_semantic_frame_exact_match"] >= 0.80
    assert selection["max_expected_calibration_error"] <= 0.04


def test_copy_script_requires_approved_hash():
    script = (
        Path(__file__).resolve().parents[1]
        / "training_workspace"
        / "COPY_BEST_TO_MODELS.ps1"
    ).read_text(encoding="utf-8")

    assert "approved.json" in script
    assert "Get-FileHash" in script
    assert "Refusing to copy stale or modified weights" in script
