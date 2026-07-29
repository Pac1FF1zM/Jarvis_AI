"""Regression tests for the portable NLU fine-tuning workspace."""
from __future__ import annotations

import copy
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
from ml.nlu.manager_train import (
    _learning_rate_multiplier,
    _manager_validation_score,
    _no_slot_false_positive_loss,
    _route_logits,
    _route_targets,
    _slot_consistency_loss,
    _update_ema,
)
from training_workspace.build_dataset import (
    APPLICATIONS,
    HARD_BOUNDARIES,
    TARGETS,
    build,
)
from training_workspace.nlu_search import confirmation_experiments, generate_phase_one
from training_workspace.run import (
    _benchmark_with_stable_latency,
    _development_score,
    _rank_phase_results,
    run,
)


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
    assert report["data"]["train_examples"] == 1120
    assert report["data"]["validation_examples"] == 210
    assert set(report["data"]["train_intents"].values()) == {160}
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


def test_hard_boundary_examples_are_pinned_to_every_generated_split():
    generated = build()
    for split, by_intent in HARD_BOUNDARIES.items():
        actual = {
            (record["intent"], record["text"].casefold())
            for record in generated[split]
        }
        for intent, texts in by_intent.items():
            assert all((intent, text.casefold()) in actual for text in texts)


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


def test_no_slot_loss_targets_the_worst_false_positive_token():
    targets = torch.zeros((2, 3), dtype=torch.long)
    intents = torch.tensor(
        [INTENTS.index("general_chat"), INTENTS.index("open_application")]
    )
    clean = torch.full((2, 3, len(SLOT_LABELS)), -4.0, requires_grad=True)
    hallucinated = torch.full(
        (2, 3, len(SLOT_LABELS)), -4.0, requires_grad=True
    )
    with torch.no_grad():
        clean[:, :, SLOT_LABELS.index("O")] = 4.0
        hallucinated[:, :, SLOT_LABELS.index("O")] = 4.0
        hallucinated[0, 1, SLOT_LABELS.index("O")] = -4.0
        hallucinated[0, 1, SLOT_LABELS.index("B-application")] = 4.0

    clean_loss = _no_slot_false_positive_loss(clean, intents, targets)
    hallucinated_loss = _no_slot_false_positive_loss(
        hallucinated, intents, targets
    )
    hallucinated_loss.backward()

    assert clean_loss < hallucinated_loss
    assert hallucinated.grad is not None
    assert torch.isfinite(hallucinated.grad).all()


def test_manager_score_penalises_a_weakest_intent_at_equal_macro_f1():
    common = {
        "intent_macro_f1": 0.95,
        "slot_entity_f1": 0.82,
        "semantic_frame_exact_match": 0.80,
        "slot_hallucination_rate": 0.01,
    }
    stable = {**common, "worst_intent_recall": 0.90}
    unstable = {**common, "worst_intent_recall": 0.60}

    assert _manager_validation_score(stable, stable) > _manager_validation_score(
        unstable, stable
    )


def test_warmup_cosine_schedule_and_ema_are_deterministic():
    multipliers = [
        _learning_rate_multiplier(
            epoch, epochs=10, warmup_epochs=2, min_lr_ratio=0.1
        )
        for epoch in range(1, 11)
    ]
    assert multipliers[:2] == [0.5, 1.0]
    assert multipliers[2] == pytest.approx(1.0)
    assert multipliers[-1] == pytest.approx(0.1)
    assert all(left >= right for left, right in zip(multipliers[2:], multipliers[3:]))

    model = torch.nn.Linear(2, 1)
    ema_model = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(1.0)
        model.bias.fill_(1.0)
        ema_model.weight.zero_()
        ema_model.bias.zero_()
    _update_ema(ema_model, model, 0.5)
    assert torch.equal(ema_model.weight, torch.full_like(ema_model.weight, 0.5))
    assert torch.equal(ema_model.bias, torch.full_like(ema_model.bias, 0.5))


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


def test_runtime_keeps_neural_slots_when_no_supported_grammar_matches():
    assert _normalise_slots(
        "сигнал восемь потом позвонить родителям",
        "set_reminder",
        {"duration": "8", "reminder_text": "позвонить родителям"},
    ) == {"minutes": "8", "reminder_text": "позвонить родителям"}
    assert _normalise_slots(
        "активируй paint",
        "open_application",
        {"application": "paint"},
    ) == {"application": "paint"}


def test_supported_grammar_repairs_incomplete_neural_spans():
    assert _normalise_slots(
        "напомни мне через 8 минут что пора позвонить родителям",
        "set_reminder",
        {"duration": "8", "reminder_text": "родителям"},
    ) == {"minutes": "8", "reminder_text": "позвонить родителям"}
    assert _normalise_slots(
        "открой для меня paint",
        "open_application",
        {"application": "для меня paint"},
    ) == {"application": "paint"}


def test_slot_fallback_covers_canonical_validation_and_holdout_templates():
    data_dir = Path(__file__).resolve().parents[1] / "training_workspace" / "data"
    for filename in ("validation.jsonl", "evaluation_holdout.jsonl"):
        for example in load_jsonl(data_dir / filename, allow_empty=False):
            if example.intent not in {"open_application", "set_reminder"}:
                continue
            expected = {
                ("minutes" if span.label == "duration" else span.label): (
                    example.text[span.start : span.end]
                )
                for span in example.spans
            }
            assert _normalise_slots(example.text, example.intent, {}) == expected


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
    assert all(
        {
            "warmup_epochs",
            "min_lr_ratio",
            "ema_decay",
            "slot_o_weight",
            "no_slot_loss_weight",
        }
        <= item.keys()
        for item in first
    )


def test_development_score_distinguishes_raw_neural_slot_quality():
    benchmarks = {
        "custom_validation": {
            "intent_macro_f1": 0.95,
            "worst_intent_recall": 0.85,
            "semantic_frame_exact_match": 0.85,
            "end_to_end_command_accuracy": 0.85,
            "slot_hallucination_rate": 0.0,
            "expected_calibration_error": 0.02,
        },
        "legacy_regression": {
            "intent_macro_f1": 0.95,
            "worst_intent_recall": 0.85,
        },
    }
    weak = {
        "custom_validation_slot_entity_f1": 0.50,
        "custom_validation_semantic_frame_exact_match": 0.40,
        "custom_validation_end_to_end_command_accuracy": 0.40,
        "custom_validation_slot_hallucination_rate": 0.20,
    }
    strong = {
        "custom_validation_slot_entity_f1": 0.82,
        "custom_validation_semantic_frame_exact_match": 0.70,
        "custom_validation_end_to_end_command_accuracy": 0.72,
        "custom_validation_slot_hallucination_rate": 0.10,
    }

    assert _development_score(benchmarks, strong) > _development_score(
        benchmarks, weak
    )


def test_development_score_penalises_raw_hallucination_above_gate():
    benchmarks = {
        "custom_validation": {
            "intent_macro_f1": 0.96,
            "worst_intent_recall": 0.90,
            "semantic_frame_exact_match": 0.85,
            "end_to_end_command_accuracy": 0.85,
            "slot_hallucination_rate": 0.0,
            "expected_calibration_error": 0.02,
        },
        "legacy_regression": {
            "intent_macro_f1": 0.96,
            "worst_intent_recall": 0.90,
        },
    }
    common = {
        "custom_validation_slot_entity_f1": 0.80,
        "custom_validation_semantic_frame_exact_match": 0.65,
        "custom_validation_end_to_end_command_accuracy": 0.70,
    }

    assert _development_score(
        benchmarks,
        {**common, "custom_validation_slot_hallucination_rate": 0.19},
    ) > _development_score(
        benchmarks,
        {**common, "custom_validation_slot_hallucination_rate": 0.21},
    )


def test_phase_ranking_prefers_hard_gate_feasibility_over_scalar_score():
    def result(name: str, *, score: float, raw_slot: float) -> dict:
        return {
            "name": name,
            "selection_score": score,
            "metrics": {
                "custom_validation_slot_entity_f1": raw_slot,
                "custom_validation_semantic_frame_exact_match": 0.65,
                "custom_validation_slot_hallucination_rate": 0.10,
            },
            "benchmarks": {
                "custom_validation": {"latency_ms_p95": 1.5}
            },
        }

    selected, failures = _rank_phase_results(
        [
            result("high_score_but_invalid", score=0.90, raw_slot=0.76),
            result("feasible", score=0.82, raw_slot=0.79),
        ],
        {
            "min_raw_slot_entity_f1": 0.78,
            "min_raw_semantic_frame_exact_match": 0.60,
            "max_raw_slot_hallucination_rate": 0.20,
            "max_p95_latency_ms": 2.0,
        },
        top_k=1,
    )

    assert selected[0]["name"] == "feasible"
    assert failures["feasible"] == []
    assert failures["high_score_but_invalid"] == ["raw_slot_entity_f1"]


def test_stable_latency_uses_median_p95(monkeypatch):
    measured = iter(
        [
            {"latency_ms_median": 1.0, "latency_ms_p95": 4.0},
            {"latency_ms_median": 1.1, "latency_ms_p95": 1.8},
            {"latency_ms_median": 1.2, "latency_ms_p95": 2.2},
        ]
    )
    monkeypatch.setattr(
        "training_workspace.run.benchmark", lambda *_args, **_kwargs: next(measured)
    )

    result = _benchmark_with_stable_latency(
        Path("unused.pt"), [], latency_trials=3, device="cpu", warmup=0, repetitions=1
    )

    assert result["latency_ms_median"] == 1.1
    assert result["latency_ms_p95"] == 2.2
    assert result["latency_trials"]["p95_ms"] == [4.0, 1.8, 2.2]


def test_reevaluation_reuses_checkpoints_without_training(tmp_path: Path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    base.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    metrics = {
        "custom_validation_slot_entity_f1": 0.80,
        "custom_validation_semantic_frame_exact_match": 0.65,
        "custom_validation_end_to_end_command_accuracy": 0.70,
        "custom_validation_slot_hallucination_rate": 0.10,
    }
    source_report = tmp_path / "source_report.json"
    source_report.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "name": "saved_seed_17",
                        "config": {"name": "saved_seed_17", "seed": 17},
                        "checkpoint": str(candidate),
                        "metrics": metrics,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config = {
        "base_checkpoint": str(base),
        "training_device": "cpu",
        "require_cuda": False,
        "runs_dir": str(tmp_path / "runs"),
        "export_dir": str(tmp_path / "export"),
        "data": {
            "train": str(root / "training_workspace/data/train.jsonl"),
            "validation": str(root / "training_workspace/data/validation.jsonl"),
            "evaluation_holdout": str(
                root / "training_workspace/data/evaluation_holdout.jsonl"
            ),
            "final_holdout": str(root / "ml/nlu/holdout_v2.jsonl"),
        },
        "search": {"enabled": False},
        "experiments": [{"name": "unused", "trainer": "manager"}],
        "benchmark": {"device": "cpu", "warmup": 0, "repetitions": 1},
        "selection": {"min_custom_macro_f1_improvement": 1.0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    measured = {
        "intent_accuracy": 0.9,
        "intent_macro_f1": 0.9,
        "worst_intent_recall": 0.8,
        "slot_entity_f1": 0.8,
        "slot_hallucination_rate": 0.0,
        "semantic_frame_exact_match": 0.8,
        "end_to_end_command_accuracy": 0.8,
        "expected_calibration_error": 0.02,
        "latency_ms_median": 0.4,
        "latency_ms_p95": 0.5,
    }
    monkeypatch.setattr("training_workspace.run.benchmark", lambda *a, **k: measured)

    def training_must_not_run(*_args, **_kwargs):
        raise AssertionError("reevaluation must not start manager training")

    monkeypatch.setattr("training_workspace.run.subprocess.run", training_must_not_run)

    report = run(config_path, reevaluate_run=source_report)

    assert report["reevaluation"]["training_performed"] is False
    assert report["reevaluation"]["checkpoints_reused"] == 1
    assert report["selection"]["status"] == "rejected"


def test_selective_confirmation_skips_phase_training_and_uses_feasible_candidate(
    tmp_path: Path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base.pt"
    base.write_bytes(b"baseline")
    metrics = {
        "custom_validation_slot_entity_f1": 0.80,
        "custom_validation_semantic_frame_exact_match": 0.65,
        "custom_validation_end_to_end_command_accuracy": 0.70,
        "custom_validation_slot_hallucination_rate": 0.10,
    }
    measured = {
        "intent_accuracy": 0.9,
        "intent_macro_f1": 0.9,
        "worst_intent_recall": 0.8,
        "slot_entity_f1": 0.8,
        "slot_hallucination_rate": 0.0,
        "semantic_frame_exact_match": 0.8,
        "end_to_end_command_accuracy": 0.8,
        "expected_calibration_error": 0.02,
        "latency_ms_median": 0.4,
        "latency_ms_p95": 0.5,
    }

    def phase_result(name: str, raw_slot: float, score: float) -> dict:
        return {
            "name": name,
            "config": {
                "name": name,
                "trainer": "manager",
                "architecture": "char_cnn",
                "method": "augmented",
                "seed": 43,
            },
            "metrics": {**metrics, "custom_validation_slot_entity_f1": raw_slot},
            "benchmarks": {"custom_validation": measured},
            "selection_score": score,
        }

    source_report = tmp_path / "source_report.json"
    source_report.write_text(
        json.dumps(
            {
                "search": {
                    "phase_one_seed": 43,
                    "phase_one": [
                        phase_result("invalid_high_score", 0.75, 0.95),
                        phase_result("feasible", 0.80, 0.85),
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    config = {
        "base_checkpoint": str(base),
        "training_device": "cpu",
        "require_cuda": False,
        "runs_dir": str(tmp_path / "runs"),
        "export_dir": str(tmp_path / "export"),
        "data": {
            "train": str(root / "training_workspace/data/train.jsonl"),
            "validation": str(root / "training_workspace/data/validation.jsonl"),
            "evaluation_holdout": str(
                root / "training_workspace/data/evaluation_holdout.jsonl"
            ),
            "final_holdout": str(root / "ml/nlu/holdout_v2.jsonl"),
        },
        "search": {
            "enabled": True,
            "trials": 2,
            "top_k": 1,
            "confirmation_seeds": [17, 43, 101],
            "confirmation_epochs": 1,
            "confirmation_patience": 1,
        },
        "benchmark": {
            "device": "cpu",
            "warmup": 0,
            "repetitions": 1,
            "latency_trials": 1,
        },
        "selection": {
            "min_custom_macro_f1_improvement": 1.0,
            "min_raw_slot_entity_f1": 0.78,
            "min_raw_semantic_frame_exact_match": 0.60,
            "max_raw_slot_hallucination_rate": 0.20,
            "max_p95_latency_ms": 2.0,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("training_workspace.run.benchmark", lambda *a, **k: measured)
    trained: list[str] = []

    def fake_training(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"checkpoint")
        output.with_suffix(".metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        trained.append(output.stem)

    monkeypatch.setattr("training_workspace.run.subprocess.run", fake_training)

    report = run(config_path, confirm_from_run=source_report)

    assert report["search"]["method"] == "selective_confirmation_feasible_first"
    assert report["search"]["finalists"] == ["feasible"]
    assert len(trained) == 3
    assert {item["candidate"] for item in report["experiments"]} == {"feasible"}


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
    assert search["space"]["method"] == ["augmented"]
    assert search["phase_one_seed"] == 43
    assert "ema_decay" in search["space"]
    assert "slot_o_weight" in search["space"]
    assert "no_slot_loss_weight" in search["space"]
    assert len(search["confirmation_seeds"]) >= 5
    assert len({experiment["seed"] for experiment in experiments}) == 1
    assert selection["max_legacy_macro_f1_drop"] == 0.0
    assert selection["min_evaluation_holdout_macro_f1_improvement"] > 0.0
    assert selection["min_holdout_worst_recall"] >= 0.6
    assert selection["min_slot_entity_f1"] >= 0.78
    assert selection["max_slot_hallucination_rate"] <= 0.02
    assert selection["min_semantic_frame_exact_match"] >= 0.80
    assert selection["max_expected_calibration_error"] <= 0.04
    assert selection["min_raw_slot_entity_f1"] >= 0.78
    assert selection["min_raw_semantic_frame_exact_match"] >= 0.60
    assert selection["max_raw_slot_hallucination_rate"] <= 0.20


def test_copy_script_requires_approved_hash():
    script = (
        Path(__file__).resolve().parents[1]
        / "training_workspace"
        / "COPY_BEST_TO_MODELS.ps1"
    ).read_text(encoding="utf-8")

    assert "approved.json" in script
    assert "Get-FileHash" in script
    assert "Refusing to copy stale or modified weights" in script


def test_start_script_exposes_checkpoint_reevaluation_mode():
    script = (
        Path(__file__).resolve().parents[1]
        / "training_workspace"
        / "START_TRAINING.ps1"
    ).read_text(encoding="utf-8")

    assert "ReevaluateRun" in script
    assert "--reevaluate-run" in script
    assert "ConfirmFromRun" in script
    assert "--confirm-from-run" in script
