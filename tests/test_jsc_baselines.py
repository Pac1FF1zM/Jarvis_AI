"""Tests for the fair from-scratch JSC baseline protocol."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ml.jsc.baseline_metrics import evaluate_program_predictions
from ml.jsc.baseline_training import (
    TrainingConfig,
    _capture_rng,
    _load_resume,
    _restore_rng,
    inspect_training,
)
from ml.jsc.data import load_jsc_jsonl
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, ToolSchemaRegistry, dumps
from ml.jsc.models import ARCHITECTURES, BaselineConfig, JSCBaselineModel
from ml.jsc.sequence_data import (
    JSCSequenceDataset,
    SequenceLimits,
    make_collate_fn,
    serialize_source,
    tokenizer_training_texts,
)
from ml.jsc.tokenizer import JSCCharTokenizer
from tools.registry import ToolRegistry


DATA_DIR = Path("training_workspace/jsc_data")


@pytest.fixture(scope="module")
def schemas() -> ToolSchemaRegistry:
    registry = ToolRegistry()
    registry.discover("tools")
    return ToolSchemaRegistry.from_tool_registry(registry)


@pytest.fixture(scope="module")
def train_examples(schemas):
    return load_jsc_jsonl(DATA_DIR / "train.jsonl", schemas, expected_split="train")


def test_character_tokenizer_is_deterministic_reversible_and_never_truncates():
    first = JSCCharTokenizer.fit(("USER:привет", '{"act":"cancel"}'))
    second = JSCCharTokenizer.fit(('{"act":"cancel"}', "USER:привет"))
    encoded = first.encode("USER:привет", max_length=32)

    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert first.decode(encoded) == "USER:привет"
    assert first.decode(first.encode("USER:я", max_length=32)).endswith("�")
    with pytest.raises(ValueError, match="silent truncation is forbidden"):
        first.encode("слишком длинно", max_length=5)


def test_sequence_dataset_preserves_dialogue_state_and_builds_dynamic_batch(train_examples):
    tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(train_examples))
    dialogue = next(example for example in train_examples if example.history and example.state)
    plain = next(example for example in train_examples if not example.history)
    dataset = JSCSequenceDataset(
        [dialogue, plain],
        tokenizer,
        SequenceLimits(source=384, target=256),
    )

    batch = make_collate_fn(tokenizer.pad_id)([dataset[0], dataset[1]])

    assert "H_USER:" in serialize_source(dialogue)
    assert "H_JARVIS:" in serialize_source(dialogue)
    assert "STATE:" in serialize_source(dialogue)
    assert batch["source_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["decoder_input_ids"].shape
    assert batch["source_mask"].dtype == torch.bool
    assert (batch["labels"] == -100).any()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_each_baseline_backpropagates_and_decodes(architecture):
    config = BaselineConfig(
        architecture=architecture,
        vocab_size=32,
        num_acts=6,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        max_source_length=24,
        max_target_length=16,
        dropout=0.0,
    )
    model = JSCBaselineModel(config)
    source = torch.randint(4, 32, (2, 10))
    decoder = torch.randint(4, 32, (2, 7))

    token_logits, act_logits = model(
        source,
        source.ne(0),
        decoder,
        decoder.ne(0),
    )
    (token_logits.mean() + act_logits.mean()).backward()
    generated, generated_acts = model.greedy_decode(
        source,
        source.ne(0),
        bos_id=1,
        eos_id=2,
        max_length=8,
    )

    assert token_logits.shape == (2, 7, 32)
    assert act_logits.shape == (2, 6)
    assert generated.shape[0] == generated_acts.shape[0] == 2
    assert generated.shape[1] <= 8
    assert model.token_head.weight is model.token_embedding.weight
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_program_metrics_penalise_invalid_jal_and_false_execution(train_examples, schemas):
    ood = next(example for example in train_examples if example.target.act == DialogueAct.REJECT)
    exact = next(example for example in train_examples if example.target.act == DialogueAct.EXECUTE)
    predictions = (
        dumps(exact.target),
        dumps(ood.target),
        dumps(
            JALPlan(
                DialogueAct.EXECUTE,
                steps=(ToolCall("get_current_time"),),
            )
        ),
        "not-json",
    )

    metrics = evaluate_program_predictions([exact, ood, ood, ood], predictions, schemas)

    assert metrics["exact_jal_accuracy"] == pytest.approx(1 / 2)
    assert metrics["codec_valid_rate"] == pytest.approx(3 / 4)
    assert metrics["schema_valid_rate"] == pytest.approx(3 / 4)
    assert metrics["ood_recall"] == pytest.approx(1 / 3)
    assert metrics["false_execution_rate"] == pytest.approx(1 / 4)
    assert metrics["execution_precision"] == pytest.approx(1 / 2)


def test_check_only_reports_locked_holdout_and_rare_act_coverage(tmp_path):
    report = inspect_training(
        TrainingConfig(
            architecture="char_cnn",
            data_dir=str(DATA_DIR),
            output_dir=str(tmp_path / "unused"),
            device="cpu",
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            attention_heads=4,
            feedforward_dim=64,
            smoke=True,
        )
    )

    assert report["data"]["holdout_loaded"] is False
    assert report["data"]["acts"]["ask"] >= 30
    assert report["data"]["acts"]["cancel"] >= 20
    assert report["protocol"]["test_used_for_selection"] is False
    assert report["protocol"]["evaluation_holdout_loaded"] is False
    assert not (tmp_path / "unused").exists()


def test_training_state_restores_model_optimizer_and_rng(tmp_path):
    config = BaselineConfig(
        architecture="char_cnn",
        vocab_size=16,
        num_acts=6,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
        feedforward_dim=32,
        max_source_length=16,
        max_target_length=16,
        dropout=0.0,
    )
    model = JSCBaselineModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "kind": "jsc_baseline_training_state",
        "run_signature": "same-run",
        "epoch": 2,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_loss": 1.5,
        "best_epoch": 1,
        "stale_epochs": 1,
        "history": [{"epoch": 0}],
        "rng_state": _capture_rng(),
    }
    path = tmp_path / "latest.pt"
    torch.save(checkpoint, path)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)

    loaded = _load_resume(
        path,
        "same-run",
        model,
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
    )
    _restore_rng(loaded["rng_state"])

    assert loaded["epoch"] == 2
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())
    with pytest.raises(ValueError, match="does not match"):
        _load_resume(
            path,
            "another-run",
            model,
            optimizer,
            scheduler,
            scaler,
            torch.device("cpu"),
        )
