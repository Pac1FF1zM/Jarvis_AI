"""Tests for the fair from-scratch JSC baseline protocol."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ml.jsc.baseline_metrics import evaluate_program_predictions
from ml.jsc.constrained_decoding import constrain_jal_predictions
from ml.jsc.structured_decoding import assemble_structured_execution
from ml.jsc.structured_labels import (
    build_parameter_labels,
    decode_parameter_logits,
    parameter_label,
)
from ml.jsc.span_labels import (
    SPAN_ARGUMENTS,
    decode_span_arguments,
    find_argument_span,
    span_tool_arguments,
)
from ml.jsc.baseline_training import (
    TrainingConfig,
    _capture_rng,
    _load_resume,
    _restore_rng,
    _token_loss,
    evaluate_locked_test,
    inspect_training,
)
from ml.jsc.data import load_jsc_jsonl
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, ToolSchemaRegistry, dumps, loads
from ml.jsc.models import ARCHITECTURES, BaselineConfig, JSCBaselineModel
from ml.jsc.project_registry import build_project_schema_registry
from core.russian_numbers import extract_russian_cardinals
from ml.jsc.sequence_data import (
    JSCSequenceDataset,
    SequenceLimits,
    make_collate_fn,
    normalize_utterance,
    serialize_source,
    tokenizer_training_texts,
)
from ml.jsc.tokenizer import JSCCharTokenizer
from tools.registry import ToolRegistry
from training_workspace import run_jsc_baselines as baseline_runner


DATA_DIR = Path("training_workspace/jsc_data")


@pytest.fixture(scope="module")
def schemas() -> ToolSchemaRegistry:
    return build_project_schema_registry()


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


def test_utterance_normalization_is_split_independent_and_stt_friendly():
    assert normalize_utterance("  Ёлка № 7; ТАЙМЕР-НАПОМИНАНИЕ  ") == (
        "елка номер 7, таймер напоминание"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("через четырнадцать минут", (14,)),
        ("через двадцать минут", (20,)),
        ("номер пятьдесят пять", (55,)),
        ("через сто двадцать пять секунд", (125,)),
        ("один два потом", (1, 2)),
        ("через 17 минут", (17,)),
    ],
)
def test_russian_cardinal_extraction(text, expected):
    assert extract_russian_cardinals(text) == expected


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
    numeric = next(
        example
        for example in train_examples
        if example.metadata.get("number_surface") == "words"
    )
    assert "USER_NUM:" in serialize_source(numeric)
    assert batch["source_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["decoder_input_ids"].shape
    assert batch["source_mask"].dtype == torch.bool
    assert batch["execution_allowed"].tolist() == [
        int(dialogue.target.act == DialogueAct.EXECUTE),
        int(plain.target.act == DialogueAct.EXECUTE),
    ]
    assert (batch["labels"] == -100).any()

    tool_names = sorted(
        {step.tool for example in train_examples for step in example.target.steps}
    )
    structured = make_collate_fn(
        tokenizer.pad_id,
        {name: index + 1 for index, name in enumerate(tool_names)},
    )([dataset[0], dataset[1]])
    assert structured["step_count"].tolist() == [len(dialogue.target.steps), len(plain.target.steps)]
    assert structured["tool_ids"].shape == (2, 8)


def test_train_tokenizer_has_zero_unknown_characters_on_every_frozen_split(
    train_examples, schemas
):
    tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(train_examples))
    for split in ("validation", "test", "evaluation_holdout"):
        examples = load_jsc_jsonl(DATA_DIR / f"{split}.jsonl", schemas)
        unknown = {
            character
            for example in examples
            for text in (serialize_source(example), dumps(example.target))
            for character in text
            if character not in tokenizer.stoi
        }
        assert unknown == set(), f"{split} contains unknown characters: {unknown}"


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


def test_copy_mechanism_returns_normalized_log_probabilities_and_backpropagates():
    config = BaselineConfig(
        architecture="tiny_transformer",
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
        copy_mechanism=True,
    )
    model = JSCBaselineModel(config)
    source = torch.randint(4, 32, (2, 10))
    decoder = torch.randint(4, 32, (2, 7))
    labels = torch.randint(4, 32, (2, 7))
    labels[1, -2:] = -100

    scores, act_logits = model(source, source.ne(0), decoder, decoder.ne(0))
    loss = _token_loss(
        scores,
        labels,
        log_probabilities=model.token_scores_are_log_probabilities,
        label_smoothing=0.05,
    ) + act_logits.mean()
    loss.backward()

    assert scores.shape == (2, 7, 32)
    assert torch.allclose(scores.exp().sum(-1), torch.ones(2, 7), atol=1e-5)
    assert model.copy_query is not None and model.copy_query.weight.grad is not None
    assert model.copy_gate is not None and model.copy_gate.weight.grad is not None


def test_structured_heads_learn_count_and_ordered_tool_labels():
    config = BaselineConfig(
        architecture="tiny_transformer",
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
        num_tools=11,
    )
    model = JSCBaselineModel(config)
    source = torch.randint(4, 32, (2, 10))
    decoder = torch.randint(4, 32, (2, 7))
    token_scores, acts, counts, tools = model.forward_structured(
        source, source.ne(0), decoder, decoder.ne(0)
    )
    (token_scores.mean() + acts.mean() + counts.mean() + tools.mean()).backward()

    assert counts.shape == (2, 9)
    assert tools.shape == (2, 8, 11)
    assert model.step_count_head is not None
    assert model.tool_sequence_head is not None
    assert any(parameter.grad is not None for parameter in model.step_count_head.parameters())


def test_schema_conditioned_parameter_head_shapes_and_backpropagates():
    config = BaselineConfig(
        architecture="tiny_transformer",
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
        num_tools=13,
        num_parameter_labels=58,
    )
    model = JSCBaselineModel(config)
    source = torch.randint(4, 32, (2, 10))
    decoder = torch.randint(4, 32, (2, 7))
    outputs = model.forward_schema_conditioned(
        source, source.ne(0), decoder, decoder.ne(0)
    )
    token_scores, acts, counts, tools, parameters = outputs
    sum(value.mean() for value in outputs).backward()

    assert token_scores.shape == (2, 7, 32)
    assert acts.shape == (2, 6)
    assert counts.shape == (2, 9)
    assert tools.shape == (2, 8, 13)
    assert parameters.shape == (2, 8, 58)
    assert model.parameter_head is not None
    assert any(parameter.grad is not None for parameter in model.parameter_head.parameters())


def test_parameter_label_space_and_dynamic_batch_are_schema_conditioned(
    schemas, train_examples
):
    labels = build_parameter_labels(schemas)
    assert parameter_label("window_control", "action", "close") in labels
    assert parameter_label("gesture_mode", "action", "enable") in labels
    assert parameter_label("file_control", "confirmed", True) in labels

    example = next(
        item
        for item in train_examples
        if any(step.tool == "gesture_mode" for step in item.target.steps)
    )
    tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(train_examples))
    row = JSCSequenceDataset(
        [example], tokenizer, SequenceLimits(384, 384)
    )[0]
    batch = make_collate_fn(
        tokenizer.pad_id,
        {name: index + 1 for index, name in enumerate(schemas.tool_names)},
        {name: index for index, name in enumerate(labels)},
    )([row])

    assert batch["parameter_targets"].shape == (1, 8, len(labels))
    assert batch["parameter_mask"].dtype == torch.bool
    assert int(batch["parameter_targets"].sum()) >= 1
    assert int(batch["parameter_mask"].sum()) >= int(
        batch["parameter_targets"].sum()
    )


def test_parameter_decoder_selects_one_value_per_tool_argument(schemas):
    labels = build_parameter_labels(schemas)
    scores = [0.01] * len(labels)
    scores[labels.index(parameter_label("window_control", "action", "close"))] = 0.91
    scores[labels.index(parameter_label("window_control", "action", "restore"))] = 0.60

    assert decode_parameter_logits(scores, labels, "window_control") == {
        "action": "close"
    }


def test_span_labels_align_free_form_values_and_dynamic_batch(
    schemas, train_examples
):
    example = next(
        item
        for item in train_examples
        if any(
            step.tool == "set_reminder"
            and isinstance(step.arguments.get("message"), str)
            and step.arguments["message"].casefold() in serialize_source(item)
            for step in item.target.steps
        )
    )
    tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(train_examples))
    row = JSCSequenceDataset([example], tokenizer, SequenceLimits(384, 384))[0]
    call = next(step for step in example.target.steps if step.tool == "set_reminder")
    start, end = find_argument_span(str(row["source_text"]), call, "message")
    batch = make_collate_fn(
        tokenizer.pad_id,
        {name: index + 1 for index, name in enumerate(schemas.tool_names)},
        {name: index for index, name in enumerate(build_parameter_labels(schemas))},
        SPAN_ARGUMENTS,
        span_tool_arguments(schemas),
    )([row])
    message_index = SPAN_ARGUMENTS.index("message")
    step_index = list(example.target.steps).index(call)

    assert str(row["source_text"])[start - 1 : end] == call.arguments["message"]
    assert batch["span_start_targets"][0, step_index, message_index] == start
    assert batch["span_end_targets"][0, step_index, message_index] == end
    assert batch["span_mask"][0, step_index, message_index]


def test_full_semantic_span_heads_shape_and_backpropagate():
    config = BaselineConfig(
        architecture="tiny_transformer",
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
        num_tools=13,
        num_parameter_labels=58,
        num_span_slots=len(SPAN_ARGUMENTS),
    )
    model = JSCBaselineModel(config)
    source = torch.randint(4, 32, (2, 10))
    decoder = torch.randint(4, 32, (2, 7))
    outputs = model.forward_full_semantic(
        source, source.ne(0), decoder, decoder.ne(0)
    )
    start_logits, end_logits = outputs[-2:]
    sum(value.mean() for value in outputs).backward()

    assert start_logits.shape == (2, 8, len(SPAN_ARGUMENTS), 10)
    assert end_logits.shape == start_logits.shape
    assert model.span_start_query is not None
    assert model.span_start_query.weight.grad is not None


def test_semantic_pooling_and_execution_verifier_shape_and_backpropagate():
    config = BaselineConfig(
        architecture="tiny_transformer",
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
        num_tools=13,
        num_parameter_labels=58,
        num_span_slots=len(SPAN_ARGUMENTS),
        semantic_pooling=True,
        execution_verifier=True,
    )
    model = JSCBaselineModel(config)
    source = torch.randint(4, 32, (2, 10))
    decoder = torch.randint(4, 32, (2, 7))
    outputs = model.forward_verified_semantic(
        source, source.ne(0), decoder, decoder.ne(0)
    )
    verifier_logits = outputs[-1]
    sum(value.mean() for value in outputs).backward()

    assert verifier_logits.shape == (2, 2)
    assert model.semantic_attention is not None
    assert model.semantic_attention.weight.grad is not None
    assert model.execution_verifier_head is not None
    assert any(
        parameter.grad is not None
        for parameter in model.execution_verifier_head.parameters()
    )


def test_span_decoder_extracts_and_canonicalizes_application(schemas):
    source = "USER:пожалуйста открой дискорд"
    length = len(source) + 2
    starts = torch.zeros(8, len(SPAN_ARGUMENTS), length)
    ends = torch.zeros_like(starts)
    start = source.index("дискорд") + 1
    end = start + len("дискорд") - 1
    slot = SPAN_ARGUMENTS.index("application")
    starts[0, slot, start] = 1.0
    ends[0, slot, end] = 1.0

    decoded = decode_span_arguments(
        starts.tolist(), ends.tolist(), source, ("open_application",), schemas
    )

    assert decoded == ({"application": "discord"},)


def test_span_decoder_ignores_structured_none_tool(schemas):
    empty = torch.zeros(1, len(SPAN_ARGUMENTS), 4).tolist()

    assert decode_span_arguments(empty, empty, "ab", ("<none>",), schemas) == ({},)


def test_constrained_decoder_prefers_span_over_wrong_raw_free_text(schemas):
    source = "USER:пожалуйста лололошка"
    raw = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(
                ToolCall("browser_control", {"action": "search", "query": "ошибка"}),
            ),
        )
    )
    tool_labels = ("<none>", *schemas.tool_names)
    count_logits = torch.full((1, 9), -5.0)
    count_logits[0, 1] = 5.0
    tool_logits = torch.full((1, 8, len(tool_labels)), -5.0)
    tool_logits[:, :, 0] = 5.0
    tool_logits[0, 0, tool_labels.index("browser_control")] = 10.0
    length = len(source) + 2
    starts = torch.full((1, 8, len(SPAN_ARGUMENTS), length), -10.0)
    ends = torch.full_like(starts, -10.0)
    start = source.index("лололошка") + 1
    end = start + len("лололошка") - 1
    slot = SPAN_ARGUMENTS.index("query")
    starts[0, 0, slot, start] = 10.0
    ends[0, 0, slot, end] = 10.0

    result = constrain_jal_predictions(
        [raw],
        torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        schemas,
        utterances=["пожалуйста лололошка"],
        step_count_logits=count_logits,
        tool_logits=tool_logits,
        tool_labels=tool_labels,
        span_start_logits=starts,
        span_end_logits=ends,
        span_slots=SPAN_ARGUMENTS,
        span_sources=[source],
    )

    assert loads(result.predictions[0]).steps[0].arguments["query"] == "лололошка"
    assert result.decisions == {"accepted_structured": 1}


def test_execution_verifier_is_an_independent_fail_closed_barrier(schemas):
    prediction = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("open_application", {"application": "paint"}),),
        )
    )
    act_logits = torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    rejected = constrain_jal_predictions(
        [prediction],
        act_logits,
        schemas,
        execution_verifier_logits=torch.tensor([[12.0, 0.0]]),
    )
    accepted = constrain_jal_predictions(
        [prediction],
        act_logits,
        schemas,
        execution_verifier_logits=torch.tensor([[0.0, 12.0]]),
    )

    assert loads(rejected.predictions[0]) == JALPlan(
        DialogueAct.REJECT, reason="low_confidence"
    )
    assert rejected.decisions == {"execution_verifier_rejected": 1}
    assert loads(accepted.predictions[0]).act == DialogueAct.EXECUTE
    assert accepted.decisions == {"accepted": 1}


def test_verified_complete_route_can_correct_neural_tool_disagreement(schemas):
    wrong = dumps(JALPlan(DialogueAct.EXECUTE, steps=(ToolCall("get_current_time"),)))
    tool_labels = ("<none>", *schemas.tool_names)
    count_logits = torch.full((1, 9), -5.0)
    count_logits[0, 1] = 5.0
    tool_logits = torch.full((1, 8, len(tool_labels)), -5.0)
    tool_logits[0, 0, tool_labels.index("get_current_time")] = 10.0

    result = constrain_jal_predictions(
        [wrong],
        torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        schemas,
        utterances=["открой paint"],
        step_count_logits=count_logits,
        tool_logits=tool_logits,
        tool_labels=tool_labels,
        execution_verifier_logits=torch.tensor([[0.0, 12.0]]),
    )

    assert loads(result.predictions[0]) == JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "paint"}),),
    )
    assert result.decisions == {"accepted_verified_explicit_route": 1}


def test_constrained_decoder_uses_parameter_head_before_wrong_raw_enum(schemas):
    raw = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(
                ToolCall(
                    "window_control", {"action": "restore", "window": "discord"}
                ),
            ),
        )
    )
    tool_labels = ("<none>", *schemas.tool_names)
    parameter_labels = build_parameter_labels(schemas)
    count_logits = torch.full((1, 9), -5.0)
    count_logits[0, 1] = 5.0
    tool_logits = torch.full((1, 8, len(tool_labels)), -5.0)
    tool_logits[:, :, 0] = 5.0
    tool_logits[0, 0, tool_labels.index("window_control")] = 10.0
    parameter_logits = torch.full((1, 8, len(parameter_labels)), -10.0)
    parameter_logits[
        0,
        0,
        parameter_labels.index(
            parameter_label("window_control", "action", "close")
        ),
    ] = 10.0

    result = constrain_jal_predictions(
        [raw],
        torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        schemas,
        utterances=["убери окно discord"],
        step_count_logits=count_logits,
        tool_logits=tool_logits,
        tool_labels=tool_labels,
        parameter_logits=parameter_logits,
        parameter_labels=parameter_labels,
    )

    plan = loads(result.predictions[0])
    assert plan.steps[0].arguments == {"action": "close", "window": "discord"}
    assert result.decisions == {"accepted_structured": 1}


def test_parameter_head_cannot_authorize_execution_without_independent_evidence(
    schemas,
):
    assert assemble_structured_execution(
        "совершенно неизвестная просьба",
        ("system_control",),
        schemas,
        structured_arguments=({"action": "volume_up"},),
    ) is None


def test_constrained_decoder_is_canonical_schema_valid_and_fail_closed(schemas):
    valid_execution = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("open_application", {"application": "paint"}),),
        )
    )
    # DialogueAct order: execute, ask, confirm, cancel, reject, dialogue.
    logits = torch.tensor(
        [
            [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 5.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    result = constrain_jal_predictions(
        [valid_execution, "not json", valid_execution],
        logits,
        schemas,
        execution_threshold=0.75,
    )

    plans = [loads(value) for value in result.predictions]
    for plan in plans:
        schemas.validate(plan)
        assert dumps(plan) in result.predictions
    assert plans[0].act == DialogueAct.EXECUTE
    assert plans[1].act == DialogueAct.REJECT
    assert plans[2].act == DialogueAct.REJECT
    assert result.decisions == {
        "accepted": 1,
        "invalid_rejected": 1,
        "low_confidence_execution_rejected": 1,
    }


def test_constrained_decoder_grounds_written_number_in_numeric_tool_slot(schemas):
    prediction = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(
                ToolCall(
                    "set_reminder",
                    {"minutes": 5, "message": "позвонить родителям"},
                ),
            ),
        )
    )
    result = constrain_jal_predictions(
        [prediction],
        torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        schemas,
        utterances=["напомни через четырнадцать минут позвонить родителям"],
    )

    grounded = loads(result.predictions[0])
    assert grounded.steps[0].arguments["minutes"] == 14
    assert result.decisions == {"accepted_numeric_grounded": 1}


def test_constrained_decoder_requires_independent_tool_sequence_agreement(schemas):
    prediction = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("open_application", {"application": "paint"}),),
        )
    )
    tool_labels = ("<none>", *schemas.tool_names)
    wrong_tool_id = tool_labels.index("get_current_time")
    count_logits = torch.full((1, 9), -5.0)
    count_logits[0, 1] = 5.0
    tool_logits = torch.full((1, 8, len(tool_labels)), -5.0)
    tool_logits[:, :, 0] = 5.0
    tool_logits[0, 0, wrong_tool_id] = 10.0

    result = constrain_jal_predictions(
        [prediction],
        torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        schemas,
        step_count_logits=count_logits,
        tool_logits=tool_logits,
        tool_labels=tool_labels,
    )

    assert loads(result.predictions[0]).act == DialogueAct.REJECT
    assert result.decisions == {"tool_disagreement_rejected": 1}


def test_structured_assembler_builds_safe_four_step_jal(schemas):
    plan = assemble_structured_execution(
        "открой браузер запусти жестовый режим "
        "напомни через четырнадцать минут о встрече закрой дискорд",
        ("open_application", "gesture_mode", "set_reminder", "window_control"),
        schemas,
    )

    assert plan == JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall("open_application", {"application": "browser"}),
            ToolCall("gesture_mode", {"action": "enable"}),
            ToolCall("set_reminder", {"minutes": 14, "message": "встрече"}),
            ToolCall("window_control", {"action": "close", "window": "discord"}),
        ),
    )
    schemas.validate(plan)


def test_structured_assembler_rejects_explicit_tool_disagreement(schemas):
    assert assemble_structured_execution(
        "закрой дискорд", ("open_application",), schemas
    ) is None


def test_constrained_decoder_blocks_destructive_ood_even_when_model_is_confident(schemas):
    hallucination = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("window_control", {"action": "restore", "window": "диск"}),),
        )
    )
    labels = ("<none>", *schemas.tool_names)
    count_logits = torch.full((1, 9), -5.0)
    count_logits[0, 1] = 5.0
    tool_logits = torch.full((1, 8, len(labels)), -5.0)
    tool_logits[:, :, 0] = 5.0
    tool_logits[0, 0, labels.index("window_control")] = 10.0

    result = constrain_jal_predictions(
        [hallucination],
        torch.tensor([[12.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        schemas,
        utterances=["очисти весь диск"],
        step_count_logits=count_logits,
        tool_logits=tool_logits,
        tool_labels=labels,
    )

    assert loads(result.predictions[0]) == JALPlan(
        DialogueAct.REJECT, reason="unsupported_tool"
    )
    assert result.decisions == {"unsafe_utterance_rejected": 1}


def test_small_jal_model_can_overfit_and_greedy_decode_two_examples(train_examples):
    torch.manual_seed(5)
    examples = [
        next(example for example in train_examples if example.target.act.value == act)
        for act in ("cancel", "reject")
    ]
    tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(examples))
    dataset = JSCSequenceDataset(examples, tokenizer, SequenceLimits(384, 256))
    batch = make_collate_fn(tokenizer.pad_id)([dataset[0], dataset[1]])
    model = JSCBaselineModel(
        BaselineConfig(
            architecture="char_cnn",
            vocab_size=tokenizer.size,
            num_acts=6,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            attention_heads=4,
            feedforward_dim=32,
            dropout=0.0,
            max_source_length=384,
            max_target_length=256,
            pad_id=tokenizer.pad_id,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.012, weight_decay=0.0)
    for _ in range(81):
        optimizer.zero_grad(set_to_none=True)
        logits, act_logits = model(
            batch["source_ids"],
            batch["source_mask"],
            batch["decoder_input_ids"],
            batch["decoder_mask"],
        )
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            batch["labels"].flatten(),
            ignore_index=-100,
        ) + 0.2 * torch.nn.functional.cross_entropy(act_logits, batch["act"])
        loss.backward()
        optimizer.step()
    generated, _ = model.greedy_decode(
        batch["source_ids"],
        batch["source_mask"],
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        max_length=128,
    )

    assert float(loss.detach()) < 0.08
    assert [tokenizer.decode(row) for row in generated] == [
        dumps(example.target) for example in examples
    ]


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


def test_check_only_does_not_read_test_and_reports_rare_act_coverage(
    tmp_path, monkeypatch
):
    original_read_bytes = Path.read_bytes
    byte_reads: list[str] = []

    def guarded_read_bytes(path: Path) -> bytes:
        byte_reads.append(path.name)
        if path.name == "test.jsonl":
            raise AssertionError("check-only attempted to open locked test bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
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

    assert report["data"]["test_loaded"] is False
    assert report["data"]["evaluation_holdout_loaded"] is False
    assert report["data"]["acts"]["ask"] >= 30
    assert report["data"]["acts"]["cancel"] >= 20
    assert report["protocol"]["test_used_for_selection"] is False
    assert report["protocol"]["evaluation_holdout_loaded"] is False
    assert "test.jsonl" not in byte_reads
    assert not (tmp_path / "unused").exists()


def test_resume_reuses_a_completed_matching_report(tmp_path, monkeypatch):
    output = tmp_path / "completed"
    output.mkdir()
    checkpoint = output / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    expected = {
        "architecture": "char_cnn",
        "seed": 17,
        "smoke": False,
        "checkpoint": str(checkpoint),
        "hyperparameters": {"learning_rate": 0.0005, "dropout": 0.1},
    }
    (output / "report.json").write_text(json.dumps(expected), encoding="utf-8")
    args = SimpleNamespace(resume_existing=True, data_dir=str(tmp_path / "data"))
    monkeypatch.setattr(
        baseline_runner,
        "train_baseline",
        lambda _config: pytest.fail("completed run was trained again"),
    )

    result = baseline_runner._run(
        args,
        "char_cnn",
        17,
        0.0005,
        0.1,
        output,
        epochs=24,
    )

    assert result == expected


def test_locked_test_rejects_smoke_checkpoint_before_opening_data(tmp_path):
    path = tmp_path / "smoke.pt"
    torch.save({"kind": "jsc_baseline_inference", "smoke": True}, path)

    with pytest.raises(ValueError, match="smoke checkpoints"):
        evaluate_locked_test(path, tmp_path, device="cpu")


def test_full_protocol_freezes_validation_selection_before_opening_test(
    tmp_path, monkeypatch
):
    architecture_score = {
        "char_cnn": 0.50,
        "bigru": 0.60,
        "tiny_transformer": 0.70,
    }

    def fake_run(
        args,
        architecture,
        seed,
        learning_rate,
        dropout,
        output_dir,
        *,
        epochs,
        smoke=False,
    ):
        tuning_bonus = (0.02 if learning_rate == 5e-4 else 0.0) + (
            0.01 if dropout == 0.10 else 0.0
        )
        score = architecture_score[architecture] + tuning_bonus
        return {
            "architecture": architecture,
            "seed": seed,
            "parameters": 100,
            "checkpoint": str(output_dir / f"{architecture}.pt"),
            "hyperparameters": {
                "learning_rate": learning_rate,
                "dropout": dropout,
            },
            "validation": {
                "generation": {
                    "exact_jal_accuracy": score,
                    "schema_valid_rate": score,
                    "false_execution_rate": 1.0 - score,
                },
                "teacher_forced": {"token_nll": 1.0 - score},
            },
        }

    test_calls: list[str] = []

    def fake_test(checkpoint, data_dir, **kwargs):
        selection_path = tmp_path / "selection_before_test.json"
        assert selection_path.is_file()
        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
        assert frozen["test_opened"] is False
        architecture = Path(checkpoint).stem
        test_calls.append(architecture)
        return {
            "architecture": architecture,
            "metrics": {
                "generation": {
                    "exact_jal_accuracy": 0.75,
                    "schema_valid_rate": 0.80,
                    "false_execution_rate": 0.01,
                }
            },
        }

    monkeypatch.setattr(baseline_runner, "_run", fake_run)
    monkeypatch.setattr(baseline_runner, "evaluate_locked_test", fake_test)
    args = SimpleNamespace(
        skip_sweep=False,
        architectures=list(ARCHITECTURES),
        learning_rates=[2e-4, 5e-4],
        dropouts=[0.10, 0.20],
        seeds=[17, 29, 41],
        pilot_epochs=2,
        epochs=3,
        data_dir=str(DATA_DIR),
        device="cpu",
        batch_size=4,
    )

    result = baseline_runner._full_protocol(args, tmp_path)

    assert result["selected_architecture"] == "tiny_transformer"
    assert result["selected_hyperparameters"]["tiny_transformer"] == {
        "learning_rate": 5e-4,
        "dropout": 0.10,
    }
    assert test_calls == ["tiny_transformer"] * 3


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
