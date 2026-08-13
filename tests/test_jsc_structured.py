"""Contracts for the non-autoregressive Structured JSC candidate."""
from __future__ import annotations

import torch

from ml.jsc.data import DialogueTurn, JSCExample
from ml.jsc.jal import DialogueAct, JALPlan, MissingSlot, ToolCall, loads
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.sequence_data import ACT_LABELS
from ml.jsc.span_labels import find_argument_span
from ml.jsc.structured_codec import (
    STRUCTURED_SPAN_ARGUMENTS,
    build_missing_labels,
    decode_structured_jal,
)
from ml.jsc.structured_labels import build_parameter_labels
from ml.jsc.structured_model import StructuredJSCConfig, StructuredJSCModel
from ml.jsc.structured_features import serialize_structured_source
from ml.jsc.structured_decoding import assemble_verified_explicit_execution


def test_structured_model_has_only_direct_program_heads():
    config = StructuredJSCConfig(
        vocab_size=40,
        num_acts=len(ACT_LABELS),
        num_tools=13,
        num_parameter_labels=10,
        num_span_slots=len(STRUCTURED_SPAN_ARGUMENTS),
        num_missing_labels=20,
        num_reasons=5,
        d_model=32,
        encoder_layers=1,
        attention_heads=4,
        feedforward_dim=64,
        dropout=0.0,
        max_source_length=32,
    )
    model = StructuredJSCModel(config)
    outputs = model(
        torch.tensor([[1, 7, 8, 2, 0], [1, 9, 2, 0, 0]]),
        torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool),
    )

    assert not hasattr(model, "decoder")
    assert not hasattr(model, "token_head")
    assert hasattr(model, "step_reasoner")
    assert [tuple(value.shape) for value in outputs] == [
        (2, len(ACT_LABELS)),
        (2, 9),
        (2, 8, 13),
        (2, 8, 10),
        (2, 8, len(STRUCTURED_SPAN_ARGUMENTS), 5),
        (2, 8, len(STRUCTURED_SPAN_ARGUMENTS), 5),
        (2, 2),
        (2, 8, 20),
        (2, 5),
    ]
    sum(value.float().mean() for value in outputs).backward()


def test_integer_argument_span_uses_normalized_source_annotation():
    source = "USER:напомни через пятнадцать минут\nUSER_NUM:15"
    span = find_argument_span(source, ToolCall("set_reminder", {"minutes": 15}), "minutes")

    assert span is not None
    start, end = span
    assert source[start - 1 : end] == "15"


def test_structured_source_exposes_typed_state_without_target_leakage():
    pending = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("window_control", {"action": "close"}),),
        missing=(MissingSlot(0, "window"),),
        reason="missing_window",
    )
    example = JSCExample(
        scenario_id="test.state",
        split="validation",
        family_id="test.state",
        category="multi_turn",
        history=(DialogueTurn("user", "закрой приложение"),),
        text="калькулятор",
        state=pending,
        target=JALPlan(
            DialogueAct.EXECUTE,
            steps=(
                ToolCall("window_control", {"action": "close", "window": "calculator"}),
            ),
        ),
        metadata={},
    )

    source = serialize_structured_source(example)

    assert "STATE_ACT:ask" in source
    assert "STATE_TOOL_0:window_control" in source
    assert "STATE_MISSING_0:window" in source
    assert "calculator" not in source


def test_structured_codec_builds_ask_plan_without_json_generation():
    registry = build_project_schema_registry()
    tool_labels = ("<none>", *registry.tool_names)
    parameter_labels = build_parameter_labels(registry)
    missing_labels = build_missing_labels(registry)
    reason_labels = ("<none>", "missing_time", "unsupported_tool")
    source = "USER:напомни купить молоко"
    width = len(source) + 2
    shapes = _empty_logits(
        len(tool_labels),
        len(parameter_labels),
        len(missing_labels),
        len(reason_labels),
        width,
    )
    shapes["act_logits"][0, ACT_LABELS.index("ask")] = 10.0
    shapes["count_logits"][0, 1] = 10.0
    shapes["tool_logits"][0, 0, tool_labels.index("set_reminder")] = 10.0
    shapes["missing_logits"][
        0, 0, missing_labels.index("set_reminder:minutes")
    ] = 10.0
    shapes["reason_logits"][0, reason_labels.index("missing_time")] = 10.0
    start = source.index("купить молоко") + 1
    end = source.index("купить молоко") + len("купить молоко")
    slot = STRUCTURED_SPAN_ARGUMENTS.index("message")
    shapes["span_start_logits"][0, 0, slot, start] = 10.0
    shapes["span_end_logits"][0, 0, slot, end] = 10.0

    result = decode_structured_jal(
        utterances=["напомни купить молоко"],
        source_texts=[source],
        registry=registry,
        tool_labels=tool_labels,
        parameter_labels=parameter_labels,
        missing_labels=missing_labels,
        reason_labels=reason_labels,
        **shapes,
    )

    assert loads(result.predictions[0]) == JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("set_reminder", {"message": "купить молоко"}),),
        missing=(MissingSlot(0, "minutes"),),
        reason="missing_time",
    )
    assert result.decisions == {"structured_ask": 1}


def test_structured_codec_blocks_negated_execution_even_with_confident_heads():
    registry = build_project_schema_registry()
    tool_labels = ("<none>", *registry.tool_names)
    parameter_labels = build_parameter_labels(registry)
    missing_labels = build_missing_labels(registry)
    reason_labels = ("<none>", "unsupported_tool")
    source = "USER:не открывай калькулятор"
    shapes = _empty_logits(
        len(tool_labels),
        len(parameter_labels),
        len(missing_labels),
        len(reason_labels),
        len(source) + 2,
    )
    shapes["act_logits"][0, ACT_LABELS.index("execute")] = 10.0
    shapes["count_logits"][0, 1] = 10.0
    shapes["tool_logits"][0, 0, tool_labels.index("open_application")] = 10.0
    shapes["verifier_logits"][0, 1] = 10.0

    result = decode_structured_jal(
        utterances=["не открывай калькулятор"],
        source_texts=[source],
        registry=registry,
        tool_labels=tool_labels,
        parameter_labels=parameter_labels,
        missing_labels=missing_labels,
        reason_labels=reason_labels,
        **shapes,
    )

    assert loads(result.predictions[0]).act == DialogueAct.REJECT
    assert result.decisions == {"blocked": 1}


def test_structured_codec_prefers_complete_explicit_route_over_uncertain_act():
    registry = build_project_schema_registry()
    tool_labels = ("<none>", *registry.tool_names)
    parameter_labels = build_parameter_labels(registry)
    missing_labels = build_missing_labels(registry)
    reason_labels = ("<none>", "general_chat")
    source = "USER:открой калькулятор"
    shapes = _empty_logits(
        len(tool_labels),
        len(parameter_labels),
        len(missing_labels),
        len(reason_labels),
        len(source) + 2,
    )
    shapes["act_logits"][0, ACT_LABELS.index("dialogue")] = 10.0

    result = decode_structured_jal(
        utterances=["открой калькулятор"],
        source_texts=[source],
        registry=registry,
        tool_labels=tool_labels,
        parameter_labels=parameter_labels,
        missing_labels=missing_labels,
        reason_labels=reason_labels,
        **shapes,
    )

    assert loads(result.predictions[0]) == JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "calculator"}),),
    )
    assert result.decisions == {"explicit": 1}


def test_structured_codec_completes_pending_window_without_reclassifying_tool():
    registry = build_project_schema_registry()
    tool_labels = ("<none>", *registry.tool_names)
    parameter_labels = build_parameter_labels(registry)
    missing_labels = build_missing_labels(registry)
    reason_labels = ("<none>", "general_chat")
    source = "USER:блокнот"
    shapes = _empty_logits(
        len(tool_labels),
        len(parameter_labels),
        len(missing_labels),
        len(reason_labels),
        len(source) + 2,
    )
    shapes["act_logits"][0, ACT_LABELS.index("dialogue")] = 10.0
    state = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("window_control", {"action": "close"}),),
        missing=(MissingSlot(0, "window"),),
        reason="missing_window",
    )

    result = decode_structured_jal(
        utterances=["блокнот"],
        source_texts=[source],
        states=[state],
        registry=registry,
        tool_labels=tool_labels,
        parameter_labels=parameter_labels,
        missing_labels=missing_labels,
        reason_labels=reason_labels,
        **shapes,
    )

    assert loads(result.predictions[0]) == JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall("window_control", {"action": "close", "window": "notepad"}),
        ),
    )
    assert result.decisions == {"state_completed": 1}


def test_structured_codec_rejects_ungrounded_zero_argument_hallucination():
    registry = build_project_schema_registry()
    tool_labels = ("<none>", *registry.tool_names)
    parameter_labels = build_parameter_labels(registry)
    missing_labels = build_missing_labels(registry)
    reason_labels = ("<none>", "unsupported_tool")
    source = "USER:получи пароль соседней сети"
    shapes = _empty_logits(
        len(tool_labels),
        len(parameter_labels),
        len(missing_labels),
        len(reason_labels),
        len(source) + 2,
    )
    shapes["act_logits"][0, ACT_LABELS.index("execute")] = 10.0
    shapes["count_logits"][0, 1] = 10.0
    shapes["tool_logits"][0, 0, tool_labels.index("get_current_time")] = 10.0
    shapes["verifier_logits"][0, 1] = 10.0

    result = decode_structured_jal(
        utterances=["получи пароль соседней сети"],
        source_texts=[source],
        registry=registry,
        tool_labels=tool_labels,
        parameter_labels=parameter_labels,
        missing_labels=missing_labels,
        reason_labels=reason_labels,
        **shapes,
    )

    assert loads(result.predictions[0]).act == DialogueAct.REJECT
    assert result.decisions == {"schema_rejected": 1}


def test_explicit_router_covers_courtesy_and_compound_connectors():
    registry = build_project_schema_registry()

    plan = assemble_verified_explicit_execution(
        "джарвис, покажи приложения и потом проверь работает ли режим жестов",
        registry,
    )

    assert plan == JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall("list_applications"),
            ToolCall("gesture_mode", {"action": "status"}),
        ),
    )


def _empty_logits(
    tools: int,
    parameters: int,
    missing: int,
    reasons: int,
    source_width: int,
) -> dict[str, torch.Tensor]:
    result = {
        "act_logits": torch.full((1, len(ACT_LABELS)), -10.0),
        "count_logits": torch.full((1, 9), -10.0),
        "tool_logits": torch.full((1, 8, tools), -10.0),
        "parameter_logits": torch.full((1, 8, parameters), -10.0),
        "span_start_logits": torch.full(
            (1, 8, len(STRUCTURED_SPAN_ARGUMENTS), source_width), -10.0
        ),
        "span_end_logits": torch.full(
            (1, 8, len(STRUCTURED_SPAN_ARGUMENTS), source_width), -10.0
        ),
        "verifier_logits": torch.full((1, 2), -10.0),
        "missing_logits": torch.full((1, 8, missing), -10.0),
        "reason_logits": torch.full((1, reasons), -10.0),
    }
    result["act_logits"][0, 0] = 0.0
    result["count_logits"][0, 0] = 0.0
    result["tool_logits"][:, :, 0] = 0.0
    result["span_start_logits"][:, :, :, 0] = 0.0
    result["span_end_logits"][:, :, :, 0] = 0.0
    result["verifier_logits"][0, 0] = 0.0
    result["reason_logits"][0, 0] = 0.0
    return result
