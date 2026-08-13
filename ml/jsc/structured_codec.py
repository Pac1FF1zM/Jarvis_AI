"""Typed decoding boundary for non-autoregressive Structured JSC."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from core.russian_numbers import extract_russian_cardinals
from tools._applications import resolve_application

from .jal import DialogueAct, JALPlan, MissingSlot, ToolCall, ToolSchemaRegistry, dumps
from .sequence_data import ACT_LABELS
from .span_labels import SPAN_ARGUMENTS, decode_span_arguments
from .structured_decoding import (
    assemble_structured_execution,
    assemble_verified_explicit_execution,
    has_explicit_execution_blocker,
)
from .structured_labels import decode_parameter_logits


NONE_REASON = "<none>"
STRUCTURED_SPAN_ARGUMENTS = SPAN_ARGUMENTS + (
    "minutes",
    "reminder_id",
    "steps",
    "undo_token",
)


@dataclass(frozen=True)
class StructuredDecodeResult:
    predictions: tuple[str, ...]
    decisions: Mapping[str, int]


def build_missing_labels(registry: ToolSchemaRegistry) -> tuple[str, ...]:
    return tuple(
        f"{tool}:{name}"
        for tool in registry.tool_names
        for name in registry.argument_names(tool)
    )


def decode_structured_jal(
    *,
    utterances: Sequence[str],
    source_texts: Sequence[str],
    act_logits: torch.Tensor,
    count_logits: torch.Tensor,
    tool_logits: torch.Tensor,
    parameter_logits: torch.Tensor,
    span_start_logits: torch.Tensor,
    span_end_logits: torch.Tensor,
    verifier_logits: torch.Tensor,
    missing_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    registry: ToolSchemaRegistry,
    tool_labels: Sequence[str],
    parameter_labels: Sequence[str],
    missing_labels: Sequence[str],
    reason_labels: Sequence[str],
    states: Sequence[JALPlan | None] | None = None,
    execution_threshold: float = 0.65,
    verifier_threshold: float = 0.50,
    parameter_threshold: float = 0.50,
    span_threshold: float = 0.35,
    missing_threshold: float = 0.45,
) -> StructuredDecodeResult:
    """Convert structured logits into canonical, schema-validated JAL."""
    rows = len(utterances)
    if len(source_texts) != rows:
        raise ValueError("source_texts must align with utterances")
    if states is not None and len(states) != rows:
        raise ValueError("states must align with utterances")
    expected_shapes = {
        "act": act_logits.shape[0],
        "count": count_logits.shape[0],
        "tool": tool_logits.shape[0],
        "parameter": parameter_logits.shape[0],
        "span_start": span_start_logits.shape[0],
        "span_end": span_end_logits.shape[0],
        "verifier": verifier_logits.shape[0],
        "missing": missing_logits.shape[0],
        "reason": reason_logits.shape[0],
    }
    if any(value != rows for value in expected_shapes.values()):
        raise ValueError(f"structured logits do not align: {expected_shapes}")
    if tool_logits.shape[-1] != len(tool_labels):
        raise ValueError("tool labels do not match logits")
    if parameter_logits.shape[-1] != len(parameter_labels):
        raise ValueError("parameter labels do not match logits")
    if missing_logits.shape[-1] != len(missing_labels):
        raise ValueError("missing labels do not match logits")
    if reason_logits.shape[-1] != len(reason_labels):
        raise ValueError("reason labels do not match logits")
    act_probabilities = act_logits.float().softmax(-1).cpu()
    verifier_probabilities = verifier_logits.float().softmax(-1).cpu()
    counts = count_logits.argmax(-1).cpu()
    tools = tool_logits.argmax(-1).cpu()
    parameters = parameter_logits.float().sigmoid().cpu()
    missing = missing_logits.float().sigmoid().cpu()
    reasons = reason_logits.argmax(-1).cpu()
    starts = span_start_logits.float().softmax(-1).cpu()
    ends = span_end_logits.float().softmax(-1).cpu()
    decisions: Counter[str] = Counter()
    predictions: list[str] = []
    for index in range(rows):
        if has_explicit_execution_blocker(utterances[index]):
            predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
            decisions["blocked"] += 1
            continue
        state = states[index] if states is not None else None
        completed = _complete_pending_state(utterances[index], state, registry)
        if completed is not None:
            predictions.append(dumps(completed))
            decisions["state_completed"] += 1
            continue
        explicit = assemble_verified_explicit_execution(utterances[index], registry)
        if explicit is not None:
            predictions.append(dumps(explicit))
            decisions["explicit"] += 1
            continue
        act_id = int(act_probabilities[index].argmax())
        act = DialogueAct(ACT_LABELS[act_id])
        act_confidence = float(act_probabilities[index, act_id])
        reason = reason_labels[int(reasons[index])]
        if reason == NONE_REASON:
            reason = _default_reason(act)
        if act == DialogueAct.CANCEL:
            predictions.append(dumps(JALPlan(act)))
            decisions["cancel"] += 1
            continue
        if act in {DialogueAct.DIALOGUE, DialogueAct.REJECT}:
            predictions.append(dumps(JALPlan(act, reason=reason)))
            decisions[act.value] += 1
            continue
        if act == DialogueAct.EXECUTE:
            if act_confidence < execution_threshold:
                predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
                decisions["low_confidence"] += 1
                continue
            verifier_row = verifier_probabilities[index]
            if int(verifier_row.argmax()) != 1 or float(verifier_row[1]) < verifier_threshold:
                predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
                decisions["verifier_rejected"] += 1
                continue
        count = int(counts[index])
        if count < 1:
            predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
            decisions["empty_steps"] += 1
            continue
        predicted_tools = tuple(
            tool_labels[int(tool_id)] for tool_id in tools[index, :count]
        )
        if any(tool == "<none>" or tool not in registry.tool_names for tool in predicted_tools):
            predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
            decisions["invalid_tool"] += 1
            continue
        arguments = tuple(
            decode_parameter_logits(
                parameters[index, step].tolist(),
                parameter_labels,
                tool,
                threshold=parameter_threshold,
            )
            for step, tool in enumerate(predicted_tools)
        )
        span_arguments = decode_span_arguments(
            starts[index].tolist(),
            ends[index].tolist(),
            source_texts[index],
            predicted_tools,
            registry,
            confidence_threshold=span_threshold,
            span_slots=STRUCTURED_SPAN_ARGUMENTS,
        )
        combined = tuple(
            {**span_values, **parameter_values}
            for span_values, parameter_values in zip(
                span_arguments, arguments, strict=True
            )
        )
        if act == DialogueAct.EXECUTE:
            plan = assemble_structured_execution(
                utterances[index],
                predicted_tools,
                registry,
                structured_arguments=combined,
                allow_neural_evidence=True,
            )
            if plan is None:
                predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
                decisions["schema_rejected"] += 1
            else:
                predictions.append(dumps(plan))
                decisions["structured_execute"] += 1
            continue
        try:
            missing_slots = _decode_missing(
                missing[index], predicted_tools, missing_labels, missing_threshold
            )
            calls = tuple(
                ToolCall(
                    tool,
                    {
                        name: value
                        for name, value in values.items()
                        if MissingSlot(step, name) not in missing_slots
                    },
                )
                for step, (tool, values) in enumerate(zip(predicted_tools, combined, strict=True))
            )
            if act == DialogueAct.ASK and not missing_slots:
                missing_slots = _infer_required_missing(calls, registry)
            plan = JALPlan(act, steps=calls, missing=missing_slots, reason=reason)
            registry.validate(plan)
        except (TypeError, ValueError):
            predictions.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
            decisions["non_execute_schema_rejected"] += 1
        else:
            predictions.append(dumps(plan))
            decisions[f"structured_{act.value}"] += 1
    return StructuredDecodeResult(tuple(predictions), dict(decisions))


def _decode_missing(
    probabilities: torch.Tensor,
    tools: Sequence[str],
    labels: Sequence[str],
    threshold: float,
) -> tuple[MissingSlot, ...]:
    result: list[MissingSlot] = []
    for step, tool in enumerate(tools):
        candidates = [
            (float(probabilities[step, index]), label.split(":", 1)[1])
            for index, label in enumerate(labels)
            if label.startswith(tool + ":")
        ]
        if not candidates:
            continue
        confidence, name = max(candidates)
        if confidence >= threshold:
            result.append(MissingSlot(step, name))
    return tuple(result)


def _infer_required_missing(
    calls: Sequence[ToolCall], registry: ToolSchemaRegistry
) -> tuple[MissingSlot, ...]:
    return tuple(
        MissingSlot(step, name)
        for step, call in enumerate(calls)
        for name in registry.required_arguments(call.tool)
        if name not in call.arguments
    )


def _default_reason(act: DialogueAct) -> str | None:
    if act == DialogueAct.DIALOGUE:
        return "general_chat"
    if act == DialogueAct.REJECT:
        return "unsupported_tool"
    if act == DialogueAct.ASK:
        return "missing_time"
    if act == DialogueAct.CONFIRM:
        return "user_confirmation"
    return None


def _complete_pending_state(
    utterance: str,
    state: JALPlan | None,
    registry: ToolSchemaRegistry,
) -> JALPlan | None:
    """Complete one unambiguous typed slot without reclassifying its tool."""
    if (
        state is None
        or state.act != DialogueAct.ASK
        or len(state.steps) != 1
        or len(state.missing) != 1
    ):
        return None
    missing = state.missing[0]
    if missing.step != 0:
        return None
    value: str | int | None = None
    if missing.name in {"application", "window"}:
        application = resolve_application(utterance)
        if application is not None:
            value = application.name
    elif missing.name in {"minutes", "reminder_id", "steps"}:
        numbers = extract_russian_cardinals(utterance)
        if len(numbers) == 1:
            value = numbers[0]
    if value is None:
        return None
    call = state.steps[0]
    try:
        plan = JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall(call.tool, {**call.arguments, missing.name: value}),),
        )
        registry.validate(plan)
    except (TypeError, ValueError):
        return None
    return plan
