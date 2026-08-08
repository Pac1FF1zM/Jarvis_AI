"""Fail-closed JAL decoding boundary for neural semantic parsers."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch

from .jal import DialogueAct, JALPlan, ToolSchemaRegistry, dumps, loads
from .sequence_data import ACT_LABELS
from .structured_decoding import (
    assemble_structured_execution,
    assemble_verified_explicit_execution,
    has_explicit_execution_blocker,
)
from .structured_labels import decode_parameter_logits
from .span_labels import SPAN_ARGUMENTS, decode_span_arguments
from core.russian_numbers import extract_russian_cardinals

_NUMERIC_ARGUMENTS = {
    "set_reminder": "minutes",
    "cancel_reminder": "reminder_id",
}
DEFAULT_EXECUTION_THRESHOLD = 0.90
DEFAULT_EXECUTION_VERIFIER_THRESHOLD = 0.50


@dataclass(frozen=True)
class ConstrainedDecodeResult:
    predictions: tuple[str, ...]
    decisions: dict[str, int]


def constrain_jal_predictions(
    predictions: Sequence[str],
    act_logits: torch.Tensor,
    registry: ToolSchemaRegistry,
    *,
    execution_threshold: float = DEFAULT_EXECUTION_THRESHOLD,
    utterances: Sequence[str] | None = None,
    step_count_logits: torch.Tensor | None = None,
    tool_logits: torch.Tensor | None = None,
    tool_labels: Sequence[str] | None = None,
    parameter_logits: torch.Tensor | None = None,
    parameter_labels: Sequence[str] | None = None,
    parameter_threshold: float = 0.5,
    span_start_logits: torch.Tensor | None = None,
    span_end_logits: torch.Tensor | None = None,
    span_slots: Sequence[str] | None = None,
    span_sources: Sequence[str] | None = None,
    span_threshold: float = 0.45,
    execution_verifier_logits: torch.Tensor | None = None,
    execution_verifier_threshold: float = DEFAULT_EXECUTION_VERIFIER_THRESHOLD,
) -> ConstrainedDecodeResult:
    """Return only canonical, schema-valid plans and reject unsafe generations.

    The generative decoder is never the final authority. An execution is admitted
    only when its JAL is valid, agrees with the independent act head, and clears
    the confidence threshold. Every other result becomes a non-executing plan.
    """
    if not 0.0 <= execution_threshold <= 1.0:
        raise ValueError("execution_threshold must be in [0, 1]")
    if act_logits.ndim != 2 or act_logits.shape[0] != len(predictions):
        raise ValueError("act logits must have one row per prediction")
    if utterances is not None and len(utterances) != len(predictions):
        raise ValueError("utterances must have one row per prediction")
    structured = step_count_logits is not None or tool_logits is not None
    if structured:
        if step_count_logits is None or tool_logits is None or tool_labels is None:
            raise ValueError("structured constraints require count logits, tool logits and labels")
        if step_count_logits.shape[0] != len(predictions) or tool_logits.shape[0] != len(
            predictions
        ):
            raise ValueError("structured logits must have one row per prediction")
        if tool_logits.shape[-1] != len(tool_labels):
            raise ValueError("tool label count does not match structured logits")
    if parameter_logits is not None:
        if parameter_labels is None or tool_labels is None:
            raise ValueError("parameter constraints require parameter and tool labels")
        if parameter_logits.shape[0] != len(predictions):
            raise ValueError("parameter logits must have one row per prediction")
        if parameter_logits.shape[-1] != len(parameter_labels):
            raise ValueError("parameter label count does not match logits")
        if not 0.0 <= parameter_threshold <= 1.0:
            raise ValueError("parameter threshold must be in [0, 1]")
    spans_enabled = span_start_logits is not None or span_end_logits is not None
    if spans_enabled:
        if span_start_logits is None or span_end_logits is None:
            raise ValueError("span decoding requires start and end logits")
        if span_slots is None or tuple(span_slots) != tuple(SPAN_ARGUMENTS):
            raise ValueError("span labels do not match runtime")
        if span_sources is None or len(span_sources) != len(predictions):
            raise ValueError("span sources must have one row per prediction")
        if span_start_logits.shape != span_end_logits.shape:
            raise ValueError("span start/end shapes differ")
        if span_start_logits.shape[0] != len(predictions):
            raise ValueError("span logits must have one row per prediction")
        if span_start_logits.shape[2] != len(span_slots):
            raise ValueError("span label count does not match logits")
    if execution_verifier_logits is not None:
        if execution_verifier_logits.shape != (len(predictions), 2):
            raise ValueError("execution verifier logits must have shape [batch, 2]")
        if not 0.0 <= execution_verifier_threshold <= 1.0:
            raise ValueError("execution verifier threshold must be in [0, 1]")
    probabilities = act_logits.float().softmax(dim=-1).cpu()
    labels = probabilities.argmax(dim=-1)
    decisions: Counter[str] = Counter()
    constrained: list[str] = []
    source_texts = utterances if utterances is not None else ("",) * len(predictions)
    predicted_counts = (
        step_count_logits.argmax(-1).cpu() if step_count_logits is not None else [None] * len(predictions)
    )
    predicted_tools = (
        tool_logits.argmax(-1).cpu() if tool_logits is not None else [None] * len(predictions)
    )
    predicted_parameters = (
        parameter_logits.float().sigmoid().cpu()
        if parameter_logits is not None
        else [None] * len(predictions)
    )
    predicted_span_starts = (
        span_start_logits.float().softmax(-1).cpu()
        if span_start_logits is not None
        else [None] * len(predictions)
    )
    predicted_span_ends = (
        span_end_logits.float().softmax(-1).cpu()
        if span_end_logits is not None
        else [None] * len(predictions)
    )
    model_source_texts = span_sources if span_sources is not None else source_texts
    verifier_probabilities = (
        execution_verifier_logits.float().softmax(-1).cpu()
        if execution_verifier_logits is not None
        else [None] * len(predictions)
    )
    for (
        prediction,
        label,
        row,
        utterance,
        predicted_count,
        predicted_tool_row,
        predicted_parameter_row,
        predicted_span_start_row,
        predicted_span_end_row,
        model_source_text,
        verifier_row,
    ) in zip(
        predictions,
        labels,
        probabilities,
        source_texts,
        predicted_counts,
        predicted_tools,
        predicted_parameters,
        predicted_span_starts,
        predicted_span_ends,
        model_source_texts,
        verifier_probabilities,
        strict=True,
    ):
        auxiliary_act = DialogueAct(ACT_LABELS[int(label)])
        auxiliary_confidence = float(row[int(label)])
        expected_tools: tuple[str, ...] | None = None
        if predicted_count is not None and predicted_tool_row is not None:
            count = int(predicted_count)
            expected_tools = tuple(
                str(tool_labels[int(tool_id)])
                for tool_id in predicted_tool_row[:count]
            )
        raw_plan: JALPlan | None = None
        raw_grounded = False
        try:
            raw_plan = loads(prediction)
            raw_plan, raw_grounded = _ground_numeric_arguments(raw_plan, utterance)
            registry.validate(raw_plan)
        except (TypeError, ValueError):
            raw_plan = None

        if auxiliary_act == DialogueAct.EXECUTE and auxiliary_confidence < execution_threshold:
            decisions["low_confidence_execution_rejected"] += 1
            constrained.append(dumps(JALPlan(DialogueAct.REJECT, reason="low_confidence")))
            continue
        if auxiliary_act == DialogueAct.EXECUTE and has_explicit_execution_blocker(utterance):
            decisions["unsafe_utterance_rejected"] += 1
            constrained.append(dumps(JALPlan(DialogueAct.REJECT, reason="unsupported_tool")))
            continue
        if auxiliary_act == DialogueAct.EXECUTE and verifier_row is not None:
            execution_confidence = float(verifier_row[1])
            if (
                int(verifier_row.argmax()) != 1
                or execution_confidence < execution_verifier_threshold
            ):
                decisions["execution_verifier_rejected"] += 1
                constrained.append(
                    dumps(JALPlan(DialogueAct.REJECT, reason="low_confidence"))
                )
                continue
            explicit_plan = assemble_verified_explicit_execution(utterance, registry)
            if explicit_plan is not None:
                decisions["accepted_verified_explicit_route"] += 1
                constrained.append(dumps(explicit_plan))
                continue
        if auxiliary_act == DialogueAct.EXECUTE and expected_tools is not None:
            structured_arguments = None
            if predicted_parameter_row is not None:
                structured_arguments = tuple(
                    decode_parameter_logits(
                        predicted_parameter_row[index].tolist(),
                        parameter_labels,
                        tool,
                        threshold=parameter_threshold,
                    )
                    for index, tool in enumerate(expected_tools)
                )
            if predicted_span_start_row is not None and predicted_span_end_row is not None:
                span_arguments = decode_span_arguments(
                    predicted_span_start_row.tolist(),
                    predicted_span_end_row.tolist(),
                    model_source_text,
                    expected_tools,
                    registry,
                    confidence_threshold=span_threshold,
                )
                if structured_arguments is None:
                    structured_arguments = span_arguments
                else:
                    structured_arguments = tuple(
                        {**span_values, **parameter_values}
                        for span_values, parameter_values in zip(
                            span_arguments, structured_arguments, strict=True
                        )
                    )
            assembled = assemble_structured_execution(
                utterance,
                expected_tools,
                registry,
                raw_plan=raw_plan,
                structured_arguments=structured_arguments,
            )
            if assembled is not None:
                decisions["accepted_structured"] += 1
                constrained.append(dumps(assembled))
                continue
        if raw_plan is None:
            decisions["invalid_rejected"] += 1
            constrained.append(dumps(_safe_fallback(auxiliary_act)))
            continue
        plan = raw_plan
        canonical = dumps(plan)
        if plan.act != auxiliary_act:
            decisions["act_disagreement_rejected"] += 1
            constrained.append(dumps(_safe_fallback(auxiliary_act)))
            continue
        if expected_tools is not None:
            actual_tools = tuple(step.tool for step in plan.steps)
            if actual_tools != expected_tools:
                decisions["tool_disagreement_rejected"] += 1
                constrained.append(dumps(_safe_fallback(auxiliary_act)))
                continue
        decisions["accepted_numeric_grounded" if raw_grounded else "accepted"] += 1
        constrained.append(canonical)
    return ConstrainedDecodeResult(tuple(constrained), dict(decisions))


def _safe_fallback(auxiliary_act: DialogueAct) -> JALPlan:
    if auxiliary_act == DialogueAct.CANCEL:
        return JALPlan(DialogueAct.CANCEL)
    if auxiliary_act == DialogueAct.REJECT:
        return JALPlan(DialogueAct.REJECT, reason="unsupported_tool")
    if auxiliary_act == DialogueAct.DIALOGUE:
        return JALPlan(DialogueAct.DIALOGUE, reason="general_chat")
    return JALPlan(DialogueAct.REJECT, reason="unsafe_model_output")


def _ground_numeric_arguments(plan: JALPlan, utterance: str) -> tuple[JALPlan, bool]:
    numbers = extract_russian_cardinals(utterance)
    if (
        len(numbers) != 1
        or not plan.steps
        or plan.act not in {DialogueAct.EXECUTE, DialogueAct.CONFIRM}
    ):
        return plan, False
    grounded = False
    steps = []
    for step in plan.steps:
        argument_name = _NUMERIC_ARGUMENTS.get(step.tool)
        arguments = dict(step.arguments)
        if argument_name is not None and arguments.get(argument_name) != numbers[0]:
            arguments[argument_name] = numbers[0]
            grounded = True
        steps.append(type(step)(step.tool, arguments))
    if not grounded:
        return plan, False
    return JALPlan(
        act=plan.act,
        steps=tuple(steps),
        missing=tuple(
            slot
            for slot in plan.missing
            if not (
                slot.step < len(steps)
                and _NUMERIC_ARGUMENTS.get(steps[slot.step].tool) == slot.name
            )
        ),
        reason=plan.reason,
        version=plan.version,
    ), True
