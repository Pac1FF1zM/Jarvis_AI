"""Exact, safety-oriented metrics for generated JAL programs."""
from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from .data import JSCExample
from .jal import DialogueAct, ToolSchemaRegistry, dumps, loads


def evaluate_program_predictions(
    examples: Sequence[JSCExample],
    predictions: Sequence[str],
    registry: ToolSchemaRegistry,
) -> dict[str, Any]:
    if len(examples) != len(predictions) or not examples:
        raise ValueError("program metrics require equal non-empty inputs")
    counts: Counter[str] = Counter()
    category_total: Counter[str] = Counter()
    category_exact: Counter[str] = Counter()
    target_tool_total: Counter[str] = Counter()
    target_tool_exact: Counter[str] = Counter()
    target_tool_present: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    for example, prediction in zip(examples, predictions):
        target_text = dumps(example.target)
        exact = prediction == target_text
        counts["examples"] += 1
        category_total[example.category] += 1
        target_tools = {step.tool for step in example.target.steps}
        target_tool_total.update(target_tools)
        number_surface = example.metadata.get("number_surface")
        if number_surface in {"digits", "words"}:
            counts[f"number_surface_{number_surface}_examples"] += 1
        if example.target.act == DialogueAct.REJECT:
            counts["ood_examples"] += 1
        if exact:
            counts["exact"] += 1
            category_exact[example.category] += 1
            target_tool_exact.update(target_tools)
            if number_surface in {"digits", "words"}:
                counts[f"number_surface_{number_surface}_exact"] += 1
        predicted_plan = None
        try:
            predicted_plan = loads(prediction)
            counts["codec_valid"] += 1
        except (TypeError, ValueError) as exc:
            failure_reasons[_reason_key(exc)] += 1
        if predicted_plan is None:
            continue
        try:
            registry.validate(predicted_plan)
            counts["schema_valid"] += 1
        except (TypeError, ValueError) as exc:
            failure_reasons[_reason_key(exc)] += 1
            continue
        if predicted_plan.act == example.target.act:
            counts["act_correct"] += 1
        if tuple(step.tool for step in predicted_plan.steps) == tuple(
            step.tool for step in example.target.steps
        ):
            counts["tool_sequence_correct"] += 1
        predicted_tools = {step.tool for step in predicted_plan.steps}
        target_tool_present.update(target_tools & predicted_tools)
        if tuple(dict(step.arguments) for step in predicted_plan.steps) == tuple(
            dict(step.arguments) for step in example.target.steps
        ):
            counts["argument_sequence_correct"] += 1
        if number_surface == "words" and _integer_arguments(predicted_plan) == _integer_arguments(
            example.target
        ):
            counts["written_number_value_correct"] += 1
        if (
            example.target.act == DialogueAct.REJECT
            and predicted_plan.act == DialogueAct.REJECT
        ):
            counts["ood_recalled"] += 1
        if example.target.act != DialogueAct.EXECUTE and predicted_plan.act == DialogueAct.EXECUTE:
            counts["false_executions"] += 1
        if predicted_plan.act == DialogueAct.EXECUTE:
            counts["predicted_executions"] += 1
            if exact:
                counts["correct_executions"] += 1
    total = counts["examples"]
    return {
        "examples": total,
        "exact_jal_accuracy": counts["exact"] / total,
        "codec_valid_rate": counts["codec_valid"] / total,
        "schema_valid_rate": counts["schema_valid"] / total,
        "act_accuracy": counts["act_correct"] / total,
        "tool_sequence_accuracy": counts["tool_sequence_correct"] / total,
        "argument_sequence_accuracy": counts["argument_sequence_correct"] / total,
        "ood_recall": counts["ood_recalled"] / max(counts["ood_examples"], 1),
        "false_execution_rate": counts["false_executions"] / total,
        "execution_precision": counts["correct_executions"]
        / max(counts["predicted_executions"], 1),
        "written_number_examples": counts["number_surface_words_examples"],
        "written_number_exact_jal_accuracy": counts["number_surface_words_exact"]
        / max(counts["number_surface_words_examples"], 1),
        "written_number_value_accuracy": counts["written_number_value_correct"]
        / max(counts["number_surface_words_examples"], 1),
        "digit_number_exact_jal_accuracy": counts["number_surface_digits_exact"]
        / max(counts["number_surface_digits_examples"], 1),
        "category_exact_jal": {
            category: category_exact[category] / category_total[category]
            for category in sorted(category_total)
        },
        "target_tool_exact_jal": {
            tool: target_tool_exact[tool] / target_tool_total[tool]
            for tool in sorted(target_tool_total)
        },
        "target_tool_presence_recall": {
            tool: target_tool_present[tool] / target_tool_total[tool]
            for tool in sorted(target_tool_total)
        },
        "invalid_reasons": dict(failure_reasons.most_common(10)),
    }


def _integer_arguments(plan) -> tuple[int, ...]:
    return tuple(
        value
        for step in plan.steps
        for value in step.arguments.values()
        if isinstance(value, int) and not isinstance(value, bool)
    )


def _reason_key(error: Exception) -> str:
    message = str(error).casefold()
    buckets = {
        "too_large": ("too large", "too many"),
        "json": ("json", "expecting", "unterminated", "extra data"),
        "fields": ("field", "key"),
        "act": ("act",),
        "tool": ("tool",),
        "argument": ("argument", "required", "exclusive", "enum"),
    }
    for name, markers in buckets.items():
        if any(marker in message for marker in markers):
            return name
    return "other"
