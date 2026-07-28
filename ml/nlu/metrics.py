"""End-to-end semantic metrics for Jarvis NLU predictions."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .data import Example
from .schema import ACTIONABLE_INTENTS, INTENT_SLOTS, NLUResult

SLOT_NAMES = ("application", "minutes", "reminder_text")


def _normalise_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.:;!?-").casefold()


def expected_runtime_slots(example: Example) -> dict[str, str]:
    """Convert annotated training spans to the slots exposed by inference."""
    result: dict[str, str] = {}
    for span in example.spans:
        value = _normalise_value(example.text[span.start : span.end])
        if span.label == "duration":
            match = re.search(r"\d+", value)
            if match:
                result["minutes"] = match.group(0)
        elif value:
            result[span.label] = value
    return result


def canonical_prediction_slots(slots: dict[str, str]) -> dict[str, str]:
    return {
        str(name): _normalise_value(str(value))
        for name, value in slots.items()
        if str(name) in SLOT_NAMES and _normalise_value(str(value))
    }


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def semantic_frame_metrics(
    examples: Iterable[Example], predictions: Iterable[NLUResult]
) -> dict[str, Any]:
    """Score the command Jarvis would execute, not only the intent label."""
    example_list = list(examples)
    prediction_list = list(predictions)
    if len(example_list) != len(prediction_list):
        raise ValueError("examples and predictions must have equal lengths")
    pairs = list(zip(example_list, prediction_list))
    if not pairs:
        raise ValueError("semantic frame metrics require at least one example")

    exact = 0
    actionable_exact = 0
    actionable_total = 0
    no_slot_total = 0
    hallucinations = 0
    illegal_predictions = 0
    predicted_slot_total = 0
    counts = {name: Counter(tp=0, fp=0, fn=0) for name in SLOT_NAMES}

    for example, prediction in pairs:
        expected_slots = expected_runtime_slots(example)
        predicted_slots = canonical_prediction_slots(prediction.slots)
        frame_matches = prediction.intent == example.intent and predicted_slots == expected_slots
        exact += int(frame_matches)
        if example.intent in ACTIONABLE_INTENTS:
            actionable_total += 1
            actionable_exact += int(frame_matches)

        allowed = INTENT_SLOTS[example.intent]
        if not allowed:
            no_slot_total += 1
            hallucinations += int(bool(predicted_slots))
        for name in predicted_slots:
            predicted_slot_total += 1
            illegal_predictions += int(name not in allowed)

        for name in SLOT_NAMES:
            expected_value = expected_slots.get(name)
            predicted_value = predicted_slots.get(name)
            if expected_value is not None and predicted_value == expected_value:
                counts[name]["tp"] += 1
            else:
                if predicted_value is not None:
                    counts[name]["fp"] += 1
                if expected_value is not None:
                    counts[name]["fn"] += 1

    aggregate = Counter(tp=0, fp=0, fn=0)
    per_slot: dict[str, dict[str, float | int]] = {}
    for name, slot_counts in counts.items():
        aggregate.update(slot_counts)
        per_slot[name] = {
            "f1": _f1(slot_counts["tp"], slot_counts["fp"], slot_counts["fn"]),
            "support": slot_counts["tp"] + slot_counts["fn"],
        }

    return {
        "semantic_frame_exact_match": exact / len(pairs),
        "end_to_end_command_accuracy": actionable_exact / max(actionable_total, 1),
        "slot_entity_f1": _f1(aggregate["tp"], aggregate["fp"], aggregate["fn"]),
        "slot_hallucination_rate": hallucinations / max(no_slot_total, 1),
        "illegal_slot_rate": illegal_predictions / max(predicted_slot_total, 1),
        "per_slot": per_slot,
    }


def expected_calibration_error(
    expected: Iterable[str], predictions: Iterable[NLUResult], *, bins: int = 10
) -> float:
    pairs = list(zip(expected, predictions))
    if not pairs:
        raise ValueError("calibration requires at least one prediction")
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            (label, result)
            for label, result in pairs
            if lower < result.confidence <= upper or (index == 0 and result.confidence == 0.0)
        ]
        if members:
            accuracy = sum(label == result.intent for label, result in members) / len(members)
            confidence = sum(result.confidence for _label, result in members) / len(members)
            error += len(members) / len(pairs) * abs(accuracy - confidence)
    return error
