"""Closed labels for schema-conditioned categorical JAL parameters."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .jal import JALScalar, ToolCall, ToolSchemaRegistry


def parameter_label(tool: str, name: str, value: JALScalar) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"{tool}|{name}={encoded}"


def parse_parameter_label(label: str) -> tuple[str, str, JALScalar]:
    try:
        tool, tail = label.split("|", 1)
        name, encoded = tail.split("=", 1)
        value = json.loads(encoded)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid structured parameter label {label!r}") from exc
    if not tool or not name or isinstance(value, (list, dict)):
        raise ValueError(f"invalid structured parameter label {label!r}")
    return tool, name, value


def build_parameter_labels(registry: ToolSchemaRegistry) -> tuple[str, ...]:
    labels = [
        parameter_label(tool, name, value)
        for tool in registry.tool_names
        for name, values in registry.categorical_values(tool).items()
        for value in values
    ]
    return tuple(sorted(labels))


def call_parameter_labels(
    call: ToolCall, registry: ToolSchemaRegistry
) -> tuple[str, ...]:
    categorical = registry.categorical_values(call.tool)
    return tuple(
        parameter_label(call.tool, name, value)
        for name, value in call.arguments.items()
        if name in categorical
    )


def decode_parameter_logits(
    scores: Sequence[float],
    labels: Sequence[str],
    tool: str,
    *,
    threshold: float = 0.5,
) -> dict[str, JALScalar]:
    """Decode at most one schema value per argument for one predicted tool."""
    if len(scores) != len(labels):
        raise ValueError("parameter scores and labels have different lengths")
    grouped: dict[str, list[tuple[float, JALScalar]]] = {}
    for score, label in zip(scores, labels):
        label_tool, name, value = parse_parameter_label(label)
        if label_tool == tool:
            grouped.setdefault(name, []).append((float(score), value))
    result: dict[str, JALScalar] = {}
    for name, candidates in grouped.items():
        score, value = max(candidates, key=lambda item: item[0])
        if score >= threshold:
            result[name] = value
    return result
