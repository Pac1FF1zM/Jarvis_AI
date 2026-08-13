"""Character-span supervision for free-form JAL arguments."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from memory.workspaces import canonical_workspace_name
from tools._applications import APPLICATIONS, normalise_name, resolve_application

from .jal import ToolCall, ToolSchemaRegistry


SPAN_ARGUMENTS = (
    "application",
    "message",
    "query",
    "url",
    "window",
    "path",
    "new_name",
    "workspace",
    "clock_time",
    "due_at",
)


def span_tool_arguments(
    registry: ToolSchemaRegistry,
    span_slots: Sequence[str] = SPAN_ARGUMENTS,
) -> dict[str, tuple[str, ...]]:
    return {
        tool: tuple(name for name in span_slots if name in registry.argument_names(tool))
        for tool in registry.tool_names
    }


def find_argument_span(
    source_text: str,
    call: ToolCall,
    argument: str,
) -> tuple[int, int] | None:
    """Return inclusive tokenizer positions (BOS is zero) for one target value."""
    value = call.arguments.get(argument)
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    value_text = str(value)
    if not value_text.strip():
        return None
    source = source_text.casefold().replace("ё", "е")
    candidates = [_normalize(value_text)]
    if argument in {"application", "window"}:
        resolved = resolve_application(value_text)
        if resolved is not None:
            for spec in APPLICATIONS:
                if spec.name == resolved.name:
                    candidates.extend(
                        normalise_name(item)
                        for item in (spec.name, spec.display_name, *spec.aliases)
                    )
                    break
    positions: list[tuple[int, int]] = []
    for candidate in set(candidates):
        start = source.rfind(candidate)
        if start >= 0:
            positions.append((start, start + len(candidate)))
    if argument == "workspace":
        target = canonical_workspace_name(value)
        words = list(re.finditer(r"[a-zа-я0-9]+", source))
        for left in range(len(words)):
            for width in range(1, min(3, len(words) - left) + 1):
                right = left + width - 1
                surface = source[words[left].start() : words[right].end()]
                if canonical_workspace_name(surface) == target:
                    positions.append((words[left].start(), words[right].end()))
    if not positions:
        return None
    start, end_exclusive = max(positions, key=lambda item: (item[0], item[1] - item[0]))
    # Character zero is tokenizer position one because position zero is BOS.
    return start + 1, end_exclusive


def decode_span_arguments(
    start_probabilities: Sequence[Sequence[Sequence[float]]],
    end_probabilities: Sequence[Sequence[Sequence[float]]],
    source_text: str,
    tools: Sequence[str],
    registry: ToolSchemaRegistry,
    *,
    confidence_threshold: float = 0.45,
    maximum_length: int = 256,
    span_slots: Sequence[str] = SPAN_ARGUMENTS,
) -> tuple[dict[str, str], ...]:
    """Decode safe, schema-applicable character spans for predicted tools."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("span confidence threshold must be in [0, 1]")
    result: list[dict[str, str]] = []
    for step_index, tool in enumerate(tools):
        if tool not in registry.tool_names:
            result.append({})
            continue
        allowed = set(registry.argument_names(tool))
        arguments: dict[str, str] = {}
        for slot_index, name in enumerate(span_slots):
            if name not in allowed:
                continue
            starts = start_probabilities[step_index][slot_index]
            ends = end_probabilities[step_index][slot_index]
            start = max(range(len(starts)), key=lambda index: float(starts[index]))
            end = max(range(len(ends)), key=lambda index: float(ends[index]))
            confidence = min(float(starts[start]), float(ends[end]))
            if (
                start <= 0
                or end < start
                or confidence < confidence_threshold
                or end - start + 1 > maximum_length
                or end > len(source_text)
            ):
                continue
            value = source_text[start - 1 : end].strip(" ,.:;!?-\n")
            if not value or "\n" in value:
                continue
            if name == "application":
                application = resolve_application(value)
                if application is None:
                    continue
                value = application.name
            elif name == "window":
                application = resolve_application(value)
                if application is not None:
                    value = application.name
            elif name == "workspace":
                value = canonical_workspace_name(value)
            arguments[name] = value
        result.append(arguments)
    return tuple(result)


def _normalize(value: str) -> str:
    normalized = value.casefold().replace("ё", "е")
    normalized = normalized.replace("№", " номер ")
    normalized = re.sub(r"[;\-–—]", " ", normalized)
    return " ".join(normalized.split())
