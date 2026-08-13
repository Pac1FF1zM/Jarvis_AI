"""Schema-conditioned assembly of executable JAL from structured predictions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from core.russian_numbers import normalize_russian_numbers
from modules.command_router import RoutedAction, route_explicit_command, split_compound_command
from tools._applications import resolve_application

from .jal import DialogueAct, JALPlan, ToolCall, ToolSchemaRegistry


_SLOT_ALIASES = {
    "set_reminder": {"reminder_text": "message"},
}
_NON_EXECUTING_ROUTES = {"negated_command", "unsupported_command"}


def assemble_verified_explicit_execution(
    utterance: str,
    registry: ToolSchemaRegistry,
) -> JALPlan | None:
    """Build a plan only when deterministic routing covers the entire command."""
    normalized = normalize_russian_numbers(utterance.casefold().replace("ё", "е"))
    parts = _meaningful_parts(split_compound_command(normalized))
    if not parts:
        return None
    actions = [route_explicit_command(part) for part in parts]
    if any(
        action is None or action.intent not in registry.tool_names
        for action in actions
    ):
        return None
    tools = tuple(action.intent for action in actions if action is not None)
    return assemble_structured_execution(utterance, tools, registry)


def has_explicit_execution_blocker(utterance: str) -> bool:
    """Return true when deterministic grammar says execution is forbidden."""
    normalized = normalize_russian_numbers(utterance.casefold().replace("ё", "е"))
    for part in _meaningful_parts(split_compound_command(normalized)):
        action = route_explicit_command(part)
        if action is not None and action.intent in _NON_EXECUTING_ROUTES:
            return True
    return False


def assemble_structured_execution(
    utterance: str,
    predicted_tools: Sequence[str],
    registry: ToolSchemaRegistry,
    *,
    raw_plan: JALPlan | None = None,
    structured_arguments: Sequence[Mapping[str, Any]] | None = None,
    allow_neural_evidence: bool = False,
) -> JALPlan | None:
    """Build an executable plan without trusting generated tool names.

    The independent structured head owns the ordered tool list. Explicit
    deterministic routing owns arguments whenever possible. A schema-valid raw
    decoder step may fill arguments only for the same tool at the same index.
    The registry is the final fail-closed boundary.
    """
    tools = tuple(predicted_tools)
    if not tools or any(tool not in registry.tool_names for tool in tools):
        return None

    normalized = normalize_russian_numbers(utterance.casefold().replace("ё", "е"))
    parts = _meaningful_parts(split_compound_command(normalized))
    routed = [route_explicit_command(part) for part in parts]
    if any(
        action is not None and action.intent in _NON_EXECUTING_ROUTES
        for action in routed
    ):
        return None

    # A deterministic parser disagreement is stronger evidence than the neural
    # tool head. Never silently replace an explicit routed action.
    routed_tools = tuple(
        action.intent for action in routed if action is not None and action.intent in registry.tool_names
    )
    if routed_tools and not _is_ordered_subsequence(routed_tools, tools):
        return None

    raw_steps = ()
    if raw_plan is not None and raw_plan.act == DialogueAct.EXECUTE:
        raw_steps = raw_plan.steps

    calls: list[ToolCall] = []
    route_cursor = 0
    for index, tool in enumerate(tools):
        action, route_cursor = _next_matching_action(routed, tool, route_cursor)
        arguments: dict[str, Any] = {}
        has_evidence = action is not None
        if action is not None:
            arguments.update(_arguments_from_route(tool, action, registry))
        if structured_arguments is not None and index < len(structured_arguments):
            allowed = set(registry.argument_names(tool))
            for name, value in structured_arguments[index].items():
                if name in allowed:
                    arguments.setdefault(name, value)
            # A direct Structured JSC call may use schema-valid neural evidence
            # after its independent act/verifier gates.  Existing callers keep
            # the historical fail-closed requirement by default.
            has_evidence = has_evidence or allow_neural_evidence
        if index < len(raw_steps) and raw_steps[index].tool == tool:
            has_evidence = True
            allowed = set(registry.argument_names(tool))
            for name, value in raw_steps[index].arguments.items():
                if name in allowed:
                    arguments.setdefault(name, value)
        if not has_evidence:
            return None
        arguments = _coerce_arguments(tool, arguments)
        calls.append(ToolCall(tool, arguments))

    try:
        plan = JALPlan(DialogueAct.EXECUTE, steps=tuple(calls))
        registry.validate(plan)
    except (TypeError, ValueError):
        return None
    return plan


def _meaningful_parts(parts: Sequence[str]) -> list[str]:
    ignored = {"затем", "потом", "после этого", "заодно", "и"}
    return [part.strip(" ,") for part in parts if part.strip(" ,").casefold() not in ignored]


def _is_ordered_subsequence(observed: Sequence[str], expected: Sequence[str]) -> bool:
    cursor = 0
    for item in observed:
        try:
            cursor = expected.index(item, cursor) + 1
        except ValueError:
            return False
    return True


def _next_matching_action(
    actions: Sequence[RoutedAction | None], tool: str, cursor: int
) -> tuple[RoutedAction | None, int]:
    for index in range(cursor, len(actions)):
        action = actions[index]
        if action is not None and action.intent == tool:
            return action, index + 1
    return None, cursor


def _arguments_from_route(
    tool: str, action: RoutedAction, registry: ToolSchemaRegistry
) -> dict[str, Any]:
    aliases = _SLOT_ALIASES.get(tool, {})
    allowed = set(registry.argument_names(tool))
    result: dict[str, Any] = {}
    for source_name, value in action.slots.items():
        name = aliases.get(source_name, source_name)
        if name in allowed:
            result[name] = value
    return result


def _coerce_arguments(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = dict(arguments)
    for name in ("minutes", "reminder_id", "steps"):
        value = result.get(name)
        if isinstance(value, str) and value.isdigit():
            result[name] = int(value)
    if tool == "gesture_mode":
        result.pop("enabled", None)
    if tool == "window_control" and isinstance(result.get("window"), str):
        application = resolve_application(result["window"])
        if application is not None:
            result["window"] = application.name
    return result
