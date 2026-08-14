"""Schema-conditioned assembly of executable JAL from structured predictions."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from core.russian_numbers import normalize_russian_numbers
from modules.command_router import RoutedAction, route_explicit_command, split_compound_command
from tools._applications import resolve_application

from .jal import DialogueAct, JALPlan, MissingSlot, ToolCall, ToolSchemaRegistry


_SLOT_ALIASES = {
    "set_reminder": {"reminder_text": "message"},
}
_NON_EXECUTING_ROUTES = {"negated_command", "unsupported_command"}
_GENERIC_APPLICATION = r"(?:приложени\w*|программ\w*)"
_GENERIC_WINDOW = r"(?:окн\w*|приложени\w*|программ\w*)"


def infer_explicit_clarification(
    utterance: str,
    registry: ToolSchemaRegistry,
) -> JALPlan | None:
    """Build a typed pending plan for deterministic incomplete commands.

    This runs before the ordinary explicit router. Generic nouns such as
    ``приложение`` are missing slots, never literal application/window names.
    Reminder requests with only one half present are handled the same way.
    """
    normalized = normalize_russian_numbers(utterance.casefold().replace("ё", "е"))
    normalized = normalized.strip(" ,.!?:;-")
    open_patterns = (
        rf"^(?:открой|открывай|запусти|запускай|включи)\s+(?:нужн\w*\s+)?{_GENERIC_APPLICATION}$",
        rf"^(?:мне\s+)?(?:нужно|надо)\s+(?:открыть|запустить)\s+(?:одно\s+|нужн\w*\s+)?{_GENERIC_APPLICATION}$",
    )
    close_patterns = (
        rf"^(?:закрой|закрывай|заверши|убери)\s+(?:одно\s+|нужн\w*\s+)?{_GENERIC_WINDOW}$",
        rf"^(?:мне\s+)?(?:нужно|надо)\s+(?:закрыть|завершить)\s+(?:одно\s+|нужн\w*\s+)?{_GENERIC_WINDOW}$",
    )
    if any(re.fullmatch(pattern, normalized) for pattern in open_patterns):
        return _validated_ask(
            registry,
            ToolCall("open_application"),
            "application",
            "missing_application",
        )
    if any(re.fullmatch(pattern, normalized) for pattern in close_patterns):
        return _validated_ask(
            registry,
            ToolCall("window_control", {"action": "close"}),
            "window",
            "missing_window",
        )
    if re.fullmatch(r"(?:отмени|удали)\s+напоминани\w*", normalized):
        return _validated_ask(
            registry,
            ToolCall("cancel_reminder"),
            "reminder_id",
            "missing_reminder_id",
        )

    action = route_explicit_command(normalized)
    if action is not None and action.intent == "set_reminder":
        arguments = _arguments_from_route("set_reminder", action, registry)
        has_message = bool(str(arguments.get("message", "")).strip())
        has_time = any(name in arguments for name in ("minutes", "due_at", "clock_time"))
        if has_message and not has_time:
            return _validated_ask(
                registry,
                ToolCall("set_reminder", arguments),
                "minutes",
                "missing_time",
            )
        if has_time and not has_message:
            return _validated_ask(
                registry,
                ToolCall("set_reminder", arguments),
                "message",
                "missing_reminder_text",
            )

    conversational = re.fullmatch(
        r"(?:мне\s+(?:нужно|надо)\s+не\s+забыть|не\s+дай\s+мне\s+забыть)\s+(.+)",
        normalized,
    )
    if conversational:
        message = conversational.group(1).strip(" ,.!?:;-")
        if message:
            return _validated_ask(
                registry,
                ToolCall("set_reminder", {"message": message}),
                "minutes",
                "missing_time",
            )
    return None


def _validated_ask(
    registry: ToolSchemaRegistry,
    call: ToolCall,
    missing_name: str,
    reason: str,
) -> JALPlan:
    plan = JALPlan(
        DialogueAct.ASK,
        steps=(call,),
        missing=(MissingSlot(0, missing_name),),
        reason=reason,
    )
    registry.validate(plan)
    return plan


def assemble_verified_explicit_execution(
    utterance: str,
    registry: ToolSchemaRegistry,
) -> JALPlan | None:
    """Build a plan only when deterministic routing covers the entire command."""
    normalized = normalize_russian_numbers(utterance.casefold().replace("ё", "е"))
    # Some single commands contain an internal comma before a pronoun
    # ("мне нужен Paint, открой его"). Route the complete utterance first so
    # the antecedent is not destroyed by compound splitting.
    direct = (
        route_explicit_command(normalized)
        if re.search(r"(?:открой|запусти|закрой|заверши)\s+(?:его|ее|её)$", normalized)
        else None
    )
    if direct is not None and direct.intent in registry.tool_names:
        try:
            direct_plan = JALPlan(
                DialogueAct.EXECUTE,
                steps=(
                    ToolCall(
                        direct.intent,
                        _coerce_arguments(
                            direct.intent,
                            _arguments_from_route(direct.intent, direct, registry),
                        ),
                    ),
                ),
            )
            registry.validate(direct_plan)
        except (TypeError, ValueError):
            pass
        else:
            return direct_plan
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
    if re.search(r"\b(?:все\s+|системн\w*\s+)?процесс\w*\b", normalized):
        return True
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
        action.intent
        for action in routed
        if action is not None and action.intent in registry.tool_names
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
            has_evidence = has_evidence or (
                allow_neural_evidence
                and _has_tool_lexical_evidence(tool, normalized)
            )
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
    ignored = {
        "затем",
        "потом",
        "после этого",
        "заодно",
        "и",
        "джарвис",
        "пожалуйста",
    }
    wrappers = re.compile(
        r"^(?:(?:джарвис|будь\s+добр|пожалуйста|сначала)\s*[,;:]?\s*|"
        r"(?:выполни\s+по\s+порядку|мне\s+нужно\s+следующее|одной\s+командой|"
        r"сделай\s+все\s+по\s+списку|последовательно|выполни\s+цепочку|"
        r"действуй\s+последовательно)\s*:\s*)",
        flags=re.IGNORECASE,
    )
    result: list[str] = []
    for raw in parts:
        part = raw.strip(" ,")
        previous = None
        while part and part != previous:
            previous = part
            part = wrappers.sub("", part).strip(" ,")
        if part and part.casefold() not in ignored:
            result.append(part)
    return result


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


def _has_zero_argument_lexical_evidence(tool: str, utterance: str) -> bool:
    patterns = {
        "get_current_time": r"\b(?:время|час|часы)\b",
        "list_applications": r"\b(?:приложени\w*|программ\w*)\b",
        "list_reminders": r"\bнапоминани\w*\b",
    }
    pattern = patterns.get(tool)
    return pattern is not None and re.search(pattern, utterance) is not None


def _has_tool_lexical_evidence(tool: str, utterance: str) -> bool:
    """Conservative independent grounding for neural-only tool choices."""
    patterns = {
        "open_application": r"\b(?:откро\w*|запуст\w*|включ\w*)\b",
        "window_control": r"\b(?:окн\w*|сверн\w*|разверн\w*|восстанов\w*|переключ\w*)\b",
        "browser_control": r"\b(?:браузер\w*|вкладк\w*|найд\w*|поищ\w*|загугл\w*)\b",
        "system_control": r"\b(?:громк\w*|тиш\w*|звук\w*|медиа\w*)\b",
        "file_control": r"\b(?:файл\w*|папк\w*|документ\w*|корзин\w*|удал\w*)\b",
        "workspace_control": r"\b(?:проект\w*|рабоч\w*|код\w*)\b",
        "gesture_mode": r"\bжест\w*\b",
        "set_reminder": r"\b(?:напомн\w*|напоминани\w*|не\s+забыть)\b",
        "cancel_reminder": r"\b(?:отмен\w*|удал\w*)\s+напоминани\w*\b",
    }
    if _has_zero_argument_lexical_evidence(tool, utterance):
        return True
    pattern = patterns.get(tool)
    return pattern is not None and re.search(pattern, utterance) is not None
