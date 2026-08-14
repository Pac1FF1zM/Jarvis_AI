"""Canonical adapters between legacy NLU payloads and typed JAL."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .jal import DialogueAct, JALPlan, MissingSlot, ToolCall, ToolSchemaRegistry, dumps


def nlu_payload_to_jal(
    payload: Mapping[str, Any], registry: ToolSchemaRegistry
) -> JALPlan:
    """Convert a production NLU result to its closest canonical JAL plan."""
    actions = payload.get("actions") or ()
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)) and actions:
        calls = tuple(
            call
            for action in actions
            if isinstance(action, Mapping)
            for call in [_action_to_call(str(action.get("intent", "")), action.get("slots") or {})]
            if call is not None
        )
        if len(calls) == len(actions):
            return _validated_execution(calls, registry)
    intent = str(payload.get("intent", "unknown"))
    slots = payload.get("slots") or {}
    if not isinstance(slots, Mapping):
        slots = {}
    call = _action_to_call(intent, slots)
    if call is not None:
        missing = _required_missing(call, registry)
        if missing:
            return JALPlan(
                DialogueAct.ASK,
                steps=(call,),
                missing=missing,
                reason=f"missing_{missing[0].name}",
            )
        return _validated_execution((call,), registry)
    if intent in {"general_chat", "wake_greeting"}:
        return JALPlan(DialogueAct.DIALOGUE, reason="general_chat")
    if intent in {"cancel", "decline"}:
        return JALPlan(DialogueAct.CANCEL)
    if intent == "negated_command":
        return JALPlan(DialogueAct.REJECT, reason="negated_command")
    return JALPlan(DialogueAct.REJECT, reason="unsupported_tool")


def jal_to_semantic_payload(text: str, plan: JALPlan) -> dict[str, Any]:
    """Bridge non-executing JAL acts to the existing dialogue response module."""
    base: dict[str, Any] = {
        "text": text,
        "confidence": 1.0,
        "intent_confidence": 1.0,
        "raw_intent": f"jal:{plan.act.value}",
        "source": "jsc",
        "jal": dumps(plan),
        "actions": [],
        "slots": {},
    }
    if plan.act == DialogueAct.DIALOGUE:
        base["intent"] = "general_chat"
    elif plan.act == DialogueAct.REJECT:
        base["intent"] = "unknown"
    elif plan.act == DialogueAct.CANCEL:
        base["intent"] = "cancel"
    elif plan.act == DialogueAct.CONFIRM:
        base["intent"] = "confirm"
    elif plan.act == DialogueAct.ASK and len(plan.steps) == 1:
        call = plan.steps[0]
        base["intent"] = call.tool
        base["slots"] = _legacy_slots(call)
    else:
        base["intent"] = "unknown"
    return base


def _action_to_call(intent: str, slots_value: Mapping[str, Any]) -> ToolCall | None:
    slots = dict(slots_value)
    if intent in {"get_current_time", "list_applications", "list_reminders"}:
        return ToolCall(intent)
    if intent == "open_application":
        return ToolCall(intent, _only(slots, "application"))
    if intent == "set_reminder":
        arguments: dict[str, Any] = {}
        message = slots.get("message", slots.get("reminder_text"))
        if message is not None:
            arguments["message"] = message
        arguments.update(_only(slots, "minutes", "due_at", "clock_time", "day"))
        return ToolCall(intent, arguments)
    if intent == "cancel_reminder":
        return ToolCall(intent, _only(slots, "reminder_id"))
    if intent == "gesture_mode":
        action = slots.get("action")
        if action is None and "enabled" in slots:
            action = "enable" if slots["enabled"] else "disable"
        return ToolCall(intent, {"action": action} if action else {})
    if intent in {
        "browser_control",
        "system_control",
        "window_control",
        "file_control",
        "workspace_control",
    }:
        return ToolCall(intent, slots)
    return None


def _legacy_slots(call: ToolCall) -> dict[str, Any]:
    slots = dict(call.arguments)
    if call.tool == "set_reminder" and "message" in slots:
        slots["reminder_text"] = slots.pop("message")
    return slots


def _only(values: Mapping[str, Any], *names: str) -> dict[str, Any]:
    return {name: values[name] for name in names if name in values}


def _required_missing(
    call: ToolCall, registry: ToolSchemaRegistry
) -> tuple[MissingSlot, ...]:
    return tuple(
        MissingSlot(0, name)
        for name in registry.required_arguments(call.tool)
        if name not in call.arguments
    )


def _validated_execution(
    calls: tuple[ToolCall, ...], registry: ToolSchemaRegistry
) -> JALPlan:
    plan = JALPlan(DialogueAct.EXECUTE, steps=calls)
    registry.validate(plan)
    return plan
