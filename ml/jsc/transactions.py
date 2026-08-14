"""Side-effect-free correction transaction and compensation policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from modules.command_router import route_explicit_command
from tools._applications import resolve_application

from .jal import DialogueAct, JALPlan, ToolCall


@dataclass(frozen=True)
class ActionReceipt:
    trace_id: str
    tool: str
    params: Mapping[str, Any]
    result: Mapping[str, Any]

    @property
    def succeeded(self) -> bool:
        return self.result.get("ok") is True


@dataclass(frozen=True)
class CompensationCall:
    tool: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class CorrectionTransaction:
    status: str
    reason: str
    original_trace_id: str | None
    replacement: ToolCall | None
    compensation: CompensationCall | None
    policy: str = "compensate_then_replace_stop_on_failure"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compensation_for(receipt: ActionReceipt) -> CompensationCall | None:
    """Return a verified compensation only when the result contains evidence."""
    if not receipt.succeeded:
        return None
    params, result = dict(receipt.params), dict(receipt.result)
    if receipt.tool == "open_application" and params.get("application"):
        return CompensationCall(
            "undo_action",
            {"action": "close_application", "application": params["application"]},
        )
    reminder = result.get("reminder")
    reminder_id = (
        reminder.get("id")
        if isinstance(reminder, Mapping)
        else result.get("reminder_id")
    )
    if receipt.tool == "set_reminder" and reminder_id is not None:
        return CompensationCall(
            "cancel_reminder", {"reminder_id": int(reminder_id)}
        )
    if receipt.tool == "cancel_reminder" and isinstance(reminder, Mapping):
        return CompensationCall(
            "undo_action",
            {"action": "restore_reminder", "reminder_id": int(reminder["id"])},
        )
    if receipt.tool == "system_control":
        inverse = {
            "volume_up": "volume_down",
            "volume_down": "volume_up",
            "volume_mute": "volume_mute",
        }.get(str(params.get("action")))
        if inverse:
            return CompensationCall(
                "system_control",
                {"action": inverse, "steps": int(params.get("steps", 1))},
            )
    if receipt.tool == "browser_control":
        inverse = {"new_tab": "close_tab", "close_tab": "reopen_tab"}.get(
            str(params.get("action"))
        )
        if inverse:
            return CompensationCall("browser_control", {"action": inverse})
    if receipt.tool == "file_control":
        action = str(params.get("action"))
        if action == "rename" and result.get("path") and result.get("previous_path"):
            return CompensationCall(
                "undo_action",
                {
                    "action": "restore_rename",
                    "path": str(result["path"]),
                    "previous_path": str(result["previous_path"]),
                },
            )
        if action == "create_folder" and result.get("path"):
            return CompensationCall(
                "undo_action",
                {"action": "remove_empty_folder", "path": str(result["path"])},
            )
    undo = result.get("undo")
    if receipt.tool == "window_control" and isinstance(undo, Mapping):
        states = undo.get("states")
        if isinstance(states, list) and states:
            return CompensationCall(
                "undo_action", {"action": "restore_windows", "states": states}
            )
    if receipt.tool == "workspace_control" and result.get("undo_token"):
        return CompensationCall(
            "workspace_control",
            {"action": "undo_launch", "undo_token": str(result["undo_token"])},
        )
    return None


def plan_correction_transaction(
    text: str,
    plan: JALPlan,
    previous_receipt: ActionReceipt | None,
) -> CorrectionTransaction | None:
    """Plan correction atomically; never execute the compensation or replacement."""
    routed = route_explicit_command(text)
    normalized = text.casefold().replace("ё", "е").strip(" ,.!?:;-")
    explicit_original = None
    if routed is not None and routed.slots.get("correction_from"):
        explicit_original = str(routed.slots["correction_from"])
    instead = re.fullmatch(
        r"стоп[, ]+вместо\s+(.+?)\s+нуж(?:ен|на|но)\s+(.+)", normalized
    )
    if instead is not None:
        application = resolve_application(instead.group(1).strip(" ,"))
        explicit_original = application.name if application is not None else None
    is_contextual = re.match(
        r"(?:я\s+)?(?:имел|имела)\s+в\s+виду\b", normalized
    ) is not None
    is_explicit_plan = normalized.startswith("поправка")
    if explicit_original is None and not is_contextual and not is_explicit_plan:
        return None
    replacement = plan.steps[-1] if plan.act == DialogueAct.EXECUTE and plan.steps else None
    if replacement is None:
        return CorrectionTransaction("blocked", "replacement_not_executable", None, None, None)
    if previous_receipt is None:
        return CorrectionTransaction("blocked", "original_action_not_observed", None, replacement, None)
    expected = explicit_original or str(previous_receipt.params.get("application", ""))
    observed = str(previous_receipt.params.get("application", ""))
    if previous_receipt.tool != "open_application" or observed != expected:
        return CorrectionTransaction(
            "blocked", "original_action_mismatch", previous_receipt.trace_id, replacement, None
        )
    compensation = compensation_for(previous_receipt)
    if compensation is None:
        return CorrectionTransaction(
            "blocked", "compensation_unavailable", previous_receipt.trace_id, replacement, None
        )
    return CorrectionTransaction(
        "ready", "verified_reversible_correction", previous_receipt.trace_id, replacement, compensation
    )
