"""Shared labels and result types for the Jarvis NLU task."""
from __future__ import annotations

from dataclasses import dataclass, field


INTENTS = (
    "get_current_time",
    "set_reminder",
    "open_application",
    "list_applications",
    "cancel",
    "general_chat",
    "unknown",
)

SLOT_LABELS = (
    "O",
    "B-duration",
    "I-duration",
    "B-reminder_text",
    "I-reminder_text",
    "B-application",
    "I-application",
)

# Slots are part of the command contract, not free-form model output.  The
# mapping is shared by training, evaluation, and runtime decoding so those
# three stages cannot silently disagree about which arguments an intent may
# produce.
INTENT_SLOTS: dict[str, frozenset[str]] = {
    "get_current_time": frozenset(),
    "set_reminder": frozenset({"minutes", "reminder_text"}),
    "open_application": frozenset({"application"}),
    "list_applications": frozenset(),
    "cancel": frozenset(),
    "general_chat": frozenset(),
    "unknown": frozenset(),
}

ACTIONABLE_INTENTS = frozenset({
    "get_current_time",
    "set_reminder",
    "open_application",
    "list_applications",
    "cancel",
})


@dataclass(frozen=True)
class NLUResult:
    """One model prediction returned to the event-driven runtime."""

    intent: str
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)
