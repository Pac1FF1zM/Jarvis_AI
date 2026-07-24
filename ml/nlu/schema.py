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


@dataclass(frozen=True)
class NLUResult:
    """One model prediction returned to the event-driven runtime."""

    intent: str
    confidence: float
    slots: dict[str, str] = field(default_factory=dict)
