"""Honest boundary for reminders until persistent delivery is implemented."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("jarvis.tools.set_reminder")

TOOL_SCHEMA: dict[str, Any] = {
    "name": "set_reminder",
    "description": "Report that persistent reminders are not available yet.",
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {
                "type": "integer",
                "description": "How many minutes from now to fire the reminder.",
            },
            "message": {
                "type": "string",
                "description": "The reminder text to surface when it fires.",
            },
        },
        "required": ["minutes", "message"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    """Reject clearly: this one-shot process has no persistent scheduler."""
    minutes = int(params.get("minutes", 0))
    message = str(params.get("message", ""))
    logger.warning("REMINDER_REJECTED in=%dmin msg=%r scheduler unavailable", minutes, message)
    return {
        "ok": False,
        "scheduled": False,
        "error": "scheduler_not_implemented",
        "minutes": minutes,
        "message": message,
        "response_text": (
            "Я пока не умею надёжно доставлять напоминания после завершения "
            "программы, поэтому ничего не запланировал."
        ),
    }
