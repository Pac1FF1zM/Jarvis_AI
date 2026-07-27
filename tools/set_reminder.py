"""Create a durable reminder through the configured scheduler service."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("jarvis.tools.set_reminder")

TOOL_SCHEMA: dict[str, Any] = {
    "name": "set_reminder",
    "description": "Create a persistent local reminder.",
    "parameters": {
        "type": "object",
        "properties": {
            "minutes": {
                "type": "integer",
                "description": "How many minutes from now to fire the reminder.",
                "minimum": 1,
            },
            "message": {
                "type": "string",
                "description": "The reminder text to surface when it fires.",
                "minLength": 1,
            },
            "due_at": {
                "type": "string",
                "description": "Optional ISO 8601 absolute date and time.",
            },
            "clock_time": {
                "type": "string",
                "description": "Optional local HH:MM time.",
            },
            "day": {
                "type": "string",
                "description": "Optional Russian day marker: сегодня/завтра.",
                "enum": ["сегодня", "завтра"],
            },
        },
        "required": ["message"],
        "x-one-of-required": ["minutes", "due_at", "clock_time"],
        "x-mutually-exclusive": ["minutes", "due_at", "clock_time"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    """Direct imports have no runtime service; never claim work was scheduled."""
    return {
        "ok": False,
        "scheduled": False,
        "error": "scheduler_not_configured",
        "response_text": "Планировщик напоминаний не запущен.",
    }


def build_executor(services: dict[str, Any]):
    scheduler = services.get("reminder_scheduler")
    if scheduler is None:
        return execute

    async def schedule(params: dict[str, Any]) -> dict[str, Any]:
        message = str(params.get("message", "")).strip()
        try:
            if params.get("minutes") is not None:
                reminder = await scheduler.create_after(int(params["minutes"]), message)
            elif params.get("due_at"):
                reminder = await scheduler.create_at(str(params["due_at"]), message)
            elif params.get("clock_time"):
                reminder = await scheduler.create_clock_time(
                    str(params["clock_time"]),
                    message,
                    day=str(params.get("day", "")),
                )
            else:
                raise ValueError("не указано время напоминания")
        except (TypeError, ValueError) as exc:
            logger.warning("REMINDER_REJECTED params=%s error=%s", params, exc)
            return {
                "ok": False,
                "scheduled": False,
                "error": "invalid_reminder",
                "response_text": f"Не удалось создать напоминание: {exc}.",
            }
        due_local = _format_due_at(reminder.due_at)
        return {
            "ok": True,
            "scheduled": True,
            "reminder": reminder.to_dict(),
            "response_text": (
                f"Напоминание номер {reminder.id} установлено на {due_local}: "
                f"{reminder.message}."
            ),
        }

    return schedule


def _format_due_at(value: str) -> str:
    from datetime import datetime

    return datetime.fromisoformat(value).astimezone().strftime("%d.%m.%Y в %H:%M")
