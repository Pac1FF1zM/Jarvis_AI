"""Cancel one pending reminder by its stable numeric id."""
from __future__ import annotations

from typing import Any


TOOL_SCHEMA: dict[str, Any] = {
    "name": "cancel_reminder",
    "description": "Cancel one pending reminder by id.",
    "parameters": {
        "type": "object",
        "properties": {
            "reminder_id": {
                "type": "integer",
                "description": "Numeric reminder id shown by list_reminders.",
                "minimum": 1,
            }
        },
        "required": ["reminder_id"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "scheduler_not_configured",
        "response_text": "Планировщик напоминаний не запущен.",
    }


def build_executor(services: dict[str, Any]):
    scheduler = services.get("reminder_scheduler")
    if scheduler is None:
        return execute

    async def cancel(params: dict[str, Any]) -> dict[str, Any]:
        try:
            reminder_id = int(params.get("reminder_id", 0))
            reminder = await scheduler.cancel(reminder_id)
        except (TypeError, ValueError) as exc:
            return {
                "ok": False,
                "error": "invalid_reminder_id",
                "response_text": f"Некорректный номер напоминания: {exc}.",
            }
        if reminder is None:
            return {
                "ok": False,
                "error": "reminder_not_pending",
                "reminder_id": reminder_id,
                "response_text": (
                    f"Активное напоминание номер {reminder_id} не найдено."
                ),
            }
        return {
            "ok": True,
            "cancelled": True,
            "reminder": reminder.to_dict(),
            "response_text": f"Напоминание номер {reminder_id} отменено.",
        }

    return cancel
