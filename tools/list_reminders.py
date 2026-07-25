"""List pending reminders for the active local profile."""
from __future__ import annotations

from datetime import datetime
from typing import Any


TOOL_SCHEMA: dict[str, Any] = {
    "name": "list_reminders",
    "description": "List pending local reminders.",
    "parameters": {"type": "object", "properties": {}, "required": []},
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

    async def list_pending(params: dict[str, Any]) -> dict[str, Any]:
        reminders = await scheduler.list_pending()
        if not reminders:
            return {
                "ok": True,
                "reminders": [],
                "response_text": "У вас нет активных напоминаний.",
            }
        lines = []
        for reminder in reminders:
            due = datetime.fromisoformat(reminder.due_at).astimezone()
            lines.append(
                f"номер {reminder.id}, {due:%d.%m.%Y в %H:%M}: {reminder.message}"
            )
        return {
            "ok": True,
            "reminders": [reminder.to_dict() for reminder in reminders],
            "response_text": "Активные напоминания: " + "; ".join(lines) + ".",
        }

    return list_pending
