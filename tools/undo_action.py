"""Internal safe inverse operations used by the dialogue undo journal."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ._windows import control_window, restore_window_states
from .file_control import _resolve_user_path


TOOL_SCHEMA: dict[str, Any] = {
    "name": "undo_action",
    "x-internal": True,
    "description": "Undo one action previously performed and journaled by Jarvis.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "close_application",
                    "restore_reminder",
                    "restore_rename",
                    "remove_empty_folder",
                    "restore_windows",
                ],
            },
            "application": {"type": "string"},
            "reminder_id": {"type": "integer"},
            "path": {"type": "string"},
            "previous_path": {"type": "string"},
        },
        "required": ["action"],
    },
}


async def execute(_params: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "error": "undo_service_unavailable",
        "response_text": "Безопасная отмена сейчас недоступна.",
    }


def build_executor(services: dict[str, Any]):
    scheduler = services.get("reminder_scheduler")

    async def undo(params: dict[str, Any]) -> dict[str, Any]:
        action = str(params.get("action", "")).casefold()
        try:
            if action == "close_application":
                application = str(params.get("application", "")).strip()
                if not application:
                    raise ValueError("не указано приложение")
                await asyncio.to_thread(control_window, "close", application)
                return {"ok": True, "response_text": f"Отменено: закрываю {application}."}
            if action == "restore_reminder":
                if scheduler is None:
                    raise ValueError("планировщик напоминаний недоступен")
                reminder_id = int(params.get("reminder_id", 0))
                reminder = await scheduler.restore(reminder_id)
                if reminder is None:
                    raise ValueError("напоминание уже нельзя восстановить")
                return {
                    "ok": True,
                    "reminder": reminder.to_dict(),
                    "response_text": f"Напоминание номер {reminder_id} восстановлено.",
                }
            if action == "restore_rename":
                current = _resolve_user_path(str(params.get("path", "")))
                previous = _resolve_user_path(
                    str(params.get("previous_path", "")), must_exist=False
                )
                if previous.exists():
                    raise ValueError("прежнее имя уже занято")
                await asyncio.to_thread(current.rename, previous)
                return {"ok": True, "response_text": f"Возвращено прежнее имя {previous.name}."}
            if action == "remove_empty_folder":
                path = _resolve_user_path(str(params.get("path", "")))
                if not path.is_dir():
                    raise ValueError("созданная папка больше не существует")
                if any(path.iterdir()):
                    raise ValueError("папка уже содержит файлы и не будет удалена")
                await asyncio.to_thread(Path.rmdir, path)
                return {"ok": True, "response_text": f"Создание папки {path.name} отменено."}
            if action == "restore_windows":
                states = params.get("states") or []
                if not isinstance(states, list):
                    raise ValueError("повреждён снимок окон")
                restored = await asyncio.to_thread(restore_window_states, states)
                if restored == 0:
                    raise ValueError("окна уже закрыты или недоступны")
                return {
                    "ok": True,
                    "response_text": f"Предыдущее расположение восстановлено, окон: {restored}.",
                }
        except (OSError, TypeError, ValueError) as exc:
            return {"ok": False, "error": "undo_failed", "response_text": str(exc)}
        return {"ok": False, "error": "unknown_undo", "response_text": "Это действие нельзя отменить."}

    return undo
