"""Profile-scoped workspace actions backed by native Windows APIs."""
from __future__ import annotations

import asyncio
from typing import Any


TOOL_SCHEMA: dict[str, Any] = {
    "name": "workspace_control",
    "description": "Capture, launch, list or finish a Jarvis Windows workspace.",
    # The existing learned JAL/NLU schema is intentionally frozen until the
    # agreed final retraining stage. Deterministic routing can use this tool now.
    "x-internal": True,
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["launch", "capture", "list", "finish", "undo_launch"],
            },
            "workspace": {"type": "string"},
            "confirmed": {"type": "boolean"},
            "undo_token": {"type": "string"},
        },
        "required": ["action"],
    },
}


def build_executor(services: dict[str, Any]):
    manager = services.get("workspace_manager")

    async def execute(params: dict[str, Any]) -> dict[str, Any]:
        if manager is None:
            return {
                "ok": False,
                "error": "workspace_service_unavailable",
                "response_text": "Управление рабочими пространствами сейчас недоступно.",
            }
        action = str(params.get("action", "")).casefold()
        name = str(params.get("workspace", "")).strip()
        try:
            if action == "list":
                rows = await asyncio.to_thread(manager.list_workspaces)
                labels = ", ".join(str(row.get("name", "")) for row in rows)
                return {
                    "ok": True,
                    "workspaces": rows,
                    "response_text": "Доступные рабочие пространства: " + labels + ".",
                }
            if action == "launch":
                if not name:
                    return {
                        "ok": False,
                        "error": "missing_workspace",
                        "response_text": "Какой режим запустить?",
                    }
                return await asyncio.to_thread(
                    manager.launch,
                    name,
                    confirmed=bool(params.get("confirmed", False)),
                )
            if action == "capture":
                if not name:
                    return {
                        "ok": False,
                        "error": "missing_workspace",
                        "response_text": "Как назвать это расположение?",
                    }
                return await asyncio.to_thread(manager.capture, name)
            if action == "finish":
                return await asyncio.to_thread(manager.finish)
            if action == "undo_launch":
                token = str(params.get("undo_token", ""))
                return await asyncio.to_thread(manager.undo_launch, token)
        except (OSError, RuntimeError, ValueError) as exc:
            return {
                "ok": False,
                "error": "workspace_error",
                "response_text": f"Не удалось выполнить действие с рабочим пространством: {exc}",
            }
        return {
            "ok": False,
            "error": "unknown_action",
            "response_text": "Неизвестное действие с рабочим пространством.",
        }

    return execute
