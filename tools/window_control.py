"""List and control visible desktop windows without command-shell access."""
from __future__ import annotations

import asyncio
from typing import Any

from ._windows import control_window, list_windows

TOOL_SCHEMA: dict[str, Any] = {
    "name": "window_control",
    "description": "List, switch, minimize, maximize, restore or close a visible window.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "switch", "minimize", "maximize", "restore", "close"]},
            "window": {"type": "string"},
        },
        "required": ["action"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).casefold()
    if action == "list":
        try:
            rows = await asyncio.to_thread(list_windows)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": "window_error", "response_text": str(exc)}
        titles = [row.title for row in rows[:20]]
        return {"ok": True, "windows": titles, "response_text": "Открытые окна: " + "; ".join(titles) + "."}
    query = str(params.get("window", "")).strip()
    if not query:
        return {"ok": False, "error": "missing_window", "response_text": "Назовите окно."}
    try:
        window = await asyncio.to_thread(control_window, action, query)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": "window_error", "response_text": str(exc)}
    verbs = {"switch": "Переключаюсь на", "minimize": "Сворачиваю", "maximize": "Разворачиваю", "restore": "Восстанавливаю", "close": "Закрываю"}
    return {"ok": True, "window": window.title, "response_text": f"{verbs[action]} {window.title}."}
