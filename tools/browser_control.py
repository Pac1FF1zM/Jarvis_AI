"""Use the Windows default browser for sites, search and basic tab controls."""
from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import quote_plus, urlparse

from ._windows import activate_default_browser, send_hotkey

TOOL_SCHEMA: dict[str, Any] = {
    "name": "browser_control",
    "description": "Search the web, open a site in the default browser, or send a browser tab shortcut.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "open_site", "new_tab", "close_tab", "reopen_tab", "next_tab", "previous_tab", "zoom_in", "zoom_out"]},
            "query": {"type": "string"},
            "url": {"type": "string"},
        },
        "required": ["action"],
    },
}


def _safe_web_url(value: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError("не указан адрес сайта")
    if "://" not in candidate:
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("разрешены только обычные http/https адреса")
    return candidate


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).casefold()
    if action == "search":
        query = str(params.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "missing_query", "response_text": "Что нужно найти?"}
        url = "https://www.google.com/search?q=" + quote_plus(query)
        await asyncio.to_thread(os.startfile, url)  # type: ignore[attr-defined]
        return {"ok": True, "query": query, "response_text": f"Ищу в интернете: {query}."}
    if action == "open_site":
        try:
            url = _safe_web_url(str(params.get("url", "")))
        except ValueError as exc:
            return {"ok": False, "error": "invalid_url", "response_text": str(exc)}
        await asyncio.to_thread(os.startfile, url)  # type: ignore[attr-defined]
        return {"ok": True, "url": url, "response_text": "Открываю сайт в браузере по умолчанию."}
    shortcuts = {
        "new_tab": (0x11, 0x54),
        "close_tab": (0x11, 0x57),
        "reopen_tab": (0x11, 0x10, 0x54),
        "next_tab": (0x11, 0x09),
        "previous_tab": (0x11, 0x10, 0x09),
        "zoom_in": (0x11, 0xBB),
        "zoom_out": (0x11, 0xBD),
    }
    if action not in shortcuts:
        return {"ok": False, "error": "unknown_action", "response_text": "Неизвестная команда браузера."}
    try:
        focused = await asyncio.to_thread(activate_default_browser)
    except (OSError, ValueError):
        focused = False
    if not focused:
        return {"ok": False, "error": "browser_not_open", "response_text": "Не найдено открытое окно браузера по умолчанию."}
    await asyncio.to_thread(send_hotkey, *shortcuts[action])
    labels = {"new_tab": "Открываю новую вкладку", "close_tab": "Закрываю текущую вкладку", "reopen_tab": "Восстанавливаю закрытую вкладку", "next_tab": "Перехожу на следующую вкладку", "previous_tab": "Перехожу на предыдущую вкладку", "zoom_in": "Увеличиваю масштаб", "zoom_out": "Уменьшаю масштаб"}
    return {"ok": True, "response_text": labels[action] + "."}
