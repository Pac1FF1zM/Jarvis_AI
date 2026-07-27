"""Conservative Windows settings, audio and media controls."""
from __future__ import annotations

import asyncio
import ctypes
import os
from typing import Any

from ._windows import send_virtual_key

_KEYS = {
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "volume_mute": 0xAD,
    "media_play_pause": 0xB3,
    "media_next": 0xB0,
    "media_previous": 0xB1,
}
_SETTINGS = {
    "settings": "ms-settings:",
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "bluetooth": "ms-settings:bluetooth",
    "network": "ms-settings:network",
    "applications": "ms-settings:appsfeatures",
    "update": "ms-settings:windowsupdate",
    "personalization": "ms-settings:personalization",
    "privacy": "ms-settings:privacy",
    "microphone": "ms-settings:privacy-microphone",
}

TOOL_SCHEMA: dict[str, Any] = {
    "name": "system_control",
    "description": "Control volume/media, open a safe Windows settings page, or lock the PC.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [*_KEYS, "open_settings", "lock"]},
            "setting": {"type": "string", "enum": list(_SETTINGS)},
            "steps": {"type": "integer", "minimum": 1, "maximum": 20},
            "confirmed": {"type": "boolean"},
        },
        "required": ["action"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).casefold()
    if action in _KEYS:
        steps = max(1, min(int(params.get("steps", 1)), 20))
        await asyncio.to_thread(lambda: [send_virtual_key(_KEYS[action]) for _ in range(steps)])
        labels = {"volume_up": "Увеличиваю громкость", "volume_down": "Уменьшаю громкость", "volume_mute": "Переключаю звук", "media_play_pause": "Переключаю воспроизведение", "media_next": "Следующий трек", "media_previous": "Предыдущий трек"}
        return {"ok": True, "response_text": labels[action] + "."}
    if action == "open_settings":
        setting = str(params.get("setting", "settings")).casefold()
        if setting not in _SETTINGS:
            return {"ok": False, "error": "unknown_setting", "response_text": "Такой раздел настроек не поддерживается."}
        await asyncio.to_thread(os.startfile, _SETTINGS[setting])  # type: ignore[attr-defined]
        return {"ok": True, "response_text": "Открываю настройки Windows."}
    if action == "lock":
        if not params.get("confirmed"):
            return {"ok": False, "confirmation_required": True, "confirmation": {"tool": "system_control", "params": {"action": "lock", "confirmed": True}}, "response_text": "Заблокировать компьютер? Скажите «подтверждаю» или «отмена»."}
        success = await asyncio.to_thread(ctypes.windll.user32.LockWorkStation)
        return {"ok": bool(success), "response_text": "Блокирую компьютер." if success else "Не удалось заблокировать компьютер."}
    return {"ok": False, "error": "unknown_action", "response_text": "Неизвестная системная команда."}

