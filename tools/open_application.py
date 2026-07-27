"""Open a fixed or Windows-registered application without a command shell."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ._applications import available_applications, launch_application, resolve_application

logger = logging.getLogger("jarvis.tools.open_application")

TOOL_SCHEMA: dict[str, Any] = {
    "name": "open_application",
    "description": "Open one supported local Windows application.",
    "parameters": {
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": "Application name requested by the user.",
            },
        },
        "required": ["application"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    requested = str(params.get("application", "")).strip()
    spec = resolve_application(requested)
    if spec is None:
        supported = [item.display_name for item in available_applications()]
        return {
            "ok": False,
            "error": "application_not_allowed",
            "application": requested,
            "supported": supported,
            "response_text": (
                f"Не удалось найти установленное приложение «{requested}»."
            ),
        }

    try:
        pid = await asyncio.to_thread(launch_application, spec)
    except (OSError, RuntimeError) as exc:
        logger.warning("Failed to open %s: %s", spec.name, exc)
        return {
            "ok": False,
            "error": "launch_failed",
            "application": spec.name,
            "response_text": f"Не удалось открыть {spec.display_name}: {exc}",
        }

    logger.info("APPLICATION_OPENED name=%s pid=%s", spec.name, pid)
    return {
        "ok": True,
        "application": spec.name,
        "display_name": spec.display_name,
        "pid": pid,
        "response_text": f"Открываю {spec.display_name}.",
    }
