"""List fixed and Windows-registered applications available to the launcher."""
from __future__ import annotations

from typing import Any

from ._applications import available_applications

TOOL_SCHEMA: dict[str, Any] = {
    "name": "list_applications",
    "description": "List local applications Jarvis is allowed to open.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    applications = [spec.display_name for spec in available_applications()]
    # Keep the complete machine-readable list in the result, but do not make
    # the user wait while TTS reads dozens of Start-menu entries aloud.
    spoken = applications[:7]
    remainder = len(applications) - len(spoken)
    suffix = f" И ещё {remainder}; назовите нужное приложение." if remainder else "."
    return {
        "ok": True,
        "applications": applications,
        "response_text": "Я могу открыть: " + ", ".join(spoken) + suffix,
    }
