"""List applications available to the safe launcher."""
from __future__ import annotations

from typing import Any

from ._applications import APPLICATIONS

TOOL_SCHEMA: dict[str, Any] = {
    "name": "list_applications",
    "description": "List local applications Jarvis is allowed to open.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    applications = [spec.display_name for spec in APPLICATIONS]
    return {
        "ok": True,
        "applications": applications,
        "response_text": "Я могу открыть: " + ", ".join(applications) + ".",
    }
