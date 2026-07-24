"""Example tool: return the current local time. No parameters."""
from __future__ import annotations

from datetime import datetime
from typing import Any

TOOL_SCHEMA: dict[str, Any] = {
    "name": "get_current_time",
    "description": "Get the current local date and time. Takes no parameters.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    """Return the current local time as an ISO 8601 string."""
    now = datetime.now()
    return {
        "iso": now.isoformat(timespec="seconds"),
        "weekday": now.strftime("%A"),
        "response_text": f"Сейчас {now:%H:%M}.",
    }
