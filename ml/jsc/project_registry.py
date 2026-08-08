"""JSC capability schemas, including runtime-routed non-public actions."""
from __future__ import annotations

from copy import deepcopy

from tools.registry import ToolRegistry
from tools.workspace_control import TOOL_SCHEMA as WORKSPACE_SCHEMA

from .jal import ToolSchemaRegistry


GESTURE_MODE_SCHEMA = {
    "name": "gesture_mode",
    "description": "Enable, disable, pause, resume or inspect Jarvis gesture mode.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["enable", "disable", "pause", "resume", "status"],
            }
        },
        "required": ["action"],
    },
}


def build_project_schema_registry() -> ToolSchemaRegistry:
    """Return every action the local orchestrator can actually route."""
    tools = ToolRegistry()
    tools.discover("tools")
    schemas = list(tools.schemas())
    workspace = deepcopy(WORKSPACE_SCHEMA)
    workspace.pop("x-internal", None)
    schemas.extend((workspace, GESTURE_MODE_SCHEMA))
    return ToolSchemaRegistry(schemas)
