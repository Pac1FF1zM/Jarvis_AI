"""Auto-discovering tool registry.

At import time the registry scans the ``tools/`` directory for every module
(excluding ``registry.py`` itself and files starting with ``_``), imports each
one, and collects its ``TOOL_SCHEMA`` + ``execute`` callable. New tools are
added simply by dropping a new file into ``tools/`` — no manual registration.

The registry is what the LLM module asks for tool schemas, and what the tool
executor (currently inside the LLM module for this stub pass) asks for to run
a named tool.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Awaitable, Callable

logger = logging.getLogger("jarvis.tools")

# A tool's execute() is an async callable mapping params -> result dict.
ToolExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolRegistry:
    """Holds discovered tools and exposes schemas + executors by name."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._executors: dict[str, ToolExecutor] = {}

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #
    def discover(self, package_name: str = "tools") -> None:
        """Import every tool module in ``package_name`` and register it."""
        package = importlib.import_module(package_name)
        for module_info in pkgutil.iter_modules(package.__path__):
            name = module_info.name
            if name == "registry" or name.startswith("_"):
                continue
            self._load_one(f"{package_name}.{name}", name)

    def _load_one(self, dotted: str, name: str) -> None:
        try:
            module = importlib.import_module(dotted)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to import tool %s", dotted)
            return
        schema = getattr(module, "TOOL_SCHEMA", None)
        execute = getattr(module, "execute", None)
        if schema is None or execute is None:
            logger.warning(
                "Skipping %s: missing TOOL_SCHEMA or execute()", dotted
            )
            return
        tool_name = schema.get("name", name)
        self._schemas[tool_name] = schema
        self._executors[tool_name] = execute
        logger.info("Discovered tool '%s'", tool_name)

    # ------------------------------------------------------------------ #
    # API for the LLM module
    # ------------------------------------------------------------------ #
    def schemas(self) -> list[dict[str, Any]]:
        """All tool schemas, ready to hand to the LLM for function calling."""
        return list(self._schemas.values())

    def names(self) -> list[str]:
        """Sorted list of registered tool names — used for keyword matching."""
        return sorted(self._schemas.keys())

    def has(self, name: str) -> bool:
        return name in self._executors

    async def execute(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Run tool ``name`` with ``params``; raises ``KeyError`` if unknown."""
        if name not in self._executors:
            raise KeyError(f"unknown tool: {name}")
        return await self._executors[name](params or {})


def load_default_registry() -> ToolRegistry:
    """Convenience: build a registry and discover the bundled tools."""
    registry = ToolRegistry()
    registry.discover("tools")
    return registry
