"""Jarvis tools package.

Each tool module in this directory (except :mod:`tools.registry` and files
starting with ``_``) exposes:

- ``TOOL_SCHEMA: dict`` — name, description, JSON-schema parameters, compatible
  with Ollama's tool-calling format.
- ``async def execute(params: dict) -> dict`` — the tool's implementation.

Tools are auto-discovered by :class:`tools.registry.ToolRegistry`; no manual
registration is needed.
"""
