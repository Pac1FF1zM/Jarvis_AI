"""Short-term conversational memory — a rolling window of recent turns.

Kept in-process (a plain list). N is configurable via ``config.memory`` /
constructor; older turns drop off the front. The list is exposed via
:meth:`as_context` in the standard OpenAI-style ``[{role, content}]`` shape so
it can be passed straight into an LLM prompt.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class ShortTermMemory:
    """Bounded FIFO of conversation turns."""

    def __init__(self, max_turns: int = 8) -> None:
        # A turn = one user message + one assistant message; we store each as
        # its own entry so the window is measured in messages, not pairs.
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._max_turns = max_turns
        self._turns: deque[dict[str, str]] = deque(maxlen=max_turns)

    @classmethod
    def from_config(cls, memory_config: dict[str, Any]) -> "ShortTermMemory":
        """Build from the ``memory:`` block of config.yaml."""
        return cls(max_turns=int(memory_config.get("short_term_turns", 8)))

    def add(self, role: str, text: str) -> None:
        """Append a turn. ``role`` is ``"user"`` / ``"assistant"`` / ``"system"``."""
        self._turns.append({"role": role, "content": text})

    def as_context(self) -> list[dict[str, str]]:
        """Return the current window as a list of ``{role, content}`` dicts."""
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
