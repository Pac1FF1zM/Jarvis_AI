"""Hard no-action guard used by every experiment benchmark entry point."""
from __future__ import annotations


class NoActionGuard:
    """The benchmark has no executor; any attempt to use one is a hard error."""

    enabled = True

    def execute(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("NO_ACTION_MODE: experimental benchmarks cannot execute actions")

    def record_interpretation(self, *, text: str, intent: str = "", slots: object = None) -> dict[str, object]:
        return {"text": text, "intent": intent, "slots": slots or {}, "execution": "blocked"}
