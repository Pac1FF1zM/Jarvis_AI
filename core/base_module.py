"""Abstract base class for every Jarvis module."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.event_bus import EventBus


class BaseModule(ABC):
    """Contract every module (STT, LLM, TTS, WakeWord, ...) must follow.

    Modules communicate **only** through the :class:`EventBus`. In ``start()``
    a module subscribes to the event types it consumes; it publishes result
    events back onto the bus. No module ever imports or calls another module.

    Subclasses set:
        name:    short identifier used for config lookup + logging.
        enabled: default enable state (config can override).
    """

    name: str = "base"
    enabled: bool = True

    def __init__(self, config: object) -> None:
        # `config` is the module-specific ModuleConfig dataclass from the loader.
        self.config = config
        self.bus: "EventBus | None" = None

    @abstractmethod
    async def start(self, bus: "EventBus") -> None:
        """Subscribe to input events and prepare any resources."""
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        """Release resources (models, connections, background tasks)."""
        raise NotImplementedError
