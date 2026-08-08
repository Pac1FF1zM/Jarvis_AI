"""Execute the six approved gesture proposals silently through Jarvis tools.

The bridge intentionally owns no camera and no neural model. Gesture actions
do not enter the conversational lifecycle: voice control remains undisturbed
and TTS never announces a gesture-triggered media key.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.base_module import BaseModule
from core.event_bus import Event, EventBus

logger = logging.getLogger("jarvis.module.gesture_bridge")


@dataclass(frozen=True)
class GestureCommand:
    text: str
    intent: str
    slots: dict[str, Any]


# Do not map pointer-like, browser or confirmation gestures. These six actions
# are reversible and already have guarded Windows-tool implementations.
GESTURE_COMMANDS: dict[str, GestureCommand] = {
    "G01": GestureCommand("переключи воспроизведение", "system_control", {"action": "media_play_pause"}),
    "G02": GestureCommand("переключи звук", "system_control", {"action": "volume_mute"}),
    "G03": GestureCommand("увеличь громкость", "system_control", {"action": "volume_up", "steps": 2}),
    "G04": GestureCommand("уменьши громкость", "system_control", {"action": "volume_down", "steps": 2}),
    "G05": GestureCommand("предыдущий трек", "system_control", {"action": "media_previous"}),
    "G06": GestureCommand("следующий трек", "system_control", {"action": "media_next"}),
}


class GestureActionBridge(BaseModule):
    """Use a completed gesture inference as a first-class Jarvis input source."""

    name = "gesture_bridge"

    def __init__(self, tools: Any) -> None:
        super().__init__(config=object())
        self.tools = tools

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("gesture_action_ready", self._on_gesture_action)
        logger.info("GestureActionBridge started safe_labels=%s", sorted(GESTURE_COMMANDS))

    async def stop(self) -> None:
        logger.info("GestureActionBridge stopped")

    async def _on_gesture_action(self, event: Event) -> None:
        if str(event.payload.get("execution", "")) != "enabled":
            return
        command = GESTURE_COMMANDS.get(str(event.payload.get("label", "")))
        if command is None:
            logger.info("GESTURE_ACTION_IGNORED label=%s reason=unmapped", event.payload.get("label"))
            return
        try:
            result = await self.tools.execute(command.intent, command.slots)
        except Exception:  # noqa: BLE001 - one media key must not stop recognition
            logger.exception(
                "GESTURE_ACTION_FAILED label=%s action=%s",
                event.payload.get("label"),
                command.slots.get("action"),
            )
            return
        logger.info(
            "GESTURE_ACTION_EXECUTED label=%s action=%s ok=%s",
            event.payload.get("label"),
            command.slots.get("action"),
            result.get("ok"),
        )
