"""Translate approved gesture proposals into ordinary Jarvis command input.

The bridge intentionally owns no camera and no neural model.  It merely maps a
small allow-list of harmless, already-supported actions onto the same
wake/transcribe/think/tool lifecycle used by a voice command.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import (
    NLUResultPayload,
    TranscriptionReadyPayload,
    WakeWordDetectedPayload,
)

logger = logging.getLogger("jarvis.module.gesture_bridge")


@dataclass(frozen=True)
class GestureCommand:
    text: str
    intent: str
    slots: dict[str, Any]


# Do not map pointer-like or confirmation gestures until they have dedicated
# desktop semantics and real-camera validation.  These actions are reversible
# and already have guarded Windows tools.
GESTURE_COMMANDS: dict[str, GestureCommand] = {
    "G03": GestureCommand("увеличь громкость", "system_control", {"action": "volume_up", "steps": 2}),
    "G04": GestureCommand("уменьши громкость", "system_control", {"action": "volume_down", "steps": 2}),
    "G05": GestureCommand("предыдущий трек", "system_control", {"action": "media_previous"}),
    "G06": GestureCommand("следующий трек", "system_control", {"action": "media_next"}),
    "G08": GestureCommand("переключи воспроизведение", "system_control", {"action": "media_play_pause"}),
    "G10": GestureCommand("увеличь масштаб", "browser_control", {"action": "zoom_in"}),
    "G11": GestureCommand("уменьши масштаб", "browser_control", {"action": "zoom_out"}),
}


class GestureActionBridge(BaseModule):
    """Use a completed gesture inference as a first-class Jarvis input source."""

    name = "gesture_bridge"

    def __init__(self) -> None:
        super().__init__(config=object())
        self._pending: dict[str, GestureCommand] = {}

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("gesture_action_ready", self._on_gesture_action)
        bus.subscribe("thinking_ready", self._on_thinking_ready)
        bus.subscribe("interaction_cancelled", self._on_trace_closed)
        bus.subscribe("interaction_failed", self._on_trace_closed)
        logger.info("GestureActionBridge started safe_labels=%s", sorted(GESTURE_COMMANDS))

    async def stop(self) -> None:
        self._pending.clear()
        logger.info("GestureActionBridge stopped")

    async def _on_gesture_action(self, event: Event) -> None:
        if str(event.payload.get("execution", "")) != "enabled":
            return
        command = GESTURE_COMMANDS.get(str(event.payload.get("label", "")))
        if command is None:
            logger.info("GESTURE_ACTION_IGNORED label=%s reason=unmapped", event.payload.get("label"))
            return
        assert self.bus is not None
        wake = self.bus.publish(
            "wake_word_detected", WakeWordDetectedPayload(source="gesture")
        )
        self._pending[wake.trace_id] = command
        # This text only advances the shared lifecycle and makes the trace
        # readable in logs. NLUModule deliberately leaves source=gesture to
        # this bridge, which supplies the already policy-mapped NLU result.
        self.bus.publish(
            "transcription_ready",
            TranscriptionReadyPayload(text=command.text, confidence=1.0, source="gesture"),
            trace_id=wake.trace_id,
        )
        logger.info("GESTURE_COMMAND_ACCEPTED label=%s trace=%s", event.payload.get("label"), wake.trace_id)

    async def _on_thinking_ready(self, event: Event) -> None:
        command = self._pending.pop(event.trace_id, None)
        if command is None or self.bus is None or self.bus.is_trace_closed(event.trace_id):
            return
        self.bus.publish(
            "nlu_result",
            NLUResultPayload(
                text=command.text,
                intent=command.intent,
                slots=command.slots,
                confidence=1.0,
                raw_intent=command.intent,
                intent_confidence=1.0,
            ),
            trace_id=event.trace_id,
        )

    async def _on_trace_closed(self, event: Event) -> None:
        self._pending.pop(event.trace_id, None)
