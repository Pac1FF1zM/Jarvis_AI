"""Console response surface used by ``main.py --text``.

It deliberately mirrors the TTS lifecycle events so the orchestrator follows
the same RESPONSE -> SPEAKING -> IDLE path without loading any audio engine.
"""
from __future__ import annotations

import logging

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import SpeechFinishedPayload, SpeechStartedPayload

logger = logging.getLogger("jarvis.module.text_output")


class TextOutputModule(BaseModule):
    name = "text_output"

    def __init__(self) -> None:
        super().__init__(config=None)

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("response_ready", self._on_response)
        bus.subscribe("notification_deliver", self._on_response)
        logger.info("TextOutputModule started")

    async def stop(self) -> None:
        logger.info("TextOutputModule stopped")

    async def _on_response(self, event: Event) -> None:
        assert self.bus is not None
        text = str(event.payload.get("text", ""))
        self.bus.publish_event(
            event.child("speech_started", SpeechStartedPayload(text=text))
        )
        print(f"Jarvis: {text}", flush=True)
        self.bus.publish_event(
            event.child("speech_finished", SpeechFinishedPayload(text=text))
        )
