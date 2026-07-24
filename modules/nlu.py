"""Jarvis-owned neural language-understanding module.

Consumes ``transcription_ready`` and publishes ``nlu_result``.  The checkpoint
contains its own vocabulary and neural weights trained from random
initialisation by :mod:`ml.nlu.train`; no downloaded model or tokenizer is
used.  Loading and inference run off the asyncio event-loop thread.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.gpu_lock import GPULock
from ml.nlu.inference import NLUPredictor
from ml.nlu.schema import NLUResult

logger = logging.getLogger("jarvis.module.nlu")


class NLUModule(BaseModule):
    name = "nlu"
    enabled = True

    def __init__(
        self,
        config: Any,
        gpu_lock: GPULock,
        predictor_factory: Callable[..., Any] = NLUPredictor,
    ) -> None:
        super().__init__(config)
        self.gpu_lock = gpu_lock
        self._predictor_factory = predictor_factory
        self._predictor: Any = None
        self._threshold = float(config.params.get("confidence_threshold", 0.55))
        self._pending_transcriptions: dict[str, Event] = {}
        self._thinking_ready: set[str] = set()
        self._pending_lock = asyncio.Lock()

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("transcription_ready", self._on_transcription)
        bus.subscribe("thinking_ready", self._on_thinking_ready)
        checkpoint = Path(self.config.model or "models/nlu_word_bigru_curriculum.pt")
        if not checkpoint.is_file():
            logger.error(
                "NLU checkpoint not found at %s; run `python -m ml.nlu.train`",
                checkpoint,
            )
        else:
            self._predictor = await asyncio.to_thread(
                self._predictor_factory, checkpoint, self.config.device
            )
        logger.info(
            "NLUModule started mode=%s checkpoint=%s threshold=%.2f",
            "neural" if self._predictor is not None else "safe-unknown",
            checkpoint,
            self._threshold,
        )

    async def stop(self) -> None:
        async with self._pending_lock:
            self._pending_transcriptions.clear()
            self._thinking_ready.clear()
        self._predictor = None
        logger.info("NLUModule stopped")

    async def _on_transcription(self, event: Event) -> None:
        await self._stage_event(transcription=event)

    async def _on_thinking_ready(self, event: Event) -> None:
        await self._stage_event(thinking_event=event)

    async def _stage_event(
        self,
        *,
        transcription: Event | None = None,
        thinking_event: Event | None = None,
    ) -> None:
        """Run inference only after both sides of the lifecycle barrier arrive."""
        trace_id = (transcription or thinking_event).trace_id  # type: ignore[union-attr]
        ready: Event | None = None
        async with self._pending_lock:
            if transcription is not None:
                self._pending_transcriptions[trace_id] = transcription
            if thinking_event is not None:
                self._thinking_ready.add(trace_id)
            if trace_id in self._pending_transcriptions and trace_id in self._thinking_ready:
                ready = self._pending_transcriptions.pop(trace_id)
                self._thinking_ready.remove(trace_id)
        if ready is not None:
            await self._process_transcription(ready)

    async def _process_transcription(self, event: Event) -> None:
        assert self.bus is not None
        text = str(event.payload.get("text", "")).strip()
        if self._predictor is None or not text:
            result = NLUResult("unknown", 0.0, {})
        else:
            try:
                async with self.gpu_lock.section("nlu"):
                    result = await asyncio.to_thread(self._predictor.predict, text)
            except Exception:  # noqa: BLE001 - keep voice pipeline responsive
                logger.exception("NLU inference failed; rejecting turn as unknown")
                result = NLUResult("unknown", 0.0, {})

        raw_intent = result.intent
        accepted_intent = raw_intent if result.confidence >= self._threshold else "unknown"
        output = event.child(
            "nlu_result",
            {
                "text": text,
                "confidence": event.payload.get("confidence", 0.0),
                "intent": accepted_intent,
                "raw_intent": raw_intent,
                "intent_confidence": result.confidence,
                "slots": result.slots if accepted_intent != "unknown" else {},
            },
        )
        self.bus.publish_event(output)
        logger.info(
            "NLU_RESULT trace=%s intent=%s raw=%s conf=%.3f slots=%s",
            event.trace_id,
            accepted_intent,
            raw_intent,
            result.confidence,
            output.payload["slots"],
        )
