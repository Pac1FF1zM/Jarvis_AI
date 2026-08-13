"""Side-effect-free Structured JSC observer for production voice traffic."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from ml.jsc.inference import StructuredJSCPredictor
from ml.jsc.project_registry import build_project_schema_registry

logger = logging.getLogger("jarvis.module.jsc_shadow")


class JSCShadowModule(BaseModule):
    """Compare JSC with deployed NLU without publishing executable events."""

    name = "jsc_shadow"

    def __init__(
        self,
        config: Any,
        *,
        predictor_factory: Callable[..., Any] = StructuredJSCPredictor,
    ) -> None:
        super().__init__(config)
        self._predictor_factory = predictor_factory
        self._predictor: Any | None = None
        self._write_lock = asyncio.Lock()
        self._log_path = Path(
            config.params.get("log_path", "logs/jsc_shadow.jsonl")
        )

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        checkpoint = Path(self.config.model)
        if not checkpoint.is_file():
            logger.warning(
                "JSC shadow disabled: checkpoint not found at %s", checkpoint
            )
            return
        thresholds = dict(self.config.params.get("thresholds") or {})
        registry = build_project_schema_registry()
        self._predictor = await asyncio.to_thread(
            self._predictor_factory,
            checkpoint,
            registry,
            device=self.config.device,
            thresholds=thresholds,
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        bus.subscribe("nlu_result", self._on_nlu_result)
        logger.info(
            "JSC_SHADOW_READY checkpoint=%s log=%s",
            checkpoint.resolve(),
            self._log_path.resolve(),
        )

    async def stop(self) -> None:
        self._predictor = None
        self.bus = None

    async def _on_nlu_result(self, event: Event) -> None:
        predictor = self._predictor
        text = str(event.payload.get("text", "")).strip()
        if predictor is None or not text:
            return
        try:
            prediction = await asyncio.to_thread(predictor.predict, text)
        except Exception:  # noqa: BLE001 - shadow must never affect production
            logger.exception("JSC_SHADOW_FAILED trace=%s", event.trace_id)
            return
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "trace_id": event.trace_id,
            "text": text,
            "production_nlu": {
                "intent": event.payload.get("intent"),
                "raw_intent": event.payload.get("raw_intent"),
                "confidence": event.payload.get("intent_confidence"),
                "slots": _json_value(event.payload.get("slots") or {}),
                "actions": _json_value(event.payload.get("actions", ())),
            },
            "jsc": {
                "jal": prediction.jal,
                "decisions": dict(prediction.decisions),
                "latency_ms": round(float(prediction.latency_ms), 3),
            },
            "executed_by_jsc": False,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            await asyncio.to_thread(self._append, line)
        logger.info(
            "JSC_SHADOW_RESULT trace=%s latency_ms=%.2f jal=%s",
            event.trace_id,
            prediction.latency_ms,
            prediction.jal,
        )

    def _append(self, line: str) -> None:
        with self._log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)


def _json_value(value: Any) -> Any:
    """Copy immutable event payload containers into JSON-native values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value
