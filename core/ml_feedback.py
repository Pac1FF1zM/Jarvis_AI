"""Local, human-reviewed Active Learning queue for Jarvis NLU.

The collector never treats a model prediction or tool outcome as a ground-truth
label.  It stores only candidates that deserve review, and the training
workspace can consume examples only after a human explicitly approves them.
No audio, credentials, LLM prompts, or tool responses are persisted here.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.event_bus import Event, EventBus

logger = logging.getLogger("jarvis.ml_feedback")


@dataclass
class _Candidate:
    text: str
    predicted_intent: str
    raw_intent: str
    intent_confidence: float
    slots: dict[str, Any]
    reasons: set[str] = field(default_factory=set)


class MLFeedbackCollector:
    """Collect uncertain NLU turns and failed commands into a bounded JSONL queue."""

    def __init__(
        self,
        queue_path: str | Path,
        *,
        enabled: bool = True,
        min_intent_confidence: float = 0.72,
        capture_tool_failures: bool = True,
        max_queue_bytes: int = 5_000_000,
    ) -> None:
        if not 0.0 < min_intent_confidence <= 1.0:
            raise ValueError("min_intent_confidence must be in (0, 1]")
        if max_queue_bytes < 1024:
            raise ValueError("max_queue_bytes must be at least 1024")
        self.queue_path = Path(queue_path)
        self.enabled = bool(enabled)
        self.min_intent_confidence = float(min_intent_confidence)
        self.capture_tool_failures = bool(capture_tool_failures)
        self.max_queue_bytes = int(max_queue_bytes)
        self._pending: dict[str, _Candidate] = {}
        # EventBus dispatches subscribers concurrently. Keep terminal facts
        # briefly so a completion cannot race ahead of its NLU subscriber.
        self._terminal_outcomes: dict[str, str] = {}
        self._deferred_reasons: dict[str, set[str]] = {}
        self._write_lock = asyncio.Lock()
        self._available = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "MLFeedbackCollector":
        values = config or {}
        return cls(
            values.get("queue_path", "data/feedback/pending.jsonl"),
            enabled=bool(values.get("enabled", True)),
            min_intent_confidence=float(values.get("min_intent_confidence", 0.72)),
            capture_tool_failures=bool(values.get("capture_tool_failures", True)),
            max_queue_bytes=int(values.get("max_queue_bytes", 5_000_000)),
        )

    async def start(self, bus: EventBus) -> None:
        if not self.enabled:
            logger.info("ML feedback collection disabled by configuration")
            return
        try:
            self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("ML feedback disabled: cannot create %s: %s", self.queue_path.parent, exc)
            return
        self._available = True
        bus.subscribe("nlu_result", self._on_nlu_result)
        bus.subscribe("tool_result", self._on_tool_result)
        bus.subscribe("interaction_failed", self._on_interaction_failed)
        bus.subscribe("interaction_completed", self._on_interaction_completed)
        logger.info(
            "ML feedback ready queue=%s min_confidence=%.2f",
            self.queue_path.resolve(),
            self.min_intent_confidence,
        )

    async def stop(self) -> None:
        self._pending.clear()
        self._terminal_outcomes.clear()
        self._deferred_reasons.clear()
        self._available = False

    async def _on_nlu_result(self, event: Event) -> None:
        if not self._available:
            return
        payload = event.payload
        text = str(payload.get("text", "")).strip()
        if not text:
            return
        predicted_intent = str(payload.get("intent", "unknown"))
        raw_intent = str(payload.get("raw_intent") or predicted_intent)
        confidence = float(payload.get("intent_confidence", 0.0))
        reasons: set[str] = set()
        if confidence < self.min_intent_confidence:
            reasons.add("low_intent_confidence")
        if predicted_intent == "unknown":
            reasons.add("unknown_or_rejected_intent")
        normalized_text = text.casefold().lstrip()
        if normalized_text.startswith(("нет", "я имел в виду", "я имела в виду")):
            # A correction is valuable supervised data even when the runtime
            # recovered successfully through its deterministic guardrail.
            reasons.add("explicit_user_correction")
        candidate = _Candidate(
            text=text,
            predicted_intent=predicted_intent,
            raw_intent=raw_intent,
            intent_confidence=confidence,
            slots=dict(payload.get("slots") or {}),
            reasons=reasons,
        )
        candidate.reasons.update(self._deferred_reasons.pop(event.trace_id, set()))
        outcome = self._terminal_outcomes.pop(event.trace_id, None)
        if outcome is None:
            # A later tool failure can turn an otherwise confident input into
            # useful feedback, so retain it only for this live trace.
            self._pending[event.trace_id] = candidate
        elif candidate.reasons:
            await self._append(event.trace_id, candidate, outcome=outcome)

    async def _on_tool_result(self, event: Event) -> None:
        if not self._available or not self.capture_tool_failures:
            return
        result = event.payload.get("result")
        if not isinstance(result, Mapping):
            return
        if result.get("ok") is False and not result.get("confirmation_required"):
            candidate = self._pending.get(event.trace_id)
            if candidate is not None:
                candidate.reasons.add("tool_execution_failed")
            else:
                self._deferred_reasons.setdefault(event.trace_id, set()).add("tool_execution_failed")

    async def _on_interaction_failed(self, event: Event) -> None:
        candidate = self._pending.pop(event.trace_id, None)
        if candidate is None:
            self._remember_terminal(event.trace_id, "failed")
            return
        candidate.reasons.add("interaction_failed")
        await self._append(event.trace_id, candidate, outcome="failed")

    async def _on_interaction_completed(self, event: Event) -> None:
        outcome = "cancelled" if event.payload.get("cancelled") else "completed"
        candidate = self._pending.pop(event.trace_id, None)
        if candidate is None:
            self._remember_terminal(event.trace_id, outcome)
            return
        if candidate.reasons:
            await self._append(event.trace_id, candidate, outcome=outcome)

    async def _append(self, trace_id: str, candidate: _Candidate, *, outcome: str) -> None:
        record = {
            "schema_version": 1,
            "status": "pending_review",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "text": candidate.text,
            "predicted_intent": candidate.predicted_intent,
            "raw_intent": candidate.raw_intent,
            "intent_confidence": round(candidate.intent_confidence, 6),
            "slots": candidate.slots,
            "reasons": sorted(candidate.reasons),
            "outcome": outcome,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            try:
                if self.queue_path.exists() and self.queue_path.stat().st_size + len(line.encode("utf-8")) > self.max_queue_bytes:
                    logger.warning("ML feedback queue is full; new candidates are skipped: %s", self.queue_path)
                    return
                await asyncio.to_thread(self._append_sync, line)
            except OSError as exc:
                logger.warning("Could not append ML feedback candidate: %s", exc)

    def _append_sync(self, line: str) -> None:
        with self.queue_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)

    def _remember_terminal(self, trace_id: str, outcome: str) -> None:
        self._terminal_outcomes[trace_id] = outcome
        if len(self._terminal_outcomes) > 1024:
            self._terminal_outcomes.pop(next(iter(self._terminal_outcomes)))
