"""Side-effect-free Structured JSC observer for production voice traffic."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import JSCCandidateReadyPayload
from ml.jsc.data import DialogueTurn
from ml.jsc.inference import StructuredJSCPredictor
from ml.jsc.jal import DialogueAct, JALPlan, dumps, loads
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.structured_decoding import plan_completeness_issues
from ml.jsc.transactions import ActionReceipt, plan_correction_transaction

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
        self._history: list[DialogueTurn] = []
        self._pending_state: JALPlan | None = None
        self._dialogue_id: str | None = None
        self._candidates: dict[str, dict[str, Any]] = {}
        self._tool_requests: dict[str, tuple[str, Mapping[str, Any]]] = {}
        self._last_receipt: ActionReceipt | None = None
        self._checkpoint_sha256: str | None = None
        self._migration_stage = str(
            config.params.get("migration_stage", "independent_shadow")
        )
        if self._migration_stage != "independent_shadow":
            raise ValueError("only the non-executing independent_shadow stage is enabled")
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
        self._checkpoint_sha256 = await asyncio.to_thread(
            lambda: hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        bus.subscribe("transcription_ready", self._on_transcription_ready)
        bus.subscribe("nlu_result", self._on_nlu_result)
        bus.subscribe("tool_call_requested", self._on_tool_call_requested)
        bus.subscribe("tool_result", self._on_tool_result)
        bus.subscribe("cancel_requested", self._on_dialogue_reset)
        bus.subscribe("session_sleep_requested", self._on_dialogue_reset)
        bus.subscribe("interaction_failed", self._on_dialogue_reset)
        logger.info(
            "JSC_SHADOW_READY checkpoint=%s log=%s",
            checkpoint.resolve(),
            self._log_path.resolve(),
        )

    async def stop(self) -> None:
        self._predictor = None
        self._history.clear()
        self._pending_state = None
        self._dialogue_id = None
        self._candidates.clear()
        self._tool_requests.clear()
        self._last_receipt = None
        self._checkpoint_sha256 = None
        self.bus = None

    async def _on_transcription_ready(self, event: Event) -> None:
        """Run JSC directly from finalized STT, independently of production NLU."""
        candidate = await self._predict_candidate(event)
        if candidate is None:
            return
        self._candidates[event.trace_id] = candidate
        assert self.bus is not None
        risk = dict(candidate["prediction"].risk)
        self.bus.publish_event(
            event.child(
                "jsc_candidate_ready",
                JSCCandidateReadyPayload(
                    text=candidate["text"],
                    jal=candidate["prediction"].jal,
                    accepted=bool(risk.get("accepted", False)),
                    risk=risk,
                ),
            )
        )

    async def _on_nlu_result(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if self._predictor is None or not text:
            return
        candidate = self._candidates.pop(event.trace_id, None)
        if candidate is None:
            candidate = await self._predict_candidate(event)
        if candidate is None:
            return
        prediction = candidate["prediction"]
        plan = candidate["plan"]
        history_before = candidate["history_before"]
        state_before = candidate["state_before"]
        dialogue_id = candidate["dialogue_id"]
        transaction = plan_correction_transaction(text, plan, self._last_receipt)
        completeness = plan_completeness_issues(text, plan)
        record = {
            "schema_version": 3,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "trace_id": event.trace_id,
            "source": "live_voice_shadow",
            "migration_stage": self._migration_stage,
            "checkpoint_sha256": self._checkpoint_sha256,
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
                "risk": _json_value(prediction.risk),
                "completeness_issues": list(completeness),
                "latency_ms": round(float(prediction.latency_ms), 3),
            },
            "correction_transaction": transaction.to_dict() if transaction else None,
            "dialogue": {
                "dialogue_id": dialogue_id,
                "history_before": [
                    {"role": turn.role, "text": turn.text}
                    for turn in history_before
                ],
                "state_before": dumps(state_before) if state_before is not None else None,
                "state_after": (
                    dumps(self._pending_state)
                    if self._pending_state is not None
                    else None
                ),
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

    async def _predict_candidate(self, event: Event) -> dict[str, Any] | None:
        predictor = self._predictor
        text = str(event.payload.get("text", "")).strip()
        if predictor is None or not text:
            return None
        history_before = tuple(self._history)
        state_before = self._pending_state
        dialogue_id = self._dialogue_id or event.trace_id
        try:
            prediction = await asyncio.to_thread(
                predictor.predict,
                text,
                history=history_before,
                state=state_before,
            )
        except Exception:  # noqa: BLE001 - shadow must never affect production
            logger.exception("JSC_SHADOW_FAILED trace=%s", event.trace_id)
            return
        try:
            plan = loads(prediction.jal)
        except (TypeError, ValueError):
            logger.error("JSC_SHADOW_INVALID_JAL trace=%s", event.trace_id)
            return None
        self._update_dialogue(event.trace_id, text, plan)
        return {
            "text": text,
            "prediction": prediction,
            "plan": plan,
            "history_before": history_before,
            "state_before": state_before,
            "dialogue_id": dialogue_id,
        }

    async def _on_tool_call_requested(self, event: Event) -> None:
        tool = str(event.payload.get("tool", ""))
        params = event.payload.get("params")
        if tool and isinstance(params, Mapping):
            self._tool_requests[event.trace_id] = (tool, params)

    async def _on_tool_result(self, event: Event) -> None:
        requested = self._tool_requests.pop(event.trace_id, None)
        result = event.payload.get("result")
        if requested is None or not isinstance(result, Mapping):
            return
        tool, params = requested
        receipt = ActionReceipt(event.trace_id, tool, dict(params), dict(result))
        if receipt.succeeded:
            self._last_receipt = receipt

    async def _on_dialogue_reset(self, _event: Event) -> None:
        self._history.clear()
        self._pending_state = None
        self._dialogue_id = None
        self._candidates.clear()

    def _update_dialogue(self, trace_id: str, text: str, plan: JALPlan) -> None:
        self._history.append(DialogueTurn("user", text))
        if plan.act == DialogueAct.ASK:
            self._pending_state = plan
            self._dialogue_id = self._dialogue_id or trace_id
            self._history.append(DialogueTurn("jarvis", _clarification_prompt(plan)))
        elif plan.act == DialogueAct.CONFIRM:
            self._pending_state = plan
            self._dialogue_id = self._dialogue_id or trace_id
            self._history.append(DialogueTurn("jarvis", "Подтвердите действие."))
        elif plan.act in {DialogueAct.EXECUTE, DialogueAct.CANCEL}:
            self._pending_state = None
            self._dialogue_id = None
        if len(self._history) > 8:
            del self._history[:-8]

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


def _clarification_prompt(plan: JALPlan) -> str:
    prompts = {
        "missing_application": "Какое приложение открыть?",
        "missing_window": "Какое окно или приложение закрыть?",
        "missing_time": "Когда вам напомнить?",
        "missing_reminder_text": "О чём вам напомнить?",
        "missing_reminder_id": "Назовите номер напоминания.",
    }
    return prompts.get(plan.reason or "", "Уточните недостающие детали.")
