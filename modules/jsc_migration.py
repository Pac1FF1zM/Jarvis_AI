"""Select NLU or JSC semantics according to audited migration gates."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import JALExecutionRequestedPayload, SemanticResultPayload
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, dumps, loads
from ml.jsc.legacy_adapter import jal_to_semantic_payload, nlu_payload_to_jal
from ml.jsc.migration import (
    MigrationStage,
    RollingErrorBudget,
    admit_stage,
    classify_reversibility,
)
from ml.jsc.project_registry import build_project_schema_registry

logger = logging.getLogger("jarvis.module.jsc_migration")


class JSCMigrationModule(BaseModule):
    """One semantic owner prevents NLU/JSC double execution during migration."""

    name = "jsc_migration"

    def __init__(self, config: Any, *, legacy_nlu_enabled: bool = True) -> None:
        super().__init__(config)
        params = dict(config.params or {})
        self.requested_stage = MigrationStage.parse(
            str(params.get("stage", "agreement_canary"))
        )
        self.evidence_path = Path(
            params.get("evidence_path", "data/jsc_migration_state.json")
        )
        self.log_path = Path(params.get("log_path", "logs/jsc_agreement.jsonl"))
        self.fallback_seconds = float(params.get("fallback_seconds", 0.75))
        self.registry = build_project_schema_registry()
        self.legacy_nlu_enabled = legacy_nlu_enabled
        self.active_stage = MigrationStage.AGREEMENT_CANARY
        self._nlu: dict[str, Mapping[str, Any]] = {}
        self._jsc: dict[str, Mapping[str, Any]] = {}
        self._handled: set[str] = set()
        self._selected_source: dict[str, str] = {}
        self._fallback_tasks: dict[str, asyncio.Task[None]] = {}
        self._write_lock = asyncio.Lock()
        self._budget = RollingErrorBudget(
            window=int(params.get("error_budget_window", 200)),
            minimum_agreement=float(params.get("minimum_agreement", 0.95)),
        )

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        evidence = self._load_evidence()
        admission = admit_stage(self.requested_stage, evidence)
        self.active_stage = admission.active
        if not self.legacy_nlu_enabled and self.active_stage != MigrationStage.NLU_REMOVED:
            raise RuntimeError(
                "legacy NLU may be disabled only after the nlu_removed stage is admitted"
            )
        if not admission.admitted:
            logger.warning(
                "JSC_STAGE_DEGRADED requested=%s active=%s reasons=%s",
                admission.requested.config_name,
                admission.active.config_name,
                admission.reasons,
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        bus.subscribe("nlu_result", self._on_nlu)
        bus.subscribe("jsc_candidate_ready", self._on_jsc)
        bus.subscribe("interaction_cancelled", self._on_trace_closed)
        bus.subscribe("interaction_failed", self._on_trace_closed)
        bus.subscribe("interaction_completed", self._on_trace_closed)
        logger.info(
            "JSC_MIGRATION_READY requested=%s active=%s",
            self.requested_stage.config_name,
            self.active_stage.config_name,
        )

    async def stop(self) -> None:
        tasks = list(self._fallback_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._fallback_tasks.clear()
        self._nlu.clear()
        self._jsc.clear()
        self._handled.clear()
        self._selected_source.clear()
        self.bus = None

    async def _on_nlu(self, event: Event) -> None:
        self._nlu[event.trace_id] = event.payload
        if self.active_stage == MigrationStage.INDEPENDENT_SHADOW:
            self._forward_nlu(event.trace_id, event.payload)
        elif self.active_stage == MigrationStage.AGREEMENT_CANARY:
            if event.trace_id in self._jsc:
                self._forward_nlu(event.trace_id, event.payload)
            else:
                self._schedule_fallback(event.trace_id)
        elif self.active_stage == MigrationStage.RESTRICTED_REVERSIBLE:
            if event.trace_id in self._jsc:
                await self._select_restricted(event.trace_id)
            else:
                self._schedule_fallback(event.trace_id)
        await self._compare_if_ready(event.trace_id)

    async def _on_jsc(self, event: Event) -> None:
        self._jsc[event.trace_id] = event.payload
        task = self._fallback_tasks.pop(event.trace_id, None)
        if task is not None:
            task.cancel()
        if self.active_stage == MigrationStage.AGREEMENT_CANARY:
            legacy = self._nlu.get(event.trace_id)
            if legacy is not None:
                self._forward_nlu(event.trace_id, legacy)
        elif self.active_stage == MigrationStage.RESTRICTED_REVERSIBLE:
            await self._select_restricted(event.trace_id)
        elif self.active_stage >= MigrationStage.JSC_PRIMARY:
            await self._select_jsc(event.trace_id)
            if self.active_stage == MigrationStage.NLU_REMOVED:
                await self._record_jsc_only(event.trace_id)
        await self._compare_if_ready(event.trace_id)

    async def _select_restricted(self, trace_id: str) -> None:
        if trace_id in self._handled:
            return
        candidate = self._jsc.get(trace_id)
        legacy = self._nlu.get(trace_id)
        if candidate is None:
            return
        plan = loads(str(candidate["jal"]))
        transaction = candidate.get("correction_transaction")
        execution_plan = _effective_execution_plan(plan, transaction)
        reversible = classify_reversibility(execution_plan)
        risk_ok = bool(candidate.get("accepted"))
        complete = not candidate.get("completeness_issues")
        transaction_ok = transaction is None or transaction.get("status") == "ready"
        if (
            execution_plan.act == DialogueAct.EXECUTE
            and reversible.eligible
            and risk_ok
            and complete
            and transaction_ok
        ):
            self._request_jal_execution(trace_id, candidate)
        elif legacy is not None:
            self._forward_nlu(trace_id, legacy)

    async def _select_jsc(self, trace_id: str) -> None:
        if trace_id in self._handled:
            return
        candidate = self._jsc.get(trace_id)
        if candidate is None:
            return
        plan = loads(str(candidate["jal"]))
        if plan.act == DialogueAct.EXECUTE:
            transaction = candidate.get("correction_transaction")
            transaction_ok = transaction is None or transaction.get("status") == "ready"
            if (
                not bool(candidate.get("accepted"))
                or candidate.get("completeness_issues")
                or not transaction_ok
            ):
                self._publish_jsc_semantic(
                    trace_id,
                    str(candidate["text"]),
                    JALPlan(DialogueAct.REJECT, reason="calibrated_abstention"),
                )
                return
            self._request_jal_execution(trace_id, candidate)
            return
        self._publish_jsc_semantic(trace_id, str(candidate["text"]), plan)

    def _forward_nlu(self, trace_id: str, payload: Mapping[str, Any]) -> None:
        if trace_id in self._handled or self.bus is None:
            return
        self._handled.add(trace_id)
        self._selected_source[trace_id] = "nlu"
        self.bus.publish(
            "semantic_result",
            SemanticResultPayload(
                text=str(payload.get("text", "")),
                intent=str(payload.get("intent", "unknown")),
                slots=dict(payload.get("slots") or {}),
                confidence=float(payload.get("confidence", 0.0)),
                raw_intent=payload.get("raw_intent"),
                intent_confidence=float(payload.get("intent_confidence", 0.0)),
                actions=[dict(item) for item in payload.get("actions") or ()],
                source="nlu",
            ),
            trace_id=trace_id,
        )

    def _publish_jsc_semantic(self, trace_id: str, text: str, plan: Any) -> None:
        if trace_id in self._handled or self.bus is None:
            return
        self._handled.add(trace_id)
        self._selected_source[trace_id] = "jsc"
        payload = jal_to_semantic_payload(text, plan)
        self.bus.publish(
            "semantic_result", SemanticResultPayload(**payload), trace_id=trace_id
        )

    def _request_jal_execution(
        self, trace_id: str, candidate: Mapping[str, Any]
    ) -> None:
        if trace_id in self._handled or self.bus is None:
            return
        self._handled.add(trace_id)
        self._selected_source[trace_id] = "jsc"
        self.bus.publish(
            "jal_execution_requested",
            JALExecutionRequestedPayload(
                text=str(candidate["text"]),
                jal=str(candidate["jal"]),
                migration_stage=self.active_stage.config_name,
                correction_transaction=candidate.get("correction_transaction"),
            ),
            trace_id=trace_id,
        )

    async def _compare_if_ready(self, trace_id: str) -> None:
        legacy, candidate = self._nlu.get(trace_id), self._jsc.get(trace_id)
        if legacy is None or candidate is None:
            return
        try:
            nlu_plan = nlu_payload_to_jal(legacy, self.registry)
            jsc_plan = loads(str(candidate["jal"]))
            exact = nlu_plan == jsc_plan
            unsafe = (
                jsc_plan.act == DialogueAct.EXECUTE
                and nlu_plan.act != DialogueAct.EXECUTE
            )
            budget_ok = self._budget.observe(
                exact_agreement=exact, unsafe_disagreement=unsafe
            )
            if not budget_ok and self.active_stage > MigrationStage.AGREEMENT_CANARY:
                logger.error("JSC_ERROR_BUDGET_TRIPPED fallback=agreement_canary")
                self.active_stage = MigrationStage.AGREEMENT_CANARY
            record = {
                "schema_version": 1,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "trace_id": trace_id,
                "input_source": candidate.get("input_source", "unknown"),
                "requested_stage": self.requested_stage.config_name,
                "active_stage": self.active_stage.config_name,
                "nlu_jal": dumps(nlu_plan),
                "jsc_jal": dumps(jsc_plan),
                "exact_agreement": exact,
                "unsafe_disagreement": unsafe,
                "selected_source": self._selected_source.get(trace_id, "pending"),
            }
        except Exception as exc:  # noqa: BLE001 - telemetry cannot break runtime
            logger.warning("JSC_AGREEMENT_COMPARE_FAILED trace=%s error=%s", trace_id, exc)
            return
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            await asyncio.to_thread(self._append, line)

    async def _record_jsc_only(self, trace_id: str) -> None:
        candidate = self._jsc.get(trace_id)
        if candidate is None:
            return
        record = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "input_source": candidate.get("input_source", "unknown"),
            "requested_stage": self.requested_stage.config_name,
            "active_stage": self.active_stage.config_name,
            "nlu_jal": None,
            "jsc_jal": str(candidate["jal"]),
            "exact_agreement": None,
            "unsafe_disagreement": None,
            "selected_source": self._selected_source.get(trace_id, "jsc"),
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._write_lock:
            await asyncio.to_thread(self._append, line)

    def _schedule_fallback(self, trace_id: str) -> None:
        if trace_id in self._fallback_tasks:
            return

        async def fallback() -> None:
            try:
                await asyncio.sleep(self.fallback_seconds)
                legacy = self._nlu.get(trace_id)
                if legacy is not None:
                    self._forward_nlu(trace_id, legacy)
            finally:
                self._fallback_tasks.pop(trace_id, None)

        self._fallback_tasks[trace_id] = asyncio.create_task(fallback())

    async def _on_trace_closed(self, event: Event) -> None:
        task = self._fallback_tasks.pop(event.trace_id, None)
        if task is not None:
            task.cancel()
        self._nlu.pop(event.trace_id, None)
        self._jsc.pop(event.trace_id, None)
        self._handled.discard(event.trace_id)
        self._selected_source.pop(event.trace_id, None)

    def _load_evidence(self) -> Mapping[str, Any]:
        if not self.evidence_path.is_file():
            return {}
        try:
            value = json.loads(self.evidence_path.read_text("utf-8"))
        except (OSError, ValueError):
            logger.exception("invalid JSC migration evidence file")
            return {}
        return value if isinstance(value, Mapping) else {}

    def _append(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)


def _effective_execution_plan(plan: JALPlan, transaction: Any) -> JALPlan:
    if not isinstance(transaction, Mapping) or transaction.get("status") != "ready":
        return plan
    replacement = transaction.get("replacement")
    if not isinstance(replacement, Mapping):
        return plan
    tool = str(replacement.get("tool", ""))
    arguments = replacement.get("arguments")
    if not tool or not isinstance(arguments, Mapping):
        return plan
    return JALPlan(
        DialogueAct.EXECUTE, steps=(ToolCall(tool, dict(arguments)),)
    )
