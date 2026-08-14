"""Transactional executor for migration-authorized JAL plans."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import (
    JALActionCommittedPayload,
    ResponseReadyPayload,
    ToolCallRequestedPayload,
    ToolResultPayload,
)
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, loads
from ml.jsc.migration import MigrationStage, classify_reversibility
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.transactions import ActionReceipt, CompensationCall, compensation_for
from tools.registry import ToolRegistry

logger = logging.getLogger("jarvis.module.jal_executor")

_READ_ONLY = frozenset({"get_current_time", "list_applications", "list_reminders"})


class JALExecutorModule(BaseModule):
    """Execute only coordinator-authorized, reversible JAL transactions."""

    name = "jal_executor"

    def __init__(self, config: Any, tools: ToolRegistry) -> None:
        super().__init__(config)
        self.tools = tools
        self.registry = build_project_schema_registry()
        self._active: dict[str, asyncio.Task[dict[str, Any]]] = {}

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("jal_execution_requested", self._on_execution_requested)
        bus.subscribe("interaction_cancelled", self._on_trace_closed)
        bus.subscribe("interaction_failed", self._on_trace_closed)
        logger.info("JAL_EXECUTOR_READY policy=restricted_reversible")

    async def stop(self) -> None:
        active = list(self._active.items())
        if active:
            await asyncio.gather(*(task for _trace, task in active), return_exceptions=True)
        self._active.clear()
        self.bus = None

    async def _on_execution_requested(self, event: Event) -> None:
        if event.trace_id in self._active:
            logger.warning("duplicate JAL execution ignored trace=%s", event.trace_id)
            return
        try:
            plan = loads(str(event.payload["jal"]))
            self.registry.validate(plan)
            if plan.act != DialogueAct.EXECUTE or not plan.steps:
                raise ValueError("JAL executor accepts only non-empty execute plans")
            migration_stage = MigrationStage.parse(
                str(event.payload.get("migration_stage", ""))
            )
            if migration_stage < MigrationStage.RESTRICTED_REVERSIBLE:
                raise ValueError("JAL execution is disabled before restricted promotion")
            effective_plan = plan
            correction = event.payload.get("correction_transaction")
            if isinstance(correction, Mapping) and correction.get("status") == "ready":
                replacement = _tool_call_from_mapping(correction.get("replacement"))
                original = _tool_call_from_mapping(correction.get("original"))
                compensation = _compensation_from_mapping(
                    correction.get("compensation")
                )
                if replacement is None or original is None or compensation is None:
                    raise ValueError("correction transaction is incomplete")
                effective_plan = JALPlan(plan.act, steps=(replacement,))
                self.registry.validate(effective_plan)
                for name in (original.tool, replacement.tool, compensation.tool):
                    if not self.tools.has(name):
                        raise ValueError(f"runtime tool unavailable: {name}")
            reversible = classify_reversibility(effective_plan)
            if (
                migration_stage == MigrationStage.RESTRICTED_REVERSIBLE
                and not reversible.eligible
            ):
                raise ValueError(f"non-reversible plan: {reversible.reasons}")
            for call in plan.steps:
                if not self.tools.has(call.tool):
                    raise ValueError(f"runtime tool unavailable: {call.tool}")
        except (KeyError, TypeError, ValueError) as exc:
            await self._publish_failure(event, f"План JAL отклонён: {exc}")
            return
        assert self.bus is not None
        self.bus.publish_event(
            event.child(
                "tool_call_requested",
                ToolCallRequestedPayload(
                    tool="jal_transaction",
                    plan=[
                        {"tool": call.tool, "params": dict(call.arguments)}
                        for call in plan.steps
                    ],
                ),
            )
        )
        task = asyncio.create_task(
            self._execute_transaction(
                event,
                plan,
                event.payload.get("correction_transaction"),
                require_compensation=(
                    migration_stage == MigrationStage.RESTRICTED_REVERSIBLE
                ),
            )
        )
        self._active[event.trace_id] = task
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            return
        finally:
            self._active.pop(event.trace_id, None)
        if self.bus.is_trace_closed(event.trace_id):
            return
        self.bus.publish_event(
            event.child(
                "tool_result",
                ToolResultPayload(
                    tool="jal_transaction", result=result, direct_response=True
                ),
            )
        )

    async def _execute_transaction(
        self,
        event: Event,
        plan: JALPlan,
        correction_value: Any,
        *,
        require_compensation: bool,
    ) -> dict[str, Any]:
        correction = correction_value if isinstance(correction_value, Mapping) else None
        if correction is not None:
            result = await self._execute_correction(event, correction)
            if result is not None:
                return result
        results: list[dict[str, Any]] = []
        compensations: list[CompensationCall] = []
        for call in plan.steps:
            result = await self._execute_call(event, call)
            results.append({"tool": call.tool, "result": result})
            if result.get("ok") is not True:
                rollback = await self._rollback(compensations)
                return self._combined(False, results, rollback=rollback)
            if call.tool not in _READ_ONLY:
                compensation = compensation_for(
                    ActionReceipt(event.trace_id, call.tool, call.arguments, result)
                )
                if compensation is None and require_compensation:
                    rollback = await self._rollback(compensations)
                    return self._combined(
                        False,
                        results,
                        rollback=rollback,
                        error="compensation_evidence_missing",
                    )
                if compensation is not None:
                    compensations.append(compensation)
        return self._combined(True, results)

    async def _execute_correction(
        self, event: Event, correction: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if correction.get("status") != "ready":
            return self._combined(False, [], error="correction_transaction_not_ready")
        compensation = _compensation_from_mapping(correction.get("compensation"))
        original = _tool_call_from_mapping(correction.get("original"))
        replacement = _tool_call_from_mapping(correction.get("replacement"))
        if compensation is None or original is None or replacement is None:
            return self._combined(False, [], error="correction_transaction_incomplete")
        compensation_result = await self.tools.execute(
            compensation.tool, dict(compensation.params)
        )
        if compensation_result.get("ok") is not True:
            return self._combined(
                False,
                [{"tool": compensation.tool, "result": compensation_result}],
                error="correction_compensation_failed",
            )
        result = await self._execute_call(event, replacement)
        results = [{"tool": replacement.tool, "result": result}]
        if result.get("ok") is not True:
            recovery = await self.tools.execute(
                original.tool, dict(original.arguments)
            )
            return self._combined(
                False,
                results,
                rollback=[{"tool": original.tool, "result": recovery}],
                error="correction_replacement_failed",
            )
        return self._combined(True, results)

    async def _execute_call(self, event: Event, call: ToolCall) -> dict[str, Any]:
        result = await self.tools.execute(call.tool, dict(call.arguments))
        if result.get("ok") is True and self.bus is not None:
            self.bus.publish_event(
                event.child(
                    "jal_action_committed",
                    JALActionCommittedPayload(
                        tool=call.tool,
                        params=dict(call.arguments),
                        result=result,
                    ),
                )
            )
        return result

    async def _rollback(
        self, compensations: list[CompensationCall]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for compensation in reversed(compensations):
            try:
                result = await self.tools.execute(
                    compensation.tool, dict(compensation.params)
                )
            except Exception as exc:  # noqa: BLE001 - continue best-effort rollback
                result = {"ok": False, "error": type(exc).__name__}
            results.append({"tool": compensation.tool, "result": result})
        return results

    @staticmethod
    def _combined(
        ok: bool,
        results: list[dict[str, Any]],
        *,
        rollback: list[dict[str, Any]] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        texts = [str(item["result"].get("response_text", "")).strip() for item in results]
        payload: dict[str, Any] = {
            "ok": ok,
            "results": results,
            "response_text": " ".join(text for text in texts if text)
            or ("Действие выполнено." if ok else "Транзакция JAL остановлена."),
        }
        if rollback is not None:
            payload["rollback"] = rollback
            payload["rollback_ok"] = all(item["result"].get("ok") is True for item in rollback)
        if error is not None:
            payload["error"] = error
        return payload

    async def _publish_failure(self, event: Event, message: str) -> None:
        assert self.bus is not None
        self.bus.publish_event(
            event.child(
                "response_ready", ResponseReadyPayload(text=message)
            )
        )

    async def _on_trace_closed(self, event: Event) -> None:
        task = self._active.get(event.trace_id)
        if task is not None and not task.done():
            logger.info("JAL_TRANSACTION_DRAINING trace=%s", event.trace_id)


def _compensation_from_mapping(value: Any) -> CompensationCall | None:
    if not isinstance(value, Mapping):
        return None
    tool = str(value.get("tool", ""))
    params = value.get("params")
    if not tool or not isinstance(params, Mapping):
        return None
    return CompensationCall(tool, dict(params))


def _tool_call_from_mapping(value: Any) -> ToolCall | None:
    if not isinstance(value, Mapping):
        return None
    tool = str(value.get("tool", ""))
    arguments = value.get("arguments")
    if not tool or not isinstance(arguments, Mapping):
        return None
    return ToolCall(tool, dict(arguments))
