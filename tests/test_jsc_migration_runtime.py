from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.event_bus import Event, EventBus
from core.event_payloads import (
    InteractionCompletedPayload,
    JALExecutionRequestedPayload,
    JSCCandidateReadyPayload,
    NLUResultPayload,
)
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, dumps
from ml.jsc.legacy_adapter import nlu_payload_to_jal
from ml.jsc.migration import (
    MigrationStage,
    RollingErrorBudget,
    admit_stage,
    classify_reversibility,
)
from ml.jsc.project_registry import build_project_schema_registry
from modules.jal_executor import JALExecutorModule
from modules.jsc_migration import JSCMigrationModule


def _evidence(**overrides):
    values = {
        "reviewed_voice_turns": 1000,
        "stable_release_cycles": 2,
        "false_execution_rate": 0.0,
        "opposite_action_rate": 0.0,
        "semantic_exact_rate": 0.95,
        "correction_accuracy": 0.98,
        "ood_recall": 0.99,
    }
    values.update(overrides)
    return values


def _config(tmp_path, stage: str, evidence: dict | None = None):
    path = tmp_path / "state.json"
    if evidence is not None:
        path.write_text(json.dumps(evidence), encoding="utf-8")
    return SimpleNamespace(
        params={
            "stage": stage,
            "evidence_path": str(path),
            "log_path": str(tmp_path / "agreement.jsonl"),
            "fallback_seconds": 0.01,
        }
    )


def test_stage_admission_requires_human_evidence_and_two_release_cycles():
    blocked = admit_stage(MigrationStage.NLU_REMOVED, {})
    admitted = admit_stage(MigrationStage.NLU_REMOVED, _evidence())

    assert blocked.active == MigrationStage.AGREEMENT_CANARY
    assert "nlu_removal_requires_two_stable_release_cycles" in blocked.reasons
    assert admitted.active == MigrationStage.NLU_REMOVED
    assert admitted.admitted is True


def test_error_budget_trips_on_any_unsafe_disagreement():
    budget = RollingErrorBudget(window=20, minimum_agreement=0.5)

    assert budget.observe(exact_agreement=True, unsafe_disagreement=False) is True
    assert budget.observe(exact_agreement=False, unsafe_disagreement=True) is False


def test_restricted_policy_accepts_reversible_and_rejects_irreversible_calls():
    reversible = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "calculator"}),),
    )
    irreversible = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("system_control", {"action": "lock"}),),
    )

    assert classify_reversibility(reversible).eligible is True
    assert classify_reversibility(irreversible).eligible is False


def test_legacy_adapter_creates_canonical_jal():
    plan = nlu_payload_to_jal(
        {
            "text": "открой калькулятор",
            "intent": "open_application",
            "slots": {"application": "calculator"},
        },
        build_project_schema_registry(),
    )

    assert plan == JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "calculator"}),),
    )


@pytest.mark.asyncio
async def test_agreement_canary_forwards_nlu_once_and_records_comparison(tmp_path):
    bus = EventBus()
    module = JSCMigrationModule(_config(tmp_path, "agreement_canary"))
    await module.start(bus)
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "calculator"}),),
    )
    nlu = Event(
        "nlu_result",
        NLUResultPayload(
            text="открой калькулятор",
            intent="open_application",
            slots={"application": "calculator"},
            confidence=1.0,
            intent_confidence=1.0,
        ),
        trace_id="canary-1",
    )
    jsc = Event(
        "jsc_candidate_ready",
        JSCCandidateReadyPayload(
            text="открой калькулятор",
            jal=dumps(plan),
            accepted=True,
            risk={"accepted": True},
            input_source="parakeet",
        ),
        trace_id="canary-1",
    )

    await module._on_nlu(nlu)
    assert bus.queue.empty()
    await module._on_jsc(jsc)

    queued = []
    while not bus.queue.empty():
        queued.append(bus.queue.get_nowait())
    assert [event.event_type for event in queued] == ["semantic_result"]
    assert queued[0].payload["source"] == "nlu"
    record = json.loads((tmp_path / "agreement.jsonl").read_text("utf-8"))
    assert record["exact_agreement"] is True
    assert record["selected_source"] == "nlu"


@pytest.mark.asyncio
async def test_unqualified_nlu_removed_request_degrades_to_canary(tmp_path):
    module = JSCMigrationModule(_config(tmp_path, "nlu_removed", evidence={}))
    await module.start(EventBus())

    assert module.requested_stage == MigrationStage.NLU_REMOVED
    assert module.active_stage == MigrationStage.AGREEMENT_CANARY


@pytest.mark.asyncio
async def test_nlu_cannot_be_disabled_before_removed_stage_is_admitted(tmp_path):
    module = JSCMigrationModule(
        _config(tmp_path, "nlu_removed", evidence={}), legacy_nlu_enabled=False
    )

    with pytest.raises(RuntimeError, match="nlu_removed stage is admitted"):
        await module.start(EventBus())


@pytest.mark.asyncio
async def test_admitted_removed_stage_starts_without_legacy_nlu(tmp_path):
    module = JSCMigrationModule(
        _config(tmp_path, "nlu_removed", evidence=_evidence()),
        legacy_nlu_enabled=False,
    )

    await module.start(EventBus())

    assert module.active_stage == MigrationStage.NLU_REMOVED


@pytest.mark.asyncio
async def test_completed_trace_releases_migration_state(tmp_path):
    module = JSCMigrationModule(_config(tmp_path, "agreement_canary"))
    await module.start(EventBus())
    module._nlu["done-1"] = {"text": "test"}
    module._jsc["done-1"] = {"jal": "test"}
    module._handled.add("done-1")

    await module._on_trace_closed(
        Event(
            "interaction_completed",
            InteractionCompletedPayload(state="IDLE", ok=True),
            trace_id="done-1",
        )
    )

    assert "done-1" not in module._nlu
    assert "done-1" not in module._jsc
    assert "done-1" not in module._handled


@pytest.mark.asyncio
async def test_restricted_stage_selects_reversible_jsc_without_forwarding_nlu(tmp_path):
    bus = EventBus()
    module = JSCMigrationModule(
        _config(tmp_path, "restricted_reversible", evidence=_evidence())
    )
    await module.start(bus)
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "calculator"}),),
    )
    await module._on_jsc(
        Event(
            "jsc_candidate_ready",
            JSCCandidateReadyPayload(
                text="открой калькулятор",
                jal=dumps(plan),
                accepted=True,
                risk={"accepted": True},
                input_source="parakeet",
            ),
            trace_id="restricted-1",
        )
    )
    await module._on_nlu(
        Event(
            "nlu_result",
            NLUResultPayload(
                text="открой калькулятор",
                intent="open_application",
                slots={"application": "calculator"},
            ),
            trace_id="restricted-1",
        )
    )

    queued = []
    while not bus.queue.empty():
        queued.append(bus.queue.get_nowait())
    assert [event.event_type for event in queued] == ["jal_execution_requested"]


class _FakeTools:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def has(self, _name):
        return True

    async def execute(self, name, params):
        self.calls.append((name, dict(params)))
        if name == "undo_action":
            return {"ok": True, "response_text": "rolled back"}
        return self.results.pop(0)


class _ScriptedTools:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def has(self, _name):
        return True

    async def execute(self, name, params):
        self.calls.append((name, dict(params)))
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_jal_executor_rolls_back_completed_steps_when_later_step_fails():
    tools = _FakeTools(
        [
            {"ok": True, "application": "calculator", "response_text": "opened"},
            {"ok": False, "response_text": "reminder failed"},
        ]
    )
    module = JALExecutorModule(SimpleNamespace(params={}), tools)
    bus = EventBus()
    await module.start(bus)
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall("open_application", {"application": "calculator"}),
            ToolCall("set_reminder", {"message": "тест", "minutes": 5}),
        ),
    )
    await module._on_execution_requested(
        Event(
            "jal_execution_requested",
            JALExecutionRequestedPayload(
                text="тест",
                jal=dumps(plan),
                migration_stage="restricted_reversible",
            ),
            trace_id="transaction-1",
        )
    )

    assert tools.calls[-1] == (
        "undo_action",
        {"action": "close_application", "application": "calculator"},
    )
    events = []
    while not bus.queue.empty():
        events.append(bus.queue.get_nowait())
    result_event = next(event for event in events if event.event_type == "tool_result")
    assert result_event.payload["result"]["ok"] is False
    assert result_event.payload["result"]["rollback_ok"] is True


@pytest.mark.asyncio
async def test_jal_executor_rejects_execution_before_restricted_stage():
    tools = _FakeTools([])
    module = JALExecutorModule(SimpleNamespace(params={}), tools)
    bus = EventBus()
    await module.start(bus)
    plan = JALPlan(DialogueAct.EXECUTE, steps=(ToolCall("get_current_time"),))

    await module._on_execution_requested(
        Event(
            "jal_execution_requested",
            JALExecutionRequestedPayload(
                text="который час",
                jal=dumps(plan),
                migration_stage="agreement_canary",
            ),
            trace_id="blocked-canary-1",
        )
    )

    events = []
    while not bus.queue.empty():
        events.append(bus.queue.get_nowait())
    assert [event.event_type for event in events] == ["response_ready"]
    assert tools.calls == []


@pytest.mark.asyncio
async def test_correction_stops_before_replacement_when_compensation_fails():
    tools = _ScriptedTools([{"ok": False, "error": "cannot_close"}])
    module = JALExecutorModule(SimpleNamespace(params={}), tools)
    bus = EventBus()
    await module.start(bus)
    replacement = ToolCall("open_application", {"application": "discord"})
    plan = JALPlan(DialogueAct.EXECUTE, steps=(replacement,))
    correction = {
        "status": "ready",
        "reason": "verified_reversible_correction",
        "original_trace_id": "original-1",
        "original": {
            "tool": "open_application",
            "arguments": {"application": "calculator"},
        },
        "replacement": {
            "tool": replacement.tool,
            "arguments": dict(replacement.arguments),
        },
        "compensation": {
            "tool": "undo_action",
            "params": {
                "action": "close_application",
                "application": "calculator",
            },
        },
    }

    await module._on_execution_requested(
        Event(
            "jal_execution_requested",
            JALExecutionRequestedPayload(
                text="нет, открой discord",
                jal=dumps(plan),
                migration_stage="restricted_reversible",
                correction_transaction=correction,
            ),
            trace_id="correction-1",
        )
    )

    assert tools.calls == [
        (
            "undo_action",
            {"action": "close_application", "application": "calculator"},
        )
    ]
    events = []
    while not bus.queue.empty():
        events.append(bus.queue.get_nowait())
    result = next(event for event in events if event.event_type == "tool_result")
    assert result.payload["result"]["error"] == "correction_compensation_failed"
