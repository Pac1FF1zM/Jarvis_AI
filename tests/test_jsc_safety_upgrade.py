from __future__ import annotations

import pytest

from ml.jsc.jal import DialogueAct, JALPlan, ToolCall
from ml.jsc.risk import SelectiveRiskPolicy, evaluate_selective_risk
from ml.jsc.structured_decoding import (
    assemble_verified_explicit_execution,
    plan_completeness_issues,
)
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.transactions import ActionReceipt, compensation_for, plan_correction_transaction
from modules.command_router import route_explicit_command


def test_selective_risk_abstains_on_ambiguous_act_distribution():
    decision = evaluate_selective_risk(
        [0.45, 0.42, 0.08, 0.05], 0.99, SelectiveRiskPolicy()
    )

    assert decision.accepted is False
    assert decision.reason == "low_act_confidence"


def test_completeness_blocks_elliptical_compound_prefix_execution():
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "calculator"}),),
    )

    assert plan_completeness_issues(
        "открой калькулятор и затем блокнот", plan
    ) == ("compound_step_count_mismatch",)
    assert (
        assemble_verified_explicit_execution(
            "открой калькулятор и затем блокнот", build_project_schema_registry()
        )
        is None
    )


def test_correction_transaction_requires_verified_compensation():
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "notepad"}),),
    )
    receipt = ActionReceipt(
        "old-trace",
        "open_application",
        {"application": "calculator"},
        {"ok": True},
    )

    transaction = plan_correction_transaction(
        "нет не калькулятор а открой блокнот", plan, receipt
    )

    assert transaction is not None
    assert transaction.status == "ready"
    assert transaction.compensation is not None
    assert transaction.compensation.tool == "undo_action"
    assert transaction.policy == "compensate_then_replace_stop_on_failure"


def test_correction_transaction_blocks_without_observed_original():
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": "notepad"}),),
    )
    transaction = plan_correction_transaction(
        "нет не калькулятор а открой блокнот", plan, None
    )

    assert transaction is not None
    assert transaction.status == "blocked"
    assert transaction.reason == "original_action_not_observed"


def test_compensation_policy_uses_verified_reminder_result_id():
    compensation = compensation_for(
        ActionReceipt(
            "reminder-trace",
            "set_reminder",
            {"message": "позвонить", "minutes": 10},
            {"ok": True, "reminder": {"id": 42}},
        )
    )

    assert compensation is not None
    assert compensation.tool == "cancel_reminder"
    assert compensation.params == {"reminder_id": 42}


@pytest.mark.parametrize(
    ("text", "intent"),
    (
        ("можешь открыть пейнт", "open_application"),
        ("проверь работает ли сейчас режим жестов", "gesture_mode"),
        ("покажи мои предстоящие напоминания", "list_reminders"),
        ("сообщи мне текущее системное время", "get_current_time"),
    ),
)
def test_fresh_probe_natural_phrases_have_grounded_routes(text: str, intent: str):
    action = route_explicit_command(text)
    assert action is not None
    assert action.intent == intent
