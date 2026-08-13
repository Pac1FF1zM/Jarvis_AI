"""Contract tests for the side-effect-free JSC migration benchmark."""
from __future__ import annotations

from argparse import Namespace

import pytest

from ml.jsc.data import JSCExample
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, dumps, loads
from ml.jsc.project_registry import build_project_schema_registry
from training_workspace.jsc_migration_benchmark import (
    _gate_report,
    _migration_metrics,
    predict_legacy_jal,
    run,
)
from training_workspace.build_jsc_migration_suite import build


class _NeverCalledPredictor:
    def predict(self, _text):
        raise AssertionError("explicit close commands must not need neural fallback")


def test_legacy_baseline_maps_compound_close_to_canonical_jal():
    registry = build_project_schema_registry()

    prediction = loads(
        predict_legacy_jal(
            "закрой калькулятор, проводник и вижу студио код",
            _NeverCalledPredictor(),
            registry,
        )
    )

    assert prediction == JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall("window_control", {"action": "close", "window": "calculator"}),
            ToolCall("window_control", {"action": "close", "window": "explorer"}),
            ToolCall(
                "window_control",
                {"action": "close", "window": "visual_studio_code"},
            ),
        ),
    )


def test_legacy_baseline_rejects_negated_close_without_prediction():
    registry = build_project_schema_registry()
    prediction = loads(
        predict_legacy_jal("не закрывай калькулятор", _NeverCalledPredictor(), registry)
    )
    assert prediction.act == DialogueAct.REJECT


def test_migration_metrics_detect_opposite_open_close_action():
    registry = build_project_schema_registry()
    example = JSCExample(
        scenario_id="audit.opposite",
        split="validation",
        family_id="audit.opposite",
        category="single",
        history=(),
        text="открой калькулятор",
        state=None,
        target=JALPlan(
            DialogueAct.EXECUTE,
            steps=(ToolCall("open_application", {"application": "calculator"}),),
        ),
        metadata={},
    )
    application = example.target.steps[0].arguments["application"]
    opposite = dumps(
        JALPlan(
            DialogueAct.EXECUTE,
            steps=(
                ToolCall(
                    "window_control", {"action": "close", "window": application}
                ),
            ),
        )
    )

    metrics = _migration_metrics([example], [opposite], registry)

    assert metrics["opposite_action_count"] == 1
    assert metrics["opposite_action_rate"] == 1.0


def test_gate_report_keeps_safety_thresholds_at_zero():
    metrics = {
        "exact_jal_accuracy": 1.0,
        "category_exact_jal": {"single": 1.0, "multi_turn": 1.0},
        "exact_jal_by_step_group": {"steps_2_3": 1.0, "steps_4_5": 1.0},
        "schema_valid_rate": 1.0,
        "false_execution_rate": 0.0,
        "opposite_action_rate": 0.01,
    }

    result = _gate_report(metrics)

    assert result["passed"] is False
    assert result["checks"]["maximum_opposite_action_rate"]["target"] == 0.0


def test_locked_split_requires_explicit_acknowledgement(tmp_path):
    args = Namespace(
        split="evaluation_holdout",
        allow_locked_split=False,
    )
    with pytest.raises(ValueError, match="locked"):
        run(args)


def test_migration_suite_has_balanced_two_to_five_step_coverage():
    rows = build()
    compounds = [row for row in rows if row.category == "compound"]

    assert len(rows) == 400
    assert {
        count: sum(len(row.target.steps) == count for row in compounds)
        for count in range(2, 6)
    } == {2: 40, 3: 40, 4: 40, 5: 40}
    assert sum(row.category == "multi_turn" for row in rows) == 40
    assert sum(row.category == "asr_noise" for row in rows) == 30
