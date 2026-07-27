"""Contract tests for the first Jarvis Semantic Core milestone: JAL v1."""
from __future__ import annotations

import json

import pytest

from ml.jsc.jal import (
    DialogueAct,
    JALCodecError,
    JALPlan,
    JALValidationError,
    MAX_STEPS,
    MissingSlot,
    ToolCall,
    ToolSchemaRegistry,
    dumps,
    loads,
)
from tools.registry import ToolRegistry


@pytest.fixture
def schemas() -> ToolSchemaRegistry:
    registry = ToolRegistry()
    registry.discover("tools")
    return ToolSchemaRegistry.from_tool_registry(registry)


def test_canonical_round_trip_is_deterministic_and_unicode_safe():
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall(
                "set_reminder",
                {"message": "позвонить другу", "minutes": 10},
            ),
        ),
    )

    encoded = dumps(plan)

    assert loads(encoded) == plan
    assert dumps(loads(encoded)) == encoded
    assert "позвонить другу" in encoded
    assert '": ' not in encoded
    assert '", ' not in encoded


def test_sequence_plan_supports_multiple_typed_tool_calls(schemas):
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall("open_application", {"application": "discord"}),
            ToolCall(
                "set_reminder",
                {"minutes": 10, "message": "закрыть Discord"},
            ),
        ),
    )

    schemas.validate(plan)


def test_ask_plan_can_carry_an_incomplete_pending_call(schemas):
    plan = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("set_reminder", {"message": "позвонить другу"}),),
        missing=(MissingSlot(0, "clock_time"),),
        reason="missing_time",
    )

    schemas.validate(plan)


def test_execute_rejects_missing_required_or_conditional_arguments(schemas):
    with pytest.raises(JALValidationError, match="requires argument 'message'"):
        schemas.validate(
            JALPlan(
                DialogueAct.EXECUTE,
                steps=(ToolCall("set_reminder", {"minutes": 10}),),
            )
        )
    with pytest.raises(JALValidationError, match="requires one of"):
        schemas.validate(
            JALPlan(
                DialogueAct.EXECUTE,
                steps=(ToolCall("set_reminder", {"message": "позвонить"}),),
            )
        )


def test_mutually_exclusive_time_sources_are_rejected(schemas):
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(
            ToolCall(
                "set_reminder",
                {
                    "message": "позвонить",
                    "minutes": 10,
                    "clock_time": "20:00",
                },
            ),
        ),
    )

    with pytest.raises(JALValidationError, match="mutually exclusive"):
        schemas.validate(plan)


@pytest.mark.parametrize("application", ["steam", "obs studio", "visual studio code"])
def test_application_schema_allows_windows_discovered_names(schemas, application):
    plan = JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall("open_application", {"application": application}),),
    )

    schemas.validate(plan)


def test_unknown_tool_argument_and_missing_reference_are_rejected(schemas):
    with pytest.raises(JALValidationError, match="unknown tool"):
        schemas.validate(
            JALPlan(
                DialogueAct.EXECUTE,
                steps=(ToolCall("run_shell", {"command": "whoami"}),),
            )
        )
    with pytest.raises(JALValidationError, match="unknown arguments"):
        schemas.validate(
            JALPlan(
                DialogueAct.EXECUTE,
                steps=(ToolCall("get_current_time", {"timezone": "UTC"}),),
            )
        )
    with pytest.raises(JALValidationError, match="unknown arguments"):
        schemas.validate(
            JALPlan(
                DialogueAct.ASK,
                steps=(ToolCall("get_current_time"),),
                missing=(MissingSlot(0, "timezone"),),
                reason="missing_timezone",
            )
        )


def test_schema_types_ranges_and_enums_are_enforced(schemas):
    invalid_calls = (
        ToolCall("cancel_reminder", {"reminder_id": True}),
        ToolCall("cancel_reminder", {"reminder_id": 0}),
        ToolCall(
            "set_reminder",
            {"message": "позвонить", "clock_time": "20:00", "day": "послезавтра"},
        ),
    )
    for call in invalid_calls:
        with pytest.raises(JALValidationError):
            schemas.validate(JALPlan(DialogueAct.EXECUTE, steps=(call,)))


def test_non_tool_dialogue_acts_cannot_smuggle_steps():
    with pytest.raises(JALCodecError, match="cannot carry tool steps"):
        JALPlan(
            DialogueAct.REJECT,
            steps=(ToolCall("get_current_time"),),
            reason="out_of_scope",
        )
    assert JALPlan(DialogueAct.REJECT, reason="out_of_scope").steps == ()
    assert JALPlan(DialogueAct.DIALOGUE, reason="general_chat").steps == ()
    assert JALPlan(DialogueAct.CANCEL).steps == ()


def test_acts_require_machine_readable_reasons_where_semantically_needed():
    call = ToolCall("get_current_time")
    with pytest.raises(JALCodecError, match="confirm requires"):
        JALPlan(DialogueAct.CONFIRM, steps=(call,))
    with pytest.raises(JALCodecError, match="dialogue requires"):
        JALPlan(DialogueAct.DIALOGUE)
    with pytest.raises(JALCodecError, match="must match"):
        JALPlan(DialogueAct.REJECT, reason="непонятная команда")
    with pytest.raises(JALCodecError, match="reason must be null"):
        JALPlan(DialogueAct.EXECUTE, steps=(call,), reason="execute_now")


def test_codec_rejects_duplicate_unknown_and_non_finite_json_fields():
    valid = dumps(JALPlan(DialogueAct.CANCEL, reason="user_requested"))
    duplicate = valid.replace('"act":"cancel"', '"act":"cancel","act":"execute"')
    unknown = json.loads(valid)
    unknown["shell"] = "whoami"

    with pytest.raises(JALCodecError, match="duplicate JSON key"):
        loads(duplicate)
    with pytest.raises(JALCodecError, match="fields mismatch"):
        loads(json.dumps(unknown))
    with pytest.raises(JALCodecError, match="non-finite"):
        loads(valid.replace('"reason":"user_requested"', '"reason":NaN'))


def test_codec_rejects_oversized_plans_and_documents():
    with pytest.raises(JALCodecError, match="at most"):
        JALPlan(
            DialogueAct.EXECUTE,
            steps=tuple(ToolCall("get_current_time") for _ in range(MAX_STEPS + 1)),
        )
    with pytest.raises(JALCodecError, match="too large"):
        loads(" " * 40_000)


def test_schema_fingerprint_is_stable_but_changes_with_contract():
    schemas = [
            {
                "name": "alpha",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "beta",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]
    first = ToolSchemaRegistry(schemas)
    reversed_order = ToolSchemaRegistry(reversed(schemas))
    changed = ToolSchemaRegistry(
        [
            {
                "name": "alpha",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": [],
                },
            }
        ]
    )

    assert first.schema_fingerprint == reversed_order.schema_fingerprint
    assert first.schema_fingerprint != changed.schema_fingerprint


def test_registry_copies_schemas_and_rejects_invalid_contract_extensions():
    mutable = {
        "name": "alpha",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    registry = ToolSchemaRegistry([mutable])
    fingerprint = registry.schema_fingerprint
    mutable["parameters"]["properties"]["late"] = {"type": "string"}
    assert registry.schema_fingerprint == fingerprint

    invalid = {
        "name": "alpha",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
            "x-mutually-exclusive": ["unknown"],
        },
    }
    with pytest.raises(JALValidationError, match="invalid x-mutually-exclusive"):
        ToolSchemaRegistry([invalid])
