"""Typed Jarvis Action Language v1 and its schema-aware safety validator."""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, TypeAlias

JAL_VERSION = 1
MAX_STEPS = 8
MAX_DOCUMENT_BYTES = 32_768
MAX_STRING_LENGTH = 4_096
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

JALScalar: TypeAlias = str | int | float | bool | None


class DialogueAct(str, Enum):
    EXECUTE = "execute"
    ASK = "ask"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    REJECT = "reject"
    DIALOGUE = "dialogue"


class JALCodecError(ValueError):
    """The serialized document violates the closed JAL v1 grammar."""


class JALValidationError(ValueError):
    """The plan is grammatical but incompatible with registered tools."""


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: Mapping[str, JALScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.tool, "tool")
        if not isinstance(self.arguments, Mapping):
            raise JALCodecError("tool arguments must be an object")
        copied: dict[str, JALScalar] = {}
        for name, value in self.arguments.items():
            _validate_name(name, "argument")
            _validate_scalar(value, f"argument {name!r}")
            copied[name] = value
        object.__setattr__(self, "arguments", copied)


@dataclass(frozen=True, order=True)
class MissingSlot:
    step: int
    name: str

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise JALCodecError("missing slot step must be an integer")
        if self.step < 0 or self.step >= MAX_STEPS:
            raise JALCodecError(f"missing slot step must be between 0 and {MAX_STEPS - 1}")
        _validate_name(self.name, "missing slot")


@dataclass(frozen=True)
class JALPlan:
    act: DialogueAct
    steps: tuple[ToolCall, ...] = ()
    missing: tuple[MissingSlot, ...] = ()
    reason: str | None = None
    version: int = JAL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.act, DialogueAct):
            raise JALCodecError("act must be a DialogueAct")
        if self.version != JAL_VERSION or isinstance(self.version, bool):
            raise JALCodecError(f"only JAL version {JAL_VERSION} is supported")
        if not isinstance(self.steps, tuple) or not all(
            isinstance(step, ToolCall) for step in self.steps
        ):
            raise JALCodecError("steps must be a tuple of ToolCall objects")
        if not isinstance(self.missing, tuple) or not all(
            isinstance(slot, MissingSlot) for slot in self.missing
        ):
            raise JALCodecError("missing must be a tuple of MissingSlot objects")
        if len(self.steps) > MAX_STEPS:
            raise JALCodecError(f"a JAL plan may contain at most {MAX_STEPS} steps")
        if len(set(self.missing)) != len(self.missing):
            raise JALCodecError("missing slots must be unique")
        if self.reason is not None:
            _validate_name(self.reason, "reason")

        if self.act == DialogueAct.EXECUTE:
            if not self.steps or self.missing:
                raise JALCodecError(f"{self.act.value} requires complete steps")
            if self.reason is not None:
                raise JALCodecError("execute reason must be null")
        elif self.act == DialogueAct.CONFIRM:
            if not self.steps or self.missing or self.reason is None:
                raise JALCodecError("confirm requires complete steps and a reason")
        elif self.act == DialogueAct.ASK:
            if not self.steps or not self.missing or self.reason is None:
                raise JALCodecError(
                    "ask requires pending steps, missing slots and a reason"
                )
        else:
            if self.steps or self.missing:
                raise JALCodecError(f"{self.act.value} cannot carry tool steps")
            needs_reason = self.act in {
                DialogueAct.REJECT,
                DialogueAct.DIALOGUE,
            }
            if needs_reason and self.reason is None:
                raise JALCodecError(
                    f"{self.act.value} requires a machine-readable reason"
                )


def dumps(plan: JALPlan) -> str:
    """Return one deterministic training/evaluation target for ``plan``."""
    if not isinstance(plan, JALPlan):
        raise TypeError("dumps expects JALPlan")
    document = {
        "version": plan.version,
        "act": plan.act.value,
        "steps": [
            {"tool": step.tool, "arguments": dict(step.arguments)}
            for step in plan.steps
        ],
        "missing": [
            {"step": slot.step, "name": slot.name} for slot in plan.missing
        ],
        "reason": plan.reason,
    }
    result = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(result.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise JALCodecError("JAL document is too large")
    return result


def loads(value: str | bytes) -> JALPlan:
    """Parse the closed JAL grammar; unknown and duplicate fields are errors."""
    if isinstance(value, bytes):
        if len(value) > MAX_DOCUMENT_BYTES:
            raise JALCodecError("JAL document is too large")
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JALCodecError("JAL bytes must be valid UTF-8") from exc
    elif isinstance(value, str):
        text = value
        if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise JALCodecError("JAL document is too large")
    else:
        raise TypeError("loads expects str or bytes")

    try:
        raw = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_raise_constant(constant)),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise JALCodecError(f"invalid JAL JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise JALCodecError("JAL root must be an object")
    _require_exact_fields(raw, {"version", "act", "steps", "missing", "reason"}, "plan")
    if raw["version"] != JAL_VERSION or isinstance(raw["version"], bool):
        raise JALCodecError(f"only JAL version {JAL_VERSION} is supported")
    try:
        act = DialogueAct(raw["act"])
    except (TypeError, ValueError) as exc:
        raise JALCodecError(f"unknown dialogue act: {raw['act']!r}") from exc
    if not isinstance(raw["steps"], list):
        raise JALCodecError("steps must be an array")
    if not isinstance(raw["missing"], list):
        raise JALCodecError("missing must be an array")

    steps: list[ToolCall] = []
    for index, item in enumerate(raw["steps"]):
        if not isinstance(item, dict):
            raise JALCodecError(f"steps[{index}] must be an object")
        _require_exact_fields(item, {"tool", "arguments"}, f"steps[{index}]")
        steps.append(ToolCall(item["tool"], item["arguments"]))
    missing: list[MissingSlot] = []
    for index, item in enumerate(raw["missing"]):
        if not isinstance(item, dict):
            raise JALCodecError(f"missing[{index}] must be an object")
        _require_exact_fields(item, {"step", "name"}, f"missing[{index}]")
        missing.append(MissingSlot(item["step"], item["name"]))
    return JALPlan(
        act=act,
        steps=tuple(steps),
        missing=tuple(missing),
        reason=raw["reason"],
        version=raw["version"],
    )


class ToolSchemaRegistry:
    """Validate JAL calls against the same JSON-like schemas as runtime tools."""

    def __init__(self, schemas: Iterable[Mapping[str, Any]]) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        for raw_schema in schemas:
            schema = deepcopy(dict(raw_schema))
            name = schema.get("name")
            try:
                _validate_name(name, "schema tool")
            except JALCodecError as exc:
                raise JALValidationError(str(exc)) from exc
            if name in self._schemas:
                raise JALValidationError(f"duplicate tool schema: {name}")
            parameters = schema.get("parameters")
            if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
                raise JALValidationError(f"tool {name} parameters must be an object schema")
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            if not isinstance(properties, Mapping) or not isinstance(required, list):
                raise JALValidationError(f"tool {name} has malformed properties/required")
            if not all(isinstance(item, str) for item in required):
                raise JALValidationError(f"tool {name} has malformed required entries")
            for property_name, property_schema in properties.items():
                try:
                    _validate_name(property_name, "schema argument")
                except JALCodecError as exc:
                    raise JALValidationError(str(exc)) from exc
                if not isinstance(property_schema, Mapping) or property_schema.get(
                    "type"
                ) not in {"string", "integer", "number", "boolean"}:
                    raise JALValidationError(
                        f"tool {name} has unsupported schema for {property_name}"
                    )
            unknown_required = set(required) - set(properties)
            if unknown_required:
                raise JALValidationError(
                    f"tool {name} requires unknown properties: {sorted(unknown_required)}"
                )
            for extension in ("x-one-of-required", "x-mutually-exclusive"):
                names = parameters.get(extension, [])
                if not isinstance(names, list) or not all(
                    isinstance(item, str) for item in names
                ):
                    raise JALValidationError(f"tool {name} has malformed {extension}")
                unknown = set(names) - set(properties)
                if unknown or len(names) != len(set(names)):
                    raise JALValidationError(
                        f"tool {name} has invalid {extension}: {names}"
                    )
            self._schemas[name] = schema

    @classmethod
    def from_tool_registry(cls, registry: Any) -> "ToolSchemaRegistry":
        return cls(registry.schemas())

    @property
    def schema_fingerprint(self) -> str:
        ordered = [self._schemas[name] for name in sorted(self._schemas)]
        canonical = json.dumps(
            ordered, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self, plan: JALPlan) -> None:
        if not isinstance(plan, JALPlan):
            raise TypeError("validate expects JALPlan")
        missing = {(slot.step, slot.name) for slot in plan.missing}
        for slot in plan.missing:
            if slot.step >= len(plan.steps):
                raise JALValidationError(
                    f"missing slot references absent step {slot.step}"
                )
            if slot.name in plan.steps[slot.step].arguments:
                raise JALValidationError(
                    f"missing slot {slot.step}.{slot.name} already has a value"
                )
        for index, call in enumerate(plan.steps):
            self._validate_call(plan.act, index, call, missing)

    def _validate_call(
        self,
        act: DialogueAct,
        index: int,
        call: ToolCall,
        missing: set[tuple[int, str]],
    ) -> None:
        schema = self._schemas.get(call.tool)
        if schema is None:
            raise JALValidationError(f"step {index}: unknown tool {call.tool!r}")
        parameters = schema["parameters"]
        properties: Mapping[str, Any] = parameters.get("properties", {})
        unknown_missing = sorted(
            name for step, name in missing if step == index and name not in properties
        )
        if unknown_missing:
            raise JALValidationError(
                f"step {index}: missing references unknown arguments {unknown_missing}"
            )
        unknown = set(call.arguments) - set(properties)
        if unknown:
            raise JALValidationError(
                f"step {index}: unknown arguments for {call.tool}: {sorted(unknown)}"
            )
        for name, value in call.arguments.items():
            self._validate_value(index, name, value, properties[name])
        for name in parameters.get("required", []):
            if name not in call.arguments and not (
                act == DialogueAct.ASK and (index, name) in missing
            ):
                raise JALValidationError(
                    f"step {index}: {call.tool} requires argument {name!r}"
                )
        alternatives = parameters.get("x-one-of-required", [])
        if alternatives:
            if not isinstance(alternatives, list) or not all(
                isinstance(name, str) and name in properties for name in alternatives
            ):
                raise JALValidationError(
                    f"tool {call.tool} has malformed x-one-of-required"
                )
            provided = [name for name in alternatives if name in call.arguments]
            requested = [name for name in alternatives if (index, name) in missing]
            if not provided and not (act == DialogueAct.ASK and requested):
                raise JALValidationError(
                    f"step {index}: {call.tool} requires one of {alternatives}"
                )
        exclusive = parameters.get("x-mutually-exclusive", [])
        if exclusive:
            if not isinstance(exclusive, list):
                raise JALValidationError(
                    f"tool {call.tool} has malformed x-mutually-exclusive"
                )
            provided = [name for name in exclusive if name in call.arguments]
            if len(provided) > 1:
                raise JALValidationError(
                    f"step {index}: mutually exclusive arguments: {provided}"
                )

    @staticmethod
    def _validate_value(index: int, name: str, value: JALScalar, spec: Any) -> None:
        if not isinstance(spec, Mapping):
            raise JALValidationError(f"argument schema for {name!r} is malformed")
        expected = spec.get("type")
        type_ok = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
        }.get(expected, False)
        if not type_ok:
            raise JALValidationError(
                f"step {index}: argument {name!r} must have type {expected}"
            )
        if isinstance(value, str):
            if len(value) < int(spec.get("minLength", 0)):
                raise JALValidationError(f"step {index}: argument {name!r} is too short")
            if len(value) > int(spec.get("maxLength", MAX_STRING_LENGTH)):
                raise JALValidationError(f"step {index}: argument {name!r} is too long")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                raise JALValidationError(f"step {index}: argument {name!r} is too small")
            if "maximum" in spec and value > spec["maximum"]:
                raise JALValidationError(f"step {index}: argument {name!r} is too large")
        if "enum" in spec and value not in spec["enum"]:
            raise JALValidationError(
                f"step {index}: argument {name!r} is not in the allowed enum"
            )


def _validate_name(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise JALCodecError(f"{label} must match {_NAME.pattern}")


def _validate_scalar(value: Any, label: str) -> None:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise JALCodecError(f"{label} must be a JSON scalar")
    if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
        raise JALCodecError(f"{label} exceeds {MAX_STRING_LENGTH} characters")
    if isinstance(value, float) and not math.isfinite(value):
        raise JALCodecError(f"{label} must be finite")


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise JALCodecError(
            f"{location} fields mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JALCodecError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _raise_constant(value: str) -> None:
    raise JALCodecError(f"non-finite JSON constant is forbidden: {value}")
