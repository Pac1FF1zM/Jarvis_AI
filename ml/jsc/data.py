"""Strict dialogue JSONL format for Jarvis Semantic Core training data."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .jal import DialogueAct, JALPlan, ToolSchemaRegistry, dumps, loads

DATA_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test", "evaluation_holdout")
CATEGORIES = frozenset(
    {
        "single",
        "compound",
        "multi_turn",
        "correction",
        "hard_negative",
        "ood",
        "asr_noise",
    }
)
_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_FIELDS = {
    "schema_version",
    "scenario_id",
    "split",
    "family_id",
    "category",
    "history",
    "text",
    "state_jal",
    "target_jal",
    "metadata",
}


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    text: str


@dataclass(frozen=True)
class JSCExample:
    scenario_id: str
    split: str
    family_id: str
    category: str
    history: tuple[DialogueTurn, ...]
    text: str
    state: JALPlan | None
    target: JALPlan
    metadata: Mapping[str, Any]

    @property
    def input_signature(self) -> str:
        parts = [
            *(f"{turn.role}:{_normalise(turn.text)}" for turn in self.history),
            "state:" + (dumps(self.state) if self.state is not None else "null"),
            "user:" + _normalise(self.text),
        ]
        return "\n".join(parts)


def load_jsc_jsonl(
    path: str | Path,
    registry: ToolSchemaRegistry,
    *,
    expected_split: str | None = None,
) -> list[JSCExample]:
    """Load, parse and schema-validate every JAL target before training."""
    source = Path(path)
    examples: list[JSCExample] = []
    scenario_ids: set[str] = set()
    signatures: set[str] = set()
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        location = f"{source}:{line_number}"
        try:
            raw = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{location}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError(f"{location}: fields must be exactly {sorted(_FIELDS)}")
        if raw["schema_version"] != DATA_SCHEMA_VERSION:
            raise ValueError(f"{location}: unsupported data schema version")
        scenario_id = _checked_id(raw["scenario_id"], location, "scenario_id")
        family_id = _checked_id(raw["family_id"], location, "family_id")
        split = str(raw["split"])
        if split not in SPLITS or (expected_split and split != expected_split):
            raise ValueError(f"{location}: unexpected split {split!r}")
        category = str(raw["category"])
        if category not in CATEGORIES:
            raise ValueError(f"{location}: unsupported category {category!r}")
        text = _checked_text(raw["text"], location, "text")
        history = _load_history(raw["history"], location)
        metadata = raw["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"{location}: metadata must be an object")
        state = _load_canonical_jal(raw["state_jal"], registry, location, "state_jal")
        target = _load_canonical_jal(
            raw["target_jal"], registry, location, "target_jal", allow_null=False
        )
        assert target is not None
        example = JSCExample(
            scenario_id=scenario_id,
            split=split,
            family_id=family_id,
            category=category,
            history=history,
            text=text,
            state=state,
            target=target,
            metadata=dict(metadata),
        )
        _validate_category_contract(example, location)
        if scenario_id in scenario_ids:
            raise ValueError(f"{location}: duplicate scenario_id {scenario_id!r}")
        if example.input_signature in signatures:
            raise ValueError(f"{location}: duplicate model input")
        scenario_ids.add(scenario_id)
        signatures.add(example.input_signature)
        examples.append(example)
    if not examples:
        raise ValueError(f"{source}: no JSC examples found")
    return examples


def validate_jsc_splits(
    split_examples: Mapping[str, Iterable[JSCExample]],
) -> dict[str, Any]:
    """Reject family/input leakage and report category/act coverage."""
    family_sets: dict[str, set[str]] = {}
    signature_sets: dict[str, set[str]] = {}
    report: dict[str, Any] = {}
    for split, values in split_examples.items():
        examples = list(values)
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        if any(example.split != split for example in examples):
            raise ValueError(f"records labeled with another split found in {split}")
        family_sets[split] = {example.family_id for example in examples}
        signature_sets[split] = {example.input_signature for example in examples}
        report[split] = {
            "examples": len(examples),
            "categories": dict(sorted(Counter(e.category for e in examples).items())),
            "acts": dict(sorted(Counter(e.target.act.value for e in examples).items())),
            "tools": dict(
                sorted(
                    Counter(
                        step.tool
                        for example in examples
                        for step in example.target.steps
                    ).items()
                )
            ),
            "families": len(family_sets[split]),
        }
    names = list(report)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            family_overlap = family_sets[left] & family_sets[right]
            input_overlap = signature_sets[left] & signature_sets[right]
            if family_overlap:
                raise ValueError(
                    f"scenario family leakage {left}/{right}: {sorted(family_overlap)[:5]}"
                )
            if input_overlap:
                raise ValueError(
                    f"model input leakage {left}/{right}: {len(input_overlap)} records"
                )
    return report


def _load_history(value: Any, location: str) -> tuple[DialogueTurn, ...]:
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError(f"{location}: history must be an array of at most 8 turns")
    turns: list[DialogueTurn] = []
    for index, raw_turn in enumerate(value):
        if not isinstance(raw_turn, dict) or set(raw_turn) != {"role", "text"}:
            raise ValueError(f"{location}: history[{index}] has invalid fields")
        role = str(raw_turn["role"])
        if role not in {"user", "jarvis"}:
            raise ValueError(f"{location}: history[{index}] has invalid role")
        turns.append(
            DialogueTurn(
                role,
                _checked_text(raw_turn["text"], location, f"history[{index}].text"),
            )
        )
    return tuple(turns)


def _validate_category_contract(example: JSCExample, location: str) -> None:
    """Keep structural labels honest instead of trusting generator metadata."""
    if example.category == "single" and (example.history or example.state is not None):
        raise ValueError(f"{location}: single example cannot carry dialogue state")
    if example.category == "compound" and not (
        example.target.act == DialogueAct.EXECUTE and len(example.target.steps) >= 2
    ):
        raise ValueError(f"{location}: compound example must execute at least two steps")
    if example.category == "multi_turn" and not (
        example.target.act in {DialogueAct.ASK, DialogueAct.CONFIRM}
        or (example.history and example.state is not None)
    ):
        raise ValueError(f"{location}: multi_turn example lacks pending dialogue context")
    if example.category == "correction" and not (
        example.history and example.state is not None
    ):
        raise ValueError(f"{location}: correction example lacks the plan being corrected")
    if example.category == "hard_negative" and example.target.act != DialogueAct.DIALOGUE:
        raise ValueError(f"{location}: hard_negative must not execute a mentioned tool")
    if example.category == "ood" and example.target.act != DialogueAct.REJECT:
        raise ValueError(f"{location}: OOD example must be rejected")
    if example.category == "asr_noise" and not isinstance(
        example.metadata.get("clean_text"), str
    ):
        raise ValueError(f"{location}: ASR-noise example requires clean_text metadata")


def _load_canonical_jal(
    value: Any,
    registry: ToolSchemaRegistry,
    location: str,
    field: str,
    *,
    allow_null: bool = True,
) -> JALPlan | None:
    if value is None and allow_null:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{location}: {field} must be canonical JAL or null")
    try:
        plan = loads(value)
        if dumps(plan) != value:
            raise ValueError("representation is not canonical")
        registry.validate(plan)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: invalid {field}: {exc}") from exc
    return plan


def _checked_id(value: Any, location: str, field: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"{location}: {field} is not a stable lowercase id")
    return value


def _checked_text(value: Any, location: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2_000:
        raise ValueError(f"{location}: {field} must be non-empty and bounded")
    return value.strip()


def _normalise(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> Any:
    raise ValueError(f"non-finite JSON value {value!r}")
