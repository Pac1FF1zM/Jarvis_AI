"""Strict JSONL loader for user-curated Jarvis NLU fine-tuning examples."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data import Example, Span
from .schema import INTENTS

SLOT_NAMES = frozenset({"duration", "reminder_text", "application"})


def load_jsonl(path: str | Path, *, allow_empty: bool = True) -> list[Example]:
    """Load friendly ``text/intent/slots`` JSONL and validate every record."""
    source = Path(path)
    if not source.is_file():
        if allow_empty:
            return []
        raise FileNotFoundError(source)
    examples: list[Example] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        try:
            record: dict[str, Any] = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        text = str(record.get("text", "")).strip()
        intent = str(record.get("intent", "")).strip()
        if not text:
            raise ValueError(f"{source}:{line_number}: text must not be empty")
        if intent not in INTENTS:
            raise ValueError(
                f"{source}:{line_number}: unknown intent {intent!r}; expected one of {INTENTS}"
            )
        key = text.casefold()
        if key in seen:
            raise ValueError(f"{source}:{line_number}: duplicate text {text!r}")
        seen.add(key)
        slots = record.get("slots") or {}
        if not isinstance(slots, dict):
            raise ValueError(f"{source}:{line_number}: slots must be an object")
        spans: list[Span] = []
        occupied: list[tuple[int, int]] = []
        for label, raw_value in slots.items():
            if label not in SLOT_NAMES:
                raise ValueError(f"{source}:{line_number}: unsupported slot {label!r}")
            value = str(raw_value).strip()
            start = text.casefold().find(value.casefold())
            if not value or start < 0:
                raise ValueError(
                    f"{source}:{line_number}: slot {label!r} value {value!r} "
                    "must occur verbatim in text"
                )
            end = start + len(value)
            if any(start < old_end and end > old_start for old_start, old_end in occupied):
                raise ValueError(f"{source}:{line_number}: slot spans overlap")
            occupied.append((start, end))
            spans.append(Span(start, end, label))
        examples.append(Example(text, intent, tuple(sorted(spans, key=lambda span: span.start))))
    if not examples and not allow_empty:
        raise ValueError(f"{source}: no examples found")
    return examples


def validate_splits(train: list[Example], validation: list[Example]) -> dict[str, Any]:
    train_text = {example.text.casefold() for example in train}
    validation_text = {example.text.casefold() for example in validation}
    overlap = sorted(train_text & validation_text)
    if overlap:
        raise ValueError(
            "custom train/validation overlap: " + ", ".join(repr(text) for text in overlap[:5])
        )
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_intents": dict(Counter(example.intent for example in train)),
        "validation_intents": dict(Counter(example.intent for example in validation)),
    }
