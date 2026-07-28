"""Review local Jarvis Active Learning candidates before they reach training.

Examples are never appended to the corpus until a person approves or corrects
their intent and optional slots.  The source queue is rewritten without the
reviewed entries; an audit trail is appended alongside it.

Usage::

    python -m training_workspace.review_feedback --summary
    python -m training_workspace.review_feedback
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ml.nlu.custom_data import SLOT_NAMES
from ml.nlu.schema import INTENTS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "data" / "feedback" / "pending.jsonl"
DEFAULT_APPROVED = ROOT / "training_workspace" / "data" / "feedback_train.jsonl"
DEFAULT_AUDIT = ROOT / "data" / "feedback" / "reviewed.jsonl"


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(record, dict) or not str(record.get("text", "")).strip():
            raise ValueError(f"{path}:{line_number}: invalid feedback record")
        records.append(record)
    return records


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pending": len(records),
        "reasons": dict(sorted(Counter(reason for item in records for reason in item.get("reasons", [])).items())),
        "predicted_intents": dict(sorted(Counter(str(item.get("predicted_intent", "unknown")) for item in records).items())),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in records)
    path.write_text(content, encoding="utf-8", newline="\n")


def _read_slots(default: dict[str, Any]) -> dict[str, str] | None:
    raw = input(f"Слоты JSON [{json.dumps(default, ensure_ascii=False)}]: ").strip()
    if not raw:
        return {str(key): str(value) for key, value in default.items()}
    try:
        slots = json.loads(raw)
    except json.JSONDecodeError:
        print("Некорректный JSON — запись пропущена.")
        return None
    if not isinstance(slots, dict) or any(key not in SLOT_NAMES for key in slots):
        print(f"Допустимые слоты: {sorted(SLOT_NAMES)}. Запись пропущена.")
        return None
    return {str(key): str(value) for key, value in slots.items()}


def review(queue: Path, approved: Path, audit: Path) -> dict[str, int]:
    pending = load_records(queue)
    reviewed: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    approved_records: list[dict[str, Any]] = []
    existing_texts = {
        str(item.get("text", "")).casefold().strip()
        for item in load_records(approved)
    }
    for index, item in enumerate(pending, 1):
        print(f"\n[{index}/{len(pending)}] {item['text']}")
        print("prediction:", item.get("predicted_intent"), "confidence:", item.get("intent_confidence"), "reasons:", ", ".join(item.get("reasons", [])))
        choice = input("Enter=accept prediction, intent / d=discard / s=skip: ").strip()
        if choice.casefold() == "s":
            kept.append(item)
            continue
        if choice.casefold() == "d":
            reviewed.append({**item, "review_status": "discarded"})
            continue
        intent = str(item.get("predicted_intent", "")) if not choice else choice
        if intent not in INTENTS:
            print(f"Допустимые intent: {', '.join(INTENTS)}. Запись оставлена в очереди.")
            kept.append(item)
            continue
        slots = _read_slots(dict(item.get("slots") or {}))
        if slots is None:
            kept.append(item)
            continue
        text = str(item["text"]).strip()
        if text.casefold() in existing_texts:
            print("Такой текст уже подтверждён; записываю только аудит.")
        else:
            approved_records.append({"text": text, "intent": intent, "slots": slots})
            existing_texts.add(text.casefold())
        reviewed.append({**item, "review_status": "approved", "reviewed_intent": intent, "reviewed_slots": slots})
    if approved_records:
        prior = load_records(approved)
        _write_jsonl(approved, prior + approved_records)
    if reviewed:
        prior_audit = load_records(audit)
        _write_jsonl(audit, prior_audit + reviewed)
    _write_jsonl(queue, kept)
    return {"approved": len(approved_records), "discarded": sum(item["review_status"] == "discarded" for item in reviewed), "skipped": len(kept)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--approved", type=Path, default=DEFAULT_APPROVED)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary", action="store_true", help="показать очередь без изменений")
    args = parser.parse_args()
    records = load_records(args.queue)
    if args.summary:
        print(json.dumps(summary(records), ensure_ascii=False, indent=2))
        return
    if not records:
        print("Очередь feedback пуста.")
        return
    print(json.dumps(review(args.queue, args.approved, args.audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
