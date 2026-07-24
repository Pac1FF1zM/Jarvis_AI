"""Evaluate a frozen NLU checkpoint on data never read by training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .inference import NLUPredictor
from .schema import INTENTS


def _load_holdout(path: Path) -> list[dict]:
    expected_path = path.with_suffix(".sha256")
    if expected_path.exists():
        expected = expected_path.read_text(encoding="ascii").strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"holdout checksum mismatch: expected {expected}, got {actual}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(checkpoint: Path, holdout: Path, threshold: float) -> dict:
    predictor = NLUPredictor(checkpoint)
    examples = _load_holdout(holdout)
    confusion: Counter[tuple[str, str]] = Counter()
    correct = frame_correct = 0
    failures: list[dict] = []
    for example in examples:
        result = predictor.predict(example["text"])
        predicted = result.intent if result.confidence >= threshold else "unknown"
        confusion[(example["intent"], predicted)] += 1
        intent_ok = predicted == example["intent"]
        slots_ok = result.slots == example.get("slots", {}) if intent_ok else False
        correct += int(intent_ok)
        frame_correct += int(intent_ok and slots_ok)
        if not intent_ok or not slots_ok:
            failures.append({
                "text": example["text"], "expected": example["intent"],
                "predicted": predicted, "confidence": round(result.confidence, 4),
                "expected_slots": example.get("slots", {}), "predicted_slots": result.slots,
            })
    f1s = []
    per_intent = {}
    for intent in INTENTS:
        tp = confusion[(intent, intent)]
        fp = sum(confusion[(other, intent)] for other in INTENTS if other != intent)
        fn = sum(confusion[(intent, other)] for other in INTENTS if other != intent)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        per_intent[intent] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "examples": len(examples), "threshold": threshold,
        "intent_accuracy": correct / len(examples),
        "intent_macro_f1": sum(f1s) / len(f1s),
        "exact_frame_accuracy": frame_correct / len(examples),
        "per_intent": per_intent, "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/nlu_word_bigru_curriculum.pt")
    parser.add_argument("--holdout", default="ml/nlu/holdout_v2.jsonl")
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()
    report = evaluate(Path(args.checkpoint), Path(args.holdout), args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
