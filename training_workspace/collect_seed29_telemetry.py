"""Collect fresh, labelled, no-action telemetry for the release seed29 JSC."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.inference import StructuredJSCPredictor
from ml.jsc.jal import dumps, loads
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.structured_decoding import plan_completeness_issues


def _known_texts(root: Path) -> set[str]:
    texts: set[str] = set()
    for path in (root / "training_workspace" / "jsc_data").glob("*.jsonl"):
        for raw in path.read_text("utf-8").splitlines():
            row = json.loads(raw)
            text = str(row.get("text", "")).strip().casefold().replace("ё", "е")
            if text:
                texts.add(text)
    return texts


def collect(probe_path: Path, checkpoint_path: Path, *, device: str = "cpu") -> dict[str, Any]:
    known = _known_texts(ROOT)
    rows = [json.loads(line) for line in probe_path.read_text("utf-8").splitlines() if line.strip()]
    overlaps = [row["id"] for row in rows if str(row["text"]).casefold().replace("ё", "е") in known]
    if overlaps:
        raise ValueError(f"probe leaks exact train/validation texts: {overlaps}")
    predictor = StructuredJSCPredictor(
        checkpoint_path, build_project_schema_registry(), device=device,
        thresholds={
            "minimum_act_confidence": 0.65,
            "minimum_act_margin": 0.15,
            "maximum_normalized_entropy": 0.65,
            "minimum_verifier_confidence": 0.90,
        },
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        text = str(row["text"])
        expected = dumps(loads(json.dumps(row["expected_jal"], ensure_ascii=False)))
        prediction = predictor.predict(text)
        plan = loads(prediction.jal)
        records.append(
            {
                "id": row["id"],
                "source": "fresh_offline_labelled_probe_not_voice",
                "text": text,
                "expected_jal": expected,
                "predicted_jal": prediction.jal,
                "exact": prediction.jal == expected,
                "decisions": dict(prediction.decisions),
                "risk": dict(prediction.risk),
                "completeness_issues": list(plan_completeness_issues(text, plan)),
                "latency_ms": round(prediction.latency_ms, 3),
                "executed_by_jsc": False,
            }
        )
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "seed": 29,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "provenance": "fresh_offline_labelled_probe_not_live_voice",
        "execution": "blocked",
        "samples": len(records),
        "exact_jal_accuracy": sum(row["exact"] for row in records) / max(len(records), 1),
        "abstentions": sum("abstention" in row["decisions"] for row in records),
        "completeness_failures": sum(bool(row["completeness_issues"]) for row in records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default="training_workspace/jsc_seed29_probe_v1.jsonl")
    parser.add_argument("--checkpoint", default="models/jsc/structured_v8_seed29.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = collect(Path(args.probe), Path(args.checkpoint), device=args.device)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
