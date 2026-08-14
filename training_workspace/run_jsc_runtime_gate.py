"""Evaluate the exact packaged Structured JSC predictor on the migration suite."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.inference import StructuredJSCPredictor
from ml.jsc.project_registry import build_project_schema_registry
from training_workspace.build_jsc_migration_suite import build
from training_workspace.jsc_migration_benchmark import _gate_report, _migration_metrics


def run(checkpoint: Path, *, device: str) -> dict[str, object]:
    registry = build_project_schema_registry()
    predictor = StructuredJSCPredictor(
        checkpoint,
        registry,
        device=device,
        thresholds={
            "execution_threshold": 0.65,
            "verifier_threshold": 0.90,
            "parameter_threshold": 0.35,
            "span_threshold": 0.25,
            "missing_threshold": 0.35,
            "minimum_act_confidence": 0.65,
            "minimum_act_margin": 0.15,
            "maximum_normalized_entropy": 0.65,
            "minimum_verifier_confidence": 0.90,
        },
    )
    examples = build()
    predictions: list[str] = []
    latencies: list[float] = []
    risks: dict[str, int] = {}
    for example in examples:
        result = predictor.predict(
            example.text, history=example.history, state=example.state
        )
        predictions.append(result.jal)
        latencies.append(result.latency_ms)
        reason = str(result.risk.get("reason", "unknown"))
        risks[reason] = risks.get(reason, 0) + 1
    metrics = _migration_metrics(examples, predictions, registry)
    gates = _gate_report(metrics)
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": "packaged_structured_jsc_no_action",
        "execution": "blocked",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "samples": len(examples),
        "metrics": metrics,
        "selective_risk_reasons": risks,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "max": max(latencies),
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/jsc/structured_v8_seed29.pt")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(Path(args.checkpoint), device=args.device)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
