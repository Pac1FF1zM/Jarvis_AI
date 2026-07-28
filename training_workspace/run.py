"""Run, compare, and safely export Jarvis NLU fine-tuning experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ml.nlu.custom_data import load_jsonl, validate_splits
from ml.nlu.data import Example, build_examples
from ml.nlu.inference import NLUPredictor
from ml.nlu.schema import INTENTS
from ml.nlu.manager_train import ROUTES

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _macro_f1(expected: list[str], predicted: list[str]) -> float:
    scores = []
    for intent in INTENTS:
        tp = sum(a == intent and b == intent for a, b in zip(expected, predicted))
        fp = sum(a != intent and b == intent for a, b in zip(expected, predicted))
        fn = sum(a == intent and b != intent for a, b in zip(expected, predicted))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return sum(scores) / len(scores)


def benchmark(
    checkpoint: Path,
    examples: list[Any],
    *,
    device: str,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    predictor = NLUPredictor(checkpoint, device=device)
    texts = [example.text for example in examples]
    expected = [example.intent for example in examples]
    predictions = [predictor.predict(text).intent for text in texts]
    for index in range(warmup):
        predictor.predict(texts[index % len(texts)])
    durations: list[float] = []
    for index in range(repetitions):
        text = texts[index % len(texts)]
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        predictor.predict(text)
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(durations)
    p95_index = min(int(len(ordered) * 0.95), len(ordered) - 1)
    per_intent: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    for intent in INTENTS:
        true_positive = sum(a == intent and b == intent for a, b in zip(expected, predictions))
        false_positive = sum(a != intent and b == intent for a, b in zip(expected, predictions))
        false_negative = sum(a == intent and b != intent for a, b in zip(expected, predictions))
        support = sum(a == intent for a in expected)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_intent[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        if support:
            recalls.append(recall)
    route_by_intent = {
        intent: route
        for route, intents in ROUTES
        for intent in intents
    }
    route_accuracy = sum(
        route_by_intent[a] == route_by_intent[b]
        for a, b in zip(expected, predictions)
    ) / len(expected)
    return {
        "examples": len(examples),
        "intent_accuracy": sum(a == b for a, b in zip(expected, predictions)) / len(expected),
        "intent_macro_f1": _macro_f1(expected, predictions),
        "worst_intent_recall": min(recalls),
        "manager_route_accuracy": route_accuracy,
        "per_intent": per_intent,
        "latency_ms_median": statistics.median(durations),
        "latency_ms_p95": ordered[p95_index],
        "device": device,
        "repetitions": repetitions,
    }


def _load_intent_only_jsonl(path: Path) -> list[Example]:
    """Load a frozen intent holdout without exposing its slots to training."""
    examples: list[Example] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        record = json.loads(raw_line)
        text = str(record.get("text", "")).strip()
        intent = str(record.get("intent", "")).strip()
        if not text or intent not in INTENTS:
            raise ValueError(f"{path}:{line_number}: invalid frozen holdout record")
        examples.append(Example(text, intent))
    if not examples:
        raise ValueError(f"{path}: frozen holdout is empty")
    return examples


def _example_record(example: Example) -> dict[str, Any]:
    """Return the friendly JSONL representation expected by manager_train."""
    slots = {
        span.label: example.text[span.start:span.end]
        for span in example.spans
    }
    return {"text": example.text, "intent": example.intent, "slots": slots}


def _write_examples_jsonl(path: Path, examples: list[Example]) -> None:
    content = "".join(
        json.dumps(_example_record(example), ensure_ascii=False, separators=(",", ":")) + "\n"
        for example in examples
    )
    path.write_text(content, encoding="utf-8", newline="\n")


def _harmonic_mean(left: float, right: float) -> float:
    return 2.0 * left * right / max(left + right, 1e-12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(require_cuda: bool) -> dict[str, Any]:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError(
            "RTX/CUDA не обнаружена. Установите CUDA-сборку PyTorch и проверьте "
            "torch.cuda.is_available() перед обучением."
        )
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if available else None,
    }


def run(config_path: Path, *, check_only: bool = False) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    environment = _environment(bool(raw.get("require_cuda", True)) and not check_only)
    base_checkpoint = _resolve(raw["base_checkpoint"], config_path)
    train_path = _resolve(raw["data"]["train"], config_path)
    validation_path = _resolve(raw["data"]["validation"], config_path)
    evaluation_holdout_path = _resolve(raw["data"]["evaluation_holdout"], config_path)
    final_holdout_path = _resolve(raw["data"]["final_holdout"], config_path)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    custom_train = load_jsonl(train_path, allow_empty=False)
    feedback_path = _resolve(
        raw["data"].get("feedback_train", "data/feedback_train.jsonl"),
        config_path,
    )
    reviewed_feedback = load_jsonl(feedback_path, allow_empty=True)
    custom_validation = load_jsonl(validation_path, allow_empty=False)
    base_texts = {example.text.casefold().strip() for example in custom_train}
    duplicate_feedback = base_texts & {
        example.text.casefold().strip() for example in reviewed_feedback
    }
    if duplicate_feedback:
        raise ValueError(
            "reviewed feedback duplicates base train examples: "
            f"{sorted(duplicate_feedback)[:3]!r}"
        )
    combined_train = custom_train + reviewed_feedback
    data_report = validate_splits(combined_train, custom_validation)
    evaluation_holdout = load_jsonl(evaluation_holdout_path, allow_empty=False)
    final_holdout = _load_intent_only_jsonl(final_holdout_path)
    # The legacy baseline predates this feedback flow and has its own frozen
    # comparison policy. Enforce the stricter no-leakage rule for newly added,
    # human-reviewed examples without retroactively invalidating that baseline.
    feedback_texts = {example.text.casefold().strip() for example in reviewed_feedback}
    for name, holdout in (("evaluation_holdout", evaluation_holdout), ("final_holdout", final_holdout)):
        overlap = feedback_texts & {example.text.casefold().strip() for example in holdout}
        if overlap:
            raise ValueError(
                f"reviewed feedback overlaps {name}: {sorted(overlap)[:3]!r}"
            )
    legacy_validation = build_examples("validation")
    legacy_regression = build_examples("test")
    data_report.update(
        {
            "evaluation_holdout_examples": len(evaluation_holdout),
            "final_holdout_examples": len(final_holdout),
            "reviewed_feedback_examples": len(reviewed_feedback),
        }
    )
    print(
        json.dumps(
            {"environment": environment, "data": data_report},
            ensure_ascii=False,
            indent=2,
        )
    )
    if check_only:
        return {"status": "configuration_ok", "environment": environment, "data": data_report}

    run_root = _resolve(raw.get("runs_dir", "runs"), config_path)
    export_dir = _resolve(raw.get("export_dir", "export"), config_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    approval_path = export_dir / "approved.json"
    # An old checkpoint may remain for forensic comparison, but it cannot be
    # copied after a new failed run because every run invalidates approval.
    approval_path.unlink(missing_ok=True)
    run_dir = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    training_path = train_path
    if reviewed_feedback:
        training_path = run_dir / "train_with_reviewed_feedback.jsonl"
        _write_examples_jsonl(training_path, combined_train)
    benchmark_cfg = raw.get("benchmark") or {}
    benchmark_device = str(benchmark_cfg.get("device", "cpu"))
    warmup = int(benchmark_cfg.get("warmup", 20))
    repetitions = int(benchmark_cfg.get("repetitions", 300))
    benchmark_kwargs = {
        "device": benchmark_device,
        "warmup": warmup,
        "repetitions": repetitions,
    }
    baseline = {
        "custom_validation": benchmark(base_checkpoint, custom_validation, **benchmark_kwargs),
        "legacy_validation": benchmark(base_checkpoint, legacy_validation, **benchmark_kwargs),
        "legacy_regression": benchmark(base_checkpoint, legacy_regression, **benchmark_kwargs),
        "evaluation_holdout": benchmark(base_checkpoint, evaluation_holdout, **benchmark_kwargs),
        "final_holdout": benchmark(base_checkpoint, final_holdout, **benchmark_kwargs),
    }
    print("BASELINE", json.dumps(baseline, ensure_ascii=False))

    results: list[dict[str, Any]] = []
    for experiment in raw.get("experiments") or []:
        name = str(experiment["name"])
        output = run_dir / f"{name}.pt"
        trainer = str(experiment.get("trainer", "manager"))
        if trainer != "manager":
            raise ValueError(f"unsupported trainer {trainer!r}; expected 'manager'")
        command = [
            sys.executable, "-m", "ml.nlu.manager_train",
            "--train-data", str(training_path),
            "--validation-data", str(validation_path),
            "--output", str(output),
            "--architecture", str(experiment.get("architecture", "char_cnn")),
            "--device", str(raw.get("training_device", "cuda")),
            "--method", str(experiment.get("method", "curriculum")),
            "--epochs", str(experiment.get("epochs", 60)),
            "--batch-size", str(experiment.get("batch_size", 256)),
            "--learning-rate", str(experiment.get("learning_rate", 1e-3)),
            "--custom-fraction", str(experiment.get("custom_fraction", 0.4)),
            "--curriculum-start-fraction", str(
                experiment.get("curriculum_start_fraction", 0.2)
            ),
            "--route-loss-weight", str(experiment.get("route_loss_weight", 0.35)),
            "--slot-loss-weight", str(experiment.get("slot_loss_weight", 0.6)),
            "--embedding-dim", str(experiment.get("embedding_dim", 48)),
            "--hidden-dim", str(experiment.get("hidden_dim", 64)),
            "--max-length", str(experiment.get("max_length", 128)),
            "--patience", str(experiment.get("patience", 12)),
            "--seed", str(experiment.get("seed", 17)),
            "--label-smoothing", str(experiment.get("label_smoothing", 0.04)),
        ]
        print(f"\n=== EXPERIMENT {name} ===", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        measured = {
            "custom_validation": benchmark(output, custom_validation, **benchmark_kwargs),
            "legacy_validation": benchmark(output, legacy_validation, **benchmark_kwargs),
            "legacy_regression": benchmark(output, legacy_regression, **benchmark_kwargs),
        }
        result = {
            "name": name,
            "checkpoint": str(output),
            "metrics": metrics,
            "benchmarks": measured,
        }
        results.append(result)
        print("RESULT", json.dumps(result, ensure_ascii=False))

    if not results:
        raise ValueError("config contains no experiments")
    selection = raw.get("selection") or {}
    min_custom_improvement = float(
        selection.get("min_custom_macro_f1_improvement", 0.05)
    )
    max_legacy_drop = float(selection.get("max_legacy_macro_f1_drop", 0.0))
    min_regression_recall = float(selection.get("min_regression_worst_recall", 0.7))
    max_p95 = float(selection.get("max_p95_latency_ms", float("inf")))
    eligible: list[dict[str, Any]] = []
    eligibility: dict[str, list[str]] = {}
    for result in results:
        measured = result["benchmarks"]
        failures: list[str] = []
        if measured["custom_validation"]["intent_macro_f1"] < (
            baseline["custom_validation"]["intent_macro_f1"] + min_custom_improvement
        ):
            failures.append("custom_validation_macro_f1")
        for dataset_name in ("legacy_validation", "legacy_regression"):
            if measured[dataset_name]["intent_macro_f1"] < (
                baseline[dataset_name]["intent_macro_f1"] - max_legacy_drop
            ):
                failures.append(f"{dataset_name}_macro_f1")
        if measured["legacy_regression"]["worst_intent_recall"] < min_regression_recall:
            failures.append("legacy_regression_worst_intent_recall")
        if measured["custom_validation"]["latency_ms_p95"] > max_p95:
            failures.append("latency_p95")
        eligibility[result["name"]] = failures
        if not failures:
            result["selection_score"] = _harmonic_mean(
                measured["custom_validation"]["intent_macro_f1"],
                measured["legacy_regression"]["intent_macro_f1"],
            )
            eligible.append(result)

    report: dict[str, Any] = {
        "environment": environment,
        "data": data_report,
        "baseline": baseline,
        "experiments": results,
        "selection": {
            "eligible": [item["name"] for item in eligible],
            "eligibility_failures": eligibility,
            "status": "rejected",
        },
    }
    if eligible:
        best = max(
            eligible,
            key=lambda item: (
                item["selection_score"],
                -item["benchmarks"]["custom_validation"]["latency_ms_p95"],
            ),
        )
        best_checkpoint = Path(best["checkpoint"])
        holdouts = {
            "evaluation_holdout": benchmark(
                best_checkpoint, evaluation_holdout, **benchmark_kwargs
            ),
            "final_holdout": benchmark(best_checkpoint, final_holdout, **benchmark_kwargs),
        }
        report["holdouts"] = holdouts
        min_evaluation_improvement = float(
            selection.get("min_evaluation_holdout_macro_f1_improvement", 0.0)
        )
        min_final_improvement = float(
            selection.get("min_final_holdout_macro_f1_improvement", 0.0)
        )
        min_holdout_recall = float(selection.get("min_holdout_worst_recall", 0.6))
        final_failures: list[str] = []
        if holdouts["evaluation_holdout"]["intent_macro_f1"] < (
            baseline["evaluation_holdout"]["intent_macro_f1"]
            + min_evaluation_improvement
        ):
            final_failures.append("evaluation_holdout_macro_f1")
        if holdouts["final_holdout"]["intent_macro_f1"] < (
            baseline["final_holdout"]["intent_macro_f1"] + min_final_improvement
        ):
            final_failures.append("final_holdout_macro_f1")
        if holdouts["final_holdout"]["intent_accuracy"] < baseline["final_holdout"]["intent_accuracy"]:
            final_failures.append("final_holdout_accuracy")
        if holdouts["evaluation_holdout"]["worst_intent_recall"] < min_holdout_recall:
            final_failures.append("evaluation_holdout_worst_intent_recall")
        if holdouts["final_holdout"]["worst_intent_recall"] < min_holdout_recall:
            final_failures.append("final_holdout_worst_intent_recall")

        report["selection"].update(
            {
                "provisional_best": best["name"],
                "final_gate_failures": final_failures,
            }
        )
        if not final_failures:
            exported = export_dir / "jarvis_nlu_best.pt"
            exported_metrics = export_dir / "jarvis_nlu_best.metrics.json"
            shutil.copy2(best_checkpoint, exported)
            shutil.copy2(best_checkpoint.with_suffix(".metrics.json"), exported_metrics)
            approval = {
                "approved": True,
                "checkpoint": str(exported),
                "sha256": _sha256(exported),
                "experiment": best["name"],
                "run": str(run_dir),
            }
            approval_path.write_text(
                json.dumps(approval, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report["selection"].update(
                {
                    "status": "approved",
                    "best": best["name"],
                    "exported": str(exported),
                    "approval": str(approval_path),
                }
            )
        else:
            report["selection"]["reason"] = "provisional winner failed final holdout gates"
    else:
        report["selection"]["reason"] = "no candidate passed development and regression gates"
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nREPORT {report_path}")
    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training_workspace/config.yaml")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(Path(args.config).resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
