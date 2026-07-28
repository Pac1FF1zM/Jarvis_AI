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
from ml.nlu.metrics import expected_calibration_error, semantic_frame_metrics
from ml.nlu.schema import INTENTS
from ml.nlu.manager_train import ROUTES
from training_workspace.nlu_search import (
    aggregate_scores,
    confirmation_experiments,
    generate_phase_one,
)

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
    prediction_results = [predictor.predict(text) for text in texts]
    predictions = [result.intent for result in prediction_results]
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
        **semantic_frame_metrics(examples, prediction_results),
        "expected_calibration_error": expected_calibration_error(
            expected, prediction_results
        ),
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


def _development_score(benchmarks: dict[str, dict[str, Any]]) -> float:
    """Rank candidates on complete development behavior, never on holdout."""
    custom = benchmarks["custom_validation"]
    regression = benchmarks["legacy_regression"]
    intent = _harmonic_mean(custom["intent_macro_f1"], regression["intent_macro_f1"])
    score = (
        0.50 * intent
        + 0.20 * custom["slot_entity_f1"]
        + 0.20 * custom["semantic_frame_exact_match"]
        + 0.10 * custom["end_to_end_command_accuracy"]
    )
    return score - 0.20 * custom["slot_hallucination_rate"] - 0.10 * custom[
        "expected_calibration_error"
    ]


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
    search = raw.get("search") or {}
    search_enabled = bool(search.get("enabled", False))
    phase_one_experiments = (
        generate_phase_one(search) if search_enabled else list(raw.get("experiments") or [])
    )
    if not phase_one_experiments:
        raise ValueError("config contains no experiments")
    data_report["search"] = (
        {
            "enabled": True,
            "phase_one_trials": len(phase_one_experiments),
            "phase_one_seed": int(search.get("phase_one_seed", 17)),
            "top_k": int(search.get("top_k", 3)),
            "confirmation_seeds": list(
                search.get("confirmation_seeds", [17, 43, 101, 211, 307])
            ),
        }
        if search_enabled
        else {"enabled": False, "experiments": len(phase_one_experiments)}
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
    }
    print("BASELINE", json.dumps(baseline, ensure_ascii=False))

    def execute_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
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
            "--weight-decay", str(experiment.get("weight_decay", 1e-4)),
            "--custom-fraction", str(experiment.get("custom_fraction", 0.4)),
            "--curriculum-start-fraction", str(
                experiment.get("curriculum_start_fraction", 0.2)
            ),
            "--route-loss-weight", str(experiment.get("route_loss_weight", 0.35)),
            "--slot-loss-weight", str(experiment.get("slot_loss_weight", 0.6)),
            "--slot-consistency-weight", str(
                experiment.get("slot_consistency_weight", 0.3)
            ),
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
            "config": dict(experiment),
            "checkpoint": str(output),
            "metrics": metrics,
            "benchmarks": measured,
            "selection_score": _development_score(measured),
        }
        print("RESULT", json.dumps(result, ensure_ascii=False))
        return result

    phase_one_results = [execute_experiment(item) for item in phase_one_experiments]
    search_report: dict[str, Any] | None = None
    if search_enabled:
        top_k = int(search.get("top_k", 3))
        if not 1 <= top_k <= len(phase_one_results):
            raise ValueError("search.top_k must be between 1 and search.trials")
        ranked = sorted(
            phase_one_results,
            key=lambda item: (
                item["selection_score"],
                -item["benchmarks"]["custom_validation"]["latency_ms_p95"],
            ),
            reverse=True,
        )
        finalist_names = {item["name"] for item in ranked[:top_k]}
        finalists = [item for item in phase_one_experiments if item["name"] in finalist_names]
        confirmation_specs = confirmation_experiments(finalists, search)
        results = []
        for experiment in confirmation_specs:
            result = execute_experiment(experiment)
            result["candidate"] = experiment["candidate"]
            result["seed"] = experiment["seed"]
            results.append(result)
        search_report = {
            "method": "two_stage_successive_halving",
            "phase_one_seed": int(search.get("phase_one_seed", 17)),
            "phase_one": phase_one_results,
            "finalists": sorted(finalist_names),
            "confirmation_seeds": list(
                search.get("confirmation_seeds", [17, 43, 101, 211, 307])
            ),
        }
    else:
        results = phase_one_results
    selection = raw.get("selection") or {}
    min_custom_improvement = float(
        selection.get("min_custom_macro_f1_improvement", 0.05)
    )
    max_legacy_drop = float(selection.get("max_legacy_macro_f1_drop", 0.0))
    min_regression_recall = float(selection.get("min_regression_worst_recall", 0.7))
    max_p95 = float(selection.get("max_p95_latency_ms", float("inf")))
    min_intent_f1 = float(selection.get("min_intent_macro_f1", 0.0))
    min_worst_recall = float(selection.get("min_worst_intent_recall", 0.0))
    min_slot_f1 = float(selection.get("min_slot_entity_f1", 0.0))
    max_hallucination = float(selection.get("max_slot_hallucination_rate", 1.0))
    min_frame_exact = float(selection.get("min_semantic_frame_exact_match", 0.0))
    min_command_accuracy = float(selection.get("min_end_to_end_command_accuracy", 0.0))
    max_ece = float(selection.get("max_expected_calibration_error", 1.0))
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
        custom = measured["custom_validation"]
        if custom["intent_macro_f1"] < min_intent_f1:
            failures.append("custom_validation_min_intent_macro_f1")
        if custom["worst_intent_recall"] < min_worst_recall:
            failures.append("custom_validation_worst_intent_recall")
        if custom["slot_entity_f1"] < min_slot_f1:
            failures.append("custom_validation_slot_entity_f1")
        if custom["slot_hallucination_rate"] > max_hallucination:
            failures.append("custom_validation_slot_hallucination_rate")
        if custom["semantic_frame_exact_match"] < min_frame_exact:
            failures.append("custom_validation_semantic_frame_exact_match")
        if custom["end_to_end_command_accuracy"] < min_command_accuracy:
            failures.append("custom_validation_end_to_end_command_accuracy")
        if custom["expected_calibration_error"] > max_ece:
            failures.append("custom_validation_expected_calibration_error")
        if custom["latency_ms_p95"] > max_p95:
            failures.append("latency_p95")
        eligibility[result["name"]] = failures
        if not failures:
            eligible.append(result)

    candidate_summaries: dict[str, dict[str, Any]] = {}
    if search_enabled:
        candidate_summaries = aggregate_scores(results)
        required_passes = int(
            search.get(
                "min_passing_seeds",
                max(1, len(set(search.get("confirmation_seeds", [17, 43, 101, 211, 307]))) - 1),
            )
        )
        for candidate, summary in candidate_summaries.items():
            passing = sum(str(item.get("candidate")) == candidate for item in eligible)
            summary["passing_runs"] = passing
            summary["eligible"] = passing >= required_passes
        robust_candidates = {
            name for name, summary in candidate_summaries.items() if summary["eligible"]
        }
        eligible = [
            item for item in eligible if str(item.get("candidate")) in robust_candidates
        ]

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
    if search_report is not None:
        search_report["candidate_summaries"] = candidate_summaries
        report["search"] = search_report
    if eligible:
        if search_enabled:
            winning_candidate = max(
                {str(item["candidate"]) for item in eligible},
                key=lambda name: (
                    candidate_summaries[name]["mean"],
                    candidate_summaries[name]["minimum"],
                    -candidate_summaries[name]["stddev"],
                ),
            )
            candidate_runs = [
                item for item in eligible if str(item["candidate"]) == winning_candidate
            ]
            median_score = statistics.median(item["selection_score"] for item in candidate_runs)
            # Export a representative run, not the luckiest seed.
            best = min(
                candidate_runs,
                key=lambda item: (
                    abs(item["selection_score"] - median_score),
                    item["benchmarks"]["custom_validation"]["latency_ms_p95"],
                ),
            )
        else:
            best = max(
                eligible,
                key=lambda item: (
                    item["selection_score"],
                    -item["benchmarks"]["custom_validation"]["latency_ms_p95"],
                ),
            )
        best_checkpoint = Path(best["checkpoint"])
        baseline.update(
            {
                "evaluation_holdout": benchmark(
                    base_checkpoint, evaluation_holdout, **benchmark_kwargs
                ),
                "final_holdout": benchmark(
                    base_checkpoint, final_holdout, **benchmark_kwargs
                ),
            }
        )
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
        evaluation = holdouts["evaluation_holdout"]
        if evaluation["slot_entity_f1"] < float(
            selection.get("min_holdout_slot_entity_f1", min_slot_f1)
        ):
            final_failures.append("evaluation_holdout_slot_entity_f1")
        if evaluation["slot_hallucination_rate"] > float(
            selection.get("max_holdout_slot_hallucination_rate", max_hallucination)
        ):
            final_failures.append("evaluation_holdout_slot_hallucination_rate")
        if evaluation["semantic_frame_exact_match"] < float(
            selection.get("min_holdout_semantic_frame_exact_match", min_frame_exact)
        ):
            final_failures.append("evaluation_holdout_semantic_frame_exact_match")
        if evaluation["end_to_end_command_accuracy"] < float(
            selection.get("min_holdout_end_to_end_command_accuracy", min_command_accuracy)
        ):
            final_failures.append("evaluation_holdout_end_to_end_command_accuracy")

        report["selection"].update(
            {
                "provisional_best": best["name"],
                "provisional_candidate": best.get("candidate", best["name"]),
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
                "candidate": best.get("candidate", best["name"]),
                "seed": best.get("seed", best["config"].get("seed")),
                "training_config": best["config"],
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
