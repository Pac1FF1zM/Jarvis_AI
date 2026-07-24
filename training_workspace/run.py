"""Run, compare, and safely export Jarvis NLU fine-tuning experiments."""
from __future__ import annotations

import argparse
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
from ml.nlu.data import build_examples
from ml.nlu.inference import NLUPredictor
from ml.nlu.schema import INTENTS

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
    return {
        "examples": len(examples),
        "intent_accuracy": sum(a == b for a, b in zip(expected, predictions)) / len(expected),
        "intent_macro_f1": _macro_f1(expected, predictions),
        "latency_ms_median": statistics.median(durations),
        "latency_ms_p95": ordered[p95_index],
        "device": device,
        "repetitions": repetitions,
    }


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
    private_holdout_path = _resolve(raw["data"]["private_holdout"], config_path)
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    custom_train = load_jsonl(train_path)
    custom_validation = load_jsonl(validation_path)
    data_report = validate_splits(custom_train, custom_validation)
    if not custom_train:
        raise ValueError(f"{train_path}: добавьте примеры для fine-tuning")
    validation_examples = build_examples("validation") + custom_validation
    private_holdout = load_jsonl(private_holdout_path)
    print(json.dumps({"environment": environment, "data": data_report}, ensure_ascii=False, indent=2))
    if check_only:
        return {"status": "configuration_ok", "environment": environment, "data": data_report}

    run_root = _resolve(raw.get("runs_dir", "runs"), config_path)
    export_dir = _resolve(raw.get("export_dir", "export"), config_path)
    run_dir = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    benchmark_cfg = raw.get("benchmark") or {}
    benchmark_device = str(benchmark_cfg.get("device", "cpu"))
    warmup = int(benchmark_cfg.get("warmup", 20))
    repetitions = int(benchmark_cfg.get("repetitions", 300))
    baseline = benchmark(
        base_checkpoint, validation_examples, device=benchmark_device,
        warmup=warmup, repetitions=repetitions,
    )
    print("BASELINE", json.dumps(baseline, ensure_ascii=False))

    results: list[dict[str, Any]] = []
    for experiment in raw.get("experiments") or []:
        name = str(experiment["name"])
        output = run_dir / f"{name}.pt"
        command = [
            sys.executable, "-m", "ml.nlu.finetune",
            "--checkpoint", str(base_checkpoint),
            "--train-data", str(train_path),
            "--validation-data", str(validation_path),
            "--output", str(output),
            "--device", str(raw.get("training_device", "cuda")),
            "--method", str(experiment.get("method", "curriculum")),
            "--epochs", str(experiment.get("epochs", 60)),
            "--batch-size", str(experiment.get("batch_size", 256)),
            "--learning-rate", str(experiment.get("learning_rate", 5e-4)),
            "--custom-repeat", str(experiment.get("custom_repeat", 4)),
            "--patience", str(experiment.get("patience", 10)),
            "--seed", str(experiment.get("seed", 17)),
            "--label-smoothing", str(experiment.get("label_smoothing", 0.05)),
            "--require-custom-data",
        ]
        print(f"\n=== EXPERIMENT {name} ===", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        metrics = json.loads(output.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        measured = benchmark(
            output, validation_examples, device=benchmark_device,
            warmup=warmup, repetitions=repetitions,
        )
        result = {"name": name, "checkpoint": str(output), "metrics": metrics, "benchmark": measured}
        results.append(result)
        print("RESULT", json.dumps(result, ensure_ascii=False))

    if not results:
        raise ValueError("config contains no experiments")
    selection = raw.get("selection") or {}
    min_improvement = float(selection.get("min_macro_f1_improvement", 0.0))
    max_p95 = float(selection.get("max_p95_latency_ms", float("inf")))
    eligible = [
        result for result in results
        if result["benchmark"]["intent_macro_f1"] >= baseline["intent_macro_f1"] + min_improvement
        and result["benchmark"]["latency_ms_p95"] <= max_p95
    ]
    report: dict[str, Any] = {
        "environment": environment,
        "data": data_report,
        "baseline": baseline,
        "experiments": results,
        "selection": {"eligible": [item["name"] for item in eligible]},
    }
    if eligible:
        best = max(
            eligible,
            key=lambda item: (
                item["benchmark"]["intent_macro_f1"],
                -item["benchmark"]["latency_ms_p95"],
            ),
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        exported = export_dir / "jarvis_nlu_best.pt"
        shutil.copy2(best["checkpoint"], exported)
        shutil.copy2(Path(best["checkpoint"]).with_suffix(".metrics.json"), exported.with_suffix(".metrics.json"))
        report["selection"].update({"best": best["name"], "exported": str(exported)})
        if private_holdout:
            report["private_holdout"] = benchmark(
                exported, private_holdout, device=benchmark_device,
                warmup=warmup, repetitions=repetitions,
            )
    else:
        report["selection"]["reason"] = "no candidate beat baseline within latency limits"
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
