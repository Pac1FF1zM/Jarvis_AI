"""Build the final frozen-baseline versus architecture-comparison report."""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.data.audit import EXPECTED_LABELS
from src.utils import resolve_from_project, write_json


MODELS = [
    {
        "name": "tsn_resnet18",
        "display": "ResNet18-TSN (reference)",
        "training": "training_report.json",
        "evaluation": "evaluation_test.json",
        "checkpoint": "checkpoints/tsn_resnet18_seed42/best.pt",
        "tensorboard": "logs/ipn_tsn/tsn_resnet18_seed42",
        "onnx": "High — 2D Conv backbone; simplest TensorRT/Jetson path",
    },
    {
        "name": "tsn_mobilenet_v3_small",
        "display": "MobileNetV3-Small-TSN",
        "training": "training_tsn_mobilenet_v3_small.json",
        "evaluation": "evaluation_test_tsn_mobilenet_v3_small.json",
        "checkpoint": "checkpoints/tsn_mobilenet_v3_small_seed42/best.pt",
        "tensorboard": "logs/ipn_tsn/tsn_mobilenet_v3_small_seed42",
        "onnx": "Very high — 2D/depthwise Conv; smallest Jetson candidate",
    },
    {
        "name": "r3d_18",
        "display": "R3D-18",
        "training": "training_r3d_18.json",
        "evaluation": "evaluation_test_r3d_18.json",
        "checkpoint": "checkpoints/r3d_18_seed42/best.pt",
        "tensorboard": "logs/ipn_tsn/r3d_18_seed42",
        "onnx": "Medium-low — 3D Conv graph; TensorRT workspace/runtime validation required",
    },
    {
        "name": "r2plus1d_18",
        "display": "R(2+1)D-18",
        "training": "training_r2plus1d_18.json",
        "evaluation": "evaluation_test_r2plus1d_18.json",
        "checkpoint": "checkpoints/r2plus1d_18_seed42/best.pt",
        "tensorboard": "logs/ipn_tsn/r2plus1d_18_seed42",
        "onnx": "Medium — factorized 3D Conv; larger graph than TSN and TensorRT validation required",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tensorboard_median_epoch_seconds(directory: Path) -> float | None:
    events: dict[int, float] = {}
    for event_file in directory.glob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        if "train/loss" not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars("train/loss"):
            events[int(event.step)] = max(events.get(int(event.step), 0.0), float(event.wall_time))
    ordered = sorted(events.items())
    intervals = [
        current[1] - previous[1]
        for previous, current in zip(ordered, ordered[1:])
        if 0 < current[1] - previous[1] < 1800
    ]
    return statistics.median(intervals) if intervals else None


def build_report() -> dict[str, Any]:
    reports = resolve_from_project("reports")
    rows: list[dict[str, Any]] = []
    for specification in MODELS:
        training = _read_json(reports / specification["training"])
        evaluation = _read_json(reports / specification["evaluation"])
        metrics = evaluation["metrics"]
        epoch_seconds = training.get("median_epoch_seconds")
        if epoch_seconds is None:
            epoch_seconds = _tensorboard_median_epoch_seconds(
                resolve_from_project(specification["tensorboard"])
            )
        checkpoint = resolve_from_project(specification["checkpoint"])
        row = {
            "model": specification["name"],
            "display_name": specification["display"],
            "selected_epoch": evaluation["selected_epoch"],
            "val_macro_f1": evaluation["selected_val_macro_f1"],
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_weighted_f1": metrics["weighted_f1"],
            "per_class_f1": {
                label: metrics["per_class"][label]["f1"] for label in EXPECTED_LABELS
            },
            "focus_f1": {
                label: metrics["per_class"][label]["f1"] for label in ("G01", "G02", "G11")
            },
            "trainable_parameters": evaluation.get("trainable_parameters", training.get("trainable_parameters")),
            "median_epoch_seconds": epoch_seconds,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "onnx_exportability": specification["onnx"],
            "test_decode_failures": evaluation["decode_failures"],
        }
        rows.append(row)
    result = {
        "status": "completed",
        "fairness": {
            "manifest": "data/splits/manifest.jsonl",
            "split_instances": {"train": 3313, "val": 726, "test": 1610},
            "strict_subject_disjoint": True,
            "seed": 42,
            "effective_batch_size": 8,
            "selection": "best checkpoint by validation macro-F1",
            "official_test_policy": "one pass per new model after checkpoint selection",
        },
        "models": rows,
    }
    write_json(reports / "architecture_comparison.json", result)
    lines = [
        "# IPN Hand architecture comparison",
        "",
        "| Model | Test acc. | Macro-F1 | Weighted-F1 | Params | Median s/epoch | Checkpoint MiB | ONNX / Jetson |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        seconds = row["median_epoch_seconds"]
        lines.append(
            f"| {row['display_name']} | {row['test_accuracy']:.4f} | {row['test_macro_f1']:.4f} "
            f"| {row['test_weighted_f1']:.4f} | {row['trainable_parameters']:,} "
            f"| {seconds:.1f} | {row['checkpoint_bytes'] / 2**20:.1f} | {row['onnx_exportability']} |"
        )
    lines.extend([
        "",
        "## Per-class F1",
        "",
        "| Model | " + " | ".join(EXPECTED_LABELS) + " |",
        "|---|" + "---:|" * len(EXPECTED_LABELS),
    ])
    for row in rows:
        values = " | ".join(f"{row['per_class_f1'][label]:.4f}" for label in EXPECTED_LABELS)
        lines.append(f"| {row['display_name']} | {values} |")
    lines.extend([
        "",
        "G01, G02, and G11 are the explicitly tracked difficult classes. TSN variants use a 2D "
        "backbone and are materially simpler to export and optimize for Jetson than either 3D-conv graph.",
        "",
    ])
    (reports / "architecture_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2))
