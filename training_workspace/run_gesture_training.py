"""Run fair from-scratch IPN Hand experiments and export only an audited winner."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ml.gesture.data import VideoGestureDataset, audit_segments, load_manifest
from ml.gesture.labels import IPN_LABELS
from ml.gesture.models import GestureModelConfig, build_model, checkpoint_payload
from ml.gesture.training import TrainingSettings, evaluate, seed_everything, train_model


def _resolve(value: str, source: Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (source.parent / candidate).resolve()


def _environment(require_cuda: bool) -> dict[str, Any]:
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError("CUDA is required by this config, but PyTorch cannot see an NVIDIA GPU.")
    return {
        "python": __import__("sys").version,
        "torch": torch.__version__,
        "cuda_available": available,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if available else None,
    }


def _gate(metrics: dict[str, Any], selection: dict[str, Any]) -> list[str]:
    failures = []
    if metrics["macro_f1"] < float(selection["min_test_macro_f1"]):
        failures.append("test macro-F1 below threshold")
    if float(metrics["no_gesture_recall"]) < float(selection["min_no_gesture_recall"]):
        failures.append("no-gesture recall below threshold")
    rate = metrics["false_trigger_rate"]
    if rate is None or float(rate) > float(selection["max_false_trigger_rate"]):
        failures.append("false-trigger rate above threshold")
    return failures


def _loader(dataset: VideoGestureDataset, config: dict[str, Any], *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=str(config.get("device", "cuda")) == "cuda",
        persistent_workers=int(config.get("num_workers", 0)) > 0,
    )


def inspect(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    manifest = _resolve(str(config["data"]["manifest"]), config_path)
    records = load_manifest(manifest)
    audit = audit_segments(records)
    required = {"train", "validation", "test"}
    missing = required - {record.split for record in records}
    if missing:
        raise ValueError(f"Manifest lacks required IPN split(s): {sorted(missing)}")
    return {"environment": _environment(bool(config.get("require_cuda", True))), "data": audit}


def run(config: dict[str, Any], config_path: Path, *, check_only: bool) -> dict[str, Any]:
    report = inspect(config, config_path)
    if check_only:
        return report
    records = load_manifest(_resolve(str(config["data"]["manifest"]), config_path))
    grouped = {split: [record for record in records if record.split == split] for split in ("train", "validation", "test")}
    data_config = config["data"]
    common = {
        "frames": int(data_config["frames"]),
        "image_size": int(data_config["image_size"]),
        "decode_retries": int(data_config.get("decode_retries", 2)),
    }
    train_data = VideoGestureDataset(grouped["train"], training=True, **common)
    validation_data = VideoGestureDataset(grouped["validation"], training=False, **common)
    test_data = VideoGestureDataset(grouped["test"], training=False, **common)

    runs_dir = _resolve(str(config["runs_dir"]), config_path) / datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, Any]] = []
    # The official test split remains unopened until all candidates have been selected on validation.
    for experiment in config["experiments"]:
        print(
            f"GESTURE_EXPERIMENT_START name={experiment['name']} "
            f"architecture={experiment['architecture']} epochs={experiment['epochs']}",
            flush=True,
        )
        seed_everything(int(experiment["seed"]))
        model_config = GestureModelConfig(
            architecture=str(experiment["architecture"]), classes=len(IPN_LABELS),
            width=int(experiment["width"]), dropout=float(experiment["dropout"]),
        )
        model = build_model(model_config)
        settings = TrainingSettings(
            epochs=int(experiment["epochs"]), learning_rate=float(experiment["learning_rate"]),
            weight_decay=float(experiment["weight_decay"]), label_smoothing=float(experiment["label_smoothing"]),
            patience=int(experiment["patience"]), device=str(config["device"]), amp=bool(config["amp"]),
            run_name=str(experiment["name"]),
        )
        trained, history = train_model(
            model,
            _loader(train_data, config, shuffle=True),
            _loader(validation_data, config, shuffle=False),
            settings,
        )
        validation = evaluate(trained, _loader(validation_data, config, shuffle=False), torch.device(settings.device))
        checkpoint = runs_dir / f"{experiment['name']}.pt"
        payload = checkpoint_payload(trained.cpu(), model_config)
        payload.update({"experiment": experiment, "validation": validation, "history": history})
        torch.save(payload, checkpoint)
        candidates.append({"name": experiment["name"], "checkpoint": checkpoint, "validation": validation})
        print(
            f"GESTURE_EXPERIMENT_DONE name={experiment['name']} "
            f"validation_macro_f1={validation['macro_f1']:.4f} "
            f"no_gesture_recall={validation['no_gesture_recall']:.4f}",
            flush=True,
        )

    winner = max(
        candidates,
        key=lambda item: 0.8 * item["validation"]["macro_f1"] + 0.2 * item["validation"]["no_gesture_recall"],
    )
    # Now, and only now, load the held-out official test clips once.
    payload = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(GestureModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    test_metrics = evaluate(model.to(config["device"]), _loader(test_data, config, shuffle=False), torch.device(config["device"]))
    failed_gates = _gate(test_metrics, config["selection"])
    result = {
        **report,
        "runs_dir": str(runs_dir),
        "candidates": [{**item, "checkpoint": str(item["checkpoint"])} for item in candidates],
        "selected": {"name": winner["name"], "checkpoint": str(winner["checkpoint"])},
        "test": test_metrics,
        "approval": {"approved": not failed_gates, "failed_gates": failed_gates},
    }
    if not failed_gates:
        export_dir = _resolve(str(config["export_dir"]), config_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        exported = export_dir / "jarvis_gesture_ipn_best.pt"
        shutil.copy2(winner["checkpoint"], exported)
        result["approval"]["checkpoint"] = str(exported)
    (runs_dir / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Jarvis Gesture Core from scratch on IPN Hand.")
    parser.add_argument("--config", type=Path, default=Path("training_workspace/gesture_config.yaml"))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.config.resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    print(json.dumps(run(config, source, check_only=args.check_only), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
