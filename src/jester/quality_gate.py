"""Short real-data optimization and resume gate before the expensive benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.utils import seed_everything
from .dataset import load_manifest
from .training import balanced_subset, config_fingerprint, configured_models, fit, load_jester_config


def quality_gate(config_path: Path) -> dict[str, object]:
    config = load_jester_config(config_path)
    seed = int(config["train"]["seed"])
    seed_everything(seed)
    manifest = Path(config["data"]["manifest"])
    train_records = balanced_subset(load_manifest(manifest, "train"), 4, seed + 10)
    val_records = balanced_subset(load_manifest(manifest, "val"), 2, seed + 11)
    epochs = 6
    root = Path(config["paths"]["runs"]) / "quality_gate"
    model_reports = []
    for model_name in configured_models(config):
        run_dir = root / model_name
        partial = fit(
            config,
            model_name=model_name,
            train_records=train_records,
            val_records=val_records,
            epochs=epochs,
            run_dir=run_dir,
            resume=False,
            max_epochs_this_run=2,
        )
        if partial["epochs_completed"] != 2:
            raise RuntimeError(f"interruption rehearsal stopped at the wrong epoch for {model_name}")
        completed = fit(
            config,
            model_name=model_name,
            train_records=train_records,
            val_records=val_records,
            epochs=epochs,
            run_dir=run_dir,
            resume=True,
        )
        if completed["resumed_from_epoch"] != 2 or completed["epochs_completed"] != epochs:
            raise RuntimeError(f"continued optimization did not reach epoch {epochs} for {model_name}")
        checkpoint = torch.load(Path(completed["checkpoint"]), map_location="cpu", weights_only=False)
        losses = [float(row["train_loss"]) for row in checkpoint["history"]]
        if len(losses) != epochs or not all(math.isfinite(value) for value in losses):
            raise RuntimeError(f"non-finite or incomplete loss history for {model_name}: {losses}")
        minimum_after_resume = min(losses[2:])
        if minimum_after_resume >= losses[0]:
            raise RuntimeError(f"optimization gate did not reduce training loss for {model_name}: {losses}")
        model_reports.append(
            {
                "model": model_name,
                "resumed_from_epoch": completed["resumed_from_epoch"],
                "epochs_completed": completed["epochs_completed"],
                "initial_train_loss": losses[0],
                "minimum_train_loss_after_resume": minimum_after_resume,
                "best_val_macro_f1": completed["best_val_macro_f1"],
            }
        )
    report: dict[str, object] = {
        "status": "passed",
        "config_fingerprint": config_fingerprint(config),
        "protocol": "real_balanced_optimization_with_epoch_2_resume",
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "models": model_reports,
    }
    output = Path(config["paths"]["reports"]) / "quality_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    args = parser.parse_args()
    quality_gate(args.config)


if __name__ == "__main__":
    main()
