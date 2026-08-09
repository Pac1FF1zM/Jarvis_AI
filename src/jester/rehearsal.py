"""Run a one-step-per-model checkpoint/resume rehearsal on real Jester clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils import seed_everything
from .dataset import load_manifest
from .models import MODEL_NAMES
from .training import balanced_subset, fit, load_jester_config


def rehearse(config_path: Path) -> dict[str, object]:
    config = load_jester_config(config_path)
    seed = int(config["train"]["seed"])
    seed_everything(seed)
    manifest = Path(config["data"]["manifest"])
    train_records = balanced_subset(load_manifest(manifest, "train"), 1, seed)
    val_records = balanced_subset(load_manifest(manifest, "val"), 1, seed + 1)
    root = Path(config["paths"]["runs"]) / "rehearsal"
    results = []
    for model_name in MODEL_NAMES:
        run_dir = root / model_name
        initial = fit(
            config,
            model_name=model_name,
            train_records=train_records,
            val_records=val_records,
            epochs=1,
            run_dir=run_dir,
            resume=False,
        )
        resumed = fit(
            config,
            model_name=model_name,
            train_records=train_records,
            val_records=val_records,
            epochs=1,
            run_dir=run_dir,
            resume=True,
        )
        if resumed["resumed_from_epoch"] != 1:
            raise RuntimeError(f"resume rehearsal failed for {model_name}")
        results.append(
            {
                "model": model_name,
                "checkpoint": initial["checkpoint"],
                "resumed_from_epoch": resumed["resumed_from_epoch"],
                "finite_macro_f1": 0.0 <= float(resumed["best_val_macro_f1"]) <= 1.0,
            }
        )
    report: dict[str, object] = {
        "status": "passed",
        "protocol": "one_balanced_real_step_then_resume",
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "models": results,
    }
    output = Path(config["paths"]["reports"]) / "rehearsal.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    args = parser.parse_args()
    rehearse(args.config)


if __name__ == "__main__":
    main()
