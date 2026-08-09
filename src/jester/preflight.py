"""Measure real Jester input throughput and CUDA batch safety before training."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.utils import seed_everything
from .dataset import JesterDataset, load_manifest
from .labels import JESTER_LABELS
from .models import JesterModelConfig, MODEL_NAMES, build_model
from .training import _dataset, _seed_worker, load_jester_config, select_safe_batch_size


def _loader_trial(
    config: dict[str, Any], dataset: JesterDataset, *, workers: int, batches: int
) -> dict[str, float | int]:
    batch_size = int(config["train"]["batch_size"])
    options: dict[str, Any] = {}
    if workers:
        options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(config["train"]["pin_memory"]),
        worker_init_fn=_seed_worker,
        **options,
    )
    iterator = iter(loader)
    started = time.perf_counter()
    first_clips, _ = next(iterator)
    first_seconds = time.perf_counter() - started
    measured = first_clips.shape[0]
    steady_started = time.perf_counter()
    for _ in range(max(0, batches - 1)):
        clips, _ = next(iterator)
        measured += clips.shape[0]
    steady_seconds = time.perf_counter() - steady_started
    steady_samples = measured - first_clips.shape[0]
    del iterator, loader
    return {
        "workers": workers,
        "first_batch_seconds": first_seconds,
        "steady_clips_per_second": steady_samples / max(steady_seconds, 1e-9),
        "measured_clips": measured,
    }


def preflight(config_path: Path, *, batches: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Jester preflight")
    config = load_jester_config(config_path)
    seed_everything(int(config["train"]["seed"]))
    records = load_manifest(Path(config["data"]["manifest"]), "train")
    needed = max(256, int(config["train"]["batch_size"]) * (batches + 2))
    dataset = _dataset(config, records[:needed], True)
    worker_trials = [_loader_trial(config, dataset, workers=value, batches=batches) for value in (0, 2, 4)]
    recommended_workers = int(max(worker_trials, key=lambda row: float(row["steady_clips_per_second"]))["workers"])

    batch_trials = []
    device = torch.device("cuda")
    batch_config = copy.deepcopy(config)
    batch_config["train"]["num_workers"] = recommended_workers
    for name in MODEL_NAMES:
        model = build_model(
            JesterModelConfig(
                name=name,
                num_classes=len(JESTER_LABELS),
                dropout=float(config["models"]["dropout"]),
            )
        ).to(device)
        batch_size, memory = select_safe_batch_size(batch_config, model, dataset, device)
        batch_trials.append({"model": name, "safe_batch_size": batch_size, **memory})
        model.zero_grad(set_to_none=True)
        del model
        torch.cuda.empty_cache()
    report: dict[str, object] = {
        "status": "passed",
        "worker_trials": worker_trials,
        "recommended_workers": recommended_workers,
        "batch_trials": batch_trials,
    }
    output = Path(config["paths"]["reports"]) / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()
    if args.batches < 2:
        parser.error("--batches must be at least 2")
    preflight(args.config, batches=args.batches)


if __name__ == "__main__":
    main()
