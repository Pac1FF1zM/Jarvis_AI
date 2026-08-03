"""Shared reproducibility, configuration, and environment helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"data", "train", "model", "overfit", "paths", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")
    return config


def resolve_from_project(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The input shape is fixed for this project, so cuDNN can safely cache the
    # fastest convolution kernels.  TF32 materially improves Ampere throughput
    # without changing the dataset, model, or optimizer schedule.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def ensure_output_directories(config: dict[str, Any]) -> None:
    for value in config["paths"].values():
        resolve_from_project(value).mkdir(parents=True, exist_ok=True)


def cuda_environment() -> dict[str, Any]:
    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if available else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if available else 0,
    }


def cuda_memory_snapshot() -> dict[str, float | int]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "total_bytes": 0,
            "peak_reserved_ratio": 0.0,
        }
    total = torch.cuda.get_device_properties(0).total_memory
    peak_reserved = torch.cuda.max_memory_reserved(0)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": peak_reserved,
        "total_bytes": total,
        "peak_reserved_ratio": peak_reserved / total,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
