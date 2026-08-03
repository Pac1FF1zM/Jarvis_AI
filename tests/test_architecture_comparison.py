"""Fairness invariants for the frozen baseline and architecture comparison."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.models import MODEL_NAMES, ModelConfig, build_model, trainable_parameter_count


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    ROOT / "configs" / "base.yaml",
    ROOT / "configs" / "mobilenet_tsn.yaml",
    ROOT / "configs" / "r3d18.yaml",
    ROOT / "configs" / "r2plus1d18.yaml",
]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_comparison_configs_preserve_data_and_optimization_protocol() -> None:
    configs = [_load(path) for path in CONFIGS]
    baseline = configs[0]
    for config in configs[1:]:
        assert config["data"] == baseline["data"]
        assert config["train"]["seed"] == baseline["train"]["seed"] == 42
        assert config["train"]["effective_batch_size"] == 8
        for key in (
            "epochs",
            "lr",
            "weight_decay",
            "optimizer",
            "scheduler",
            "warmup_epochs",
            "min_lr_ratio",
            "class_weight_power",
            "amp",
            "num_workers",
        ):
            assert config["train"][key] == baseline["train"][key]
        assert config["evaluation"] == baseline["evaluation"]


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_registry_builds_every_architecture_without_downloading(name: str) -> None:
    model = build_model(ModelConfig(name=name, pretrained=False))
    assert trainable_parameter_count(model) > 0
    assert model.config.name == name
    assert model.config.num_classes == 14
