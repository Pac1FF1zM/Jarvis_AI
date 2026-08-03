"""Comparable pretrained architectures sharing the audited IPN clip contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.video import (
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    r2plus1d_18,
    r3d_18,
)

from .tsn import TSNConfig, TSNResNet18


MODEL_NAMES = (
    "tsn_resnet18",
    "tsn_mobilenet_v3_small",
    "r3d_18",
    "r2plus1d_18",
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int = 14
    pretrained: bool = True
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if self.name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model name {self.name!r}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class _CheckpointModel(nn.Module):
    config: ModelConfig

    def checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "kind": "ipn_gesture_architecture_v1",
            "model_config": asdict(self.config),
            "state_dict": self.state_dict(),
            **extra,
        }


class TSNMobileNetV3Small(_CheckpointModel):
    """ImageNet MobileNetV3-Small per frame followed by temporal mean pooling."""

    feature_dim = 1024

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if config.pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.feature_dim, config.num_classes),
        )

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        _validate_clip(clip)
        batch, time, channels, height, width = clip.shape
        frames = clip.reshape(batch * time, channels, height, width).contiguous(
            memory_format=torch.channels_last
        )
        features = self.backbone(frames)
        return features.reshape(batch, time, self.feature_dim).mean(dim=1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(clip))


class VideoConvNet(_CheckpointModel):
    """Torchvision 3D backbone adapting the common `(B,T,C,H,W)` dataset layout."""

    feature_dim = 512

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.name == "r3d_18":
            weights = R3D_18_Weights.KINETICS400_V1 if config.pretrained else None
            backbone = r3d_18(weights=weights)
        elif config.name == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.KINETICS400_V1 if config.pretrained else None
            backbone = r2plus1d_18(weights=weights)
        else:  # pragma: no cover - guarded by ModelConfig/factory
            raise ValueError(f"Not a 3D architecture: {config.name}")
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.feature_dim, config.num_classes),
        )

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        _validate_clip(clip)
        return self.backbone(clip.permute(0, 2, 1, 3, 4).contiguous())

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(clip))


def _validate_clip(clip: torch.Tensor) -> None:
    if clip.ndim != 5:
        raise ValueError(f"Expected clip shape (B,T,C,H,W), got {tuple(clip.shape)}")
    if clip.shape[2] != 3:
        raise ValueError(f"Expected 3 RGB channels, got {clip.shape[2]}")


def build_model(config: ModelConfig) -> nn.Module:
    if config.name == "tsn_resnet18":
        return TSNResNet18(
            TSNConfig(
                name=config.name,
                num_classes=config.num_classes,
                pretrained=config.pretrained,
                dropout=config.dropout,
            )
        )
    if config.name == "tsn_mobilenet_v3_small":
        return TSNMobileNetV3Small(config)
    return VideoConvNet(config)


def model_config(value: dict[str, Any], *, load_pretrained: bool | None = None) -> ModelConfig:
    fields = dict(value)
    if load_pretrained is not None:
        fields["pretrained"] = load_pretrained
    return ModelConfig(**fields)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
