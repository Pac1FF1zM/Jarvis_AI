"""Temporal Segment Network baseline with an ImageNet ResNet-18 backbone."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True)
class TSNConfig:
    name: str = "tsn_resnet18"
    num_classes: int = 14
    pretrained: bool = True
    dropout: float = 0.20

    def __post_init__(self) -> None:
        if self.name != "tsn_resnet18":
            raise ValueError(f"Unsupported model name {self.name!r}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class TSNResNet18(nn.Module):
    """Encode frames independently, mean-pool features over time, classify."""

    feature_dim = 512

    def __init__(self, config: TSNConfig | None = None) -> None:
        super().__init__()
        self.config = config or TSNConfig()
        weights = ResNet18_Weights.IMAGENET1K_V1 if self.config.pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(self.config.dropout),
            nn.Linear(self.feature_dim, self.config.num_classes),
        )

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"Expected clip shape (B,T,C,H,W), got {tuple(clip.shape)}")
        batch, time, channels, height, width = clip.shape
        if channels != 3:
            raise ValueError(f"Expected 3 RGB channels, got {channels}")
        frame_features = self.backbone(clip.reshape(batch * time, channels, height, width))
        return frame_features.reshape(batch, time, self.feature_dim).mean(dim=1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(clip))

    def checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "kind": "ipn_tsn_resnet18_v1",
            "model_config": asdict(self.config),
            "state_dict": self.state_dict(),
            **extra,
        }
