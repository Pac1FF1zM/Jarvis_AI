"""Three laptop-sized Jester candidates initialized only from random weights."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small


MODEL_NAMES = ("tiny_3d_cnn", "cnn_bigru", "mobilenet_tsm_attention")


@dataclass(frozen=True)
class JesterModelConfig:
    name: str
    num_classes: int = 27
    dropout: float = 0.25

    def __post_init__(self) -> None:
        if self.name not in MODEL_NAMES:
            raise ValueError(f"unsupported Jester model {self.name!r}")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


class JesterModel(nn.Module):
    config: JesterModelConfig

    def checkpoint_payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "kind": "jarvis_jester_from_scratch_v1",
            "model_config": asdict(self.config),
            "pretrained": False,
            "state_dict": self.state_dict(),
            **extra,
        }


class Conv3DBlock(nn.Sequential):
    def __init__(self, incoming: int, outgoing: int, stride: tuple[int, int, int]) -> None:
        super().__init__(
            nn.Conv3d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outgoing),
            nn.GELU(),
        )


class Tiny3DCNN(JesterModel):
    def __init__(self, config: JesterModelConfig) -> None:
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            Conv3DBlock(3, 24, (1, 2, 2)),
            Conv3DBlock(24, 48, (2, 2, 2)),
            Conv3DBlock(48, 96, (2, 2, 2)),
            Conv3DBlock(96, 128, (2, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(config.dropout), nn.Linear(128, config.num_classes))

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        return self.features(clip.permute(0, 2, 1, 3, 4)).flatten(1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(clip))


class FrameEncoder(nn.Module):
    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, self.output_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.output_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clip.shape
        encoded = self.layers(clip.reshape(batch * time, channels, height, width)).flatten(1)
        return encoded.reshape(batch, time, self.output_dim)


class CNNBiGRU(JesterModel):
    def __init__(self, config: JesterModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = FrameEncoder()
        self.temporal = nn.GRU(FrameEncoder.output_dim, 128, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(256, config.num_classes))

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.temporal(self.encoder(clip))
        return sequence.mean(dim=1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(clip))


def temporal_shift(sequence: torch.Tensor, fold_div: int = 8) -> torch.Tensor:
    result = torch.zeros_like(sequence)
    fold = max(1, sequence.shape[-1] // fold_div)
    result[:, :-1, :fold] = sequence[:, 1:, :fold]
    result[:, 1:, fold : 2 * fold] = sequence[:, :-1, fold : 2 * fold]
    result[:, :, 2 * fold :] = sequence[:, :, 2 * fold :]
    return result


class MobileNetTSMAttention(JesterModel):
    feature_dim = 1024

    def __init__(self, config: JesterModelConfig) -> None:
        super().__init__()
        self.config = config
        backbone = mobilenet_v3_small(weights=None)
        backbone.classifier[-1] = nn.Identity()
        self.backbone = backbone
        self.temporal = nn.Conv1d(
            self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=self.feature_dim
        )
        self.norm = nn.LayerNorm(self.feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.feature_dim, config.num_classes))

    def forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = clip.shape
        frames = clip.reshape(batch * time, channels, height, width).contiguous(memory_format=torch.channels_last)
        sequence = self.backbone(frames).reshape(batch, time, self.feature_dim)
        shifted = temporal_shift(sequence)
        mixed = shifted + self.temporal(shifted.transpose(1, 2)).transpose(1, 2)
        mixed = self.norm(mixed)
        weights = torch.softmax(self.attention(mixed), dim=1)
        return (mixed * weights).sum(dim=1)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(clip))


def build_model(config: JesterModelConfig) -> JesterModel:
    if config.name == "tiny_3d_cnn":
        return Tiny3DCNN(config)
    if config.name == "cnn_bigru":
        return CNNBiGRU(config)
    return MobileNetTSMAttention(config)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
