"""Small spatio-temporal neural networks initialized from random weights."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GestureModelConfig:
    architecture: str
    classes: int
    width: int = 32
    dropout: float = 0.20


class Conv3DBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, *, stride: tuple[int, int, int]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(incoming, outgoing, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outgoing),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class Tiny3DCNN(nn.Module):
    """A real 3D convolutional baseline, not a reused vision backbone."""

    def __init__(self, config: GestureModelConfig) -> None:
        super().__init__()
        w = config.width
        self.features = nn.Sequential(
            Conv3DBlock(3, w, stride=(1, 2, 2)),
            Conv3DBlock(w, w * 2, stride=(2, 2, 2)),
            Conv3DBlock(w * 2, w * 4, stride=(2, 2, 2)),
            Conv3DBlock(w * 4, w * 4, stride=(2, 2, 2)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(config.dropout), nn.Linear(w * 4, config.classes))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(clip))


class FrameEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_dim = width * 4

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        batch, channels, time, height, width = clip.shape
        features = self.features(clip.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width))
        return features.flatten(1).reshape(batch, time, -1)


class CNNBiGRU(nn.Module):
    def __init__(self, config: GestureModelConfig) -> None:
        super().__init__()
        self.encoder = FrameEncoder(config.width)
        hidden = self.encoder.output_dim // 2
        self.temporal = nn.GRU(self.encoder.output_dim, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(hidden * 2, config.classes))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.temporal(self.encoder(clip))
        return self.head(sequence.mean(dim=1))


class CNNTemporalTransformer(nn.Module):
    def __init__(self, config: GestureModelConfig) -> None:
        super().__init__()
        self.encoder = FrameEncoder(config.width)
        dimension = self.encoder.output_dim
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=4,
            dim_feedforward=dimension * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(dimension), nn.Dropout(config.dropout), nn.Linear(dimension, config.classes))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.head(self.temporal(self.encoder(clip)).mean(dim=1))


ARCHITECTURES = {
    "tiny_3d_cnn": Tiny3DCNN,
    "cnn_bigru": CNNBiGRU,
    "cnn_temporal_transformer": CNNTemporalTransformer,
}


def build_model(config: GestureModelConfig) -> nn.Module:
    try:
        return ARCHITECTURES[config.architecture](config)
    except KeyError as error:
        raise ValueError(f"Unsupported gesture architecture: {config.architecture}") from error


def checkpoint_payload(model: nn.Module, config: GestureModelConfig) -> dict[str, object]:
    return {
        "kind": "jarvis_gesture_from_scratch_v1",
        "model_config": asdict(config),
        "state_dict": model.state_dict(),
        "pretrained": False,
    }
