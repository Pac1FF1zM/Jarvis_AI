"""Clip-consistent RGB transforms for IPN Hand."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ClipTransformConfig:
    frame_size: int = 112
    resize_size: int = 128
    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.20
    hue: float = 0.05

    def __post_init__(self) -> None:
        if self.frame_size < 32:
            raise ValueError("frame_size must be at least 32")
        if self.resize_size < self.frame_size:
            raise ValueError("resize_size must be >= frame_size")


class ClipTransform:
    """Apply one sampled crop and one color transform to every clip frame.

    Horizontal flip is intentionally absent because IPN contains directional
    gesture labels whose meaning would change under reflection.
    """

    def __init__(self, config: ClipTransformConfig, *, training: bool) -> None:
        self.config = config
        self.training = training
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    def _color_jitter(self, clip: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        operations = [
            (F.adjust_brightness, random.uniform(max(0.0, 1 - cfg.brightness), 1 + cfg.brightness)),
            (F.adjust_contrast, random.uniform(max(0.0, 1 - cfg.contrast), 1 + cfg.contrast)),
            (F.adjust_saturation, random.uniform(max(0.0, 1 - cfg.saturation), 1 + cfg.saturation)),
            (F.adjust_hue, random.uniform(-cfg.hue, cfg.hue)),
        ]
        random.shuffle(operations)
        for operation, factor in operations:
            clip = operation(clip, factor)
        return clip

    def __call__(self, frames: list[np.ndarray]) -> torch.Tensor:
        if not frames:
            raise ValueError("Cannot transform an empty clip")
        if any(frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
            raise ValueError("Every frame must be an HxWx3 RGB array")
        clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
        if min(clip.shape[-2:]) != self.config.resize_size:
            clip = F.resize(
                clip,
                self.config.resize_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        height, width = clip.shape[-2:]
        crop = self.config.frame_size
        if self.training:
            top = random.randint(0, height - crop)
            left = random.randint(0, width - crop)
        else:
            top = (height - crop) // 2
            left = (width - crop) // 2
        clip = F.crop(clip, top, left, crop, crop)
        if self.training:
            clip = self._color_jitter(clip)
        return (clip - self.mean) / self.std


def resize_clip_for_cache(frames: list[np.ndarray], resize_size: int) -> np.ndarray:
    """Resize RGB frames once with the same kernel used by ClipTransform."""
    clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()
    clip = F.resize(
        clip,
        resize_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return clip.clamp(0, 255).round().byte().permute(0, 2, 3, 1).numpy()
