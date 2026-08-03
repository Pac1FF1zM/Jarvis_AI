from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.dataset import (
    ClipRecord,
    VideoGestureDataset,
    collate_skip_bad,
    uniform_frame_indices,
)
from src.data.transforms import ClipTransform, ClipTransformConfig
from src.models.tsn import TSNConfig, TSNResNet18
from src.metrics import classification_metrics
from src.train import warmup_cosine_lambda


def test_uniform_sampler_covers_entire_inclusive_segment():
    indices = uniform_frame_indices(11, 210, 16)
    assert len(indices) == 16
    assert indices[0] == 11
    assert indices[-1] == 210
    assert all(left < right for left, right in zip(indices, indices[1:]))


def test_clip_transform_uses_identical_parameters_for_every_frame():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[:, :60, 0] = 255
    random.seed(42)
    transform = ClipTransform(
        ClipTransformConfig(frame_size=64, resize_size=72),
        training=True,
    )
    clip = transform([frame.copy() for _ in range(4)])
    assert clip.shape == (4, 3, 64, 64)
    assert torch.equal(clip[0], clip[1])
    assert torch.equal(clip[1], clip[2])
    assert torch.equal(clip[2], clip[3])


def test_dataset_decodes_rgb_clip_and_collates(tmp_path: Path):
    video = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(24):
        frame = np.full((48, 64, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    record = ClipRecord(
        video=video.name,
        video_id=video.stem,
        subject_id="subject_1",
        label="G01",
        class_id=3,
        start_frame=1,
        end_frame=24,
        split="train",
    )
    dataset = VideoGestureDataset(
        [record],
        data_root=tmp_path,
        clip_len=16,
        frame_size=64,
        training=False,
        cache_dir=tmp_path / "cache",
        cache_resize_size=72,
    )
    item = dataset[0]
    assert item is not None
    clip, label = item
    assert clip.shape == (16, 3, 64, 64)
    assert clip.dtype == torch.float32
    assert label == 3
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 1
    repeated = dataset[0]
    assert repeated is not None
    assert torch.equal(clip, repeated[0])
    batch = collate_skip_bad([item, None])
    assert batch is not None
    assert batch[0].shape == (1, 16, 3, 64, 64)
    assert batch[1].tolist() == [3]
    dataset.assert_decode_health()


def test_decode_guard_hard_fails_above_one_percent(tmp_path: Path):
    record = ClipRecord(
        video="missing.avi",
        video_id="missing",
        subject_id="subject_1",
        label="D0X",
        class_id=0,
        start_frame=1,
        end_frame=16,
        split="train",
    )
    dataset = VideoGestureDataset([record], data_root=tmp_path, max_decode_error_rate=0.01)
    assert dataset[0] is None
    with pytest.raises(RuntimeError, match="exceeds limit"):
        dataset.assert_decode_health()


def test_tsn_resnet18_smoke_shape_and_backward():
    model = TSNResNet18(TSNConfig(num_classes=14, pretrained=False, dropout=0.0))
    clip = torch.randn(2, 4, 3, 64, 64)
    logits = model(clip)
    assert logits.shape == (2, 14)
    logits.sum().backward()
    assert model.head[1].weight.grad is not None


def test_warmup_cosine_schedule_and_metrics_cover_all_classes():
    values = [
        warmup_cosine_lambda(
            step, total_steps=20, warmup_steps=4, min_lr_ratio=0.01
        )
        for step in range(20)
    ]
    assert values[0] == pytest.approx(0.25)
    assert values[3] == pytest.approx(1.0)
    assert values[-1] < values[4]
    metrics = classification_metrics([0, 1, 1], [0, 1, 0])
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert set(metrics["per_class"]) == {
        "D0X", "B0A", "B0B", "G01", "G02", "G03", "G04",
        "G05", "G06", "G07", "G08", "G09", "G10", "G11",
    }
