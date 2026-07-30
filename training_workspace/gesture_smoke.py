"""Create a disposable AVI corpus for an end-to-end gesture pipeline smoke run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.gesture.labels import IPN_LABELS


def create_smoke_config(
    output_root: Path,
    *,
    device: str = "cuda",
    epochs: int = 3,
) -> Path:
    """Generate small real AVI files and return a runnable training config."""
    if epochs < 1:
        raise ValueError("Smoke epochs must be at least 1")
    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Gesture smoke test requires opencv-python-headless in .venv-training"
        ) from error

    session = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    workspace = output_root.resolve() / "smoke" / session
    videos = workspace / "videos"
    videos.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []

    for label_index, label in enumerate(IPN_LABELS):
        for split in ("train", "validation", "test"):
            path = videos / f"{split}_{label}.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 64)
            )
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not create smoke AVI {path}")
            try:
                for frame_index in range(1, 17):
                    frame = np.zeros((64, 64, 3), dtype=np.uint8)
                    color = (
                        (label_index * 37) % 255,
                        (label_index * 71) % 255,
                        (label_index * 113) % 255,
                    )
                    frame[:] = tuple(channel // 4 for channel in color)
                    x = (frame_index * 3 + label_index * 2) % 48
                    y = (frame_index * 2 + label_index * 3) % 48
                    cv2.rectangle(frame, (x, y), (x + 15, y + 15), color, -1)
                    writer.write(frame)
            finally:
                writer.release()
            records.append(
                {
                    "video": str(path.resolve()),
                    "label": label,
                    "start_frame": 1,
                    "end_frame": 16,
                    "split": split,
                    "video_id": f"{split}_{label}",
                }
            )

    manifest = workspace / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    experiments = [
        ("tiny_3d_cnn", 17),
        ("cnn_bigru", 29),
        ("cnn_temporal_transformer", 43),
    ]
    config = {
        "require_cuda": device == "cuda",
        "device": device,
        "amp": device == "cuda",
        "runs_dir": str(workspace / "runs"),
        "export_dir": str(workspace / "export-disabled"),
        "data": {
            "manifest": str(manifest),
            "frames": 8,
            "image_size": 32,
            "decode_retries": 2,
        },
        "batch_size": len(IPN_LABELS),
        "num_workers": 0,
        "experiments": [
            {
                "name": name,
                "architecture": name,
                "width": 8,
                "dropout": 0.10,
                "epochs": epochs,
                "patience": epochs,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "label_smoothing": 0.0,
                "seed": seed,
            }
            for name, seed in experiments
        ],
        "selection": {
            "min_test_macro_f1": 0.0,
            "min_no_gesture_recall": 0.0,
            "max_false_trigger_rate": 1.0,
        },
    }
    config_path = workspace / "smoke_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path
