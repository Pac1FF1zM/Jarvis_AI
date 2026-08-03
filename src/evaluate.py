"""Evaluate the selected ResNet18-TSN checkpoint once on the official test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.audit import EXPECTED_LABELS
from src.data.dataset import VideoGestureDataset, collate_skip_bad, load_manifest
from src.metrics import classification_metrics
from src.models import build_model, model_config, trainable_parameter_count
from src.utils import load_config, resolve_from_project, seed_everything, write_json


def load_selected_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("kind") not in {"ipn_tsn_resnet18_v1", "ipn_gesture_architecture_v1"}:
        raise ValueError(f"Unsupported checkpoint kind: {payload.get('kind')!r}")
    labels = payload.get("labels")
    if labels != list(EXPECTED_LABELS):
        raise ValueError("Checkpoint label order does not match the audited dataset")
    checkpoint_config = model_config(payload["model_config"], load_pretrained=False)
    model = build_model(checkpoint_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def save_confusion_matrix(matrix: list[list[int]], output: Path) -> None:
    values = np.asarray(matrix, dtype=np.int64)
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title="IPN Hand — official test confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(EXPECTED_LABELS)),
        yticks=np.arange(len(EXPECTED_LABELS)),
        xticklabels=EXPECTED_LABELS,
        yticklabels=EXPECTED_LABELS,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[row, column] > threshold else "black",
            )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate(config_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    seed_everything(int(config["train"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_selected_model(checkpoint_path.resolve(), device)
    records = load_manifest(resolve_from_project(config["data"]["manifest"]), split="test")
    dataset = VideoGestureDataset(
        records,
        data_root=resolve_from_project(config["data"]["root"]),
        clip_len=int(config["data"]["clip_len"]),
        frame_size=int(config["data"]["frame_size"]),
        cache_dir=resolve_from_project(config["data"]["cache_dir"]),
        cache_resize_size=int(config["data"]["cache_resize_size"]),
        training=False,
        decode_retries=int(config["data"]["decode_retries"]),
        max_decode_error_rate=float(config["data"]["max_decode_error_rate"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(payload.get("batch_size", config["train"]["batch_size"])),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=bool(config["train"]["pin_memory"]),
        collate_fn=collate_skip_bad,
    )
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    decoded = 0
    amp = bool(config["train"]["amp"]) and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            if batch is None:
                continue
            clips, labels = batch
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(clips)
                loss = nn.functional.cross_entropy(logits, labels)
            total_loss += float(loss) * labels.numel()
            decoded += labels.numel()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    dataset.assert_decode_health()
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / max(decoded, 1)
    reports = resolve_from_project(config["paths"]["reports"])
    model_name = str(payload["model_config"]["name"])
    confusion_name = (
        "confusion_matrix_test.png"
        if model_name == "tsn_resnet18"
        else f"confusion_matrix_test_{model_name}.png"
    )
    confusion_path = reports / confusion_name
    save_confusion_matrix(metrics["confusion_matrix"], confusion_path)
    report = {
        "status": "completed",
        "protocol": "official_test_once_after_train_val_selection",
        "test_split_opened": True,
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_epoch": int(payload["epoch"]),
        "selection_metric": "val_macro_f1",
        "selected_val_macro_f1": float(payload["best_macro_f1"]),
        "device": str(device),
        "model_name": model_name,
        "trainable_parameters": trainable_parameter_count(model),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "test_instances_expected": len(records),
        "test_instances_decoded": decoded,
        "decode_failures": dataset.decode_failures,
        "decode_error_rate": dataset.decode_error_rate,
        "metrics": metrics,
        "confusion_matrix_path": str(confusion_path.resolve()),
    }
    output_name = (
        "evaluation_test.json"
        if model_name == "tsn_resnet18"
        else f"evaluation_test_{model_name}.json"
    )
    output = reports / output_name
    write_json(output, report)
    print(json.dumps({**report, "metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"}}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tsn_resnet18_seed42/best.pt"),
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
