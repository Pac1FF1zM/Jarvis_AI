"""Mandatory pre-training gates for every comparable IPN architecture."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from src.data.audit import EXPECTED_LABELS
from src.data.dataset import VideoGestureDataset, load_manifest, uniform_frame_indices
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model, model_config, trainable_parameter_count
from src.train import _preflight_batch_size
from src.utils import (
    cuda_environment,
    cuda_memory_snapshot,
    ensure_output_directories,
    load_config,
    resolve_from_project,
    seed_everything,
    write_json,
)


def _save_sample_grid(
    clip: torch.Tensor,
    indices: list[int],
    record: Any,
    output: Path,
) -> None:
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    rgb = ((clip.cpu() * std + mean).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()
    frame_size = rgb.shape[1]
    header = 34
    tile_header = 16
    canvas = Image.new("RGB", (frame_size * 4, header + (frame_size + tile_header) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), f"{record.video_id} | {record.label} | {record.start_frame}-{record.end_frame}", fill="black")
    for position, (frame, frame_index) in enumerate(zip(rgb, indices, strict=True)):
        row, column = divmod(position, 4)
        x = column * frame_size
        y = header + row * (frame_size + tile_header)
        canvas.paste(Image.fromarray(frame), (x, y + tile_header))
        draw.text((x + 3, y + 1), f"frame {frame_index}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _save_overfit_curve(history: list[dict[str, float | int]], output: Path) -> None:
    steps = [int(row["step"]) for row in history]
    losses = [float(row["loss"]) for row in history]
    accuracies = [float(row["accuracy"]) for row in history]
    figure, loss_axis = plt.subplots(figsize=(8, 5))
    accuracy_axis = loss_axis.twinx()
    loss_axis.plot(steps, losses, color="tab:red", label="loss")
    accuracy_axis.plot(steps, accuracies, color="tab:blue", label="accuracy")
    loss_axis.set_xlabel("Optimization step")
    loss_axis.set_ylabel("Cross-entropy loss", color="tab:red")
    accuracy_axis.set_ylabel("Training accuracy", color="tab:blue")
    accuracy_axis.set_ylim(0, 1.02)
    loss_axis.grid(alpha=0.25)
    figure.suptitle("Mandatory overfit gate: 14 fixed IPN Hand clips")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _fixed_class_balanced_subset(records: list[Any]) -> list[Any]:
    selected = []
    for label in EXPECTED_LABELS:
        selected.append(next(record for record in records if record.label == label))
    return selected


def run_gates(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_output_directories(config)
    seed = int(config["train"]["seed"])
    seed_everything(seed)
    data_root = resolve_from_project(config["data"]["root"])
    manifest_path = resolve_from_project(config["data"]["manifest"])
    report_dir = resolve_from_project(config["paths"]["reports"])
    checkpoint_dir = resolve_from_project(config["paths"]["checkpoints"])
    split_report = json.loads((manifest_path.parent / "split_report.json").read_text(encoding="utf-8"))
    if split_report["subject_intersections"] != {
        "train_val": [], "train_test": [], "val_test": []
    }:
        raise RuntimeError("Subject leakage gate failed")
    if not split_report["matches_expectation"]:
        raise RuntimeError("Manifest expectation gate failed")

    train_records = load_manifest(manifest_path, split="train")
    grid_record = next(record for record in train_records if record.label == "G08")
    grid_dataset = VideoGestureDataset(
        [grid_record],
        data_root=data_root,
        clip_len=int(config["data"]["clip_len"]),
        frame_size=int(config["data"]["frame_size"]),
        cache_dir=resolve_from_project(config["data"]["cache_dir"]),
        cache_resize_size=int(config["data"]["cache_resize_size"]),
        training=False,
        decode_retries=int(config["data"]["decode_retries"]),
        max_decode_error_rate=float(config["data"]["max_decode_error_rate"]),
    )
    grid_item = grid_dataset[0]
    if grid_item is None:
        raise RuntimeError("Sample-grid clip could not be decoded")
    grid_clip, _ = grid_item
    grid_dataset.assert_decode_health()
    grid_indices = uniform_frame_indices(
        grid_record.start_frame, grid_record.end_frame, int(config["data"]["clip_len"])
    )
    grid_path = report_dir / "sample_clip_grid.png"
    _save_sample_grid(grid_clip, grid_indices, grid_record, grid_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured gates")
    device = torch.device("cuda")
    architecture_config = model_config(config["model"])
    model_name = architecture_config.name
    selected_batch, batch_preflight = _preflight_batch_size(config, train_records)
    effective_batch = int(config["train"].get("effective_batch_size", selected_batch))
    accumulation = effective_batch // selected_batch
    smoke_model = build_model(architecture_config).to(device).eval()
    smoke_batch = min(2, selected_batch)
    smoke_input = torch.randn(
        smoke_batch,
        int(config["data"]["clip_len"]),
        3,
        int(config["data"]["frame_size"]),
        int(config["data"]["frame_size"]),
        device=device,
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        smoke_output = smoke_model(smoke_input)
    if tuple(smoke_output.shape) != (smoke_batch, int(config["model"]["num_classes"])):
        raise RuntimeError(f"{model_name} smoke shape gate failed: {tuple(smoke_output.shape)}")
    del smoke_model, smoke_input, smoke_output
    torch.cuda.empty_cache()

    overfit_records = _fixed_class_balanced_subset(train_records)
    overfit_dataset = VideoGestureDataset(
        overfit_records,
        data_root=data_root,
        clip_len=int(config["data"]["clip_len"]),
        frame_size=int(config["data"]["frame_size"]),
        cache_dir=resolve_from_project(config["data"]["cache_dir"]),
        cache_resize_size=int(config["data"]["cache_resize_size"]),
        training=False,
        decode_retries=int(config["data"]["decode_retries"]),
        max_decode_error_rate=float(config["data"]["max_decode_error_rate"]),
    )
    loaded = [overfit_dataset[index] for index in range(len(overfit_dataset))]
    if any(item is None for item in loaded):
        overfit_dataset.assert_decode_health()
        raise RuntimeError("Overfit subset contains an unreadable clip")
    overfit_dataset.assert_decode_health()
    clips = torch.stack([item[0] for item in loaded if item is not None])
    labels = torch.tensor([item[1] for item in loaded if item is not None])

    overfit_config = replace(architecture_config, dropout=0.0)
    model = build_model(overfit_config).to(device)
    learning_rate = float(config["overfit"]["lr"])
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": learning_rate * 0.1},
            {"params": model.head.parameters(), "lr": learning_rate},
        ],
        weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]))
    target_accuracy = float(config["overfit"]["target_accuracy"])
    target_loss = float(config["overfit"]["target_loss"])
    max_steps = int(config["overfit"]["steps"])
    history: list[dict[str, float | int]] = []
    passed = False
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_logits: list[torch.Tensor] = []
        weighted_loss = torch.zeros((), device=device)
        for start in range(0, len(clips), selected_batch):
            clip_chunk = clips[start : start + selected_batch].to(device)
            label_chunk = labels[start : start + selected_batch].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config["train"]["amp"])):
                chunk_logits = model(clip_chunk)
                chunk_loss = torch.nn.functional.cross_entropy(chunk_logits, label_chunk)
                scaled_chunk_loss = chunk_loss * (label_chunk.numel() / len(labels))
            scaler.scale(scaled_chunk_loss).backward()
            weighted_loss = weighted_loss + scaled_chunk_loss.detach()
            step_logits.append(chunk_logits.detach().cpu())
        scaler.step(optimizer)
        scaler.update()
        logits = torch.cat(step_logits)
        accuracy = float((logits.argmax(dim=1) == labels).float().mean())
        row = {"step": step, "loss": float(weighted_loss), "accuracy": accuracy}
        history.append(row)
        if step == 1 or step % 10 == 0:
            print(
                f"OVERFIT step={step}/{max_steps} loss={row['loss']:.6f} "
                f"accuracy={accuracy:.4f}",
                flush=True,
            )
        if accuracy >= target_accuracy and row["loss"] <= target_loss:
            passed = True
            break

    curve_path = report_dir / f"overfit_curve_{model_name}.png"
    _save_overfit_curve(history, curve_path)
    checkpoint_path = checkpoint_dir / f"overfit_{model_name}.pt"
    torch.save(
        model.checkpoint_payload(
            gate="overfit",
            passed=passed,
            samples=[record.video_id for record in overfit_records],
            final=history[-1],
        ),
        checkpoint_path,
    )
    gate_report = {
        "passed": passed,
        "model_name": model_name,
        "trainable_parameters": trainable_parameter_count(model),
        "environment": cuda_environment(),
        "manifest_gate": {
            "instances": split_report["instance_count"],
            "classes": split_report["class_count"],
            "split_instance_counts": split_report["split_instance_counts"],
            "split_subject_counts": split_report["split_subject_counts"],
            "subject_intersections": split_report["subject_intersections"],
            "frame_bound_corrections": split_report["frame_bound_corrections"],
        },
        "sample_grid_gate": {
            "path": str(grid_path.resolve()),
            "video_id": grid_record.video_id,
            "label": grid_record.label,
            "sampled_indices": grid_indices,
            "clip_shape": list(grid_clip.shape),
        },
        "model_smoke_gate": {"input_batch": smoke_batch, "output_shape": [smoke_batch, 14]},
        "vram_preflight_gate": {
            **batch_preflight,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": selected_batch * accumulation,
        },
        "overfit_gate": {
            "samples": len(overfit_records),
            "one_per_class": [record.label for record in overfit_records],
            "steps_run": len(history),
            "target_accuracy": target_accuracy,
            "target_loss": target_loss,
            "final": history[-1],
            "curve": str(curve_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "decode_attempts": overfit_dataset.decode_attempts,
            "decode_failures": overfit_dataset.decode_failures,
        },
        "peak_vram": cuda_memory_snapshot(),
    }
    report_name = "gates.json" if model_name == "tsn_resnet18" else f"gates_{model_name}.json"
    write_json(report_dir / report_name, gate_report)
    print(json.dumps(gate_report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError(
            f"Mandatory overfit gate failed after {len(history)} steps: {history[-1]}"
        )
    return gate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    args = parser.parse_args()
    run_gates(args.config)


if __name__ == "__main__":
    main()
