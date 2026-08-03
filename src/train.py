"""Train an audited pretrained gesture architecture on IPN Hand."""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.audit import EXPECTED_LABELS
from src.data.dataset import VideoGestureDataset, collate_skip_bad, load_manifest
from src.metrics import classification_metrics
from src.models import build_model, model_config, trainable_parameter_count
from src.utils import (
    cuda_environment,
    cuda_memory_snapshot,
    ensure_output_directories,
    load_config,
    resolve_from_project,
    seed_everything,
    write_json,
)


def class_weights(records: list[Any], power: float) -> tuple[list[float], dict[str, int]]:
    counts = Counter(record.label for record in records)
    missing = set(EXPECTED_LABELS) - set(counts)
    if missing:
        raise ValueError(f"Training split lacks classes: {sorted(missing)}")
    total = len(records)
    raw = [(total / (len(EXPECTED_LABELS) * counts[label])) ** power for label in EXPECTED_LABELS]
    scale = len(raw) / sum(raw)
    return [value * scale for value in raw], dict(counts)


def _loader(
    dataset: VideoGestureDataset,
    *,
    batch_size: int,
    shuffle: bool,
    config: dict[str, Any],
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=bool(config["train"]["pin_memory"]),
        collate_fn=collate_skip_bad,
        drop_last=False,
    )


def _datasets(config: dict[str, Any]) -> tuple[VideoGestureDataset, VideoGestureDataset]:
    manifest = resolve_from_project(config["data"]["manifest"])
    root = resolve_from_project(config["data"]["root"])
    common = {
        "data_root": root,
        "clip_len": int(config["data"]["clip_len"]),
        "frame_size": int(config["data"]["frame_size"]),
        "cache_dir": resolve_from_project(config["data"]["cache_dir"]),
        "cache_resize_size": int(config["data"]["cache_resize_size"]),
        "decode_retries": int(config["data"]["decode_retries"]),
        "max_decode_error_rate": float(config["data"]["max_decode_error_rate"]),
    }
    return (
        VideoGestureDataset(load_manifest(manifest, split="train"), training=True, **common),
        VideoGestureDataset(load_manifest(manifest, split="val"), training=False, **common),
    )


def warmup_cosine_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1 / warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


def _preflight_batch_size(config: dict[str, Any], records: list[Any]) -> tuple[int, dict[str, Any]]:
    requested = int(config["train"]["batch_size"])
    minimum = int(config["train"]["min_batch_size"])
    safety_ratio = float(config["train"]["vram_safety_ratio"])
    root = resolve_from_project(config["data"]["root"])
    result: dict[str, Any] = {}
    effective = int(config["train"].get("effective_batch_size", requested))
    candidates = [
        value
        for value in range(requested, minimum - 1, -1)
        if effective % value == 0
    ]
    if not candidates:
        raise ValueError(
            f"No batch size in [{minimum}, {requested}] evenly divides effective batch {effective}"
        )
    for batch_size in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        probe_dataset = VideoGestureDataset(
            records[:batch_size],
            data_root=root,
            clip_len=int(config["data"]["clip_len"]),
            frame_size=int(config["data"]["frame_size"]),
            cache_dir=resolve_from_project(config["data"]["cache_dir"]),
            cache_resize_size=int(config["data"]["cache_resize_size"]),
            training=True,
            decode_retries=int(config["data"]["decode_retries"]),
            max_decode_error_rate=float(config["data"]["max_decode_error_rate"]),
        )
        probe = collate_skip_bad([probe_dataset[index] for index in range(len(probe_dataset))])
        probe_dataset.assert_decode_health()
        if probe is None:
            raise RuntimeError("Batch preflight could not decode any sample")
        clips, labels = probe
        try:
            model = build_model(model_config(config["model"])).cuda().train()
            clips, labels = clips.cuda(), labels.cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config["train"]["amp"])):
                logits = model(clips)
                loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            torch.cuda.synchronize()
            memory = cuda_memory_snapshot()
            result = {
                "requested_batch_size": requested,
                "selected_batch_size": batch_size,
                "peak_vram": memory,
                "within_safety_ratio": memory["peak_reserved_ratio"] <= safety_ratio,
            }
            del model, clips, labels, logits, loss
            torch.cuda.empty_cache()
            if result["within_safety_ratio"]:
                return batch_size, result
        except torch.OutOfMemoryError as error:
            result = {
                "requested_batch_size": requested,
                "selected_batch_size": batch_size,
                "oom": True,
                "error": str(error),
            }
            torch.cuda.empty_cache()
        if not bool(config["train"]["auto_reduce_batch_on_oom"]):
            break
    raise RuntimeError(f"No safe CUDA batch size found: {result}")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    dataset: VideoGestureDataset,
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    count = 0
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
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
            count += labels.numel()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    dataset.assert_decode_health()
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / max(count, 1)
    metrics["decoded_samples"] = count
    metrics["decode_failures"] = dataset.decode_failures
    return metrics


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    best_macro_f1: float,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    run_name: str,
    batch_size: int,
) -> dict[str, Any]:
    return model.checkpoint_payload(  # type: ignore[attr-defined]
        epoch=epoch,
        best_macro_f1=best_macro_f1,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        history=history,
        config=config,
        labels=list(EXPECTED_LABELS),
        run_name=run_name,
        batch_size=batch_size,
        smoke=False,
    )


def train(config_path: Path, *, resume: Path | None = None, run_name: str | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_output_directories(config)
    seed_everything(int(config["train"]["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Full configured training requires CUDA")
    device = torch.device("cuda")
    train_dataset, validation_dataset = _datasets(config)
    # Test records are deliberately not loaded here.
    selected_batch, batch_preflight = _preflight_batch_size(config, train_dataset.records)
    print(f"BATCH_PREFLIGHT {json.dumps(batch_preflight)}", flush=True)
    if bool(config["train"].get("auto_gradient_accumulation", False)):
        effective_batch = int(config["train"]["effective_batch_size"])
        if effective_batch % selected_batch:
            raise RuntimeError(
                f"Selected batch {selected_batch} does not divide effective batch {effective_batch}"
            )
        accumulation = effective_batch // selected_batch
    else:
        accumulation = int(config["train"]["gradient_accumulation_steps"])
    train_loader = _loader(train_dataset, batch_size=selected_batch, shuffle=True, config=config)
    validation_loader = _loader(validation_dataset, batch_size=selected_batch, shuffle=False, config=config)

    model = build_model(model_config(config["model"])).to(device)
    weights, counts = class_weights(
        train_dataset.records, float(config["train"]["class_weight_power"])
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * int(config["train"]["epochs"])
    warmup_steps = updates_per_epoch * int(config["train"]["warmup_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_ratio=float(config["train"]["min_lr_ratio"]),
        ),
    )
    amp = bool(config["train"]["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch = 1
    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []
    if resume is not None:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload["scaler_state_dict"])
        start_epoch = int(payload["epoch"]) + 1
        best_macro_f1 = float(payload["best_macro_f1"])
        history = list(payload["history"])
        run_name = str(payload["run_name"])

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = resolve_from_project(config["paths"]["checkpoints"]) / run_name
    log_dir = resolve_from_project(config["paths"]["logs"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    writer = SummaryWriter(log_dir=str(log_dir))
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        count = 0
        targets: list[int] = []
        predictions: list[int] = []
        micro_batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            clips, labels = batch
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    logits = model(clips)
                    raw_loss = criterion(logits, labels)
                    loss = raw_loss / accumulation
                scaler.scale(loss).backward()
            except torch.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA OOM despite preflight at batch_size={selected_batch}; "
                    "resume from last.pt after lowering train.batch_size"
                ) from error
            micro_batches += 1
            if micro_batches % accumulation == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += float(raw_loss.detach()) * labels.numel()
            count += labels.numel()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        if micro_batches % accumulation:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        train_dataset.assert_decode_health()
        train_metrics = classification_metrics(targets, predictions)
        train_metrics["loss"] = total_loss / max(count, 1)
        train_metrics["decoded_samples"] = count
        train_metrics["decode_failures"] = train_dataset.decode_failures
        validation_metrics = _evaluate(
            model, validation_loader, validation_dataset, device, amp=amp
        )
        vram = cuda_memory_snapshot()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": validation_metrics,
            "vram": vram if epoch == 1 else None,
            "epoch_duration_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        for split, metrics in (("train", train_metrics), ("val", validation_metrics)):
            writer.add_scalar(f"{split}/loss", metrics["loss"], epoch)
            writer.add_scalar(f"{split}/accuracy", metrics["accuracy"], epoch)
            writer.add_scalar(f"{split}/macro_f1", metrics["macro_f1"], epoch)
            writer.add_scalar(f"{split}/weighted_f1", metrics["weighted_f1"], epoch)
        writer.add_scalar("train/learning_rate", row["learning_rate"], epoch)
        if epoch == 1 and bool(config["train"]["log_vram_first_epoch"]):
            print(f"VRAM_FIRST_EPOCH {json.dumps(vram)}", flush=True)
        print(
            f"EPOCH epoch={epoch}/{config['train']['epochs']} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_accuracy={train_metrics['accuracy']:.4f} "
            f"train_macro_f1={train_metrics['macro_f1']:.4f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.4f} "
            f"seconds={row['epoch_duration_seconds']:.1f}",
            flush=True,
        )
        improved = validation_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = validation_metrics["macro_f1"]
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            best_macro_f1=best_macro_f1,
            history=history,
            config=config,
            run_name=run_name,
            batch_size=selected_batch,
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        writer.flush()

    writer.close()
    report = {
        "status": "completed",
        "run_name": run_name,
        "environment": cuda_environment(),
        "batch_preflight": batch_preflight,
        "selected_batch_size": selected_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": selected_batch * accumulation,
        "trainable_parameters": trainable_parameter_count(model),
        "class_counts": counts,
        "class_weights": dict(zip(EXPECTED_LABELS, weights, strict=True)),
        "epochs_completed": history[-1]["epoch"],
        "best_val_macro_f1": best_macro_f1,
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "config_copy": str((run_dir / "config.yaml").resolve()),
        "tensorboard_log_dir": str(log_dir.resolve()),
        "history": history,
        "test_split_opened": False,
    }
    model_name = str(config["model"]["name"])
    report_name = "training_report.json" if model_name == "tsn_resnet18" else f"training_{model_name}.json"
    report["mean_epoch_seconds"] = sum(
        float(row.get("epoch_duration_seconds", 0.0)) for row in history
    ) / max(len(history), 1)
    measured_durations = [
        float(row["epoch_duration_seconds"])
        for row in history
        if row.get("epoch_duration_seconds") is not None
    ]
    report["median_epoch_seconds"] = (
        statistics.median(measured_durations) if measured_durations else None
    )
    report["checkpoint_bytes"] = best_path.stat().st_size
    report_path = resolve_from_project(config["paths"]["reports"]) / report_name
    write_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    train(args.config, resume=args.resume, run_name=args.run_name)


if __name__ == "__main__":
    main()
