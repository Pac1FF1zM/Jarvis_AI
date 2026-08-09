"""Benchmark Jester candidates or train the selected model without test leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
import yaml
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import seed_everything
from .dataset import JesterDataset, ManifestRecord, load_manifest
from .labels import JESTER_LABELS, NEGATIVE_LABELS
from .models import JesterModelConfig, build_model, parameter_count


def load_jester_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"data", "train", "models", "paths", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Jester config is missing sections: {sorted(missing)}")
    return config


def metrics(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    labels = list(range(len(JESTER_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    negative_indices = [JESTER_LABELS.index(label) for label in NEGATIVE_LABELS]
    negative_support = sum(int(support[index]) for index in negative_indices)
    negative_recall = (
        sum(float(recall[index]) * int(support[index]) for index in negative_indices)
        / max(negative_support, 1)
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1.mean()),
        "negative_recall": negative_recall,
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(JESTER_LABELS)
        },
    }


def balanced_subset(records: list[ManifestRecord], per_class: int, seed: int) -> list[ManifestRecord]:
    grouped: dict[int, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_id].append(record)
    rng = random.Random(seed)
    selected: list[ManifestRecord] = []
    for class_id in range(len(JESTER_LABELS)):
        rows = grouped[class_id]
        if not rows:
            raise ValueError(f"class {JESTER_LABELS[class_id]!r} is missing")
        rng.shuffle(rows)
        selected.extend(rows[: min(per_class, len(rows))])
    rng.shuffle(selected)
    return selected


def _dataset(config: dict[str, Any], records: list[ManifestRecord], training: bool) -> JesterDataset:
    data = config["data"]
    return JesterDataset(
        records,
        frames_root=Path(data["frames_root"]),
        clip_len=int(data["clip_len"]),
        frame_size=int(data["frame_size"]),
        resize_size=int(data["resize_size"]),
        training=training,
    )


def _loader(config: dict[str, Any], dataset: JesterDataset, batch_size: int, shuffle: bool) -> DataLoader:
    workers = int(config["train"]["num_workers"])
    options: dict[str, Any] = {}
    if workers > 0:
        # On Windows each spawned worker imports the CUDA-enabled torch DLLs.
        # Keeping train workers alive while validation workers start can exhaust
        # the system commit/page file, especially with 8+ workers.
        options["persistent_workers"] = bool(config["train"].get("persistent_workers", False))
        options["prefetch_factor"] = int(config["train"].get("prefetch_factor", 2))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["train"]["pin_memory"]),
        drop_last=False,
        worker_init_fn=_seed_worker,
        **options,
    )


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _records_fingerprint(records: list[ManifestRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.clip_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.class_id).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_safe_batch_size(
    config: dict[str, Any], model: nn.Module, dataset: JesterDataset, device: torch.device
) -> tuple[int, dict[str, float | int]]:
    requested = int(config["train"]["batch_size"])
    effective = int(config["train"]["effective_batch_size"])
    limit = float(config["train"]["max_vram_ratio"])
    candidates = [value for value in (requested, requested // 2, requested // 4, 1) if value > 0 and effective % value == 0]
    seen: set[int] = set()
    last: dict[str, float | int] = {}
    for batch_size in candidates:
        if batch_size in seen:
            continue
        seen.add(batch_size)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            clips, labels = next(iter(_loader(config, dataset, batch_size, False)))
            clips = clips.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config["train"]["amp"])):
                loss = nn.functional.cross_entropy(model(clips), labels)
            loss.backward()
            torch.cuda.synchronize()
            total = torch.cuda.get_device_properties(0).total_memory
            peak = torch.cuda.max_memory_reserved(0)
            last = {"batch_size": batch_size, "peak_reserved_bytes": peak, "total_bytes": total, "ratio": peak / total}
            model.zero_grad(set_to_none=True)
            del clips, labels, loss
            torch.cuda.empty_cache()
            if float(last["ratio"]) <= limit:
                return batch_size, last
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError(f"no safe Jester batch size found: {last}")


@torch.inference_mode()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> dict[str, Any]:
    model.eval()
    total_loss = torch.zeros((), device=device)
    count = 0
    targets: list[int] = []
    predictions: list[int] = []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(clips)
            loss = nn.functional.cross_entropy(logits, labels)
        total_loss += loss * labels.numel()
        count += labels.numel()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
    result = metrics(targets, predictions)
    result["loss"] = float(total_loss) / max(count, 1)
    result["samples"] = count
    return result


def fit(
    config: dict[str, Any],
    *,
    model_name: str,
    train_records: list[ManifestRecord],
    val_records: list[ManifestRecord],
    epochs: int,
    run_dir: Path,
    resume: bool = True,
    max_epochs_this_run: int | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Jester training requires the configured NVIDIA CUDA environment")
    train_cfg = config["train"]
    effective_batch = int(train_cfg["effective_batch_size"])
    amp = bool(train_cfg["amp"])
    train_dataset = _dataset(config, train_records, True)
    val_dataset = _dataset(config, val_records, False)
    model_config = JesterModelConfig(
        name=model_name,
        num_classes=len(JESTER_LABELS),
        dropout=float(config["models"]["dropout"]),
    )
    model = build_model(model_config).to(device)
    batch_size, vram_preflight = select_safe_batch_size(config, model, train_dataset, device)
    accumulation = effective_batch // batch_size
    train_loader = _loader(config, train_dataset, batch_size, True)
    val_loader = _loader(config, val_dataset, batch_size, False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = updates_per_epoch * int(train_cfg["warmup_epochs"])

    def lr_scale(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max(1 / warmup_updates, (step + 1) / warmup_updates)
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(train_cfg["label_smoothing"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    latest_path = run_dir / "latest.pt"
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    start_epoch = 1
    resumed_from_epoch = 0
    patience = int(train_cfg["patience"])
    train_fingerprint = _records_fingerprint(train_records)
    val_fingerprint = _records_fingerprint(val_records)
    if resume and latest_path.is_file():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state.get("kind") != "jarvis_jester_training_state_v1":
            raise ValueError(f"unsupported resume checkpoint: {latest_path}")
        if state.get("model_config") != model_config.__dict__:
            raise ValueError("resume checkpoint model configuration differs from the requested run")
        if state.get("training_config") != config:
            raise ValueError("resume checkpoint training configuration differs from the requested run")
        if state.get("train_fingerprint") != train_fingerprint or state.get("val_fingerprint") != val_fingerprint:
            raise ValueError("resume checkpoint dataset split differs from the requested run")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        best_score = float(state["best_score"])
        best_state = {name: value.detach().cpu().clone() for name, value in state["best_state"].items()}
        history = list(state["history"])
        stale = int(state["stale"])
        resumed_from_epoch = int(state["epoch"])
        start_epoch = resumed_from_epoch + 1
        _restore_rng_state(state["rng_state"])
        print(f"JESTER_RESUME model={model_name} epoch={resumed_from_epoch}", flush=True)
    started = time.perf_counter()
    end_epoch = epochs
    if max_epochs_this_run is not None:
        if max_epochs_this_run < 1:
            raise ValueError("max_epochs_this_run must be positive")
        end_epoch = min(epochs, start_epoch + max_epochs_this_run - 1)
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = torch.zeros((), device=device)
        seen = 0
        for step, (clips, labels) in enumerate(train_loader, 1):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(clips)
                raw_loss = criterion(logits, labels)
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            epoch_loss += raw_loss.detach() * labels.numel()
            seen += labels.numel()
        validation = evaluate_loader(model, val_loader, device, amp)
        row = {
            "epoch": epoch,
            "train_loss": float(epoch_loss) / max(seen, 1),
            "validation": validation,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        writer.add_scalar("train/loss", row["train_loss"], epoch)
        writer.add_scalar("validation/macro_f1", validation["macro_f1"], epoch)
        writer.add_scalar("validation/negative_recall", validation["negative_recall"], epoch)
        print(
            f"JESTER_EPOCH model={model_name} epoch={epoch}/{epochs} "
            f"loss={row['train_loss']:.5f} val_macro_f1={validation['macro_f1']:.4f} "
            f"negative_recall={validation['negative_recall']:.4f}",
            flush=True,
        )
        score = float(validation["macro_f1"])
        should_stop = False
        if score > best_score + 1e-6:
            best_score = score
            best_state = _cpu_state_dict(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                should_stop = True
        _atomic_torch_save(
            {
                "kind": "jarvis_jester_training_state_v1",
                "epoch": epoch,
                "model_config": model_config.__dict__,
                "training_config": config,
                "train_fingerprint": train_fingerprint,
                "val_fingerprint": val_fingerprint,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_score": best_score,
                "best_state": best_state,
                "history": history,
                "stale": stale,
                "rng_state": _rng_state(),
            },
            latest_path,
        )
        if should_stop:
            break
    writer.close()
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_validation = evaluate_loader(model, val_loader, device, amp)
    checkpoint = model.checkpoint_payload(
        labels=list(JESTER_LABELS),
        best_val_macro_f1=best_score,
        best_validation=best_validation,
        history=history,
        training_config=config,
    )
    checkpoint_path = run_dir / "best.pt"
    _atomic_torch_save(checkpoint, checkpoint_path)
    report = {
        "model": model_name,
        "pretrained": False,
        "parameters": parameter_count(model),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "vram_preflight": vram_preflight,
        "epochs_completed": len(history),
        "resumed_from_epoch": resumed_from_epoch,
        "best_val_macro_f1": best_score,
        "validation": best_validation,
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path.resolve()),
        "test_split_opened": False,
    }
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def benchmark(config_path: Path, *, resume: bool = True) -> dict[str, Any]:
    config = load_jester_config(config_path)
    seed = int(config["train"]["seed"])
    seed_everything(seed)
    manifest = Path(config["data"]["manifest"])
    per_class = int(config["train"]["benchmark_samples_per_class"])
    train_records = balanced_subset(load_manifest(manifest, "train"), per_class, seed)
    val_records = balanced_subset(load_manifest(manifest, "val"), max(32, per_class // 4), seed + 1)
    root = Path(config["paths"]["runs"]) / "benchmark"
    reports = [
        fit(
            config,
            model_name=name,
            train_records=train_records,
            val_records=val_records,
            epochs=int(config["train"]["benchmark_epochs"]),
            run_dir=root / name,
            resume=resume,
        )
        for name in config["models"]["candidates"]
    ]
    ranking = sorted(reports, key=lambda row: (-float(row["best_val_macro_f1"]), float(row["seconds"])))
    result = {
        "status": "completed",
        "config_fingerprint": config_fingerprint(config),
        "protocol": "balanced_equal_budget_candidate_benchmark",
        "ranking": ranking,
        "recommended_winner": ranking[0]["model"],
        "test_split_opened": False,
    }
    output = Path(config["paths"]["reports"]) / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def train_winner(config_path: Path, model_name: str | None = None, *, resume: bool = True) -> dict[str, Any]:
    config = load_jester_config(config_path)
    seed_everything(int(config["train"]["seed"]))
    manifest = Path(config["data"]["manifest"])
    selected = model_name or config["models"].get("winner")
    if not selected:
        benchmark_report = Path(config["paths"]["reports"]) / "benchmark.json"
        if not benchmark_report.is_file():
            raise RuntimeError("run the candidate benchmark before full training")
        benchmark_payload = json.loads(benchmark_report.read_text(encoding="utf-8"))
        if benchmark_payload.get("config_fingerprint") != config_fingerprint(config):
            raise RuntimeError("candidate benchmark belongs to a different training configuration")
        selected = benchmark_payload["recommended_winner"]
    selected = str(selected)
    return fit(
        config,
        model_name=selected,
        train_records=load_manifest(manifest, "train"),
        val_records=load_manifest(manifest, "val"),
        epochs=int(config["train"]["epochs"]),
        run_dir=Path(config["paths"]["runs"]) / "full" / selected,
        resume=resume,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "train"))
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    parser.add_argument("--model", choices=("tiny_3d_cnn", "cnn_bigru", "mobilenet_tsm_attention"))
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing latest.pt checkpoint.")
    args = parser.parse_args()
    if args.stage == "benchmark":
        benchmark(args.config, resume=not args.fresh)
    else:
        print(json.dumps(train_winner(args.config, args.model, resume=not args.fresh), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
