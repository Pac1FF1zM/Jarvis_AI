"""Fair training protocol shared by all JSC sequence baselines."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from tools.registry import ToolRegistry

from .baseline_metrics import evaluate_program_predictions
from .data import JSCExample, load_jsc_jsonl
from .jal import ToolSchemaRegistry
from .models import ARCHITECTURES, BaselineConfig, JSCBaselineModel
from .sequence_data import (
    ACT_LABELS,
    JSCSequenceDataset,
    SequenceLimits,
    make_collate_fn,
    tokenizer_training_texts,
)
from .tokenizer import JSCCharTokenizer


@dataclass(frozen=True)
class TrainingConfig:
    architecture: str
    data_dir: str
    output_dir: str
    seed: int = 17
    device: str = "auto"
    epochs: int = 24
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.05
    act_loss_weight: float = 0.25
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.15
    max_source_length: int = 384
    max_target_length: int = 256
    patience: int = 6
    gradient_clip: float = 1.0
    warmup_ratio: float = 0.08
    num_workers: int = 0
    use_amp: bool = True
    resume: str | None = None
    smoke: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if self.epochs < 1 or self.batch_size < 1 or self.patience < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are invalid")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.act_loss_weight < 0 or self.gradient_clip <= 0:
            raise ValueError("loss/gradient parameters are invalid")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


@dataclass(frozen=True)
class TrainingContext:
    registry: ToolSchemaRegistry
    train: tuple[JSCExample, ...]
    validation: tuple[JSCExample, ...]
    test: tuple[JSCExample, ...]
    tokenizer: JSCCharTokenizer
    limits: SequenceLimits
    data_fingerprint: str
    manifest: Mapping[str, Any]


def inspect_training(config: TrainingConfig) -> dict[str, Any]:
    context = _load_context(config)
    model_config = _model_config(config, context.tokenizer)
    device = _resolve_device(config.device)
    model = JSCBaselineModel(model_config).to(device)
    return {
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "requested_device": config.device,
            "resolved_device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "train": len(context.train),
            "validation": len(context.validation),
            "test": len(context.test),
            "holdout_loaded": False,
            "fingerprint": context.data_fingerprint,
            "acts": dict(sorted(Counter(e.target.act.value for e in context.train).items())),
        },
        "tokenizer": {
            "type": "jsc_char_v1",
            "vocabulary": context.tokenizer.size,
            "fingerprint": context.tokenizer.fingerprint,
        },
        "model": {
            "architecture": config.architecture,
            "parameters": model.parameter_count(),
            "config": model_config.to_dict(),
        },
        "protocol": {
            "selection": "minimum validation teacher-forced NLL",
            "test_used_for_selection": False,
            "evaluation_holdout_loaded": False,
            "sampling": "inverse-sqrt target-act weighted",
            "amp_dtype": _amp_dtype_name(device, config.use_amp),
            "deterministic_algorithms": True,
            "smoke": config.smoke,
        },
    }


def train_baseline(config: TrainingConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    context = _load_context(config)
    device = _resolve_device(config.device)
    model_config = _model_config(config, context.tokenizer)
    model = JSCBaselineModel(model_config).to(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    report_path = output / "report.json"
    if config.resume is None and any(path.exists() for path in (latest_path, best_path, report_path)):
        raise FileExistsError(
            f"run directory {output} already contains checkpoints; use --resume or a new path"
        )
    train_dataset = JSCSequenceDataset(context.train, context.tokenizer, context.limits)
    validation_dataset = JSCSequenceDataset(
        context.validation, context.tokenizer, context.limits
    )
    collate = make_collate_fn(context.tokenizer.pad_id)
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    if config.smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(config.epochs * steps_per_epoch, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _schedule_lambda(total_steps, warmup_steps),
    )
    amp_dtype = _amp_dtype(device, config.use_amp)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_dtype == torch.float16,
    )
    run_signature = _run_signature(config, context, model_config)
    start_epoch = 0
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if config.resume is not None:
        resumed = _load_resume(
            Path(config.resume),
            run_signature,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        start_epoch = resumed["epoch"] + 1
        best_loss = resumed["best_loss"]
        best_epoch = resumed["best_epoch"]
        stale_epochs = resumed["stale_epochs"]
        history = list(resumed["history"])
        _restore_rng(resumed["rng_state"])
    started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        train_loader = _train_loader(
            train_dataset,
            context.train,
            collate,
            config,
            epoch,
        )
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate,
        )
        validation_metrics = _teacher_forced_metrics(
            model,
            validation_loader,
            device,
            config,
        )
        epoch_report = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_report)
        improved = validation_metrics["token_nll"] < best_loss - 1e-6
        if improved:
            best_loss = validation_metrics["token_nll"]
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(
                best_path,
                _inference_checkpoint(
                    config,
                    context,
                    model_config,
                    model,
                    epoch,
                    validation_metrics,
                    run_signature,
                ),
            )
        else:
            stale_epochs += 1
        _atomic_torch_save(
            latest_path,
            {
                "format_version": 1,
                "kind": "jsc_baseline_training_state",
                "run_signature": run_signature,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "rng_state": _capture_rng(),
            },
        )
        print(
            f"epoch={epoch + 1}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_nll={validation_metrics['token_nll']:.4f} "
            f"val_token_acc={validation_metrics['token_accuracy']:.4f} "
            f"val_act_acc={validation_metrics['aux_act_accuracy']:.4f}",
            flush=True,
        )
        if stale_epochs >= config.patience:
            break
    if not best_path.is_file():
        raise RuntimeError("training produced no best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    validation_final = _final_metrics(
        model,
        context.validation,
        context,
        device,
        config,
    )
    test_final = _final_metrics(
        model,
        context.test,
        context,
        device,
        config,
    )
    report = {
        "format_version": 1,
        "architecture": config.architecture,
        "seed": config.seed,
        "smoke": config.smoke,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameters": model.parameter_count(),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "data_fingerprint": context.data_fingerprint,
        "tokenizer_fingerprint": context.tokenizer.fingerprint,
        "tool_schema_sha256": context.registry.schema_fingerprint,
        "selection": {
            "metric": "validation_teacher_forced_token_nll",
            "best": best_loss,
            "test_used": False,
            "evaluation_holdout_loaded": False,
        },
        "validation": validation_final,
        "test": test_final,
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    _atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _load_context(config: TrainingConfig) -> TrainingContext:
    data_dir = Path(config.data_dir)
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tools = ToolRegistry()
    tools.discover("tools")
    registry = ToolSchemaRegistry.from_tool_registry(tools)
    if manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("dataset tool schema fingerprint does not match current runtime")
    loaded: dict[str, tuple[JSCExample, ...]] = {}
    digest = hashlib.sha256()
    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}.jsonl"
        content = path.read_bytes()
        expected_hash = manifest["splits"][split]["sha256"]
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"{split} hash does not match dataset manifest")
        digest.update(split.encode("ascii") + b"\0" + content)
        loaded[split] = tuple(load_jsc_jsonl(path, registry, expected_split=split))
    digest.update(registry.schema_fingerprint.encode("ascii"))
    tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(loaded["train"]))
    limits = SequenceLimits(config.max_source_length, config.max_target_length)
    for split, examples in loaded.items():
        dataset = JSCSequenceDataset(examples, tokenizer, limits)
        for index in range(len(dataset)):
            dataset[index]
    return TrainingContext(
        registry=registry,
        train=loaded["train"],
        validation=loaded["validation"],
        test=loaded["test"],
        tokenizer=tokenizer,
        limits=limits,
        data_fingerprint=digest.hexdigest(),
        manifest=manifest,
    )


def _model_config(
    config: TrainingConfig, tokenizer: JSCCharTokenizer
) -> BaselineConfig:
    return BaselineConfig(
        architecture=config.architecture,
        vocab_size=tokenizer.size,
        num_acts=len(ACT_LABELS),
        d_model=config.d_model,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        attention_heads=config.attention_heads,
        feedforward_dim=config.feedforward_dim,
        dropout=config.dropout,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
        pad_id=tokenizer.pad_id,
    )


def _train_loader(
    dataset: JSCSequenceDataset,
    examples: tuple[JSCExample, ...],
    collate,
    config: TrainingConfig,
    epoch: int,
) -> DataLoader:
    counts = Counter(example.target.act.value for example in examples)
    weights = [math.sqrt(len(examples) / counts[e.target.act.value]) for e in examples]
    generator = torch.Generator().manual_seed(config.seed + epoch * 10_007)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=config.device != "cpu" and torch.cuda.is_available(),
    )


def _train_epoch(
    model: JSCBaselineModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    total_loss = total_tokens = correct_tokens = batches = optimizer_steps = skipped_steps = 0
    for batch_index, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        source_ids, source_mask, decoder_ids, decoder_mask, labels, acts = _move_batch(
            batch, device
        )
        with torch.autocast(
            device_type=device.type,
            dtype=_amp_dtype(device, config.use_amp),
            enabled=device.type == "cuda" and config.use_amp,
        ):
            logits, act_logits = model(
                source_ids, source_mask, decoder_ids, decoder_mask
            )
            token_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=config.label_smoothing,
            )
            act_loss = nn.functional.cross_entropy(
                act_logits,
                acts,
                label_smoothing=config.label_smoothing,
            )
            loss = token_loss + config.act_loss_weight * act_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            scheduler.step()
            optimizer_steps += 1
        else:
            skipped_steps += 1
        valid = labels.ne(-100)
        total_tokens += int(valid.sum())
        correct_tokens += int((logits.argmax(-1).eq(labels) & valid).sum())
        total_loss += float(loss.detach())
        batches += 1
        if config.smoke and batch_index >= 1:
            break
    return {
        "loss": total_loss / max(batches, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "batches": batches,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }


@torch.inference_mode()
def _teacher_forced_metrics(
    model: JSCBaselineModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    model.eval()
    nll_sum = correct_tokens = total_tokens = act_correct = act_total = batches = 0
    for batch_index, batch in enumerate(loader):
        source_ids, source_mask, decoder_ids, decoder_mask, labels, acts = _move_batch(
            batch, device
        )
        logits, act_logits = model(source_ids, source_mask, decoder_ids, decoder_mask)
        nll_sum += float(
            nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        )
        valid = labels.ne(-100)
        total_tokens += int(valid.sum())
        correct_tokens += int((logits.argmax(-1).eq(labels) & valid).sum())
        act_correct += int(act_logits.argmax(-1).eq(acts).sum())
        act_total += acts.numel()
        batches += 1
        if config.smoke and batch_index >= 0:
            break
    return {
        "token_nll": nll_sum / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "aux_act_accuracy": act_correct / max(act_total, 1),
        "batches": batches,
    }


def _final_metrics(
    model: JSCBaselineModel,
    examples: tuple[JSCExample, ...],
    context: TrainingContext,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, Any]:
    selected = examples[: min(len(examples), config.batch_size)] if config.smoke else examples
    dataset = JSCSequenceDataset(selected, context.tokenizer, context.limits)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=make_collate_fn(context.tokenizer.pad_id),
    )
    teacher = _teacher_forced_metrics(model, loader, device, config)
    predictions: list[str] = []
    ordered_examples: list[JSCExample] = []
    by_id = {example.scenario_id: example for example in selected}
    decode_limit = min(config.max_target_length, 64) if config.smoke else config.max_target_length
    with torch.inference_mode():
        for batch in loader:
            source_ids = batch["source_ids"].to(device)
            source_mask = batch["source_mask"].to(device)
            generated, _ = model.greedy_decode(
                source_ids,
                source_mask,
                bos_id=context.tokenizer.bos_id,
                eos_id=context.tokenizer.eos_id,
                max_length=decode_limit,
            )
            predictions.extend(
                context.tokenizer.decode(row.tolist()) for row in generated.cpu()
            )
            ordered_examples.extend(by_id[scenario_id] for scenario_id in batch["scenario_id"])
    return {
        "teacher_forced": teacher,
        "generation": evaluate_program_predictions(
            ordered_examples,
            predictions,
            context.registry,
        ),
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device):
    return (
        batch["source_ids"].to(device),
        batch["source_mask"].to(device),
        batch["decoder_input_ids"].to(device),
        batch["decoder_mask"].to(device),
        batch["labels"].to(device),
        batch["act"].to(device),
    )


def _schedule_lambda(total_steps: int, warmup_steps: int):
    def value(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))), 0.05)

    return value


def _run_signature(
    config: TrainingConfig,
    context: TrainingContext,
    model_config: BaselineConfig,
) -> str:
    ignored = {"output_dir", "device", "resume"}
    training = {key: value for key, value in asdict(config).items() if key not in ignored}
    payload = {
        "training": training,
        "model": model_config.to_dict(),
        "data": context.data_fingerprint,
        "tokenizer": context.tokenizer.fingerprint,
        "tools": context.registry.schema_fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inference_checkpoint(
    config: TrainingConfig,
    context: TrainingContext,
    model_config: BaselineConfig,
    model: JSCBaselineModel,
    epoch: int,
    validation_metrics: Mapping[str, float],
    run_signature: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "jsc_baseline_inference",
        "architecture": config.architecture,
        "model_config": model_config.to_dict(),
        "model_state": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "tokenizer": context.tokenizer.to_dict(),
        "tokenizer_fingerprint": context.tokenizer.fingerprint,
        "data_fingerprint": context.data_fingerprint,
        "tool_schema_sha256": context.registry.schema_fingerprint,
        "run_signature": run_signature,
        "seed": config.seed,
        "epoch": epoch,
        "smoke": config.smoke,
        "validation_teacher_forced": dict(validation_metrics),
    }


def _load_resume(
    path: Path,
    run_signature: str,
    model: JSCBaselineModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "jsc_baseline_training_state":
        raise ValueError("resume file is not a JSC training-state checkpoint")
    if checkpoint.get("run_signature") != run_signature:
        raise ValueError("resume checkpoint does not match data/model/training configuration")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _amp_dtype(device: torch.device, enabled: bool) -> torch.dtype:
    if enabled and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _amp_dtype_name(device: torch.device, enabled: bool) -> str | None:
    if not enabled or device.type != "cuda":
        return None
    return str(_amp_dtype(device, enabled)).removeprefix("torch.")


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
