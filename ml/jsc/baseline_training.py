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
from .constrained_decoding import constrain_jal_predictions
from .data import JSCExample, load_jsc_jsonl
from .jal import ToolSchemaRegistry
from .models import ARCHITECTURES, BaselineConfig, JSCBaselineModel
from .project_registry import build_project_schema_registry
from .structured_labels import build_parameter_labels
from .span_labels import SPAN_ARGUMENTS, span_tool_arguments
from .sequence_data import (
    ACT_LABELS,
    JSCSequenceDataset,
    SequenceLimits,
    make_collate_fn,
    serialize_source,
    tokenizer_training_texts,
    SOURCE_FORMAT_VERSION,
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
    max_target_length: int = 384
    patience: int = 6
    gradient_clip: float = 1.0
    warmup_ratio: float = 0.08
    num_workers: int = 0
    use_amp: bool = True
    copy_mechanism: bool = False
    structured_heads: bool = False
    parameter_heads: bool = False
    span_heads: bool = False
    semantic_pooling: bool = False
    execution_verifier: bool = False
    step_count_loss_weight: float = 0.35
    tool_sequence_loss_weight: float = 0.60
    parameter_loss_weight: float = 0.45
    span_loss_weight: float = 0.50
    execution_verifier_loss_weight: float = 0.75
    parameter_lr_multiplier: float = 3.0
    span_lr_multiplier: float = 3.0
    freeze_base_for_parameters: bool = False
    freeze_base_for_spans: bool = False
    freeze_base_for_semantics: bool = False
    init_checkpoint: str | None = None
    resume: str | None = None
    smoke: bool = False
    final_generation_metrics: bool = True

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
        if (
            self.step_count_loss_weight < 0
            or self.tool_sequence_loss_weight < 0
            or self.parameter_loss_weight < 0
            or self.span_loss_weight < 0
            or self.execution_verifier_loss_weight < 0
        ):
            raise ValueError("structured loss weights cannot be negative")
        if self.parameter_heads and not self.structured_heads:
            raise ValueError("parameter heads require structured heads")
        if self.span_heads and not (self.structured_heads and self.parameter_heads):
            raise ValueError("span heads require structured and parameter heads")
        if self.init_checkpoint is not None and self.resume is not None:
            raise ValueError("init_checkpoint and resume are mutually exclusive")
        if self.parameter_lr_multiplier <= 0:
            raise ValueError("parameter_lr_multiplier must be positive")
        if self.span_lr_multiplier <= 0:
            raise ValueError("span_lr_multiplier must be positive")
        if self.freeze_base_for_parameters and not (
            self.parameter_heads and self.init_checkpoint is not None
        ):
            raise ValueError(
                "freeze_base_for_parameters requires parameter heads and init checkpoint"
            )
        if self.freeze_base_for_spans and not (
            self.span_heads and self.init_checkpoint is not None
        ):
            raise ValueError("freeze_base_for_spans requires span heads and init checkpoint")
        if self.freeze_base_for_spans and self.freeze_base_for_parameters:
            raise ValueError("only one staged-freezing mode may be active")
        if self.freeze_base_for_semantics and not (
            self.execution_verifier
            and self.init_checkpoint is not None
        ):
            raise ValueError(
                "freeze_base_for_semantics requires verifier and init checkpoint"
            )
        if self.freeze_base_for_semantics and (
            self.freeze_base_for_spans or self.freeze_base_for_parameters
        ):
            raise ValueError("only one staged-freezing mode may be active")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")


@dataclass(frozen=True)
class TrainingContext:
    registry: ToolSchemaRegistry
    train: tuple[JSCExample, ...]
    validation: tuple[JSCExample, ...]
    test: tuple[JSCExample, ...] | None
    tokenizer: JSCCharTokenizer
    limits: SequenceLimits
    data_fingerprint: str
    manifest: Mapping[str, Any]


def inspect_training(config: TrainingConfig) -> dict[str, Any]:
    context = _load_context(config)
    model_config = _model_config(config, context.tokenizer, context.registry)
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
            "test": int(context.manifest["splits"]["test"]["examples"]),
            "test_loaded": context.test is not None,
            "evaluation_holdout_loaded": False,
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
            "sampling": "inverse-sqrt act/step-count/rarest-tool weighted",
            "categorical_parameters": len(build_parameter_labels(context.registry))
            if config.parameter_heads
            else 0,
            "span_slots": len(SPAN_ARGUMENTS) if config.span_heads else 0,
            "semantic_pooling": config.semantic_pooling,
            "execution_verifier": config.execution_verifier,
            "amp_dtype": _amp_dtype_name(device, config.use_amp),
            "deterministic_algorithms": True,
            "smoke": config.smoke,
        },
    }


def train_baseline(config: TrainingConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    context = _load_context(config)
    device = _resolve_device(config.device)
    model_config = _model_config(config, context.tokenizer, context.registry)
    model = JSCBaselineModel(model_config).to(device)
    initialization = None
    if config.init_checkpoint is not None:
        initialization = _load_initial_weights(
            Path(config.init_checkpoint), model, context, model_config, device
        )
    if config.freeze_base_for_parameters:
        assert model.parameter_head is not None
        for name, value in model.named_parameters():
            value.requires_grad_(name.startswith("parameter_head."))
    if config.freeze_base_for_spans:
        for name, value in model.named_parameters():
            value.requires_grad_(name.startswith("span_"))
    if config.freeze_base_for_semantics:
        trainable_prefixes = ("execution_verifier_head.",)
        if config.semantic_pooling:
            trainable_prefixes += (
                "semantic_",
                "act_head.",
                "step_count_head.",
                "tool_sequence_head.",
            )
        for name, value in model.named_parameters():
            value.requires_grad_(name.startswith(trainable_prefixes))
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
    tool_to_id = {name: index + 1 for index, name in enumerate(context.registry.tool_names)}
    parameter_labels = build_parameter_labels(context.registry)
    collate = make_collate_fn(
        context.tokenizer.pad_id,
        tool_to_id if config.structured_heads else None,
        {name: index for index, name in enumerate(parameter_labels)}
        if config.parameter_heads
        else None,
        SPAN_ARGUMENTS if config.span_heads else None,
        span_tool_arguments(context.registry) if config.span_heads else None,
    )
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    if config.smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    if config.freeze_base_for_semantics:
        optimizer_groups = [
            {
                "params": [value for value in model.parameters() if value.requires_grad],
                "lr": config.learning_rate,
            }
        ]
    elif config.freeze_base_for_spans:
        span_parameters = [
            value
            for name, value in model.named_parameters()
            if name.startswith("span_")
        ]
        optimizer_groups = [
            {
                "params": span_parameters,
                "lr": config.learning_rate * config.span_lr_multiplier,
            }
        ]
    elif config.freeze_base_for_parameters:
        assert model.parameter_head is not None
        optimizer_groups = [
            {
                "params": list(model.parameter_head.parameters()),
                "lr": config.learning_rate * config.parameter_lr_multiplier,
            }
        ]
    elif config.parameter_heads:
        assert model.parameter_head is not None
        parameter_ids = {id(value) for value in model.parameter_head.parameters()}
        optimizer_groups = [
            {
                "params": [
                    value for value in model.parameters() if id(value) not in parameter_ids
                ],
                "lr": config.learning_rate,
            },
            {
                "params": list(model.parameter_head.parameters()),
                "lr": config.learning_rate * config.parameter_lr_multiplier,
            },
        ]
    else:
        optimizer_groups = [{"params": list(model.parameters()), "lr": config.learning_rate}]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
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
        selection_nll = validation_metrics.get("selection_nll", validation_metrics["token_nll"])
        improved = selection_nll < best_loss - 1e-6
        if improved:
            best_loss = selection_nll
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
            f"val_act_acc={validation_metrics['aux_act_accuracy']:.4f}"
            + (
                f" val_count_acc={validation_metrics['step_count_accuracy']:.4f}"
                f" val_tool_seq_acc={validation_metrics['tool_sequence_head_accuracy']:.4f}"
                + (
                    f" val_param_acc={validation_metrics['parameter_head_accuracy']:.4f}"
                    if config.parameter_heads
                    else ""
                )
                + (
                    f" val_span_acc={validation_metrics['span_head_accuracy']:.4f}"
                    if config.span_heads
                    else ""
                )
                + (
                    " val_verify_fpr="
                    f"{validation_metrics['execution_verifier_false_positive_rate']:.4f}"
                    if config.execution_verifier
                    else ""
                )
                if config.structured_heads
                else ""
            ),
            flush=True,
        )
        if stale_epochs >= config.patience:
            break
    if not best_path.is_file():
        raise RuntimeError("training produced no best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    validation_final = (
        _final_metrics(
            model,
            context.validation,
            context,
            device,
            config,
        )
        if config.final_generation_metrics
        else {
            "teacher_forced": dict(best["validation_teacher_forced"]),
            "generation": None,
            "constrained_generation": None,
        }
    )
    report = {
        "format_version": 1,
        "architecture": config.architecture,
        "seed": config.seed,
        "smoke": config.smoke,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameters": model.parameter_count(),
        "initialization": initialization,
        "hyperparameters": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "label_smoothing": config.label_smoothing,
            "act_loss_weight": config.act_loss_weight,
            "d_model": config.d_model,
            "encoder_layers": config.encoder_layers,
            "decoder_layers": config.decoder_layers,
            "attention_heads": config.attention_heads,
            "feedforward_dim": config.feedforward_dim,
            "dropout": config.dropout,
            "copy_mechanism": config.copy_mechanism,
            "structured_heads": config.structured_heads,
            "parameter_heads": config.parameter_heads,
            "span_heads": config.span_heads,
            "semantic_pooling": config.semantic_pooling,
            "execution_verifier": config.execution_verifier,
            "step_count_loss_weight": config.step_count_loss_weight,
            "tool_sequence_loss_weight": config.tool_sequence_loss_weight,
            "parameter_loss_weight": config.parameter_loss_weight,
            "span_loss_weight": config.span_loss_weight,
            "execution_verifier_loss_weight": config.execution_verifier_loss_weight,
            "parameter_lr_multiplier": config.parameter_lr_multiplier,
            "span_lr_multiplier": config.span_lr_multiplier,
            "freeze_base_for_parameters": config.freeze_base_for_parameters,
            "freeze_base_for_spans": config.freeze_base_for_spans,
            "freeze_base_for_semantics": config.freeze_base_for_semantics,
            "final_generation_metrics": config.final_generation_metrics,
        },
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "data_fingerprint": context.data_fingerprint,
        "tokenizer_fingerprint": context.tokenizer.fingerprint,
        "tool_schema_sha256": context.registry.schema_fingerprint,
        "selection": {
            "metric": "validation_structured_selection_nll"
            if config.structured_heads
            else "validation_teacher_forced_token_nll",
            "best": best_loss,
            "test_used": False,
            "evaluation_holdout_loaded": False,
        },
        "validation": validation_final,
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    _atomic_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def evaluate_locked_test(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    *,
    device: str = "auto",
    batch_size: int = 32,
    allow_smoke: bool = False,
) -> dict[str, Any]:
    """Open test only for an already selected immutable checkpoint."""
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=resolved_device,
        weights_only=False,
    )
    if checkpoint.get("kind") != "jsc_baseline_inference":
        raise ValueError("checkpoint is not a JSC inference checkpoint")
    if checkpoint.get("smoke") and not allow_smoke:
        raise ValueError("smoke checkpoints cannot be evaluated as trained models")
    directory = Path(data_dir)
    manifest = json.loads(
        (directory / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    registry = build_project_schema_registry()
    if manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("dataset tool schema fingerprint does not match current runtime")
    if checkpoint.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("checkpoint tool schema fingerprint does not match current runtime")
    data_fingerprint = _manifest_data_fingerprint(manifest, registry)
    if checkpoint.get("data_fingerprint") != data_fingerprint:
        raise ValueError("checkpoint was trained against another dataset manifest")
    test_path = directory / "test.jsonl"
    test_content = test_path.read_bytes()
    if hashlib.sha256(test_content).hexdigest() != manifest["splits"]["test"]["sha256"]:
        raise ValueError("test hash does not match dataset manifest")
    test_examples = tuple(load_jsc_jsonl(test_path, registry, expected_split="test"))
    tokenizer = JSCCharTokenizer.from_dict(checkpoint["tokenizer"])
    if tokenizer.fingerprint != checkpoint.get("tokenizer_fingerprint"):
        raise ValueError("checkpoint tokenizer fingerprint is corrupt")
    model_config = BaselineConfig.from_dict(checkpoint["model_config"])
    model = JSCBaselineModel(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    expected_tool_labels = ("<none>", *registry.tool_names)
    if model_config.num_tools and tuple(checkpoint.get("tool_labels", ())) != expected_tool_labels:
        raise ValueError("checkpoint structured tool labels do not match current runtime")
    expected_parameter_labels = build_parameter_labels(registry)
    if model_config.num_parameter_labels and tuple(
        checkpoint.get("parameter_labels", ())
    ) != expected_parameter_labels:
        raise ValueError("checkpoint parameter labels do not match current runtime")
    if model_config.num_span_slots and tuple(checkpoint.get("span_slots", ())) != tuple(
        SPAN_ARGUMENTS
    ):
        raise ValueError("checkpoint span labels do not match runtime")
    evaluation_config = TrainingConfig(
        architecture=model_config.architecture,
        data_dir=str(directory),
        output_dir=".",
        device=device,
        batch_size=batch_size,
        d_model=model_config.d_model,
        encoder_layers=model_config.encoder_layers,
        decoder_layers=model_config.decoder_layers,
        attention_heads=model_config.attention_heads,
        feedforward_dim=model_config.feedforward_dim,
        dropout=model_config.dropout,
        max_source_length=model_config.max_source_length,
        max_target_length=model_config.max_target_length,
        copy_mechanism=model_config.copy_mechanism,
        structured_heads=bool(model_config.num_tools),
        parameter_heads=bool(model_config.num_parameter_labels),
        span_heads=bool(model_config.num_span_slots),
        semantic_pooling=model_config.semantic_pooling,
        execution_verifier=model_config.execution_verifier,
        smoke=bool(checkpoint.get("smoke")),
    )
    context = TrainingContext(
        registry=registry,
        train=(),
        validation=(),
        test=test_examples,
        tokenizer=tokenizer,
        limits=SequenceLimits(
            model_config.max_source_length,
            model_config.max_target_length,
        ),
        data_fingerprint=data_fingerprint,
        manifest=manifest,
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "architecture": model_config.architecture,
        "seed": checkpoint["seed"],
        "selected_before_test": True,
        "metrics": _final_metrics(
            model,
            test_examples,
            context,
            resolved_device,
            evaluation_config,
        ),
    }


def _load_context(config: TrainingConfig) -> TrainingContext:
    data_dir = Path(config.data_dir)
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = build_project_schema_registry()
    if manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("dataset tool schema fingerprint does not match current runtime")
    loaded: dict[str, tuple[JSCExample, ...]] = {}
    for split in ("train", "validation"):
        path = data_dir / f"{split}.jsonl"
        content = path.read_bytes()
        expected_hash = manifest["splits"][split]["sha256"]
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"{split} hash does not match dataset manifest")
        loaded[split] = tuple(load_jsc_jsonl(path, registry, expected_split=split))
    data_fingerprint = _manifest_data_fingerprint(manifest, registry)
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
        test=None,
        tokenizer=tokenizer,
        limits=limits,
        data_fingerprint=data_fingerprint,
        manifest=manifest,
    )


def _manifest_data_fingerprint(
    manifest: Mapping[str, Any],
    registry: ToolSchemaRegistry,
) -> str:
    payload = {
        "version": manifest.get("version"),
        "data_schema_version": manifest.get("data_schema_version"),
        "source_format_version": SOURCE_FORMAT_VERSION,
        "tool_schema_sha256": registry.schema_fingerprint,
        "splits": {
            split: manifest["splits"][split]["sha256"]
            for split in ("train", "validation", "test")
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_config(
    config: TrainingConfig,
    tokenizer: JSCCharTokenizer,
    registry: ToolSchemaRegistry,
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
        copy_mechanism=config.copy_mechanism,
        num_tools=(len(registry.tool_names) + 1) if config.structured_heads else 0,
        num_parameter_labels=(
            len(build_parameter_labels(registry)) if config.parameter_heads else 0
        ),
        num_span_slots=len(SPAN_ARGUMENTS) if config.span_heads else 0,
        semantic_pooling=config.semantic_pooling,
        execution_verifier=config.execution_verifier,
    )


def _train_loader(
    dataset: JSCSequenceDataset,
    examples: tuple[JSCExample, ...],
    collate,
    config: TrainingConfig,
    epoch: int,
) -> DataLoader:
    counts = Counter(example.target.act.value for example in examples)
    step_counts = Counter(len(example.target.steps) for example in examples)
    tool_counts = Counter(
        step.tool for example in examples for step in example.target.steps
    )
    weights = []
    for example in examples:
        components = [
            math.sqrt(len(examples) / counts[example.target.act.value]),
            math.sqrt(len(examples) / step_counts[len(example.target.steps)]),
        ]
        components.extend(
            math.sqrt(len(examples) / tool_counts[step.tool])
            for step in example.target.steps
        )
        weights.append(min(max(components), 12.0))
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
    count_correct = count_total = tool_sequence_correct = 0
    parameter_correct = parameter_total = 0
    span_correct = span_total = 0
    verifier_correct = verifier_total = verifier_false_positive = verifier_negative = 0
    verifier_true_positive = verifier_positive = 0
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
            if config.execution_verifier:
                (
                    logits,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                    span_start_logits,
                    span_end_logits,
                    verifier_logits,
                ) = model.forward_verified_semantic(
                    source_ids, source_mask, decoder_ids, decoder_mask
                )
            elif config.span_heads:
                (
                    logits,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                    span_start_logits,
                    span_end_logits,
                ) = model.forward_full_semantic(
                    source_ids, source_mask, decoder_ids, decoder_mask
                )
            elif config.parameter_heads:
                (
                    logits,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                ) = model.forward_schema_conditioned(
                    source_ids, source_mask, decoder_ids, decoder_mask
                )
            elif config.structured_heads:
                logits, act_logits, count_logits, tool_logits = model.forward_structured(
                    source_ids, source_mask, decoder_ids, decoder_mask
                )
            else:
                logits, act_logits = model(
                    source_ids, source_mask, decoder_ids, decoder_mask
                )
            token_loss = _token_loss(
                logits,
                labels,
                log_probabilities=model.token_scores_are_log_probabilities,
                label_smoothing=config.label_smoothing,
            )
            act_loss = nn.functional.cross_entropy(
                act_logits,
                acts,
                label_smoothing=config.label_smoothing,
            )
            loss = token_loss + config.act_loss_weight * act_loss
            if config.execution_verifier:
                execution_targets = batch["execution_allowed"].to(device)
                verifier_weights = torch.tensor(
                    (3.0, 1.0), device=device, dtype=verifier_logits.dtype
                )
                verifier_loss = nn.functional.cross_entropy(
                    verifier_logits, execution_targets, weight=verifier_weights
                )
                loss = (
                    loss
                    + config.execution_verifier_loss_weight * verifier_loss
                )
            if config.structured_heads:
                step_counts = batch["step_count"].to(device)
                tool_ids = batch["tool_ids"].to(device)
                count_loss = nn.functional.cross_entropy(count_logits, step_counts)
                tool_weights = torch.ones(
                    tool_logits.shape[-1], device=device, dtype=tool_logits.dtype
                )
                tool_weights[0] = 0.05
                tool_loss = nn.functional.cross_entropy(
                    tool_logits.flatten(0, 1),
                    tool_ids.flatten(),
                    weight=tool_weights,
                )
                loss = (
                    loss
                    + config.step_count_loss_weight * count_loss
                    + config.tool_sequence_loss_weight * tool_loss
                )
                if config.parameter_heads:
                    parameter_targets = batch["parameter_targets"].to(device)
                    parameter_mask = batch["parameter_mask"].to(device)
                    parameter_loss_values = nn.functional.binary_cross_entropy_with_logits(
                        parameter_logits,
                        parameter_targets,
                        reduction="none",
                    )
                    positive_weights = torch.where(
                        parameter_targets.bool(),
                        torch.full_like(parameter_targets, 6.0),
                        torch.ones_like(parameter_targets),
                    )
                    selected_parameter_loss = (
                        parameter_loss_values * positive_weights
                    ).masked_select(parameter_mask)
                    parameter_loss = selected_parameter_loss.sum() / max(
                        selected_parameter_loss.numel(), 1
                    )
                    loss = loss + config.parameter_loss_weight * parameter_loss
                if config.span_heads:
                    span_start_targets = batch["span_start_targets"].to(device)
                    span_end_targets = batch["span_end_targets"].to(device)
                    span_mask = batch["span_mask"].to(device)
                    if bool(span_mask.any()):
                        span_start_loss = nn.functional.cross_entropy(
                            span_start_logits[span_mask], span_start_targets[span_mask]
                        )
                        span_end_loss = nn.functional.cross_entropy(
                            span_end_logits[span_mask], span_end_targets[span_mask]
                        )
                        loss = loss + config.span_loss_weight * (
                            span_start_loss + span_end_loss
                        )
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
        if config.structured_heads:
            predicted_counts = count_logits.argmax(-1)
            predicted_tools = tool_logits.argmax(-1)
            count_correct += int(predicted_counts.eq(step_counts).sum())
            count_total += step_counts.numel()
            tool_sequence_correct += int(
                predicted_tools.eq(tool_ids).all(dim=1).sum()
            )
            if config.parameter_heads:
                predicted_parameter_values = parameter_logits.sigmoid().ge(0.5)
                target_parameter_values = parameter_targets.bool()
                rows_with_parameters = parameter_mask.any(dim=(1, 2))
                row_correct = (
                    predicted_parameter_values.eq(target_parameter_values)
                    | ~parameter_mask
                ).all(dim=(1, 2))
                parameter_correct += int(
                    (row_correct & rows_with_parameters).sum()
                )
                parameter_total += int(rows_with_parameters.sum())
            if config.span_heads:
                span_mask = batch["span_mask"].to(device)
                rows_with_spans = span_mask.any(dim=(1, 2))
                span_row_correct = (
                    (
                        span_start_logits.argmax(-1).eq(span_start_targets)
                        & span_end_logits.argmax(-1).eq(span_end_targets)
                    )
                    | ~span_mask
                ).all(dim=(1, 2))
                span_correct += int((span_row_correct & rows_with_spans).sum())
                span_total += int(rows_with_spans.sum())
        if config.execution_verifier:
            verifier_predictions = verifier_logits.argmax(-1)
            verifier_correct += int(verifier_predictions.eq(execution_targets).sum())
            verifier_total += execution_targets.numel()
            negative = execution_targets.eq(0)
            positive = execution_targets.eq(1)
            verifier_false_positive += int(
                (verifier_predictions.eq(1) & negative).sum()
            )
            verifier_negative += int(negative.sum())
            verifier_true_positive += int(
                (verifier_predictions.eq(1) & positive).sum()
            )
            verifier_positive += int(positive.sum())
        total_loss += float(loss.detach())
        batches += 1
        if config.smoke and batch_index >= 1:
            break
    result = {
        "loss": total_loss / max(batches, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "batches": batches,
        "optimizer_steps": optimizer_steps,
        "skipped_optimizer_steps": skipped_steps,
    }
    if config.structured_heads:
        result.update(
            {
                "step_count_accuracy": count_correct / max(count_total, 1),
                "tool_sequence_head_accuracy": tool_sequence_correct
                / max(count_total, 1),
            }
        )
        if config.parameter_heads:
            result["parameter_head_accuracy"] = parameter_correct / max(
                parameter_total, 1
            )
        if config.span_heads:
            result["span_head_accuracy"] = span_correct / max(span_total, 1)
    if config.execution_verifier:
        result.update(
            {
                "execution_verifier_accuracy": verifier_correct
                / max(verifier_total, 1),
                "execution_verifier_false_positive_rate": verifier_false_positive
                / max(verifier_negative, 1),
                "execution_verifier_recall": verifier_true_positive
                / max(verifier_positive, 1),
            }
        )
    return result


@torch.inference_mode()
def _teacher_forced_metrics(
    model: JSCBaselineModel,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    model.eval()
    nll_sum = correct_tokens = total_tokens = act_correct = act_total = batches = 0
    count_correct = count_total = tool_sequence_correct = 0
    parameter_correct = parameter_total = parameter_label_total = 0
    span_correct = span_total = span_label_total = 0
    count_nll_sum = tool_nll_sum = parameter_nll_sum = span_nll_sum = 0.0
    verifier_correct = verifier_total = verifier_false_positive = verifier_negative = 0
    verifier_true_positive = verifier_positive = 0
    verifier_nll_sum = 0.0
    for batch_index, batch in enumerate(loader):
        source_ids, source_mask, decoder_ids, decoder_mask, labels, acts = _move_batch(
            batch, device
        )
        if config.execution_verifier:
            (
                logits,
                act_logits,
                count_logits,
                tool_logits,
                parameter_logits,
                span_start_logits,
                span_end_logits,
                verifier_logits,
            ) = model.forward_verified_semantic(
                source_ids, source_mask, decoder_ids, decoder_mask
            )
        elif config.span_heads:
            (
                logits,
                act_logits,
                count_logits,
                tool_logits,
                parameter_logits,
                span_start_logits,
                span_end_logits,
            ) = model.forward_full_semantic(
                source_ids, source_mask, decoder_ids, decoder_mask
            )
        elif config.parameter_heads:
            (
                logits,
                act_logits,
                count_logits,
                tool_logits,
                parameter_logits,
            ) = model.forward_schema_conditioned(
                source_ids, source_mask, decoder_ids, decoder_mask
            )
        elif config.structured_heads:
            logits, act_logits, count_logits, tool_logits = model.forward_structured(
                source_ids, source_mask, decoder_ids, decoder_mask
            )
        else:
            logits, act_logits = model(source_ids, source_mask, decoder_ids, decoder_mask)
        nll_sum += float(
            _token_loss(
                logits,
                labels,
                log_probabilities=model.token_scores_are_log_probabilities,
                reduction="sum",
            )
        )
        valid = labels.ne(-100)
        total_tokens += int(valid.sum())
        correct_tokens += int((logits.argmax(-1).eq(labels) & valid).sum())
        act_correct += int(act_logits.argmax(-1).eq(acts).sum())
        act_total += acts.numel()
        if config.execution_verifier:
            execution_targets = batch["execution_allowed"].to(device)
            verifier_weights = torch.tensor(
                (3.0, 1.0), device=device, dtype=verifier_logits.dtype
            )
            verifier_nll_sum += float(
                nn.functional.cross_entropy(
                    verifier_logits,
                    execution_targets,
                    weight=verifier_weights,
                    reduction="sum",
                )
            )
            verifier_predictions = verifier_logits.argmax(-1)
            verifier_correct += int(verifier_predictions.eq(execution_targets).sum())
            verifier_total += execution_targets.numel()
            negative = execution_targets.eq(0)
            positive = execution_targets.eq(1)
            verifier_false_positive += int(
                (verifier_predictions.eq(1) & negative).sum()
            )
            verifier_negative += int(negative.sum())
            verifier_true_positive += int(
                (verifier_predictions.eq(1) & positive).sum()
            )
            verifier_positive += int(positive.sum())
        if config.structured_heads:
            step_counts = batch["step_count"].to(device)
            tool_ids = batch["tool_ids"].to(device)
            count_correct += int(count_logits.argmax(-1).eq(step_counts).sum())
            count_total += step_counts.numel()
            tool_sequence_correct += int(
                tool_logits.argmax(-1).eq(tool_ids).all(dim=1).sum()
            )
            count_nll_sum += float(
                nn.functional.cross_entropy(count_logits, step_counts, reduction="sum")
            )
            tool_nll_sum += float(
                nn.functional.cross_entropy(
                    tool_logits.flatten(0, 1), tool_ids.flatten(), reduction="sum"
                )
            )
            if config.parameter_heads:
                parameter_targets = batch["parameter_targets"].to(device)
                parameter_mask = batch["parameter_mask"].to(device)
                parameter_values = nn.functional.binary_cross_entropy_with_logits(
                    parameter_logits,
                    parameter_targets,
                    reduction="none",
                )
                parameter_weights = torch.where(
                    parameter_targets.bool(),
                    torch.full_like(parameter_targets, 6.0),
                    torch.ones_like(parameter_targets),
                )
                parameter_nll_sum += float(
                    (parameter_values * parameter_weights)
                    .masked_select(parameter_mask)
                    .sum()
                )
                parameter_label_total += int(parameter_mask.sum())
                predicted_parameter_values = parameter_logits.sigmoid().ge(0.5)
                target_parameter_values = parameter_targets.bool()
                rows_with_parameters = parameter_mask.any(dim=(1, 2))
                row_correct = (
                    predicted_parameter_values.eq(target_parameter_values)
                    | ~parameter_mask
                ).all(dim=(1, 2))
                parameter_correct += int(
                    (row_correct & rows_with_parameters).sum()
                )
                parameter_total += int(rows_with_parameters.sum())
            if config.span_heads:
                span_start_targets = batch["span_start_targets"].to(device)
                span_end_targets = batch["span_end_targets"].to(device)
                span_mask = batch["span_mask"].to(device)
                if bool(span_mask.any()):
                    span_start_nll = nn.functional.cross_entropy(
                        span_start_logits[span_mask],
                        span_start_targets[span_mask],
                        reduction="sum",
                    )
                    span_end_nll = nn.functional.cross_entropy(
                        span_end_logits[span_mask],
                        span_end_targets[span_mask],
                        reduction="sum",
                    )
                    span_nll_sum += float(span_start_nll + span_end_nll)
                    span_label_total += int(span_mask.sum()) * 2
                rows_with_spans = span_mask.any(dim=(1, 2))
                span_row_correct = (
                    (
                        span_start_logits.argmax(-1).eq(span_start_targets)
                        & span_end_logits.argmax(-1).eq(span_end_targets)
                    )
                    | ~span_mask
                ).all(dim=(1, 2))
                span_correct += int((span_row_correct & rows_with_spans).sum())
                span_total += int(rows_with_spans.sum())
        batches += 1
        if config.smoke and batch_index >= 0:
            break
    result = {
        "token_nll": nll_sum / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "aux_act_accuracy": act_correct / max(act_total, 1),
        "batches": batches,
    }
    if config.structured_heads:
        result.update(
            {
                "step_count_accuracy": count_correct / max(count_total, 1),
                "tool_sequence_head_accuracy": tool_sequence_correct
                / max(count_total, 1),
                "step_count_nll": count_nll_sum / max(count_total, 1),
                "tool_token_nll": tool_nll_sum
                / max(count_total * model.config.max_steps, 1),
            }
        )
        result["selection_nll"] = (
            result["token_nll"]
            + config.step_count_loss_weight * result["step_count_nll"]
            + config.tool_sequence_loss_weight * result["tool_token_nll"]
        )
        if config.parameter_heads:
            result["parameter_head_accuracy"] = parameter_correct / max(
                parameter_total, 1
            )
            result["parameter_token_nll"] = parameter_nll_sum / max(
                parameter_label_total, 1
            )
            result["selection_nll"] += (
                config.parameter_loss_weight * result["parameter_token_nll"]
            )
        if config.span_heads:
            result["span_head_accuracy"] = span_correct / max(span_total, 1)
            result["span_token_nll"] = span_nll_sum / max(span_label_total, 1)
            result["selection_nll"] += (
                config.span_loss_weight * result["span_token_nll"]
            )
    if config.execution_verifier:
        result.update(
            {
                "execution_verifier_accuracy": verifier_correct
                / max(verifier_total, 1),
                "execution_verifier_false_positive_rate": verifier_false_positive
                / max(verifier_negative, 1),
                "execution_verifier_recall": verifier_true_positive
                / max(verifier_positive, 1),
                "execution_verifier_nll": verifier_nll_sum
                / max(verifier_total, 1),
            }
        )
        result["selection_nll"] = result.get("selection_nll", result["token_nll"])
        result["selection_nll"] += (
            config.execution_verifier_loss_weight
            * result["execution_verifier_nll"]
        )
    return result


def _token_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    log_probabilities: bool,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    flattened_scores = scores.reshape(-1, scores.shape[-1])
    flattened_labels = labels.reshape(-1)
    if not log_probabilities:
        return nn.functional.cross_entropy(
            flattened_scores,
            flattened_labels,
            ignore_index=-100,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )
    valid = flattened_labels.ne(-100)
    if not bool(valid.any()):
        return flattened_scores.sum() * 0.0
    selected_scores = flattened_scores[valid]
    selected_labels = flattened_labels[valid]
    negative_log_likelihood = -selected_scores.gather(
        1, selected_labels.unsqueeze(1)
    ).squeeze(1)
    if label_smoothing:
        smooth_loss = -selected_scores.mean(dim=1)
        negative_log_likelihood = (
            (1.0 - label_smoothing) * negative_log_likelihood
            + label_smoothing * smooth_loss
        )
    if reduction == "sum":
        return negative_log_likelihood.sum()
    if reduction == "mean":
        return negative_log_likelihood.mean()
    raise ValueError(f"unsupported token loss reduction {reduction!r}")


def _final_metrics(
    model: JSCBaselineModel,
    examples: tuple[JSCExample, ...],
    context: TrainingContext,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, Any]:
    selected = examples[: min(len(examples), config.batch_size)] if config.smoke else examples
    dataset = JSCSequenceDataset(selected, context.tokenizer, context.limits)
    parameter_labels = build_parameter_labels(context.registry)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=make_collate_fn(
            context.tokenizer.pad_id,
            {name: index + 1 for index, name in enumerate(context.registry.tool_names)}
            if config.structured_heads
            else None,
            {name: index for index, name in enumerate(parameter_labels)}
            if config.parameter_heads
            else None,
            SPAN_ARGUMENTS if config.span_heads else None,
            span_tool_arguments(context.registry) if config.span_heads else None,
        ),
    )
    teacher = _teacher_forced_metrics(model, loader, device, config)
    predictions: list[str] = []
    act_outputs: list[torch.Tensor] = []
    count_outputs: list[torch.Tensor] = []
    tool_outputs: list[torch.Tensor] = []
    parameter_outputs: list[torch.Tensor] = []
    span_start_outputs: list[torch.Tensor] = []
    span_end_outputs: list[torch.Tensor] = []
    verifier_outputs: list[torch.Tensor] = []
    ordered_examples: list[JSCExample] = []
    by_id = {example.scenario_id: example for example in selected}
    decode_limit = min(config.max_target_length, 64) if config.smoke else config.max_target_length
    with torch.inference_mode():
        for batch in loader:
            source_ids = batch["source_ids"].to(device)
            source_mask = batch["source_mask"].to(device)
            if config.execution_verifier:
                (
                    generated,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                    span_start_logits,
                    span_end_logits,
                    verifier_logits,
                ) = model.greedy_decode_verified_semantic(
                    source_ids,
                    source_mask,
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=decode_limit,
                )
                count_outputs.append(count_logits.cpu())
                tool_outputs.append(tool_logits.cpu())
                parameter_outputs.append(parameter_logits.cpu())
                span_start_outputs.append(span_start_logits.cpu())
                span_end_outputs.append(span_end_logits.cpu())
                verifier_outputs.append(verifier_logits.cpu())
            elif config.span_heads:
                (
                    generated,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                    span_start_logits,
                    span_end_logits,
                ) = model.greedy_decode_full_semantic(
                    source_ids,
                    source_mask,
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=decode_limit,
                )
                count_outputs.append(count_logits.cpu())
                tool_outputs.append(tool_logits.cpu())
                parameter_outputs.append(parameter_logits.cpu())
                span_start_outputs.append(span_start_logits.cpu())
                span_end_outputs.append(span_end_logits.cpu())
            elif config.parameter_heads:
                (
                    generated,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                ) = model.greedy_decode_schema_conditioned(
                    source_ids,
                    source_mask,
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=decode_limit,
                )
                count_outputs.append(count_logits.cpu())
                tool_outputs.append(tool_logits.cpu())
                parameter_outputs.append(parameter_logits.cpu())
            elif config.structured_heads:
                generated, act_logits, count_logits, tool_logits = (
                    model.greedy_decode_structured(
                        source_ids,
                        source_mask,
                        bos_id=context.tokenizer.bos_id,
                        eos_id=context.tokenizer.eos_id,
                        max_length=decode_limit,
                    )
                )
                count_outputs.append(count_logits.cpu())
                tool_outputs.append(tool_logits.cpu())
            else:
                generated, act_logits = model.greedy_decode(
                    source_ids,
                    source_mask,
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=decode_limit,
                )
            predictions.extend(
                context.tokenizer.decode(row.tolist()) for row in generated.cpu()
            )
            act_outputs.append(act_logits.cpu())
            ordered_examples.extend(by_id[scenario_id] for scenario_id in batch["scenario_id"])
    constrained = constrain_jal_predictions(
        predictions,
        torch.cat(act_outputs, dim=0),
        context.registry,
        utterances=[example.text for example in ordered_examples],
        step_count_logits=torch.cat(count_outputs, dim=0) if count_outputs else None,
        tool_logits=torch.cat(tool_outputs, dim=0) if tool_outputs else None,
        tool_labels=("<none>", *context.registry.tool_names),
        parameter_logits=(
            torch.cat(parameter_outputs, dim=0) if parameter_outputs else None
        ),
        parameter_labels=parameter_labels if parameter_outputs else None,
        span_start_logits=(
            _concatenate_span_logits(span_start_outputs)
            if span_start_outputs
            else None
        ),
        span_end_logits=(
            _concatenate_span_logits(span_end_outputs)
            if span_end_outputs
            else None
        ),
        span_slots=SPAN_ARGUMENTS if span_start_outputs else None,
        span_sources=(
            [serialize_source(example) for example in ordered_examples]
            if span_start_outputs
            else None
        ),
        execution_verifier_logits=(
            torch.cat(verifier_outputs, dim=0) if verifier_outputs else None
        ),
    )
    return {
        "teacher_forced": teacher,
        "generation": evaluate_program_predictions(
            ordered_examples,
            predictions,
            context.registry,
        ),
        "constrained_generation": {
            **evaluate_program_predictions(
                ordered_examples,
                constrained.predictions,
                context.registry,
            ),
            "decoder_decisions": constrained.decisions,
        },
    }


def _concatenate_span_logits(values: list[torch.Tensor]) -> torch.Tensor:
    """Pad dynamic source axes with impossible logits before concatenation."""
    if not values:
        raise ValueError("span logits cannot be empty")
    maximum = max(value.shape[-1] for value in values)
    padded = [
        nn.functional.pad(
            value,
            (0, maximum - value.shape[-1]),
            value=torch.finfo(value.dtype).min,
        )
        for value in values
    ]
    return torch.cat(padded, dim=0)


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
    ignored = {"output_dir", "device", "resume", "final_generation_metrics"}
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
        "tool_labels": ("<none>", *context.registry.tool_names)
        if model_config.num_tools
        else (),
        "parameter_labels": build_parameter_labels(context.registry)
        if model_config.num_parameter_labels
        else (),
        "span_slots": SPAN_ARGUMENTS if model_config.num_span_slots else (),
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


def _load_initial_weights(
    path: Path,
    model: JSCBaselineModel,
    context: TrainingContext,
    model_config: BaselineConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Warm-start a schema-parameter model from a compatible JSC checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "jsc_baseline_inference":
        raise ValueError("initial checkpoint is not a JSC inference checkpoint")
    if checkpoint.get("data_fingerprint") != context.data_fingerprint:
        raise ValueError("initial checkpoint was trained against another dataset")
    if checkpoint.get("tool_schema_sha256") != context.registry.schema_fingerprint:
        raise ValueError("initial checkpoint tool schema does not match runtime")
    if checkpoint.get("tokenizer_fingerprint") != context.tokenizer.fingerprint:
        raise ValueError("initial checkpoint tokenizer does not match training data")
    source_config = BaselineConfig.from_dict(checkpoint["model_config"])
    source_values = source_config.to_dict()
    target_values = model_config.to_dict()
    source_parameter_labels = source_values["num_parameter_labels"]
    source_span_slots = source_values["num_span_slots"]
    source_semantic_pooling = source_values["semantic_pooling"]
    source_execution_verifier = source_values["execution_verifier"]
    source_values["num_parameter_labels"] = target_values["num_parameter_labels"]
    source_values["num_span_slots"] = target_values["num_span_slots"]
    source_values["semantic_pooling"] = target_values["semantic_pooling"]
    source_values["execution_verifier"] = target_values["execution_verifier"]
    if source_values != target_values:
        raise ValueError("initial checkpoint architecture is incompatible")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_prefixes: tuple[str, ...] = ()
    if not source_parameter_labels and target_values["num_parameter_labels"]:
        allowed_prefixes += ("parameter_head.",)
    if not source_span_slots and target_values["num_span_slots"]:
        allowed_prefixes += (
            "span_slot_embeddings.",
            "span_start_query.",
            "span_end_query.",
            "span_memory.",
        )
    if not source_semantic_pooling and target_values["semantic_pooling"]:
        allowed_prefixes += ("semantic_attention.", "semantic_projection.")
    if not source_execution_verifier and target_values["execution_verifier"]:
        allowed_prefixes += ("execution_verifier_head.",)
    allowed_missing = {
        name for name in model.state_dict() if name.startswith(allowed_prefixes)
    }
    if set(missing) != allowed_missing or unexpected:
        raise ValueError(
            f"unexpected warm-start state: missing={missing}, unexpected={unexpected}"
        )
    return {
        "checkpoint": str(path.resolve()),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "reused_parameters": len(model.state_dict()) - len(missing),
        "new_parameter_tensors": len(missing),
    }


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
