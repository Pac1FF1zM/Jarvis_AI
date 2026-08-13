"""Training protocol for non-autoregressive Structured JSC."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import JSCExample, load_jsc_jsonl
from .jal import DialogueAct, JALPlan, ToolSchemaRegistry, dumps
from .project_registry import build_project_schema_registry
from .sequence_data import ACT_LABELS, ACT_TO_ID, make_collate_fn
from .span_labels import span_tool_arguments
from .structured_codec import (
    STRUCTURED_SPAN_ARGUMENTS,
    build_missing_labels,
    decode_structured_jal,
)
from .structured_features import (
    serialize_structured_source,
    structured_segment_targets,
)
from .structured_labels import build_parameter_labels
from .structured_model import StructuredJSCConfig, StructuredJSCModel
from .tokenizer import JSCCharTokenizer


@dataclass(frozen=True)
class StructuredTrainingConfig:
    data_dir: str = "training_workspace/jsc_data"
    output_dir: str = "training_workspace/jsc_structured_runs/seed17"
    seed: int = 17
    device: str = "auto"
    epochs: int = 36
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    d_model: int = 192
    encoder_layers: int = 3
    attention_heads: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.12
    max_source_length: int = 416
    patience: int = 8
    warmup_ratio: float = 0.08
    gradient_clip: float = 1.0
    use_amp: bool = True
    execution_threshold: float = 0.55
    verifier_threshold: float = 0.50
    span_threshold: float = 0.30
    parameter_threshold: float = 0.50
    missing_threshold: float = 0.45
    resume: str | None = None
    smoke: bool = False
    segmented_router: bool = True

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if min(self.epochs, self.batch_size, self.patience) < 1:
            raise ValueError("epochs, batch size and patience must be positive")


@dataclass(frozen=True)
class _Context:
    registry: ToolSchemaRegistry
    train: tuple[JSCExample, ...]
    validation: tuple[JSCExample, ...]
    tokenizer: JSCCharTokenizer
    tool_labels: tuple[str, ...]
    parameter_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    reason_labels: tuple[str, ...]
    data_fingerprint: str


@dataclass(frozen=True)
class StructuredLogitCache:
    """Source-only model outputs reusable for validation threshold selection."""

    utterances: tuple[str, ...]
    source_texts: tuple[str, ...]
    states: tuple[JALPlan | None, ...]
    outputs: tuple[torch.Tensor, ...]
    registry: ToolSchemaRegistry
    tool_labels: tuple[str, ...]
    parameter_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    reason_labels: tuple[str, ...]


class _StructuredDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[JSCExample],
        tokenizer: JSCCharTokenizer,
        max_source_length: int,
    ) -> None:
        self.examples = tuple(examples)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        source_text = serialize_structured_source(example)
        source_ids = self.tokenizer.encode(
            source_text, max_length=self.max_source_length
        )
        return {
            "source_ids": source_ids,
            "source_text": source_text,
            # Compatibility fields for the shared structured collator.  These
            # carry only BOS/EOS controls, never a serialized JAL target.
            "decoder_input_ids": [self.tokenizer.bos_id],
            "labels": [self.tokenizer.eos_id],
            "act": ACT_TO_ID[example.target.act.value],
            "scenario_id": example.scenario_id,
            "category": example.category,
            "target_text": example.scenario_id,
            "tools": [step.tool for step in example.target.steps],
            "calls": list(example.target.steps),
        }


def train_structured(config: StructuredTrainingConfig) -> dict[str, Any]:
    _set_seed(config.seed)
    context = _load_context(config)
    device = _device(config.device)
    model_config = StructuredJSCConfig(
        vocab_size=context.tokenizer.size,
        num_acts=len(ACT_LABELS),
        num_tools=len(context.tool_labels),
        num_parameter_labels=len(context.parameter_labels),
        num_span_slots=len(STRUCTURED_SPAN_ARGUMENTS),
        num_missing_labels=len(context.missing_labels),
        num_reasons=len(context.reason_labels),
        d_model=32 if config.smoke else config.d_model,
        encoder_layers=1 if config.smoke else config.encoder_layers,
        step_layers=1 if config.smoke else 2,
        attention_heads=config.attention_heads,
        feedforward_dim=64 if config.smoke else config.feedforward_dim,
        dropout=config.dropout,
        max_source_length=config.max_source_length,
        pad_id=context.tokenizer.pad_id,
        segmented_router=config.segmented_router,
    )
    model = StructuredJSCModel(model_config).to(device)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    latest_path = output / "latest.pt"
    report_path = output / "report.json"
    if config.resume is None and any(
        path.exists() for path in (best_path, latest_path, report_path)
    ):
        raise FileExistsError(f"structured run already exists: {output}")
    collate = _collate(context)
    train_dataset = _StructuredDataset(
        context.train, context.tokenizer, config.max_source_length
    )
    validation_dataset = _StructuredDataset(
        context.validation, context.tokenizer, config.max_source_length
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4 if config.smoke else config.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_dataset) / config.batch_size)
    total_steps = max(steps_per_epoch * (1 if config.smoke else config.epochs), 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _schedule(total_steps, warmup_steps)
    )
    # CUDA autocast uses bfloat16 here, which does not need loss scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    signature = _signature(config, context, model_config)
    start_epoch = 0
    best_rank: tuple[float, ...] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    if config.resume:
        state = torch.load(config.resume, map_location=device, weights_only=False)
        if state.get("run_signature") != signature:
            raise ValueError("structured resume signature mismatch")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = state["epoch"] + 1
        best_rank = tuple(state["best_rank"])
        best_epoch = state["best_epoch"]
        stale = state["stale"]
        history = list(state["history"])
    started = time.perf_counter()
    class_weights = _class_weights(context, device)
    for epoch in range(start_epoch, 1 if config.smoke else config.epochs):
        train_loader = _train_loader(
            train_dataset, context.train, collate, config, epoch
        )
        train_metrics = _epoch(
            model,
            train_loader,
            device,
            config,
            class_weights,
            optimizer,
            scheduler,
            scaler,
        )
        validation_loss = _epoch(
            model,
            validation_loader,
            device,
            config,
            class_weights,
        )
        program = evaluate_structured(
            model, context.validation, context, device, config.batch_size, config
        )
        rank = (
            float(program["opposite_action_rate"] > 0.0),
            -float(program["exact_jal_accuracy"]),
            float(program["false_execution_rate"]),
            -float(program["tool_sequence_accuracy"]),
            float(validation_loss["loss"]),
        )
        improved = best_rank is None or rank < best_rank
        if improved:
            best_rank = rank
            best_epoch = epoch
            stale = 0
            _save_checkpoint(
                best_path, model, model_config, context, config, epoch, signature
            )
        else:
            stale += 1
        epoch_report = {
            "epoch": epoch,
            "train": train_metrics,
            "validation_loss": validation_loss,
            "validation_program": program,
            "selection_rank": rank,
        }
        history.append(epoch_report)
        torch.save(
            {
                "kind": "jsc_structured_training_state",
                "run_signature": signature,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_rank": best_rank,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
            },
            latest_path,
        )
        print(
            f"epoch={epoch + 1}/{1 if config.smoke else config.epochs} "
            f"loss={train_metrics['loss']:.4f} val_loss={validation_loss['loss']:.4f} "
            f"val_exact={program['exact_jal_accuracy']:.4f} "
            f"val_tool={program['tool_sequence_accuracy']:.4f} "
            f"val_false={program['false_execution_rate']:.4f}",
            flush=True,
        )
        if stale >= config.patience:
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final = evaluate_structured(
        model, context.validation, context, device, config.batch_size, config
    )
    report = {
        "format_version": 2 if model.config.segmented_router else 1,
        "architecture": (
            "structured_jsc_segmented_router"
            if model.config.segmented_router
            else "structured_jsc_no_json"
        ),
        "seed": config.seed,
        "parameters": model.parameter_count(),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(best_path.resolve()),
        "model_config": model_config.to_dict(),
        "training_config": asdict(config),
        "selection": {
            "metric": "validation_program_exact_jal_safety_first",
            "rank": best_rank,
            "test_opened": False,
            "evaluation_holdout_opened": False,
        },
        "validation": final,
        "history": history,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


@torch.inference_mode()
def evaluate_structured_checkpoint(
    checkpoint_path: str | Path,
    examples: Sequence[JSCExample],
    registry: ToolSchemaRegistry,
    *,
    device: str = "auto",
    batch_size: int = 64,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    resolved = _device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
    if checkpoint.get("kind") != "jsc_structured_inference":
        raise ValueError("not a Structured JSC checkpoint")
    if checkpoint.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("structured checkpoint schema mismatch")
    tokenizer = JSCCharTokenizer.from_dict(checkpoint["tokenizer"])
    model = StructuredJSCModel(
        StructuredJSCConfig.from_dict(checkpoint["model_config"])
    ).to(resolved)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    config = StructuredTrainingConfig(**checkpoint["training_config"])
    if thresholds:
        config = StructuredTrainingConfig(
            **{**asdict(config), **dict(thresholds), "resume": None}
        )
    context = _Context(
        registry=registry,
        train=(),
        validation=tuple(examples),
        tokenizer=tokenizer,
        tool_labels=tuple(checkpoint["tool_labels"]),
        parameter_labels=tuple(checkpoint["parameter_labels"]),
        missing_labels=tuple(checkpoint["missing_labels"]),
        reason_labels=tuple(checkpoint["reason_labels"]),
        data_fingerprint=str(checkpoint["data_fingerprint"]),
    )
    predictions, decisions = _predict(
        model, examples, context, resolved, batch_size, config
    )
    return predictions, decisions


@torch.inference_mode()
def cache_structured_checkpoint_logits(
    checkpoint_path: str | Path,
    examples: Sequence[JSCExample],
    registry: ToolSchemaRegistry,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> tuple[StructuredLogitCache, StructuredTrainingConfig, Mapping[str, Any]]:
    """Run a checkpoint once and retain CPU logits for cheap calibration."""
    resolved = _device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved, weights_only=False)
    if checkpoint.get("kind") != "jsc_structured_inference":
        raise ValueError("not a Structured JSC checkpoint")
    if checkpoint.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("structured checkpoint schema mismatch")
    tokenizer = JSCCharTokenizer.from_dict(checkpoint["tokenizer"])
    model = StructuredJSCModel(
        StructuredJSCConfig.from_dict(checkpoint["model_config"])
    ).to(resolved)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    config = StructuredTrainingConfig(**checkpoint["training_config"])
    context = _Context(
        registry=registry,
        train=(),
        validation=tuple(examples),
        tokenizer=tokenizer,
        tool_labels=tuple(checkpoint["tool_labels"]),
        parameter_labels=tuple(checkpoint["parameter_labels"]),
        missing_labels=tuple(checkpoint["missing_labels"]),
        reason_labels=tuple(checkpoint["reason_labels"]),
        data_fingerprint=str(checkpoint["data_fingerprint"]),
    )
    dataset = _StructuredDataset(examples, tokenizer, config.max_source_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate(context)
    )
    by_id = {example.scenario_id: example for example in examples}
    ordered_examples: list[JSCExample] = []
    chunks: list[list[torch.Tensor]] = [
        [] for _ in range(10 if model.config.segmented_router else 9)
    ]
    for batch in loader:
        outputs = model(
            batch["source_ids"].to(resolved), batch["source_mask"].to(resolved)
        )
        for index, value in enumerate(outputs):
            chunks[index].append(value.float().cpu())
        ordered_examples.extend(by_id[value] for value in batch["scenario_id"])
    maximum_source_width = max(
        value.shape[-1] for index in (4, 5) for value in chunks[index]
    )
    variable_width_outputs = (4, 5, 9) if len(chunks) > 9 else (4, 5)
    for index in variable_width_outputs:
        chunks[index] = [
            F.pad(
                value,
                (0, maximum_source_width - value.shape[-1]),
                value=torch.finfo(value.dtype).min,
            )
            for value in chunks[index]
        ]
    cache = StructuredLogitCache(
        utterances=tuple(example.text for example in ordered_examples),
        source_texts=tuple(
            serialize_structured_source(example) for example in ordered_examples
        ),
        states=tuple(example.state for example in ordered_examples),
        outputs=tuple(torch.cat(values) for values in chunks),
        registry=registry,
        tool_labels=context.tool_labels,
        parameter_labels=context.parameter_labels,
        missing_labels=context.missing_labels,
        reason_labels=context.reason_labels,
    )
    return cache, config, checkpoint


def decode_structured_cache(
    cache: StructuredLogitCache,
    config: StructuredTrainingConfig,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Decode cached logits under validation-selected confidence thresholds."""
    values = asdict(config)
    if thresholds:
        values.update(thresholds)
    tuned = StructuredTrainingConfig(**values)
    outputs = cache.outputs
    decoded = decode_structured_jal(
        utterances=cache.utterances,
        source_texts=cache.source_texts,
        act_logits=outputs[0],
        count_logits=outputs[1],
        tool_logits=outputs[2],
        parameter_logits=outputs[3],
        span_start_logits=outputs[4],
        span_end_logits=outputs[5],
        verifier_logits=outputs[6],
        missing_logits=outputs[7],
        reason_logits=outputs[8],
        registry=cache.registry,
        tool_labels=cache.tool_labels,
        parameter_labels=cache.parameter_labels,
        missing_labels=cache.missing_labels,
        reason_labels=cache.reason_labels,
        states=cache.states,
        execution_threshold=tuned.execution_threshold,
        verifier_threshold=tuned.verifier_threshold,
        parameter_threshold=tuned.parameter_threshold,
        span_threshold=tuned.span_threshold,
        missing_threshold=tuned.missing_threshold,
    )
    return list(decoded.predictions), dict(decoded.decisions)


def evaluate_structured(
    model: StructuredJSCModel,
    examples: Sequence[JSCExample],
    context: _Context,
    device: torch.device,
    batch_size: int,
    config: StructuredTrainingConfig,
) -> dict[str, Any]:
    from training_workspace.jsc_migration_benchmark import _migration_metrics

    predictions, decisions = _predict(
        model, examples, context, device, batch_size, config
    )
    return {
        **_migration_metrics(examples, predictions, context.registry),
        "decoder_decisions": decisions,
    }


@torch.inference_mode()
def _predict(
    model: StructuredJSCModel,
    examples: Sequence[JSCExample],
    context: _Context,
    device: torch.device,
    batch_size: int,
    config: StructuredTrainingConfig,
) -> tuple[list[str], dict[str, int]]:
    model.eval()
    dataset = _StructuredDataset(examples, context.tokenizer, config.max_source_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate(context))
    by_id = {example.scenario_id: example for example in examples}
    predictions: list[str] = []
    decisions: Counter[str] = Counter()
    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        outputs = model(source_ids, source_mask)
        ordered = [by_id[value] for value in batch["scenario_id"]]
        decoded = decode_structured_jal(
            utterances=[example.text for example in ordered],
            source_texts=[serialize_structured_source(example) for example in ordered],
            act_logits=outputs[0].cpu(),
            count_logits=outputs[1].cpu(),
            tool_logits=outputs[2].cpu(),
            parameter_logits=outputs[3].cpu(),
            span_start_logits=outputs[4].cpu(),
            span_end_logits=outputs[5].cpu(),
            verifier_logits=outputs[6].cpu(),
            missing_logits=outputs[7].cpu(),
            reason_logits=outputs[8].cpu(),
            registry=context.registry,
            tool_labels=context.tool_labels,
            parameter_labels=context.parameter_labels,
            missing_labels=context.missing_labels,
            reason_labels=context.reason_labels,
            states=[example.state for example in ordered],
            execution_threshold=config.execution_threshold,
            verifier_threshold=config.verifier_threshold,
            parameter_threshold=config.parameter_threshold,
            span_threshold=config.span_threshold,
            missing_threshold=config.missing_threshold,
        )
        predictions.extend(decoded.predictions)
        decisions.update(decoded.decisions)
    return predictions, dict(decisions)


def _epoch(
    model: StructuredJSCModel,
    loader: DataLoader,
    device: torch.device,
    config: StructuredTrainingConfig,
    weights: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Counter[str] = Counter()
    for batch_index, batch in enumerate(loader):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        source = batch["source_ids"].to(device)
        mask = batch["source_mask"].to(device)
        tools = batch["tool_ids"].to(device)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=device.type == "cuda" and config.use_amp,
        ):
            outputs = model(
                source,
                mask,
                conditioning_tool_ids=tools,
                conditioning_segment_ids=(
                    batch["segment_ids"].to(device)
                    if model.config.segmented_router
                    else None
                ),
            )
            loss_parts = _loss(outputs, batch, device, weights)
            loss = sum(loss_parts.values())
        if training:
            assert optimizer is not None and scheduler is not None and scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        totals["loss"] += float(loss.detach())
        for name, value in loss_parts.items():
            totals[name] += float(value.detach())
        totals["batches"] += 1
        if config.smoke and batch_index >= 1:
            break
    return {
        name: value / max(totals["batches"], 1)
        for name, value in totals.items()
        if name != "batches"
    } | {"batches": int(totals["batches"])}


def _loss(
    outputs: tuple[torch.Tensor, ...],
    batch: Mapping[str, Any],
    device: torch.device,
    weights: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    act, count, tools, parameters, starts, ends, verifier, missing, reason = outputs[:9]
    acts = batch["act"].to(device)
    counts = batch["step_count"].to(device)
    tool_targets = batch["tool_ids"].to(device)
    active_steps = torch.arange(tools.shape[1], device=device)[None] < counts[:, None]
    result = {
        "act": 2.0 * F.cross_entropy(act, acts, weight=weights["act"]),
        "count": 2.0 * F.cross_entropy(count, counts, weight=weights["count"]),
        "verifier": F.cross_entropy(
            verifier, batch["execution_allowed"].to(device), weight=weights["verifier"]
        ),
        "reason": 0.5 * F.cross_entropy(
            reason, batch["reason_targets"].to(device), weight=weights["reason"]
        ),
    }
    if active_steps.any():
        result["tool"] = 2.0 * F.cross_entropy(
            tools[active_steps], tool_targets[active_steps], weight=weights["tool"]
        )
    parameter_mask = batch["parameter_mask"].to(device)
    if parameter_mask.any():
        raw = F.binary_cross_entropy_with_logits(
            parameters,
            batch["parameter_targets"].to(device),
            reduction="none",
            pos_weight=weights["parameter_pos"],
        )
        result["parameter"] = raw[parameter_mask].mean()
    span_mask = batch["span_mask"].to(device)
    if span_mask.any():
        result["span_start"] = F.cross_entropy(
            starts[span_mask], batch["span_start_targets"].to(device)[span_mask]
        )
        result["span_end"] = F.cross_entropy(
            ends[span_mask], batch["span_end_targets"].to(device)[span_mask]
        )
    missing_mask = batch["missing_mask"].to(device)
    if missing_mask.any():
        raw = F.binary_cross_entropy_with_logits(
            missing,
            batch["missing_targets"].to(device),
            reduction="none",
            pos_weight=weights["missing_pos"],
        )
        result["missing"] = raw[missing_mask].mean()
    if len(outputs) > 9:
        boundary_mask = batch["boundary_mask"].to(device)
        if boundary_mask.any():
            targets = batch["boundary_targets"].to(device)
            positives = targets[boundary_mask].sum()
            negatives = boundary_mask.sum() - positives
            positive_weight = (negatives / positives.clamp_min(1.0)).clamp(
                1.0, 32.0
            )
            result["boundary"] = 0.75 * F.binary_cross_entropy_with_logits(
                outputs[9][boundary_mask],
                targets[boundary_mask],
                pos_weight=positive_weight,
            )
    return result


def _collate(context: _Context):
    tool_to_id = {name: index for index, name in enumerate(context.tool_labels)}
    base = make_collate_fn(
        context.tokenizer.pad_id,
        tool_to_id,
        {name: index for index, name in enumerate(context.parameter_labels)},
        STRUCTURED_SPAN_ARGUMENTS,
        span_tool_arguments(context.registry, STRUCTURED_SPAN_ARGUMENTS),
    )
    examples = {
        example.scenario_id: example
        for example in (*context.train, *context.validation)
    }
    missing_to_id = {name: index for index, name in enumerate(context.missing_labels)}
    reason_to_id = {name: index for index, name in enumerate(context.reason_labels)}

    def collate(rows: list[dict[str, object]]) -> dict[str, Any]:
        batch = base(rows)
        missing_targets = torch.zeros(
            len(rows), 8, len(context.missing_labels), dtype=torch.float32
        )
        missing_mask = torch.zeros_like(missing_targets, dtype=torch.bool)
        reason_targets = []
        for row_index, scenario_id in enumerate(batch["scenario_id"]):
            example = examples[scenario_id]
            for step_index, call in enumerate(example.target.steps):
                prefix = call.tool + ":"
                for label, label_id in missing_to_id.items():
                    if label.startswith(prefix):
                        missing_mask[row_index, step_index, label_id] = True
            for slot in example.target.missing:
                tool = example.target.steps[slot.step].tool
                missing_targets[
                    row_index, slot.step, missing_to_id[f"{tool}:{slot.name}"]
                ] = 1.0
            reason_targets.append(reason_to_id[example.target.reason or "<none>"])
        batch["missing_targets"] = missing_targets
        batch["missing_mask"] = missing_mask
        batch["reason_targets"] = torch.tensor(reason_targets, dtype=torch.long)
        width = int(batch["source_ids"].shape[1])
        segment_rows: list[list[int]] = []
        boundary_rows: list[list[float]] = []
        boundary_mask_rows: list[list[bool]] = []
        for row, scenario_id in zip(rows, batch["scenario_id"]):
            example = examples[scenario_id]
            segments, boundaries, supervised = structured_segment_targets(
                str(row["source_text"]), len(example.target.steps)
            )
            padding = width - len(segments)
            segment_rows.append(segments + [-1] * padding)
            boundary_rows.append(boundaries + [0.0] * padding)
            boundary_mask_rows.append(supervised + [False] * padding)
        batch["segment_ids"] = torch.tensor(segment_rows, dtype=torch.long)
        batch["boundary_targets"] = torch.tensor(
            boundary_rows, dtype=torch.float32
        )
        batch["boundary_mask"] = torch.tensor(
            boundary_mask_rows, dtype=torch.bool
        )
        return batch

    return collate


def _load_context(config: StructuredTrainingConfig) -> _Context:
    directory = Path(config.data_dir)
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    registry = build_project_schema_registry()
    if manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("dataset schema mismatch")
    loaded = {}
    for split in ("train", "validation"):
        path = directory / f"{split}.jsonl"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["splits"][split]["sha256"]:
            raise ValueError(f"{split} hash mismatch")
        loaded[split] = tuple(load_jsc_jsonl(path, registry, expected_split=split))
    tokenizer = JSCCharTokenizer.fit(
        serialize_structured_source(example) for example in loaded["train"]
    )
    reasons = ("<none>",) + tuple(
        sorted({example.target.reason for example in loaded["train"] if example.target.reason})
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "train": manifest["splits"]["train"]["sha256"],
                "validation": manifest["splits"]["validation"]["sha256"],
                "schema": registry.schema_fingerprint,
                "structured_format": 5,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return _Context(
        registry=registry,
        train=loaded["train"],
        validation=loaded["validation"],
        tokenizer=tokenizer,
        tool_labels=("<none>", *registry.tool_names),
        parameter_labels=build_parameter_labels(registry),
        missing_labels=build_missing_labels(registry),
        reason_labels=reasons,
        data_fingerprint=fingerprint,
    )


def _class_weights(context: _Context, device: torch.device) -> dict[str, torch.Tensor]:
    def weights(values: Sequence[int], size: int) -> torch.Tensor:
        counts = Counter(values)
        result = torch.ones(size, dtype=torch.float32, device=device)
        total = len(values)
        for index in range(size):
            result[index] = math.sqrt(total / max(counts[index], 1))
        return result / result.mean()

    act_values = [ACT_TO_ID[example.target.act.value] for example in context.train]
    count_values = [len(example.target.steps) for example in context.train]
    tool_to_id = {name: index for index, name in enumerate(context.tool_labels)}
    tool_values = [
        tool_to_id[step.tool]
        for example in context.train
        for step in example.target.steps
    ]
    reason_to_id = {name: index for index, name in enumerate(context.reason_labels)}
    reason_values = [reason_to_id[example.target.reason or "<none>"] for example in context.train]
    execution = [int(example.target.act == DialogueAct.EXECUTE) for example in context.train]
    return {
        # Apply imbalance correction once, at the loss.  Combining this with
        # replacement sampling previously distorted the real command prior.
        "act": weights(act_values, len(ACT_LABELS)),
        "count": weights(count_values, 9),
        "tool": weights(tool_values, len(context.tool_labels)),
        "reason": weights(reason_values, len(context.reason_labels)),
        "verifier": weights(execution, 2),
        "parameter_pos": torch.full(
            (len(context.parameter_labels),), 2.0, device=device
        ),
        "missing_pos": torch.full(
            (len(context.missing_labels),), 8.0, device=device
        ),
    }


def _train_loader(dataset, examples, collate, config, epoch):
    return DataLoader(
        dataset,
        batch_size=4 if config.smoke else config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + epoch * 10_007),
        collate_fn=collate,
    )


def _save_checkpoint(path, model, model_config, context, config, epoch, signature):
    torch.save(
        {
            "format_version": 2 if model.config.segmented_router else 1,
            "kind": "jsc_structured_inference",
            "model_config": model_config.to_dict(),
            "model_state": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "tokenizer": context.tokenizer.to_dict(),
            "tool_labels": context.tool_labels,
            "parameter_labels": context.parameter_labels,
            "missing_labels": context.missing_labels,
            "reason_labels": context.reason_labels,
            "tool_schema_sha256": context.registry.schema_fingerprint,
            "data_fingerprint": context.data_fingerprint,
            "training_config": asdict(config) | {"resume": None},
            "run_signature": signature,
            "seed": config.seed,
            "epoch": epoch,
        },
        path,
    )


def _signature(config, context, model_config):
    payload = {
        "training": asdict(config) | {"output_dir": None, "device": None, "resume": None},
        "model": model_config.to_dict(),
        "data": context.data_fingerprint,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _schedule(total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + math.cos(math.pi * min(progress, 1.0))), 0.05)

    return schedule
