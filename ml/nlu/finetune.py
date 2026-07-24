"""Continue training a Jarvis-owned NLU checkpoint on local JSONL data."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .custom_data import load_jsonl, validate_splits
from .data import Example, build_examples, iter_text
from .models import build_model
from .schema import INTENTS, SLOT_LABELS
from .tokenizer import CharTokenizer, WordTokenizer, IGNORE_INDEX
from .train import (
    NLUDataset,
    _accuracy,
    _expected_calibration_error,
    _fit_temperature,
    _validation_logits,
    set_seed,
)


def _make_grad_scaler(enabled: bool):
    """Use the current AMP API while retaining compatibility with PyTorch 2.0."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - old PyTorch only
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(
    examples: list[Example], tokenizer: Any, batch_size: int, device: torch.device
) -> DataLoader:
    counts = {intent: 0 for intent in INTENTS}
    for example in examples:
        counts[example.intent] += 1
    weights = [1.0 / max(counts[example.intent], 1) for example in examples]
    sampler = WeightedRandomSampler(weights, num_samples=len(examples), replacement=True)
    return DataLoader(
        NLUDataset(examples, tokenizer),
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )


def _restore_with_expanded_vocabulary(
    model: nn.Module, checkpoint_state: dict[str, torch.Tensor]
) -> None:
    """Restore all weights and copy old embedding rows into the enlarged table."""
    target = model.state_dict()
    for name, old_value in checkpoint_state.items():
        if name not in target:
            raise ValueError(f"checkpoint contains unexpected tensor {name!r}")
        if target[name].shape == old_value.shape:
            target[name] = old_value
        elif name == "embedding.weight" and target[name].shape[1:] == old_value.shape[1:]:
            if target[name].shape[0] < old_value.shape[0]:
                raise ValueError("fine-tuning vocabulary cannot shrink")
            target[name][: old_value.shape[0]] = old_value
        else:
            raise ValueError(
                f"incompatible tensor {name}: checkpoint={tuple(old_value.shape)} "
                f"target={tuple(target[name].shape)}"
            )
    model.load_state_dict(target)


def finetune(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but PyTorch cannot see the GPU. Install a CUDA-enabled "
            "PyTorch build and verify `python -c \"import torch; print(torch.cuda.is_available())\"`."
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer_cls = WordTokenizer if checkpoint.get("tokenizer_type") == "word" else CharTokenizer
    tokenizer = tokenizer_cls.from_dict(checkpoint["tokenizer"])
    custom_train = load_jsonl(args.train_data)
    custom_validation = load_jsonl(args.validation_data)
    split_report = validate_splits(custom_train, custom_validation)
    if args.require_custom_data and not custom_train:
        raise ValueError(f"{args.train_data}: add at least one fine-tuning example")

    clean_base = build_examples("train", augmented=False, seed=args.seed)
    augmented_base = build_examples("train", augmented=True, seed=args.seed)
    repeated_custom = custom_train * max(args.custom_repeat, 1)
    clean_train = clean_base + repeated_custom
    augmented_train = augmented_base + repeated_custom
    validation = build_examples("validation", seed=args.seed) + custom_validation
    development_test = build_examples("test", seed=args.seed)
    added_tokens = tokenizer.extend(iter_text(clean_train + custom_validation))

    model = build_model(
        checkpoint["architecture"],
        vocab_size=len(tokenizer.stoi),
        num_intents=len(INTENTS),
        num_slots=len(SLOT_LABELS),
        pad_id=tokenizer.pad_id,
        **checkpoint["model_config"],
    )
    _restore_with_expanded_vocabulary(model, checkpoint["model_state"])
    model.to(device)

    validation_loader = DataLoader(
        NLUDataset(validation, tokenizer), batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        NLUDataset(development_test, tokenizer), batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    intent_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    slot_weights = torch.tensor((0.2, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0), device=device)
    slot_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=slot_weights)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = _make_grad_scaler(amp_enabled)

    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if args.method == "standard":
            epoch_examples = clean_train
        elif args.method == "augmented":
            epoch_examples = augmented_train
        else:
            epoch_examples = clean_train if epoch <= args.epochs // 2 else augmented_train
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in _loader(epoch_examples, tokenizer, args.batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            intents = batch["intent"].to(device, non_blocking=True)
            slots = batch["slots"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                intent_logits, slot_logits = model(ids, mask)
                loss = intent_loss(intent_logits, intents)
                loss = loss + args.slot_loss_weight * slot_loss(
                    slot_logits.reshape(-1, len(SLOT_LABELS)), slots.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach())
            batches += 1

        validation_metrics = _accuracy(model, validation_loader, device)
        score = validation_metrics["intent_macro_f1"]
        history.append({
            "epoch": float(epoch),
            "loss": running_loss / max(batches, 1),
            **validation_metrics,
        })
        print(
            f"epoch={epoch:03d} loss={history[-1]['loss']:.4f} "
            f"val_macro_f1={score:.4f} val_slot_f1={validation_metrics['slot_entity_f1']:.4f}",
            flush=True,
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    calibration_logits, calibration_labels = _validation_logits(
        model, validation_loader, device
    )
    temperature = _fit_temperature(calibration_logits, calibration_labels)
    elapsed = time.perf_counter() - started
    metrics: dict[str, Any] = {
        **{f"validation_{key}": value for key, value in _accuracy(model, validation_loader, device).items()},
        **{f"test_{key}": value for key, value in _accuracy(model, test_loader, device).items()},
        "temperature": temperature,
        "validation_ece_before": _expected_calibration_error(
            calibration_logits, calibration_labels, 1.0
        ),
        "validation_ece_after": _expected_calibration_error(
            calibration_logits, calibration_labels, temperature
        ),
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "training_seconds": elapsed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": len(tokenizer.stoi),
        "added_tokens": added_tokens,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp_enabled,
        "tf32": bool(args.tf32 and device.type == "cuda"),
        "custom_data": split_report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 3,
            "architecture": checkpoint["architecture"],
            "tokenizer_type": checkpoint.get("tokenizer_type", "char"),
            "method": f"finetune_{args.method}",
            "model_config": checkpoint["model_config"],
            "model_state": {name: tensor.cpu() for name, tensor in best_state.items()},
            "tokenizer": tokenizer.to_dict(),
            "metrics": metrics,
            "seed": args.seed,
            "temperature": temperature,
            "parent_checkpoint": str(Path(args.checkpoint).resolve()),
            "history": history,
            "environment": {"python": platform.python_version(), "torch": torch.__version__},
        },
        output,
    )
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--method", choices=("standard", "augmented", "curriculum"), default="curriculum")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--slot-loss-weight", type=float, default=0.6)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--custom-repeat", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-custom-data", action="store_true")
    return parser


def main() -> None:
    metrics = finetune(build_parser().parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
