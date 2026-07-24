"""Train Jarvis' fast hierarchical command manager entirely from scratch.

The manager learns two related decisions from project-owned data:

1. route the utterance to a tool, control action, dialogue, or safe rejection;
2. choose the concrete intent and extract its slots.

No pretrained model, embedding, tokenizer, or external dataset is used.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .custom_data import load_jsonl, validate_splits
from .data import Example, build_examples, iter_text
from .models import build_model
from .schema import INTENTS, SLOT_LABELS
from .tokenizer import CharTokenizer, IGNORE_INDEX, WordTokenizer
from .train import (
    NLUDataset,
    _accuracy,
    _expected_calibration_error,
    _fit_temperature,
    _validation_logits,
    set_seed,
)

# The auxiliary route task teaches the model Jarvis' manager role without
# changing the runtime checkpoint interface.
ROUTES = (
    ("tool", ("get_current_time", "set_reminder", "open_application", "list_applications")),
    ("control", ("cancel",)),
    ("dialogue", ("general_chat",)),
    ("reject", ("unknown",)),
)
INTENT_TO_ROUTE = {
    INTENTS.index(intent): route_index
    for route_index, (_route, intents) in enumerate(ROUTES)
    for intent in intents
}


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - PyTorch < 2.0
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _route_logits(intent_logits: torch.Tensor) -> torch.Tensor:
    """Aggregate intent evidence into four differentiable manager routes."""
    groups = []
    for _route, intents in ROUTES:
        indices = torch.tensor(
            [INTENTS.index(intent) for intent in intents],
            dtype=torch.long,
            device=intent_logits.device,
        )
        groups.append(torch.logsumexp(intent_logits.index_select(1, indices), dim=1))
    return torch.stack(groups, dim=1)


def _route_targets(intent_targets: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(
        [INTENT_TO_ROUTE[index] for index in range(len(INTENTS))],
        dtype=torch.long,
        device=intent_targets.device,
    )
    return mapping[intent_targets]


def _source_balanced_loader(
    base_examples: list[Example],
    custom_examples: list[Example],
    tokenizer: Any,
    *,
    batch_size: int,
    custom_fraction: float,
    device: torch.device,
    seed: int,
) -> DataLoader:
    """Balance intents while explicitly controlling old/new data exposure."""
    if not 0.0 < custom_fraction < 1.0:
        raise ValueError("custom_fraction must be between 0 and 1")
    examples = base_examples + custom_examples
    sources = ["base"] * len(base_examples) + ["custom"] * len(custom_examples)
    counts = Counter(
        (source, example.intent) for source, example in zip(sources, examples)
    )
    source_probability = {"base": 1.0 - custom_fraction, "custom": custom_fraction}
    weights = [
        source_probability[source] / (len(INTENTS) * counts[(source, example.intent)])
        for source, example in zip(sources, examples)
    ]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(examples),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        NLUDataset(examples, tokenizer),
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )


def _balanced_score(custom_macro_f1: float, legacy_macro_f1: float) -> float:
    """Harmonic mean: a model cannot hide weakness on either validation domain."""
    return 2.0 * custom_macro_f1 * legacy_macro_f1 / max(
        custom_macro_f1 + legacy_macro_f1, 1e-12
    )


def _model_and_tokenizer(args: argparse.Namespace, texts: list[str]):
    tokenizer_class = WordTokenizer if args.architecture == "word_bigru" else CharTokenizer
    tokenizer = tokenizer_class.fit(texts, max_length=args.max_length)
    model_config: dict[str, int] = {"embedding_dim": args.embedding_dim}
    if args.architecture == "char_cnn":
        model_config["channels"] = args.hidden_dim
    else:
        model_config["hidden_dim"] = args.hidden_dim
    model = build_model(
        args.architecture,
        vocab_size=len(tokenizer.stoi),
        num_intents=len(INTENTS),
        num_slots=len(SLOT_LABELS),
        pad_id=tokenizer.pad_id,
        **model_config,
    )
    return model, tokenizer, model_config


def train_manager(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but PyTorch cannot see an NVIDIA GPU")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    custom_train = load_jsonl(args.train_data, allow_empty=False)
    custom_validation = load_jsonl(args.validation_data, allow_empty=False)
    split_report = validate_splits(custom_train, custom_validation)
    base_clean = build_examples("train", augmented=False, seed=args.seed)
    base_augmented = build_examples("train", augmented=True, seed=args.seed)
    legacy_validation = build_examples("validation", seed=args.seed)
    legacy_regression = build_examples("test", seed=args.seed)

    # Validation vocabulary must not leak into training. Character models can
    # still process unseen words because their alphabet comes from train only.
    tokenizer_texts = list(iter_text(base_augmented + custom_train))
    model, tokenizer, model_config = _model_and_tokenizer(args, tokenizer_texts)
    model.to(device)

    custom_validation_loader = DataLoader(
        NLUDataset(custom_validation, tokenizer),
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )
    legacy_validation_loader = DataLoader(
        NLUDataset(legacy_validation, tokenizer),
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )
    combined_validation_loader = DataLoader(
        NLUDataset(legacy_validation + custom_validation, tokenizer),
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )
    legacy_regression_loader = DataLoader(
        NLUDataset(legacy_regression, tokenizer),
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    intent_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    route_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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
            epoch_base = base_clean
            custom_fraction = args.custom_fraction
        elif args.method == "augmented":
            epoch_base = base_augmented
            custom_fraction = args.custom_fraction
        else:
            progress = epoch / args.epochs
            epoch_base = base_clean if progress <= 0.4 else base_augmented
            custom_fraction = (
                args.curriculum_start_fraction
                + (args.custom_fraction - args.curriculum_start_fraction) * progress
            )

        model.train()
        running_loss = 0.0
        batches = 0
        loader = _source_balanced_loader(
            epoch_base,
            custom_train,
            tokenizer,
            batch_size=args.batch_size,
            custom_fraction=custom_fraction,
            device=device,
            seed=args.seed * 10_000 + epoch,
        )
        for batch in loader:
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
                loss = loss + args.route_loss_weight * route_loss(
                    _route_logits(intent_logits), _route_targets(intents)
                )
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

        custom_metrics = _accuracy(model, custom_validation_loader, device)
        legacy_metrics = _accuracy(model, legacy_validation_loader, device)
        score = _balanced_score(
            custom_metrics["intent_macro_f1"], legacy_metrics["intent_macro_f1"]
        )
        history.append(
            {
                "epoch": float(epoch),
                "loss": running_loss / max(batches, 1),
                "balanced_score": score,
                "custom_macro_f1": custom_metrics["intent_macro_f1"],
                "legacy_macro_f1": legacy_metrics["intent_macro_f1"],
                "custom_slot_f1": custom_metrics["slot_entity_f1"],
            }
        )
        print(
            f"epoch={epoch:03d} loss={history[-1]['loss']:.4f} "
            f"score={score:.4f} custom_f1={custom_metrics['intent_macro_f1']:.4f} "
            f"legacy_f1={legacy_metrics['intent_macro_f1']:.4f} "
            f"slot_f1={custom_metrics['slot_entity_f1']:.4f}",
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
        raise RuntimeError("manager training produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    calibration_logits, calibration_labels = _validation_logits(
        model, combined_validation_loader, device
    )
    temperature = _fit_temperature(calibration_logits, calibration_labels)
    elapsed = time.perf_counter() - started

    metrics: dict[str, Any] = {
        **{
            f"custom_validation_{key}": value
            for key, value in _accuracy(model, custom_validation_loader, device).items()
        },
        **{
            f"legacy_validation_{key}": value
            for key, value in _accuracy(model, legacy_validation_loader, device).items()
        },
        **{
            f"legacy_regression_{key}": value
            for key, value in _accuracy(model, legacy_regression_loader, device).items()
        },
        "balanced_validation_score": best_score,
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
        "architecture": args.architecture,
        "method": args.method,
        "manager_routes": [route for route, _intents in ROUTES],
        "route_loss_weight": args.route_loss_weight,
        "custom_fraction": args.custom_fraction,
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
            "format_version": 4,
            "architecture": args.architecture,
            "tokenizer_type": "word" if args.architecture == "word_bigru" else "char",
            "method": f"manager_{args.method}",
            "model_config": model_config,
            "model_state": {name: tensor.cpu() for name, tensor in best_state.items()},
            "tokenizer": tokenizer.to_dict(),
            "metrics": metrics,
            "seed": args.seed,
            "temperature": temperature,
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
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=("char_cnn", "bigru", "word_bigru"), default="char_cnn")
    parser.add_argument("--method", choices=("standard", "augmented", "curriculum"), default="curriculum")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--slot-loss-weight", type=float, default=0.6)
    parser.add_argument("--route-loss-weight", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--custom-fraction", type=float, default=0.4)
    parser.add_argument("--curriculum-start-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    metrics = train_manager(build_parser().parse_args())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
