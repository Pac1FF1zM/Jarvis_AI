"""Training utilities and metrics for Jarvis Gesture Core experiments."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .labels import IPN_LABELS, NO_GESTURE_LABEL


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 35
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    label_smoothing: float = 0.03
    patience: int = 8
    device: str = "cuda"
    amp: bool = True
    run_name: str = "gesture"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _metrics(target: list[int], predicted: list[int]) -> dict[str, Any]:
    if not target:
        raise ValueError("Cannot calculate metrics for zero samples")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores = []
    no_gesture_index = IPN_LABELS.index(NO_GESTURE_LABEL)
    for index, name in enumerate(IPN_LABELS):
        tp = sum(actual == index and guess == index for actual, guess in zip(target, predicted))
        fp = sum(actual != index and guess == index for actual, guess in zip(target, predicted))
        fn = sum(actual == index and guess != index for actual, guess in zip(target, predicted))
        support = sum(actual == index for actual in target)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1_scores.append(f1)
    non_gesture = [guess for actual, guess in zip(target, predicted) if actual == no_gesture_index]
    false_trigger_rate = (
        sum(guess != no_gesture_index for guess in non_gesture) / len(non_gesture) if non_gesture else None
    )
    return {
        "accuracy": sum(actual == guess for actual, guess in zip(target, predicted)) / len(target),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "no_gesture_recall": per_class[NO_GESTURE_LABEL]["recall"],
        "false_trigger_rate": false_trigger_rate,
        "per_class": per_class,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    total_loss = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    for clips, labels in loader:
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(clips)
        total_loss += float(criterion(logits, labels)) * labels.numel()
        count += labels.numel()
        expected.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    result = _metrics(expected, predicted)
    result["loss"] = total_loss / max(count, 1)
    return result


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    settings: TrainingSettings,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train only on train; validation selects the epoch and is never merged in."""
    device = torch.device(settings.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=settings.label_smoothing)
    amp_enabled = settings.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * labels.numel()
            count += labels.numel()
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        # A classifier that fires during idle motion is unsafe even if its raw F1 is high.
        score = 0.8 * validation["macro_f1"] + 0.2 * float(validation["no_gesture_recall"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(count, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "selection_score": score,
            "validation": validation,
        }
        history.append(row)
        print(
            "GESTURE_EPOCH "
            f"experiment={settings.run_name} epoch={epoch}/{settings.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"validation_macro_f1={validation['macro_f1']:.4f} "
            f"no_gesture_recall={validation['no_gesture_recall']:.4f} "
            f"selection_score={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= settings.patience:
                print(
                    f"GESTURE_EARLY_STOP experiment={settings.run_name} epoch={epoch}",
                    flush=True,
                )
                break
    if best_state is None:  # defensive; a non-empty dataset always creates one
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history
