"""Reproducible training CLI for Jarvis models built from random weights.

Examples::

    python -m ml.nlu.train --architecture char_cnn --method standard
    python -m ml.nlu.train --architecture bigru --method augmented
    python -m ml.nlu.train --architecture char_cnn --method curriculum
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import Example, build_examples, iter_text
from .models import build_model
from .schema import ACTIONABLE_INTENTS, INTENTS, INTENT_SLOTS, SLOT_LABELS
from .tokenizer import CharTokenizer, WordTokenizer, IGNORE_INDEX


class NLUDataset(Dataset):
    def __init__(self, examples: list[Example], tokenizer: CharTokenizer) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.intent_to_id = {name: index for index, name in enumerate(INTENTS)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        ids, mask = self.tokenizer.encode(example.text)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "intent": torch.tensor(self.intent_to_id[example.intent], dtype=torch.long),
            "slots": torch.tensor(self.tokenizer.encode_slots(example), dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Fair hyperparameter comparisons require repeatable kernels.  These
    # switches trade a little peak throughput for substantially less seed noise.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    intent_correct = intent_total = slot_correct = slot_total = 0
    true_positive = false_positive = false_negative = 0
    frame_correct = actionable_correct = actionable_total = 0
    no_slot_total = hallucinated_frames = 0
    per_slot_counts = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in ("duration", "reminder_text", "application")
    }
    confusion = torch.zeros((len(INTENTS), len(INTENTS)), dtype=torch.long)
    with torch.inference_mode():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            intent = batch["intent"].to(device)
            slots = batch["slots"].to(device)
            intent_logits, slot_logits = model(ids, mask)
            intent_predictions = intent_logits.argmax(-1)
            intent_correct += int((intent_predictions == intent).sum())
            intent_total += intent.numel()
            for expected, predicted in zip(intent.cpu(), intent_predictions.cpu()):
                confusion[int(expected), int(predicted)] += 1
            valid = slots != IGNORE_INDEX
            slot_predictions = slot_logits.argmax(-1)
            slot_correct += int(((slot_predictions == slots) & valid).sum())
            slot_total += int(valid.sum())
            expected_entity = (slots != 0) & valid
            predicted_entity = (slot_predictions != 0) & valid
            true_positive += int(((slot_predictions == slots) & expected_entity).sum())
            false_positive += int((predicted_entity & (slot_predictions != slots)).sum())
            false_negative += int((expected_entity & (slot_predictions != slots)).sum())
            for row in range(intent.numel()):
                valid_row = valid[row]
                expected_row = slots[row][valid_row]
                predicted_row = slot_predictions[row][valid_row]
                intent_name = INTENTS[int(intent[row])]
                exact = bool(intent_predictions[row] == intent[row]) and torch.equal(
                    predicted_row, expected_row
                )
                frame_correct += int(exact)
                if intent_name in ACTIONABLE_INTENTS:
                    actionable_total += 1
                    actionable_correct += int(exact)
                if not INTENT_SLOTS[intent_name]:
                    no_slot_total += 1
                    hallucinated_frames += int(bool((predicted_row != 0).any()))
                for slot_name in per_slot_counts:
                    expected_type = torch.tensor(
                        [
                            (
                                SLOT_LABELS[int(value)].partition("-")[2]
                                if int(value) else ""
                            )
                            == slot_name
                            for value in expected_row.cpu()
                        ],
                        dtype=torch.bool,
                    )
                    predicted_type = torch.tensor(
                        [
                            (
                                SLOT_LABELS[int(value)].partition("-")[2]
                                if int(value) else ""
                            )
                            == slot_name
                            for value in predicted_row.cpu()
                        ],
                        dtype=torch.bool,
                    )
                    counts = per_slot_counts[slot_name]
                    counts["tp"] += int((expected_type & predicted_type).sum())
                    counts["fp"] += int((~expected_type & predicted_type).sum())
                    counts["fn"] += int((expected_type & ~predicted_type).sum())
    f1_values: list[float] = []
    recall_values: list[float] = []
    per_intent: dict[str, dict[str, float | int]] = {}
    for label_id in range(len(INTENTS)):
        tp = int(confusion[label_id, label_id])
        fp = int(confusion[:, label_id].sum()) - tp
        fn = int(confusion[label_id, :].sum()) - tp
        support = tp + fn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(support, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        if support:
            recall_values.append(recall)
        per_intent[INTENTS[label_id]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    entity_precision = true_positive / max(true_positive + false_positive, 1)
    entity_recall = true_positive / max(true_positive + false_negative, 1)
    per_slot = {}
    for name, counts in per_slot_counts.items():
        precision = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        recall = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        per_slot[name] = {
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "support_tokens": counts["tp"] + counts["fn"],
        }
    return {
        "intent_accuracy": intent_correct / max(intent_total, 1),
        "intent_macro_f1": sum(f1_values) / len(f1_values),
        "worst_intent_recall": min(recall_values, default=0.0),
        "per_intent": per_intent,
        "slot_token_accuracy": slot_correct / max(slot_total, 1),
        "slot_entity_f1": 2 * entity_precision * entity_recall / max(
            entity_precision + entity_recall, 1e-12
        ),
        "semantic_frame_exact_match": frame_correct / max(intent_total, 1),
        "end_to_end_command_accuracy": actionable_correct / max(actionable_total, 1),
        "slot_hallucination_rate": hallucinated_frames / max(no_slot_total, 1),
        "per_slot": per_slot,
    }


def _validation_logits(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            intent_logits, _ = model(
                batch["input_ids"].to(device), batch["mask"].to(device)
            )
            logits.append(intent_logits.cpu())
            labels.append(batch["intent"])
    return torch.cat(logits), torch.cat(labels)


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Choose a deterministic scalar temperature by validation NLL."""
    candidates = torch.linspace(0.25, 5.0, 476)
    losses = torch.stack(
        [nn.functional.cross_entropy(logits / value, labels) for value in candidates]
    )
    return float(candidates[int(losses.argmin())])


def _expected_calibration_error(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> float:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, predictions = probabilities.max(dim=-1)
    correct = predictions.eq(labels)
    error = torch.tensor(0.0)
    for lower in torch.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        members = (confidence > lower) & (confidence <= upper)
        if members.any():
            error += members.float().mean() * (
                confidence[members].mean() - correct[members].float().mean()
            ).abs()
    return float(error)


def _balanced_loader(
    examples: list[Example], tokenizer: CharTokenizer, batch_size: int
) -> DataLoader:
    counts = {intent: 0 for intent in INTENTS}
    for example in examples:
        counts[example.intent] += 1
    weights = [1.0 / counts[example.intent] for example in examples]
    sampler = WeightedRandomSampler(weights, num_samples=len(examples), replacement=True)
    return DataLoader(NLUDataset(examples, tokenizer), batch_size=batch_size, sampler=sampler)


def train(args: argparse.Namespace) -> dict[str, float]:
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    clean_train = build_examples("train", augmented=False, seed=args.seed)
    augmented_train = build_examples("train", augmented=True, seed=args.seed)
    validation = build_examples("validation", seed=args.seed)
    test = build_examples("test", seed=args.seed)
    tokenizer_class = WordTokenizer if args.architecture.startswith("word_") else CharTokenizer
    tokenizer = tokenizer_class.fit(iter_text(augmented_train), max_length=args.max_length)

    model_config = {"embedding_dim": args.embedding_dim}
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
    ).to(device)

    validation_loader = DataLoader(NLUDataset(validation, tokenizer), batch_size=args.batch_size)
    test_loader = DataLoader(NLUDataset(test, tokenizer), batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    intent_loss = nn.CrossEntropyLoss()
    # O is abundant even in balanced intent batches.  Lowering its weight
    # prevents a superficially high slot score from a model that predicts O
    # for every character.
    slot_weights = torch.tensor((0.2, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0), device=device)
    slot_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=slot_weights)

    for epoch in range(args.epochs):
        if args.method == "standard":
            epoch_examples = clean_train
        elif args.method == "augmented":
            epoch_examples = augmented_train
        else:  # curriculum: clean foundation, noisy variants in second half
            epoch_examples = clean_train if epoch < args.epochs // 2 else augmented_train
        loader = _balanced_loader(epoch_examples, tokenizer, args.batch_size)
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            intents = batch["intent"].to(device)
            slots = batch["slots"].to(device)
            intent_logits, slot_logits = model(ids, mask)
            loss = intent_loss(intent_logits, intents)
            loss += args.slot_loss_weight * slot_loss(
                slot_logits.reshape(-1, len(SLOT_LABELS)), slots.reshape(-1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    calibration_logits, calibration_labels = _validation_logits(
        model, validation_loader, device
    )
    temperature = _fit_temperature(calibration_logits, calibration_labels)

    metrics = {
        **{f"validation_{k}": v for k, v in _accuracy(model, validation_loader, device).items()},
        **{f"test_{k}": v for k, v in _accuracy(model, test_loader, device).items()},
        "train_examples": len(augmented_train if args.method != "standard" else clean_train),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "temperature": temperature,
        "validation_ece_before": _expected_calibration_error(
            calibration_logits, calibration_labels, 1.0
        ),
        "validation_ece_after": _expected_calibration_error(
            calibration_logits, calibration_labels, temperature
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "architecture": args.architecture,
            "tokenizer_type": "word" if args.architecture.startswith("word_") else "char",
            "method": args.method,
            "model_config": model_config,
            "model_state": model.cpu().state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "metrics": metrics,
            "seed": args.seed,
            "temperature": temperature,
        },
        output,
    )
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("char_cnn", "bigru", "word_bigru"), default="char_cnn")
    parser.add_argument("--method", choices=("standard", "augmented", "curriculum"), default="standard")
    parser.add_argument("--output", default="models/nlu_cnn_standard.pt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--slot-loss-weight", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
