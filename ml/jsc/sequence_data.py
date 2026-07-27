"""Sequence representation and dynamic batches for JSC semantic parsing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .data import JSCExample
from .jal import DialogueAct, dumps
from .tokenizer import JSCCharTokenizer


ACT_LABELS = tuple(act.value for act in DialogueAct)
ACT_TO_ID = {name: index for index, name in enumerate(ACT_LABELS)}
_UTTERANCE_TRANSLATION = str.maketrans(
    {
        "ё": "е",
        "№": " номер ",
        ";": ",",
        "-": " ",
        "–": " ",
        "—": " ",
    }
)


def normalize_utterance(value: str) -> str:
    """Apply a split-independent normalization suitable for STT-like text."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("utterance must be a non-empty string")
    normalized = value.casefold().translate(_UTTERANCE_TRANSLATION)
    return " ".join(normalized.split())


def serialize_source(example: JSCExample) -> str:
    """Create an unambiguous model input from history, state and current text."""
    lines = [
        f"H_{turn.role.upper()}:{normalize_utterance(turn.text)}"
        for turn in example.history
    ]
    if example.state is not None:
        lines.append(f"STATE:{dumps(example.state)}")
    lines.append(f"USER:{normalize_utterance(example.text)}")
    return "\n".join(lines)


def tokenizer_training_texts(examples: Iterable[JSCExample]) -> Iterable[str]:
    for example in examples:
        yield serialize_source(example)
        yield dumps(example.target)


@dataclass(frozen=True)
class SequenceLimits:
    source: int = 384
    target: int = 256

    def __post_init__(self) -> None:
        if self.source < 8 or self.target < 8:
            raise ValueError("sequence limits are implausibly small")


class JSCSequenceDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[JSCExample],
        tokenizer: JSCCharTokenizer,
        limits: SequenceLimits,
    ) -> None:
        if not examples:
            raise ValueError("JSC sequence dataset cannot be empty")
        self.examples = tuple(examples)
        self.tokenizer = tokenizer
        self.limits = limits

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        source_ids = self.tokenizer.encode(
            serialize_source(example), max_length=self.limits.source
        )
        target_text = dumps(example.target)
        target_ids = self.tokenizer.encode(target_text, max_length=self.limits.target)
        return {
            "source_ids": source_ids,
            "decoder_input_ids": target_ids[:-1],
            "labels": target_ids[1:],
            "act": ACT_TO_ID[example.target.act.value],
            "scenario_id": example.scenario_id,
            "category": example.category,
            "target_text": target_text,
        }


def make_collate_fn(pad_id: int):
    def collate(rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        source_ids = _pad([row["source_ids"] for row in rows], pad_id)
        decoder_ids = _pad([row["decoder_input_ids"] for row in rows], pad_id)
        labels = _pad([row["labels"] for row in rows], -100)
        return {
            "source_ids": source_ids,
            "source_mask": source_ids.ne(pad_id),
            "decoder_input_ids": decoder_ids,
            "decoder_mask": decoder_ids.ne(pad_id),
            "labels": labels,
            "act": torch.tensor([row["act"] for row in rows], dtype=torch.long),
            "scenario_id": [str(row["scenario_id"]) for row in rows],
            "category": [str(row["category"]) for row in rows],
            "target_text": [str(row["target_text"]) for row in rows],
        }

    return collate


def _pad(values: list[object], padding_value: int) -> torch.Tensor:
    tensors = [torch.tensor(value, dtype=torch.long) for value in values]
    return torch.nn.utils.rnn.pad_sequence(
        tensors,
        batch_first=True,
        padding_value=padding_value,
    )
