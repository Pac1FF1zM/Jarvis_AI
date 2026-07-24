"""A tiny character tokenizer trained only on the Jarvis corpus."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from .data import Example
from .schema import SLOT_LABELS


PAD = "<pad>"
UNK = "<unk>"
IGNORE_INDEX = -100


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    max_length: int = 128

    @classmethod
    def fit(
        cls, texts: Iterable[str], *, max_length: int = 128, min_frequency: int = 1
    ) -> "CharTokenizer":
        counts = Counter(char.lower() for text in texts for char in text)
        chars = sorted(char for char, count in counts.items() if count >= min_frequency)
        return cls({PAD: 0, UNK: 1, **{char: i + 2 for i, char in enumerate(chars)}}, max_length)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        chars = list(text.lower())[: self.max_length]
        ids = [self.stoi.get(char, self.stoi[UNK]) for char in chars]
        mask = [1] * len(ids)
        padding = self.max_length - len(ids)
        return ids + [self.pad_id] * padding, mask + [0] * padding

    def extend(self, texts: Iterable[str]) -> int:
        """Append unseen local-corpus symbols without changing existing IDs."""
        unseen = sorted(
            {char.lower() for text in texts for char in text} - set(self.stoi)
        )
        for char in unseen:
            self.stoi[char] = len(self.stoi)
        return len(unseen)

    def encode_slots(self, example: Example) -> list[int]:
        label_to_id = {label: idx for idx, label in enumerate(SLOT_LABELS)}
        labels = [label_to_id["O"]] * min(len(example.text), self.max_length)
        for span in example.spans:
            if span.start >= self.max_length:
                continue
            end = min(span.end, self.max_length)
            labels[span.start] = label_to_id[f"B-{span.label}"]
            for index in range(span.start + 1, end):
                labels[index] = label_to_id[f"I-{span.label}"]
        return labels + [IGNORE_INDEX] * (self.max_length - len(labels))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(min(len(text), self.max_length))]

    def to_dict(self) -> dict:
        return {"stoi": self.stoi, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        return cls(stoi=dict(data["stoi"]), max_length=int(data["max_length"]))


_WORD_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass
class WordTokenizer:
    """Word vocabulary learned from the local corpus, with span offsets."""

    stoi: dict[str, int]
    max_length: int = 32

    @classmethod
    def fit(
        cls, texts: Iterable[str], *, max_length: int = 32, min_frequency: int = 1
    ) -> "WordTokenizer":
        counts = Counter(
            match.group(0).lower()
            for text in texts
            for match in _WORD_RE.finditer(text)
        )
        words = sorted(word for word, count in counts.items() if count >= min_frequency)
        return cls({PAD: 0, UNK: 1, **{word: i + 2 for i, word in enumerate(words)}}, max_length)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [match.span() for match in list(_WORD_RE.finditer(text))[: self.max_length]]

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        matches = list(_WORD_RE.finditer(text))[: self.max_length]
        ids = [self.stoi.get(match.group(0).lower(), self.stoi[UNK]) for match in matches]
        mask = [1] * len(ids)
        padding = self.max_length - len(ids)
        return ids + [self.pad_id] * padding, mask + [0] * padding

    def extend(self, texts: Iterable[str]) -> int:
        """Append unseen local-corpus tokens while preserving checkpoint IDs."""
        unseen = sorted(
            {
                match.group(0).lower()
                for text in texts
                for match in _WORD_RE.finditer(text)
            }
            - set(self.stoi)
        )
        for word in unseen:
            self.stoi[word] = len(self.stoi)
        return len(unseen)

    def encode_slots(self, example: Example) -> list[int]:
        label_to_id = {label: idx for idx, label in enumerate(SLOT_LABELS)}
        offsets = self.offsets(example.text)
        labels = [label_to_id["O"]] * len(offsets)
        for span in example.spans:
            matching = [
                index
                for index, (start, end) in enumerate(offsets)
                if start < span.end and end > span.start
            ]
            for position, index in enumerate(matching):
                prefix = "B" if position == 0 else "I"
                labels[index] = label_to_id[f"{prefix}-{span.label}"]
        return labels + [IGNORE_INDEX] * (self.max_length - len(labels))

    def to_dict(self) -> dict:
        return {"stoi": self.stoi, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(stoi=dict(data["stoi"]), max_length=int(data["max_length"]))
