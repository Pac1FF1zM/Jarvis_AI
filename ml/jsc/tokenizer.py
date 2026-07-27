"""Project-owned reversible character tokenizer for JSC sequence models."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = (PAD, BOS, EOS, UNK)


@dataclass(frozen=True)
class JSCCharTokenizer:
    """A deterministic character vocabulary fitted only on project data."""

    stoi: Mapping[str, int]

    def __post_init__(self) -> None:
        vocabulary = dict(self.stoi)
        if len(vocabulary) != len(set(vocabulary.values())):
            raise ValueError("tokenizer ids must be unique")
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("tokenizer ids must be contiguous from zero")
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            if vocabulary.get(token) != expected_id:
                raise ValueError(f"special token {token!r} must have id {expected_id}")
        object.__setattr__(self, "stoi", vocabulary)

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        *,
        min_frequency: int = 1,
    ) -> "JSCCharTokenizer":
        counts = Counter(character for text in texts for character in text)
        characters = sorted(
            character
            for character, count in counts.items()
            if count >= min_frequency and character not in SPECIAL_TOKENS
        )
        vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        vocabulary.update(
            {character: index + len(SPECIAL_TOKENS) for index, character in enumerate(characters)}
        )
        return cls(vocabulary)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def size(self) -> int:
        return len(self.stoi)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode(self, text: str, *, max_length: int) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("tokenizer input must be a non-empty string")
        required = len(text) + 2
        if required > max_length:
            raise ValueError(
                f"encoded sequence needs {required} tokens, limit is {max_length}; "
                "silent truncation is forbidden"
            )
        return [
            self.bos_id,
            *(self.stoi.get(character, self.unk_id) for character in text),
            self.eos_id,
        ]

    def decode(self, token_ids: Iterable[int]) -> str:
        inverse = {index: token for token, index in self.stoi.items()}
        characters: list[str] = []
        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id == self.eos_id:
                break
            if token_id in {self.pad_id, self.bos_id}:
                continue
            token = inverse.get(token_id, UNK)
            characters.append("�" if token == UNK else token)
        return "".join(characters)

    def to_dict(self) -> dict[str, object]:
        return {"type": "jsc_char_v1", "stoi": dict(self.stoi)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "JSCCharTokenizer":
        if set(raw) != {"type", "stoi"} or raw["type"] != "jsc_char_v1":
            raise ValueError("unsupported tokenizer payload")
        stoi = raw["stoi"]
        if not isinstance(stoi, Mapping):
            raise ValueError("tokenizer stoi must be an object")
        return cls({str(token): int(index) for token, index in stoi.items()})
