"""Sequence representation and dynamic batches for JSC semantic parsing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .data import JSCExample
from .jal import DialogueAct, dumps
from .structured_labels import parameter_label, parse_parameter_label
from .span_labels import find_argument_span
from core.russian_numbers import extract_russian_cardinals
from .tokenizer import JSCCharTokenizer


ACT_LABELS = tuple(act.value for act in DialogueAct)
ACT_TO_ID = {name: index for index, name in enumerate(ACT_LABELS)}
SOURCE_FORMAT_VERSION = 2
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
    lines: list[str] = []
    for turn in example.history:
        prefix = f"H_{turn.role.upper()}"
        normalized = normalize_utterance(turn.text)
        lines.append(f"{prefix}:{normalized}")
        numbers = extract_russian_cardinals(normalized)
        if numbers:
            lines.append(f"{prefix}_NUM:{','.join(map(str, numbers))}")
    if example.state is not None:
        lines.append(f"STATE:{dumps(example.state)}")
    normalized_user = normalize_utterance(example.text)
    lines.append(f"USER:{normalized_user}")
    numbers = extract_russian_cardinals(normalized_user)
    if numbers:
        lines.append(f"USER_NUM:{','.join(map(str, numbers))}")
    return "\n".join(lines)


def tokenizer_training_texts(examples: Iterable[JSCExample]) -> Iterable[str]:
    for example in examples:
        yield serialize_source(example)
        yield dumps(example.target)


@dataclass(frozen=True)
class SequenceLimits:
    source: int = 384
    target: int = 384

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
        source_text = serialize_source(example)
        source_ids = self.tokenizer.encode(
            source_text, max_length=self.limits.source
        )
        target_text = dumps(example.target)
        target_ids = self.tokenizer.encode(target_text, max_length=self.limits.target)
        return {
            "source_ids": source_ids,
            "source_text": source_text,
            "decoder_input_ids": target_ids[:-1],
            "labels": target_ids[1:],
            "act": ACT_TO_ID[example.target.act.value],
            "scenario_id": example.scenario_id,
            "category": example.category,
            "target_text": target_text,
            "tools": [step.tool for step in example.target.steps],
            "calls": list(example.target.steps),
        }


def make_collate_fn(
    pad_id: int,
    tool_to_id: dict[str, int] | None = None,
    parameter_to_id: dict[str, int] | None = None,
    span_slots: Sequence[str] | None = None,
    span_arguments_by_tool: Mapping[str, Sequence[str]] | None = None,
    *,
    max_steps: int = 8,
):
    def collate(rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        source_ids = _pad([row["source_ids"] for row in rows], pad_id)
        decoder_ids = _pad([row["decoder_input_ids"] for row in rows], pad_id)
        labels = _pad([row["labels"] for row in rows], -100)
        act_targets = torch.tensor([row["act"] for row in rows], dtype=torch.long)
        batch = {
            "source_ids": source_ids,
            "source_mask": source_ids.ne(pad_id),
            "decoder_input_ids": decoder_ids,
            "decoder_mask": decoder_ids.ne(pad_id),
            "labels": labels,
            "act": act_targets,
            "execution_allowed": act_targets.eq(ACT_TO_ID[DialogueAct.EXECUTE.value]).long(),
            "scenario_id": [str(row["scenario_id"]) for row in rows],
            "category": [str(row["category"]) for row in rows],
            "target_text": [str(row["target_text"]) for row in rows],
        }
        if tool_to_id is not None:
            tool_rows: list[list[int]] = []
            for row in rows:
                tools = list(row["tools"])
                if len(tools) > max_steps:
                    raise ValueError("tool sequence exceeds structured max_steps")
                try:
                    encoded = [tool_to_id[str(tool)] for tool in tools]
                except KeyError as exc:
                    raise ValueError(f"unknown structured tool label {exc.args[0]!r}") from exc
                tool_rows.append(encoded + [0] * (max_steps - len(encoded)))
            batch["step_count"] = torch.tensor(
                [len(row["tools"]) for row in rows], dtype=torch.long
            )
            batch["tool_ids"] = torch.tensor(tool_rows, dtype=torch.long)
            if parameter_to_id is not None:
                parameter_count = len(parameter_to_id)
                targets = torch.zeros(
                    len(rows), max_steps, parameter_count, dtype=torch.float32
                )
                applicable = torch.zeros_like(targets, dtype=torch.bool)
                labels_by_tool: dict[str, list[int]] = {}
                for label, label_id in parameter_to_id.items():
                    tool, _name, _value = parse_parameter_label(label)
                    labels_by_tool.setdefault(tool, []).append(label_id)
                for row_index, row in enumerate(rows):
                    calls = list(row["calls"])
                    for step_index, call in enumerate(calls):
                        for label_id in labels_by_tool.get(call.tool, ()):
                            applicable[row_index, step_index, label_id] = True
                        for name, value in call.arguments.items():
                            label = parameter_label(call.tool, name, value)
                            label_id = parameter_to_id.get(label)
                            if label_id is not None:
                                targets[row_index, step_index, label_id] = 1.0
                batch["parameter_targets"] = targets
                batch["parameter_mask"] = applicable
            if span_slots is not None:
                if span_arguments_by_tool is None:
                    raise ValueError("span slots require tool argument mapping")
                start_targets = torch.zeros(
                    len(rows), max_steps, len(span_slots), dtype=torch.long
                )
                end_targets = torch.zeros_like(start_targets)
                span_mask = torch.zeros_like(start_targets, dtype=torch.bool)
                for row_index, row in enumerate(rows):
                    source_text = str(row["source_text"])
                    for step_index, call in enumerate(list(row["calls"])):
                        applicable_slots = set(
                            span_arguments_by_tool.get(call.tool, ())
                        )
                        for slot_index, name in enumerate(span_slots):
                            if name not in applicable_slots:
                                continue
                            if name not in call.arguments:
                                span_mask[row_index, step_index, slot_index] = True
                                continue
                            span = find_argument_span(source_text, call, name)
                            if span is None:
                                continue
                            start, end = span
                            if end >= len(row["source_ids"]):
                                # The value was truncated from this model input.
                                continue
                            span_mask[row_index, step_index, slot_index] = True
                            start_targets[row_index, step_index, slot_index] = start
                            end_targets[row_index, step_index, slot_index] = end
                batch["span_start_targets"] = start_targets
                batch["span_end_targets"] = end_targets
                batch["span_mask"] = span_mask
        return batch

    return collate


def _pad(values: list[object], padding_value: int) -> torch.Tensor:
    tensors = [torch.tensor(value, dtype=torch.long) for value in values]
    return torch.nn.utils.rnn.pad_sequence(
        tensors,
        batch_first=True,
        padding_value=padding_value,
    )
