"""Runtime inference for locally trained Jarvis NLU checkpoints."""
from __future__ import annotations

import re
from pathlib import Path

import torch

from .models import build_model
from .schema import INTENTS, SLOT_LABELS, NLUResult
from .tokenizer import CharTokenizer, WordTokenizer


class NLUPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        tokenizer_class = (
            WordTokenizer if checkpoint.get("tokenizer_type") == "word" else CharTokenizer
        )
        self.tokenizer = tokenizer_class.from_dict(checkpoint["tokenizer"])
        config = checkpoint["model_config"]
        self.model = build_model(
            checkpoint["architecture"],
            vocab_size=len(self.tokenizer.stoi),
            num_intents=len(INTENTS),
            num_slots=len(SLOT_LABELS),
            pad_id=self.tokenizer.pad_id,
            **config,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.temperature = max(float(checkpoint.get("temperature", 1.0)), 1e-6)

    @torch.inference_mode()
    def predict(self, text: str) -> NLUResult:
        ids, mask = self.tokenizer.encode(text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention = torch.tensor([mask], dtype=torch.bool, device=self.device)
        intent_logits, slot_logits = self.model(input_ids, attention)
        probabilities = torch.softmax(intent_logits[0] / self.temperature, dim=-1)
        confidence, intent_id = probabilities.max(dim=-1)
        slot_ids = slot_logits[0].argmax(dim=-1).tolist()
        slots = _decode_slots(text, slot_ids, self.tokenizer)
        slots = _normalise_slots(text, INTENTS[int(intent_id)], slots)
        return NLUResult(INTENTS[int(intent_id)], float(confidence), slots)


def _decode_slots(text: str, slot_ids: list[int], tokenizer) -> dict[str, str]:
    slots: dict[str, str] = {}
    current_label: str | None = None
    start = 0
    end = 0
    offsets = tokenizer.offsets(text)

    def finish() -> None:
        nonlocal current_label
        if current_label is not None and end > start:
            value = text[start:end].strip(" ,.:;!?")
            if value:
                slots[current_label] = value
        current_label = None

    for index, (token_start, token_end) in enumerate(offsets):
        tag = SLOT_LABELS[slot_ids[index]]
        if tag == "O":
            finish()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or label != current_label:
            finish()
            current_label = label
            start = token_start
        end = token_end
    finish()
    return slots


def _normalise_slots(text: str, intent: str, slots: dict[str, str]) -> dict[str, str]:
    if "duration" in slots:
        match = re.search(r"\d+", slots["duration"])
        if match:
            slots["minutes"] = match.group(0)
        slots.pop("duration", None)
    if intent == "set_reminder":
        # The first baseline is deliberately hybrid: the neural slot head is
        # trained and evaluated, while a constrained decoder guarantees that
        # the two parameters required by today's only reminder tool are not
        # truncated by a weak sequence tagger.  This is not presented as an
        # ML result; raw slot F1 remains in the training report.
        duration = re.search(
            r"(?:через|на|спустя|in)\s+(\d+)\s+(?:минут(?:у|ы)?|minutes?)\b",
            text,
            flags=re.IGNORECASE,
        )
        if duration:
            slots["minutes"] = duration.group(1)
            remainder = text[duration.end():].strip(" ,.:;!?-")
            remainder = re.sub(
                r"^(?:напомни(?:\s+мне)?|скажи\s+мне|о\s+том\s+чтобы|to)\s+",
                "",
                remainder,
                flags=re.IGNORECASE,
            ).strip(" ,.:;!?-")
            if remainder:
                slots["reminder_text"] = remainder
    elif intent == "open_application":
        # Never trust an unconstrained neural span for process launching. The
        # intent is learned, but an explicit imperative form is also required
        # before an application value is allowed to reach the launcher.
        slots.pop("application", None)
        match = re.search(
            r"(?:(?:открой|открыть|запусти|запустить|запустим|включи|open|launch)"
            r"(?:\s+мне)?(?:\s+приложение)?\s+|"
            r"мне\s+нужно\s+приложение\s+)(.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            application = re.sub(
                r"\s+(?:пожалуйста|джарвис)$", "", match.group(1), flags=re.IGNORECASE
            ).strip(" ,.:;!?-")
            if application:
                slots["application"] = application
    return slots
