"""Runtime inference for locally trained Jarvis NLU checkpoints."""
from __future__ import annotations

import re
from pathlib import Path

import torch

from .models import build_model
from .schema import INTENTS, INTENT_SLOTS, SLOT_LABELS, NLUResult
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
    # Neural spans remain the general signal.  For a complete command grammar
    # that Jarvis explicitly supports, deterministic parsing validates the
    # tool arguments; raw neural quality is scored separately during training.
    slots = dict(slots)
    if "duration" in slots:
        match = re.search(r"\d+", slots["duration"])
        if match:
            slots["minutes"] = match.group(0)
        slots.pop("duration", None)
    if intent == "set_reminder":
        duration = _relative_duration(text)
        if duration is not None:
            slots["minutes"] = duration.group(1)
        if duration is not None:
            remainder = text[duration.end():].strip(" ,.:;!?-")
            remainder = re.sub(
                r"^(?:(?:и\s+)?напомни(?:\s+мне)?|"
                r"сообщи(?:\s+мне)?(?:\s+что\s+нужно)?|"
                r"скажи(?:\s+мне)?|что\s+пора|"
                r"не\s+забудь\s+сказать|вспомнить|"
                r"напоминани[ея]\s+чтобы|о\s+том\s+чтобы|to)\s+",
                "",
                remainder,
                flags=re.IGNORECASE,
            ).strip(" ,.:;!?-")
            if remainder:
                slots["reminder_text"] = remainder
    elif intent == "open_application":
        # The launcher independently resolves this value through its strict
        # allow-list.  Keeping the neural span here is therefore both safer
        # than arbitrary process execution and more accurate than discarding it.
        match = _application_fallback(text)
        if match is not None:
            application = re.sub(
                r"\s+(?:пожалуйста|джарвис)$", "", match.group(1), flags=re.IGNORECASE
            ).strip(" ,.:;!?-")
            if application:
                slots["application"] = application
    # A slot from the wrong intent must never become a tool argument.  The raw
    # slot head is still measured during training, while runtime uses this
    # deterministic contract boundary as a final safety net.
    allowed = INTENT_SLOTS[intent]
    return {name: value for name, value in slots.items() if name in allowed}


def _relative_duration(text: str) -> re.Match[str] | None:
    patterns = (
        r"(?:через|на|спустя|in)\s+(\d+)\s+(?:минут(?:у|ы)?|minutes?)\b",
        r"(?:отсчитай|когда\s+пройд[её]т)\s+(\d+)\s+(?:минут(?:у|ы)?|minutes?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match
    return None


def _application_fallback(text: str) -> re.Match[str] | None:
    patterns = (
        r"(?:(?:открой(?:-ка)?|открыть|запусти|запустить|запустим|включи|"
        r"включим|open|launch)(?:\s+мне|\s+для\s+меня)?"
        r"(?:\s+приложение|\s+программу)?\s+)(.+)$",
        r"(?:я\s+хочу\s+чтобы\s+ты\s+открыл|будь\s+добр\s+открой|"
        r"можно\s+запустить|пора\s+открыть|прошу\s+запустить|"
        r"мне\s+сейчас\s+нужен|мне\s+нужно\s+приложение)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return match
    return None
