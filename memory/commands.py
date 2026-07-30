"""Deterministic Russian command grammar for explicit memory management."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryCommand:
    action: Literal["remember", "recall", "list", "forget", "clear"]
    value: str = ""


_PREFIX = r"(?:джарвис[,.]?\s+)?(?:пожалуйста[,.]?\s+)?"
_REMEMBER = re.compile(
    rf"^{_PREFIX}(?:запомни|запиши\s+в\s+память|сохрани\s+в\s+памяти)"
    r"(?:[,:]?\s+(?:то[, ]+)?(?:что\s+)?)?(.*)$",
    flags=re.IGNORECASE,
)
_FORGET = re.compile(
    rf"^{_PREFIX}(?:забудь|удали\s+из\s+памяти)"
    r"(?:[,:]?\s+(?:то[, ]+)?(?:что\s+)?)?(.*)$",
    flags=re.IGNORECASE,
)
_RECALL = re.compile(
    rf"^{_PREFIX}(?:что\s+ты\s+(?:помнишь|знаешь)\s+(?:про|об)\s+|"
    r"напомни[, ]+что\s+ты\s+знаешь\s+(?:про|об)\s+)(.+)$",
    flags=re.IGNORECASE,
)
_LIST = re.compile(
    rf"^{_PREFIX}(?:что\s+ты\s+(?:помнишь|запомнил|знаешь)(?:\s+обо\s+мне)?|"
    r"что\s+ты\s+обо\s+мне\s+(?:помнишь|знаешь)|"
    r"покажи\s+(?:свою\s+)?память|что\s+хранится\s+в\s+(?:твоей\s+)?памяти|"
    r"перечисли\s+(?:все\s+)?(?:факты|то[, ]+что\s+ты\s+помнишь))$",
    flags=re.IGNORECASE,
)
_DIRECT_RECALL = re.compile(
    rf"^{_PREFIX}(?:как\s+меня\s+зовут|како(?:й|е|я)\s+(?:у\s+меня|мо[йяе])\s+.+)$",
    flags=re.IGNORECASE,
)
_CLEAR_VALUES = {
    "все",
    "всё",
    "всю информацию",
    "все обо мне",
    "всё обо мне",
    "всю память",
}
_COMPOUND_SIDE_EFFECT = re.compile(
    r"(?:,|\s+и\s+|\s+(?:затем|потом)\s+)"
    r"(?:открой|запусти|включи|закрой|удали|поставь|напомни|найди)\b",
    flags=re.IGNORECASE,
)


def parse_memory_command(text: str) -> MemoryCommand | None:
    """Parse only explicit memory language; ordinary dialogue returns ``None``."""
    value = re.sub(r"\s+", " ", str(text)).strip(" \t\r\n.!?")
    if not value:
        return None
    if _LIST.fullmatch(value):
        return MemoryCommand("list")
    match = _REMEMBER.fullmatch(value)
    if match:
        fact = match.group(1).strip(" ,.:;!?")
        # Do not accidentally persist the tail of a compound side-effect
        # command as personal data. Compound memory plans are intentionally not
        # supported until they can be atomic.
        if _COMPOUND_SIDE_EFFECT.search(fact):
            return None
        return MemoryCommand("remember", fact)
    match = _FORGET.fullmatch(value)
    if match:
        target = match.group(1).strip(" ,.:;!?")
        if target.casefold().replace("ё", "е") in {
            item.casefold().replace("ё", "е") for item in _CLEAR_VALUES
        }:
            return MemoryCommand("clear")
        return MemoryCommand("forget", target)
    match = _RECALL.fullmatch(value)
    if match:
        return MemoryCommand("recall", match.group(1).strip(" ,.:;!?"))
    if _DIRECT_RECALL.fullmatch(value):
        return MemoryCommand("recall", value)
    return None
