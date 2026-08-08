"""Small deterministic Russian cardinal-number normalizer for command slots."""
from __future__ import annotations

import re


_VALUES = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "две": 2, "два": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
    "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11,
    "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14,
    "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20,
    "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
}
_NUMBER_SEQUENCE = re.compile(
    r"\b(" + "|".join(sorted(_VALUES, key=len, reverse=True)) + r")"
    r"(?:\s+(" + "|".join(sorted(_VALUES, key=len, reverse=True)) + r")){0,3}"
    r"(?=\s+(?:секунд|секунду|секунды|минут|минуту|минуты|час|часа|часов|лет|год|года)\b)",
    flags=re.IGNORECASE,
)
_TOKEN = re.compile(r"\d+|[а-яё]+", re.IGNORECASE)


def extract_russian_cardinals(text: str) -> tuple[int, ...]:
    """Extract digit or Russian cardinal sequences up to 999, in spoken order."""
    if not isinstance(text, str):
        raise TypeError("number extraction expects text")
    result: list[int] = []
    current: int | None = None
    previous_kind: str | None = None

    def flush() -> None:
        nonlocal current, previous_kind
        if current is not None:
            result.append(current)
        current = None
        previous_kind = None

    for token in _TOKEN.findall(text.casefold().replace("ё", "е")):
        if token.isdigit():
            flush()
            result.append(int(token))
            continue
        number = _VALUES.get(token)
        if number is None:
            flush()
            continue
        kind = "hundred" if number >= 100 else "tens" if number >= 20 else "unit"
        if current is None:
            current = number
        elif kind == "unit" and (
            previous_kind in {"tens", "hundred"}
            and (previous_kind != "tens" or current % 10 == 0)
        ):
            current += number
        elif kind == "tens" and previous_kind == "hundred":
            current += number
        else:
            flush()
            current = number
        previous_kind = kind
    flush()
    return tuple(result)


def normalize_russian_numbers(text: str) -> str:
    """Replace bounded number-word sequences used by durations and age."""
    def replace(match: re.Match[str]) -> str:
        words = match.group(0).casefold().split()
        total = sum(_VALUES[word] for word in words)
        return str(total)

    return _NUMBER_SEQUENCE.sub(replace, text)
