"""Conservative extraction of important profile facts from Russian dialogue."""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.russian_numbers import normalize_russian_numbers


@dataclass(frozen=True)
class PersonalFact:
    category: str
    text: str


_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "name",
        re.compile(r"\b(?:меня зовут|мое имя|моё имя)\s+([а-яa-z][а-яa-z-]{1,39})\b", re.I),
        "Пользователя зовут {0}",
    ),
    (
        "age",
        re.compile(r"\bмне\s+(\d{1,3})\s+(?:лет|года|год)\b", re.I),
        "Возраст пользователя: {0}",
    ),
    (
        "study",
        re.compile(r"\bя\s+учусь\s+(.{3,140})$", re.I),
        "Пользователь учится {0}",
    ),
    (
        "work",
        re.compile(r"\bя\s+работаю\s+(.{3,140})$", re.I),
        "Пользователь работает {0}",
    ),
    (
        "goal",
        re.compile(r"\b(?:моя цель|моей целью является)\s*[:—-]?\s*(.{3,180})$", re.I),
        "Цель пользователя: {0}",
    ),
    (
        "city",
        re.compile(r"\bя\s+живу\s+(.{3,100})$", re.I),
        "Пользователь живёт {0}",
    ),
)


def extract_personal_facts(text: str) -> list[PersonalFact]:
    """Return only explicit identity/life facts; ordinary dialogue yields none."""
    value = normalize_russian_numbers(
        re.sub(r"\s+", " ", str(text)).strip(" \t\r\n.!?")
    )
    if not value or len(value) > 500:
        return []
    output: list[PersonalFact] = []
    for category, pattern, template in _PATTERNS:
        match = pattern.search(value)
        if match is None:
            continue
        captured = match.group(1).strip(" ,.:;!?—-")
        if category == "age":
            age = int(captured)
            if not 1 <= age <= 120:
                continue
            captured = str(age)
        elif category == "name":
            captured = captured[:1].upper() + captured[1:]
        output.append(PersonalFact(category, template.format(captured)))
    return output
