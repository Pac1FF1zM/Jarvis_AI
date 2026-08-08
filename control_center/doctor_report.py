"""Pure parsing helpers for Control Center diagnostic reports."""
from __future__ import annotations

import json
from typing import Any


def parse_doctor_output(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "checks" in value:
            return value
    raise ValueError("Диагностика не вернула корректный JSON-отчёт")
