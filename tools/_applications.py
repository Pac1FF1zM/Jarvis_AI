"""Safe application allow-list shared by application tools."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ApplicationSpec:
    name: str
    display_name: str
    command: tuple[str, ...] | None = None
    url: str | None = None
    uri: str | None = None
    aliases: tuple[str, ...] = ()


APPLICATIONS = (
    ApplicationSpec(
        "calculator",
        "Калькулятор",
        ("calc.exe",),
        aliases=("калькулятор", "калькуляторы", "calc"),
    ),
    ApplicationSpec(
        "notepad",
        "Блокнот",
        ("notepad.exe",),
        aliases=(
            "блокнот", "блокноты", "блакнот", "блекнот", "notepad",
            "black note", "blacknote",
        ),
    ),
    ApplicationSpec(
        "explorer",
        "Проводник",
        ("explorer.exe",),
        aliases=("проводник", "файлы", "file explorer", "explorer"),
    ),
    ApplicationSpec(
        "paint",
        "Пейнт",
        uri="ms-paint:",
        aliases=("paint", "паинт", "пайнт", "пейнт", "пеинт", "пэйнт"),
    ),
    ApplicationSpec(
        "discord",
        "Дискорд",
        uri="discord://",
        aliases=(
            "discord", "дискорд", "дисорд", "дискод", "дискор", "дискорд",
        ),
    ),
    ApplicationSpec(
        "task_manager",
        "Диспетчер задач",
        ("taskmgr.exe",),
        aliases=("диспетчер задач", "task manager"),
    ),
    ApplicationSpec(
        "browser",
        "Браузер",
        # A normal HTTPS URL is handed to Windows' default browser.  Using
        # ``about:blank`` here made Windows look for an ``about:`` protocol
        # handler and offer the Microsoft Store instead.
        url="https://www.google.com/",
        aliases=("браузер", "browser", "интернет"),
    ),
)


def normalise_name(value: str) -> str:
    value = value.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip(" .,!?:;\"'")


def _compact_name(value: str) -> str:
    """Remove separators so Whisper's ``к алкулятор`` can match safely."""
    return re.sub(r"[^a-zа-я0-9]+", "", normalise_name(value))


def _edit_distance(left: str, right: str) -> int:
    """Small Levenshtein distance helper for the tiny fixed allow-list."""
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def resolve_application(value: str) -> ApplicationSpec | None:
    requested = normalise_name(value)
    requested_compact = _compact_name(requested)
    if not requested_compact:
        return None

    ranked: list[tuple[int, ApplicationSpec]] = []
    for spec in APPLICATIONS:
        candidates = (spec.name, spec.display_name, *spec.aliases)
        normalised = {normalise_name(candidate) for candidate in candidates}
        compact = {_compact_name(candidate) for candidate in candidates}
        if requested in normalised or requested_compact in compact:
            return spec
        if len(requested_compact) >= 4:
            distance = min(
                _edit_distance(requested_compact, candidate)
                for candidate in compact
            )
            ranked.append((distance, spec))

    # Fuzzy matching never creates a launch target: it can only select one of
    # the fixed entries above. Require a strong, unambiguous match so an
    # unrelated application name cannot accidentally launch something else.
    ranked.sort(key=lambda item: item[0])
    if not ranked or ranked[0][0] > 1:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def launch_application(spec: ApplicationSpec) -> int | None:
    """Launch one pre-approved target without invoking a command shell."""
    if spec.command is not None:
        process = subprocess.Popen(  # noqa: S603 - command comes only from allow-list
            list(spec.command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return process.pid
    if spec.url is not None:
        os.startfile(spec.url)  # type: ignore[attr-defined]  # Windows-only runtime
        return None
    if spec.uri is not None:
        os.startfile(spec.uri)  # type: ignore[attr-defined]  # Windows-only runtime
        return None
    raise RuntimeError(f"application {spec.name!r} has no launch target")
