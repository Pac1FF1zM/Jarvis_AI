"""Safe application allow-list shared by application tools."""
from __future__ import annotations

import os
import re
import subprocess
import webbrowser
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
        "calculator", "Калькулятор", ("calc.exe",), aliases=("калькулятор", "calc"),
    ),
    ApplicationSpec(
        "notepad", "Блокнот", ("notepad.exe",), aliases=("блокнот", "notepad"),
    ),
    ApplicationSpec(
        "explorer",
        "Проводник",
        ("explorer.exe",),
        aliases=("проводник", "файлы", "file explorer", "explorer"),
    ),
    ApplicationSpec(
        "paint", "Paint", uri="ms-paint:", aliases=("paint", "паинт", "пейнт"),
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
        url="about:blank",
        aliases=("браузер", "browser", "интернет"),
    ),
)


def normalise_name(value: str) -> str:
    value = value.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip(" .,!?:;\"'")


def resolve_application(value: str) -> ApplicationSpec | None:
    requested = normalise_name(value)
    for spec in APPLICATIONS:
        candidates = (spec.name, spec.display_name, *spec.aliases)
        if requested in {normalise_name(candidate) for candidate in candidates}:
            return spec
    return None


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
        if not webbrowser.open(spec.url, new=0, autoraise=True):
            raise RuntimeError("default browser refused the request")
        return None
    if spec.uri is not None:
        os.startfile(spec.uri)  # type: ignore[attr-defined]  # Windows-only runtime
        return None
    raise RuntimeError(f"application {spec.name!r} has no launch target")
