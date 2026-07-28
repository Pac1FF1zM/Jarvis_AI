"""Deterministic safety router around the learned NLU manager.

The neural model remains authoritative for learned intents.  This module only
recognises explicit command grammar for OS side effects and converts compound
utterances into a serial action plan.  It never executes anything itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tools._applications import resolve_application


@dataclass(frozen=True)
class RoutedAction:
    intent: str
    slots: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.99

    def payload(self) -> dict[str, Any]:
        return {"intent": self.intent, "slots": dict(self.slots), "confidence": self.confidence}


_COMMAND_START = (
    r"(?:открой|открыть|запусти|включи|поставь|создай|скажи|покажи|"
    r"найди|поищи|загугли|переключи|перейди|сверни|разверни|закрой|"
    r"увеличь|уменьши|сделай|заблокируй|удали|переименуй|напомни)"
)
_COMPOUND_SEPARATOR = re.compile(
    rf"\s*(?:;\s*(?:(?:а\s+)?(?:затем|потом)\s+)?|,\s*(?={_COMMAND_START})|"
    rf"\s+(?:а\s+)?(?:затем|потом)\s+|\s+и\s+(?={_COMMAND_START}))",
    flags=re.IGNORECASE,
)


def split_compound_command(text: str) -> list[str]:
    parts = [part.strip(" ,") for part in _COMPOUND_SEPARATOR.split(text) if part.strip(" ,")]
    return parts if len(parts) > 1 else [text.strip()]


def route_explicit_command(
    text: str, *, previous_action: RoutedAction | None = None
) -> RoutedAction | None:
    value = re.sub(r"\s+", " ", text.casefold().replace("ё", "е")).strip(" .,!?:;")
    if not value:
        return None

    if value in {"подтверждаю", "подтвердить", "да подтверждаю", "выполняй", "давай"}:
        return RoutedAction("confirm")
    if value in {"отмена", "не подтверждаю", "не надо", "передумал", "передумала"}:
        return RoutedAction("decline")

    gesture_mode = re.fullmatch(
        r"(?:(включи|запусти|активируй|включить|запустить|активировать)|"
        r"(выключи|отключи|останови|выключить|отключить|остановить))\s+"
        r"(?:режим\s+)?жест(?:ов|ами)",
        value,
    )
    if gesture_mode:
        return RoutedAction("gesture_mode", {"enabled": bool(gesture_mode.group(1))})

    correction = re.match(
        r"^(?:(?:нет[, ]+)(?:(?:я\s+)?(?:имел|имела)\s+в\s+виду\s+)?|"
        r"(?:я\s+)?(?:имел|имела)\s+в\s+виду\s+)(.+)$",
        value,
    )
    if correction and previous_action is not None:
        replacement = correction.group(1).strip(" ,")
        if previous_action.intent == "open_application":
            application = resolve_application(replacement)
            if application is not None:
                return RoutedAction("open_application", {"application": application.name})
        if previous_action.intent == "browser_control" and previous_action.slots.get("action") == "search":
            return RoutedAction("browser_control", {"action": "search", "query": replacement})

    if value in {"скажи время", "текущее время"}:
        return RoutedAction("get_current_time")
    if re.fullmatch(r"(?:покажи|назови|перечисли|какие) (?:доступные |установленные )?(?:приложения|программы)(?: ты (?:можешь )?открыть)?", value):
        return RoutedAction("list_applications")

    search = re.match(r"^(?:найди|поищи|загугли)(?:\s+в\s+интернете|\s+в\s+сети)?\s+(.+)$", value)
    if search and not re.match(r"^(?:файл|папк)", search.group(1)):
        return RoutedAction("browser_control", {"action": "search", "query": search.group(1)})
    site = re.match(r"^(?:открой|перейди\s+на)\s+(?:сайт\s+)?((?:https?://)?[a-z0-9а-я.-]+\.[a-zа-я]{2,}(?:/\S*)?)$", value)
    if site:
        return RoutedAction("browser_control", {"action": "open_site", "url": site.group(1)})

    browser_patterns = (
        (r"^(?:открой|создай) (?:новую )?вкладку$", "new_tab"),
        (r"^закрой (?:текущую )?вкладку$", "close_tab"),
        (r"^(?:верни|восстанови) (?:закрытую |последнюю )?вкладку$", "reopen_tab"),
        (r"^(?:следующая вкладка|перейди на следующую вкладку)$", "next_tab"),
        (r"^(?:предыдущая вкладка|перейди на предыдущую вкладку)$", "previous_tab"),
    )
    for pattern, action in browser_patterns:
        if re.match(pattern, value):
            return RoutedAction("browser_control", {"action": action})

    system_patterns = (
        (r"^(?:(?:увеличь|прибавь) громкость|сделай громче)$", "volume_up"),
        (r"^(?:(?:уменьши|убавь) громкость|сделай тише)$", "volume_down"),
        (r"^(?:выключи|включи|переключи) звук$", "volume_mute"),
        (r"^(?:пауза|продолжи|включи музыку|останови музыку|воспроизведение)$", "media_play_pause"),
        (r"^(?:следующий трек|следующая песня)$", "media_next"),
        (r"^(?:предыдущий трек|предыдущая песня)$", "media_previous"),
        (r"^(?:заблокируй|блокируй) (?:компьютер|пк)$", "lock"),
    )
    for pattern, action in system_patterns:
        if re.match(pattern, value):
            return RoutedAction("system_control", {"action": action})

    setting_match = re.match(r"^открой настройки(?: (экрана|звука|bluetooth|блютуз|сети|приложений|обновления|персонализации|приватности|микрофона))?$", value)
    if setting_match:
        mapping = {None: "settings", "экрана": "display", "звука": "sound", "bluetooth": "bluetooth", "блютуз": "bluetooth", "сети": "network", "приложений": "applications", "обновления": "update", "персонализации": "personalization", "приватности": "privacy", "микрофона": "microphone"}
        return RoutedAction("system_control", {"action": "open_settings", "setting": mapping[setting_match.group(1)]})

    if re.fullmatch(r"(?:покажи|перечисли) (?:открытые )?окна", value):
        return RoutedAction("window_control", {"action": "list"})
    window_match = re.match(r"^(переключись на|перейди в|сверни|разверни|восстанови|закрой) (?:окно )?(.+)$", value)
    if window_match:
        actions = {"переключись на": "switch", "перейди в": "switch", "сверни": "minimize", "разверни": "maximize", "восстанови": "restore", "закрой": "close"}
        return RoutedAction("window_control", {"action": actions[window_match.group(1)], "window": window_match.group(2)})

    file_search = re.match(r"^(?:найди|поищи) (?:файл|папку)\s+(.+)$", value)
    if file_search:
        return RoutedAction("file_control", {"action": "find", "query": file_search.group(1)})
    file_patterns = (
        (r"^(?:покажи содержимое|что в папке)\s+(.+)$", "list", "path"),
        (r"^открой (?:файл|папку)\s+(.+)$", "open", "path"),
        (r"^покажи (?:файл|папку)\s+(.+) в проводнике$", "reveal", "path"),
        (r"^создай папку\s+(.+)$", "create_folder", "path"),
        (r"^удали (?:файл|папку)\s+(.+)$", "delete", "path"),
    )
    for pattern, action, slot in file_patterns:
        match = re.match(pattern, value)
        if match:
            return RoutedAction("file_control", {"action": action, slot: match.group(1)})
    rename = re.match(r"^переименуй (?:файл|папку)\s+(.+)\s+в\s+([^\\/]+)$", value)
    if rename:
        return RoutedAction("file_control", {"action": "rename", "path": rename.group(1), "new_name": rename.group(2)})

    open_match = re.match(r"^(?:открой|запусти|включи)(?: мне)?(?: приложение| программу)?\s+(.+)$", value)
    if open_match:
        application = resolve_application(open_match.group(1))
        if application is not None:
            return RoutedAction("open_application", {"application": application.name})
    return None
