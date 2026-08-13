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
from memory.workspaces import canonical_workspace_name
from core.russian_numbers import normalize_russian_numbers


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
    r"увеличь|уменьши|сделай|размести|сохрани|заверши|заблокируй|удали|переименуй|напомни)"
)
_COMPOUND_SEPARATOR = re.compile(
    rf"\s*(?:;\s*(?:(?:(?:а|и)\s+)?(?:затем|потом|после\s+этого|заодно)\s+)?|"
    rf",\s*(?:(?:заодно|после\s+этого)\s+|(?={_COMMAND_START}))|"
    rf"\s+(?:(?:а|и)\s+)?(?:затем|потом|после\s+этого|заодно)\s+|"
    rf"\s+и\s+(?={_COMMAND_START}))",
    flags=re.IGNORECASE,
)
_RAW_COMMAND_BOUNDARY = re.compile(
    r"\s+(?=(?:открой|запусти|включи|поставь|создай|покажи|найди|поищи|"
    r"загугли|переключи|перейди|сверни|разверни|закрой|увеличь|уменьши|"
    r"заблокируй|удали|переименуй|напомни|активируй|останови|размести|"
    r"сохрани|заверши)\b)",
    flags=re.IGNORECASE,
)


def split_compound_command(text: str) -> list[str]:
    parts = [part.strip(" ,") for part in _COMPOUND_SEPARATOR.split(text) if part.strip(" ,")]
    if len(parts) > 1:
        flattened = [
            item.strip(" ,")
            for part in parts
            for item in _RAW_COMMAND_BOUNDARY.split(part)
            if item.strip(" ,")
        ]
        return [item for part in flattened for item in _split_coordinated_applications(part)]
    # Speech recognizers can return a flat string without punctuation.
    # Finite imperative verbs are reliable action boundaries; infinitives are
    # deliberately excluded so "напомни открыть дверь" remains one reminder.
    raw = [part.strip(" ,") for part in _RAW_COMMAND_BOUNDARY.split(text) if part.strip(" ,")]
    if (
        len(raw) >= 2
        and re.fullmatch(r"через\s+\d+\s+минут(?:у|ы)?", raw[0], flags=re.IGNORECASE)
        and raw[1].casefold().startswith("напомни")
    ):
        raw = [raw[0] + " " + raw[1], *raw[2:]]
    parts = raw if len(raw) > 1 else [text.strip()]
    return [item for part in parts for item in _split_coordinated_applications(part)]


def _split_coordinated_applications(text: str) -> list[str]:
    """Expand one explicit app verb over an allow-listed enumeration.

    This runs only after the utterance has ended.  Every object must resolve
    independently; a partly unknown list is preserved as one ambiguous phrase
    instead of guessing or executing its known prefix.
    """
    match = re.fullmatch(
        r"(?P<prefix>(?:(?:джарвис|пожалуйста)\s+)*"
        r"(?:открой|запусти|включи|закрой|сверни|разверни|восстанови|"
        r"переключись\s+на|перейди\s+в)"
        r"(?:\s+(?:приложение|программу|окно))?)\s+(?P<objects>.+)",
        text.strip(" ,"),
        flags=re.IGNORECASE,
    )
    if match is None:
        return [text]
    objects = [value.strip(" ,") for value in re.split(r"\s+и\s+|,\s*", match.group("objects"))]
    if len(objects) < 2 or any(not value for value in objects):
        return [text]
    if any(resolve_application(value) is None for value in objects):
        return [text]
    prefix = match.group("prefix")
    return [f"{prefix} {value}" for value in objects]


def route_explicit_command(
    text: str, *, previous_action: RoutedAction | None = None
) -> RoutedAction | None:
    value = normalize_russian_numbers(
        re.sub(r"\s+", " ", text.casefold().replace("ё", "е"))
    ).strip(" .,!?:;")
    if not value:
        return None

    if value in {"подтверждаю", "подтвердить", "да подтверждаю", "выполняй", "давай"}:
        return RoutedAction("confirm")
    if value in {"отмена", "не подтверждаю", "не надо", "передумал", "передумала"}:
        return RoutedAction("decline")
    if value in {
        "отмени последнее действие",
        "отменить последнее действие",
        "верни последнее действие",
        "сделай как было",
    }:
        return RoutedAction("undo")
    if value in {"jarvis", "джарвис", "железяка проснись", "доброе утро"}:
        return RoutedAction("wake_greeting")
    value = re.sub(
        r"^(?:(?:джарвис|пожалуйста|сначала|затем|потом|после этого|заодно)\s+)+",
        "",
        value,
    )
    value = re.sub(r"\s+(?:пожалуйста|если можно)$", "", value)
    if re.fullmatch(
        r"(?:(?:полностью )?(?:очисти|форматируй|сотри) (?:весь )?(?:диск|системный диск)|"
        r"удали (?:все|всё) (?:файлы|данные)|уничтожь (?:все|всё) (?:файлы|данные))",
        value,
    ):
        return RoutedAction("unsupported_command")
    if re.match(r"^не\s+(?:закрывай|открывай|запускай|удаляй|выключай|включай)\b", value):
        return RoutedAction("negated_command")
    if value in {"закрой его", "закрой ее", "закрой её", "закрой это приложение"}:
        if previous_action is not None and previous_action.intent == "open_application":
            application = previous_action.slots.get("application")
            if application:
                return RoutedAction("window_control", {"action": "close", "window": application})
    if value in {"покажи подешевле", "найди подешевле", "а подешевле"}:
        if previous_action is not None and previous_action.intent == "browser_control":
            query = str(previous_action.slots.get("query", "")).strip()
            if query:
                return RoutedAction("browser_control", {"action": "search", "query": query + " подешевле"})

    gesture_target = (
        r"(?:режим\s+жестов|жестовый\s+режим|управление\s+(?:жестами|руками)|"
        r"распознавание\s+жестов|жесты)"
    )
    gesture_patterns = (
        (
            rf"^(?:(?:включи|запусти|активируй|задействуй|начни|открой)\s+{gesture_target}|"
            rf"перейди\s+в\s+{gesture_target}|начни\s+распознавать\s+жесты)$",
            "enable",
        ),
        (
            rf"^(?:(?:выключи|отключи|останови|заверши|закрой|прекрати|временно\s+останови)\s+{gesture_target}|"
            rf"выйди\s+из\s+{gesture_target}|хватит\s+распознавать\s+жесты)$",
            "disable",
        ),
        (
            rf"^(?:(?:поставь|переведи)\s+{gesture_target}\s+на\s+паузу|"
            rf"приостанови\s+{gesture_target})$",
            "pause",
        ),
        (
            rf"^(?:(?:продолжи|возобнови)\s+{gesture_target}|"
            rf"сними\s+{gesture_target}\s+с\s+паузы)$",
            "resume",
        ),
        (
            rf"^(?:(?:работает|включен|активен|активна)\s+ли\s+{gesture_target}|"
            rf"проверь\s+(?:работает|включен|активен)\s+ли\s+{gesture_target}|"
            rf"{gesture_target}\s+(?:сейчас\s+)?(?:работает|работают|включен|включены|активен|активны)|"
            r"(?:включена|активна)\s+ли\s+камера\s+жестов)$",
            "status",
        ),
    )
    for pattern, action in gesture_patterns:
        if re.fullmatch(pattern, value):
            slots: dict[str, Any] = {"action": action}
            if action in {"enable", "disable"}:
                slots["enabled"] = action == "enable"
            return RoutedAction("gesture_mode", slots)

    if re.fullmatch(
        r"(?:покажи|перечисли|какие есть|какие) (?:мои )?(?:рабочие пространства|рабочие режимы|режимы)(?: у меня есть)?",
        value,
    ):
        return RoutedAction("workspace_control", {"action": "list"})
    workspace_capture = re.match(
        r"^сохрани (?:это |текущее )?(?:расположение|рабочее пространство) "
        r"(?:как|под названием) (.+)$",
        value,
    )
    if workspace_capture:
        return RoutedAction(
            "workspace_control",
            {
                "action": "capture",
                "workspace": canonical_workspace_name(workspace_capture.group(1)),
            },
        )
    workspace_exit = re.match(
        r"^(?:выйди из|заверши|закрой) (?:текущий )?"
        r"(?:режим|режима|рабочее пространство|рабочего пространства)(?: .+)?$",
        value,
    )
    if workspace_exit:
        return RoutedAction("workspace_control", {"action": "finish"})
    workspace_launch = re.match(
        r"^(?:запусти|включи|активируй|открой|перейди в) "
        r"(?:(?:рабочее пространство|пространство|окружение|режим) (.+)|(.+?) режим)$",
        value,
    )
    if workspace_launch:
        name = workspace_launch.group(1) or workspace_launch.group(2)
        if name:
            return RoutedAction(
                "workspace_control",
                {"action": "launch", "workspace": canonical_workspace_name(name)},
            )

    explicit_correction = re.match(
        r"^(?:нет[, ]+)?не\s+(.+?)[, ]+(?:а\s+)?(?:открой|запусти|включи)\s+(.+)$",
        value,
    )
    if explicit_correction:
        previous = resolve_application(explicit_correction.group(1))
        replacement = resolve_application(explicit_correction.group(2))
        if replacement is not None:
            slots = {"application": replacement.name}
            if previous is not None:
                slots["correction_from"] = previous.name
            return RoutedAction("open_application", slots)

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
    if re.fullmatch(
        r"(?:скажи|назови|озвучь|покажи|проверь|сообщи|уточни)(?: мне)? "
        r"(?:(?:точное|текущее|системное) )?(?:время|час|показания часов)",
        value,
    ):
        return RoutedAction("get_current_time")
    if re.fullmatch(r"(?:покажи|назови|перечисли|какие) (?:доступные |установленные )?(?:приложения|программы)(?: ты (?:можешь )?открыть)?", value):
        return RoutedAction("list_applications")

    if re.fullmatch(
        r"(?:покажи|перечисли|назови|какие|список) "
        r"(?:(?:у меня|мои|активные|предстоящие) )?напоминани(?:я|й)",
        value,
    ):
        return RoutedAction("list_reminders")
    cancel_reminder = re.fullmatch(
        r"(?:отмени|удали|сними|убери|отключи) напоминание"
        r"(?: (?:номер|№|под номером|с номером))? (\d+)",
        value,
    )
    if cancel_reminder:
        return RoutedAction(
            "cancel_reminder", {"reminder_id": int(cancel_reminder.group(1))}
        )
    relative_reminder_patterns = (
        r"^напомни(?: мне)? через (\d+) минут(?:у|ы)? (.+)$",
        r"^через (\d+) минут(?:у|ы)? напомни(?: мне)? (.+)$",
    )
    for pattern in relative_reminder_patterns:
        relative = re.match(pattern, value)
        if relative:
            message = re.sub(
                r"^(?:о том,? чтобы|о том что|про|что|о)\s+", "", relative.group(2)
            ).strip(" ,.:;!?-")
            if message:
                return RoutedAction(
                    "set_reminder",
                    {"minutes": int(relative.group(1)), "reminder_text": message},
                )
    absolute_reminder_patterns = (
        r"^(?:напомни(?: мне)?|поставь напоминание) "
        r"(?:(сегодня|завтра) )?(?:в|на) (\d{1,2}[.:]\d{2}) (.+)$",
        r"^(?:(сегодня|завтра) )?в (\d{1,2}[.:]\d{2}) напомни(?: мне)? (.+)$",
    )
    for pattern in absolute_reminder_patterns:
        absolute = re.match(pattern, value)
        if absolute:
            message = re.sub(
                r"^(?:о том,? чтобы|о том что|про|что|о)\s+", "", absolute.group(3)
            ).strip(" ,.:;!?-")
            if message:
                slots: dict[str, Any] = {
                    "clock_time": absolute.group(2).replace(".", ":"),
                    "reminder_text": message,
                }
                if absolute.group(1):
                    slots["day"] = absolute.group(1)
                return RoutedAction("set_reminder", slots)

    incomplete_relative = re.match(
        r"^напомни(?:\s+мне)?\s+через\s+(\d+)\s+минут(?:у|ы)?$", value
    )
    if incomplete_relative:
        return RoutedAction("set_reminder", {"minutes": incomplete_relative.group(1)})
    incomplete_reminder = re.match(r"^напомни(?:\s+мне)?\s+(?:о|про|что)?\s*(.+)$", value)
    if incomplete_reminder and not re.search(r"\b(?:через|сегодня|завтра|\d{1,2}:\d{2})\b", value):
        return RoutedAction("set_reminder", {"reminder_text": incomplete_reminder.group(1)})

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
    if re.fullmatch(r"(?:размести|поставь) два окна рядом", value):
        return RoutedAction(
            "window_control", {"action": "arrange_two", "layout": "two_columns"}
        )
    layout_match = re.match(
        r"^(?:размести|расположи) окна (?:по схеме |как )?"
        r"(три столбца|четыре окна|сетка|главное слева|главное справа)$",
        value,
    )
    if layout_match:
        layouts = {
            "три столбца": "three_columns",
            "четыре окна": "grid_4",
            "сетка": "grid_4",
            "главное слева": "main_left",
            "главное справа": "main_right",
        }
        return RoutedAction(
            "window_control",
            {"action": "layout", "layout": layouts[layout_match.group(1)]},
        )
    position_match = re.match(
        r"^(?:(?:перемести|поставь) (?:окно )?(.+?) |(.+?) )"
        r"(влево|налево|вправо|направо|по центру|в левый верхний угол|"
        r"в правый верхний угол|в левый нижний угол|в правый нижний угол)$",
        value,
    )
    if position_match:
        placements = {
            "влево": "left",
            "налево": "left",
            "вправо": "right",
            "направо": "right",
            "по центру": "center",
            "в левый верхний угол": "top_left",
            "в правый верхний угол": "top_right",
            "в левый нижний угол": "bottom_left",
            "в правый нижний угол": "bottom_right",
        }
        return RoutedAction(
            "window_control",
            {
                "action": "move",
                "window": (position_match.group(1) or position_match.group(2)).strip(),
                "placement": placements[position_match.group(3)],
            },
        )
    relative_active = re.match(
        r"^(?:перемести|сдвинь)(?: окно)? (?:немного |чуть )?"
        r"(левее|правее|выше|ниже)$",
        value,
    )
    relative_move = None if relative_active else re.match(
        r"^(?:перемести|сдвинь) (?:окно )?(.+?) (?:немного |чуть )?"
        r"(левее|правее|выше|ниже)$",
        value,
    )
    if relative_active:
        directions = {
            "левее": "slightly_left",
            "правее": "slightly_right",
            "выше": "slightly_up",
            "ниже": "slightly_down",
        }
        return RoutedAction(
            "window_control",
            {"action": "move", "placement": directions[relative_active.group(1)]},
        )
    if relative_move:
        directions = {
            "левее": "slightly_left",
            "правее": "slightly_right",
            "выше": "slightly_up",
            "ниже": "slightly_down",
        }
        slots: dict[str, Any] = {
            "action": "move",
            "placement": directions[relative_move.group(2)],
        }
        if relative_move.group(1):
            slots["window"] = relative_move.group(1).strip()
        return RoutedAction("window_control", slots)
    monitor_move = re.match(
        r"^перемести (?:окно )?(.+?) на (первый|второй|третий|четвертый) монитор$",
        value,
    )
    if monitor_move:
        numbers = {"первый": 1, "второй": 2, "третий": 3, "четвертый": 4}
        return RoutedAction(
            "window_control",
            {
                "action": "move",
                "window": monitor_move.group(1),
                "placement": "monitor",
                "monitor": numbers[monitor_move.group(2)],
            },
        )
    resize_match = re.match(
        r"^(?:сделай|увеличь|уменьши) (?:(?:окно )?(.+?) )?"
        r"(шире|уже|выше|ниже|больше|меньше)$",
        value,
    )
    if resize_match:
        directions = {
            "шире": "wider",
            "уже": "narrower",
            "выше": "taller",
            "ниже": "shorter",
            "больше": "larger",
            "меньше": "smaller",
        }
        slots = {"action": "resize_relative", "direction": directions[resize_match.group(2)]}
        if resize_match.group(1):
            slots["window"] = resize_match.group(1).strip()
        return RoutedAction("window_control", slots)
    topmost_match = re.match(
        r"^(?:закрепи|оставь) (?:окно )?(.+?) поверх (?:всех|остальных окон)$",
        value,
    )
    if topmost_match:
        return RoutedAction(
            "window_control",
            {"action": "topmost", "window": topmost_match.group(1)},
        )
    not_topmost = re.match(
        r"^убери (?:окно )?(.+?) из режима поверх (?:всех|остальных окон)$",
        value,
    )
    if not_topmost:
        return RoutedAction(
            "window_control",
            {"action": "not_topmost", "window": not_topmost.group(1)},
        )
    if value in {"сверни все окна", "покажи рабочий стол"}:
        return RoutedAction(
            "window_control",
            {"action": "minimize_all" if value.startswith("сверни") else "show_desktop"},
        )
    window_match = re.match(r"^(переключись на|перейди в|сверни|разверни|восстанови|закрой) (?:окно )?(.+)$", value)
    if window_match:
        actions = {"переключись на": "switch", "перейди в": "switch", "сверни": "minimize", "разверни": "maximize", "восстанови": "restore", "закрой": "close"}
        target = window_match.group(2)
        application = resolve_application(target)
        return RoutedAction(
            "window_control",
            {
                "action": actions[window_match.group(1)],
                "window": application.name if application is not None else target,
            },
        )

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
