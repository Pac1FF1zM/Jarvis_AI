"""Build the deterministic, project-owned Jarvis Semantic Core v3 corpus."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Iterable

from ml.jsc.data import DATA_SCHEMA_VERSION, SPLITS, load_jsc_jsonl, validate_jsc_splits
from ml.jsc.jal import (
    DialogueAct,
    JALPlan,
    MissingSlot,
    ToolCall,
    ToolSchemaRegistry,
    dumps,
)
from tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "jsc_data"
SEEDS = {"train": 3101, "validation": 4201, "test": 5301, "evaluation_holdout": 6401}
TARGETS = {
    "train": {
        "single": 650,
        "compound": 120,
        "multi_turn": 180,
        "correction": 100,
        "hard_negative": 120,
        "ood": 100,
        "asr_noise": 130,
    },
    "validation": {
        "single": 130,
        "compound": 25,
        "multi_turn": 35,
        "correction": 20,
        "hard_negative": 25,
        "ood": 25,
        "asr_noise": 30,
    },
    "test": {
        "single": 130,
        "compound": 25,
        "multi_turn": 35,
        "correction": 20,
        "hard_negative": 25,
        "ood": 25,
        "asr_noise": 30,
    },
    "evaluation_holdout": {
        "single": 95,
        "compound": 18,
        "multi_turn": 25,
        "correction": 15,
        "hard_negative": 20,
        "ood": 17,
        "asr_noise": 20,
    },
}

CATEGORY_ACT_MINIMUMS = {
    "train": {"single": {"cancel": 20}, "multi_turn": {"ask": 30, "confirm": 8}},
    "validation": {"single": {"cancel": 5}, "multi_turn": {"ask": 8, "confirm": 3}},
    "test": {"single": {"cancel": 5}, "multi_turn": {"ask": 8, "confirm": 3}},
    "evaluation_holdout": {"single": {"cancel": 4}, "multi_turn": {"ask": 6, "confirm": 2}},
}

CATEGORY_TOOL_MINIMUMS = {
    "train": {"single": {"browser_control": 20, "file_control": 20, "system_control": 20, "window_control": 20}, "multi_turn": {"file_control": 4, "window_control": 4}},
    "validation": {"single": {"browser_control": 5, "file_control": 5, "system_control": 5, "window_control": 5}, "multi_turn": {"file_control": 2, "window_control": 2}},
    "test": {"single": {"browser_control": 5, "file_control": 5, "system_control": 5, "window_control": 5}, "multi_turn": {"file_control": 2, "window_control": 2}},
    "evaluation_holdout": {"single": {"browser_control": 4, "file_control": 4, "system_control": 4, "window_control": 4}, "multi_turn": {"file_control": 1, "window_control": 1}},
}

DESKTOP_VALUES = {
    "train": {
        "queries": ("погода в ташкенте", "новости науки", "курс доллара", "рецепт плова"),
        "sites": ("github.com", "wikipedia.org", "youtube.com"),
        "windows": ("дискорд", "браузер", "блокнот", "проводник"),
        "files": ("диплом", "отчёт", "фотография", "презентация"),
        "folders": ("документы", "загрузки", "рабочий стол"),
    },
    "validation": {
        "queries": ("прогноз на неделю", "расписание поездов", "новости технологий"),
        "sites": ("python.org", "maps.google.com"),
        "windows": ("калькулятор", "paint", "диспетчер задач"),
        "files": ("курсовая", "смета", "скриншот"),
        "folders": ("изображения", "музыка"),
    },
    "test": {
        "queries": ("температура завтра", "афиша кино", "документация python"),
        "sites": ("stackoverflow.com", "openai.com"),
        "windows": ("текстовый редактор", "яндекс", "discord"),
        "files": ("резюме", "чертёж", "таблица"),
        "folders": ("видео", "documents"),
    },
    "evaluation_holdout": {
        "queries": ("погода на выходные", "последние матчи", "учебник pytorch"),
        "sites": ("pytorch.org", "github.io"),
        "windows": ("окно браузера", "графический редактор", "файлы"),
        "files": ("портфолио", "конспект", "архив"),
        "folders": ("downloads", "pictures"),
    },
}

APPLICATION_WORDS = {
    "train": {
        "calculator": ("калькулятор", "calc"),
        "notepad": ("блокнот", "notepad"),
        "explorer": ("проводник", "файлы"),
        "paint": ("пейнт", "paint"),
        "discord": ("дискорд", "discord"),
        "task_manager": ("диспетчер задач", "task manager"),
        "browser": ("браузер", "интернет"),
    },
    "validation": {
        "calculator": ("калькулятор на компьютере",),
        "notepad": ("системный блокнот",),
        "explorer": ("обозреватель файлов",),
        "paint": ("паинт",),
        "discord": ("дискор",),
        "task_manager": ("окно диспетчера задач",),
        "browser": ("мой браузер",),
    },
    "test": {
        "calculator": ("приложение калькулятора",),
        "notepad": ("текстовый блокнот",),
        "explorer": ("проводник windows",),
        "paint": ("пеинт",),
        "discord": ("дисорд",),
        "task_manager": ("task manager",),
        "browser": ("browser",),
    },
    "evaluation_holdout": {
        "calculator": ("калькуляторы",),
        "notepad": ("блакнот",),
        "explorer": ("file explorer",),
        "paint": ("пэйнт",),
        "discord": ("дискод",),
        "task_manager": ("диспетчер запущенных задач",),
        "browser": ("браузер по умолчанию",),
    },
}

LEXICON = {
    "train": {
        "open": ("открой {app}", "запусти {app}", "включи пожалуйста {app}", "джарвис открой {app}"),
        "time": ("который сейчас час", "скажи точное время", "сколько времени джарвис", "проверь системные часы"),
        "list_apps": ("какие приложения ты можешь открыть", "перечисли доступные программы", "покажи белый список приложений"),
        "list_reminders": ("покажи мои напоминания", "какие напоминания активны", "перечисли напоминания"),
        "reminder": (
            "через {minutes} минут напомни {message}",
            "напомни через {minutes} минут {message}",
            "поставь на {minutes} минут напоминание {message}",
            "скажи через {minutes} минут что нужно {message}",
            "не дай забыть через {minutes} минут {message}",
        ),
        "absolute": ("напомни завтра в {clock} {message}", "сегодня в {clock} напомни {message}"),
        "cancel_reminder": ("отмени напоминание номер {number}", "удали напоминание {number}"),
        "cancel": (
            "стоп джарвис",
            "отмени текущую команду",
            "я передумал",
            "ничего не делай",
            "прекрати выполнение",
            "хватит работать над этим",
            "сбрось текущее действие",
            "останови операцию",
            "не выполняй эту просьбу",
            "прерви активную задачу",
            "отставить команду",
            "забудь последний запрос",
            "вернись в режим ожидания",
            "сними текущую задачу",
            "остановись пожалуйста",
            "аннулируй моё действие",
            "не продолжай эту операцию",
            "закрой текущий запрос",
            "прервись и жди",
            "полная отмена действия",
        ),
        "dialogue": ("расскажи про {topic}", "объясни простыми словами {topic}", "давай обсудим {topic}"),
    },
    "validation": {
        "open": ("будь добр открой {app}", "можешь запустить {app}", "я хочу чтобы ты включил {app}"),
        "time": ("сориентируй по времени", "озвучь показания часов", "назови время без объяснений"),
        "list_apps": ("что доступно для запуска", "озвучь перечень программ"),
        "list_reminders": ("назови предстоящие напоминания", "что у меня запланировано из напоминаний"),
        "reminder": ("отсчитай {minutes} минут и напомни {message}", "мне нужно через {minutes} минут вспомнить {message}"),
        "absolute": ("поставь напоминание завтра на {clock} {message}",),
        "cancel_reminder": ("сними напоминание №{number}",),
        "cancel": ("прерви всё что делаешь", "сбрось мою команду", "аннулируй запрос", "останови начатую операцию", "не продолжай выполнение"),
        "dialogue": ("проведи небольшой ликбез про {topic}", "что полезно знать о {topic}"),
    },
    "test": {
        "open": ("прошу запустить {app}", "открой-ка {app}", "мне сейчас нужен {app}"),
        "time": ("глянь что показывают часы", "хочу знать время в эту минуту", "можешь проверить текущий час"),
        "list_apps": ("с чем из приложений ты работаешь", "покажи меню программ"),
        "list_reminders": ("прочитай список будущих напоминаний", "есть ли у меня напоминания"),
        "reminder": ("когда пройдёт {minutes} минут напомни {message}", "через {minutes} минут не забудь сказать {message}"),
        "absolute": ("завтра к {clock} напомни {message}",),
        "cancel_reminder": ("убери напоминание под номером {number}",),
        "cancel": ("вернись в ожидание", "не выполняй прошлый запрос", "прекрати активность", "отмени последнее поручение", "сразу останови задачу"),
        "dialogue": ("как устроено {topic}", "можем поговорить про {topic}"),
    },
    "evaluation_holdout": {
        "open": ("пора открыть {app}", "давай включим {app}", "открой для меня {app}"),
        "time": ("какой час на компьютере", "озвучь время прямо сейчас", "сколько на системных часах"),
        "list_apps": ("какой софт подключён к джарвису", "что умеешь запускать на этом компьютере"),
        "list_reminders": ("напомни какие события ожидают", "что сейчас стоит в напоминаниях"),
        "reminder": ("спустя {minutes} минут скажи что пора {message}", "проконтролируй чтобы через {minutes} минут я вспомнил {message}"),
        "absolute": ("на завтра в {clock} создай напоминание {message}",),
        "cancel_reminder": ("отключи напоминание с номером {number}",),
        "cancel": ("сейчас же прекрати операцию", "отставить прошлое действие", "сверни выполнение команды", "больше ничего не предпринимай"),
        "dialogue": ("объясни принцип работы {topic}", "поделись знаниями о {topic}"),
    },
}

VALUES = {
    "train": {
        "minutes": (1, 2, 5, 10, 15, 20, 30, 45),
        "messages": ("выпить воды", "проверить духовку", "позвонить другу", "ответить коллеге", "покормить кота", "проверить почту", "начать встречу", "сделать перерыв"),
        "clocks": ("08:00", "09:30", "14:15", "18:00", "20:45"),
        "topics": ("нейронные сети", "космос", "браузеры", "программирование", "голосовые ассистенты", "дискорд"),
    },
    "validation": {
        "minutes": (3, 8, 17, 35),
        "messages": ("полить цветы", "забрать заказ", "проверить окна", "позвонить родителям"),
        "clocks": ("07:40", "12:25", "19:10"),
        "topics": ("цифровую приватность", "квантовые компьютеры", "домашние сети"),
    },
    "test": {
        "minutes": (4, 11, 27, 50),
        "messages": ("выключить утюг", "сохранить документ", "проверить батарею", "достать ключи"),
        "clocks": ("06:50", "13:35", "21:20"),
        "topics": ("обучение с подкреплением", "устройство микрофона", "автоматизацию дома"),
    },
    "evaluation_holdout": {
        "minutes": (6, 14, 23, 55),
        "messages": ("снять кастрюлю", "открыть дверь курьеру", "вынуть еду", "проверить заряд"),
        "clocks": ("10:05", "16:40", "22:15"),
        "topics": ("компьютерное зрение", "историю интернета", "защиту паролей"),
    },
}

HARD_NEGATIVES = {
    "train": ("зачем нужен {app}", "расскажи как работает {app}", "почему {app} иногда зависает", "мне нравится {app}", "можно ли удалить {app}", "слово напоминание означает что", "почему люди смотрят на время", "не открывай {app} просто расскажи о нём"),
    "validation": ("какая польза от {app}", "сравни {app} с другими программами", "я не просил открывать {app}", "как устроены системные часы"),
    "test": ("что будет если закрыть {app}", "объясни назначение {app}", "не запускай {app} я только спрашиваю", "поговорим о планировании времени"),
    "evaluation_holdout": ("стоит ли пользоваться {app}", "почему называется {app}", "без запуска опиши {app}", "как не забывать о делах"),
}

OOD_PHRASES = {
    "train": ("выключи компьютер", "удали системную папку", "отправь сообщение начальнику", "переведи деньги", "закажи такси", "включи конкретный альбом", "измени яркость монитора", "покажи пароль от вайфая", "напиши код за меня", "найди билет на самолёт", "смени пароль учётной записи", "скачай неизвестную программу"),
    "validation": ("перезагрузи ноутбук", "очисти весь диск", "позвони в банк", "купи продукты онлайн", "включи любимую песню", "измени яркость экрана"),
    "test": ("отформатируй диск", "отправь письмо клиенту", "оплати счёт", "поставь фильм", "выключи интернет", "установи драйвер"),
    "evaluation_holdout": ("заверши все процессы", "удали мою учётную запись", "закажи еду", "сделай банковский перевод", "позвони маме", "подключись к чужой сети"),
}

INCOMPLETE_OOD = {
    "train": (
        "сделай как вчера",
        "открой это",
        "ну ту штуку",
        "эээ дальше",
        "неразборчивые слова",
        "один два потом",
    ),
    "validation": (
        "повтори то действие",
        "запусти вон то",
        "эм продолжай",
        "сделай что-нибудь",
        "слова потерялись",
        "первое а затем это",
    ),
    "test": (
        "верни прошлую штуку",
        "включи его",
        "так ну дальше",
        "выполни как обычно",
        "часть команды не слышно",
        "после второго делай",
    ),
    "evaluation_holdout": (
        "сделай то самое",
        "открой нужное",
        "хм продолжи потом",
        "разберись сам",
        "команда оборвалась",
        "сначала это и дальше",
    ),
}

CANCEL_ID_ANSWERS = {
    "train": "номер {number}",
    "validation": "это номер {number}",
    "test": "напоминание {number}",
    "evaluation_holdout": "под номером {number}",
}

COMPOUND_FRAMES = {
    "train": (
        "открой {app}, а через {minutes} минут напомни {message}",
        "запусти {app} и поставь напоминание через {minutes} минут {message}",
        "сначала включи {app}, потом через {minutes} минут напомни {message}",
    ),
    "validation": (
        "пожалуйста открой {app}, затем через {minutes} минут дай знать что пора {message}",
        "включи {app}; ещё создай напоминание на {minutes} минут: {message}",
        "мне нужен {app}, и спустя {minutes} минут напомни про {message}",
    ),
    "test": (
        "запусти {app}, после этого через {minutes} минут скажи что нужно {message}",
        "открой {app} и по истечении {minutes} минут сообщи: {message}",
        "сделай два дела: включи {app}; через {minutes} минут напомни {message}",
    ),
    "evaluation_holdout": (
        "сначала подготовь {app}; спустя {minutes} минут напомни мне {message}",
        "открой для меня {app}, а через {minutes} минут предупреди о том чтобы {message}",
        "нужно включить {app} и завести напоминание через {minutes} минут насчёт {message}",
    ),
}

TIME_AND_APP_FRAMES = {
    "train": "скажи который час и открой {app}",
    "validation": "сначала назови время затем запусти {app}",
    "test": "проверь часы после чего включи {app}",
    "evaluation_holdout": "озвучь текущее время заодно открой {app}",
}

MULTI_TURN_FRAMES = {
    "train": {
        "requests": (
            "напомни мне {message}",
            "создай напоминание {message}",
            "мне нужно не забыть {message}",
            "поставь напоминание про {message}",
        ),
        "question": "Когда напомнить?",
        "answers": ("через {minutes} минут", "спустя {minutes} минут", "минут через {minutes}"),
        "app_request": "открой приложение",
        "app_question": "Какое приложение открыть?",
    },
    "validation": {
        "requests": ("сохрани напоминание {message}", "нужно напомнить {message}", "добавь в напоминания {message}"),
        "question": "На какое время поставить напоминание?",
        "answers": ("ровно через {minutes} минут", "подожди {minutes} минут", "отсчитай {minutes} минут"),
        "app_request": "запусти какую-нибудь программу",
        "app_question": "Назовите нужную программу.",
    },
    "test": {
        "requests": ("помоги не забыть {message}", "запиши напоминание {message}", "уведоми меня про {message}"),
        "question": "Через сколько минут сообщить об этом?",
        "answers": ("через {minutes}", "{minutes} минут от сейчас", "пусть будет через {minutes} минут"),
        "app_request": "мне нужно открыть программу",
        "app_question": "Уточните название приложения.",
    },
    "evaluation_holdout": {
        "requests": ("оставь напоминание о том чтобы {message}", "зафиксируй что надо {message}", "мне потребуется напомнить {message}"),
        "question": "Когда это напомнить?",
        "answers": ("спустя ровно {minutes} минут", "срок {minutes} минут", "по истечении {minutes} минут"),
        "app_request": "включи одно приложение",
        "app_question": "Какое именно приложение вам нужно?",
    },
}

CANCEL_MULTI_TURN_FRAMES = {
    "train": ("отмени напоминание", "Назовите номер напоминания."),
    "validation": ("убери одно напоминание", "Какой у него номер?"),
    "test": ("нужно удалить напоминание", "Уточните идентификатор напоминания."),
    "evaluation_holdout": ("сними запланированное напоминание", "Сообщите его номер."),
}

CORRECTION_FRAMES = {
    "train": {
        "app_request": "открой {old}",
        "app_question": "Открыть {old}?",
        "app_fix": "нет, лучше {new}",
        "reminder_request": "напомни через {old} минут {message}",
        "reminder_question": "Через {old} минут?",
        "reminder_fix": "нет, через {new} минут",
    },
    "validation": {
        "app_request": "запусти {old}",
        "app_question": "Вы имеете в виду {old}?",
        "app_fix": "исправь на {new}",
        "reminder_request": "поставь напоминание на {old} минут {message}",
        "reminder_question": "Оставить срок {old} минут?",
        "reminder_fix": "поменяй на {new} минут",
    },
    "test": {
        "app_request": "мне нужен {old}",
        "app_question": "Запустить {old}?",
        "app_fix": "нет выбери {new}",
        "reminder_request": "через {old} минут скажи {message}",
        "reminder_question": "Напомнить спустя {old} минут?",
        "reminder_fix": "исправление: через {new} минут",
    },
    "evaluation_holdout": {
        "app_request": "включи программу {old}",
        "app_question": "Подтвердите запуск {old}.",
        "app_fix": "замени её на {new}",
        "reminder_request": "создай таймер на {old} минут чтобы напомнить {message}",
        "reminder_question": "Сохранить интервал {old} минут?",
        "reminder_fix": "скорректируй интервал до {new} минут",
    },
}


@dataclass(frozen=True)
class Candidate:
    category: str
    family: str
    text: str
    target: JALPlan
    history: tuple[tuple[str, str], ...] = ()
    state: JALPlan | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> str:
        history = "|".join(f"{role}:{_norm(text)}" for role, text in self.history)
        state = dumps(self.state) if self.state else "null"
        return f"{history}|{state}|{_norm(self.text)}"


def _registry() -> ToolSchemaRegistry:
    tools = ToolRegistry()
    tools.discover("tools")
    return ToolSchemaRegistry.from_tool_registry(tools)


def generate() -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Return deterministic records without touching the filesystem."""
    registry = _registry()
    generated: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        pools = _candidate_pools(split)
        selected: list[Candidate] = []
        used: set[str] = set()
        for category, count in TARGETS[split].items():
            candidates = sorted(pools[category], key=lambda item: item.signature)
            random.Random(SEEDS[split] + len(category) * 97).shuffle(candidates)
            accepted = [candidate for candidate in candidates if candidate.signature not in used]
            if len(accepted) < count:
                raise RuntimeError(
                    f"{split}/{category}: need {count} unique candidates, have {len(accepted)}"
                )
            chosen: list[Candidate] = []
            for act, minimum in CATEGORY_ACT_MINIMUMS[split].get(category, {}).items():
                matching = [candidate for candidate in accepted if candidate.target.act.value == act]
                if len(matching) < minimum:
                    raise RuntimeError(
                        f"{split}/{category}: need {minimum} examples for act={act}, "
                        f"have {len(matching)}"
                    )
                chosen.extend(matching[:minimum])
            for tool, minimum in CATEGORY_TOOL_MINIMUMS[split].get(category, {}).items():
                already = {candidate.signature for candidate in chosen}
                matching = [
                    candidate for candidate in accepted
                    if candidate.signature not in already
                    and any(step.tool == tool for step in candidate.target.steps)
                ]
                if len(matching) < minimum:
                    raise RuntimeError(
                        f"{split}/{category}: need {minimum} examples for tool={tool}, "
                        f"have {len(matching)}"
                    )
                chosen.extend(matching[:minimum])
            chosen_signatures = {candidate.signature for candidate in chosen}
            chosen.extend(
                candidate
                for candidate in accepted
                if candidate.signature not in chosen_signatures
            )
            chosen = chosen[:count]
            selected.extend(chosen)
            used.update(candidate.signature for candidate in chosen)
        random.Random(SEEDS[split]).shuffle(selected)
        records: list[dict[str, Any]] = []
        counters: Counter[str] = Counter()
        for candidate in selected:
            registry.validate(candidate.target)
            if candidate.state is not None:
                registry.validate(candidate.state)
            counters[candidate.category] += 1
            records.append(
                _to_record(
                    split,
                    candidate,
                    counters[candidate.category],
                )
            )
        generated[split] = records
    _audit_raw_splits(generated)
    return generated, registry.schema_fingerprint


def write_dataset(output_dir: str | Path = DATA_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated, schema_fingerprint = generate()
    split_files: dict[str, Path] = {}
    for split, records in generated.items():
        path = output / f"{split}.jsonl"
        content = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        )
        _atomic_write(path, content)
        split_files[split] = path
    loaded = {
        split: load_jsc_jsonl(path, _registry(), expected_split=split)
        for split, path in split_files.items()
    }
    report = validate_jsc_splits(loaded)
    manifest = {
        "version": 3,
        "data_schema_version": DATA_SCHEMA_VERSION,
        "generator": "training_workspace.build_jsc_dataset",
        "seeded": True,
        "external_sources": False,
        "synthetic_holdout": True,
        "family_id_scheme": "sha256(category+roles+slot-masked-dialogue)[:16]",
        "split_policy": "no structural-family or exact-model-input overlap",
        "tool_schema_sha256": schema_fingerprint,
        "splits": {},
    }
    for split, path in split_files.items():
        content = path.read_bytes()
        manifest["splits"][split] = {
            **report[split],
            "file": path.name,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    _atomic_write(
        output / "dataset_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _candidate_pools(split: str) -> dict[str, list[Candidate]]:
    single = _single_candidates(split)
    return {
        "single": single,
        "compound": _compound_candidates(split),
        "multi_turn": _multi_turn_candidates(split),
        "correction": _correction_candidates(split),
        "hard_negative": _hard_negative_candidates(split),
        "ood": _ood_candidates(split),
        "asr_noise": _asr_noise_candidates(split, single),
    }


def _single_candidates(split: str) -> list[Candidate]:
    lex = LEXICON[split]
    values = VALUES[split]
    candidates: list[Candidate] = []
    for tool_name, words in APPLICATION_WORDS[split].items():
        for template_index, template in enumerate(lex["open"]):
            for word in words:
                candidates.append(
                    Candidate(
                        "single",
                        f"{split}.single.open_{template_index}",
                        template.format(app=word),
                        _execute("open_application", application=tool_name),
                    )
                )
    candidates.extend(
        Candidate("single", f"{split}.single.time_{index}", text, _execute("get_current_time"))
        for index, text in enumerate(lex["time"])
    )
    candidates.extend(
        Candidate("single", f"{split}.single.apps_{index}", text, _execute("list_applications"))
        for index, text in enumerate(lex["list_apps"])
    )
    candidates.extend(
        Candidate("single", f"{split}.single.reminders_{index}", text, _execute("list_reminders"))
        for index, text in enumerate(lex["list_reminders"])
    )
    for template_index, template in enumerate(lex["reminder"]):
        for minutes in values["minutes"]:
            for message in values["messages"]:
                candidates.append(
                    Candidate(
                        "single",
                        f"{split}.single.relative_{template_index}",
                        template.format(minutes=minutes, message=message),
                        _execute("set_reminder", minutes=minutes, message=message),
                    )
                )
    for template_index, template in enumerate(lex["absolute"]):
        day = "завтра" if "завтра" in template or "на завтра" in template else "сегодня"
        for clock in values["clocks"]:
            for message in values["messages"]:
                candidates.append(
                    Candidate(
                        "single",
                        f"{split}.single.absolute_{template_index}",
                        template.format(clock=clock, message=message),
                        _execute("set_reminder", clock_time=clock, day=day, message=message),
                    )
                )
    for template_index, template in enumerate(lex["cancel_reminder"]):
        for number in range(1, 21):
            candidates.append(
                Candidate(
                    "single",
                    f"{split}.single.cancel_reminder_{template_index}",
                    template.format(number=number),
                    _execute("cancel_reminder", reminder_id=number),
                )
            )
    candidates.extend(
        Candidate("single", f"{split}.single.cancel_{index}", text, JALPlan(DialogueAct.CANCEL, reason="user_requested"))
        for index, text in enumerate(lex["cancel"])
    )
    for template_index, template in enumerate(lex["dialogue"]):
        for topic in values["topics"]:
            candidates.append(
                Candidate(
                    "single",
                    f"{split}.single.dialogue_{template_index}",
                    template.format(topic=topic),
                    JALPlan(DialogueAct.DIALOGUE, reason="general_chat"),
                )
            )
    candidates.extend(_desktop_single_candidates(split))
    return candidates


def _desktop_single_candidates(split: str) -> list[Candidate]:
    """Project-owned desktop scenarios; every split uses distinct phrasing."""
    values = DESKTOP_VALUES[split]
    candidates: list[Candidate] = []
    search_templates = {
        "train": ("найди в интернете {value}", "поищи в сети {value}"),
        "validation": ("выполни веб поиск по теме {value}", "разыщи онлайн {value}"),
        "test": ("проверь через браузер {value}", "отыщи в интернете сведения {value}"),
        "evaluation_holdout": ("сделай сетевой поиск {value}", "посмотри онлайн информацию {value}"),
    }[split]
    for index, template in enumerate(search_templates):
        for query in values["queries"]:
            text = template.format(value=query)
            for wrapper_index, wrapped in enumerate((text, f"джарвис {text}", f"пожалуйста {text}")):
                candidates.append(Candidate("single", f"{split}.single.browser_search_{index}_{wrapper_index}", wrapped, _execute("browser_control", action="search", query=query)))
    site_templates = {
        "train": "открой сайт {value}",
        "validation": "перейди на веб адрес {value}",
        "test": "загрузи страницу {value}",
        "evaluation_holdout": "покажи в браузере ресурс {value}",
    }
    for site in values["sites"]:
        site_text = site_templates[split].format(value=site)
        candidates.append(Candidate("single", f"{split}.single.browser_site", site_text, _execute("browser_control", action="open_site", url=site)))
        candidates.append(Candidate("single", f"{split}.single.browser_site_polite", f"джарвис {site_text} пожалуйста", _execute("browser_control", action="open_site", url=site)))
    tab_phrases = {
        "train": (("открой новую вкладку", "new_tab"), ("закрой текущую вкладку", "close_tab"), ("верни закрытую вкладку", "reopen_tab"), ("следующая вкладка", "next_tab"), ("предыдущая вкладка", "previous_tab")),
        "validation": (("создай ещё одну вкладку", "new_tab"), ("убери эту вкладку", "close_tab"), ("восстанови последнюю вкладку", "reopen_tab"), ("переключись на вкладку справа", "next_tab"), ("переключись на вкладку слева", "previous_tab")),
        "test": (("добавь чистую вкладку", "new_tab"), ("заверши активную вкладку", "close_tab"), ("открой недавно закрытую вкладку", "reopen_tab"), ("листай вкладки вперёд", "next_tab"), ("листай вкладки назад", "previous_tab")),
        "evaluation_holdout": (("нужна новая страница в браузере", "new_tab"), ("закрой страницу браузера", "close_tab"), ("возврати прошлую страницу", "reopen_tab"), ("покажи соседнюю вкладку далее", "next_tab"), ("покажи соседнюю вкладку ранее", "previous_tab")),
    }[split]
    for text, action in tab_phrases:
        candidates.append(Candidate("single", f"{split}.single.browser_tab_{action}", text, _execute("browser_control", action=action)))

    system_phrases = {
        "train": (("прибавь громкость", "volume_up"), ("убавь громкость", "volume_down"), ("переключи звук", "volume_mute"), ("поставь музыку на паузу", "media_play_pause"), ("следующий трек", "media_next"), ("предыдущий трек", "media_previous")),
        "validation": (("сделай звук громче", "volume_up"), ("сделай звук тише", "volume_down"), ("отключи динамики", "volume_mute"), ("продолжи воспроизведение", "media_play_pause"), ("перейди к следующей песне", "media_next"), ("верни прошлую песню", "media_previous")),
        "test": (("подними уровень звука", "volume_up"), ("снизь уровень звука", "volume_down"), ("заглуши аудио", "volume_mute"), ("переключи воспроизведение", "media_play_pause"), ("листай музыку вперёд", "media_next"), ("листай музыку назад", "media_previous")),
        "evaluation_holdout": (("добавь звука", "volume_up"), ("приглуши звук", "volume_down"), ("переключи беззвучный режим", "volume_mute"), ("останови или запусти музыку", "media_play_pause"), ("включи композицию далее", "media_next"), ("включи композицию ранее", "media_previous")),
    }[split]
    for text, action in system_phrases:
        for wrapped in (text, f"джарвис {text}"):
            candidates.append(Candidate("single", f"{split}.single.system_{action}", wrapped, _execute("system_control", action=action)))
    settings = (("звук", "sound"), ("экран", "display"), ("сеть", "network"), ("микрофон", "microphone"), ("обновления", "update"))
    setting_frames = {"train": "открой настройки {label}", "validation": "покажи параметры раздела {label}", "test": "перейди к системным параметрам {label}", "evaluation_holdout": "запусти страницу конфигурации {label}"}
    for label, setting in settings:
        setting_text = setting_frames[split].format(label=label)
        candidates.append(Candidate("single", f"{split}.single.setting_{setting}", setting_text, _execute("system_control", action="open_settings", setting=setting)))
        candidates.append(Candidate("single", f"{split}.single.setting_{setting}_polite", f"пожалуйста {setting_text}", _execute("system_control", action="open_settings", setting=setting)))

    window_frames = {
        "train": (("переключись на {value}", "switch"), ("сверни {value}", "minimize"), ("разверни {value}", "maximize"), ("восстанови окно {value}", "restore"), ("закрой окно {value}", "close")),
        "validation": (("покажи окно {value}", "switch"), ("убери с экрана {value}", "minimize"), ("раскрой на весь экран {value}", "maximize"), ("верни окно {value}", "restore"), ("заверши окно {value}", "close")),
        "test": (("перейди в окно {value}", "switch"), ("минимизируй {value}", "minimize"), ("максимизируй {value}", "maximize"), ("подними окно {value}", "restore"), ("закрой приложение с окном {value}", "close")),
        "evaluation_holdout": (("выведи вперёд {value}", "switch"), ("спрячь окно {value}", "minimize"), ("растяни окно {value}", "maximize"), ("возврати видимость {value}", "restore"), ("заверши работу окна {value}", "close")),
    }[split]
    for template, action in window_frames:
        for window in values["windows"]:
            text = template.format(value=window)
            candidates.append(Candidate("single", f"{split}.single.window_{action}", text, _execute("window_control", action=action, window=window)))
            candidates.append(Candidate("single", f"{split}.single.window_{action}_polite", f"джарвис {text} пожалуйста", _execute("window_control", action=action, window=window)))

    file_frames = {
        "train": ("найди файл {value}", "поищи папку {value}"),
        "validation": ("разыщи документ {value}", "отыщи каталог {value}"),
        "test": ("проверь где лежит файл {value}", "покажи расположение папки {value}"),
        "evaluation_holdout": ("выполни поиск файла {value}", "обнаружь директорию {value}"),
    }[split]
    for template in file_frames:
        for query in values["files"]:
            text = template.format(value=query)
            candidates.append(Candidate("single", f"{split}.single.file_find", text, _execute("file_control", action="find", query=query)))
            candidates.append(Candidate("single", f"{split}.single.file_find_polite", f"джарвис {text}", _execute("file_control", action="find", query=query)))
    folder_frames = {"train": "покажи содержимое папки {value}", "validation": "перечисли файлы внутри {value}", "test": "что хранится в каталоге {value}", "evaluation_holdout": "прочитай список объектов из {value}"}
    for folder in values["folders"]:
        folder_text = folder_frames[split].format(value=folder)
        candidates.append(Candidate("single", f"{split}.single.file_list", folder_text, _execute("file_control", action="list", path=folder)))
        candidates.append(Candidate("single", f"{split}.single.file_list_polite", f"пожалуйста {folder_text}", _execute("file_control", action="list", path=folder)))
    return candidates


def _compound_candidates(split: str) -> list[Candidate]:
    values = VALUES[split]
    applications = list(APPLICATION_WORDS[split].items())
    candidates: list[Candidate] = []
    frames = COMPOUND_FRAMES[split]
    for frame_index, frame in enumerate(frames):
        for app_name, words in applications:
            for minutes in values["minutes"]:
                for message in values["messages"]:
                    candidates.append(
                        Candidate(
                            "compound",
                            f"{split}.compound.open_remind_{frame_index}",
                            frame.format(app=words[0], minutes=minutes, message=message),
                            JALPlan(
                                DialogueAct.EXECUTE,
                                steps=(
                                    ToolCall("open_application", {"application": app_name}),
                                    ToolCall("set_reminder", {"minutes": minutes, "message": message}),
                                ),
                            ),
                            metadata={"difficulty": "compositional"},
                        )
                    )
    for app_name, words in applications:
        candidates.append(
            Candidate(
                "compound",
                f"{split}.compound.time_open",
                TIME_AND_APP_FRAMES[split].format(app=words[-1]),
                JALPlan(
                    DialogueAct.EXECUTE,
                    steps=(
                        ToolCall("get_current_time"),
                        ToolCall("open_application", {"application": app_name}),
                    ),
                ),
                metadata={"difficulty": "compositional"},
            )
        )
    return candidates


def _multi_turn_candidates(split: str) -> list[Candidate]:
    values = VALUES[split]
    frames = MULTI_TURN_FRAMES[split]
    candidates: list[Candidate] = []
    for message in values["messages"]:
        pending = JALPlan(
            DialogueAct.ASK,
            steps=(ToolCall("set_reminder", {"message": message}),),
            missing=(MissingSlot(0, "minutes"),),
            reason="missing_time",
        )
        for request_index, request_template in enumerate(frames["requests"]):
            candidates.append(
                Candidate(
                    "multi_turn",
                    f"{split}.multi.reminder_ask_{request_index}",
                    request_template.format(message=message),
                    pending,
                    metadata={"turn": "request"},
                )
            )
        for minutes in values["minutes"]:
            for answer_index, answer_template in enumerate(frames["answers"]):
                answer = answer_template.format(minutes=minutes)
                candidates.append(
                    Candidate(
                        "multi_turn",
                        f"{split}.multi.reminder_fill_{answer_index}",
                        answer,
                        _execute("set_reminder", minutes=minutes, message=message),
                        history=(
                            ("user", frames["requests"][0].format(message=message)),
                            ("jarvis", frames["question"]),
                        ),
                        state=pending,
                        metadata={"turn": "slot_fill"},
                    )
                )
    for app_name, words in APPLICATION_WORDS[split].items():
        pending = JALPlan(
            DialogueAct.ASK,
            steps=(ToolCall("open_application"),),
            missing=(MissingSlot(0, "application"),),
            reason="missing_application",
        )
        if app_name == next(iter(APPLICATION_WORDS[split])):
            candidates.append(
                Candidate(
                    "multi_turn",
                    f"{split}.multi.application_ask",
                    frames["app_request"],
                    pending,
                    metadata={"turn": "request"},
                )
            )
        candidates.append(
            Candidate(
                "multi_turn",
                f"{split}.multi.application_fill",
                words[0],
                _execute("open_application", application=app_name),
                history=(
                    ("user", frames["app_request"]),
                    ("jarvis", frames["app_question"]),
                ),
                state=pending,
                metadata={"turn": "slot_fill"},
            )
        )
    for number in range(1, 31):
        pending = JALPlan(
            DialogueAct.ASK,
            steps=(ToolCall("cancel_reminder"),),
            missing=(MissingSlot(0, "reminder_id"),),
            reason="missing_reminder_id",
        )
        request, question = CANCEL_MULTI_TURN_FRAMES[split]
        if number == 1:
            candidates.append(
                Candidate(
                    "multi_turn",
                    f"{split}.multi.cancel_reminder_ask",
                    request,
                    pending,
                    metadata={"turn": "request"},
                )
            )
        candidates.append(
            Candidate(
                "multi_turn",
                f"{split}.multi.cancel_reminder_fill",
                CANCEL_ID_ANSWERS[split].format(number=number),
                _execute("cancel_reminder", reminder_id=number),
                history=(("user", request), ("jarvis", question)),
                state=pending,
                metadata={"turn": "slot_fill"},
            )
        )
    confirmation_frames = {
        "train": {
            "window_request": "закрой окно", "window_question": "Какое окно закрыть?",
            "delete": "удали файл {value}", "delete_alt": "перемести файл {value} в корзину",
            "delete_question": "Переместить {value} в корзину?", "delete_answer": "подтверждаю удаление",
        },
        "validation": {
            "window_request": "заверши одно окно", "window_question": "Назовите нужное окно.",
            "delete": "убери документ {value}", "delete_alt": "отправь документ {value} в корзину",
            "delete_question": "Подтвердите отправку {value} в корзину.", "delete_answer": "согласен отправляй в корзину",
        },
        "test": {
            "window_request": "закрой приложение", "window_question": "Какое приложение нужно закрыть?",
            "delete": "перемести в корзину файл {value}", "delete_alt": "удали с компьютера файл {value}",
            "delete_question": "Действительно удалить {value}?", "delete_answer": "да перемещай",
        },
        "evaluation_holdout": {
            "window_request": "закрой активную программу", "window_question": "Уточните название окна.",
            "delete": "отправь в корзину документ {value}", "delete_alt": "сотри документ {value}",
            "delete_question": "Разрешаете убрать {value}?", "delete_answer": "разрешаю удалить",
        },
    }[split]
    window_pending = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("window_control", {"action": "close"}),),
        missing=(MissingSlot(0, "window"),),
        reason="missing_window",
    )
    for window in DESKTOP_VALUES[split]["windows"]:
        candidates.append(Candidate("multi_turn", f"{split}.multi.window_close_fill", window, _execute("window_control", action="close", window=window), history=(("user", confirmation_frames["window_request"]), ("jarvis", confirmation_frames["window_question"])), state=window_pending, metadata={"turn": "slot_fill"}))
    for file_name in DESKTOP_VALUES[split]["files"]:
        path = f"документы/{file_name}.txt"
        pending = JALPlan(
            DialogueAct.CONFIRM,
            steps=(ToolCall("file_control", {"action": "delete", "path": path}),),
            reason="user_confirmation",
        )
        for request_key in ("delete", "delete_alt"):
            request = confirmation_frames[request_key].format(value=path)
            candidates.append(Candidate("multi_turn", f"{split}.multi.file_delete_confirm_{request_key}", request, pending, metadata={"turn": "confirmation_request"}))
            candidates.append(Candidate("multi_turn", f"{split}.multi.file_delete_execute_{request_key}", confirmation_frames["delete_answer"], _execute("file_control", action="delete", path=path, confirmed=True), history=(("user", request), ("jarvis", confirmation_frames["delete_question"].format(value=path))), state=pending, metadata={"turn": "confirmation_answer"}))
    return candidates


def _correction_candidates(split: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    frames = CORRECTION_FRAMES[split]
    apps = list(APPLICATION_WORDS[split].items())
    for old_name, old_words in apps:
        old_plan = JALPlan(
            DialogueAct.CONFIRM,
            steps=(ToolCall("open_application", {"application": old_name}),),
            reason="user_confirmation",
        )
        for new_name, new_words in apps:
            if new_name == old_name:
                continue
            candidates.append(
                Candidate(
                    "correction",
                    f"{split}.correction.application",
                    frames["app_fix"].format(new=new_words[0]),
                    _execute("open_application", application=new_name),
                    history=(
                        ("user", frames["app_request"].format(old=old_words[0])),
                        ("jarvis", frames["app_question"].format(old=old_words[0])),
                    ),
                    state=old_plan,
                    metadata={"correction": "replace_application"},
                )
            )
    values = VALUES[split]
    for message in values["messages"]:
        for old_minutes in values["minutes"]:
            old_plan = JALPlan(
                DialogueAct.CONFIRM,
                steps=(ToolCall("set_reminder", {"minutes": old_minutes, "message": message}),),
                reason="user_confirmation",
            )
            for new_minutes in values["minutes"]:
                if new_minutes != old_minutes:
                    candidates.append(
                        Candidate(
                            "correction",
                            f"{split}.correction.reminder_time",
                            frames["reminder_fix"].format(new=new_minutes),
                            _execute("set_reminder", minutes=new_minutes, message=message),
                            history=(
                                (
                                    "user",
                                    frames["reminder_request"].format(
                                        old=old_minutes,
                                        message=message,
                                    ),
                                ),
                                (
                                    "jarvis",
                                    frames["reminder_question"].format(old=old_minutes),
                                ),
                            ),
                            state=old_plan,
                            metadata={"correction": "replace_time"},
                        )
                    )
    return candidates


def _hard_negative_candidates(split: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    wrappers = (
        "{text}",
        "джарвис {text}",
        "{text} просто интересно",
        "объясни пожалуйста {text}",
    )
    for template_index, template in enumerate(HARD_NEGATIVES[split]):
        aliases = (
            tuple(word for words in APPLICATION_WORDS[split].values() for word in words)
            if "{app}" in template
            else (None,)
        )
        for alias in aliases:
            base_text = template.format(app=alias) if alias is not None else template
            for wrapper_index, wrapper in enumerate(wrappers):
                candidates.append(
                    Candidate(
                        "hard_negative",
                        f"{split}.hard_negative.mention_{template_index}_{wrapper_index}",
                        wrapper.format(text=base_text),
                        JALPlan(DialogueAct.DIALOGUE, reason="general_chat"),
                        metadata={"contrast": "tool_mention_without_command"},
                    )
                )
    return candidates


def _ood_candidates(split: str) -> list[Candidate]:
    wrappers = (
        "{text}",
        "джарвис {text}",
        "пожалуйста {text}",
        "можешь {text}",
        "мне нужно чтобы ты {text}",
        "давай {text}",
        "прямо сейчас {text}",
        "помоги {text}",
        "срочно {text}",
        "выполни команду {text}",
    )
    candidates: list[Candidate] = []
    for phrase_index, phrase in enumerate(OOD_PHRASES[split]):
        for wrapper_index, wrapper in enumerate(wrappers):
            candidates.append(
                Candidate(
                    "ood",
                    f"{split}.ood.unsupported_{wrapper_index}",
                    wrapper.format(text=phrase),
                    JALPlan(DialogueAct.REJECT, reason="unsupported_tool"),
                    metadata={"ood_source": phrase_index},
                )
            )
    for index, text in enumerate(INCOMPLETE_OOD[split]):
        candidates.append(
            Candidate(
                "ood",
                f"{split}.ood.incomplete_{index}",
                text,
                JALPlan(DialogueAct.REJECT, reason="out_of_scope"),
                metadata={"ood_source": "incomplete"},
            )
        )
    return candidates


def _asr_noise_candidates(split: str, singles: Iterable[Candidate]) -> list[Candidate]:
    replacements = {
        "калькулятор": ("к алкулятор", "calculator"),
        "блокнот": ("блакнот", "notepad"),
        "пейнт": ("пеинт", "paint"),
        "дискорд": ("дисорд", "discord"),
        "открой": ("отпрой", "open"),
        "сколько": ("колька", "time_question"),
    }
    fillers = {
        "train": ("ээ {text}", "{text} пожалуйста", "джарвис {text}"),
        "validation": ("эм {text}", "{text} если можно"),
        "test": ("слушай {text}", "{text} сейчас"),
        "evaluation_holdout": ("так {text}", "{text} джарвис"),
    }[split]
    candidates: list[Candidate] = []
    for base_index, base in enumerate(singles):
        if base.target.act != DialogueAct.EXECUTE:
            continue
        changed = base.text
        applied = "filler"
        for source, (replacement, replacement_id) in replacements.items():
            if source in changed.casefold():
                changed = changed.casefold().replace(source, replacement, 1)
                applied = f"replace_{replacement_id}"
                break
        if changed == base.text:
            frame = fillers[base_index % len(fillers)]
            changed = frame.format(text=base.text)
        candidates.append(
            Candidate(
                "asr_noise",
                f"{split}.asr_noise.{applied}",
                changed,
                base.target,
                history=base.history,
                state=base.state,
                metadata={"noise": applied, "clean_text": base.text},
            )
        )
    return candidates


def _execute(tool: str, **arguments: Any) -> JALPlan:
    return JALPlan(
        DialogueAct.EXECUTE,
        steps=(ToolCall(tool, arguments),),
    )


def _to_record(split: str, candidate: Candidate, index: int) -> dict[str, Any]:
    structural_signature = _structural_signature(candidate)
    family_hash = hashlib.sha256(structural_signature.encode("utf-8")).hexdigest()[:16]
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "scenario_id": f"{split}.{candidate.category}.{index:05d}",
        "split": split,
        "family_id": f"{candidate.category}.{family_hash}",
        "category": candidate.category,
        "history": [
            {"role": role, "text": text} for role, text in candidate.history
        ],
        "text": candidate.text,
        "state_jal": dumps(candidate.state) if candidate.state else None,
        "target_jal": dumps(candidate.target),
        "metadata": {
            "synthetic": True,
            "generator_family": candidate.family,
            **candidate.metadata,
        },
    }


def _structural_signature(candidate: Candidate) -> str:
    turns = [
        [role, _surface_shape(text)] for role, text in candidate.history
    ]
    turns.append(["user", _surface_shape(candidate.text)])
    return json.dumps(
        [candidate.category, turns],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _surface_shape(value: str) -> str:
    """Mask slot values so paraphrase-family leakage is visible across splits."""
    shaped = _norm(value)
    shaped = re.sub(r"\b\d{1,2}:\d{2}\b", "{clock}", shaped)
    for term in _slot_terms():
        shaped = shaped.replace(_norm(term), "{slot}")
    return re.sub(r"\b\d+\b", "{number}", shaped)


@cache
def _slot_terms() -> tuple[str, ...]:
    terms = {
        term
        for split in SPLITS
        for words in APPLICATION_WORDS[split].values()
        for term in words
    }
    for split in SPLITS:
        terms.update(VALUES[split]["messages"])
        terms.update(VALUES[split]["topics"])
        for values in DESKTOP_VALUES[split].values():
            terms.update(values)
    return tuple(sorted(terms, key=len, reverse=True))


def _audit_raw_splits(generated: dict[str, list[dict[str, Any]]]) -> None:
    families: dict[str, set[str]] = {}
    signatures: dict[str, set[str]] = {}
    for split, records in generated.items():
        families[split] = {record["family_id"] for record in records}
        signatures[split] = {
            json.dumps(
                [record["history"], record["state_jal"], _norm(record["text"])],
                ensure_ascii=False,
                sort_keys=True,
            )
            for record in records
        }
        if len(signatures[split]) != len(records):
            raise RuntimeError(f"duplicate model inputs inside {split}")
        counts = Counter(record["category"] for record in records)
        if dict(counts) != TARGETS[split]:
            raise RuntimeError(f"category imbalance in {split}: {counts}")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            if families[left] & families[right]:
                raise RuntimeError(f"family leakage between {left} and {right}")
            if signatures[left] & signatures[right]:
                raise RuntimeError(f"input leakage between {left} and {right}")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _norm(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").split())


def main() -> None:
    manifest = write_dataset()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
