"""Build the deterministic, project-owned Jarvis NLU fine-tuning corpus.

The corpus is generated only from the phrase families below.  It downloads
nothing and uses no external model, tokenizer, dataset, or Hugging Face code.
Run it from the repository root with::

    python -m training_workspace.build_dataset
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ml.nlu.data import build_examples
from ml.nlu.schema import INTENTS

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TARGETS = {"train": 120, "validation": 30, "evaluation_holdout": 15}
SEEDS = {"train": 1701, "validation": 2903, "evaluation_holdout": 4307}

APPLICATIONS = {
    "calculator": ("калькулятор", "калькуляторы", "calc"),
    "notepad": ("блокнот", "блокноты", "блакнот", "блекнот", "notepad", "black note"),
    "explorer": ("проводник", "файлы", "explorer", "file explorer"),
    "paint": ("paint", "пейнт", "паинт", "пайнт", "пеинт", "пэйнт"),
    "discord": ("discord", "дискорд", "дисорд", "дискод", "дискор"),
    "task_manager": ("диспетчер задач", "task manager"),
    "browser": ("браузер", "browser", "интернет"),
}

OPEN_TEMPLATES = {
    "train": (
        "открой {application}", "запусти {application}",
        "открой мне {application}", "запусти мне {application}",
        "открой приложение {application}", "запусти программу {application}",
        "включи {application}", "включи пожалуйста {application}",
        "пожалуйста открой {application}", "пожалуйста запусти {application}",
        "джарвис открой {application}", "джарвис запусти {application}",
        "открой пожалуйста {application}", "запусти пожалуйста {application}",
        "давай откроем {application}", "давай запустим {application}",
        "хочу открыть {application}", "мне нужно открыть {application}",
        "нужно запустить {application}", "можешь запустить {application}",
        "open {application}", "launch {application}",
    ),
    "validation": (
        "будь добр открой {application}", "можно запустить {application}",
        "открой для меня {application}", "включи мне приложение {application}",
        "я хочу чтобы ты открыл {application}", "джарвис включи {application}",
    ),
    "evaluation_holdout": (
        "пора открыть {application}", "прошу запустить {application}",
        "открой-ка {application}", "мне сейчас нужен {application}",
        "давай включим {application}",
    ),
}

REMINDER_TEMPLATES = {
    "train": (
        "через {duration} минут напомни {message}",
        "напомни через {duration} минут {message}",
        "через {duration} минут скажи мне {message}",
        "поставь напоминание через {duration} минут {message}",
        "создай напоминание на {duration} минут {message}",
        "спустя {duration} минут напомни мне {message}",
        "на {duration} минут поставь напоминание {message}",
        "джарвис через {duration} минут напомни {message}",
        "пожалуйста напомни через {duration} минут {message}",
        "не дай забыть через {duration} минут {message}",
    ),
    "validation": (
        "через {duration} минут сообщи что нужно {message}",
        "установи напоминание спустя {duration} минут {message}",
        "хочу напоминание через {duration} минут {message}",
        "отсчитай {duration} минут и напомни {message}",
        "напомни мне через {duration} минут что пора {message}",
    ),
    "evaluation_holdout": (
        "когда пройдёт {duration} минут напомни {message}",
        "поставь на {duration} минут напоминание чтобы {message}",
        "через {duration} минут не забудь сказать {message}",
        "мне надо через {duration} минут вспомнить {message}",
    ),
}

DURATIONS = {
    "train": ("1", "2", "3", "5", "7", "10", "12", "15", "20", "25", "30", "45", "60"),
    "validation": ("4", "8", "14", "19", "35", "50"),
    "evaluation_holdout": ("6", "11", "17", "27", "40", "55"),
}

MESSAGES = {
    "train": (
        "выпить воды", "проверить чайник", "выключить духовку",
        "закрыть окно", "позвонить другу", "ответить коллеге",
        "встать и размяться", "проверить почту", "отправить отчёт",
        "покормить кота", "забрать документы", "проверить загрузку",
        "начать встречу", "поставить телефон на зарядку",
        "достать бельё", "сделать короткий перерыв",
    ),
    "validation": (
        "проверить замок", "полить цветы", "выключить утюг",
        "позвонить родителям", "достать ключи", "проверить окна",
        "подготовиться к звонку", "забрать заказ",
    ),
    "evaluation_holdout": (
        "снять кастрюлю с плиты", "дать лекарство собаке",
        "открыть дверь курьеру", "сохранить рабочий файл",
        "проверить заряд батареи", "вынуть еду из духовки",
    ),
}

TIME_PARTS = {
    "train": (
        ("скажи", "который сейчас час"), ("подскажи", "сколько сейчас времени"),
        ("сообщи", "текущее время"), ("назови", "точное время"),
        ("проверь", "что сейчас на часах"), ("покажи", "время на данный момент"),
        ("можешь сказать", "который час"), ("хочу узнать", "сколько времени"),
        ("джарвис скажи", "время сейчас"), ("пожалуйста уточни", "который час"),
        ("мне нужно знать", "текущее время"), ("посмотри", "сколько времени на часах"),
    ),
    "validation": (
        ("будь добр скажи", "время прямо сейчас"),
        ("можно узнать", "который теперь час"),
        ("глянь пожалуйста", "что показывают часы"),
        ("сообщи без лишних слов", "точное время"),
        ("проверь для меня", "сколько времени сейчас"),
        ("джарвис уточни", "текущий час"),
    ),
    "evaluation_holdout": (
        ("сориентируй", "по текущему времени"),
        ("озвучь", "показания часов"),
        ("напомни", "какой сейчас час"),
        ("хотелось бы узнать", "время в эту минуту"),
        ("можешь проверить", "час на компьютере"),
    ),
}

TIME_ENDINGS = {
    "train": ("", " пожалуйста", " джарвис", " на компьютере", " по местному времени", " без даты", " точно", " сейчас", " для меня", " одним предложением", " и ответь голосом"),
    "validation": ("", " пожалуйста", " в моём часовом поясе", " без объяснений", " на системных часах", " джарвис"),
    "evaluation_holdout": ("", " пожалуйста", " именно сейчас", " коротко", " на этом компьютере"),
}

LIST_PARTS = {
    "train": (
        "какие приложения ты можешь открыть", "перечисли доступные приложения",
        "покажи список разрешённых программ", "что из программ можно запустить",
        "назови поддерживаемые приложения", "какие программы тебе доступны",
        "что ты умеешь открывать", "покажи варианты для запуска",
        "дай список приложений", "какой софт ты можешь включить",
        "что доступно из программ", "перечисли свой список программ",
    ),
    "validation": (
        "какие программы входят в белый список", "покажи весь доступный софт",
        "что можно открыть через джарвис", "назови программы для быстрого запуска",
        "какие приложения поддерживает ассистент", "выведи перечень доступных программ",
    ),
    "evaluation_holdout": (
        "с чем из приложений ты работаешь", "какие программы у тебя подключены",
        "покажи меню доступных приложений", "что умеешь запускать на этом компьютере",
        "озвучь список разрешённого софта",
    ),
}

LIST_ENDINGS = {
    "train": ("", " пожалуйста", " джарвис", " на моём компьютере", " прямо сейчас", " для запуска", " коротко", " полностью", " без лишнего текста", " голосом", " одним списком"),
    "validation": ("", " пожалуйста", " целиком", " на данный момент", " джарвис", " для меня"),
    "evaluation_holdout": ("", " пожалуйста", " без пояснений", " сейчас", " одним ответом"),
}

CANCEL_PARTS = {
    "train": (
        "отмени текущую команду", "останови выполнение", "прерви эту операцию",
        "ничего не запускай", "я передумал", "не надо это делать",
        "сбрось текущее действие", "прекрати выполнение команды",
        "стоп джарвис", "отставить команду", "не открывай дискорд",
        "не запускай браузер", "остановись", "отмена",
    ),
    "validation": (
        "прерви всё что сейчас делаешь", "сними последнюю команду",
        "пожалуйста останови операцию", "я отменяю запрос",
        "не продолжай выполнение", "передумал открывать приложение",
    ),
    "evaluation_holdout": (
        "давай отменим это действие", "вернись в режим ожидания",
        "сейчас же прекрати команду", "не выполняй мой прошлый запрос",
        "закрой текущую операцию",
    ),
}

CANCEL_ENDINGS = {
    "train": ("", " пожалуйста", " прямо сейчас", " немедленно", " джарвис", " я передумал", " и вернись в ожидание", " без подтверждения", " это ошибка", " пока не нужно"),
    "validation": ("", " пожалуйста", " сейчас", " джарвис", " я ошибся", " немедленно"),
    "evaluation_holdout": ("", " пожалуйста", " в эту секунду", " пока что", " джарвис"),
}

CHAT_FRAMES = {
    "train": (
        "расскажи про {topic}", "объясни простыми словами {topic}",
        "что ты думаешь про {topic}", "помоги разобраться в теме {topic}",
        "почему важно понимать {topic}", "давай поговорим про {topic}",
        "можешь кратко описать {topic}", "хочу узнать больше про {topic}",
    ),
    "validation": (
        "поделись интересным фактом про {topic}", "как новичку понять {topic}",
        "объясни мне принцип работы {topic}", "какое твоё мнение насчёт {topic}",
        "с чего начать изучение темы {topic}",
    ),
    "evaluation_holdout": (
        "проведи небольшой ликбез про {topic}", "расскажи что полезно знать о {topic}",
        "можем обсудить {topic}", "как устроено {topic}",
    ),
}

CHAT_TOPICS = {
    "train": (
        "машинное обучение", "нейронные сети", "космос", "программирование",
        "компьютерные игры", "роботы", "интернет", "безопасность данных",
        "операционные системы", "голосовые ассистенты", "погоду и климат",
        "историю компьютеров", "музыку", "фильмы", "здоровый сон",
        "изучение английского", "правильное питание", "работу процессора",
        "видеокарты", "локальные модели", "почему дискорд иногда не открывается",
        "как выбрать браузер по умолчанию", "назначение калькулятора",
        "разницу между файлами и папками",
    ),
    "validation": (
        "квантовые компьютеры", "солнечную систему", "языки программирования",
        "цифровую приватность", "архитектуру нейросетей", "домашние сети",
        "энергопотребление компьютера", "облачные технологии",
    ),
    "evaluation_holdout": (
        "обучение с подкреплением", "устройство микрофона", "историю интернета",
        "защиту паролей", "автоматизацию дома", "компьютерное зрение",
    ),
}

UNKNOWN_FRAGMENTS = {
    "train": (
        "открой", "запусти", "напомни", "через несколько минут", "сделай это",
        "там потом", "ну вот это", "не знаю", "какая-то команда",
        "приложение без названия", "неразборчивые слова", "эээ ммм",
        "что-нибудь такое", "туда и обратно", "один два три",
        "команда без объекта", "действие без деталей", "бла бла бла",
        "продолжай туда", "выполни штуку", "непонятный шум",
        "открой правый", "включи эту самую", "потом сделай как вчера",
    ),
    "validation": (
        "ну это самое сделай", "открой что-нибудь", "напомни когда-нибудь",
        "запусти ту программу", "не расслышал сам себя", "случайный набор слов",
        "ээ дальше туда", "команда потерялась",
    ),
    "evaluation_holdout": (
        "сделай как обычно", "включи неизвестно что", "потом напомни",
        "вот та штука", "слова без смысла", "открой этот",
    ),
}

UNKNOWN_WRAPPERS = {
    "train": ("{fragment}", "эм {fragment}", "джарвис {fragment}", "ну {fragment}", "{fragment} пожалуйста", "кажется {fragment}"),
    "validation": ("{fragment}", "ээ {fragment}", "в общем {fragment}", "{fragment} наверное"),
    "evaluation_holdout": ("{fragment}", "мм {fragment}", "короче {fragment}"),
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _record(text: str, intent: str, slots: dict[str, str] | None = None) -> dict[str, Any]:
    return {"text": _clean(text), "intent": intent, "slots": slots or {}}


def _simple_candidates(split: str) -> dict[str, list[dict[str, Any]]]:
    time = [
        _record(f"{start} {subject}{ending}", "get_current_time")
        for start, subject in TIME_PARTS[split]
        for ending in TIME_ENDINGS[split]
    ]
    applications = [
        _record(f"{phrase}{ending}", "list_applications")
        for phrase in LIST_PARTS[split]
        for ending in LIST_ENDINGS[split]
    ]
    cancel = [
        _record(f"{phrase}{ending}", "cancel")
        for phrase in CANCEL_PARTS[split]
        for ending in CANCEL_ENDINGS[split]
    ]
    chat = [
        _record(frame.format(topic=topic), "general_chat")
        for frame in CHAT_FRAMES[split]
        for topic in CHAT_TOPICS[split]
    ]
    unknown = [
        _record(wrapper.format(fragment=fragment), "unknown")
        for wrapper in UNKNOWN_WRAPPERS[split]
        for fragment in UNKNOWN_FRAGMENTS[split]
    ]
    return {
        "get_current_time": time,
        "list_applications": applications,
        "cancel": cancel,
        "general_chat": chat,
        "unknown": unknown,
    }


def _select_open_candidates(
    split: str,
    *,
    count: int,
    seed: int,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    """Select equal coverage per allow-listed application, not per alias."""
    selected: list[dict[str, Any]] = []
    base_quota, remainder = divmod(count, len(APPLICATIONS))
    for app_index, aliases in enumerate(APPLICATIONS.values()):
        quota = base_quota + (app_index < remainder)
        candidates = [
            _record(
                template.format(application=application),
                "open_application",
                {"application": application},
            )
            for template in OPEN_TEMPLATES[split]
            for application in aliases
        ]
        selected.extend(
            _select(
                candidates,
                count=quota,
                seed=seed + app_index * 37,
                forbidden=forbidden,
            )
        )
    random.Random(seed).shuffle(selected)
    return selected


def _reminder_candidates(split: str) -> list[dict[str, Any]]:
    return [
        _record(
            template.format(duration=duration, message=message),
            "set_reminder",
            {"duration": duration, "reminder_text": message},
        )
        for template in REMINDER_TEMPLATES[split]
        for duration in DURATIONS[split]
        for message in MESSAGES[split]
    ]


def _base_texts() -> set[str]:
    return {
        example.text.casefold()
        for split in ("train", "validation", "test")
        for example in build_examples(split)
    }


def _select(
    candidates: Iterable[dict[str, Any]],
    *,
    count: int,
    seed: int,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for record in candidates:
        key = record["text"].casefold()
        if key not in forbidden:
            unique.setdefault(key, record)
    values = list(unique.values())
    random.Random(seed).shuffle(values)
    if len(values) < count:
        raise ValueError(f"only {len(values)} unique candidates available; need {count}")
    selected = values[:count]
    forbidden.update(record["text"].casefold() for record in selected)
    return selected


def build() -> dict[str, list[dict[str, Any]]]:
    forbidden = _base_texts()
    result: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "evaluation_holdout"):
        by_intent = _simple_candidates(split)
        by_intent["set_reminder"] = _reminder_candidates(split)
        records: list[dict[str, Any]] = []
        for intent_index, intent in enumerate(INTENTS):
            seed = SEEDS[split] + intent_index * 101
            if intent == "open_application":
                selected = _select_open_candidates(
                    split,
                    count=TARGETS[split],
                    seed=seed,
                    forbidden=forbidden,
                )
            else:
                selected = _select(
                    by_intent[intent],
                    count=TARGETS[split],
                    seed=seed,
                    forbidden=forbidden,
                )
            records.extend(selected)
        random.Random(SEEDS[split]).shuffle(records)
        result[split] = records
    return result


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = build()
    manifest: dict[str, Any] = {
        "version": 2,
        "generator": "training_workspace.build_dataset",
        "external_sources": False,
        "splits": {},
    }
    for split, records in splits.items():
        filename = f"{split}.jsonl"
        content = _jsonl(records)
        (DATA_DIR / filename).write_text(content, encoding="utf-8", newline="\n")
        manifest["splits"][split] = {
            "file": filename,
            "examples": len(records),
            "intents": dict(sorted(Counter(record["intent"] for record in records).items())),
            "sha256": _sha256(content),
        }
    (DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    print(json.dumps(write_dataset(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
