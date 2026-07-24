"""Own, deterministic training corpus for the first Jarvis NLU baseline.

No external datasets or pretrained tokenizers are used.  Examples are built
from independently curated train/validation/test phrase families.  Keeping
templates separate by split prevents an optimistic score caused by the same
sentence pattern appearing in both training and evaluation data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Example:
    text: str
    intent: str
    spans: tuple[Span, ...] = ()


_SIMPLE: dict[str, dict[str, tuple[str, ...]]] = {
    "train": {
        "get_current_time": (
            "который сейчас час", "сколько сейчас времени", "скажи время",
            "подскажи текущее время", "время сейчас", "который час джарвис",
            "можешь назвать время", "what is the time",
            "покажи время на часах", "сообщи текущее время",
            "назови время пожалуйста", "хочу проверить который час",
            "время на данный момент", "tell me the current time",
            "time now please", "current time please", "дай точное время",
            "проверь сколько времени", "сообщи который теперь час",
            "сколько времени прямо сейчас", "глянь на часы",
            "хочу узнать текущее время", "хочу знать который час",
        ),
        "list_applications": (
            "какие приложения ты можешь открыть",
            "покажи список доступных приложений",
            "что из программ тебе доступно",
            "перечисли приложения",
            "какие программы поддерживаются",
            "list applications",
            "что можно запускать",
            "покажи доступные программы",
            "что ты умеешь открывать",
            "назови программы для запуска",
            "список приложений джарвис",
            "what applications are available",
            "что доступно для запуска", "покажи возможности запуска программ",
            "что умеешь запускать из программ", "перечень разрешенных приложений",
            "какие программы можно включить", "show supported applications",
        ),
        "cancel": (
            "отмена", "отмени это", "остановись", "прекрати выполнение",
            "не надо", "забудь", "cancel", "стоп джарвис",
            "прерви операцию", "хватит это делать", "перестань выполнять",
            "сбрось текущую команду", "отставить", "stop the action",
            "прерви текущее действие", "хватит выполнять команду",
            "перестань пожалуйста", "сбрось операцию", "останови выполнение",
            "abort the command",
            "отменить команду", "отменяю", "стоп", "прекрати",
            "ничего не делай", "не запускай", "я передумал",
            "останови это", "не делай этого", "abort current task",
            "terminate the action", "please stop", "сними задачу",
        ),
        "general_chat": (
            "привет", "как твои дела", "расскажи интересный факт",
            "объясни что такое нейронная сеть", "почему небо синее",
            "давай поговорим", "кто ты", "hello there",
            "расскажи про технологии", "можешь мне помочь",
            "объясни простыми словами", "tell me something interesting",
            "как устроены приложения", "расскажи про браузеры",
            "зачем нужен калькулятор", "как работает проводник",
            "почему время быстро идет", "что такое диспетчер задач",
            "что ты вообще умеешь", "опиши свои возможности",
            "на что ты способен", "помоги придумать идею",
            "расскажи шутку", "как начать программировать",
            "что ты думаешь о роботах", "поговорим о космосе",
        ),
        "unknown": (
            "эээ", "неразборчивый шум", "что-нибудь", "дальше",
            "оно там", "сделай как вчера", "не знаю", "один два три",
            "непонятно что сказано", "неопределённая команда",
            "открыть", "запустить", "приложение", "время", "напомнить",
            "команда без продолжения", "неполная фраза",
            "эм", "ну", "как бы", "это", "там", "сюда", "потом",
            "сделай какую-нибудь штуку", "что-то такое", "туда сюда",
            "ничего не расслышал", "бла бла", "какая-то команда",
            "команда без объекта", "действие без объекта", "через какое-то время",
            "непонятный набор слов", "повтори команду", "действие без деталей",
        ),
    },
    "validation": {
        "get_current_time": (
            "назови который час", "хочу узнать время", "сколько времени на часах",
        ),
        "list_applications": (
            "что ты умеешь запускать", "назови доступные программы",
            "какие приложения можно открыть",
        ),
        "cancel": (
            "отставить команду", "перестань это делать", "сбрось действие",
        ),
        "general_chat": (
            "расскажи про космос", "чем ты можешь помочь", "как работает интернет",
        ),
        "unknown": (
            "вот это самое", "продолжай туда", "непонятная команда",
        ),
    },
    "test": {
        "get_current_time": (
            "покажи мне текущее время", "что сейчас на часах",
            "джарвис сообщи время", "time please",
        ),
        "list_applications": (
            "покажи что можно запустить", "список программ для открытия",
            "что из приложений ты открываешь", "available applications",
        ),
        "cancel": (
            "отменяй", "хватит выполнять", "прерви текущую задачу", "stop it",
        ),
        "general_chat": (
            "поговори со мной", "объясни квантовую физику", "что ты умеешь",
            "tell me a story",
        ),
        "unknown": (
            "это самое там", "ну давай уже", "абракадабра", "не расслышал",
        ),
    },
}

_REMINDER_TEMPLATES = {
    "train": (
        "напомни через {minutes} минут {message}",
        "поставь напоминание через {minutes} минут {message}",
        "через {minutes} минут напомни {message}",
        "создай напоминание на {minutes} минут: {message}",
        "не дай забыть через {minutes} минут {message}",
        "remind me in {minutes} minutes to {message}",
        "напомни мне через {minutes} минут про {message}",
        "установи напоминание спустя {minutes} минут {message}",
        "через {minutes} минут сообщи {message}",
        "запланируй напоминание на {minutes} минут {message}",
        "поставь таймер на {minutes} минут и напомни {message}",
    ),
    "validation": (
        "напомни мне спустя {minutes} минут {message}",
        "установи таймер-напоминание на {minutes} минут {message}",
    ),
    "test": (
        "через {minutes} минут скажи мне {message}",
        "сделай напоминание через {minutes} минут о том чтобы {message}",
        "не забудь напомнить через {minutes} минут {message}",
    ),
}

_APPLICATION_TEMPLATES = {
    "train": (
        "открой {application}",
        "запусти {application}",
        "открой приложение {application}",
        "джарвис запусти {application}",
        "можешь открыть {application}",
        "open {application}",
        "включи {application}",
        "я хочу открыть {application}",
        "открой мне программу {application}",
        "запусти программу {application}",
        "начни работу с {application}",
        "мне нужен {application}",
        "давай откроем {application}",
        "launch application {application}",
        "хочу открыть программу {application}",
        "нужно запустить {application}",
        "включи программу {application}",
        "пожалуйста запусти {application}",
        "открой пожалуйста {application}",
        "давай запустим программу {application}",
        "мне нужно открыть {application}",
        "start application {application}",
    ),
    "validation": (
        "пожалуйста открой {application}",
        "хочу запустить {application}",
        "включи приложение {application}",
    ),
    "test": (
        "открой мне {application}",
        "давай запустим {application}",
        "мне нужно приложение {application}",
        "launch {application}",
    ),
}

_APPLICATION_VALUES = {
    "train": ("калькулятор", "блокнот", "проводник", "paint", "диспетчер задач", "браузер"),
    "validation": ("calc", "notepad", "файлы", "пейнт", "task manager", "интернет"),
    "test": ("калькулятор", "notepad", "проводник", "паинт", "диспетчер задач", "browser"),
}

_VALUES = {
    "train": {
        "minutes": ("1", "2", "3", "5", "7", "10", "15", "20", "30", "45"),
        "message": (
            "встать и размяться", "проверить почту", "выключить духовку",
            "позвонить маме", "начать встречу", "выпить воды",
            "проверить загрузку", "отправить отчёт",
            "закончить работу", "сделать короткий перерыв",
        ),
    },
    "validation": {
        "minutes": ("4", "12", "25"),
        "message": ("закрыть окно", "ответить коллеге", "сделать перерыв"),
    },
    "test": {
        "minutes": ("6", "18", "40"),
        "message": ("проверить чайник", "подготовиться к звонку", "покормить кота"),
    },
}

_PREFIXES = ("", "пожалуйста ", "джарвис ")
_SUFFIXES = ("", " пожалуйста", " джарвис")


def _render_reminder(template: str, minutes: str, message: str, prefix: str = "") -> Example:
    text = prefix + template.format(minutes=minutes, message=message)
    duration_start = text.index(minutes, len(prefix))
    message_start = text.index(message, duration_start + len(minutes))
    return Example(
        text=text,
        intent="set_reminder",
        spans=(
            Span(duration_start, duration_start + len(minutes), "duration"),
            Span(message_start, message_start + len(message), "reminder_text"),
        ),
    )


def _render_application(template: str, application: str) -> Example:
    text = template.format(application=application)
    start = text.index(application)
    return Example(
        text=text,
        intent="open_application",
        spans=(Span(start, start + len(application), "application"),),
    )


def build_examples(split: str, *, augmented: bool = False, seed: int = 17) -> list[Example]:
    """Return a deterministic split; augmentation is training-only."""
    if split not in _SIMPLE:
        raise ValueError(f"unknown split: {split}")
    if augmented and split != "train":
        raise ValueError("augmentation is allowed only for the training split")

    examples: list[Example] = []
    for intent, phrases in _SIMPLE[split].items():
        for phrase in phrases:
            examples.append(Example(phrase, intent))
            if split == "train":
                examples.append(Example(f"пожалуйста {phrase}", intent))
                examples.append(Example(f"{phrase} джарвис", intent))

    values = _VALUES[split]
    for template in _REMINDER_TEMPLATES[split]:
        for minutes in values["minutes"]:
            for message in values["message"]:
                examples.append(_render_reminder(template, minutes, message))
                if split == "train":
                    examples.append(_render_reminder(template, minutes, message, "пожалуйста "))

    for template in _APPLICATION_TEMPLATES[split]:
        for application in _APPLICATION_VALUES[split]:
            examples.append(_render_application(template, application))

    if augmented:
        rng = random.Random(seed)
        augmented_examples: list[Example] = []
        for example in examples:
            # Safe STT-style augmentation: case and punctuation changes keep
            # character offsets stable; doubled spaces are avoided for spans.
            if not example.spans and rng.random() < 0.7:
                text = example.text
                variant = text.upper() if rng.random() < 0.25 else text
                if rng.random() < 0.6:
                    variant += rng.choice(("?", "!", "."))
                augmented_examples.append(Example(variant, example.intent))
            elif example.spans and rng.random() < 0.35:
                prefix = rng.choice(_PREFIXES)
                shifted = tuple(
                    Span(s.start + len(prefix), s.end + len(prefix), s.label)
                    for s in example.spans
                )
                augmented_examples.append(
                    Example(prefix + example.text, example.intent, shifted)
                )
        examples.extend(augmented_examples)

    # Stable shuffle prevents batches from being grouped by class.
    random.Random(seed).shuffle(examples)
    return examples


def iter_text(examples: Iterable[Example]) -> Iterable[str]:
    for example in examples:
        yield example.text
