"""Build train-only family augmentation for the Structured JSC bottlenecks.

Only the public train and validation splits are read.  The migration suite,
test and evaluation_holdout are not inputs to this generator.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.data import (
    DATA_SCHEMA_VERSION,
    DialogueTurn,
    JSCExample,
    load_jsc_jsonl,
    validate_jsc_splits,
)
from ml.jsc.jal import DialogueAct, JALPlan, MissingSlot, ToolCall, dumps
from ml.jsc.project_registry import build_project_schema_registry


SOURCE = Path("training_workspace/jsc_data")
OUTPUT = Path("training_workspace/jsc_structured_data")
SEED = 73_019
APPS = (
    ("calculator", "калькулятор"),
    ("notepad", "блокнот"),
    ("explorer", "проводник"),
    ("paint", "пейнт"),
    ("discord", "дискорд"),
    ("visual_studio_code", "вс код"),
    ("telegram", "телеграм"),
    ("browser", "браузер"),
)


def _execute(*calls: ToolCall) -> JALPlan:
    return JALPlan(DialogueAct.EXECUTE, steps=tuple(calls))


def _example(
    index: int,
    category: str,
    text: str,
    target: JALPlan,
    family: str,
    *,
    history: tuple[DialogueTurn, ...] = (),
    state: JALPlan | None = None,
) -> JSCExample:
    return JSCExample(
        scenario_id=f"train.structured_aug.{index:05d}",
        split="train",
        family_id=f"structured_aug.{family}",
        category=category,
        history=history,
        text=text,
        state=state,
        target=target,
        metadata={"synthetic": True, "structured_augmentation": True},
    )


def _action_bank() -> list[tuple[str, ToolCall]]:
    rows: list[tuple[str, ToolCall]] = []
    for canonical, spoken in APPS:
        rows.extend(
            (
                (
                    f"включи для меня {spoken}",
                    ToolCall("open_application", {"application": canonical}),
                ),
                (
                    f"поработаем в {spoken}, запусти его",
                    ToolCall("open_application", {"application": canonical}),
                ),
                (
                    f"выведи на экран {spoken}",
                    ToolCall("open_application", {"application": canonical}),
                ),
                (
                    f"убери окно {spoken}",
                    ToolCall("window_control", {"action": "close", "window": canonical}),
                ),
                (
                    f"заверши именно {spoken}",
                    ToolCall("window_control", {"action": "close", "window": canonical}),
                ),
                (
                    f"это окно {spoken} можно закрыть",
                    ToolCall("window_control", {"action": "close", "window": canonical}),
                ),
            )
        )
    rows.extend(
        (
            ("сообщи текущее время", ToolCall("get_current_time")),
            ("перечисли программы которые доступны", ToolCall("list_applications")),
            ("покажи список напоминаний", ToolCall("list_reminders")),
            ("создай новую вкладку", ToolCall("browser_control", {"action": "new_tab"})),
            ("перейди на следующую вкладку", ToolCall("browser_control", {"action": "next_tab"})),
            ("верни предыдущую вкладку", ToolCall("browser_control", {"action": "previous_tab"})),
            ("прибавь громкость", ToolCall("system_control", {"action": "volume_up"})),
            ("убавь громкость", ToolCall("system_control", {"action": "volume_down"})),
            ("поставь звук на паузу", ToolCall("system_control", {"action": "media_play_pause"})),
            ("включи управление жестами", ToolCall("gesture_mode", {"action": "enable"})),
            ("отключи управление жестами", ToolCall("gesture_mode", {"action": "disable"})),
            ("узнай состояние жестов", ToolCall("gesture_mode", {"action": "status"})),
            (
                "найди в сети расписание автобусов",
                ToolCall(
                    "browser_control",
                    {"action": "search", "query": "расписание автобусов"},
                ),
            ),
            (
                "отыщи в интернете прогноз погоды",
                ToolCall(
                    "browser_control",
                    {"action": "search", "query": "прогноз погоды"},
                ),
            ),
            (
                "через 7 минут напомни сделать разминку",
                ToolCall("set_reminder", {"minutes": 7, "message": "сделать разминку"}),
            ),
            (
                "через 25 минут напомни проверить чайник",
                ToolCall("set_reminder", {"minutes": 25, "message": "проверить чайник"}),
            ),
            (
                "поставь напоминание на 40 минут позвонить домой",
                ToolCall("set_reminder", {"minutes": 40, "message": "позвонить домой"}),
            ),
            (
                "отмени напоминание номер 31",
                ToolCall("cancel_reminder", {"reminder_id": 31}),
            ),
        )
    )
    return rows


def build_augmentations() -> list[JSCExample]:
    rng = random.Random(SEED)
    rows: list[JSCExample] = []
    signatures: set[str] = set()

    def add(
        category: str,
        text: str,
        target: JALPlan,
        family: str,
        *,
        history: tuple[DialogueTurn, ...] = (),
        state: JALPlan | None = None,
    ) -> None:
        candidate = _example(
            len(rows), category, text, target, family, history=history, state=state
        )
        if candidate.input_signature not in signatures:
            signatures.add(candidate.input_signature)
            rows.append(candidate)

    bank = _action_bank()
    single_wrappers = (
        "джарвис, {action}",
        "сделай пожалуйста: {action}",
        "сейчас нужно вот что — {action}",
        "выполни одну задачу: {action}",
    )
    for wrapper_index, wrapper in enumerate(single_wrappers):
        for action_index, (text, call) in enumerate(bank):
            add(
                "single",
                wrapper.format(action=text),
                _execute(call),
                f"single.wrapper_{wrapper_index}.action_{action_index}",
            )

    connectors = (" и затем ", ", после чего ", "; потом ", ", а следом ")
    compound_prefixes = (
        "выполни цепочку: ",
        "за один подход сделай так: ",
        "у меня несколько задач: ",
        "действуй последовательно: ",
        "вот полный порядок: ",
    )
    for step_count in range(2, 6):
        for index in range(180):
            selected = rng.sample(bank, step_count)
            connector = connectors[(index + step_count) % len(connectors)]
            prefix = compound_prefixes[(index // len(connectors)) % len(compound_prefixes)]
            add(
                "compound",
                prefix + connector.join(text for text, _call in selected),
                _execute(*(call for _text, call in selected)),
                f"compound.steps_{step_count}.layout_{index % 20}",
            )

    close_state = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("window_control", {"action": "close"}),),
        missing=(MissingSlot(0, "window"),),
        reason="missing_window",
    )
    open_state = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("open_application"),),
        missing=(MissingSlot(0, "application"),),
        reason="missing_application",
    )
    prompts = (
        "заверши одно из приложений",
        "закрой нужное окно",
        "убери программу с экрана",
    )
    for app_index, (canonical, spoken) in enumerate(APPS):
        for prompt_index, prompt in enumerate(prompts):
            history = (
                DialogueTurn("user", prompt),
                DialogueTurn("jarvis", "Уточните название приложения."),
            )
            add(
                "multi_turn",
                f"я имею в виду {spoken}",
                _execute(
                    ToolCall("window_control", {"action": "close", "window": canonical})
                ),
                f"multi.close.prompt_{prompt_index}",
                history=history,
                state=close_state,
            )
            add(
                "multi_turn",
                f"нужен именно {spoken}",
                _execute(ToolCall("open_application", {"application": canonical})),
                f"multi.open.prompt_{prompt_index}",
                history=(
                    DialogueTurn("user", "запусти приложение"),
                    DialogueTurn("jarvis", "Какое приложение открыть?"),
                ),
                state=open_state,
            )

    messages = (
        "проверить документы",
        "ответить руководителю",
        "выключить духовку",
        "сделать короткий перерыв",
        "забрать посылку",
        "позвонить в сервис",
        "оплатить интернет",
        "подготовить заметки",
    )
    ask_frames = (
        "создай напоминание: {message}",
        "не разреши забыть {message}",
        "мне понадобится напоминание {message}",
        "запиши задачу {message} и напомни",
    )
    for message_index, message in enumerate(messages):
        for frame_index, frame in enumerate(ask_frames):
            add(
                "multi_turn",
                frame.format(message=message),
                JALPlan(
                    DialogueAct.ASK,
                    steps=(ToolCall("set_reminder", {"message": message}),),
                    missing=(MissingSlot(0, "minutes"),),
                    reason="missing_time",
                ),
                f"multi.ask.frame_{frame_index}.message_{message_index}",
            )
        pending = JALPlan(
            DialogueAct.ASK,
            steps=(ToolCall("set_reminder", {"message": message}),),
            missing=(MissingSlot(0, "minutes"),),
            reason="missing_time",
        )
        for minutes in (6, 12, 18, 35, 50):
            add(
                "multi_turn",
                f"давай через {minutes} минут",
                _execute(
                    ToolCall("set_reminder", {"minutes": minutes, "message": message})
                ),
                f"multi.reminder_fill.message_{message_index}",
                history=(
                    DialogueTurn("user", f"напомни {message}"),
                    DialogueTurn("jarvis", "Через сколько минут напомнить?"),
                ),
                state=pending,
            )

    files = ("черновик", "старый отчёт", "временная таблица", "копия презентации")
    for file_index, name in enumerate(files):
        path = f"документы/{name}.txt"
        for frame_index, frame in enumerate(
            ("удали документ {name}", "отправь в корзину файл {name}")
        ):
            add(
                "multi_turn",
                frame.format(name=name),
                JALPlan(
                    DialogueAct.CONFIRM,
                    steps=(
                        ToolCall("file_control", {"action": "delete", "path": path}),
                    ),
                    reason="user_confirmation",
                ),
                f"multi.confirm.frame_{frame_index}.file_{file_index}",
            )

    negative_frames = (
        "объясни как устроено приложение {app}",
        "что умеет программа {app}",
        "почему люди иногда закрывают {app}",
        "сравни открытие и закрытие окна {app}",
        "произнеси название {app}, но команду не выполняй",
        "можно ли пользоваться {app} без запуска",
    )
    for repeat in range(4):
        for app_index, (_canonical, spoken) in enumerate(APPS):
            for frame_index, frame in enumerate(negative_frames):
                add(
                    "hard_negative",
                    ("пожалуйста, " if repeat % 2 else "")
                    + frame.format(app=spoken)
                    + (" сегодня" if repeat >= 2 else ""),
                    JALPlan(DialogueAct.DIALOGUE, reason="general_chat"),
                    f"negative.frame_{frame_index}.repeat_{repeat}",
                )

    ood_objects = (
        "кофеварку",
        "ворота гаража",
        "кондиционер",
        "настольную лампу",
        "робот пылесос",
        "телевизор",
    )
    ood_frames = (
        "включи {item}",
        "настрой {item}",
        "выключи {item}",
        "подключись к {item}",
    )
    for item_index, item in enumerate(ood_objects):
        for frame_index, frame in enumerate(ood_frames):
            for suffix_index, suffix in enumerate((" сейчас", " пожалуйста", " немедленно")):
                add(
                    "ood",
                    frame.format(item=item) + suffix,
                    JALPlan(DialogueAct.REJECT, reason="out_of_scope"),
                    f"ood.frame_{frame_index}.suffix_{suffix_index}.item_{item_index}",
                )

    for index in range(36):
        add(
            "single",
            f"отмени эту команду вариант {index + 1}",
            JALPlan(DialogueAct.CANCEL),
            f"cancel.variant_{index}",
        )

    # Voice-derived v2 curriculum. These templates come from observed failure
    # families, not from migration/test/holdout files. They deliberately teach
    # the initial ASK turn as well as the slot-filling answer; the previous
    # augmentation mostly contained only the second half of application
    # clarifications, which made the offline multi-turn score misleading.
    open_prompts = (
        "открой приложение",
        "открой нужное приложение",
        "запусти программу",
        "запусти нужную программу",
        "мне нужно открыть приложение",
        "надо запустить одну программу",
    )
    close_prompts = (
        "закрой приложение",
        "закрой нужное окно",
        "заверши нужную программу",
        "нужно закрыть одно окно",
        "мне надо завершить приложение",
        "убери программу с экрана",
    )
    for prompt_index, prompt in enumerate(open_prompts):
        add(
            "multi_turn",
            prompt,
            open_state,
            f"voice.ask.open.prompt_{prompt_index}",
        )
    for prompt_index, prompt in enumerate(close_prompts):
        add(
            "multi_turn",
            prompt,
            close_state,
            f"voice.ask.close.prompt_{prompt_index}",
        )

    app_surfaces = {
        "calculator": ("калькулятор", "калькулятором"),
        "notepad": ("блокнот", "блокнотом"),
        "explorer": ("проводник", "проводником"),
        "paint": ("пейнт", "paint"),
        "discord": ("дискорд", "discord"),
        "visual_studio_code": (
            "вэскод",
            "вс код",
            "visual studio code",
            "вижуал студио код",
            "вест код",
            "вес код",
        ),
        "telegram": ("телеграм", "telegram"),
        "browser": ("браузер", "интернет браузер"),
    }
    answer_frames = ("{app}", "это {app}", "нужен {app}", "я имею в виду {app}")
    for canonical, surfaces in app_surfaces.items():
        for surface_index, surface in enumerate(surfaces):
            for frame_index, frame in enumerate(answer_frames):
                answer = frame.format(app=surface)
                add(
                    "multi_turn",
                    answer,
                    _execute(ToolCall("open_application", {"application": canonical})),
                    f"voice.fill.open.{canonical}.surface_{surface_index}.frame_{frame_index}",
                    history=(
                        DialogueTurn("user", open_prompts[(surface_index + frame_index) % len(open_prompts)]),
                        DialogueTurn("jarvis", "Какое приложение открыть?"),
                    ),
                    state=open_state,
                )
                add(
                    "multi_turn",
                    answer,
                    _execute(
                        ToolCall(
                            "window_control",
                            {"action": "close", "window": canonical},
                        )
                    ),
                    f"voice.fill.close.{canonical}.surface_{surface_index}.frame_{frame_index}",
                    history=(
                        DialogueTurn("user", close_prompts[(surface_index + frame_index) % len(close_prompts)]),
                        DialogueTurn("jarvis", "Какое окно или приложение закрыть?"),
                    ),
                    state=close_state,
                )

    reminder_messages = (
        "проверить духовку",
        "позвонить другу",
        "ответить коллеге",
        "размяться",
        "проверить почту",
        "выключить чайник",
        "забрать посылку",
        "принять лекарство",
        "начать встречу",
        "оплатить интернет",
    )
    reminder_frames = (
        "напомни мне {message}",
        "создай напоминание {message}",
        "мне нужно не забыть {message}",
        "не дай мне забыть {message}",
        "поставь напоминание {message}",
    )
    time_answers = (
        (5, "через пять минут"),
        (10, "через десять минут"),
        (20, "спустя двадцать минут"),
        (25, "через двадцать пять минут"),
        (40, "давай через сорок минут"),
    )
    for message_index, message in enumerate(reminder_messages):
        pending = JALPlan(
            DialogueAct.ASK,
            steps=(ToolCall("set_reminder", {"message": message}),),
            missing=(MissingSlot(0, "minutes"),),
            reason="missing_time",
        )
        for frame_index, frame in enumerate(reminder_frames):
            request = frame.format(message=message)
            add(
                "multi_turn",
                request,
                pending,
                f"voice.ask.reminder.message_{message_index}.frame_{frame_index}",
            )
        for time_index, (minutes, answer) in enumerate(time_answers):
            add(
                "multi_turn",
                answer,
                _execute(
                    ToolCall(
                        "set_reminder",
                        {"minutes": minutes, "message": message},
                    )
                ),
                f"voice.fill.reminder.message_{message_index}.time_{time_index}",
                history=(
                    DialogueTurn("user", reminder_frames[message_index % len(reminder_frames)].format(message=message)),
                    DialogueTurn("jarvis", "Когда вам напомнить?"),
                ),
                state=pending,
            )

    for time_index, (minutes, answer) in enumerate(time_answers):
        missing_message = JALPlan(
            DialogueAct.ASK,
            steps=(ToolCall("set_reminder", {"minutes": minutes}),),
            missing=(MissingSlot(0, "message"),),
            reason="missing_reminder_text",
        )
        add(
            "multi_turn",
            f"{answer} напомни",
            missing_message,
            f"voice.ask.reminder_text.time_{time_index}",
        )
        for message_index, message in enumerate(reminder_messages):
            add(
                "multi_turn",
                message,
                _execute(
                    ToolCall(
                        "set_reminder",
                        {"minutes": minutes, "message": message},
                    )
                ),
                f"voice.fill.reminder_text.time_{time_index}.message_{message_index}",
                history=(
                    DialogueTurn("user", f"{answer} напомни"),
                    DialogueTurn("jarvis", "О чём вам напомнить?"),
                ),
                state=missing_message,
            )

    for previous, previous_surfaces in app_surfaces.items():
        previous_surface = previous_surfaces[0]
        previous_plan = _execute(
            ToolCall("open_application", {"application": previous})
        )
        for replacement, replacement_surfaces in app_surfaces.items():
            if replacement == previous:
                continue
            replacement_surface = replacement_surfaces[0]
            for frame_index, frame in enumerate(
                (
                    "нет, я имел в виду {app}",
                    "поправка, нужен {app}",
                    "не то, открой {app}",
                )
            ):
                add(
                    "correction",
                    frame.format(app=replacement_surface),
                    _execute(
                        ToolCall(
                            "open_application",
                            {"application": replacement},
                        )
                    ),
                    f"voice.correction.{previous}.to_{replacement}.frame_{frame_index}",
                    history=(
                        DialogueTurn("user", f"открой {previous_surface}"),
                        DialogueTurn("jarvis", f"Открываю {previous_surface}."),
                    ),
                    state=previous_plan,
                )

    gesture_surfaces = {
        "enable": ("включи жесты", "активируй жестовый режим"),
        "status": ("проверь жесты", "покажи состояние жестов"),
        "pause": ("поставь жесты на паузу", "временно останови жесты"),
        "resume": ("возобнови жесты", "продолжи распознавание жестов"),
        "disable": ("отключи жесты", "выключи жестовый режим"),
    }
    gesture_order = ("enable", "status", "pause", "resume", "disable")
    for variant in range(32):
        count = 2 + variant % 4
        actions = gesture_order[:count]
        phrases = [
            gesture_surfaces[action][(variant // (index + 1)) % 2]
            for index, action in enumerate(actions)
        ]
        text = ", ".join(phrases[:-1]) + " и " + phrases[-1]
        add(
            "compound",
            text,
            _execute(
                *(ToolCall("gesture_mode", {"action": action}) for action in actions)
            ),
            f"voice.gesture_sequence.steps_{count}.variant_{variant}",
        )

    voice_bank = (
        ("открой калькулятор", ToolCall("open_application", {"application": "calculator"})),
        ("открой пейнт", ToolCall("open_application", {"application": "paint"})),
        ("открой блокнот", ToolCall("open_application", {"application": "notepad"})),
        ("открой проводник", ToolCall("open_application", {"application": "explorer"})),
        ("скажи время", ToolCall("get_current_time")),
        ("покажи приложения", ToolCall("list_applications")),
        ("покажи напоминания", ToolCall("list_reminders")),
        ("сделай громче", ToolCall("system_control", {"action": "volume_up"})),
        ("сделай тише", ToolCall("system_control", {"action": "volume_down"})),
        ("включи жесты", ToolCall("gesture_mode", {"action": "enable"})),
        ("отключи жесты", ToolCall("gesture_mode", {"action": "disable"})),
        ("создай вкладку", ToolCall("browser_control", {"action": "new_tab"})),
    )
    for step_count in range(2, 6):
        for variant in range(320):
            selected = rng.sample(voice_bank, step_count)
            phrases = [text for text, _call in selected]
            connector = (", ", ", затем ", " и потом ")[variant % 3]
            text = connector.join(phrases[:-1]) + " и " + phrases[-1]
            add(
                "compound",
                text,
                _execute(*(call for _text, call in selected)),
                f"voice.compound.steps_{step_count}.variant_{variant}",
            )

    for index, text in enumerate(
        (
            "проверить духовку",
            "ответить коллеге",
            "создай напоминание ответить коллеге",
            "мне нужно не забыть позвонить другу",
            "отмени напоминание",
            "номер тридцать один",
        )
    ):
        # These contrastive surfaces were involved in false global-cancel
        # predictions. Their true targets are already represented above; the
        # standalone fragments must never teach an unrelated destructive tool.
        if text in {"проверить духовку", "ответить коллеге", "номер тридцать один"}:
            add(
                "hard_negative",
                text,
                JALPlan(DialogueAct.DIALOGUE, reason="general_chat"),
                f"voice.cancel_negative.fragment_{index}",
            )
    return rows


def _record(example: JSCExample) -> dict[str, Any]:
    return {
        "schema_version": DATA_SCHEMA_VERSION,
        "scenario_id": example.scenario_id,
        "split": example.split,
        "family_id": example.family_id,
        "category": example.category,
        "history": [
            {"role": turn.role, "text": turn.text} for turn in example.history
        ],
        "text": example.text,
        "state_jal": dumps(example.state) if example.state is not None else None,
        "target_jal": dumps(example.target),
        "metadata": dict(example.metadata),
    }


def _jsonl(examples: Iterable[JSCExample]) -> str:
    return "".join(
        json.dumps(
            _record(example),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for example in examples
    )


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    temporary.replace(path)


def build(source: Path = SOURCE, output: Path = OUTPUT) -> dict[str, Any]:
    registry = build_project_schema_registry()
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    train = tuple(load_jsc_jsonl(source / "train.jsonl", registry, expected_split="train"))
    validation = tuple(
        load_jsc_jsonl(source / "validation.jsonl", registry, expected_split="validation")
    )
    blocked_signatures = {example.input_signature for example in (*train, *validation)}
    augmentations = tuple(
        example
        for example in build_augmentations()
        if example.input_signature not in blocked_signatures
    )
    combined = (*train, *augmentations)
    split_report = validate_jsc_splits({"train": combined, "validation": validation})
    train_content = _jsonl(combined)
    validation_content = (source / "validation.jsonl").read_bytes()
    _write(output / "train.jsonl", train_content)
    _write(output / "validation.jsonl", validation_content)
    result = deepcopy(manifest)
    result["generator"] = "training_workspace.build_jsc_structured_dataset"
    result["structured_augmentation"] = {
        "seed": SEED,
        "base_train_examples": len(train),
        "added_train_examples": len(augmentations),
        "migration_suite_opened": False,
        "test_opened": False,
        "evaluation_holdout_opened": False,
    }
    result["splits"] = {
        "train": {
            **split_report["train"],
            "file": "train.jsonl",
            "sha256": hashlib.sha256(train_content.encode("utf-8")).hexdigest(),
        },
        "validation": {
            **split_report["validation"],
            "file": "validation.jsonl",
            "sha256": hashlib.sha256(validation_content).hexdigest(),
        },
    }
    _write(
        output / "dataset_manifest.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = build(args.source, args.output)
    print(json.dumps(report["structured_augmentation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
