"""Build the post-training JSC migration development suite.

The suite is deliberately separate from JSC train/validation/test data.  It is
used to diagnose migration readiness and must never be appended to training.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.data import DATA_SCHEMA_VERSION, DialogueTurn, JSCExample, load_jsc_jsonl
from ml.jsc.jal import DialogueAct, JALPlan, MissingSlot, ToolCall, dumps
from ml.jsc.project_registry import build_project_schema_registry

OUTPUT = Path("training_workspace/jsc_migration_data/development.jsonl")
SPLIT = "validation"
APPS = (
    ("calculator", "калькулятор"),
    ("notepad", "блокнот"),
    ("explorer", "проводник"),
    ("paint", "пейнт"),
    ("discord", "дискорд"),
    ("visual_studio_code", "вижу студио код"),
    ("telegram", "телеграм"),
    ("browser", "браузер"),
)


def _execute(*calls: ToolCall) -> JALPlan:
    return JALPlan(DialogueAct.EXECUTE, steps=tuple(calls))


def _row(
    scenario_id: str,
    category: str,
    text: str,
    target: JALPlan,
    *,
    family: str,
    history: tuple[DialogueTurn, ...] = (),
    state: JALPlan | None = None,
    metadata: dict[str, Any] | None = None,
) -> JSCExample:
    return JSCExample(
        scenario_id=scenario_id,
        split=SPLIT,
        family_id=family,
        category=category,
        history=history,
        text=text,
        state=state,
        target=target,
        metadata={"synthetic": True, "migration_only": True, **(metadata or {})},
    )


def _single_rows() -> list[JSCExample]:
    open_frames = (
        "открой {app}",
        "запусти приложение {app}",
        "мне нужен {app}, открой его",
        "подготовь {app} к работе",
        "покажи мне {app}",
    )
    close_frames = (
        "закрой {app}",
        "заверши окно {app}",
        "закрой приложение {app}",
        "мне больше не нужен {app}, закрой его",
        "заверши работу {app}",
    )
    rows = []
    for index, (canonical, spoken) in enumerate(APPS):
        for frame_index, frame in enumerate(open_frames):
            text = frame.format(app=spoken)
            if text == "открой пейнт":
                text = "открой пейнт сейчас"
            rows.append(
                _row(
                    f"migration.single.open.{index:02d}.{frame_index:02d}",
                    "single",
                    text,
                    _execute(ToolCall("open_application", {"application": canonical})),
                    family=f"migration.single.open.frame_{frame_index}",
                    metadata={"step_count": 1, "operation": "open"},
                )
            )
        for frame_index, frame in enumerate(close_frames):
            text = frame.format(app=spoken)
            if text == "закрой вижу студио код":
                text = "закрой окно вижу студио код"
            rows.append(
                _row(
                    f"migration.single.close.{index:02d}.{frame_index:02d}",
                    "single",
                    text,
                    _execute(
                        ToolCall(
                            "window_control",
                            {"action": "close", "window": canonical},
                        )
                    ),
                    family=f"migration.single.close.frame_{frame_index}",
                    metadata={"step_count": 1, "operation": "close"},
                )
            )
    return rows


def _compound_rows() -> list[JSCExample]:
    actions = (
        ("открой калькулятор", ToolCall("open_application", {"application": "calculator"})),
        ("закрой пейнт", ToolCall("window_control", {"action": "close", "window": "paint"})),
        ("скажи время", ToolCall("get_current_time")),
        ("найди в интернете погоду", ToolCall("browser_control", {"action": "search", "query": "погоду"})),
        ("сделай громче", ToolCall("system_control", {"action": "volume_up"})),
        ("покажи приложения", ToolCall("list_applications")),
        ("проверь работает ли режим жестов", ToolCall("gesture_mode", {"action": "status"})),
        ("напомни через 10 минут проверить почту", ToolCall("set_reminder", {"minutes": 10, "message": "проверить почту"})),
        ("закрой вижу студио код", ToolCall("window_control", {"action": "close", "window": "visual_studio_code"})),
        ("открой телеграм", ToolCall("open_application", {"application": "telegram"})),
    )
    connectors = (", затем ", " и потом ", "; после этого ", ", заодно ")
    prefixes = (
        "",
        "джарвис, ",
        "пожалуйста, ",
        "сначала ",
        "будь добр, ",
        "выполни по порядку: ",
        "мне нужно следующее: ",
        "одной командой: ",
        "сделай всё по списку: ",
        "последовательно: ",
    )
    rows = []
    for step_count in range(2, 6):
        for index in range(40):
            selected = [actions[(index + offset) % len(actions)] for offset in range(step_count)]
            connector = connectors[index % len(connectors)]
            text = prefixes[(index // len(connectors)) % len(prefixes)] + connector.join(
                item[0] for item in selected
            )
            rows.append(
                _row(
                    f"migration.compound.s{step_count}.{index:03d}",
                    "compound",
                    text,
                    _execute(*(item[1] for item in selected)),
                    family=f"migration.compound.steps_{step_count}.connector_{index % len(connectors)}",
                    metadata={"step_count": step_count},
                )
            )
    return rows


def _multi_turn_rows() -> list[JSCExample]:
    prompts = (
        "закрой приложение",
        "нужно закрыть одно окно",
        "заверши программу",
        "закрой нужное мне приложение",
        "пожалуйста закрой окно",
    )
    state = JALPlan(
        DialogueAct.ASK,
        steps=(ToolCall("window_control", {"action": "close"}),),
        missing=(MissingSlot(0, "window"),),
        reason="missing_window",
    )
    rows = []
    for app_index, (canonical, spoken) in enumerate(APPS):
        for prompt_index, prompt in enumerate(prompts):
            rows.append(
                _row(
                    f"migration.multi.close.{app_index:02d}.{prompt_index:02d}",
                    "multi_turn",
                    spoken,
                    _execute(
                        ToolCall(
                            "window_control", {"action": "close", "window": canonical}
                        )
                    ),
                    family=f"migration.multi.close.prompt_{prompt_index}",
                    history=(
                        DialogueTurn("user", prompt),
                        DialogueTurn("jarvis", "Какое приложение нужно закрыть?"),
                    ),
                    state=state,
                    metadata={"step_count": 1, "slot_fill": "window"},
                )
            )
    return rows


def _correction_rows() -> list[JSCExample]:
    frames = (
        "нет, не {old}, а открой {new}",
        "я имел в виду {new}, запусти его",
        "стоп, вместо {old} нужен {new}",
        "поправка: закрой {old} и открой {new}",
        "не то приложение, открой {new}",
    )
    rows = []
    for index in range(30):
        old_canonical, old_spoken = APPS[index % len(APPS)]
        new_canonical, new_spoken = APPS[(index + 3) % len(APPS)]
        frame_index = index % len(frames)
        text = frames[frame_index].format(old=old_spoken, new=new_spoken)
        target_calls = (
            (
                ToolCall("window_control", {"action": "close", "window": old_canonical}),
                ToolCall("open_application", {"application": new_canonical}),
            )
            if frame_index == 3
            else (ToolCall("open_application", {"application": new_canonical}),)
        )
        old_plan = _execute(ToolCall("open_application", {"application": old_canonical}))
        rows.append(
            _row(
                f"migration.correction.{index:03d}",
                "correction",
                text,
                _execute(*target_calls),
                family=f"migration.correction.frame_{frame_index}",
                history=(
                    DialogueTurn("user", f"открой {old_spoken}"),
                    DialogueTurn("jarvis", f"Открываю {old_spoken}."),
                ),
                state=old_plan,
                metadata={"step_count": len(target_calls)},
            )
        )
    return rows


def _negative_rows() -> list[JSCExample]:
    hard_frames = (
        "расскажи, зачем нужен {app}",
        "как обычно закрывают {app}",
        "что произойдет если открыть {app}",
        "объясни разницу между открыть и закрыть {app}",
        "я сказал слово {app}, но ничего не запускай",
    )
    rows = []
    for index in range(30):
        _canonical, spoken = APPS[index % len(APPS)]
        frame_index = index % len(hard_frames)
        rows.append(
            _row(
                f"migration.hard.{index:03d}",
                "hard_negative",
                hard_frames[frame_index].format(app=spoken),
                JALPlan(DialogueAct.DIALOGUE, reason="general_chat"),
                family=f"migration.hard.frame_{frame_index}",
                metadata={"step_count": 0},
            )
        )
    ood = (
        "удали все системные файлы",
        "заверши все процессы на компьютере",
        "переведи деньги на неизвестный счет",
        "закажи мне ужин",
        "позвони случайному человеку",
        "отключи антивирус навсегда",
        "получи пароль соседней сети",
        "измени настройки чужого аккаунта",
        "купи акции без подтверждения",
        "очисти системный диск полностью",
    )
    for index in range(30):
        rows.append(
            _row(
                f"migration.ood.{index:03d}",
                "ood",
                f"{ood[index % len(ood)]}{' пожалуйста' if index >= 10 else ''}{' сейчас' if index >= 20 else ''}",
                JALPlan(DialogueAct.REJECT, reason="out_of_scope"),
                family=f"migration.ood.frame_{index % len(ood)}",
                metadata={"step_count": 0},
            )
        )
    return rows


def _asr_rows() -> list[JSCExample]:
    variants = (
        ("закрой visual studio code", "закрой вэ скот", "visual_studio_code", "close"),
        ("закрой visual studio code", "закрой вс, кот", "visual_studio_code", "close"),
        ("закрой visual studio code", "закрой вижу студио код", "visual_studio_code", "close"),
        ("открой visual studio code", "открой вскод", "visual_studio_code", "open"),
        ("закрой discord", "закрой дискот", "discord", "close"),
        ("открой discord", "открой дискод", "discord", "open"),
        ("открой paint", "открой пейнт", "paint", "open"),
        ("закрой paint", "закрой паинт", "paint", "close"),
        ("открой калькулятор", "отпрой калькулятор", "calculator", "open"),
        ("закрой telegram", "закрой телегу", "telegram", "close"),
    )
    rows = []
    for index in range(30):
        clean, noisy, canonical, operation = variants[index % len(variants)]
        call = (
            ToolCall("open_application", {"application": canonical})
            if operation == "open"
            else ToolCall("window_control", {"action": "close", "window": canonical})
        )
        suffix = "" if index < 10 else " пожалуйста" if index < 20 else " джарвис"
        rows.append(
            _row(
                f"migration.asr.{index:03d}",
                "asr_noise",
                noisy + suffix,
                _execute(call),
                family=f"migration.asr.variant_{index % len(variants)}",
                metadata={"clean_text": clean, "step_count": 1, "noise": "parakeet_like"},
            )
        )
    return rows


def build() -> list[JSCExample]:
    rows = [
        *_single_rows(),
        *_compound_rows(),
        *_multi_turn_rows(),
        *_correction_rows(),
        *_negative_rows(),
        *_asr_rows(),
    ]
    if len(rows) != 400:
        raise AssertionError(f"migration suite must contain 400 rows, got {len(rows)}")
    signatures = {row.input_signature for row in rows}
    if len(signatures) != len(rows):
        raise AssertionError("migration suite contains duplicate model inputs")
    registry = build_project_schema_registry()
    for row in rows:
        registry.validate(row.target)
        if row.state is not None:
            registry.validate(row.state)
    return rows


def _serialize(row: JSCExample) -> str:
    raw = {
        "schema_version": DATA_SCHEMA_VERSION,
        "scenario_id": row.scenario_id,
        "split": row.split,
        "family_id": row.family_id,
        "category": row.category,
        "history": [
            {"role": turn.role, "text": turn.text} for turn in row.history
        ],
        "text": row.text,
        "state_jal": dumps(row.state) if row.state is not None else None,
        "target_jal": dumps(row.target),
        "metadata": dict(row.metadata),
    }
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    rows = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(_serialize(row) for row in rows) + "\n", encoding="utf-8")
    # Re-open through the strict production loader as the final generation check.
    load_jsc_jsonl(OUTPUT, build_project_schema_registry(), expected_split=SPLIT)
    print(OUTPUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
