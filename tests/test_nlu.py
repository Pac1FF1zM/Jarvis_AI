"""Regression tests for the project-owned NLU baseline."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading

from core.config_loader import ModuleConfig
from core.event_bus import Event, EventBus
from core.gpu_lock import GPULock
from ml.nlu.data import build_examples
from ml.nlu.inference import NLUPredictor
from ml.nlu.schema import NLUResult
from modules.nlu import (
    _apply_reminder_guardrails,
    NLUModule,
    _apply_runtime_command_guardrails,
    _normalise_transcription_for_nlu,
)
from modules.command_router import RoutedAction, route_explicit_command, split_compound_command


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "nlu_word_bigru_curriculum.pt"
)


def test_dataset_splits_have_no_identical_texts():
    splits = [
        {example.text.lower() for example in build_examples(name)}
        for name in ("train", "validation", "test")
    ]
    assert splits[0].isdisjoint(splits[1])
    assert splits[0].isdisjoint(splits[2])
    assert splits[1].isdisjoint(splits[2])


def test_flat_whisper_transcript_is_split_into_ordered_actions_without_commas():
    text = (
        "открой браузер запусти жестовый режим "
        "напомни через двадцать минут о встрече закрой дискорд"
    )
    normalized = _normalise_transcription_for_nlu(text)
    assert split_compound_command(normalized) == [
        "открой браузер",
        "запусти жестовый режим",
        "напомни через 20 минут о встрече",
        "закрой дискорд",
    ]


def test_context_pronoun_negation_followup_and_correction_are_deterministic():
    previous = RoutedAction("open_application", {"application": "browser"})
    assert route_explicit_command("закрой его", previous_action=previous) == RoutedAction(
        "window_control", {"action": "close", "window": "browser"}
    )
    assert route_explicit_command("не закрывай браузер").intent == "negated_command"
    correction = route_explicit_command("нет не paint открой калькулятор", previous_action=previous)
    assert correction is not None
    assert correction.intent == "open_application"
    assert correction.slots["correction_from"] == "paint"
    assert correction.slots["application"] == "calculator"


def test_frozen_holdout_has_no_exact_development_overlap():
    holdout_path = Path(__file__).resolve().parents[1] / "ml" / "nlu" / "holdout_v2.jsonl"
    holdout = {
        json.loads(line)["text"].casefold().strip()
        for line in holdout_path.read_text(encoding="utf-8").splitlines()
        if line
    }
    development = {
        example.text.casefold().strip()
        for split in ("train", "validation", "test")
        for example in build_examples(split, augmented=(split == "train"))
    }
    assert len(holdout) == 49
    assert holdout.isdisjoint(development)


def test_trained_checkpoint_routes_core_intents():
    predictor = NLUPredictor(MODEL_PATH)
    assert predictor.predict("сколько сейчас времени").intent == "get_current_time"
    assert predictor.predict("привет как дела").intent == "general_chat"
    assert predictor.predict("остановись").intent == "cancel"
    assert predictor.predict("открой калькулятор").intent == "open_application"
    assert predictor.predict("какие приложения можно открыть").intent == "list_applications"


def test_reminder_parameters_are_complete_on_unseen_phrase_family():
    predictor = NLUPredictor(MODEL_PATH)
    result = predictor.predict("через 18 минут скажи мне проверить чайник")
    assert result.intent == "set_reminder"
    assert result.slots == {"minutes": "18", "reminder_text": "проверить чайник"}


def test_application_name_is_extracted_on_unseen_phrase_family():
    predictor = NLUPredictor(MODEL_PATH)
    result = predictor.predict("открой мне калькулятор")
    assert result.intent == "open_application"
    assert result.slots == {"application": "калькулятор"}


def test_talking_about_application_does_not_trigger_launch_intent():
    predictor = NLUPredictor(MODEL_PATH)
    for text in (
        "как работает интернет",
        "расскажи про браузеры",
        "зачем нужен калькулятор",
    ):
        assert predictor.predict(text).intent == "general_chat"


def test_observed_whisper_errors_are_normalised_before_neural_routing():
    assert (
        _normalise_transcription_for_nlu("Отпрой к алкулятор.")
        == "открой калькулятор."
    )
    assert _normalise_transcription_for_nlu("Запусти блокноты") == "запусти блокнот"
    assert _normalise_transcription_for_nlu("Открой пеинт") == "открой paint"
    assert _normalise_transcription_for_nlu("Колька времени") == "сколько времени"


def test_explicit_phonetic_allowlisted_app_rescues_bad_neural_intent():
    bad_prediction = NLUResult("cancel", 0.999, {})
    for text, expected in (
        ("Отпрой к алкулятор", "calculator"),
        ("Запусти блокноты", "notepad"),
        ("Открой пеинт", "paint"),
        ("Открой дисорд", "discord"),
        ("Открой дискод", "discord"),
        ("Запусти пожалуйста дискорд", "discord"),
        ("Будь добр открой Paint", "paint"),
        ("Мне сейчас нужен браузер", "browser"),
        ("Давай включим калькулятор", "calculator"),
        ("Открой-ка блокнот", "notepad"),
    ):
        normalised = _normalise_transcription_for_nlu(text)
        rescued = _apply_runtime_command_guardrails(normalised, bad_prediction)
        assert rescued.intent == "open_application"
        assert rescued.slots == {"application": expected}


def test_guardrail_never_rescues_non_imperative_or_unknown_application(monkeypatch):
    import tools._applications as applications

    monkeypatch.setattr(applications, "discover_installed_applications", lambda: ())
    prediction = NLUResult("general_chat", 0.9, {})
    for text in (
        "расскажи про калькулятор",
        "как открыть калькулятор",
        "мне нравится дискорд",
        "не открывай браузер",
        "открой powershell",
    ):
        normalised = _normalise_transcription_for_nlu(text)
        assert (
            _apply_runtime_command_guardrails(normalised, prediction)
            == prediction
        )


def test_reminder_guardrails_cover_create_list_cancel_and_absolute_time():
    fallback = NLUResult("general_chat", 0.7, {})
    cases = (
        (
            "через 12 минут напомни проверить духовку",
            "set_reminder",
            {"minutes": "12", "reminder_text": "проверить духовку"},
        ),
        (
            "напомни завтра в 18:30 позвонить маме",
            "set_reminder",
            {
                "clock_time": "18:30",
                "day": "завтра",
                "reminder_text": "позвонить маме",
            },
        ),
        ("покажи мои напоминания", "list_reminders", {}),
        ("отмени напоминание номер 7", "cancel_reminder", {"reminder_id": "7"}),
    )

    for text, intent, slots in cases:
        result = _apply_reminder_guardrails(text, fallback)
        assert result.intent == intent
        assert result.slots == slots
        assert result.confidence >= 0.99


def test_reminder_guardrail_does_not_treat_discussion_as_action():
    fallback = NLUResult("general_chat", 0.91, {})
    for text in (
        "расскажи как работают напоминания",
        "я не хочу ставить напоминание",
        "напоминание без времени",
        "удали все данные",
    ):
        assert _apply_reminder_guardrails(text, fallback) == fallback


async def test_module_publishes_nlu_result_with_same_trace():
    event_loop_thread = threading.get_ident()
    prediction_threads: list[int] = []

    class FakePredictor:
        def __init__(self, checkpoint, device):
            self.checkpoint = checkpoint
            self.device = device

        def predict(self, text):
            prediction_threads.append(threading.get_ident())
            return NLUResult("get_current_time", 0.91, {})

    bus = EventBus()
    module = NLUModule(
        ModuleConfig(device="cpu", model=str(MODEL_PATH)),
        GPULock(),
        predictor_factory=FakePredictor,
    )
    output: list[Event] = []

    async def record(event: Event) -> None:
        output.append(event)

    bus.subscribe("nlu_result", record)
    await module.start(bus)
    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "который час", "confidence": 0.88},
        trace_id="nlu-trace",
    )
    await asyncio.sleep(0.05)
    assert output == [], "NLU must wait until the orchestrator reaches THINKING"
    bus.publish("thinking_ready", trace_id="nlu-trace")
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task
    await module.stop()

    assert len(output) == 1
    assert output[0].trace_id == "nlu-trace"
    assert output[0].payload["intent"] == "get_current_time"
    assert output[0].payload["intent_confidence"] == 0.91
    assert prediction_threads
    assert all(thread_id != event_loop_thread for thread_id in prediction_threads)


async def test_low_confidence_prediction_is_rejected_as_unknown():
    class UncertainPredictor:
        def __init__(self, checkpoint, device):
            pass

        def predict(self, text):
            return NLUResult("set_reminder", 0.2, {"minutes": "5"})

    bus = EventBus()
    module = NLUModule(
        ModuleConfig(
            device="cpu",
            model=str(MODEL_PATH),
            params={"confidence_threshold": 0.55},
        ),
        GPULock(),
        predictor_factory=UncertainPredictor,
    )
    output: list[Event] = []

    async def record(event: Event) -> None:
        output.append(event)

    bus.subscribe("nlu_result", record)
    await module.start(bus)
    run_task = asyncio.create_task(bus.run())
    bus.publish("transcription_ready", {"text": "неясно"}, trace_id="reject")
    bus.publish("thinking_ready", trace_id="reject")
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task
    await module.stop()

    assert output[0].payload["intent"] == "unknown"
    assert output[0].payload["raw_intent"] == "set_reminder"
    assert output[0].payload["slots"] == {}


async def test_cancel_intent_publishes_control_event_instead_of_nlu_result():
    class CancelPredictor:
        def __init__(self, checkpoint, device):
            pass

        def predict(self, text):
            return NLUResult("cancel", 0.99, {})

    bus = EventBus()
    module = NLUModule(
        ModuleConfig(device="cpu", model=str(MODEL_PATH)),
        GPULock(),
        predictor_factory=CancelPredictor,
    )
    cancel_events: list[Event] = []
    nlu_events: list[Event] = []

    async def record_cancel(event: Event) -> None:
        cancel_events.append(event)

    async def record_nlu(event: Event) -> None:
        nlu_events.append(event)

    bus.subscribe("cancel_requested", record_cancel)
    bus.subscribe("nlu_result", record_nlu)
    await module.start(bus)
    run_task = asyncio.create_task(bus.run())
    bus.publish("transcription_ready", {"text": "стоп"}, trace_id="stop-trace")
    bus.publish("thinking_ready", trace_id="stop-trace")
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task
    await module.stop()

    assert nlu_events == []
    assert len(cancel_events) == 1
    assert cancel_events[0].trace_id == "stop-trace"
    assert cancel_events[0].payload["reason"] == "user_requested"
