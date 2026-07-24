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
from modules.nlu import NLUModule


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
