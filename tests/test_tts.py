"""Tests for modules/tts.py.

Covers: response_ready -> speech_started then speech_finished (in order),
trace_id propagation, and speaking duration being non-trivial.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import pytest

from core.config_loader import ModuleConfig
from core.event_bus import EventBus, Event
import modules.tts as tts_mod
from modules.tts import TTSModule, _prepare_russian_speech_text


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


async def test_speech_start_then_finish_in_order(bus: EventBus):
    mod = TTSModule(config=ModuleConfig())
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)

    bus.subscribe("speech_started", record)
    bus.subscribe("speech_finished", record)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    started = asyncio.get_event_loop().time()
    bus.publish("response_ready", {"text": "hello world"}, trace_id="tts-tr")
    await asyncio.sleep(0.5)
    await bus.stop()
    await run_task
    await mod.stop()
    elapsed = asyncio.get_event_loop().time() - started

    types = [e.event_type for e in events]
    assert types == ["speech_started", "speech_finished"], types
    assert all(e.trace_id == "tts-tr" for e in events)
    # ~25ms/char * 11 chars capped at 0.4s — should take a measurable moment.
    assert elapsed >= 0.05


async def test_empty_text_does_not_crash(bus: EventBus):
    mod = TTSModule(config=ModuleConfig())
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)

    bus.subscribe("speech_finished", record)
    await mod.start(bus)
    run_task = asyncio.create_task(bus.run())
    bus.publish("response_ready", {"text": ""}, trace_id="empty-tr")
    await asyncio.sleep(0.2)
    await bus.stop()
    await run_task
    await mod.stop()
    assert events, "speech_finished should still fire for empty text"


# =========================================================================== #
# Real Silero path (mocked; no package, model download, or audio device needed)
# =========================================================================== #
class _FakeSileroModel:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.apply_calls: list[dict[str, Any]] = []
        self.apply_thread_ids: list[int] = []
        self.error: BaseException | None = None

    def to(self, device: str) -> None:
        self.to_calls.append(device)

    def apply_tts(self, **kwargs):
        self.apply_calls.append(kwargs)
        self.apply_thread_ids.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        return [0.0, 0.1, -0.1]


class _FakeSileroFactory:
    def __init__(self, model: _FakeSileroModel) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.model, "example text"


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.play_calls: list[dict[str, Any]] = []
        self.play_thread_ids: list[int] = []
        self.stop_calls = 0

    def play(self, audio, samplerate, blocking):
        self.play_calls.append(
            {"audio": audio, "samplerate": samplerate, "blocking": blocking}
        )
        self.play_thread_ids.append(threading.get_ident())

    def stop(self):
        self.stop_calls += 1


@pytest.fixture
def fake_silero(monkeypatch):
    model = _FakeSileroModel()
    factory = _FakeSileroFactory(model)
    sounddevice = _FakeSoundDevice()
    monkeypatch.setattr(tts_mod, "_SILERO_TTS", factory)
    monkeypatch.setattr(tts_mod, "_SOUNDDEVICE", sounddevice)
    return model, factory, sounddevice


async def test_real_model_loaded_once_in_start(bus: EventBus, fake_silero):
    model, factory, _sounddevice = fake_silero
    mod = TTSModule(
        ModuleConfig(
            device="cpu",
            model="v3_en",
            params={"language": "en", "speaker": "en_7", "sample_rate": 24000},
        )
    )

    await mod.start(bus)
    await mod.stop()

    assert len(factory.calls) == 1, "Silero model must load once in start()"
    assert factory.calls[0] == {"language": "en", "speaker": "v3_en"}
    assert model.to_calls == ["cpu"]


async def test_real_synthesis_runs_off_event_loop(bus: EventBus, fake_silero):
    model, _factory, sounddevice = fake_silero
    mod = TTSModule(ModuleConfig(device="cpu", model="v4_ru"))
    await mod.start(bus)

    finished = asyncio.Event()
    output: list[Event] = []

    async def record(event: Event) -> None:
        output.append(event)
        finished.set()

    bus.subscribe("speech_finished", record)
    event_loop_thread_id = threading.get_ident()
    run_task = asyncio.create_task(bus.run())
    bus.publish("response_ready", {"text": "Привет"}, trace_id="real-tts")
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()

    assert output[0].payload["text"] == "Привет"
    assert model.apply_calls == [
        {"text": "Привет", "speaker": "xenia", "sample_rate": 48000}
    ]
    assert model.apply_thread_ids
    assert all(tid != event_loop_thread_id for tid in model.apply_thread_ids)
    assert sounddevice.play_thread_ids
    assert all(tid != event_loop_thread_id for tid in sounddevice.play_thread_ids)


def test_russian_speech_text_spells_clock_time_and_digits():
    assert _prepare_russian_speech_text("Сейчас 20:55.") == (
        "Сейчас двадцать часов пятьдесят пять минут."
    )
    assert _prepare_russian_speech_text("Через 12 минут") == (
        "Через двенадцать минут"
    )


async def test_russian_silero_receives_pronounceable_time(bus: EventBus, fake_silero):
    model, _factory, _sounddevice = fake_silero
    mod = TTSModule(ModuleConfig(device="cpu", model="v4_ru"))
    await mod.start(bus)
    finished = asyncio.Event()

    async def record(_event: Event) -> None:
        finished.set()

    bus.subscribe("speech_finished", record)
    run_task = asyncio.create_task(bus.run())
    bus.publish("response_ready", {"text": "Сейчас 20:55."}, trace_id="time-tts")
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()

    assert model.apply_calls[0]["text"] == (
        "Сейчас двадцать часов пятьдесят пять минут."
    )


async def test_russian_configuration_loads_matching_model(bus: EventBus, fake_silero):
    _model, factory, _sounddevice = fake_silero
    mod = TTSModule(
        ModuleConfig(
            device="cpu",
            model="v4_ru",
            params={"language": "ru", "speaker": "xenia", "sample_rate": 48000},
        )
    )

    await mod.start(bus)
    await mod.stop()

    assert factory.calls == [{"language": "ru", "speaker": "v4_ru"}]


def test_russian_configuration_rejects_english_model():
    with pytest.raises(ValueError, match="requires a Russian Silero model"):
        TTSModule(
            ModuleConfig(
                model="v3_en",
                params={"language": "ru", "speaker": "xenia"},
            )
        )


async def test_silero_missing_falls_back_to_stub(
    bus: EventBus, monkeypatch, caplog
):
    monkeypatch.setattr(tts_mod, "_SILERO_TTS", None)
    monkeypatch.setattr(tts_mod, "_SOUNDDEVICE", _FakeSoundDevice())
    mod = TTSModule(ModuleConfig())

    with caplog.at_level(logging.WARNING, logger="jarvis.module.tts"):
        await mod.start(bus)

    assert mod._model is None
    assert any("silero not installed" in r.message for r in caplog.records)

    finished = asyncio.Event()
    bus.subscribe("speech_finished", lambda event: _set_event(finished))
    run_task = asyncio.create_task(bus.run())
    bus.publish("response_ready", {"text": "hi"}, trace_id="stub-tts")
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()


async def test_runtime_error_falls_back_for_that_turn(
    bus: EventBus, fake_silero, caplog
):
    model, _factory, sounddevice = fake_silero
    model.error = RuntimeError("synthesis exploded")
    mod = TTSModule(ModuleConfig())
    await mod.start(bus)

    finished = asyncio.Event()
    output: list[Event] = []

    async def record(event: Event) -> None:
        output.append(event)
        finished.set()

    bus.subscribe("speech_finished", record)
    with caplog.at_level(logging.ERROR, logger="jarvis.module.tts"):
        run_task = asyncio.create_task(bus.run())
        bus.publish("response_ready", {"text": "boom"}, trace_id="error-tts")
        await asyncio.wait_for(finished.wait(), timeout=1.0)
        await bus.stop()
        await run_task
    await mod.stop()

    assert output and output[0].trace_id == "error-tts"
    assert sounddevice.play_calls == []
    assert any("falling back to stub" in r.message for r in caplog.records)


async def test_barge_in_cancels_old_trace_without_stale_finish(
    bus: EventBus, monkeypatch
):
    monkeypatch.setattr(tts_mod, "_SILERO_TTS", None)
    monkeypatch.setattr(tts_mod, "_SOUNDDEVICE", None)
    mod = TTSModule(ModuleConfig())
    await mod.start(bus)

    first_entered = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_finished = asyncio.Event()
    natural_first_completions = 0
    speak_tasks: dict[str, asyncio.Task] = {}
    finished_events: list[Event] = []

    async def controlled_speak(text: str, session) -> None:
        nonlocal natural_first_completions
        speak_tasks[text] = asyncio.current_task()
        if text == "first response":
            first_entered.set()
            try:
                await asyncio.Event().wait()
                natural_first_completions += 1
            except asyncio.CancelledError:
                first_cancelled.set()
                raise

    async def record_finished(event: Event) -> None:
        finished_events.append(event)
        if event.trace_id == "trace-b":
            second_finished.set()

    monkeypatch.setattr(mod, "_speak", controlled_speak)
    bus.subscribe("speech_finished", record_finished)
    run_task = asyncio.create_task(bus.run())

    bus.publish(
        "response_ready", {"text": "first response"}, trace_id="trace-a"
    )
    await asyncio.wait_for(first_entered.wait(), timeout=1.0)
    first_task = speak_tasks["first response"]

    bus.publish("wake_word_detected", {}, trace_id="trace-b")
    await asyncio.wait_for(first_cancelled.wait(), timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await first_task

    bus.publish(
        "response_ready", {"text": "second response"}, trace_id="trace-b"
    )
    await asyncio.wait_for(second_finished.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()

    assert first_task.cancelled(), "trace-A speak task was not actually cancelled"
    assert natural_first_completions == 0
    assert not any(event.trace_id == "trace-a" for event in finished_events)
    assert any(event.trace_id == "trace-b" for event in finished_events)


async def _set_event(event: asyncio.Event) -> None:
    event.set()


# =========================================================================== #
# Adversarial cancellation/ownership races (deterministic synchronization)
# =========================================================================== #
class _BlockingSileroModel(_FakeSileroModel):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def apply_tts(self, **kwargs):
        self.apply_calls.append(kwargs)
        self.apply_thread_ids.append(threading.get_ident())
        self.started.set()
        self.release.wait(timeout=5.0)
        if self.error is not None:
            raise self.error
        return [0.0, 0.1]


async def test_double_wake_serializes_until_synthesis_worker_is_drained(
    bus: EventBus, monkeypatch
):
    model = _BlockingSileroModel()
    factory = _FakeSileroFactory(model)
    sounddevice = _FakeSoundDevice()
    monkeypatch.setattr(tts_mod, "_SILERO_TTS", factory)
    monkeypatch.setattr(tts_mod, "_SOUNDDEVICE", sounddevice)
    mod = TTSModule(ModuleConfig())
    await mod.start(bus)

    published: list[Event] = []
    original_publish_event = bus.publish_event

    def record_publish(event: Event) -> None:
        published.append(event)
        original_publish_event(event)

    monkeypatch.setattr(bus, "publish_event", record_publish)

    response_task = asyncio.create_task(
        mod._on_response(
            Event("response_ready", {"text": "blocking"}, trace_id="trace-a")
        )
    )
    await _wait_until(model.started.is_set)
    session = mod._session
    assert session is not None

    wake_b = asyncio.create_task(
        mod._on_wake(Event("wake_word_detected", trace_id="trace-b"))
    )
    await _wait_until(lambda: session.cancel_requested)
    assert mod._synthesis_worker is not None
    assert not mod._synthesis_worker.done()

    # The second wake must wait on the single-writer lock; it must not issue
    # an independent second task.cancel() while trace A is draining.
    wake_c = asyncio.create_task(
        mod._on_wake(Event("wake_word_detected", trace_id="trace-c"))
    )
    await asyncio.sleep(0)
    assert not wake_c.done()
    assert mod._owner_trace_id == "trace-a"

    model.release.set()
    results = await asyncio.gather(wake_b, wake_c, response_task, return_exceptions=True)
    await mod.stop()

    assert not any(isinstance(result, BaseException) for result in results), results
    assert mod._synthesis_worker is None
    assert session.synthesis_worker is None
    assert mod._session is None
    assert mod._owner_trace_id == "trace-c"
    assert not any(
        event.event_type == "speech_finished" and event.trace_id == "trace-a"
        for event in published
    )


async def test_completion_boundary_wake_suppresses_stale_speech_finished(
    bus: EventBus, monkeypatch
):
    monkeypatch.setattr(tts_mod, "_SILERO_TTS", None)
    monkeypatch.setattr(tts_mod, "_SOUNDDEVICE", None)
    mod = TTSModule(ModuleConfig())
    await mod.start(bus)

    entered = asyncio.Event()
    release = asyncio.Event()
    published: list[Event] = []
    original_publish_event = bus.publish_event

    async def finish_on_release(text: str, session) -> None:
        entered.set()
        await release.wait()

    def record_publish(event: Event) -> None:
        published.append(event)
        original_publish_event(event)

    monkeypatch.setattr(mod, "_speak", finish_on_release)
    monkeypatch.setattr(bus, "publish_event", record_publish)

    response_task = asyncio.create_task(
        mod._on_response(
            Event("response_ready", {"text": "almost done"}, trace_id="trace-a")
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    session = mod._session
    assert session is not None and session.task is not None

    # Hold the state lock, queue wake B first, then let the child become done.
    # When the lock is released, wake B must invalidate the completed session
    # before _on_response gets its final-publication turn.
    await mod._state_lock.acquire()
    try:
        wake_task = asyncio.create_task(
            mod._on_wake(Event("wake_word_detected", trace_id="trace-b"))
        )
        await asyncio.sleep(0)
        release.set()
        await _wait_until(session.task.done)
        assert not any(
            event.event_type == "speech_finished" for event in published
        )
    finally:
        mod._state_lock.release()

    await asyncio.gather(wake_task, response_task)
    await mod.stop()

    assert mod._owner_trace_id == "trace-b"
    assert not any(
        event.event_type == "speech_finished" and event.trace_id == "trace-a"
        for event in published
    )


async def test_old_generation_cannot_stop_new_device_owner(
    bus: EventBus, fake_silero
):
    _model, _factory, sounddevice = fake_silero
    mod = TTSModule(ModuleConfig())
    await mod.start(bus)

    async with mod._state_lock:
        mod._device_owner_generation = 2
        old_stopped = await mod._stop_audio_if_owner_locked(1)
        current_stopped = await mod._stop_audio_if_owner_locked(2)
    await mod.stop()

    assert old_stopped is False
    assert current_stopped is True
    assert sounddevice.stop_calls == 1


async def test_interaction_failure_cancels_current_speech_without_stale_finish(
    bus: EventBus, monkeypatch
):
    monkeypatch.setattr(tts_mod, "_SILERO_TTS", None)
    monkeypatch.setattr(tts_mod, "_SOUNDDEVICE", None)
    mod = TTSModule(ModuleConfig())
    await mod.start(bus)
    entered = asyncio.Event()
    published: list[Event] = []
    original_publish = bus.publish_event

    async def blocked_speak(text: str, session) -> None:
        entered.set()
        await asyncio.Event().wait()

    def record(event: Event) -> None:
        published.append(event)
        original_publish(event)

    monkeypatch.setattr(mod, "_speak", blocked_speak)
    monkeypatch.setattr(bus, "publish_event", record)
    response_task = asyncio.create_task(
        mod._on_response(
            Event(
                "response_ready",
                {"text": "never finish"},
                trace_id="failed-trace",
            )
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)

    await mod._on_interaction_failed(
        Event(
            "interaction_failed",
            {"reason": "interaction_timeout"},
            trace_id="failed-trace",
        )
    )
    await asyncio.wait_for(response_task, timeout=1.0)
    await mod.stop()

    assert mod._session is None
    assert mod._owner_trace_id is None
    assert not any(
        event.event_type == "speech_finished" and event.trace_id == "failed-trace"
        for event in published
    )


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition was not reached before timeout")
        await asyncio.sleep(0)
