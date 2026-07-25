"""End-to-end adversarial checks for trace cancellation and supersession."""
from __future__ import annotations

import asyncio
import threading

from core.config_loader import ModuleConfig
from core.event_bus import Event, EventBus
from core.gpu_lock import GPULock
from core.orchestrator import Orchestrator, State
from memory.short_term import ShortTermMemory
import modules.llm as llm_mod
from modules.llm import LLMModule
from tools.registry import ToolRegistry


async def test_wake_during_blocking_llm_drains_worker_without_stale_response(
    monkeypatch,
):
    started = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()

    class BlockingOllama:
        @staticmethod
        def chat(**kwargs):
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test did not release blocking Ollama")
            worker_finished.set()
            return {"message": {"content": "stale answer"}}

    monkeypatch.setattr(llm_mod, "_OLLAMA", BlockingOllama)
    bus = EventBus()
    orch = Orchestrator(
        bus,
        {"listening_timeout_seconds": 2, "interaction_timeout_seconds": 2},
    )
    llm = LLMModule(
        ModuleConfig(
            model="fake",
            params={"input_event": "nlu_result"},
        ),
        GPULock(),
        ToolRegistry(),
        ShortTermMemory(max_turns=4),
    )
    responses: list[Event] = []
    completions: list[Event] = []

    async def record_response(event: Event) -> None:
        responses.append(event)

    async def record_completion(event: Event) -> None:
        completions.append(event)

    await orch.start()
    await llm.start(bus)
    bus.subscribe("response_ready", record_response)
    bus.subscribe("interaction_completed", record_completion)
    run_task = asyncio.create_task(bus.run())

    bus.publish("wake_word_detected", {}, trace_id="trace-a")
    await asyncio.sleep(0)
    bus.publish("audio_captured", {"audio": b"pcm"}, trace_id="trace-a")
    bus.publish(
        "transcription_ready", {"text": "поговорим"}, trace_id="trace-a"
    )
    await asyncio.sleep(0.02)
    bus.publish(
        "nlu_result",
        {
            "text": "поговорим",
            "intent": "general_chat",
            "intent_confidence": 0.99,
            "slots": {},
        },
        trace_id="trace-a",
    )
    assert await asyncio.to_thread(started.wait, 1.0)

    bus.publish("wake_word_detected", {}, trace_id="trace-b")
    await asyncio.sleep(0.05)
    assert orch.state == State.LISTENING
    assert orch._current_trace == "trace-b"
    assert bus.is_trace_cancelled("trace-a")

    release.set()
    assert await asyncio.to_thread(worker_finished.wait, 1.0)
    await asyncio.sleep(0.05)

    assert responses == []
    cancelled = [event for event in completions if event.trace_id == "trace-a"]
    assert len(cancelled) == 1
    assert cancelled[0].payload["reason"] == "superseded"

    await bus.stop()
    await run_task
    await llm.stop()
    await orch.stop()


async def test_two_rapid_wakes_transfer_ownership_once_per_trace():
    bus = EventBus()
    orch = Orchestrator(
        bus,
        {"listening_timeout_seconds": 2, "interaction_timeout_seconds": 2},
    )
    completions: list[Event] = []

    async def record(event: Event) -> None:
        completions.append(event)

    await orch.start()
    bus.subscribe("interaction_completed", record)
    run_task = asyncio.create_task(bus.run())

    bus.publish("wake_word_detected", {}, trace_id="trace-a")
    await asyncio.sleep(0.01)
    bus.publish("audio_captured", {"audio": b"pcm"}, trace_id="trace-a")
    bus.publish(
        "transcription_ready", {"text": "old"}, trace_id="trace-a"
    )
    await asyncio.sleep(0.01)

    bus.publish("wake_word_detected", {}, trace_id="trace-b")
    bus.publish("wake_word_detected", {}, trace_id="trace-c")
    await asyncio.sleep(0.05)

    assert orch.state == State.LISTENING
    assert orch._current_trace == "trace-c"
    assert bus.is_trace_cancelled("trace-a")
    assert bus.is_trace_cancelled("trace-b")
    assert not bus.is_trace_closed("trace-c")
    assert [event.trace_id for event in completions] == ["trace-a", "trace-b"]
    assert all(event.payload["cancelled"] for event in completions)

    await bus.stop()
    await run_task
    await orch.stop()
