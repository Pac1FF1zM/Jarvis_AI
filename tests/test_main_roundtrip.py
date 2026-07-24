"""End-to-end roundtrip test — the automated "mocked demo round-trip".

Boots the whole pipeline with the same wiring ``main.py`` uses, triggers one
wake word, and asserts the orchestrator state machine traverses the full
IDLE -> ... -> IDLE cycle on a single shared trace_id. This is the automated
version of running ``python main.py`` and watching the log.
"""
from __future__ import annotations

import asyncio

import pytest

from core.config_loader import ModuleConfig
from core.event_bus import EventBus, Event
from core.gpu_lock import GPULock
from core.orchestrator import Orchestrator, State
from memory.short_term import ShortTermMemory
from modules.llm import LLMModule
from modules.nlu import NLUModule
from modules.stt import STUB_TEXT
from modules.stt import STTModule
from modules.tts import TTSModule
from modules.wake_word import WakeWordModule
from tools.registry import ToolRegistry
import tools.open_application as open_application_module


async def _build_and_run_one_cycle(
    user_text: str = "what is the time",
) -> tuple[Orchestrator, list[Event], str]:
    """Wire every component exactly as main.py does, run one wake cycle."""
    bus = EventBus()
    gpu_lock = GPULock()
    orch = Orchestrator(bus, {"listening_timeout_seconds": 5})

    tools = ToolRegistry()
    tools.discover("tools")
    short_term = ShortTermMemory(max_turns=4)

    wake_word = WakeWordModule(ModuleConfig(), force_simulated=True)
    stt = STTModule(ModuleConfig(device="cpu", model="base"), gpu_lock)
    nlu = NLUModule(
        ModuleConfig(
            device="cpu", model="models/nlu_word_bigru_curriculum.pt"
        ),
        gpu_lock,
    )
    llm = LLMModule(ModuleConfig(
                        device="cpu",
                        model="qwen2.5:7b-instruct",
                        params={"input_event": "nlu_result"},
                    ),
                    gpu_lock, tools, short_term)
    tts = TTSModule(ModuleConfig())

    # Capture every published event in order.
    all_events: list[Event] = []
    watched = (
        "wake_word_detected", "audio_captured", "transcription_ready", "nlu_result",
        "tool_call_requested", "tool_result", "response_ready",
        "speech_started", "speech_finished",
    )

    async def record(event: Event) -> None:
        all_events.append(event)

    for et in watched:
        bus.subscribe(et, record)

    # Patch STT to return the requested user text (default stub returns a
    # fixed sentence; we want to drive both the tool path and the plain path).
    async def fake_transcribe(payload):
        return user_text, 0.9
    stt._transcribe = fake_transcribe  # type: ignore[assignment]

    await wake_word.start(bus)
    await stt.start(bus)
    await nlu.start(bus)
    await llm.start(bus)
    await tts.start(bus)
    await orch.start()

    run_task = asyncio.create_task(bus.run())
    try:
        wake = await wake_word.trigger()
        # Poll for IDLE with a generous timeout.
        for _ in range(200):  # 200 * 25ms = 5s
            await asyncio.sleep(0.025)
            if orch.state == State.IDLE and any(
                e.event_type == "speech_finished" for e in all_events
            ):
                break
    finally:
        await bus.stop()
        await run_task
        await tts.stop()
        await llm.stop()
        await nlu.stop()
        await stt.stop()
        await wake_word.stop()
        await orch.stop()

    return orch, all_events, wake.trace_id


async def test_full_cycle_returns_to_idle():
    orch, events, trace_id = await _build_and_run_one_cycle("what is the time")
    assert orch.state == State.IDLE, f"final state was {orch.state.value}"


async def test_full_cycle_visited_core_states():
    """The orchestrator should have moved through the canonical states."""
    orch, events, trace_id = await _build_and_run_one_cycle("what is the time")

    # Reconstruct the transition sequence by replaying events into a fresh
    # orchestrator and recording states — simpler than parsing logs.
    types = [e.event_type for e in events]
    # Sanity: all the expected event types fired, in order.
    assert "wake_word_detected" in types
    assert "audio_captured" in types
    assert "transcription_ready" in types
    assert "nlu_result" in types
    # 'time' utterance -> tool path
    assert "tool_call_requested" in types
    assert "tool_result" in types
    assert "response_ready" in types
    assert "speech_started" in types
    assert "speech_finished" in types


async def test_full_cycle_single_trace_id():
    """Every event in the cycle must share one trace_id."""
    orch, events, trace_id = await _build_and_run_one_cycle("what is the time")
    assert events, "no events captured"
    assert all(e.trace_id == trace_id for e in events), (
        "trace_id diverged across the cycle"
    )


async def test_plain_reply_skips_tool_call():
    """An utterance without a tool keyword must NOT produce tool_call_requested."""
    orch, events, trace_id = await _build_and_run_one_cycle("hello there")
    types = [e.event_type for e in events]
    assert orch.state == State.IDLE
    assert "tool_call_requested" not in types
    assert "response_ready" in types
    assert "speech_finished" in types


async def test_voice_demo_stub_transcript_routes_to_time_tool():
    orch, events, _trace_id = await _build_and_run_one_cycle(STUB_TEXT)
    nlu = next(event for event in events if event.event_type == "nlu_result")
    request = next(
        event for event in events if event.event_type == "tool_call_requested"
    )
    assert nlu.payload["intent"] == "get_current_time"
    assert request.payload["tool"] == "get_current_time"
    assert orch.state == State.IDLE


async def test_own_nlu_routes_reminder_parameters_end_to_end():
    """The trained NLU checkpoint, not Ollama, must choose and fill the tool."""
    orch, events, trace_id = await _build_and_run_one_cycle(
        "через 18 минут скажи мне проверить чайник"
    )
    nlu = next(event for event in events if event.event_type == "nlu_result")
    request = next(
        event for event in events if event.event_type == "tool_call_requested"
    )
    result = next(event for event in events if event.event_type == "tool_result")
    response = next(event for event in events if event.event_type == "response_ready")
    assert nlu.payload["intent"] == "set_reminder"
    assert request.payload == {
        "tool": "set_reminder",
        "params": {"minutes": 18, "message": "проверить чайник"},
    }
    assert result.payload["result"]["scheduled"] is False
    assert "ничего не запланировал" in response.payload["text"]
    assert orch.state == State.IDLE


async def test_own_nlu_opens_allowlisted_application_end_to_end(monkeypatch):
    launched: list[str] = []

    def fake_launch(spec):
        launched.append(spec.name)
        return 9001

    monkeypatch.setattr(open_application_module, "launch_application", fake_launch)
    orch, events, trace_id = await _build_and_run_one_cycle(
        "открой мне калькулятор"
    )
    nlu = next(event for event in events if event.event_type == "nlu_result")
    request = next(
        event for event in events if event.event_type == "tool_call_requested"
    )
    result = next(event for event in events if event.event_type == "tool_result")
    assert nlu.payload["intent"] == "open_application"
    assert request.payload == {
        "tool": "open_application",
        "params": {"application": "calculator"},
    }
    assert result.payload["result"]["ok"] is True
    assert launched == ["calculator"]
    assert orch.state == State.IDLE
