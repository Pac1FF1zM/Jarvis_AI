"""Tests for modules/llm.py.

Existing coverage (unchanged behavior, stub/fallback path):
  - plain reply path,
  - tool-call path (tool_call_requested -> tool_result -> response_ready re-entry),
  - trace_id propagation,
  - short-term memory recording,
  - GPU lock acquisition on the inference path.

New coverage for the real Ollama wiring (all mocked — no Ollama install, no
running server, no model download):
  - a mocked response with ``tool_calls`` drives the existing
    ``tool_call_requested`` -> execute -> ``tool_result`` -> re-entry flow,
    ending in ``response_ready``,
  - a mocked plain-text response goes straight to ``response_ready``,
  - ``asyncio.to_thread`` is actually used (the blocking call isn't awaited
    directly on the event loop),
  - the ``ollama`` package missing, and separately a connection error on the
    first call, both fall back to stub mode without raising and without
    crashing the handlers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from core.config_loader import ModuleConfig
from core.event_bus import EventBus, Event
from core.gpu_lock import GPULock
from core.event_payloads import GestureModeChangedPayload
import modules.llm as llm_mod
from memory.short_term import ShortTermMemory
from modules.llm import LLMModule
from tools.registry import ToolRegistry


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


def _build_llm(bus, gpu_lock=None, tools=None, stm=None):
    # NOTE: use `is not None` for stm — an empty ShortTermMemory is falsy
    # (len()==0), so `stm or ShortTermMemory(...)` would silently replace a
    # caller-provided empty memory with a fresh throwaway.
    if tools is None:
        tools = ToolRegistry()
        tools.discover("tools")
    return LLMModule(
        config=ModuleConfig(device="cpu", model="qwen2.5:7b-instruct"),
        gpu_lock=gpu_lock if gpu_lock is not None else GPULock(),
        tools=tools,
        short_term=stm if stm is not None else ShortTermMemory(max_turns=4),
    )


def _recorder(sink: list):
    """Async handler that appends each event into ``sink``."""
    async def _record(event: Event) -> None:
        sink.append(event)
    return _record


async def _noop(event: Event) -> None:
    pass


async def test_trace_cancellation_stops_async_tool_and_suppresses_output(
    bus: EventBus,
):
    started = asyncio.Event()
    tool_cancelled = asyncio.Event()
    tools = ToolRegistry()

    async def slow_tool(params):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise

    tools._schemas["slow_tool"] = {
        "name": "slow_tool",
        "x-cancellation-mode": "cancel",
    }
    tools._executors["slow_tool"] = slow_tool
    mod = _build_llm(bus, tools=tools)
    outputs: list[Event] = []
    bus.subscribe("tool_result", _recorder(outputs))
    bus.subscribe("response_ready", _recorder(outputs))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    input_event = Event("transcription_ready", {"text": "slow"}, trace_id="slow-trace")
    tool_handler = asyncio.create_task(
        mod._request_tool(input_event, "slow_tool", {}, direct_response=True)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    bus.cancel_trace("slow-trace", reason="user_requested")
    await asyncio.wait_for(tool_cancelled.wait(), timeout=1.0)
    await asyncio.wait_for(tool_handler, timeout=1.0)
    await asyncio.sleep(0.02)
    await bus.stop()
    await run_task
    await mod.stop()

    assert outputs == []
    assert mod._active_tool_tasks == {}


async def test_non_preemptible_tool_drains_after_cancel_without_stale_output(
    bus: EventBus,
):
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    tools = ToolRegistry()

    async def thread_style_tool(params):
        started.set()
        await release.wait()
        finished.set()
        return {"response_text": "stale side effect result"}

    # No opt-in cancellation flag: side-effecting tools default to drain.
    tools._schemas["thread_tool"] = {"name": "thread_tool"}
    tools._executors["thread_tool"] = thread_style_tool
    mod = _build_llm(bus, tools=tools)
    outputs: list[Event] = []
    bus.subscribe("tool_result", _recorder(outputs))
    bus.subscribe("response_ready", _recorder(outputs))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    input_event = Event("transcription_ready", {"text": "run"}, trace_id="drain-tool")
    tool_handler = asyncio.create_task(
        mod._request_tool(input_event, "thread_tool", {}, direct_response=True)
    )
    await asyncio.wait_for(started.wait(), timeout=1.0)
    bus.cancel_trace("drain-tool", reason="user_requested")
    await asyncio.sleep(0.02)

    assert not tool_handler.done()
    assert not finished.is_set()
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=1.0)
    await asyncio.wait_for(tool_handler, timeout=1.0)
    await asyncio.sleep(0.02)

    assert outputs == []
    assert mod._active_tool_tasks == {}
    await bus.stop()
    await run_task
    await mod.stop()


# =========================================================================== #
# EXISTING TESTS — stub/fallback path (must keep passing unchanged)
# =========================================================================== #
async def test_plain_reply_path(bus: EventBus):
    mod = _build_llm(bus)
    out: list[Event] = []
    bus.subscribe("response_ready", _recorder(out))
    bus.subscribe("tool_call_requested", _recorder(out))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "hello there", "confidence": 0.9},
        trace_id="plain-tr",
    )
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    await mod.stop()

    types = [e.event_type for e in out]
    assert "response_ready" in types
    assert "tool_call_requested" not in types, "no tool keyword -> no tool call"
    resp = next(e for e in out if e.event_type == "response_ready")
    assert resp.trace_id == "plain-tr"
    assert "text" in resp.payload


async def test_tool_call_then_response(bus: EventBus):
    """A 'time' utterance triggers get_current_time, then a final response."""
    mod = _build_llm(bus)
    out: list[Event] = []
    for et in ("response_ready", "tool_call_requested", "tool_result"):
        bus.subscribe(et, _recorder(out))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "what is the time", "confidence": 0.9},
        trace_id="tool-tr",
    )
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    await mod.stop()

    types = [e.event_type for e in out]
    assert "tool_call_requested" in types, "expected a tool call for 'time'"
    assert "tool_result" in types
    assert "response_ready" in types, "expected final response after tool"
    # All on the same trace.
    assert all(e.trace_id == "tool-tr" for e in out)


async def test_short_term_memory_records_turn(bus: EventBus):
    stm = ShortTermMemory(max_turns=4)
    mod = _build_llm(bus, stm=stm)
    bus.subscribe("response_ready", _noop)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "hello", "confidence": 0.9},
        trace_id="mem-tr",
    )
    await asyncio.sleep(0.2)
    await bus.stop()
    await run_task
    await mod.stop()

    ctx = stm.as_context()
    roles = [c["role"] for c in ctx]
    assert roles == ["user", "assistant"], roles


async def test_gpu_lock_acquired_on_inference(bus: EventBus, caplog):
    mod = _build_llm(bus)
    bus.subscribe("response_ready", _noop)
    await mod.start(bus)

    with caplog.at_level(logging.INFO, logger="jarvis.gpu"):
        run_task = asyncio.create_task(bus.run())
        bus.publish(
            "transcription_ready",
            {"text": "hello", "confidence": 0.9},
            trace_id="gpu-tr",
        )
        await asyncio.sleep(0.2)
        await bus.stop()
        await run_task
        await mod.stop()

    assert any(
        "GPU_ACQUIRE label=llm" in r.message for r in caplog.records
    ), "LLM must acquire the GPU lock for inference"


# =========================================================================== #
# NEW TESTS — real Ollama path (mocked; no install / server / model)
# =========================================================================== #
class _FakeFunction:
    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeOllamaPkg:
    """Stands in for the `ollama` package; records chat() calls."""

    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.chat_calls: list[dict] = []
        self.show_calls: list[str] = []
        self.show_error: BaseException | None = None

    def chat(self, model, messages, tools=None):
        self.chat_calls.append(
            {"model": model, "messages": messages, "tools": tools}
        )
        if self.error is not None:
            raise self.error
        return self.response

    def show(self, model):
        self.show_calls.append(model)
        if self.show_error is not None:
            raise self.show_error
        return {"model": model}


@pytest.fixture
def fake_ollama(monkeypatch):
    """Inject a controllable fake `ollama` package into modules.llm."""
    pkg = _FakeOllamaPkg()
    monkeypatch.setattr(llm_mod, "_OLLAMA", pkg)
    return pkg


async def test_startup_probe_caches_unavailable_ollama_before_first_turn(
    bus: EventBus, fake_ollama, caplog
):
    fake_ollama.show_error = ConnectionError("connection refused")
    tools = ToolRegistry()
    mod = LLMModule(
        config=ModuleConfig(
            device="cpu",
            model="qwen2.5:7b-instruct",
            params={"probe_on_start": True},
        ),
        gpu_lock=GPULock(),
        tools=tools,
        short_term=ShortTermMemory(max_turns=4),
    )

    with caplog.at_level(logging.WARNING, logger="jarvis.module.llm"):
        await mod.start(bus)

    assert fake_ollama.show_calls == ["qwen2.5:7b-instruct"]
    assert mod._server_down is True
    assert any("startup probe" in record.message for record in caplog.records)
    await mod.stop()


async def test_mocked_tool_calls_drive_full_tool_flow(bus: EventBus, fake_ollama):
    """A response carrying tool_calls drives tool_call_requested -> tool_result
    -> re-entry -> response_ready, with the tool name/args flowing through."""
    # First call: model requests a tool. Second call: model emits final text.
    responses = iter(
        [
            _FakeResponse(
                _FakeMessage(
                    tool_calls=[_FakeToolCall("get_current_time", {})]
                )
            ),
            _FakeResponse(_FakeMessage(content="It is noon.")),
        ]
    )
    fake_ollama.response = next(responses)

    # Swap response on each call so the second (re-entry) call gets final text.
    original_chat = fake_ollama.chat

    def chat_swapper(model, messages, tools):
        result = original_chat(model, messages, tools)
        try:
            fake_ollama.response = next(responses)
        except StopIteration:
            pass
        return result

    fake_ollama.chat = chat_swapper

    mod = _build_llm(bus)
    out: list[Event] = []
    for et in ("response_ready", "tool_call_requested", "tool_result"):
        bus.subscribe(et, _recorder(out))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "what is the time", "confidence": 0.9},
        trace_id="m-tool",
    )
    await asyncio.sleep(0.4)
    await bus.stop()
    await run_task
    await mod.stop()

    types = [e.event_type for e in out]
    assert "tool_call_requested" in types
    assert "tool_result" in types
    assert "response_ready" in types

    # The tool_call_requested payload carries the model-emitted name/args.
    req = next(e for e in out if e.event_type == "tool_call_requested")
    assert req.payload["tool"] == "get_current_time"
    assert req.payload["params"] == {}

    # The final response text is the model's second-call content.
    resp = next(e for e in out if e.event_type == "response_ready")
    assert resp.payload["text"] == "It is noon."
    # All events on one trace.
    assert all(e.trace_id == "m-tool" for e in out)


async def test_mocked_plain_text_goes_straight_to_response(bus: EventBus, fake_ollama):
    """A plain-text response bypasses tool_call_requested entirely."""
    fake_ollama.response = _FakeResponse(_FakeMessage(content="Hi there."))

    mod = _build_llm(bus)
    out: list[Event] = []
    bus.subscribe("response_ready", _recorder(out))
    bus.subscribe("tool_call_requested", _recorder(out))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "hello", "confidence": 0.9},
        trace_id="m-plain",
    )
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    await mod.stop()

    types = [e.event_type for e in out]
    assert "tool_call_requested" not in types
    assert "response_ready" in types
    resp = next(e for e in out if e.event_type == "response_ready")
    assert resp.payload["text"] == "Hi there."
    assert resp.trace_id == "m-plain"


async def test_asyncio_to_thread_is_used(bus: EventBus, fake_ollama, monkeypatch):
    """The blocking ollama.chat call must go via asyncio.to_thread."""
    fake_ollama.response = _FakeResponse(_FakeMessage(content="via thread"))

    mod = _build_llm(bus)
    bus.subscribe("response_ready", _noop)
    await mod.start(bus)

    to_thread_calls: list[Any] = []
    original_to_thread = asyncio.to_thread

    async def spy_to_thread(fn, *args, **kwargs):
        to_thread_calls.append(fn)
        return await original_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(llm_mod.asyncio, "to_thread", spy_to_thread)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "hello", "confidence": 0.9},
        trace_id="m-thread",
    )
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    await mod.stop()

    assert to_thread_calls, (
        "ollama.chat must be called via asyncio.to_thread, not awaited directly"
    )
    # The first to_thread call should be the ollama.chat bound method. Bound
    # methods create a new object on each attribute access, so we compare by
    # equality (==) rather than identity (is).
    first_fn = to_thread_calls[0]
    assert first_fn == llm_mod._OLLAMA.chat, (
        f"expected first to_thread call to be ollama.chat, got {first_fn!r}"
    )


async def test_package_missing_falls_back_to_stub(bus: EventBus, monkeypatch, caplog):
    """Simulate ollama not installed: module stays in stub mode, no crash."""
    monkeypatch.setattr(llm_mod, "_OLLAMA", None)
    mod = _build_llm(bus)

    with caplog.at_level(logging.WARNING, logger="jarvis.module.llm"):
        await mod.start(bus)
    assert any(
        "ollama package not installed" in r.message for r in caplog.records
    ), "expected actionable not-installed warning"

    out: list[Event] = []
    bus.subscribe("response_ready", _recorder(out))
    bus.subscribe("tool_call_requested", _recorder(out))
    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "what is the time", "confidence": 0.9},
        trace_id="missing-tr",
    )
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    await mod.stop()

    # Stub heuristic still produces a tool call for "time".
    types = [e.event_type for e in out]
    assert "tool_call_requested" in types
    assert "response_ready" in types


async def test_connection_error_falls_back_and_caches_down_state(
    bus: EventBus, fake_ollama, caplog
):
    """First-call connection error -> actionable warning + stub fallback, and
    the server-down flag is cached so subsequent turns skip the retry."""
    fake_ollama.error = ConnectionError("connection refused")

    mod = _build_llm(bus)
    out: list[Event] = []
    for et in ("response_ready", "tool_call_requested"):
        bus.subscribe(et, _recorder(out))
    await mod.start(bus)
    assert mod._server_down is False

    with caplog.at_level(logging.WARNING, logger="jarvis.module.llm"):
        run_task = asyncio.create_task(bus.run())
        bus.publish(
            "transcription_ready",
            {"text": "hello", "confidence": 0.9},
            trace_id="conn-tr-1",
        )
        await asyncio.sleep(0.3)
        # Server-down flag should now be cached.
        assert mod._server_down is True
        # Fire a second turn — it should NOT re-invoke ollama.chat.
        chat_calls_before = len(fake_ollama.chat_calls)
        bus.publish(
            "transcription_ready",
            {"text": "again", "confidence": 0.9},
            trace_id="conn-tr-2",
        )
        await asyncio.sleep(0.3)
        await bus.stop()
        await run_task
        await mod.stop()

    assert any(
        "Ollama server unreachable" in r.message for r in caplog.records
    ), "expected actionable unreachable warning"
    # Second turn must not have made another chat call (cached down state).
    assert len(fake_ollama.chat_calls) == chat_calls_before, (
        "cached server-down should skip the retry on turn 2"
    )
    # And both turns still produced responses via stub fallback.
    types = [e.event_type for e in out]
    assert types.count("response_ready") == 2


async def test_non_connection_error_is_reraised(bus: EventBus, fake_ollama):
    """A non-connection error from ollama.chat must not be silently swallowed."""
    fake_ollama.error = ValueError("malformed request")

    mod = _build_llm(bus)
    bus.subscribe("response_ready", _noop)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "hello", "confidence": 0.9},
        trace_id="err-tr",
    )
    await asyncio.sleep(0.3)
    await bus.stop()
    await run_task
    # The handler error is caught by EventBus._safe_dispatch and logged, but
    # the module must NOT have silently switched to stub mode for a
    # non-connection error.
    assert mod._server_down is False, (
        "non-connection errors must not cache the server-down flag"
    )


async def test_tool_result_message_role_is_tool(bus: EventBus, fake_ollama):
    """On re-entry after a tool call, the tool output must reach the model as
    a role='tool' message (the OpenAI/Ollama tool-message convention)."""
    responses = iter(
        [
            _FakeResponse(
                _FakeMessage(tool_calls=[_FakeToolCall("get_current_time", {})])
            ),
            _FakeResponse(_FakeMessage(content="It is noon.")),
        ]
    )
    fake_ollama.response = next(responses)
    original_chat = fake_ollama.chat

    def chat_swapper(model, messages, tools):
        result = original_chat(model, messages, tools)
        try:
            fake_ollama.response = next(responses)
        except StopIteration:
            pass
        return result

    fake_ollama.chat = chat_swapper

    mod = _build_llm(bus)
    bus.subscribe("response_ready", _noop)
    bus.subscribe("tool_call_requested", _noop)
    bus.subscribe("tool_result", _noop)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "transcription_ready",
        {"text": "what is the time", "confidence": 0.9},
        trace_id="role-tr",
    )
    await asyncio.sleep(0.4)
    await bus.stop()
    await run_task
    await mod.stop()

    # The second chat call (re-entry) must include a role="tool" message.
    assert len(fake_ollama.chat_calls) == 2
    second_messages = fake_ollama.chat_calls[1]["messages"]
    tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
    assert tool_msgs, (
        "tool re-entry must pass the tool output as a role='tool' message"
    )


async def test_nlu_routing_does_not_expose_or_honor_ollama_tools(
    bus: EventBus, fake_ollama
):
    """NLU owns actions: chat cannot smuggle an Ollama tool call through."""
    fake_ollama.response = _FakeResponse(
        _FakeMessage(
            content="Обычный ответ.",
            tool_calls=[_FakeToolCall("open_application", {"application": "calc"})],
        )
    )
    tools = ToolRegistry()
    tools.discover("tools")
    mod = LLMModule(
        config=ModuleConfig(
            device="cpu",
            model="qwen2.5:7b-instruct",
            params={"input_event": "nlu_result"},
        ),
        gpu_lock=GPULock(),
        tools=tools,
        short_term=ShortTermMemory(max_turns=4),
    )
    responses: list[Event] = []
    tool_requests: list[Event] = []
    bus.subscribe("response_ready", _recorder(responses))
    bus.subscribe("tool_call_requested", _recorder(tool_requests))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "nlu_result",
        {"text": "расскажи шутку", "intent": "general_chat", "slots": {}},
        trace_id="nlu-authority",
    )
    await asyncio.sleep(0.2)
    await bus.stop()
    await run_task
    await mod.stop()

    assert fake_ollama.chat_calls[0]["tools"] is None
    assert not tool_requests
    assert responses[0].payload["text"] == "Обычный ответ."


async def test_nlu_tool_result_is_returned_without_second_ollama_call(
    bus: EventBus, fake_ollama
):
    tools = ToolRegistry()
    tools.discover("tools")
    mod = LLMModule(
        config=ModuleConfig(
            device="cpu",
            model="qwen2.5:7b-instruct",
            params={"input_event": "nlu_result"},
        ),
        gpu_lock=GPULock(),
        tools=tools,
        short_term=ShortTermMemory(max_turns=4),
    )
    responses: list[Event] = []
    bus.subscribe("response_ready", _recorder(responses))
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish(
        "nlu_result",
        {"text": "сколько времени", "intent": "get_current_time", "slots": {}},
        trace_id="direct-tool",
    )
    await asyncio.sleep(0.2)
    await bus.stop()
    await run_task
    await mod.stop()

    assert fake_ollama.chat_calls == []
    assert len(responses) == 1
    assert responses[0].payload["text"].startswith("Сейчас ")


async def test_voice_gesture_mode_request_waits_for_the_runtime_result(bus: EventBus):
    """Jarvis confirms gesture mode only after the CV runtime accepts it."""
    tools = ToolRegistry()
    mod = LLMModule(
        config=ModuleConfig(params={"input_event": "nlu_result"}),
        gpu_lock=GPULock(),
        tools=tools,
        short_term=ShortTermMemory(max_turns=4),
        gesture_enabled=True,
    )
    responses: list[Event] = []

    async def accept_mode(event: Event) -> None:
        bus.publish_event(
            event.child(
                "gesture_mode_changed",
                GestureModeChangedPayload(armed=True, source="voice"),
            )
        )

    bus.subscribe("response_ready", _recorder(responses))
    bus.subscribe("gesture_mode_requested", accept_mode)
    await mod.start(bus)
    runner = asyncio.create_task(bus.run())
    bus.publish(
        "nlu_result",
        {
            "text": "включи режим жестов",
            "intent": "gesture_mode",
            "slots": {"enabled": True},
        },
        trace_id="gesture-voice",
    )
    await asyncio.sleep(0.08)
    await bus.stop()
    await runner
    await mod.stop()

    assert len(responses) == 1
    assert "Режим жестов включен" in responses[0].payload["text"]
