"""Regression tests for desktop control, plans, corrections and wake phrase."""
from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from core.config_loader import ModuleConfig
from core.event_bus import Event, EventBus
from core.gpu_lock import GPULock
from memory.short_term import ShortTermMemory
from modules.command_router import (
    RoutedAction,
    route_explicit_command,
    split_compound_command,
)
from modules.llm import LLMModule
from modules.wake_word import WakeWordModule
from tools.registry import ToolRegistry


def test_compound_splitter_only_breaks_at_a_new_command():
    assert split_compound_command(
        "напомни через 10 минут купить хлеб и молоко"
    ) == ["напомни через 10 минут купить хлеб и молоко"]
    assert split_compound_command(
        "открой браузер, поставь напоминание; затем скажи время"
    ) == ["открой браузер", "поставь напоминание", "скажи время"]


def test_explicit_router_distinguishes_web_and_file_search():
    web = route_explicit_command("найди в интернете погоду в Ташкенте")
    file = route_explicit_command("найди файл диплом")

    assert web == RoutedAction(
        "browser_control", {"action": "search", "query": "погоду в ташкенте"}
    )
    assert file == RoutedAction(
        "file_control", {"action": "find", "query": "диплом"}
    )
    assert route_explicit_command("не открывай браузер") is None


def test_dialogue_correction_reuses_previous_action_type(monkeypatch):
    from modules import command_router
    from tools._applications import ApplicationSpec

    discord = ApplicationSpec("discord", "Дискорд", uri="discord://")
    monkeypatch.setattr(command_router, "resolve_application", lambda value: discord if value == "дискорд" else None)
    corrected = route_explicit_command(
        "нет, я имел в виду дискорд",
        previous_action=RoutedAction("open_application", {"application": "paint"}),
    )
    assert corrected == RoutedAction("open_application", {"application": "discord"})


async def test_compound_plan_uses_one_lifecycle_envelope_and_keeps_order():
    bus = EventBus()
    registry = ToolRegistry()
    calls: list[str] = []

    async def first(_params: dict[str, Any]) -> dict[str, Any]:
        calls.append("first")
        # Legacy read-only tools predate the optional `ok` convention.
        return {"response_text": "Первое выполнено."}

    async def second(_params: dict[str, Any]) -> dict[str, Any]:
        calls.append("second")
        return {"ok": True, "response_text": "Второе выполнено."}

    registry._schemas.update({"get_current_time": {"name": "get_current_time"}, "list_applications": {"name": "list_applications"}})
    registry._executors.update({"get_current_time": first, "list_applications": second})
    module = LLMModule(ModuleConfig(params={"input_event": "nlu_result"}), GPULock(), registry, ShortTermMemory(4))
    observed: list[Event] = []

    async def record(event: Event) -> None:
        observed.append(event)

    bus.subscribe("tool_call_requested", record)
    bus.subscribe("response_ready", record)
    await module.start(bus)
    runner = asyncio.create_task(bus.run())
    bus.publish(
        "nlu_result",
        {
            "text": "составная команда",
            "intent": "get_current_time",
            "actions": [
                {"intent": "get_current_time", "slots": {}},
                {"intent": "list_applications", "slots": {}},
            ],
        },
        trace_id="plan-trace",
    )
    await asyncio.sleep(0.1)
    await bus.stop()
    await runner
    await module.stop()

    assert calls == ["first", "second"]
    lifecycle = [event for event in observed if event.event_type == "tool_call_requested"]
    assert len(lifecycle) == 1
    assert lifecycle[0].payload["tool"] == "compound_plan"
    response = next(event for event in observed if event.event_type == "response_ready")
    assert response.payload["text"] == "Первое выполнено. Второе выполнено."


async def test_compound_plan_stops_after_failure_without_later_side_effects():
    bus = EventBus()
    registry = ToolRegistry()
    calls: list[str] = []

    async def failure(_params):
        calls.append("failure")
        return {"ok": False, "response_text": "Не выполнено."}

    async def forbidden(_params):
        calls.append("forbidden")
        return {"ok": True, "response_text": "Не должно выполниться."}

    registry._schemas.update({"get_current_time": {"name": "get_current_time"}, "list_applications": {"name": "list_applications"}})
    registry._executors.update({"get_current_time": failure, "list_applications": forbidden})
    module = LLMModule(ModuleConfig(params={"input_event": "nlu_result"}), GPULock(), registry, ShortTermMemory(4))
    await module.start(bus)
    runner = asyncio.create_task(bus.run())
    bus.publish("nlu_result", {"text": "plan", "intent": "get_current_time", "actions": [{"intent": "get_current_time", "slots": {}}, {"intent": "list_applications", "slots": {}}]}, trace_id="failed-plan")
    await asyncio.sleep(0.1)
    await bus.stop()
    await runner
    await module.stop()

    assert calls == ["failure"]


async def test_dangerous_system_action_requires_a_separate_confirmation_turn():
    bus = EventBus()
    registry = ToolRegistry()
    confirmed_values: list[bool] = []

    async def guarded(params):
        confirmed = bool(params.get("confirmed"))
        confirmed_values.append(confirmed)
        if not confirmed:
            return {"ok": False, "confirmation_required": True, "confirmation": {"tool": "system_control", "params": {"action": "lock", "confirmed": True}}, "response_text": "Подтвердите блокировку."}
        return {"ok": True, "response_text": "Компьютер заблокирован."}

    registry._schemas["system_control"] = {"name": "system_control"}
    registry._executors["system_control"] = guarded
    module = LLMModule(ModuleConfig(params={"input_event": "nlu_result"}), GPULock(), registry, ShortTermMemory(4))
    responses: list[Event] = []

    async def record(event: Event):
        responses.append(event)

    bus.subscribe("response_ready", record)
    await module.start(bus)
    runner = asyncio.create_task(bus.run())
    bus.publish("nlu_result", {"text": "заблокируй компьютер", "intent": "system_control", "slots": {"action": "lock"}}, trace_id="ask")
    await asyncio.sleep(0.05)
    assert confirmed_values == [False]
    assert module._pending_confirmation is not None
    bus.publish("nlu_result", {"text": "подтверждаю", "intent": "confirm", "slots": {}}, trace_id="confirm")
    await asyncio.sleep(0.05)
    await bus.stop()
    await runner
    await module.stop()

    assert confirmed_values == [False, True]
    assert [event.payload["text"] for event in responses] == ["Подтвердите блокировку.", "Компьютер заблокирован."]


async def test_closing_regular_window_needs_no_jarvis_confirmation(monkeypatch):
    from tools import window_control
    from tools._windows import WindowInfo

    closed: list[tuple[str, str]] = []

    def fake_control(action: str, query: str) -> WindowInfo:
        closed.append((action, query))
        return WindowInfo(
            handle=123,
            title="Telegram",
            process_id=456,
            executable="Telegram.exe",
        )

    monkeypatch.setattr(window_control, "control_window", fake_control)

    result = await window_control.execute({"action": "close", "window": "Telegram"})

    assert result["ok"] is True
    assert "confirmation_required" not in result
    assert closed == [("close", "Telegram")]


class _WakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, frames: int):
        return np.zeros((frames, 1), dtype=np.int16), False


class _WakeSoundDevice:
    def InputStream(self, **kwargs):
        assert kwargs["blocksize"] == 1280
        return _WakeStream()


class _WakeModel:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def predict(self, samples):
        assert samples.dtype == np.int16
        self.calls += 1
        return {"hey_jarvis": 0.8}


def test_wake_phrase_requires_consecutive_positive_frames():
    module = WakeWordModule(
        ModuleConfig(params={"wake_phrase_threshold": 0.55, "wake_phrase_frames": 2})
    )
    module._sounddevice = _WakeSoundDevice()
    module._wake_model = _WakeModel()

    assert module._listen_for_wake_sync() is True
    assert module._wake_model.calls == 2
