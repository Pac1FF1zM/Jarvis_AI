"""Persistent-mode lifecycle tests with no microphone or hotkey hardware."""
from __future__ import annotations

import asyncio

import pytest

import main as jarvis_main
from modules.llm import LLMModule
from modules.nlu import NLUModule
from modules.stt import STTModule
from modules.tts import TTSModule
import modules.wake_word as wake_word_module
from modules.wake_word import WakeWordModule


class _TrackingSimulatedWakeWord(WakeWordModule):
    instances: list["_TrackingSimulatedWakeWord"] = []

    def __init__(self, config, *, force_simulated: bool = False) -> None:
        # Persistent main normally requests real input. Force only this test
        # double onto the deterministic trigger path so no OS hook is touched.
        super().__init__(config, force_simulated=True)
        self.stop_calls = 0
        self.__class__.instances.append(self)

    async def stop(self) -> None:
        self.stop_calls += 1
        await super().stop()


async def _wait_until_ready() -> _TrackingSimulatedWakeWord:
    for _ in range(500):
        if _TrackingSimulatedWakeWord.instances:
            wake = _TrackingSimulatedWakeWord.instances[-1]
            if wake.bus is not None and wake.bus._running:
                return wake
        await asyncio.sleep(0.01)
    raise AssertionError("persistent pipeline did not become ready")


async def test_persistent_mode_handles_repeated_interactions_and_clean_shutdown(
    monkeypatch,
):
    _TrackingSimulatedWakeWord.instances.clear()
    monkeypatch.setattr(
        wake_word_module, "WakeWordModule", _TrackingSimulatedWakeWord
    )
    shutdown = asyncio.Event()
    pipeline = asyncio.create_task(
        jarvis_main.run_pipeline("config.yaml", shutdown_event=shutdown)
    )
    wake = await _wait_until_ready()
    completions: asyncio.Queue[str] = asyncio.Queue()

    async def record_completion(event) -> None:
        completions.put_nowait(event.trace_id)

    wake.bus.subscribe("interaction_completed", record_completion)
    trace_ids: list[str] = []
    for _ in range(3):
        activation = await wake.trigger()
        completed_trace = await asyncio.wait_for(completions.get(), timeout=5.0)
        assert completed_trace == activation.trace_id
        trace_ids.append(completed_trace)

    assert len(set(trace_ids)) == 3
    shutdown.set()
    await asyncio.wait_for(pipeline, timeout=5.0)

    assert wake.stop_calls == 1
    assert not wake.bus._running
    assert not wake.bus._tasks


async def test_persistent_mode_cancellation_uses_clean_shutdown_path(monkeypatch):
    _TrackingSimulatedWakeWord.instances.clear()
    monkeypatch.setattr(
        wake_word_module, "WakeWordModule", _TrackingSimulatedWakeWord
    )
    pipeline = asyncio.create_task(jarvis_main.run_pipeline("config.yaml"))
    wake = await _wait_until_ready()

    # asyncio.run translates Ctrl+C/SIGINT into cancellation of its main task
    # on modern Python, so cancellation exercises the same branch deterministically.
    pipeline.cancel()
    await asyncio.wait_for(pipeline, timeout=5.0)

    assert not pipeline.cancelled()
    assert wake.stop_calls == 1
    assert not wake.bus._running
    assert not wake.bus._tasks


async def test_voice_modules_initialize_in_parallel(monkeypatch):
    active = 0
    max_active = 0
    started: set[str] = set()

    async def delayed_start(self, bus) -> None:
        nonlocal active, max_active
        self.bus = bus
        active += 1
        max_active = max(max_active, active)
        started.add(self.name)
        await asyncio.sleep(0.03)
        active -= 1

    for module_type in (WakeWordModule, STTModule, NLUModule, LLMModule, TTSModule):
        monkeypatch.setattr(module_type, "start", delayed_start)

    shutdown = asyncio.Event()
    shutdown.set()
    await jarvis_main.run_pipeline("config.yaml", shutdown_event=shutdown)

    assert started == {"wake_word", "stt", "nlu", "llm", "tts"}
    assert max_active == 5


async def test_parallel_startup_cleans_every_module_when_one_engine_fails(
    monkeypatch,
):
    stopped: list[str] = []

    async def start_with_stt_failure(self, bus) -> None:
        self.bus = bus
        await asyncio.sleep(0.01 if self.name == "stt" else 0.02)
        if self.name == "stt":
            raise RuntimeError("synthetic STT startup failure")

    async def tracked_stop(self) -> None:
        stopped.append(self.name)

    module_types = (WakeWordModule, STTModule, NLUModule, LLMModule, TTSModule)
    for module_type in module_types:
        monkeypatch.setattr(module_type, "start", start_with_stt_failure)
        monkeypatch.setattr(module_type, "stop", tracked_stop)

    with pytest.raises(RuntimeError, match="synthetic STT startup failure"):
        await jarvis_main.run_pipeline("config.yaml", shutdown_event=asyncio.Event())

    assert sorted(stopped) == ["llm", "nlu", "stt", "tts", "wake_word"]
