"""Hardware-free tests for push-to-talk microphone capture and VAD bounds."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pytest

from core.config_loader import ModuleConfig
from core.event_bus import EventBus, Event
from core.orchestrator import Orchestrator, State
from core.profile_manager import device_fingerprint
import modules.wake_word as wake_mod
from modules.wake_word import WakeWordModule


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


class _FakeVAD:
    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.calls = 0
        self.reset_calls = 0

    def reset_states(self) -> None:
        self.reset_calls += 1

    def __call__(self, tensor, sample_rate: int) -> float:
        assert tensor.numel() == 512
        assert sample_rate == 16_000
        score = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return score


class _FakeStream:
    def __init__(self, owner: "_FakeSoundDevice") -> None:
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, frames: int):
        self.owner.read_count += 1
        return np.full((frames, 1), self.owner.sample_value, dtype=np.int16), False


class _FakeSoundDevice:
    def __init__(self) -> None:
        self.read_count = 0
        self.sample_value = 1000
        self.stream_kwargs: list[dict[str, Any]] = []
        self.query_error: BaseException | None = None

    def query_devices(self, device=None, kind=None):
        if self.query_error is not None:
            raise self.query_error
        assert kind == "input"
        return {"name": "Fake microphone", "max_input_channels": 1}

    def InputStream(self, **kwargs):
        self.stream_kwargs.append(kwargs)
        return _FakeStream(self)


class _FakeHotkeys:
    instances: list["_FakeHotkeys"] = []

    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _FakeKeyboard:
    GlobalHotKeys = _FakeHotkeys


def _config(**overrides) -> ModuleConfig:
    params = {
        "hotkey": "<ctrl>+<alt>+<space>",
        "vad_threshold": 0.5,
        "end_silence_ms": 64,
        "speech_start_timeout_ms": 160,
        "min_speech_ms": 64,
        "pre_roll_ms": 32,
        "max_duration_ms": 320,
    }
    params.update(overrides)
    return ModuleConfig(params=params)


def _install_real_fakes(monkeypatch, scores: list[float]):
    vad = _FakeVAD(scores)
    sounddevice = _FakeSoundDevice()
    _FakeHotkeys.instances.clear()
    monkeypatch.setattr(wake_mod, "_SOUNDDEVICE", sounddevice)
    monkeypatch.setattr(wake_mod, "_PYNPUT_KEYBOARD", _FakeKeyboard)
    monkeypatch.setattr(wake_mod, "_LOAD_SILERO_VAD", lambda **kwargs: vad)
    return vad, sounddevice


async def _start_bus_with_recorders(bus: EventBus):
    events: list[Event] = []
    audio_ready = asyncio.Event()

    async def record(event: Event) -> None:
        events.append(event)
        if event.event_type == "audio_captured":
            audio_ready.set()

    bus.subscribe("wake_word_detected", record)
    bus.subscribe("audio_captured", record)
    return events, audio_ready, asyncio.create_task(bus.run())


async def test_hotkey_captures_real_pcm_until_vad_silence(
    bus: EventBus, monkeypatch
):
    vad, sounddevice = _install_real_fakes(
        monkeypatch, [0.1, 0.9, 0.8, 0.1, 0.1]
    )
    mod = WakeWordModule(_config())
    events, audio_ready, run_task = await _start_bus_with_recorders(bus)
    await mod.start(bus)
    listener = _FakeHotkeys.instances[-1]
    assert listener.started

    listener.callbacks["<ctrl>+<alt>+<space>"]()
    # The first torch import can be relatively slow on a cold Windows process.
    await asyncio.wait_for(audio_ready.wait(), timeout=5.0)
    await bus.stop()
    await run_task
    await mod.stop()

    assert [event.event_type for event in events] == [
        "wake_word_detected", "audio_captured"
    ]
    assert events[0].trace_id == events[1].trace_id
    payload = events[1].payload
    assert payload["source"] == "microphone"
    assert payload["sample_rate"] == 16_000
    assert payload["channels"] == 1
    assert payload["sample_width"] == 2
    assert payload["capture_end"] == "vad_silence"
    assert payload["audio"] and len(payload["audio"]) % 2 == 0
    assert sounddevice.read_count == 5
    assert sounddevice.stream_kwargs == [{
        "samplerate": 16_000, "channels": 1, "dtype": "int16",
        "blocksize": 512, "latency": "low",
    }]
    assert vad.reset_calls == 1
    assert listener.stopped


async def test_max_duration_caps_recording(bus: EventBus, monkeypatch):
    _vad, sounddevice = _install_real_fakes(monkeypatch, [0.9])
    mod = WakeWordModule(
        _config(max_duration_ms=96, min_speech_ms=32, end_silence_ms=500)
    )
    events, audio_ready, run_task = await _start_bus_with_recorders(bus)
    await mod.start(bus)

    await mod.activate()
    await asyncio.wait_for(audio_ready.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()

    audio = next(event for event in events if event.event_type == "audio_captured")
    assert sounddevice.read_count == 3
    assert audio.payload["duration_ms"] == 96
    assert audio.payload["capture_end"] == "max_duration"


async def test_matching_microphone_applies_profile_vad_and_gain(
    bus: EventBus, monkeypatch
):
    device = {"name": "Fake microphone", "max_input_channels": 1}
    calibration = {
        "device_fingerprint": device_fingerprint(device),
        "vad_start_threshold": 0.7,
        "vad_end_threshold": 0.3,
        "end_silence_ms": 64,
        "min_speech_ms": 32,
        "pre_roll_ms": 32,
        "pcm_gain_db": 6.0,
    }
    _vad, sounddevice = _install_real_fakes(monkeypatch, [0.8, 0.4, 0.2, 0.2])
    other = dict(calibration, device_fingerprint="0000000000000000", pcm_gain_db=-6)
    mod = WakeWordModule(
        _config(
            voice_calibrations={
                other["device_fingerprint"]: other,
                calibration["device_fingerprint"]: calibration,
            }
        )
    )
    events, audio_ready, run_task = await _start_bus_with_recorders(bus)
    await mod.start(bus)

    await mod.activate()
    await asyncio.wait_for(audio_ready.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()

    audio = next(event for event in events if event.event_type == "audio_captured")
    samples = np.frombuffer(audio.payload["audio"], dtype=np.int16)
    assert audio.payload["voice_calibrated"] is True
    assert audio.payload["input_device_fingerprint"] == device_fingerprint(device)
    assert samples[0] == 1995
    # 0.4 keeps speech alive through the lower calibrated exit threshold.
    assert sounddevice.read_count == 4


async def test_calibration_for_another_microphone_is_ignored(
    bus: EventBus, monkeypatch, caplog
):
    calibration = {
        "device_fingerprint": device_fingerprint(
            {"name": "Another microphone", "max_input_channels": 1}
        ),
        "vad_start_threshold": 0.9,
        "vad_end_threshold": 0.8,
        "pcm_gain_db": 12,
    }
    _vad, _sounddevice = _install_real_fakes(monkeypatch, [0.9])
    mod = WakeWordModule(
        _config(
            voice_calibrations={calibration["device_fingerprint"]: calibration}
        )
    )

    with caplog.at_level(logging.WARNING, logger="jarvis.module.wake_word"):
        await mod.start(bus)
        await mod.activate()
    await mod.stop()

    assert mod._calibration_applied is False
    assert mod._speech_threshold == 0.5
    assert "VOICE_CALIBRATION_SKIPPED" in caplog.text


async def test_silent_recording_publishes_no_audio_and_does_not_crash(
    bus: EventBus, monkeypatch, caplog
):
    _vad, sounddevice = _install_real_fakes(monkeypatch, [0.0])
    mod = WakeWordModule(_config(speech_start_timeout_ms=96))
    events, _audio_ready, run_task = await _start_bus_with_recorders(bus)
    await mod.start(bus)

    with caplog.at_level(logging.WARNING, logger="jarvis.module.wake_word"):
        wake = await mod.activate()
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task
    await mod.stop()

    assert wake is not None
    assert [event.event_type for event in events] == ["wake_word_detected"]
    assert sounddevice.read_count == 3
    assert "MICROPHONE_EMPTY" in caplog.text


async def test_microphone_permission_error_is_logged_and_returns(
    bus: EventBus, monkeypatch, caplog
):
    _vad, sounddevice = _install_real_fakes(monkeypatch, [0.9])
    sounddevice.query_error = PermissionError("microphone denied")
    mod = WakeWordModule(_config())
    events, _audio_ready, run_task = await _start_bus_with_recorders(bus)
    await mod.start(bus)

    with caplog.at_level(logging.ERROR, logger="jarvis.module.wake_word"):
        await mod.activate()
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task
    await mod.stop()

    assert [event.event_type for event in events] == ["wake_word_detected"]
    assert "MICROPHONE_CAPTURE_FAILED" in caplog.text
    assert "permissions" in caplog.text


async def test_missing_libraries_keep_simulated_trigger(
    bus: EventBus, monkeypatch, caplog
):
    monkeypatch.setattr(wake_mod, "_SOUNDDEVICE", None)
    monkeypatch.setattr(wake_mod, "_PYNPUT_KEYBOARD", None)
    monkeypatch.setattr(wake_mod, "_LOAD_SILERO_VAD", None)
    mod = WakeWordModule(_config())
    events, audio_ready, run_task = await _start_bus_with_recorders(bus)

    with caplog.at_level(logging.WARNING, logger="jarvis.module.wake_word"):
        await mod.start(bus)
    wake = await mod.trigger()
    await asyncio.wait_for(audio_ready.wait(), timeout=1.0)
    await bus.stop()
    await run_task
    await mod.stop()

    assert not mod.real_activation_enabled
    assert [event.event_type for event in events] == [
        "wake_word_detected", "audio_captured"
    ]
    assert all(event.trace_id == wake.trace_id for event in events)
    assert events[1].payload["source"] == "simulated"
    assert events[1].payload["audio"] == b"<stub-pcm-chunks>"
    assert "real push-to-talk unavailable" in caplog.text


async def test_trigger_requires_start_first(bus: EventBus):
    mod = WakeWordModule(_config(), force_simulated=True)
    with pytest.raises(RuntimeError, match="start"):
        await mod.trigger()


async def test_wake_word_subscribes_to_all_trace_termination_cleanup(bus: EventBus):
    mod = WakeWordModule(_config(), force_simulated=True)
    await mod.start(bus)

    assert set(bus._subscribers) == {
        "interaction_cancelled",
        "interaction_completed",
        "interaction_failed",
        "speech_started",
        "speech_finished",
    }
    assert bus._subscribers["interaction_failed"] == [mod._on_interaction_failed]


async def test_successful_voice_turn_opens_bounded_active_session(
    bus: EventBus, monkeypatch
):
    mod = WakeWordModule(
        _config(active_session_enabled=True, active_session_timeout_seconds=0.1),
        force_simulated=True,
    )
    await mod.start(bus)
    mod.real_activation_enabled = True
    mod._trace_sources["first-turn"] = "push_to_talk"
    monkeypatch.setattr(
        mod,
        "_record_microphone_sync",
        lambda timeout_ms=None: wake_mod._CaptureResult(
            pcm=b"\x01\x00" * 512,
            duration_ms=32,
            end_reason="vad_silence",
        ),
    )
    events: list[Event] = []
    audio_ready = asyncio.Event()

    async def record(event: Event) -> None:
        events.append(event)
        if event.event_type == "audio_captured":
            audio_ready.set()

    bus.subscribe("wake_word_detected", record)
    bus.subscribe("audio_captured", record)
    runner = asyncio.create_task(bus.run())
    bus.publish(
        "interaction_completed",
        {"state": "IDLE", "ok": True},
        trace_id="first-turn",
    )

    await asyncio.wait_for(audio_ready.wait(), timeout=1.0)
    await mod.stop()
    await bus.stop()
    await runner

    assert [event.event_type for event in events] == [
        "wake_word_detected",
        "audio_captured",
    ]
    assert events[0].payload["source"] == "active_session"
    assert events[1].trace_id == events[0].trace_id


async def test_active_session_silence_returns_to_sleep_without_audio(
    bus: EventBus, monkeypatch
):
    mod = WakeWordModule(
        _config(active_session_enabled=True, active_session_timeout_seconds=0.1),
        force_simulated=True,
    )
    await mod.start(bus)
    mod.real_activation_enabled = True
    mod._trace_sources["first-turn"] = "wake_phrase"
    monkeypatch.setattr(mod, "_record_microphone_sync", lambda timeout_ms=None: None)
    events: list[Event] = []
    cancelled = asyncio.Event()

    async def record(event: Event) -> None:
        events.append(event)
        if event.event_type == "cancel_requested":
            cancelled.set()

    bus.subscribe("wake_word_detected", record)
    bus.subscribe("audio_captured", record)
    bus.subscribe("cancel_requested", record)
    runner = asyncio.create_task(bus.run())
    bus.publish(
        "interaction_completed",
        {"state": "IDLE", "ok": True},
        trace_id="first-turn",
    )

    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    await mod.stop()
    await bus.stop()
    await runner

    assert [event.event_type for event in events] == [
        "wake_word_detected",
        "cancel_requested",
    ]
    assert events[-1].payload["reason"] == "active_session_timeout"


async def test_active_session_timeout_completes_cleanly_back_in_idle(
    bus: EventBus, monkeypatch
):
    """The seven-second polling window is a clean session boundary, not a failure."""
    mod = WakeWordModule(
        _config(active_session_enabled=True, active_session_timeout_seconds=0.1),
        force_simulated=True,
    )
    orchestrator = Orchestrator(
        bus,
        {"listening_timeout_seconds": 1, "interaction_timeout_seconds": 2},
    )
    await mod.start(bus)
    await orchestrator.start()
    mod.real_activation_enabled = True
    mod._trace_sources["spoken-turn"] = "push_to_talk"
    monkeypatch.setattr(mod, "_record_microphone_sync", lambda timeout_ms=None: None)
    completed: asyncio.Queue[Event] = asyncio.Queue()
    bus.subscribe("interaction_completed", completed.put)
    runner = asyncio.create_task(bus.run())

    orchestrator.state = State.SPEAKING
    orchestrator._current_trace = "spoken-turn"
    bus.publish("speech_finished", {"text": "Готово."}, trace_id="spoken-turn")

    first = await asyncio.wait_for(completed.get(), timeout=1.0)
    second = await asyncio.wait_for(completed.get(), timeout=1.0)
    await mod.stop()
    await bus.stop()
    await runner
    await orchestrator.stop()

    assert first.trace_id == "spoken-turn"
    assert first.payload["ok"] is True
    assert second.trace_id != first.trace_id
    assert second.payload["cancelled"] is True
    assert second.payload["reason"] == "active_session_timeout"
    assert orchestrator.state == State.IDLE
    assert orchestrator._current_trace is None


async def test_detected_wake_phrase_enters_real_microphone_capture(
    bus: EventBus, monkeypatch
):
    mod = WakeWordModule(_config(), force_simulated=True)
    await mod.start(bus)
    mod.real_activation_enabled = True
    detection_calls = 0

    def detect_once() -> bool:
        nonlocal detection_calls
        detection_calls += 1
        if detection_calls == 1:
            return True
        mod._shutdown_requested.set()
        return False

    monkeypatch.setattr(mod, "_listen_for_wake_sync", detect_once)
    monkeypatch.setattr(
        mod,
        "_record_microphone_sync",
        lambda timeout_ms=None: wake_mod._CaptureResult(
            pcm=b"\x01\x00" * 512,
            duration_ms=32,
            end_reason="vad_silence",
        ),
    )
    audio_ready = asyncio.Event()
    events: list[Event] = []

    async def record(event: Event) -> None:
        events.append(event)
        if event.event_type == "audio_captured":
            mod._shutdown_requested.set()
            audio_ready.set()

    bus.subscribe("wake_word_detected", record)
    bus.subscribe("audio_captured", record)
    runner = asyncio.create_task(bus.run())
    listener = asyncio.create_task(mod._wake_listener_loop())

    await asyncio.wait_for(audio_ready.wait(), timeout=1.0)
    await listener
    await mod.stop()
    await bus.stop()
    await runner

    assert [event.event_type for event in events] == [
        "wake_word_detected",
        "audio_captured",
    ]
    assert events[0].payload["source"] == "wake_phrase"
    assert events[1].trace_id == events[0].trace_id
