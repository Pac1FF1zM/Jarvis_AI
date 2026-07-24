"""Tests for the orchestrator hardening (fixes #1, #2, #3).

#1 — orchestrator must NOT publish fabricated module payloads (audio_captured).
#2 — invalid transitions are authoritative: no side effects leak past a reject.
#3 — barge-in is SPEAKING-only; wake in other active states is a clean no-op.
"""
from __future__ import annotations

import asyncio

import pytest

from core.event_bus import EventBus
from core.orchestrator import Orchestrator, State


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def orch(bus: EventBus) -> Orchestrator:
    return Orchestrator(bus, {"listening_timeout_seconds": 60})


# --------------------------------------------------------------------------- #
# Fix #1 — orchestrator must not originate module payload data
# --------------------------------------------------------------------------- #
async def test_wake_does_not_publish_audio_captured(bus: EventBus, orch: Orchestrator):
    """The orchestrator reacting to a wake word must NOT emit audio_captured.

    Audio capture triggering belongs to a separate audio-capture module that
    subscribes to wake_word_detected — the orchestrator only tracks state.
    """
    published_types: list[str] = []
    bus.subscribe(
        "audio_captured", lambda e: asyncio.sleep(0, result=None)  # noop
    )
    # Wrap publish to record everything the orchestrator emits.
    original_publish = bus.publish

    def spy_publish(event_type, payload=None, trace_id=None):
        published_types.append(event_type)
        return original_publish(event_type, payload, trace_id)

    bus.publish = spy_publish  # type: ignore[assignment]

    await orch.start()
    bus.publish("wake_word_detected", {}, trace_id="t1")
    # Let the dispatch loop process the event.
    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task

    assert "audio_captured" not in published_types, (
        "orchestrator must not fabricate audio_captured (fix #1)"
    )


# --------------------------------------------------------------------------- #
# Fix #2 — invalid transitions reject authoritatively + emit a diagnostic
# --------------------------------------------------------------------------- #
async def test_invalid_transition_publishes_diagnostic(bus: EventBus, orch: Orchestrator):
    """An invalid transition must publish invalid_transition and not move state."""
    diagnostics: list[dict] = []
    captured_events = []

    async def capture(event):
        captured_events.append(event)

    async def capture_diag(event):
        diagnostics.append(event.payload)

    await orch.start()
    bus.subscribe("invalid_transition", capture_diag)
    # Park the orchestrator in THINKING, then try to go LISTENING (not allowed
    # from THINKING — only SPEAKING -> LISTENING is the barge-in path).
    orch.state = State.THINKING
    # _transition returns False for invalid attempts.
    ok = orch._transition(State.LISTENING, "trace-x")
    assert ok is False
    assert orch.state == State.THINKING  # unchanged

    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task

    assert diagnostics, "expected an invalid_transition diagnostic event"
    assert diagnostics[0]["current_state"] == "THINKING"
    assert diagnostics[0]["attempted_target"] == "LISTENING"


async def test_duplicate_wake_while_listening_is_noop(bus: EventBus, orch: Orchestrator):
    """Fix #2 regression: a second wake word while LISTENING must not re-arm.

    The original code armed a fresh listening timeout AND published a fake
    audio_captured on duplicate wake. After the fix, the duplicate wake is a
    clean no-op: state stays LISTENING and no audio_captured is emitted.
    """
    audio_events: list[str] = []
    bus.subscribe(
        "audio_captured",
        lambda e: audio_events.append("audio_captured"),  # noqa: E731
    )
    await orch.start()
    bus.publish("wake_word_detected", {}, trace_id="dup1")
    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    assert orch.state == State.LISTENING
    assert orch._current_trace == "dup1"
    # Fire a duplicate wake.
    bus.publish("wake_word_detected", {}, trace_id="dup2")
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task

    assert orch.state == State.LISTENING  # unchanged, not reset/re-armed badly
    assert orch._current_trace == "dup1", (
        "ignored wake must not steal trace ownership"
    )
    assert audio_events == [], "no audio_captured should be published by orchestrator"


# --------------------------------------------------------------------------- #
# Fix #3 — barge-in is SPEAKING-only
# --------------------------------------------------------------------------- #
async def test_barge_in_from_speaking(bus: EventBus, orch: Orchestrator):
    """Wake while SPEAKING must transition to LISTENING (barge-in)."""
    await orch.start()
    orch.state = State.SPEAKING
    bus.publish("wake_word_detected", {}, trace_id="barge")
    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task
    assert orch.state == State.LISTENING


@pytest.mark.parametrize(
    "blocked_state",
    [State.TRANSCRIBING, State.THINKING, State.TOOL_CALL, State.LISTENING, State.WAKE_DETECTED],
)
async def test_wake_in_non_speaking_active_state_is_noop(
    bus: EventBus, orch: Orchestrator, blocked_state: State
):
    """Fix #3: wake in any active state other than SPEAKING must not move state."""
    await orch.start()
    orch.state = blocked_state
    orch._current_trace = "active"
    bus.publish("wake_word_detected", {}, trace_id="noop")
    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task
    assert orch.state == blocked_state, (
        f"state changed from {blocked_state.value} on wake — barge-in is "
        f"SPEAKING-only, this should have been a no-op"
    )
    assert orch._current_trace == "active"


async def test_interaction_completed_is_published_only_after_idle(
    bus: EventBus, orch: Orchestrator
):
    completed = []

    async def record(event):
        completed.append((event.trace_id, orch.state, event.payload))

    await orch.start()
    bus.subscribe("interaction_completed", record)
    orch.state = State.SPEAKING
    orch._current_trace = "complete-tr"
    run_task = asyncio.create_task(bus.run())
    bus.publish("speech_finished", {"text": "done"}, trace_id="complete-tr")
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task

    assert completed == [
        ("complete-tr", State.IDLE, {"state": "IDLE", "ok": True})
    ]


async def test_thinking_ready_is_published_after_authoritative_transition(
    bus: EventBus, orch: Orchestrator
):
    observed = []

    async def record(event):
        observed.append((event.trace_id, orch.state, event.payload))

    await orch.start()
    bus.subscribe("thinking_ready", record)
    orch.state = State.LISTENING
    orch._current_trace = "think-tr"
    run_task = asyncio.create_task(bus.run())
    bus.publish("transcription_ready", {"text": "привет"}, trace_id="think-tr")
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task

    assert observed == [("think-tr", State.THINKING, {})]


async def test_audio_captured_cancels_listening_timeout_during_slow_stt(bus: EventBus):
    orch = Orchestrator(bus, {"listening_timeout_seconds": 0.05})
    await orch.start()
    run_task = asyncio.create_task(bus.run())
    bus.publish("wake_word_detected", {}, trace_id="slow-stt")
    await asyncio.sleep(0.01)
    bus.publish("audio_captured", {"audio": b"pcm"}, trace_id="slow-stt")
    await asyncio.sleep(0.1)

    assert orch.state == State.TRANSCRIBING
    bus.publish(
        "transcription_ready", {"text": "который час"}, trace_id="slow-stt"
    )
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task
    await orch.stop()

    assert orch.state == State.THINKING


async def test_handler_failure_recovers_to_idle_and_allows_next_trace(bus: EventBus):
    orch = Orchestrator(
        bus,
        {"listening_timeout_seconds": 1, "interaction_timeout_seconds": 1},
    )
    completed: asyncio.Queue = asyncio.Queue()

    async def broken_handler(event):
        raise RuntimeError("tool exploded")

    await orch.start()
    bus.subscribe("explode", broken_handler)
    bus.subscribe("interaction_completed", lambda event: completed.put(event))
    run_task = asyncio.create_task(bus.run())

    bus.publish("wake_word_detected", {}, trace_id="broken-trace")
    await asyncio.sleep(0)
    bus.publish("audio_captured", {"audio": b"pcm"}, trace_id="broken-trace")
    bus.publish("transcription_ready", {"text": "test"}, trace_id="broken-trace")
    bus.publish("explode", {}, trace_id="broken-trace")

    failure = await asyncio.wait_for(completed.get(), timeout=1.0)
    assert failure.trace_id == "broken-trace"
    assert failure.payload == {
        "state": "IDLE",
        "ok": False,
        "reason": "handler_exception",
        "failed_state": "THINKING",
    }
    assert orch.state == State.IDLE
    assert orch._current_trace is None

    # Late output from the failed handler chain cannot resurrect the trace.
    bus.publish("response_ready", {"text": "stale"}, trace_id="broken-trace")
    bus.publish("speech_finished", {}, trace_id="broken-trace")
    await asyncio.sleep(0.05)
    assert orch.state == State.IDLE

    bus.publish("wake_word_detected", {}, trace_id="next-trace")
    await asyncio.sleep(0.05)
    assert orch.state == State.LISTENING
    assert orch._current_trace == "next-trace"

    await bus.stop()
    await run_task
    await orch.stop()


async def test_interaction_watchdog_recovers_thinking_trace(bus: EventBus):
    orch = Orchestrator(
        bus,
        {"listening_timeout_seconds": 1, "interaction_timeout_seconds": 0.05},
    )
    completed: asyncio.Queue = asyncio.Queue()
    await orch.start()
    bus.subscribe("interaction_completed", lambda event: completed.put(event))
    run_task = asyncio.create_task(bus.run())

    bus.publish("wake_word_detected", {}, trace_id="timeout-trace")
    await asyncio.sleep(0)
    bus.publish("audio_captured", {"audio": b"pcm"}, trace_id="timeout-trace")
    bus.publish("transcription_ready", {"text": "test"}, trace_id="timeout-trace")

    failure = await asyncio.wait_for(completed.get(), timeout=1.0)
    assert failure.payload["ok"] is False
    assert failure.payload["reason"] == "interaction_timeout"
    assert failure.payload["failed_state"] == "THINKING"
    assert orch.state == State.IDLE

    await bus.stop()
    await run_task
    await orch.stop()
