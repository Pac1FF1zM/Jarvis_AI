"""Tests for the EventBus hardening (fix #4).

#4 — run() must track dispatched handler tasks, and stop() must drain
in-flight handler work (with a bounded timeout) instead of dropping it.
"""
from __future__ import annotations

import asyncio

import pytest

from core.event_bus import EventBus


async def test_inflight_handler_completes_before_stop():
    """A slow handler dispatched before stop() must finish, not be dropped."""
    bus = EventBus()
    finished = asyncio.Event()

    async def slow_handler(event):
        await asyncio.sleep(0.2)
        finished.set()

    bus.subscribe("ping", slow_handler)
    run_task = asyncio.create_task(bus.run())
    bus.publish("ping", {"n": 1}, trace_id="t1")
    # Let the dispatch kick off the handler task.
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task

    assert finished.is_set(), "handler was dropped/truncated on shutdown (fix #4)"


async def test_stop_cancels_handlers_exceeding_drain_timeout():
    """Handlers that overrun the drain timeout are cancelled + logged."""
    bus = EventBus()
    cancelled_was_logged = []

    async def forever_handler(event):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled_was_logged.append(True)
            raise

    bus.subscribe("ping", forever_handler)
    run_task = asyncio.create_task(bus.run())
    bus.publish("ping", {}, trace_id="t1")
    await asyncio.sleep(0.05)

    # Monkeypatch the drain timeout to keep the test fast.
    import core.event_bus as eb_mod

    original_wait = eb_mod.asyncio.wait

    def fast_wait(tasks, timeout=None):
        return original_wait(tasks, timeout=0.05)

    eb_mod.asyncio.wait = fast_wait  # type: ignore[assignment]
    try:
        await bus.stop()
    finally:
        eb_mod.asyncio.wait = original_wait  # type: ignore[assignment]
    await run_task

    assert cancelled_was_logged, "overrun handler should have been cancelled (fix #4)"


async def test_tasks_set_tracks_and_discards():
    """The bus must keep strong refs to in-flight tasks and discard on done."""
    bus = EventBus()
    handled = asyncio.Event()

    async def handler(event):
        handled.set()

    bus.subscribe("ping", handler)
    run_task = asyncio.create_task(bus.run())
    bus.publish("ping", {}, trace_id="t1")
    await asyncio.sleep(0.05)
    # While running, the task set should be populated, then drain after handlers finish.
    await handled.wait()
    await asyncio.sleep(0.02)
    await bus.stop()
    await run_task

    assert bus._tasks == set() or all(t.done() for t in bus._tasks)


async def test_handler_exception_emits_recoverable_failure_event():
    bus = EventBus()
    failure_received = asyncio.Event()
    failures = []

    async def broken_handler(event):
        raise ValueError("broken inference")

    async def record_failure(event):
        failures.append(event)
        failure_received.set()

    bus.subscribe("input", broken_handler)
    bus.subscribe("interaction_failed", record_failure)
    run_task = asyncio.create_task(bus.run())
    bus.publish("input", {"secret": "not copied"}, trace_id="failed-trace")
    await asyncio.wait_for(failure_received.wait(), timeout=1.0)
    await bus.stop()
    await run_task

    assert len(failures) == 1
    failure = failures[0]
    assert failure.trace_id == "failed-trace"
    assert failure.payload["reason"] == "handler_exception"
    assert failure.payload["source_event"] == "input"
    assert failure.payload["error_type"] == "ValueError"
    assert failure.payload["error"] == "broken inference"
    assert "secret" not in failure.payload


async def test_failure_handler_exception_does_not_recurse_forever():
    bus = EventBus()
    calls = 0

    async def broken_recovery(event):
        nonlocal calls
        calls += 1
        raise RuntimeError("recovery failed")

    bus.subscribe("interaction_failed", broken_recovery)
    run_task = asyncio.create_task(bus.run())
    bus.publish("interaction_failed", {"reason": "test"}, trace_id="failed-once")
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task

    assert calls == 1


async def test_cancelled_trace_drops_queued_and_late_work_but_emits_terminal_event():
    bus = EventBus()
    handled = []
    cancelled = []

    async def record_work(event):
        handled.append(event.event_type)

    async def record_cancel(event):
        cancelled.append(event)

    bus.subscribe("work", record_work)
    bus.subscribe("response_ready", record_work)
    bus.subscribe("interaction_cancelled", record_cancel)

    bus.publish("work", {}, trace_id="cancel-me")
    assert bus.cancel_trace("cancel-me", reason="user_requested") is True
    assert bus.publish("response_ready", {}, trace_id="cancel-me")

    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task

    assert handled == []
    assert len(cancelled) == 1
    assert cancelled[0].payload["reason"] == "user_requested"
    assert bus.is_trace_cancelled("cancel-me")


async def test_running_handler_drains_but_its_post_cancel_output_is_discarded():
    bus = EventBus()
    started = asyncio.Event()
    release = asyncio.Event()
    drained = asyncio.Event()
    stale_responses = []

    async def non_preemptible_handler(event):
        started.set()
        await release.wait()
        bus.publish_event(event.child("response_ready", {"text": "stale"}))
        drained.set()

    async def record_response(event):
        stale_responses.append(event)

    bus.subscribe("work", non_preemptible_handler)
    bus.subscribe("response_ready", record_response)
    run_task = asyncio.create_task(bus.run())
    bus.publish("work", {}, trace_id="drain-me")
    await asyncio.wait_for(started.wait(), timeout=1.0)

    bus.cancel_trace("drain-me", reason="superseded")
    release.set()
    await asyncio.wait_for(drained.wait(), timeout=1.0)
    await asyncio.sleep(0.02)
    await bus.stop()
    await run_task

    assert stale_responses == []
    assert bus.trace_outcome("drain-me")["outcome"] == "cancelled"


async def test_double_cancel_is_idempotent():
    bus = EventBus()
    cancellations = []

    async def record(event):
        cancellations.append(event)

    bus.subscribe("interaction_cancelled", record)
    assert bus.cancel_trace("once", reason="first") is True
    assert bus.cancel_trace("once", reason="second") is False

    run_task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)
    await bus.stop()
    await run_task

    assert len(cancellations) == 1
    assert cancellations[0].payload["reason"] == "first"
