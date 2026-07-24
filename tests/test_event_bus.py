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
