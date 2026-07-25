"""Persistent reminder storage, cancellation and delivery integration."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone

from core.event_bus import EventBus
from core.orchestrator import Orchestrator, State
from modules.reminders import ReminderScheduler
from modules.text_output import TextOutputModule


async def test_reminder_persists_across_scheduler_restart_and_can_be_cancelled(
    tmp_path,
):
    db_path = tmp_path / "persistent-reminders.db"
    first = ReminderScheduler(db_path, poll_interval_seconds=0.01)
    bus = EventBus()
    await first.start(bus, delivery_enabled=False)
    created = await first.create_at(
        datetime.now(timezone.utc) + timedelta(hours=1), "проверить духовку"
    )
    await first.stop()

    second = ReminderScheduler(db_path, poll_interval_seconds=0.01)
    await second.start(bus, delivery_enabled=False)
    pending = await second.list_pending()
    assert pending == [created]

    cancelled = await second.cancel(created.id)
    assert cancelled is not None
    assert cancelled.id == created.id
    assert cancelled.status == "cancelled"
    assert await second.list_pending() == []
    await second.stop()

    with sqlite3.connect(db_path) as connection:
        status = connection.execute(
            "SELECT status FROM reminders WHERE id=?", (created.id,)
        ).fetchone()[0]
    assert status == "cancelled"


async def test_overdue_reminder_is_delivered_after_restart(tmp_path, capsys):
    db_path = tmp_path / "overdue-reminders.db"
    setup_bus = EventBus()
    setup = ReminderScheduler(db_path, poll_interval_seconds=0.01)
    await setup.start(setup_bus, delivery_enabled=False)
    reminder = await setup.create_at(
        datetime.now(timezone.utc) + timedelta(milliseconds=50),
        "выпить воды",
    )
    await setup.stop()
    await asyncio.sleep(0.06)

    bus = EventBus()
    scheduler = ReminderScheduler(db_path, poll_interval_seconds=0.01)
    orchestrator = Orchestrator(
        bus,
        {"listening_timeout_seconds": 1, "interaction_timeout_seconds": 1},
    )
    output = TextOutputModule()
    completed: asyncio.Queue = asyncio.Queue()
    await scheduler.start(bus, delivery_enabled=True)
    await output.start(bus)
    await orchestrator.start()
    bus.subscribe("interaction_completed", lambda event: completed.put(event))
    run_task = asyncio.create_task(bus.run())

    delivered = await asyncio.wait_for(completed.get(), timeout=1.0)
    assert delivered.payload["ok"] is True
    assert delivered.trace_id
    assert orchestrator.state == State.IDLE
    assert await scheduler.list_pending() == []
    assert f"Напоминание: {reminder.message}" in capsys.readouterr().out

    await scheduler.stop()
    await bus.stop()
    await run_task
    await output.stop()
    await orchestrator.stop()


async def test_due_reminder_waits_for_active_interaction(tmp_path):
    db_path = tmp_path / "queued-reminders.db"
    bus = EventBus()
    scheduler = ReminderScheduler(db_path, poll_interval_seconds=0.01)
    orchestrator = Orchestrator(
        bus,
        {"listening_timeout_seconds": 1, "interaction_timeout_seconds": 1},
    )
    output = TextOutputModule()
    completed: asyncio.Queue = asyncio.Queue()
    await scheduler.start(bus, delivery_enabled=True)
    await output.start(bus)
    await orchestrator.start()
    bus.subscribe("interaction_completed", lambda event: completed.put(event))
    run_task = asyncio.create_task(bus.run())

    bus.publish("wake_word_detected", {}, trace_id="user-trace")
    deadline = asyncio.get_running_loop().time() + 1.0
    while orchestrator.state != State.LISTENING:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("user interaction did not reach LISTENING")
        await asyncio.sleep(0)
    assert orchestrator.state == State.LISTENING
    reminder = await scheduler.create_at(
        datetime.now(timezone.utc) + timedelta(milliseconds=30), "позвонить маме"
    )

    deadline = asyncio.get_running_loop().time() + 1.0
    while not orchestrator._pending_notifications:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("due reminder was not queued")
        await asyncio.sleep(0)
    assert orchestrator.state == State.LISTENING

    bus.publish(
        "interaction_failed",
        {"reason": "test_release"},
        trace_id="user-trace",
    )
    user_failure = await asyncio.wait_for(completed.get(), timeout=1.0)
    notification_completion = await asyncio.wait_for(completed.get(), timeout=1.0)

    assert user_failure.trace_id == "user-trace"
    assert user_failure.payload["ok"] is False
    assert notification_completion.trace_id != "user-trace"
    assert notification_completion.payload["ok"] is True
    assert await scheduler.list_pending() == []
    assert reminder.id not in scheduler._queued_ids

    await scheduler.stop()
    await bus.stop()
    await run_task
    await output.stop()
    await orchestrator.stop()


async def test_cancelled_reminder_is_never_delivered(tmp_path):
    db_path = tmp_path / "cancelled-reminders.db"
    bus = EventBus()
    scheduler = ReminderScheduler(db_path, poll_interval_seconds=0.01)
    delivered = asyncio.Event()
    bus.subscribe("notification_ready", lambda _event: _set(delivered))
    await scheduler.start(bus, delivery_enabled=True)
    run_task = asyncio.create_task(bus.run())
    reminder = await scheduler.create_at(
        datetime.now(timezone.utc) + timedelta(milliseconds=80), "не доставлять"
    )
    assert await scheduler.cancel(reminder.id) is not None
    await asyncio.sleep(0.12)

    assert not delivered.is_set()
    await scheduler.stop()
    await bus.stop()
    await run_task


async def _set(event: asyncio.Event) -> None:
    event.set()
