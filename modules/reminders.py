"""Persistent SQLite reminder scheduler and event-driven delivery service."""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.event_bus import Event, EventBus
from core.event_payloads import (
    InteractionFailedPayload,
    NotificationDeliverPayload,
    NotificationPayload,
    ReminderCancelledPayload,
)

logger = logging.getLogger("jarvis.module.reminders")


@dataclass(frozen=True)
class Reminder:
    id: int
    message: str
    due_at: str
    status: str
    created_at: str
    profile_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReminderScheduler:
    """Store reminders durably and coordinate one-shot local delivery."""

    def __init__(
        self,
        db_path: str | Path = "reminders.db",
        *,
        poll_interval_seconds: float = 0.5,
        profile_id: str = "default",
    ) -> None:
        self.db_path = str(Path(db_path))
        self.poll_interval = float(poll_interval_seconds)
        self.profile_id = str(profile_id).strip() or "default"
        if self.poll_interval <= 0:
            raise ValueError("reminder poll interval must be positive")
        self.bus: EventBus | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._queued_ids: set[int] = set()

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "ReminderScheduler":
        import os

        config = config or {}
        return cls(
            os.environ.get("JARVIS_REMINDERS_DB")
            or config.get("db_path", "reminders.db"),
            poll_interval_seconds=float(
                config.get("poll_interval_seconds", 0.5)
            ),
            profile_id=str(config.get("profile_id", "default")),
        )

    async def start(self, bus: EventBus, *, delivery_enabled: bool = True) -> None:
        self.bus = bus
        bus.subscribe("notification_authorized", self._on_notification_authorized)
        await asyncio.to_thread(self._initialize_schema)
        if delivery_enabled:
            self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "ReminderScheduler started db=%s profile=%s delivery=%s",
            Path(self.db_path).resolve(),
            self.profile_id,
            delivery_enabled,
        )

    async def stop(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._queued_ids.clear()
        logger.info("ReminderScheduler stopped")

    async def create_after(self, minutes: int, message: str) -> Reminder:
        minutes = int(minutes)
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        due_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        return await self.create_at(due_at, message)

    async def create_at(self, due_at: datetime | str, message: str) -> Reminder:
        text = self._validate_message(message)
        due = self._coerce_datetime(due_at)
        if due <= datetime.now(timezone.utc):
            raise ValueError("reminder time must be in the future")
        return await asyncio.to_thread(self._insert, due, text)

    async def create_clock_time(
        self, clock_time: str, message: str, *, day: str = ""
    ) -> Reminder:
        try:
            hour_text, minute_text = clock_time.replace(".", ":").split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("clock_time must have HH:MM format") from exc
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("clock_time is outside 00:00-23:59")

        now_local = datetime.now().astimezone()
        target = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        normalized_day = day.casefold().strip()
        if normalized_day == "завтра":
            target += timedelta(days=1)
        elif normalized_day == "сегодня":
            if target <= now_local:
                raise ValueError("указанное время сегодня уже прошло")
        elif target <= now_local:
            target += timedelta(days=1)
        elif normalized_day:
            raise ValueError(f"unsupported reminder day: {day}")
        return await self.create_at(target, message)

    async def list_pending(self) -> list[Reminder]:
        return await asyncio.to_thread(self._select_pending)

    async def cancel(self, reminder_id: int) -> Reminder | None:
        reminder_id = int(reminder_id)
        if reminder_id <= 0:
            raise ValueError("reminder id must be positive")
        reminder = await asyncio.to_thread(self._cancel_pending, reminder_id)
        if reminder is not None:
            self._queued_ids.discard(reminder_id)
            if self.bus is not None:
                self.bus.publish(
                    "reminder_cancelled",
                    ReminderCancelledPayload(reminder_id=reminder_id),
                )
        return reminder

    async def _poll_loop(self) -> None:
        try:
            while True:
                await self._publish_due()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            return

    async def _publish_due(self) -> None:
        if self.bus is None:
            return
        due = await asyncio.to_thread(self._select_due)
        for reminder in due:
            if reminder.id in self._queued_ids:
                continue
            self._queued_ids.add(reminder.id)
            self.bus.publish(
                "notification_ready",
                NotificationPayload(
                    reminder_id=reminder.id,
                    text=f"Напоминание: {reminder.message}",
                    message=reminder.message,
                    due_at=reminder.due_at,
                ),
            )
            logger.info(
                "REMINDER_DUE id=%d due_at=%s", reminder.id, reminder.due_at
            )

    async def _on_notification_authorized(self, event: Event) -> None:
        reminder_id = event.payload.get("reminder_id")
        if reminder_id is None or event.payload.get("source") != "reminder":
            return
        reminder_id = int(reminder_id)
        delivered = await asyncio.to_thread(self._mark_fired, reminder_id)
        self._queued_ids.discard(reminder_id)
        if not delivered:
            assert self.bus is not None
            self.bus.publish_event(
                event.child(
                    "interaction_failed",
                    InteractionFailedPayload(
                        reason="reminder_not_pending", reminder_id=reminder_id
                    ),
                )
            )
            return
        assert self.bus is not None
        self.bus.publish_event(
            event.child(
                "notification_deliver",
                NotificationDeliverPayload(**dict(event.payload)),
            )
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','fired','cancelled')),
                    created_at TEXT NOT NULL,
                    fired_at TEXT,
                    cancelled_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_due "
                "ON reminders(profile_id, status, due_at)"
            )

    @staticmethod
    def _validate_message(message: str) -> str:
        text = str(message).strip()
        if not text:
            raise ValueError("reminder message must not be empty")
        if len(text) > 500:
            raise ValueError("reminder message must be at most 500 characters")
        return text

    @staticmethod
    def _coerce_datetime(value: datetime | str) -> datetime:
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("due_at must be an ISO 8601 datetime") from exc
        if value.tzinfo is None:
            value = value.astimezone()
        return value.astimezone(timezone.utc)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=int(row["id"]),
            message=str(row["message"]),
            due_at=str(row["due_at"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            profile_id=str(row["profile_id"]),
        )

    def _insert(self, due_at: datetime, message: str) -> Reminder:
        due_text = due_at.astimezone(timezone.utc).isoformat()
        created_text = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reminders(profile_id,message,due_at,status,created_at) "
                "VALUES(?,?,?,'pending',?)",
                (self.profile_id, message, due_text, created_text),
            )
            row = connection.execute(
                "SELECT * FROM reminders WHERE id=?", (cursor.lastrowid,)
            ).fetchone()
        assert row is not None
        reminder = self._from_row(row)
        logger.info(
            "REMINDER_CREATED id=%d due_at=%s message=%r",
            reminder.id,
            reminder.due_at,
            reminder.message,
        )
        return reminder

    def _select_pending(self) -> list[Reminder]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE profile_id=? AND status='pending' "
                "ORDER BY due_at,id",
                (self.profile_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _select_due(self) -> list[Reminder]:
        now_text = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE profile_id=? AND status='pending' "
                "AND due_at<=? ORDER BY due_at,id",
                (self.profile_id, now_text),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _cancel_pending(self, reminder_id: int) -> Reminder | None:
        cancelled_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reminders WHERE id=? AND profile_id=? "
                "AND status='pending'",
                (reminder_id, self.profile_id),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE reminders SET status='cancelled',cancelled_at=? WHERE id=?",
                (cancelled_at, reminder_id),
            )
            updated = connection.execute(
                "SELECT * FROM reminders WHERE id=?", (reminder_id,)
            ).fetchone()
        assert updated is not None
        return self._from_row(updated)

    def _mark_fired(self, reminder_id: int) -> bool:
        fired_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE reminders SET status='fired',fired_at=? "
                "WHERE id=? AND profile_id=? AND status='pending'",
                (fired_at, reminder_id, self.profile_id),
            )
        return cursor.rowcount == 1
