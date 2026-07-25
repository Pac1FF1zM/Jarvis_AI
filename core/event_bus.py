"""Async pub/sub event bus.

The bus is the *only* communication channel between modules. No module ever
imports or calls another module directly — they publish events here and
subscribe to the event types they care about.

Events carry a ``trace_id`` so a single user interaction can be reconstructed
end-to-end across every module (used for latency benchmarking).
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger("jarvis.bus")

_TERMINAL_EVENTS = frozenset(
    {"interaction_cancelled", "interaction_failed", "interaction_completed"}
)
_MAX_CLOSED_TRACES = 1024

# A handler is any async callable that accepts an Event.
Handler = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    """A single event flowing through the bus.

    Attributes:
        event_type: logical name, e.g. ``"transcription_ready"``.
        payload: arbitrary dict of data for the event.
        timestamp: epoch seconds at publish time.
        trace_id: groups every event belonging to one user interaction.
    """

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def child(self, event_type: str, payload: dict[str, Any] | None = None) -> "Event":
        """Create a follow-up event that inherits this event's trace_id."""
        return Event(
            event_type=event_type,
            payload=payload or {},
            trace_id=self.trace_id,
        )


class EventBus:
    """In-process pub/sub bus built on :class:`asyncio.Queue`.

    No external broker (Redis/RabbitMQ) is used at this stage — everything runs
    inside one process. Multiple handlers may subscribe to the same event type;
    each is dispatched as its own task so handlers run concurrently.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        # Created lazily so the bus can be instantiated before the loop runs.
        self._queue: asyncio.Queue[Event] | None = None
        self._stop_event: asyncio.Event | None = None
        self._running: bool = False
        # In-flight dispatch tasks (fix #4): kept so stop() can drain them
        # instead of dropping/truncating in-flight handler work on shutdown.
        self._tasks: set[asyncio.Task] = set()
        # A closed trace is a tombstone, not a force-cancelled asyncio task.
        # Blocking STT/LLM workers cannot be pre-empted safely, so they drain
        # while every queued or late downstream event for that trace is
        # rejected here. Bounded retention prevents growth in persistent mode.
        self._closed_traces: OrderedDict[str, dict[str, Any]] = OrderedDict()

    # ------------------------------------------------------------------ #
    # Lazy properties (avoid binding asyncio primitives to the wrong loop)
    # ------------------------------------------------------------------ #
    @property
    def queue(self) -> asyncio.Queue[Event]:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    @property
    def stop_event(self) -> asyncio.Event:
        if self._stop_event is None:
            self._stop_event = asyncio.Event()
        return self._stop_event

    # ------------------------------------------------------------------ #
    # Pub/sub API
    # ------------------------------------------------------------------ #
    def subscribe(self, event_type: str, handler: Handler) -> None:
        """Register ``handler`` to be awaited for every ``event_type`` event."""
        self._subscribers[event_type].append(handler)
        logger.debug(
            "subscribed handler to '%s' (total=%d)",
            event_type,
            len(self._subscribers[event_type]),
        )

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> Event:
        """Publish a new event. ``trace_id`` is auto-generated if omitted.

        Carrying an explicit ``trace_id`` (e.g. from the triggering event) keeps
        a whole interaction grouped together for latency analysis.
        """
        event = Event(
            event_type=event_type,
            payload=payload or {},
            trace_id=trace_id or uuid.uuid4().hex[:12],
        )
        self._publish_or_drop(event)
        return event

    def publish_event(self, event: Event) -> bool:
        """Publish an already-constructed :class:`Event` (keeps its trace_id)."""
        return self._publish_or_drop(event)

    def _publish_or_drop(self, event: Event) -> bool:
        if event.event_type == "interaction_cancelled":
            details = dict(event.payload)
            reason = str(details.pop("reason", "cancelled"))
            return self.cancel_trace(
                event.trace_id,
                reason=reason,
                details=details,
            )
        if event.event_type == "interaction_failed":
            return self.fail_trace(event.trace_id, event.payload)
        if (
            self.is_trace_closed(event.trace_id)
            and event.event_type not in _TERMINAL_EVENTS
        ):
            logger.info(
                "CLOSED_TRACE_EVENT_DROPPED type=%s trace=%s outcome=%s",
                event.event_type,
                event.trace_id,
                self._closed_traces[event.trace_id]["outcome"],
            )
            return False
        self.queue.put_nowait(event)
        logger.info("PUBLISH %s trace=%s", event.event_type, event.trace_id)
        return True

    # ------------------------------------------------------------------ #
    # Trace lifecycle / cooperative cancellation
    # ------------------------------------------------------------------ #
    def is_trace_closed(self, trace_id: str) -> bool:
        return trace_id in self._closed_traces

    def is_trace_cancelled(self, trace_id: str) -> bool:
        record = self._closed_traces.get(trace_id)
        return bool(record and record["outcome"] in {"cancelled", "failed"})

    def trace_outcome(self, trace_id: str) -> dict[str, Any] | None:
        record = self._closed_traces.get(trace_id)
        return dict(record) if record is not None else None

    def cancel_trace(
        self,
        trace_id: str,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically tombstone ``trace_id`` and announce its cancellation.

        Already-running handlers are deliberately not force-cancelled: a
        ``to_thread`` worker would continue outside asyncio and could release
        a shared GPU/device lock too early. Their results are discarded by the
        tombstone instead. Async tools receive the cancellation event and may
        stop cooperatively.
        """
        payload = {"reason": reason, **(details or {})}
        if not self._close_trace(trace_id, "cancelled", payload):
            return False
        self._enqueue_terminal(
            Event("interaction_cancelled", payload, trace_id=trace_id)
        )
        return True

    def fail_trace(self, trace_id: str, payload: dict[str, Any]) -> bool:
        """Atomically tombstone a failed trace before recovery is dispatched."""
        failure = dict(payload)
        failure.setdefault("reason", "unknown")
        if not self._close_trace(trace_id, "failed", failure):
            logger.info("DUPLICATE_TRACE_FAILURE_IGNORED trace=%s", trace_id)
            return False
        self._enqueue_terminal(
            Event("interaction_failed", failure, trace_id=trace_id)
        )
        return True

    def complete_trace(self, trace_id: str) -> bool:
        """Close a successful trace so delayed duplicate output is rejected."""
        return self._close_trace(trace_id, "completed", {})

    def _close_trace(
        self, trace_id: str, outcome: str, details: dict[str, Any]
    ) -> bool:
        if trace_id in self._closed_traces:
            return False
        self._closed_traces[trace_id] = {
            "outcome": outcome,
            "closed_at": time.time(),
            **details,
        }
        while len(self._closed_traces) > _MAX_CLOSED_TRACES:
            self._closed_traces.popitem(last=False)
        logger.info("TRACE_CLOSED trace=%s outcome=%s", trace_id, outcome)
        return True

    def _enqueue_terminal(self, event: Event) -> None:
        self.queue.put_nowait(event)
        logger.info("PUBLISH %s trace=%s", event.event_type, event.trace_id)

    async def run(self) -> None:
        """Main dispatch loop. Run as a task; stop via :meth:`stop`."""
        self._running = True
        self.stop_event.clear()
        logger.info("EventBus running")
        while self._running:
            try:
                # Short wait_for so we can notice stop_event between events.
                event = await asyncio.wait_for(self.queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                if self.stop_event.is_set():
                    break
                continue
            handlers = list(self._subscribers.get(event.event_type, []))
            for handler in handlers:
                task = asyncio.create_task(self._safe_dispatch(handler, event))
                # Fix #4: hold a strong reference + discard on done so the set
                # doesn't grow unboundedly and tasks aren't GC'd mid-flight.
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        logger.info("EventBus run loop exited")

    async def _safe_dispatch(self, handler: Handler, event: Event) -> None:
        if (
            self.is_trace_closed(event.trace_id)
            and event.event_type not in _TERMINAL_EVENTS
        ):
            logger.info(
                "CLOSED_TRACE_DISPATCH_SKIPPED type=%s trace=%s",
                event.event_type,
                event.trace_id,
            )
            return
        try:
            await handler(event)
        except Exception as exc:  # noqa: BLE001 — report and keep the bus alive
            logger.exception(
                "handler error on '%s' trace=%s", event.event_type, event.trace_id
            )
            # Turn an otherwise terminal handler exception into an explicit
            # lifecycle signal.  The orchestrator can then reset the current
            # trace instead of remaining stuck in TRANSCRIBING/THINKING/etc.
            # Never recurse if the recovery handler itself is defective.
            if event.event_type not in _TERMINAL_EVENTS and not self.is_trace_closed(
                event.trace_id
            ):
                handler_name = getattr(
                    handler,
                    "__qualname__",
                    getattr(handler, "__name__", repr(handler)),
                )
                self.fail_trace(
                    event.trace_id,
                    {
                        "reason": "handler_exception",
                        "source_event": event.event_type,
                        "handler": handler_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )

    async def stop(self) -> None:
        """Signal the run loop to exit, then drain in-flight handler tasks.

        Fix #4: tasks that are already dispatched when shutdown is requested
        represent real in-flight handler work — we wait for them (with a bounded
        timeout) rather than dropping or truncating them. Anything still
        unfinished when the timeout lapses is cancelled and logged.
        """
        self._running = False
        self.stop_event.set()
        # Give the run loop a chance to flush its current dispatch batch.
        if self._tasks:
            drain_timeout = 5.0
            done, pending = await asyncio.wait(
                list(self._tasks), timeout=drain_timeout
            )
            for task in pending:
                logger.warning(
                    "EventBus.stop: cancelling unfinished handler task %r "
                    "(exceeded drain timeout %.1fs)",
                    task,
                    drain_timeout,
                )
                task.cancel()
            # Await cancellation completion so callers see clean teardown.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._tasks.clear()
