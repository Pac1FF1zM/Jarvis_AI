"""Orchestrator — the core finite state machine.

The orchestrator tracks the assistant's lifecycle state and enforces valid
transitions. It contains **no** business logic for any module and originates
**no** payload data for other modules (fix #1): it only reacts to lifecycle
events by transitioning state, logging, and handling the two cross-cutting
concerns:

  * **Listening timeout** — revert to IDLE if nothing happens for N seconds.
  * **Barge-in** — a wake word while SPEAKING interrupts and returns to
    LISTENING. Barge-in is SPEAKING-only by design (fix #3, decision below):
    interrupting during THINKING/TOOL_CALL would require cancelling in-flight
    LLM inference, which is a separate concern. Wake-during-any-other-state is
    an explicit, logged no-op rather than a silent partial state change.

Every state transition is **authoritative** (fix #2): ``_transition()`` returns
``False`` for an invalid attempt, publishes an ``invalid_transition`` diagnostic
event (carrying trace_id, current state, attempted target), and every handler
stops immediately on ``False`` — no timers armed, no follow-up events published.

State graph::

    IDLE -> WAKE_DETECTED -> LISTENING -> TRANSCRIBING -> THINKING
                                      audio_captured     transcription_ready
          ^                                                |
          |                                                v
          +-- SPEAKING <---- response_ready ---- (response / tool_result)
                   ^
                   |
              (barge-in: SPEAKING -> LISTENING on wake_word_detected)
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any

from core.event_bus import EventBus, Event

logger = logging.getLogger("jarvis.orch")


class State(str, enum.Enum):
    IDLE = "IDLE"
    WAKE_DETECTED = "WAKE_DETECTED"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    TOOL_CALL = "TOOL_CALL"
    SPEAKING = "SPEAKING"


# Allowed source -> {target states} transitions.
#
# Barge-in (fix #3) is SPEAKING-only: SPEAKING -> LISTENING is the sole
# interrupt path. A wake_word_detected arriving in any other active state
# (LISTENING / TRANSCRIBING / THINKING / TOOL_CALL / WAKE_DETECTED) is an
# explicit no-op handled in _on_wake, so those states deliberately do NOT
# list a wake-driven target here.
VALID_TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.WAKE_DETECTED, State.LISTENING},
    State.WAKE_DETECTED: {State.LISTENING, State.IDLE},
    State.LISTENING: {State.TRANSCRIBING, State.IDLE},
    State.TRANSCRIBING: {State.THINKING, State.IDLE},
    State.THINKING: {State.TOOL_CALL, State.SPEAKING, State.IDLE},
    State.TOOL_CALL: {State.THINKING, State.IDLE},
    State.SPEAKING: {State.IDLE, State.LISTENING},  # LISTENING = barge-in
}


class Orchestrator:
    """Subscribes to lifecycle events and drives the state machine."""

    def __init__(self, bus: EventBus, config: dict[str, Any] | None = None) -> None:
        self.bus = bus
        self.config = config or {}
        self.state: State = State.IDLE
        self._current_trace: str | None = None
        self._listening_timeout: float = float(
            self.config.get("listening_timeout_seconds", 8)
        )
        self._timeout_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Subscribe to every lifecycle event the state machine cares about."""
        self.bus.subscribe("wake_word_detected", self._on_wake)
        self.bus.subscribe("audio_captured", self._on_audio_captured)
        self.bus.subscribe("transcription_ready", self._on_transcription)
        self.bus.subscribe("response_ready", self._on_response)
        self.bus.subscribe("tool_call_requested", self._on_tool_call)
        self.bus.subscribe("tool_result", self._on_tool_result)
        self.bus.subscribe("speech_started", self._on_speech_started)
        self.bus.subscribe("speech_finished", self._on_speech_finished)
        logger.info("Orchestrator started (initial=%s)", self.state.value)

    async def stop(self) -> None:
        self._cancel_timeout()

    # ------------------------------------------------------------------ #
    # Authoritative transition helper (fix #2)
    # ------------------------------------------------------------------ #
    def _transition(self, target: State, trace_id: str | None) -> bool:
        """Move to ``target`` if allowed; otherwise publish a diagnostic.

        Returns ``True`` on a successful transition. On failure it logs a
        warning and publishes an ``invalid_transition`` event carrying the
        trace_id, current state, and attempted target, so callers can simply
        ``if not self._transition(...): return`` and be sure no side effects
        leak past an invalid attempt.
        """
        if target not in VALID_TRANSITIONS.get(self.state, set()):
            logger.warning(
                "INVALID_TRANSITION %s -> %s (trace=%s)",
                self.state.value,
                target.value,
                trace_id,
            )
            self.bus.publish(
                "invalid_transition",
                {
                    "current_state": self.state.value,
                    "attempted_target": target.value,
                },
                trace_id=trace_id,
            )
            return False
        old = self.state
        self.state = target
        logger.info(
            "TRANSITION %s -> %s trace=%s ts=%.3f",
            old.value,
            target.value,
            trace_id,
            time.time(),
        )
        return True

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    async def _on_wake(self, event: Event) -> None:
        """Handle a wake word.

        - IDLE      -> WAKE_DETECTED -> LISTENING  (normal wake)
        - SPEAKING  -> LISTENING                   (barge-in, SPEAKING-only)
        - any other active state                   (explicit no-op)

        Per fix #1 this handler does NOT publish ``audio_captured``. Triggering
        audio capture is the job of a dedicated audio-capture module that
        subscribes to ``wake_word_detected`` like any other module — the
        orchestrator never originates payload data for other modules.
        """
        self._current_trace = event.trace_id
        current = self.state

        if current == State.IDLE:
            # Normal wake: IDLE -> WAKE_DETECTED -> LISTENING.
            if not self._transition(State.WAKE_DETECTED, event.trace_id):
                return
            if not self._transition(State.LISTENING, event.trace_id):
                return
            self._arm_listening_timeout(event.trace_id)
        elif current == State.SPEAKING:
            # Barge-in (SPEAKING-only, per fix #3).
            logger.info("BARGE-IN during SPEAKING trace=%s", event.trace_id)
            if not self._transition(State.LISTENING, event.trace_id):
                return
            self._arm_listening_timeout(event.trace_id)
        else:
            # Wake during LISTENING / TRANSCRIBING / THINKING / TOOL_CALL /
            # WAKE_DETECTED: clean, explicit no-op. Barge-in is SPEAKING-only,
            # so we log + ignore rather than attempting a partial state change.
            logger.info(
                "WAKE_IGNORED state=%s trace=%s (barge-in is SPEAKING-only)",
                current.value,
                event.trace_id,
            )

    async def _on_audio_captured(self, event: Event) -> None:
        """End LISTENING as soon as bounded audio exists, before slow STT."""
        if self._current_trace is not None and event.trace_id != self._current_trace:
            logger.info(
                "STALE_AUDIO_IGNORED trace=%s current_trace=%s",
                event.trace_id,
                self._current_trace,
            )
            return
        if self.state != State.LISTENING:
            logger.info(
                "AUDIO_LIFECYCLE_IGNORED state=%s trace=%s",
                self.state.value,
                event.trace_id,
            )
            return
        self._cancel_timeout()
        self._transition(State.TRANSCRIBING, event.trace_id)

    async def _on_transcription(self, event: Event) -> None:
        if self._current_trace is not None and event.trace_id != self._current_trace:
            logger.info(
                "STALE_TRANSCRIPTION_IGNORED trace=%s current_trace=%s",
                event.trace_id,
                self._current_trace,
            )
            return
        self._cancel_timeout()
        # Text mode intentionally has no audio_captured event, so retain its
        # direct LISTENING -> TRANSCRIBING path. Real audio already moved the
        # state in _on_audio_captured.
        if self.state == State.LISTENING:
            if not self._transition(State.TRANSCRIBING, event.trace_id):
                return
        elif self.state != State.TRANSCRIBING:
            self._transition(State.TRANSCRIBING, event.trace_id)
            return
        # TRANSCRIBING -> THINKING is always valid from TRANSCRIBING, but we
        # still honour the authoritative contract.
        if not self._transition(State.THINKING, event.trace_id):
            return
        # Lifecycle barrier for NLU. Both handlers receive transcription_ready
        # concurrently, so NLU must not publish a result until this transition
        # has authoritatively completed.
        self.bus.publish_event(event.child("thinking_ready"))

    async def _on_tool_call(self, event: Event) -> None:
        if not self._transition(State.TOOL_CALL, event.trace_id):
            return

    async def _on_tool_result(self, event: Event) -> None:
        # LLM re-invokes with the tool output appended — back to THINKING.
        self._transition(State.THINKING, event.trace_id)

    async def _on_response(self, event: Event) -> None:
        if not self._transition(State.SPEAKING, event.trace_id):
            return

    async def _on_speech_started(self, event: Event) -> None:
        # response_ready already moved us into SPEAKING; keep it idempotent but
        # still authoritative — a stray speech_started from the wrong state is
        # rejected and diagnosed rather than silently applied.
        if self.state != State.SPEAKING:
            self._transition(State.SPEAKING, event.trace_id)

    async def _on_speech_finished(self, event: Event) -> None:
        if not self._transition(State.IDLE, event.trace_id):
            return
        self._current_trace = None
        # Authoritative end-of-interaction signal: published only after the
        # state machine has actually reached IDLE for this trace.
        self.bus.publish_event(
            event.child("interaction_completed", {"state": State.IDLE.value})
        )

    # ------------------------------------------------------------------ #
    # Listening timeout
    # ------------------------------------------------------------------ #
    def _arm_listening_timeout(self, trace_id: str) -> None:
        self._cancel_timeout()
        self._timeout_task = asyncio.create_task(self._listening_watchdog(trace_id))

    async def _listening_watchdog(self, trace_id: str) -> None:
        try:
            await asyncio.sleep(self._listening_timeout)
        except asyncio.CancelledError:
            return
        if self.state == State.LISTENING:
            logger.info("LISTENING_TIMEOUT trace=%s -> IDLE", trace_id)
            if self._transition(State.IDLE, trace_id):
                self._current_trace = None

    def _cancel_timeout(self) -> None:
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
        self._timeout_task = None
