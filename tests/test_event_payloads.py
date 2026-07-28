"""Runtime guarantees for typed EventBus payload contracts."""
from __future__ import annotations

import asyncio
from types import MappingProxyType

import pytest

from core.event_bus import Event, EventBus
from core.event_payloads import (
    ResponseReadyPayload,
    TranscriptionReadyPayload,
)


def test_typed_payload_is_bound_to_exact_event_name():
    with pytest.raises(ValueError, match="belongs to 'response_ready'"):
        Event(
            "transcription_ready",
            ResponseReadyPayload(text="не тот контракт"),
        )


def test_critical_values_are_validated_at_producer_boundary():
    with pytest.raises(ValueError, match="between 0 and 1"):
        TranscriptionReadyPayload(text="тест", confidence=1.5)
    with pytest.raises(TypeError, match="text must be str"):
        TranscriptionReadyPayload(text=None, confidence=0.5)  # type: ignore[arg-type]


def test_published_payload_is_detached_and_recursively_immutable():
    producer_owned = {"params": {"steps": 2}, "plan": [{"tool": "clock"}]}
    event = Event("legacy_extension_event", producer_owned)
    producer_owned["params"]["steps"] = 9

    assert isinstance(event.payload, MappingProxyType)
    assert event.payload["params"]["steps"] == 2
    with pytest.raises(TypeError):
        event.payload["params"]["steps"] = 3
    assert event.payload["plan"] == (MappingProxyType({"tool": "clock"}),)


def test_typed_payload_keeps_mapping_compatibility_for_incremental_migration():
    event = Event(
        "transcription_ready",
        TranscriptionReadyPayload(text="привет", confidence=0.75),
    )

    assert event.payload["text"] == "привет"
    assert event.payload.get("confidence") == pytest.approx(0.75)
    assert dict(event.payload) == {"text": "привет", "confidence": 0.75}


async def test_wrong_contract_inside_handler_becomes_recoverable_trace_failure():
    bus = EventBus()
    failures: asyncio.Queue[Event] = asyncio.Queue()

    async def broken_producer(event: Event) -> None:
        bus.publish_event(
            event.child(
                "response_ready",
                TranscriptionReadyPayload(text="wrong contract", confidence=0.8),
            )
        )

    bus.subscribe("input", broken_producer)
    bus.subscribe("interaction_failed", failures.put)
    run_task = asyncio.create_task(bus.run())
    bus.publish("input", {}, trace_id="typed-contract-failure")

    failure = await asyncio.wait_for(failures.get(), timeout=1.0)
    await bus.stop()
    await run_task

    assert failure.trace_id == "typed-contract-failure"
    assert failure.payload["reason"] == "handler_exception"
    assert failure.payload["error_type"] == "ValueError"
