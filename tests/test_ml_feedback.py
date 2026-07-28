"""Regression coverage for the local, human-reviewed NLU feedback queue."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.event_bus import EventBus
from core.ml_feedback import MLFeedbackCollector


async def test_low_confidence_turn_is_queued_only_after_completion(tmp_path: Path):
    queue_path = tmp_path / "feedback" / "pending.jsonl"
    bus = EventBus()
    collector = MLFeedbackCollector(queue_path, min_intent_confidence=0.72)
    await collector.start(bus)
    runner = asyncio.create_task(bus.run())

    bus.publish(
        "nlu_result",
        {
            "text": "открой дискор",
            "intent": "unknown",
            "raw_intent": "open_application",
            "intent_confidence": 0.41,
            "slots": {},
        },
        trace_id="feedback-trace",
    )
    await asyncio.sleep(0.03)
    assert not queue_path.exists(), "unfinished interactions must not be used as feedback"
    bus.publish(
        "interaction_completed",
        {"state": "IDLE", "ok": True},
        trace_id="feedback-trace",
    )
    await asyncio.sleep(0.05)

    await bus.stop()
    await runner
    await collector.stop()

    records = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["text"] == "открой дискор"
    assert record["status"] == "pending_review"
    assert set(record["reasons"]) == {"low_intent_confidence", "unknown_or_rejected_intent"}
    assert "audio" not in record
    assert "reviewed_intent" not in record


async def test_confident_successful_turn_is_not_collected(tmp_path: Path):
    queue_path = tmp_path / "pending.jsonl"
    bus = EventBus()
    collector = MLFeedbackCollector(queue_path)
    await collector.start(bus)
    runner = asyncio.create_task(bus.run())
    bus.publish(
        "nlu_result",
        {
            "text": "который час",
            "intent": "get_current_time",
            "raw_intent": "get_current_time",
            "intent_confidence": 0.99,
            "slots": {},
        },
        trace_id="confident-trace",
    )
    bus.publish(
        "interaction_completed",
        {"state": "IDLE", "ok": True},
        trace_id="confident-trace",
    )
    await asyncio.sleep(0.05)
    await bus.stop()
    await runner
    await collector.stop()
    assert not queue_path.exists()


async def test_tool_failure_promotes_a_confident_turn_to_review(tmp_path: Path):
    queue_path = tmp_path / "pending.jsonl"
    bus = EventBus()
    collector = MLFeedbackCollector(queue_path)
    await collector.start(bus)
    runner = asyncio.create_task(bus.run())
    bus.publish(
        "nlu_result",
        {
            "text": "открой браузер",
            "intent": "open_application",
            "raw_intent": "open_application",
            "intent_confidence": 0.99,
            "slots": {"application": "browser"},
        },
        trace_id="tool-failure",
    )
    bus.publish(
        "tool_result",
        {
            "tool": "open_application",
            "result": {"ok": False, "error": "not_found"},
        },
        trace_id="tool-failure",
    )
    bus.publish("interaction_completed", {"state": "IDLE", "ok": True}, trace_id="tool-failure")
    await asyncio.sleep(0.08)
    await bus.stop()
    await runner
    await collector.stop()

    record = json.loads(queue_path.read_text(encoding="utf-8").strip())
    assert record["reasons"] == ["tool_execution_failed"]
