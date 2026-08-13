from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.event_bus import Event, EventBus
from core.event_payloads import NLUResultPayload
from ml.jsc.inference import StructuredPrediction
from modules.jsc_shadow import JSCShadowModule


def _config(tmp_path, checkpoint):
    return SimpleNamespace(
        model=str(checkpoint),
        device="cpu",
        params={"log_path": str(tmp_path / "jsc_shadow.jsonl")},
    )


@pytest.mark.asyncio
async def test_shadow_records_comparison_without_publishing_execution(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test checkpoint sentinel")
    calls: list[str] = []

    class FakePredictor:
        def predict(self, text):
            calls.append(text)
            return StructuredPrediction(
                '{"act":"execute","steps":[]}',
                {"accepted": 1},
                12.3456,
            )

    def factory(*args, **kwargs):
        return FakePredictor()

    bus = EventBus()
    module = JSCShadowModule(_config(tmp_path, checkpoint), predictor_factory=factory)
    await module.start(bus)
    event = Event(
        "nlu_result",
        NLUResultPayload(
            text="открой калькулятор",
            intent="open_application",
            raw_intent="open_application",
            intent_confidence=0.91,
            slots={"application": "calculator"},
            actions=[
                {
                    "intent": "open_application",
                    "slots": {"application": "calculator"},
                    "confidence": 0.99,
                }
            ],
        ),
        trace_id="voice001",
    )

    await module._on_nlu_result(event)

    record = json.loads((tmp_path / "jsc_shadow.jsonl").read_text("utf-8"))
    assert calls == ["открой калькулятор"]
    assert record["trace_id"] == "voice001"
    assert record["production_nlu"]["intent"] == "open_application"
    assert record["jsc"]["decisions"] == {"accepted": 1}
    assert record["jsc"]["latency_ms"] == 12.346
    assert record["executed_by_jsc"] is False
    assert bus.queue.empty()


@pytest.mark.asyncio
async def test_shadow_stays_disabled_when_checkpoint_is_missing(tmp_path):
    created = False

    def factory(*args, **kwargs):
        nonlocal created
        created = True
        raise AssertionError("predictor must not be created")

    bus = EventBus()
    module = JSCShadowModule(
        _config(tmp_path, tmp_path / "missing.pt"), predictor_factory=factory
    )

    await module.start(bus)

    assert created is False
    assert module._predictor is None
    assert not (tmp_path / "jsc_shadow.jsonl").exists()


@pytest.mark.asyncio
async def test_shadow_prediction_failure_never_escapes_to_runtime(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"test checkpoint sentinel")

    class BrokenPredictor:
        def predict(self, text):
            raise RuntimeError("broken experimental model")

    bus = EventBus()
    module = JSCShadowModule(
        _config(tmp_path, checkpoint),
        predictor_factory=lambda *args, **kwargs: BrokenPredictor(),
    )
    await module.start(bus)

    await module._on_nlu_result(
        Event(
            "nlu_result",
            NLUResultPayload(text="привет", intent="chat"),
            trace_id="voice002",
        )
    )

    assert not (tmp_path / "jsc_shadow.jsonl").exists()
