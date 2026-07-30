"""Safety tests for the webcam Gesture Core runtime shell."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.event_bus import EventBus
from core.orchestrator import Orchestrator
from core.gpu_lock import GPULock
from core.event_payloads import GestureActionReadyPayload
from ml.gesture.labels import IPN_LABELS
from ml.gesture.models import GestureModelConfig, build_model, checkpoint_payload
from modules.command_router import route_explicit_command
from modules.gesture_bridge import GestureActionBridge
from modules.gesture_control import GestureControlModule, TemporalGestureGate


def _candidate_files(tmp_path, *, approved: bool = False):
    model_config = GestureModelConfig(
        architecture="tiny_3d_cnn", classes=len(IPN_LABELS), width=4, dropout=0.0
    )
    payload = checkpoint_payload(build_model(model_config), model_config)
    payload.update(
        {
            "experiment": {"name": "candidate"},
            "smoke": False,
            "history": [],
            "validation": {"macro_f1": 0.1},
        }
    )
    checkpoint = tmp_path / "candidate.pt"
    torch.save(payload, checkpoint)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "selected": {"name": "candidate", "checkpoint": str(checkpoint)},
                "test": {"macro_f1": 0.1},
                "smoke": False,
                "approval": {
                    "approved": approved,
                    "failed_gates": [] if approved else ["test macro-F1 below threshold"],
                },
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return checkpoint, report, digest


def test_temporal_gate_requires_stable_windows_and_respects_cooldown():
    gate = TemporalGestureGate(0.9, consecutive_windows=3, cooldown_seconds=1.0)

    assert not gate.observe("G01", 0.95, now=0.0)
    assert not gate.observe("G01", 0.95, now=0.1)
    assert gate.observe("G01", 0.95, now=0.2)
    # A normal frame resets evidence. A new gesture cannot reuse old evidence.
    assert not gate.observe("D0X", 1.0, now=0.3)
    assert not gate.observe("G01", 0.95, now=0.4)
    assert not gate.observe("G01", 0.95, now=0.5)
    assert not gate.observe("G01", 0.95, now=0.6)  # still cooling down
    assert gate.observe("G01", 0.95, now=1.3)


class _AlwaysClick(torch.nn.Module):
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        logits = torch.full((clip.shape[0], len(IPN_LABELS)), -12.0)
        logits[:, IPN_LABELS.index("G01")] = 12.0
        return logits


async def test_runtime_publishes_a_proposal_only_after_three_windows():
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={
            "frames": 4,
            "image_size": 32,
            "confidence_threshold": 0.9,
            "consecutive_windows": 3,
            "cooldown_seconds": 0.0,
        },
    )
    bus = EventBus()
    module = GestureControlModule(config, GPULock())
    module.bus = bus
    module._model = _AlwaysClick().eval()
    module._model_ready = True
    module._armed = True
    module._generation = 1
    frames = __import__("numpy").zeros((4, 32, 32, 3), dtype="uint8")

    await module._infer_clip(frames, 1)
    await module._infer_clip(frames, 1)
    assert bus.queue.empty()
    await module._infer_clip(frames, 1)

    event = bus.queue.get_nowait()
    assert event.event_type == "gesture_action_ready"
    assert event.payload == {
        "label": "G01",
        "action_hint": "confirm",
        "confidence": 1.0,
        "consecutive_windows": 3,
        "execution": "disabled_pending_real_camera_validation",
    }


async def test_arm_is_rejected_when_no_approved_model_is_loaded():
    config = SimpleNamespace(device="cpu", model="", params={})
    bus = EventBus()
    module = GestureControlModule(config, GPULock())
    module.bus = bus

    await module._set_armed(True, source="test")

    event = bus.queue.get_nowait()
    assert event.event_type == "gesture_mode_changed"
    assert event.payload["armed"] is False
    assert event.payload["reason"] == "model_unavailable"


def test_voice_router_treats_gesture_mode_as_a_first_class_command():
    assert route_explicit_command("включи режим жестов").payload() == {
        "intent": "gesture_mode", "slots": {"enabled": True}, "confidence": 0.99
    }
    assert route_explicit_command("отключи жестами").slots == {"enabled": False}


def test_runtime_refuses_a_synthetic_smoke_checkpoint(monkeypatch):
    config = SimpleNamespace(device="cpu", model="", params={})
    module = GestureControlModule(config, GPULock())
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {
            "kind": "jarvis_gesture_from_scratch_v1",
            "pretrained": False,
            "smoke": True,
        },
    )

    with pytest.raises(ValueError, match="smoke checkpoint"):
        module._load_checkpoint(Path("synthetic.pt"))


async def test_failed_gate_checkpoint_loads_only_as_observer(tmp_path, monkeypatch):
    checkpoint, report, digest = _candidate_files(tmp_path)
    config = SimpleNamespace(
        device="cpu",
        model=str(checkpoint),
        params={
            "quality_report": str(report),
            "checkpoint_sha256": digest,
            "allow_unapproved_observer": True,
            "execution_enabled": True,
            "frames": 4,
            "image_size": 32,
        },
    )
    module = GestureControlModule(config, GPULock())
    bus = EventBus()
    await module.start(bus)

    async def no_camera(_generation):
        return None

    monkeypatch.setattr(module, "_camera_loop_async", no_camera)
    await module._set_armed(True, source="voice")
    event = bus.queue.get_nowait()

    assert module._model_ready is True
    assert module._observer_only is True
    assert module._execution_enabled is False
    assert event.event_type == "gesture_mode_changed"
    assert event.payload["armed"] is True
    assert event.payload["reason"] == "observer_unapproved_model"
    await module.stop()


async def test_failed_gate_checkpoint_is_refused_without_observer_opt_in(tmp_path):
    checkpoint, report, digest = _candidate_files(tmp_path)
    config = SimpleNamespace(
        device="cpu",
        model=str(checkpoint),
        params={
            "quality_report": str(report),
            "checkpoint_sha256": digest,
            "allow_unapproved_observer": False,
        },
    )
    module = GestureControlModule(config, GPULock())
    await module.start(EventBus())
    assert module._model_ready is False
    assert module._model is None


async def test_checkpoint_hash_mismatch_is_refused_even_in_observer_mode(tmp_path):
    checkpoint, report, _digest = _candidate_files(tmp_path)
    config = SimpleNamespace(
        device="cpu",
        model=str(checkpoint),
        params={
            "quality_report": str(report),
            "checkpoint_sha256": "0" * 64,
            "allow_unapproved_observer": True,
        },
    )
    module = GestureControlModule(config, GPULock())
    await module.start(EventBus())
    assert module._model_ready is False
    assert module._model is None


async def test_enabled_gesture_uses_the_normal_jarvis_lifecycle():
    bus = EventBus()
    orchestrator = Orchestrator(bus)
    bridge = GestureActionBridge()
    received = []

    async def record(event):
        received.append(event)

    bus.subscribe("nlu_result", record)
    await orchestrator.start()
    await bridge.start(bus)
    runner = __import__("asyncio").create_task(bus.run())
    bus.publish(
        "gesture_action_ready",
        GestureActionReadyPayload(
            label="G03",
            action_hint="navigate_up",
            confidence=0.99,
            consecutive_windows=3,
            execution="enabled",
        ),
    )
    for _ in range(50):
        if received:
            break
        await __import__("asyncio").sleep(0.01)
    await bus.stop()
    await runner
    await bridge.stop()
    await orchestrator.stop()

    assert len(received) == 1
    assert received[0].payload["intent"] == "system_control"
    assert received[0].payload["slots"] == {"action": "volume_up", "steps": 2}
