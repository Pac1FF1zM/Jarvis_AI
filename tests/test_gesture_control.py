"""Safety tests for the webcam Gesture Core runtime shell."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from core.event_bus import EventBus
from core.gpu_lock import GPULock
from ml.gesture.labels import IPN_LABELS
from modules.gesture_control import GestureControlModule, TemporalGestureGate


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
