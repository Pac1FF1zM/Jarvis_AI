"""Isolated ``main.py --gesture_mode`` runtime regression tests."""
from __future__ import annotations

import asyncio
import sys

import pytest

import main as main_module
import modules.gesture_control as gesture_control_module
from core.event_payloads import (
    GestureActionReadyPayload,
    GestureModeChangedPayload,
    GestureRuntimeStatusPayload,
)


def test_gesture_cli_dispatches_only_the_isolated_runtime(monkeypatch):
    calls: list[str] = []

    async def fake_gesture(config_path: str) -> None:
        calls.append(config_path)

    async def forbidden_pipeline(*_args, **_kwargs) -> None:
        raise AssertionError("full Jarvis pipeline started in --gesture_mode")

    monkeypatch.setattr(main_module, "run_gesture_mode", fake_gesture)
    monkeypatch.setattr(main_module, "run_pipeline", forbidden_pipeline)
    monkeypatch.setattr(sys, "argv", ["main.py", "--gesture_mode"])

    main_module.main()

    assert calls == ["config.yaml"]


async def test_isolated_runtime_arms_reports_prediction_and_stops_cleanly(
    tmp_path, monkeypatch, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "modules:",
                "  gesture:",
                "    enabled: true",
                "    device: cpu",
                "    model: candidate.pt",
                "logging:",
                f"  log_file: '{(tmp_path / 'jarvis.log').as_posix()}'",
                f"  session_log_dir: '{(tmp_path / 'sessions').as_posix()}'",
                "  console: false",
            )
        ),
        encoding="utf-8",
    )
    stopped = asyncio.Event()
    shutdown = asyncio.Event()

    class FakeGestureControlModule:
        def __init__(self, config, gpu_lock) -> None:
            assert config.params["armed_on_start"] is False
            self.bus = None

        async def start(self, bus) -> None:
            self.bus = bus
            bus.subscribe("gesture_mode_requested", self._on_requested)

        async def _on_requested(self, event) -> None:
            assert event.payload["source"] == "standalone_cli"
            self.bus.publish(
                "gesture_mode_changed",
                GestureModeChangedPayload(
                    armed=True,
                    source="standalone_cli",
                    reason="observer_unapproved_model",
                ),
            )
            self.bus.publish(
                "gesture_runtime_status",
                GestureRuntimeStatusPayload(
                    status="camera_ready", detail="camera_index=0"
                ),
            )
            self.bus.publish(
                "gesture_action_ready",
                GestureActionReadyPayload(
                    label="G03",
                    action_hint="navigate_up",
                    confidence=0.96,
                    consecutive_windows=3,
                    execution="observer_unapproved_model",
                ),
            )
            shutdown.set()

        async def stop(self) -> None:
            stopped.set()

    monkeypatch.setattr(
        gesture_control_module, "GestureControlModule", FakeGestureControlModule
    )

    await main_module.run_gesture_mode(str(config_path), shutdown_event=shutdown)

    output = capsys.readouterr().out
    assert "Gesture Core активирован" in output
    assert "Gesture Core: camera_ready: camera_index=0" in output
    assert "observer-режиме" in output
    assert "Жест: G03 (navigate_up), уверенность 96.0%" in output
    assert stopped.is_set()


async def test_isolated_runtime_reports_unavailable_model(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "modules:\n  gesture:\n    enabled: true\n    device: cpu\n",
        encoding="utf-8",
    )

    class MissingGestureControlModule:
        def __init__(self, _config, _gpu_lock) -> None:
            self.bus = None

        async def start(self, bus) -> None:
            self.bus = bus
            bus.subscribe("gesture_mode_requested", self._on_requested)

        async def _on_requested(self, _event) -> None:
            self.bus.publish(
                "gesture_mode_changed",
                GestureModeChangedPayload(
                    armed=False, source="standalone_cli", reason="model_unavailable"
                ),
            )

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(
        gesture_control_module, "GestureControlModule", MissingGestureControlModule
    )

    with pytest.raises(main_module.GestureModeError, match="model_unavailable"):
        await main_module.run_gesture_mode(str(config_path))
