"""Safety tests for the webcam Gesture Core runtime shell."""
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from core.event_bus import Event, EventBus
from core.gpu_lock import GPULock
from core.event_payloads import GestureActionReadyPayload
from ml.gesture.labels import IPN_LABELS
from ml.gesture.models import GestureModelConfig, build_model, checkpoint_payload
from modules.command_router import route_explicit_command
from modules.gesture_bridge import GestureActionBridge
from modules.gesture_bridge import GESTURE_COMMANDS
from modules.gesture_control import (
    GestureControlModule,
    TemporalGestureGate,
    _opencv_gui_available,
)
from modules.gesture_ui import (
    EmbeddedGesturePreview,
    GesturePreviewState,
    build_gesture_preview,
    decode_gesture_datagram,
)
from src.jester.labels import JESTER_LABELS
from src.jester.models import JesterModelConfig, build_model as build_jester_model


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


def test_temporal_gate_can_require_d0x_before_repeating_an_action():
    gate = TemporalGestureGate(
        0.8,
        consecutive_windows=2,
        cooldown_seconds=0.0,
        require_neutral_rearm=True,
    )

    assert not gate.observe("G07", 0.95, now=0.0)
    assert gate.observe("G07", 0.95, now=0.1)
    assert not gate.observe("G07", 0.99, now=0.2)
    assert not gate.observe("G07", 0.99, now=0.3)
    assert not gate.observe("D0X", 0.70, now=0.4)
    assert not gate.observe("G07", 0.95, now=0.5)
    assert gate.observe("G07", 0.95, now=0.6)


def test_headless_opencv_is_detected_before_preview_start():
    class HeadlessCV2:
        @staticmethod
        def getBuildInformation():
            return "Video I/O:\n  GUI: NONE\n"

    assert _opencv_gui_available(HeadlessCV2()) is False


def test_auto_device_keeps_installer_usable_with_or_without_cuda(monkeypatch):
    config = SimpleNamespace(
        device="auto",
        model="",
        params={"preview_enabled": False, "frames": 4, "image_size": 32},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert GestureControlModule(config, GPULock())._device == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert GestureControlModule(config, GPULock())._device == "cuda"


def test_preview_shows_raw_prediction_and_q_requests_exit():
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={"preview_enabled": True, "frames": 4, "image_size": 32},
    )
    module = GestureControlModule(config, GPULock())
    module._latest_prediction = ("G03", 0.76)
    module._latest_top3 = (("G03", 0.76), ("G07", 0.14), ("D0X", 0.10))
    rendered_text: list[str] = []

    class FakeCV2:
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0
        WND_PROP_VISIBLE = 0

        @staticmethod
        def rectangle(*_args):
            return None

        @staticmethod
        def putText(_frame, value, *_args):
            rendered_text.append(value)

        @staticmethod
        def imshow(*_args):
            return None

        @staticmethod
        def waitKey(_delay):
            return ord("q")

    keep_running = module._render_preview(
        FakeCV2(), np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert keep_running is False
    assert any("Prediction: G03 (volume_up) 76.0%" in text for text in rendered_text)
    assert any("Top-3: G03 76.0% | G07 14.0% | D0X 10.0%" in text for text in rendered_text)
    assert any("OBSERVER ONLY" in text for text in rendered_text)


def test_tsn_clip_preprocessing_uses_time_first_imagenet_contract():
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={"frames": 4, "window_frames": 8, "image_size": 32, "resize_size": 40},
    )
    module = GestureControlModule(config, GPULock())
    module._runtime_kind = "ipn_architecture"
    frames = np.zeros((4, 40, 60, 3), dtype=np.uint8)

    clip = module._prepare_clip(frames)

    assert clip.shape == (1, 4, 3, 32, 32)
    assert clip.dtype == torch.float32
    assert float(clip.max()) < 0.0  # zero RGB after ImageNet normalization


def test_tsn_evaluation_report_is_observer_only(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    report = tmp_path / "evaluation.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "protocol": "official_test_once_after_train_val_selection",
                "test_split_opened": True,
                "checkpoint": str(checkpoint),
                "test_instances_expected": 1610,
                "test_instances_decoded": 1610,
                "decode_failures": 0,
                "metrics": {"macro_f1": 0.5858},
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        device="cpu",
        model=str(checkpoint),
        params={
            "quality_report": str(report),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    module = GestureControlModule(config, GPULock())

    quality = module._verify_quality_report(checkpoint)

    assert quality.approved is False
    assert quality.selected_name == tmp_path.name
    assert quality.test_macro_f1 == pytest.approx(0.5858)
    assert quality.failed_gates == ("live webcam action gate pending",)


def test_jester_final_evaluation_report_is_observer_only(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"jester checkpoint")
    report = tmp_path / "final_test.json"
    report.write_text(
        json.dumps(
            {
                "status": "completed",
                "protocol": "official_test_once_after_train_validation_selection",
                "checkpoint": str(checkpoint),
                "metrics": {
                    "samples": 14_743,
                    "macro_f1": 0.7871,
                    "negative_recall": 0.8962,
                },
                "test_split_opened": True,
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        device="cpu",
        model=str(checkpoint),
        params={
            "quality_report": str(report),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
    )
    module = GestureControlModule(config, GPULock())

    quality = module._verify_quality_report(checkpoint)

    assert quality.approved is False
    assert quality.selected_name == tmp_path.name
    assert quality.test_macro_f1 == pytest.approx(0.7871)
    assert quality.failed_gates == ("live webcam action gate pending",)


def test_jester_tiny3d_checkpoint_loads_with_training_preprocessing(tmp_path):
    model_config = JesterModelConfig(name="tiny_3d_cnn", num_classes=len(JESTER_LABELS))
    payload = build_jester_model(model_config).checkpoint_payload(
        labels=list(JESTER_LABELS),
        training_config={
            "data": {"clip_len": 4, "frame_size": 32, "resize_size": 40}
        },
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(payload, checkpoint)
    config = SimpleNamespace(
        device="cpu",
        model=str(checkpoint),
        params={"frames": 4, "image_size": 32, "resize_size": 40},
    )
    module = GestureControlModule(config, GPULock())

    model = module._load_checkpoint(checkpoint)
    clip = module._prepare_clip(np.zeros((4, 40, 60, 3), dtype=np.uint8))

    assert model.training is False
    assert module._runtime_kind == "jester_tiny3d"
    assert module._model_name == "tiny_3d_cnn"
    assert clip.shape == (1, 4, 3, 32, 32)
    assert float(clip.max()) < 0.0


@pytest.mark.parametrize(
    ("jester_label", "runtime_label"),
    [("Stop Sign", "G01"), ("Thumb Up", "G03"), ("Drumming Fingers", "D0X")],
)
def test_jester_predictions_are_restricted_to_safe_runtime_labels(
    jester_label, runtime_label
):
    class FixedJesterPrediction(torch.nn.Module):
        def forward(self, clip):
            logits = torch.full((clip.shape[0], len(JESTER_LABELS)), -12.0)
            logits[:, JESTER_LABELS.index(jester_label)] = 12.0
            return logits

    config = SimpleNamespace(device="cpu", model="", params={})
    module = GestureControlModule(config, GPULock())
    module._model = FixedJesterPrediction().eval()
    module._runtime_kind = "jester_tiny3d"

    label, confidence, top3 = module._predict_sync(torch.zeros(1, 4, 3, 32, 32))

    assert label == runtime_label
    assert confidence == pytest.approx(1.0)
    assert set(item[0] for item in top3) <= {"D0X", "G01", "G02", "G03", "G04", "G05", "G06"}


def test_tsn_checkpoint_loader_accepts_audited_contract(monkeypatch):
    class TinyTSN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(len(IPN_LABELS)))

        def forward(self, clip):
            return self.bias.unsqueeze(0).expand(clip.shape[0], -1)

    tiny = TinyTSN()
    payload = {
        "kind": "ipn_tsn_resnet18_v1",
        "run_name": "reference",
        "smoke": False,
        "labels": list(IPN_LABELS),
        "model_config": {
            "name": "tsn_resnet18",
            "num_classes": len(IPN_LABELS),
            "pretrained": True,
            "dropout": 0.2,
        },
        "state_dict": tiny.state_dict(),
        "config": {
            "data": {"clip_len": 4, "frame_size": 32, "cache_resize_size": 40}
        },
    }
    import src.models

    monkeypatch.setattr(src.models, "build_model", lambda _config: TinyTSN())
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={"frames": 4, "window_frames": 8, "image_size": 32, "resize_size": 40},
    )
    module = GestureControlModule(config, GPULock())

    model = module._load_ipn_architecture_checkpoint(
        payload,
        expected_experiment="reference",
    )

    assert isinstance(model, TinyTSN)
    assert module._runtime_kind == "ipn_architecture"
    assert module._model_name == "tsn_resnet18"


class _AlwaysClick(torch.nn.Module):
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        logits = torch.full((clip.shape[0], len(IPN_LABELS)), -12.0)
        logits[:, IPN_LABELS.index("G01")] = 12.0
        return logits


class _AlwaysOpenTwice(torch.nn.Module):
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        logits = torch.full((clip.shape[0], len(IPN_LABELS)), -12.0)
        logits[:, IPN_LABELS.index("G07")] = 12.0
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
    assert module._latest_prediction is not None
    assert module._latest_prediction[0] == "G01"
    assert module._latest_prediction[1] == pytest.approx(1.0)
    await module._infer_clip(frames, 1)
    assert bus.queue.empty()
    await module._infer_clip(frames, 1)

    event = bus.queue.get_nowait()
    assert event.event_type == "gesture_action_ready"
    assert event.payload == {
        "label": "G01",
        "action_hint": "media_play_pause",
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


async def test_camera_failure_refuses_activation_and_cleans_session(tmp_path, monkeypatch):
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={"log_dir": str(tmp_path / "gesture-logs")},
    )
    bus = EventBus()
    module = GestureControlModule(config, GPULock())
    module.bus = bus
    module._model_ready = True

    async def unavailable_camera(_generation):
        module._camera_start_status = ("camera_unavailable", "test camera")
        module._camera_start_event.set()

    monkeypatch.setattr(module, "_camera_loop_async", unavailable_camera)

    await module._set_armed(
        True, source="voice", action="enable", trace_id="camera-failure"
    )

    event = bus.queue.get_nowait()
    assert event.trace_id == "camera-failure"
    assert event.payload["armed"] is False
    assert event.payload["reason"] == "camera_unavailable"
    assert module._preview_requested.is_set() is False
    assert module._gesture_log.path is None


async def test_repeated_voice_enable_reopens_preview_only_after_confirmation():
    bus = EventBus()
    module = GestureControlModule(
        SimpleNamespace(device="cpu", model="", params={}), GPULock()
    )
    module.bus = bus
    module._model_ready = True
    module._armed = True

    await module._set_armed(
        True, source="voice", action="enable", trace_id="repeat-enable"
    )
    assert module._preview_requested.is_set() is False

    await module._on_speech_finished(
        Event(
            "speech_finished",
            {"text": "Жестовый режим активирован"},
            trace_id="repeat-enable",
        )
    )
    assert module._preview_requested.is_set() is True


def test_toggle_hotkey_publishes_a_scoped_mode_request():
    bus = EventBus()
    module = GestureControlModule(
        SimpleNamespace(device="cpu", model="", params={}), GPULock()
    )
    module.bus = bus

    module._publish_hotkey_toggle()

    event = bus.queue.get_nowait()
    assert event.event_type == "gesture_mode_requested"
    assert event.payload == {"action": "toggle", "source": "hotkey"}


def test_toggle_hotkey_uses_physical_keys_and_debounces_held_chord(monkeypatch):
    module = GestureControlModule(
        SimpleNamespace(device="cpu", model="", params={}), GPULock()
    )
    toggles = []
    monkeypatch.setattr(module, "_on_toggle_hotkey_thread", lambda: toggles.append(True))
    ctrl = SimpleNamespace(vk=0xA2)
    alt = SimpleNamespace(vk=0xA4)
    slash = SimpleNamespace(vk=0xBF)

    module._on_hotkey_press_thread(ctrl)
    module._on_hotkey_press_thread(alt)
    module._on_hotkey_press_thread(slash)
    module._on_hotkey_press_thread(slash)
    assert toggles == [True]

    module._on_hotkey_release_thread(slash)
    module._on_hotkey_press_thread(slash)
    assert toggles == [True, True]


async def test_camera_shutdown_waits_before_forcing_driver_release(monkeypatch):
    module = GestureControlModule(
        SimpleNamespace(device="cpu", model="", params={}), GPULock()
    )
    module._stop_camera.set()
    released = []

    async def camera_worker():
        await __import__("asyncio").sleep(0)

    module._camera_task = __import__("asyncio").create_task(camera_worker())
    monkeypatch.setattr(module, "_release_capture", lambda: released.append(True))

    await module._drain_camera_worker()

    assert released == []
    assert module._camera_task is None


def test_voice_router_treats_gesture_mode_as_a_first_class_command():
    assert route_explicit_command("включи режим жестов").payload() == {
        "intent": "gesture_mode",
        "slots": {"action": "enable", "enabled": True},
        "confidence": 0.99,
    }
    assert route_explicit_command("отключи управление жестами").slots == {
        "action": "disable",
        "enabled": False,
    }
    assert route_explicit_command("перейди в режим жестов").slots["action"] == "enable"
    assert route_explicit_command("начни жестовый режим").slots["action"] == "enable"
    assert route_explicit_command("запусти жестовый режим").slots["action"] == "enable"
    assert route_explicit_command("поставь жесты на паузу").slots == {"action": "pause"}
    assert route_explicit_command("возобнови режим жестов").slots == {"action": "resume"}
    assert route_explicit_command("жесты сейчас работают").slots == {"action": "status"}


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
            "log_dir": str(tmp_path / "gesture-logs"),
        },
    )
    module = GestureControlModule(config, GPULock())
    bus = EventBus()
    await module.start(bus)

    async def no_camera(_generation):
        module._camera_start_status = ("camera_ready", "test camera")
        module._camera_start_event.set()
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


async def test_observer_action_allowlist_keeps_only_selected_safe_labels_executable():
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={
            "frames": 4,
            "image_size": 32,
            "confidence_threshold": 0.8,
            "consecutive_windows": 2,
            "cooldown_seconds": 0.0,
            "require_neutral_rearm": True,
            "execution_enabled": True,
            "observer_action_allowlist": ["G01"],
        },
    )
    bus = EventBus()
    module = GestureControlModule(config, GPULock())
    module.bus = bus
    module._model = _AlwaysClick().eval()
    module._model_ready = True
    module._observer_only = True
    module._armed = True
    module._generation = 1
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    await module._infer_clip(frames, 1)
    await module._infer_clip(frames, 1)

    event = bus.queue.get_nowait()
    assert event.payload["label"] == "G01"
    assert event.payload["execution"] == "enabled"

    allowed = GestureControlModule(config, GPULock())
    allowed.bus = bus
    allowed._model = _AlwaysOpenTwice().eval()
    allowed._model_ready = True
    allowed._observer_only = True
    allowed._armed = True
    allowed._generation = 1
    await allowed._infer_clip(frames, 1)
    await allowed._infer_clip(frames, 1)

    event = bus.queue.get_nowait()
    assert event.payload["label"] == "G07"
    assert event.payload["execution"] == "observer_unapproved_model"


async def test_pause_keeps_camera_armed_but_blocks_gesture_actions():
    config = SimpleNamespace(
        device="cpu",
        model="",
        params={
            "frames": 4,
            "image_size": 32,
            "confidence_threshold": 0.8,
            "consecutive_windows": 2,
            "cooldown_seconds": 0.0,
            "execution_enabled": True,
        },
    )
    bus = EventBus()
    module = GestureControlModule(config, GPULock())
    module.bus = bus
    module._model = _AlwaysClick().eval()
    module._model_ready = True
    module._armed = True
    module._paused = True
    module._generation = 1
    frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)

    await module._infer_clip(frames, 1)
    await module._infer_clip(frames, 1)

    assert module._armed is True
    assert module._latest_prediction[0] == "G01"
    assert bus.queue.empty()


def test_g01_to_g06_are_mapped_only_to_reversible_media_actions():
    assert set(GESTURE_COMMANDS).issuperset({"G01", "G02", "G03", "G04", "G05", "G06"})
    assert "G07" not in GESTURE_COMMANDS
    assert GESTURE_COMMANDS["G01"].slots == {"action": "media_play_pause"}
    assert GESTURE_COMMANDS["G02"].slots == {"action": "volume_mute"}
    assert GESTURE_COMMANDS["G03"].slots == {"action": "volume_up", "steps": 2}
    assert GESTURE_COMMANDS["G04"].slots == {"action": "volume_down", "steps": 2}
    assert GESTURE_COMMANDS["G05"].slots == {"action": "media_previous"}
    assert GESTURE_COMMANDS["G06"].slots == {"action": "media_next"}


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


async def test_enabled_gesture_executes_silently_without_voice_lifecycle():
    bus = EventBus()
    calls = []

    class FakeTools:
        async def execute(self, name, params):
            calls.append((name, params))
            return {"ok": True, "response_text": "should stay silent"}

    bridge = GestureActionBridge(FakeTools())
    voice_events = []

    async def record_voice(event):
        voice_events.append(event)

    bus.subscribe("response_ready", record_voice)
    bus.subscribe("wake_word_detected", record_voice)
    await bridge.start(bus)
    runner = __import__("asyncio").create_task(bus.run())
    bus.publish(
        "gesture_action_ready",
        GestureActionReadyPayload(
            label="G03",
            action_hint="volume_up",
            confidence=0.99,
            consecutive_windows=3,
            execution="enabled",
        ),
    )
    for _ in range(50):
        if calls:
            break
        await __import__("asyncio").sleep(0.01)
    await bus.stop()
    await runner
    await bridge.stop()

    assert calls == [("system_control", {"action": "volume_up", "steps": 2})]
    assert voice_events == []


def test_embedded_preview_streams_authenticated_metadata_and_frame():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)

    class FakeCV2:
        IMWRITE_JPEG_QUALITY = 1

        @staticmethod
        def imencode(_extension, _frame, _settings):
            return True, np.frombuffer(b"jpeg-data", dtype=np.uint8)

        @staticmethod
        def resize(frame, _size):
            return frame

    preview = EmbeddedGesturePreview(receiver.getsockname()[1], "secret-token")
    assert preview.open(FakeCV2()) == "control_center"
    opened, _address = receiver.recvfrom(60_000)
    opened_metadata, opened_payload = decode_gesture_datagram(opened)
    assert opened_metadata == {"event": "opened", "token": "secret-token"}
    assert opened_payload == b""

    state = GesturePreviewState(
        status="АКТИВЕН",
        label="G03",
        action="громкость выше",
        confidence=0.93,
        top3=(("G03", 0.93), ("D0X", 0.04)),
        stable_count=2,
        stable_required=3,
        last_action="volume_up",
        fps=24.0,
        latency_ms=8.5,
        model="tsn_resnet18",
        camera="#0 · dshow",
    )
    assert preview.render(np.zeros((360, 640, 3), dtype=np.uint8), state)
    packet, _address = receiver.recvfrom(60_000)
    metadata, jpeg = decode_gesture_datagram(packet)
    assert metadata["token"] == "secret-token"
    assert metadata["event"] == "frame"
    assert metadata["label"] == "G03"
    assert metadata["confidence"] == pytest.approx(0.93)
    assert jpeg == b"jpeg-data"
    preview.close()
    receiver.close()


def test_control_center_environment_selects_embedded_preview(monkeypatch):
    monkeypatch.setenv("JARVIS_GESTURE_PREVIEW_PORT", "45678")
    monkeypatch.setenv("JARVIS_GESTURE_PREVIEW_TOKEN", "local-token")

    preview = build_gesture_preview("separate window must not open")

    assert isinstance(preview, EmbeddedGesturePreview)
    assert preview.port == 45678
    assert preview.token == "local-token"
