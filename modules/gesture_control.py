"""Safe webcam runtime shell for the locally trained Jarvis Gesture Core.

The CV module reads local RGB frames and emits temporally gated proposals. It
never imports Windows-control tools or treats one frame as a command; the
separate bridge accepts only the configured reversible G01-G06 test actions.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import (
    GestureActionReadyPayload,
    GestureModeChangedPayload,
    GestureRuntimeStatusPayload,
)
from core.gpu_lock import GPULock
from ml.gesture.labels import (
    IPN_LABELS,
    JESTER_SAFE_RUNTIME_MAP,
    JARVIS_ACTION_HINTS,
    NO_GESTURE_LABEL,
    SAFE_RUNTIME_LABELS,
)
from ml.gesture.models import GestureModelConfig, build_model
from modules.gesture_session_log import GestureSessionLog
from modules.gesture_ui import GesturePreviewState, build_gesture_preview

logger = logging.getLogger("jarvis.module.gesture")

_ACTION_DISPLAY_NAMES = {
    "idle": "ожидание",
    "media_play_pause": "пауза / продолжить",
    "volume_mute": "включить / выключить звук",
    "volume_up": "громкость выше",
    "volume_down": "громкость ниже",
    "media_previous": "предыдущий трек",
    "media_next": "следующий трек",
    "arm_gesture_mode": "служебный жест",
    "activate": "активация",
    "secondary_activate": "дополнительная активация",
    "zoom_in": "увеличить масштаб",
    "zoom_out": "уменьшить масштаб",
}


def _opencv_gui_available(cv2: Any) -> bool:
    """Return false for headless OpenCV wheels before ``namedWindow`` crashes."""
    try:
        build_info = str(cv2.getBuildInformation())
    except Exception:  # noqa: BLE001 - older/fake OpenCV builds may omit it
        return True
    for line in build_info.splitlines():
        if line.strip().casefold().startswith("gui:"):
            value = line.split(":", 1)[1].strip().casefold()
            return value not in {"none", "no", "false"}
    return True


@dataclass
class TemporalGestureGate:
    """Require a stable, confident temporal prediction before emitting a proposal."""

    confidence_threshold: float
    consecutive_windows: int
    cooldown_seconds: float
    require_neutral_rearm: bool = False
    _candidate: str | None = None
    _count: int = 0
    _last_emitted_at: float = float("-inf")
    _needs_neutral: bool = False

    def observe(self, label: str, confidence: float, *, now: float) -> bool:
        """Return ``True`` once for a safe action proposal, otherwise ``False``."""
        if label == NO_GESTURE_LABEL:
            self._candidate, self._count = None, 0
            self._needs_neutral = False
            return False
        if confidence < self.confidence_threshold:
            self._candidate, self._count = None, 0
            return False
        if self._needs_neutral:
            return False
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1
        if self._count < self.consecutive_windows:
            return False
        if now - self._last_emitted_at < self.cooldown_seconds:
            return False
        self._last_emitted_at = now
        # A held pose must pass the evidence requirement again after cooldown.
        self._candidate, self._count = None, 0
        self._needs_neutral = self.require_neutral_rearm
        return True

    def reset(self) -> None:
        self._candidate, self._count = None, 0
        self._needs_neutral = False


@dataclass(frozen=True)
class GestureQualityStatus:
    """Audited relationship between a checkpoint and its training report."""

    approved: bool
    selected_name: str
    test_macro_f1: float
    failed_gates: tuple[str, ...]


class GestureControlModule(BaseModule):
    """Own local webcam capture and gated inference for a trained gesture model."""

    name = "gesture"
    enabled = False

    def __init__(self, config: Any, gpu_lock: GPULock) -> None:
        super().__init__(config)
        params = getattr(config, "params", {}) or {}
        self.gpu_lock = gpu_lock
        requested_device = str(getattr(config, "device", "cpu")).casefold()
        self._device = (
            "cuda"
            if requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if requested_device == "auto"
            else requested_device
        )
        self._camera_index = int(params.get("camera_index", 0))
        default_backend = "dshow" if sys.platform == "win32" else "auto"
        self._camera_backend = str(
            params.get("camera_backend", default_backend)
        ).strip().casefold()
        self._preview_enabled = bool(params.get("preview_enabled", False))
        self._preview_window = (
            str(params.get("preview_window", "Jarvis Gesture Core")).strip()
            or "Jarvis Gesture Core"
        )
        self._toggle_hotkey = str(
            params.get("toggle_hotkey", "")
        ).strip()
        self._camera_start_timeout = float(params.get("camera_start_timeout", 8.0))
        self._gesture_log = GestureSessionLog(
            str(params.get("log_dir", "logs/gestures")),
            retention_days=int(params.get("log_retention_days", 30)),
            max_total_bytes=int(params.get("log_max_bytes", 100 * 1024 * 1024)),
            max_file_bytes=int(params.get("log_file_max_bytes", 10 * 1024 * 1024)),
        )
        self._frames_configured = "frames" in params
        self._image_size_configured = "image_size" in params
        self._resize_size_configured = "resize_size" in params
        self._window_frames_configured = "window_frames" in params
        self._frames = int(params.get("frames", 32))
        self._window_frames = int(params.get("window_frames", self._frames))
        self._image_size = int(params.get("image_size", 112))
        self._resize_size = int(params.get("resize_size", self._image_size))
        self._window_stride = int(params.get("window_stride", 4))
        self._armed_on_start = bool(params.get("armed_on_start", False))
        self._execution_enabled = bool(params.get("execution_enabled", False))
        raw_observer_allowlist = params.get("observer_action_allowlist", [])
        if not isinstance(raw_observer_allowlist, (list, tuple, set)):
            raise ValueError("gesture observer_action_allowlist must be a list")
        self._observer_action_allowlist = frozenset(
            str(label).strip() for label in raw_observer_allowlist if str(label).strip()
        )
        raw_action_allowlist = params.get(
            "action_allowlist", raw_observer_allowlist
        )
        if not isinstance(raw_action_allowlist, (list, tuple, set)):
            raise ValueError("gesture action_allowlist must be a list")
        self._action_allowlist = frozenset(
            str(label).strip() for label in raw_action_allowlist if str(label).strip()
        )
        self._quality_report = Path(str(params.get("quality_report", ""))).expanduser()
        self._expected_sha256 = str(params.get("checkpoint_sha256", "")).strip().casefold()
        self._allow_unapproved_observer = bool(
            params.get("allow_unapproved_observer", False)
        )
        self._gate = TemporalGestureGate(
            confidence_threshold=float(params.get("confidence_threshold", 0.90)),
            consecutive_windows=int(params.get("consecutive_windows", 3)),
            cooldown_seconds=float(params.get("cooldown_seconds", 1.5)),
            require_neutral_rearm=bool(params.get("require_neutral_rearm", False)),
        )
        self._validate_settings()
        self._model: torch.nn.Module | None = None
        self._runtime_kind = "legacy_3d"
        self._model_name = "unloaded"
        self._model_ready = False
        self._observer_only = False
        self._quality: GestureQualityStatus | None = None
        self._armed = False
        self._paused = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._camera_task: asyncio.Task[None] | None = None
        self._camera_start_event: asyncio.Event | None = None
        self._camera_start_status: tuple[str, str] | None = None
        self._preview_requested = threading.Event()
        self._preview_trace_id: str | None = None
        self._hotkey_listener: Any = None
        self._hotkey_tokens: set[str] = set()
        self._hotkey_chord_active = False
        self._inference_pending = threading.Event()
        self._stop_camera = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture: Any = None
        self._prediction_lock = threading.Lock()
        self._latest_prediction: tuple[str, float] | None = None
        self._latest_top3: tuple[tuple[str, float], ...] = ()
        self._latest_latency_ms = 0.0
        self._latest_fps = 0.0
        self._last_action = ""
        self._generation = 0

    def _validate_settings(self) -> None:
        if self._device not in {"cpu", "cuda"}:
            raise ValueError("gesture device must be 'auto', 'cpu' or 'cuda'")
        if self._camera_index < 0:
            raise ValueError("gesture camera_index must be >= 0")
        if self._camera_backend not in {"auto", "dshow", "msmf"}:
            raise ValueError("gesture camera_backend must be auto, dshow or msmf")
        if self._frames < 4 or self._image_size < 32:
            raise ValueError("gesture frames must be >= 4 and image_size must be >= 32")
        if self._window_frames < self._frames:
            raise ValueError("gesture window_frames must be >= frames")
        if self._resize_size < self._image_size:
            raise ValueError("gesture resize_size must be >= image_size")
        if self._window_stride < 1:
            raise ValueError("gesture window_stride must be positive")
        if not 0.0 < self._gate.confidence_threshold <= 1.0:
            raise ValueError("gesture confidence_threshold must be in (0, 1]")
        if self._gate.consecutive_windows < 2:
            raise ValueError("gesture consecutive_windows must be >= 2")
        if self._gate.cooldown_seconds < 0:
            raise ValueError("gesture cooldown_seconds must be >= 0")
        if self._camera_start_timeout <= 0:
            raise ValueError("gesture camera_start_timeout must be positive")
        invalid_observer_actions = self._observer_action_allowlist - SAFE_RUNTIME_LABELS
        if invalid_observer_actions or NO_GESTURE_LABEL in self._observer_action_allowlist:
            raise ValueError(
                "gesture observer_action_allowlist contains invalid labels: "
                f"{sorted(invalid_observer_actions | ({NO_GESTURE_LABEL} if NO_GESTURE_LABEL in self._observer_action_allowlist else set()))}"
            )
        invalid_actions = self._action_allowlist - SAFE_RUNTIME_LABELS
        if invalid_actions or NO_GESTURE_LABEL in self._action_allowlist:
            raise ValueError(
                "gesture action_allowlist contains invalid labels: "
                f"{sorted(invalid_actions | ({NO_GESTURE_LABEL} if NO_GESTURE_LABEL in self._action_allowlist else set()))}"
            )
        if self._expected_sha256 and (
            len(self._expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self._expected_sha256)
        ):
            raise ValueError("gesture checkpoint_sha256 must be a 64-character hex digest")

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        self._loop = asyncio.get_running_loop()
        bus.subscribe("gesture_mode_requested", self._on_mode_requested)
        bus.subscribe("speech_finished", self._on_speech_finished)
        checkpoint = Path(str(getattr(self.config, "model", ""))).expanduser()
        if not checkpoint.is_file():
            logger.warning(
                "GestureControlModule inactive: checkpoint is missing at %s. "
                "Train and approve Gesture Core before enabling this module.",
                checkpoint or "<empty path>",
            )
            return
        try:
            quality = await asyncio.to_thread(self._verify_quality_report, checkpoint)
            if not quality.approved and not self._allow_unapproved_observer:
                raise ValueError(
                    "gesture checkpoint failed quality gates: "
                    + "; ".join(quality.failed_gates or ("not approved",))
                )
            self._model = await asyncio.to_thread(
                self._load_checkpoint,
                checkpoint,
                expected_experiment=quality.selected_name,
            )
        except Exception:  # noqa: BLE001 - external model file must not break Jarvis startup
            logger.exception("GestureControlModule inactive: refused checkpoint %s", checkpoint)
            self._model = None
            return
        self._quality = quality
        self._observer_only = not quality.approved
        if (
            self._observer_only
            and self._execution_enabled
            and not self._observer_action_allowlist
        ):
            logger.warning(
                "Gesture execution forced off: checkpoint is observer-only "
                "(test_macro_f1=%.4f)",
                quality.test_macro_f1,
            )
            self._execution_enabled = False
        elif self._observer_only and self._execution_enabled:
            logger.warning(
                "Restricted observer action test enabled only for labels=%s",
                sorted(self._observer_action_allowlist),
            )
        self._model_ready = True
        logger.info(
            "GestureControlModule ready model=%s runtime=%s device=%s "
            "sampled_frames=%d window_frames=%d image_size=%d "
            "quality=%s test_macro_f1=%.4f; camera remains off until "
            "gesture_mode_requested",
            self._model_name,
            self._runtime_kind,
            self._device,
            self._frames,
            self._window_frames,
            self._image_size,
            "approved" if quality.approved else "observer-unapproved",
            quality.test_macro_f1,
        )
        self._start_hotkey_listener()
        if self._armed_on_start:
            await self._set_armed(True, source="config", action="enable")

    async def stop(self) -> None:
        listener, self._hotkey_listener = self._hotkey_listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:  # noqa: BLE001 - best effort Windows hook teardown
                logger.exception("failed to stop gesture hotkey listener")
        self._hotkey_tokens.clear()
        self._hotkey_chord_active = False
        await self._set_armed(False, source="shutdown", action="disable")
        self._gesture_log.close()
        self._model = None
        self._model_ready = False
        self._observer_only = False
        self._quality = None
        self._loop = None
        logger.info("GestureControlModule stopped")

    def _start_hotkey_listener(self) -> None:
        if not self._toggle_hotkey or self._hotkey_listener is not None:
            return
        try:
            keyboard = importlib.import_module("pynput.keyboard")
            self._hotkey_listener = keyboard.Listener(
                on_press=self._on_hotkey_press_thread,
                on_release=self._on_hotkey_release_thread,
            )
            self._hotkey_listener.start()
            logger.info(
                "Gesture toggle hotkey ready hotkey=%s mode=physical_vk",
                self._toggle_hotkey,
            )
        except (ImportError, OSError, ValueError):
            self._hotkey_listener = None
            logger.exception("gesture toggle hotkey unavailable")

    @staticmethod
    def _gesture_hotkey_token(key: Any) -> str | None:
        """Map a pynput key to the physical Ctrl/Alt/slash chord.

        Matching the slash character through ``GlobalHotKeys`` is unreliable
        on Windows once Ctrl+Alt or a non-English layout is active. Virtual-key
        codes keep the configured shortcut stable across keyboard layouts.
        """
        value = getattr(key, "value", key)
        vk = getattr(key, "vk", None)
        if vk is None:
            vk = getattr(value, "vk", None)
        if vk in {0x11, 0xA2, 0xA3}:
            return "ctrl"
        if vk in {0x12, 0xA4, 0xA5}:
            return "alt"
        if vk in {0xBF, 0x6F}:
            return "slash"
        character = str(getattr(key, "char", "") or "")
        if character in {"/", "?"}:
            return "slash"
        return None

    def _on_hotkey_press_thread(self, key: Any) -> None:
        token = self._gesture_hotkey_token(key)
        if token is None:
            return
        self._hotkey_tokens.add(token)
        complete = {"ctrl", "alt", "slash"} <= self._hotkey_tokens
        if complete and not self._hotkey_chord_active:
            self._hotkey_chord_active = True
            logger.info("GESTURE_TOGGLE_HOTKEY_PRESSED")
            self._on_toggle_hotkey_thread()

    def _on_hotkey_release_thread(self, key: Any) -> None:
        token = self._gesture_hotkey_token(key)
        if token is not None:
            self._hotkey_tokens.discard(token)
        if not {"ctrl", "alt", "slash"} <= self._hotkey_tokens:
            self._hotkey_chord_active = False

    def _on_toggle_hotkey_thread(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._publish_hotkey_toggle)
        except RuntimeError:
            logger.debug("gesture hotkey arrived while event loop was closing")

    def _publish_hotkey_toggle(self) -> None:
        if self.bus is None:
            return
        self.bus.publish(
            "gesture_mode_requested",
            {
                "action": "toggle",
                "source": "hotkey",
            },
        )

    def _verify_quality_report(self, checkpoint: Path) -> GestureQualityStatus:
        if not self._quality_report.is_file():
            raise ValueError(
                f"gesture quality report is missing at {self._quality_report or '<empty path>'}"
            )
        if not self._expected_sha256:
            raise ValueError("gesture checkpoint_sha256 is required")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self._expected_sha256:
            raise ValueError("gesture checkpoint SHA-256 does not match config")
        report = json.loads(self._quality_report.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict) or report.get("smoke") is True:
            raise ValueError("gesture quality report is invalid or synthetic")
        if report.get("protocol") == "official_test_once_after_train_val_selection":
            if report.get("status") != "completed" or report.get("test_split_opened") is not True:
                raise ValueError("TSN evaluation report is incomplete")
            selected_checkpoint = Path(str(report.get("checkpoint", ""))).name
            if selected_checkpoint != checkpoint.name:
                raise ValueError("TSN evaluation report does not select this checkpoint")
            metrics = report.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("TSN evaluation report is missing metrics")
            macro_f1 = float(metrics.get("macro_f1", -1.0))
            if not 0.0 <= macro_f1 <= 1.0:
                raise ValueError("TSN evaluation report has invalid test macro-F1")
            expected = int(report.get("test_instances_expected", 0))
            decoded = int(report.get("test_instances_decoded", -1))
            if expected < 1 or decoded != expected or int(report.get("decode_failures", -1)) != 0:
                raise ValueError("TSN evaluation report did not decode the full official test split")
            # Official isolated-gesture gates passed, but live webcam execution
            # remains deliberately unapproved until real-camera gates exist.
            return GestureQualityStatus(
                approved=False,
                selected_name=checkpoint.parent.name,
                test_macro_f1=macro_f1,
                failed_gates=("live webcam action gate pending",),
            )
        if report.get("protocol") == "official_test_once_after_train_validation_selection":
            if report.get("status") != "completed" or report.get("test_split_opened") is not True:
                raise ValueError("Jester evaluation report is incomplete")
            selected_checkpoint = Path(str(report.get("checkpoint", ""))).name
            if selected_checkpoint != checkpoint.name:
                raise ValueError("Jester evaluation report does not select this checkpoint")
            metrics = report.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("Jester evaluation report is missing metrics")
            samples = int(metrics.get("samples", 0))
            macro_f1 = float(metrics.get("macro_f1", -1.0))
            negative_recall = float(metrics.get("negative_recall", -1.0))
            if samples != 14_743:
                raise ValueError("Jester evaluation did not cover the full official test split")
            if not 0.0 <= macro_f1 <= 1.0 or not 0.0 <= negative_recall <= 1.0:
                raise ValueError("Jester evaluation report has invalid metrics")
            return GestureQualityStatus(
                approved=False,
                selected_name=checkpoint.parent.name,
                test_macro_f1=macro_f1,
                failed_gates=("live webcam action gate pending",),
            )
        selected = report.get("selected")
        approval = report.get("approval")
        test_metrics = report.get("test")
        if not all(isinstance(value, dict) for value in (selected, approval, test_metrics)):
            raise ValueError("gesture quality report is missing selected/test/approval")
        selected_name = str(selected.get("name", "")).strip()
        selected_checkpoint = Path(str(selected.get("checkpoint", ""))).name
        if not selected_name or selected_checkpoint != checkpoint.name:
            raise ValueError("gesture quality report does not select this checkpoint")
        macro_f1 = float(test_metrics.get("macro_f1", -1.0))
        if not 0.0 <= macro_f1 <= 1.0:
            raise ValueError("gesture quality report has invalid test macro-F1")
        failed_gates = approval.get("failed_gates", [])
        if not isinstance(failed_gates, list) or not all(
            isinstance(item, str) and item.strip() for item in failed_gates
        ):
            raise ValueError("gesture quality report has invalid failed_gates")
        approved = approval.get("approved")
        if not isinstance(approved, bool):
            raise ValueError("gesture quality report has invalid approval flag")
        if approved == bool(failed_gates):
            raise ValueError("gesture quality report approval contradicts failed_gates")
        return GestureQualityStatus(
            approved=approved,
            selected_name=selected_name,
            test_macro_f1=macro_f1,
            failed_gates=tuple(failed_gates),
        )

    def _load_checkpoint(
        self,
        checkpoint: Path,
        *,
        expected_experiment: str | None = None,
    ) -> torch.nn.Module:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("gesture checkpoint payload must be a dictionary")
        kind = payload.get("kind")
        if kind in {"ipn_tsn_resnet18_v1", "ipn_gesture_architecture_v1"}:
            return self._load_ipn_architecture_checkpoint(
                payload,
                expected_experiment=expected_experiment,
            )
        if kind == "jarvis_jester_from_scratch_v1":
            return self._load_jester_checkpoint(payload)
        if kind != "jarvis_gesture_from_scratch_v1":
            raise ValueError(f"unsupported gesture checkpoint kind: {kind!r}")
        if payload.get("pretrained") is not False:
            raise ValueError("gesture checkpoint must declare pretrained=false")
        if payload.get("smoke") is True:
            raise ValueError("synthetic smoke checkpoint cannot control Jarvis")
        if expected_experiment is not None and payload.get("experiment", {}).get("name") != expected_experiment:
            raise ValueError("gesture checkpoint experiment does not match quality report")
        raw_config = payload.get("model_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise ValueError("gesture checkpoint is missing model_config or state_dict")
        model_config = GestureModelConfig(**raw_config)
        if model_config.classes != len(IPN_LABELS):
            raise ValueError("gesture checkpoint has an incompatible class count")
        model = build_model(model_config)
        model.load_state_dict(state_dict, strict=True)
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("gesture config requests CUDA but PyTorch cannot see a CUDA device")
        self._runtime_kind = "legacy_3d"
        self._model_name = model_config.architecture
        return model.to(self._device).eval()

    def _load_jester_checkpoint(self, payload: dict[str, Any]) -> torch.nn.Module:
        """Load the audited 27-class Tiny3D checkpoint under a restricted map."""
        from src.jester.labels import JESTER_LABELS
        from src.jester.models import JesterModelConfig, build_model as build_jester_model

        if payload.get("pretrained") is not False:
            raise ValueError("Jester checkpoint must declare pretrained=false")
        if payload.get("smoke") is True:
            raise ValueError("synthetic Jester checkpoint cannot control Jarvis")
        if payload.get("labels") != list(JESTER_LABELS):
            raise ValueError("Jester checkpoint label order is incompatible with Jarvis")
        raw_config = payload.get("model_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise ValueError("Jester checkpoint is missing model_config or state_dict")
        model_config = JesterModelConfig(**raw_config)
        if model_config.name != "tiny_3d_cnn":
            raise ValueError("Jarvis runtime currently accepts only the audited Tiny3D model")
        if model_config.num_classes != len(JESTER_LABELS):
            raise ValueError("Jester checkpoint has an incompatible class count")

        data_config = payload.get("training_config", {}).get("data", {})
        expected_frames = int(data_config.get("clip_len", self._frames))
        expected_image_size = int(data_config.get("frame_size", self._image_size))
        expected_resize_size = int(data_config.get("resize_size", self._resize_size))
        if self._frames_configured and self._frames != expected_frames:
            raise ValueError(
                f"Jester runtime frames={self._frames} but checkpoint requires {expected_frames}"
            )
        if self._image_size_configured and self._image_size != expected_image_size:
            raise ValueError(
                "Jester runtime image_size="
                f"{self._image_size} but checkpoint requires {expected_image_size}"
            )
        if self._resize_size_configured and self._resize_size != expected_resize_size:
            raise ValueError(
                "Jester runtime resize_size="
                f"{self._resize_size} but checkpoint requires {expected_resize_size}"
            )
        self._frames = expected_frames
        self._image_size = expected_image_size
        self._resize_size = expected_resize_size
        if not self._window_frames_configured:
            self._window_frames = max(self._window_frames, self._frames)

        model = build_jester_model(model_config)
        model.load_state_dict(state_dict, strict=True)
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("gesture config requests CUDA but PyTorch cannot see a CUDA device")
        self._runtime_kind = "jester_tiny3d"
        self._model_name = model_config.name
        return model.to(self._device).eval()

    def _load_ipn_architecture_checkpoint(
        self,
        payload: dict[str, Any],
        *,
        expected_experiment: str | None,
    ) -> torch.nn.Module:
        if payload.get("smoke") is True:
            raise ValueError("synthetic smoke checkpoint cannot run Jarvis")
        if payload.get("labels") != list(IPN_LABELS):
            raise ValueError("TSN checkpoint label order is incompatible with Jarvis")
        if expected_experiment is not None and payload.get("run_name") != expected_experiment:
            raise ValueError("TSN checkpoint run does not match the evaluation report")
        raw_config = payload.get("model_config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, dict) or not isinstance(state_dict, dict):
            raise ValueError("TSN checkpoint is missing model_config or state_dict")
        data_config = payload.get("config", {}).get("data", {})
        expected_frames = int(data_config.get("clip_len", self._frames))
        expected_image_size = int(data_config.get("frame_size", self._image_size))
        expected_resize_size = int(data_config.get("cache_resize_size", self._resize_size))
        if self._frames_configured and self._frames != expected_frames:
            raise ValueError(
                f"TSN runtime frames={self._frames} but checkpoint requires {expected_frames}"
            )
        if self._image_size_configured and self._image_size != expected_image_size:
            raise ValueError(
                f"TSN runtime image_size={self._image_size} but checkpoint "
                f"requires {expected_image_size}"
            )
        if self._resize_size_configured and self._resize_size != expected_resize_size:
            raise ValueError(
                f"TSN runtime resize_size={self._resize_size} but checkpoint "
                f"requires {expected_resize_size}"
            )
        self._frames = expected_frames
        self._image_size = expected_image_size
        self._resize_size = expected_resize_size
        if not self._window_frames_configured:
            self._window_frames = max(self._window_frames, self._frames)
        try:
            from src.models import build_model as build_ipn_model
            from src.models import model_config as parse_ipn_model_config
        except ImportError as exc:
            raise ImportError(
                "TSN runtime requires torchvision; install the runtime dependencies"
            ) from exc
        checkpoint_config = parse_ipn_model_config(raw_config, load_pretrained=False)
        if checkpoint_config.num_classes != len(IPN_LABELS):
            raise ValueError("TSN checkpoint has an incompatible class count")
        model = build_ipn_model(checkpoint_config)
        model.load_state_dict(state_dict, strict=True)
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("gesture config requests CUDA but PyTorch cannot see a CUDA device")
        self._runtime_kind = "ipn_architecture"
        self._model_name = checkpoint_config.name
        return model.to(self._device).eval()

    async def _on_mode_requested(self, event: Event) -> None:
        action = str(event.payload.get("action") or "").casefold()
        if not action:
            action = "enable" if bool(event.payload.get("enabled", False)) else "disable"
        if action == "toggle":
            action = "disable" if self._armed else "enable"
        source = str(event.payload.get("source", "event"))
        if action == "status":
            self._publish_mode_changed(
                source=source,
                action=action,
                trace_id=event.trace_id,
            )
            return
        if action == "pause":
            if not self._armed:
                self._publish_mode_changed(
                    source=source,
                    action=action,
                    trace_id=event.trace_id,
                    reason="not_active",
                )
                return
            self._paused = True
            self._gate.reset()
            self._gesture_log.write("mode_paused", source=source)
            self._publish_mode_changed(
                source=source,
                action=action,
                trace_id=event.trace_id,
            )
            return
        if action == "resume":
            if not self._armed:
                self._publish_mode_changed(
                    source=source,
                    action=action,
                    trace_id=event.trace_id,
                    reason="not_active",
                )
                return
            self._paused = False
            self._gate.reset()
            self._gesture_log.write("mode_resumed", source=source)
            self._publish_mode_changed(
                source=source,
                action=action,
                trace_id=event.trace_id,
            )
            return
        await self._set_armed(
            action == "enable",
            source=source,
            action=action,
            trace_id=event.trace_id,
        )

    async def _on_speech_finished(self, event: Event) -> None:
        if self._armed and event.trace_id == self._preview_trace_id:
            self._preview_trace_id = None
            self._preview_requested.set()

    async def _set_armed(
        self,
        enabled: bool,
        *,
        source: str,
        action: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        assert self.bus is not None
        action = action or ("enable" if enabled else "disable")
        if enabled and not self._model_ready:
            self._publish_mode_changed(
                source=source,
                action=action,
                trace_id=trace_id,
                reason="model_unavailable",
            )
            logger.warning("GESTURE_MODE_REJECTED source=%s reason=model_unavailable", source)
            return
        if enabled == self._armed:
            if enabled and action == "enable":
                self._paused = False
                if source == "voice":
                    self._preview_trace_id = trace_id
                else:
                    self._preview_requested.set()
            self._publish_mode_changed(
                source=source,
                action=action,
                trace_id=trace_id,
            )
            return
        self._armed = enabled
        self._paused = False
        self._generation += 1
        self._gate.reset()
        with self._prediction_lock:
            self._latest_prediction = None
            self._latest_top3 = ()
            self._latest_latency_ms = 0.0
            self._latest_fps = 0.0
            self._last_action = ""
        if enabled:
            log_path = self._gesture_log.start(
                model=self._model_name,
                camera_index=self._camera_index,
                camera_backend=self._camera_backend,
                threshold=self._gate.confidence_threshold,
                stable_windows=self._gate.consecutive_windows,
                execution_labels=sorted(self._action_allowlist),
                source=source,
            )
            logger.info("GESTURE_SESSION_LOG_READY file=%s", log_path.resolve())
            self._stop_camera.clear()
            self._preview_requested.clear()
            self._preview_trace_id = trace_id if source == "voice" else None
            if source != "voice":
                self._preview_requested.set()
            self._camera_start_event = asyncio.Event()
            self._camera_start_status = None
            self._camera_task = asyncio.create_task(self._camera_loop_async(self._generation))
            try:
                await asyncio.wait_for(
                    self._camera_start_event.wait(), timeout=self._camera_start_timeout
                )
            except asyncio.TimeoutError:
                self._camera_start_status = (
                    "camera_unavailable",
                    "camera startup timed out",
                )
            status, detail = self._camera_start_status or (
                "camera_unavailable",
                "camera startup ended without status",
            )
            self._camera_start_event = None
            if status != "camera_ready":
                self._armed = False
                self._preview_trace_id = None
                self._preview_requested.clear()
                self._stop_camera.set()
                await self._drain_camera_worker()
                self._gesture_log.write(
                    "session_rejected", reason=status, detail=detail
                )
                self._gesture_log.close()
                self._publish_mode_changed(
                    source=source,
                    action=action,
                    trace_id=trace_id,
                    reason=status,
                )
                logger.warning(
                    "GESTURE_MODE_REJECTED source=%s reason=%s detail=%s",
                    source,
                    status,
                    detail,
                )
                return
        else:
            self._preview_trace_id = None
            self._preview_requested.clear()
            self._stop_camera.set()
            await self._drain_camera_worker()
            self._gesture_log.write("session_stopped", source=source)
            self._gesture_log.close()
        self._publish_mode_changed(
            source=source,
            action=action,
            trace_id=trace_id,
            reason=("observer_unapproved_model" if enabled and self._observer_only else None),
        )
        logger.info(
            "GESTURE_MODE armed=%s paused=%s source=%s action=%s",
            enabled,
            self._paused,
            source,
            action,
        )

    async def _drain_camera_worker(self) -> None:
        """Stop capture without blocking the event loop on a camera driver."""
        task = self._camera_task
        if task is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except asyncio.TimeoutError:
            logger.warning(
                "gesture camera worker did not drain; releasing capture asynchronously"
            )
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._release_capture), timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.error("gesture camera driver did not release within 2 seconds")
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
            except asyncio.TimeoutError:
                logger.error("gesture camera worker remains blocked after release")
        if task.done():
            await asyncio.gather(task, return_exceptions=True)
            self._camera_task = None

    def _publish_mode_changed(
        self,
        *,
        source: str,
        action: str,
        trace_id: str | None,
        reason: str | None = None,
    ) -> None:
        assert self.bus is not None
        self.bus.publish(
            "gesture_mode_changed",
            GestureModeChangedPayload(
                armed=self._armed,
                source=source,
                action=action,
                paused=self._paused,
                reason=reason,
            ),
            trace_id=trace_id,
        )

    async def _camera_loop_async(self, generation: int) -> None:
        await asyncio.to_thread(self._camera_loop_sync, generation)

    def _camera_loop_sync(self, generation: int) -> None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            self._publish_status_from_thread("dependency_missing", detail="opencv-python is not installed")
            return
        backend_ids = {
            "auto": cv2.CAP_ANY,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
        }
        backend_id = backend_ids[self._camera_backend]
        capture = (
            cv2.VideoCapture(self._camera_index)
            if self._camera_backend == "auto"
            else cv2.VideoCapture(self._camera_index, backend_id)
        )
        with self._capture_lock:
            self._capture = capture
        if not capture.isOpened():
            self._publish_status_from_thread(
                "camera_unavailable",
                detail=(
                    f"camera_index={self._camera_index} "
                    f"backend={self._camera_backend}"
                ),
            )
            self._release_capture()
            return
        self._publish_status_from_thread(
            "camera_ready",
            detail=(
                f"camera_index={self._camera_index} "
                f"backend={self._camera_backend}"
            ),
        )
        frames: deque[np.ndarray] = deque(maxlen=self._window_frames)
        seen = 0
        fps_frames = 0
        fps_started = time.perf_counter()
        preview: Any = None
        try:
            while not self._stop_camera.is_set() and generation == self._generation:
                ok, frame = capture.read()
                if not ok:
                    self._publish_status_from_thread("camera_read_failed")
                    return
                camera_frame = frame
                model_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                if self._runtime_kind in {"ipn_architecture", "jester_tiny3d"}:
                    model_frame = self._resize_short_side(cv2, model_frame, self._resize_size)
                else:
                    model_frame = cv2.resize(
                        model_frame,
                        (self._image_size, self._image_size),
                        interpolation=cv2.INTER_AREA,
                    )
                frames.append(model_frame)
                seen += 1
                fps_frames += 1
                fps_elapsed = time.perf_counter() - fps_started
                if fps_elapsed >= 1.0:
                    with self._prediction_lock:
                        self._latest_fps = fps_frames / fps_elapsed
                    fps_frames = 0
                    fps_started = time.perf_counter()
                if (
                    len(frames) == self._window_frames
                    and seen % self._window_stride == 0
                    and not self._inference_pending.is_set()
                    and self._loop is not None
                ):
                    self._inference_pending.set()
                    buffered = list(frames)
                    indices = np.linspace(0, len(buffered) - 1, self._frames).round().astype(int)
                    sampled = np.stack([buffered[int(index)] for index in indices]).copy()
                    future = asyncio.run_coroutine_threadsafe(
                        self._infer_clip(sampled, generation), self._loop
                    )
                    future.add_done_callback(lambda _future: self._inference_pending.clear())
                if self._preview_enabled and self._preview_requested.is_set():
                    if preview is None:
                        preview = build_gesture_preview(self._preview_window)
                        try:
                            backend = preview.open(cv2)
                        except Exception as exc:  # noqa: BLE001 - optional GUI fallback
                            preview = None
                            self._preview_requested.clear()
                            self._publish_status_from_thread(
                                "preview_unavailable",
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                            continue
                        self._gesture_log.write("preview_opened", backend=backend)
                        self._publish_status_from_thread(
                            "preview_ready", detail=f"backend={backend}"
                        )
                    if not preview.render(camera_frame, self._preview_state()):
                        preview.close()
                        preview = None
                        self._preview_requested.clear()
                        self._gesture_log.write("preview_closed", source="window")
                        self._publish_status_from_thread(
                            "preview_closed",
                            detail="window hidden; recognition remains active",
                        )
        finally:
            if preview is not None:
                preview.close()
            self._release_capture()

    def _preview_state(self) -> GesturePreviewState:
        with self._prediction_lock:
            prediction = self._latest_prediction
            top3 = self._latest_top3
            latency_ms = self._latest_latency_ms
            fps = self._latest_fps
            last_action = self._last_action
        if prediction is None:
            label, confidence = NO_GESTURE_LABEL, 0.0
        else:
            label, confidence = prediction
        return GesturePreviewState(
            status="ПАУЗА" if self._paused else "АКТИВЕН",
            label=label,
            action=_ACTION_DISPLAY_NAMES.get(
                JARVIS_ACTION_HINTS.get(label, "idle"), "ожидание"
            ),
            confidence=confidence,
            top3=top3,
            stable_count=self._gate._count,
            stable_required=self._gate.consecutive_windows,
            last_action=last_action,
            fps=fps,
            latency_ms=latency_ms,
            model=self._model_name,
            camera=f"#{self._camera_index} · {self._camera_backend}",
        )

    def _render_preview(self, cv2: Any, frame: np.ndarray) -> bool:
        """Render the raw classifier output; return false when the user exits."""
        with self._prediction_lock:
            prediction = self._latest_prediction
            top3 = self._latest_top3
        if prediction is None:
            primary = f"Collecting {self._window_frames}-frame window..."
            ranking = "Top-3: waiting for first inference"
            confidence = 0.0
            label = NO_GESTURE_LABEL
        else:
            label, confidence = prediction
            primary = (
                f"Prediction: {label} ({JARVIS_ACTION_HINTS[label]}) "
                f"{confidence:.1%}"
            )
            ranking = "Top-3: " + " | ".join(
                f"{item_label} {item_confidence:.1%}"
                for item_label, item_confidence in top3
            )
        gate_ready = (
            prediction is not None
            and label != NO_GESTURE_LABEL
            and confidence >= self._gate.confidence_threshold
        )
        color = (80, 220, 80) if gate_ready else (0, 210, 255)
        if self._observer_only and self._execution_enabled:
            safety_text = (
                "TEST ACTIONS: "
                + ", ".join(sorted(self._observer_action_allowlist))
                + " | neutral rearm required"
            )
        else:
            safety_text = (
                f"Stable display gate: >= {self._gate.confidence_threshold:.0%} x "
                f"{self._gate.consecutive_windows} | OBSERVER ONLY"
            )
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, min(height, 137)), (20, 20, 20), -1)
        cv2.putText(
            frame,
            primary,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            ranking,
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            safety_text,
            (16, 91),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Model: {self._model_name} | Q / ESC / close window: exit",
            (16, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self._preview_window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:
            return False
        try:
            return cv2.getWindowProperty(
                self._preview_window, cv2.WND_PROP_VISIBLE
            ) >= 1.0
        except Exception:  # noqa: BLE001 - unsupported property means keep running
            return True

    async def _infer_clip(self, frames: np.ndarray, generation: int) -> None:
        if not self._armed or generation != self._generation or self._model is None:
            return
        inference_started = time.perf_counter()
        clip = await asyncio.to_thread(self._prepare_clip, frames)
        async with self.gpu_lock.section("gesture"):
            label, confidence, top3 = await asyncio.to_thread(self._predict_sync, clip)
        if not self._armed or generation != self._generation:
            return
        latency_ms = (time.perf_counter() - inference_started) * 1000
        with self._prediction_lock:
            self._latest_prediction = (label, confidence)
            self._latest_top3 = top3
            self._latest_latency_ms = latency_ms
        if self._paused:
            self._gate.reset()
            emitted = False
        else:
            emitted = self._gate.observe(label, confidence, now=time.monotonic())
        execution_allowed = (
            emitted
            and self._execution_enabled
            and not self._paused
            and label in self._action_allowlist
            and (not self._observer_only or label in self._observer_action_allowlist)
        )
        self._gesture_log.write(
            "prediction",
            label=label,
            confidence=round(confidence, 6),
            top3=[
                {"label": item_label, "confidence": round(item_confidence, 6)}
                for item_label, item_confidence in top3
            ],
            latency_ms=round(latency_ms, 3),
            paused=self._paused,
            stable_count=self._gate._count,
            stable_required=self._gate.consecutive_windows,
            neutral_rearm_pending=self._gate._needs_neutral,
            emitted=emitted,
            execution_allowed=execution_allowed,
        )
        if not emitted:
            return
        assert self.bus is not None
        with self._prediction_lock:
            self._last_action = JARVIS_ACTION_HINTS[label]
        # Desktop control remains outside the CV module. An optional bridge
        # consumes only explicitly enabled, safe proposals.
        self.bus.publish(
            "gesture_action_ready",
            GestureActionReadyPayload(
                label=label,
                action_hint=JARVIS_ACTION_HINTS[label],
                confidence=round(confidence, 4),
                consecutive_windows=self._gate.consecutive_windows,
                execution=(
                    "enabled"
                    if execution_allowed
                    else (
                        "observer_unapproved_model"
                        if self._observer_only
                        else "disabled_pending_real_camera_validation"
                    )
                ),
            ),
        )
        self._gesture_log.write(
            "action_proposed",
            label=label,
            action=JARVIS_ACTION_HINTS[label],
            confidence=round(confidence, 6),
            execution="enabled" if execution_allowed else "blocked",
        )
        logger.info("GESTURE_ACTION_READY label=%s confidence=%.3f", label, confidence)

    def _prepare_clip(self, frames: np.ndarray) -> torch.Tensor:
        if self._runtime_kind in {"ipn_architecture", "jester_tiny3d"}:
            from src.data.transforms import ClipTransform, ClipTransformConfig

            transform = ClipTransform(
                ClipTransformConfig(
                    frame_size=self._image_size,
                    resize_size=self._resize_size,
                ),
                training=False,
            )
            return transform(list(frames)).unsqueeze(0)
        return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255.0)

    @staticmethod
    def _resize_short_side(cv2: Any, frame: np.ndarray, target: int) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = target / min(height, width)
        resized = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(frame, resized, interpolation=cv2.INTER_LINEAR)

    def _predict_sync(
        self, clip: torch.Tensor
    ) -> tuple[str, float, tuple[tuple[str, float], ...]]:
        if self._model is None:
            raise RuntimeError("gesture model disappeared during inference")
        with torch.inference_mode():
            with torch.autocast(
                device_type=self._device,
                dtype=torch.float16,
                enabled=self._device == "cuda" and self._runtime_kind in {
                    "ipn_architecture", "jester_tiny3d"
                },
            ):
                probabilities = self._model(clip.to(self._device)).softmax(dim=1)[0].cpu()
        output_labels = IPN_LABELS
        if self._runtime_kind == "jester_tiny3d":
            from src.jester.labels import JESTER_LABELS

            output_labels = (NO_GESTURE_LABEL, *sorted(SAFE_RUNTIME_LABELS))
            runtime_probabilities = torch.zeros(len(output_labels), dtype=probabilities.dtype)
            output_index = {label: index for index, label in enumerate(output_labels)}
            for source_index, source_label in enumerate(JESTER_LABELS):
                runtime_label = JESTER_SAFE_RUNTIME_MAP.get(source_label, NO_GESTURE_LABEL)
                runtime_probabilities[output_index[runtime_label]] += probabilities[source_index]
            probabilities = runtime_probabilities
        scores, indices = probabilities.topk(min(3, len(output_labels)))
        top3 = tuple(
            (output_labels[int(index)], float(score))
            for score, index in zip(scores, indices, strict=True)
        )
        return top3[0][0], top3[0][1], top3

    def _publish_status_from_thread(self, status: str, *, detail: str = "") -> None:
        if self.bus is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            self._handle_runtime_status,
            status,
            detail,
        )
        log_status = logger.info if status == "camera_ready" else logger.warning
        log_status("GESTURE_RUNTIME_STATUS status=%s detail=%s", status, detail)

    def _handle_runtime_status(self, status: str, detail: str) -> None:
        if self.bus is None:
            return
        self.bus.publish(
            "gesture_runtime_status",
            GestureRuntimeStatusPayload(status=status, detail=detail),
        )
        self._gesture_log.write("runtime_status", status=status, detail=detail)
        startup_terminal = {
            "camera_ready",
            "camera_unavailable",
            "camera_read_failed",
            "dependency_missing",
        }
        if self._camera_start_event is not None and status in startup_terminal:
            self._camera_start_status = (status, detail)
            self._camera_start_event.set()
            return
        if self._armed and status in {"camera_read_failed", "camera_unavailable"}:
            asyncio.create_task(
                self._set_armed(
                    False,
                    source="runtime_failure",
                    action="disable",
                )
            )

    def _release_capture(self) -> None:
        with self._capture_lock:
            capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - a driver error must not block shutdown
                logger.debug("error releasing gesture camera", exc_info=True)
