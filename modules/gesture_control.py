"""Safe webcam runtime shell for the locally trained Jarvis Gesture Core.

The module is intentionally disabled in the stock configuration.  When a
trained checkpoint is supplied, it reads RGB frames from a local webcam and
publishes *proposals* only.  It never imports a Windows-control tool, presses
keys, opens applications, or treats one frame as a command.  A future action
bridge must explicitly subscribe to ``gesture_action_ready`` after real-camera
validation is complete.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
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
from ml.gesture.labels import IPN_LABELS, JARVIS_ACTION_HINTS, NO_GESTURE_LABEL
from ml.gesture.models import GestureModelConfig, build_model

logger = logging.getLogger("jarvis.module.gesture")


@dataclass
class TemporalGestureGate:
    """Require a stable, confident temporal prediction before emitting a proposal."""

    confidence_threshold: float
    consecutive_windows: int
    cooldown_seconds: float
    _candidate: str | None = None
    _count: int = 0
    _last_emitted_at: float = float("-inf")

    def observe(self, label: str, confidence: float, *, now: float) -> bool:
        """Return ``True`` once for a safe action proposal, otherwise ``False``."""
        if label == NO_GESTURE_LABEL or confidence < self.confidence_threshold:
            self.reset()
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
        return True

    def reset(self) -> None:
        self._candidate, self._count = None, 0


class GestureControlModule(BaseModule):
    """Own local webcam capture and gated inference for a trained gesture model."""

    name = "gesture"
    enabled = False

    def __init__(self, config: Any, gpu_lock: GPULock) -> None:
        super().__init__(config)
        params = getattr(config, "params", {}) or {}
        self.gpu_lock = gpu_lock
        self._device = str(getattr(config, "device", "cpu")).casefold()
        self._camera_index = int(params.get("camera_index", 0))
        self._frames = int(params.get("frames", 32))
        self._image_size = int(params.get("image_size", 112))
        self._window_stride = int(params.get("window_stride", 4))
        self._armed_on_start = bool(params.get("armed_on_start", False))
        self._execution_enabled = bool(params.get("execution_enabled", False))
        self._gate = TemporalGestureGate(
            confidence_threshold=float(params.get("confidence_threshold", 0.90)),
            consecutive_windows=int(params.get("consecutive_windows", 3)),
            cooldown_seconds=float(params.get("cooldown_seconds", 1.5)),
        )
        self._validate_settings()
        self._model: torch.nn.Module | None = None
        self._model_ready = False
        self._armed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._camera_task: asyncio.Task[None] | None = None
        self._inference_pending = threading.Event()
        self._stop_camera = threading.Event()
        self._capture_lock = threading.Lock()
        self._capture: Any = None
        self._generation = 0

    def _validate_settings(self) -> None:
        if self._device not in {"cpu", "cuda"}:
            raise ValueError("gesture device must be 'cpu' or 'cuda'")
        if self._camera_index < 0:
            raise ValueError("gesture camera_index must be >= 0")
        if self._frames < 4 or self._image_size < 32:
            raise ValueError("gesture frames must be >= 4 and image_size must be >= 32")
        if self._window_stride < 1:
            raise ValueError("gesture window_stride must be positive")
        if not 0.0 < self._gate.confidence_threshold <= 1.0:
            raise ValueError("gesture confidence_threshold must be in (0, 1]")
        if self._gate.consecutive_windows < 2:
            raise ValueError("gesture consecutive_windows must be >= 2")
        if self._gate.cooldown_seconds < 0:
            raise ValueError("gesture cooldown_seconds must be >= 0")

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        self._loop = asyncio.get_running_loop()
        bus.subscribe("gesture_mode_requested", self._on_mode_requested)
        checkpoint = Path(str(getattr(self.config, "model", ""))).expanduser()
        if not checkpoint.is_file():
            logger.warning(
                "GestureControlModule inactive: checkpoint is missing at %s. "
                "Train and approve Gesture Core before enabling this module.",
                checkpoint or "<empty path>",
            )
            return
        try:
            self._model = await asyncio.to_thread(self._load_checkpoint, checkpoint)
        except Exception:  # noqa: BLE001 - external model file must not break Jarvis startup
            logger.exception("GestureControlModule inactive: refused checkpoint %s", checkpoint)
            self._model = None
            return
        self._model_ready = True
        logger.info(
            "GestureControlModule ready device=%s frames=%d image_size=%d; "
            "camera remains off until gesture_mode_requested",
            self._device,
            self._frames,
            self._image_size,
        )
        if self._armed_on_start:
            await self._set_armed(True, source="config")

    async def stop(self) -> None:
        await self._set_armed(False, source="shutdown")
        self._model = None
        self._model_ready = False
        self._loop = None
        logger.info("GestureControlModule stopped")

    def _load_checkpoint(self, checkpoint: Path) -> torch.nn.Module:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("kind") != "jarvis_gesture_from_scratch_v1":
            raise ValueError("checkpoint is not a Jarvis Gesture Core v1 model")
        if payload.get("pretrained") is not False:
            raise ValueError("gesture checkpoint must declare pretrained=false")
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
        return model.to(self._device).eval()

    async def _on_mode_requested(self, event: Event) -> None:
        requested = bool(event.payload.get("enabled", False))
        await self._set_armed(requested, source=str(event.payload.get("source", "event")))

    async def _set_armed(self, enabled: bool, *, source: str) -> None:
        assert self.bus is not None
        if enabled and not self._model_ready:
            self.bus.publish(
                "gesture_mode_changed",
                GestureModeChangedPayload(
                    armed=False, reason="model_unavailable", source=source
                ),
            )
            logger.warning("GESTURE_MODE_REJECTED source=%s reason=model_unavailable", source)
            return
        if enabled == self._armed:
            return
        self._armed = enabled
        self._generation += 1
        self._gate.reset()
        if enabled:
            self._stop_camera.clear()
            self._camera_task = asyncio.create_task(self._camera_loop_async(self._generation))
        else:
            self._stop_camera.set()
            self._release_capture()
            if self._camera_task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(self._camera_task), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("gesture camera worker did not drain within 2 seconds")
                self._camera_task = None
        self.bus.publish(
            "gesture_mode_changed",
            GestureModeChangedPayload(armed=enabled, source=source),
        )
        logger.info("GESTURE_MODE armed=%s source=%s", enabled, source)

    async def _camera_loop_async(self, generation: int) -> None:
        await asyncio.to_thread(self._camera_loop_sync, generation)

    def _camera_loop_sync(self, generation: int) -> None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError:
            self._publish_status_from_thread("dependency_missing", detail="opencv-python is not installed")
            return
        capture = cv2.VideoCapture(self._camera_index)
        with self._capture_lock:
            self._capture = capture
        if not capture.isOpened():
            self._publish_status_from_thread("camera_unavailable", detail=f"camera_index={self._camera_index}")
            self._release_capture()
            return
        frames: deque[np.ndarray] = deque(maxlen=self._frames)
        seen = 0
        try:
            while not self._stop_camera.is_set() and generation == self._generation:
                ok, frame = capture.read()
                if not ok:
                    self._publish_status_from_thread("camera_read_failed")
                    return
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA)
                frames.append(frame)
                seen += 1
                if (
                    len(frames) == self._frames
                    and seen % self._window_stride == 0
                    and not self._inference_pending.is_set()
                    and self._loop is not None
                ):
                    self._inference_pending.set()
                    future = asyncio.run_coroutine_threadsafe(
                        self._infer_clip(np.stack(frames).copy(), generation), self._loop
                    )
                    future.add_done_callback(lambda _future: self._inference_pending.clear())
        finally:
            self._release_capture()

    async def _infer_clip(self, frames: np.ndarray, generation: int) -> None:
        if not self._armed or generation != self._generation or self._model is None:
            return
        clip = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255.0)
        async with self.gpu_lock.section("gesture"):
            label, confidence = await asyncio.to_thread(self._predict_sync, clip)
        if not self._armed or generation != self._generation:
            return
        if not self._gate.observe(label, confidence, now=time.monotonic()):
            return
        assert self.bus is not None
        # Desktop control remains outside the CV module. An optional bridge
        # consumes only explicitly enabled, safe proposals.
        self.bus.publish(
            "gesture_action_ready",
            GestureActionReadyPayload(
                label=label,
                action_hint=JARVIS_ACTION_HINTS[label],
                confidence=round(confidence, 4),
                consecutive_windows=self._gate.consecutive_windows,
                execution="enabled" if self._execution_enabled else "disabled_pending_real_camera_validation",
            ),
        )
        logger.info("GESTURE_ACTION_READY label=%s confidence=%.3f", label, confidence)

    def _predict_sync(self, clip: torch.Tensor) -> tuple[str, float]:
        if self._model is None:
            raise RuntimeError("gesture model disappeared during inference")
        with torch.inference_mode():
            probabilities = self._model(clip.to(self._device)).softmax(dim=1)[0].cpu()
        index = int(probabilities.argmax())
        return IPN_LABELS[index], float(probabilities[index])

    def _publish_status_from_thread(self, status: str, *, detail: str = "") -> None:
        if self.bus is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(
            self.bus.publish,
            "gesture_runtime_status",
            GestureRuntimeStatusPayload(status=status, detail=detail),
        )
        logger.warning("GESTURE_RUNTIME_STATUS status=%s detail=%s", status, detail)

    def _release_capture(self) -> None:
        with self._capture_lock:
            capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - a driver error must not block shutdown
                logger.debug("error releasing gesture camera", exc_info=True)
