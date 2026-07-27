"""Local wake-phrase/push-to-talk activation and bounded microphone capture.

A global hotkey press publishes ``wake_word_detected`` immediately, then a
16 kHz mono ``sounddevice.InputStream`` is captured in a worker thread until
streaming Silero VAD observes end-of-speech. The resulting signed 16-bit PCM
bytes are published as ``audio_captured`` on the same trace.

openWakeWord listens locally when configured; the global hotkey remains a
fallback. Optional dependencies are guarded. ``trigger()`` deliberately stays
a deterministic simulated path for CI and ``python main.py --demo``.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from core.base_module import BaseModule
from core.event_bus import EventBus, Event
from core.profile_manager import device_fingerprint
from core.voice_calibration import apply_pcm_gain

logger = logging.getLogger("jarvis.module.wake_word")

_UNSET = object()
_SOUNDDEVICE: Any = _UNSET
_PYNPUT_KEYBOARD: Any = _UNSET
_LOAD_SILERO_VAD: Any = _UNSET
_OPENWAKEWORD: Any = _UNSET

_SAMPLE_RATE = 16_000
_BLOCK_SIZE = 512  # 32 ms; the native Silero VAD streaming window at 16 kHz.
_PCM_WIDTH_BYTES = 2
_STUB_AUDIO = b"<stub-pcm-chunks>"


@dataclass(frozen=True)
class _CaptureResult:
    pcm: bytes
    duration_ms: int
    end_reason: str


def _resolve_optional_dependencies() -> tuple[Any | None, Any | None, Any | None]:
    """Resolve capture/hotkey/VAD packages without making import-time mandatory."""
    global _SOUNDDEVICE, _PYNPUT_KEYBOARD, _LOAD_SILERO_VAD
    if _SOUNDDEVICE is _UNSET:
        try:
            _SOUNDDEVICE = importlib.import_module("sounddevice")
        except (ImportError, OSError):
            _SOUNDDEVICE = None
    if _PYNPUT_KEYBOARD is _UNSET:
        try:
            _PYNPUT_KEYBOARD = importlib.import_module("pynput.keyboard")
        except (ImportError, OSError):
            _PYNPUT_KEYBOARD = None
    if _LOAD_SILERO_VAD is _UNSET:
        try:
            _LOAD_SILERO_VAD = importlib.import_module(
                "silero_vad"
            ).load_silero_vad
        except (ImportError, AttributeError, OSError):
            _LOAD_SILERO_VAD = None
    return _SOUNDDEVICE, _PYNPUT_KEYBOARD, _LOAD_SILERO_VAD


class WakeWordModule(BaseModule):
    """Own the push-to-talk hotkey and produce bounded audio turns."""

    name = "wake_word"
    enabled = True

    def __init__(self, config: Any, *, force_simulated: bool = False) -> None:
        super().__init__(config)
        params = getattr(config, "params", {}) or {}
        self._hotkey = str(params.get("hotkey", "<ctrl>+<alt>+<space>"))
        self._sample_rate = _SAMPLE_RATE
        self._block_size = _BLOCK_SIZE
        self._speech_threshold = float(params.get("vad_threshold", 0.5))
        self._silence_threshold = self._speech_threshold
        self._end_silence_ms = int(params.get("end_silence_ms", 800))
        self._speech_start_timeout_ms = int(
            params.get("speech_start_timeout_ms", 5000)
        )
        self._min_speech_ms = int(params.get("min_speech_ms", 250))
        self._pre_roll_ms = int(params.get("pre_roll_ms", 320))
        self._max_duration_ms = int(params.get("max_duration_ms", 15000))
        self._input_device = params.get("input_device")
        self._wake_phrase_enabled = bool(params.get("wake_phrase_enabled", False))
        self._wake_phrase_model = str(params.get("wake_phrase_model", "hey_jarvis"))
        self._wake_phrase_threshold = float(params.get("wake_phrase_threshold", 0.35))
        self._wake_phrase_frames = int(params.get("wake_phrase_frames", 1))
        self._wake_phrase_vad_threshold = float(
            params.get("wake_phrase_vad_threshold", 0.3)
        )
        self._wake_phrase_auto_download = bool(params.get("wake_phrase_auto_download", True))
        self._active_session_enabled = bool(params.get("active_session_enabled", True))
        self._active_session_timeout_ms = round(
            float(params.get("active_session_timeout_seconds", 7.0)) * 1000
        )
        self._calibrations = params.get("voice_calibrations") or {}
        legacy_calibration = params.get("voice_calibration") or {}
        if legacy_calibration and not self._calibrations:
            fingerprint = str(legacy_calibration.get("device_fingerprint", ""))
            self._calibrations = {fingerprint: legacy_calibration}
        self._calibration_applied = False
        self._device_fingerprint: str | None = None
        self._pcm_gain_db = 0.0
        self._default_capture_settings = {
            "speech_threshold": self._speech_threshold,
            "silence_threshold": self._silence_threshold,
            "end_silence_ms": self._end_silence_ms,
            "min_speech_ms": self._min_speech_ms,
            "pre_roll_ms": self._pre_roll_ms,
        }
        self._force_simulated = force_simulated
        self._validate_settings()

        self._sounddevice: Any = None
        self._vad_model: Any = None
        self._hotkey_listener: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._activation_task: asyncio.Task[Event | None] | None = None
        self._wake_listener_task: asyncio.Task[None] | None = None
        self._wake_model: Any = None
        self._capture_lock = asyncio.Lock()
        self._stop_capture = threading.Event()
        self._microphone_lock = threading.Lock()
        self._wake_listener_pause = threading.Event()
        self._shutdown_requested = threading.Event()
        self._active_trace_id: str | None = None
        self._trace_sources: dict[str, str] = {}
        self.real_activation_enabled = False
        self.wake_phrase_activation_enabled = False

    def _validate_settings(self) -> None:
        if not 0.0 < self._speech_threshold < 1.0:
            raise ValueError("vad_threshold must be between 0 and 1")
        for name, value in (
            ("end_silence_ms", self._end_silence_ms),
            ("speech_start_timeout_ms", self._speech_start_timeout_ms),
            ("min_speech_ms", self._min_speech_ms),
            ("max_duration_ms", self._max_duration_ms),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self._max_duration_ms < self._min_speech_ms:
            raise ValueError("max_duration_ms must be >= min_speech_ms")
        if not 0.0 < self._wake_phrase_threshold < 1.0:
            raise ValueError("wake_phrase_threshold must be between 0 and 1")
        if self._wake_phrase_frames < 1:
            raise ValueError("wake_phrase_frames must be positive")
        if not 0.0 <= self._wake_phrase_vad_threshold < 1.0:
            raise ValueError("wake_phrase_vad_threshold must be between 0 and 1")
        if self._active_session_timeout_ms <= 0:
            raise ValueError("active_session_timeout_seconds must be positive")

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("interaction_cancelled", self._on_interaction_failed)
        bus.subscribe("interaction_failed", self._on_interaction_failed)
        bus.subscribe("interaction_completed", self._on_interaction_completed)
        bus.subscribe("speech_started", self._on_speech_started)
        bus.subscribe("speech_finished", self._on_speech_finished)
        self._loop = asyncio.get_running_loop()
        self._shutdown_requested.clear()
        if self._force_simulated:
            logger.info("WakeWordModule started mode=simulated")
            return
        sounddevice, keyboard, load_silero_vad = _resolve_optional_dependencies()
        missing: list[str] = []
        if sounddevice is None:
            missing.append("sounddevice")
        if keyboard is None:
            missing.append("pynput")
        if load_silero_vad is None:
            missing.append("silero-vad[onnx-cpu]")
        if missing:
            logger.warning(
                "real push-to-talk unavailable; install %s. "
                "Use `python main.py --demo` or trigger() for simulated input",
                ", ".join(missing),
            )
            return

        try:
            self._vad_model = await asyncio.to_thread(load_silero_vad, onnx=True)
            self._sounddevice = sounddevice
            self._hotkey_listener = keyboard.GlobalHotKeys(
                {self._hotkey: self._on_hotkey_thread}
            )
            self._hotkey_listener.start()
        except Exception:  # noqa: BLE001 - optional hardware path must degrade
            self._vad_model = None
            self._sounddevice = None
            self._hotkey_listener = None
            logger.exception(
                "push-to-talk initialization failed; check input permissions "
                "and packages, then use `python main.py --demo` as fallback"
            )
            return
        self.real_activation_enabled = True
        if self._wake_phrase_enabled:
            try:
                self._wake_model = await asyncio.to_thread(self._load_wake_model_sync)
            except Exception:  # noqa: BLE001 - optional phrase path degrades to hotkey
                self._wake_model = None
                logger.exception(
                    "wake phrase unavailable; Ctrl+Alt+Space remains active. "
                    "Install openwakeword and check network access for the one-time model download"
                )
            if self._wake_model is not None:
                self._wake_listener_task = asyncio.create_task(self._wake_listener_loop())
                self.wake_phrase_activation_enabled = True
        logger.info(
            "WakeWordModule started mode=%s hotkey=%s sample_rate=%d calibration=%s",
            "wake-phrase+push-to-talk" if self._wake_model is not None else "push-to-talk",
            self._hotkey,
            self._sample_rate,
            "pending-device-check" if self._calibrations else "defaults",
        )

    def _apply_device_calibration(self, device_info: Any) -> None:
        """Apply a profile only when it was measured on this microphone."""
        info = dict(device_info)
        fingerprint = device_fingerprint(info)
        self._device_fingerprint = fingerprint
        defaults = self._default_capture_settings
        self._speech_threshold = float(defaults["speech_threshold"])
        self._silence_threshold = float(defaults["silence_threshold"])
        self._end_silence_ms = int(defaults["end_silence_ms"])
        self._min_speech_ms = int(defaults["min_speech_ms"])
        self._pre_roll_ms = int(defaults["pre_roll_ms"])
        self._pcm_gain_db = 0.0
        self._calibration_applied = False
        calibrations = self._calibrations
        if not isinstance(calibrations, dict) or not calibrations:
            return
        calibration = calibrations.get(fingerprint)
        if not isinstance(calibration, dict):
            logger.warning(
                "VOICE_CALIBRATION_SKIPPED microphone changed current=%s known=%s; "
                "run `python main.py --calibrate-voice`",
                fingerprint,
                ",".join(sorted(str(key) for key in calibrations)) or "none",
            )
            return
        self._speech_threshold = float(
            calibration.get("vad_start_threshold", self._speech_threshold)
        )
        self._silence_threshold = float(
            calibration.get("vad_end_threshold", self._speech_threshold)
        )
        self._end_silence_ms = int(
            calibration.get("end_silence_ms", self._end_silence_ms)
        )
        self._min_speech_ms = int(
            calibration.get("min_speech_ms", self._min_speech_ms)
        )
        self._pre_roll_ms = int(calibration.get("pre_roll_ms", self._pre_roll_ms))
        self._pcm_gain_db = float(calibration.get("pcm_gain_db", 0.0))
        if not 0.0 < self._silence_threshold <= self._speech_threshold < 1.0:
            raise ValueError("calibrated VAD thresholds are invalid")
        self._validate_settings()
        self._calibration_applied = True

    async def stop(self) -> None:
        self.real_activation_enabled = False
        self.wake_phrase_activation_enabled = False
        self._shutdown_requested.set()
        self._wake_listener_pause.set()
        self._stop_capture.set()
        listener = self._hotkey_listener
        self._hotkey_listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:  # noqa: BLE001 - best-effort OS hook teardown
                logger.exception("failed to stop global hotkey listener cleanly")
        task = self._activation_task
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)
        self._activation_task = None
        listener_task = self._wake_listener_task
        if listener_task is not None and not listener_task.done():
            await asyncio.gather(listener_task, return_exceptions=True)
        self._wake_listener_task = None
        self._wake_model = None
        self._trace_sources.clear()
        self._vad_model = None
        self._sounddevice = None
        logger.info("WakeWordModule stopped")

    def _on_hotkey_thread(self) -> None:
        """Move immediately from pynput's Windows hook thread to asyncio."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._schedule_activation)
        except RuntimeError:
            logger.debug("hotkey arrived while event loop was closing")

    def _schedule_activation(self) -> None:
        if not self.real_activation_enabled:
            return
        if (
            self._capture_lock.locked()
            or self._activation_task is not None
            and not self._activation_task.done()
        ):
            logger.info("HOTKEY_IGNORED capture already active")
            return
        self._wake_listener_pause.set()
        self._activation_task = asyncio.create_task(self.activate())
        self._activation_task.add_done_callback(self._activation_finished)

    def _activation_finished(self, task: asyncio.Task[Event | None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - callback must consume task failure
            logger.exception("push-to-talk activation task failed")
        finally:
            # Keep the always-on listener paused through STT, tool execution
            # and TTS. The trace completion handler either opens the bounded
            # follow-up window or returns to wake-only sleep mode.
            self._resume_wake_listener_if_idle()

    def _resume_wake_listener_if_idle(self) -> None:
        """Resume wake-only listening when no voice trace or capture owns input."""
        if (
            not self._shutdown_requested.is_set()
            and not self._capture_lock.locked()
            and not self._trace_sources
        ):
            self._wake_listener_pause.clear()

    def _notify_speech_capture_started(self) -> None:
        """Move the VAD start edge safely from the audio thread to asyncio."""
        loop = self._loop
        trace_id = self._active_trace_id
        if loop is None or loop.is_closed() or trace_id is None:
            return
        try:
            loop.call_soon_threadsafe(self._publish_speech_capture_started, trace_id)
        except RuntimeError:
            logger.debug("speech start arrived while event loop was closing")

    def _publish_speech_capture_started(self, trace_id: str) -> None:
        if (
            self.bus is not None
            and trace_id in self._trace_sources
            and not self.bus.is_trace_closed(trace_id)
        ):
            self.bus.publish(
                "speech_capture_started",
                {"source": "microphone"},
                trace_id=trace_id,
            )

    async def activate(
        self,
        *,
        source: str = "push_to_talk",
        speech_start_timeout_ms: int | None = None,
    ) -> Event | None:
        """Handle one real hotkey activation and publish captured PCM."""
        if self.bus is None:
            raise RuntimeError("WakeWordModule.start() must be called first")
        if not self.real_activation_enabled:
            logger.warning("real activation unavailable; using simulated trigger()")
            return await self.trigger()
        async with self._capture_lock:
            self._stop_capture.clear()
            self._wake_listener_pause.set()
            logger.info("VOICE_ACTIVATED source=%s hotkey=%s", source, self._hotkey)
            wake_event = self.bus.publish(
                "wake_word_detected", {"source": source}
            )
            self._trace_sources[wake_event.trace_id] = source
            self._active_trace_id = wake_event.trace_id
            try:
                result = await asyncio.to_thread(
                    self._record_microphone_sync, speech_start_timeout_ms
                )
            except Exception as exc:  # noqa: BLE001 - PortAudio errors vary
                logger.error(
                    "MICROPHONE_CAPTURE_FAILED: %s. Check the default input "
                    "device and Windows microphone permissions",
                    exc,
                    exc_info=True,
                )
                self.bus.publish(
                    "cancel_requested",
                    {
                        "reason": "microphone_capture_failed",
                        "target_trace_id": wake_event.trace_id,
                    },
                    trace_id=wake_event.trace_id,
                )
                return wake_event
            finally:
                self._active_trace_id = None
            if result is None:
                logger.warning(
                    "MICROPHONE_EMPTY source=%s no speech detected within %.1fs; "
                    "returning to sleep mode",
                    source,
                    (speech_start_timeout_ms or self._speech_start_timeout_ms) / 1000,
                )
                if source == "active_session":
                    # A bounded follow-up session ends with an audible state
                    # change. This remains a normal successful interaction so
                    # TTS can finish before wake-word listening resumes.
                    self.bus.publish(
                        "session_sleep_requested",
                        {"text": "Отключаюсь.", "reason": "active_session_timeout"},
                        trace_id=wake_event.trace_id,
                    )
                else:
                    self.bus.publish(
                        "cancel_requested",
                        {
                            "reason": "no_speech",
                            "target_trace_id": wake_event.trace_id,
                        },
                        trace_id=wake_event.trace_id,
                    )
                return wake_event
            audio_event = wake_event.child(
                "audio_captured",
                {
                    "audio": apply_pcm_gain(result.pcm, self._pcm_gain_db),
                    "sample_rate": self._sample_rate,
                    "channels": 1,
                    "sample_width": _PCM_WIDTH_BYTES,
                    "duration_ms": result.duration_ms,
                    "source": "microphone",
                    "capture_end": result.end_reason,
                    "voice_calibrated": self._calibration_applied,
                    "input_device_fingerprint": self._device_fingerprint,
                },
            )
            self.bus.publish_event(audio_event)
            logger.info(
                "AUDIO_CAPTURED real trace=%s bytes=%d duration_ms=%d end=%s",
                audio_event.trace_id,
                len(result.pcm),
                result.duration_ms,
                result.end_reason,
            )
            return wake_event

    async def _on_interaction_failed(self, event: Event) -> None:
        if self._active_trace_id == event.trace_id:
            self._stop_capture.set()

    async def _on_interaction_completed(self, event: Event) -> None:
        """Keep a successful voice conversation open for one bounded next turn."""
        source = self._trace_sources.pop(event.trace_id, None)
        if self._shutdown_requested.is_set():
            return
        if source is None:
            # Reminder notifications have no microphone activation source but
            # still pause wake detection while their TTS is playing.
            self._resume_wake_listener_if_idle()
            return
        if event.payload.get("sleep_mode"):
            logger.info("ACTIVE_SESSION_CLOSED reason=active_session_timeout")
            self._resume_wake_listener_if_idle()
            return
        if event.payload.get("ok", True) is False or event.payload.get("cancelled"):
            logger.info(
                "ACTIVE_SESSION_CLOSED reason=%s",
                event.payload.get("reason", "interaction_failed"),
            )
            self._resume_wake_listener_if_idle()
            return
        if not self._active_session_enabled or not self.real_activation_enabled:
            self._resume_wake_listener_if_idle()
            return
        if self._activation_task is not None and not self._activation_task.done():
            logger.info("ACTIVE_SESSION_SKIPPED capture already active")
            return
        self._wake_listener_pause.set()
        logger.info(
            "ACTIVE_SESSION_LISTENING timeout=%.1fs",
            self._active_session_timeout_ms / 1000,
        )
        self._activation_task = asyncio.create_task(
            self.activate(
                source="active_session",
                speech_start_timeout_ms=self._active_session_timeout_ms,
            )
        )
        self._activation_task.add_done_callback(self._activation_finished)

    async def _on_speech_started(self, event: Event) -> None:
        # Without acoustic echo cancellation the assistant's own speaker can
        # produce false wakes. Hotkey barge-in remains available during TTS.
        self._wake_listener_pause.set()

    async def _on_speech_finished(self, event: Event) -> None:
        # Orchestrator publishes interaction_completed immediately after this
        # event. Keeping the listener paused closes the race where the wake
        # model could reopen the microphone between playback and follow-up.
        return

    def _record_microphone_sync(
        self, speech_start_timeout_ms: int | None = None
    ) -> _CaptureResult | None:
        """Record and run streaming VAD; called only in a worker thread."""
        with self._microphone_lock:
            return self._record_microphone_unlocked(speech_start_timeout_ms)

    def _record_microphone_unlocked(
        self, speech_start_timeout_ms: int | None = None
    ) -> _CaptureResult | None:
        sd = self._sounddevice
        model = self._vad_model
        if sd is None or model is None:
            raise RuntimeError("real microphone dependencies are not initialized")
        device_info = sd.query_devices(self._input_device, kind="input")
        if not device_info or int(device_info.get("max_input_channels", 0)) < 1:
            raise RuntimeError("no usable default input device found")
        self._apply_device_calibration(device_info)
        reset = getattr(model, "reset_states", None)
        if callable(reset):
            reset()

        pre_roll_blocks = max(1, self._pre_roll_ms * self._sample_rate // 1000 // self._block_size)
        pre_roll: deque[bytes] = deque(maxlen=pre_roll_blocks)
        recorded: list[bytes] = []
        speech_started = False
        voiced_samples = 0
        silent_samples = 0
        total_samples = 0
        max_samples = self._max_duration_ms * self._sample_rate // 1000
        start_timeout_samples = (
            speech_start_timeout_ms or self._speech_start_timeout_ms
        ) * self._sample_rate // 1000
        min_speech_samples = self._min_speech_ms * self._sample_rate // 1000
        end_silence_samples = self._end_silence_ms * self._sample_rate // 1000
        end_reason = "max_duration"

        stream_kwargs: dict[str, Any] = {
            "samplerate": self._sample_rate,
            "channels": 1,
            "dtype": "int16",
            "blocksize": self._block_size,
            "latency": "low",
        }
        if self._input_device is not None:
            stream_kwargs["device"] = self._input_device
        with sd.InputStream(**stream_kwargs) as stream:
            while total_samples < max_samples and not self._stop_capture.is_set():
                block, overflowed = stream.read(self._block_size)
                if overflowed:
                    logger.warning("microphone input overflow; audio may contain a gap")
                pcm = block.tobytes()
                samples = int(getattr(block, "size", self._block_size))
                total_samples += samples
                probability = self._speech_probability(model, block)

                if not speech_started:
                    pre_roll.append(pcm)
                    if probability >= self._speech_threshold:
                        speech_started = True
                        self._notify_speech_capture_started()
                        voiced_samples = samples
                        recorded.extend(pre_roll)
                        pre_roll.clear()
                    elif total_samples >= start_timeout_samples:
                        return None
                    continue

                recorded.append(pcm)
                if probability >= self._silence_threshold:
                    voiced_samples += samples
                    silent_samples = 0
                else:
                    silent_samples += samples
                    if silent_samples >= end_silence_samples:
                        if voiced_samples >= min_speech_samples:
                            end_reason = "vad_silence"
                            break
                        # Reject a click/noise burst and continue waiting for
                        # actual speech within the same bounded activation.
                        speech_started = False
                        voiced_samples = 0
                        silent_samples = 0
                        recorded.clear()
                        pre_roll.clear()

        if not recorded or voiced_samples < min_speech_samples:
            return None
        pcm = b"".join(recorded)
        captured_samples = len(pcm) // _PCM_WIDTH_BYTES
        return _CaptureResult(
            pcm=pcm,
            duration_ms=round(captured_samples * 1000 / self._sample_rate),
            end_reason=end_reason,
        )

    def _load_wake_model_sync(self) -> Any:
        global _OPENWAKEWORD
        if _OPENWAKEWORD is _UNSET:
            try:
                _OPENWAKEWORD = importlib.import_module("openwakeword")
            except (ImportError, OSError):
                _OPENWAKEWORD = None
        if _OPENWAKEWORD is None:
            raise ImportError("openwakeword is not installed")
        if self._wake_phrase_auto_download:
            utilities = importlib.import_module("openwakeword.utils")
            utilities.download_models([self._wake_phrase_model])
        model_class = importlib.import_module("openwakeword.model").Model
        return model_class(
            wakeword_models=[self._wake_phrase_model],
            inference_framework="onnx",
            vad_threshold=self._wake_phrase_vad_threshold,
        )

    async def _wake_listener_loop(self) -> None:
        while not self._shutdown_requested.is_set():
            if self._wake_listener_pause.is_set():
                await asyncio.sleep(0.05)
                continue
            try:
                detected = await asyncio.to_thread(self._listen_for_wake_sync)
            except Exception:  # noqa: BLE001 - retain hotkey when listener fails
                logger.exception("wake phrase listener failed; disabling it for this session")
                return
            if not detected or self._shutdown_requested.is_set():
                continue
            logger.info(
                "WAKE_PHRASE_DETECTED model=%s threshold=%.2f",
                self._wake_phrase_model,
                self._wake_phrase_threshold,
            )
            try:
                await self.activate(source="wake_phrase")
            finally:
                if not self._shutdown_requested.is_set():
                    self._wake_listener_pause.clear()

    def _listen_for_wake_sync(self) -> bool:
        sd = self._sounddevice
        model = self._wake_model
        if sd is None or model is None:
            return False
        import numpy as np

        stream_kwargs: dict[str, Any] = {
            "samplerate": self._sample_rate,
            "channels": 1,
            "dtype": "int16",
            "blocksize": 1280,
            "latency": "low",
        }
        if self._input_device is not None:
            stream_kwargs["device"] = self._input_device
        consecutive = 0
        last_candidate_log = 0.0
        reset = getattr(model, "reset", None)
        if callable(reset):
            reset()
        with self._microphone_lock:
            with sd.InputStream(**stream_kwargs) as stream:
                while not self._shutdown_requested.is_set() and not self._wake_listener_pause.is_set():
                    block, overflowed = stream.read(1280)
                    if overflowed:
                        logger.debug("wake phrase input overflow")
                    samples = np.asarray(block).reshape(-1).astype(np.int16, copy=False)
                    scores = model.predict(samples)
                    score = max((float(value) for value in scores.values()), default=0.0)
                    now = time.monotonic()
                    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                    if rms >= 250.0 and now - last_candidate_log >= 5.0:
                        logger.info(
                            "WAKE_PHRASE_CANDIDATE score=%.3f threshold=%.3f rms=%.0f",
                            score,
                            self._wake_phrase_threshold,
                            rms,
                        )
                        last_candidate_log = now
                    consecutive = consecutive + 1 if score >= self._wake_phrase_threshold else 0
                    if consecutive >= self._wake_phrase_frames:
                        return True
        return False

    def _speech_probability(self, model: Any, block: Any) -> float:
        import torch

        tensor = torch.from_numpy(block.copy()).reshape(-1).float().div_(32768.0)
        output = model(tensor, self._sample_rate)
        item = getattr(output, "item", None)
        return float(item() if callable(item) else output)

    async def trigger(self) -> Event:
        """Publish one deterministic simulated turn for CI and ``--demo``."""
        if self.bus is None:
            raise RuntimeError("WakeWordModule.start() must be called first")
        logger.info("WAKE_DETECTED (simulated trigger)")
        wake_event = self.bus.publish(
            "wake_word_detected", {"source": "simulated"}
        )
        await asyncio.sleep(0.05)
        audio_event = wake_event.child(
            "audio_captured",
            {
                "audio": _STUB_AUDIO,
                "sample_rate": self._sample_rate,
                "channels": 1,
                "sample_width": _PCM_WIDTH_BYTES,
                "duration_ms": 50,
                "source": "simulated",
            },
        )
        self.bus.publish_event(audio_event)
        logger.info(
            "AUDIO_CAPTURED simulated trace=%s bytes=%d",
            audio_event.trace_id,
            len(_STUB_AUDIO),
        )
        return wake_event


async def _standalone_test() -> None:
    mod = WakeWordModule(config=None)
    bus = EventBus()
    seen: list[str] = []

    async def record(event: Event) -> None:
        seen.append(f"{event.event_type}:{event.trace_id}")

    bus.subscribe("wake_word_detected", record)
    bus.subscribe("audio_captured", record)
    await mod.start(bus)
    run_task = asyncio.create_task(bus.run())
    wake = await mod.trigger()
    await asyncio.sleep(0.1)
    await bus.stop()
    await run_task
    await mod.stop()
    assert len(seen) == 2, seen
    assert all(wake.trace_id in event for event in seen)
    print(f"trace_id={wake.trace_id}")
    print(f"events_seen={seen}")
    print("OK wake_word standalone")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        asyncio.run(_standalone_test())
    else:
        print("usage: python -m modules.wake_word --test")
