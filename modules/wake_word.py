"""Push-to-talk activation and bounded real microphone capture.

A global hotkey press publishes ``wake_word_detected`` immediately, then a
16 kHz mono ``sounddevice.InputStream`` is captured in a worker thread until
streaming Silero VAD observes end-of-speech. The resulting signed 16-bit PCM
bytes are published as ``audio_captured`` on the same trace.

The optional hardware dependencies are guarded. ``trigger()`` deliberately
remains a deterministic simulated path for CI and ``python main.py --demo``.
Wake-word recognition itself is out of scope for this step.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

from core.base_module import BaseModule
from core.event_bus import EventBus, Event

logger = logging.getLogger("jarvis.module.wake_word")

_UNSET = object()
_SOUNDDEVICE: Any = _UNSET
_PYNPUT_KEYBOARD: Any = _UNSET
_LOAD_SILERO_VAD: Any = _UNSET

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
        self._end_silence_ms = int(params.get("end_silence_ms", 800))
        self._speech_start_timeout_ms = int(
            params.get("speech_start_timeout_ms", 5000)
        )
        self._min_speech_ms = int(params.get("min_speech_ms", 250))
        self._pre_roll_ms = int(params.get("pre_roll_ms", 320))
        self._max_duration_ms = int(params.get("max_duration_ms", 15000))
        self._input_device = params.get("input_device")
        self._force_simulated = force_simulated
        self._validate_settings()

        self._sounddevice: Any = None
        self._vad_model: Any = None
        self._hotkey_listener: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._activation_task: asyncio.Task[Event | None] | None = None
        self._capture_lock = asyncio.Lock()
        self._stop_capture = threading.Event()
        self._active_trace_id: str | None = None
        self.real_activation_enabled = False

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

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("interaction_cancelled", self._on_interaction_failed)
        bus.subscribe("interaction_failed", self._on_interaction_failed)
        self._loop = asyncio.get_running_loop()
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
        logger.info(
            "WakeWordModule started mode=push-to-talk hotkey=%s sample_rate=%d",
            self._hotkey,
            self._sample_rate,
        )

    async def stop(self) -> None:
        self.real_activation_enabled = False
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
        if self._activation_task is not None and not self._activation_task.done():
            logger.info("HOTKEY_IGNORED capture already active")
            return
        self._activation_task = asyncio.create_task(self.activate())
        self._activation_task.add_done_callback(self._activation_finished)

    @staticmethod
    def _activation_finished(task: asyncio.Task[Event | None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:  # noqa: BLE001 - callback must consume task failure
            logger.exception("push-to-talk activation task failed")

    async def activate(self) -> Event | None:
        """Handle one real hotkey activation and publish captured PCM."""
        if self.bus is None:
            raise RuntimeError("WakeWordModule.start() must be called first")
        if not self.real_activation_enabled:
            logger.warning("real activation unavailable; using simulated trigger()")
            return await self.trigger()
        async with self._capture_lock:
            self._stop_capture.clear()
            logger.info("PUSH_TO_TALK_ACTIVATED hotkey=%s", self._hotkey)
            wake_event = self.bus.publish(
                "wake_word_detected", {"source": "push_to_talk"}
            )
            self._active_trace_id = wake_event.trace_id
            try:
                result = await asyncio.to_thread(self._record_microphone_sync)
            except Exception as exc:  # noqa: BLE001 - PortAudio errors vary
                logger.error(
                    "MICROPHONE_CAPTURE_FAILED: %s. Check the default input "
                    "device and Windows microphone permissions",
                    exc,
                    exc_info=True,
                )
                return wake_event
            finally:
                self._active_trace_id = None
            if result is None:
                logger.warning(
                    "MICROPHONE_EMPTY no speech detected within %.1fs; returning to waiting",
                    self._speech_start_timeout_ms / 1000,
                )
                return wake_event
            audio_event = wake_event.child(
                "audio_captured",
                {
                    "audio": result.pcm,
                    "sample_rate": self._sample_rate,
                    "channels": 1,
                    "sample_width": _PCM_WIDTH_BYTES,
                    "duration_ms": result.duration_ms,
                    "source": "microphone",
                    "capture_end": result.end_reason,
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

    def _record_microphone_sync(self) -> _CaptureResult | None:
        """Record and run streaming VAD; called only in a worker thread."""
        sd = self._sounddevice
        model = self._vad_model
        if sd is None or model is None:
            raise RuntimeError("real microphone dependencies are not initialized")
        device_info = sd.query_devices(self._input_device, kind="input")
        if not device_info or int(device_info.get("max_input_channels", 0)) < 1:
            raise RuntimeError("no usable default input device found")
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
        start_timeout_samples = self._speech_start_timeout_ms * self._sample_rate // 1000
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
                        voiced_samples = samples
                        recorded.extend(pre_roll)
                        pre_roll.clear()
                    elif total_samples >= start_timeout_samples:
                        return None
                    continue

                recorded.append(pcm)
                if probability >= self._speech_threshold:
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
