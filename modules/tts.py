"""TTS module — Silero synthesis with interruptible audio playback.

Subscribes to ``response_ready`` and publishes ``speech_started`` immediately,
then ``speech_finished`` after playback completes.  A newer
``wake_word_detected`` event cancels playback for the older trace (barge-in)
without emitting a stale ``speech_finished`` event.

Silero and sounddevice are optional at runtime.  If either package is absent,
or real synthesis/playback fails for a turn, the module retains the original
timed stub behavior so the event pipeline remains usable.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from core.base_module import BaseModule
from core.event_bus import EventBus, Event

logger = logging.getLogger("jarvis.module.tts")

# Optional dependencies are resolved in start(), not at module import time.
# The sentinel distinguishes "not attempted" from a test/runtime ImportError.
_UNSET = object()
_SILERO_TTS: Any = _UNSET
_SOUNDDEVICE: Any = _UNSET

_DEFAULT_MODEL = "v4_ru"
_DEFAULT_LANGUAGE = "ru"
_DEFAULT_SPEAKER = "xenia"
_DEFAULT_SAMPLE_RATE = 48000
_RUSSIAN_SPEAKERS = frozenset({"aidar", "baya", "eugene", "kseniya", "xenia"})
_RUSSIAN_SAMPLE_RATES = frozenset({8000, 24000, 48000})

_RU_ONES = (
    "", "один", "два", "три", "четыре", "пять", "шесть", "семь",
    "восемь", "девять",
)
_RU_TEENS = (
    "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
    "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
)
_RU_TENS = (
    "", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят",
    "семьдесят", "восемьдесят", "девяносто",
)
_RU_HUNDREDS = (
    "", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот",
    "семьсот", "восемьсот", "девятьсот",
)


def _ru_plural(value: int, one: str, few: str, many: str) -> str:
    last_two = value % 100
    if 11 <= last_two <= 14:
        return many
    last = value % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


def _ru_under_thousand(value: int, *, feminine: bool = False) -> str:
    words: list[str] = []
    if value >= 100:
        words.append(_RU_HUNDREDS[value // 100])
        value %= 100
    if 10 <= value <= 19:
        words.append(_RU_TEENS[value - 10])
        return " ".join(words)
    if value >= 20:
        words.append(_RU_TENS[value // 10])
        value %= 10
    if value:
        if feminine and value == 1:
            words.append("одна")
        elif feminine and value == 2:
            words.append("две")
        else:
            words.append(_RU_ONES[value])
    return " ".join(words)


def _integer_to_russian(value: int) -> str:
    """Spell a non-negative integer for the Russian Silero frontend."""
    if value == 0:
        return "ноль"
    if value < 0:
        return "минус " + _integer_to_russian(-value)
    if value >= 1_000_000:
        return " ".join(_RU_ONES[int(digit)] or "ноль" for digit in str(value))
    words: list[str] = []
    thousands, remainder = divmod(value, 1000)
    if thousands:
        words.append(_ru_under_thousand(thousands, feminine=True))
        words.append(_ru_plural(thousands, "тысяча", "тысячи", "тысяч"))
    if remainder:
        words.append(_ru_under_thousand(remainder))
    return " ".join(words)


def _prepare_russian_speech_text(text: str) -> str:
    """Convert clock times and remaining digits into pronounceable words."""
    def replace_time(match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        return " ".join(
            (
                _integer_to_russian(hours),
                _ru_plural(hours, "час", "часа", "часов"),
                _integer_to_russian(minutes),
                _ru_plural(minutes, "минута", "минуты", "минут"),
            )
        )

    prepared = re.sub(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)", replace_time, text)
    prepared = re.sub(
        r"(?<![\w])\d+(?![\w])",
        lambda match: _integer_to_russian(int(match.group(0))),
        prepared,
    )
    return prepared


@dataclass
class _SpeechSession:
    """All mutable state belonging to one response/output generation."""

    generation: int
    trace_id: str
    text: str
    task: asyncio.Task[None] | None = None
    synthesis_worker: asyncio.Task[Any] | None = None
    playback_worker: asyncio.Task[None] | None = None
    cancel_requested: bool = False
    device_cleanup_required: bool = False


def _resolve_optional_dependencies() -> tuple[Any | None, Any | None]:
    """Resolve the official Silero factory and sounddevice lazily."""
    global _SILERO_TTS, _SOUNDDEVICE

    if _SILERO_TTS is _UNSET:
        try:
            _SILERO_TTS = importlib.import_module("silero").silero_tts
        except (ImportError, AttributeError):
            _SILERO_TTS = None

    if _SOUNDDEVICE is _UNSET:
        try:
            _SOUNDDEVICE = importlib.import_module("sounddevice")
        except ImportError:
            _SOUNDDEVICE = None

    return _SILERO_TTS, _SOUNDDEVICE


class TTSModule(BaseModule):
    """Speak LLM responses using Silero, with a timed stub fallback."""

    name = "tts"
    enabled = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        params = getattr(config, "params", {}) or {}
        self._model_id = str(getattr(config, "model", "") or _DEFAULT_MODEL)
        self._language = str(params.get("language", _DEFAULT_LANGUAGE))
        self._speaker = str(params.get("speaker", _DEFAULT_SPEAKER))
        self._sample_rate = int(params.get("sample_rate", _DEFAULT_SAMPLE_RATE))
        self._validate_configuration()

        # Silero is intentionally CPU-only here.  On the target GTX 1060 3 GB,
        # scarce GPU memory is reserved for STT/LLM; CPU Silero is lightweight
        # and therefore does not need the shared GPULock.
        requested_device = str(getattr(config, "device", "cpu") or "cpu")
        self._device = "cpu"
        if requested_device.lower() != "cpu":
            logger.warning(
                "TTS device=%s requested, but Silero TTS is pinned to CPU to "
                "preserve the 3 GB GPU budget",
                requested_device,
            )

        self._model: Any = None
        self._sounddevice: Any = None
        self._state_lock = asyncio.Lock()
        self._generation = 0
        self._owner_trace_id: str | None = None
        self._device_owner_generation: int | None = None
        self._session: _SpeechSession | None = None
        # Compatibility/diagnostic mirrors used by tests and runtime logging.
        self._speak_task: asyncio.Task[None] | None = None
        self._speak_trace_id: str | None = None
        self._synthesis_worker: asyncio.Task[Any] | None = None
        self._playback_worker: asyncio.Task[None] | None = None

    def _validate_configuration(self) -> None:
        """Reject known-incompatible Russian Silero settings before download."""
        if self._language != "ru":
            return
        if not (self._model_id.endswith("_ru") or self._model_id.startswith("ru_")):
            raise ValueError(
                "Russian TTS requires a Russian Silero model (for example "
                f"v4_ru), got {self._model_id!r}"
            )
        if self._speaker not in _RUSSIAN_SPEAKERS:
            choices = ", ".join(sorted(_RUSSIAN_SPEAKERS))
            raise ValueError(
                f"Unsupported Russian Silero speaker {self._speaker!r}; "
                f"choose one of: {choices}"
            )
        if self._sample_rate not in _RUSSIAN_SAMPLE_RATES:
            choices = ", ".join(str(rate) for rate in sorted(_RUSSIAN_SAMPLE_RATES))
            raise ValueError(
                f"Unsupported sample_rate {self._sample_rate} for "
                f"{self._model_id}; choose one of: {choices}"
            )

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("response_ready", self._on_response)
        bus.subscribe("wake_word_detected", self._on_wake)
        bus.subscribe("interaction_failed", self._on_interaction_failed)
        bus.subscribe("notification_deliver", self._on_response)

        silero_tts, sounddevice = _resolve_optional_dependencies()
        if silero_tts is None:
            logger.warning(
                "silero not installed — pip install silero; "
                "TTS will run in stub-only mode"
            )
        elif sounddevice is None:
            logger.warning(
                "sounddevice not installed — pip install sounddevice; "
                "TTS will run in stub-only mode"
            )
        else:
            self._sounddevice = sounddevice
            try:
                self._model = await asyncio.to_thread(self._load_model_sync, silero_tts)
            except Exception:  # noqa: BLE001 — retain a working voice pipeline
                self._model = None
                self._sounddevice = None
                logger.exception(
                    "Silero model load failed — TTS will run in stub-only mode"
                )

        logger.info(
            "TTSModule started (mode=%s) device=%s model=%s speaker=%s",
            "real" if self._model is not None else "stub",
            self._device,
            self._model_id,
            self._speaker,
        )

    async def stop(self) -> None:
        async with self._state_lock:
            session = self._session
            if session is not None:
                await self._cancel_and_drain_locked(session)
                if self._session is session:
                    self._session = None
            self._clear_session_mirrors(session)
            self._device_owner_generation = None
        logger.info("TTSModule stopped")

    def _load_model_sync(self, silero_tts: Any) -> Any:
        """Load the official pip-package model once, outside the event loop."""
        model, _example_text = silero_tts(
            language=self._language,
            speaker=self._model_id,
        )
        to_device = getattr(model, "to", None)
        if callable(to_device):
            to_device(self._device)
        return model

    # ------------------------------------------------------------------ #
    # Event handlers
    # ------------------------------------------------------------------ #
    async def _on_response(self, event: Event) -> None:
        assert self.bus is not None
        text = str(event.payload.get("text", ""))

        # Single-writer section: an older trace is completely cancelled and
        # drained before the new trace can own or use the output device.
        async with self._state_lock:
            old_session = self._session
            if old_session is not None:
                await self._cancel_and_drain_locked(old_session)

            self._generation += 1
            session = _SpeechSession(self._generation, event.trace_id, text)
            self._session = session
            self._owner_trace_id = event.trace_id
            self._device_owner_generation = session.generation

            self.bus.publish_event(event.child("speech_started", {"text": text}))
            logger.info(
                "SPEECH_STARTED trace=%s generation=%d chars=%d",
                event.trace_id,
                session.generation,
                len(text),
            )
            session.task = asyncio.create_task(self._speak(text, session))
            self._sync_session_mirrors(session)

        try:
            await asyncio.shield(session.task)
        except asyncio.CancelledError:
            # A serialized barge-in marks the session before cancelling it.
            # Unexpected cancellation of this handler is cleaned up below and
            # then propagated instead of being mistaken for a barge-in.
            if not session.cancel_requested:
                async with self._state_lock:
                    if self._session is session:
                        await self._cancel_and_drain_locked(session)
                raise

        # Completion and final publication are atomic with respect to wake/new
        # response handlers.  A done task is not enough: this exact generation
        # must still own output at the moment speech_finished is published.
        async with self._state_lock:
            is_current = (
                self._session is session
                and self._owner_trace_id == session.trace_id
                and self._device_owner_generation == session.generation
                and not session.cancel_requested
            )
            if not is_current:
                logger.info(
                    "SPEECH_FINISH_SUPPRESSED trace=%s generation=%d",
                    session.trace_id,
                    session.generation,
                )
                return

            if session.device_cleanup_required:
                await self._stop_audio_if_owner_locked(session.generation)
            self.bus.publish_event(event.child("speech_finished", {"text": text}))
            logger.info(
                "SPEECH_FINISHED trace=%s generation=%d",
                event.trace_id,
                session.generation,
            )
            self._session = None
            self._device_owner_generation = None
            self._clear_session_mirrors(session)

    async def _on_wake(self, event: Event) -> None:
        """Serialize barge-in cancellation, draining, and ownership transfer."""
        async with self._state_lock:
            session = self._session
            if session is not None and session.trace_id != event.trace_id:
                logger.info(
                    "BARGE_IN_CANCEL old_trace=%s new_trace=%s generation=%d",
                    session.trace_id,
                    event.trace_id,
                    session.generation,
                )
                await self._cancel_and_drain_locked(session)
                if self._session is session:
                    self._session = None
                self._clear_session_mirrors(session)

            # Every distinct wake becomes the current logical owner even when
            # it has not produced a response yet.  A second wake waits for the
            # first drain, then atomically supersedes it here.
            if self._owner_trace_id != event.trace_id:
                self._generation += 1
                self._owner_trace_id = event.trace_id
            self._device_owner_generation = None

    async def _on_interaction_failed(self, event: Event) -> None:
        """Stop failed-trace audio before a later interaction can own output."""
        async with self._state_lock:
            session = self._session
            if session is not None and session.trace_id == event.trace_id:
                await self._cancel_and_drain_locked(session)
                if self._session is session:
                    self._session = None
                self._clear_session_mirrors(session)
            if self._owner_trace_id == event.trace_id:
                self._owner_trace_id = None
                self._device_owner_generation = None

    async def _cancel_and_drain_locked(self, session: _SpeechSession) -> None:
        """Cancel and fully drain one session while ``_state_lock`` is held."""
        session.cancel_requested = True
        task = session.task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

        # The child deliberately leaves playback cleanup to this serialized
        # owner.  Stop is generation-guarded, then the blocking worker is
        # awaited before any newer session may start.
        await self._drain_playback_locked(session)
        if session.device_cleanup_required:
            await self._stop_audio_if_owner_locked(session.generation)

        # Synthesis normally drains inside the child because Silero has no
        # pre-emption API.  This is a final invariant check/drain for unusual
        # external cancellation paths.
        worker = session.synthesis_worker
        if worker is not None:
            await self._await_worker_after_cancel(worker)
            session.synthesis_worker = None
            if self._synthesis_worker is worker:
                self._synthesis_worker = None

    async def _drain_playback_locked(self, session: _SpeechSession) -> None:
        worker = session.playback_worker
        if worker is None:
            return
        if not worker.done():
            await self._stop_audio_if_owner_locked(session.generation)
        try:
            await asyncio.shield(worker)
        except Exception:  # noqa: BLE001 — retrieve worker failure after cancel
            logger.debug(
                "discarding playback error from cancelled trace=%s",
                session.trace_id,
                exc_info=True,
            )
        session.playback_worker = None
        if self._playback_worker is worker:
            self._playback_worker = None

    async def _stop_audio_if_owner_locked(self, generation: int) -> bool:
        """Stop shared output only when ``generation`` still owns the device."""
        if (
            self._sounddevice is None
            or self._device_owner_generation != generation
        ):
            return False
        await asyncio.to_thread(self._sounddevice.stop)
        return True

    def _sync_session_mirrors(self, session: _SpeechSession) -> None:
        self._speak_task = session.task
        self._speak_trace_id = session.trace_id

    def _clear_session_mirrors(self, session: _SpeechSession | None) -> None:
        if session is None or self._speak_task is session.task:
            self._speak_task = None
            self._speak_trace_id = None

    # ------------------------------------------------------------------ #
    # Synthesis + playback
    # ------------------------------------------------------------------ #
    async def _speak(self, text: str, session: _SpeechSession) -> None:
        if self._model is None or self._sounddevice is None or not text:
            await self._speak_stub(text)
            return

        try:
            speech_text = (
                _prepare_russian_speech_text(text)
                if self._language == "ru"
                else text
            )
            audio = await self._synthesize_audio(speech_text, session)
            await self._play_audio(audio, session)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — one bad turn must not kill TTS
            logger.exception(
                "Silero synthesis/playback failed — falling back to stub "
                "speech for this turn"
            )
            # Runtime failure belongs to the current session.  The serialized
            # finalizer/canceller remains responsible for shared-device stop.
            await self._speak_stub(text)

    async def _synthesize_audio(self, text: str, session: _SpeechSession) -> Any:
        """Run inference off-loop and drain it if the owning turn is cancelled.

        Silero's synchronous ``apply_tts`` API has no pre-emption hook.  A
        Python thread already inside it cannot be force-killed safely, so on
        cancellation we prevent playback and drain that worker before the
        speak task finishes.  This avoids leaving untracked inference running
        in the background.
        """
        worker = asyncio.create_task(asyncio.to_thread(self._synthesize_sync, text))
        session.synthesis_worker = worker
        self._synthesis_worker = worker
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            if not session.cancel_requested:
                raise
            # Barge-in cancellation is intentional.  Repeated wakes cannot
            # cancel this task concurrently because they serialize on the
            # state lock, but the loop also tolerates unrelated repeat cancel
            # requests without abandoning the non-preemptible worker.
            await self._await_worker_after_cancel(worker)
            raise
        finally:
            if worker.done():
                session.synthesis_worker = None
            if worker.done() and self._synthesis_worker is worker:
                self._synthesis_worker = None

    @staticmethod
    async def _await_worker_after_cancel(worker: asyncio.Task[Any]) -> None:
        """Drain a shielded worker and retrieve any discarded exception."""
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                return
        if worker.cancelled():
            return
        try:
            worker.result()
        except Exception:
            # Cancellation intentionally discards synthesis output/errors;
            # retrieving the exception prevents "never retrieved" noise.
            return

    def _synthesize_sync(self, text: str) -> Any:
        """Run blocking Silero inference and return CPU audio samples."""
        audio = self._model.apply_tts(
            text=text,
            speaker=self._speaker,
            sample_rate=self._sample_rate,
        )
        # Silero commonly returns a torch.Tensor.  sounddevice accepts numpy
        # arrays, so detach and move it to CPU before leaving the worker.
        detach = getattr(audio, "detach", None)
        if callable(detach):
            audio = detach()
        cpu = getattr(audio, "cpu", None)
        if callable(cpu):
            audio = cpu()
        numpy = getattr(audio, "numpy", None)
        if callable(numpy):
            audio = numpy()
        return audio

    async def _play_audio(self, audio: Any, session: _SpeechSession) -> None:
        """Play samples off-loop; serialized owner performs cancellation cleanup."""
        worker = asyncio.create_task(asyncio.to_thread(self._play_audio_sync, audio))
        session.playback_worker = worker
        self._playback_worker = worker
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Do not stop or clear here.  _cancel_and_drain_locked owns the
            # shared device and will generation-check, stop, and drain it.
            raise
        except Exception:
            session.device_cleanup_required = True
            raise
        finally:
            if worker.done():
                session.playback_worker = None
            if worker.done() and self._playback_worker is worker:
                self._playback_worker = None

    def _play_audio_sync(self, audio: Any) -> None:
        self._sounddevice.play(audio, samplerate=self._sample_rate, blocking=True)

    @staticmethod
    async def _speak_stub(text: str) -> None:
        """Simulate the original playback duration for fallback mode."""
        duration = min(0.4, max(0.03, len(text) * 0.025))
        await asyncio.sleep(duration)


# ---------------------------------------------------------------------------- #
# Standalone test entry: `python -m modules.tts --test`
# ---------------------------------------------------------------------------- #
async def _standalone_test() -> None:
    from core.config_loader import ModuleConfig

    mod = TTSModule(config=ModuleConfig())
    bus = EventBus()
    seen: list[Event] = []

    async def record(event: Event) -> None:
        seen.append(event)

    bus.subscribe("speech_started", record)
    bus.subscribe("speech_finished", record)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    bus.publish("response_ready", {"text": "hi there"}, trace_id="tts-only")
    await asyncio.sleep(0.5)
    await bus.stop()
    await run_task
    await mod.stop()

    types = [e.event_type for e in seen]
    print(f"events={[(e.event_type, e.trace_id) for e in seen]}")
    assert types == ["speech_started", "speech_finished"], types
    assert all(e.trace_id == "tts-only" for e in seen)
    print("OK tts standalone")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        asyncio.run(_standalone_test())
    else:
        print("usage: python -m modules.tts --test")
