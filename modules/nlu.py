"""Jarvis-owned neural language-understanding module.

Consumes ``transcription_ready`` and publishes ``nlu_result``.  The checkpoint
contains its own vocabulary and neural weights trained from random
initialisation by :mod:`ml.nlu.train`; no downloaded model or tokenizer is
used.  Loading and inference run off the asyncio event-loop thread.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Callable

from core.base_module import BaseModule
from core.event_bus import Event, EventBus
from core.event_payloads import CancelRequestedPayload, NLUResultPayload
from core.gpu_lock import GPULock
from ml.nlu.inference import NLUPredictor
from ml.nlu.schema import NLUResult
from modules.command_router import RoutedAction, route_explicit_command, split_compound_command
from tools._applications import resolve_application

logger = logging.getLogger("jarvis.module.nlu")

_OPEN_REQUESTS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        # Direct imperative, including polite words before or after the verb.
        r"^\s*(?:(?:джарвис|пожалуйста|будь\s+добр)\s+)*"
        r"(?:открой(?:-ка)?|открыть|запусти|запустить|запустим|включи|open|launch)"
        r"(?:\s+(?:мне|для\s+меня|пожалуйста|приложение|программу))*\s+(.+?)\s*$",
        # Explicit request forms that still contain an unambiguous launch verb.
        r"^\s*(?:(?:джарвис|пожалуйста)\s+)*"
        r"(?:давай\s+(?:откроем|запустим|включим)|хочу\s+открыть|"
        r"мне\s+нужно\s+открыть|мне\s+(?:сейчас\s+)?нуж(?:ен|на|но)|"
        r"нужно\s+запустить|можешь\s+(?:открыть|запустить)|"
        r"можно\s+(?:открыть|запустить)|пора\s+открыть|"
        r"прошу\s+(?:открыть|запустить)|я\s+хочу\s+чтобы\s+ты\s+открыл)"
        r"(?:\s+(?:мне|приложение|программу))*\s+(.+?)\s*$",
    )
)

_LIST_REMINDERS = re.compile(
    r"^\s*(?:покажи|перечисли|назови|какие|список)\s+"
    r"(?:(?:у\s+меня|мои|активные)\s+)?напоминани(?:я|й)\s*$",
    flags=re.IGNORECASE,
)
_CANCEL_REMINDER = re.compile(
    r"^\s*(?:отмени|удали|сними)\s+напоминание"
    r"(?:\s+(?:номер|№))?\s+(\d+)\s*$",
    flags=re.IGNORECASE,
)
_RELATIVE_REMINDERS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^\s*напомни(?:\s+мне)?\s+через\s+(\d+)\s+минут(?:у|ы)?\s+(.+?)\s*$",
        r"^\s*через\s+(\d+)\s+минут(?:у|ы)?\s+напомни(?:\s+мне)?\s+(.+?)\s*$",
    )
)
_ABSOLUTE_REMINDERS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^\s*(?:напомни(?:\s+мне)?|поставь\s+напоминание)\s+"
        r"(?:(сегодня|завтра)\s+)?(?:в|на)\s+(\d{1,2}[.:]\d{2})\s+(.+?)\s*$",
        r"^\s*(?:(сегодня|завтра)\s+)?в\s+(\d{1,2}[.:]\d{2})\s+"
        r"напомни(?:\s+мне)?\s+(.+?)\s*$",
    )
)


def _normalise_transcription_for_nlu(text: str) -> str:
    """Repair a few observed Russian Whisper errors before neural routing."""
    corrected = text.lower().replace("ё", "е")
    replacements = (
        (r"\bотпрой\b", "открой"),
        (r"\bколька\s+времени\b", "сколько времени"),
        (r"\bк\s+а(?:л|ль)кулятор(?:ы|а|ом)?\b", "калькулятор"),
        (r"\bкалькуляторы\b", "калькулятор"),
        (r"\bблокноты\b", "блокнот"),
        (r"\b(?:паинт|пайнт|пейнт|пеинт|пэйнт)\b", "paint"),
        (r"\b(?:дисорд|дискод|дискор)\b", "дискорд"),
    )
    for pattern, replacement in replacements:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", corrected).strip()


def _apply_runtime_command_guardrails(
    normalised_text: str,
    result: NLUResult,
) -> NLUResult:
    """Rescue explicit safe launches while keeping the allow-list authoritative.

    A distorted verb/name may fool the neural intent head, but it still cannot
    launch an arbitrary executable: correction is allowed only when an
    imperative is present and the requested tail resolves to the fixed list.
    """
    match = None
    for pattern in _OPEN_REQUESTS:
        match = pattern.match(normalised_text)
        if match:
            break
    if not match:
        return result
    requested = re.sub(
        r"\s+(?:пожалуйста|джарвис)$", "", match.group(1), flags=re.IGNORECASE
    ).strip(" ,.:;!?-")
    application = resolve_application(requested)
    if application is None:
        return result
    return NLUResult(
        "open_application",
        max(result.confidence, 0.99),
        {"application": application.name},
    )


def _clean_reminder_text(value: str) -> str:
    return re.sub(
        r"^(?:о\s+том,?\s+чтобы|о\s+том\s+что|про|что|о)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip(" ,.:;!?-")


def _apply_reminder_guardrails(
    normalised_text: str, result: NLUResult
) -> NLUResult:
    """Recognize exact persistent-reminder controls around the neural model."""
    if _LIST_REMINDERS.match(normalised_text):
        return NLUResult("list_reminders", 0.99, {})
    cancel = _CANCEL_REMINDER.match(normalised_text)
    if cancel:
        return NLUResult(
            "cancel_reminder", 0.99, {"reminder_id": cancel.group(1)}
        )
    for pattern in _RELATIVE_REMINDERS:
        match = pattern.match(normalised_text)
        if match:
            message = _clean_reminder_text(match.group(2))
            if message:
                return NLUResult(
                    "set_reminder",
                    max(result.confidence, 0.99),
                    {"minutes": match.group(1), "reminder_text": message},
                )
    for pattern in _ABSOLUTE_REMINDERS:
        match = pattern.match(normalised_text)
        if match:
            message = _clean_reminder_text(match.group(3))
            if message:
                slots = {
                    "clock_time": match.group(2).replace(".", ":"),
                    "reminder_text": message,
                }
                if match.group(1):
                    slots["day"] = match.group(1).casefold()
                return NLUResult(
                    "set_reminder", max(result.confidence, 0.99), slots
                )
    return result


class NLUModule(BaseModule):
    name = "nlu"
    enabled = True

    def __init__(
        self,
        config: Any,
        gpu_lock: GPULock,
        predictor_factory: Callable[..., Any] = NLUPredictor,
    ) -> None:
        super().__init__(config)
        self.gpu_lock = gpu_lock
        self._predictor_factory = predictor_factory
        self._predictor: Any = None
        self._threshold = float(config.params.get("confidence_threshold", 0.55))
        self._pending_transcriptions: dict[str, Event] = {}
        self._thinking_ready: set[str] = set()
        self._pending_lock = asyncio.Lock()
        self._last_action: RoutedAction | None = None

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe("transcription_ready", self._on_transcription)
        bus.subscribe("thinking_ready", self._on_thinking_ready)
        bus.subscribe("interaction_cancelled", self._on_trace_closed)
        bus.subscribe("interaction_failed", self._on_trace_closed)
        checkpoint = Path(self.config.model or "models/nlu_word_bigru_curriculum.pt")
        if not checkpoint.is_file():
            logger.error(
                "NLU checkpoint not found at %s; run `python -m ml.nlu.train`",
                checkpoint,
            )
        else:
            self._predictor = await asyncio.to_thread(
                self._predictor_factory, checkpoint, self.config.device
            )
        logger.info(
            "NLUModule started mode=%s checkpoint=%s threshold=%.2f",
            "neural" if self._predictor is not None else "safe-unknown",
            checkpoint,
            self._threshold,
        )

    async def stop(self) -> None:
        async with self._pending_lock:
            self._pending_transcriptions.clear()
            self._thinking_ready.clear()
        self._predictor = None
        logger.info("NLUModule stopped")

    async def _on_transcription(self, event: Event) -> None:
        await self._stage_event(transcription=event)

    async def _on_thinking_ready(self, event: Event) -> None:
        await self._stage_event(thinking_event=event)

    async def _stage_event(
        self,
        *,
        transcription: Event | None = None,
        thinking_event: Event | None = None,
    ) -> None:
        """Run inference only after both sides of the lifecycle barrier arrive."""
        trace_id = (transcription or thinking_event).trace_id  # type: ignore[union-attr]
        assert self.bus is not None
        if self.bus.is_trace_closed(trace_id):
            return
        ready: Event | None = None
        async with self._pending_lock:
            if transcription is not None:
                self._pending_transcriptions[trace_id] = transcription
            if thinking_event is not None:
                self._thinking_ready.add(trace_id)
            if trace_id in self._pending_transcriptions and trace_id in self._thinking_ready:
                ready = self._pending_transcriptions.pop(trace_id)
                self._thinking_ready.remove(trace_id)
        if ready is not None:
            await self._process_transcription(ready)

    async def _on_trace_closed(self, event: Event) -> None:
        """Discard either half of the NLU barrier for a terminated trace."""
        async with self._pending_lock:
            self._pending_transcriptions.pop(event.trace_id, None)
            self._thinking_ready.discard(event.trace_id)

    async def _process_transcription(self, event: Event) -> None:
        assert self.bus is not None
        text = str(event.payload.get("text", "")).strip()
        normalised_text = _normalise_transcription_for_nlu(text)
        parts = split_compound_command(normalised_text)
        actions: list[RoutedAction] = []
        try:
            for part in parts:
                routed = route_explicit_command(part, previous_action=self._last_action)
                if routed is None:
                    routed = await self._predict_action(part)
                actions.append(routed)
        except Exception:  # noqa: BLE001 - keep voice pipeline responsive
            logger.exception("NLU inference failed; rejecting turn as unknown")
            actions = [RoutedAction("unknown", {}, 0.0)]
        result_action = actions[0]
        result = NLUResult(result_action.intent, result_action.confidence, result_action.slots)

        if normalised_text != text.casefold():
            logger.info("NLU_NORMALIZED original=%r normalized=%r", text, normalised_text)

        raw_intent = result.intent
        accepted_intent = raw_intent if result.confidence >= self._threshold else "unknown"
        if self.bus.is_trace_closed(event.trace_id):
            logger.info("NLU_RESULT_DISCARDED cancelled trace=%s", event.trace_id)
            return
        if accepted_intent == "cancel":
            self.bus.publish_event(
                event.child(
                    "cancel_requested",
                    CancelRequestedPayload(
                        reason="user_requested",
                        text=text,
                        intent_confidence=result.confidence,
                    ),
                )
            )
            logger.info(
                "CANCEL_REQUESTED trace=%s conf=%.3f",
                event.trace_id,
                result.confidence,
            )
            return
        output = event.child(
            "nlu_result",
            NLUResultPayload(
                text=text,
                confidence=float(event.payload.get("confidence", 0.0)),
                intent=accepted_intent,
                raw_intent=raw_intent,
                intent_confidence=result.confidence,
                slots=result.slots if accepted_intent != "unknown" else {},
                actions=[
                    {
                        **action.payload(),
                        "raw_intent": action.intent,
                        "intent": action.intent if action.confidence >= self._threshold else "unknown",
                        "slots": dict(action.slots) if action.confidence >= self._threshold else {},
                    }
                    for action in actions
                ] if len(actions) > 1 else [],
            ),
        )
        remembered = next(
            (
                action for action in reversed(actions)
                if action.confidence >= self._threshold
                and action.intent not in {"unknown", "general_chat", "confirm", "decline", "cancel"}
            ),
            None,
        )
        if remembered is not None:
            self._last_action = remembered
        self.bus.publish_event(output)
        logger.info(
            "NLU_RESULT trace=%s intent=%s raw=%s conf=%.3f slots=%s",
            event.trace_id,
            accepted_intent,
            raw_intent,
            result.confidence,
            output.payload["slots"],
        )

    async def _predict_action(self, text: str) -> RoutedAction:
        if self._predictor is None or not text:
            return RoutedAction("unknown", {}, 0.0)
        async with self.gpu_lock.section("nlu"):
            result = await asyncio.to_thread(self._predictor.predict, text)
        result = _apply_runtime_command_guardrails(text, result)
        result = _apply_reminder_guardrails(text, result)
        return RoutedAction(result.intent, dict(result.slots), result.confidence)
