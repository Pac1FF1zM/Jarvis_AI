"""Typed, immutable-at-publication payload contracts for Jarvis events.

The event bus still exposes payloads through the familiar Mapping interface,
so consumers can migrate independently.  Producers construct these frozen
dataclasses: missing arguments, misspelled names and invalid critical values
then fail at the producer boundary instead of becoming a distant fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, ClassVar, Iterator, Mapping


class EventPayload(Mapping[str, Any]):
    """Base class for a payload bound to exactly one event type."""

    event_type: ClassVar[str]

    def __getitem__(self, key: str) -> Any:
        if key not in {item.name for item in fields(self)}:
            raise KeyError(key)
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self) if getattr(self, item.name) is not None)

    def __len__(self) -> int:
        return sum(getattr(self, item.name) is not None for item in fields(self))

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType({key: _plain_copy(self[key]) for key in self})


def _plain_copy(value: Any) -> Any:
    """Detach and recursively freeze producer-owned containers."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _plain_copy(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_plain_copy(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_plain_copy(item) for item in value)
    return value


def freeze_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an outer-immutable, detached mapping for queue ownership."""
    return MappingProxyType({str(key): _plain_copy(value) for key, value in payload.items()})


def _require_text(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _require_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class WakeWordDetectedPayload(EventPayload):
    event_type: ClassVar[str] = "wake_word_detected"
    source: str = "unknown"

    def __post_init__(self) -> None:
        _require_text("source", self.source)


@dataclass(frozen=True)
class SpeechCaptureStartedPayload(EventPayload):
    event_type: ClassVar[str] = "speech_capture_started"
    source: str = "microphone"


@dataclass(frozen=True)
class AudioCapturedPayload(EventPayload):
    event_type: ClassVar[str] = "audio_captured"
    audio: bytes
    sample_rate: int | None = None
    channels: int | None = None
    sample_width: int | None = None
    duration_ms: int | None = None
    source: str | None = None
    capture_end: str | None = None
    voice_calibrated: bool | None = None
    input_device_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes):
            raise TypeError("audio must be bytes")
        for name in ("sample_rate", "channels", "sample_width"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")


@dataclass(frozen=True)
class TranscriptionReadyPayload(EventPayload):
    event_type: ClassVar[str] = "transcription_ready"
    text: str
    confidence: float = 0.0
    source: str | None = None

    def __post_init__(self) -> None:
        _require_text("text", self.text, allow_empty=True)
        object.__setattr__(self, "confidence", _require_probability("confidence", self.confidence))


@dataclass(frozen=True)
class ThinkingReadyPayload(EventPayload):
    event_type: ClassVar[str] = "thinking_ready"


@dataclass(frozen=True)
class InvalidTransitionPayload(EventPayload):
    event_type: ClassVar[str] = "invalid_transition"
    current_state: str
    attempted_target: str


@dataclass(frozen=True)
class InteractionCancelledPayload(EventPayload):
    event_type: ClassVar[str] = "interaction_cancelled"
    reason: str
    cancelled_state: str | None = None
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        _require_text("reason", self.reason)


@dataclass(frozen=True)
class InteractionFailedPayload(EventPayload):
    event_type: ClassVar[str] = "interaction_failed"
    reason: str
    state: str | None = None
    reminder_id: int | None = None
    source_event: str | None = None
    handler: str | None = None
    error_type: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_text("reason", self.reason)


@dataclass(frozen=True)
class InteractionCompletedPayload(EventPayload):
    event_type: ClassVar[str] = "interaction_completed"
    state: str
    ok: bool
    cancelled: bool | None = None
    reason: str | None = None
    cancelled_state: str | None = None
    superseded_by: str | None = None
    failed_state: str | None = None
    sleep_mode: bool | None = None

    def __post_init__(self) -> None:
        _require_text("state", self.state)
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")


@dataclass(frozen=True)
class NLUResultPayload(EventPayload):
    event_type: ClassVar[str] = "nlu_result"
    text: str
    intent: str
    slots: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_intent: str | None = None
    intent_confidence: float = 0.0
    actions: list[Mapping[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_text("text", self.text, allow_empty=True)
        _require_text("intent", self.intent)
        if self.raw_intent is not None:
            _require_text("raw_intent", self.raw_intent)
        object.__setattr__(self, "confidence", _require_probability("confidence", self.confidence))
        object.__setattr__(
            self,
            "intent_confidence",
            _require_probability("intent_confidence", self.intent_confidence),
        )
        if not isinstance(self.slots, Mapping):
            raise TypeError("slots must be a mapping")
        if not isinstance(self.actions, list) or not all(isinstance(item, Mapping) for item in self.actions):
            raise TypeError("actions must be a list of mappings")


@dataclass(frozen=True)
class ToolCallRequestedPayload(EventPayload):
    event_type: ClassVar[str] = "tool_call_requested"
    tool: str
    params: Mapping[str, Any] | None = None
    plan: list[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        _require_text("tool", self.tool)
        if (self.params is None) == (self.plan is None):
            raise ValueError("tool_call_requested requires exactly one of params or plan")
        if self.params is not None and not isinstance(self.params, Mapping):
            raise TypeError("params must be a mapping")
        if self.plan is not None and (
            not isinstance(self.plan, list) or not all(isinstance(item, Mapping) for item in self.plan)
        ):
            raise TypeError("plan must be a list of mappings")


@dataclass(frozen=True)
class ToolResultPayload(EventPayload):
    event_type: ClassVar[str] = "tool_result"
    tool: str
    result: Mapping[str, Any]
    direct_response: bool = False

    def __post_init__(self) -> None:
        _require_text("tool", self.tool)
        if not isinstance(self.result, Mapping):
            raise TypeError("result must be a mapping")
        if not isinstance(self.direct_response, bool):
            raise TypeError("direct_response must be bool")


@dataclass(frozen=True)
class ResponseReadyPayload(EventPayload):
    event_type: ClassVar[str] = "response_ready"
    text: str = ""

    def __post_init__(self) -> None:
        _require_text("text", self.text, allow_empty=True)


@dataclass(frozen=True)
class SpeechStartedPayload(EventPayload):
    event_type: ClassVar[str] = "speech_started"
    text: str = ""

    def __post_init__(self) -> None:
        _require_text("text", self.text, allow_empty=True)


@dataclass(frozen=True)
class SpeechFinishedPayload(EventPayload):
    event_type: ClassVar[str] = "speech_finished"
    text: str = ""

    def __post_init__(self) -> None:
        _require_text("text", self.text, allow_empty=True)


@dataclass(frozen=True)
class CancelRequestedPayload(EventPayload):
    event_type: ClassVar[str] = "cancel_requested"
    reason: str
    target_trace_id: str | None = None
    text: str | None = None
    intent_confidence: float | None = None

    def __post_init__(self) -> None:
        _require_text("reason", self.reason)
        if self.intent_confidence is not None:
            object.__setattr__(
                self,
                "intent_confidence",
                _require_probability("intent_confidence", self.intent_confidence),
            )


@dataclass(frozen=True)
class SessionSleepRequestedPayload(EventPayload):
    event_type: ClassVar[str] = "session_sleep_requested"
    text: str
    reason: str

    def __post_init__(self) -> None:
        _require_text("text", self.text)
        _require_text("reason", self.reason)


@dataclass(frozen=True)
class ReminderCancelledPayload(EventPayload):
    event_type: ClassVar[str] = "reminder_cancelled"
    reminder_id: int

    def __post_init__(self) -> None:
        if isinstance(self.reminder_id, bool) or self.reminder_id <= 0:
            raise ValueError("reminder_id must be a positive integer")


@dataclass(frozen=True)
class NotificationPayload(EventPayload):
    """Shared shape for ready/authorized/deliver; bind it explicitly with ``for_event``."""

    event_type: ClassVar[str] = "notification_ready"
    reminder_id: int
    text: str
    message: str
    due_at: str
    source: str = "reminder"

    def __post_init__(self) -> None:
        if isinstance(self.reminder_id, bool) or self.reminder_id <= 0:
            raise ValueError("reminder_id must be a positive integer")
        _require_text("text", self.text)
        _require_text("message", self.message)
        _require_text("due_at", self.due_at)
        _require_text("source", self.source)


@dataclass(frozen=True)
class NotificationAuthorizedPayload(NotificationPayload):
    event_type: ClassVar[str] = "notification_authorized"


@dataclass(frozen=True)
class NotificationDeliverPayload(NotificationPayload):
    event_type: ClassVar[str] = "notification_deliver"


@dataclass(frozen=True)
class GestureModeRequestedPayload(EventPayload):
    event_type: ClassVar[str] = "gesture_mode_requested"
    enabled: bool | None = None
    action: str | None = None
    source: str = "event"

    def __post_init__(self) -> None:
        if self.enabled is not None and not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool or None")
        if self.action is not None and self.action not in {
            "enable", "disable", "pause", "resume", "status", "toggle"
        }:
            raise ValueError("unsupported gesture mode action")
        if self.enabled is None and self.action is None:
            raise ValueError("gesture mode request needs enabled or action")
        if self.enabled is not None and self.action not in {None, "enable", "disable"}:
            raise ValueError("enabled is only compatible with enable/disable")
        if self.enabled is True and self.action == "disable":
            raise ValueError("enabled=True contradicts action=disable")
        if self.enabled is False and self.action == "enable":
            raise ValueError("enabled=False contradicts action=enable")
        _require_text("source", self.source)


@dataclass(frozen=True)
class GestureModeChangedPayload(EventPayload):
    event_type: ClassVar[str] = "gesture_mode_changed"
    armed: bool
    source: str
    action: str | None = None
    paused: bool = False
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.armed, bool):
            raise TypeError("armed must be bool")
        if not isinstance(self.paused, bool):
            raise TypeError("paused must be bool")
        if self.action is not None and self.action not in {
            "enable", "disable", "pause", "resume", "status", "toggle"
        }:
            raise ValueError("unsupported gesture mode action")
        _require_text("source", self.source)


@dataclass(frozen=True)
class GestureActionReadyPayload(EventPayload):
    event_type: ClassVar[str] = "gesture_action_ready"
    label: str
    action_hint: str
    confidence: float
    consecutive_windows: int
    execution: str

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_text("action_hint", self.action_hint)
        _require_text("execution", self.execution)
        object.__setattr__(self, "confidence", _require_probability("confidence", self.confidence))
        if self.consecutive_windows < 2:
            raise ValueError("consecutive_windows must be >= 2")


@dataclass(frozen=True)
class GestureRuntimeStatusPayload(EventPayload):
    event_type: ClassVar[str] = "gesture_runtime_status"
    status: str
    detail: str = ""

    def __post_init__(self) -> None:
        _require_text("status", self.status)
        _require_text("detail", self.detail, allow_empty=True)


PAYLOAD_CONTRACTS: Mapping[str, type[EventPayload]] = MappingProxyType(
    {
        contract.event_type: contract
        for contract in (
            WakeWordDetectedPayload,
            SpeechCaptureStartedPayload,
            AudioCapturedPayload,
            TranscriptionReadyPayload,
            ThinkingReadyPayload,
            InvalidTransitionPayload,
            InteractionCancelledPayload,
            InteractionFailedPayload,
            InteractionCompletedPayload,
            NLUResultPayload,
            ToolCallRequestedPayload,
            ToolResultPayload,
            ResponseReadyPayload,
            SpeechStartedPayload,
            SpeechFinishedPayload,
            CancelRequestedPayload,
            SessionSleepRequestedPayload,
            ReminderCancelledPayload,
            NotificationPayload,
            NotificationAuthorizedPayload,
            NotificationDeliverPayload,
            GestureModeRequestedPayload,
            GestureModeChangedPayload,
            GestureActionReadyPayload,
            GestureRuntimeStatusPayload,
        )
    }
)


def coerce_event_payload(
    event_type: str,
    payload: Mapping[str, Any] | EventPayload,
) -> Mapping[str, Any]:
    """Validate known events and freeze both typed and legacy producers.

    Unknown event names remain extensible but still receive immutable snapshots.
    Known names reject extra keys, wrong types and missing required fields.
    """
    if isinstance(payload, EventPayload):
        if payload.event_type != event_type:
            raise ValueError(
                f"payload contract {type(payload).__name__} belongs to "
                f"{payload.event_type!r}, not {event_type!r}"
            )
        return payload.to_mapping()
    if not isinstance(payload, Mapping):
        raise TypeError("event payload must be an EventPayload or Mapping")
    contract = PAYLOAD_CONTRACTS.get(event_type)
    if contract is None:
        return freeze_payload(payload)
    try:
        typed = contract(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {event_type} payload: {exc}") from exc
    return typed.to_mapping()
