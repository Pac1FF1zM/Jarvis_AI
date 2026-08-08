"""Evidence-based latency analysis for the local Jarvis voice pipeline.

The module deliberately reads only developer session logs.  It does not store
audio or user transcripts in reports, which keeps acceptance artefacts safe to
share and commit.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


_LINE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\S+ \S+ \| (?P<message>.*)$"
)
_TRACE = re.compile(r"trace=(?P<trace>[0-9a-f]+)")
_DURATION = re.compile(r"duration_ms=(?P<duration>\d+)")
_SOURCE = re.compile(r"VOICE_ACTIVATED source=(?P<source>[a-z_]+)")


@dataclass
class _RawTurn:
    source: str = "unknown"
    timestamps: dict[str, float] = field(default_factory=dict)
    capture_duration_ms: float | None = None
    capture_end: str = "unknown"
    real_audio: bool = False


@dataclass(frozen=True)
class VoiceTurn:
    """An anonymised timing-only representation of one real voice turn."""

    source: str
    acknowledgement_ms: float
    speech_start_ms: float
    capture_ms: float
    stt_ms: float
    nlu_ms: float
    action_ms: float
    response_tts_ms: float
    command_ready_ms: float
    end_to_end_ms: float
    capture_end: str


@dataclass(frozen=True)
class Candidate:
    name: str
    description: str
    end_silence_saving_ms: float = 0.0
    max_capture_ms: float | None = None
    start_confirmation_cost_ms: float = 0.0
    acknowledgement_target_ms: float | None = None
    stt_factor: float = 1.0
    echo_risk: float = 0.0
    accuracy_loss: float = 0.0
    preserves_required_acknowledgement: bool = True


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    median_command_ready_ms: float
    p95_command_ready_ms: float
    median_end_to_end_ms: float
    p95_end_to_end_ms: float
    score: float
    eligible: bool


DEFAULT_CANDIDATES = (
    Candidate(
        "current",
        "Текущий последовательный pipeline: полный ответ, статический VAD, Whisper small.",
    ),
    Candidate(
        "balanced_vad",
        "Полный кэшированный ответ, подтверждение начала речи, 650 мс тишины и лимит 12 с.",
        end_silence_saving_ms=150.0,
        max_capture_ms=12_000.0,
        start_confirmation_cost_ms=32.0,
    ),
    Candidate(
        "short_ack",
        "Короткое подтверждение до записи; быстрее, но меняет согласованные реплики.",
        end_silence_saving_ms=150.0,
        max_capture_ms=12_000.0,
        start_confirmation_cost_ms=32.0,
        acknowledgement_target_ms=900.0,
        preserves_required_acknowledgement=False,
    ),
    Candidate(
        "duplex_overlap",
        "Запись одновременно с ответом; минимальная задержка, высокий риск эха без AEC.",
        end_silence_saving_ms=150.0,
        max_capture_ms=12_000.0,
        acknowledgement_target_ms=0.0,
        echo_risk=0.35,
    ),
    Candidate(
        "smaller_stt",
        "Более лёгкая STT-модель; быстрее, но ожидаемо теряет качество русских команд.",
        end_silence_saving_ms=150.0,
        max_capture_ms=12_000.0,
        start_confirmation_cost_ms=32.0,
        stt_factor=0.58,
        accuracy_loss=0.07,
    ),
)


def _millis(timestamp: str) -> float:
    return datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1000.0


def parse_session_logs(paths: Iterable[Path]) -> list[VoiceTurn]:
    """Extract timing signals without retaining transcript text."""
    raw: dict[str, _RawTurn] = {}
    pending_source = "unknown"
    marker_names = (
        ("PUBLISH wake_word_detected", "wake"),
        ("PUBLISH wake_acknowledgement_finished", "ack"),
        ("PUBLISH speech_capture_started", "capture_start"),
        ("AUDIO_CAPTURED real", "audio"),
        ("TRANSCRIPTION_READY", "transcription"),
        ("NLU_RESULT", "nlu"),
        ("PUBLISH response_ready", "response"),
        ("SPEECH_FINISHED", "speech_finished"),
    )
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _LINE.match(line)
            if not parsed:
                continue
            message = parsed.group("message")
            source_match = _SOURCE.search(message)
            if source_match:
                pending_source = source_match.group("source")
                continue
            trace_match = _TRACE.search(message)
            if not trace_match:
                continue
            trace = trace_match.group("trace")
            turn = raw.setdefault(trace, _RawTurn())
            when = _millis(parsed.group("timestamp"))
            for needle, name in marker_names:
                if needle in message:
                    turn.timestamps.setdefault(name, when)
                    if name == "wake":
                        turn.source = pending_source
                        pending_source = "unknown"
                    break
            if "AUDIO_CAPTURED real" in message:
                turn.real_audio = True
                duration = _DURATION.search(message)
                if duration:
                    turn.capture_duration_ms = float(duration.group("duration"))
                if "end=" in message:
                    turn.capture_end = message.rsplit("end=", 1)[1].split()[0]

    turns: list[VoiceTurn] = []
    required = {"wake", "audio", "transcription", "nlu", "response", "speech_finished"}
    for turn in raw.values():
        ts = turn.timestamps
        if not turn.real_audio or not required.issubset(ts):
            continue
        capture_start = ts.get("capture_start", ts["wake"])
        ack = max(0.0, ts.get("ack", ts["wake"]) - ts["wake"])
        speech_start = max(0.0, capture_start - ts.get("ack", ts["wake"]))
        capture = turn.capture_duration_ms or max(0.0, ts["audio"] - capture_start)
        turns.append(
            VoiceTurn(
                source=turn.source,
                acknowledgement_ms=ack,
                speech_start_ms=speech_start,
                capture_ms=capture,
                stt_ms=max(0.0, ts["transcription"] - ts["audio"]),
                nlu_ms=max(0.0, ts["nlu"] - ts["transcription"]),
                action_ms=max(0.0, ts["response"] - ts["nlu"]),
                response_tts_ms=max(0.0, ts["speech_finished"] - ts["response"]),
                command_ready_ms=max(0.0, ts["response"] - ts["wake"]),
                end_to_end_ms=max(0.0, ts["speech_finished"] - ts["wake"]),
                capture_end=turn.capture_end,
            )
        )
    return turns


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("at least one voice turn is required")
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _simulate(turn: VoiceTurn, candidate: Candidate) -> tuple[float, float]:
    observed_components = (
        turn.acknowledgement_ms
        + turn.speech_start_ms
        + turn.capture_ms
        + turn.stt_ms
        + turn.nlu_ms
        + turn.action_ms
    )
    # Preserve scheduler/dispatch overhead that is visible in the real trace
    # but not one of the explicit stages below. This also makes ``current`` an
    # exact replay of the measured baseline instead of an approximation.
    unmodelled_ms = turn.command_ready_ms - observed_components
    ack = turn.acknowledgement_ms
    if candidate.acknowledgement_target_ms is not None:
        ack = min(ack, candidate.acknowledgement_target_ms)
    capture = turn.capture_ms
    if candidate.max_capture_ms is not None:
        capture = min(capture, candidate.max_capture_ms)
    capture = max(0.0, capture - candidate.end_silence_saving_ms)
    capture += candidate.start_confirmation_cost_ms
    # Whisper latency is approximately proportional to the bounded audio
    # window on this machine. Preserve a small fixed model/dispatch floor.
    capture_ratio = capture / turn.capture_ms if turn.capture_ms else 1.0
    stt = max(350.0, turn.stt_ms * min(1.0, capture_ratio)) * candidate.stt_factor
    command_ready = (
        ack
        + turn.speech_start_ms
        + capture
        + stt
        + turn.nlu_ms
        + turn.action_ms
        + unmodelled_ms
    )
    end_to_end = command_ready + (turn.end_to_end_ms - turn.command_ready_ms)
    return command_ready, end_to_end


def evaluate_candidates(
    turns: Iterable[VoiceTurn],
    candidates: Iterable[Candidate] = DEFAULT_CANDIDATES,
) -> list[CandidateResult]:
    samples = list(turns)
    if not samples:
        raise ValueError("no complete real voice turns found")
    results: list[CandidateResult] = []
    for candidate in candidates:
        simulated = [_simulate(turn, candidate) for turn in samples]
        ready = [item[0] for item in simulated]
        total = [item[1] for item in simulated]
        eligible = (
            candidate.preserves_required_acknowledgement
            and candidate.echo_risk <= 0.10
            and candidate.accuracy_loss <= 0.03
        )
        score = (
            _percentile(ready, 0.95)
            + candidate.echo_risk * 20_000.0
            + candidate.accuracy_loss * 20_000.0
            + (0.0 if candidate.preserves_required_acknowledgement else 100_000.0)
        )
        results.append(
            CandidateResult(
                candidate,
                statistics.median(ready),
                _percentile(ready, 0.95),
                statistics.median(total),
                _percentile(total, 0.95),
                score,
                eligible,
            )
        )
    return sorted(results, key=lambda item: (not item.eligible, item.score))


def summarise_turns(turns: Iterable[VoiceTurn]) -> Mapping[str, float | int]:
    samples = list(turns)
    if not samples:
        raise ValueError("no complete real voice turns found")
    return {
        "turns": len(samples),
        "max_duration_captures": sum(turn.capture_end == "max_duration" for turn in samples),
        "median_capture_ms": round(statistics.median(turn.capture_ms for turn in samples), 2),
        "p95_capture_ms": round(_percentile([turn.capture_ms for turn in samples], 0.95), 2),
        "median_stt_ms": round(statistics.median(turn.stt_ms for turn in samples), 2),
        "p95_stt_ms": round(_percentile([turn.stt_ms for turn in samples], 0.95), 2),
        "median_command_ready_ms": round(statistics.median(turn.command_ready_ms for turn in samples), 2),
        "p95_command_ready_ms": round(_percentile([turn.command_ready_ms for turn in samples], 0.95), 2),
    }
