"""Frozen, independent audio -> STT -> JSC benchmark contracts."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.parakeet.benchmarks.compare_stt import error_counts
from ml.jsc.jal import DialogueAct, loads
from ml.jsc.structured_decoding import plan_completeness_issues


@dataclass(frozen=True)
class VoiceBenchmarkSample:
    sample_id: str
    audio_path: Path
    audio_sha256: str
    reference_text: str
    expected_jal: str
    speaker_id: str


VOICE_E2E_GATES: Mapping[str, float] = {
    "maximum_wer": 0.12,
    "minimum_oracle_semantic_exact": 0.95,
    "minimum_voice_semantic_exact": 0.90,
    "maximum_false_execution_rate": 0.0,
    "minimum_complete_execution_rate": 1.0,
}


def load_independent_manifest(
    path: Path, *, enforce_population: bool = True
) -> tuple[VoiceBenchmarkSample, ...]:
    """Load a frozen human-voice manifest and verify every audio hash."""
    rows: list[VoiceBenchmarkSample] = []
    ids: set[str] = set()
    for line_number, raw in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if row.get("provenance") != "independent_human" or row.get("consent") is not True:
            raise ValueError(f"line {line_number}: independent provenance and consent are required")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or sample_id in ids:
            raise ValueError(f"line {line_number}: id is missing or duplicated")
        ids.add(sample_id)
        audio = Path(str(row.get("path", "")))
        if not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        expected_hash = str(row.get("audio_sha256", "")).lower()
        actual_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            raise ValueError(f"line {line_number}: audio hash mismatch")
        reference = str(row.get("reference_text", "")).strip()
        expected_jal = str(row.get("expected_jal", "")).strip()
        speaker = str(row.get("speaker_id", "")).strip()
        if not reference or not expected_jal or not speaker:
            raise ValueError(f"line {line_number}: reference, expected_jal and speaker_id are required")
        loads(expected_jal)
        rows.append(
            VoiceBenchmarkSample(
                sample_id, audio, actual_hash, reference, expected_jal, speaker
            )
        )
    if not rows:
        raise ValueError("benchmark manifest is empty")
    if enforce_population and (len(rows) < 30 or len({row.speaker_id for row in rows}) < 3):
        raise ValueError("independent benchmark requires >=30 samples from >=3 speakers")
    return tuple(rows)


def evaluate_voice_e2e(
    samples: Sequence[VoiceBenchmarkSample],
    *,
    transcribe: Callable[[Path], tuple[str, float]],
    predict: Callable[[str], tuple[str, float]],
) -> dict[str, Any]:
    """Evaluate both oracle semantics and the real audio/STT semantic path."""
    details: list[dict[str, Any]] = []
    totals = {"word_errors": 0, "reference_words": 0, "char_errors": 0, "reference_chars": 0}
    for sample in samples:
        hypothesis, stt_ms = transcribe(sample.audio_path)
        oracle_jal, oracle_ms = predict(sample.reference_text)
        voice_jal, semantic_ms = predict(hypothesis)
        counts = error_counts(sample.reference_text, hypothesis)
        for name in totals:
            totals[name] += counts[name]
        expected = loads(sample.expected_jal)
        oracle = loads(oracle_jal)
        voice = loads(voice_jal)
        issues = plan_completeness_issues(hypothesis, voice)
        false_execution = expected.act != DialogueAct.EXECUTE and voice.act == DialogueAct.EXECUTE
        details.append(
            {
                "id": sample.sample_id,
                "speaker_id": sample.speaker_id,
                "reference_text": sample.reference_text,
                "hypothesis": hypothesis,
                "expected_jal": sample.expected_jal,
                "oracle_jal": oracle_jal,
                "voice_jal": voice_jal,
                "oracle_semantic_exact": oracle == expected,
                "voice_semantic_exact": voice == expected,
                "false_execution": false_execution,
                "completeness_issues": list(issues),
                "stt_ms": stt_ms,
                "oracle_semantic_ms": oracle_ms,
                "voice_semantic_ms": semantic_ms,
                **counts,
            }
        )
    count = len(details)
    execution_rows = [row for row in details if loads(row["expected_jal"]).act == DialogueAct.EXECUTE]
    metrics = {
        "samples": count,
        "speakers": len({sample.speaker_id for sample in samples}),
        "wer": totals["word_errors"] / max(totals["reference_words"], 1),
        "cer": totals["char_errors"] / max(totals["reference_chars"], 1),
        "oracle_semantic_exact": sum(row["oracle_semantic_exact"] for row in details) / count,
        "voice_semantic_exact": sum(row["voice_semantic_exact"] for row in details) / count,
        "false_execution_rate": sum(row["false_execution"] for row in details) / count,
        "complete_execution_rate": (
            sum(not row["completeness_issues"] for row in execution_rows) / len(execution_rows)
            if execution_rows else 1.0
        ),
        "p95_end_to_end_ms": _percentile(
            [row["stt_ms"] + row["voice_semantic_ms"] for row in details], 0.95
        ),
    }
    checks = {
        "maximum_wer": metrics["wer"] <= VOICE_E2E_GATES["maximum_wer"],
        "minimum_oracle_semantic_exact": metrics["oracle_semantic_exact"] >= VOICE_E2E_GATES["minimum_oracle_semantic_exact"],
        "minimum_voice_semantic_exact": metrics["voice_semantic_exact"] >= VOICE_E2E_GATES["minimum_voice_semantic_exact"],
        "maximum_false_execution_rate": metrics["false_execution_rate"] <= VOICE_E2E_GATES["maximum_false_execution_rate"],
        "minimum_complete_execution_rate": metrics["complete_execution_rate"] >= VOICE_E2E_GATES["minimum_complete_execution_rate"],
    }
    return {
        "schema_version": 1,
        "mode": "independent_human_voice_no_action",
        "execution": "blocked",
        "metrics": metrics,
        "gates": {"passed": all(checks.values()), "checks": checks, "targets": dict(VOICE_E2E_GATES)},
        "results": details,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else 0.0
