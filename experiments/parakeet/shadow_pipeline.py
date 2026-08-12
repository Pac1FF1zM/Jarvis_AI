"""No-action Parakeet -> production NLU diagnostic pipeline.

This module has no publisher, LLM, tool registry, reminder service, or
executor. It returns semantic frames as data and cannot execute them.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.nlu.inference import NLUPredictor
from modules.semantic_commit import prepare_final_utterance
from modules.command_router import RoutedAction, route_explicit_command, split_compound_command
from modules.nlu import (
    _apply_reminder_guardrails,
    _apply_runtime_command_guardrails,
    _normalise_transcription_for_nlu,
)
from tools._applications import resolve_application
from modules.parakeet_client import (
    DEFAULT_MODEL_DIR as PARAKEET_MODEL_DIR,
    PersistentParakeetClient,
)


@dataclass(frozen=True)
class ShadowAction:
    raw_intent: str
    intent: str
    slots: dict[str, Any]
    confidence: float
    execution: str = "blocked"


class ShadowNLU:
    """Production-equivalent prediction/routing with no publisher or executor."""

    def __init__(self, checkpoint: Path, *, threshold: float = 0.55) -> None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"NLU checkpoint not found: {checkpoint}")
        self.predictor = NLUPredictor(checkpoint, "cpu")
        self.threshold = threshold

    def predict(self, text: str) -> dict[str, Any]:
        gate = prepare_final_utterance(text)
        normalized = _normalise_transcription_for_nlu(gate.route_text or text)
        if gate.state != "analyze":
            return {
                "input_text": text,
                "normalized_text": normalized,
                "threshold": self.threshold,
                "commit_state": gate.state,
                "commit_reason": gate.reason,
                "actions": [],
                "execution": "blocked",
            }
        actions: list[ShadowAction] = []
        previous: RoutedAction | None = None
        for part in split_compound_command(normalized):
            routed = route_explicit_command(part, previous_action=previous)
            if routed is None:
                result = self.predictor.predict(part)
                result = _apply_runtime_command_guardrails(part, result)
                result = _apply_reminder_guardrails(part, result)
                routed = RoutedAction(result.intent, dict(result.slots), result.confidence)
            accepted = routed.intent if routed.confidence >= self.threshold else "unknown"
            slots = dict(routed.slots) if accepted != "unknown" else {}
            if accepted == "open_application":
                application = resolve_application(str(slots.get("application", "")))
                if application is None:
                    accepted, slots = "unknown", {}
                else:
                    slots["application"] = application.name
            actions.append(
                ShadowAction(
                    raw_intent=routed.intent,
                    intent=accepted,
                    slots=slots,
                    confidence=float(routed.confidence),
                )
            )
            if accepted not in {"unknown", "general_chat", "confirm", "decline", "cancel"}:
                previous = routed
        unresolved = any(action.intent == "unknown" for action in actions)
        return {
            "input_text": text,
            "normalized_text": normalized,
            "threshold": self.threshold,
            "commit_state": "clarify" if unresolved else "ready",
            "commit_reason": "unresolved_action" if unresolved else "all_actions_resolved",
            "actions": [] if unresolved else [asdict(action) for action in actions],
            "candidate_actions": [asdict(action) for action in actions] if unresolved else [],
            "execution": "blocked",
        }


def decode_in_child(
    wav: Path,
    *,
    repository_root: Path,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    jarvis_python = repository_root / "venv/Scripts/python.exe"
    model_dir = repository_root / PARAKEET_MODEL_DIR
    if not jarvis_python.is_file():
        raise FileNotFoundError(f"Jarvis interpreter not found: {jarvis_python}")
    command = [
        str(jarvis_python),
        "-m",
        "experiments.parakeet.worker.decode_file",
        "--wav",
        str(wav.resolve()),
        "--model-dir",
        str(model_dir.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Parakeet child exceeded {timeout_seconds:.1f}s and was terminated") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Parakeet child failed with code {completed.returncode}: {detail}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Parakeet child returned no JSON")
    return json.loads(lines[-1])


PersistentParakeetDecoder = PersistentParakeetClient
