"""No-action Parakeet -> production NLU diagnostic pipeline.

This module has no publisher, LLM, tool registry, reminder service, or
executor. It returns semantic frames as data and cannot execute them.
"""
from __future__ import annotations

import json
import base64
import queue
import subprocess
import threading
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


PARAKEET_MODEL_DIR = Path(".local/parakeet/models/nvidia--parakeet-tdt-0.6b-v3")


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


def _readline_with_timeout(stream: Any, timeout_seconds: float) -> str:
    result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(stream.readline())
        except BaseException as exc:  # Preserve the worker I/O error for the caller.
            result.put(exc)

    threading.Thread(target=read, daemon=True).start()
    try:
        value = result.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(f"Parakeet worker did not respond within {timeout_seconds:.1f}s") from exc
    if isinstance(value, BaseException):
        raise value
    return value


class PersistentParakeetDecoder:
    """Keep the isolated model warm while exchanging in-memory WAV payloads."""

    def __init__(
        self,
        *,
        repository_root: Path,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self.startup: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        jarvis_python = self.repository_root / "venv/Scripts/python.exe"
        model_dir = self.repository_root / PARAKEET_MODEL_DIR
        if not jarvis_python.is_file():
            raise FileNotFoundError(f"Jarvis interpreter not found: {jarvis_python}")
        command = [
            str(jarvis_python),
            "-m",
            "experiments.parakeet.worker.serve",
            "--model-dir",
            str(model_dir),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            shell=False,
        )
        try:
            self.startup = self._receive()
            if self.startup.get("event") != "ready":
                raise RuntimeError(f"unexpected Parakeet startup response: {self.startup}")
            return self.startup
        except Exception:
            self.close(force=True)
            raise

    def decode(self, wav_bytes: bytes) -> dict[str, Any]:
        if not wav_bytes:
            raise ValueError("captured audio is empty")
        if self.process is None:
            self.start()
        self._send({"op": "decode", "audio_b64": base64.b64encode(wav_bytes).decode("ascii")})
        try:
            response = self._receive()
        except TimeoutError:
            # A timed-out native/CUDA generation cannot be safely interrupted
            # in-process. Kill the isolated worker, discard this capture, and
            # lazily start a clean worker for the next phrase.
            self.close(force=True)
            raise
        if response.get("event") == "error":
            raise RuntimeError(str(response.get("error", "Parakeet worker error")))
        if response.get("event") != "result":
            raise RuntimeError(f"unexpected Parakeet response: {response}")
        return response

    def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("Parakeet worker is not running")
        process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        process.stdin.flush()

    def _receive(self) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("Parakeet worker is not running")
        line = _readline_with_timeout(process.stdout, self.timeout_seconds)
        if not line:
            # stdout may reach EOF just before poll() observes process exit.
            # Wait briefly so the real worker traceback is never hidden by
            # that race, then drain stderr once the pipe is closed.
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            detail = process.stderr.read().strip() if process.stderr is not None else ""
            raise RuntimeError(f"Parakeet worker exited without a response: {detail}")
        return json.loads(line)

    def close(self, *, force: bool = False) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if not force and process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write('{"op":"close"}\n')
                process.stdin.flush()
                process.wait(timeout=3.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                force = True
        if force and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()

    def __enter__(self) -> "PersistentParakeetDecoder":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
