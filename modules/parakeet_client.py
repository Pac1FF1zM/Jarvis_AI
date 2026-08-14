"""Isolated persistent client for the production Parakeet STT worker."""
from __future__ import annotations

import base64
from collections import deque
import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path(
    ".local/parakeet/models/nvidia--parakeet-tdt-0.6b-v3"
)


def _readline_with_timeout(stream: Any, timeout_seconds: float) -> str:
    result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            result.put(stream.readline())
        except BaseException as exc:
            result.put(exc)

    threading.Thread(target=read, daemon=True).start()
    try:
        value = result.get(timeout=timeout_seconds)
    except queue.Empty as exc:
        raise TimeoutError(
            f"Parakeet worker did not respond within {timeout_seconds:.1f}s"
        ) from exc
    if isinstance(value, BaseException):
        raise value
    return value


class PersistentParakeetClient:
    """Keep the model warm in an isolated interpreter and exchange WAV bytes."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        python_executable: str | Path = "venv/Scripts/python.exe",
        provider: str = "cuda",
        timeout_seconds: float = 45.0,
    ) -> None:
        root = Path(repository_root).resolve()
        model_path = Path(model_dir)
        python_path = Path(python_executable)
        self.repository_root = root
        self.model_dir = (
            model_path.resolve()
            if model_path.is_absolute()
            else (root / model_path).resolve()
        )
        self.python_executable = (
            python_path.resolve()
            if python_path.is_absolute()
            else (root / python_path).resolve()
        )
        self.provider = provider.casefold()
        self.timeout_seconds = float(timeout_seconds)
        self.process: subprocess.Popen[str] | None = None
        self.startup: dict[str, Any] | None = None
        self._stderr_tail: deque[str] = deque(maxlen=80)
        self._stderr_thread: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        if self.process is not None:
            return dict(self.startup or {})
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"Parakeet interpreter not found: {self.python_executable}"
            )
        if self.provider not in {"cuda", "cpu"}:
            raise ValueError("Parakeet provider must be 'cuda' or 'cpu'")
        command = [
            str(self.python_executable),
            "-m",
            "experiments.parakeet.worker.serve",
            "--model-dir",
            str(self.model_dir),
            "--provider",
            self.provider,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Control Center is a windowed application.  Without this flag
            # Windows creates a separate console for the long-lived Parakeet
            # interpreter; closing that console kills STT during startup.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            text=True,
            encoding="utf-8",
            bufsize=1,
            shell=False,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self.process,),
            daemon=True,
            name="parakeet-stderr",
        )
        self._stderr_thread.start()
        try:
            self.startup = self._receive()
            if self.startup.get("event") != "ready":
                raise RuntimeError(
                    f"unexpected Parakeet startup response: {self.startup}"
                )
            return dict(self.startup)
        except Exception:
            self.close(force=True)
            raise

    def decode(self, wav_bytes: bytes) -> dict[str, Any]:
        if not wav_bytes:
            raise ValueError("captured audio is empty")
        if self.process is None:
            self.start()
        self._send(
            {
                "op": "decode",
                "audio_b64": base64.b64encode(wav_bytes).decode("ascii"),
            }
        )
        try:
            response = self._receive()
        except TimeoutError:
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
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
            detail = "\n".join(self._stderr_tail).strip()
            raise RuntimeError(
                f"Parakeet worker exited without a response: {detail}"
            )
        return json.loads(line)

    def _drain_stderr(self, process: subprocess.Popen[str]) -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            for line in stream:
                line = line.strip()
                if line:
                    self._stderr_tail.append(line)
        except (OSError, ValueError):
            # Shutdown closes the stream to unblock this daemon reader.
            return

    def close(self, *, force: bool = False) -> None:
        process, self.process = self.process, None
        self.startup = None
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
        stderr_thread, self._stderr_thread = self._stderr_thread, None
        if (
            stderr_thread is not None
            and stderr_thread is not threading.current_thread()
        ):
            stderr_thread.join(timeout=1.0)

    def __enter__(self) -> "PersistentParakeetClient":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
