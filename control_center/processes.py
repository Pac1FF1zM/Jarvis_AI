"""Non-blocking subprocess boundaries used by the desktop UI."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from control_center.doctor_report import parse_doctor_output


class JarvisRuntimeProcess(QObject):
    output = Signal(str)
    state_changed = Signal(str)
    failed = Signal(str)

    def __init__(self, project_root: str | Path) -> None:
        super().__init__()
        self.root = Path(project_root).resolve()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.root))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        self.stop_file: Path | None = None
        self._scan_tail = ""
        self._ready_seen = False
        self._stop_requested = False

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(
        self,
        config_path: str | Path,
        *,
        gesture_preview_port: int | None = None,
        gesture_preview_token: str | None = None,
    ) -> None:
        if self.running:
            return
        control_dir = self.root / "logs" / "control-center"
        control_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file = control_dir / f"stop-{uuid.uuid4().hex}.request"
        self.stop_file.unlink(missing_ok=True)
        self._scan_tail = ""
        self._ready_seen = False
        self._stop_requested = False
        arguments = [
            str(self.root / "main.py"),
            "--config",
            str(Path(config_path).resolve()),
            "--stop-file",
            str(self.stop_file),
        ]
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        if gesture_preview_port and gesture_preview_token:
            environment.insert("JARVIS_GESTURE_PREVIEW_PORT", str(gesture_preview_port))
            environment.insert("JARVIS_GESTURE_PREVIEW_TOKEN", gesture_preview_token)
        self.process.setProcessEnvironment(environment)
        self.state_changed.emit("starting")
        self.process.start(sys.executable, arguments)

    def stop(self) -> None:
        if not self.running:
            return
        self._stop_requested = True
        self.state_changed.emit("stopping")
        if self.stop_file is not None:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop\n", encoding="utf-8")
        QTimer.singleShot(45_000, self._kill_if_still_running)

    def _kill_if_still_running(self) -> None:
        if self.running:
            self.output.emit("[CONTROL] Graceful stop timed out; terminating runtime.\n")
            self.process.kill()

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._scan_tail = (self._scan_tail + data)[-2000:]
            if not self._ready_seen and "JARVIS_RUNTIME_READY" in self._scan_tail:
                self._ready_seen = True
                if not self._stop_requested:
                    self.state_changed.emit("running")
            self.output.emit(data)

    def _finished(self, exit_code: int, _status: Any) -> None:
        if self.stop_file is not None:
            self.stop_file.unlink(missing_ok=True)
        self.state_changed.emit("stopped" if exit_code == 0 else "failed")

    def _error(self, error: Any) -> None:
        self.failed.emit(f"Не удалось запустить Jarvis: {error}")
        self.state_changed.emit("failed")


class DoctorProcess(QObject):
    completed = Signal(dict)
    output = Signal(str)
    failed = Signal(str)
    state_changed = Signal(bool)

    def __init__(self, project_root: str | Path) -> None:
        super().__init__()
        self.root = Path(project_root).resolve()
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.root))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(
            lambda error: self.failed.emit(f"Doctor не запустился: {error}")
        )
        self._buffer = ""

    @property
    def running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def start(self, config_path: str | Path) -> None:
        if self.running:
            return
        self._buffer = ""
        self.state_changed.emit(True)
        self.process.start(
            sys.executable,
            [
                str(self.root / "main.py"),
                "--doctor",
                "--json",
                "--config",
                str(Path(config_path).resolve()),
            ],
        )

    def _read(self) -> None:
        value = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._buffer += value
        self.output.emit(value)

    def _finished(self, _exit_code: int, _status: Any) -> None:
        self.state_changed.emit(False)
        try:
            self.completed.emit(parse_doctor_output(self._buffer))
        except ValueError as exc:
            self.failed.emit(str(exc))
