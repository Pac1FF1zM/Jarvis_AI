"""Bounded JSONL telemetry for live Gesture Core sessions."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class GestureSessionLog:
    """Write privacy-safe gesture predictions and keep storage bounded."""

    def __init__(
        self,
        root: str | Path,
        *,
        retention_days: int = 30,
        max_total_bytes: int = 100 * 1024 * 1024,
        max_file_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.root = Path(root)
        self.retention_days = max(1, int(retention_days))
        self.max_total_bytes = max(1024, int(max_total_bytes))
        self.max_file_bytes = max(
            1024, min(int(max_file_bytes), self.max_total_bytes)
        )
        self._lock = threading.RLock()
        self._stream: Any = None
        self._path: Path | None = None
        self._part = 0
        self._metadata: dict[str, Any] = {}

    @property
    def path(self) -> Path | None:
        return self._path

    def start(self, **metadata: Any) -> Path:
        with self._lock:
            self.close()
            self.root.mkdir(parents=True, exist_ok=True)
            self._metadata = dict(metadata)
            self._part = 0
            self._cleanup_locked()
            self._open_part_locked()
            assert self._path is not None
            self.write("session_started", **self._metadata)
            return self._path

    def write(self, event: str, **fields: Any) -> None:
        with self._lock:
            if self._stream is None:
                return
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "event": str(event),
                **fields,
            }
            encoded = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), default=str
            )
            self._stream.write(encoded + "\n")
            self._stream.flush()
            if self._path is not None and self._path.stat().st_size >= self.max_file_bytes:
                self._stream.close()
                self._stream = None
                self._part += 1
                self._open_part_locked()
                self._cleanup_locked()

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
            self._stream = None
            self._path = None

    def cleanup(self) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._cleanup_locked()

    def _open_part_locked(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = f"_part{self._part:02d}" if self._part else ""
        self._path = self.root / f"gesture_session_{stamp}{suffix}.jsonl"
        self._stream = self._path.open("a", encoding="utf-8", newline="\n")

    def _cleanup_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        files = sorted(
            self.root.glob("gesture_session_*.jsonl"),
            key=lambda item: item.stat().st_mtime,
        )
        for path in list(files):
            if path == self._path:
                continue
            modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            if modified < cutoff:
                path.unlink(missing_ok=True)
                files.remove(path)
        total = sum(path.stat().st_size for path in files if path.exists())
        for path in files:
            if total <= self.max_total_bytes:
                break
            if path == self._path or not path.exists():
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size
