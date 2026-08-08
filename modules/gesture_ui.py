"""Modern optional PySide6 preview for the live gesture classifier."""
from __future__ import annotations

import json
import os
import socket
import struct
import time
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


_DATAGRAM_LIMIT = 60_000
_EMBEDDED_FPS = 12.0


@dataclass(frozen=True)
class GesturePreviewState:
    status: str
    label: str
    action: str
    confidence: float
    top3: tuple[tuple[str, float], ...]
    stable_count: int
    stable_required: int
    last_action: str
    fps: float
    latency_ms: float
    model: str
    camera: str


class ModernGesturePreview:
    """Own a PySide6 window in the camera worker, with an OpenCV fallback."""

    def __init__(self, title: str = "Jarvis Gesture Core") -> None:
        self.title = title
        self.backend = "unopened"
        self._app: Any = None
        self._window: Any = None
        self._cv2: Any = None
        self._closed = False

    def open(self, cv2: Any) -> str:
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
            from PySide6.QtWidgets import QApplication, QFrame, QLabel, QWidget

            app = QApplication.instance() or QApplication([])

            class GestureWindow(QWidget):
                def __init__(self, title: str) -> None:
                    super().__init__()
                    self.user_closed = False
                    self.setWindowTitle(title)
                    self.resize(1100, 720)
                    self.setMinimumSize(760, 500)
                    self.setStyleSheet("background:#080b12;")
                    self.video = QLabel(self)
                    self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.video.setStyleSheet("background:#080b12;")
                    self.panel = QFrame(self)
                    self.panel.setFixedSize(440, 330)
                    self.panel.setStyleSheet(
                        "QFrame{background:rgba(12,17,27,225);border:1px solid "
                        "rgba(101,118,255,150);border-radius:18px;}"
                    )
                    self.heading = QLabel("GESTURE CORE", self.panel)
                    self.heading.setGeometry(22, 16, 396, 30)
                    self.heading.setFont(QFont("Segoe UI Semibold", 14))
                    self.heading.setStyleSheet(
                        "color:#8fa1ff;background:transparent;border:none;border-radius:0;"
                    )
                    self.details = QLabel(self.panel)
                    self.details.setGeometry(22, 52, 396, 258)
                    self.details.setFont(QFont("Segoe UI", 11))
                    self.details.setWordWrap(True)
                    self.details.setAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
                    )
                    self.details.setStyleSheet(
                        "color:#eef2ff;background:transparent;border:none;border-radius:0;"
                    )

                def resizeEvent(self, event: Any) -> None:
                    self.video.setGeometry(self.rect())
                    margin = 24
                    self.panel.move(
                        max(margin, self.width() - self.panel.width() - margin),
                        max(margin, self.height() - self.panel.height() - margin),
                    )
                    super().resizeEvent(event)

                def closeEvent(self, event: Any) -> None:
                    self.user_closed = True
                    event.accept()

                def update_view(self, frame: Any, state: GesturePreviewState) -> None:
                    height, width, channels = frame.shape
                    image = QImage(
                        frame.data,
                        width,
                        height,
                        channels * width,
                        QImage.Format.Format_BGR888,
                    ).copy()
                    pixmap = QPixmap.fromImage(image).scaled(
                        self.video.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    canvas = QPixmap(self.video.size())
                    canvas.fill(QColor("#080b12"))
                    painter = QPainter(canvas)
                    painter.drawPixmap(
                        (canvas.width() - pixmap.width()) // 2,
                        (canvas.height() - pixmap.height()) // 2,
                        pixmap,
                    )
                    painter.end()
                    self.video.setPixmap(canvas)
                    ranking = " · ".join(
                        f"{label} {confidence:.0%}"
                        for label, confidence in state.top3
                    ) or "ожидание"
                    self.details.setText(
                        f"Состояние   {state.status}\n"
                        f"Жест        {state.label}  ·  {state.confidence:.1%}\n"
                        f"Действие    {state.action}\n"
                        f"Стабильность {state.stable_count}/{state.stable_required}\n\n"
                        f"TOP-3       {ranking}\n"
                        f"Последнее   {state.last_action or '—'}\n"
                        f"Скорость    {state.fps:.1f} FPS  ·  {state.latency_ms:.1f} ms\n"
                        f"Модель      {state.model}\n"
                        f"Камера      {state.camera}"
                    )

            self._app = app
            self._window = GestureWindow(self.title)
            self._window.show()
            self._app.processEvents()
            self.backend = "pyside6"
            return self.backend
        except (ImportError, OSError, RuntimeError):
            self._cv2 = cv2
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.title, 1100, 720)
            self.backend = "opencv"
            return self.backend

    def render(self, frame: Any, state: GesturePreviewState) -> bool:
        if self._closed:
            return False
        if self.backend == "pyside6":
            self._window.update_view(frame, state)
            self._app.processEvents()
            if self._window.user_closed or not self._window.isVisible():
                self._closed = True
                return False
            return True
        if self.backend == "opencv":
            rendered = frame.copy()
            height, width = rendered.shape[:2]
            panel_width, panel_height = min(500, width - 24), min(250, height - 24)
            left, top = width - panel_width - 12, height - panel_height - 12
            overlay = rendered.copy()
            self._cv2.rectangle(
                overlay, (left, top), (width - 12, height - 12), (18, 18, 28), -1
            )
            self._cv2.addWeighted(overlay, 0.84, rendered, 0.16, 0, rendered)
            lines = (
                f"GESTURE CORE  |  {state.status}",
                f"{state.label}  {state.confidence:.1%}  ->  {state.action}",
                f"Stable {state.stable_count}/{state.stable_required}",
                "Top-3: " + " | ".join(f"{x} {p:.0%}" for x, p in state.top3),
                f"Last: {state.last_action or '-'}",
                f"{state.fps:.1f} FPS | {state.latency_ms:.1f} ms | {state.model}",
            )
            for index, line in enumerate(lines):
                self._cv2.putText(
                    rendered,
                    line,
                    (left + 18, top + 34 + index * 34),
                    self._cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (238, 242, 255),
                    1,
                    self._cv2.LINE_AA,
                )
            self._cv2.imshow(self.title, rendered)
            key = self._cv2.waitKey(1) & 0xFF
            try:
                visible = self._cv2.getWindowProperty(
                    self.title, self._cv2.WND_PROP_VISIBLE
                ) >= 1.0
            except Exception:
                visible = True
            if key in {ord("q"), 27} or not visible:
                self._closed = True
                return False
            return True
        return False

    def close(self) -> None:
        if self.backend == "pyside6" and self._window is not None:
            self._window.close()
            self._app.processEvents()
        elif self.backend == "opencv" and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.title)
                self._cv2.waitKey(1)
            except Exception:
                pass
        self._closed = True


def decode_gesture_datagram(packet: bytes) -> tuple[dict[str, Any], bytes]:
    """Decode one authenticated metadata + JPEG preview datagram."""
    if len(packet) < 4:
        raise ValueError("gesture preview datagram is truncated")
    metadata_size = struct.unpack("!I", packet[:4])[0]
    metadata_end = 4 + metadata_size
    if metadata_size <= 0 or metadata_end > len(packet):
        raise ValueError("gesture preview metadata size is invalid")
    metadata = json.loads(packet[4:metadata_end].decode("utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("gesture preview metadata must be an object")
    return metadata, packet[metadata_end:]


class EmbeddedGesturePreview:
    """Stream a rate-limited preview to the local Control Center over UDP."""

    def __init__(self, port: int, token: str) -> None:
        self.port = int(port)
        self.token = token
        self.backend = "unopened"
        self._socket: socket.socket | None = None
        self._cv2: Any = None
        self._last_sent_at = 0.0
        self._closed = False

    def open(self, cv2: Any) -> str:
        self._cv2 = cv2
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.backend = "control_center"
        self._send({"event": "opened", "token": self.token}, b"")
        return self.backend

    def render(self, frame: Any, state: GesturePreviewState) -> bool:
        if self._closed or self._socket is None:
            return False
        now = time.monotonic()
        if now - self._last_sent_at < 1.0 / _EMBEDDED_FPS:
            return True
        self._last_sent_at = now
        try:
            preview = self._resize_for_transport(frame)
            jpeg = self._encode_jpeg(preview)
            metadata = asdict(state)
            metadata.update({"event": "frame", "token": self.token})
            self._send(metadata, jpeg)
        except Exception:  # noqa: BLE001 - preview loss must never stop recognition
            return True
        return True

    def _resize_for_transport(self, frame: Any) -> Any:
        height, width = frame.shape[:2]
        if width <= 640:
            return frame
        scale = 640.0 / float(width)
        return self._cv2.resize(frame, (640, max(1, int(height * scale))))

    def _encode_jpeg(self, frame: Any) -> bytes:
        candidates = [frame]
        height, width = frame.shape[:2]
        for target_width in (480, 360):
            if width > target_width:
                scale = target_width / float(width)
                candidates.append(
                    self._cv2.resize(frame, (target_width, max(1, int(height * scale))))
                )
        for candidate in candidates:
            for quality in (68, 52, 38):
                ok, encoded = self._cv2.imencode(
                    ".jpg",
                    candidate,
                    [int(self._cv2.IMWRITE_JPEG_QUALITY), quality],
                )
                if not ok:
                    continue
                value = bytes(encoded)
                if len(value) < _DATAGRAM_LIMIT - 2_000:
                    return value
        raise RuntimeError("gesture preview frame is too large for local transport")

    def _send(self, metadata: dict[str, Any], payload: bytes) -> None:
        if self._socket is None:
            return
        encoded = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        packet = struct.pack("!I", len(encoded)) + encoded + payload
        if len(packet) > _DATAGRAM_LIMIT:
            raise RuntimeError("gesture preview datagram exceeds the transport limit")
        self._socket.sendto(packet, ("127.0.0.1", self.port))

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._send({"event": "closed", "token": self.token}, b"")
        except OSError:
            pass
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._closed = True


def build_gesture_preview(title: str) -> ModernGesturePreview | EmbeddedGesturePreview:
    """Use the embedded app surface when Control Center supplied an endpoint."""
    port_text = os.environ.get("JARVIS_GESTURE_PREVIEW_PORT", "").strip()
    token = os.environ.get("JARVIS_GESTURE_PREVIEW_TOKEN", "").strip()
    if port_text and token:
        try:
            port = int(port_text)
        except ValueError:
            port = 0
        if 0 < port <= 65_535:
            return EmbeddedGesturePreview(port, token)
    return ModernGesturePreview(title)
