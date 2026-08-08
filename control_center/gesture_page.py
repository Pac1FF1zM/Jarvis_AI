"""Embedded Gesture Core surface and local preview receiver."""
from __future__ import annotations

import secrets
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QResizeEvent
from PySide6.QtNetwork import QHostAddress, QUdpSocket
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from modules.gesture_ui import decode_gesture_datagram


def _label(text: str, object_name: str | None = None) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


class VideoSurface(QLabel):
    """Keep the most recent camera frame correctly scaled during window resize."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("gestureVideo")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(520, 360)
        self._source = QPixmap()
        self.show_placeholder()

    def show_frame(self, image: QImage) -> None:
        self._source = QPixmap.fromImage(image)
        self.setText("")
        self._rescale()

    def show_placeholder(self, text: str = "КАМЕРА НЕ АКТИВНА\n\nЗапустите Jarvis и скажите:\n«запусти жестовый режим»") -> None:
        self._source = QPixmap()
        self.clear()
        self.setText(text)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source.isNull():
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class GestureModePage(QWidget):
    """One Control Center page for video and gesture model telemetry."""

    preview_activated = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview_token = secrets.token_hex(16)
        self._active = False
        self._socket = QUdpSocket(self)
        if not self._socket.bind(QHostAddress.SpecialAddress.LocalHost, 0):
            raise RuntimeError(f"gesture preview port is unavailable: {self._socket.errorString()}")
        self.preview_port = int(self._socket.localPort())
        self._socket.readyRead.connect(self._read_datagrams)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 28)
        layout.setSpacing(12)
        layout.addWidget(_label("GESTURE CORE // 05", "eyebrow"))
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(_label("Жестовый режим", "title"))
        title.addWidget(
            _label(
                "Камера, распознавание и статистика работают внутри Control Center.",
                "muted",
            )
        )
        header.addLayout(title)
        header.addStretch(1)
        self.mode_badge = _label("●  ОЖИДАНИЕ", "gestureBadgeOffline")
        header.addWidget(self.mode_badge, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(header)

        workspace = QHBoxLayout()
        workspace.setSpacing(14)
        video_card = QFrame()
        video_card.setObjectName("gestureVideoCard")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(10, 10, 10, 10)
        video_layout.setSpacing(8)
        self.video = VideoSurface()
        video_layout.addWidget(self.video, 1)
        footer = QHBoxLayout()
        self.camera_label = _label("CAMERA // OFFLINE", "mutedSmall")
        self.model_label = _label("MODEL // WAITING", "mutedSmall")
        footer.addWidget(self.camera_label)
        footer.addStretch(1)
        footer.addWidget(self.model_label)
        video_layout.addLayout(footer)
        workspace.addWidget(video_card, 7)

        telemetry = QFrame()
        telemetry.setObjectName("gestureTelemetry")
        telemetry.setFixedWidth(340)
        panel = QVBoxLayout(telemetry)
        panel.setContentsMargins(20, 18, 20, 18)
        panel.setSpacing(10)
        panel.addWidget(_label("LIVE TELEMETRY", "eyebrow"))
        self.gesture_label = _label("D0X", "gestureLabel")
        self.action_label = _label("Ожидание камеры", "sectionTitle")
        self.action_label.setWordWrap(True)
        panel.addWidget(self.gesture_label)
        panel.addWidget(self.action_label)
        panel.addWidget(_label("УВЕРЕННОСТЬ", "mutedSmall"))
        self.confidence = QProgressBar()
        self.confidence.setObjectName("gestureConfidence")
        self.confidence.setRange(0, 1000)
        self.confidence.setValue(0)
        self.confidence.setFormat("0.0%")
        panel.addWidget(self.confidence)

        self.stability_label = _label("СТАБИЛЬНОСТЬ  0/3", "metricCaption")
        panel.addWidget(self.stability_label)
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        panel.addWidget(divider)
        panel.addWidget(_label("TOP-3 ПРЕДСКАЗАНИЯ", "eyebrow"))
        self.top3_labels = [_label("—", "gestureRank") for _ in range(3)]
        for label in self.top3_labels:
            panel.addWidget(label)
        panel.addStretch(1)
        self.last_action_label = _label("Последнее действие: —", "muted")
        self.last_action_label.setWordWrap(True)
        self.performance_label = _label("0.0 FPS  ·  0.0 ms", "gesturePerformance")
        panel.addWidget(self.last_action_label)
        panel.addWidget(self.performance_label)
        workspace.addWidget(telemetry, 3)
        layout.addLayout(workspace, 1)

    def _read_datagrams(self) -> None:
        while self._socket.hasPendingDatagrams():
            datagram = self._socket.receiveDatagram()
            try:
                metadata, jpeg = decode_gesture_datagram(bytes(datagram.data()))
            except (ValueError, UnicodeError):
                continue
            if metadata.get("token") != self.preview_token:
                continue
            event = metadata.get("event")
            if event == "closed":
                self.runtime_stopped()
                continue
            if event not in {"opened", "frame"}:
                continue
            if not self._active:
                self._active = True
                self._set_badge("●  АКТИВЕН", online=True)
                self.preview_activated.emit()
            if event == "frame" and jpeg:
                # PySide6 6.10 rejects the otherwise documented bytes format
                # argument (``b"JPG"``) with ValueError.  Qt reliably detects
                # JPEG from its header, so let it auto-detect the format.
                image = QImage.fromData(jpeg)
                if not image.isNull():
                    self.video.show_frame(image)
                self._render_state(metadata)

    def _render_state(self, state: dict[str, Any]) -> None:
        confidence = max(0.0, min(1.0, float(state.get("confidence", 0.0))))
        self.gesture_label.setText(str(state.get("label", "D0X")))
        self.action_label.setText(str(state.get("action", "ожидание")))
        self.confidence.setValue(round(confidence * 1000))
        self.confidence.setFormat(f"{confidence:.1%}")
        stable = int(state.get("stable_count", 0))
        required = int(state.get("stable_required", 0))
        self.stability_label.setText(f"СТАБИЛЬНОСТЬ  {stable}/{required}")
        top3 = list(state.get("top3") or [])
        for index, label in enumerate(self.top3_labels):
            if index < len(top3) and len(top3[index]) >= 2:
                label.setText(f"{index + 1:02d}   {top3[index][0]}   {float(top3[index][1]):.1%}")
            else:
                label.setText(f"{index + 1:02d}   —")
        self.last_action_label.setText(f"Последнее действие: {state.get('last_action') or '—'}")
        self.performance_label.setText(
            f"{float(state.get('fps', 0.0)):.1f} FPS  ·  "
            f"{float(state.get('latency_ms', 0.0)):.1f} ms"
        )
        self.camera_label.setText(f"CAMERA // {state.get('camera', '—')}")
        self.model_label.setText(f"MODEL // {state.get('model', '—')}")
        status = str(state.get("status", "АКТИВЕН"))
        self._set_badge(f"●  {status}", online=status != "ПАУЗА")

    def runtime_stopped(self) -> None:
        self._active = False
        self._set_badge("●  ОЖИДАНИЕ", online=False)
        self.video.show_placeholder()
        self.camera_label.setText("CAMERA // OFFLINE")
        self.model_label.setText("MODEL // WAITING")
        self.gesture_label.setText("D0X")
        self.action_label.setText("Ожидание камеры")
        self.confidence.setValue(0)
        self.confidence.setFormat("0.0%")
        self.stability_label.setText("СТАБИЛЬНОСТЬ  0/3")
        for index, label in enumerate(self.top3_labels):
            label.setText(f"{index + 1:02d}   —")
        self.last_action_label.setText("Последнее действие: —")
        self.performance_label.setText("0.0 FPS  ·  0.0 ms")

    def _set_badge(self, text: str, *, online: bool) -> None:
        self.mode_badge.setText(text)
        self.mode_badge.setObjectName("gestureBadgeOnline" if online else "gestureBadgeOffline")
        self.mode_badge.style().unpolish(self.mode_badge)
        self.mode_badge.style().polish(self.mode_badge)
