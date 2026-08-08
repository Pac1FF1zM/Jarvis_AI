"""Button-driven voice calibration wizard with no retained raw recordings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from core.config_loader import load_config
from core.profile_manager import ProfileManager
from core.voice_calibration import (
    CalibrationQualityError,
    SAMPLE_RATE,
    analyze_signal,
    derive_calibration,
    _vad_scores,
)


class _WorkerSignals(QObject):
    completed = Signal(object)
    failed = Signal(str)


class _FunctionWorker(QRunnable):
    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function()
        except Exception as exc:  # noqa: BLE001 - surfaced in the wizard
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.completed.emit(result)


def list_input_devices() -> list[tuple[int, dict[str, Any]]]:
    import sounddevice as sd

    devices: list[tuple[int, dict[str, Any]]] = []
    for index, device in enumerate(sd.query_devices()):
        value = dict(device)
        if int(value.get("max_input_channels", 0) or 0) > 0:
            devices.append((index, value))
    return devices


def record_voice_step(device_index: int, seconds: float) -> Any:
    import sounddevice as sd

    return sd.rec(
        round(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=device_index,
        blocking=True,
    ).reshape(-1).copy()


def finalize_calibration(
    config_path: str | Path,
    profile_id: str,
    profile_name: str,
    device: dict[str, Any],
    recordings: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from silero_vad import load_silero_vad

    expected = {"silence", "normal", "quiet", "loud", "validation"}
    if set(recordings) != expected:
        raise CalibrationQualityError("не все упражнения калибровки записаны")
    model = load_silero_vad(onnx=True)
    silence_pcm = recordings["silence"]
    speech_pcm = np.concatenate(
        [recordings["normal"], recordings["quiet"], recordings["loud"]]
    )
    validation_pcm = recordings["validation"]
    silence = analyze_signal(silence_pcm, _vad_scores(model, silence_pcm))
    speech = analyze_signal(speech_pcm, _vad_scores(model, speech_pcm))
    calibration = derive_calibration(device, silence, speech)
    validation = analyze_signal(
        validation_pcm, _vad_scores(model, validation_pcm)
    )
    if (
        validation.vad_speech_ratio < 0.10
        or validation.vad_p95 < calibration["vad_start_threshold"]
    ):
        raise CalibrationQualityError(
            "проверочная фраза не прошла выбранный VAD-порог"
        )
    if validation.clipping_ratio > 0.01:
        raise CalibrationQualityError("проверочная фраза перегружает микрофон")
    calibration["quality"]["validation_vad_speech_ratio"] = (
        validation.vad_speech_ratio
    )
    cfg = load_config(str(config_path))
    profile_root = str(cfg.profiles.get("root", "")).strip() or None
    manager = ProfileManager(profile_root)
    manager.ensure_profile(profile_id, profile_name or None)
    manager.save_calibration(profile_id, calibration)
    manager.set_active(profile_id)
    return calibration


_STEPS = (
    ("silence", 4.0, "ТИШИНА", "Не говорите и не двигайтесь четыре секунды."),
    ("normal", 5.0, "ОБЫЧНЫЙ ГОЛОС", "Скажите: «Джарвис, открой браузер»."),
    ("quiet", 5.0, "ТИХИЙ ГОЛОС", "Скажите тише: «Джарвис, который сейчас час»."),
    ("loud", 5.0, "ГРОМКИЙ ГОЛОС", "Скажите громче обычного фразу с напоминанием."),
    ("validation", 5.0, "ПРОВЕРКА", "Скажите: «Джарвис, открой калькулятор»."),
)


class VoiceCalibrationDialog(QDialog):
    calibration_saved = Signal(dict)

    def __init__(self, config_path: str | Path, parent: Any = None) -> None:
        super().__init__(parent)
        self.config_path = Path(config_path).resolve()
        self.setWindowTitle("Jarvis // Voice Calibration")
        self.setMinimumSize(680, 560)
        self.resize(720, 600)
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.recordings: dict[str, Any] = {}
        self.devices: dict[int, dict[str, Any]] = {}
        self.step_index = 0
        self.countdown = 0
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self._tick_countdown)
        self._build_ui()
        self._load_devices()
        self._show_step()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 28, 30, 28)
        layout.setSpacing(18)
        eyebrow = QLabel("VOICE SYSTEM // CALIBRATION")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Настройка голоса")
        title.setObjectName("title")
        subtitle = QLabel(
            "Записи обрабатываются только в оперативной памяти и никогда не сохраняются."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        identity = QFrame()
        identity.setObjectName("card")
        identity_layout = QVBoxLayout(identity)
        identity_layout.setContentsMargins(18, 16, 18, 16)
        row = QHBoxLayout()
        self.profile_id = QLineEdit("default")
        self.profile_id.setPlaceholderText("ID профиля")
        self.profile_name = QLineEdit("Основной пользователь")
        self.profile_name.setPlaceholderText("Имя профиля")
        row.addWidget(self.profile_id)
        row.addWidget(self.profile_name)
        identity_layout.addLayout(row)
        self.device_combo = QComboBox()
        identity_layout.addWidget(self.device_combo)
        layout.addWidget(identity)

        stage = QFrame()
        stage.setObjectName("accentCard")
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(24, 22, 24, 22)
        stage_layout.setSpacing(10)
        self.step_label = QLabel()
        self.step_label.setObjectName("sectionTitle")
        self.instruction = QLabel()
        self.instruction.setWordWrap(True)
        self.instruction.setMinimumHeight(48)
        self.recording_status = QLabel("ГОТОВ")
        self.recording_status.setObjectName("eyebrow")
        stage_layout.addWidget(self.step_label)
        stage_layout.addWidget(self.instruction)
        stage_layout.addWidget(self.recording_status)
        layout.addWidget(stage, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, len(_STEPS))
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        cancel = QPushButton("Закрыть")
        cancel.clicked.connect(self.reject)
        self.record_button = QPushButton("Начать этап")
        self.record_button.setObjectName("primary")
        self.record_button.clicked.connect(self._prepare_recording)
        buttons.addWidget(cancel)
        buttons.addStretch(1)
        buttons.addWidget(self.record_button)
        layout.addLayout(buttons)

    def _load_devices(self) -> None:
        try:
            devices = list_input_devices()
        except Exception as exc:  # noqa: BLE001
            self.record_button.setEnabled(False)
            self.device_combo.addItem(f"Микрофоны недоступны: {exc}")
            return
        for index, device in devices:
            self.devices[index] = device
            self.device_combo.addItem(str(device.get("name", f"Микрофон {index}")), index)
        if not devices:
            self.record_button.setEnabled(False)
            self.device_combo.addItem("Входные устройства не найдены")

    def _show_step(self) -> None:
        if self.step_index >= len(_STEPS):
            return
        _key, seconds, title, prompt = _STEPS[self.step_index]
        self.step_label.setText(f"ЭТАП {self.step_index + 1}/{len(_STEPS)} // {title}")
        self.instruction.setText(f"{prompt}\nДлительность записи: {seconds:.0f} секунд.")
        self.recording_status.setText("ГОТОВ К ЗАПИСИ")
        self.record_button.setText("Начать этап")
        self.record_button.setEnabled(self.device_combo.currentData() is not None)

    def _prepare_recording(self) -> None:
        if self.device_combo.currentData() is None:
            return
        self.device_combo.setEnabled(False)
        self.profile_id.setEnabled(False)
        self.profile_name.setEnabled(False)
        self.record_button.setEnabled(False)
        self.countdown = 3
        self.recording_status.setText("ПРИГОТОВЬТЕСЬ // 3")
        self.countdown_timer.start(1000)

    def _tick_countdown(self) -> None:
        self.countdown -= 1
        if self.countdown > 0:
            self.recording_status.setText(f"ПРИГОТОВЬТЕСЬ // {self.countdown}")
            return
        self.countdown_timer.stop()
        self._start_recording()

    def _start_recording(self) -> None:
        key, seconds, _title, _prompt = _STEPS[self.step_index]
        device_index = int(self.device_combo.currentData())
        self.recording_status.setText("● ИДЁТ ЗАПИСЬ")
        worker = _FunctionWorker(lambda: record_voice_step(device_index, seconds))
        worker.signals.completed.connect(lambda samples, name=key: self._recorded(name, samples))
        worker.signals.failed.connect(self._failed)
        self.thread_pool.start(worker)

    def _recorded(self, key: str, samples: Any) -> None:
        self.recordings[key] = samples
        self.step_index += 1
        self.progress.setValue(self.step_index)
        if self.step_index < len(_STEPS):
            self._show_step()
            return
        self.recording_status.setText("АНАЛИЗ СИГНАЛА И VAD...")
        self.record_button.setEnabled(False)
        device = dict(self.devices[int(self.device_combo.currentData())])
        recordings = dict(self.recordings)
        profile_id = self.profile_id.text().strip() or "default"
        profile_name = self.profile_name.text().strip()
        worker = _FunctionWorker(
            lambda: finalize_calibration(
                self.config_path,
                profile_id,
                profile_name,
                device,
                recordings,
            )
        )
        worker.signals.completed.connect(self._saved)
        worker.signals.failed.connect(self._failed)
        self.thread_pool.start(worker)

    def _saved(self, calibration: dict[str, Any]) -> None:
        self.recordings.clear()
        quality = calibration.get("quality", {})
        self.recording_status.setText(
            "СОХРАНЕНО // SNR {:.1f} dB // VAD {:.2f}".format(
                float(quality.get("snr_db", 0.0)),
                float(calibration.get("vad_start_threshold", 0.0)),
            )
        )
        self.record_button.setText("Готово")
        self.record_button.setEnabled(True)
        self.record_button.clicked.disconnect()
        self.record_button.clicked.connect(self.accept)
        self.calibration_saved.emit(calibration)

    def _failed(self, message: str) -> None:
        self.recording_status.setText("ОШИБКА КАЛИБРОВКИ")
        self.record_button.setEnabled(True)
        self.record_button.setText("Повторить этап")
        QMessageBox.critical(self, "Калибровка не сохранена", message)

    def reject(self) -> None:
        self.recordings.clear()
        super().reject()
