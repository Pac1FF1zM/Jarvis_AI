"""Main PySide6 window for the Jarvis Control Center."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QTimer,
)
from PySide6.QtGui import QColor, QCloseEvent, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from control_center.calibration import VoiceCalibrationDialog
from control_center.config_model import ConfigEditorError, ConfigRepository
from control_center.dialog_page import DialogHistoryPage
from control_center.gesture_page import GestureModePage
from control_center.processes import DoctorProcess, JarvisRuntimeProcess
from control_center.workspace_page import WorkspacePage
from control_center.theme import CYAN, GREEN, MAGENTA, MUTED, TEXT, YELLOW
from core.config_loader import load_config
from core.profile_manager import ProfileManager


def _label(text: str, object_name: str | None = None) -> QLabel:
    value = QLabel(text)
    if object_name:
        value.setObjectName(object_name)
    return value


def _card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(10)
    return frame, layout


class AnimatedBackdrop(QWidget):
    """Quiet moving grid used only on the overview screen."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._timer.start(42)

    def _advance(self) -> None:
        if self.isVisible():
            self._phase = (self._phase + 0.006) % 1.0
            self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#07090f"))
        grid = QColor(CYAN)
        grid.setAlpha(13)
        painter.setPen(QPen(grid, 1))
        step = 44
        offset = int(self._phase * step)
        for x in range(-step + offset, self.width() + step, step):
            painter.drawLine(x, 0, x, self.height())
        for y in range(-step + offset, self.height() + step, step):
            painter.drawLine(0, y, self.width(), y)

        scan_y = int(self._phase * (self.height() + 180)) - 90
        gradient = QLinearGradient(0, scan_y - 45, 0, scan_y + 45)
        gradient.setColorAt(0.0, QColor(25, 230, 242, 0))
        gradient.setColorAt(0.5, QColor(25, 230, 242, 15))
        gradient.setColorAt(1.0, QColor(25, 230, 242, 0))
        painter.fillRect(QRect(0, scan_y - 45, self.width(), 90), gradient)


class InteractiveCard(QFrame):
    """Dashboard card with a restrained animated hover glow."""

    def __init__(self, accent: str = CYAN, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("quickCard")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        color = QColor(accent)
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setColor(QColor(color.red(), color.green(), color.blue(), 100))
        self._shadow.setBlurRadius(0)
        self._shadow.setOffset(0, 0)
        self.setGraphicsEffect(self._shadow)
        self._glow = QPropertyAnimation(self._shadow, b"blurRadius", self)
        self._glow.setDuration(220)
        self._glow.setEasingCurve(QEasingCurve.Type.OutCubic)

    def enterEvent(self, event: Any) -> None:
        self._animate_glow(22.0)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._animate_glow(0.0)
        super().leaveEvent(event)

    def _animate_glow(self, target: float) -> None:
        self._glow.stop()
        self._glow.setStartValue(self._shadow.blurRadius())
        self._glow.setEndValue(target)
        self._glow.start()


class PulseCore(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self._phase = 0.0
        self._active = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.timer.start(32)

    def _advance(self) -> None:
        self._phase = (self._phase + (0.055 if self._active else 0.018)) % (math.pi * 2)
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        size = min(self.width(), self.height())
        color = QColor(CYAN if self._active else "#34445b")
        pulse = (math.sin(self._phase * 1.7) + 1.0) / 2.0
        glow = QColor(color)
        glow.setAlpha(24 + int(34 * pulse))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        glow_radius = size * (0.34 + pulse * 0.015)
        painter.drawEllipse(center, glow_radius, glow_radius)
        for radius, width, alpha, speed in (
            (0.32, 2.0, 190, 1.0),
            (0.265, 1.0, 105, -1.5),
            (0.205, 3.0, 220, 0.65),
        ):
            ring = QColor(color)
            ring.setAlpha(alpha)
            pen = QPen(ring, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setDashPattern([5, 6] if radius != 0.205 else [18, 5, 2, 5])
            pen.setDashOffset(self._phase * 9 * speed)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = size * radius
            painter.drawEllipse(center, r, r)
        orbit_radius = size * 0.32
        orbit_angle = self._phase * (2.1 if self._active else 0.7)
        node = QPoint(
            int(center.x() + math.cos(orbit_angle) * orbit_radius),
            int(center.y() + math.sin(orbit_angle) * orbit_radius),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(YELLOW if self._active else MUTED))
        node_radius = 4 if self._active else 3
        painter.drawEllipse(node, node_radius, node_radius)
        painter.setPen(QColor(YELLOW if self._active else MUTED))
        font = QFont("Segoe UI Variable", 16, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(0, center.y() - 28, self.width(), 28),
            Qt.AlignmentFlag.AlignCenter,
            "ONLINE" if self._active else "STANDBY",
        )
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Cascadia Mono", 8))
        painter.drawText(
            QRectF(0, center.y() + 5, self.width(), 24),
            Qt.AlignmentFlag.AlignCenter,
            "JARVIS CORE // 01",
        )


class ControlCenterWindow(QMainWindow):
    def __init__(self, project_root: str | Path, config_path: str | Path) -> None:
        super().__init__()
        self.root = Path(project_root).resolve()
        self.config_path = Path(config_path).resolve()
        initial_config = load_config(str(self.config_path))
        profile_root = str(initial_config.profiles.get("root", "")).strip() or None
        self.profile_manager = ProfileManager(profile_root)
        self.profile_id = self.profile_manager.active_profile_id()
        self.profile_manager.ensure_profile(self.profile_id)
        self.workspace_store_path = self.profile_manager.profile_dir(self.profile_id) / "workspaces.json"
        self.memory_db_path = str(initial_config.memory.get("db_path", self.root / "memory.db"))
        self.runtime = JarvisRuntimeProcess(self.root)
        self.doctor = DoctorProcess(self.root)
        self._page_animation: QParallelAnimationGroup | None = None
        self._nav_animation: QPropertyAnimation | None = None
        self._nav_buttons: list[QPushButton] = []
        self._controls: dict[str, Any] = {}
        self._close_after_stop = False
        self.setWindowTitle("Jarvis // Control Center")
        self.resize(1380, 850)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        self._wire_processes()
        self._load_config_form()
        self._refresh_calibration_status()
        self._set_runtime_state("stopped")

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(225)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(14, 24, 14, 20)
        side.setSpacing(6)
        brand = _label("JARVIS", "brand")
        brand.setContentsMargins(12, 0, 0, 0)
        edition = _label("CONTROL CENTER // DEV", "eyebrow")
        edition.setContentsMargins(12, 0, 0, 20)
        side.addWidget(brand)
        side.addWidget(edition)
        entries = (
            ("ОБЗОР", 0),
            ("КОНФИГУРАЦИЯ", 1),
            ("ДИАГНОСТИКА", 2),
            ("КАЛИБРОВКА", 3),
            ("ЖЕСТОВЫЙ РЕЖИМ", 4),
            ("ПРОСТРАНСТВА", 5),
            ("ДИАЛОГИ", 6),
            ("ЖУРНАЛ", 7),
        )
        for text, index in entries:
            button = QPushButton(text)
            button.setObjectName("nav")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=index: self._show_page(value))
            self._nav_buttons.append(button)
            side.addWidget(button)
        side.addStretch(1)
        build = _label("LOCAL RUNTIME\nBUILD 2026.08", "muted")
        build.setContentsMargins(12, 0, 0, 0)
        side.addWidget(build)
        self.nav_indicator = QFrame(self.sidebar)
        self.nav_indicator.setObjectName("navIndicator")
        self.nav_indicator.setFixedWidth(3)
        self.nav_indicator.hide()
        outer.addWidget(self.sidebar)

        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(70)
        top = QHBoxLayout(topbar)
        top.setContentsMargins(26, 0, 28, 0)
        self.path_label = _label(str(self.config_path), "muted")
        self.path_label.setToolTip(str(self.config_path))
        top.addWidget(_label("ACTIVE CONFIG", "eyebrow"))
        top.addWidget(self.path_label, 1)
        self.runtime_status = _label("OFFLINE", "statusOffline")
        top.addWidget(self.runtime_status)
        shell_layout.addWidget(topbar)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._dashboard_page())
        self.pages.addWidget(self._config_page())
        self.pages.addWidget(self._doctor_page())
        self.pages.addWidget(self._calibration_page())
        self.gesture_view = GestureModePage()
        self.gesture_view.preview_activated.connect(self._gesture_preview_activated)
        self.pages.addWidget(self.gesture_view)
        self.workspace_view = WorkspacePage(self.workspace_store_path)
        self.pages.addWidget(self.workspace_view)
        self.dialog_view = DialogHistoryPage(self.memory_db_path, self.profile_id)
        self.pages.addWidget(self.dialog_view)
        self.pages.addWidget(self._logs_page())
        shell_layout.addWidget(self.pages, 1)
        outer.addWidget(shell, 1)
        self._nav_buttons[0].setChecked(True)
        QTimer.singleShot(0, lambda: self._move_nav_indicator(0, animate=False))

    def _page_shell(self, eyebrow: str, title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 26, 30, 28)
        layout.setSpacing(12)
        layout.addWidget(_label(eyebrow, "eyebrow"))
        layout.addWidget(_label(title, "title"))
        text = _label(description, "muted")
        text.setWordWrap(True)
        layout.addWidget(text)
        return page, layout

    def _dashboard_page(self) -> QWidget:
        page = AnimatedBackdrop()
        page.setObjectName("dashboardPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 24, 30, 26)
        layout.setSpacing(14)

        heading = QHBoxLayout()
        heading_text = QVBoxLayout()
        heading_text.setSpacing(2)
        heading_text.addWidget(_label("SYSTEM OVERVIEW // 01", "eyebrow"))
        heading_text.addWidget(_label("Центр управления", "title"))
        heading.addLayout(heading_text)
        heading.addStretch(1)
        heading.addWidget(_label("●  LOCAL / PRIVATE", "privacyBadge"), 0, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(heading)

        self.dashboard_hero = QFrame()
        self.dashboard_hero.setObjectName("heroCard")
        hero = QHBoxLayout(self.dashboard_hero)
        hero.setContentsMargins(28, 18, 34, 20)
        hero.setSpacing(28)
        self.pulse_core = PulseCore()
        hero.addWidget(self.pulse_core, 5)

        command = QVBoxLayout()
        command.setSpacing(8)
        command.addStretch(1)
        command.addWidget(_label("LOCAL INTELLIGENCE NODE", "eyebrow"))
        command.addWidget(_label("JARVIS CORE", "heroTitle"))
        self.hero_state_label = _label("СТАНДБАЙ", "heroStateOffline")
        command.addWidget(self.hero_state_label)
        self.runtime_detail = _label(
            "Голосовые модули ожидают запуска. Все вычисления остаются на этом компьютере.",
            "muted",
        )
        self.runtime_detail.setWordWrap(True)
        command.addWidget(self.runtime_detail)

        metrics = QHBoxLayout()
        metrics.setSpacing(8)
        for value, caption in (("LOCAL", "РЕЖИМ"), ("5", "МОДУЛЕЙ"), ("PRIVATE", "ДАННЫЕ")):
            metric = QFrame()
            metric.setObjectName("metric")
            metric_box = QVBoxLayout(metric)
            metric_box.setContentsMargins(12, 8, 12, 8)
            metric_box.setSpacing(0)
            metric_box.addWidget(_label(value, "metricValue"))
            metric_box.addWidget(_label(caption, "metricCaption"))
            metrics.addWidget(metric, 1)
        command.addLayout(metrics)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.start_button = QPushButton("ЗАПУСТИТЬ")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start_runtime)
        self.stop_button = QPushButton("ОСТАНОВИТЬ")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.runtime.stop)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        command.addLayout(controls)
        command.addWidget(_label("VOICE: «ДЖАРВИС»   ·   GESTURES: CTRL + ALT + /", "commandHint"))
        command.addStretch(1)
        hero.addLayout(command, 6)
        layout.addWidget(self.dashboard_hero, 6)

        quick_heading = QHBoxLayout()
        quick_heading.addWidget(_label("БЫСТРЫЙ ДОСТУП", "eyebrow"))
        quick_heading.addStretch(1)
        quick_heading.addWidget(_label("НАВЕДИТЕ ДЛЯ ПОДСВЕТКИ", "mutedSmall"))
        layout.addLayout(quick_heading)

        quick_grid = QHBoxLayout()
        quick_grid.setSpacing(14)
        self.dashboard_cards: list[InteractiveCard] = []

        self.doctor_summary_card = InteractiveCard(CYAN)
        doctor_layout = QVBoxLayout(self.doctor_summary_card)
        doctor_layout.setContentsMargins(18, 15, 18, 15)
        doctor_layout.addWidget(_label("SYSTEM DOCTOR", "eyebrow"))
        self.doctor_summary = _label("Проверка ещё не запускалась", "sectionTitle")
        doctor_layout.addWidget(self.doctor_summary)
        doctor_layout.addStretch(1)
        doctor_quick = QPushButton("Запустить проверку")
        doctor_quick.clicked.connect(lambda: (self._show_page(2), self._run_doctor()))
        doctor_layout.addWidget(doctor_quick)
        quick_grid.addWidget(self.doctor_summary_card, 1)
        self.dashboard_cards.append(self.doctor_summary_card)

        calibration_card = InteractiveCard(YELLOW)
        calibration_layout = QVBoxLayout(calibration_card)
        calibration_layout.setContentsMargins(18, 15, 18, 15)
        calibration_layout.addWidget(_label("VOICE PROFILE", "eyebrowYellow"))
        self.dashboard_calibration = _label("Проверка профиля...", "sectionTitle")
        calibration_layout.addWidget(self.dashboard_calibration)
        calibration_layout.addStretch(1)
        calibration_quick = QPushButton("Открыть мастер")
        calibration_quick.clicked.connect(lambda: (self._show_page(3), self._open_calibration()))
        calibration_layout.addWidget(calibration_quick)
        quick_grid.addWidget(calibration_card, 1)
        self.dashboard_cards.append(calibration_card)

        config_card = InteractiveCard(MAGENTA)
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(18, 15, 18, 15)
        config_layout.addWidget(_label("RUNTIME CONFIG", "eyebrowMagenta"))
        config_layout.addWidget(_label("Безопасные параметры и полный YAML", "sectionTitle"))
        config_layout.addStretch(1)
        config_quick = QPushButton("Настроить")
        config_quick.clicked.connect(lambda: self._show_page(1))
        config_layout.addWidget(config_quick)
        quick_grid.addWidget(config_card, 1)
        self.dashboard_cards.append(config_card)
        layout.addLayout(quick_grid, 3)
        return page

    def _config_page(self) -> QWidget:
        page, layout = self._page_shell(
            "RUNTIME CONFIG // 02",
            "Конфигурация",
            "Основные параметры сохраняются без удаления комментариев. Для полного контроля доступен YAML.",
        )
        path_row = QHBoxLayout()
        self.config_path_edit = QLineEdit(str(self.config_path))
        browse = QPushButton("Выбрать файл")
        browse.clicked.connect(self._browse_config)
        reload_button = QPushButton("Перезагрузить")
        reload_button.clicked.connect(self._reload_config_from_path)
        path_row.addWidget(self.config_path_edit, 1)
        path_row.addWidget(browse)
        path_row.addWidget(reload_button)
        layout.addLayout(path_row)

        tabs = QTabWidget()
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setContentsMargins(14, 14, 14, 14)
        form_layout.setSpacing(14)
        form_layout.addWidget(self._config_group("Жестовый режим", (
            ("Модуль включён", "modules.gesture.enabled", self._check()),
            ("Устройство", "modules.gesture.device", self._combo(("cpu", "cuda"))),
            ("Показывать интерфейс", "modules.gesture.params.preview_enabled", self._check()),
            ("Индекс камеры", "modules.gesture.params.camera_index", self._spin(0, 16)),
            ("Порог уверенности", "modules.gesture.params.confidence_threshold", self._double(0.1, 1.0, 0.01)),
            ("Стабильных окон", "modules.gesture.params.consecutive_windows", self._spin(2, 12)),
            ("Cooldown, секунд", "modules.gesture.params.cooldown_seconds", self._double(0.0, 10.0, 0.1)),
            ("Выполнять G01–G06", "modules.gesture.params.execution_enabled", self._check()),
        )))
        form_layout.addWidget(self._config_group("Голос и диалог", (
            ("Wake phrase", "modules.wake_word.params.wake_phrase_enabled", self._check()),
            ("Порог wake phrase", "modules.wake_word.params.wake_phrase_threshold", self._double(0.05, 0.95, 0.01)),
            ("Активная сессия", "modules.wake_word.params.active_session_enabled", self._check()),
            ("Таймаут продолжения", "modules.wake_word.params.active_session_timeout_seconds", self._spin(2, 60)),
            ("Устройство STT", "modules.stt.device", self._combo(("auto", "cpu", "cuda"))),
            ("Модель Parakeet", "modules.stt.params.model_dir", QLineEdit()),
            ("Устройство LLM", "modules.llm.device", self._combo(("cpu", "cuda"))),
            ("Модель Ollama", "modules.llm.model", QLineEdit()),
            ("Уровень логов", "logging.level", self._combo(("DEBUG", "INFO", "WARNING", "ERROR"))),
        )))
        save_form = QPushButton("СОХРАНИТЬ ОСНОВНЫЕ ПАРАМЕТРЫ")
        save_form.setObjectName("primary")
        save_form.clicked.connect(self._save_config_form)
        form_layout.addWidget(save_form)
        form_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(form_container)
        tabs.addTab(scroll, "Основные")

        yaml_page = QWidget()
        yaml_layout = QVBoxLayout(yaml_page)
        yaml_layout.setContentsMargins(14, 14, 14, 14)
        warning = _label(
            "Расширенный режим: перед сохранением YAML проверяется, но смысл параметров остаётся вашей ответственностью.",
            "muted",
        )
        warning.setWordWrap(True)
        self.yaml_editor = QPlainTextEdit()
        self.yaml_editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        save_yaml = QPushButton("ПРОВЕРИТЬ И СОХРАНИТЬ YAML")
        save_yaml.setObjectName("primary")
        save_yaml.clicked.connect(self._save_raw_yaml)
        yaml_layout.addWidget(warning)
        yaml_layout.addWidget(self.yaml_editor, 1)
        yaml_layout.addWidget(save_yaml)
        tabs.addTab(yaml_page, "YAML")
        layout.addWidget(tabs, 1)
        return page

    def _config_group(self, title: str, rows: tuple[tuple[str, str, QWidget], ...]) -> QFrame:
        frame, layout = _card()
        layout.addWidget(_label(title, "sectionTitle"))
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        for caption, key, widget in rows:
            self._controls[key] = widget
            form.addRow(caption, widget)
        layout.addLayout(form)
        return frame

    @staticmethod
    def _check() -> QCheckBox:
        return QCheckBox("Включено")

    @staticmethod
    def _combo(values: tuple[str, ...]) -> QComboBox:
        widget = QComboBox()
        widget.addItems(values)
        return widget

    @staticmethod
    def _spin(minimum: int, maximum: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        return widget

    @staticmethod
    def _double(minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setDecimals(2)
        return widget

    def _doctor_page(self) -> QWidget:
        page, layout = self._page_shell(
            "SYSTEM DOCTOR // 03",
            "Диагностика",
            "Та же проверка --doctor, но с понятным отчётом и действиями прямо в приложении.",
        )
        toolbar = QHBoxLayout()
        self.doctor_button = QPushButton("ЗАПУСТИТЬ ПОЛНУЮ ПРОВЕРКУ")
        self.doctor_button.setObjectName("primary")
        self.doctor_button.clicked.connect(self._run_doctor)
        self.doctor_progress = QProgressBar()
        self.doctor_progress.setRange(0, 1)
        self.doctor_progress.setValue(0)
        self.doctor_counts = _label("PASS 0  //  WARN 0  //  FAIL 0", "eyebrow")
        toolbar.addWidget(self.doctor_button)
        toolbar.addWidget(self.doctor_progress, 1)
        toolbar.addWidget(self.doctor_counts)
        layout.addLayout(toolbar)
        self.doctor_table = QTableWidget(0, 5)
        self.doctor_table.setHorizontalHeaderLabels(
            ["СТАТУС", "КАТЕГОРИЯ", "ПРОВЕРКА", "РЕЗУЛЬТАТ", "ЧТО ДЕЛАТЬ"]
        )
        self.doctor_table.setAlternatingRowColors(True)
        self.doctor_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.doctor_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.doctor_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.doctor_table, 1)
        return page

    def _calibration_page(self) -> QWidget:
        page, layout = self._page_shell(
            "VOICE CALIBRATION // 04",
            "Калибровка голоса",
            "Мастер измерит фон, тихую и громкую речь, настроит VAD и усиление для текущего микрофона.",
        )
        card, content = _card()
        content.addWidget(_label("АКТИВНЫЙ ПРОФИЛЬ", "eyebrow"))
        self.calibration_status = _label("Проверка...", "sectionTitle")
        self.calibration_detail = _label("", "muted")
        self.calibration_detail.setWordWrap(True)
        start = QPushButton("ОТКРЫТЬ МАСТЕР КАЛИБРОВКИ")
        start.setObjectName("primary")
        start.clicked.connect(self._open_calibration)
        content.addWidget(self.calibration_status)
        content.addWidget(self.calibration_detail)
        content.addStretch(1)
        content.addWidget(start)
        layout.addWidget(card, 1)
        return page

    def _logs_page(self) -> QWidget:
        page, layout = self._page_shell(
            "LIVE TELEMETRY // 08",
            "Журнал",
            "Вывод runtime отображается здесь. Подробные файлы продолжают храниться в logs/.",
        )
        buttons = QHBoxLayout()
        clear = QPushButton("Очистить экран")
        clear.clicked.connect(lambda: self.log_view.clear())
        open_folder = QPushButton("Открыть папку logs")
        open_folder.clicked.connect(self._open_logs_folder)
        buttons.addWidget(clear)
        buttons.addWidget(open_folder)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setPlaceholderText("Runtime ещё не запускался...")
        layout.addWidget(self.log_view, 1)
        return page

    def _gesture_preview_activated(self) -> None:
        self._show_page(4)
        self._toast("Жестовый режим активирован — камера открыта внутри Control Center")

    def _wire_processes(self) -> None:
        self.runtime.output.connect(self._append_log)
        self.runtime.state_changed.connect(self._set_runtime_state)
        self.runtime.failed.connect(lambda message: QMessageBox.critical(self, "Ошибка запуска", message))
        self.doctor.output.connect(self._append_log)
        self.doctor.completed.connect(self._render_doctor)
        self.doctor.failed.connect(self._doctor_failed)
        self.doctor.state_changed.connect(self._doctor_busy)

    def _show_page(self, index: int) -> None:
        if index == self.pages.currentIndex():
            return
        for button_index, button in enumerate(self._nav_buttons):
            button.setChecked(button_index == index)
        self.pages.setCurrentIndex(index)
        page = self.pages.currentWidget()
        self._move_nav_indicator(index)
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)
        end_position = page.pos()
        page.move(end_position + QPoint(24, 0))
        opacity = QPropertyAnimation(effect, b"opacity", self)
        opacity.setDuration(300)
        opacity.setStartValue(0.0)
        opacity.setEndValue(1.0)
        opacity.setEasingCurve(QEasingCurve.Type.OutCubic)
        slide = QPropertyAnimation(page, b"pos", self)
        slide.setDuration(360)
        slide.setStartValue(page.pos())
        slide.setEndValue(end_position)
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        group = QParallelAnimationGroup(self)
        group.addAnimation(opacity)
        group.addAnimation(slide)
        group.finished.connect(lambda: self._finish_page_animation(page, end_position))
        self._page_animation = group
        group.start()

    def _finish_page_animation(self, page: QWidget, position: QPoint) -> None:
        page.move(position)
        page.setGraphicsEffect(None)

    def _move_nav_indicator(self, index: int, *, animate: bool = True) -> None:
        if not 0 <= index < len(self._nav_buttons):
            return
        button = self._nav_buttons[index]
        top_left = button.mapTo(self.sidebar, QPoint(0, 0))
        target = QRect(14, top_left.y() + 7, 3, max(18, button.height() - 14))
        self.nav_indicator.show()
        self.nav_indicator.raise_()
        current = self.nav_indicator.geometry()
        if not animate or not current.isValid():
            self.nav_indicator.setGeometry(target)
            return
        animation = QPropertyAnimation(self.nav_indicator, b"geometry", self)
        animation.setDuration(280)
        animation.setStartValue(current)
        animation.setEndValue(target)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._nav_animation = animation
        animation.start()

    def _start_runtime(self) -> None:
        if not self.config_path.is_file():
            QMessageBox.warning(self, "Конфиг не найден", str(self.config_path))
            return
        self.runtime.start(
            self.config_path,
            gesture_preview_port=self.gesture_view.preview_port,
            gesture_preview_token=self.gesture_view.preview_token,
        )

    def _set_runtime_state(self, state: str) -> None:
        running = state in {"starting", "running", "stopping"}
        active = state == "running"
        labels = {
            "starting": "ЗАПУСК...",
            "running": "ONLINE",
            "stopping": "ОСТАНОВКА...",
            "stopped": "OFFLINE",
            "failed": "ERROR",
        }
        self.runtime_status.setText(labels.get(state, state.upper()))
        self.runtime_status.setObjectName("statusOnline" if active else "statusOffline")
        self.runtime_status.style().unpolish(self.runtime_status)
        self.runtime_status.style().polish(self.runtime_status)
        self.pulse_core.set_active(active)
        hero_labels = {
            "starting": "ИНИЦИАЛИЗАЦИЯ",
            "running": "СИСТЕМА АКТИВНА",
            "stopping": "ЗАВЕРШЕНИЕ СЕССИИ",
            "stopped": "СТАНДБАЙ",
            "failed": "ТРЕБУЕТСЯ ВНИМАНИЕ",
        }
        hero_details = {
            "starting": "Поднимаю локальные модули и проверяю готовность голосового контура…",
            "running": "Голосовое управление активно. Jarvis готов принимать команды.",
            "stopping": "Безопасно останавливаю фоновые модули и сохраняю журналы…",
            "stopped": "Голосовые модули ожидают запуска. Все вычисления остаются на этом компьютере.",
            "failed": "Runtime не запустился. Откройте журнал или выполните диагностику.",
        }
        self.hero_state_label.setText(hero_labels.get(state, state.upper()))
        self.hero_state_label.setObjectName("heroStateOnline" if active else "heroStateOffline")
        self.hero_state_label.style().unpolish(self.hero_state_label)
        self.hero_state_label.style().polish(self.hero_state_label)
        self.runtime_detail.setText(hero_details.get(state, ""))
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running and state != "stopping")
        if state in {"stopped", "failed"}:
            self.gesture_view.runtime_stopped()
        if state in {"stopped", "failed"} and self._close_after_stop:
            self._close_after_stop = False
            QTimer.singleShot(0, self.close)

    def _append_log(self, text: str) -> None:
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _browse_config(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Выберите config.yaml", str(self.config_path.parent), "YAML (*.yaml *.yml)"
        )
        if path:
            self.config_path_edit.setText(path)
            self._reload_config_from_path()

    def _reload_config_from_path(self) -> None:
        candidate = Path(self.config_path_edit.text().strip()).expanduser().resolve()
        if not candidate.is_file():
            QMessageBox.warning(self, "Конфиг не найден", str(candidate))
            return
        self.config_path = candidate
        self.path_label.setText(str(candidate))
        self.path_label.setToolTip(str(candidate))
        self._load_config_form()
        self._refresh_calibration_status()

    def _load_config_form(self) -> None:
        try:
            repository = ConfigRepository(self.config_path)
            values = repository.form_values()
            self.yaml_editor.setPlainText(repository.raw_text())
        except (OSError, ConfigEditorError) as exc:
            QMessageBox.critical(self, "Ошибка конфигурации", str(exc))
            return
        for key, widget in self._controls.items():
            value = values[key]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                index = widget.findText(str(value))
                if index < 0:
                    widget.addItem(str(value))
                    index = widget.count() - 1
                widget.setCurrentIndex(index)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def _control_values(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, widget in self._controls.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                result[key] = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                result[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                result[key] = widget.text().strip()
        return result

    def _save_config_form(self) -> None:
        if self.runtime.running:
            QMessageBox.information(
                self,
                "Jarvis запущен",
                "Остановите Jarvis перед изменением runtime-конфигурации.",
            )
            return
        try:
            repository = ConfigRepository(self.config_path)
            repository.save_form(self._control_values())
            self.yaml_editor.setPlainText(repository.raw_text())
        except (OSError, ConfigEditorError) as exc:
            QMessageBox.critical(self, "Конфигурация не сохранена", str(exc))
            return
        self._toast("Конфигурация сохранена")

    def _save_raw_yaml(self) -> None:
        if self.runtime.running:
            QMessageBox.information(self, "Jarvis запущен", "Сначала остановите runtime.")
            return
        try:
            ConfigRepository(self.config_path).save_raw(self.yaml_editor.toPlainText())
            self._load_config_form()
        except (OSError, ConfigEditorError) as exc:
            QMessageBox.critical(self, "YAML не сохранён", str(exc))
            return
        self._toast("YAML проверен и сохранён")

    def _run_doctor(self) -> None:
        self.doctor.start(self.config_path)

    def _doctor_busy(self, busy: bool) -> None:
        self.doctor_button.setEnabled(not busy)
        self.doctor_progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.doctor_progress.setValue(1)

    def _render_doctor(self, report: dict[str, Any]) -> None:
        checks = list(report.get("checks") or [])
        self.doctor_table.setRowCount(len(checks))
        colors = {"pass": GREEN, "warn": YELLOW, "fail": MAGENTA, "skip": MUTED}
        for row, check in enumerate(checks):
            values = (
                str(check.get("status", "?")),
                str(check.get("category", "")),
                str(check.get("check_id", "")),
                str(check.get("summary", "")) + (
                    "\n" + str(check.get("detail")) if check.get("detail") else ""
                ),
                str(check.get("action", "")),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setForeground(QColor(colors.get(values[0], TEXT)))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.doctor_table.setItem(row, column, item)
        self.doctor_table.resizeRowsToContents()
        counts = report.get("counts") or {}
        self.doctor_counts.setText(
            "PASS {}  //  WARN {}  //  FAIL {}".format(
                counts.get("pass", 0), counts.get("warn", 0), counts.get("fail", 0)
            )
        )
        overall = str(report.get("overall", "unknown")).upper()
        self.doctor_summary.setText(f"Последний результат: {overall}")
        self._toast(f"Диагностика завершена: {overall}")

    def _doctor_failed(self, message: str) -> None:
        self._doctor_busy(False)
        QMessageBox.critical(self, "Ошибка диагностики", message)

    def _open_calibration(self) -> None:
        if self.runtime.running:
            QMessageBox.information(
                self,
                "Микрофон занят",
                "Сначала остановите Jarvis, затем запустите калибровку.",
            )
            return
        dialog = VoiceCalibrationDialog(self.config_path, self)
        dialog.calibration_saved.connect(lambda _value: self._refresh_calibration_status())
        dialog.exec()

    def _refresh_calibration_status(self) -> None:
        try:
            cfg = load_config(str(self.config_path))
            profile_root = str(cfg.profiles.get("root", "")).strip() or None
            manager = ProfileManager(profile_root)
            profile_id = manager.active_profile_id()
            calibration = manager.active_calibration(profile_id)
        except Exception as exc:  # noqa: BLE001 - status card remains usable
            self.calibration_status.setText("Профиль недоступен")
            self.calibration_detail.setText(str(exc))
            self.dashboard_calibration.setText("Требуется проверка")
            return
        if calibration is None:
            self.calibration_status.setText(f"{profile_id} // не откалиброван")
            self.calibration_detail.setText("Запустите мастер для текущего микрофона.")
            self.dashboard_calibration.setText("Не настроен")
            return
        quality = calibration.get("quality", {})
        device = calibration.get("device", {})
        self.calibration_status.setText(f"{profile_id} // готов")
        self.calibration_detail.setText(
            "{}\nSNR {:.1f} dB · VAD {:.2f}/{:.2f} · gain {:+.1f} dB".format(
                device.get("name", "Микрофон"),
                float(quality.get("snr_db", 0.0)),
                float(calibration.get("vad_start_threshold", 0.0)),
                float(calibration.get("vad_end_threshold", 0.0)),
                float(calibration.get("pcm_gain_db", 0.0)),
            )
        )
        self.dashboard_calibration.setText("Профиль готов")

    def _open_logs_folder(self) -> None:
        import os

        path = self.root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)  # type: ignore[attr-defined]

    def _toast(self, text: str) -> None:
        self.statusBar().showMessage(text, 3500)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.runtime.running:
            choice = QMessageBox.question(
                self,
                "Jarvis ещё работает",
                "Остановить Jarvis и закрыть Control Center?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if choice != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._close_after_stop = True
            self.runtime.stop()
            event.ignore()
            return
        if self.doctor.running:
            self.doctor.process.kill()
            self.doctor.process.waitForFinished(1_000)
        self.dialog_view.shutdown()
        self.workspace_view.shutdown()
        event.accept()


def animate_window_entrance(window: QMainWindow) -> QPropertyAnimation:
    window.setWindowOpacity(0.0)
    animation = QPropertyAnimation(window, b"windowOpacity", window)
    animation.setDuration(360)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start()
    return animation
