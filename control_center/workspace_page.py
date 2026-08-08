"""Interactive workspace cards, capture controls and draggable monitor preview."""
from __future__ import annotations

from copy import deepcopy
import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from control_center.theme import CYAN, GREEN, MAGENTA, MUTED, TEXT, YELLOW
from core.workspace_manager import WorkspaceManager
from memory.workspaces import WorkspaceStore
from tools._applications import resolve_application
from tools._windows import list_monitors


def _label(text: str, object_name: str | None = None) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


class WorkspaceCanvas(QWidget):
    """A small drag-and-resize editor for normalized window rectangles."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceCanvas")
        self.setMinimumHeight(255)
        self._workspace: dict[str, Any] = {}
        self._hit_rects: list[tuple[int, QRectF]] = []
        self._active_index = -1
        self._resize = False
        self._last = QPointF()
        try:
            self._monitor_count = max(1, len(list_monitors()))
        except OSError:
            self._monitor_count = 1

    def set_workspace(self, workspace: dict[str, Any] | None) -> None:
        self._workspace = deepcopy(workspace or {})
        self._active_index = -1
        self.update()

    def workspace(self) -> dict[str, Any]:
        return deepcopy(self._workspace)

    def _monitor_rects(self) -> list[QRectF]:
        gap = 16.0
        available = max(200.0, self.width() - 60.0 - gap * (self._monitor_count - 1))
        width = available / self._monitor_count
        height = max(130.0, self.height() - 52.0)
        return [QRectF(30 + index * (width + gap), 24, width, height) for index in range(self._monitor_count)]

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#080c13"))
        monitors = self._monitor_rects()
        for index, monitor in enumerate(monitors, start=1):
            painter.setBrush(QColor("#0d1420"))
            painter.setPen(QPen(QColor("#34475e"), 2))
            painter.drawRoundedRect(monitor, 10, 10)
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(monitor.left(), 3, monitor.width(), 20),
                Qt.AlignmentFlag.AlignCenter,
                f"МОНИТОР {index}",
            )
        placements = self._workspace.get("placements", [])
        labels = {
            str(item.get("key")): str(item.get("label") or item.get("query") or item.get("key"))
            for item in self._workspace.get("applications", [])
        }
        colors = [CYAN, MAGENTA, YELLOW, GREEN]
        self._hit_rects.clear()
        for index, placement in enumerate(placements):
            monitor_number = max(1, min(len(monitors), int(placement.get("monitor", 1))))
            monitor = monitors[monitor_number - 1]
            x = float(placement.get("x", 0.0))
            y = float(placement.get("y", 0.0))
            width = float(placement.get("width", 0.45))
            height = float(placement.get("height", 0.7))
            rect = QRectF(
                monitor.left() + x * monitor.width(),
                monitor.top() + y * monitor.height(),
                max(42, width * monitor.width()),
                max(34, height * monitor.height()),
            ).intersected(monitor)
            self._hit_rects.append((index, rect))
            color = QColor(colors[index % len(colors)])
            fill = QColor(color)
            fill.setAlpha(35 if index != self._active_index else 62)
            painter.setBrush(fill)
            painter.setPen(QPen(color, 2 if index == self._active_index else 1))
            painter.drawRoundedRect(rect.adjusted(3, 3, -3, -3), 6, 6)
            painter.setPen(QColor(TEXT))
            label = labels.get(str(placement.get("key")), str(placement.get("title", "Окно")))
            painter.drawText(rect.adjusted(11, 8, -10, -8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, label[:30])
            handle = QRectF(rect.right() - 15, rect.bottom() - 15, 11, 11)
            painter.fillRect(handle, color)
        if not placements:
            painter.setPen(QColor(MUTED))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Снимок пока пуст\nНажмите «Снять текущее расположение»")
        else:
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(0, self.height() - 24, self.width(), 20),
                Qt.AlignmentFlag.AlignCenter,
                "ПЕРЕТАСКИВАЙТЕ ОКНА; МАРКЕР В УГЛУ МЕНЯЕТ РАЗМЕР",
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        point = event.position()
        self._active_index = -1
        for index, rect in reversed(self._hit_rects):
            if rect.contains(point):
                self._active_index = index
                self._resize = QRectF(rect.right() - 22, rect.bottom() - 22, 22, 22).contains(point)
                self._last = point
                self.update()
                return

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active_index < 0:
            return
        placements = self._workspace.get("placements", [])
        if self._active_index >= len(placements):
            return
        placement = placements[self._active_index]
        monitor_number = max(1, min(self._monitor_count, int(placement.get("monitor", 1))))
        monitors = self._monitor_rects()
        monitor = monitors[monitor_number - 1]
        delta = event.position() - self._last
        self._last = event.position()
        if not self._resize:
            target = next(
                (index for index, rect in enumerate(monitors, start=1) if rect.contains(event.position())),
                None,
            )
            if target is not None and target != monitor_number:
                placement["monitor"] = target
                monitor = monitors[target - 1]
                placement["x"] = 0.05
                placement["y"] = 0.05
        if self._resize:
            placement["width"] = max(0.12, min(1.0 - float(placement.get("x", 0)), float(placement.get("width", 0.5)) + delta.x() / monitor.width()))
            placement["height"] = max(0.12, min(1.0 - float(placement.get("y", 0)), float(placement.get("height", 0.5)) + delta.y() / monitor.height()))
        else:
            width = float(placement.get("width", 0.5))
            height = float(placement.get("height", 0.5))
            placement["x"] = max(0.0, min(1.0 - width, float(placement.get("x", 0)) + delta.x() / monitor.width()))
            placement["y"] = max(0.0, min(1.0 - height, float(placement.get("y", 0)) + delta.y() / monitor.height()))
        self.update()
        self.changed.emit()

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._active_index = -1
        self._resize = False
        self.update()


class _WorkspaceTask(QThread):
    completed = Signal(dict)

    def __init__(self, operation: Callable[[], dict[str, Any]], parent: QWidget) -> None:
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            result = self.operation()
        except Exception as exc:  # noqa: BLE001 - surface a UI-safe error
            result = {"ok": False, "response_text": str(exc)}
        self.completed.emit(result)


class WorkspacePage(QWidget):
    def __init__(self, store_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = WorkspaceStore(store_path)
        self.manager = WorkspaceManager(self.store)
        self._current_id = ""
        self._tasks: list[_WorkspaceTask] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 28)
        layout.setSpacing(12)
        layout.addWidget(_label("WINDOW WORKSPACES // 06", "eyebrow"))
        header = QHBoxLayout()
        headings = QVBoxLayout()
        headings.addWidget(_label("Рабочие пространства", "title"))
        headings.addWidget(_label("Нативные схемы Windows: приложения, файлы, сайты и временный рабочий стол.", "muted"))
        header.addLayout(headings)
        header.addStretch(1)
        capture_new = QPushButton("СОЗДАТЬ ИЗ ТЕКУЩИХ ОКОН")
        capture_new.setObjectName("primary")
        capture_new.clicked.connect(self._capture_new)
        header.addWidget(capture_new)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QFrame()
        left.setObjectName("workspaceListCard")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(13, 13, 13, 13)
        left_layout.addWidget(_label("РЕЖИМЫ", "eyebrow"))
        self.workspace_list = QListWidget()
        self.workspace_list.setObjectName("workspaceList")
        self.workspace_list.currentItemChanged.connect(self._selected)
        left_layout.addWidget(self.workspace_list, 1)
        self.finish_button = QPushButton("ЗАВЕРШИТЬ ТЕКУЩИЙ РЕЖИМ")
        self.finish_button.clicked.connect(self._finish)
        left_layout.addWidget(self.finish_button)
        splitter.addWidget(left)

        right = QFrame()
        right.setObjectName("workspaceEditorCard")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(18, 16, 18, 16)
        title_row = QHBoxLayout()
        title_box = QVBoxLayout()
        self.title = _label("Выберите режим", "sectionTitle")
        self.description = _label("", "muted")
        self.description.setWordWrap(True)
        title_box.addWidget(self.title)
        title_box.addWidget(self.description)
        title_row.addLayout(title_box, 1)
        self.temp_desktop = QCheckBox("Временный рабочий стол")
        title_row.addWidget(self.temp_desktop)
        right_layout.addLayout(title_row)
        self.canvas = WorkspaceCanvas()
        right_layout.addWidget(self.canvas, 2)

        resources = QHBoxLayout()
        app_column = QVBoxLayout()
        app_header = QHBoxLayout()
        app_header.addWidget(_label("ПРИЛОЖЕНИЯ", "eyebrow"))
        app_header.addStretch(1)
        add_app = QPushButton("+ приложение")
        add_app.clicked.connect(self._add_application)
        app_header.addWidget(add_app)
        app_column.addLayout(app_header)
        self.apps = QListWidget()
        self.apps.setObjectName("workspaceResources")
        app_column.addWidget(self.apps)
        resources.addLayout(app_column, 1)
        site_column = QVBoxLayout()
        site_header = QHBoxLayout()
        site_header.addWidget(_label("САЙТЫ И ФАЙЛЫ", "eyebrow"))
        site_header.addStretch(1)
        add_site = QPushButton("+ сайт")
        add_site.clicked.connect(self._add_site)
        add_file = QPushButton("+ файл")
        add_file.clicked.connect(self._add_files)
        remove_resource = QPushButton("−")
        remove_resource.setToolTip("Убрать выбранный сайт или файл")
        remove_resource.clicked.connect(self._remove_resource)
        site_header.addWidget(add_site)
        site_header.addWidget(add_file)
        site_header.addWidget(remove_resource)
        site_column.addLayout(site_header)
        self.resources = QListWidget()
        self.resources.setObjectName("workspaceResources")
        site_column.addWidget(self.resources)
        resources.addLayout(site_column, 1)
        right_layout.addLayout(resources, 1)

        actions = QHBoxLayout()
        self.status = _label("ГОТОВО", "workspaceStatus")
        actions.addWidget(self.status, 1)
        self.capture_button = QPushButton("Снять текущее расположение")
        self.capture_button.clicked.connect(self._capture_current)
        self.save_button = QPushButton("СОХРАНИТЬ ИЗМЕНЕНИЯ")
        self.save_button.clicked.connect(self._save)
        self.launch_button = QPushButton("ЗАПУСТИТЬ РЕЖИМ")
        self.launch_button.setObjectName("primary")
        self.launch_button.clicked.connect(self._launch)
        actions.addWidget(self.capture_button)
        actions.addWidget(self.save_button)
        actions.addWidget(self.launch_button)
        right_layout.addLayout(actions)
        splitter.addWidget(right)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter, 1)

    def refresh(self, selected_id: str | None = None) -> None:
        selected = selected_id or self._current_id
        rows = self.store.list()
        self.workspace_list.blockSignals(True)
        self.workspace_list.clear()
        target = 0
        for index, workspace in enumerate(rows):
            apps = sum(bool(item.get("enabled", True)) for item in workspace.get("applications", []))
            item = QListWidgetItem(f"{workspace['name']}\n{apps} приложений")
            item.setData(Qt.ItemDataRole.UserRole, workspace["id"])
            self.workspace_list.addItem(item)
            if workspace["id"] == selected:
                target = index
        self.workspace_list.blockSignals(False)
        if rows:
            self.workspace_list.setCurrentRow(target)
        active = self.manager.active_status()
        self.finish_button.setEnabled(active is not None)
        if active is None:
            self.finish_button.setText("ЗАВЕРШИТЬ ТЕКУЩИЙ РЕЖИМ")
        else:
            self.finish_button.setText(
                f"ЗАВЕРШИТЬ: {active.get('workspace_name', 'РЕЖИМ')}"
            )

    def _selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        self._current_id = str(current.data(Qt.ItemDataRole.UserRole))
        workspace = self.store.get(self._current_id)
        if workspace is None:
            return
        self.title.setText(str(workspace["name"]))
        recommendations = workspace.get("recommendations", [])
        suffix = "\nМожно добавить: " + ", ".join(recommendations) if recommendations else ""
        self.description.setText(str(workspace.get("description", "")) + suffix)
        self.temp_desktop.setChecked(bool(workspace.get("temporary_desktop", True)))
        self.canvas.set_workspace(workspace)
        self.apps.clear()
        for app in workspace.get("applications", []):
            installed = resolve_application(str(app.get("query", ""))) is not None
            item = QListWidgetItem(
                f"{app.get('label', app.get('query', 'Приложение'))}  {'● установлено' if installed else '○ не найдено'}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(app.get("key", "")))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if app.get("enabled", True) else Qt.CheckState.Unchecked)
            self.apps.addItem(item)
        self.resources.clear()
        for site in workspace.get("sites", []):
            item = QListWidgetItem(f"WEB  //  {site.get('label', site.get('url', ''))}")
            item.setData(Qt.ItemDataRole.UserRole, ("site", str(site.get("key", ""))))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if site.get("enabled", True) else Qt.CheckState.Unchecked)
            self.resources.addItem(item)
        for path in workspace.get("files", []):
            item = QListWidgetItem(f"FILE //  {path}")
            item.setData(Qt.ItemDataRole.UserRole, ("file", str(path)))
            self.resources.addItem(item)

    def _workspace_from_controls(self) -> dict[str, Any] | None:
        workspace = self.canvas.workspace()
        if not workspace:
            return None
        workspace["temporary_desktop"] = self.temp_desktop.isChecked()
        app_states = {
            str(self.apps.item(index).data(Qt.ItemDataRole.UserRole)): self.apps.item(index).checkState() == Qt.CheckState.Checked
            for index in range(self.apps.count())
        }
        for app in workspace.get("applications", []):
            app["enabled"] = app_states.get(str(app.get("key", "")), bool(app.get("enabled", True)))
        site_states = {}
        files = []
        for index in range(self.resources.count()):
            item = self.resources.item(index)
            kind, value = item.data(Qt.ItemDataRole.UserRole)
            if kind == "site":
                site_states[value] = item.checkState() == Qt.CheckState.Checked
            else:
                files.append(value)
        for site in workspace.get("sites", []):
            site["enabled"] = site_states.get(str(site.get("key", "")), bool(site.get("enabled", True)))
        workspace["files"] = files
        return workspace

    def _save(self) -> None:
        workspace = self._workspace_from_controls()
        if workspace is None:
            return
        result = self.manager.update_workspace(workspace)
        self.status.setText(result["response_text"].upper())
        self.refresh(str(workspace["id"]))

    def _launch(self) -> None:
        self._save()
        self._run(lambda: self.manager.launch(self._current_id), retry_workspace=self._current_id)

    def _finish(self) -> None:
        self._run(self.manager.finish)

    def _capture_current(self) -> None:
        workspace = self.store.get(self._current_id)
        if workspace is None:
            return
        self._run(lambda: self.manager.capture(str(workspace["name"])), refresh_id=self._current_id)

    def _capture_new(self) -> None:
        name, accepted = QInputDialog.getText(self, "Новое пространство", "Название:")
        if accepted and name.strip():
            self._run(lambda: self.manager.capture(name.strip()), refresh_id="")

    def _add_site(self) -> None:
        workspace = self.canvas.workspace()
        if not workspace:
            return
        url, accepted = QInputDialog.getText(self, "Добавить сайт", "Адрес https://…:")
        if not accepted or not url.strip():
            return
        label, label_ok = QInputDialog.getText(self, "Название сайта", "Название:")
        if not label_ok:
            return
        key = "site-" + str(len(workspace.get("sites", [])) + 1)
        workspace.setdefault("sites", []).append(
            {"key": key, "label": label.strip() or url.strip(), "url": url.strip(), "enabled": True}
        )
        self.canvas.set_workspace(workspace)
        self.manager.update_workspace(workspace)
        self._selected(self.workspace_list.currentItem(), None)

    def _add_application(self) -> None:
        workspace = self.canvas.workspace()
        if not workspace:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Добавить приложение",
            "Название установленной программы:",
        )
        if not accepted or not name.strip():
            return
        spec = resolve_application(name.strip())
        if spec is None:
            QMessageBox.warning(
                self,
                "Приложение не найдено",
                "Сначала установите или запустите приложение, затем повторите поиск.",
            )
            return
        key_base = "app-" + "".join(
            character for character in spec.name.casefold() if character.isalnum()
        )[:24]
        known = {str(item.get("key")) for item in workspace.get("applications", [])}
        key = key_base
        suffix = 2
        while key in known:
            key = f"{key_base}-{suffix}"
            suffix += 1
        workspace.setdefault("applications", []).append(
            {
                "key": key,
                "query": spec.name,
                "label": spec.display_name,
                "optional": True,
                "enabled": True,
                "strategy": "normal",
            }
        )
        count = len(workspace["applications"])
        workspace.setdefault("placements", []).append(
            {
                "key": key,
                "monitor": 1,
                "x": 0.68,
                "y": min(0.72, max(0.0, (count - 2) * 0.18)),
                "width": 0.32,
                "height": 0.28,
            }
        )
        self.canvas.set_workspace(workspace)
        self.manager.update_workspace(workspace)
        self._selected(self.workspace_list.currentItem(), None)

    def _add_files(self) -> None:
        workspace = self.canvas.workspace()
        if not workspace:
            return
        paths, _filter = QFileDialog.getOpenFileNames(self, "Добавить файлы в рабочее пространство")
        if not paths:
            return
        current = list(workspace.get("files", []))
        current.extend(path for path in paths if path not in current)
        workspace["files"] = current
        self.canvas.set_workspace(workspace)
        self.manager.update_workspace(workspace)
        self._selected(self.workspace_list.currentItem(), None)

    def _remove_resource(self) -> None:
        item = self.resources.currentItem()
        workspace = self.canvas.workspace()
        if item is None or not workspace:
            return
        kind, value = item.data(Qt.ItemDataRole.UserRole)
        if kind == "site":
            workspace["sites"] = [
                site for site in workspace.get("sites", []) if str(site.get("key")) != value
            ]
        else:
            workspace["files"] = [
                path for path in workspace.get("files", []) if str(path) != value
            ]
        self.canvas.set_workspace(workspace)
        self.manager.update_workspace(workspace)
        self._selected(self.workspace_list.currentItem(), None)

    def _run(
        self,
        operation: Callable[[], dict[str, Any]],
        *,
        retry_workspace: str = "",
        refresh_id: str | None = None,
    ) -> None:
        self._set_busy(True)
        task = _WorkspaceTask(operation, self)
        self._tasks.append(task)

        def completed(result: dict[str, Any]) -> None:
            self._tasks.remove(task)
            self._set_busy(False)
            if result.get("confirmation_required") and retry_workspace:
                if QMessageBox.question(self, "Закрыть игры", str(result.get("response_text", ""))) == QMessageBox.StandardButton.Yes:
                    self._run(lambda: self.manager.launch(retry_workspace, confirmed=True))
                return
            self.status.setText(str(result.get("response_text", "Готово")).upper())
            if not result.get("ok"):
                QMessageBox.warning(self, "Рабочее пространство", str(result.get("response_text", "Ошибка")))
            if result.get("elevation_required"):
                self._offer_elevated_restart()
            if refresh_id is not None:
                workspace = result.get("workspace") or {}
                selected = refresh_id or (workspace.get("id") if isinstance(workspace, dict) else "")
                self.refresh(str(selected or self._current_id))
            else:
                self.refresh(self._current_id)
        task.completed.connect(completed)
        task.finished.connect(task.deleteLater)
        task.start()

    def _offer_elevated_restart(self) -> None:
        if QMessageBox.question(
            self,
            "Нужны права администратора",
            "Защищённое окно нельзя переместить с обычными правами. Перезапустить Control Center от имени администратора?",
        ) != QMessageBox.StandardButton.Yes:
            return
        main_window = self.window()
        runtime = getattr(main_window, "runtime", None)
        if runtime is not None and bool(getattr(runtime, "running", False)):
            QMessageBox.information(
                self,
                "Сначала остановите Jarvis",
                "Остановите Jarvis кнопкой в Control Center, затем повторите запуск режима. Это предотвращает два одновременных голосовых процесса.",
            )
            return
        root = Path(__file__).resolve().parents[1]
        script = root / "jarvis_control.py"
        parameters = subprocess.list2cmdline([str(script)])
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            parameters,
            str(root),
            1,
        )
        if int(result) <= 32:
            QMessageBox.warning(self, "Повышение прав", "Windows отклонила перезапуск от имени администратора.")
            return
        main_window.close()

    def _set_busy(self, busy: bool) -> None:
        self.launch_button.setEnabled(not busy)
        self.capture_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        self.finish_button.setEnabled(not busy)
        if busy:
            self.status.setText("ВЫПОЛНЯЮ…")

    def shutdown(self) -> None:
        for task in list(self._tasks):
            task.wait(20_000)
        self.manager.shutdown()
