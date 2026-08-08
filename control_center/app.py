"""Application bootstrap for the Jarvis Control Center."""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from control_center.theme import BACKGROUND, STYLE_SHEET, TEXT
from control_center.window import ControlCenterWindow, animate_window_entrance


def run(project_root: str | Path | None = None, config_path: str | Path | None = None) -> int:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    selected_config = Path(config_path or root / "config.yaml").resolve()
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Jarvis Control Center")
    app.setOrganizationName("Jarvis")
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    app.setPalette(palette)
    app.setStyleSheet(STYLE_SHEET)
    window = ControlCenterWindow(root, selected_config)
    window.show()
    animation = animate_window_entrance(window)
    window._entrance_animation = animation  # keep animation alive for its duration
    return app.exec()
