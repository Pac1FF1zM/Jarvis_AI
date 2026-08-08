"""Regression coverage for the desktop Control Center boundaries."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from control_center.config_model import ConfigEditorError, ConfigRepository
from control_center.doctor_report import parse_doctor_output


ROOT = Path(__file__).resolve().parents[1]


def test_config_form_preserves_comments_and_unrelated_values(tmp_path):
    path = tmp_path / "config.yaml"
    original = (ROOT / "config.yaml").read_text(encoding="utf-8")
    path.write_text(original, encoding="utf-8")
    repository = ConfigRepository(path)

    repository.save_form(
        {
            "modules.gesture.enabled": False,
            "modules.wake_word.params.wake_phrase_threshold": 0.42,
        }
    )

    updated = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(updated)
    assert updated.startswith("# Jarvis configuration")
    assert "# DirectShow avoids slow/hanging" in updated
    assert parsed["modules"]["gesture"]["enabled"] is False
    assert parsed["modules"]["wake_word"]["params"]["wake_phrase_threshold"] == 0.42
    assert parsed["modules"]["nlu"]["model"] == "models/nlu_manager_finetuned.pt"


def test_raw_config_is_validated_before_atomic_save(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("modules: {}\n", encoding="utf-8")
    repository = ConfigRepository(path)

    with pytest.raises(ConfigEditorError, match="YAML содержит ошибку"):
        repository.save_raw("modules: [\n")

    assert path.read_text(encoding="utf-8") == "modules: {}\n"


def test_doctor_parser_finds_structured_report_after_process_noise():
    report = {"overall": "degraded", "checks": [], "counts": {"warn": 1}}
    output = "runtime prelude\n" + json.dumps(report, ensure_ascii=False) + "\n"

    assert parse_doctor_output(output) == report


def test_control_center_window_smoke_without_starting_runtime(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt = pytest.importorskip("PySide6.QtWidgets")
    from control_center.theme import STYLE_SHEET
    from control_center.window import ControlCenterWindow

    app = qt.QApplication.instance() or qt.QApplication([])
    app.setStyleSheet(STYLE_SHEET)
    window = ControlCenterWindow(ROOT, ROOT / "config.yaml")
    window.show()
    app.processEvents()

    assert window.pages.count() == 8
    assert window.dashboard_hero.objectName() == "heroCard"
    assert len(window.dashboard_cards) == 3
    assert window.nav_indicator.isVisible()
    assert window.gesture_view.preview_port > 0
    assert window.runtime.running is False
    assert window.runtime_status.text() == "OFFLINE"
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()

    window._set_runtime_state("starting")
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()

    window._set_runtime_state("running")
    assert not window.start_button.isEnabled()
    assert window.stop_button.isEnabled()
    assert window.hero_state_label.text() == "СИСТЕМА АКТИВНА"

    window._set_runtime_state("stopped")
    assert window.start_button.isEnabled()
    assert not window.stop_button.isEnabled()
    assert window.hero_state_label.text() == "СТАНДБАЙ"

    from modules.gesture_ui import EmbeddedGesturePreview

    preview = EmbeddedGesturePreview(
        window.gesture_view.preview_port,
        window.gesture_view.preview_token,
    )
    preview.open(object())
    app.processEvents()
    assert window.pages.currentIndex() == 4
    assert window.gesture_view.mode_badge.text() == "●  АКТИВЕН"
    preview.close()
    app.processEvents()

    window._show_page(1)
    app.processEvents()
    assert window.pages.currentIndex() == 1
    assert window._nav_buttons[1].isChecked()

    window.close()
