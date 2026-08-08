"""Workspace persistence, routing and destructive-transition safety."""
from __future__ import annotations

from pathlib import Path

from core.workspace_manager import WorkspaceManager
from memory.workspaces import WorkspaceStore, canonical_workspace_name
from modules.command_router import RoutedAction, route_explicit_command
from tools._windows import MonitorInfo, WindowInfo, WindowState


def test_default_workspaces_are_created_and_editable(tmp_path):
    store = WorkspaceStore(tmp_path / "workspaces.json")
    rows = store.list()

    assert {row["id"] for row in rows} >= {"programming", "gaming", "study"}
    assert store.get("игровой")["id"] == "gaming"
    assert store.get("учёба")["id"] == "study"
    programming = store.get("programming")
    assert programming is not None
    assert {site["key"] for site in programming["sites"]} == {
        "chatgpt",
        "claude",
        "deepseek",
    }
    programming["temporary_desktop"] = False
    store.save(programming)
    assert store.get("программирование")["temporary_desktop"] is False


def test_workspace_aliases_and_voice_commands_are_deterministic():
    assert canonical_workspace_name("Учёба") == "study"
    assert route_explicit_command("включи игровой режим") == RoutedAction(
        "workspace_control", {"action": "launch", "workspace": "gaming"}
    )
    assert route_explicit_command("запусти режим программирование") == RoutedAction(
        "workspace_control", {"action": "launch", "workspace": "programming"}
    )
    assert route_explicit_command("сохрани это расположение как работа") == RoutedAction(
        "workspace_control", {"action": "capture", "workspace": "работа"}
    )
    assert route_explicit_command("выйди из режима игры") == RoutedAction(
        "workspace_control", {"action": "finish"}
    )
    assert route_explicit_command("Discord направо") == RoutedAction(
        "window_control",
        {"action": "move", "window": "discord", "placement": "right"},
    )
    assert route_explicit_command("сделай Discord шире") == RoutedAction(
        "window_control",
        {"action": "resize_relative", "direction": "wider", "window": "discord"},
    )


def test_capture_saves_monitor_relative_geometry(tmp_path, monkeypatch):
    from core import workspace_manager as module

    window = WindowInfo(100, "Code — project", 200, "Code.exe", r"C:\Code.exe")
    monitor = MonitorInfo(1, 1, 0, 0, 1920, 1040, True, "DISPLAY1")
    state = WindowState(100, window.title, 0, 0, 960, 1040)
    monkeypatch.setattr(module, "list_windows", lambda: [window])
    monkeypatch.setattr(module, "monitor_for_window", lambda _handle: monitor)
    monkeypatch.setattr(module, "capture_window_state", lambda _handle, _title: state)
    monkeypatch.setattr(module.WorkspaceManager, "_application_for_window", staticmethod(lambda _row: None))

    manager = WorkspaceManager(WorkspaceStore(tmp_path / "workspaces.json"))
    result = manager.capture("Моя работа")

    assert result["ok"] is True
    saved = manager.store.get("Моя работа")
    assert saved is not None
    placement = saved["placements"][0]
    assert placement["x"] == 0.0
    assert placement["width"] == 0.5
    assert placement["height"] == 1.0


def test_study_mode_lists_games_before_any_side_effect(tmp_path, monkeypatch):
    manager = WorkspaceManager(WorkspaceStore(tmp_path / "workspaces.json"))
    game = WindowInfo(
        99,
        "Example Game",
        100,
        "game.exe",
        r"C:\Program Files\Steam\steamapps\common\Example\game.exe",
    )
    monkeypatch.setattr(manager, "_game_windows", lambda: [game])

    result = manager.launch("study")

    assert result["confirmation_required"] is True
    assert result["confirmation"]["params"]["confirmed"] is True
    assert "Example Game" in result["response_text"]


def test_launch_reuses_existing_application_without_duplicate(tmp_path, monkeypatch):
    from core import workspace_manager as module

    store = WorkspaceStore(tmp_path / "workspaces.json")
    store.save(
        {
            "id": "reuse-test",
            "name": "Reuse test",
            "temporary_desktop": False,
            "applications": [
                {"key": "code", "query": "Visual Studio Code", "label": "VS Code", "enabled": True}
            ],
            "sites": [],
            "files": [],
            "placements": [{"key": "code", "monitor": 1, "x": 0, "y": 0, "width": 1, "height": 1}],
            "close_groups": [],
        }
    )
    manager = WorkspaceManager(store)
    window = WindowInfo(100, "Code — project", 200, "Code.exe", r"C:\Code.exe")
    monkeypatch.setattr(module, "list_windows", lambda: [window])
    monkeypatch.setattr(manager, "_matching_workspace_windows", lambda _workspace: {"code": window})
    monkeypatch.setattr(module, "capture_window_state", lambda *_args: WindowState(100, window.title, 0, 0, 800, 600))
    launched = []
    monkeypatch.setattr(manager, "_launch_workspace_application", lambda app: launched.append(app))
    monkeypatch.setattr(manager, "_apply_placements", lambda *_args: (1, []))

    result = manager.launch("reuse-test")

    assert result["ok"] is True
    assert result["reused"] == ["VS Code"]
    assert launched == []


def test_workspace_ui_starts_with_three_presets(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    import pytest

    qt = pytest.importorskip("PySide6.QtWidgets")
    from control_center.workspace_page import WorkspacePage

    app = qt.QApplication.instance() or qt.QApplication([])
    page = WorkspacePage(tmp_path / "workspaces.json")
    page.show()
    app.processEvents()

    assert page.workspace_list.count() == 3
    assert page.title.text() == "Игры"  # presets are sorted alphabetically for the user
    assert page.canvas.workspace()["placements"]
    page.close()
