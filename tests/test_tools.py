"""Tests for the tools subsystem: registry auto-discovery + the two tools."""
from __future__ import annotations

import pytest

from core.event_bus import EventBus
from modules.reminders import ReminderScheduler
from tools.get_current_time import execute as execute_time
from tools.get_current_time import TOOL_SCHEMA as TIME_SCHEMA
from tools.set_reminder import execute as execute_reminder
from tools.open_application import execute as execute_open_application
from tools.list_applications import execute as execute_list_applications
import tools.open_application as open_application_module
import tools._applications as applications_module
import tools.browser_control as browser_control_module
from tools.browser_control import execute as execute_browser_control
from tools._applications import resolve_application
from tools.registry import ToolRegistry


def test_registry_auto_discovers_all_tools():
    reg = ToolRegistry()
    reg.discover("tools")
    names = reg.names()
    assert "get_current_time" in names
    assert "set_reminder" in names
    assert "list_reminders" in names
    assert "cancel_reminder" in names
    assert "open_application" in names
    assert "list_applications" in names
    assert "browser_control" in names
    assert "file_control" in names
    assert "system_control" in names
    assert "window_control" in names


def test_registry_does_not_register_itself():
    reg = ToolRegistry()
    reg.discover("tools")
    # registry.py must not register itself as a tool.
    assert "registry" not in reg.names()


async def test_get_current_time_returns_iso():
    result = await execute_time({})
    assert "iso" in result
    assert "weekday" in result
    # ISO should look like a date.
    assert len(result["iso"]) >= 10


async def test_set_reminder_does_not_claim_unscheduled_work():
    result = await execute_reminder({"minutes": 10, "message": "stand up"})
    assert result["ok"] is False
    assert result["scheduled"] is False
    assert result["error"] == "scheduler_not_configured"
    assert "не запущен" in result["response_text"]


async def test_registry_uses_configured_persistent_reminder_service(tmp_path):
    scheduler = ReminderScheduler(tmp_path / "tools-reminders.db")
    await scheduler.start(EventBus(), delivery_enabled=False)
    registry = ToolRegistry({"reminder_scheduler": scheduler})
    registry.discover("tools")

    created = await registry.execute(
        "set_reminder", {"minutes": 10, "message": "проверить духовку"}
    )
    reminder_id = created["reminder"]["id"]
    listed = await registry.execute("list_reminders", {})
    cancelled = await registry.execute(
        "cancel_reminder", {"reminder_id": reminder_id}
    )

    assert created["ok"] is True
    assert [item["id"] for item in listed["reminders"]] == [reminder_id]
    assert cancelled["ok"] is True
    assert cancelled["reminder"]["status"] == "cancelled"
    assert await scheduler.list_pending() == []
    await scheduler.stop()


async def test_registry_execute_runs_tool():
    reg = ToolRegistry()
    reg.discover("tools")
    out = await reg.execute("get_current_time", {})
    assert "iso" in out


async def test_registry_execute_unknown_tool_raises():
    reg = ToolRegistry()
    reg.discover("tools")
    with pytest.raises(KeyError):
        await reg.execute("does_not_exist", {})


def test_schemas_match_documented_shape():
    """Each schema must carry name, description, parameters (README §6 contract)."""
    reg = ToolRegistry()
    reg.discover("tools")
    for schema in reg.schemas():
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"


def test_time_schema_has_no_required_params():
    assert TIME_SCHEMA["parameters"].get("required", []) == []


async def test_open_application_launches_only_allowlisted_target(monkeypatch):
    launched = []

    def fake_launch(spec):
        launched.append(spec.name)
        return 4242

    monkeypatch.setattr(open_application_module, "launch_application", fake_launch)
    result = await execute_open_application({"application": "калькулятор"})
    assert result["ok"] is True
    assert result["application"] == "calculator"
    assert result["pid"] == 4242
    assert launched == ["calculator"]


async def test_open_application_rejects_unknown_target(monkeypatch):
    def must_not_launch(spec):
        raise AssertionError("unknown applications must never be launched")

    monkeypatch.setattr(open_application_module, "launch_application", must_not_launch)
    monkeypatch.setattr(applications_module, "discover_installed_applications", lambda: ())
    result = await execute_open_application({"application": "powershell"})
    assert result["ok"] is False
    assert result["error"] == "application_not_allowed"
    assert "Калькулятор" in result["supported"]


async def test_list_applications_returns_safe_allowlist():
    result = await execute_list_applications({})
    assert result["ok"] is True
    assert "Калькулятор" in result["applications"]
    assert "Блокнот" in result["applications"]
    assert "PowerShell" not in result["applications"]


def test_paint_uses_fixed_windows_uri_instead_of_shell():
    paint = resolve_application("paint")
    assert paint is not None
    assert paint.command is None
    assert paint.uri == "ms-paint:"


def test_discord_uses_fixed_windows_uri_instead_of_shell():
    discord = resolve_application("дисорд")
    assert discord is not None
    assert discord.command is None
    assert discord.uri == "discord://"


def test_running_discord_is_activated_without_secondary_instance(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(
        applications_module, "_activate_windows_process_window", lambda _name: True
    )
    monkeypatch.setattr(applications_module, "_open_windows_uri_detached", opened.append)
    discord = resolve_application("дискорд")

    assert discord is not None
    applications_module.launch_application(discord)

    assert opened == []


def test_cold_discord_launch_waits_for_visible_window(monkeypatch):
    opened: list[str] = []
    sleeps: list[float] = []
    activation_results = iter((False, False, True, True))
    monkeypatch.setattr(
        applications_module,
        "_activate_windows_process_window",
        lambda _name: next(activation_results),
    )
    monkeypatch.setattr(applications_module, "_open_windows_uri_detached", opened.append)
    monkeypatch.setattr(applications_module.time, "sleep", sleeps.append)
    discord = resolve_application("discord")

    assert discord is not None
    applications_module.launch_application(discord)

    assert opened == ["discord://"]
    assert applications_module._DISCORD_COLD_START_SETTLE_SECONDS in sleeps


def test_discord_uri_launch_does_not_inherit_jarvis_console(monkeypatch):
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return object()

    monkeypatch.setattr(applications_module.subprocess, "Popen", fake_popen)

    applications_module._open_windows_uri_detached("discord://")

    command, kwargs = calls[0]
    assert command == ["explorer.exe", "discord://"]
    assert kwargs["stdin"] is applications_module.subprocess.DEVNULL
    assert kwargs["stdout"] is applications_module.subprocess.DEVNULL
    assert kwargs["stderr"] is applications_module.subprocess.DEVNULL


def test_browser_uses_windows_default_https_handler(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(applications_module.os, "startfile", opened.append)
    browser = resolve_application("браузер")

    assert browser is not None
    assert browser.url == "https://www.google.com/"
    assert browser.command is None

    applications_module.launch_application(browser)

    assert opened == ["https://www.google.com/"]


@pytest.mark.parametrize(
    ("heard", "expected"),
    [
        ("блокноты", "notepad"),
        ("black note", "notepad"),
        ("к алкулятор", "calculator"),
        ("пеинт", "paint"),
        ("пейнт", "paint"),
        ("дисорд", "discord"),
        ("дискод", "discord"),
        ("дискорти", "discord"),
    ],
)
def test_allowlist_resolves_common_russian_whisper_variants(heard, expected):
    resolved = resolve_application(heard)
    assert resolved is not None
    assert resolved.name == expected


def test_fuzzy_resolver_still_rejects_unrelated_or_dangerous_names(monkeypatch):
    monkeypatch.setattr(applications_module, "discover_installed_applications", lambda: ())
    for name in ("powershell", "командная строка", "дисковод"):
        assert resolve_application(name) is None


def test_resolver_accepts_exact_windows_discovered_application(monkeypatch):
    discovered = applications_module.ApplicationSpec(
        name="Example App",
        display_name="Example App",
        path=r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Example App.lnk",
        aliases=("пример",),
        discovered=True,
    )
    monkeypatch.setattr(
        applications_module, "discover_installed_applications", lambda: (discovered,)
    )

    assert resolve_application("example app") == discovered
    assert resolve_application("пример") == discovered


async def test_browser_shortcut_is_not_sent_to_an_unrelated_foreground_window(monkeypatch):
    sent: list[tuple[int, ...]] = []
    monkeypatch.setattr(browser_control_module, "activate_default_browser", lambda: False)
    monkeypatch.setattr(browser_control_module, "send_hotkey", lambda *keys: sent.append(keys))

    result = await execute_browser_control({"action": "close_tab"})

    assert result["ok"] is False
    assert result["error"] == "browser_not_open"
    assert sent == []
