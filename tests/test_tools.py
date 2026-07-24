"""Tests for the tools subsystem: registry auto-discovery + the two tools."""
from __future__ import annotations

import pytest

from tools.get_current_time import execute as execute_time
from tools.get_current_time import TOOL_SCHEMA as TIME_SCHEMA
from tools.set_reminder import execute as execute_reminder
from tools.open_application import execute as execute_open_application
from tools.list_applications import execute as execute_list_applications
import tools.open_application as open_application_module
import tools._applications as applications_module
from tools._applications import resolve_application
from tools.registry import ToolRegistry


def test_registry_auto_discovers_all_tools():
    reg = ToolRegistry()
    reg.discover("tools")
    names = reg.names()
    assert "get_current_time" in names
    assert "set_reminder" in names
    assert "open_application" in names
    assert "list_applications" in names


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
    assert result["error"] == "scheduler_not_implemented"
    assert result["minutes"] == 10
    assert result["message"] == "stand up"
    assert "ничего не запланировал" in result["response_text"]


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
    ],
)
def test_allowlist_resolves_common_russian_whisper_variants(heard, expected):
    resolved = resolve_application(heard)
    assert resolved is not None
    assert resolved.name == expected


def test_fuzzy_resolver_still_rejects_unrelated_or_dangerous_names():
    for name in ("powershell", "командная строка", "дисковод", "steam"):
        assert resolve_application(name) is None
