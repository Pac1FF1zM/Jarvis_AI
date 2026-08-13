"""Tests for the config loader hardening (fix #5).

#5 — the loader warns about unknown module keys at load time, and ``module()``
warns the first time an unrecognized name is looked up. Typos must not be
masked by a silent default.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from core.config_loader import Config, EXPECTED_MODULE_NAMES, load_config

# Warning format from config_loader.load_config():
#   "Unknown module key '<name>' in config.yaml 'modules:' — not one of [...]"
# Match just the quoted <name> so we can assert on the *unknown key itself*,
# not on substrings of the full rendered message (which legitimately lists every
# expected name — including 'wake_word' — as context).
_UNKNOWN_KEY_RE = re.compile(r"Unknown module key '([^']+)'")


def _write_config(tmp_path: Path, body: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_warns_on_unknown_module_key_at_load(tmp_path, caplog):
    """A typo'd key under modules: must produce a warning at load time."""
    cfg_path = _write_config(
        tmp_path,
        """
modules:
  wake_word:
    enabled: true
  sttt:                 # typo
    enabled: true
""",
    )
    with caplog.at_level(logging.WARNING, logger="jarvis.config"):
        cfg = load_config(cfg_path)

    # FIX A: parse the actual unknown key out of each "Unknown module key '<name>'"
    # warning, rather than substring-matching the whole rendered message. The
    # implementation correctly lists every EXPECTED_MODULE_NAME (including
    # 'wake_word') as context, so a naive substring check would false-positive.
    unknown_keys: list[str] = []
    for record in caplog.records:
        match = _UNKNOWN_KEY_RE.search(record.getMessage())
        if match:
            unknown_keys.append(match.group(1))

    # The typo'd key must be flagged.
    assert "sttt" in unknown_keys, (
        "expected a warning for unknown module key 'sttt'"
    )
    # Known keys must not be reported as unknown. Checking the parsed key
    # (not the message text) is what makes this assertion sound.
    assert "wake_word" not in unknown_keys, (
        "'wake_word' should not be flagged as an unknown key"
    )
    # Sanity: typo still parses into a ModuleConfig (no crash).
    assert cfg.module("sttt").enabled is True


def test_module_lookup_warns_once_for_unknown_name(caplog):
    """Looking up a never-configured name warns exactly once."""
    cfg = Config()
    with caplog.at_level(logging.WARNING, logger="jarvis.config"):
        first = cfg.module("totally_made_up")
        second = cfg.module("totally_made_up")
    assert first.enabled is True and first.device == "cpu"  # safe default
    # Exactly one warning record for that name.
    warnings = [r for r in caplog.records if "totally_made_up" in r.message]
    assert len(warnings) == 1, "should warn exactly once per unknown name (fix #5)"


def test_known_names_do_not_warn_on_lookup(caplog):
    """Pre-seeded config sections must not warn on lookup."""
    cfg = Config(modules={})  # no sections
    with caplog.at_level(logging.WARNING, logger="jarvis.config"):
        # EXPECTED_MODULE_NAMES are the canonical names; even if absent from
        # config they represent "intentionally defaulted" only if explicitly
        # marked so. Here we just assert the mechanism: a configured name
        # produces no warning.
        pass
    from core.config_loader import ModuleConfig

    cfg2 = Config(modules={"stt": ModuleConfig(enabled=False, device="cuda")})
    with caplog.at_level(logging.WARNING, logger="jarvis.config"):
        cfg2.module("stt")
    assert "stt" not in caplog.text


def test_expected_module_names_is_canonical_set():
    """Guardrail: the known set includes every canonical pipeline module."""
    assert EXPECTED_MODULE_NAMES == frozenset(
        {"gesture", "wake_word", "stt", "nlu", "jsc_shadow", "llm", "tts"}
    )


def test_installed_runtime_keeps_user_state_outside_application_dir(
    tmp_path, monkeypatch
):
    config_path = _write_config(
        tmp_path,
        "modules:\n  gesture:\n    enabled: true\n    params:\n      log_dir: logs/gestures\n",
    )
    data_dir = tmp_path / "user-data"
    monkeypatch.setenv("JARVIS_DATA_DIR", str(data_dir))

    config = load_config(config_path)

    assert config.logging["log_file"] == str(data_dir / "logs" / "jarvis.log")
    assert config.logging["session_log_dir"] == str(
        data_dir / "logs" / "sessions"
    )
    assert config.memory["db_path"] == str(data_dir / "memory.db")
    assert config.reminders["db_path"] == str(data_dir / "reminders.db")
    assert config.module("gesture").params["log_dir"] == str(
        data_dir / "logs" / "gestures"
    )
