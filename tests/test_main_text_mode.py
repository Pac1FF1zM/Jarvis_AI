"""Integration checks for the audio-free ``main.py --text`` surface."""
from __future__ import annotations

import logging
import sys

import pytest

from core.config_loader import Config
from core.event_bus import Event
from core.orchestrator import State
import main as main_module
from modules.stt import STTModule
from modules.tts import TTSModule
from modules.wake_word import WakeWordModule


def test_help_is_short_user_facing_and_does_not_start_runtime(monkeypatch, capsys):
    async def forbidden_run(*args, **kwargs):
        raise AssertionError("runtime initialized while rendering --help")

    monkeypatch.setattr(main_module, "run_pipeline", forbidden_run)
    monkeypatch.setattr(sys, "argv", ["main.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Jarvis — локальный голосовой помощник" in output
    assert "Без параметров запускается голосовой режим" in output
    assert "--text КОМАНДА" in output
    assert "--doctor" in output
    assert "--calibrate-voice" in output
    assert "--profiles" in output
    assert "Что умеет:" in output
    assert "составные команды" in output
    assert len(output.splitlines()) <= 40


def test_setup_logging_creates_shareable_per_session_transcript(tmp_path):
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        session_path = main_module.setup_logging(
            Config(
                logging={
                    "level": "INFO",
                    "log_file": str(tmp_path / "jarvis.log"),
                    "session_log_dir": str(tmp_path / "sessions"),
                    "console": False,
                }
            )
        )
        logging.getLogger("jarvis.test").info("TRANSCRIPTION_READY text='тест'")
        for handler in root.handlers:
            handler.flush()

        assert session_path.parent == tmp_path / "sessions"
        assert session_path.name.startswith("jarvis_session_")
        assert session_path.suffix == ".txt"
        transcript = session_path.read_text(encoding="utf-8")
        assert "SESSION_LOG_READY" in transcript
        assert "TRANSCRIPTION_READY text='тест'" in transcript
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        root.setLevel(original_level)
        for handler in original_handlers:
            root.addHandler(handler)


async def test_text_mode_never_starts_voice_modules(monkeypatch, capsys):
    async def forbidden_start(self, bus):
        raise AssertionError("voice module initialized during --text mode")

    monkeypatch.setattr(WakeWordModule, "start", forbidden_start)
    monkeypatch.setattr(STTModule, "start", forbidden_start)
    monkeypatch.setattr(TTSModule, "start", forbidden_start)

    await main_module.run_pipeline(
        "config.yaml", text_input="какие приложения ты можешь открыть"
    )

    captured = capsys.readouterr()
    assert "Jarvis: Я могу открыть:" in captured.out


async def test_text_mode_waiter_surfaces_recovered_interaction_failure():
    completion = main_module._TraceCompletion()
    await completion.record(
        Event(
            "interaction_completed",
            {
                "state": "IDLE",
                "ok": False,
                "reason": "handler_exception",
                "failed_state": "THINKING",
            },
            trace_id="failed-text",
        )
    )

    class IdleOrchestrator:
        state = State.IDLE

    with pytest.raises(RuntimeError, match="failed and recovered to IDLE"):
        await completion.wait("failed-text", IdleOrchestrator(), timeout=0.1)


async def test_text_mode_creates_lists_and_cancels_persistent_reminder(capsys):
    await main_module.run_pipeline(
        "config.yaml", text_input="через 10 минут напомни проверить духовку"
    )
    created_output = capsys.readouterr().out
    assert "Напоминание номер 1 установлено" in created_output
    assert "проверить духовку" in created_output

    await main_module.run_pipeline(
        "config.yaml", text_input="покажи мои напоминания"
    )
    listed_output = capsys.readouterr().out
    assert "Активные напоминания: номер 1" in listed_output
    assert "проверить духовку" in listed_output

    await main_module.run_pipeline(
        "config.yaml", text_input="отмени напоминание номер 1"
    )
    cancelled_output = capsys.readouterr().out
    assert "Напоминание номер 1 отменено" in cancelled_output

    await main_module.run_pipeline(
        "config.yaml", text_input="покажи мои напоминания"
    )
    assert "У вас нет активных напоминаний" in capsys.readouterr().out
