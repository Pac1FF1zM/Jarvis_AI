"""Integration checks for the audio-free ``main.py --text`` surface."""
from __future__ import annotations

import logging

from core.config_loader import Config
import main as main_module


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

    monkeypatch.setattr(main_module.WakeWordModule, "start", forbidden_start)
    monkeypatch.setattr(main_module.STTModule, "start", forbidden_start)
    monkeypatch.setattr(main_module.TTSModule, "start", forbidden_start)

    await main_module.run_pipeline(
        "config.yaml", text_input="какие приложения ты можешь открыть"
    )

    captured = capsys.readouterr()
    assert "Jarvis: Я могу открыть:" in captured.out
