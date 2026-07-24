"""Integration checks for the audio-free ``main.py --text`` surface."""
from __future__ import annotations

import main as main_module


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
