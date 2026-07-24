"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable so `from core.event_bus import ...` works
# when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_optional_runtime_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every pytest run offline and independent of installed engines.

    Optional packages are intentionally detected at module import/startup in
    production.  Without this fixture, installing Whisper, Ollama, Silero or
    sounddevice changes integration-test behaviour and can trigger downloads,
    network calls, microphone access or real playback.  Tests for a real code
    path explicitly replace these ``None`` values with their own fakes.
    """
    import modules.llm as llm_module
    import modules.stt as stt_module
    import modules.tts as tts_module
    import modules.wake_word as wake_word_module

    monkeypatch.setattr(llm_module, "_OLLAMA", None)
    monkeypatch.setattr(stt_module, "_WHISPER", None)
    monkeypatch.setattr(tts_module, "_SILERO_TTS", None)
    monkeypatch.setattr(tts_module, "_SOUNDDEVICE", None)
    monkeypatch.setattr(wake_word_module, "_SOUNDDEVICE", None)
    monkeypatch.setattr(wake_word_module, "_PYNPUT_KEYBOARD", None)
    monkeypatch.setattr(wake_word_module, "_LOAD_SILERO_VAD", None)
