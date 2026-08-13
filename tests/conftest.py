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
def isolate_optional_runtime_engines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Keep every pytest run offline and independent of installed engines.

    Optional packages are intentionally detected at module import/startup in
    production.  Without this fixture, installing Parakeet, Ollama, Silero or
    sounddevice changes integration-test behaviour and can trigger downloads,
    network calls, microphone access or real playback.  Tests for a real code
    path explicitly replaces these disabled values with its own fakes.
    """
    import modules.llm as llm_module
    import modules.stt as stt_module
    import modules.tts as tts_module
    import modules.wake_word as wake_word_module

    class _DisabledParakeetClient:
        """Fast offline stand-in for integration tests using production config."""

        def __init__(self, **_kwargs) -> None:
            pass

        def start(self):
            return {
                "event": "ready",
                "model_load_ms": 0.0,
                "warm_up_ms": 0.0,
                "health": {
                    "model_id": "pytest-disabled-parakeet",
                    "model_revision": "offline",
                },
            }

        def decode(self, _wav_bytes):
            return {"event": "result", "text": "", "decode_ms": 0.0}

        def close(self):
            return None

    monkeypatch.setattr(llm_module, "_OLLAMA", None)
    monkeypatch.setattr(
        stt_module, "PersistentParakeetClient", _DisabledParakeetClient
    )
    monkeypatch.setattr(tts_module, "_SILERO_TTS", None)
    monkeypatch.setattr(tts_module, "_SOUNDDEVICE", None)
    monkeypatch.setattr(wake_word_module, "_SOUNDDEVICE", None)
    monkeypatch.setattr(wake_word_module, "_PYNPUT_KEYBOARD", None)
    monkeypatch.setattr(wake_word_module, "_LOAD_SILERO_VAD", None)
    monkeypatch.setenv("JARVIS_REMINDERS_DB", str(tmp_path / "reminders.db"))
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path / "jarvis-data"))
