"""Regression checks for the hermetic default pytest environment."""
from __future__ import annotations

import modules.llm as llm_module
import modules.stt as stt_module
import modules.tts as tts_module
import modules.wake_word as wake_word_module


def test_optional_runtime_engines_are_disabled_unless_explicitly_faked():
    assert llm_module._OLLAMA is None
    assert stt_module.PersistentParakeetClient.__name__ == "_DisabledParakeetClient"
    assert tts_module._SILERO_TTS is None
    assert tts_module._SOUNDDEVICE is None
    assert wake_word_module._SOUNDDEVICE is None
    assert wake_word_module._PYNPUT_KEYBOARD is None
    assert wake_word_module._LOAD_SILERO_VAD is None
