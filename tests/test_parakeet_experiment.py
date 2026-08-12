"""Safety-focused Parakeet shadow tests; model weights are not required."""
from __future__ import annotations

import io
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.parakeet import fixtures as fixture_mod
from experiments.parakeet import shadow_pipeline as shadow_mod
from modules.semantic_commit import prepare_final_utterance
from experiments.parakeet.benchmarks.no_action import NoActionGuard
from experiments.parakeet.benchmarks.compare_stt import (
    error_counts,
    load_manifest,
    normalize_text,
    summarize,
)
from experiments.parakeet.scripts import fixture_tool, model_acquisition, shadow_test
from experiments.parakeet.worker import backend as backend_mod


def _wav(*, rate: int = 16_000, channels: int = 1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * rate * channels)
    return output.getvalue()


def test_no_action_guard_is_a_hard_stop():
    guard = NoActionGuard()
    assert guard.record_interpretation(text="открой браузер")["execution"] == "blocked"
    with pytest.raises(RuntimeError, match="NO_ACTION_MODE"):
        guard.execute("open_browser")


def test_parakeet_accepts_only_16khz_mono_pcm_wav():
    assert len(backend_mod._decode_wav(_wav())) == 16_000
    with pytest.raises(backend_mod.AudioFormatError, match="16 kHz"):
        backend_mod._decode_wav(_wav(rate=8_000))
    with pytest.raises(backend_mod.AudioFormatError, match="16 kHz"):
        backend_mod._decode_wav(_wav(channels=2))
    with pytest.raises(backend_mod.AudioFormatError):
        backend_mod._decode_wav(b"not a wav")


def test_backend_passes_numpy_audio_to_parakeet_processor(monkeypatch):
    import numpy as np

    observed = {}

    class Inputs(dict):
        def to(self, **_kwargs):
            return self

    class Processor:
        def __call__(self, audio, **_kwargs):
            observed["audio"] = audio
            return Inputs()

        def decode(self, *_args, **_kwargs):
            return "тест"

    class Model:
        device = "cpu"
        dtype = "float32"

        def generate(self, **_kwargs):
            observed["generation"] = _kwargs
            return SimpleNamespace(sequences=[1])

    class Inference:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    backend = backend_mod.ParakeetBackend(model_dir=".", provider="cpu")
    backend._processor, backend._model = Processor(), Model()
    backend._torch = SimpleNamespace(inference_mode=lambda: Inference())
    assert backend.transcribe(_wav()) == "тест"
    assert isinstance(observed["audio"][0], np.ndarray)
    assert observed["audio"][0].dtype == np.float32
    assert observed["generation"]["max_new_tokens"] == backend_mod.MAX_NEW_TOKENS == 96


def test_model_and_revision_are_pinned_consistently():
    assert backend_mod.MODEL_ID == model_acquisition.MODEL_ID
    assert backend_mod.MODEL_REVISION == model_acquisition.MODEL_REVISION
    assert len(backend_mod.MODEL_REVISION) == 40


def test_shadow_nlu_returns_production_shape_but_blocks_execution(tmp_path, monkeypatch):
    checkpoint = tmp_path / "nlu.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setattr(shadow_mod, "NLUPredictor", lambda *_args: object())

    result = shadow_mod.ShadowNLU(checkpoint).predict("открой калькулятор")

    assert result["execution"] == "blocked"
    assert result["commit_state"] == "ready"
    assert result["actions"] == [{
        "raw_intent": "open_application",
        "intent": "open_application",
        "slots": {"application": "calculator"},
        "confidence": 0.99,
        "execution": "blocked",
    }]


@pytest.mark.parametrize(
    ("text", "state", "reason"),
    (
        ("Запусти телегу и.", "wait", "unfinished_utterance"),
        ("Открой.", "wait", "unfinished_utterance"),
        ("Не открывай браузер.", "rejected", "negated_command"),
        ("Он сказал: открой калькулятор", "rejected", "mentioned_or_quoted_command"),
        ("Что будет, если сказать «открой браузер»?", "rejected", "mentioned_or_quoted_command"),
    ),
)
def test_commit_gate_blocks_incomplete_negated_and_mentioned_commands(text, state, reason):
    result = prepare_final_utterance(text)
    assert (result.state, result.reason, result.route_text) == (state, reason, "")


def test_commit_gate_does_not_confuse_infinitive_question_with_incomplete_command():
    result = prepare_final_utterance("какие приложения ты можешь открыть")

    assert result.state == "analyze"
    assert result.route_text == "какие приложения ты можешь открыть"


@pytest.mark.parametrize(
    ("text", "route_text"),
    (
        ("Открой калькулятор. Нет, лучше блокнот.", "открой блокнот"),
        ("Запусти телеграмм. Ой, закрой телеграмм.", "закрой телеграмм"),
    ),
)
def test_commit_gate_keeps_only_final_self_correction(text, route_text):
    result = prepare_final_utterance(text)
    assert (result.state, result.route_text) == ("analyze", route_text)


def test_shadow_nlu_never_commits_part_of_unresolved_compound(tmp_path, monkeypatch):
    checkpoint = tmp_path / "nlu.pt"
    checkpoint.write_bytes(b"checkpoint")

    class Predictor:
        def predict(self, _text):
            return SimpleNamespace(intent="unknown", slots={}, confidence=0.99)

    monkeypatch.setattr(shadow_mod, "NLUPredictor", lambda *_args: Predictor())
    result = shadow_mod.ShadowNLU(checkpoint).predict(
        "открой браузер и запусти совершенно неизвестное приложение"
    )
    assert result["commit_state"] == "clarify"
    assert result["actions"] == []
    assert len(result["candidate_actions"]) == 2


def test_shadow_decoder_uses_jarvis_interpreter_without_shell(tmp_path, monkeypatch):
    interpreter = tmp_path / "venv/Scripts/python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"")
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout='{"text":"тест"}', stderr="")

    monkeypatch.setattr(shadow_mod.subprocess, "run", fake_run)
    assert shadow_mod.decode_in_child(wav, repository_root=tmp_path)["text"] == "тест"
    assert Path(observed["command"][0]) == interpreter
    assert "experiments.parakeet.worker.decode_file" in observed["command"]
    assert observed["kwargs"]["shell"] is False


def test_shadow_pipeline_has_no_action_pipeline_hooks():
    source = Path(shadow_mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("EventBus", "ToolRegistry", ".publish(", "main.py", "shell=True"):
        assert forbidden not in source


def test_worker_eof_reports_stderr_after_exit(tmp_path):
    decoder = shadow_mod.PersistentParakeetDecoder(repository_root=tmp_path)
    decoder._stderr_tail.append("real startup failure")
    decoder.process = SimpleNamespace(
        stdout=io.StringIO(""),
        stderr=io.StringIO(""),
        wait=lambda timeout: 1,
    )
    with pytest.raises(RuntimeError, match="real startup failure"):
        decoder._receive()


def test_decode_timeout_kills_worker_and_next_capture_restarts(monkeypatch, tmp_path):
    decoder = shadow_mod.PersistentParakeetDecoder(repository_root=tmp_path)
    decoder.process = object()
    events = []
    monkeypatch.setattr(decoder, "_send", lambda _payload: events.append("send"))
    monkeypatch.setattr(decoder, "_receive", lambda: (_ for _ in ()).throw(TimeoutError("slow")))
    monkeypatch.setattr(decoder, "close", lambda **_kwargs: (events.append("close"), setattr(decoder, "process", None)))
    with pytest.raises(TimeoutError, match="slow"):
        decoder.decode(b"wav")
    assert events == ["send", "close"]


def test_live_capture_builds_expected_pcm_wav():
    payload = shadow_test._pcm_to_wav(b"\x00\x00" * 1600)
    assert payload.startswith(b"RIFF")
    assert len(backend_mod._decode_wav(payload)) == 1600


def test_closed_input_exits_without_recording(monkeypatch, tmp_path, capsys):
    class Decoder:
        startup = {"model_load_ms": 1.0}

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(shadow_test, "PersistentParakeetDecoder", Decoder)
    monkeypatch.setattr("builtins.input", lambda *_args: (_ for _ in ()).throw(EOFError()))
    shadow_test._run_microphone(SimpleNamespace(), device=None, timeout=1.0)
    assert "Никакие команды не выполнялись" in capsys.readouterr().out


def test_model_download_requires_separate_explicit_approval(tmp_path):
    with pytest.raises(ValueError, match="explicit CC-BY-4.0 acceptance"):
        model_acquisition.download_model(tmp_path)


def test_model_license_acceptance_is_immutable(tmp_path):
    review = tmp_path / "license-review"
    review.mkdir()
    license_path = review / "CC-BY-4.0.txt"
    license_path.write_text("terms", encoding="utf-8")
    provenance = {"license_sha256": model_acquisition._sha256(license_path)}
    (review / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    record = model_acquisition.accept_license(tmp_path, "CC-BY-4.0")
    assert record["approved"] is True
    with pytest.raises(ValueError, match="immutable approval"):
        model_acquisition.accept_license(tmp_path, "CC-BY-4.0")


def test_fixture_validator_requires_ordered_canonical_actions():
    with pytest.raises(ValueError, match="action must contain exactly"):
        fixture_mod.validate_action({"intent": "open_application"})
    with pytest.raises(ValueError, match="unknown production intent"):
        fixture_mod.validate_action({"intent": "open_app", "slots": {}})
    with pytest.raises(ValueError, match="unknown slots"):
        fixture_mod.validate_action({"intent": "open_application", "slots": {"app": "calculator"}})


def test_fixture_tool_help_exposes_required_recorder_operations(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fixture_tool.py", "--help"])
    with pytest.raises(SystemExit):
        fixture_tool.main()
    text = capsys.readouterr().out
    for command in ("devices", "test-device", "record-next", "replay", "re-record", "delete", "validate"):
        assert command in text


def test_recorder_source_has_no_jarvis_execution_imports():
    source = Path(fixture_tool.__file__).read_text(encoding="utf-8")
    for forbidden in ("EventBus", "NLUModule", "ToolRegistry", "run_pipeline", "modules.nlu"):
        assert forbidden not in source


def test_optional_local_model_smoke():
    import os

    configured = os.environ.get("JARVIS_PARAKEET_MODEL_DIR", "").strip()
    if not configured:
        pytest.skip("set JARVIS_PARAKEET_MODEL_DIR to run the approved local model smoke test")
    model_dir = Path(configured)
    if not model_dir.is_dir():
        pytest.skip("set JARVIS_PARAKEET_MODEL_DIR to run the approved local model smoke test")
    backend = backend_mod.ParakeetBackend(model_dir=model_dir, provider="cuda")
    backend.start()
    try:
        backend.warm_up()
        assert backend.health()["state"] == "ready"
    finally:
        backend.close()


def test_benchmark_normalization_and_edit_metrics_are_deterministic():
    assert normalize_text("  Ёж, ОТКРОЙ Browser! ") == "еж открой browser"
    counts = error_counts("открой браузер", "открой новый браузер")
    assert counts["word_errors"] == 1
    assert counts["reference_words"] == 2
    assert counts["exact"] == 0


def test_benchmark_manifest_requires_real_references_and_audio(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(_wav())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {"id": "one", "path": "sample.wav", "reference_text": "тест"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert load_manifest(manifest)[0]["path"] == str(audio.resolve())
    manifest.write_text(
        json.dumps({"id": "bad", "path": "sample.wav", "reference_text": ""}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="human reference_text"):
        load_manifest(manifest)


def test_benchmark_summary_uses_corpus_wer_and_latency():
    rows = [
        {
            "word_errors": 1,
            "reference_words": 2,
            "char_errors": 1,
            "reference_chars": 10,
            "exact": 0,
            "latency_ms": 100.0,
            "audio_seconds": 1.0,
        },
        {
            "word_errors": 0,
            "reference_words": 3,
            "char_errors": 0,
            "reference_chars": 12,
            "exact": 1,
            "latency_ms": 200.0,
            "audio_seconds": 2.0,
        },
    ]
    result = summarize("test", rows)
    assert result["wer"] == pytest.approx(0.2)
    assert result["cer"] == pytest.approx(1 / 22)
    assert result["exact_match_rate"] == 0.5
    assert result["latency_ms"]["median"] == 150.0
    assert result["rtf"] == pytest.approx(0.1)
