"""Offline tests for private profiles and microphone calibration math."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from core.config_loader import Config, ModuleConfig
from core.profile_manager import (
    ProfileError,
    ProfileManager,
    apply_profile_to_config,
    device_fingerprint,
)
from core.voice_calibration import (
    CalibrationQualityError,
    SignalMetrics,
    analyze_signal,
    apply_pcm_gain,
    derive_calibration,
    run_interactive_calibration,
)
import core.voice_calibration as calibration_module
import main as main_module


def _device(name: str = "USB Microphone") -> dict:
    return {
        "name": name,
        "max_input_channels": 1,
        "default_samplerate": 48_000,
    }


def _calibration(device: dict | None = None) -> dict:
    device = device or _device()
    return {
        "schema_version": 1,
        "device_fingerprint": device_fingerprint(device),
        "device": device,
        "vad_start_threshold": 0.56,
        "vad_end_threshold": 0.41,
        "pcm_gain_db": 3.0,
    }


def test_profile_round_trip_is_device_specific_and_never_stores_audio(tmp_path):
    manager = ProfileManager(tmp_path / "profiles")
    manager.ensure_profile("mikhail", "Михаил")
    manager.save_calibration("mikhail", _calibration())
    manager.save_calibration("mikhail", _calibration(_device("Laptop Mic")))
    manager.save_aliases("mikhail", {"Discord": ["дискорд", "дисорд"]})
    manager.set_active("mikhail")

    assert manager.active_profile_id() == "mikhail"
    assert manager.calibration_for("mikhail", device_fingerprint(_device())) is not None
    assert manager.calibration_for(
        "mikhail", device_fingerprint(_device("Laptop Mic"))
    ) is not None
    assert len(manager.calibrations("mikhail")) == 2
    assert manager.load_aliases("mikhail")["Discord"] == ["дискорд", "дисорд"]
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob("*.pcm"))


def test_profile_id_blocks_path_traversal(tmp_path):
    manager = ProfileManager(tmp_path)
    for unsafe in ("../other", "user/name", "", "имя"):
        with pytest.raises(ProfileError):
            manager.ensure_profile(unsafe)


def test_corrupt_profile_personalization_falls_back_without_crashing(tmp_path, caplog):
    manager = ProfileManager(tmp_path)
    profile = manager.profile_dir("default")
    profile.mkdir(parents=True)
    (profile / "voice_calibration.json").write_text("{broken", encoding="utf-8")
    cfg = Config(
        modules={"wake_word": ModuleConfig(), "stt": ModuleConfig()},
        reminders={},
    )

    assert apply_profile_to_config(cfg, manager) == "default"
    assert cfg.reminders["profile_id"] == "default"
    assert "voice_calibrations" not in cfg.module("wake_word").params
    assert "PROFILE_PERSONALIZATION_IGNORED" in caplog.text


def test_profile_overlay_connects_reminders_calibration_and_whisper_aliases(tmp_path):
    manager = ProfileManager(tmp_path)
    manager.ensure_profile("anna")
    manager.save_calibration("anna", _calibration())
    manager.save_aliases("anna", {"Discord": ["дисорд", "дискод"]})
    manager.set_active("anna")
    cfg = Config(
        modules={
            "wake_word": ModuleConfig(params={}),
            "stt": ModuleConfig(params={"initial_prompt": "Команды Джарвиса."}),
        }
    )

    assert apply_profile_to_config(cfg, manager) == "anna"
    assert cfg.reminders["profile_id"] == "anna"
    selected = cfg.module("wake_word").params["voice_calibrations"]
    assert selected[device_fingerprint(_device())]["pcm_gain_db"] == 3.0
    assert "Discord: дисорд, дискод" in cfg.module("stt").params["initial_prompt"]


def test_signal_metrics_and_calibration_are_deterministic():
    rng = np.random.default_rng(42)
    silence_pcm = rng.normal(0, 120, 16_000).astype(np.int16)
    t = np.arange(16_000) / 16_000
    speech_pcm = (7000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    silence = analyze_signal(silence_pcm, [0.02] * 31)
    speech = analyze_signal(speech_pcm, [0.92] * 31)

    result = derive_calibration(_device(), silence, speech)

    assert result["quality"]["snr_db"] > 20
    assert 0.35 <= result["vad_start_threshold"] <= 0.75
    assert result["vad_end_threshold"] < result["vad_start_threshold"]
    assert -6 <= result["pcm_gain_db"] <= 12


def test_bad_snr_and_clipping_are_rejected():
    quiet = SignalMetrics(-30, -25, -10, 0.0, 0.02, 0.1)
    weak = SignalMetrics(-29, -23, -5, 0.0, 0.2, 0.4)
    with pytest.raises(CalibrationQualityError, match="SNR"):
        derive_calibration(_device(), quiet, weak)
    good_noise = SignalMetrics(-60, -50, -30, 0.0, 0.01, 0.05)
    clipped = SignalMetrics(-20, -10, 0, 0.02, 0.8, 0.95)
    with pytest.raises(CalibrationQualityError, match="перегружен"):
        derive_calibration(_device(), good_noise, clipped)


def test_pcm_gain_is_bounded_and_saturates_without_wraparound():
    pcm = np.array([1000, 20_000, -20_000, 32_000, -32_000], dtype=np.int16).tobytes()
    boosted = np.frombuffer(apply_pcm_gain(pcm, 30), dtype=np.int16)

    assert boosted[0] == 3981  # +12 dB hard limit
    assert boosted[1] == 32767
    assert boosted[2] == -32768
    assert boosted[3] == 32767
    assert boosted[4] == -32768


def test_profile_json_is_versioned_and_atomic_temp_is_removed(tmp_path):
    manager = ProfileManager(tmp_path)
    manager.ensure_profile("default")
    payload = json.loads((tmp_path / "default" / "profile.json").read_text("utf-8"))
    assert payload["schema_version"] == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_profile_listing_is_read_only_and_returns_persisted_ids(tmp_path):
    manager = ProfileManager(tmp_path / "profiles")
    manager.ensure_profile("default", "Основной")
    manager.ensure_profile("mikhail", "Михаил")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    profiles = manager.list_profiles()

    assert [(item["profile_id"], item["name"]) for item in profiles] == [
        ("default", "Основной"),
        ("mikhail", "Михаил"),
    ]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before


def test_main_profiles_cli_prints_ids_without_starting_runtime(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("JARVIS_DATA_DIR", raising=False)
    profile_root = tmp_path / "profiles"
    manager = ProfileManager(profile_root)
    manager.ensure_profile("default", "Основной")
    manager.ensure_profile("mikhail", "Михаил")
    manager.set_active("mikhail")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"profiles": {"root": str(profile_root)}}),
        encoding="utf-8",
    )

    async def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("profile listing must not start runtime")

    monkeypatch.setattr(main_module, "run_pipeline", forbidden_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--profiles", "--config", str(config_path)],
    )

    main_module.main()

    output = capsys.readouterr().out
    assert "  default — Основной" in output
    assert "* mikhail — Михаил" in output
    assert "* — активный профиль" in output


def test_standalone_profile_argument_is_rejected_before_runtime(monkeypatch, capsys):
    async def forbidden_pipeline(*args, **kwargs):
        raise AssertionError("invalid profile arguments must not start runtime")

    monkeypatch.setattr(main_module, "run_pipeline", forbidden_pipeline)
    monkeypatch.setattr(sys, "argv", ["main.py", "--profile", "ID"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "только с --calibrate-voice" in error
    assert "python main.py --profiles" in error


def test_interactive_calibration_uses_fakes_and_persists_only_aggregates(
    tmp_path, monkeypatch
):
    class FakeSoundDevice:
        calls = 0

        @staticmethod
        def query_devices(device=None, kind=None):
            assert kind == "input"
            return _device()

        @classmethod
        def rec(cls, frames, **kwargs):
            cls.calls += 1
            if cls.calls == 1:
                return np.full((frames, 1), 50, dtype=np.int16)
            t = np.arange(frames) / 16_000
            return (6000 * np.sin(2 * np.pi * 220 * t)).astype(np.int16).reshape(-1, 1)

    class FakeVAD:
        def reset_states(self):
            pass

        def __call__(self, tensor, sample_rate):
            assert sample_rate == 16_000
            return 0.9 if float(tensor.abs().mean()) > 0.01 else 0.01

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setitem(
        sys.modules,
        "silero_vad",
        SimpleNamespace(load_silero_vad=lambda **kwargs: FakeVAD()),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    manager = ProfileManager(tmp_path / "profiles")

    result = run_interactive_calibration(manager, "default")

    assert FakeSoundDevice.calls == 5
    assert result["quality"]["validation_vad_speech_ratio"] == 1.0
    assert manager.active_calibration("default") is not None
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob("*.pcm"))


def test_main_calibration_cli_delegates_without_starting_pipeline(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_DATA_DIR", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "profiles": {"root": str(tmp_path / "private")},
                "modules": {"wake_word": {"params": {}}},
            }
        ),
        encoding="utf-8",
    )
    called = {}

    def fake_calibrate(manager, profile_id, **kwargs):
        called.update(root=manager.root, profile_id=profile_id, kwargs=kwargs)

    monkeypatch.setattr(calibration_module, "run_interactive_calibration", fake_calibrate)
    monkeypatch.setattr(
        main_module,
        "run_pipeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("pipeline must not start during calibration")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--calibrate-voice",
            "--config",
            str(config_path),
            "--profile",
            "mikhail",
            "--profile-name",
            "Михаил",
        ],
    )

    main_module.main()

    assert called["root"] == tmp_path / "private"
    assert called["profile_id"] == "mikhail"
    assert called["kwargs"]["profile_name"] == "Михаил"
