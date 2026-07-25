"""Offline tests for the public ``main.py --doctor`` runtime diagnostics."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from core.config_loader import Config, ModuleConfig
from core.runtime_diagnostics import (
    DiagnosticStatus,
    RuntimeDiagnosticRunner,
    render_report,
    run_doctor,
)
import main as main_module


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "NVIDIA GeForce RTX 3090 Ti"

    @staticmethod
    def get_device_properties(index: int):
        assert index == 0
        return SimpleNamespace(total_memory=24 * 1024**3)


class _FakeSoundDevice:
    def __init__(self, *, input_error: Exception | None = None) -> None:
        self.input_error = input_error
        self.queries: list[str] = []

    def query_devices(self, *, kind: str):
        self.queries.append(kind)
        if kind == "input" and self.input_error is not None:
            raise self.input_error
        if kind == "input":
            return {"name": "USB Microphone", "max_input_channels": 1}
        return {"name": "Default Speakers", "max_output_channels": 2}


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def _config(tmp_path: Path) -> Config:
    checkpoint = tmp_path / "models" / "nlu.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"controlled test checkpoint")
    whisper_root = tmp_path / "models" / "whisper"
    whisper_root.mkdir()
    (whisper_root / "small.pt").write_bytes(b"controlled whisper cache")
    return Config(
        modules={
            "wake_word": ModuleConfig(enabled=True, device="cpu"),
            "stt": ModuleConfig(
                enabled=True,
                device="auto",
                model="small",
                params={"download_root": str(whisper_root)},
            ),
            "nlu": ModuleConfig(
                enabled=True, device="cpu", model=str(checkpoint)
            ),
            "llm": ModuleConfig(
                enabled=True, device="cpu", model="qwen2.5:7b-instruct"
            ),
            "tts": ModuleConfig(enabled=True, device="cpu", model="v4_ru"),
        },
        logging={"log_file": str(tmp_path / "logs" / "jarvis.log")},
        memory={"db_path": str(tmp_path / "state" / "memory.db")},
        reminders={"db_path": str(tmp_path / "state" / "reminders.db")},
    )


def _healthy_runner(
    tmp_path: Path,
    *,
    sounddevice: _FakeSoundDevice | None = None,
    config: Config | None = None,
) -> RuntimeDiagnosticRunner:
    fake_sounddevice = sounddevice or _FakeSoundDevice()
    modules = {
        "torch": SimpleNamespace(cuda=_FakeCuda()),
        "sounddevice": fake_sounddevice,
        "pynput": SimpleNamespace(),
        "silero_vad": SimpleNamespace(),
        "onnxruntime": SimpleNamespace(),
        "whisper": SimpleNamespace(),
        "silero": SimpleNamespace(),
        "ollama": SimpleNamespace(),
    }

    def import_module(name: str):
        return modules[name]

    return RuntimeDiagnosticRunner(
        config or _config(tmp_path),
        project_root=tmp_path,
        import_module=import_module,
        distribution_version=lambda name: f"{name}-test-version",
        disk_usage=lambda _path: SimpleNamespace(free=20 * 1024**3),
        urlopen=lambda _request, timeout: _FakeResponse(
            {"models": [{"name": "qwen2.5:7b-instruct"}]}
        ),
        checkpoint_validator=lambda path: f"validated {path.name}",
        memory_probe=lambda: 32 * 1024**3,
        python_version=(3, 12, 9),
        platform_name="Windows",
    )


def test_healthy_runtime_report_is_ready_without_real_engines(tmp_path):
    report = _healthy_runner(tmp_path).run()

    assert report.overall == "ready"
    assert report.exit_code == 0
    assert all(
        check.status in {DiagnosticStatus.PASS, DiagnosticStatus.SKIP}
        for check in report.checks
    )
    by_id = {check.check_id: check for check in report.checks}
    assert "RTX 3090 Ti" in by_id["compute.cuda"].detail
    assert by_id["engine.nlu"].status == DiagnosticStatus.PASS
    assert by_id["model.ollama"].status == DiagnosticStatus.PASS


def test_missing_core_and_voice_dependencies_are_actionable_failures(tmp_path):
    config = _config(tmp_path)
    Path(config.module("nlu").model).unlink()

    def missing(name: str):
        raise ModuleNotFoundError(f"No module named {name}")

    runner = RuntimeDiagnosticRunner(
        config,
        project_root=tmp_path,
        import_module=missing,
        distribution_version=lambda _name: "unused",
        disk_usage=lambda _path: SimpleNamespace(free=20 * 1024**3),
        urlopen=lambda _request, timeout: (_ for _ in ()).throw(
            ConnectionRefusedError("offline")
        ),
        memory_probe=lambda: 4 * 1024**3,
        python_version=(3, 12, 9),
        platform_name="Windows",
    )
    report = runner.run()
    by_id = {check.check_id: check for check in report.checks}

    assert report.overall == "failed"
    assert report.exit_code == 2
    assert by_id["engine.torch"].status == DiagnosticStatus.FAIL
    assert by_id["engine.nlu"].status == DiagnosticStatus.FAIL
    assert by_id["engine.whisper"].status == DiagnosticStatus.FAIL
    assert by_id["engine.hotkey"].status == DiagnosticStatus.FAIL
    assert by_id["service.ollama"].status == DiagnosticStatus.WARN
    assert all(
        check.action
        for check in report.checks
        if check.status == DiagnosticStatus.FAIL
        and check.check_id != "audio.input"
    )


def test_audio_device_failure_is_distinguished_from_installed_package(tmp_path):
    runner = _healthy_runner(
        tmp_path,
        sounddevice=_FakeSoundDevice(
            input_error=PermissionError("microphone permission denied")
        ),
    )
    report = runner.run()
    by_id = {check.check_id: check for check in report.checks}

    assert by_id["engine.sounddevice"].status == DiagnosticStatus.PASS
    assert by_id["audio.input"].status == DiagnosticStatus.FAIL
    assert "PermissionError" in by_id["audio.input"].detail
    assert by_id["audio.output"].status == DiagnosticStatus.PASS


def test_invalid_runtime_config_is_reported_before_module_start(tmp_path):
    runner = _healthy_runner(tmp_path)
    runner.config.modules["stt"].device = "cdua"
    runner.config.modules["tts"].params.update(
        {"language": "ru", "speaker": "unknown", "sample_rate": 44100}
    )

    report = runner.run()
    check = next(item for item in report.checks if item.check_id == "config.values")

    assert report.overall == "failed"
    assert check.status == DiagnosticStatus.FAIL
    assert "device='cdua'" in check.detail
    assert "speaker='unknown'" in check.detail
    assert "sample_rate=44100" in check.detail


def test_wrong_yaml_value_types_do_not_crash_doctor(tmp_path):
    config = _config(tmp_path)
    config.logging = []  # type: ignore[assignment]
    config.modules["stt"].device = None  # type: ignore[assignment]
    config.modules["tts"].params = []  # type: ignore[assignment]
    runner = _healthy_runner(tmp_path, config=config)

    report = runner.run()
    check = next(item for item in report.checks if item.check_id == "config.values")

    assert report.overall == "failed"
    assert "секция logging должна быть объектом YAML" in check.detail
    assert "device=None" in check.detail
    assert "modules.tts.params должна быть объектом YAML" in check.detail


def test_json_report_has_stable_schema_and_human_report_has_actions(tmp_path):
    report = _healthy_runner(tmp_path).run()
    json_stream = io.StringIO()
    render_report(report, json_output=True, stream=json_stream)
    payload = json.loads(json_stream.getvalue())

    assert payload["schema_version"] == 1
    assert payload["overall"] == "ready"
    assert payload["exit_code"] == 0
    assert payload["counts"]["pass"] > 0
    assert {item["check_id"] for item in payload["checks"]} >= {
        "system.python",
        "engine.nlu",
        "audio.input",
        "model.ollama",
    }

    human_stream = io.StringIO()
    render_report(report, stream=human_stream)
    human = human_stream.getvalue()
    assert "Jarvis Runtime Doctor" in human
    assert "[OK]" in human
    assert "Итог: READY" in human


def test_malformed_config_returns_failure_without_traceback(tmp_path, capsys):
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("modules: [this is: not valid", encoding="utf-8")

    exit_code = run_doctor(str(config_path))
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "[FAIL] Не удалось прочитать config.yaml" in output
    assert "Действие:" in output


def test_main_doctor_does_not_start_pipeline(monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_doctor(config_path: str, *, json_output: bool = False) -> int:
        calls.append((config_path, json_output))
        return 2

    async def forbidden_pipeline(*_args, **_kwargs):
        raise AssertionError("doctor must not start the Jarvis pipeline")

    monkeypatch.setattr(main_module, "run_doctor", fake_doctor)
    monkeypatch.setattr(main_module, "run_pipeline", forbidden_pipeline)
    monkeypatch.setattr(
        "sys.argv", ["main.py", "--doctor", "--json", "--config", "custom.yaml"]
    )

    with pytest.raises(SystemExit) as stopped:
        main_module.main()

    assert stopped.value.code == 2
    assert calls == [("custom.yaml", True)]


def test_main_doctor_bootstraps_even_when_torch_import_is_broken():
    script = """
import builtins
import sys
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('controlled broken torch')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = ['main.py', '--doctor']
import main
main.run_doctor = lambda config_path, json_output=False: 0
main.main()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
