"""Static safety contracts for the distributable Windows installer."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "installer"


def test_lite_and_full_dependencies_are_deliberately_separated():
    lite = (INSTALLER / "requirements-lite.txt").read_text(encoding="utf-8")
    full = (INSTALLER / "requirements-full.txt").read_text(encoding="utf-8")

    assert "ollama" not in lite.casefold()
    assert "ollama==" in full.casefold()
    assert "huggingface" not in (lite + full).casefold()
    assert "torch" not in {
        line.split("==", 1)[0].casefold()
        for line in lite.splitlines()
        if line and not line.startswith("#")
    }


def test_setup_does_not_package_private_or_generated_workspace_data():
    script = (INSTALLER / "Jarvis.iss").read_text(encoding="utf-8").casefold()

    for forbidden in (
        ".env",
        "venv\\*",
        "logs\\*",
        "training_workspace",
        "tests\\*",
        "memory.db",
        "reminders.db",
        "__pycache__",
        "ml\\*",
        "models\\*.pt",
        "latest_silero_models.yml",
    ):
        assert forbidden not in script
    assert "models\\nlu_manager_finetuned.pt" in script
    assert "ml\\nlu\\inference.py" in script
    assert "ml\\nlu\\train.py" not in script
    assert "holdout_v2" not in script
    assert "python-3.12.9-amd64.exe" in script


def test_installed_launchers_isolate_user_state_and_use_private_python():
    launchers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (INSTALLER / "launchers").glob("*.cmd")
    ).casefold()

    assert "jarvis_data_dir=%appdata%\\jarvis" in launchers
    assert "runtime\\python\\python.exe" in launchers
    assert "enable jarvis full" not in (
        INSTALLER / "launchers" / "Jarvis.cmd"
    ).read_text(encoding="utf-8").casefold()


def test_bootstrap_verifies_doctor_and_keeps_ollama_optional():
    bootstrap = (INSTALLER / "bootstrap_runtime.ps1").read_text(
        encoding="utf-8"
    ).casefold()

    assert "[switch]$installollama" in bootstrap
    assert "[switch]$ollamaonly" in bootstrap
    assert "--doctor" in bootstrap
    assert "doctor-report.json" in bootstrap
    assert "get-authenticodesignature" in bootstrap
    assert "ollamasetup.exe" in bootstrap
    assert "set-location -literalpath $appdir" in bootstrap
    assert "falling back to cpu pytorch" in bootstrap
    assert "ollama app.exe" not in bootstrap


def test_build_pins_python_hash_and_excludes_generated_artifacts_from_git():
    build = (INSTALLER / "BUILD_INSTALLER.ps1").read_text(
        encoding="utf-8"
    )
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "2A52993092A19CFDFFE126E2EEAC46A4265E25705614546604AD44988E040C0F" in build
    assert "Get-AuthenticodeSignature" in build
    assert "installer/cache/" in ignored
    assert "installer/output/" in ignored
    assert "installer/.tools/" in ignored
