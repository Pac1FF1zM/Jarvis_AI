"""Static safety contracts for the distributable Windows installer."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "installer"


def test_lite_and_full_dependencies_are_deliberately_separated():
    lite = (INSTALLER / "requirements-lite.txt").read_text(encoding="utf-8")
    full = (INSTALLER / "requirements-full.txt").read_text(encoding="utf-8")

    assert "ollama" not in lite.casefold()
    assert "ollama==" in full.casefold()
    assert "huggingface" not in (lite + full).casefold()
    assert "openai-whisper" not in lite.casefold()
    assert "transformers==5.15.0" in lite.casefold()
    assert "librosa==0.11.0" in lite.casefold()
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
    assert "jarvis_control.py" in script
    assert "control_center\\*.py" in script
    assert "ml\\gesture\\models.py" in script
    assert "ml\\jsc\\*.py" in script
    assert "models\\jsc\\*" in script
    assert "models\\gesture\\20260812_jester_tiny3d\\*" in script
    assert "checkpoints\\tsn_resnet18_seed42\\best.pt" not in script
    assert "reports\\evaluation_test.json" not in script
    assert "ml\\nlu\\train.py" not in script
    assert "holdout_v2" not in script
    assert "python-3.12.9-amd64.exe" in script
    assert "setup parakeet.cmd" in script
    assert "prepare_whisper.py" not in script


def test_packaged_nlu_runtime_runs_inference_without_training_workspace(tmp_path):
    """Smoke the exact ML subset declared by Inno Setup, not the source tree."""
    setup = (INSTALLER / "Jarvis.iss").read_text(encoding="utf-8")
    sources = re.findall(r'^Source: "\.\.\\(ml\\[^"*]+\.py)";', setup, re.M)
    assert sources

    for windows_relative in sources:
        relative = Path(*windows_relative.split("\\"))
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    checkpoint = tmp_path / "models" / "nlu_manager_finetuned.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "models" / checkpoint.name, checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from ml.nlu.inference import NLUPredictor; "
                "result = NLUPredictor(sys.argv[2]).predict('который час'); "
                "assert result.intent == 'get_current_time', result"
            ),
            str(tmp_path),
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_setup_declares_every_structured_jsc_runtime_module():
    """The installed shadow must not depend on the source checkout."""
    setup = (INSTALLER / "Jarvis.iss").read_text(encoding="utf-8").casefold()

    assert 'source: "..\\ml\\jsc\\*.py"' in setup
    assert 'source: "..\\models\\jsc\\*"' in setup
    assert (ROOT / "models" / "jsc" / "structured_v8_seed29.pt").is_file()
    assert (
        ROOT
        / "models"
        / "gesture"
        / "20260812_jester_tiny3d"
        / "best.pt"
    ).is_file()


def test_packaged_jsc_runtime_runs_without_training_workspace(tmp_path):
    """Smoke the release checkpoint from an installer-shaped source tree."""
    for package in ("core", "memory", "modules", "tools"):
        shutil.copytree(ROOT / package, tmp_path / package)
    (tmp_path / "ml").mkdir()
    shutil.copy2(ROOT / "ml" / "__init__.py", tmp_path / "ml" / "__init__.py")
    shutil.copytree(ROOT / "ml" / "jsc", tmp_path / "ml" / "jsc")
    checkpoint = tmp_path / "models" / "jsc" / "structured_v8_seed29.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "models" / "jsc" / checkpoint.name, checkpoint)

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys; sys.path.insert(0, sys.argv[1]); "
                "from ml.jsc.inference import StructuredJSCPredictor; "
                "from ml.jsc.jal import loads; "
                "from ml.jsc.project_registry import build_project_schema_registry; "
                "predictor = StructuredJSCPredictor(sys.argv[2], "
                "build_project_schema_registry(), device='cpu'); "
                "plan = loads(predictor.predict('открой калькулятор').jal); "
                "assert plan.act.value == 'execute' and len(plan.steps) == 1, plan"
            ),
            str(tmp_path),
            str(checkpoint),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_release_model_manifest_matches_packaged_artifacts():
    manifest = json.loads(
        (ROOT / "models" / "RELEASE_MODELS.json").read_text(encoding="utf-8")
    )
    assert manifest["release"] == "0.7.0"

    for model in manifest["models"]:
        path = ROOT / Path(model["path"])
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == model["sha256"]


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
    assert "jarvis_control.py" in (
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
