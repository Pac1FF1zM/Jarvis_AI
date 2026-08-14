import json
from pathlib import Path


def test_release_version_is_one_value_across_code_models_and_installer():
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION").read_text("utf-8").strip()
    manifest = json.loads((root / "models" / "RELEASE_MODELS.json").read_text("utf-8"))
    migration = json.loads(
        (root / "models" / "JSC_MIGRATION_STATE.json").read_text("utf-8")
    )
    installer = (root / "installer" / "Jarvis.iss").read_text("utf-8")

    assert version == "0.9.0"
    assert manifest["release"] == version
    assert migration["active_release"] == version
    assert 'FileOpen("..\\\\VERSION")' in installer
