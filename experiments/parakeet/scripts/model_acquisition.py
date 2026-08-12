"""Pinned Parakeet license review, explicit acceptance, and local download."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
MODEL_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
LICENSE_SPDX = "CC-BY-4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.txt"
MODEL_CARD_URL = f"https://huggingface.co/{MODEL_ID}/raw/{MODEL_REVISION}/README.md"
EXPECTED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(root: Path) -> tuple[Path, Path, Path]:
    review = root / "license-review"
    approval = root / "approvals" / "model-license-approval.json"
    model = root / "models" / "nvidia--parakeet-tdt-0.6b-v3"
    return review, approval, model


def review_license(root: Path) -> dict[str, Any]:
    review, _approval, _model = _paths(root)
    review.mkdir(parents=True, exist_ok=True)
    for url, output in ((LICENSE_URL, review / "CC-BY-4.0.txt"), (MODEL_CARD_URL, review / "MODEL_CARD.md")):
        request = urllib.request.Request(url, headers={"User-Agent": "Jarvis-Parakeet-License-Review"})
        with urllib.request.urlopen(request, timeout=30) as response:
            output.write_bytes(response.read())
    evidence = {
        "stage": "ModelLicenseReview",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license_spdx": LICENSE_SPDX,
        "license_url": LICENSE_URL,
        "license_sha256": _sha256(review / "CC-BY-4.0.txt"),
        "model_card_url": MODEL_CARD_URL,
        "model_card_sha256": _sha256(review / "MODEL_CARD.md"),
        "note": "Review evidence only. It does not approve or download model weights.",
    }
    (review / "provenance.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def accept_license(root: Path, accepted_spdx: str) -> dict[str, Any]:
    review, approval, _model = _paths(root)
    provenance_path = review / "provenance.json"
    if accepted_spdx != LICENSE_SPDX:
        raise ValueError(f"type the exact license identifier: {LICENSE_SPDX}")
    if not provenance_path.is_file():
        raise ValueError("run ReviewLicense before accepting the model license")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if _sha256(review / "CC-BY-4.0.txt") != provenance.get("license_sha256"):
        raise ValueError("reviewed license evidence changed; acceptance blocked")
    if approval.exists():
        raise ValueError(f"immutable approval already exists and will not be overwritten: {approval}")
    record = {
        "stage": "InstallApprovedModel",
        "approved": True,
        "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_by": "local_repository_owner_via_explicit_cli_flag",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license_spdx": LICENSE_SPDX,
        "license_url": LICENSE_URL,
        "license_sha256": provenance["license_sha256"],
        "scope": "Local Parakeet diagnostic model installation; unrelated to fixture consent or retention.",
    }
    approval.parent.mkdir(parents=True, exist_ok=True)
    approval.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def _approval(root: Path) -> dict[str, Any]:
    review, approval_path, _model = _paths(root)
    if not approval_path.is_file():
        raise ValueError("model download blocked: explicit CC-BY-4.0 acceptance is missing")
    record = json.loads(approval_path.read_text(encoding="utf-8"))
    expected = {
        "stage": "InstallApprovedModel",
        "approved": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license_spdx": LICENSE_SPDX,
        "license_sha256": _sha256(review / "CC-BY-4.0.txt"),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"model approval mismatch for {key}; download blocked")
    return record


def download_model(root: Path) -> dict[str, Any]:
    _approval(root)
    _review, _approval_path, model = _paths(root)
    from huggingface_hub import snapshot_download

    model.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model,
        allow_patterns=list(EXPECTED_FILES),
    )
    missing = [name for name in EXPECTED_FILES if not (model / name).is_file()]
    if missing:
        raise ValueError(f"downloaded snapshot is incomplete: {missing}")
    evidence = {
        "stage": "InstalledApprovedModel",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license_spdx": LICENSE_SPDX,
        "files": {name: _sha256(model / name) for name in EXPECTED_FILES},
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        "authenticity_note": "Pinned official HTTPS Hugging Face repository revision; no independent signature was published.",
    }
    (model / "installation.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    return evidence


def status(root: Path) -> dict[str, Any]:
    review, approval, model = _paths(root)
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license_reviewed": (review / "provenance.json").is_file(),
        "license_accepted": approval.is_file(),
        "model_installed": all((model / name).is_file() for name in EXPECTED_FILES),
        "model_dir": str(model),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("ReviewLicense", "AcceptLicense", "DownloadModel", "Status"))
    parser.add_argument("--root", default=".local/parakeet")
    parser.add_argument("--accept-license", default="")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.stage == "ReviewLicense":
        result = review_license(root)
    elif args.stage == "AcceptLicense":
        result = accept_license(root, args.accept_license)
    elif args.stage == "DownloadModel":
        result = download_model(root)
    else:
        result = status(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
