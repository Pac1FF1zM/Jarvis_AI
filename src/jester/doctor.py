"""Read-only readiness report for the Jester environment, data and pipeline."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from .acquire import PART_NAMES, PART_SIZES
from .labels import JESTER_LABELS
from .prepare import read_labels
from .training import load_jester_config


def doctor(config_path: Path) -> dict[str, object]:
    config = load_jester_config(config_path)
    downloads = Path(config["data"]["downloads"])
    parts = []
    for name, expected in zip(PART_NAMES, PART_SIZES, strict=True):
        path = downloads / name
        actual = path.stat().st_size if path.is_file() else 0
        parts.append({"name": name, "bytes": actual, "expected": expected, "complete": actual == expected})
    labels = read_labels(Path(config["data"]["metadata"]) / "labels.csv")
    frames_root = Path(config["data"]["frames_root"])
    manifest = Path(config["data"]["manifest"])
    license_acceptance = downloads.parent / "RESEARCH_LICENSE_ACCEPTED.json"
    free = shutil.disk_usage(Path.cwd()).free
    data_ready = (
        license_acceptance.is_file()
        and all(part["complete"] for part in parts)
        and frames_root.is_dir()
        and manifest.is_file()
    )
    report: dict[str, object] = {
        "status": "ready" if data_ready else "preparing",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "free_disk_bytes": free,
        "labels": len(labels),
        "expected_labels": len(JESTER_LABELS),
        "archive_parts": parts,
        "research_license_accepted": license_acceptance.is_file(),
        "frames_extracted": frames_root.is_dir(),
        "manifest_ready": manifest.is_file(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    args = parser.parse_args()
    doctor(args.config)


if __name__ == "__main__":
    main()
