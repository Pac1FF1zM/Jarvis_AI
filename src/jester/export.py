"""Export a validated Jester backbone for licensed research use only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .labels import JESTER_LABELS


def export_backbone(checkpoint: Path, output: Path) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("kind") != "jarvis_jester_from_scratch_v1":
        raise ValueError("unsupported Jester checkpoint")
    state = {
        name: value
        for name, value in payload["state_dict"].items()
        if not name.startswith("head.")
    }
    exported = {
        "kind": "jarvis_jester_backbone_v1",
        "source_model_config": payload["model_config"],
        "source_labels": list(JESTER_LABELS),
        "pretrained": False,
        "backbone_state_dict": state,
        "target": "research_only_pending_separate_license_review",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output)
    report: dict[str, object] = {
        "status": "exported",
        "output": str(output.resolve()),
        "tensors": len(state),
        "bytes": output.stat().st_size,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("training_workspace/jester/export/jester_backbone.pt"))
    args = parser.parse_args()
    export_backbone(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
