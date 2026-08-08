"""Open the official Jester test split once after model selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import JesterDataset, load_manifest
from .labels import JESTER_LABELS
from .models import JesterModelConfig, build_model
from .training import _loader, evaluate_loader, load_jester_config


def evaluate(config_path: Path, checkpoint_path: Path) -> dict[str, object]:
    config = load_jester_config(config_path)
    output = Path(config["paths"]["reports"]) / "final_test.json"
    if output.exists():
        raise RuntimeError(
            f"sealed Jester test was already evaluated; preserved report: {output.resolve()}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("kind") != "jarvis_jester_from_scratch_v1" or payload.get("pretrained") is not False:
        raise ValueError("checkpoint is not a from-scratch Jester model")
    if payload.get("labels") != list(JESTER_LABELS):
        raise ValueError("checkpoint label order differs from official Jester labels")
    model = build_model(JesterModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    records = load_manifest(Path(config["data"]["manifest"]), "test")
    dataset = JesterDataset(
        records,
        frames_root=Path(config["data"]["frames_root"]),
        clip_len=int(config["data"]["clip_len"]),
        frame_size=int(config["data"]["frame_size"]),
        resize_size=int(config["data"]["resize_size"]),
        training=False,
    )
    loader = _loader(config, dataset, int(config["train"]["batch_size"]), False)
    result: dict[str, object] = {
        "status": "completed",
        "protocol": "official_test_once_after_train_validation_selection",
        "checkpoint": str(checkpoint_path.resolve()),
        "metrics": evaluate_loader(model, loader, device, bool(config["train"]["amp"]) and device.type == "cuda"),
        "test_split_opened": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
