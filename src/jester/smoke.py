"""One CUDA forward/backward pass for every candidate at the real clip shape."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .dataset import JesterDataset, load_manifest
from .labels import JESTER_LABELS
from .models import JesterModelConfig, build_model, parameter_count
from .training import configured_models, load_jester_config


def smoke(config_path: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Jester smoke test")
    config = load_jester_config(config_path)
    data = config["data"]
    device = torch.device("cuda")
    manifest = Path(data["manifest"])
    frames_root = Path(data["frames_root"])
    if manifest.is_file() and frames_root.is_dir():
        record = load_manifest(manifest, "val")[0]
        dataset = JesterDataset(
            [record],
            frames_root=frames_root,
            clip_len=int(data["clip_len"]),
            frame_size=int(data["frame_size"]),
            resize_size=int(data["resize_size"]),
            training=False,
        )
        sample, class_id = dataset[0]
        clip = sample.unsqueeze(0).to(device)
        target = torch.tensor([class_id], dtype=torch.long, device=device)
        input_source = f"validation_clip:{record.clip_id}"
    else:
        clip = torch.randn(
            1, int(data["clip_len"]), 3, int(data["frame_size"]), int(data["frame_size"]), device=device
        )
        target = torch.zeros(1, dtype=torch.long, device=device)
        input_source = "synthetic_fallback"
    results = []
    for name in configured_models(config):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(
            JesterModelConfig(name=name, num_classes=len(JESTER_LABELS), dropout=float(config["models"]["dropout"]))
        ).to(device).train()
        # Exclude one-time cuDNN algorithm selection and CUDA kernel loading
        # from the reported steady-state timing.
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            warmup_loss = torch.nn.functional.cross_entropy(model(clip), target)
        warmup_loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(clip), target)
        loss.backward()
        torch.cuda.synchronize()
        results.append(
            {
                "model": name,
                "parameters": parameter_count(model),
                "milliseconds": (time.perf_counter() - started) * 1000,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "finite_loss": bool(torch.isfinite(loss)),
            }
        )
        del model, loss, warmup_loss
    report: dict[str, object] = {
        "status": "passed",
        "input_source": input_source,
        "shape": list(clip.shape),
        "models": results,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/jester_from_scratch.yaml"))
    args = parser.parse_args()
    smoke(args.config)


if __name__ == "__main__":
    main()
