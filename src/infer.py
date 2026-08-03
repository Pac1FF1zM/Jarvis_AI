"""Run end-to-end isolated-gesture inference with the selected TSN checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import torch

from src.data.audit import EXPECTED_LABELS
from src.data.dataset import decode_clip, uniform_frame_indices
from src.data.transforms import ClipTransform, ClipTransformConfig
from src.evaluate import load_selected_model
from src.utils import load_config, resolve_from_project, write_json


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count < 1:
        raise ValueError(f"Video reports no frames: {path}")
    return count


def infer_video(
    config_path: Path,
    checkpoint_path: Path,
    video_path: Path,
    *,
    start_frame: int = 1,
    end_frame: int | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    config = load_config(config_path)
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    total_frames = video_frame_count(video_path)
    end_frame = total_frames if end_frame is None else end_frame
    if start_frame < 1 or end_frame < start_frame or end_frame > total_frames:
        raise ValueError(
            f"Invalid inclusive frame range {start_frame}..{end_frame} for {total_frames} frames"
        )
    top_k = min(max(top_k, 1), len(EXPECTED_LABELS))
    indices = uniform_frame_indices(start_frame, end_frame, int(config["data"]["clip_len"]))
    frames = decode_clip(
        video_path,
        indices,
        decode_retries=int(config["data"]["decode_retries"]),
    )
    transform = ClipTransform(
        ClipTransformConfig(
            frame_size=int(config["data"]["frame_size"]),
            resize_size=int(config["data"]["cache_resize_size"]),
        ),
        training=False,
    )
    clip = transform(frames).unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_selected_model(checkpoint_path.resolve(), device)
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(config["train"]["amp"]) and device.type == "cuda",
        ):
            logits = model(clip.to(device))
        probabilities = logits.softmax(dim=1)[0]
    scores, class_ids = probabilities.topk(top_k)
    ranking = [
        {
            "rank": rank,
            "class_id": int(class_id),
            "label": EXPECTED_LABELS[int(class_id)],
            "confidence": float(score),
        }
        for rank, (score, class_id) in enumerate(zip(scores.cpu(), class_ids.cpu(), strict=True), start=1)
    ]
    return {
        "status": "completed",
        "video": str(video_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_epoch": int(payload["epoch"]),
        "device": str(device),
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "sampled_frame_indices": indices,
        "clip_shape": list(clip.shape),
        "prediction": ranking[0],
        "top_k": ranking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/tsn_resnet18_seed42/best.pt"),
    )
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = infer_video(
        args.config,
        args.checkpoint,
        args.video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        top_k=args.top_k,
    )
    if args.output is not None:
        write_json(resolve_from_project(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
