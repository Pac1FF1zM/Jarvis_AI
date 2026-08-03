"""Evaluate the selected gesture checkpoint on prepared user recordings."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.audit import EXPECTED_LABELS
from src.data.dataset import decode_clip, uniform_frame_indices
from src.data.transforms import ClipTransform, ClipTransformConfig
from src.evaluate import load_selected_model
from src.utils import load_config, write_json


def frame_count(path: Path) -> int:
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


def evaluate(
    labels_path: Path,
    video_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    with labels_path.open(encoding="utf-8", newline="") as handle:
        samples = list(csv.DictReader(handle))
    if not samples:
        raise ValueError(f"No samples in {labels_path}")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_selected_model(checkpoint_path.resolve(), device)
    model.eval()
    transform = ClipTransform(
        ClipTransformConfig(
            frame_size=int(config["data"]["frame_size"]),
            resize_size=int(config["data"]["cache_resize_size"]),
        ),
        training=False,
    )

    predictions: list[dict[str, Any]] = []
    target_ids: list[int] = []
    predicted_ids: list[int] = []
    top3_hits = 0
    started = time.perf_counter()

    for sample in samples:
        label = sample["label"]
        if label not in EXPECTED_LABELS or label == "D0X":
            raise ValueError(f"Unsupported custom-test label: {label}")
        path = video_dir / sample["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        total_frames = frame_count(path)
        indices = uniform_frame_indices(1, total_frames, int(config["data"]["clip_len"]))
        frames = decode_clip(path, indices, decode_retries=int(config["data"]["decode_retries"]))
        clip = transform(frames).unsqueeze(0).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(config["train"]["amp"]) and device.type == "cuda",
            ):
                probabilities = model(clip).softmax(dim=1)[0]
        scores, class_ids = probabilities.topk(3)
        top3 = [
            {"label": EXPECTED_LABELS[int(class_id)], "confidence": float(score)}
            for score, class_id in zip(scores.cpu(), class_ids.cpu(), strict=True)
        ]
        target_id = EXPECTED_LABELS.index(label)
        predicted_id = int(class_ids[0])
        top3_correct = any(item["label"] == label for item in top3)
        top3_hits += int(top3_correct)
        target_ids.append(target_id)
        predicted_ids.append(predicted_id)
        predictions.append(
            {
                "file": sample["file"],
                "expected": label,
                "predicted": EXPECTED_LABELS[predicted_id],
                "confidence": float(scores[0]),
                "correct": predicted_id == target_id,
                "top3_correct": top3_correct,
                "top3": top3,
                "sampled_frame_indices": indices,
            }
        )

    gesture_ids = list(range(1, len(EXPECTED_LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        target_ids,
        predicted_ids,
        labels=gesture_ids,
        zero_division=0,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": "completed",
        "checkpoint": str(checkpoint_path.resolve()),
        "selected_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "sample_count": len(samples),
        "accuracy": float(accuracy_score(target_ids, predicted_ids)),
        "top3_accuracy": top3_hits / len(samples),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(sum(value * count for value, count in zip(f1, support, strict=True)) / len(samples)),
        "elapsed_seconds": elapsed,
        "per_class": {
            EXPECTED_LABELS[class_id]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_id in enumerate(gesture_ids)
        },
        "predictions": predictions,
    }


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("file", "expected", "predicted", "confidence", "correct", "top3_correct"),
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({field: prediction[field] for field in writer.fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/custom_test/labels.csv"))
    parser.add_argument("--video-dir", type=Path, default=Path("data/custom_test/prepared"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tsn_resnet18_seed42/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/custom_test/evaluation.json"))
    args = parser.parse_args()
    result = evaluate(args.labels, args.video_dir, args.config, args.checkpoint)
    write_json(args.output, result)
    write_predictions_csv(args.output.with_name("predictions.csv"), result["predictions"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
