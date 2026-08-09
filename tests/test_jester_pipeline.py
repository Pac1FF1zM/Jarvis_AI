"""Regression tests for the isolated Jester training pipeline."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.jester.acquire import MultipartReader, extract_multipart
from src.jester.dataset import JesterDataset, ManifestRecord, temporal_indices
from src.jester.download import aligned_resume_size
from src.jester.labels import JESTER_LABELS
from src.jester.models import JesterModelConfig, build_model
from src.jester.prepare import read_labeled_split, read_labels
from src.jester.training import (
    _atomic_torch_save,
    _records_fingerprint,
    balanced_subset,
    config_fingerprint,
    load_jester_config,
    metrics,
    train_winner,
)


def test_official_label_order_and_metadata_parser(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("\n".join(JESTER_LABELS) + "\n", encoding="utf-8")
    split = tmp_path / "train.csv"
    split.write_text("1;Swiping Left\n2;No gesture\n", encoding="utf-8")

    assert read_labels(labels) == JESTER_LABELS
    rows = read_labeled_split(split, "train")
    assert [(row.clip_id, row.class_id) for row in rows] == [("1", 0), ("2", 25)]


def test_temporal_sampling_is_bounded_and_deterministic_for_evaluation():
    indices = temporal_indices(7, 16, training=False)
    assert len(indices) == 16
    assert indices == sorted(indices)
    assert min(indices) >= 1 and max(indices) <= 7


def test_dataset_decodes_numbered_jpegs(tmp_path):
    directory = tmp_path / "123"
    directory.mkdir()
    for index in range(1, 6):
        image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"{index:05d}.jpg"), image)
    record = ManifestRecord("123", JESTER_LABELS[0], 0, "train", "123", 5)
    dataset = JesterDataset(
        [record], frames_root=tmp_path, clip_len=4, frame_size=32, resize_size=40, training=False
    )

    clip, label = dataset[0]

    assert clip.shape == (4, 3, 32, 32)
    assert label == 0
    assert torch.isfinite(clip).all()


@pytest.mark.parametrize("name", ("tiny_3d_cnn", "cnn_bigru", "mobilenet_tsm_attention"))
def test_all_candidates_are_from_scratch_and_share_clip_contract(name):
    model = build_model(JesterModelConfig(name=name, num_classes=27, dropout=0.0)).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 4, 3, 32, 32))
    payload = model.checkpoint_payload(labels=list(JESTER_LABELS))

    assert logits.shape == (1, 27)
    assert payload["pretrained"] is False
    assert payload["kind"] == "jarvis_jester_from_scratch_v1"


def test_balanced_benchmark_subset_is_equal_per_class():
    records = [
        ManifestRecord(f"{class_id}-{sample}", label, class_id, "train", "x", 3)
        for class_id, label in enumerate(JESTER_LABELS)
        for sample in range(3)
    ]
    selected = balanced_subset(records, 2, seed=7)
    counts = {class_id: sum(row.class_id == class_id for row in selected) for class_id in range(27)}
    assert set(counts.values()) == {2}


def test_metrics_reports_negative_recall():
    target = list(range(27))
    result = metrics(target, target)
    assert result["macro_f1"] == pytest.approx(1.0)
    assert result["negative_recall"] == pytest.approx(1.0)


def test_multipart_stream_extracts_without_combined_archive(tmp_path):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        data = b"jpeg-data"
        info = tarfile.TarInfo("20bn-jester-v1/1/00001.jpg")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    raw = payload.getvalue()
    parts = []
    for index, chunk in enumerate((raw[:10], raw[10:31], raw[31:])):
        path = tmp_path / f"part-{index}"
        path.write_bytes(chunk)
        parts.append(path)

    output = tmp_path / "frames"
    assert extract_multipart(parts, output) == 1
    assert (output / "20bn-jester-v1/1/00001.jpg").read_bytes() == b"jpeg-data"


def test_multipart_reader_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        MultipartReader([])


def test_resumable_download_discards_only_incomplete_tail(tmp_path):
    path = tmp_path / "part"
    path.write_bytes(b"x" * 25)

    assert aligned_resume_size(path, expected_size=100, chunk_size=10) == 20
    assert path.stat().st_size == 20


def test_atomic_checkpoint_write_leaves_no_temporary_file(tmp_path):
    output = tmp_path / "latest.pt"
    _atomic_torch_save({"value": torch.tensor([1, 2, 3])}, output)

    assert torch.equal(torch.load(output, weights_only=True)["value"], torch.tensor([1, 2, 3]))
    assert not output.with_suffix(".pt.tmp").exists()


def test_split_fingerprint_detects_order_and_label_changes():
    base = [ManifestRecord("1", JESTER_LABELS[0], 0, "train", "1", 3)]
    changed = [ManifestRecord("1", JESTER_LABELS[1], 1, "train", "1", 3)]

    assert _records_fingerprint(base) != _records_fingerprint(changed)
    assert _records_fingerprint(base + changed) != _records_fingerprint(changed + base)


def test_config_fingerprint_is_order_independent_and_change_sensitive():
    assert config_fingerprint({"train": {"batch": 32}, "seed": 42}) == config_fingerprint(
        {"seed": 42, "train": {"batch": 32}}
    )
    assert config_fingerprint({"train": {"batch": 32}}) != config_fingerprint(
        {"train": {"batch": 64}}
    )


def test_benchmark_extends_beyond_warmup():
    config = load_jester_config(Path("configs/jester_from_scratch.yaml"))
    assert config["train"]["benchmark_epochs"] > config["train"]["warmup_epochs"]
    assert config["models"]["winner"] is None


def test_windows_training_launchers_enforce_preparation_order():
    root = Path(__file__).resolve().parents[1]
    setup = (root / "SETUP_JESTER_TRAINING.cmd").read_text(encoding="utf-8").casefold()
    prepare = (root / "PREPARE_JESTER_TRAINING.cmd").read_text(encoding="utf-8").casefold()
    benchmark = (root / "START_JESTER_BENCHMARK.cmd").read_text(encoding="utf-8").casefold()
    training = (root / "START_JESTER_TRAINING.cmd").read_text(encoding="utf-8").casefold()

    assert "python 3.10 is missing" in setup
    assert "torch.cuda.is_available" in setup
    assert "jester.quality_gate" in prepare
    assert "--write-profile configs/jester_hardware.yaml" in prepare
    assert "--require-ready" in benchmark
    assert "benchmark.json" in training


def test_full_training_rejects_stale_benchmark_report(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "benchmark.json").write_text(
        json.dumps({"recommended_winner": "tiny_3d_cnn", "config_fingerprint": "stale"}),
        encoding="utf-8",
    )
    config = {
        "data": {"manifest": str(tmp_path / "missing.jsonl")},
        "train": {"seed": 42, "epochs": 1},
        "models": {"winner": None},
        "paths": {"reports": str(reports), "runs": str(tmp_path / "runs")},
        "evaluation": {},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RuntimeError, match="different training configuration"):
        train_winner(config_path)
