"""Fast regression coverage for the from-scratch gesture training components."""
from __future__ import annotations

import json

import pytest
import torch

from ml.gesture.data import (
    GestureSegment,
    _read_video_frame,
    _sample_indices,
    audit_segments,
    import_ipn_segments,
    write_manifest,
)
from ml.gesture.labels import IPN_LABELS
from ml.gesture.models import ARCHITECTURES, GestureModelConfig, build_model, checkpoint_payload
from ml.gesture.training import _metrics


def _segment(path, split: str, label: str, video_id: str) -> GestureSegment:
    path.write_bytes(b"placeholder")
    return GestureSegment(
        video=str(path), label=label, start_frame=1, end_frame=4, split=split, video_id=video_id
    )


def test_ipn_import_uses_official_splits_and_writes_auditable_manifest(tmp_path):
    videos = tmp_path / "videos"
    annotations = tmp_path / "annotations"
    videos.mkdir()
    annotations.mkdir()
    for name in ("subject_a.mp4", "subject_b.mp4", "subject_c.mp4"):
        (videos / name).write_bytes(b"not decoded by import")
    source = {
        "labels": list(IPN_LABELS),
        "database": {
            "subject_a.mp4^first": {
                "subset": "training",
                "annotations": {"label": "D0X", "start_frame": 1, "end_frame": 12},
            },
            "subject_b.mp4^first": {
                "subset": "validation",
                "annotations": {"label": "G01", "start_frame": 2, "end_frame": 15},
            },
            "subject_c.mp4^first": {
                "subset": "testing",
                "annotations": {"label": "G11", "start_frame": 3, "end_frame": 16},
            },
        },
    }
    (annotations / "annotations.json").write_text(json.dumps(source), encoding="utf-8")

    records = import_ipn_segments(videos, annotations)
    report = write_manifest(records, tmp_path / "manifest.jsonl")

    assert [record.split for record in records] == ["train", "validation", "test"]
    assert report["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["videos_per_split"] == {"train": 1, "validation": 1, "test": 1}


def test_ipn_import_reads_google_drive_text_annotations_without_subject_leakage(
    tmp_path,
):
    videos = tmp_path / "videos"
    annotations = tmp_path / "annotations"
    videos.mkdir()
    annotations.mkdir()
    class_index = "id,label\n" + "\n".join(
        f"{index},{label}" for index, label in enumerate(IPN_LABELS, 1)
    )
    (annotations / "classIdx.txt").write_text(class_index, encoding="utf-8")

    train_rows = []
    for subject in range(1, 6):
        name = f"1CM42_{subject}_R_#{subject}"
        (videos / f"{name}.avi").write_bytes(b"not decoded by import")
        label = IPN_LABELS[subject - 1]
        train_rows.append(f"{name},{label},{subject},1,12,12")
    test_rows = []
    for subject in range(6, 8):
        name = f"1CM42_{subject}_R_#{subject}"
        (videos / f"{name}.mp4").write_bytes(b"not decoded by import")
        label = IPN_LABELS[subject - 1]
        test_rows.append(f"{name},{label},{subject},2,15,14")
    (annotations / "Annot_TrainList.txt").write_text(
        "\n".join(train_rows), encoding="utf-8"
    )
    (annotations / "Annot_TestList.txt").write_text(
        "\n".join(test_rows), encoding="utf-8"
    )

    records = import_ipn_segments(videos, annotations)
    repeated = import_ipn_segments(videos, annotations)

    def subjects(split):
        return {
            "_".join(
                record.video_id.replace("\\", "/").split("/")[-1].split("_")[:2]
            )
            for record in records
            if record.split == split
        }

    assert records == repeated
    assert {record.split for record in records} == {"train", "validation", "test"}
    assert subjects("train").isdisjoint(subjects("validation"))
    assert subjects("train").isdisjoint(subjects("test"))
    assert subjects("validation").isdisjoint(subjects("test"))
    assert len(subjects("validation")) == 1
    assert len(subjects("test")) == 2


def test_audit_rejects_video_leakage_and_temporal_sampler_is_bounded(tmp_path):
    video = tmp_path / "shared.mp4"
    records = [_segment(video, "train", "D0X", "same"), _segment(video, "test", "G01", "same")]
    with pytest.raises(ValueError, match="leakage"):
        audit_segments(records)

    assert _sample_indices(4, 7, 8, training=False) == [4, 4, 5, 5, 6, 6, 7, 7]
    assert all(4 <= item <= 20 for item in _sample_indices(4, 20, 8, training=True))


def test_video_decoder_backtracks_only_near_an_avi_boundary():
    class Capture:
        def __init__(self):
            self.position = None

        def set(self, _property, position):
            self.position = int(position)

        def read(self):
            if self.position == 9:
                return False, None
            return True, torch.tensor(self.position).numpy()

    frame, decoded_index = _read_video_frame(
        Capture(), 10, position_property=1, max_backtrack=4
    )

    assert decoded_index == 9
    assert int(frame) == 8


def test_video_decoder_can_reach_first_frame_from_early_avi_annotation():
    class Capture:
        def __init__(self):
            self.position = None

        def set(self, _property, position):
            self.position = int(position)

        def read(self):
            return self.position == 0, torch.tensor(self.position).numpy()

    frame, decoded_index = _read_video_frame(
        Capture(), 5, position_property=1, max_backtrack=4
    )

    assert decoded_index == 1
    assert int(frame) == 0


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_each_from_scratch_video_model_backpropagates_and_serializes(architecture):
    config = GestureModelConfig(architecture=architecture, classes=len(IPN_LABELS), width=8, dropout=0.0)
    model = build_model(config)
    clip = torch.rand(2, 3, 8, 32, 32)
    logits = model(clip)
    logits.mean().backward()
    payload = checkpoint_payload(model, config)

    assert logits.shape == (2, len(IPN_LABELS))
    assert payload["pretrained"] is False
    assert payload["model_config"]["architecture"] == architecture
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_metrics_make_false_trigger_rate_explicit():
    d0x, g01 = IPN_LABELS.index("D0X"), IPN_LABELS.index("G01")
    metrics = _metrics([d0x, d0x, g01], [d0x, g01, g01])

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["no_gesture_recall"] == pytest.approx(1 / 2)
    assert metrics["false_trigger_rate"] == pytest.approx(1 / 2)
