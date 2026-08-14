from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.voice_e2e_benchmark import evaluate_voice_e2e, load_independent_manifest
from ml.jsc.jal import DialogueAct, JALPlan, dumps


def test_independent_voice_manifest_is_hash_frozen_and_evaluates(tmp_path: Path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"independent human audio sentinel")
    expected = dumps(JALPlan(DialogueAct.DIALOGUE, reason="general_chat"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "voice-1",
                "path": audio.name,
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "reference_text": "привет",
                "expected_jal": expected,
                "speaker_id": "speaker-a",
                "provenance": "independent_human",
                "consent": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    samples = load_independent_manifest(manifest, enforce_population=False)
    report = evaluate_voice_e2e(
        samples,
        transcribe=lambda _path: ("привет", 10.0),
        predict=lambda _text: (expected, 2.0),
    )

    assert report["execution"] == "blocked"
    assert report["metrics"]["wer"] == 0.0
    assert report["metrics"]["voice_semantic_exact"] == 1.0
    assert report["gates"]["passed"] is True
