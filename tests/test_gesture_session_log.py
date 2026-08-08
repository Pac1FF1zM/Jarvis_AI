from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

from modules.gesture_session_log import GestureSessionLog


def test_gesture_session_log_writes_jsonl_without_frames(tmp_path):
    log = GestureSessionLog(tmp_path, retention_days=30, max_total_bytes=100_000)
    path = log.start(model="test", camera_index=0)
    log.write(
        "prediction",
        label="G01",
        confidence=0.95,
        top3=[{"label": "G01", "confidence": 0.95}],
    )
    log.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event"] == "session_started"
    assert rows[1]["event"] == "prediction"
    assert rows[1]["label"] == "G01"
    assert all("frame" not in row and "video" not in row for row in rows)


def test_gesture_log_cleanup_applies_age_and_total_size_limits(tmp_path):
    old = tmp_path / "gesture_session_old.jsonl"
    old.write_bytes(b"x" * 2000)
    old_time = (datetime.now(timezone.utc) - timedelta(days=31)).timestamp()
    os.utime(old, (old_time, old_time))
    recent_a = tmp_path / "gesture_session_recent_a.jsonl"
    recent_b = tmp_path / "gesture_session_recent_b.jsonl"
    recent_a.write_bytes(b"a" * 1200)
    recent_b.write_bytes(b"b" * 1200)

    log = GestureSessionLog(tmp_path, retention_days=30, max_total_bytes=1800)
    log.cleanup()

    assert not old.exists()
    assert sum(path.stat().st_size for path in tmp_path.glob("*.jsonl")) <= 1800
