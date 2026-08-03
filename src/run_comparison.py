"""Fail-fast autonomous runner for the approved architecture comparison."""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import resolve_from_project, write_json


STATE_PATH = resolve_from_project("reports/comparison_orchestrator.json")
LOG_DIR = resolve_from_project("logs/ipn_tsn/comparison_orchestrator")


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save(state: dict[str, Any]) -> None:
    state["updated_at"] = _stamp()
    write_json(STATE_PATH, state)


def _run(state: dict[str, Any], name: str, arguments: list[str], timeout_hours: float) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{name}.out.log"
    stderr_path = LOG_DIR / f"{name}.err.log"
    state["current_step"] = name
    state["steps"][name] = {"status": "running", "started_at": _stamp()}
    _save(state)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            [sys.executable, "-u", *arguments],
            cwd=resolve_from_project("."),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout_hours * 3600,
            check=False,
        )
    state["steps"][name].update(
        status="completed" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        finished_at=_stamp(),
        stdout=str(stdout_path),
        stderr=str(stderr_path),
    )
    _save(state)
    if completed.returncode:
        raise RuntimeError(f"Step {name} failed with return code {completed.returncode}; see {stderr_path}")


def _wait_for_existing_mobile_training(state: dict[str, Any]) -> None:
    report = resolve_from_project("reports/training_tsn_mobilenet_v3_small.json")
    name = "wait_mobile_training"
    state["current_step"] = name
    state["steps"][name] = {"status": "waiting", "started_at": _stamp(), "target": str(report)}
    _save(state)
    deadline = time.monotonic() + 12 * 3600
    while not report.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"MobileNet training report did not appear within 12 hours: {report}")
        time.sleep(60)
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("status") != "completed" or payload.get("epochs_completed") != 40:
        raise RuntimeError(f"MobileNet training report is not a completed 40-epoch run: {report}")
    state["steps"][name].update(status="completed", finished_at=_stamp())
    _save(state)


def main() -> None:
    state: dict[str, Any] = {
        "status": "running",
        "started_at": _stamp(),
        "current_step": None,
        "steps": {},
        "policy": "gates -> full train/val -> one official test per new model",
    }
    _save(state)
    try:
        _wait_for_existing_mobile_training(state)
        _run(state, "test_mobile_once", ["-m", "src.evaluate", "--config", "configs/mobilenet_tsn.yaml", "--checkpoint", "checkpoints/tsn_mobilenet_v3_small_seed42/best.pt"], 12)
        _run(state, "gates_r3d18", ["-m", "src.gates", "--config", "configs/r3d18.yaml"], 6)
        _run(state, "train_r3d18", ["-m", "src.train", "--config", "configs/r3d18.yaml", "--run-name", "r3d_18_seed42"], 72)
        _run(state, "test_r3d18_once", ["-m", "src.evaluate", "--config", "configs/r3d18.yaml", "--checkpoint", "checkpoints/r3d_18_seed42/best.pt"], 12)
        _run(state, "gates_r2plus1d18", ["-m", "src.gates", "--config", "configs/r2plus1d18.yaml"], 6)
        _run(state, "train_r2plus1d18", ["-m", "src.train", "--config", "configs/r2plus1d18.yaml", "--run-name", "r2plus1d_18_seed42"], 72)
        _run(state, "test_r2plus1d18_once", ["-m", "src.evaluate", "--config", "configs/r2plus1d18.yaml", "--checkpoint", "checkpoints/r2plus1d_18_seed42/best.pt"], 12)
        _run(state, "comparison_report", ["-m", "src.compare_report"], 1)
        state["status"] = "completed"
        state["current_step"] = None
    except Exception as error:  # noqa: BLE001 - state must capture any fail-fast stop
        state["status"] = "failed"
        state["error"] = str(error)
        state["traceback"] = traceback.format_exc()
        raise
    finally:
        _save(state)


if __name__ == "__main__":
    main()
