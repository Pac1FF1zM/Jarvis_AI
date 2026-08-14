"""Report which NLU -> JSC migration stages current audited evidence admits."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.migration import MigrationStage, admit_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", default="models/JSC_MIGRATION_STATE.json")
    args = parser.parse_args()
    path = Path(args.evidence)
    evidence = json.loads(path.read_text("utf-8"))
    release = (ROOT / "VERSION").read_text("utf-8").strip()
    stages = {}
    for stage in MigrationStage:
        admission = admit_stage(stage, evidence)
        stages[stage.config_name] = {
            "admitted": admission.admitted,
            "active_stage": admission.active.config_name,
            "reasons": list(admission.reasons),
        }
    report = {
        "schema_version": 1,
        "evidence": str(path),
        "active_status": evidence.get("status", "unknown"),
        "release": release,
        "evidence_release_matches": evidence.get("active_release") == release,
        "stages": stages,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["evidence_release_matches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
