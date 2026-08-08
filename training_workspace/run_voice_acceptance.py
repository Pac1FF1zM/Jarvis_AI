"""Analyse real Jarvis sessions and compare safe voice-pipeline candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.voice_acceptance import evaluate_candidates, parse_session_logs, summarise_turns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default="logs/sessions", help="Directory with session .txt logs")
    parser.add_argument(
        "--since",
        help="Optional inclusive YYYYMMDD filename cutoff (for example 20260807)",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    paths = sorted(log_dir.glob("*.txt"))
    if args.since:
        paths = [path for path in paths if path.name >= f"jarvis_session_{args.since}"]
    turns = parse_session_logs(paths)
    results = evaluate_candidates(turns)
    report = {
        "baseline": dict(summarise_turns(turns)),
        "winner": results[0].candidate.name,
        "candidates": [
            {
                "name": result.candidate.name,
                "eligible": result.eligible,
                "median_command_ready_ms": round(result.median_command_ready_ms, 2),
                "p95_command_ready_ms": round(result.p95_command_ready_ms, 2),
                "median_end_to_end_ms": round(result.median_end_to_end_ms, 2),
                "p95_end_to_end_ms": round(result.p95_end_to_end_ms, 2),
                "description": result.candidate.description,
            }
            for result in results
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
