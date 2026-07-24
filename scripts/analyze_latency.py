"""Parse a Jarvis log file and print per-module average latency.

Reads ``logs/jarvis.log`` (or a path argument), groups events by ``trace_id``,
and computes:

  * per-stage latency, derived from the gap between consecutive PUBLISH lines
    within a trace (each stage = a module's work between receiving an event
    and publishing the next),
  * GPU **wait** vs **execution** time, taken directly from the
    ``GPU_ACQUIRE``/``GPU_RELEASE`` lines that ``core/gpu_lock.py`` emits,
  * end-to-end latency per trace.

Usage::

    python scripts/analyze_latency.py
    python scripts/analyze_latency.py path/to/jarvis.log
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = "logs/jarvis.log"

# A log line looks like:
#   2026-07-18 01:42:03.123 INFO jarvis.bus | PUBLISH transcription_ready trace=abc12
# We need the timestamp, the marker keyword, and any key=val tokens.
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+\w+\s+[\w.]+\s*\|\s+(?P<rest>.*)$"
)
_TRACE_RE = re.compile(r"\btrace=(?P<trace>\w+)")
_KV_RE = re.compile(r"(?P<key>\w+)=(?P<val>[-\w.]+)")


def _parse_ts(ts: str) -> float:
    # datetime.fromisoformat handles the "YYYY-MM-DD HH:MM:SS.fff" form.
    import datetime as dt

    return dt.datetime.fromisoformat(ts).timestamp()


def parse_log(path: Path) -> tuple[
    dict[str, list[tuple[float, str]]],
    dict[str, list[tuple[str, float, float]]],
]:
    """Return ``(per_trace_events, gpu_samples)``.

    - ``per_trace_events``: trace_id -> list of (timestamp, event_type).
    - ``gpu_samples``: list of (label, exec_ms, wait_ms) (no trace grouping;
      gpu_lock logs don't carry trace_id, which is fine — we aggregate by label).
    """
    per_trace: dict[str, list[tuple[float, str]]] = defaultdict(list)
    gpu: list[tuple[str, float, float]] = []

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            m = _LINE_RE.match(raw.rstrip("\n"))
            if not m:
                continue
            ts = _parse_ts(m.group("ts"))
            rest = m.group("rest")
            trace_match = _TRACE_RE.search(rest)
            trace_id = trace_match.group("trace") if trace_match else None

            if rest.startswith("PUBLISH "):
                event_type = rest[len("PUBLISH ") :].split()[0]
                if trace_id is not None:
                    per_trace[trace_id].append((ts, event_type))
            elif rest.startswith("GPU_RELEASE"):
                kv = {k: v for k, v in _KV_RE.findall(rest)}
                try:
                    label = kv.get("label", "?")
                    exec_ms = float(kv.get("exec_ms", 0))
                    wait_ms = float(kv.get("wait_ms", 0))
                    gpu.append((label, exec_ms, wait_ms))
                except ValueError:
                    continue
    return per_trace, gpu


def stage_latency(per_trace: dict[str, list[tuple[float, str]]]) -> dict[str, list[float]]:
    """For each event type, the time between it and the *previous* event in
    the same trace — i.e. how long the producing module took to respond.
    """
    stages: dict[str, list[float]] = defaultdict(list)
    for events in per_trace.values():
        events.sort(key=lambda x: x[0])
        for i in range(1, len(events)):
            prev_ts, _ = events[i - 1]
            cur_ts, cur_type = events[i]
            stages[cur_type].append((cur_ts - prev_ts) * 1000.0)
    return stages


def gpu_latency(gpu: list[tuple[str, float, float]]) -> dict[str, dict[str, float]]:
    """Aggregate GPU samples by label -> {exec_ms, wait_ms} averages."""
    by_label: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"exec": [], "wait": []}
    )
    for label, exec_ms, wait_ms in gpu:
        by_label[label]["exec"].append(exec_ms)
        by_label[label]["wait"].append(wait_ms)
    return {
        label: {
            "exec_ms": statistics.mean(v["exec"]) if v["exec"] else 0.0,
            "wait_ms": statistics.mean(v["wait"]) if v["wait"] else 0.0,
            "samples": len(v["exec"]),
        }
        for label, v in by_label.items()
    }


def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"avg={statistics.mean(values):7.2f}ms "
        f"min={min(values):7.2f}ms "
        f"max={max(values):7.2f}ms "
        f"n={len(values)}"
    )


def report(path: Path) -> int:
    if not path.exists():
        print(f"log file not found: {path}", file=sys.stderr)
        return 1
    per_trace, gpu = parse_log(path)

    print(f"\n=== Latency report for {path} ===")
    print(f"traces seen: {len(per_trace)}\n")

    print("-- per-stage latency (gap from previous event in same trace) --")
    stages = stage_latency(per_trace)
    if not stages:
        print("  (no PUBLISH lines found)")
    for event_type in sorted(stages):
        print(f"  {event_type:24s} {_fmt(stages[event_type])}")

    print("\n-- GPU contention (from GPU_ACQUIRE/GPU_RELEASE lines) --")
    gpu_stats = gpu_latency(gpu)
    if not gpu_stats:
        print("  (no GPU_RELEASE lines found)")
    for label in sorted(gpu_stats):
        s = gpu_stats[label]
        print(
            f"  {label:8s} exec_avg={s['exec_ms']:7.2f}ms "
            f"wait_avg={s['wait_ms']:7.2f}ms samples={s['samples']}"
        )

    print("\n-- end-to-end latency per trace --")
    if not per_trace:
        print("  (no traces)")
    for trace_id, events in sorted(per_trace.items()):
        events.sort(key=lambda x: x[0])
        if len(events) >= 2:
            e2e = (events[-1][0] - events[0][0]) * 1000.0
            print(f"  trace={trace_id} events={len(events)} e2e={e2e:.2f}ms")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", default=DEFAULT_LOG)
    args = parser.parse_args()
    sys.exit(report(Path(args.log)))


if __name__ == "__main__":
    main()
