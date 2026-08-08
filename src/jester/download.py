"""Resumable official Jester downloader using verified HTTP byte ranges."""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .acquire import PART_NAMES, PART_SIZES


BASE_URL = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester"
API_NAMES = tuple(f"20bnjester-v1-{index:02d}" for index in range(3))
REFERER = "https://www.qualcomm.com/developer/software/jester-dataset/downloads"
LICENSE_URL = (
    "https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/"
    "jester_something_something_exercise_research_license_final_qti_28jul2022.pdf"
)
DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024


def _fetch_range(url: str, start: int, end: int, retries: int = 10) -> bytes:
    expected = end - start + 1
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": REFERER,
                "Range": f"bytes={start}-{end}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 206:
                    raise RuntimeError(f"server returned HTTP {response.status}, expected 206")
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected Content-Range: {content_range!r}")
                data = response.read()
                if len(data) != expected:
                    raise RuntimeError(f"short range response: {len(data)} != {expected}")
                return data
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if attempt == retries:
                raise RuntimeError(f"range {start}-{end} failed after {retries} attempts") from error
            time.sleep(min(5 * attempt, 30))
    raise AssertionError("unreachable")


def aligned_resume_size(path: Path, expected_size: int, chunk_size: int) -> int:
    if not path.exists():
        return 0
    current = path.stat().st_size
    if current > expected_size:
        raise ValueError(f"download is larger than the official part: {path}")
    if current == expected_size:
        return current
    aligned = current - current % chunk_size
    if aligned != current:
        with path.open("r+b") as handle:
            handle.truncate(aligned)
    return aligned


def download_part(
    destination: Path,
    *,
    api_name: str,
    expected_size: int,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    workers: int = 3,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    current = aligned_resume_size(destination, expected_size, chunk_size)
    if current == expected_size:
        return {"name": destination.name, "bytes": current, "status": "already_complete"}
    url = f"{BASE_URL}/{api_name}"
    with destination.open("ab", buffering=0) as output:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while current < expected_size:
                ranges = []
                for _ in range(workers):
                    if current + sum(end - start + 1 for start, end in ranges) >= expected_size:
                        break
                    start = current + sum(end - begin + 1 for begin, end in ranges)
                    end = min(expected_size - 1, start + chunk_size - 1)
                    ranges.append((start, end))
                chunks = list(pool.map(lambda bounds: _fetch_range(url, *bounds), ranges))
                for (start, end), data in zip(ranges, chunks, strict=True):
                    if start != current:
                        raise RuntimeError("range ordering invariant failed")
                    output.write(data)
                    current = end + 1
                print(
                    f"JESTER_DOWNLOAD part={destination.name} bytes={current}/{expected_size} "
                    f"progress={current / expected_size:.1%}",
                    flush=True,
                )
    return {"name": destination.name, "bytes": current, "status": "complete"}


def download_all(downloads: Path, *, chunk_size: int, workers_per_part: int) -> dict[str, object]:
    with ThreadPoolExecutor(max_workers=len(PART_NAMES)) as pool:
        futures = [
            pool.submit(
                download_part,
                downloads / name,
                api_name=api_name,
                expected_size=size,
                chunk_size=chunk_size,
                workers=workers_per_part,
            )
            for name, api_name, size in zip(PART_NAMES, API_NAMES, PART_SIZES, strict=True)
        ]
        parts = [future.result() for future in futures]
    report: dict[str, object] = {"status": "complete", "parts": parts}
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("data/raw/jester/downloads"))
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--workers-per-part", type=int, default=3)
    parser.add_argument(
        "--accept-research-license",
        action="store_true",
        help="Confirm that the user has reviewed and accepts Qualcomm's Jester Research Use License.",
    )
    args = parser.parse_args()
    if args.chunk_mib < 1 or args.workers_per_part < 1:
        parser.error("chunk size and worker count must be positive")
    if not args.accept_research_license:
        parser.error(
            "download is disabled until the Qualcomm Jester Research Use License is accepted explicitly; "
            f"review {LICENSE_URL} and rerun with --accept-research-license"
        )
    acceptance_path = args.downloads.parent / "RESEARCH_LICENSE_ACCEPTED.json"
    acceptance_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path.write_text(
        json.dumps(
            {
                "license_url": LICENSE_URL,
                "accepted_explicitly": True,
                "scope": "internal non-profit research use only",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    download_all(
        args.downloads,
        chunk_size=args.chunk_mib * 1024 * 1024,
        workers_per_part=args.workers_per_part,
    )


if __name__ == "__main__":
    main()
