"""Verify and safely stream-extract the official three-part Jester TGZ."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import BinaryIO


PART_NAMES = tuple(f"20bn-jester-v1-{index:02d}" for index in range(3))
PART_SIZES = (10_000_000_000, 10_000_000_000, 2_930_724_987)


class MultipartReader(io.RawIOBase):
    """Read several binary files as one non-seekable stream."""

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        if not paths:
            raise ValueError("at least one archive part is required")
        self._paths = paths
        self._index = 0
        self._handle: BinaryIO | None = paths[0].open("rb")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer)
        written = 0
        while written < len(view) and self._handle is not None:
            count = self._handle.readinto(view[written:])
            if count:
                written += count
                continue
            self._handle.close()
            self._index += 1
            self._handle = (
                self._paths[self._index].open("rb")
                if self._index < len(self._paths)
                else None
            )
        return written

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        super().close()


def official_parts(downloads: Path) -> list[Path]:
    paths = [downloads / name for name in PART_NAMES]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing official Jester archive parts: {missing}")
    wrong_sizes = {
        path.name: {"actual": path.stat().st_size, "expected": expected}
        for path, expected in zip(paths, PART_SIZES, strict=True)
        if path.stat().st_size != expected
    }
    if wrong_sizes:
        raise ValueError(f"Jester archive parts are incomplete or changed: {wrong_sizes}")
    return paths


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_download_manifest(paths: list[Path], output: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": "Qualcomm Jester Dataset official download",
        "parts": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in paths
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"unsafe archive path: {member_name!r}")
    return target


def extract_multipart(parts: list[Path], output: Path) -> int:
    """Extract regular files/directories only; reject links and special entries."""
    output.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    with MultipartReader(parts) as raw, io.BufferedReader(raw, 8 * 1024 * 1024) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                target = _safe_target(output, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError(f"unsupported archive member: {member.name!r}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member has no data: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
                extracted_files += 1
    return extracted_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("data/raw/jester/downloads"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/jester/frames"))
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/jester/download_manifest.json"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    parts = official_parts(args.downloads)
    manifest = write_download_manifest(parts, args.manifest)
    if args.verify_only:
        print(json.dumps(manifest, indent=2))
        return
    free = shutil.disk_usage(args.output.parent.resolve()).free
    if free < 24 * 1024**3:
        raise RuntimeError(f"at least 24 GiB free is required for extraction; available={free / 1024**3:.2f}")
    count = extract_multipart(parts, args.output)
    print(json.dumps({"status": "extracted", "files": count, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
