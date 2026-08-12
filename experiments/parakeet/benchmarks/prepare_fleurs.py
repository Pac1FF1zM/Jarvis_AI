"""Build a deterministic local JSONL benchmark from a downloaded FLEURS split."""
from __future__ import annotations

import argparse
import csv
import json
import tarfile
from pathlib import Path


def select_rows(
    tsv_path: Path, limit: int, *, max_audio_seconds: float | None = None
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_sentences: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row in reader:
            if len(row) < 7:
                raise ValueError(f"invalid FLEURS row with {len(row)} columns")
            sentence_id, filename, _raw, normalized, _chars, samples, gender = row[
                :7
            ]
            if (
                max_audio_seconds is not None
                and int(samples) / 16_000 > max_audio_seconds
            ):
                continue
            if sentence_id in seen_sentences:
                continue
            seen_sentences.add(sentence_id)
            selected.append(
                {
                    "id": f"fleurs-ru-dev-{sentence_id}",
                    "filename": filename,
                    "reference_text": normalized,
                    "gender": gender.casefold(),
                }
            )
            if len(selected) >= limit:
                break
    if len(selected) < limit:
        raise ValueError(f"requested {limit} unique sentences, found {len(selected)}")
    return selected


def build_manifest(
    tsv_path: Path,
    archive_path: Path,
    output_dir: Path,
    limit: int,
    *,
    max_audio_seconds: float | None = None,
) -> Path:
    rows = select_rows(tsv_path, limit, max_audio_seconds=max_audio_seconds)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {
            member.name: member
            for member in archive.getmembers()
            if member.isfile()
        }
        for row in rows:
            member_name = f"dev/{row['filename']}"
            member = members.get(member_name)
            if member is None:
                raise FileNotFoundError(f"{member_name} is absent from {archive_path}")
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"cannot read {member_name}")
            target = audio_dir / row["filename"]
            target.write_bytes(source.read())

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            payload = {
                "id": row["id"],
                "path": f"audio/{row['filename']}",
                "reference_text": row["reference_text"],
                "dataset": "google/fleurs",
                "split": "ru_ru/dev",
                "gender": row["gender"],
            }
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-audio-seconds", type=float)
    args = parser.parse_args()
    manifest = build_manifest(
        Path(args.tsv).resolve(),
        Path(args.archive).resolve(),
        Path(args.output_dir).resolve(),
        args.limit,
        max_audio_seconds=args.max_audio_seconds,
    )
    print(manifest)


if __name__ == "__main__":
    main()
