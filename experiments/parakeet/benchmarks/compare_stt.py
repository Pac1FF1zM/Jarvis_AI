"""Paired, no-action Parakeet vs Whisper benchmark on a JSONL manifest."""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import re
import statistics
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any, Callable

from experiments.parakeet.benchmarks.no_action import NoActionGuard
from modules.parakeet_client import DEFAULT_MODEL_DIR, PersistentParakeetClient


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str) -> dict[str, int]:
    expected = normalize_text(reference)
    actual = normalize_text(hypothesis)
    ref_words, hyp_words = expected.split(), actual.split()
    ref_chars = list(expected.replace(" ", ""))
    hyp_chars = list(actual.replace(" ", ""))
    return {
        "word_errors": edit_distance(ref_words, hyp_words),
        "reference_words": len(ref_words),
        "char_errors": edit_distance(ref_chars, hyp_chars),
        "reference_chars": len(ref_chars),
        "exact": int(expected == actual),
    }


def load_manifest(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        row = json.loads(line)
        if not str(row.get("reference_text", "")).strip():
            raise ValueError(f"manifest line {line_number} has no human reference_text")
        audio_path = Path(str(row.get("path", "")))
        if not audio_path.is_absolute():
            audio_path = (path.parent / audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"manifest line {line_number}: {audio_path}")
        rows.append({**row, "path": str(audio_path)})
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("benchmark manifest contains no usable rows")
    return rows


def audio_as_16k_wav(path: Path) -> tuple[bytes, Any, float]:
    import librosa
    import numpy as np
    samples, _rate = librosa.load(path, sr=16_000, mono=True)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError(f"empty audio: {path}")
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(pcm)
    return output.getvalue(), samples, len(samples) / 16_000.0


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(engine: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    word_errors = sum(row["word_errors"] for row in rows)
    reference_words = sum(row["reference_words"] for row in rows)
    char_errors = sum(row["char_errors"] for row in rows)
    reference_chars = sum(row["reference_chars"] for row in rows)
    latencies = [float(row["latency_ms"]) for row in rows]
    total_audio = sum(float(row["audio_seconds"]) for row in rows)
    return {
        "engine": engine,
        "utterances": len(rows),
        "wer": word_errors / max(reference_words, 1),
        "cer": char_errors / max(reference_chars, 1),
        "exact_match_rate": sum(row["exact"] for row in rows) / len(rows),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
        },
        "rtf": sum(latencies) / 1000.0 / max(total_audio, 0.001),
    }


def _run_engine(
    engine: str,
    fixtures: list[dict[str, Any]],
    transcribe: Callable[[dict[str, Any]], tuple[str, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        started = time.perf_counter()
        text, reported_ms = transcribe(fixture)
        wall_ms = (time.perf_counter() - started) * 1000.0
        counts = error_counts(str(fixture["reference_text"]), text)
        results.append(
            {
                "id": fixture.get("id", Path(fixture["path"]).stem),
                "reference_text": fixture["reference_text"],
                "hypothesis": text,
                "audio_seconds": fixture["audio_seconds"],
                "latency_ms": reported_ms or wall_ms,
                **counts,
            }
        )
    return results, summarize(engine, results)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    NoActionGuard()
    repository_root = Path(args.repository_root).resolve()
    manifest = Path(args.manifest).resolve()
    rows = load_manifest(manifest, limit=args.limit)
    fixtures: list[dict[str, Any]] = []
    for row in rows:
        wav_bytes, samples, seconds = audio_as_16k_wav(Path(row["path"]))
        fixtures.append(
            {
                **row,
                "wav_bytes": wav_bytes,
                "samples": samples,
                "audio_seconds": seconds,
            }
        )

    engines: dict[str, Any] = {}
    details: dict[str, Any] = {}

    # Run heavyweight engines sequentially so their VRAM footprints never overlap.
    if args.order == "whisper-first":
        order = ("whisper", "parakeet")
    else:
        order = ("parakeet", "whisper")
    for engine in order:
        if engine == "whisper":
            import torch
            import whisper

            load_started = time.perf_counter()
            model = whisper.load_model(
                args.whisper_model,
                device=args.provider,
                download_root=str(repository_root / "models/openai-whisper"),
            )
            load_ms = (time.perf_counter() - load_started) * 1000.0

            def transcribe_whisper(fixture: dict[str, Any]) -> tuple[str, float]:
                result = model.transcribe(
                    fixture["samples"],
                    language="ru",
                    task="transcribe",
                    fp16=args.provider == "cuda",
                    verbose=None,
                    temperature=0.0,
                    beam_size=3,
                    patience=1.0,
                    condition_on_previous_text=False,
                )
                return str(result.get("text", "")).strip(), 0.0

            engine_rows, summary = _run_engine(engine, fixtures, transcribe_whisper)
            summary["model"] = args.whisper_model
            summary["model_load_ms"] = load_ms
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            with PersistentParakeetClient(
                repository_root=repository_root,
                model_dir=args.parakeet_model_dir,
                python_executable=args.parakeet_python,
                provider=args.provider,
                timeout_seconds=args.timeout,
            ) as client:
                startup = dict(client.startup or {})

                def transcribe_parakeet(fixture: dict[str, Any]) -> tuple[str, float]:
                    result = client.decode(fixture["wav_bytes"])
                    return str(result.get("text", "")).strip(), float(
                        result.get("decode_ms") or 0.0
                    )

                engine_rows, summary = _run_engine(engine, fixtures, transcribe_parakeet)
                summary["model"] = startup.get("health", {}).get("model_id")
                summary["model_revision"] = startup.get("health", {}).get(
                    "model_revision"
                )
                summary["model_load_ms"] = startup.get("model_load_ms")
                summary["warm_up_ms"] = startup.get("warm_up_ms")
        engines[engine] = summary
        details[engine] = engine_rows

    return {
        "schema_version": 1,
        "mode": "paired_no_action_stt_benchmark",
        "manifest": str(manifest),
        "provider": args.provider,
        "execution": "blocked",
        "summaries": engines,
        "winner_by_wer": min(engines, key=lambda item: engines[item]["wer"]),
        "results": details,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repository-root", default=str(root))
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--parakeet-model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--parakeet-python", default="venv/Scripts/python.exe")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--order",
        choices=("parakeet-first", "whisper-first"),
        default="parakeet-first",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_benchmark(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
