"""Run the frozen independent human-voice Parakeet -> Structured JSC benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.voice_e2e_benchmark import evaluate_voice_e2e, load_independent_manifest
from experiments.parakeet.benchmarks.compare_stt import audio_as_16k_wav
from experiments.parakeet.benchmarks.no_action import NoActionGuard
from ml.jsc.inference import StructuredJSCPredictor
from ml.jsc.project_registry import build_project_schema_registry
from modules.parakeet_client import DEFAULT_MODEL_DIR, PersistentParakeetClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", default="models/jsc/structured_v8_seed29.pt")
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--parakeet-python", default="venv/Scripts/python.exe")
    parser.add_argument("--parakeet-model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    NoActionGuard()
    manifest = Path(args.manifest).resolve()
    samples = load_independent_manifest(manifest)
    predictor = StructuredJSCPredictor(
        Path(args.checkpoint), build_project_schema_registry(), device="cpu"
    )
    with PersistentParakeetClient(
        repository_root=ROOT,
        model_dir=args.parakeet_model_dir,
        python_executable=args.parakeet_python,
        provider=args.provider,
    ) as client:
        def transcribe(path: Path) -> tuple[str, float]:
            wav, _samples, _seconds = audio_as_16k_wav(path)
            result = client.decode(wav)
            return str(result.get("text", "")), float(result.get("decode_ms", 0.0))

        def predict(text: str) -> tuple[str, float]:
            result = predictor.predict(text)
            return result.jal, result.latency_ms

        report = evaluate_voice_e2e(samples, transcribe=transcribe, predict=predict)
    report["manifest_sha256"] = __import__("hashlib").sha256(manifest.read_bytes()).hexdigest()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
