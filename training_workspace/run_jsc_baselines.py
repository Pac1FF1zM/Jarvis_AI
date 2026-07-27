"""Run the fair multi-architecture, multi-seed JSC baseline protocol."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ml.jsc.baseline_training import TrainingConfig, inspect_training, train_baseline
from ml.jsc.models import ARCHITECTURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training_workspace/jsc_data")
    parser.add_argument("--output-root", default="training_workspace/jsc_runs")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 41))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    reports: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    for architecture in args.architectures:
        for seed in args.seeds:
            if (args.check_only or args.smoke) and seed != args.seeds[0]:
                continue
            run_dir = output_root / f"{architecture}_seed{seed}"
            latest = run_dir / "latest.pt"
            resume = str(latest) if args.resume_existing and latest.is_file() else None
            config = TrainingConfig(
                architecture=architecture,
                data_dir=args.data_dir,
                output_dir=str(run_dir),
                seed=seed,
                device=args.device,
                epochs=1 if args.smoke else args.epochs,
                batch_size=min(args.batch_size, 4) if args.smoke else args.batch_size,
                learning_rate=args.learning_rate,
                d_model=min(args.d_model, 32) if args.smoke else args.d_model,
                encoder_layers=1 if args.smoke else args.encoder_layers,
                decoder_layers=1 if args.smoke else args.decoder_layers,
                attention_heads=args.attention_heads,
                feedforward_dim=min(args.feedforward_dim, 64) if args.smoke else args.feedforward_dim,
                patience=args.patience,
                use_amp=not args.no_amp,
                resume=resume,
                smoke=args.smoke,
            )
            if args.check_only:
                inspection = inspect_training(config)
                inspection["planned_seeds"] = list(args.seeds)
                inspections.append(inspection)
            else:
                reports.append(train_baseline(config))
    if args.check_only:
        result: dict[str, Any] = {
            "check_only": True,
            "planned_runs": len(args.architectures) * len(args.seeds),
            "inspected_models": len(inspections),
            "inspections": inspections,
        }
    else:
        leaderboard = _leaderboard(reports)
        result = {
            "check_only": False,
            "smoke": args.smoke,
            "selection_uses": "validation generation metrics only",
            "test_used_for_selection": False,
            "evaluation_holdout_loaded": False,
            "runs": reports,
            "leaderboard": leaderboard,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            output_root / "leaderboard.json",
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _leaderboard(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        matching = [report for report in reports if report["architecture"] == architecture]
        if not matching:
            continue
        exact = [report["validation"]["generation"]["exact_jal_accuracy"] for report in matching]
        valid = [report["validation"]["generation"]["schema_valid_rate"] for report in matching]
        false_execution = [report["validation"]["generation"]["false_execution_rate"] for report in matching]
        rows.append(
            {
                "architecture": architecture,
                "seeds": [report["seed"] for report in matching],
                "validation_exact_jal_mean": statistics.fmean(exact),
                "validation_exact_jal_std": statistics.pstdev(exact),
                "validation_schema_valid_mean": statistics.fmean(valid),
                "validation_false_execution_mean": statistics.fmean(false_execution),
                "parameters": matching[0]["parameters"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -row["validation_exact_jal_mean"],
            -row["validation_schema_valid_mean"],
            row["validation_false_execution_mean"],
            row["parameters"],
        ),
    )


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
