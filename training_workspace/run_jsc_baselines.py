"""Run validation-only tuning and multi-seed JSC baseline confirmation."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ml.jsc.baseline_training import (
    TrainingConfig,
    evaluate_locked_test,
    inspect_training,
    train_baseline,
)
from ml.jsc.models import ARCHITECTURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training_workspace/jsc_data")
    parser.add_argument("--output-root", default="training_workspace/jsc_runs")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(17, 29, 41))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--pilot-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=(2e-4, 5e-4),
    )
    parser.add_argument("--dropouts", nargs="+", type=float, default=(0.10, 0.20))
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--copy-mechanism", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--structured-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--parameter-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    output_root = Path(args.output_root)
    if args.check_only:
        result = _check_only(args)
    elif args.smoke:
        smoke_session = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        reports = [
            _run(
                args,
                architecture,
                args.seeds[0],
                args.learning_rates[0],
                args.dropouts[0],
                output_root
                / "smoke"
                / smoke_session
                / f"{architecture}_seed{args.seeds[0]}",
                epochs=1,
                smoke=True,
            )
            for architecture in args.architectures
        ]
        result = {
            "check_only": False,
            "smoke": True,
            "test_opened": False,
            "evaluation_holdout_loaded": False,
            "runs": reports,
            "leaderboard": _leaderboard(reports, args.architectures),
        }
    else:
        result = _full_protocol(args, output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            output_root / "leaderboard.json",
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _full_protocol(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    if args.skip_sweep:
        selected_hyperparameters = {
            architecture: {
                "learning_rate": args.learning_rates[0],
                "dropout": args.dropouts[0],
            }
            for architecture in args.architectures
        }
        pilot_reports: list[dict[str, Any]] = []
    else:
        pilot_reports = []
        for architecture in args.architectures:
            for learning_rate in args.learning_rates:
                for dropout in args.dropouts:
                    name = _configuration_name(architecture, learning_rate, dropout)
                    pilot_reports.append(
                        _run(
                            args,
                            architecture,
                            args.seeds[0],
                            learning_rate,
                            dropout,
                            output_root / "pilot" / name,
                            epochs=args.pilot_epochs,
                        )
                    )
        selected_hyperparameters = _select_hyperparameters(
            pilot_reports,
            args.architectures,
        )
    confirmation_reports: list[dict[str, Any]] = []
    for architecture in args.architectures:
        selected = selected_hyperparameters[architecture]
        for seed in args.seeds:
            name = _configuration_name(
                architecture,
                selected["learning_rate"],
                selected["dropout"],
                seed,
            )
            confirmation_reports.append(
                _run(
                    args,
                    architecture,
                    seed,
                    selected["learning_rate"],
                    selected["dropout"],
                    output_root / "confirmation" / name,
                    epochs=args.epochs,
                )
            )
    leaderboard = _leaderboard(confirmation_reports, args.architectures)
    selected_architecture = leaderboard[0]["architecture"]
    selection = {
        "selected_architecture": selected_architecture,
        "selected_hyperparameters": selected_hyperparameters,
        "leaderboard": leaderboard,
        "selection_source": "validation only",
        "test_opened": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        output_root / "selection_before_test.json",
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
    )
    selected_reports = [
        report
        for report in confirmation_reports
        if report["architecture"] == selected_architecture
    ]
    test_runs = [
        evaluate_locked_test(
            report["checkpoint"],
            args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
        for report in selected_reports
    ]
    return {
        "format_version": 1,
        "smoke": False,
        "pilot_sweep_enabled": not args.skip_sweep,
        "test_used_for_selection": False,
        "test_opened_after_selection": True,
        "evaluation_holdout_loaded": False,
        "pilot_runs": pilot_reports,
        "selected_hyperparameters": selected_hyperparameters,
        "confirmation_runs": confirmation_reports,
        "validation_leaderboard": leaderboard,
        "selected_architecture": selected_architecture,
        "selected_test_runs": test_runs,
        "selected_test_summary": _test_summary(test_runs),
    }


def _check_only(args: argparse.Namespace) -> dict[str, Any]:
    inspections: list[dict[str, Any]] = []
    for architecture in args.architectures:
        config = _config(
            args,
            architecture,
            args.seeds[0],
            args.learning_rates[0],
            args.dropouts[0],
            output_dir="unused",
            epochs=args.epochs,
        )
        inspection = inspect_training(config)
        inspection["planned_seeds"] = list(args.seeds)
        inspections.append(inspection)
    pilot_count = 0 if args.skip_sweep else (
        len(args.architectures) * len(args.learning_rates) * len(args.dropouts)
    )
    confirmation_count = len(args.architectures) * len(args.seeds)
    return {
        "check_only": True,
        "planned_pilot_runs": pilot_count,
        "planned_confirmation_runs": confirmation_count,
        "planned_total_runs": pilot_count + confirmation_count,
        "test_loaded": False,
        "evaluation_holdout_loaded": False,
        "inspections": inspections,
    }


def _run(
    args: argparse.Namespace,
    architecture: str,
    seed: int,
    learning_rate: float,
    dropout: float,
    output_dir: Path,
    *,
    epochs: int,
    smoke: bool = False,
) -> dict[str, Any]:
    completed = _load_completed_report(
        args,
        output_dir,
        architecture=architecture,
        seed=seed,
        learning_rate=learning_rate,
        dropout=dropout,
        smoke=smoke,
    )
    if completed is not None:
        print(f"resume: using completed report {Path(output_dir) / 'report.json'}")
        return completed
    config = _config(
        args,
        architecture,
        seed,
        learning_rate,
        dropout,
        output_dir=str(output_dir),
        epochs=epochs,
        smoke=smoke,
    )
    return train_baseline(config)


def _load_completed_report(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    architecture: str,
    seed: int,
    learning_rate: float,
    dropout: float,
    smoke: bool,
) -> dict[str, Any] | None:
    """Reuse a finished run; ``latest.pt`` remains the path for partial runs."""
    if not args.resume_existing:
        return None
    report_path = Path(output_dir) / "report.json"
    if not report_path.is_file():
        return None
    data_dir = Path(args.data_dir)
    inputs = [
        data_dir / "dataset_manifest.json",
        data_dir / "train.jsonl",
        data_dir / "validation.jsonl",
    ]
    if any(path.is_file() and path.stat().st_mtime > report_path.stat().st_mtime for path in inputs):
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        hyperparameters = report["hyperparameters"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        report.get("architecture") != architecture
        or int(report.get("seed", -1)) != seed
        or bool(report.get("smoke", False)) != smoke
        or float(hyperparameters.get("learning_rate", -1.0)) != learning_rate
        or float(hyperparameters.get("dropout", -1.0)) != dropout
        or bool(hyperparameters.get("copy_mechanism", False))
        != bool(getattr(args, "copy_mechanism", False))
        or bool(hyperparameters.get("structured_heads", False))
        != bool(getattr(args, "structured_heads", False))
        or bool(hyperparameters.get("parameter_heads", False))
        != bool(getattr(args, "parameter_heads", False))
        or not Path(str(report.get("checkpoint", ""))).is_file()
    ):
        return None
    return report


def _config(
    args: argparse.Namespace,
    architecture: str,
    seed: int,
    learning_rate: float,
    dropout: float,
    *,
    output_dir: str,
    epochs: int,
    smoke: bool = False,
) -> TrainingConfig:
    run_dir = Path(output_dir)
    latest = run_dir / "latest.pt"
    resume = str(latest) if args.resume_existing and latest.is_file() else None
    return TrainingConfig(
        architecture=architecture,
        data_dir=args.data_dir,
        output_dir=output_dir,
        seed=seed,
        device=args.device,
        epochs=epochs,
        batch_size=min(args.batch_size, 4) if smoke else args.batch_size,
        learning_rate=learning_rate,
        d_model=min(args.d_model, 32) if smoke else args.d_model,
        encoder_layers=1 if smoke else args.encoder_layers,
        decoder_layers=1 if smoke else args.decoder_layers,
        attention_heads=args.attention_heads,
        feedforward_dim=min(args.feedforward_dim, 64) if smoke else args.feedforward_dim,
        dropout=dropout,
        patience=args.patience,
        use_amp=not args.no_amp,
        copy_mechanism=bool(getattr(args, "copy_mechanism", False)),
        structured_heads=bool(getattr(args, "structured_heads", False)),
        parameter_heads=bool(getattr(args, "parameter_heads", False)),
        resume=resume,
        smoke=smoke,
    )


def _select_hyperparameters(
    reports: list[dict[str, Any]],
    architectures: Iterable[str],
) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    for architecture in architectures:
        candidates = [report for report in reports if report["architecture"] == architecture]
        if not candidates:
            raise ValueError(f"no pilot reports for {architecture}")
        best = min(candidates, key=_validation_rank)
        selected[architecture] = {
            "learning_rate": float(best["hyperparameters"]["learning_rate"]),
            "dropout": float(best["hyperparameters"]["dropout"]),
        }
    return selected


def _validation_rank(report: dict[str, Any]) -> tuple[bool, bool, float, float, float]:
    generation = _preferred_generation(report["validation"])
    teacher = report["validation"]["teacher_forced"]
    return (
        generation["false_execution_rate"] > 0.001,
        generation["schema_valid_rate"] < 1.0,
        -generation["exact_jal_accuracy"],
        generation["false_execution_rate"],
        teacher["token_nll"],
    )


def _leaderboard(
    reports: list[dict[str, Any]],
    architectures: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for architecture in architectures:
        matching = [report for report in reports if report["architecture"] == architecture]
        if not matching:
            continue
        exact = [_preferred_generation(report["validation"])["exact_jal_accuracy"] for report in matching]
        valid = [_preferred_generation(report["validation"])["schema_valid_rate"] for report in matching]
        false_execution = [
            _preferred_generation(report["validation"])["false_execution_rate"]
            for report in matching
        ]
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
            row["validation_false_execution_mean"] > 0.001,
            row["validation_schema_valid_mean"] < 1.0,
            -row["validation_exact_jal_mean"],
            row["validation_false_execution_mean"],
            row["parameters"],
        ),
    )


def _test_summary(test_runs: list[dict[str, Any]]) -> dict[str, float]:
    generations = [_preferred_generation(run["metrics"]) for run in test_runs]
    return {
        "exact_jal_mean": statistics.fmean(
            result["exact_jal_accuracy"] for result in generations
        ),
        "exact_jal_std": statistics.pstdev(
            result["exact_jal_accuracy"] for result in generations
        ),
        "schema_valid_mean": statistics.fmean(
            result["schema_valid_rate"] for result in generations
        ),
        "false_execution_mean": statistics.fmean(
            result["false_execution_rate"] for result in generations
        ),
    }


def _preferred_generation(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics.get("constrained_generation", metrics["generation"])


def _configuration_name(
    architecture: str,
    learning_rate: float,
    dropout: float,
    seed: int | None = None,
) -> str:
    suffix = f"_seed{seed}" if seed is not None else ""
    return (
        f"{architecture}_lr{learning_rate:.0e}_drop{dropout:.2f}{suffix}"
        .replace("+", "")
        .replace(".", "p")
    )


def _validate_args(args: argparse.Namespace) -> None:
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if args.epochs < 1 or args.pilot_epochs < 1:
        raise ValueError("epoch counts must be positive")
    if any(value <= 0 for value in args.learning_rates):
        raise ValueError("learning rates must be positive")
    if any(not 0 <= value < 1 for value in args.dropouts):
        raise ValueError("dropouts must be in [0, 1)")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    main()
