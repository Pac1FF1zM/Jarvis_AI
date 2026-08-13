"""Train and benchmark the direct Structured JSC architecture without JSON.

Checkpoint epochs and confidence thresholds are selected only on validation.
The separate migration development suite is opened afterwards for the final
comparison.  Locked test and evaluation_holdout splits are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.data import JSCExample, load_jsc_jsonl
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.structured_training import (
    StructuredTrainingConfig,
    cache_structured_checkpoint_logits,
    decode_structured_cache,
    train_structured,
)
from training_workspace.jsc_migration_benchmark import (
    _failure_analysis,
    _gate_report,
    _migration_metrics,
)


DEFAULT_SEEDS = (17, 29, 41)
FORMAT_VERSION = 1
THRESHOLD_GRID = {
    "execution_threshold": (0.45, 0.50, 0.55, 0.60, 0.65),
    "verifier_threshold": (0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    "parameter_threshold": (0.35, 0.45, 0.50, 0.60),
    "span_threshold": (0.20, 0.25, 0.30, 0.35, 0.40),
    "missing_threshold": (0.35, 0.40, 0.45, 0.50, 0.60),
}


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _load_development_data(
    data_dir: Path, migration_suite: Path
) -> tuple[tuple[JSCExample, ...], tuple[JSCExample, ...], Any, Mapping[str, Any]]:
    registry = build_project_schema_registry()
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("dataset schema does not match runtime")
    validation_path = data_dir / "validation.jsonl"
    expected = manifest["splits"]["validation"]["sha256"]
    if hashlib.sha256(validation_path.read_bytes()).hexdigest() != expected:
        raise ValueError("validation hash mismatch")
    validation = tuple(
        load_jsc_jsonl(validation_path, registry, expected_split="validation")
    )
    migration = tuple(
        load_jsc_jsonl(migration_suite, registry, expected_split="validation")
    )
    return validation, migration, registry, manifest


def _training_config(args: argparse.Namespace, seed: int, output: Path) -> StructuredTrainingConfig:
    latest = output / "latest.pt"
    return StructuredTrainingConfig(
        data_dir=str(args.data_dir),
        output_dir=str(output),
        seed=seed,
        device=args.device,
        epochs=1 if args.smoke else args.epochs,
        batch_size=4 if args.smoke else args.batch_size,
        learning_rate=args.learning_rate,
        d_model=32 if args.smoke else args.d_model,
        encoder_layers=1 if args.smoke else args.encoder_layers,
        attention_heads=4,
        feedforward_dim=64 if args.smoke else args.feedforward_dim,
        patience=1 if args.smoke else args.patience,
        use_amp=not args.no_amp,
        resume=str(latest) if latest.is_file() else None,
        smoke=args.smoke,
    )


def _completed_report(output: Path, requested: StructuredTrainingConfig) -> dict[str, Any] | None:
    path = output / "report.json"
    checkpoint = output / "best.pt"
    if not (path.is_file() and checkpoint.is_file()):
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    saved = report.get("training_config", {})
    comparable = ("seed", "d_model", "encoder_layers", "feedforward_dim", "smoke")
    if any(saved.get(name) != getattr(requested, name) for name in comparable):
        raise ValueError(f"completed Structured JSC run has incompatible config: {output}")
    return report


def _rank(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["opposite_action_rate"] > 0.0),
        float(metrics["false_execution_rate"] > 0.002),
        -float(metrics["exact_jal_accuracy"]),
        -float(metrics["tool_sequence_accuracy"]),
        -float(metrics["argument_sequence_accuracy"]),
        -float(metrics["act_accuracy"]),
    )


def tune_thresholds(
    cache: Any,
    examples: Sequence[JSCExample],
    base: StructuredTrainingConfig,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    """Coordinate-search thresholds using validation program metrics only."""
    selected = {
        name: float(getattr(base, name)) for name in THRESHOLD_GRID
    }
    trace: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    # Two deterministic passes capture the small interactions between act,
    # verifier, span and missing-slot thresholds without an expensive grid.
    for pass_index in range(2):
        changed = False
        for name, candidates in THRESHOLD_GRID.items():
            rows = []
            for value in candidates:
                trial = {**selected, name: value}
                predictions, decisions = decode_structured_cache(cache, base, trial)
                metrics = _migration_metrics(examples, predictions, cache.registry)
                rows.append(((_rank(metrics), value), metrics, decisions))
            (_candidate_rank, value), metrics, decisions = min(rows, key=lambda row: row[0])
            changed = changed or selected[name] != value
            selected[name] = value
            best_metrics = metrics
            trace.append(
                {
                    "pass": pass_index + 1,
                    "field": name,
                    "selected": value,
                    "exact_jal_accuracy": metrics["exact_jal_accuracy"],
                    "tool_sequence_accuracy": metrics["tool_sequence_accuracy"],
                    "false_execution_rate": metrics["false_execution_rate"],
                    "opposite_action_rate": metrics["opposite_action_rate"],
                    "decisions": decisions,
                }
            )
        if not changed:
            break
    predictions, decisions = decode_structured_cache(cache, base, selected)
    best_metrics = _migration_metrics(examples, predictions, cache.registry)
    best_metrics["decoder_decisions"] = decisions
    return selected, best_metrics, trace


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "mean": statistics.fmean(rows),
        "std": statistics.stdev(rows) if len(rows) > 1 else 0.0,
        "min": min(rows),
        "max": max(rows),
    }


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = (
        "exact_jal_accuracy",
        "act_accuracy",
        "tool_sequence_accuracy",
        "argument_sequence_accuracy",
        "schema_valid_rate",
        "false_execution_rate",
        "opposite_action_rate",
    )
    return {
        "seeds": [row["seed"] for row in runs],
        "validation": {
            name: _mean_std(row["validation"][name] for row in runs) for name in names
        },
        "migration": {
            name: _mean_std(row["migration"][name] for row in runs) for name in names
        },
    }


def _reference_metrics() -> dict[str, Any]:
    path = Path("reports/jsc_migration_development.json")
    if not path.is_file():
        return {"available": False}
    content = path.read_bytes()
    report = json.loads(content)
    systems = report.get("systems", {})
    return {
        "available": True,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "production_nlu_exact_jal": systems.get("production_nlu", {})
        .get("metrics", {})
        .get("exact_jal_accuracy"),
        "jsc_v8_structured_exact_jal": systems.get("jsc_v8_structured_only", {})
        .get("metrics", {})
        .get("exact_jal_accuracy"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seeds = tuple(dict.fromkeys(args.seeds))
    if args.smoke:
        seeds = seeds[:1]
    validation, migration, registry, manifest = _load_development_data(
        args.data_dir, args.migration_suite
    )
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        output = args.output_root / ("smoke" if args.smoke else "runs") / f"seed{seed}"
        config = _training_config(args, seed, output)
        print(f"structured-jsc: seed={seed} output={output}", flush=True)
        training = _completed_report(output, config)
        if training is None:
            training = train_structured(config)
        checkpoint = Path(training["checkpoint"])
        validation_cache, inference_config, checkpoint_payload = (
            cache_structured_checkpoint_logits(
                checkpoint,
                validation,
                registry,
                device=args.device,
                batch_size=args.eval_batch_size,
            )
        )
        thresholds, validation_metrics, calibration = tune_thresholds(
            validation_cache, validation, inference_config
        )
        started = time.perf_counter()
        migration_cache, _config, _payload = cache_structured_checkpoint_logits(
            checkpoint,
            migration,
            registry,
            device=args.device,
            batch_size=args.eval_batch_size,
        )
        cold_inference_seconds = time.perf_counter() - started
        migration_predictions, migration_decisions = decode_structured_cache(
            migration_cache, inference_config, thresholds
        )
        migration_metrics = _migration_metrics(
            migration, migration_predictions, registry
        )
        migration_metrics["decoder_decisions"] = migration_decisions
        runs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "parameters": training["parameters"],
                "best_epoch": training["best_epoch"],
                "epochs_completed": training["epochs_completed"],
                "training_elapsed_seconds": training["elapsed_seconds"],
                "thresholds": thresholds,
                "calibration_trace": calibration,
                "validation": validation_metrics,
                "migration": migration_metrics,
                "migration_failure_analysis": _failure_analysis(
                    migration, migration_predictions, registry
                ),
                "migration_cold_load_and_inference_seconds": cold_inference_seconds,
                "checkpoint_format": checkpoint_payload.get("kind"),
            }
        )
    selected = min(runs, key=lambda row: (_rank(row["validation"]), row["seed"]))
    gates = _gate_report(selected["migration"])
    report = {
        "format_version": FORMAT_VERSION,
        "architecture": {
            "name": "structured_jsc_no_json",
            "autoregressive_decoder": False,
            "generated_json": False,
            "direct_heads": [
                "dialogue_act",
                "step_count",
                "ordered_tools",
                "categorical_arguments",
                "argument_spans",
                "missing_slots",
                "reason",
                "execution_verifier",
            ],
        },
        "protocol": {
            "seeds": list(seeds),
            "training": {
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "d_model": args.d_model,
                "encoder_layers": args.encoder_layers,
                "feedforward_dim": args.feedforward_dim,
                "amp": not args.no_amp,
                "device": args.device,
            },
            "selection": "checkpoint_epoch_and_thresholds_on_validation_only",
            "migration_suite_used_after_selection": True,
            "validation_examples": len(validation),
            "migration_examples": len(migration),
            "validation_sha256": manifest["splits"]["validation"]["sha256"],
            "migration_sha256": hashlib.sha256(args.migration_suite.read_bytes()).hexdigest(),
            "locked_test_opened": False,
            "evaluation_holdout_opened": False,
            "runtime_tools_called": False,
        },
        "runs": runs,
        "aggregate": _aggregate(runs),
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_thresholds": selected["thresholds"],
        "selected_migration_gates": gates,
        "reference": _reference_metrics(),
        "production_decision": (
            "eligible_for_shadow_wiring_review"
            if gates["passed"]
            else "keep_current_production_routing"
        ),
        "production_routing_changed": False,
    }
    _atomic_write(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(args.markdown, _render_markdown(report))
    return report


def _render_markdown(report: Mapping[str, Any]) -> str:
    selected = next(
        row for row in report["runs"] if row["seed"] == report["selected_seed"]
    )
    training = report["protocol"]["training"]
    gates = report["selected_migration_gates"]
    status = "PASS" if gates["passed"] else "FAIL"
    lines = [
        "# Structured JSC без JSON — benchmark",
        "",
        "## Итог",
        "",
        f"**Migration gates: {status}.** Production routing не переключался. "
        f"Выбран seed `{report['selected_seed']}` только по validation; его migration "
        f"Exact JAL — {selected['migration']['exact_jal_accuracy']:.2%}.",
        "",
        "Модель не содержит autoregressive decoder/token head и не генерирует JSON: "
        "она напрямую предсказывает act, число и порядок шагов, tools, аргументы, "
        "missing slots, reason и независимый execution verifier. JAL собирается "
        "типизированным schema validator в fail-closed режиме.",
        "",
        "## Протокол",
        "",
        f"- Seeds: `{report['protocol']['seeds']}`; validation: "
        f"{report['protocol']['validation_examples']}; migration development: "
        f"{report['protocol']['migration_examples']}.",
        f"- Topology: d_model={training['d_model']}, encoder_layers="
        f"{training['encoder_layers']}, FFN={training['feedforward_dim']}; batch="
        f"{training['batch_size']}; lr={training['learning_rate']}.",
        "- Epoch checkpoint и confidence thresholds выбирались только на validation.",
        "- Migration suite оценивался после выбора; приложения/tools не запускались.",
        "- Locked `test` и `evaluation_holdout` не открывались.",
        "",
        "## Результаты по seed",
        "",
        "| Seed | Params | Epoch | Validation Exact | Migration Exact | Tool seq | "
        "Arguments | False execute | Opposite |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        metric = row["migration"]
        lines.append(
            f"| {row['seed']} | {row['parameters']:,} | {row['best_epoch'] + 1} | "
            f"{row['validation']['exact_jal_accuracy']:.2%} | "
            f"{metric['exact_jal_accuracy']:.2%} | {metric['tool_sequence_accuracy']:.2%} | "
            f"{metric['argument_sequence_accuracy']:.2%} | "
            f"{metric['false_execution_rate']:.2%} | {metric['opposite_action_rate']:.2%} |"
        )
    aggregate = report["aggregate"]["migration"]
    failures = selected["migration_failure_analysis"]["counts"]
    lines.extend(
        [
            "",
            f"Среднее Migration Exact JAL: **{aggregate['exact_jal_accuracy']['mean']:.2%} "
            f"± {aggregate['exact_jal_accuracy']['std']:.2%}**.",
            "",
            "## Главные ошибки выбранного seed",
            "",
            f"- Wrong dialogue act: {failures.get('wrong_dialogue_act', 0)}.",
            f"- Wrong arguments: {failures.get('wrong_arguments', 0)}.",
            f"- Wrong tool sequence: {failures.get('wrong_tool_sequence', 0)}.",
            f"- Validation-selected thresholds: `{report['selected_thresholds']}`.",
            "",
            "## Обязательные gates выбранного кандидата",
            "",
            "| Gate | Actual | Target | Status |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name, row in gates["checks"].items():
        lines.append(
            f"| {name} | {row['actual']:.2%} | {row['target']:.2%} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    reference = report["reference"]
    if reference.get("available"):
        lines.extend(
            [
                "",
                "## Сравнение с предыдущим этапом",
                "",
                "- Production NLU reference: "
                f"{reference['production_nlu_exact_jal']:.2%} Exact JAL.",
                "- JSC v8 structured-head reference: "
                f"{reference['jsc_v8_structured_exact_jal']:.2%}.",
                "- Новый Structured JSC, выбранный seed: "
                f"{selected['migration']['exact_jal_accuracy']:.2%}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Решение",
            "",
            f"`{report['production_decision']}`. До прохождения всех gates новый "
            "checkpoint остаётся экспериментальным; production wiring не изменён.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training_workspace/jsc_data"))
    parser.add_argument(
        "--migration-suite",
        type=Path,
        default=Path("training_workspace/jsc_migration_data/development.jsonl"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("training_workspace/jsc_structured_runs_v3")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=192)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("reports/jsc_structured.json"))
    parser.add_argument(
        "--markdown", type=Path, default=Path("docs/JSC_STRUCTURED_BENCHMARK_RU.md")
    )
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "selected_seed": report["selected_seed"],
                "gates": report["selected_migration_gates"],
                "decision": report["production_decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
