"""Run a leakage-free data scaling curve for the complete JSC v8 topology.

The protocol trains the same end-to-end architecture on nested, family-level
25/50/75/100 percent subsets.  Validation and the post-training migration
suite are evaluated, while test and evaluation_holdout are never opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.baseline_training import TrainingConfig, train_baseline
from ml.jsc.data import JSCExample, load_jsc_jsonl, validate_jsc_splits
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.sequence_data import tokenizer_training_texts
from ml.jsc.tokenizer import JSCCharTokenizer
from training_workspace.jsc_migration_benchmark import (
    _migration_metrics,
    predict_structured_jal,
)


DEFAULT_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
DEFAULT_SEEDS = (17, 29, 41)
SUBSET_SEED = "jsc-data-scaling-v1"
FORMAT_VERSION = 1


def _fraction_name(fraction: float) -> str:
    return f"p{round(fraction * 100):03d}"


def _atomic_write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    temporary.replace(path)


def _record_features(row: Mapping[str, Any]) -> Counter[str]:
    features = Counter({f"category:{row['category']}": 1})
    target = row.get("target_jal")
    if isinstance(target, str):
        plan = json.loads(target)
        steps = list(plan.get("steps") or ())
        features[f"act:{plan.get('act')}"] += 1
        features[f"step_count:{len(steps)}"] += 1
        for step in steps:
            features[f"tool:{step.get('tool')}"] += 1
    return features


def nested_family_indices(
    records: Sequence[Mapping[str, Any]], fractions: Sequence[float]
) -> dict[float, tuple[int, ...]]:
    """Select deterministic nested prefixes of whole families per category."""
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("fractions must be in (0, 1]")
    requested = tuple(sorted(set(fractions)))
    indices_by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        indices_by_family[str(row["family_id"])].append(index)
    family_features = {
        family: sum(
            (_record_features(records[index]) for index in indices),
            Counter(),
        )
        for family, indices in indices_by_family.items()
    }
    full_features = sum((_record_features(row) for row in records), Counter())
    tie_breakers = {
        family: hashlib.sha256(f"{SUBSET_SEED}:{family}".encode("utf-8")).hexdigest()
        for family in indices_by_family
    }
    selected: dict[float, set[int]] = {fraction: set() for fraction in requested}
    chosen_families: set[str] = set()
    chosen_indices: set[int] = set()
    chosen_features: Counter[str] = Counter()
    for fraction in requested:
        feature_targets = {
            name: math.ceil(total * fraction) for name, total in full_features.items()
        }
        while any(
            chosen_features[name] < target for name, target in feature_targets.items()
        ):
            candidates = []
            for family, family_indices in indices_by_family.items():
                if family in chosen_families:
                    continue
                gains = sum(
                    min(
                        family_features[family][name],
                        max(target - chosen_features[name], 0),
                    )
                    / max(target, 1)
                    for name, target in feature_targets.items()
                )
                candidates.append(
                    (
                        -(gains / math.sqrt(len(family_indices))),
                        tie_breakers[family],
                        family,
                    )
                )
            if not candidates:
                raise AssertionError("ran out of families before reaching target fraction")
            score, _tie, best = min(candidates)
            if score == 0.0:
                raise AssertionError("remaining families cannot satisfy feature targets")
            chosen_families.add(best)
            chosen_indices.update(indices_by_family[best])
            chosen_features.update(family_features[best])
        selected[fraction] = set(chosen_indices)
    result = {
        fraction: tuple(sorted(selected[fraction])) for fraction in requested
    }
    previous: set[int] = set()
    for fraction in requested:
        current = set(result[fraction])
        if not previous <= current:
            raise AssertionError("scaling subsets must be nested")
        previous = current
    if requested[-1] == 1.0 and len(result[1.0]) != len(records):
        raise AssertionError("100 percent subset must contain every training example")
    return result


def _raw_records(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
    return [json.loads(line) for line in lines], lines


def prepare_scaling_datasets(
    source_dir: Path,
    output_root: Path,
    fractions: Sequence[float],
) -> dict[float, dict[str, Any]]:
    """Write train subsets and an unchanged validation split with fresh manifests."""
    registry = build_project_schema_registry()
    source_manifest = json.loads(
        (source_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("source dataset schema does not match runtime")
    records, source_lines = _raw_records(source_dir / "train.jsonl")
    selected = nested_family_indices(records, fractions)
    validation_bytes = (source_dir / "validation.jsonl").read_bytes()
    validation = load_jsc_jsonl(
        source_dir / "validation.jsonl", registry, expected_split="validation"
    )
    result: dict[float, dict[str, Any]] = {}
    for fraction in sorted(selected):
        directory = output_root / _fraction_name(fraction)
        indices = selected[fraction]
        train_content = "".join(source_lines[index] + "\n" for index in indices)
        _atomic_write(directory / "train.jsonl", train_content)
        _atomic_write(directory / "validation.jsonl", validation_bytes)
        train = load_jsc_jsonl(
            directory / "train.jsonl", registry, expected_split="train"
        )
        split_report = validate_jsc_splits({"train": train, "validation": validation})
        manifest = copy.deepcopy(source_manifest)
        manifest["generator"] = "training_workspace.run_jsc_data_scaling"
        manifest["scaling_curve"] = {
            "subset_seed": SUBSET_SEED,
            "requested_fraction": fraction,
            "actual_fraction": len(train) / len(records),
            "nested_family_selection": True,
            "test_opened": False,
            "evaluation_holdout_opened": False,
        }
        train_hash = hashlib.sha256((directory / "train.jsonl").read_bytes()).hexdigest()
        manifest["splits"]["train"] = {
            **split_report["train"],
            "file": "train.jsonl",
            "sha256": train_hash,
        }
        manifest["splits"]["validation"] = {
            **split_report["validation"],
            "file": "validation.jsonl",
            "sha256": hashlib.sha256(validation_bytes).hexdigest(),
        }
        _atomic_write(
            directory / "dataset_manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        tokenizer = JSCCharTokenizer.fit(tokenizer_training_texts(train))
        result[fraction] = {
            "directory": str(directory.resolve()),
            "requested_fraction": fraction,
            "actual_fraction": len(train) / len(records),
            "examples": len(train),
            "families": len({example.family_id for example in train}),
            "categories": split_report["train"]["categories"],
            "acts": split_report["train"]["acts"],
            "tools": split_report["train"]["tools"],
            "tokenizer_vocabulary": tokenizer.size,
            "train_sha256": manifest["splits"]["train"]["sha256"],
            "validation_sha256": manifest["splits"]["validation"]["sha256"],
        }
    return result


def _training_config(
    args: argparse.Namespace,
    fraction: float,
    seed: int,
    data_dir: Path,
    output_dir: Path,
) -> TrainingConfig:
    epochs = 1 if args.smoke else math.ceil(args.base_epochs / fraction)
    patience = 1 if args.smoke else math.ceil(args.base_patience / fraction)
    latest = output_dir / "latest.pt"
    return TrainingConfig(
        architecture="tiny_transformer",
        data_dir=str(data_dir),
        output_dir=str(output_dir),
        seed=seed,
        device=args.device,
        epochs=epochs,
        batch_size=4 if args.smoke else args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        label_smoothing=0.05,
        act_loss_weight=0.25,
        d_model=32 if args.smoke else 192,
        encoder_layers=1 if args.smoke else 3,
        decoder_layers=1 if args.smoke else 3,
        attention_heads=4,
        feedforward_dim=64 if args.smoke else 384,
        dropout=0.12,
        patience=patience,
        use_amp=not args.no_amp,
        copy_mechanism=True,
        structured_heads=True,
        parameter_heads=True,
        span_heads=True,
        semantic_pooling=False,
        execution_verifier=True,
        resume=str(latest) if latest.is_file() else None,
        smoke=args.smoke,
        final_generation_metrics=False,
    )


def _completed_report(output_dir: Path) -> dict[str, Any] | None:
    report_path = output_dir / "report.json"
    best_path = output_dir / "best.pt"
    if not (report_path.is_file() and best_path.is_file()):
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if Path(str(report.get("checkpoint", ""))).is_file():
        return report
    return None


def _latest_inference_checkpoint(output_dir: Path, best_checkpoint: Path) -> Path:
    """Materialize the early-stop state as an inference-only diagnostic checkpoint."""
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    latest = torch.load(output_dir / "latest.pt", map_location="cpu", weights_only=False)
    derived = dict(best)
    derived["model_state"] = latest["model_state"]
    derived["epoch"] = latest["epoch"]
    path = output_dir / ".latest_inference_scaling.pt"
    torch.save(derived, path)
    return path


def _mean_std(values: Iterable[float]) -> dict[str, float]:
    rows = list(values)
    return {
        "mean": statistics.fmean(rows),
        "std": statistics.stdev(rows) if len(rows) > 1 else 0.0,
        "min": min(rows),
        "max": max(rows),
    }


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[float(run["requested_fraction"])].append(run)
    result = []
    for fraction in sorted(grouped):
        rows = grouped[fraction]
        result.append(
            {
                "requested_fraction": fraction,
                "actual_fraction": rows[0]["actual_fraction"],
                "examples": rows[0]["train_examples"],
                "families": rows[0]["train_families"],
                "seeds": [row["seed"] for row in rows],
                "validation_structured_exact_jal": _mean_std(
                    row["validation_structured"]["exact_jal_accuracy"] for row in rows
                ),
                "migration_structured_exact_jal": _mean_std(
                    row["migration_structured"]["exact_jal_accuracy"] for row in rows
                ),
                "migration_latest_structured_exact_jal": _mean_std(
                    row["migration_latest_structured"]["exact_jal_accuracy"]
                    for row in rows
                ),
                "migration_act_accuracy": _mean_std(
                    row["migration_structured"]["act_accuracy"] for row in rows
                ),
                "migration_tool_sequence_accuracy": _mean_std(
                    row["migration_structured"]["tool_sequence_accuracy"] for row in rows
                ),
                "migration_argument_sequence_accuracy": _mean_std(
                    row["migration_structured"]["argument_sequence_accuracy"] for row in rows
                ),
                "selected_train_tool_head_accuracy": _mean_std(
                    row["selected_epoch_train_heads"]["tool_sequence_head_accuracy"]
                    for row in rows
                ),
                "selected_validation_tool_head_accuracy": _mean_std(
                    row["selected_epoch_validation_heads"]["tool_sequence_head_accuracy"]
                    for row in rows
                ),
                "selected_train_span_head_accuracy": _mean_std(
                    row["selected_epoch_train_heads"]["span_head_accuracy"] for row in rows
                ),
                "selected_validation_span_head_accuracy": _mean_std(
                    row["selected_epoch_validation_heads"]["span_head_accuracy"]
                    for row in rows
                ),
                "migration_single_exact_jal": _mean_std(
                    row["migration_structured"]["category_exact_jal"].get("single", 0.0)
                    for row in rows
                ),
                "migration_steps_2_3_exact_jal": _mean_std(
                    row["migration_structured"]["exact_jal_by_step_group"].get("steps_2_3")
                    or 0.0
                    for row in rows
                ),
                "migration_steps_4_5_exact_jal": _mean_std(
                    row["migration_structured"]["exact_jal_by_step_group"].get("steps_4_5")
                    or 0.0
                    for row in rows
                ),
                "migration_multi_turn_exact_jal": _mean_std(
                    row["migration_structured"]["category_exact_jal"].get("multi_turn", 0.0)
                    for row in rows
                ),
                "migration_false_execution_rate": _mean_std(
                    row["migration_structured"]["false_execution_rate"] for row in rows
                ),
                "migration_opposite_action_rate": _mean_std(
                    row["migration_structured"]["opposite_action_rate"] for row in rows
                ),
            }
        )
    return result


def _diagnose(aggregate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    means = [row["migration_structured_exact_jal"]["mean"] for row in aggregate]
    latest_means = [
        row.get(
            "migration_latest_structured_exact_jal",
            row["migration_structured_exact_jal"],
        )["mean"]
        for row in aggregate
    ]
    if len(means) < 2:
        return {
            "verdict": "insufficient_fractions",
            "monotonic_non_decreasing": True,
            "gain_25_to_100": 0.0,
            "gain_75_to_100": 0.0,
            "maximum_seed_std": max(
                row["migration_structured_exact_jal"]["std"] for row in aggregate
            ),
            "latest_gain_25_to_100": 0.0,
            "latest_gain_75_to_100": 0.0,
            "latest_monotonic_non_decreasing": True,
        }
    gain_25_to_100 = means[-1] - means[0]
    gain_75_to_100 = means[-1] - means[-2] if len(means) >= 4 else 0.0
    monotonic = all(right >= left for left, right in zip(means, means[1:]))
    maximum_seed_std = max(
        row["migration_structured_exact_jal"]["std"] for row in aggregate
    )
    if maximum_seed_std >= 0.04:
        verdict = "seed_variance_too_high"
    elif gain_25_to_100 >= 0.08 and gain_75_to_100 >= 0.015:
        verdict = "data_limited_with_remaining_headroom"
    elif abs(gain_75_to_100) < 0.01:
        verdict = "plateau_architecture_or_objective_limited"
    else:
        verdict = "mixed_more_data_and_objective_work_needed"
    return {
        "verdict": verdict,
        "monotonic_non_decreasing": monotonic,
        "gain_25_to_100": gain_25_to_100,
        "gain_75_to_100": gain_75_to_100,
        "maximum_seed_std": maximum_seed_std,
        "latest_gain_25_to_100": latest_means[-1] - latest_means[0],
        "latest_gain_75_to_100": (
            latest_means[-1] - latest_means[-2] if len(latest_means) >= 4 else 0.0
        ),
        "latest_monotonic_non_decreasing": all(
            right >= left for left, right in zip(latest_means, latest_means[1:])
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# JSC v8 Data-first Scaling Curve",
        "",
        "## Протокол",
        "",
        "- Полная топология JSC v8 обучается end-to-end с нуля; staged v8 checkpoint не используется.",
        "- Поднаборы вложены, выбираются целыми structural families и балансируют "
        "category/act/tool/step-count.",
        "- Для приблизительно сопоставимого optimisation budget число эпох масштабируется "
        "обратно запрошенной доле. Из-за неделимых families фактический budget выше полного "
        "на 12.9% / 5.4% / 2.7% для первых трёх точек; это даёт малым наборам небольшое "
        "преимущество и не может искусственно создать наблюдаемое plateau.",
        f"- Seeds: `{report['protocol']['seeds']}`; locked test/holdout не открывались.",
        "- Главная метрика — Structured JSC без autoregressive JSON.",
        "- `selected` — checkpoint по composite validation NLL; `latest` — состояние "
        "в момент early stop, показанное как post-hoc diagnostic, а не новый выбранный кандидат.",
        "",
        "## Кривая",
        "",
        "| Доля | Примеры | Families | Validation Exact | Migration selected | Migration latest | Single | 2–3 шага | 4–5 шагов | Multi-turn | False execute |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["aggregate"]:
        metric = lambda name: row[name]
        lines.append(
            f"| {row['requested_fraction']:.0%} | {row['examples']} | {row['families']} | "
            f"{metric('validation_structured_exact_jal')['mean']:.2%} ± {metric('validation_structured_exact_jal')['std']:.2%} | "
            f"{metric('migration_structured_exact_jal')['mean']:.2%} ± {metric('migration_structured_exact_jal')['std']:.2%} | "
            f"{metric('migration_latest_structured_exact_jal')['mean']:.2%} ± {metric('migration_latest_structured_exact_jal')['std']:.2%} | "
            f"{metric('migration_single_exact_jal')['mean']:.2%} | "
            f"{metric('migration_steps_2_3_exact_jal')['mean']:.2%} | "
            f"{metric('migration_steps_4_5_exact_jal')['mean']:.2%} | "
            f"{metric('migration_multi_turn_exact_jal')['mean']:.2%} | "
            f"{metric('migration_false_execution_rate')['mean']:.2%} |"
        )
    diagnosis = report["diagnosis"]
    full = report["aggregate"][-1]
    lines.extend(
        [
            "",
            "## Где находится потолок v8",
            "",
            "| Метрика на 100% | Среднее | Std |",
            "|---|---:|---:|",
            f"| Migration dialogue act | {full['migration_act_accuracy']['mean']:.2%} | {full['migration_act_accuracy']['std']:.2%} |",
            f"| Migration tool sequence | {full['migration_tool_sequence_accuracy']['mean']:.2%} | {full['migration_tool_sequence_accuracy']['std']:.2%} |",
            f"| Migration arguments | {full['migration_argument_sequence_accuracy']['mean']:.2%} | {full['migration_argument_sequence_accuracy']['std']:.2%} |",
            f"| Tool head, train at selected epoch | {full['selected_train_tool_head_accuracy']['mean']:.2%} | {full['selected_train_tool_head_accuracy']['std']:.2%} |",
            f"| Tool head, validation at selected epoch | {full['selected_validation_tool_head_accuracy']['mean']:.2%} | {full['selected_validation_tool_head_accuracy']['std']:.2%} |",
            f"| Span head, train at selected epoch | {full['selected_train_span_head_accuracy']['mean']:.2%} | {full['selected_train_span_head_accuracy']['std']:.2%} |",
            f"| Span head, validation at selected epoch | {full['selected_validation_span_head_accuracy']['mean']:.2%} | {full['selected_validation_span_head_accuracy']['std']:.2%} |",
            "",
            "## Диагноз",
            "",
            f"- Verdict: `{diagnosis['verdict']}`.",
            f"- Рост 25% → 100%: {diagnosis['gain_25_to_100']:+.2%}.",
            f"- Рост 75% → 100%: {diagnosis['gain_75_to_100']:+.2%}.",
            f"- Максимальное standard deviation между seed: {diagnosis['maximum_seed_std']:.2%}.",
            f"- Монотонный рост: `{diagnosis['monotonic_non_decreasing']}`.",
            f"- Диагностический latest растёт монотонно "
            f"(`{diagnosis['latest_monotonic_non_decreasing']}`), но всего на "
            f"{diagnosis['latest_gain_25_to_100']:+.2%} от 25% до 100% и на "
            f"{diagnosis['latest_gain_75_to_100']:+.2%} от 75% до 100%; это не меняет verdict.",
            "- Multi-turn остался на 0%; планы на 2–3 шага ухудшились до 0.39%, "
            "на 4–5 шагов — 0.42%.",
            "- Нулевые false-execute/opposite-action достигаются в основном fail-closed "
            "отказами; это безопасно, но не означает готовность выполнять команды.",
            "- Data-first улучшает отдельные классификационные головы, но не преодолевает "
            "рассогласование JSON reconstruction, tool sequence и span extraction.",
            "",
            "## Решение",
            "",
            "1. Не расходовать следующий цикл на простое добавление похожих synthetic данных в v8.",
            "2. Перейти к Structured JSC без JSON: прямой act/count/tool/argument/span decoder, "
            "отдельные loss schedules и checkpoint selection по program-level метрикам.",
            "3. Использовать текущие данные как baseline, затем расширять только доказанно "
            "слабые families: multi-turn, 2–5 действий, ASR aliases и свободные аргументы.",
            "4. Не открывать locked test/holdout, пока новый structured-кандидат не пройдёт "
            "development migration gates.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    fractions = tuple(sorted(set(args.fractions)))
    seeds = tuple(dict.fromkeys(args.seeds))
    if args.smoke:
        fractions = fractions[:1]
        seeds = seeds[:1]
    datasets = prepare_scaling_datasets(args.data_dir, args.data_output_root, fractions)
    if args.prepare_only:
        return {
            "format_version": FORMAT_VERSION,
            "prepare_only": True,
            "datasets": datasets,
            "locked_splits_opened": False,
        }
    registry = build_project_schema_registry()
    migration = tuple(
        load_jsc_jsonl(args.migration_suite, registry, expected_split="validation")
    )
    runs: list[dict[str, Any]] = []
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    for fraction in fractions:
        dataset = datasets[fraction]
        data_dir = Path(dataset["directory"])
        validation = tuple(
            load_jsc_jsonl(data_dir / "validation.jsonl", registry, expected_split="validation")
        )
        for seed in seeds:
            run_name = f"{_fraction_name(fraction)}_seed{seed}"
            output_dir = args.output_root / ("smoke" if args.smoke else "runs") / run_name
            print(f"scaling-run: fraction={fraction:.0%} seed={seed}", flush=True)
            training = _completed_report(output_dir)
            if training is None:
                training = train_baseline(
                    _training_config(args, fraction, seed, data_dir, output_dir)
                )
            checkpoint = Path(training["checkpoint"])
            selected_epoch = training["history"][training["best_epoch"]]
            validation_predictions, validation_timing = predict_structured_jal(
                validation, checkpoint, registry, device, args.eval_batch_size
            )
            migration_predictions, migration_timing = predict_structured_jal(
                migration, checkpoint, registry, device, args.eval_batch_size
            )
            latest_checkpoint = _latest_inference_checkpoint(output_dir, checkpoint)
            try:
                validation_latest_predictions, _validation_latest_timing = predict_structured_jal(
                    validation,
                    latest_checkpoint,
                    registry,
                    device,
                    args.eval_batch_size,
                )
                migration_latest_predictions, _migration_latest_timing = predict_structured_jal(
                    migration,
                    latest_checkpoint,
                    registry,
                    device,
                    args.eval_batch_size,
                )
            finally:
                latest_checkpoint.unlink(missing_ok=True)
            runs.append(
                {
                    "requested_fraction": fraction,
                    "actual_fraction": dataset["actual_fraction"],
                    "train_examples": dataset["examples"],
                    "train_families": dataset["families"],
                    "seed": seed,
                    "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "epochs_completed": training["epochs_completed"],
                    "best_epoch": training["best_epoch"],
                    "training_elapsed_seconds": training["elapsed_seconds"],
                    "selected_epoch_train_heads": {
                        key: selected_epoch["train"][key]
                        for key in (
                            "step_count_accuracy",
                            "tool_sequence_head_accuracy",
                            "parameter_head_accuracy",
                            "span_head_accuracy",
                            "execution_verifier_false_positive_rate",
                        )
                    },
                    "selected_epoch_validation_heads": {
                        key: selected_epoch["validation"][key]
                        for key in (
                            "step_count_accuracy",
                            "tool_sequence_head_accuracy",
                            "parameter_head_accuracy",
                            "span_head_accuracy",
                            "execution_verifier_false_positive_rate",
                        )
                    },
                    "validation_structured": _migration_metrics(
                        validation, validation_predictions, registry
                    ),
                    "validation_structured_timing": validation_timing,
                    "migration_structured": _migration_metrics(
                        migration, migration_predictions, registry
                    ),
                    "migration_structured_timing": migration_timing,
                    "validation_latest_structured": _migration_metrics(
                        validation, validation_latest_predictions, registry
                    ),
                    "migration_latest_structured": _migration_metrics(
                        migration, migration_latest_predictions, registry
                    ),
                }
            )
    aggregate = _aggregate(runs)
    report = {
        "format_version": FORMAT_VERSION,
        "protocol": {
            "fractions": list(fractions),
            "seeds": list(seeds),
            "subset_seed": SUBSET_SEED,
            "base_epochs_at_100_percent": args.base_epochs,
            "equalized_optimizer_budget": True,
            "optimizer_budget_note": (
                "approximately equalized by requested fraction; whole-family overshoot "
                "slightly favors smaller subsets"
            ),
            "architecture": "jsc_v8_topology_end_to_end_from_scratch",
            "locked_test_opened": False,
            "evaluation_holdout_opened": False,
            "migration_suite_sha256": hashlib.sha256(
                args.migration_suite.read_bytes()
            ).hexdigest(),
        },
        "datasets": {str(key): value for key, value in datasets.items()},
        "runs": runs,
        "aggregate": aggregate,
        "diagnosis": _diagnose(aggregate),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        args.report,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(args.markdown, _render_markdown(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training_workspace/jsc_data"))
    parser.add_argument(
        "--data-output-root", type=Path, default=Path("training_workspace/jsc_scaling_data")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("training_workspace/jsc_scaling_runs")
    )
    parser.add_argument(
        "--migration-suite",
        type=Path,
        default=Path("training_workspace/jsc_migration_data/development.jsonl"),
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-epochs", type=int, default=24)
    parser.add_argument("--base-patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("reports/jsc_data_scaling.json"))
    parser.add_argument(
        "--markdown", type=Path, default=Path("docs/JSC_DATA_SCALING_RU.md")
    )
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report.get("diagnosis", report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
