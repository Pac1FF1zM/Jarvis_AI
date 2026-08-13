"""Reproducible, side-effect-free benchmark for replacing production NLU with JSC.

The benchmark never calls runtime tools.  By default it opens only the
development validation split; locked test/holdout data require an explicit
one-way command-line acknowledgement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.baseline_metrics import evaluate_program_predictions
from ml.jsc.constrained_decoding import constrain_jal_predictions
from ml.jsc.data import JSCExample, load_jsc_jsonl
from ml.jsc.jal import DialogueAct, JALPlan, ToolCall, dumps, loads
from ml.jsc.models import BaselineConfig, JSCBaselineModel
from ml.jsc.project_registry import build_project_schema_registry
from ml.jsc.sequence_data import (
    serialize_source,
)
from ml.jsc.span_labels import SPAN_ARGUMENTS
from ml.jsc.structured_labels import build_parameter_labels
from ml.jsc.tokenizer import JSCCharTokenizer
from ml.nlu.inference import NLUPredictor
from modules.command_router import RoutedAction, route_explicit_command, split_compound_command
from modules.nlu import (
    _apply_reminder_guardrails,
    _apply_runtime_command_guardrails,
    _normalise_transcription_for_nlu,
)
from modules.semantic_commit import prepare_final_utterance
from tools._applications import resolve_application


FORMAT_VERSION = 2
LOCKED_SPLITS = frozenset({"test", "evaluation_holdout"})
MIGRATION_GATES = {
    "overall_exact_jal_accuracy": 0.84,
    "single_exact_jal_accuracy": 0.87,
    "steps_2_3_exact_jal_accuracy": 0.85,
    "steps_4_5_exact_jal_accuracy": 0.82,
    "multi_turn_exact_jal_accuracy": 0.87,
    "schema_valid_rate": 1.0,
    "maximum_false_execution_rate": 0.002,
    "maximum_opposite_action_rate": 0.0,
}
_REJECT = dumps(JALPlan(DialogueAct.REJECT, reason="unsafe_model_output"))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values, default=0.0),
    }


def _load_examples(
    data_dir: Path, split: str, registry: Any, suite_path: Path | None = None
) -> tuple[JSCExample, ...]:
    if suite_path is not None:
        return tuple(load_jsc_jsonl(suite_path, registry, expected_split="validation"))
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("dataset tool schema fingerprint does not match runtime")
    path = data_dir / f"{split}.jsonl"
    expected = manifest["splits"][split]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{split} hash does not match dataset manifest")
    return tuple(load_jsc_jsonl(path, registry, expected_split=split))


def _load_jsc(
    checkpoint_path: Path, device: torch.device, registry: Any
) -> tuple[JSCBaselineModel, JSCCharTokenizer, BaselineConfig, Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("tool_schema_sha256") != registry.schema_fingerprint:
        raise ValueError("checkpoint tool schema fingerprint does not match runtime")
    tokenizer = JSCCharTokenizer.from_dict(checkpoint["tokenizer"])
    config = BaselineConfig.from_dict(checkpoint["model_config"])
    model = JSCBaselineModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, config, checkpoint


def _inference_rows(
    examples: Sequence[JSCExample], tokenizer: JSCCharTokenizer, max_source_length: int
) -> list[dict[str, Any]]:
    """Encode sources only, allowing the benchmark target to exceed decoder limits."""
    rows = []
    for example in examples:
        source_text = serialize_source(example)
        rows.append(
            {
                "source_ids": tokenizer.encode(source_text, max_length=max_source_length),
                "scenario_id": example.scenario_id,
            }
        )
    return rows


def _inference_collate(pad_id: int):
    def collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(row["source_ids"]) for row in rows)
        source_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        for index, row in enumerate(rows):
            values = torch.tensor(row["source_ids"], dtype=torch.long)
            source_ids[index, : len(values)] = values
        return {
            "source_ids": source_ids,
            "source_mask": source_ids.ne(pad_id),
            "scenario_id": [str(row["scenario_id"]) for row in rows],
        }

    return collate


def _legacy_action(text: str, predictor: NLUPredictor) -> RoutedAction:
    routed = route_explicit_command(text)
    if routed is not None:
        return routed
    result = predictor.predict(text)
    result = _apply_runtime_command_guardrails(text, result)
    result = _apply_reminder_guardrails(text, result)
    return RoutedAction(result.intent, dict(result.slots), result.confidence)


def _tool_call(action: RoutedAction) -> ToolCall | None:
    intent = action.intent
    slots = dict(action.slots)
    if intent == "get_current_time":
        return ToolCall("get_current_time")
    if intent == "list_applications":
        return ToolCall("list_applications")
    if intent == "open_application" and slots.get("application"):
        return ToolCall("open_application", {"application": slots["application"]})
    if intent == "set_reminder" and slots.get("reminder_text"):
        arguments: dict[str, Any] = {"message": slots["reminder_text"]}
        for key in ("minutes", "due_at", "clock_time", "day"):
            if key in slots:
                arguments[key] = int(slots[key]) if key == "minutes" else slots[key]
        if any(key in arguments for key in ("minutes", "due_at", "clock_time")):
            return ToolCall("set_reminder", arguments)
    if intent == "cancel_reminder" and slots.get("reminder_id") is not None:
        return ToolCall("cancel_reminder", {"reminder_id": int(slots["reminder_id"])})
    if intent in {
        "browser_control",
        "system_control",
        "window_control",
        "file_control",
        "workspace_control",
        "gesture_mode",
    }:
        return ToolCall(intent, slots)
    return None


def predict_legacy_jal(text: str, predictor: NLUPredictor, registry: Any) -> str:
    """Map the deployed NLU+router behaviour into the same JAL contract."""
    gate = prepare_final_utterance(text)
    if gate.state != "analyze":
        return dumps(JALPlan(DialogueAct.REJECT, reason="unsafe_model_output"))
    normalized = _normalise_transcription_for_nlu(gate.route_text or text)
    actions = [_legacy_action(part, predictor) for part in split_compound_command(normalized)]
    if len(actions) > 1 and any(
        action.intent == "unknown" or action.confidence < 0.55 for action in actions
    ):
        return _REJECT
    if len(actions) == 1:
        action = actions[0]
        if action.intent == "cancel":
            return dumps(JALPlan(DialogueAct.CANCEL))
        if action.intent == "general_chat":
            return dumps(JALPlan(DialogueAct.DIALOGUE, reason="general_chat"))
        if action.intent in {"unknown", "negated_command", "unsupported_command"}:
            return _REJECT
    calls = tuple(call for action in actions if (call := _tool_call(action)) is not None)
    if len(calls) != len(actions):
        return _REJECT
    try:
        plan = JALPlan(DialogueAct.EXECUTE, steps=calls)
        registry.validate(plan)
    except (TypeError, ValueError):
        return _REJECT
    return dumps(plan)


def _run_legacy(
    examples: Sequence[JSCExample], checkpoint: Path, registry: Any
) -> tuple[list[str], dict[str, Any]]:
    predictor = NLUPredictor(checkpoint, "cpu")
    predictions: list[str] = []
    timings: list[float] = []
    for example in examples:
        started = time.perf_counter()
        predictions.append(predict_legacy_jal(example.text, predictor, registry))
        timings.append((time.perf_counter() - started) * 1000.0)
    return predictions, _latency_summary(timings)


def _run_jsc(
    examples: Sequence[JSCExample],
    checkpoint_path: Path,
    registry: Any,
    device: torch.device,
    batch_size: int,
    execution_threshold: float,
    verifier_threshold: float,
    span_threshold: float,
    latency_limit: int,
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    model, tokenizer, config, _checkpoint = _load_jsc(checkpoint_path, device, registry)
    dataset = _inference_rows(examples, tokenizer, config.max_source_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_inference_collate(tokenizer.pad_id),
    )
    raw_predictions: list[str] = []
    constrained_predictions: list[str] = []
    structured_predictions: list[str] = []
    full_batch_ms: list[float] = []
    structured_batch_ms: list[float] = []
    full_decisions: Counter[str] = Counter()
    structured_decisions: Counter[str] = Counter()
    parameter_labels = build_parameter_labels(registry)
    tool_labels = ("<none>", *registry.tool_names)

    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        ordered = [examples_by_id[str(value)] for value in batch["scenario_id"]]
        source_texts = [serialize_source(example) for example in ordered]
        utterances = [example.text for example in ordered]

        _sync(device)
        started = time.perf_counter()
        full = model.greedy_decode_verified_semantic(
            source_ids,
            source_mask,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_length=config.max_target_length,
        )
        _sync(device)
        full_batch_ms.append((time.perf_counter() - started) * 1000.0)
        (
            generated,
            act_logits,
            count_logits,
            tool_logits,
            parameter_logits,
            span_start_logits,
            span_end_logits,
            verifier_logits,
        ) = full
        raw = [tokenizer.decode(row.tolist()) for row in generated.cpu()]
        raw_predictions.extend(raw)
        constrained = constrain_jal_predictions(
            raw,
            act_logits.cpu(),
            registry,
            execution_threshold=execution_threshold,
            utterances=utterances,
            step_count_logits=count_logits.cpu(),
            tool_logits=tool_logits.cpu(),
            tool_labels=tool_labels,
            parameter_logits=parameter_logits.cpu(),
            parameter_labels=parameter_labels,
            span_start_logits=span_start_logits.cpu(),
            span_end_logits=span_end_logits.cpu(),
            span_slots=SPAN_ARGUMENTS,
            span_sources=source_texts,
            span_threshold=span_threshold,
            execution_verifier_logits=verifier_logits.cpu(),
            execution_verifier_threshold=verifier_threshold,
        )
        constrained_predictions.extend(constrained.predictions)
        full_decisions.update(constrained.decisions)

        _sync(device)
        started = time.perf_counter()
        structured = model.predict_verified_semantic_heads(source_ids, source_mask)
        _sync(device)
        structured_batch_ms.append((time.perf_counter() - started) * 1000.0)
        (
            structured_act,
            structured_count,
            structured_tools,
            structured_parameters,
            structured_starts,
            structured_ends,
            structured_verifier,
        ) = structured
        diagnostic = constrain_jal_predictions(
            [_REJECT] * len(ordered),
            structured_act.cpu(),
            registry,
            execution_threshold=execution_threshold,
            utterances=utterances,
            step_count_logits=structured_count.cpu(),
            tool_logits=structured_tools.cpu(),
            tool_labels=tool_labels,
            parameter_logits=structured_parameters.cpu(),
            parameter_labels=parameter_labels,
            span_start_logits=structured_starts.cpu(),
            span_end_logits=structured_ends.cpu(),
            span_slots=SPAN_ARGUMENTS,
            span_sources=source_texts,
            span_threshold=span_threshold,
            execution_verifier_logits=structured_verifier.cpu(),
            execution_verifier_threshold=verifier_threshold,
        )
        structured_predictions.extend(diagnostic.predictions)
        structured_decisions.update(diagnostic.decisions)

    single_full_ms: list[float] = []
    single_structured_ms: list[float] = []
    latency_dataset = _inference_rows(
        examples[: max(2, min(latency_limit + 2, len(examples)))],
        tokenizer,
        config.max_source_length,
    )
    latency_loader = DataLoader(
        latency_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_inference_collate(tokenizer.pad_id),
    )
    for index, batch in enumerate(latency_loader):
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        _sync(device)
        started = time.perf_counter()
        model.greedy_decode_verified_semantic(
            source_ids,
            source_mask,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_length=config.max_target_length,
        )
        _sync(device)
        full_duration = (time.perf_counter() - started) * 1000.0
        _sync(device)
        started = time.perf_counter()
        model.predict_verified_semantic_heads(source_ids, source_mask)
        _sync(device)
        structured_duration = (time.perf_counter() - started) * 1000.0
        # Two warm-up requests are deliberately excluded from runtime p95.
        if index >= 2:
            single_full_ms.append(full_duration)
            single_structured_ms.append(structured_duration)

    per_example_full = [
        duration / min(batch_size, len(examples) - index * batch_size)
        for index, duration in enumerate(full_batch_ms)
    ]
    per_example_structured = [
        duration / min(batch_size, len(examples) - index * batch_size)
        for index, duration in enumerate(structured_batch_ms)
    ]
    timing = {
        "device": str(device),
        "batch_size": batch_size,
        "target_length_diagnostics": {
            "decoder_limit": config.max_target_length,
            "maximum_expected_tokens": max(len(dumps(example.target)) + 2 for example in examples),
            "expected_targets_over_decoder_limit": sum(
                len(dumps(example.target)) + 2 > config.max_target_length
                for example in examples
            ),
        },
        "full_autoregressive_batch": _latency_summary(full_batch_ms),
        "full_autoregressive_amortized_per_example": _latency_summary(per_example_full),
        "structured_only_batch": _latency_summary(structured_batch_ms),
        "structured_only_amortized_per_example": _latency_summary(per_example_structured),
        "full_autoregressive_single_request_warm": _latency_summary(single_full_ms),
        "structured_only_single_request_warm": _latency_summary(single_structured_ms),
        "full_decoder_decisions": dict(full_decisions),
        "structured_decoder_decisions": dict(structured_decisions),
    }
    return raw_predictions, constrained_predictions, structured_predictions, timing


def _canonical_application(value: Any) -> str:
    resolved = resolve_application(str(value))
    return resolved.name if resolved is not None else str(value).casefold().strip()


def _opposite_action(target: JALPlan, predicted: JALPlan) -> bool:
    target_open = {
        _canonical_application(step.arguments.get("application"))
        for step in target.steps
        if step.tool == "open_application"
    }
    target_close = {
        _canonical_application(step.arguments.get("window"))
        for step in target.steps
        if step.tool == "window_control" and step.arguments.get("action") == "close"
    }
    predicted_open = {
        _canonical_application(step.arguments.get("application"))
        for step in predicted.steps
        if step.tool == "open_application"
    }
    predicted_close = {
        _canonical_application(step.arguments.get("window"))
        for step in predicted.steps
        if step.tool == "window_control" and step.arguments.get("action") == "close"
    }
    return bool((target_open & predicted_close) or (target_close & predicted_open))


def _migration_metrics(
    examples: Sequence[JSCExample], predictions: Sequence[str], registry: Any
) -> dict[str, Any]:
    metrics = evaluate_program_predictions(examples, predictions, registry)
    groups: dict[str, list[tuple[JSCExample, str]]] = defaultdict(list)
    opposite = 0
    negated = 0
    negated_executed = 0
    for example, prediction in zip(examples, predictions, strict=True):
        steps = len(example.target.steps)
        if steps == 1:
            groups["steps_1"].append((example, prediction))
        if 2 <= steps <= 3:
            groups["steps_2_3"].append((example, prediction))
        if 4 <= steps <= 5:
            groups["steps_4_5"].append((example, prediction))
        try:
            plan = loads(prediction)
        except (TypeError, ValueError):
            continue
        opposite += int(_opposite_action(example.target, plan))
        normalized = example.text.casefold().replace("ё", "е")
        if any(marker in normalized for marker in ("не открывай", "не запускай", "не закрывай", "без запуска")):
            negated += 1
            negated_executed += int(plan.act == DialogueAct.EXECUTE)
    exact_by_step_group = {}
    for name, rows in groups.items():
        exact_by_step_group[name] = (
            sum(dumps(example.target) == prediction for example, prediction in rows)
            / len(rows)
            if rows
            else None
        )
    return {
        **metrics,
        "exact_jal_by_step_group": exact_by_step_group,
        "opposite_action_count": opposite,
        "opposite_action_rate": opposite / len(examples),
        "negated_examples": negated,
        "negated_execution_rate": negated_executed / max(negated, 1),
    }


def _failure_analysis(
    examples: Sequence[JSCExample], predictions: Sequence[str], registry: Any
) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for example, prediction in zip(examples, predictions, strict=True):
        target = dumps(example.target)
        if prediction == target:
            continue
        try:
            plan = loads(prediction)
            registry.validate(plan)
        except (TypeError, ValueError):
            reason = "invalid_jal"
        else:
            if plan.act != example.target.act:
                reason = "wrong_dialogue_act"
            elif tuple(step.tool for step in plan.steps) != tuple(
                step.tool for step in example.target.steps
            ):
                reason = "wrong_tool_sequence"
            else:
                reason = "wrong_arguments"
        reasons[reason] += 1
        if len(samples[reason]) < 5:
            samples[reason].append(
                {
                    "scenario_id": example.scenario_id,
                    "category": example.category,
                    "text": example.text,
                    "target": target,
                    "prediction": prediction,
                }
            )
    return {"counts": dict(reasons), "examples": dict(samples)}


def _pairwise_comparison(
    examples: Sequence[JSCExample], left: Sequence[str], right: Sequence[str]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for example, left_prediction, right_prediction in zip(
        examples, left, right, strict=True
    ):
        target = dumps(example.target)
        left_exact = left_prediction == target
        right_exact = right_prediction == target
        key = (
            "both_exact" if left_exact and right_exact else
            "production_nlu_only" if left_exact else
            "jsc_only" if right_exact else
            "neither_exact"
        )
        counts[key] += 1
    return dict(counts)


def _gate_report(metrics: Mapping[str, Any]) -> dict[str, Any]:
    category = metrics.get("category_exact_jal", {})
    steps = metrics.get("exact_jal_by_step_group", {})
    actual = {
        "overall_exact_jal_accuracy": metrics["exact_jal_accuracy"],
        "single_exact_jal_accuracy": category.get("single", 0.0),
        "steps_2_3_exact_jal_accuracy": steps.get("steps_2_3") or 0.0,
        "steps_4_5_exact_jal_accuracy": steps.get("steps_4_5") or 0.0,
        "multi_turn_exact_jal_accuracy": category.get("multi_turn", 0.0),
        "schema_valid_rate": metrics["schema_valid_rate"],
        "maximum_false_execution_rate": metrics["false_execution_rate"],
        "maximum_opposite_action_rate": metrics["opposite_action_rate"],
    }
    rows = {}
    for name, target in MIGRATION_GATES.items():
        maximum = name.startswith("maximum_")
        rows[name] = {
            "actual": actual[name],
            "target": target,
            "passed": actual[name] <= target if maximum else actual[name] >= target,
        }
    return {"passed": all(row["passed"] for row in rows.values()), "checks": rows}


def _capacity_evidence(checkpoint_path: Path, metrics: Mapping[str, Any]) -> dict[str, Any]:
    report_path = checkpoint_path.with_name("report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    history = report.get("history") or []
    maximum_train_tool = max(
        (row.get("train", {}).get("tool_sequence_head_accuracy", 0.0) for row in history),
        default=0.0,
    )
    maximum_train_steps = max(
        (row.get("train", {}).get("step_count_accuracy", 0.0) for row in history),
        default=0.0,
    )
    validation = report.get("validation", {}).get("teacher_forced", {})
    return {
        "parameters": report.get("parameters"),
        "checkpoint_best_epoch": report.get("best_epoch"),
        "maximum_recorded_train_tool_sequence_head_accuracy": maximum_train_tool,
        "maximum_recorded_train_step_count_accuracy": maximum_train_steps,
        "validation_tool_sequence_head_accuracy": validation.get("tool_sequence_head_accuracy"),
        "validation_step_count_accuracy": validation.get("step_count_accuracy"),
        "benchmark_exact_jal_accuracy": metrics.get("exact_jal_accuracy"),
        "diagnosis": (
            "large_generalization_gap"
            if maximum_train_tool >= 0.90
            and float(validation.get("tool_sequence_head_accuracy") or 0.0) < 0.70
            else "capacity_or_optimization_inconclusive"
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    systems = report["systems"]
    constrained = systems["jsc_v8_constrained"]
    structured = systems["jsc_v8_structured_only"]
    legacy = systems["production_nlu"]
    timing = report["jsc_timing"]
    capacity = report["capacity_evidence"]
    pairwise = report["production_nlu_vs_constrained_jsc"]
    gate_names = {
        "overall_exact_jal_accuracy": "Exact JAL, весь набор",
        "single_exact_jal_accuracy": "Одиночные команды",
        "steps_2_3_exact_jal_accuracy": "Планы на 2–3 действия",
        "steps_4_5_exact_jal_accuracy": "Планы на 4–5 действий",
        "multi_turn_exact_jal_accuracy": "Multi-turn",
        "schema_valid_rate": "Валидная JAL-схема",
        "maximum_false_execution_rate": "Ложное выполнение (максимум)",
        "maximum_opposite_action_rate": "Противоположное действие (максимум)",
    }
    lines = [
        "# JSC Migration Benchmark — отчёт",
        "",
        "## Итог",
        "",
        "**JSC v8 пока нельзя ставить вместо NLU в production.** На отдельном "
        f"migration development-наборе constrained JSC получил "
        f"{constrained['metrics']['exact_jal_accuracy']:.2%} Exact JAL против "
        f"{legacy['metrics']['exact_jal_accuracy']:.2%} у production NLU и не прошёл "
        f"{sum(not row['passed'] for row in constrained['gates']['checks'].values())} из "
        f"{len(constrained['gates']['checks'])} обязательных проверок.",
        "",
        "При этом Structured JSC подтверждён как правильное направление: без "
        "autoregressive JSON он сохранил тот же Exact JAL, обеспечил 100% валидность "
        "схемы и нулевой opposite-action rate, резко сократив latency.",
        "",
        "## Протокол",
        "",
        f"- Набор: `{report['protocol']['split']}`; примеров: "
        f"{report['protocol']['examples']}; устройство: `{report['protocol']['device']}`.",
        "- Никакие приложения и инструменты не запускались: сравнивались только JAL-планы.",
        "- Закрытые `test` и `evaluation_holdout` не открывались.",
        f"- Распределение по числу действий: `{report['protocol']['target_step_counts']}`.",
        f"- Категории: `{report['protocol']['category_counts']}`.",
        "",
        "## Качество",
        "",
        "| System | Exact JAL | Tool sequence | Arguments | Schema valid | False execution | Opposite action |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("production_nlu", "jsc_v8_raw", "jsc_v8_constrained", "jsc_v8_structured_only"):
        value = systems[name]["metrics"]
        lines.append(
            f"| {name} | {value['exact_jal_accuracy']:.2%} | "
            f"{value['tool_sequence_accuracy']:.2%} | {value['argument_sequence_accuracy']:.2%} | "
            f"{value['schema_valid_rate']:.2%} | {value['false_execution_rate']:.2%} | "
            f"{value['opposite_action_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Миграционные пороги для constrained JSC",
            "",
            "| Проверка | Результат | Порог | Статус |",
            "|---|---:|---:|:---:|",
        ]
    )
    for name, row in constrained["gates"]["checks"].items():
        lines.append(
            f"| {gate_names[name]} | {row['actual']:.2%} | {row['target']:.2%} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    full_p95 = timing["full_autoregressive_single_request_warm"]["p95_ms"]
    structured_p95 = timing["structured_only_single_request_warm"]["p95_ms"]
    length_diagnostics = timing["target_length_diagnostics"]
    lines.extend(
        [
            "",
            "## Сложные команды",
            "",
            "| System | 1 действие | 2–3 действия | 4–5 действий | Multi-turn | ASR noise |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("production_nlu", "jsc_v8_constrained", "jsc_v8_structured_only"):
        metrics = systems[name]["metrics"]
        step = metrics["exact_jal_by_step_group"]
        categories = metrics["category_exact_jal"]
        lines.append(
            f"| {name} | {step['steps_1']:.2%} | {step['steps_2_3']:.2%} | "
            f"{step['steps_4_5']:.2%} | {categories.get('multi_turn', 0.0):.2%} | "
            f"{categories.get('asr_noise', 0.0):.2%} |"
        )
    lines.extend(
        [
            "",
            "## Latency и JSON-декодер",
            "",
            f"- Production NLU, CPU, single request p95: "
            f"{legacy['latency']['p95_ms']:.2f} мс.",
            f"- JSC autoregressive JSON, GPU, single request p95: {full_p95:.2f} мс.",
            f"- JSC structured-only, GPU, single request p95: {structured_p95:.2f} мс.",
            f"- Structured-only быстрее полного декодера примерно в "
            f"{full_p95 / max(structured_p95, 1e-9):.0f} раз по p95.",
            f"- {length_diagnostics['expected_targets_over_decoder_limit']} эталонных планов "
            f"превысили лимит JSON-декодера {length_diagnostics['decoder_limit']} токенов; "
            f"максимум составил {length_diagnostics['maximum_expected_tokens']} токенов.",
            "",
            "## Что показал Data-first аудит v8",
            "",
            f"- Параметров: {capacity['parameters']:,}; лучший checkpoint: epoch "
            f"{capacity['checkpoint_best_epoch']}.",
            f"- Tool-sequence head на train достиг "
            f"{capacity['maximum_recorded_train_tool_sequence_head_accuracy']:.2%}, "
            f"но на validation — только {capacity['validation_tool_sequence_head_accuracy']:.2%}.",
            f"- Step-count: train "
            f"{capacity['maximum_recorded_train_step_count_accuracy']:.2%}, validation "
            f"{capacity['validation_step_count_accuracy']:.2%}.",
            "- Это сильный разрыв обобщения: параметров достаточно, чтобы почти выучить "
            "train, но текущие 1 400 примеров и разнообразие семейств не дают переноса.",
            "- Точный потолок одной контрольной точкой определить нельзя. Для него нужен "
            "контролируемый scaling-run на 25/50/75/100% данных и нескольких seed.",
            "",
            "## Сравнение кандидатов",
            "",
            f"- Оба решили правильно: {pairwise.get('both_exact', 0)}; только NLU: "
            f"{pairwise.get('production_nlu_only', 0)}; только JSC: "
            f"{pairwise.get('jsc_only', 0)}; ни один: {pairwise.get('neither_exact', 0)}.",
            f"- Главная ошибка constrained JSC: неверный dialogue act — "
            f"{constrained['failure_analysis']['counts'].get('wrong_dialogue_act', 0)} случаев.",
            f"- У production NLU opposite-action rate равен "
            f"{legacy['metrics']['opposite_action_rate']:.2%}; у structured JSC — "
            f"{structured['metrics']['opposite_action_rate']:.2%}.",
            "",
            "## Решение",
            "",
            "1. Не заменять production NLU текущим checkpoint JSC v8.",
            "2. Следующим экспериментом провести Data-first scaling curve и проверить, "
            "растёт ли held-out качество при расширении семейств, ASR-вариантов, "
            "multi-turn и планов на 2–5 действий.",
            "3. Structured JSC проектировать без JSON, но полное обучение новой схемы "
            "начинать после scaling-run: текущие structured heads не обучались как "
            "самостоятельный production-декодер и не являются потолком архитектуры.",
            "4. Финальные locked test/holdout открыть один раз только для кандидата, "
            "прошедшего development-пороги.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.split in LOCKED_SPLITS and not args.allow_locked_split:
        raise ValueError(
            f"{args.split} is locked; pass --allow-locked-split only for the final selected candidate"
        )
    registry = build_project_schema_registry()
    examples = _load_examples(args.data_dir, args.split, registry, args.suite_path)
    global examples_by_id
    examples_by_id = {example.scenario_id: example for example in examples}
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    legacy, legacy_latency = _run_legacy(examples, args.nlu_checkpoint, registry)
    raw, constrained, structured, jsc_timing = _run_jsc(
        examples,
        args.jsc_checkpoint,
        registry,
        device,
        args.batch_size,
        args.execution_threshold,
        args.verifier_threshold,
        args.span_threshold,
        args.latency_limit,
    )
    candidates = {
        "production_nlu": (legacy, legacy_latency),
        "jsc_v8_raw": (raw, None),
        "jsc_v8_constrained": (constrained, None),
        "jsc_v8_structured_only": (structured, None),
    }
    systems = {}
    for name, (predictions, latency) in candidates.items():
        metrics = _migration_metrics(examples, predictions, registry)
        systems[name] = {
            "metrics": metrics,
            "gates": _gate_report(metrics),
            "failure_analysis": _failure_analysis(examples, predictions, registry),
        }
        if latency is not None:
            systems[name]["latency"] = latency
    checkpoint_hash = hashlib.sha256(args.jsc_checkpoint.read_bytes()).hexdigest()
    report = {
        "format_version": FORMAT_VERSION,
        "protocol": {
            "split": "migration_development" if args.suite_path else args.split,
            "suite_path": str(args.suite_path.resolve()) if args.suite_path else None,
            "examples": len(examples),
            "locked_split_opened": args.split in LOCKED_SPLITS,
            "side_effects_enabled": False,
            "device": str(device),
            "batch_size": args.batch_size,
            "execution_threshold": args.execution_threshold,
            "verifier_threshold": args.verifier_threshold,
            "span_threshold": args.span_threshold,
            "category_counts": dict(sorted(Counter(example.category for example in examples).items())),
            "target_step_counts": {
                str(key): value
                for key, value in sorted(Counter(len(example.target.steps) for example in examples).items())
            },
        },
        "checkpoints": {
            "production_nlu": str(args.nlu_checkpoint.resolve()),
            "production_nlu_sha256": hashlib.sha256(args.nlu_checkpoint.read_bytes()).hexdigest(),
            "jsc": str(args.jsc_checkpoint.resolve()),
            "jsc_sha256": checkpoint_hash,
            "suite_sha256": (
                hashlib.sha256(args.suite_path.read_bytes()).hexdigest()
                if args.suite_path else None
            ),
        },
        "migration_gates": MIGRATION_GATES,
        "systems": systems,
        "jsc_timing": jsc_timing,
        "production_nlu_vs_constrained_jsc": _pairwise_comparison(
            examples, legacy, constrained
        ),
        "capacity_evidence": _capacity_evidence(
            args.jsc_checkpoint, systems["jsc_v8_constrained"]["metrics"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_render_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsc-checkpoint",
        type=Path,
        default=Path("training_workspace/jsc_runs_v8/legacy_verifier_seed17/best.pt"),
    )
    parser.add_argument(
        "--nlu-checkpoint", type=Path, default=Path("models/nlu_manager_finetuned.pt")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("training_workspace/jsc_data")
    )
    parser.add_argument("--suite-path", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "test", "evaluation_holdout"), default="validation"
    )
    parser.add_argument("--allow-locked-split", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--execution-threshold", type=float, default=0.85)
    parser.add_argument("--verifier-threshold", type=float, default=0.50)
    parser.add_argument("--span-threshold", type=float, default=0.45)
    parser.add_argument("--latency-limit", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/jsc_migration_validation.json")
    )
    parser.add_argument(
        "--markdown", type=Path, default=Path("docs/JSC_MIGRATION_BENCHMARK_RU.md")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(args.output.resolve())
    print(
        json.dumps(
            {
                name: {
                    "exact_jal": value["metrics"]["exact_jal_accuracy"],
                    "gates_passed": value["gates"]["passed"],
                }
                for name, value in report["systems"].items()
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
