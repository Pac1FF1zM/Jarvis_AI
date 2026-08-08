"""Inspect free-running JSC predictions without opening the locked test split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.jsc.baseline_training import (
    TrainingConfig,
    _concatenate_span_logits,
    _load_context,
    _resolve_device,
)
from ml.jsc.baseline_metrics import evaluate_program_predictions
from ml.jsc.constrained_decoding import constrain_jal_predictions
from ml.jsc.jal import dumps, loads
from ml.jsc.models import BaselineConfig, JSCBaselineModel
from ml.jsc.sequence_data import JSCSequenceDataset, make_collate_fn, serialize_source
from ml.jsc.span_labels import SPAN_ARGUMENTS
from ml.jsc.structured_labels import build_parameter_labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("training_workspace/jsc_data"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=(0.75, 0.80, 0.85, 0.90, 0.95, 0.98),
    )
    parser.add_argument(
        "--span-thresholds",
        nargs="+",
        type=float,
        default=(0.45,),
    )
    parser.add_argument(
        "--verifier-thresholds",
        nargs="+",
        type=float,
        default=(0.50,),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = BaselineConfig.from_dict(checkpoint["model_config"])
    config = TrainingConfig(
        architecture=model_config.architecture,
        data_dir=str(args.data_dir),
        output_dir=".",
        device=args.device,
        batch_size=args.batch_size,
        d_model=model_config.d_model,
        encoder_layers=model_config.encoder_layers,
        decoder_layers=model_config.decoder_layers,
        attention_heads=model_config.attention_heads,
        feedforward_dim=model_config.feedforward_dim,
        dropout=model_config.dropout,
        max_source_length=model_config.max_source_length,
        max_target_length=model_config.max_target_length,
        copy_mechanism=model_config.copy_mechanism,
        structured_heads=bool(model_config.num_tools),
        parameter_heads=bool(model_config.num_parameter_labels),
        span_heads=bool(model_config.num_span_slots),
        semantic_pooling=model_config.semantic_pooling,
        execution_verifier=model_config.execution_verifier,
    )
    context = _load_context(config)
    if checkpoint["tokenizer_fingerprint"] != context.tokenizer.fingerprint:
        raise ValueError("checkpoint tokenizer does not match current training data")
    model = JSCBaselineModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = JSCSequenceDataset(context.validation, context.tokenizer, context.limits)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(context.tokenizer.pad_id),
    )
    examples_by_id = {item.scenario_id: item for item in context.validation}
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    predictions: list[str] = []
    ordered_examples = []
    all_act_logits: list[torch.Tensor] = []
    all_count_logits: list[torch.Tensor] = []
    all_tool_logits: list[torch.Tensor] = []
    all_parameter_logits: list[torch.Tensor] = []
    all_span_start_logits: list[torch.Tensor] = []
    all_span_end_logits: list[torch.Tensor] = []
    all_verifier_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            if model_config.execution_verifier:
                (
                    generated,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                    span_start_logits,
                    span_end_logits,
                    verifier_logits,
                ) = model.greedy_decode_verified_semantic(
                    batch["source_ids"].to(device),
                    batch["source_mask"].to(device),
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=model_config.max_target_length,
                )
                all_count_logits.append(count_logits.cpu())
                all_tool_logits.append(tool_logits.cpu())
                all_parameter_logits.append(parameter_logits.cpu())
                all_span_start_logits.append(span_start_logits.cpu())
                all_span_end_logits.append(span_end_logits.cpu())
                all_verifier_logits.append(verifier_logits.cpu())
            elif model_config.num_span_slots:
                (
                    generated,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                    span_start_logits,
                    span_end_logits,
                ) = model.greedy_decode_full_semantic(
                    batch["source_ids"].to(device),
                    batch["source_mask"].to(device),
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=model_config.max_target_length,
                )
                all_count_logits.append(count_logits.cpu())
                all_tool_logits.append(tool_logits.cpu())
                all_parameter_logits.append(parameter_logits.cpu())
                all_span_start_logits.append(span_start_logits.cpu())
                all_span_end_logits.append(span_end_logits.cpu())
            elif model_config.num_parameter_labels:
                (
                    generated,
                    act_logits,
                    count_logits,
                    tool_logits,
                    parameter_logits,
                ) = model.greedy_decode_schema_conditioned(
                    batch["source_ids"].to(device),
                    batch["source_mask"].to(device),
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=model_config.max_target_length,
                )
                all_count_logits.append(count_logits.cpu())
                all_tool_logits.append(tool_logits.cpu())
                all_parameter_logits.append(parameter_logits.cpu())
            elif model_config.num_tools:
                generated, act_logits, count_logits, tool_logits = model.greedy_decode_structured(
                    batch["source_ids"].to(device),
                    batch["source_mask"].to(device),
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=model_config.max_target_length,
                )
                all_count_logits.append(count_logits.cpu())
                all_tool_logits.append(tool_logits.cpu())
            else:
                generated, act_logits = model.greedy_decode(
                    batch["source_ids"].to(device),
                    batch["source_mask"].to(device),
                    bos_id=context.tokenizer.bos_id,
                    eos_id=context.tokenizer.eos_id,
                    max_length=model_config.max_target_length,
                )
            all_act_logits.append(act_logits.cpu())
            for scenario_id, token_ids, act_id in zip(
                batch["scenario_id"], generated.cpu(), act_logits.argmax(-1).cpu()
            ):
                example = examples_by_id[scenario_id]
                prediction = context.tokenizer.decode(token_ids.tolist())
                predictions.append(prediction)
                ordered_examples.append(example)
                target = dumps(example.target)
                status = "exact" if prediction == target else "mismatch"
                if status != "exact":
                    try:
                        plan = loads(prediction)
                        context.registry.validate(plan)
                        status = "valid_mismatch"
                    except ValueError as exc:
                        status = type(exc).__name__
                counts[status] += 1
                if len(rows) < args.limit and prediction != target:
                    rows.append(
                        {
                            "scenario_id": scenario_id,
                            "category": example.category,
                            "text": example.text,
                            "target_act": example.target.act.value,
                            "aux_act": list(type(example.target.act))[int(act_id)].value,
                            "target": target,
                            "prediction": prediction,
                            "status": status,
                        }
                    )
    act_tensor = torch.cat(all_act_logits, dim=0)
    count_tensor = torch.cat(all_count_logits, dim=0) if all_count_logits else None
    tool_tensor = torch.cat(all_tool_logits, dim=0) if all_tool_logits else None
    parameter_tensor = (
        torch.cat(all_parameter_logits, dim=0) if all_parameter_logits else None
    )
    span_start_tensor = (
        _concatenate_span_logits(all_span_start_logits)
        if all_span_start_logits
        else None
    )
    span_end_tensor = (
        _concatenate_span_logits(all_span_end_logits)
        if all_span_end_logits
        else None
    )
    verifier_tensor = (
        torch.cat(all_verifier_logits, dim=0) if all_verifier_logits else None
    )
    parameter_labels = build_parameter_labels(context.registry)
    constrained_reports = {}
    for span_threshold in args.span_thresholds:
        for verifier_threshold in args.verifier_thresholds:
            for threshold in args.thresholds:
                constrained = constrain_jal_predictions(
                predictions,
                act_tensor,
                context.registry,
                execution_threshold=threshold,
                utterances=[example.text for example in ordered_examples],
                step_count_logits=count_tensor,
                tool_logits=tool_tensor,
                tool_labels=("<none>", *context.registry.tool_names),
                parameter_logits=parameter_tensor,
                parameter_labels=(
                    parameter_labels if parameter_tensor is not None else None
                ),
                span_start_logits=span_start_tensor,
                span_end_logits=span_end_tensor,
                span_slots=SPAN_ARGUMENTS if span_start_tensor is not None else None,
                span_sources=(
                    [serialize_source(example) for example in ordered_examples]
                    if span_start_tensor is not None
                    else None
                ),
                    span_threshold=span_threshold,
                    execution_verifier_logits=verifier_tensor,
                    execution_verifier_threshold=verifier_threshold,
                )
                key = f"{threshold:.3f}"
                if len(args.span_thresholds) > 1 or len(args.verifier_thresholds) > 1:
                    key = (
                        f"exec={threshold:.3f},span={span_threshold:.3f},"
                        f"verifier={verifier_threshold:.3f}"
                    )
                constrained_reports[key] = {
                    **evaluate_program_predictions(
                        ordered_examples, constrained.predictions, context.registry
                    ),
                    "decoder_decisions": constrained.decisions,
                }
    report = {
        "counts": dict(counts),
        "raw_metrics": evaluate_program_predictions(
            ordered_examples, predictions, context.registry
        ),
        "constrained_metrics_by_threshold": constrained_reports,
        "examples": rows,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
