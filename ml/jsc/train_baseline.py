"""Train or inspect one from-scratch JSC baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline_training import TrainingConfig, inspect_training, train_baseline
from .models import ARCHITECTURES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--data-dir", default="training_workspace/jsc_data")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--act-loss-weight", type=float, default=0.25)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-source-length", type=int, default=384)
    parser.add_argument("--max-target-length", type=int, default=384)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--copy-mechanism", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--structured-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--parameter-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--span-heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--semantic-pooling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--execution-verifier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--freeze-base-for-parameters", action="store_true")
    parser.add_argument("--freeze-base-for-spans", action="store_true")
    parser.add_argument("--freeze-base-for-semantics", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    output = args.output_dir or str(
        Path("training_workspace/jsc_runs")
        / f"{args.architecture}_seed{args.seed}"
    )
    compact = args.smoke
    return TrainingConfig(
        architecture=args.architecture,
        data_dir=args.data_dir,
        output_dir=output,
        seed=args.seed,
        device=args.device,
        epochs=1 if compact else args.epochs,
        batch_size=min(args.batch_size, 4) if compact else args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        act_loss_weight=args.act_loss_weight,
        d_model=min(args.d_model, 32) if compact else args.d_model,
        encoder_layers=1 if compact else args.encoder_layers,
        decoder_layers=1 if compact else args.decoder_layers,
        attention_heads=min(args.attention_heads, 4),
        feedforward_dim=min(args.feedforward_dim, 64) if compact else args.feedforward_dim,
        dropout=args.dropout,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        patience=args.patience,
        gradient_clip=args.gradient_clip,
        warmup_ratio=args.warmup_ratio,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        copy_mechanism=args.copy_mechanism,
        structured_heads=args.structured_heads,
        parameter_heads=args.parameter_heads,
        span_heads=args.span_heads,
        semantic_pooling=args.semantic_pooling,
        execution_verifier=args.execution_verifier,
        init_checkpoint=args.init_checkpoint,
        freeze_base_for_parameters=args.freeze_base_for_parameters,
        freeze_base_for_spans=args.freeze_base_for_spans,
        freeze_base_for_semantics=args.freeze_base_for_semantics,
        resume=args.resume,
        smoke=compact,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    result = inspect_training(config) if args.check_only else train_baseline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
