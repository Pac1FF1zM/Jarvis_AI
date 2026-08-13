"""Production-shaped inference wrapper for experimental Structured JSC."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .data import DialogueTurn
from .jal import JALPlan, ToolSchemaRegistry
from .structured_codec import decode_structured_jal
from .structured_features import serialize_structured_input
from .structured_model import StructuredJSCConfig, StructuredJSCModel
from .tokenizer import JSCCharTokenizer


@dataclass(frozen=True)
class StructuredPrediction:
    jal: str
    decisions: Mapping[str, int]
    latency_ms: float


class StructuredJSCPredictor:
    """Load one inference checkpoint and predict typed JAL without side effects."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        registry: ToolSchemaRegistry,
        *,
        device: str = "cpu",
        thresholds: Mapping[str, float] | None = None,
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("kind") != "jsc_structured_inference":
            raise ValueError("not a Structured JSC inference checkpoint")
        if checkpoint.get("tool_schema_sha256") != registry.schema_fingerprint:
            raise ValueError("Structured JSC checkpoint schema mismatch")
        self.registry = registry
        self.tokenizer = JSCCharTokenizer.from_dict(checkpoint["tokenizer"])
        self.model = StructuredJSCModel(
            StructuredJSCConfig.from_dict(checkpoint["model_config"])
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.tool_labels = tuple(checkpoint["tool_labels"])
        self.parameter_labels = tuple(checkpoint["parameter_labels"])
        self.missing_labels = tuple(checkpoint["missing_labels"])
        self.reason_labels = tuple(checkpoint["reason_labels"])
        training = dict(checkpoint["training_config"])
        self.max_source_length = int(training["max_source_length"])
        defaults = {
            "execution_threshold": training["execution_threshold"],
            "verifier_threshold": training["verifier_threshold"],
            "parameter_threshold": training["parameter_threshold"],
            "span_threshold": training["span_threshold"],
            "missing_threshold": training["missing_threshold"],
        }
        defaults.update(thresholds or {})
        self.thresholds = defaults

    @torch.inference_mode()
    def predict(
        self,
        text: str,
        *,
        history: tuple[DialogueTurn, ...] = (),
        state: JALPlan | None = None,
    ) -> StructuredPrediction:
        source = serialize_structured_input(text, history, state)
        ids = self.tokenizer.encode(source, max_length=self.max_source_length)
        source_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        source_mask = source_ids.ne(self.tokenizer.pad_id)
        started = time.perf_counter()
        outputs = self.model(source_ids, source_mask)
        decoded = decode_structured_jal(
            utterances=[text],
            source_texts=[source],
            act_logits=outputs[0].cpu(),
            count_logits=outputs[1].cpu(),
            tool_logits=outputs[2].cpu(),
            parameter_logits=outputs[3].cpu(),
            span_start_logits=outputs[4].cpu(),
            span_end_logits=outputs[5].cpu(),
            verifier_logits=outputs[6].cpu(),
            missing_logits=outputs[7].cpu(),
            reason_logits=outputs[8].cpu(),
            registry=self.registry,
            tool_labels=self.tool_labels,
            parameter_labels=self.parameter_labels,
            missing_labels=self.missing_labels,
            reason_labels=self.reason_labels,
            states=[state],
            **self.thresholds,
        )
        return StructuredPrediction(
            decoded.predictions[0],
            decoded.decisions,
            (time.perf_counter() - started) * 1000.0,
        )
