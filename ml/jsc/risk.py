"""Calibrated selective-execution policy for Structured JSC.

The policy is intentionally independent from tool execution.  It turns model
scores into a small, auditable accept/abstain decision which can be calibrated
on a frozen validation set and logged in shadow telemetry.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SelectiveRiskPolicy:
    minimum_act_confidence: float = 0.65
    minimum_act_margin: float = 0.15
    maximum_normalized_entropy: float = 0.65
    minimum_verifier_confidence: float = 0.90

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> "SelectiveRiskPolicy":
        source = dict(values or {})
        return cls(
            minimum_act_confidence=float(source.get("minimum_act_confidence", 0.65)),
            minimum_act_margin=float(source.get("minimum_act_margin", 0.15)),
            maximum_normalized_entropy=float(source.get("maximum_normalized_entropy", 0.65)),
            minimum_verifier_confidence=float(source.get("minimum_verifier_confidence", 0.90)),
        )


@dataclass(frozen=True)
class SelectiveRiskDecision:
    accepted: bool
    reason: str
    act_confidence: float
    act_margin: float
    normalized_entropy: float
    verifier_confidence: float

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


def evaluate_selective_risk(
    act_probabilities: Sequence[float],
    verifier_confidence: float,
    policy: SelectiveRiskPolicy,
) -> SelectiveRiskDecision:
    """Evaluate an execution candidate; failed checks always abstain."""
    if not act_probabilities:
        raise ValueError("act_probabilities must not be empty")
    ordered = sorted((float(value) for value in act_probabilities), reverse=True)
    confidence = ordered[0]
    margin = confidence - (ordered[1] if len(ordered) > 1 else 0.0)
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in ordered)
    normalized_entropy = entropy / math.log(len(ordered)) if len(ordered) > 1 else 0.0
    checks = (
        (confidence >= policy.minimum_act_confidence, "low_act_confidence"),
        (margin >= policy.minimum_act_margin, "low_act_margin"),
        (normalized_entropy <= policy.maximum_normalized_entropy, "high_act_entropy"),
        (
            verifier_confidence >= policy.minimum_verifier_confidence,
            "low_verifier_confidence",
        ),
    )
    reason = next((name for passed, name in checks if not passed), "accepted")
    return SelectiveRiskDecision(
        accepted=reason == "accepted",
        reason=reason,
        act_confidence=confidence,
        act_margin=margin,
        normalized_entropy=normalized_entropy,
        verifier_confidence=float(verifier_confidence),
    )
