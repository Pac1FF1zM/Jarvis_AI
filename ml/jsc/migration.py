"""Fail-closed migration stages and admission gates for replacing legacy NLU."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

from .jal import DialogueAct, JALPlan


class MigrationStage(IntEnum):
    INDEPENDENT_SHADOW = 0
    AGREEMENT_CANARY = 1
    RESTRICTED_REVERSIBLE = 2
    JSC_PRIMARY = 3
    NLU_REMOVED = 4

    @classmethod
    def parse(cls, value: str) -> "MigrationStage":
        try:
            return cls[
                value.strip().replace("-", "_").upper()
            ]
        except KeyError as exc:
            raise ValueError(f"unknown JSC migration stage: {value}") from exc

    @property
    def config_name(self) -> str:
        return self.name.casefold()


@dataclass(frozen=True)
class StageAdmission:
    requested: MigrationStage
    active: MigrationStage
    admitted: bool
    reasons: tuple[str, ...]


def admit_stage(
    requested: MigrationStage, evidence: Mapping[str, Any]
) -> StageAdmission:
    """Admit only stages supported by reviewed human evidence and release history."""
    if requested <= MigrationStage.AGREEMENT_CANARY:
        return StageAdmission(requested, requested, True, ())
    reasons: list[str] = []
    checks = (
        (int(evidence.get("reviewed_voice_turns", 0)) >= 1000, "reviewed_voice_turns_below_1000"),
        (float(evidence.get("false_execution_rate", 1.0)) == 0.0, "false_execution_not_zero"),
        (float(evidence.get("opposite_action_rate", 1.0)) == 0.0, "opposite_action_not_zero"),
        (float(evidence.get("semantic_exact_rate", 0.0)) >= 0.90, "semantic_exact_below_0_90"),
        (float(evidence.get("correction_accuracy", 0.0)) >= 0.95, "correction_below_0_95"),
        (float(evidence.get("ood_recall", 0.0)) >= 0.98, "ood_recall_below_0_98"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    stable_cycles = int(evidence.get("stable_release_cycles", 0))
    if requested >= MigrationStage.JSC_PRIMARY and stable_cycles < 1:
        reasons.append("jsc_primary_requires_one_stable_release_cycle")
    if requested >= MigrationStage.NLU_REMOVED and stable_cycles < 2:
        reasons.append("nlu_removal_requires_two_stable_release_cycles")
    if reasons:
        return StageAdmission(
            requested, MigrationStage.AGREEMENT_CANARY, False, tuple(reasons)
        )
    return StageAdmission(requested, requested, True, ())


@dataclass(frozen=True)
class ReversibilityDecision:
    eligible: bool
    reasons: tuple[str, ...]


def classify_reversibility(plan: JALPlan) -> ReversibilityDecision:
    """Allow only read-only or explicitly compensatable JAL calls."""
    if plan.act != DialogueAct.EXECUTE:
        return ReversibilityDecision(False, ("plan_is_not_execute",))
    reasons: list[str] = []
    for index, call in enumerate(plan.steps):
        args = dict(call.arguments)
        allowed = False
        if call.tool in {"get_current_time", "list_applications", "list_reminders"}:
            allowed = True
        elif call.tool in {"open_application", "set_reminder", "cancel_reminder"}:
            allowed = True
        elif call.tool == "system_control":
            allowed = args.get("action") in {"volume_up", "volume_down", "volume_mute"}
        elif call.tool == "browser_control":
            allowed = args.get("action") in {"new_tab", "close_tab"}
        elif call.tool == "file_control":
            allowed = args.get("action") in {"create_folder", "rename"}
        elif call.tool == "window_control":
            allowed = args.get("action") in {
                "minimize", "maximize", "restore", "switch", "snap_left", "snap_right"
            }
        elif call.tool == "workspace_control":
            allowed = args.get("action") == "launch"
        if not allowed:
            reasons.append(f"step_{index}_{call.tool}_not_reversible")
    return ReversibilityDecision(not reasons, tuple(reasons))


class RollingErrorBudget:
    """In-memory circuit breaker for unsafe disagreements after promotion."""

    def __init__(self, *, window: int = 200, minimum_agreement: float = 0.95) -> None:
        if window < 10:
            raise ValueError("error-budget window must be >= 10")
        if not 0.0 <= minimum_agreement <= 1.0:
            raise ValueError("minimum_agreement must be between 0 and 1")
        self.window = window
        self.minimum_agreement = minimum_agreement
        self._observations: list[tuple[bool, bool]] = []

    def observe(self, *, exact_agreement: bool, unsafe_disagreement: bool) -> bool:
        self._observations.append((exact_agreement, unsafe_disagreement))
        self._observations[:] = self._observations[-self.window :]
        if any(unsafe for _exact, unsafe in self._observations):
            return False
        if len(self._observations) < min(20, self.window):
            return True
        agreement = sum(exact for exact, _unsafe in self._observations) / len(
            self._observations
        )
        return agreement >= self.minimum_agreement
