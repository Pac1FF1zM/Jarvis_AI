"""Deterministic two-stage hyperparameter search for the local NLU manager."""
from __future__ import annotations

import math
import random
import statistics
from copy import deepcopy
from typing import Any


def _choice(rng: random.Random, values: list[Any], name: str) -> Any:
    if not values:
        raise ValueError(f"search space {name!r} must not be empty")
    return rng.choice(values)


def _uniform(rng: random.Random, bounds: list[float], name: str, *, log: bool = False) -> float:
    if len(bounds) != 2 or bounds[0] > bounds[1]:
        raise ValueError(f"search range {name!r} must be [minimum, maximum]")
    low, high = map(float, bounds)
    if log:
        if low <= 0:
            raise ValueError(f"log search range {name!r} must be positive")
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    return rng.uniform(low, high)


def generate_phase_one(search: dict[str, Any]) -> list[dict[str, Any]]:
    """Sample comparable candidates; every trial uses the exact same data seed."""
    trials = int(search.get("trials", 24))
    if trials < 1:
        raise ValueError("search.trials must be positive")
    seed = int(search.get("search_seed", 20260729))
    training_seed = int(search.get("phase_one_seed", 17))
    space = search.get("space") or {}
    rng = random.Random(seed)
    methods = list(space.get("method", ["standard", "augmented", "curriculum"]))
    candidates: list[dict[str, Any]] = []
    for index in range(trials):
        # Ensure all training methods are represented before random sampling.
        method = methods[index] if index < len(methods) else _choice(rng, methods, "method")
        candidate = {
            "name": f"search_{index + 1:02d}",
            "trainer": "manager",
            "architecture": "char_cnn",
            "method": method,
            "epochs": int(search.get("phase_one_epochs", 28)),
            "patience": int(search.get("phase_one_patience", 6)),
            "batch_size": int(_choice(rng, list(space.get("batch_size", [128, 256])), "batch_size")),
            "learning_rate": _uniform(
                rng, list(space.get("learning_rate", [0.0004, 0.0015])),
                "learning_rate", log=True,
            ),
            "weight_decay": _uniform(
                rng, list(space.get("weight_decay", [0.00003, 0.0003])),
                "weight_decay", log=True,
            ),
            "custom_fraction": _uniform(
                rng, list(space.get("custom_fraction", [0.30, 0.60])), "custom_fraction"
            ),
            "curriculum_start_fraction": _uniform(
                rng,
                list(space.get("curriculum_start_fraction", [0.15, 0.30])),
                "curriculum_start_fraction",
            ),
            "route_loss_weight": _uniform(
                rng, list(space.get("route_loss_weight", [0.20, 0.45])), "route_loss_weight"
            ),
            "slot_loss_weight": _uniform(
                rng, list(space.get("slot_loss_weight", [0.60, 1.60])), "slot_loss_weight"
            ),
            "slot_o_weight": _uniform(
                rng, list(space.get("slot_o_weight", [0.22, 0.36])), "slot_o_weight"
            ),
            "slot_consistency_weight": _uniform(
                rng,
                list(space.get("slot_consistency_weight", [0.10, 0.80])),
                "slot_consistency_weight",
            ),
            "no_slot_loss_weight": _uniform(
                rng,
                list(space.get("no_slot_loss_weight", [0.05, 0.30])),
                "no_slot_loss_weight",
            ),
            "embedding_dim": int(
                _choice(rng, list(space.get("embedding_dim", [48, 64, 96])), "embedding_dim")
            ),
            "hidden_dim": int(
                _choice(rng, list(space.get("hidden_dim", [64, 96, 128])), "hidden_dim")
            ),
            "label_smoothing": _uniform(
                rng, list(space.get("label_smoothing", [0.0, 0.06])), "label_smoothing"
            ),
            "warmup_epochs": int(
                _choice(
                    rng,
                    list(space.get("warmup_epochs", [3, 5, 7])),
                    "warmup_epochs",
                )
            ),
            "min_lr_ratio": _uniform(
                rng, list(space.get("min_lr_ratio", [0.05, 0.20])), "min_lr_ratio"
            ),
            "ema_decay": _uniform(
                rng, list(space.get("ema_decay", [0.992, 0.998])), "ema_decay"
            ),
            "max_length": int(search.get("max_length", 128)),
            "seed": training_seed,
        }
        if candidate["curriculum_start_fraction"] >= candidate["custom_fraction"]:
            candidate["curriculum_start_fraction"] = max(
                0.05, candidate["custom_fraction"] * 0.5
            )
        candidates.append(candidate)
    return candidates


def confirmation_experiments(
    finalists: list[dict[str, Any]], search: dict[str, Any]
) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in search.get("confirmation_seeds", [17, 43, 101, 211, 307])]
    if len(set(seeds)) < 3:
        raise ValueError("search.confirmation_seeds must contain at least three unique seeds")
    min_passing = int(search.get("min_passing_seeds", max(1, len(set(seeds)) - 1)))
    if not 1 <= min_passing <= len(set(seeds)):
        raise ValueError(
            "search.min_passing_seeds must be between 1 and the number of unique seeds"
        )
    result: list[dict[str, Any]] = []
    for finalist in finalists:
        for seed in seeds:
            experiment = deepcopy(finalist)
            experiment["candidate"] = finalist["name"]
            experiment["name"] = f"{finalist['name']}_seed_{seed}"
            experiment["seed"] = seed
            experiment["epochs"] = int(search.get("confirmation_epochs", 70))
            experiment["patience"] = int(search.get("confirmation_patience", 12))
            result.append(experiment)
    return result


def aggregate_scores(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = {}
    for result in results:
        grouped.setdefault(str(result["candidate"]), []).append(float(result["selection_score"]))
    return {
        name: {
            "runs": len(scores),
            "mean": statistics.fmean(scores),
            "stddev": statistics.pstdev(scores),
            "minimum": min(scores),
            "maximum": max(scores),
        }
        for name, scores in grouped.items()
    }
