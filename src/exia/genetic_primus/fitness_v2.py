from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .fitness import FitnessResult, score_simulation
from .simulator import PrimusSimulation


SLICE_FAILURE_PREFIXES = (
    "nonpositive_market_state_lcb:",
    "nonpositive_direction_lcb:",
    "nonpositive_symbol_lcb:",
)
OVERALL_FAILURES = ("nonpositive_mean", "nonpositive_lcb")


def lexicographic_key_from_values(
    *,
    constraint_violation: float,
    overall_failure_count: int,
    slice_failure_count: int,
    worst_slice_lcb_bps: float,
    fitness: float,
    overall_lcb_bps: float,
) -> tuple[float, ...]:
    feasible = float(constraint_violation) <= 1.0e-12
    return (
        float(feasible),
        -float(constraint_violation),
        -float(overall_failure_count),
        -float(slice_failure_count),
        float(worst_slice_lcb_bps),
        float(fitness),
        float(overall_lcb_bps),
    )


def v2_ranking_key(score: FitnessResult) -> tuple[float, ...]:
    metrics = score.metrics
    return lexicographic_key_from_values(
        constraint_violation=float(score.constraint_violation),
        overall_failure_count=int(metrics["overall_failure_count"]),
        slice_failure_count=int(metrics["slice_failure_count"]),
        worst_slice_lcb_bps=float(metrics["worst_supported_slice_lcb_bps"]),
        fitness=float(score.fitness),
        overall_lcb_bps=float(metrics["lcb_bps"] or -1.0e9),
    )


def ordinal_selection_scores(keys: list[tuple[float, ...]]) -> np.ndarray:
    order = sorted(range(len(keys)), key=lambda index: (keys[index], -index))
    ranks = np.empty(len(keys), dtype=np.float64)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def score_simulation_v2(
    simulation: PrimusSimulation,
    *,
    config: Mapping[str, Any],
    key: str,
) -> FitnessResult:
    base = score_simulation(
        simulation,
        config=config,
        key=key,
        final_gate=True,
    )
    minimum_slice = int(config["minimum_slice_trades"])
    supported = base.slices.loc[base.slices["trades"] >= minimum_slice]
    finite_lcb = supported["lcb_bps"].dropna().astype("float64")
    worst_lcb = float(finite_lcb.min()) if len(finite_lcb) else -1.0e9
    slice_failures = tuple(
        failure
        for failure in base.failed_gates
        if failure.startswith(SLICE_FAILURE_PREFIXES)
    )
    overall_failures = tuple(
        failure for failure in base.failed_gates if failure in OVERALL_FAILURES
    )
    diagnostic_score = (
        -1.0e12 * float(base.constraint_violation)
        -1.0e9 * len(overall_failures)
        -1.0e6 * len(slice_failures)
        +1.0e3 * float(np.clip(worst_lcb, -1_000.0, 1_000.0))
        +float(np.clip(base.fitness, -100_000.0, 100_000.0))
    )
    metrics = dict(base.metrics)
    metrics.update(
        {
            "selection_score": diagnostic_score,
            "selection_contract": "risk>overall_gate>failed_slices>worst_slice_lcb>monthly_fitness",
            "overall_failure_count": len(overall_failures),
            "slice_failure_count": len(slice_failures),
            "supported_slice_count": int(len(supported)),
            "worst_supported_slice_lcb_bps": worst_lcb,
        }
    )
    return replace(base, selection_score=diagnostic_score, metrics=metrics)
