from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from .fitness import FitnessResult
from .fitness_v2 import score_simulation_v2
from .simulator import PrimusSimulation


def lexicographic_key_v3_from_values(
    *,
    constraint_violation: float,
    overall_failure_count: int,
    temporal_fitness: float,
    overall_median_bps: float,
    slice_failure_count: int,
    worst_slice_lcb_bps: float,
    overall_lcb_bps: float,
) -> tuple[float, ...]:
    feasible = float(constraint_violation) <= 1.0e-12
    temporal_margin = min(float(temporal_fitness), float(overall_median_bps))
    temporal_passed = temporal_margin > 0.0
    return (
        float(feasible),
        -float(constraint_violation),
        -float(overall_failure_count),
        float(temporal_passed),
        0.0 if temporal_passed else temporal_margin,
        -float(slice_failure_count),
        float(worst_slice_lcb_bps),
        float(temporal_fitness),
        float(overall_lcb_bps),
    )


def v3_ranking_key(score: FitnessResult) -> tuple[float, ...]:
    metrics = score.metrics
    return lexicographic_key_v3_from_values(
        constraint_violation=float(score.constraint_violation),
        overall_failure_count=int(metrics["overall_failure_count"]),
        temporal_fitness=float(score.fitness),
        overall_median_bps=float(metrics["median_bps"] or -1.0e9),
        slice_failure_count=int(metrics["slice_failure_count"]),
        worst_slice_lcb_bps=float(metrics["worst_supported_slice_lcb_bps"]),
        overall_lcb_bps=float(metrics["lcb_bps"] or -1.0e9),
    )


def ordinal_selection_scores_v3(keys: list[tuple[float, ...]]) -> np.ndarray:
    order = sorted(range(len(keys)), key=lambda index: (keys[index], -index))
    ranks = np.empty(len(keys), dtype=np.float64)
    for rank, index in enumerate(order):
        ranks[index] = float(rank)
    return ranks


def score_simulation_v3(
    simulation: PrimusSimulation,
    *,
    config: Mapping[str, Any],
    key: str,
) -> FitnessResult:
    base = score_simulation_v2(simulation, config=config, key=key)
    temporal_margin = min(float(base.fitness), float(base.metrics["median_bps"] or -1.0e9))
    temporal_passed = temporal_margin > 0.0
    diagnostic_score = (
        float(base.selection_score)
        + (1.0e11 if temporal_passed else 1.0e6 * float(np.clip(temporal_margin, -10_000.0, 0.0)))
    )
    metrics = dict(base.metrics)
    metrics.update(
        {
            "selection_score": diagnostic_score,
            "selection_contract": (
                "risk>overall_gate>temporal_gate>failed_slices>"
                "worst_slice_lcb>monthly_fitness"
            ),
            "temporal_gate_passed": temporal_passed,
            "temporal_margin_bps": temporal_margin,
        }
    )
    return replace(base, selection_score=diagnostic_score, metrics=metrics)
