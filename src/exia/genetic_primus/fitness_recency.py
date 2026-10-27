from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .measurement import BootstrapBank
from .simulator import PrimusSimulation


DAY_MS = 86_400_000.0


@dataclass(frozen=True)
class RecencyFitness:
    fitness: float
    metrics: Mapping[str, Any]
    failed_gates: tuple[str, ...]


def recency_ess_threshold(
    config: Mapping[str, Any], *, half_life_days: int, observed_days: int
) -> float:
    threshold = float(config["minimum_ess_days"][str(half_life_days)])
    fraction = config.get("minimum_ess_fraction_of_observed_days")
    if fraction is not None:
        threshold = min(threshold, float(fraction) * int(observed_days))
    return threshold


def anchored_recency_weights(
    timestamps: Sequence[int],
    *,
    cutoff_ms: int,
    half_life_days: float,
    uniform_anchor: float,
) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.int64)
    if not len(values):
        return np.empty(0, dtype=np.float64)
    if half_life_days <= 0.0:
        raise ValueError("half-life must be positive")
    if not 0.0 <= uniform_anchor <= 1.0:
        raise ValueError("uniform anchor must be between zero and one")
    age_days = np.maximum(0.0, (float(cutoff_ms) - values.astype(float)) / DAY_MS)
    decay = np.exp2(-age_days / float(half_life_days))
    decay /= decay.sum()
    return (1.0 - uniform_anchor) * decay + uniform_anchor / len(values)


def weighted_block_lcb(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    resample_indices: np.ndarray,
    alpha: float,
) -> float:
    data = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    indices = np.asarray(resample_indices, dtype=np.int64)
    if data.ndim != 1 or weight.shape != data.shape:
        raise ValueError("weighted LCB values and weights must align")
    if indices.ndim != 2 or indices.shape[1] != len(data):
        raise ValueError("weighted LCB bootstrap shape mismatch")
    if not len(data):
        raise ValueError("weighted LCB requires observations")
    return float(np.quantile(data[indices] @ weight, float(alpha)))


def score_recency_fitness(
    daily: pd.DataFrame,
    simulation: PrimusSimulation,
    *,
    fold_lcbs: Sequence[float],
    config: Mapping[str, Any],
    bootstrap_bank: BootstrapBank,
    alpha: float,
    complexity_parameter_count: int,
    active_parameter_count: int | None = None,
    active_feature_count: int | None = None,
    total_feature_count: int | None = None,
) -> RecencyFitness:
    timestamps = daily["timestamp"].to_numpy(np.int64)
    values = daily["net_bps"].to_numpy(np.float64)
    if not len(values):
        return RecencyFitness(
            fitness=-1.0e9,
            metrics={"recency_primary_lcb_bps": -1.0e9},
            failed_gates=("recency_no_daily_observations",),
        )
    indices = bootstrap_bank.indices_for("overall", timestamps)
    anchor = float(config["uniform_anchor"])
    half_lives = tuple(int(value) for value in config["half_life_days"])
    metrics: dict[str, Any] = {}
    failures: list[str] = []
    lcbs: list[float] = []
    for half_life in half_lives:
        weights = anchored_recency_weights(
            timestamps,
            cutoff_ms=bootstrap_bank.cutoff_ms,
            half_life_days=half_life,
            uniform_anchor=anchor,
        )
        lcb = weighted_block_lcb(values, weights, resample_indices=indices, alpha=alpha)
        ess = float(1.0 / np.square(weights).sum())
        lcbs.append(lcb)
        metrics[f"recency_mean_{half_life}d_bps"] = float(values @ weights)
        metrics[f"recency_lcb_{half_life}d_bps"] = lcb
        metrics[f"recency_ess_{half_life}d"] = ess
        ess_threshold = recency_ess_threshold(
            config,
            half_life_days=half_life,
            observed_days=len(values),
        )
        metrics[f"recency_ess_threshold_{half_life}d"] = ess_threshold
        if ess < ess_threshold:
            failures.append(f"recency_ess_{half_life}d")

    primary_lcb = min(lcbs)
    finite_folds = np.asarray(list(fold_lcbs), dtype=np.float64)
    finite_folds = finite_folds[np.isfinite(finite_folds)]
    worst_window_lcb = float(finite_folds.min()) if len(finite_folds) else -1.0e9
    days = max(1, len(values))
    fills_per_day = float(simulation.turnover_fills / days)
    total_actions = int(sum(simulation.action_counts.values()))
    action_instability = float(simulation.turnover_fills / max(1, total_actions))
    max_drawdown = (
        float(-simulation.portfolio["drawdown"].min()) if not simulation.portfolio.empty else 0.0
    )
    drawdown_penalty = float(config["drawdown_penalty_bps_per_pct"]) * max_drawdown * 100.0
    turnover_penalty = float(config["turnover_penalty_bps_per_fill_per_day"]) * fills_per_day
    complexity_penalty = (
        float(config["complexity_penalty_bps_per_100_parameters"])
        * max(0, int(complexity_parameter_count))
        / 100.0
    )
    parameter_count = max(0, int(complexity_parameter_count))
    active_parameters = min(
        parameter_count,
        max(0, int(parameter_count if active_parameter_count is None else active_parameter_count)),
    )
    active_features = max(0, int(0 if active_feature_count is None else active_feature_count))
    total_features = max(active_features, int(0 if total_feature_count is None else total_feature_count))
    active_parameter_penalty = (
        float(config.get("active_parameter_penalty_bps_per_100", 0.0))
        * active_parameters
        / 100.0
    )
    active_feature_penalty = float(config.get("active_feature_penalty_bps", 0.0)) * active_features
    instability_penalty = float(config["action_instability_penalty_bps"]) * action_instability
    total_penalty = (
        drawdown_penalty
        + turnover_penalty
        + complexity_penalty
        + active_parameter_penalty
        + active_feature_penalty
        + instability_penalty
    )
    fitness = (
        float(config["recent_lcb_weight"]) * primary_lcb
        + float(config["worst_window_lcb_weight"]) * worst_window_lcb
        - total_penalty
    )
    metrics.update(
        {
            "recency_primary_lcb_bps": primary_lcb,
            "recency_worst_window_lcb_bps": worst_window_lcb,
            "recency_action_instability": action_instability,
            "recency_drawdown_penalty_bps": drawdown_penalty,
            "recency_turnover_penalty_bps": turnover_penalty,
            "recency_complexity_penalty_bps": complexity_penalty,
            "recency_active_parameter_count": active_parameters,
            "recency_parameter_sparsity": float(
                1.0 - active_parameters / max(1, parameter_count)
            ),
            "recency_active_feature_count": active_features,
            "recency_feature_sparsity": float(1.0 - active_features / max(1, total_features)),
            "recency_active_parameter_penalty_bps": active_parameter_penalty,
            "recency_active_feature_penalty_bps": active_feature_penalty,
            "recency_parsimony_penalty_bps": (
                complexity_penalty + active_parameter_penalty + active_feature_penalty
            ),
            "recency_instability_penalty_bps": instability_penalty,
            "recency_total_penalty_bps": total_penalty,
        }
    )
    return RecencyFitness(float(fitness), metrics, tuple(failures))
