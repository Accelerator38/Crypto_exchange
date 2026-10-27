from __future__ import annotations

import numpy as np
import pandas as pd

from exia.genetic_primus.fitness_recency import anchored_recency_weights, score_recency_fitness
from exia.genetic_primus.fitness_v4 import score_simulation_v4
from exia.genetic_primus.measurement import BootstrapBank, build_bootstrap_bank
from exia.genetic_primus.simulator import PrimusSimulation


DAY_MS = 86_400_000


def _daily(days: int = 360) -> pd.DataFrame:
    timestamps = np.arange(days, dtype=np.int64) * DAY_MS
    returns = np.r_[np.ones(days - 30), np.full(30, -3.0)]
    return pd.DataFrame({"timestamp": timestamps, "net_bps": returns})


def _bank(daily: pd.DataFrame, samples: int = 4) -> BootstrapBank:
    timestamps = daily["timestamp"].to_numpy(np.int64)
    indices = np.tile(np.arange(len(timestamps), dtype=np.int64), (samples, 1))
    return BootstrapBank(
        bank_id="unit-recency",
        seed=1,
        samples=samples,
        block_days=1,
        cutoff_ms=int(timestamps[-1] + DAY_MS),
        folds_sha256="unit",
        scope_timestamps={"overall": timestamps},
        scope_indices={"overall": indices},
        content_sha256="unit",
    )


def _config() -> dict:
    return {
        "half_life_days": [30, 90, 180],
        "uniform_anchor": 0.2,
        "recent_lcb_weight": 0.7,
        "worst_window_lcb_weight": 0.3,
        "minimum_ess_days": {"30": 1.0, "90": 1.0, "180": 1.0},
        "drawdown_penalty_bps_per_pct": 1.0,
        "turnover_penalty_bps_per_fill_per_day": 0.5,
        "complexity_penalty_bps_per_100_parameters": 1.0,
        "active_weight_threshold": 0.05,
        "active_parameter_penalty_bps_per_100": 0.5,
        "active_feature_penalty_bps": 0.1,
        "action_instability_penalty_bps": 20.0,
    }


def _simulation(*, fills: int = 0, drawdown: float = 0.0) -> PrimusSimulation:
    portfolio = pd.DataFrame({"drawdown": [-drawdown]})
    return PrimusSimulation(
        ledger=pd.DataFrame(),
        portfolio=portfolio,
        action_counts={"SHORT": 100, "FLAT": 100, "LONG": 100},
        turnover_fills=fills,
    )


def test_anchored_weights_are_normalized_and_retain_history() -> None:
    daily = _daily()
    weights = anchored_recency_weights(
        daily["timestamp"], cutoff_ms=_bank(daily).cutoff_ms, half_life_days=30, uniform_anchor=0.2
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] >= 0.2 / len(weights)
    assert weights[-1] > weights[0]


def test_short_half_life_sees_recent_losses_and_ess_is_ordered() -> None:
    daily = _daily()
    score = score_recency_fitness(
        daily,
        _simulation(),
        fold_lcbs=[0.2, 0.4, 0.1],
        config=_config(),
        bootstrap_bank=_bank(daily),
        alpha=0.05,
        complexity_parameter_count=100,
    )
    assert score.metrics["recency_lcb_30d_bps"] < score.metrics["recency_lcb_180d_bps"]
    assert score.metrics["recency_ess_30d"] < score.metrics["recency_ess_90d"]
    assert score.metrics["recency_ess_90d"] < score.metrics["recency_ess_180d"]
    assert score.metrics["recency_primary_lcb_bps"] == score.metrics["recency_lcb_30d_bps"]


def test_drawdown_turnover_complexity_and_instability_reduce_fitness() -> None:
    daily = _daily()
    bank = _bank(daily)
    baseline = score_recency_fitness(
        daily,
        _simulation(),
        fold_lcbs=[0.1],
        config=_config(),
        bootstrap_bank=bank,
        alpha=0.05,
        complexity_parameter_count=0,
    )
    penalized = score_recency_fitness(
        daily,
        _simulation(fills=30, drawdown=0.1),
        fold_lcbs=[0.1],
        config=_config(),
        bootstrap_bank=bank,
        alpha=0.05,
        complexity_parameter_count=200,
    )
    assert penalized.fitness < baseline.fitness
    assert penalized.metrics["recency_drawdown_penalty_bps"] == 10.0
    assert penalized.metrics["recency_complexity_penalty_bps"] == 2.0
    assert penalized.metrics["recency_instability_penalty_bps"] == 2.0


def test_v4_integration_uses_common_bank_and_exposes_recency_metrics() -> None:
    timestamps = np.arange(61, dtype=np.int64) * DAY_MS
    frame = pd.DataFrame({"timestamp": timestamps, "symbol": "LINKUSDT"})
    folds = [
        {"name": f"window_{index}", "start": pd.Timestamp((1 + 20 * index) * DAY_MS, unit="ms", tz="UTC").isoformat(), "end": pd.Timestamp((21 + 20 * index) * DAY_MS, unit="ms", tz="UTC").isoformat()}
        for index in range(3)
    ]
    bank = build_bootstrap_bank(
        frame,
        start_timestamp=0,
        cutoff_timestamp=61 * DAY_MS,
        folds=folds,
        bank_id="integration",
        seed=7,
        samples=4,
        block_days=1,
    )
    portfolio = pd.DataFrame(
        {
            "timestamp": timestamps[1:],
            "net_return": np.full(60, 0.0001),
            "drawdown": np.zeros(60),
        }
    )
    ledger = pd.DataFrame(
        {
            "entry_timestamp": [10 * DAY_MS, 40 * DAY_MS],
            "direction": ["LONG", "SHORT"],
            "market_state": ["trend", "range"],
            "symbol": ["LINKUSDT", "LINKUSDT"],
        }
    )
    simulation = PrimusSimulation(
        ledger=ledger,
        portfolio=portfolio,
        action_counts={"SHORT": 20, "FLAT": 20, "LONG": 20},
        turnover_fills=2,
    )
    config = {
        "bootstrap_samples": 4,
        "bootstrap_alpha": 0.05,
        "bootstrap_block_days": 1,
        "minimum_trades": 1,
        "max_drawdown": 1.0,
        "max_turnover_fills_per_day": 10.0,
        "max_direction_share": 1.0,
        "max_state_share": 1.0,
        "max_symbol_share": 1.0,
        "minimum_positive_fold_means": 1,
        "turnover_penalty_bps_per_fill_per_day": 0.0,
        "recency": _config(),
        "complexity_parameter_count": 100,
    }
    score = score_simulation_v4(
        simulation, folds=folds, config=config, key="ignored-with-bank", bootstrap_bank=bank
    )
    assert score.metrics["recency_primary_lcb_bps"] > 0.0
    assert score.metrics["recency_worst_window_lcb_bps"] > 0.0
    assert score.metrics["recency_complexity_penalty_bps"] == 1.0


def test_sparse_genome_beats_dense_genome_when_returns_are_equal() -> None:
    daily = _daily()
    common = {
        "daily": daily,
        "simulation": _simulation(),
        "fold_lcbs": [0.1],
        "config": _config(),
        "bootstrap_bank": _bank(daily),
        "alpha": 0.05,
        "complexity_parameter_count": 200,
        "total_feature_count": 30,
    }
    dense = score_recency_fitness(
        **common, active_parameter_count=200, active_feature_count=30
    )
    sparse = score_recency_fitness(
        **common, active_parameter_count=20, active_feature_count=5
    )
    assert sparse.fitness > dense.fitness
    assert sparse.metrics["recency_feature_sparsity"] == 25 / 30
    assert sparse.metrics["recency_parsimony_penalty_bps"] < dense.metrics[
        "recency_parsimony_penalty_bps"
    ]
