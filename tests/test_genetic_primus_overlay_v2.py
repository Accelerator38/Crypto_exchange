from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exia.genetic_primus.fitness_v4 import score_simulation_v4
from exia.genetic_primus.measurement import build_bootstrap_bank
from exia.genetic_primus.policy_v4 import (
    V4NetworkShape,
    genome_sparsity_v4,
    supervised_seed_overlay_v4,
)
from exia.genetic_primus.rolling import (
    build_rolling_origins,
    load_rolling_protocol,
    summarize_outer_deployment_evidence,
)
from exia.genetic_primus.simulator import PrimusSimulation


ROOT = Path(__file__).resolve().parents[1]
DAY_MS = 86_400_000


def test_overlay_v2_seals_rolling_730d_and_three_day_deployments() -> None:
    protocol = load_rolling_protocol(ROOT / "configs/genetic_primus_overlay_3d_v2.json")
    origin = build_rolling_origins(protocol)[0]
    assert (origin.train_end - origin.train_start) // DAY_MS == 730
    assert (origin.deployment_end - origin.deployment_start) // DAY_MS == 3
    assert protocol.payload["safety"]["promotion_authority"] is False


def test_binary_overlay_has_31_parameters_and_at_most_eight_features() -> None:
    shape = V4NetworkShape(30, (), 1, maximum_active_features=8)
    x = np.vstack([np.full((8, 30), -1.0), np.full((8, 30), 1.0)]).astype(np.float32)
    y = np.r_[np.zeros(8), np.ones(8)]
    policy = supervised_seed_overlay_v4(x, y, shape=shape, seed=7, epochs=3)
    metrics = genome_sparsity_v4(
        policy.genome, shape=shape, active_weight_threshold=0.0001
    )
    assert shape.genome_size == 31
    assert metrics["active_feature_count"] <= 8
    assert policy.allow(x[0]) is False
    assert policy.allow(x[-1]) is True


def test_outer_evidence_uses_completed_deployments_not_training_folds() -> None:
    rows = []
    for origin, challenger, active in ((0, 2.0, 1.0), (1, -1.0, 0.0)):
        rows.extend(
            [
                {
                    "origin_index": origin,
                    "agent_id": "GeneticPrimusChallenger",
                    "net_bps": challenger,
                    "evidence_role": "outer_deployment",
                },
                {
                    "origin_index": origin,
                    "agent_id": "GeneticPrimusActive",
                    "net_bps": active,
                    "evidence_role": "outer_deployment",
                },
            ]
        )
    evidence = summarize_outer_deployment_evidence(
        pd.DataFrame(rows), minimum_completed_origins=60, minimum_positive_fraction=0.65
    )
    assert evidence["completed_outer_origins"] == 2
    assert evidence["positive_outer_slice_fraction"] == 0.5
    assert evidence["evidence_complete"] is False
    assert evidence["promotion_approved"] is False


def _score(*, daily_bps: float, allowed_fraction: float):
    timestamps = np.arange(61, dtype=np.int64) * DAY_MS
    folds = [
        {
            "name": f"window_{index}",
            "start": pd.Timestamp((1 + 20 * index) * DAY_MS, unit="ms", tz="UTC").isoformat(),
            "end": pd.Timestamp((21 + 20 * index) * DAY_MS, unit="ms", tz="UTC").isoformat(),
        }
        for index in range(3)
    ]
    bank = build_bootstrap_bank(
        pd.DataFrame({"timestamp": timestamps, "symbol": "LINKUSDT"}),
        start_timestamp=0,
        cutoff_timestamp=61 * DAY_MS,
        folds=folds,
        bank_id=f"coverage-{allowed_fraction}",
        seed=7,
        samples=4,
        block_days=1,
    )
    portfolio = pd.DataFrame(
        {
            "timestamp": timestamps[1:],
            "net_return": np.full(60, daily_bps / 10_000.0),
            "drawdown": np.zeros(60),
        }
    )
    no_trade = portfolio.assign(net_return=0.0)
    ledger = pd.DataFrame(
        {
            "entry_timestamp": [10 * DAY_MS],
            "direction": ["LONG"],
            "market_state": ["trend"],
            "symbol": ["LINKUSDT"],
        }
    )
    simulation = PrimusSimulation(
        ledger=ledger,
        portfolio=portfolio,
        action_counts={"SHORT": 0, "FLAT": 59, "LONG": 1},
        turnover_fills=2,
        diagnostics={"overlay_opportunities": 100, "overlay_allowed": int(allowed_fraction * 100)},
    )
    recency = {
        "half_life_days": [180],
        "uniform_anchor": 1.0,
        "minimum_ess_days": {"180": 1.0},
        "recent_lcb_weight": 0.7,
        "worst_window_lcb_weight": 0.3,
        "drawdown_penalty_bps_per_pct": 0.0,
        "turnover_penalty_bps_per_fill_per_day": 0.0,
        "complexity_penalty_bps_per_100_parameters": 0.0,
        "active_parameter_penalty_bps_per_100": 0.0,
        "active_feature_penalty_bps": 0.0,
        "action_instability_penalty_bps": 0.0,
    }
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
        "recency": recency,
        "complexity_parameter_count": 31,
        "abstention": {
            "minimum_allowed_fraction": 0.05,
            "maximum_allowed_fraction": 0.35,
            "coverage_penalty_bps": 40.0,
        },
        "paired_incremental": {
            "references": ["NoTrade", "Momentum_return_1", "Incumbent"],
            "absolute_reference": "NoTrade",
            "baseline_reference": "Momentum_return_1",
            "minimum_lcb_bps": 0.0,
        },
        "adaptive_multiple_testing": {
            "max_hypotheses_per_origin": 107,
            "selection_bias_penalty_multiplier": 1.0,
        },
    }
    return score_simulation_v4(
        simulation,
        folds=folds,
        config=config,
        key="coverage",
        bootstrap_bank=bank,
        paired_reference_portfolios={
            "NoTrade": no_trade,
            "Momentum_return_1": portfolio,
            "Incumbent": no_trade.copy(),
        },
    )


def test_coverage_constraint_prevents_flat_search_collapse() -> None:
    admissible = _score(daily_bps=1.0, allowed_fraction=0.10)
    flat_like = _score(daily_bps=0.0, allowed_fraction=0.00)
    assert admissible.metrics["coverage_constraint_passed"] is True
    assert flat_like.metrics["coverage_constraint_passed"] is False
    assert admissible.metrics["paired_reference_aliases"] == {"Incumbent": "NoTrade"}
    assert admissible.ranking_key > flat_like.ranking_key
