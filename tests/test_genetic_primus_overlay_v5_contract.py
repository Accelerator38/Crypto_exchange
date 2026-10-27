from pathlib import Path

import numpy as np

from exia.genetic_primus.fitness_v4 import _opportunity_normalized_concentration
from exia.genetic_primus.global_multiple_testing import (
    build_global_multiple_testing_evidence,
)
from exia.genetic_primus.objective_registry import (
    deflated_sharpe_probability,
    load_immutable_objective_registry,
    probability_of_backtest_overfitting,
)
from exia.genetic_primus.policy_v4 import V4NetworkShape, stability_selection_v4
from exia.genetic_primus.promotion_v4 import _drawdown_gate_failures
from exia.genetic_primus.rolling import genetic_simulation_kwargs, load_rolling_protocol
from exia.genetic_primus.simulator import overlay_permission_step


ROOT = Path(__file__).resolve().parents[1]


def test_v5_contract_and_global_registry_are_hash_locked() -> None:
    protocol = load_rolling_protocol(
        ROOT / "configs/genetic_primus_overlay_3d_v5_sealed.json"
    )
    registry = load_immutable_objective_registry(
        ROOT / "configs/genetic_primus_objective_registry_v2.json", workspace=ROOT
    )
    assert registry["objective_count"] == 20
    assert registry["objective_ids"][-1] == "overlay_3d_v5_sealed"
    kwargs = genetic_simulation_kwargs(protocol)
    assert kwargs["overlay_entry_logit_threshold"] == 0.15
    assert kwargs["overlay_exit_logit_threshold"] == -0.05
    assert kwargs["overlay_cooldown_bars"] == 2


def test_no_trade_transition_uses_absolute_drawdown_budget() -> None:
    config = {
        "max_absolute_drawdown": 0.10,
        "max_drawdown_degradation": 0.0,
        "incumbent_minimum_turnover_fills": 1,
    }
    failures, incumbent_trades, mode = _drawdown_gate_failures(
        challenger_drawdown=0.03,
        incumbent_drawdown=0.0,
        incumbent_turnover_fills=0,
        config=config,
    )
    assert failures == []
    assert not incumbent_trades
    assert mode == "absolute_no_trade_transition"
    failures, incumbent_trades, _ = _drawdown_gate_failures(
        challenger_drawdown=0.03,
        incumbent_drawdown=0.02,
        incumbent_turnover_fills=1,
        config=config,
    )
    assert incumbent_trades
    assert "drawdown_vs_incumbent" in failures


def test_stability_selection_requires_frequency_and_sign_agreement() -> None:
    shape = V4NetworkShape(inputs=4, hidden=(), outputs=1, maximum_active_features=2)
    genomes = (
        np.asarray([0.8, 0.7, 0.01, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.9, 0.6, 0.02, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.7, -0.8, 0.01, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.6, 0.9, 0.01, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.5, 0.8, 0.01, 0.0, 0.0], dtype=np.float32),
    )
    result = stability_selection_v4(
        genomes,
        shape=shape,
        minimum_active_frequency=0.60,
        minimum_sign_agreement=0.80,
        active_weight_threshold=0.05,
    )
    assert result["stable_feature_indices"] == [0, 1]
    assert result["passed"]


def test_opportunity_normalization_removes_market_supply_bias() -> None:
    balanced = _opportunity_normalized_concentration(
        {"A": 90, "B": 10}, {"A": 45, "B": 5}, minimum_bucket_opportunities=5
    )
    concentrated = _opportunity_normalized_concentration(
        {"A": 90, "B": 10}, {"A": 40, "B": 10}, minimum_bucket_opportunities=5
    )
    assert balanced == 1.0
    assert concentrated == 2.0


def test_registered_dsr_and_pbo_are_deterministic() -> None:
    dsr = deflated_sharpe_probability(
        [0.01, -0.005, 0.02, 0.004, -0.002, 0.015],
        registered_trial_sharpes=[-0.1, 0.0, 0.1, 0.2],
    )
    matrix = np.asarray(
        [[0.01 * ((row + column) % 5 - 2) for column in range(3)] for row in range(24)]
    )
    pbo = probability_of_backtest_overfitting(matrix, slice_count=4)
    assert 0.0 <= dsr["deflated_sharpe_probability"] <= 1.0
    assert 0.0 <= pbo["pbo"] <= 1.0


def test_hysteresis_and_cooldown_transition_is_deterministic() -> None:
    state = overlay_permission_step(
        baseline_target=1, score=0.16, permission=False, permission_direction=0,
        cooldown_remaining=0, entry_threshold=0.15, exit_threshold=-0.05,
        cooldown_bars=2,
    )
    assert state == (True, 1, 0)
    state = overlay_permission_step(
        baseline_target=1, score=0.0, permission=state[0], permission_direction=state[1],
        cooldown_remaining=state[2], entry_threshold=0.15, exit_threshold=-0.05,
        cooldown_bars=2,
    )
    assert state == (True, 1, 0)
    state = overlay_permission_step(
        baseline_target=-1, score=1.0, permission=state[0], permission_direction=state[1],
        cooldown_remaining=state[2], entry_threshold=0.15, exit_threshold=-0.05,
        cooldown_bars=2,
    )
    assert state == (False, 0, 2)


def test_historical_global_multiple_testing_is_diagnostic_only() -> None:
    evidence = build_global_multiple_testing_evidence(
        ROOT / "configs/genetic_primus_outcome_catalog_v1.json", workspace=ROOT
    )
    assert evidence["status"] == "DIAGNOSTIC_ONLY"
    assert not evidence["promotion_eligible"]
    assert evidence["registered_trial_count"] == 20.0
    assert evidence["materialized_objective_count"] == 3
    assert evidence["observation_count"] == 6
    assert "minimum_observations" in evidence["failed_readiness_gates"]
