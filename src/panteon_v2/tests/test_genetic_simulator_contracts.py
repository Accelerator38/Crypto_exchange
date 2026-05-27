from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_crypto_genetics():
    root = Path(__file__).resolve().parents[3]
    genetics_dir = root / "Genetics_DL_Agents"
    if str(genetics_dir) not in sys.path:
        sys.path.insert(0, str(genetics_dir))
    return importlib.import_module("crypto_genetics")


def _load_contract_eval_tool():
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("tools.evaluate_genetics_contract")


def _write_retrodate_csv(path: Path, *, timestamp: int, dt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "open", "high", "low", "close", "volume", "symbol", "datetime"],
        )
        writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "symbol": "BTC/USDT",
            "datetime": dt,
        })


def _patch_minimal_sim_config(monkeypatch, cg, *, spot_fee, futures_fee):
    monkeypatch.setattr(cg, "TRAIN_FEE", spot_fee)
    monkeypatch.setattr(cg, "TRAIN_FUTURES_FEE", futures_fee)
    monkeypatch.setattr(cg, "TRAIN_SLIPPAGE", 0.0)
    monkeypatch.setattr(cg, "TRAIN_FRACTION", 1.0)
    monkeypatch.setattr(cg, "TRAIN_LEVERAGE", 1.0)
    monkeypatch.setattr(cg, "TRAIN_STOP_PCT", 0.0)
    monkeypatch.setattr(cg, "TRAIN_STOP_MODE", "from_peak")
    monkeypatch.setattr(cg, "TRAIN_FUNDING_RATE", 0.0)
    monkeypatch.setattr(cg, "TRAIN_APY_RATE", 0.0)
    monkeypatch.setattr(cg, "TRAIN_LIQ_FEE", 0.0)
    monkeypatch.setattr(cg, "TRAIN_BAR", 1)
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 1)
    monkeypatch.setattr(cg, "SWAP_FEE_MULTIPLIER", 1.0)


def test_futures_simulation_uses_futures_fee_not_spot_fee(monkeypatch):
    cg = _load_crypto_genetics()
    _patch_minimal_sim_config(monkeypatch, cg, spot_fee=0.10, futures_fee=0.0)

    prices = np.array([[[100.0]], [[100.0]]], dtype=np.float64).reshape(2, 1)
    spot_buy = np.array([[[2], [0]]], dtype=np.int32)
    futures_long = np.array([[[5], [0]]], dtype=np.int32)

    spot_rets, *_ = cg.simulate_batch(
        spot_buy,
        prices,
        initial_capital=1000.0,
        execution_lag_bars=0,
        use_numba=False,
    )
    fut_rets, *_ = cg.simulate_batch(
        futures_long,
        prices,
        initial_capital=1000.0,
        execution_lag_bars=0,
        use_numba=False,
    )

    assert spot_rets[0] < -15.0
    assert abs(fut_rets[0]) < 1e-9


def test_next_bar_execution_lag_delays_actions(monkeypatch):
    cg = _load_crypto_genetics()
    _patch_minimal_sim_config(monkeypatch, cg, spot_fee=0.0, futures_fee=0.0)

    prices = np.array([[100.0], [200.0]], dtype=np.float64)
    buy_first_bar = np.array([[[2], [0]]], dtype=np.int32)

    same_bar_rets, *_ = cg.simulate_batch(
        buy_first_bar,
        prices,
        initial_capital=1000.0,
        execution_lag_bars=0,
        use_numba=False,
    )
    next_bar_rets, *_ = cg.simulate_batch(
        buy_first_bar,
        prices,
        initial_capital=1000.0,
        execution_lag_bars=1,
        use_numba=False,
    )

    assert same_bar_rets[0] == 100.0
    assert next_bar_rets[0] == 0.0


def _patch_robust_fitness_weights(monkeypatch, cg, *, concentration_penalty_w=0.25):
    monkeypatch.setattr(cg, "ROBUST_TRIM_FRACTION", 0.10)
    monkeypatch.setattr(cg, "ROBUST_MEDIAN_RET_W", 0.80)
    monkeypatch.setattr(cg, "ROBUST_TRIMMED_MEAN_W", 0.60)
    monkeypatch.setattr(cg, "ROBUST_CVAR5_RET_W", 0.20)
    monkeypatch.setattr(cg, "ROBUST_CONC_MAX_PCT", 30.0)
    monkeypatch.setattr(cg, "ROBUST_CONC_PENALTY_W", concentration_penalty_w)
    monkeypatch.setattr(cg, "ROBUST_POS_PERIOD_TARGET", 0.55)
    monkeypatch.setattr(cg, "ROBUST_POS_PERIOD_PENALTY_W", 5.0)


def test_robust_fitness_adjustment_penalizes_single_outlier_concentration(monkeypatch):
    cg = _load_crypto_genetics()
    _patch_robust_fitness_weights(monkeypatch, cg)

    concentrated = [0.5] * 20 + [100.0]
    stable = [2.0] * 21
    adjustments = cg._compute_robust_fitness_adjustment(
        np.array([concentrated, stable], dtype=np.float64)
    )

    assert adjustments[0] < 0.0
    assert adjustments[1] > 0.0
    assert adjustments[1] > adjustments[0]


def test_compute_fitness_applies_robust_concentration_penalty(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([[0.5] * 20 + [100.0]], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.full_like(period_rets, 0.05)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    base = cg._compute_fitness(period_rets, period_dds, trade_rates)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", True)
    monkeypatch.setattr(cg, "ROBUST_MEDIAN_RET_W", 0.0)
    monkeypatch.setattr(cg, "ROBUST_TRIMMED_MEAN_W", 0.0)
    monkeypatch.setattr(cg, "ROBUST_CVAR5_RET_W", 0.0)
    monkeypatch.setattr(cg, "ROBUST_CONC_MAX_PCT", 30.0)
    monkeypatch.setattr(cg, "ROBUST_CONC_PENALTY_W", 1.0)
    monkeypatch.setattr(cg, "ROBUST_POS_PERIOD_TARGET", 0.0)
    monkeypatch.setattr(cg, "ROBUST_POS_PERIOD_PENALTY_W", 0.0)
    robust = cg._compute_fitness(period_rets, period_dds, trade_rates)

    assert robust[0] < base[0] - 50.0


def test_core_fitness_prefers_stable_returns_over_single_outlier(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([
        [2.0] * 24,
        [0.5] * 23 + [60.0],
    ], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.full_like(period_rets, 0.05)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    monkeypatch.setattr(cg, "TRADE_REWARD_W", 0.0)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_MAX_PENALTY_W", 0.0, raising=False)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_MAX_PENALTY_W", 0.0, raising=False)
    monkeypatch.setattr(cg, "ACTION_FEASIBILITY_PENALTY_W", 0.0, raising=False)
    monkeypatch.setattr(cg, "FITNESS_CORE_ROBUST_BLEND", 1.0, raising=False)
    monkeypatch.setattr(cg, "FITNESS_OUTLIER_CONC_MAX_PCT", 25.0, raising=False)
    monkeypatch.setattr(cg, "FITNESS_OUTLIER_CONC_PENALTY_W", 0.50, raising=False)

    fits = cg._compute_fitness(period_rets, period_dds, trade_rates)

    assert fits[0] > fits[1]


def test_action_contract_metrics_counts_max_position_saturation(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 1)

    actions = np.array([[
        [5, 5, 0],
        [5, 5, 0],
        [0, 0, 0],
    ]], dtype=np.int32)

    metrics = cg._action_contract_metrics(actions)

    assert metrics["turnover_rates"][0] == 4 / 9
    assert metrics["saturation_rates"][0] == 3 / 9


def test_position_aware_policy_suppresses_duplicate_and_saturated_opens(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 1)

    actions = np.array([[
        [5, 5, 0],
        [5, 5, 0],
        [8, 5, 0],
    ]], dtype=np.int32)

    adjusted = cg._apply_position_aware_action_policy(actions)

    assert adjusted.tolist() == [[
        [5, 0, 0],
        [0, 0, 0],
        [8, 5, 0],
    ]]


def test_position_state_features_encode_symbol_and_capacity_state(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 2)

    x = np.zeros((1, 2, cg.N_INPUT), dtype=np.float32)
    prices = np.array([110.0, 90.0], dtype=np.float64)
    spot_open = np.array([[True, False]])
    spot_entry = np.array([[100.0, 0.0]], dtype=np.float64)
    fut_side = np.array([[0, -1]], dtype=np.int8)
    fut_entry = np.array([[0.0, 100.0]], dtype=np.float64)
    n_pos = np.array([2], dtype=np.int32)

    cg._inject_position_state_features(
        x,
        prices,
        spot_open,
        spot_entry,
        fut_side,
        fut_entry,
        n_pos,
    )

    assert x[0, 0, 12] > 0.0
    assert x[0, 1, 13] > 0.0
    assert x[0, :, 14].tolist() == [1.0, 1.0]
    assert x[0, :, 15].tolist() == [1.0, 1.0]


def test_position_aware_forward_does_not_return_saturated_opens(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 1)

    genome = np.zeros(cg.GENOME_SIZE, dtype=np.float32)
    b4_start = (
        cg._W1s + cg._b1s
        + cg._W2s + cg._b2s
        + cg._W3s + cg._b3s
        + cg._W4s
    )
    genome[b4_start + 5] = 10.0

    feat = np.zeros((3, 2, cg.N_INPUT), dtype=np.float32)
    prices = np.full((3, 2), 100.0, dtype=np.float64)

    actions, suppression_metrics = cg._batch_forward_position_aware_numpy(
        genome[None],
        feat,
        prices,
        return_suppression_metrics=True,
    )
    metrics = cg._action_contract_metrics(actions.astype(np.int32))

    assert actions.tolist() == [[
        [5, 0],
        [0, 0],
        [0, 0],
    ]]
    assert metrics["saturation_rates"][0] == 0.0
    assert metrics["turnover_rates"][0] == 1 / 6
    assert suppression_metrics["saturation_rates"][0] == 5 / 6
    assert suppression_metrics["invalid_open_logit_pressures"][0] > 6.0


def test_invalid_open_pressure_counts_same_bar_overcapacity(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 1)

    genome = np.zeros(cg.GENOME_SIZE, dtype=np.float32)
    b4_start = (
        cg._W1s + cg._b1s
        + cg._W2s + cg._b2s
        + cg._W3s + cg._b3s
        + cg._W4s
    )
    genome[b4_start + 5] = 10.0

    feat = np.zeros((3, 2, cg.N_INPUT), dtype=np.float32)
    prices = np.full((3, 2), 100.0, dtype=np.float64)

    _, suppression_metrics = cg._batch_forward_position_aware_numpy(
        genome[None],
        feat,
        prices,
        return_suppression_metrics=True,
    )

    assert suppression_metrics["invalid_open_logit_pressures"][0] > 8.0


def test_compute_fitness_penalizes_excess_turnover_and_saturation(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.array([[0.03, 0.03], [0.50, 0.50]], dtype=np.float64)
    saturation_rates = np.array([[0.0, 0.0], [0.35, 0.35]], dtype=np.float64)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    monkeypatch.setattr(cg, "TRADE_REWARD_W", 0.0)
    monkeypatch.setattr(cg, "TURNOVER_TARGET_RATE", 0.10)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 10.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 20.0)

    fits = cg._compute_fitness(
        period_rets,
        period_dds,
        trade_rates,
        period_saturation_rates=saturation_rates,
    )

    assert fits[0] > fits[1] + 5.0


def test_compute_fitness_does_not_prefer_low_turnover_lossy_policy(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([
        [0.45, 0.45, 0.45, 0.45, 0.45, 0.45],
        [0.10, 0.10, 0.10, 0.10, -0.70, 0.10],
    ], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.array([
        [0.14] * 6,
        [0.01] * 6,
    ], dtype=np.float64)
    saturation_rates = np.array([
        [0.06] * 6,
        [0.00] * 6,
    ], dtype=np.float64)
    invalid_open_pressure = np.array([
        [0.0] * 6,
        [0.0] * 6,
    ], dtype=np.float64)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", True)
    monkeypatch.setattr(cg, "FITNESS_CORE_ROBUST_BLEND", 0.45, raising=False)
    monkeypatch.setattr(cg, "FITNESS_MIN_CORE_MEAN_RET", 0.20, raising=False)
    monkeypatch.setattr(cg, "FITNESS_MEAN_RET_FLOOR_PENALTY_W", 20.0, raising=False)
    monkeypatch.setattr(cg, "FITNESS_NEGATIVE_PERIOD_TARGET", 0.0, raising=False)
    monkeypatch.setattr(cg, "FITNESS_NEGATIVE_PERIOD_PENALTY_W", 6.0, raising=False)
    monkeypatch.setattr(cg, "TURNOVER_TARGET_RATE", 0.10)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 8.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 25.0)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_PENALTY_W", 4.0)
    monkeypatch.setattr(cg, "ACTION_FEASIBILITY_PENALTY_W", 120.0, raising=False)

    fits = cg._compute_fitness(
        period_rets,
        period_dds,
        trade_rates,
        period_saturation_rates=saturation_rates,
        period_invalid_open_logit_pressures=invalid_open_pressure,
    )

    assert fits[0] > fits[1]


def test_compute_fitness_penalizes_invalid_open_logit_pressure(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.full_like(period_rets, 0.03)
    invalid_open_pressure = np.array([[0.0, 0.0], [2.5, 2.5]], dtype=np.float64)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    monkeypatch.setattr(cg, "TRADE_REWARD_W", 0.0)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_PENALTY_W", 3.0)

    fits = cg._compute_fitness(
        period_rets,
        period_dds,
        trade_rates,
        period_invalid_open_logit_pressures=invalid_open_pressure,
    )

    assert fits[0] > fits[1] + 7.0


def test_compute_fitness_penalizes_spiky_max_saturation(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([[3.0, 3.0, 3.0, 3.0],
                            [3.0, 3.0, 3.0, 3.0]], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.full_like(period_rets, 0.03)
    saturation_rates = np.array([[0.25, 0.25, 0.25, 0.25],
                                 [1.00, 0.00, 0.00, 0.00]], dtype=np.float64)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    monkeypatch.setattr(cg, "TRADE_REWARD_W", 0.0)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_MAX_PENALTY_W", 8.0, raising=False)

    fits = cg._compute_fitness(
        period_rets,
        period_dds,
        trade_rates,
        period_saturation_rates=saturation_rates,
    )

    assert fits[0] > fits[1] + 5.0


def test_compute_fitness_penalizes_spiky_invalid_open_pressure(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([[3.0, 3.0, 3.0, 3.0],
                            [3.0, 3.0, 3.0, 3.0]], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.full_like(period_rets, 0.03)
    invalid_open_pressure = np.array([[0.20, 0.20, 0.20, 0.20],
                                      [0.80, 0.00, 0.00, 0.00]], dtype=np.float64)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    monkeypatch.setattr(cg, "TRADE_REWARD_W", 0.0)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_MAX_PENALTY_W", 6.0, raising=False)

    fits = cg._compute_fitness(
        period_rets,
        period_dds,
        trade_rates,
        period_invalid_open_logit_pressures=invalid_open_pressure,
    )

    assert fits[0] > fits[1] + 3.0


def test_compute_fitness_prefers_executable_policy_over_high_return_invalid_policy(monkeypatch):
    cg = _load_crypto_genetics()
    period_rets = np.array([[1.0, 1.0, 1.0, 1.0],
                            [8.0, 8.0, 8.0, 8.0]], dtype=np.float64)
    period_dds = np.zeros_like(period_rets)
    trade_rates = np.full_like(period_rets, 0.03)
    saturation_rates = np.array([[0.02, 0.02, 0.02, 0.02],
                                 [0.70, 0.70, 0.70, 0.70]], dtype=np.float64)
    invalid_open_pressure = np.array([[0.00, 0.00, 0.00, 0.00],
                                      [0.80, 0.80, 0.80, 0.80]], dtype=np.float64)

    monkeypatch.setattr(cg, "ROBUST_FITNESS_ENABLED", False)
    monkeypatch.setattr(cg, "TRADE_REWARD_W", 0.0)
    monkeypatch.setattr(cg, "TURNOVER_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "MAX_POSITION_SATURATION_MAX_PENALTY_W", 0.0, raising=False)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_PENALTY_W", 0.0)
    monkeypatch.setattr(cg, "INVALID_OPEN_LOGIT_MAX_PENALTY_W", 0.0, raising=False)
    monkeypatch.setattr(cg, "ACTION_FEASIBILITY_PENALTY_W", 60.0, raising=False)
    monkeypatch.setattr(cg, "ACTION_FEASIBILITY_SATURATION_TARGET", 0.25, raising=False)
    monkeypatch.setattr(cg, "ACTION_FEASIBILITY_INVALID_OPEN_TARGET", 0.05, raising=False)

    fits = cg._compute_fitness(
        period_rets,
        period_dds,
        trade_rates,
        period_saturation_rates=saturation_rates,
        period_invalid_open_logit_pressures=invalid_open_pressure,
    )

    assert fits[0] > fits[1]


def test_island_regime_genomes_require_positive_contract_fitness(tmp_path, monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "AGENTS_DIR", str(tmp_path))
    monkeypatch.setattr(cg, "REGIME_GENOME_MIN_FITNESS", 0.0, raising=False)

    trainer = object.__new__(cg.GeneticTrainer)
    trainer.gen = 1
    trainer.islands = [[0], [1], [2]]
    trainer.pop = np.zeros((3, cg.GENOME_SIZE), dtype=np.float32)
    trainer._island_best_fits = {}
    trainer._island_best_genomes = {}

    trainer._save_island_bests(np.array([-10.0, -0.5, 0.25], dtype=np.float64))

    assert not (tmp_path / "best_genome_regime_bearish.npy").exists()
    assert not (tmp_path / "best_genome_regime_neutral.npy").exists()
    assert (tmp_path / "best_genome_regime_bullish.npy").exists()


def test_per_regime_genomes_require_positive_contract_fitness(tmp_path, monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "AGENTS_DIR", str(tmp_path))
    monkeypatch.setattr(cg, "REGIME_GENOME_MIN_FITNESS", 0.0, raising=False)

    trainer = object.__new__(cg.GeneticTrainer)
    trainer.gen = 1
    trainer.best_g = np.zeros(cg.GENOME_SIZE, dtype=np.float32)
    trainer.precomp = [(None, None, None, None, None, 1.0, "bullish")]
    trainer.best_per_regime = {}
    trainer._island_best_fits = {}
    trainer._island_best_genomes = {}

    trainer._save_best_per_regime(
        [5.0],
        genome=np.ones(cg.GENOME_SIZE, dtype=np.float32),
        candidate_fitness=-1.0,
    )

    assert not (tmp_path / "best_genome_regime_bullish.npy").exists()

    trainer._save_best_per_regime(
        [5.0],
        genome=np.ones(cg.GENOME_SIZE, dtype=np.float32),
        candidate_fitness=0.25,
    )

    assert (tmp_path / "best_genome_regime_bullish.npy").exists()


def test_de_mutant_handles_single_member_island_without_crashing():
    cg = _load_crypto_genetics()
    trainer = object.__new__(cg.GeneticTrainer)
    trainer.rng = np.random.default_rng(123)
    trainer.stagnation = 0

    best = np.full(cg.GENOME_SIZE, 0.25, dtype=np.float32)
    island_pop = np.full((1, cg.GENOME_SIZE), -0.50, dtype=np.float32)
    island_fits = np.array([1.0], dtype=np.float64)

    mutant = trainer._de_mutant(island_fits, island_pop, best)

    assert mutant.dtype == np.float32
    np.testing.assert_allclose(mutant, best)


def test_contract_evaluator_reports_action_contract_metrics(monkeypatch):
    tool = _load_contract_eval_tool()
    cg = tool.cg
    genome = np.zeros(cg.GENOME_SIZE, dtype=np.float32)

    def fake_evaluate_population(population, precomp):
        assert population.shape == (1, cg.GENOME_SIZE)
        assert cg.POSITION_STATE_FEATURES_ENABLED is False
        return np.array([1.25], dtype=np.float64), [[0.5, -0.1]]

    def fake_contract_metrics(*, genome, precomp, execution_lag_bars):
        assert execution_lag_bars == 1
        assert cg.POSITION_STATE_FEATURES_ENABLED is False
        return {
            "mean_turnover_rate": 0.12,
            "mean_saturation_rate": 0.03,
            "max_saturation_rate": 0.04,
            "mean_invalid_open_logit_pressure": 0.21,
            "max_invalid_open_logit_pressure": 0.25,
            "periods": [{
                "period": "2025-01",
                "turnover_rate": 0.12,
                "saturation_rate": 0.03,
                "invalid_open_logit_pressure": 0.21,
            }],
        }

    monkeypatch.setattr(tool, "_evaluate_population", fake_evaluate_population)
    monkeypatch.setattr(tool, "_contract_metrics_for_genome", fake_contract_metrics, raising=False)

    report = tool._evaluate_mode(
        genome=genome,
        precomp=[],
        mode_name="fee_fixed_nextbar",
        execution_lag_bars=1,
        futures_fee=0.0002,
        position_state_features_enabled=False,
    )

    assert report["contract_metrics"]["mean_turnover_rate"] == 0.12
    assert report["contract_metrics"]["mean_saturation_rate"] == 0.03
    assert report["contract_metrics"]["max_saturation_rate"] == 0.04
    assert report["contract_metrics"]["mean_invalid_open_logit_pressure"] == 0.21
    assert report["contract_metrics"]["max_invalid_open_logit_pressure"] == 0.25
    assert report["position_state_features_enabled"] is False


def test_contract_evaluator_uses_crash_aware_regime_labels():
    tool = _load_contract_eval_tool()

    bearish_crash = (None, None, None, 1, "2024-04", 2.5, "bearish", -27.5)
    bearish_non_crash = (None, None, None, 2, "2024-06", 2.5, "bearish", -9.0)
    raw_crash = (None, None, None, 3, "2022-06", 2.5, "crash")

    assert tool._contract_regime_label(bearish_crash) == "crash"
    assert tool._contract_regime_label(bearish_non_crash) == "bearish"
    assert tool._contract_regime_label(raw_crash) == "crash"


def test_contract_evaluator_infers_position_state_flag_from_genome_metadata(tmp_path):
    tool = _load_contract_eval_tool()

    genome_path = tmp_path / "archive_rank1.npy"
    np.save(genome_path, np.zeros(tool.cg.GENOME_SIZE, dtype=np.float32))
    (tmp_path / "best_genome_meta.json").write_text(
        '{"position_state_features_enabled": true}',
        encoding="utf-8",
    )

    assert tool._infer_position_state_features_enabled(genome_path) is True
    assert tool._infer_position_state_features_enabled(tmp_path / "missing.npy") is False


def test_contract_evaluator_blocks_requested_retrodate_year_mismatch(tmp_path):
    tool = _load_contract_eval_tool()
    _write_retrodate_csv(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        timestamp=1640995200000,
        dt="2022-01-01 00:00:00+00:00",
    )

    with pytest.raises(tool.RetrodateValidationError) as exc:
        tool._validate_requested_retrodate_files(
            tmp_path,
            timeframe="1m",
            start_date="2026-01-01",
            end_date="2026-02-28",
        )

    message = str(exc.value)
    assert "crypto_1m_2026_all_symbols.csv" in message
    assert "year_mismatch" in message


def test_contract_evaluator_ignores_unrequested_invalid_retrodate_year(tmp_path):
    tool = _load_contract_eval_tool()
    _write_retrodate_csv(
        tmp_path / "crypto_1m_2025_all_symbols.csv",
        timestamp=1735689600000,
        dt="2025-01-01 00:00:00+00:00",
    )
    _write_retrodate_csv(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        timestamp=1640995200000,
        dt="2022-01-01 00:00:00+00:00",
    )

    selection = tool._validate_requested_retrodate_files(
        tmp_path,
        timeframe="1m",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    assert [item.path.name for item in selection.valid_reports] == [
        "crypto_1m_2025_all_symbols.csv"
    ]
    assert selection.missing_years == ()


def test_contract_evaluator_preflight_avoids_full_retrodate_directory_scan(tmp_path, monkeypatch):
    tool = _load_contract_eval_tool()
    _write_retrodate_csv(
        tmp_path / "crypto_1m_2025_all_symbols.csv",
        timestamp=1735689600000,
        dt="2025-01-01 00:00:00+00:00",
    )
    _write_retrodate_csv(
        tmp_path / "crypto_1m_2026_all_symbols.csv",
        timestamp=1640995200000,
        dt="2022-01-01 00:00:00+00:00",
    )

    def fail_full_scan(_path):
        raise AssertionError("genetics evaluator preflight must not full-scan Retrodate directory")

    monkeypatch.setattr(tool, "validate_retrodate_dir", fail_full_scan)

    selection = tool._validate_requested_retrodate_files(
        tmp_path,
        timeframe="1m",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )

    assert [item.path.name for item in selection.valid_reports] == [
        "crypto_1m_2025_all_symbols.csv"
    ]


def test_contract_evaluator_can_override_data_dir(tmp_path, monkeypatch):
    tool = _load_contract_eval_tool()
    default_dir = tmp_path / "default"
    override_dir = tmp_path / "override"
    default_dir.mkdir()
    override_dir.mkdir()

    monkeypatch.setattr(tool.cg, "_resolve_data_dir", lambda: str(default_dir))

    assert tool._resolve_eval_data_dir(None) == default_dir
    assert tool._resolve_eval_data_dir(str(override_dir)) == override_dir


def test_contract_metrics_report_effective_turnover_after_position_policy(monkeypatch):
    tool = _load_contract_eval_tool()
    cg = tool.cg
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 1)
    monkeypatch.setattr(cg, "CURRENCY_SELECTION_ENABLED", False)
    monkeypatch.setattr(cg, "POSITION_STATE_FEATURES_ENABLED", False)

    feat = np.zeros((3, 2, cg.N_INPUT), dtype=np.float32)
    precomp = [(feat, np.ones((3, 2), dtype=np.float64), ["BTC", "ETH"], 1, "2025-01", 1.0, "bullish")]

    def fake_forward(population, features, prices=None):
        assert features is feat
        assert prices is precomp[0][1]
        return np.array([[
            [5, 5],
            [5, 5],
            [8, 5],
        ]], dtype=np.int8)

    monkeypatch.setattr(cg.GeneticTrainer, "_batch_forward_numpy", staticmethod(fake_forward))

    metrics = tool._contract_metrics_for_genome(
        genome=np.zeros(cg.GENOME_SIZE, dtype=np.float32),
        precomp=precomp,
        execution_lag_bars=1,
    )

    assert metrics["mean_turnover_rate"] == 5 / 6
    assert metrics["mean_saturation_rate"] == 3 / 6
    assert metrics["mean_effective_turnover_rate"] == 2 / 6
    assert metrics["mean_invalid_open_logit_pressure"] == 0.0


def test_smoke_training_feasible_best_seed_biases_open_action_logits():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    cg = tool.cg
    best = np.zeros(cg.GENOME_SIZE, dtype=np.float32)
    b4_start = cg.GENOME_SIZE - cg.N_ACTIONS
    best[b4_start:b4_start + cg.N_ACTIONS] = 1.0
    population = np.full((5, cg.GENOME_SIZE), 99.0, dtype=np.float32)

    inserted = tool._seed_feasible_best_population(
        population,
        best,
        np.random.default_rng(7),
        start_slot=1,
        seed_frac=0.4,
        sigma=0.0,
        open_logit_bias=2.5,
    )

    open_actions = [code for code in cg._OPEN_ACTION_CODES if code < cg.N_ACTIONS]
    non_open_actions = [code for code in range(cg.N_ACTIONS) if code not in open_actions]
    assert inserted == 2
    np.testing.assert_allclose(population[1:3, b4_start + np.array(open_actions)], -1.5)
    np.testing.assert_allclose(population[1:3, b4_start + np.array(non_open_actions)], 1.0)
    assert np.all(population[3:] == 99.0)


def test_smoke_training_can_seed_output_layer_only_mutants():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    cg = tool.cg
    best = np.linspace(-0.25, 0.25, cg.GENOME_SIZE, dtype=np.float32)
    population = np.full((4, cg.GENOME_SIZE), 99.0, dtype=np.float32)
    output_start = cg.GENOME_SIZE - (cg.N_HIDDEN3 * cg.N_ACTIONS + cg.N_ACTIONS)

    inserted = tool._seed_feasible_best_population(
        population,
        best,
        np.random.default_rng(13),
        start_slot=1,
        seed_frac=0.75,
        sigma=0.02,
        open_logit_bias=0.0,
        mutation_scope="output",
    )

    assert inserted == 3
    expected_prefix = np.broadcast_to(best[:output_start], (3, output_start))
    np.testing.assert_allclose(population[1:, :output_start], expected_prefix)
    assert np.max(np.abs(population[1:, output_start:] - best[output_start:])) > 0.0


def test_smoke_training_staged_mutation_schedule_moves_from_explore_to_calibration():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")

    early = tool.build_staged_mutation_schedule(
        generation_index=0,
        total_generations=10,
        base_sigma=0.10,
        base_mutation_rate=0.20,
    )
    mid = tool.build_staged_mutation_schedule(
        generation_index=5,
        total_generations=10,
        base_sigma=0.10,
        base_mutation_rate=0.20,
    )
    late = tool.build_staged_mutation_schedule(
        generation_index=9,
        total_generations=10,
        base_sigma=0.10,
        base_mutation_rate=0.20,
    )

    assert early["stage"] == "explore"
    assert early["mutation_scope"] == "all"
    assert early["mutation_sigma"] > mid["mutation_sigma"] > late["mutation_sigma"]
    assert early["mutation_rate"] > mid["mutation_rate"] >= late["mutation_rate"]
    assert late["stage"] == "output_calibration"
    assert late["mutation_scope"] == "output"
    assert late["output_layer_only"] is True


def test_smoke_training_staged_schedule_preserves_regime_elites_and_diversity_penalty():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")

    schedule = tool.build_staged_mutation_schedule(
        generation_index=2,
        total_generations=12,
        base_sigma=0.25,
        base_mutation_rate=0.20,
        diversity_penalty_weight=0.075,
    )

    assert schedule["preserve_regime_elites"] is True
    assert schedule["diversity_penalty_enabled"] is True
    assert schedule["diversity_penalty_weight"] == pytest.approx(0.075)
    assert schedule["full_genome_mutation_primary"] is False


def test_smoke_training_staged_schedule_late_stage_restricts_mutation_to_output_layer():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    cg = tool.cg
    output_start = tool._output_layer_start_index()

    class FakeTrainer:
        def __init__(self):
            self.gen = 10
            self.sigma = 0.25

        def _mut(self, genome):
            return np.asarray(genome, dtype=np.float32) + 1.0

        def _mut_layerwise(self, genome):
            return np.asarray(genome, dtype=np.float32) + 2.0

        def _local_mutant(self, genome):
            return np.asarray(genome, dtype=np.float32) + 3.0

        def _cma_mutant(self, genome):
            return np.asarray(genome, dtype=np.float32) + 4.0

        def _levy_mutant(self, genome):
            return np.asarray(genome, dtype=np.float32) + 5.0

        def _next(self, _fits):
            base = np.zeros(cg.GENOME_SIZE, dtype=np.float32)
            self.mutated = self._mut(base)
            self.layerwise = self._mut_layerwise(base)
            self.local = self._local_mutant(base)
            self.cma = self._cma_mutant(base)
            self.levy = self._levy_mutant(base)
            return self.mutated[None]

    trainer = FakeTrainer()

    tool._install_staged_mutation_schedule(
        trainer,
        total_generations=10,
        base_sigma=0.25,
        base_mutation_rate=0.20,
    )
    trainer._next(np.array([1.0], dtype=np.float64))

    for mutated in (trainer.mutated, trainer.layerwise, trainer.local, trainer.cma, trainer.levy):
        np.testing.assert_allclose(mutated[:output_start], 0.0)
        assert float(np.max(mutated[output_start:])) > 0.0
    assert trainer.staged_mutation_history[-1]["stage"] == "output_calibration"


def test_smoke_training_staged_schedule_maps_last_reproductive_step_to_calibration():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")

    class FakeTrainer:
        def __init__(self):
            self.gen = 2
            self.sigma = 0.25

        def _mut(self, genome):
            return genome

        def _next(self, _fits):
            return np.zeros((1, tool.cg.GENOME_SIZE), dtype=np.float32)

    trainer = FakeTrainer()

    tool._install_staged_mutation_schedule(
        trainer,
        total_generations=3,
        base_sigma=0.25,
        base_mutation_rate=0.20,
        early_frac=0.34,
        late_frac=0.34,
    )
    trainer._next(np.array([1.0], dtype=np.float64))

    assert trainer.staged_mutation_history[-1]["stage"] == "output_calibration"
    assert trainer.staged_mutation_history[-1]["training_generations"] == 3


def test_smoke_training_staged_output_calibration_freezes_direct_noise_hidden_trunk():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    cg = tool.cg
    output_start = tool._output_layer_start_index()

    class FakeTrainer:
        def __init__(self):
            self.gen = 2
            self.sigma = 0.25
            self.pop = np.zeros((3, cg.GENOME_SIZE), dtype=np.float32)
            self.pop[0, :output_start] = 7.0
            self.pop[1, :output_start] = 3.0

        def _mut(self, genome):
            return genome

        def _next(self, _fits):
            nxt = np.ones_like(self.pop)
            nxt[:, output_start:] = 11.0
            return nxt

    trainer = FakeTrainer()

    tool._install_staged_mutation_schedule(
        trainer,
        total_generations=3,
        base_sigma=0.25,
        base_mutation_rate=0.20,
        early_frac=0.34,
        late_frac=0.34,
    )
    nxt = trainer._next(np.array([2.0, 1.0, 0.0], dtype=np.float64))

    np.testing.assert_allclose(nxt[:, :output_start], 7.0)
    np.testing.assert_allclose(nxt[:, output_start:], 11.0)


def test_smoke_training_position_state_mode_uses_source_metadata(tmp_path):
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    (tmp_path / "best_genome_meta.json").write_text(
        '{"position_state_features_enabled": false}',
        encoding="utf-8",
    )

    assert tool._resolve_position_state_features_enabled(
        "source-meta",
        tmp_path,
        settings_default=True,
    ) is False


def test_smoke_training_position_state_mode_can_force_on_off(tmp_path):
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    (tmp_path / "best_genome_meta.json").write_text(
        '{"position_state_features_enabled": false}',
        encoding="utf-8",
    )

    assert tool._resolve_position_state_features_enabled(
        "on",
        tmp_path,
        settings_default=False,
    ) is True
    assert tool._resolve_position_state_features_enabled(
        "off",
        tmp_path,
        settings_default=True,
    ) is False
    assert tool._resolve_position_state_features_enabled(
        "settings",
        tmp_path,
        settings_default=True,
    ) is True


def test_smoke_training_can_filter_precomp_by_market_regime():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    precomp = [
        ("feat", "prices", ["BTC"], 1, "2025-01", 1.0, "bullish"),
        ("feat", "prices", ["BTC"], 2, "2025-02", 2.5, "bearish"),
        ("feat", "prices", ["BTC"], 3, "2025-03", 1.5, "neutral"),
        ("feat", "prices", ["BTC"], 4, "2025-04", 2.5, "crash"),
    ]

    bearish = tool._filter_precomp_by_regime(precomp, "bearish")
    all_periods = tool._filter_precomp_by_regime(precomp, "all")

    assert [row[4] for row in bearish] == ["2025-02", "2025-04"]
    assert all_periods == precomp


def test_behavior_cloning_seed_names_use_curated_list_without_auto_discovery(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(cg, "AGENT_SEED_LIST", ["MomentumScalper", "LiveOIBreakout"])
    monkeypatch.setattr(cg, "BC_AUTO_DISCOVERY", False, raising=False)

    def fail_discovery():
        raise AssertionError("auto discovery should stay disabled for curated BC")

    monkeypatch.setattr(cg, "_discover_agent_classes", fail_discovery)

    assert cg._agent_seed_names() == ["MomentumScalper", "LiveOIBreakout"]


def test_behavior_cloning_seed_names_expand_curated_panteon_actor_aliases(monkeypatch):
    cg = _load_crypto_genetics()
    monkeypatch.setattr(
        cg,
        "AGENT_SEED_LIST",
        [
            "Antonius_conservative",
            "Solo_MomentumScalper",
            "ResearchValidatorAgent",
            "LiveOIBreakout",
            "VolBreakoutHunter",
        ],
    )
    monkeypatch.setattr(cg, "BC_AUTO_DISCOVERY", False, raising=False)

    assert cg._agent_seed_names() == [
        "VolBreakoutHunter",
        "FundingArb",
        "ResearchValidatorAgent",
        "CrashPanicShortAgent",
        "MomentumScalper",
        "LiveOIBreakout",
    ]


def test_behavior_cloning_resolves_panteon_runtime_agent_classes(monkeypatch):
    import sys
    import types

    cg = _load_crypto_genetics()
    fake_parent = types.ModuleType("panteon_runtime")
    fake_parent.__path__ = []
    fake_module = types.ModuleType("panteon_runtime.panteon_agents")

    class LiveOIBreakout:
        def act(self, market):
            return {}

    LiveOIBreakout.__module__ = fake_module.__name__
    fake_module.LiveOIBreakout = LiveOIBreakout
    monkeypatch.setitem(sys.modules, "panteon_runtime", fake_parent)
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    assert cg._resolve_bc_agent_class("LiveOIBreakout") is LiveOIBreakout


def test_behavior_cloning_bootstrap_exposes_real_panteon_agents():
    cg = _load_crypto_genetics()
    root = Path(__file__).resolve().parents[3]
    runtime_dir = root / "src" / "panteon_runtime"

    assert str(runtime_dir) in sys.path
    resolved = cg._resolve_bc_agent_class("MomentumScalper")
    assert resolved is not None
    assert resolved.__name__ == "MomentumScalper"


@pytest.mark.skipif(os.name != "nt", reason="Windows evaluator branch")
def test_cpu_evaluator_uses_thread_pool_for_large_windows_precomp():
    cg = _load_crypto_genetics()
    ev = cg.CPUEvaluator([], 2)
    try:
        assert ev._ex is not None
        assert ev._pkl is None
    finally:
        ev.shutdown()


def test_genetic_settings_can_load_from_env_path(tmp_path, monkeypatch):
    cg = _load_crypto_genetics()
    settings_path = tmp_path / "settings_genetic_wide.txt"
    settings_path.write_text(
        "n_hidden1 = 192\n"
        "n_hidden2 = 96\n"
        "n_hidden3 = 48\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GENETICS_SETTINGS_FILE", str(settings_path))

    loaded = cg._load_genetic_settings()

    assert loaded["n_hidden1"] == "192"
    assert loaded["n_hidden2"] == "96"
    assert loaded["n_hidden3"] == "48"
    assert "pop_size" in loaded


def test_training_window_prefers_genetics_env_over_exchange_config(monkeypatch):
    cg = _load_crypto_genetics()

    monkeypatch.setitem(cg._GS, "train_start_date", "2021-01-01")
    monkeypatch.setitem(cg._GS, "train_end_date", "2021-12-31")
    cfg = {"start_date": "2018-01-01", "end_date": "2025-12-31"}

    assert cg._training_window_from_config(cfg) == ("2021-01-01", "2021-12-31")

    monkeypatch.setenv("GENETICS_TRAIN_START_DATE", "2022-01-01")
    monkeypatch.setenv("GENETICS_TRAIN_END_DATE", "2023-12-31")

    assert cg._training_window_from_config(cfg) == ("2022-01-01", "2023-12-31")


def test_cpu_batch_eval_applies_regime_weight_override(monkeypatch):
    cg = _load_crypto_genetics()
    trainer = object.__new__(cg.GeneticTrainer)
    trainer.mode = "cpu"

    monkeypatch.setattr(cg, "POSITION_STATE_FEATURES_ENABLED", False)
    monkeypatch.setattr(cg, "CURRENCY_SELECTION_ENABLED", False)
    monkeypatch.setattr(cg, "TRAIN_MAX_POS", 2)

    def fake_forward(population, feat, prices=None):
        return np.zeros((len(population), feat.shape[0], feat.shape[1]), dtype=np.int8)

    captured = {}

    def fake_simulate(actions, prices, initial_capital, snap_every):
        g = actions.shape[0]
        return (
            np.ones(g, dtype=np.float64),
            np.zeros(g, dtype=np.float64),
            np.zeros(g, dtype=np.float64),
            None,
        )

    def fake_compute(ret_mat, dd_mat, tr_mat, dpv, rw_arr, sat_mat, pressure_mat):
        captured["rw"] = rw_arr.copy()
        return np.zeros(ret_mat.shape[0], dtype=np.float64)

    monkeypatch.setattr(cg.GeneticTrainer, "_batch_forward_numpy", staticmethod(fake_forward))
    monkeypatch.setattr(cg, "simulate_batch", fake_simulate)
    monkeypatch.setattr(cg, "_compute_fitness", fake_compute)

    population = np.zeros((2, cg.GENOME_SIZE), dtype=np.float32)
    feat = np.zeros((2, 1, cg.N_INPUT), dtype=np.float32)
    prices = np.ones((2, 1), dtype=np.float64)
    precomp = [
        (feat, prices, ["BTC"], 1, "2022-01", 1.0, "bearish"),
        (feat, prices, ["BTC"], 2, "2022-02", 1.0, "bullish"),
    ]

    trainer._eval_island_batch(
        population,
        precomp,
        rw_override={"bearish": 3.0, "bullish": 0.5},
    )

    np.testing.assert_allclose(captured["rw"], np.array([3.0, 0.5]))


def _test_mlp_forward(genome: np.ndarray, arch: tuple[int, int, int, int, int],
                      x: np.ndarray) -> np.ndarray:
    n_in, h1, h2, h3, n_actions = arch
    i = 0
    w1 = genome[i:i + n_in * h1].reshape(n_in, h1); i += n_in * h1
    b1 = genome[i:i + h1]; i += h1
    w2 = genome[i:i + h1 * h2].reshape(h1, h2); i += h1 * h2
    b2 = genome[i:i + h2]; i += h2
    w3 = genome[i:i + h2 * h3].reshape(h2, h3); i += h2 * h3
    b3 = genome[i:i + h3]; i += h3
    w4 = genome[i:i + h3 * n_actions].reshape(h3, n_actions); i += h3 * n_actions
    b4 = genome[i:i + n_actions]

    def elu(v):
        return np.where(v >= 0, v, np.exp(v) - 1.0)

    return elu(elu(elu(x @ w1 + b1) @ w2 + b2) @ w3 + b3) @ w4 + b4


def test_smoke_training_transfers_genome_to_wider_architecture():
    import importlib

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    source_arch = (3, 4, 3, 2, 2)
    target_arch = (3, 6, 5, 4, 2)
    rng = np.random.default_rng(11)
    source = rng.normal(0.0, 0.2, tool._genome_size_for_arch(source_arch)).astype(np.float32)

    transferred = tool._transfer_mlp_genome(source, source_arch, target_arch)

    assert transferred.shape == (tool._genome_size_for_arch(target_arch),)
    x = rng.normal(0.0, 1.0, (5, source_arch[0])).astype(np.float32)
    np.testing.assert_allclose(
        _test_mlp_forward(source, source_arch, x),
        _test_mlp_forward(transferred, target_arch, x),
        atol=1e-6,
    )


def test_contract_evaluator_reports_directional_exposure_metrics():
    import importlib

    tool = importlib.import_module("tools.evaluate_genetics_contract")
    actions = np.zeros((1, 4, 2), dtype=np.int32)
    actions[0, 0, 0] = 4  # open futures long on symbol 0
    actions[0, 0, 1] = 6  # open futures short on symbol 1
    actions[0, 2, 0] = 8  # close futures long
    actions[0, 3, 1] = 8  # close futures short

    metrics = tool._position_exposure_metrics(actions, max_positions=4)

    assert metrics["mean_long_slot_rate"] == pytest.approx(2 / 8)
    assert metrics["mean_short_slot_rate"] == pytest.approx(3 / 8)
    assert metrics["mean_position_slot_rate"] == pytest.approx(5 / 8)
    assert metrics["max_open_positions"] == 2
    assert metrics["mean_net_direction_bias"] == pytest.approx((2 - 3) / 5)

    by_symbol = tool._position_exposure_metrics(
        actions,
        max_positions=4,
        symbols=["BTC", "ETH"],
        top_n_symbols=2,
    )

    assert by_symbol["top_position_symbols"][0]["symbol"] == "ETH"
    assert by_symbol["top_position_symbols"][0]["position_slot_rate"] == pytest.approx(3 / 4)
    assert by_symbol["top_position_symbols"][0]["short_slot_rate"] == pytest.approx(3 / 4)
    assert by_symbol["top_position_symbols"][1]["symbol"] == "BTC"
    assert by_symbol["top_position_symbols"][1]["position_slot_rate"] == pytest.approx(2 / 4)
    assert by_symbol["top_position_symbols"][1]["long_slot_rate"] == pytest.approx(2 / 4)


def test_smoke_training_writes_warm_start_artifacts(tmp_path):
    import importlib
    import json

    tool = importlib.import_module("tools.run_genetics_smoke_training")
    genome = np.arange(6, dtype=np.float32)

    written = tool._write_warm_start_artifacts(
        tmp_path,
        genome,
        {
            "source_arch": {"in": 1, "h1": 2, "h2": 1, "h3": 1, "out": 1},
            "target_arch": {"in": 1, "h1": 3, "h2": 2, "h3": 1, "out": 1},
            "source_genome_transferred": True,
        },
    )

    np.testing.assert_allclose(np.load(written["genome"]), genome)
    meta = json.loads(written["meta"].read_text(encoding="utf-8"))
    assert meta["role"] == "warm_start_baseline"
    assert meta["source_genome_transferred"] is True
    assert meta["genome_size"] == 6


def test_output_calibration_changes_only_open_action_output_biases():
    import importlib

    tool = importlib.import_module("tools.calibrate_genetics_output_thresholds")
    cg = tool.cg
    genome = np.linspace(-0.25, 0.25, cg.GENOME_SIZE, dtype=np.float32)
    calibrated = tool.calibrate_open_action_bias(genome, open_bias=0.75)

    changed = np.flatnonzero(np.abs(calibrated - genome) > 1e-7)
    expected = tool.open_output_bias_indices()

    np.testing.assert_array_equal(changed, expected)
    np.testing.assert_allclose(calibrated[expected], genome[expected] - 0.75)


def test_output_calibration_writes_manifest_only_under_neiro_genetics(tmp_path):
    import importlib

    tool = importlib.import_module("tools.calibrate_genetics_output_thresholds")
    genome = np.zeros(tool.cg.GENOME_SIZE, dtype=np.float32)
    source = tmp_path / "source.npy"
    np.save(source, genome)

    outside = tmp_path / "outside_results"
    with pytest.raises(ValueError):
        tool.write_calibrated_variants(
            source_genome=source,
            out_dir=outside,
            open_biases=[0.25],
        )

    out_dir = Path("Results") / "neiro_genetics" / "calibration_test"
    result = tool.write_calibrated_variants(
        source_genome=source,
        out_dir=out_dir,
        open_biases=[0.25, 0.50],
    )

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["promotion_allowed"] is False
    assert manifest["live_trading_eligible"] is False
    assert manifest["requires_validation"] is True
    assert len(manifest["variants"]) == 2
    assert all("Results\\neiro_genetics" in item["path"] for item in manifest["variants"])


def test_behavior_cloning_flash_seed_alias_expands_to_best_components():
    cg = _load_crypto_genetics()

    expanded = cg._expand_bc_agent_seed_name("Panteon_Flash")

    assert "MomentumScalper" in expanded
    assert "LiveOIBreakout" in expanded
    assert "VolBreakoutHunter" in expanded
    assert "FundingArb" in expanded
    assert "ResearchValidatorAgent" in expanded
    assert "CrashPanicShortAgent" in expanded


def _genetics_selection_report(*, baseline_mean: float, candidate_mean: float,
                              baseline_min: float = 0.0, candidate_min: float = 0.0,
                              candidate_turnover: float = 0.01) -> dict:
    def genome(path: str, mean: float, min_ret: float, turnover: float) -> dict:
        return {
            "path": path,
            "position_state_features_enabled": False,
            "modes": [{
                "mode": "fee_fixed_nextbar",
                "fitness": mean,
                "period_stats": {
                    "mean_ret": mean,
                    "min_ret": min_ret,
                    "max_ret": mean,
                    "positive_period_pct": 100.0 if min_ret >= 0 else 66.67,
                },
                "robust_score": {
                    "max_positive_contribution_pct": 30.0,
                    "passes_default_gates": True,
                },
                "contract_metrics": {
                    "mean_turnover_rate": turnover,
                    "mean_saturation_rate": 0.01,
                    "mean_invalid_open_logit_pressure": 0.0,
                },
                "period_rets": [mean],
            }],
        }

    return {
        "genomes": [
            genome("baseline.npy", baseline_mean, baseline_min, 0.01),
            genome("candidate.npy", candidate_mean, candidate_min, candidate_turnover),
        ]
    }


def _genetics_regime_selection_report(
    *,
    baseline_rets: list[float],
    candidate_rets: list[float],
    regimes: list[str],
    baseline_turnovers: list[float] | None = None,
    candidate_turnovers: list[float] | None = None,
    baseline_saturations: list[float] | None = None,
    candidate_saturations: list[float] | None = None,
) -> dict:
    baseline_turnovers = baseline_turnovers or [0.01] * len(regimes)
    candidate_turnovers = candidate_turnovers or [0.01] * len(regimes)
    baseline_saturations = baseline_saturations or [0.01] * len(regimes)
    candidate_saturations = candidate_saturations or [0.01] * len(regimes)

    def genome(path: str, rets: list[float], turnovers: list[float], saturations: list[float]) -> dict:
        return {
            "path": path,
            "position_state_features_enabled": False,
            "modes": [{
                "mode": "fee_fixed_nextbar",
                "fitness": float(np.mean(rets)),
                "period_stats": {
                    "mean_ret": float(np.mean(rets)),
                    "min_ret": float(np.min(rets)),
                    "max_ret": float(np.max(rets)),
                    "positive_period_pct": float(np.mean(np.asarray(rets) > 0.0) * 100.0),
                },
                "robust_score": {
                    "max_positive_contribution_pct": 30.0,
                    "passes_default_gates": True,
                },
                "contract_metrics": {
                    "mean_turnover_rate": float(np.mean(turnovers)),
                    "mean_saturation_rate": float(np.mean(saturations)),
                    "mean_invalid_open_logit_pressure": 0.0,
                    "periods": [
                        {
                            "period": f"p{i}",
                            "regime": regime,
                            "turnover_rate": float(turnovers[i]),
                            "saturation_rate": float(saturations[i]),
                            "invalid_open_logit_pressure": 0.0,
                        }
                        for i, regime in enumerate(regimes)
                    ],
                },
                "period_rets": list(rets),
            }],
        }

    return {
        "genomes": [
            genome("baseline.npy", baseline_rets, baseline_turnovers, baseline_saturations),
            genome("candidate.npy", candidate_rets, candidate_turnovers, candidate_saturations),
        ]
    }


def test_genetics_candidate_selector_keeps_baseline_when_validation_loses():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_selection_report(baseline_mean=0.2, candidate_mean=0.5)
    validation = _genetics_selection_report(
        baseline_mean=1.0,
        candidate_mean=-0.2,
        baseline_min=0.5,
        candidate_min=-1.0,
    )

    selected = selector.select_candidate(train, validation)

    assert selected["selected_is_baseline"] is True
    assert selected["selected_path"] == "baseline.npy"
    assert selected["candidates"][0]["accepted"] is False
    assert "validation_mean_ret" in selected["candidates"][0]["failures"]


def test_genetics_candidate_selector_keeps_baseline_on_validation_tie():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_selection_report(baseline_mean=0.2, candidate_mean=0.5)
    validation = _genetics_selection_report(
        baseline_mean=1.0,
        candidate_mean=1.0,
        baseline_min=0.5,
        candidate_min=0.5,
    )

    selected = selector.select_candidate(train, validation)

    assert selected["selected_is_baseline"] is True
    assert selected["selected_path"] == "baseline.npy"
    assert selected["candidates"][0]["accepted"] is False
    assert "validation_tie" in selected["candidates"][0]["failures"]


def test_genetics_candidate_selector_accepts_validation_winner():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_selection_report(baseline_mean=0.2, candidate_mean=0.5)
    validation = _genetics_selection_report(
        baseline_mean=0.1,
        candidate_mean=0.4,
        baseline_min=0.0,
        candidate_min=0.2,
    )

    selected = selector.select_candidate(
        train,
        validation,
        min_validation_mean_delta=0.05,
        min_validation_min_ret_delta=0.0,
        min_positive_period_pct=100.0,
    )

    assert selected["selected_is_baseline"] is False
    assert selected["selected_path"] == "candidate.npy"
    assert selected["candidates"][0]["accepted"] is True


def test_genetics_candidate_selector_fitness_v3_rejects_costly_validation_winner():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")

    def report(candidate_cost: float) -> dict:
        regimes = ["crash", "bearish", "neutral", "bullish"]

        def genome(path: str, rets: list[float], cost: float) -> dict:
            return {
                "path": path,
                "modes": [{
                    "mode": "fee_fixed_nextbar",
                    "fitness": float(np.mean(rets)),
                    "period_stats": {
                        "mean_ret": float(np.mean(rets)),
                        "min_ret": float(np.min(rets)),
                        "max_ret": float(np.max(rets)),
                        "positive_period_pct": 100.0,
                    },
                    "robust_score": {"passes_default_gates": True},
                    "contract_metrics": {
                        "mean_turnover_rate": 0.03,
                        "max_turnover_rate": 0.03,
                        "mean_saturation_rate": 0.00,
                        "max_saturation_rate": 0.00,
                        "mean_invalid_open_logit_pressure": 0.00,
                        "max_invalid_open_logit_pressure": 0.00,
                        "periods": [
                            {
                                "period": f"2024-{idx + 1:02d}",
                                "regime": regime,
                                "turnover_rate": 0.03,
                                "effective_turnover_rate": 0.03,
                                "saturation_rate": 0.00,
                                "invalid_open_logit_pressure": 0.00,
                                "max_drawdown_pct": 0.10,
                                "cost_pct": cost,
                                "slippage_pct": 0.00,
                            }
                            for idx, regime in enumerate(regimes)
                        ],
                    },
                    "period_rets": rets,
                }],
            }

        return {
            "genomes": [
                genome("baseline.npy", [0.8, 0.8, 0.8, 0.8], 0.01),
                genome("candidate.npy", [1.0, 1.0, 1.0, 1.0], candidate_cost),
            ]
        }

    selected = selector.select_candidate(
        report(candidate_cost=0.01),
        report(candidate_cost=2.00),
        use_fitness_v3_robust=True,
    )

    assert selected["selected_is_baseline"] is True
    assert "validation_fitness_v3" in selected["candidates"][0]["failures"]
    assert selected["candidates"][0]["validation"]["fitness_v3_robust"] < selected[
        "baseline_validation"
    ]["fitness_v3_robust"]


def test_genetics_candidate_selector_cli_can_enable_fitness_v3_gate(tmp_path):
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    regimes = ["crash", "bearish", "neutral", "bullish"]

    def genome(path: str, rets: list[float], cost: float) -> dict:
        return {
            "path": path,
            "modes": [{
                "mode": "fee_fixed_nextbar",
                "fitness": float(np.mean(rets)),
                "period_stats": {
                    "mean_ret": float(np.mean(rets)),
                    "min_ret": float(np.min(rets)),
                    "max_ret": float(np.max(rets)),
                    "positive_period_pct": 100.0,
                },
                "robust_score": {"passes_default_gates": True},
                "contract_metrics": {
                    "mean_turnover_rate": 0.03,
                    "mean_saturation_rate": 0.00,
                    "mean_invalid_open_logit_pressure": 0.00,
                    "periods": [
                        {
                            "period": f"2024-{idx + 1:02d}",
                            "regime": regime,
                            "turnover_rate": 0.03,
                            "effective_turnover_rate": 0.03,
                            "saturation_rate": 0.00,
                            "invalid_open_logit_pressure": 0.00,
                            "max_drawdown_pct": 0.10,
                            "cost_pct": cost,
                            "slippage_pct": 0.00,
                        }
                        for idx, regime in enumerate(regimes)
                    ],
                },
                "period_rets": rets,
            }],
        }

    report = {
        "genomes": [
            genome("baseline.npy", [0.8, 0.8, 0.8, 0.8], 0.01),
            genome("candidate.npy", [1.0, 1.0, 1.0, 1.0], 2.00),
        ]
    }
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    out_path = tmp_path / "selection.json"
    train_path.write_text(json.dumps(report), encoding="utf-8")
    validation_path.write_text(json.dumps(report), encoding="utf-8")

    rc = selector.main([
        "--train-report",
        str(train_path),
        "--validation-report",
        str(validation_path),
        "--out",
        str(out_path),
        "--use-fitness-v3-robust",
    ])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["selected_is_baseline"] is True
    assert "validation_fitness_v3" in payload["candidates"][0]["failures"]


def test_genetics_candidate_selector_fitness_v4_rejects_outlier_validation_winner():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    regimes = ["crash", "bearish", "neutral", "bullish"]

    def genome(path: str, rets: list[float]) -> dict:
        return {
            "path": path,
            "modes": [{
                "mode": "fee_fixed_nextbar",
                "fitness": float(np.mean(rets)),
                "period_stats": {
                    "mean_ret": float(np.mean(rets)),
                    "min_ret": float(np.min(rets)),
                    "max_ret": float(np.max(rets)),
                    "positive_period_pct": 100.0,
                },
                "robust_score": {"passes_default_gates": True},
                "contract_metrics": {
                    "mean_turnover_rate": 0.03,
                    "max_turnover_rate": 0.03,
                    "mean_saturation_rate": 0.00,
                    "max_saturation_rate": 0.00,
                    "mean_invalid_open_logit_pressure": 0.00,
                    "max_invalid_open_logit_pressure": 0.00,
                    "periods": [
                        {
                            "period": f"2024-{idx + 1:02d}",
                            "regime": regime,
                            "turnover_rate": 0.03,
                            "effective_turnover_rate": 0.03,
                            "saturation_rate": 0.00,
                            "invalid_open_logit_pressure": 0.00,
                        }
                        for idx, regime in enumerate(regimes)
                    ],
                },
                "period_rets": rets,
            }],
        }

    report = {
        "genomes": [
            genome("baseline.npy", [0.8, 0.8, 0.8, 0.8]),
            genome("candidate.npy", [0.8, 0.8, 0.8, 4.0]),
        ]
    }

    selected = selector.select_candidate(
        report,
        report,
        use_fitness_v4_robust=True,
    )

    assert selected["selected_is_baseline"] is True
    assert "fitness_v4_gates" in selected["candidates"][0]["failures"]
    assert "validation_fitness_v4" in selected["candidates"][0]["failures"]
    assert selected["candidates"][0]["validation"]["fitness_v4_robust"] < selected[
        "baseline_validation"
    ]["fitness_v4_robust"]


def test_single_candidate_oos_gate_rejects_oos_loss_and_crash_floor_loss():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    regimes = ["crash", "bearish", "neutral", "bullish"]

    def report(baseline_rets: list[float], candidate_rets: list[float]) -> dict:
        def genome(path: str, rets: list[float]) -> dict:
            return {
                "path": path,
                "modes": [{
                    "mode": "fee_fixed_nextbar",
                    "fitness": float(np.mean(rets)),
                    "period_stats": {
                        "mean_ret": float(np.mean(rets)),
                        "min_ret": float(np.min(rets)),
                        "max_ret": float(np.max(rets)),
                        "positive_period_pct": float(
                            np.mean(np.asarray(rets) > 0.0) * 100.0
                        ),
                    },
                    "robust_score": {"passes_default_gates": True},
                    "contract_metrics": {
                        "mean_turnover_rate": 0.03,
                        "max_turnover_rate": 0.03,
                        "mean_saturation_rate": 0.00,
                        "max_saturation_rate": 0.00,
                        "mean_invalid_open_logit_pressure": 0.00,
                        "max_invalid_open_logit_pressure": 0.00,
                        "periods": [
                            {
                                "period": f"2025-{idx + 1:02d}",
                                "regime": regime,
                                "turnover_rate": 0.03,
                                "effective_turnover_rate": 0.03,
                                "saturation_rate": 0.00,
                                "invalid_open_logit_pressure": 0.00,
                                "max_drawdown_pct": 0.10,
                            }
                            for idx, regime in enumerate(regimes)
                        ],
                    },
                    "period_rets": rets,
                }],
            }

        return {
            "genomes": [
                genome("baseline.npy", baseline_rets),
                genome("candidate.npy", candidate_rets),
            ]
        }

    gate = selector.evaluate_single_candidate_multisplit_gate(
        {
            "promotion_eligible": True,
            "selected_path": "candidate.npy",
            "baseline_path": "baseline.npy",
        },
        [report([0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.4, 0.4])],
        use_fitness_v3_robust=True,
    )

    assert gate["promotion_eligible"] is False
    assert "oos_0_mean_ret" in gate["promotion_failures"]
    assert "oos_0_crash_floor" in gate["promotion_failures"]


def test_regime_router_selector_can_improve_unvalidated_regime_without_validation_degradation():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[1.0, 0.1],
        candidate_rets=[-1.0, 0.5],
        regimes=["bearish", "neutral"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[1.0],
        candidate_rets=[-1.0],
        regimes=["bearish"],
    )

    selected = selector.select_regime_router(train, validation)

    assert selected["selected_is_baseline"] is False
    assert selected["promotion_eligible"] is False
    assert "validation_tie" in selected["promotion_failures"]
    assert selected["selected_regime_map"]["bearish"] == "baseline.npy"
    assert selected["selected_regime_map"]["neutral"] == "candidate.npy"
    assert selected["validation"]["mean_ret"] == pytest.approx(1.0)
    assert selected["train"]["mean_ret"] == pytest.approx(0.75)


def test_regime_router_selector_rejects_train_gain_with_validation_degradation():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[1.0],
        candidate_rets=[1.2],
        regimes=["bearish"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[1.0],
        candidate_rets=[-1.0],
        regimes=["bearish"],
    )

    selected = selector.select_regime_router(train, validation)

    assert selected["selected_is_baseline"] is True
    assert selected["selected_regime_map"]["bearish"] == "baseline.npy"


def test_regime_router_selector_requires_multiple_validation_periods_for_promotion():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.40, 0.40],
        regimes=["neutral", "bullish"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.40],
        regimes=["neutral"],
    )

    selected = selector.select_regime_router(train, validation)

    assert selected["selected_is_baseline"] is False
    assert selected["promotion_eligible"] is False
    assert "validation_period_count" in selected["promotion_failures"]


def test_regime_router_selector_requires_validation_coverage_for_selected_regimes():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.40, 0.40],
        regimes=["neutral", "bullish"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.40, 0.30],
        regimes=["neutral", "neutral"],
    )

    selected = selector.select_regime_router(train, validation)

    assert selected["selected_is_baseline"] is False
    assert selected["selected_regime_map"]["bullish"] == "candidate.npy"
    assert selected["promotion_eligible"] is False
    assert "validation_regime_coverage:bullish" in selected["promotion_failures"]


def test_regime_router_selector_uses_guard_reports_to_reject_fragile_maps():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.45, 0.45],
        regimes=["neutral", "bullish"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.40, 0.35],
        regimes=["neutral", "bullish"],
    )
    guard = _genetics_regime_selection_report(
        baseline_rets=[0.20, 0.20],
        candidate_rets=[-0.20, -0.10],
        regimes=["neutral", "bullish"],
    )

    selected = selector.select_regime_router(train, validation, guard_reports=[guard])

    assert selected["selected_is_baseline"] is True
    assert selected["selected_regime_map"]["neutral"] == "baseline.npy"
    assert selected["selected_regime_map"]["bullish"] == "baseline.npy"
    assert "guard_0_mean_ret" in selected["candidates"][0]["failures"]


def test_regime_router_selector_can_restrict_candidate_regimes_for_specialists():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.50, 0.50],
        regimes=["neutral", "bullish"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10, 0.10],
        candidate_rets=[0.40, 0.40],
        regimes=["neutral", "bullish"],
    )

    selected = selector.select_regime_router(
        train,
        validation,
        allowed_candidate_regimes=["neutral"],
    )

    assert selected["selected_is_baseline"] is False
    assert selected["selected_regime_map"]["neutral"] == "candidate.npy"
    assert selected["selected_regime_map"]["bullish"] == "baseline.npy"


def test_regime_router_selector_can_route_crash_specialist():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.35],
        regimes=["crash"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.32],
        regimes=["crash"],
    )

    selected = selector.select_regime_router(
        train,
        validation,
        allowed_candidate_regimes=["crash"],
        min_validation_periods=1,
    )

    assert selected["selected_is_baseline"] is False
    assert selected["selected_regime_map"]["crash"] == "candidate.npy"
    assert selected["baseline_regime_map"]["crash"] == "baseline.npy"


def test_regime_router_cli_accepts_crash_candidate_regime(tmp_path):
    import importlib
    import json

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.35],
        regimes=["crash"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.32],
        regimes=["crash"],
    )
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    out_path = tmp_path / "selection.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")

    rc = selector.main([
        "--train-report",
        str(train_path),
        "--validation-report",
        str(validation_path),
        "--regime-router",
        "--allowed-candidate-regime",
        "crash",
        "--min-validation-periods",
        "1",
        "--out",
        str(out_path),
    ])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["selected_regime_map"]["crash"] == "candidate.npy"


def test_regime_router_selector_rejects_guard_contract_breaches_before_selection():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.40],
        regimes=["neutral"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.35],
        regimes=["neutral"],
    )
    guard = _genetics_regime_selection_report(
        baseline_rets=[0.20],
        candidate_rets=[0.30],
        regimes=["neutral"],
        baseline_turnovers=[0.02],
        candidate_turnovers=[0.20],
        baseline_saturations=[0.02],
        candidate_saturations=[0.18],
    )

    selected = selector.select_regime_router(
        train,
        validation,
        guard_reports=[guard],
        allowed_candidate_regimes=["neutral"],
        min_validation_periods=1,
        max_turnover_rate=0.10,
        max_saturation_rate=0.10,
    )

    assert selected["selected_is_baseline"] is True
    assert "guard_0_turnover" in selected["candidates"][0]["failures"]
    assert "guard_0_saturation" in selected["candidates"][0]["failures"]


def test_regime_router_cli_passes_contract_limits_into_selection(tmp_path):
    import importlib
    import json

    selector = importlib.import_module("tools.select_genetics_candidate")
    train = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.40],
        regimes=["neutral"],
    )
    validation = _genetics_regime_selection_report(
        baseline_rets=[0.10],
        candidate_rets=[0.35],
        regimes=["neutral"],
    )
    guard = _genetics_regime_selection_report(
        baseline_rets=[0.20],
        candidate_rets=[0.30],
        regimes=["neutral"],
        baseline_turnovers=[0.02],
        candidate_turnovers=[0.20],
        baseline_saturations=[0.02],
        candidate_saturations=[0.18],
    )
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    guard_path = tmp_path / "guard.json"
    out_path = tmp_path / "selection.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    guard_path.write_text(json.dumps(guard), encoding="utf-8")

    rc = selector.main([
        "--train-report",
        str(train_path),
        "--validation-report",
        str(validation_path),
        "--guard-report",
        str(guard_path),
        "--regime-router",
        "--allowed-candidate-regime",
        "neutral",
        "--min-validation-periods",
        "1",
        "--max-turnover-rate",
        "0.10",
        "--max-saturation-rate",
        "0.10",
        "--out",
        str(out_path),
    ])

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["selected_is_baseline"] is True
    assert "guard_0_turnover" in payload["candidates"][0]["failures"]


def test_regime_router_multisplit_gate_rejects_oos_degradation():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    selection = {
        "promotion_eligible": True,
        "selected_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "candidate.npy",
            "bullish": "baseline.npy",
        },
        "baseline_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "baseline.npy",
            "bullish": "baseline.npy",
        },
    }
    oos = _genetics_regime_selection_report(
        baseline_rets=[0.30],
        candidate_rets=[-0.10],
        regimes=["neutral"],
    )

    gate = selector.evaluate_regime_router_multisplit_gate(selection, [oos])

    assert gate["promotion_eligible"] is False
    assert "oos_0_mean_ret" in gate["promotion_failures"]
    assert gate["holdouts"][0]["baseline"]["mean_ret"] == pytest.approx(0.30)
    assert gate["holdouts"][0]["candidate"]["mean_ret"] == pytest.approx(-0.10)


def test_regime_router_multisplit_gate_requires_oos_coverage_for_selected_regimes():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    selection = {
        "promotion_eligible": True,
        "selected_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "candidate.npy",
            "bullish": "candidate.npy",
        },
        "baseline_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "baseline.npy",
            "bullish": "baseline.npy",
        },
    }
    oos = _genetics_regime_selection_report(
        baseline_rets=[0.30, 0.20],
        candidate_rets=[0.30, 0.20],
        regimes=["bearish", "bearish"],
    )

    gate = selector.evaluate_regime_router_multisplit_gate(selection, [oos])

    assert gate["promotion_eligible"] is False
    assert "oos_regime_coverage:neutral" in gate["promotion_failures"]
    assert "oos_regime_coverage:bullish" in gate["promotion_failures"]


def test_regime_router_multisplit_gate_rejects_selected_map_contract_breaches():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    selection = {
        "promotion_eligible": True,
        "selected_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "candidate.npy",
            "bullish": "baseline.npy",
        },
        "baseline_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "baseline.npy",
            "bullish": "baseline.npy",
        },
    }
    oos = _genetics_regime_selection_report(
        baseline_rets=[0.20, 0.20],
        candidate_rets=[0.20, 0.20],
        regimes=["bearish", "neutral"],
        baseline_turnovers=[0.01, 0.01],
        candidate_turnovers=[0.01, 0.25],
        baseline_saturations=[0.01, 0.01],
        candidate_saturations=[0.01, 0.23],
    )

    gate = selector.evaluate_regime_router_multisplit_gate(
        selection,
        [oos],
        max_turnover_rate=0.10,
        max_saturation_rate=0.10,
    )

    assert gate["promotion_eligible"] is False
    assert "oos_0_turnover" in gate["promotion_failures"]
    assert "oos_0_saturation" in gate["promotion_failures"]
    assert gate["holdouts"][0]["candidate"]["mean_turnover_rate"] == pytest.approx(0.13)
    assert gate["holdouts"][0]["candidate"]["mean_saturation_rate"] == pytest.approx(0.12)


def test_regime_router_paper_gate_marks_paper_only_after_no_degradation():
    import importlib

    selector = importlib.import_module("tools.select_genetics_candidate")
    selection = {
        "promotion_eligible": True,
        "selected_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "candidate.npy",
            "bullish": "baseline.npy",
        },
        "baseline_regime_map": {
            "bearish": "baseline.npy",
            "neutral": "baseline.npy",
            "bullish": "baseline.npy",
        },
    }
    paper = _genetics_regime_selection_report(
        baseline_rets=[0.20, 0.20],
        candidate_rets=[0.20, 0.25],
        regimes=["bearish", "neutral"],
        baseline_turnovers=[0.01, 0.01],
        candidate_turnovers=[0.01, 0.08],
        baseline_saturations=[0.01, 0.01],
        candidate_saturations=[0.01, 0.07],
    )

    gate = selector.evaluate_regime_router_paper_gate(selection, [paper])

    assert gate["paper_trading_eligible"] is True
    assert gate["live_trading_eligible"] is False
    assert gate["paper_failures"] == []
    assert gate["paper_reports"][0]["candidate"]["mean_ret"] == pytest.approx(0.225)
