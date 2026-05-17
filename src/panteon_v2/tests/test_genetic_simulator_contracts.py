from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


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
