from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from panteon_v2.analysis.cointegration_development import (
    CointegrationModel,
    CointegrationSignal,
    _cointegration_metrics,
    _close_position,
    _open_position,
    fit_best_cointegration_pair,
    model_entry_signal,
)
from panteon_v2.analysis.strategy_lab import (
    MarketBar,
    MarketFrame,
    _build_symbol_series,
)


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    ROOT / "configs" / "strategy_candidate_p6_cointegration_spread_v1.json"
)


def _candidate():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["candidate"]


def _frames(price_rows: list[dict[str, float]]) -> list[MarketFrame]:
    result = []
    for index, prices in enumerate(price_rows):
        timestamp = index * 4 * 3_600_000
        result.append(
            MarketFrame(
                timestamp_ms=timestamp,
                bars={
                    symbol: MarketBar(
                        timestamp_ms=timestamp,
                        open=price,
                        high=price * 1.001,
                        low=price * 0.999,
                        close=price,
                        volume=100.0,
                    )
                    for symbol, price in prices.items()
                },
            )
        )
    return result


def test_walk_forward_fit_selects_cointegrated_pair_without_future_data():
    rng = np.random.default_rng(7)
    x_log = np.cumsum(rng.normal(0.0002, 0.006, 300)) + math.log(100.0)
    residual = np.zeros(300)
    for index in range(1, len(residual)):
        residual[index] = (
            0.8 * residual[index - 1] + rng.normal(0.0, 0.002)
        )
    y_log = 0.3 + 1.1 * x_log + residual
    sol_log = np.cumsum(rng.normal(0.0001, 0.009, 300)) + math.log(50.0)
    xrp_log = np.cumsum(rng.normal(-0.0001, 0.01, 300)) + math.log(20.0)
    frames = _frames(
        [
            {
                "BTC": math.exp(y_log[index]),
                "ETH": math.exp(x_log[index]),
                "SOL": math.exp(sol_log[index]),
                "XRP": math.exp(xrp_log[index]),
            }
            for index in range(300)
        ]
    )
    series = {
        symbol: _build_symbol_series(symbol, frames)
        for symbol in ("BTC", "ETH", "SOL", "XRP")
    }

    model, audit = fit_best_cointegration_pair(
        _candidate(),
        index=299,
        series=series,
    )

    assert model is not None
    assert model.pair_id == "BTC|ETH"
    assert model.formation_start_index == 48
    assert model.formation_end_index == 299
    assert model.pvalue <= 0.05
    assert 2.0 <= model.half_life_bars <= 42.0
    assert audit["tested_pairs"] == 6


def test_entry_signal_requires_cost_aware_spread_reversion():
    frames = _frames(
        [
            {"BTC": 100.0, "ETH": 100.0},
            {"BTC": 100.0 * math.exp(0.03), "ETH": 100.0},
        ]
    )
    series = {
        symbol: _build_symbol_series(symbol, frames)
        for symbol in ("BTC", "ETH")
    }
    model = CointegrationModel(
        y_symbol="BTC",
        x_symbol="ETH",
        formation_start_index=0,
        formation_end_index=0,
        alpha=0.0,
        beta=1.0,
        spread_mean=0.0,
        spread_std=0.01,
        pvalue=0.01,
        half_life_bars=4.0,
    )

    signal = model_entry_signal(
        _candidate(),
        model=model,
        index=1,
        series=series,
    )

    assert signal is not None
    assert signal.direction == "SHORT_SPREAD"
    assert signal.zscore == pytest.approx(3.0)
    assert signal.estimated_reversion_bps == pytest.approx(250.0)


def test_hedge_weighted_close_applies_costs_to_both_legs():
    model = CointegrationModel(
        y_symbol="BTC",
        x_symbol="ETH",
        formation_start_index=0,
        formation_end_index=0,
        alpha=0.0,
        beta=1.0,
        spread_mean=0.0,
        spread_std=0.01,
        pvalue=0.01,
        half_life_bars=4.0,
    )
    signal = CointegrationSignal(
        model=model,
        direction="SHORT_SPREAD",
        signal_index=0,
        zscore=2.5,
        estimated_reversion_bps=200.0,
    )
    entry = _frames([{"BTC": 100.0, "ETH": 100.0}])[0]
    exit_frame = _frames([{"BTC": 90.0, "ETH": 110.0}])[0]
    position = _open_position(
        signal,
        frame=entry,
        index=1,
        gross_notional_usd=20.0,
    )

    trade, legs = _close_position(
        position,
        frame=exit_frame,
        exit_index=2,
        exit_reason="mean_reversion",
        real_cost_bps=12.0,
        stressed_cost_bps=18.0,
    )

    expected_gross = ((100.0 / 90.0 - 1.0) + (110.0 / 100.0 - 1.0)) / 2
    assert position.y_notional_usd == pytest.approx(10.0)
    assert position.x_notional_usd == pytest.approx(10.0)
    assert trade["gross_bps"] == pytest.approx(expected_gross * 10_000.0)
    assert trade["net_bps"] == pytest.approx(
        expected_gross * 10_000.0 - 12.0
    )
    assert [row["direction"] for row in legs] == ["SHORT", "LONG"]
    assert _cointegration_metrics(
        [trade],
        net_field="net_bps",
    )["fills"] == 4
