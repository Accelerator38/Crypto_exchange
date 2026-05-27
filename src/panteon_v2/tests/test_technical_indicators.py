from __future__ import annotations

import pytest

from panteon_v2.analysis.technical_indicators import (
    TechnicalIndicatorConfig,
    TechnicalIndicatorState,
)


def test_rsi_becomes_available_after_period_plus_one_closes():
    state = TechnicalIndicatorState(TechnicalIndicatorConfig(rsi_period=14))

    last = None
    for idx in range(15):
        price = 100.0 + idx
        last = state.update_symbol("BTC/USDT", high=price, low=price, close=price)

    assert last is not None
    assert last.rsi_14 == pytest.approx(100.0)


def test_rsi_handles_mixed_gains_and_losses():
    state = TechnicalIndicatorState(TechnicalIndicatorConfig(rsi_period=14))
    prices = [
        100.0,
        102.0,
        101.0,
        103.0,
        104.0,
        102.0,
        105.0,
        107.0,
        106.0,
        108.0,
        109.0,
        107.0,
        110.0,
        112.0,
        111.0,
    ]

    last = None
    for price in prices:
        last = state.update_symbol(
            "ETH/USDT",
            high=price + 1.0,
            low=price - 1.0,
            close=price,
        )

    assert last is not None
    assert 60.0 < last.rsi_14 < 75.0


def test_macd_is_normalized_as_percent_of_close():
    state = TechnicalIndicatorState()

    last = None
    for idx in range(40):
        price = 100.0 + idx * 0.5
        last = state.update_symbol("BTC/USDT", high=price, low=price, close=price)

    assert last is not None
    assert last.macd_line_pct is not None
    assert last.macd_signal_pct is not None
    assert last.macd_histogram_pct is not None
    assert last.macd_line_pct > 0.0
    assert abs(last.macd_histogram_pct) < abs(last.macd_line_pct)


def test_atr_percent_uses_true_range_and_previous_close_gap():
    state = TechnicalIndicatorState(TechnicalIndicatorConfig(atr_period=3))
    candles = [
        (101.0, 99.0, 100.0),
        (103.0, 99.0, 102.0),
        (110.0, 108.0, 109.0),
        (111.0, 107.0, 108.0),
    ]

    last = None
    for high, low, close in candles:
        last = state.update_symbol("SOL/USDT", high=high, low=low, close=close)

    assert last is not None
    assert last.atr_14_pct is not None
    assert last.atr_14_pct == pytest.approx(
        ((4.0 + 8.0 + 4.0) / 3.0) / 108.0 * 100.0
    )
