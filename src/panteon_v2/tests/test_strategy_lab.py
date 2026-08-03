from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from panteon_v2.analysis import strategy_lab
from panteon_v2.analysis.strategy_lab import (
    EntrySignal,
    MarketBar,
    MarketFrame,
    OpenPosition,
    PairSignal,
    PortfolioSignal,
    _build_symbol_series,
    _metrics,
    aggregate_market_frames,
    detect_entry_signals,
    protective_exit,
)
from panteon_v2.policy.strategy_candidate_registry import (
    StrategyCandidateRegistry,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = StrategyCandidateRegistry.from_json(
    ROOT / "configs" / "strategy_candidates_v1.json"
)


def _frames(closes: list[float], volumes: list[float] | None = None):
    values = volumes or [100.0] * len(closes)
    return [
        MarketFrame(
            timestamp_ms=index * 3_600_000,
            bars={
                "BTC": MarketBar(
                    timestamp_ms=index * 3_600_000,
                    open=close,
                    high=close * 1.001,
                    low=close * 0.999,
                    close=close,
                    volume=volume,
                )
            },
        )
        for index, (close, volume) in enumerate(zip(closes, values))
    ]


def test_pullback_signal_uses_closed_bar_recross():
    closes = [100.0 + index * 0.12 for index in range(140)]
    closes[-2] -= 1.55
    closes[-1] += 0.4
    frames = _frames(closes)
    series = {"BTC": _build_symbol_series("BTC", frames)}

    signals = detect_entry_signals(
        REGISTRY.candidate("regime_pullback_hourly_v1"),
        index=len(frames) - 1,
        series=series,
    )

    assert len(signals) == 1
    assert signals[0].direction == "LONG"
    assert signals[0].regime == "bullish"


def test_compression_requires_confirmed_close_and_volume_expansion():
    noisy = [100.0 + np.sin(index / 2.0) for index in range(190)]
    closes = noisy + [100.0 + np.sin(index) * 0.02 for index in range(30)]
    closes[-1] = 101.5
    volumes = [100.0] * len(closes)
    volumes[-1] = 250.0
    frames = _frames(closes, volumes)
    last = frames[-1].bars["BTC"]
    frames[-1] = MarketFrame(
        timestamp_ms=frames[-1].timestamp_ms,
        bars={
            "BTC": MarketBar(
                timestamp_ms=last.timestamp_ms,
                open=100.0,
                high=101.7,
                low=99.9,
                close=101.5,
                volume=250.0,
            )
        },
    )
    series = {"BTC": _build_symbol_series("BTC", frames)}

    signals = detect_entry_signals(
        REGISTRY.candidate("ohlcv_compression_transition_hourly_v1"),
        index=len(frames) - 1,
        series=series,
    )

    assert len(signals) == 1
    assert signals[0].direction == "LONG"


def test_intrabar_stop_has_priority_over_target():
    position = OpenPosition(
        signal=EntrySignal(
            symbol="BTC",
            direction="LONG",
            signal_index=10,
            score=1.0,
            atr=1.0,
            regime="bullish",
        ),
        entry_index=11,
        entry_timestamp_ms=11,
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
    )
    bar = MarketBar(
        timestamp_ms=12,
        open=100.0,
        high=105.0,
        low=97.0,
        close=102.0,
        volume=100.0,
    )

    price, reason = protective_exit(position, bar)

    assert price == 98.0
    assert reason == "stop_loss"


def test_metrics_apply_costs_and_drawdown():
    trades = [
        {
            "gross_bps": 20.0,
            "net_bps": 8.0,
            "stressed_net_bps": 2.0,
            "net_pnl_usd": 0.008,
            "stressed_net_pnl_usd": 0.002,
        },
        {
            "gross_bps": -10.0,
            "net_bps": -22.0,
            "stressed_net_bps": -28.0,
            "net_pnl_usd": -0.022,
            "stressed_net_pnl_usd": -0.028,
        },
    ]

    metrics = _metrics(trades, net_field="net_bps")

    assert metrics["mean_net_bps"] == -7.0
    assert metrics["net_pnl_usd"] == pytest.approx(-0.014)
    assert metrics["max_drawdown_usd"] == pytest.approx(0.022)


def test_aggregates_hourly_market_into_exact_4h_bars():
    frames = _frames([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])

    aggregated = aggregate_market_frames(frames, bars_per_frame=4)

    assert len(aggregated) == 2
    first = aggregated[0].bars["BTC"]
    assert first.timestamp_ms == 0
    assert first.open == 100.0
    assert first.close == 103.0
    assert first.high == pytest.approx(103.0 * 1.001)
    assert first.low == pytest.approx(100.0 * 0.999)
    assert first.volume == 400.0


def test_cross_sectional_trend_selects_strongest_long_and_short():
    frames = []
    for index in range(200):
        timestamp = index * 4 * 3_600_000
        btc = 100.0 + index * 0.4
        eth = 200.0 - index * 0.4
        frames.append(
            MarketFrame(
                timestamp_ms=timestamp,
                bars={
                    "BTC": MarketBar(
                        timestamp_ms=timestamp,
                        open=btc,
                        high=btc * 1.001,
                        low=btc * 0.999,
                        close=btc,
                        volume=100.0,
                    ),
                    "ETH": MarketBar(
                        timestamp_ms=timestamp,
                        open=eth,
                        high=eth * 1.001,
                        low=eth * 0.999,
                        close=eth,
                        volume=100.0,
                    ),
                },
            )
        )
    series = {
        symbol: _build_symbol_series(symbol, frames)
        for symbol in ("BTC", "ETH")
    }
    candidate = json.loads(
        (
            ROOT / "configs" / "strategy_candidate_p3_cross_sectional_v1.json"
        ).read_text(encoding="utf-8")
    )["candidate"]

    signals = detect_entry_signals(
        candidate,
        index=191,
        series=series,
    )

    assert [(row.symbol, row.direction) for row in signals] == [
        ("BTC", "LONG"),
        ("ETH", "SHORT"),
    ]


def test_market_neutral_signal_selects_strongest_and_weakest_assets():
    frames = []
    for index in range(200):
        timestamp = index * 4 * 3_600_000
        prices = {
            "BTC": 100.0 + index * 0.5,
            "ETH": 150.0 + index * 0.05,
            "DOGE": 200.0 - index * 0.5,
        }
        frames.append(
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
    series = {
        symbol: _build_symbol_series(symbol, frames)
        for symbol in ("BTC", "ETH", "DOGE")
    }
    candidate = json.loads(
        (
            ROOT
            / "configs"
            / "strategy_candidate_p4_market_neutral_pair_v1.json"
        ).read_text(encoding="utf-8")
    )["candidate"]

    signal = strategy_lab._market_neutral_pair_signal(
        candidate,
        index=191,
        series=series,
    )
    snapshot = strategy_lab._relative_momentum_snapshot(
        candidate,
        index=191,
        series=series,
    )
    strongest = max(
        snapshot,
        key=lambda symbol: (snapshot[symbol]["rank_score"], symbol),
    )
    weakest = min(
        snapshot,
        key=lambda symbol: (snapshot[symbol]["rank_score"], symbol),
    )

    assert signal is not None
    assert signal.long_symbol == strongest
    assert signal.short_symbol == weakest
    assert signal.momentum_spread_bps >= 128.0


def test_market_neutral_pair_fills_and_exits_both_legs_atomically(monkeypatch):
    frames = []
    for index, (btc, eth) in enumerate(
        ((100.0, 200.0), (101.0, 198.0), (102.0, 196.0))
    ):
        timestamp = index * 4 * 3_600_000
        frames.append(
            MarketFrame(
                timestamp_ms=timestamp,
                bars={
                    "BTC": MarketBar(
                        timestamp_ms=timestamp,
                        open=btc,
                        high=btc,
                        low=btc,
                        close=btc,
                        volume=100.0,
                    ),
                    "ETH": MarketBar(
                        timestamp_ms=timestamp,
                        open=eth,
                        high=eth,
                        low=eth,
                        close=eth,
                        volume=100.0,
                    ),
                },
            )
        )
    series = {
        symbol: _build_symbol_series(symbol, frames)
        for symbol in ("BTC", "ETH")
    }
    candidate = json.loads(
        (
            ROOT
            / "configs"
            / "strategy_candidate_p4_market_neutral_pair_v1.json"
        ).read_text(encoding="utf-8")
    )["candidate"]
    candidate["exit_contract"]["max_holding_bars"] = 1
    candidate["exit_contract"]["rank_inversion_exit"] = False

    def first_pair(candidate, *, index, series):
        del candidate, series
        if index == 0:
            return PairSignal(
                long_symbol="BTC",
                short_symbol="ETH",
                signal_index=0,
                score_spread=1.0,
                momentum_spread_bps=500.0,
            )
        return None

    monkeypatch.setattr(
        strategy_lab,
        "_market_neutral_pair_signal",
        first_pair,
    )
    result = strategy_lab._simulate_pair_split(
        candidate,
        split_name="test",
        frames=frames,
        series=series,
        start_index=0,
        end_index=2,
        costs={
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 4.0,
            "safety_buffer_bps": 4.0,
            "cost_stress_multiplier": 1.5,
        },
        portfolio={"notional_per_trade_usd": 10.0},
    )

    assert result["entry_fills"] == 2
    assert result["exit_fills"] == 2
    assert result["closed_pairs"] == 1
    assert result["paired_fill_atomicity_violations"] == 0
    trade = result["trades"][0]
    expected_gross = (
        (102.0 / 101.0 - 1.0) + (198.0 / 196.0 - 1.0)
    ) * 5_000.0
    assert trade["gross_bps"] == pytest.approx(expected_gross)
    assert trade["net_bps"] == pytest.approx(expected_gross - 12.0)
    assert trade["net_pnl_usd"] == pytest.approx(
        20.0 * (expected_gross - 12.0) / 10_000.0
    )


def test_market_neutral_portfolio_selects_disjoint_top2_bottom2():
    symbols = ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "BNB", "LINK")
    drifts = {
        symbol: 0.002 - index * 0.00055
        for index, symbol in enumerate(symbols)
    }
    frames = []
    for index in range(210):
        timestamp = index * 4 * 3_600_000
        prices = {
            symbol: 100.0
            * math.exp(
                drift * index + 0.002 * math.sin(index / 3.0)
            )
            for symbol, drift in drifts.items()
        }
        frames.append(
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
    series = {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }
    candidate = json.loads(
        (
            ROOT
            / "configs"
            / "strategy_candidate_p5_market_neutral_portfolio_v1.json"
        ).read_text(encoding="utf-8")
    )["candidate"]
    decision_index = next(
        index
        for index in range(180, len(frames))
        if strategy_lab._portfolio_decision_index(
            candidate,
            index=index,
            series=series,
        )
    )

    signal = strategy_lab._market_neutral_portfolio_signal(
        candidate,
        index=decision_index,
        series=series,
    )

    assert signal is not None
    assert len(signal.long_symbols) == 2
    assert len(signal.short_symbols) == 2
    assert set(signal.long_symbols).isdisjoint(signal.short_symbols)
    assert signal.momentum_dispersion_bps >= 128.0


def test_market_neutral_portfolio_rebalances_all_four_legs(monkeypatch):
    symbols = ("BTC", "ETH", "SOL", "XRP")
    prices = (
        {"BTC": 100.0, "ETH": 200.0, "SOL": 50.0, "XRP": 20.0},
        {"BTC": 101.0, "ETH": 202.0, "SOL": 49.0, "XRP": 19.8},
        {"BTC": 103.0, "ETH": 203.0, "SOL": 48.0, "XRP": 19.4},
    )
    frames = []
    for index, row in enumerate(prices):
        timestamp = index * 4 * 3_600_000
        frames.append(
            MarketFrame(
                timestamp_ms=timestamp,
                bars={
                    symbol: MarketBar(
                        timestamp_ms=timestamp,
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=100.0,
                    )
                    for symbol, price in row.items()
                },
            )
        )
    series = {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }
    candidate = json.loads(
        (
            ROOT
            / "configs"
            / "strategy_candidate_p5_market_neutral_portfolio_v1.json"
        ).read_text(encoding="utf-8")
    )["candidate"]
    signal = PortfolioSignal(
        long_symbols=("BTC", "ETH"),
        short_symbols=("SOL", "XRP"),
        signal_index=0,
        score_dispersion=1.0,
        momentum_dispersion_bps=500.0,
    )

    monkeypatch.setattr(
        strategy_lab,
        "_portfolio_decision_index",
        lambda candidate, *, index, series: index in {0, 1},
    )
    monkeypatch.setattr(
        strategy_lab,
        "_market_neutral_portfolio_signal",
        lambda candidate, *, index, series: signal if index in {0, 1} else None,
    )
    result = strategy_lab._simulate_portfolio_split(
        candidate,
        split_name="test",
        frames=frames,
        series=series,
        start_index=0,
        end_index=2,
        costs={
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 4.0,
            "safety_buffer_bps": 4.0,
            "cost_stress_multiplier": 1.5,
        },
        portfolio={"notional_per_trade_usd": 10.0},
    )

    assert result["entry_fills"] == 8
    assert result["exit_fills"] == 4
    assert result["closed_portfolios"] == 1
    assert result["remaining_open_positions"] == 4
    assert result["portfolio_fill_atomicity_violations"] == 0
    assert result["net_notional_usd"] == 0.0
    trade = result["trades"][0]
    leg_returns = [
        103.0 / 101.0 - 1.0,
        203.0 / 202.0 - 1.0,
        49.0 / 48.0 - 1.0,
        19.8 / 19.4 - 1.0,
    ]
    expected_gross = sum(leg_returns) / 4.0 * 10_000.0
    assert trade["gross_bps"] == pytest.approx(expected_gross)
    assert trade["net_bps"] == pytest.approx(expected_gross - 12.0)
    assert trade["net_pnl_usd"] == pytest.approx(
        40.0 * (expected_gross - 12.0) / 10_000.0
    )


def test_funding_carry_collects_settlements_after_next_bar_entry():
    frames = _frames([100.0] * 60)
    series = {"BTC": _build_symbol_series("BTC", frames)}
    candidate = REGISTRY.candidate("funding_carry_hourly_v1")
    funding = {
        "BTC": {
            index * 3_600_000: 0.001
            for index in (16, 24, 32, 40, 48, 56)
        }
    }

    signals = detect_entry_signals(
        candidate,
        index=32,
        series=series,
        funding_by_symbol=funding,
    )
    assert len(signals) == 1
    assert signals[0].direction == "SHORT"

    result = strategy_lab._simulate_split(
        candidate,
        split_name="recent_screen",
        frames=frames,
        series=series,
        start_index=0,
        end_index=59,
        costs={
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 4.0,
            "safety_buffer_bps": 4.0,
            "cost_stress_multiplier": 1.5,
        },
        portfolio={"notional_per_trade_usd": 10.0},
        funding_by_symbol=funding,
    )

    assert result["closed_trades"] == 1
    trade = result["trades"][0]
    assert trade["entry_index"] == 33
    assert trade["exit_index"] == 57
    assert trade["funding_bps"] == pytest.approx(30.0)
    assert trade["net_bps"] == pytest.approx(18.0)


def test_split_enters_next_bar_open_and_right_censors_position(monkeypatch):
    frames = _frames([100.0, 101.0, 102.0])
    series = {"BTC": _build_symbol_series("BTC", frames)}
    candidate = REGISTRY.candidate("regime_pullback_hourly_v1")
    candidate["exit_contract"] = {
        **candidate["exit_contract"],
        "stop_atr": 100.0,
        "target_atr": 100.0,
        "max_holding_bars": 1,
    }

    def signals_at_first_bar(
        candidate,
        *,
        index,
        series,
        funding_by_symbol=None,
    ):
        del candidate, series, funding_by_symbol
        if index != 0:
            return []
        return [
            EntrySignal(
                symbol="BTC",
                direction="LONG",
                signal_index=0,
                score=1.0,
                atr=1.0,
                regime="bullish",
            )
        ]

    monkeypatch.setattr(strategy_lab, "detect_entry_signals", signals_at_first_bar)

    closed_result = strategy_lab._simulate_split(
        candidate,
        split_name="test",
        frames=frames,
        series=series,
        start_index=0,
        end_index=2,
        costs={
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 4.0,
            "safety_buffer_bps": 10.0,
            "cost_stress_multiplier": 1.5,
        },
        portfolio={"notional_per_trade_usd": 10.0},
    )

    assert closed_result["closed_trades"] == 1
    assert closed_result["trades"][0]["entry_index"] == 1
    assert closed_result["trades"][0]["entry_price"] == 101.0
    assert closed_result["trades"][0]["exit_index"] == 2
    assert closed_result["trades"][0]["exit_price"] == 102.0
    assert closed_result["trades"][0]["cost_bps"] == 12.0
    assert closed_result["trades"][0]["stressed_cost_bps"] == 22.0

    candidate["exit_contract"]["max_holding_bars"] = 10
    censored_result = strategy_lab._simulate_split(
        candidate,
        split_name="test",
        frames=frames,
        series=series,
        start_index=0,
        end_index=2,
        costs={
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 4.0,
            "safety_buffer_bps": 10.0,
            "cost_stress_multiplier": 1.5,
        },
        portfolio={"notional_per_trade_usd": 10.0},
    )

    assert censored_result["selected_signals"] == 1
    assert censored_result["entry_fills"] == 1
    assert censored_result["exit_fills"] == 0
    assert censored_result["closed_trades"] == 0
    assert censored_result["remaining_open_positions"] == 1
    assert censored_result["right_censored_positions"] == 1
