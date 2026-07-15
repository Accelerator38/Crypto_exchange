from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "analyze_bitget_range_breakout.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("bitget_range_breakout", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bar(index: int, *, close: float, high: float, low: float, volume: float) -> dict[str, object]:
    return {
        "timestamp": index,
        "datetime": f"t{index}",
        "symbol": "TEST/USDT",
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _repeating_long_breakout_bars(
    *,
    symbol: str = "TEST/USDT",
    repetitions: int = 20,
    exit_close: float = 102.0,
) -> list[dict[str, object]]:
    bars = []
    cursor = 0
    for _ in range(repetitions):
        bars.extend(
            {
                **_bar(cursor + index, close=100.0, high=100.1, low=99.9, volume=100.0),
                "symbol": symbol,
            }
            for index in range(30)
        )
        cursor += 30
        bars.append({
            **_bar(cursor, close=101.0, high=101.2, low=100.8, volume=250.0),
            "symbol": symbol,
        })
        cursor += 1
        bars.extend(
            {
                **_bar(
                    cursor + index,
                    close=exit_close,
                    high=exit_close + 0.2,
                    low=exit_close - 0.3,
                    volume=120.0,
                ),
                "symbol": symbol,
            }
            for index in range(5)
        )
        cursor += 5
    return bars


def test_scan_breakouts_finds_confirmed_long_after_compression():
    tool = _load_tool()
    bars = [
        _bar(index, close=100.0, high=100.1, low=99.9, volume=100.0)
        for index in range(30)
    ]
    bars.append(_bar(30, close=101.0, high=101.2, low=100.8, volume=250.0))
    bars.extend(
        _bar(31 + index, close=102.0, high=102.2, low=101.7, volume=120.0)
        for index in range(5)
    )

    trades = tool.scan_breakouts(
        bars,
        symbol="TEST/USDT",
        lookback_bars=30,
        hold_bars=5,
        max_range_pct=0.5,
        volume_mult=2.0,
        breakout_buffer_bps=2.0,
        round_trip_cost_bps=8.0,
        notional_usd=5.0,
    )

    assert len(trades) == 1
    assert trades[0]["direction"] == "LONG"
    assert trades[0]["net_usd"] > 0.0


def test_scan_breakouts_requires_volume_spike():
    tool = _load_tool()
    bars = [
        _bar(index, close=100.0, high=100.1, low=99.9, volume=100.0)
        for index in range(30)
    ]
    bars.append(_bar(30, close=101.0, high=101.2, low=100.8, volume=120.0))
    bars.extend(
        _bar(31 + index, close=102.0, high=102.2, low=101.7, volume=120.0)
        for index in range(5)
    )

    trades = tool.scan_breakouts(
        bars,
        symbol="TEST/USDT",
        lookback_bars=30,
        hold_bars=5,
        max_range_pct=0.5,
        volume_mult=2.0,
        breakout_buffer_bps=2.0,
        round_trip_cost_bps=8.0,
        notional_usd=5.0,
    )

    assert trades == []


def test_summarize_trades_blocks_sparse_positive_candidate():
    tool = _load_tool()
    summary = tool.summarize_trades(
        [{"net_usd": 0.05, "net_pct": 1.0}],
        min_closed=10,
        max_drawdown_usd=0.2,
    )

    assert summary["eligible"] is False
    assert "min_closed" in summary["fail_reasons"]
    assert summary["expectancy_usd"] > 0.0


def test_analyze_bars_exports_eligible_symbol_direction_candidate():
    tool = _load_tool()
    bars = _repeating_long_breakout_bars(repetitions=10)

    report = tool.analyze_bars(
        {"TEST/USDT": bars},
        lookback_grid=(30,),
        hold_grid=(5,),
        max_range_grid=(0.5,),
        volume_mult_grid=(2.0,),
        breakout_buffer_grid=(2.0,),
        round_trip_cost_bps=8.0,
        notional_usd=5.0,
        min_closed=10,
        max_drawdown_usd=0.2,
        top_n=5,
    )

    assert report["eligible_count"] == 1
    candidate = report["eligible_candidates"][0]
    assert candidate["symbol"] == "TEST"
    assert candidate["direction"] == "LONG"
    assert candidate["closed_trades"] == 10


def test_analyze_bars_exports_aggregate_candidate_across_symbols():
    tool = _load_tool()

    def build_symbol_bars(symbol: str) -> list[dict[str, object]]:
        return _repeating_long_breakout_bars(symbol=symbol, repetitions=5)

    report = tool.analyze_bars(
        {
            "AAA/USDT": build_symbol_bars("AAA/USDT"),
            "BBB/USDT": build_symbol_bars("BBB/USDT"),
        },
        lookback_grid=(30,),
        hold_grid=(5,),
        max_range_grid=(0.5,),
        volume_mult_grid=(2.0,),
        breakout_buffer_grid=(2.0,),
        round_trip_cost_bps=8.0,
        notional_usd=5.0,
        min_closed=10,
        max_drawdown_usd=0.2,
        top_n=5,
    )

    assert report["eligible_count"] == 0
    assert report["aggregate_eligible_count"] == 1
    aggregate = report["aggregate_eligible_candidates"][0]
    assert aggregate["symbol"] == "MULTI"
    assert aggregate["symbols"] == {"AAA": 5, "BBB": 5}
    assert aggregate["closed_trades"] == 10


def test_analyze_bars_marks_candidate_robust_when_train_oos_and_stress_pass():
    tool = _load_tool()
    report = tool.analyze_bars(
        {"TEST/USDT": _repeating_long_breakout_bars(repetitions=20)},
        lookback_grid=(30,),
        hold_grid=(5,),
        max_range_grid=(0.5,),
        volume_mult_grid=(2.0,),
        breakout_buffer_grid=(2.0,),
        round_trip_cost_bps=8.0,
        stress_round_trip_cost_bps=(12.0,),
        notional_usd=5.0,
        min_closed=20,
        max_drawdown_usd=0.2,
        oos_fraction=0.35,
        top_n=5,
    )

    assert report["eligible_count"] == 1
    assert report["robust_eligible_count"] == 1
    candidate = report["robust_eligible_candidates"][0]
    assert candidate["oos_summary"]["closed_trades"] >= 7
    assert candidate["stress_summaries"]["12"]["eligible"] is True


def test_analyze_bars_blocks_robust_when_stress_cost_erases_edge():
    tool = _load_tool()
    report = tool.analyze_bars(
        {"TEST/USDT": _repeating_long_breakout_bars(repetitions=20, exit_close=101.25)},
        lookback_grid=(30,),
        hold_grid=(5,),
        max_range_grid=(0.5,),
        volume_mult_grid=(2.0,),
        breakout_buffer_grid=(2.0,),
        round_trip_cost_bps=8.0,
        stress_round_trip_cost_bps=(120.0,),
        notional_usd=5.0,
        min_closed=20,
        max_drawdown_usd=0.2,
        oos_fraction=0.35,
        top_n=5,
    )

    assert report["eligible_count"] == 1
    assert report["robust_eligible_count"] == 0
    candidate = report["eligible_candidates"][0]
    assert candidate["robust_eligible"] is False
    assert "stress_120_nonpositive_expectancy" in candidate["robust_fail_reasons"]
