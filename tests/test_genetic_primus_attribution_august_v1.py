"""Direct deterministic regressions, no pytest, collector, market data or network."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from tools.analyze_genetic_primus_attribution_august_v1 import (
    bar_attribution, blocks, describe, trade_buckets,
)


def test_blocks_cover_exactly_once():
    expected = [(0, 91), (91, 182), (182, 273), (273, 365)]
    assert list(blocks(365, [0, 91, 182, 273, 365])) == expected
    assert sum(b - a for a, b in expected) == 365
    for edges in ([0, 90, 90, 365], [1, 365], [0, 366]):
        try:
            blocks(365, edges)
        except ValueError:
            continue
        raise AssertionError("invalid boundaries accepted")


def test_gap_fees_terminal_and_additivity():
    bars = pd.DataFrame({"timestamp": [0, 1, 2], "symbol": ["X"] * 3,
                         "open": [100, 115, 110], "close": [110, 120, 100]})
    decisions = pd.DataFrame({"execution_timestamp": [0, 1, 2], "symbol": ["X"] * 3,
                              "quantity_after": [.005, .005, -.004], "desired_position": [1, 1, -1]})
    trades = pd.DataFrame([
        {"symbol": "X", "entry_timestamp": 0, "exit_timestamp": 2, "quantity": .005,
         "entry_price": 100, "exit_price": 110, "entry_fee": .001, "exit_fee": .002,
         "gross_pnl": .05, "net_pnl": .047, "direction": 1},
        {"symbol": "X", "entry_timestamp": 2, "exit_timestamp": 3, "quantity": -.004,
         "entry_price": 110, "exit_price": 100, "entry_fee": .003, "exit_fee": .004,
         "gross_pnl": .04, "net_pnl": .033, "direction": -1}])
    portfolio = pd.DataFrame({"timestamp": [1, 2, 3], "equity": [1.049, 1.099, 1.080]})
    a = bar_attribution(bars, decisions, trades, portfolio, ("X",), 0, 3, 1)
    np.testing.assert_allclose(a["gross"].ravel(), [.05, .05, -.01], atol=1e-12)
    assert abs(a["fees"].sum() - .01) < 1e-12
    assert abs(describe(a, 0, 1)["net_pnl"] + describe(a, 1, 3)["net_pnl"] - .08) < 1e-12
    damaged = portfolio.copy()
    damaged.loc[1, "equity"] += .001
    try:
        bar_attribution(bars, decisions, trades, damaged, ("X",), 0, 3, 1)
    except ArithmeticError:
        pass
    else:
        raise AssertionError("nonreconciling equity accepted")


def test_trade_buckets_half_open_and_entry_partition():
    trades = pd.DataFrame([
        {"symbol": "X", "entry_timestamp": 0, "exit_timestamp": 2, "direction": 1, "net_pnl": .1, "gross_pnl": .11},
        {"symbol": "X", "entry_timestamp": 2, "exit_timestamp": 4, "direction": -1, "net_pnl": -.1, "gross_pnl": -.09},
        {"symbol": "X", "entry_timestamp": 4, "exit_timestamp": 6, "direction": 1, "net_pnl": .2, "gross_pnl": .21}])
    target = np.array([[0], [0], [-1], [0], [1], [1]])
    result = trade_buckets(trades, target, ("X",), 0, 6, 1)
    assert all(v["count"] == 1 for v in result.values())
    left = trade_buckets(trades, target, ("X",), 0, 6, 1, 0, 2)
    right = trade_buckets(trades, target, ("X",), 0, 6, 1, 2, 6)
    assert sum(v["count"] for v in left.values()) == 1
    assert sum(v["count"] for v in right.values()) == 2


def test_zero_trade_attribution():
    bars = pd.DataFrame({"timestamp": [0], "symbol": ["X"], "open": [100], "close": [200]})
    decisions = pd.DataFrame({"execution_timestamp": [0], "symbol": ["X"],
                              "quantity_after": [0.0], "desired_position": [0]})
    portfolio = pd.DataFrame({"timestamp": [1], "equity": [1.0]})
    a = bar_attribution(bars, decisions, pd.DataFrame(), portfolio, ("X",), 0, 1, 1)
    assert a["net"].sum() == 0 and a["turnover"].sum() == 0
