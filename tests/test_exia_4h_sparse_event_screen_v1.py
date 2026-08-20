from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simple_research.position_policy import simulate_ohlc_position_policy
from simple_research.strategies import build_signal


FAMILY_PATH = ROOT / "configs" / "exia_4h_sparse_event_family_v1.json"
RUNNER_PATH = ROOT / "tools" / "run_exia_4h_sparse_event_screen_v1.py"
BAR_MS = 14_400_000


def _load_runner():
    spec = importlib.util.spec_from_file_location("exia_4h_sparse_event_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _panel(bars: int = 100) -> pd.DataFrame:
    rows = []
    for bar in range(bars):
        cycle = -1.5 if bar % 11 == 0 else 0.0
        for symbol_index in range(8):
            price = 100.0 + symbol_index + bar * (0.8 + symbol_index * 0.015) + cycle
            rows.append(
                {
                    "timestamp": bar * BAR_MS,
                    "symbol": f"S{symbol_index}/USDT",
                    "open": price - 0.2,
                    "high": price + 0.7,
                    "low": price - 0.7,
                    "close": price,
                    "volume": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _shared_params() -> dict[str, object]:
    return {
        "fast": 3,
        "slow": 8,
        "threshold_bps": 1.0,
        "slope_bars": 2,
        "slope_threshold_bps": 1.0,
        "market_lookback": 4,
        "market_threshold_bps": 50.0,
        "market_breadth": 0.625,
        "market_minimum_symbols": 6,
        "market_confirmation_bars": 2,
        "direction": "long",
    }


def _policy() -> dict[str, object]:
    return {
        "atr_period": 2,
        "hard_stop_atr": 10.0,
        "breakeven_trigger_atr": None,
        "trailing_trigger_atr": None,
        "trailing_distance_atr": None,
        "no_progress_bars": None,
        "no_progress_mfe_atr": None,
        "max_holding_bars": None,
        "take_profit_atr": None,
    }


def test_sparse_event_family_reuses_sealed_inputs_and_is_safe() -> None:
    family = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))
    runner = _load_runner()
    exit_family, entry_family, position_policy = runner.validate_event_family(family)

    assert len(family["events"]) == 4
    assert sum(event["comparison_only"] for event in family["events"]) == 1
    assert family["position_policy_source"]["entry_mode"] == "event"
    assert position_policy["policy_id"] == "ATR_BREAKEVEN_PROGRESS24"
    assert exit_family["entry_source"]["candidate_id"] == "MQ_EMA12_48_LONG"
    assert entry_family["timeframe_minutes"] == 240
    assert set(family["safety"].values()) == {False}


def test_event_mode_zero_does_not_close_an_open_position() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": bar * BAR_MS,
                "symbol": "BTC/USDT",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            }
            for bar in range(6)
        ]
    )
    signal = pd.Series([0, 0, 1, 0, 0, 0], dtype="int8")
    event_ledger = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=0,
        end_timestamp=6 * BAR_MS,
        policy=_policy(),
        entry_mode="event",
    )
    target_ledger = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=0,
        end_timestamp=6 * BAR_MS,
        policy=_policy(),
        entry_mode="target",
    )

    assert event_ledger.iloc[0]["entry_timestamp"] == 3 * BAR_MS
    assert event_ledger.iloc[0]["exit_reason"] == "split_end"
    assert event_ledger.iloc[0]["exit_timestamp"] == 5 * BAR_MS
    assert target_ledger.iloc[0]["exit_reason"] == "signal_exit"
    assert target_ledger.iloc[0]["exit_timestamp"] == 4 * BAR_MS


def test_sparse_events_have_no_future_dependency() -> None:
    frame = _panel()
    changed = frame.copy()
    cutoff = 65 * BAR_MS
    future = changed["timestamp"] > cutoff
    changed.loc[future, ["open", "high", "low", "close"]] *= np.linspace(
        0.25, 2.5, int(future.sum())
    )[:, None]
    strategies = [
        ("ema_activation_event", {}),
        ("market_regime_transition_event", {}),
        (
            "cross_sectional_breakout_event",
            {"breakout_window": 10, "momentum_lookback": 4, "rank_threshold": 0.75},
        ),
        ("trend_pullback_continuation_event", {}),
    ]
    total_events = 0
    prefix = frame["timestamp"] <= cutoff
    for kind, extras in strategies:
        params = {**_shared_params(), **extras}
        original = build_signal(frame, {"kind": kind, "params": params})
        mutated = build_signal(changed, {"kind": kind, "params": params})
        pd.testing.assert_series_equal(original.loc[prefix], mutated.loc[prefix])
        assert set(original.unique()) <= {0, 1}
        total_events += int(original.sum())
    assert total_events > 0


def test_breakout_event_is_a_fresh_edge_not_a_continuous_state() -> None:
    frame = _panel(60)
    signal = build_signal(
        frame,
        {
            "kind": "cross_sectional_breakout_event",
            "params": {
                **_shared_params(),
                "breakout_window": 10,
                "momentum_lookback": 4,
                "rank_threshold": 0.75,
            },
        },
    )
    for _, group in frame.assign(signal=signal).groupby("symbol", sort=True):
        ordered = group.sort_values("timestamp", kind="stable")
        assert not ((ordered["signal"] == 1) & (ordered["signal"].shift(1) == 1)).any()


def test_event_gate_rejects_control_and_negative_distribution() -> None:
    runner = _load_runner()
    gate = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))["stage_gate"]
    metric = {
        "comparison_only": False,
        "closed_trades": 40,
        "mean_stress_net_bps": 20.0,
        "median_stress_net_bps": -1.0,
        "stress_lcb_bps": 5.0,
        "familywise_stress_lcb_bps": 2.0,
        "p01_stress_net_bps": -400.0,
    }
    assert runner._gate_pass(metric, gate, control_p01_bps=-600.0) is False
    metric["median_stress_net_bps"] = 1.0
    assert runner._gate_pass(metric, gate, control_p01_bps=-600.0) is True
    metric["comparison_only"] = True
    assert runner._gate_pass(metric, gate, control_p01_bps=-600.0) is False
