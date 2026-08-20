from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from simple_research.position_policy import simulate_ohlc_position_policy


FAMILY_PATH = ROOT / "configs" / "exia_4h_exit_policy_family_v1.json"
RUNNER_PATH = ROOT / "tools" / "run_exia_4h_exit_policy_screen_v1.py"
BAR_MS = 14_400_000


def _load_runner():
    spec = importlib.util.spec_from_file_location("exia_4h_exit_policy_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": index * BAR_MS,
                "symbol": "BTC/USDT",
                "open": prices[0],
                "high": prices[1],
                "low": prices[2],
                "close": prices[3],
                "volume": 1000.0,
            }
            for index, prices in enumerate(bars)
        ]
    )


def _policy(**updates):
    policy = {
        "atr_period": 2,
        "hard_stop_atr": 1.0,
        "breakeven_trigger_atr": None,
        "trailing_trigger_atr": None,
        "trailing_distance_atr": None,
        "no_progress_bars": None,
        "no_progress_mfe_atr": None,
        "max_holding_bars": None,
        "take_profit_atr": None,
    }
    policy.update(updates)
    return policy


def test_exit_family_is_fixed_bounded_and_safe() -> None:
    family = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))
    runner = _load_runner()
    entry_family = runner.validate_exit_family(family)

    assert family["entry_source"]["candidate_id"] == "MQ_EMA12_48_LONG"
    assert family["timeframe_minutes"] == 240
    assert len(family["policies"]) == 4
    assert sum(item["comparison_only"] for item in family["policies"]) == 1
    assert set(family["safety"].values()) == {False}
    assert family["stage_gate"]["minimum_p01_improvement_vs_control_bps"] == 100.0
    assert entry_family["source"]["dataset_sha256"] == (
        "f66529b4135d04ddc35b7a6535667d1c6d53cd39c34adb86f23e54d4082b2de7"
    )


def test_same_bar_stop_and_target_uses_stop_first() -> None:
    frame = _frame(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 103.0, 97.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
        ]
    )
    signal = pd.Series([0, 0, 1, 1, 0], index=frame.index, dtype="int8")
    ledger = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=0,
        end_timestamp=5 * BAR_MS,
        policy=_policy(take_profit_atr=1.0),
    )

    assert len(ledger) >= 1
    first = ledger.iloc[0]
    assert first["entry_timestamp"] == 3 * BAR_MS
    assert first["exit_timestamp"] == 3 * BAR_MS
    assert first["exit_reason"] == "stop"
    assert first["entry_atr"] == 2.0
    assert first["exit_price"] == 98.0


def test_gap_through_stop_fills_at_open() -> None:
    frame = _frame(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (95.0, 96.0, 94.0, 95.0),
        ]
    )
    signal = pd.Series([0, 0, 1, 1, 1], index=frame.index, dtype="int8")
    ledger = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=0,
        end_timestamp=5 * BAR_MS,
        policy=_policy(),
    )

    assert len(ledger) == 1
    trade = ledger.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_timestamp"] == 4 * BAR_MS
    assert trade["exit_price"] == 95.0


def test_future_ohlc_changes_do_not_change_completed_prefix_trades() -> None:
    bars = [(100.0, 101.0, 99.0, 100.0)] * 12
    frame = _frame(bars)
    signal = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype="int8")
    original = simulate_ohlc_position_policy(
        frame,
        signal,
        start_timestamp=0,
        end_timestamp=12 * BAR_MS,
        policy=_policy(hard_stop_atr=4.0),
    )
    changed = frame.copy()
    changed.loc[changed["timestamp"] > 9 * BAR_MS, ["open", "high", "low", "close"]] *= 3.0
    changed_result = simulate_ohlc_position_policy(
        changed,
        signal,
        start_timestamp=0,
        end_timestamp=12 * BAR_MS,
        policy=_policy(hard_stop_atr=4.0),
    )

    prefix_original = original.loc[original["exit_timestamp"] <= 9 * BAR_MS].reset_index(drop=True)
    prefix_changed = changed_result.loc[changed_result["exit_timestamp"] <= 9 * BAR_MS].reset_index(drop=True)
    pd.testing.assert_frame_equal(prefix_original, prefix_changed)
    assert len(prefix_original) == 2


def test_gate_requires_positive_median_and_material_p01_improvement() -> None:
    runner = _load_runner()
    gate = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))["stage_gate"]
    metrics = {
        "comparison_only": False,
        "closed_trades": 100,
        "mean_stress_net_bps": 10.0,
        "median_stress_net_bps": -0.1,
        "stress_lcb_bps": 2.0,
        "familywise_stress_lcb_bps": 1.0,
        "p01_stress_net_bps": -500.0,
    }

    assert runner._gate_pass(metrics, gate, control_p01_bps=-700.0) is False
    metrics["median_stress_net_bps"] = 0.1
    metrics["p01_stress_net_bps"] = -650.0
    assert runner._gate_pass(metrics, gate, control_p01_bps=-700.0) is False
    metrics["p01_stress_net_bps"] = -600.0
    assert runner._gate_pass(metrics, gate, control_p01_bps=-700.0) is True
    metrics["comparison_only"] = True
    assert runner._gate_pass(metrics, gate, control_p01_bps=-700.0) is False
