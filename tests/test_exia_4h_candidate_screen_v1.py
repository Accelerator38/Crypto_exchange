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

from simple_research.strategies import build_market_regime, build_signal


FAMILY_PATH = ROOT / "configs" / "exia_4h_candidate_family_v1.json"
RUNNER_PATH = ROOT / "tools" / "run_exia_4h_candidate_screen_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("exia_4h_candidate_screen_test", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _panel(bars: int = 40) -> pd.DataFrame:
    rows = []
    for bar in range(bars):
        for symbol_index in range(8):
            price = 100.0 + symbol_index + bar * (1.0 + symbol_index * 0.02)
            rows.append(
                {
                    "timestamp": bar * 14_400_000,
                    "symbol": f"S{symbol_index}/USDT",
                    "open": price - 0.2,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 1000.0 + symbol_index,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_family_is_bounded_preregistered_and_safe() -> None:
    family = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))
    runner = _load_runner()
    runner.validate_family(family)

    assert family["profile_id"] == "exia_4h_v1"
    assert family["timeframe_minutes"] == 240
    assert len(family["candidates"]) == 8
    assert set(family["safety"].values()) == {False}
    assert family["stage_gate"] == {
        "minimum_closed_trades": 20,
        "require_positive_stress_mean": True,
        "require_positive_stress_lcb": True,
        "require_positive_familywise_lcb": True,
        "development_to_validation_limit": 4,
        "validation_to_oos_limit": 2,
        "oos_to_sanity_limit": 1,
    }


def test_market_regime_uses_only_completed_current_and_past_bars() -> None:
    frame = _panel()
    params = {
        "market_lookback": 4,
        "market_threshold_bps": 100.0,
        "market_breadth": 0.625,
        "market_minimum_symbols": 6,
        "market_confirmation_bars": 2,
    }
    original = build_market_regime(frame, params)
    cutoff = 24 * 14_400_000
    changed = frame.copy()
    changed.loc[changed["timestamp"] > cutoff, "close"] *= np.linspace(
        0.2,
        3.0,
        int((changed["timestamp"] > cutoff).sum()),
    )
    changed_regime = build_market_regime(changed, params)

    pd.testing.assert_series_equal(
        original.loc[original.index <= cutoff],
        changed_regime.loc[changed_regime.index <= cutoff],
    )
    assert (original.loc[original.index >= 5 * 14_400_000] == 1).all()


def test_market_quorum_signal_has_no_future_dependency_and_honors_direction() -> None:
    frame = _panel()
    params = {
        "fast": 3,
        "slow": 8,
        "threshold_bps": 1.0,
        "slope_bars": 2,
        "slope_threshold_bps": 1.0,
        "market_lookback": 4,
        "market_threshold_bps": 100.0,
        "market_breadth": 0.625,
        "market_minimum_symbols": 6,
        "market_confirmation_bars": 2,
        "direction": "long",
    }
    strategy = {"kind": "ema_market_quorum", "params": params}
    original = build_signal(frame, strategy)
    cutoff = 24 * 14_400_000
    changed = frame.copy()
    changed.loc[changed["timestamp"] > cutoff, "close"] *= 0.1
    changed_signal = build_signal(changed, strategy)
    prefix = frame["timestamp"] <= cutoff

    pd.testing.assert_series_equal(original.loc[prefix], changed_signal.loc[prefix])
    assert set(original.unique()) <= {0, 1}
    assert int((original == 1).sum()) > 0


def test_stage_gate_requires_familywise_positive_lcb() -> None:
    runner = _load_runner()
    gate = json.loads(FAMILY_PATH.read_text(encoding="utf-8"))["stage_gate"]
    metrics = {
        "closed_trades": 100,
        "mean_stress_net_bps": 10.0,
        "stress_lcb_bps": 2.0,
        "familywise_stress_lcb_bps": -0.1,
    }

    assert runner._gate_pass(metrics, gate) is False
    metrics["familywise_stress_lcb_bps"] = 0.1
    assert runner._gate_pass(metrics, gate) is True
