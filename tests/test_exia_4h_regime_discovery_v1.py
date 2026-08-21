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

SPEC_PATH = ROOT / "configs" / "exia_4h_regime_discovery_v1.json"
PREPARER_PATH = ROOT / "tools" / "prepare_exia_4h_regime_discovery_v1.py"
ANALYZER_PATH = ROOT / "tools" / "analyze_exia_4h_regime_discovery_v1.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _panel(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(20260820)
    records = []
    for symbol_index, symbol in enumerate(("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")):
        returns = rng.normal(0.0002 * (symbol_index - 1), 0.012, rows)
        close = (100.0 + symbol_index * 10.0) * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        spread = np.maximum(close * rng.uniform(0.002, 0.02, rows), 0.01)
        volume = rng.lognormal(8.0 + symbol_index * 0.1, 0.45, rows)
        for index in range(rows):
            records.append(
                {
                    "timestamp": index * 240 * 60 * 1000,
                    "symbol": symbol,
                    "open": open_[index],
                    "high": max(open_[index], close[index]) + spread[index],
                    "low": min(open_[index], close[index]) - spread[index],
                    "close": close[index],
                    "volume": volume[index],
                }
            )
    return pd.DataFrame(records).sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)


def test_preregistered_discovery_has_expected_coverage_and_no_regime_gate() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    _load_module("prepare_regime_discovery_test", PREPARER_PATH).validate_discovery_spec(payload)

    assert payload["experiment_counts"] == {
        "current_components": 43,
        "current_runtime_agents": 22,
        "current_simple_strategies": 8,
        "additional_variants": 90,
        "new_agents": 10,
        "total_components": 143,
    }
    assert payload["selection_policy"]["external_regime_gate"] is False
    assert payload["selection_policy"]["no_trade_by_market_regime"] is False
    assert payload["regime_attribution"]["label_only"] is True
    assert set(payload["safety"].values()) == {False}
    assert {row["variant_id"] for row in payload["agents"]} >= {
        "BASE",
        "FAST067",
        "SLOW150",
        "SLOW200",
        "NEW_BASE",
    }


def test_ten_new_agents_are_distinct_causal_and_emit_valid_targets() -> None:
    from simple_research.strategies import build_signal

    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    agents = [row for row in payload["agents"] if row["variant_id"] == "NEW_BASE"]
    assert len(agents) == 10
    assert len({row["strategy_kind"] for row in agents}) == 10

    panel = _panel()
    cutoff_timestamp = int(panel["timestamp"].unique()[139])
    prefix = panel.loc[panel["timestamp"] <= cutoff_timestamp].copy()
    for agent in agents:
        strategy = {"kind": agent["strategy_kind"], "params": agent["params"]}
        full_signal = build_signal(panel, strategy)
        prefix_signal = build_signal(prefix, strategy)
        assert set(full_signal.unique()) <= {-1, 0, 1}
        assert full_signal.loc[prefix.index].tolist() == prefix_signal.tolist()


def test_mean_and_median_are_separate_and_ranking_rejects_outlier_only_mean() -> None:
    analyzer = _load_module("analyze_regime_discovery_test", ANALYZER_PATH)
    trades = pd.DataFrame(
        {
            "stress_net_bps": [-20.0, -10.0, 100.0],
            "gross_bps": [-4.0, 6.0, 116.0],
            "base_net_bps": [-12.0, -2.0, 108.0],
            "fills": [2, 2, 2],
        }
    )
    summary = analyzer._basic_summary(trades)
    assert summary["mean_stress_net_bps"] > 0
    assert summary["median_stress_net_bps"] == -10.0

    ranking_row = pd.Series(
        {
            "closed_trades": 30,
            "mean_stress_net_bps": 5.0,
            "median_stress_net_bps": -1.0,
            "stress_lcb_bps": 1.0,
            "familywise_stress_lcb_bps": 0.5,
        }
    )
    assert analyzer._ranking_status(ranking_row) == "MIXED"
