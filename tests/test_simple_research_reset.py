from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simple_research.batch import validate_registry
from simple_research.dataset import build_feature_tape, load_feature_tape
from simple_research.simulator import CostModel, simulate_targets
from simple_research.strategies import build_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _frame(rows: int = 160) -> pd.DataFrame:
    timestamp = np.arange(rows, dtype=np.int64) * 3_600_000
    close = 100.0 + np.linspace(0.0, 10.0, rows) + np.sin(np.arange(rows) / 5.0)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 100.0),
            "symbol": "BTC/USDT",
        }
    )


def test_registry_is_bounded_and_read_only():
    path = (
        PROJECT_ROOT
        / "configs"
        / "simple_research"
        / "preregistered_strategies_v1.json"
    )
    registry = json.loads(path.read_text(encoding="utf-8"))

    validate_registry(registry)

    assert len(registry["strategies"]) == 10
    assert registry["orders_enabled"] is False
    assert registry["promotion_authority"] is False


def test_signals_do_not_change_when_only_future_bars_change():
    original = _frame()
    changed = original.copy()
    changed.loc[121:, ["open", "high", "low", "close"]] *= 10.0
    strategy = {
        "kind": "ema_trend",
        "params": {"fast": 12, "slow": 48, "threshold_bps": 5.0},
    }

    original_signal = build_signal(original, strategy)
    changed_signal = build_signal(changed, strategy)

    pd.testing.assert_series_equal(original_signal.iloc[:121], changed_signal.iloc[:121])


def test_simulator_delays_signal_to_next_open_and_charges_round_trip_cost():
    frame = pd.DataFrame(
        {
            "timestamp": [0, 3_600_000, 7_200_000, 10_800_000],
            "open": [100.0, 100.0, 110.0, 110.0],
            "high": [101.0, 101.0, 111.0, 111.0],
            "low": [99.0, 99.0, 109.0, 109.0],
            "close": [100.0, 100.0, 110.0, 110.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
            "symbol": ["BTC/USDT"] * 4,
        }
    )
    signal = pd.Series([1, 0, 0, 0], dtype="int8")

    result = simulate_targets(
        frame,
        signal,
        start_timestamp=0,
        end_timestamp=14_400_000,
        costs=CostModel(fee_bps_per_fill=4.0, slippage_bps_per_fill=2.0),
    )

    assert len(result.ledger) == 1
    trade = result.ledger.iloc[0]
    assert trade["entry_timestamp"] == 3_600_000
    assert trade["exit_timestamp"] == 7_200_000
    assert trade["gross_bps"] == pytest.approx(1000.0)
    assert trade["cost_bps"] == pytest.approx(12.0)
    assert trade["net_bps"] == pytest.approx(988.0)


def test_feature_tape_round_trip(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    frame = _frame(120)
    frame.to_csv(source / "crypto_1m_2026_all_symbols.csv", index=False)
    output = tmp_path / "feature.parquet"

    manifest = build_feature_tape(
        source_dir=source,
        output_path=output,
        timeframe="1h",
        timeframe_ms=3_600_000,
        expected_symbols=("BTC/USDT",),
    )
    loaded = load_feature_tape(output)

    assert manifest["validation"]["passed"] is True
    assert manifest["future_labels_included"] is False
    assert manifest["orders_enabled"] is False
    assert len(loaded) == 120
    assert "donchian_high_55" in loaded.columns
