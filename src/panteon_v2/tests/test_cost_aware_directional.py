from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from panteon_v2.analysis import cost_aware_directional


ROOT = Path(__file__).resolve().parents[3]


def _synthetic_history(days: int = 35) -> pd.DataFrame:
    rows = days * 24 * 60
    timestamps = 1_800_000_000_000 + np.arange(rows, dtype=np.int64) * 60_000
    phase = np.arange(rows) % 180
    returns = np.where(
        phase < 30,
        0.00010,
        np.where((phase >= 60) & (phase < 90), -0.00010, 0.0),
    )
    close = 100.0 * np.exp(np.cumsum(returns))
    open_price = np.concatenate(([close[0]], close[:-1]))
    high = np.maximum(open_price, close) * 1.00002
    low = np.minimum(open_price, close) * 0.99998
    volume = 1000.0 + 100.0 * np.sin(np.arange(rows) / 17.0)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "BTC/USDT",
        }
    )


def test_labeled_dataset_has_three_classes_and_entry_only_features():
    labeled, feature_names = cost_aware_directional._build_labeled_dataset(
        _synthetic_history()
    )

    assert set(labeled["label_class"]) == {-1, 0, 1}
    assert set(cost_aware_directional.FEATURE_NAMES).issubset(feature_names)
    assert labeled[feature_names].notna().all().all()
    assert (labeled["exit_timestamp_ms"] > labeled["timestamp"]).all()
    assert labeled.attrs["ambiguous_rows_removed"] == 0


def test_directional_report_is_anchored_and_never_creates_runtime_policy(
    monkeypatch,
):
    history = _synthetic_history()
    monkeypatch.setattr(
        cost_aware_directional,
        "_load_history",
        lambda _: (
            history,
            {
                "dataset_dir": "fixture",
                "manifest_sha256": "a" * 64,
                "dataset_sha256": "b" * 64,
                "requested_start_date": "fixture",
                "requested_end_date": "fixture",
                "timeframe": "1m",
                "exchange": "BITGET",
            },
        ),
    )

    report = cost_aware_directional.evaluate_cost_aware_directional("fixture")

    assert len(report["folds"]) == 2
    assert report["research_only"] is True
    assert report["runtime_profile_created"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    for fold in report["folds"]:
        assert fold["train_dates"][1] < fold["validation_dates"][0]
        assert fold["validation_dates"][1] < fold["oos_dates"][0]
        assert fold["selected_threshold"] in cost_aware_directional.THRESHOLD_GRID


def test_directional_cli_bootstraps_outside_repository(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_bitget_cost_aware_directional_v1.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "anchored walk-forward" in result.stdout
