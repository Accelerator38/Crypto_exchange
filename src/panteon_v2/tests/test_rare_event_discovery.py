from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from panteon_v2.analysis import rare_event_discovery


ROOT = Path(__file__).resolve().parents[3]


def _synthetic_history(days: int = 28) -> pd.DataFrame:
    count = days * 24 * 60
    timestamps = 1_800_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    phase = np.arange(count) % 360
    returns = np.where(
        phase < 45,
        0.00008,
        np.where((phase >= 180) & (phase < 225), -0.00008, 0.0),
    )
    close = 100.0 * np.exp(np.cumsum(returns))
    open_price = np.concatenate(([close[0]], close[:-1]))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.00005,
            "low": np.minimum(open_price, close) * 0.99995,
            "close": close,
            "volume": 1_000.0 + 500.0 * (phase == 44),
            "symbol": "BTC/USDT",
        }
    )


def test_event_features_exclude_current_bar_from_prior_range():
    history = _synthetic_history(days=1)
    frame = rare_event_discovery._build_event_frame(history)
    row = frame.iloc[120]

    assert row["prior_high_120m"] == history.iloc[:120]["high"].max()
    assert row["prior_low_120m"] == history.iloc[:120]["low"].min()


def test_same_bar_take_and_stop_resolves_to_stop():
    timestamps = 1_800_000_000_000 + np.arange(241, dtype=np.int64) * 60_000
    bars = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": np.full(241, 100.0),
            "high": np.full(241, 100.0),
            "low": np.full(241, 100.0),
        }
    )
    bars.loc[1, "high"] = 100.7
    bars.loc[1, "low"] = 99.6

    outcome = rare_event_discovery._barrier_outcome(
        bars,
        entry_index=0,
        direction="LONG",
    )

    assert outcome is not None
    assert outcome["exit_reason"] == "ambiguous_stop"
    assert outcome["gross_bps"] == -30.0
    assert outcome["stress_net_bps"] == -46.0


def test_fixed_campaign_has_five_candidates_and_no_oos_tuning():
    contracts = rare_event_discovery._candidate_contracts()

    assert len(contracts) == 5
    assert len({row[0] for row in contracts}) == 5
    assert rare_event_discovery.BASE_ROUNDTRIP_COST_BPS == 12.0
    assert rare_event_discovery.STRESS_ROUNDTRIP_COST_BPS == 16.0
    assert rare_event_discovery.MAX_TRADES_PER_DAY == 4.0


def test_report_never_creates_runtime_policy(monkeypatch):
    history = _synthetic_history()
    monkeypatch.setattr(
        rare_event_discovery,
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

    report = rare_event_discovery.evaluate_rare_event_discovery("fixture")

    assert report["research_only"] is True
    assert report["runtime_profile_created"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert report["fixed_contract"]["oos_tuning"] is False
    assert len(report["candidates"]) == 5


def test_rare_event_cli_bootstraps_outside_repository(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_bitget_rare_event_discovery_v1.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "five fixed low-turnover" in result.stdout
