from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exia.contracts import CandidateSpec, ExperimentManifest, load_candidate_spec


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = ROOT / "freqtrade_pilot" / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from exia_market_mode import (  # noqa: E402
    MarketMode,
    append_market_mode_columns,
)


def _candles(close: np.ndarray) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC")
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
        }
    )


def test_candidate_spec_is_strict_and_pins_sources() -> None:
    path = (
        ROOT
        / "freqtrade_pilot"
        / "candidates"
        / "exia_market_mode_foundation_v1.json"
    )
    spec = load_candidate_spec(path, root=ROOT)

    assert spec.status == "DEVELOPMENT"
    assert spec.allowed_market_states == tuple(mode.value for mode in MarketMode)
    assert spec.risk["max_open_trades"] == 1
    assert spec.risk["leverage"] == 1
    assert spec.source["paper_allowed"] is False
    assert spec.source["live_allowed"] is False
    assert spec.source["orders_enabled"] is False
    assert spec.source["promotion_authority"] is False


def test_candidate_spec_rejects_unknown_settings_and_live_flags() -> None:
    path = (
        ROOT
        / "freqtrade_pilot"
        / "candidates"
        / "exia_market_mode_foundation_v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["new_runtime_threshold"] = 1
    with pytest.raises(ValueError, match="keys mismatch"):
        CandidateSpec.from_mapping(payload)

    payload.pop("new_runtime_threshold")
    payload["live_allowed"] = True
    with pytest.raises(ValueError, match="safety flags"):
        CandidateSpec.from_mapping(payload)


def test_market_mode_classifies_trends_range_and_shock() -> None:
    count = 900
    up = 100.0 * np.exp(np.arange(count) * 0.0005)
    down = 100.0 * np.exp(np.arange(count) * -0.0005)
    flat = np.full(count, 100.0)

    assert append_market_mode_columns(_candles(up)).iloc[-1][
        "exia_market_mode"
    ] == MarketMode.TREND_UP.value
    assert append_market_mode_columns(_candles(down)).iloc[-1][
        "exia_market_mode"
    ] == MarketMode.TREND_DOWN.value
    assert append_market_mode_columns(_candles(flat)).iloc[-1][
        "exia_market_mode"
    ] == MarketMode.RANGE.value

    shock = flat.copy()
    shock[-1] = 130.0
    result = append_market_mode_columns(_candles(shock))
    assert result.iloc[-1]["exia_market_mode"] == MarketMode.UNSAFE.value
    assert bool(result.iloc[-1]["exia_volatility_shock"]) is True


def test_market_mode_is_prefix_invariant() -> None:
    rng = np.random.default_rng(41)
    first = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=900)))
    future = first[-1] * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=100)))
    prefix = append_market_mode_columns(_candles(first))
    extended = append_market_mode_columns(_candles(np.r_[first, future])).iloc[:900]

    columns = [
        "exia_market_mode",
        "exia_previous_mode",
        "exia_state_age_bars",
        "exia_volatility_threshold",
    ]
    pd.testing.assert_frame_equal(
        prefix[columns].reset_index(drop=True),
        extended[columns].reset_index(drop=True),
    )


def test_market_mode_fails_closed_after_a_data_gap() -> None:
    frame = _candles(100.0 * np.exp(np.arange(900) * 0.0005))
    frame.loc[800:, "date"] += pd.Timedelta(hours=1)
    result = append_market_mode_columns(frame)

    assert result.iloc[800]["exia_market_mode"] == MarketMode.UNSAFE.value
    assert result.iloc[800]["exia_contiguous_bars"] == 1


def test_experiment_manifest_is_non_authoritative() -> None:
    digest = "a" * 64
    payload = {
        "schema_version": "exia.experiment_manifest.v1",
        "experiment_id": "exia_test_001",
        "candidate_id": "exia_market_mode_foundation_v1",
        "candidate_spec_sha256": digest,
        "strategy_sha256": digest,
        "taxonomy_sha256": digest,
        "dataset_sha256": digest,
        "freqtrade_revision": "b" * 40,
        "experiment_code_sha256": digest,
        "runtime_config_sha256": digest,
        "historical_config_sha256": digest,
        "timeranges": {"development": "20220101-20240101"},
        "costs": {"base": 0.0004, "stress": 0.0008},
        "command": ["freqtrade", "backtesting"],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest = ExperimentManifest.from_mapping(payload)
    assert manifest.experiment_id == "exia_test_001"

    payload["orders_enabled"] = True
    with pytest.raises(ValueError, match="cannot authorize"):
        ExperimentManifest.from_mapping(payload)


def test_foundation_strategy_is_no_trade_and_dry_run_only() -> None:
    source = (
        STRATEGY_DIR / "ExiaMarketModeStrategyV1.py"
    ).read_text(encoding="utf-8")

    assert "class ExiaMarketModeStrategyV1(IStrategy)" in source
    assert 'self.config.get("dry_run") is not True' in source
    assert 'dataframe["enter_long"] = 0' in source
    assert 'dataframe["enter_short"] = 0' in source
    assert "return 1.0" in source
    assert '"stoploss_on_exchange": True' in source


def test_distribution_report_is_fail_closed_when_present() -> None:
    path = (
        ROOT
        / "Reports"
        / "Exia"
        / "market_mode_foundation_v1"
        / "distribution.json"
    )
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == "FOUNDATION_READY_FOR_CANDIDATE_DESIGN"
    assert all(report["checks"].values())
    assert report["paper_allowed"] is False
    assert report["live_allowed"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert set(report["observed_states"]) == {
        "TREND_UP",
        "TREND_DOWN",
        "RANGE",
        "UNSAFE",
    }
