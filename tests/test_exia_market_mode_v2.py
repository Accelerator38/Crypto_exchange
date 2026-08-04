from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from exia.contracts import load_candidate_spec
from tools.run_exia_experiment_v1 import _taxonomy_api


ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = ROOT / "freqtrade_pilot" / "user_data" / "strategies"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from exia_market_mode_v2 import (  # noqa: E402
    DEFAULT_MARKET_MODE_CONTRACT,
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


def test_v2_contract_fits_bitget_startup_limit() -> None:
    contract = DEFAULT_MARKET_MODE_CONTRACT

    assert contract.volatility_lookback_hours == 720
    assert contract.minimum_history_bars == 744
    assert contract.runtime_startup_bars == 999
    assert contract.runtime_startup_bars <= 999
    assert contract.state_age_cap_bars == 168


def test_v2_restart_preserves_current_state_and_diagnostics() -> None:
    rng = np.random.default_rng(91)
    close = 100.0 * np.exp(
        np.cumsum(rng.normal(0.00015, 0.004, size=4000))
    )
    source = _candles(close)
    full = append_market_mode_columns(source)
    restarted = append_market_mode_columns(
        source.iloc[-DEFAULT_MARKET_MODE_CONTRACT.runtime_startup_bars :]
    )

    for column in (
        "exia_market_mode",
        "exia_previous_mode",
        "exia_volatility_shock",
        "exia_contiguous_bars",
        "exia_state_age_bars",
    ):
        assert restarted.iloc[-1][column] == full.iloc[-1][column]


def test_v2_runtime_burn_in_does_not_create_false_state_onset() -> None:
    count = DEFAULT_MARKET_MODE_CONTRACT.runtime_startup_bars
    close = 100.0 * np.exp(np.arange(count) * 0.0005)
    result = append_market_mode_columns(_candles(close))

    assert result.iloc[-1]["exia_market_mode"] == MarketMode.TREND_UP.value
    assert result.iloc[-1]["exia_previous_mode"] == MarketMode.TREND_UP.value
    assert result.iloc[-1]["exia_state_age_bars"] > 1


def test_v2_candidate_and_runner_pin_the_same_taxonomy() -> None:
    candidate_path = (
        ROOT
        / "freqtrade_pilot"
        / "candidates"
        / "exia_market_mode_foundation_v2.json"
    )
    spec = load_candidate_spec(candidate_path, root=ROOT)
    order, classify = _taxonomy_api(spec)

    assert spec.taxonomy["version"] == "exia.market_mode.v2"
    assert spec.strategy["class_name"] == "ExiaMarketModeStrategyV2"
    assert order == ("TREND_UP", "TREND_DOWN", "RANGE", "UNSAFE")
    assert classify.__module__.startswith("_exia_taxonomy_9643eec662c9")
    assert spec.source["paper_allowed"] is False
    assert spec.source["live_allowed"] is False
    assert spec.source["orders_enabled"] is False
    assert spec.source["promotion_authority"] is False


def test_v2_foundation_strategy_uses_burn_in_and_never_enters() -> None:
    source = (STRATEGY_DIR / "ExiaMarketModeStrategyV2.py").read_text(
        encoding="utf-8"
    )

    assert "runtime_startup_bars" in source
    assert 'self.config.get("dry_run") is not True' in source
    assert 'dataframe["enter_long"] = 0' in source
    assert 'dataframe["enter_short"] = 0' in source
    assert "return 1.0" in source
    assert '"stoploss_on_exchange": True' in source


def test_generated_v2_foundation_report_is_fail_closed_when_present() -> None:
    path = (
        ROOT
        / "Reports"
        / "Exia"
        / "market_mode_foundation_v2"
        / "distribution.json"
    )
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["status"] == "FOUNDATION_READY_FOR_CANDIDATE_DESIGN"
    assert all(report["checks"].values())
    assert all(item["passed"] for item in report["restart_parity"].values())
    assert report["paper_allowed"] is False
    assert report["live_allowed"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
