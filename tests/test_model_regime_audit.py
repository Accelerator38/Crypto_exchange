from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from simple_research.regimes import (
    REGIME_ORDER,
    attribute_trade_regimes,
    build_market_regime_lookup,
)


ROOT = Path(__file__).resolve().parents[1]


def test_regime_lookup_is_causal_and_trade_attribution_is_lagged() -> None:
    timestamps = pd.Series(range(5000), dtype="int64") * 3_600_000
    close = pd.Series(range(5000), dtype="float64") * 0.1 + 100.0
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTC/USDT",
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 1.0,
        }
    )
    lookup = build_market_regime_lookup(frame)
    ledger = pd.DataFrame(
        {
            "entry_timestamp": [timestamps.iloc[-1] + 3_600_000],
            "net_bps": [1.0],
        }
    )

    attributed = attribute_trade_regimes(ledger, lookup)

    assert attributed.loc[0, "market_regime"] == lookup[int(timestamps.iloc[-1])]
    assert attributed.loc[0, "market_regime"] in REGIME_ORDER


def test_model_regime_audit_is_fail_closed_when_present() -> None:
    path = ROOT / "Reports" / "ModelRegimeAudit" / "current_models_by_regime_v1.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["paper_allowed"] is False
    assert report["live_allowed"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert report["regime_contract"]["used_by_strategy"] is False
    assert report["confirmed_specializations"] == []
    assert report["model_inventory"]["evaluated"] == 10
    assert report["model_inventory"]["uniformly_ranked"] == 8

    for model in report["models"].values():
        for costs in model["windows"].values():
            for metrics in costs.values():
                regime_count = sum(
                    regime["closed_trades"]
                    for regime in metrics["regimes"].values()
                )
                assert regime_count == metrics["overall"]["closed_trades"]
                assert metrics["regimes"]["unknown"]["closed_trades"] == 0
