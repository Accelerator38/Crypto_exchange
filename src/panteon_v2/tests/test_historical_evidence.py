from __future__ import annotations

from datetime import datetime, timezone

from panteon_v2.policy.historical_evidence import (
    HISTORICAL_CAPABILITY_SCHEMA_VERSION,
    audit_bitget_historical_capabilities,
)


class _Exchange:
    def __init__(self, *, open_interest_history: bool) -> None:
        self.has = {
            "fetchOHLCV": True,
            "fetchMarkOHLCV": True,
            "fetchIndexOHLCV": True,
            "fetchFundingRateHistory": True,
            "fetchLongShortRatioHistory": True,
            "fetchOpenInterestHistory": open_interest_history,
        }

    @staticmethod
    def fetch_ohlcv(_symbol, *, timeframe, limit):
        assert timeframe == "1h"
        assert limit == 2
        return [[1_700_000_000_000, 1, 2, 1, 2, 3]]

    fetch_mark_ohlcv = fetch_ohlcv
    fetch_index_ohlcv = fetch_ohlcv

    @staticmethod
    def fetch_funding_rate_history(_symbol, *, limit):
        assert limit == 2
        return [{"timestamp": 1_700_000_000_000}]

    @staticmethod
    def fetch_long_short_ratio_history(_symbol, *, timeframe, limit):
        assert timeframe == "1h"
        assert limit == 2
        return [{"timestamp": 1_700_000_000_000}]

    @staticmethod
    def fetch_open_interest_history(_symbol, *, timeframe, limit):
        assert timeframe == "1h"
        assert limit == 2
        return [{"timestamp": 1_700_000_000_000}]


def _audit(open_interest_history: bool):
    return audit_bitget_historical_capabilities(
        _Exchange(open_interest_history=open_interest_history),
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        generated_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )


def test_historical_audit_hard_blocks_missing_open_interest_history():
    report = _audit(False)

    assert report["schema_version"] == HISTORICAL_CAPABILITY_SCHEMA_VERSION
    assert report["research_backfill_possible"] is True
    assert report["promotion_backfill_possible"] is False
    assert report["evidence_tape_creation_allowed"] is False
    assert report["allowed_use"] == "diagnostic_only"
    assert report["primary_failure"] == "historical_open_interest_unavailable"
    assert report["missing_required_fields"] == ["open_interest"]
    assert report["capabilities"]["open_interest"]["status"] == "not_supported"
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False


def test_historical_audit_requires_timestamped_history_for_every_field():
    report = _audit(True)

    assert report["missing_required_fields"] == []
    assert report["promotion_backfill_possible"] is True
    assert report["evidence_tape_creation_allowed"] is True
    assert report["allowed_use"] == "promotion_evidence"
    assert all(
        row["status"] == "available"
        for row in report["capabilities"].values()
    )
