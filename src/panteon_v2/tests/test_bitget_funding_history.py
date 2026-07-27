from __future__ import annotations

import json

import pytest

from panteon_v2.analysis.bitget_funding_history import (
    BITGET_FUNDING_HISTORY_ENDPOINT,
    BITGET_INSTRUMENTS_ENDPOINT,
    FundingHistoryError,
    build_or_verify_funding_snapshot,
    load_funding_snapshot,
    verify_funding_snapshot,
)


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _fake_get(endpoint, *, params, timeout):
    del timeout
    if endpoint == BITGET_INSTRUMENTS_ENDPOINT:
        return _Response(
            {
                "code": "00000",
                "data": [
                    {
                        "symbol": "BTCUSDT",
                        "fundInterval": "8",
                        "status": "online",
                    }
                ],
            }
        )
    assert endpoint == BITGET_FUNDING_HISTORY_ENDPOINT
    assert params["symbol"] == "BTCUSDT"
    return _Response(
        {
            "code": "00000",
            "data": {
                "resultList": [
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.001",
                        "fundingRateTimestamp": "1782864000000",
                    },
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0005",
                        "fundingRateTimestamp": "1782835200000",
                    },
                    {
                        "symbol": "BTCUSDT",
                        "fundingRate": "0.0004",
                        "fundingRateTimestamp": "1782806400000",
                    },
                ]
            },
        }
    )


def test_builds_sealed_recent_snapshot_without_promotion_authority(tmp_path):
    manifest = build_or_verify_funding_snapshot(
        tmp_path,
        symbols=["BTC"],
        required_start_at="2022-01-01T00:00:00+00:00",
        required_end_at="2026-07-14T23:00:00+00:00",
        request_get=_fake_get,
        sleep_seconds=0.0,
    )

    assert manifest["verified"] is True
    assert manifest["registered_oos_coverage_complete"] is False
    assert manifest["allowed_use"] == "recent_diagnostic_screen_only"
    assert manifest["orders_enabled"] is False
    assert manifest["promotion_authority"] is False
    assert manifest["coverage"]["BTC"]["observed_gap_hours"] == {"8": 2}

    funding, loaded = load_funding_snapshot(
        tmp_path,
        expected_symbols=["BTC"],
    )
    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]
    assert funding["BTC"][1782864000000] == 0.001


def test_snapshot_verification_detects_tampering(tmp_path):
    build_or_verify_funding_snapshot(
        tmp_path,
        symbols=["BTC"],
        required_start_at="2022-01-01T00:00:00+00:00",
        required_end_at="2026-07-14T23:00:00+00:00",
        request_get=_fake_get,
        sleep_seconds=0.0,
    )
    path = tmp_path / "funding_btcusdt.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(FundingHistoryError, match="SHA mismatch"):
        verify_funding_snapshot(tmp_path, expected_symbols=["BTC"])


def test_nonempty_unsealed_directory_fails_closed(tmp_path):
    (tmp_path / "partial.json").write_text(
        json.dumps({"partial": True}),
        encoding="utf-8",
    )

    with pytest.raises(FundingHistoryError, match="not sealed"):
        build_or_verify_funding_snapshot(
            tmp_path,
            symbols=["BTC"],
            required_start_at="2022-01-01T00:00:00+00:00",
            required_end_at="2026-07-14T23:00:00+00:00",
            request_get=_fake_get,
        )
