from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone

import pytest

from panteon_v2.app.agent_bootstrap import _ensure_paths
from panteon_v2.policy.derivatives_context import (
    DERIVATIVES_CONTEXT_CSV_FIELDS,
    DerivativesContextError,
    HistoricalDerivativesContext,
)


_ensure_paths()

import panteon_agents
from panteon_agents import CarryFlowAgentV2


NOW = datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc)


def _row(timestamp: datetime, *, funding_rate: float, complete: bool = True):
    return {
        "timestamp_ms": int(timestamp.timestamp() * 1000),
        "datetime_utc": timestamp.isoformat(),
        "symbol": "BTC",
        "source_exchange": "BITGET",
        "market_type": "swap",
        "funding_rate": funding_rate,
        "open_interest_usdt": 1_000_000.0 if complete else "",
        "long_ratio": 0.65 if complete else "",
        "short_ratio": 0.35 if complete else "",
        "mark_price": 101.0 if complete else "",
        "index_price": 100.0 if complete else "",
        "last_price": 100.5 if complete else "",
        "next_funding_ts": 0,
        "long_short_ratio_ts": int(timestamp.timestamp() * 1000),
        "source_updated_ts": timestamp.timestamp(),
        "context_complete": "true" if complete else "false",
        "retrieval_status": "complete" if complete else "missing",
    }


def _write_context(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DERIVATIVES_CONTEXT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_historical_context_is_point_in_time_and_rejects_time_reversal(tmp_path):
    path = tmp_path / "context.csv"
    _write_context(
        path,
        [
            _row(NOW, funding_rate=0.0001),
            _row(NOW + timedelta(minutes=10), funding_rate=0.0002),
        ],
    )
    context = HistoricalDerivativesContext.from_csv(path)

    context.advance(NOW + timedelta(minutes=5))
    first = context.get("BTC/USDT")
    assert first["funding_rate"] == 0.0001
    assert first["age_sec"] == 300.0

    context.advance(NOW + timedelta(minutes=10))
    second = context.get("BTC")
    assert second["funding_rate"] == 0.0002
    assert second["age_sec"] == 0.0

    with pytest.raises(DerivativesContextError, match="moved backwards"):
        context.advance(NOW + timedelta(minutes=9))


def test_historical_context_age_controls_carryflow_freshness(tmp_path):
    path = tmp_path / "context.csv"
    _write_context(path, [_row(NOW, funding_rate=0.0001)])
    context = HistoricalDerivativesContext.from_csv(path)
    actor = CarryFlowAgentV2()
    actor.MAX_DATA_AGE_SEC = 60
    panteon_agents.set_fetcher(context)
    try:
        context.advance(NOW + timedelta(seconds=60))
        assert actor._fresh_funding("BTC")["age_sec"] == 60.0
        context.advance(NOW + timedelta(seconds=61))
        assert actor._fresh_funding("BTC") == {}
    finally:
        panteon_agents.set_fetcher(None)


def test_incomplete_rows_are_retained_but_never_marked_complete(tmp_path):
    path = tmp_path / "context.csv"
    _write_context(path, [_row(NOW, funding_rate=0.0, complete=False)])
    context = HistoricalDerivativesContext.from_csv(path)

    context.advance(NOW)
    payload = context.get("BTC")

    assert payload
    assert payload["context_complete"] is False
    assert payload["open_interest_usdt"] == 0.0


def test_duplicate_symbol_timestamp_is_rejected(tmp_path):
    path = tmp_path / "context.csv"
    row = _row(NOW, funding_rate=0.0001)
    _write_context(path, [row, row])

    with pytest.raises(DerivativesContextError, match="duplicate"):
        HistoricalDerivativesContext.from_csv(path)


def test_wrong_exchange_or_timestamp_contract_is_rejected(tmp_path):
    wrong_exchange = _row(NOW, funding_rate=0.0001)
    wrong_exchange["source_exchange"] = "MEXC"
    path = tmp_path / "wrong_exchange.csv"
    _write_context(path, [wrong_exchange])
    with pytest.raises(DerivativesContextError, match="source_exchange"):
        HistoricalDerivativesContext.from_csv(path)

    wrong_timestamp = _row(NOW, funding_rate=0.0001)
    wrong_timestamp["datetime_utc"] = (NOW + timedelta(seconds=1)).isoformat()
    path = tmp_path / "wrong_timestamp.csv"
    _write_context(path, [wrong_timestamp])
    with pytest.raises(DerivativesContextError, match="does not match"):
        HistoricalDerivativesContext.from_csv(path)
