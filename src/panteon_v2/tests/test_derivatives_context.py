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


def test_carryflow_short_basis_floor_is_independent_and_diagnostic():
    context = {
        "funding_rate": 0.0002,
        "open_interest_usdt": 1_000_000.0,
        "long_ratio": 0.65,
        "short_ratio": 0.35,
        "mark_price": 99.95,
        "index_price": 100.0,
        "last_price": 100.0,
        "age_sec": 0.0,
        "context_complete": True,
    }

    class Fetcher:
        def get(self, symbol):
            return dict(context)

    actor = CarryFlowAgentV2()
    actor.CHECK_INT = 1
    actor.EMA_FAST = 2
    actor.EMA_SLOW = 6
    actor.EXTREME_EXT = 0.0
    actor.RSI_N = 2
    actor.RSI_OB = 50
    actor.OI_SPIKE = 0.02
    actor.MAX_POS = 1
    panteon_agents.set_fetcher(Fetcher())
    try:
        for bar in range(1, 11):
            actor.act(
                {"BTC": 100.0 + bar},
                {"BTC": 1.0},
                bar_index=bar,
            )
        context["open_interest_usdt"] *= 1.03
        blocked = actor.act(
            {"BTC": 111.0},
            {"BTC": 1.0},
            bar_index=11,
        )
        assert blocked["BTC"] == 0
        assert (
            actor.last_signal_diagnostics["BTC"]["reason"]
            == "short_basis_below_floor"
        )

        actor.SHORT_BASIS_FLOOR = -0.0015
        context["open_interest_usdt"] *= 1.03
        allowed = actor.act(
            {"BTC": 112.0},
            {"BTC": 1.0},
            bar_index=12,
        )
        assert allowed["BTC"] == 7
        assert actor.last_signal_diagnostics["BTC"]["reason"] == "candidate_short"
    finally:
        panteon_agents.set_fetcher(None)


def test_carryflow_normalization_exit_respects_minimum_hold():
    context = {
        "funding_rate": 0.0002,
        "open_interest_usdt": 1_000_000.0,
        "long_ratio": 0.65,
        "short_ratio": 0.35,
        "mark_price": 99.95,
        "index_price": 100.0,
        "last_price": 100.0,
        "age_sec": 0.0,
        "context_complete": True,
    }

    class Fetcher:
        def get(self, symbol):
            return dict(context)

    actor = CarryFlowAgentV2()
    actor.CHECK_INT = 1
    actor.EMA_FAST = 2
    actor.EMA_SLOW = 6
    actor.EXTREME_EXT = 1.0
    actor.RSI_N = 2
    actor.RSI_OB = 50
    actor.OI_SPIKE = 0.02
    actor.MAX_POS = 1
    actor.HOLD = 10
    actor.STOP = 1.0
    actor.TARGET = 1.0
    actor.SHORT_BASIS_FLOOR = -0.0015
    actor.EXIT_ON_NORMALIZATION = True
    actor.MIN_HOLD_BEFORE_NORMALIZATION = 2
    panteon_agents.set_fetcher(Fetcher())
    try:
        for bar in range(1, 11):
            actor.act({"BTC": 100.0 + bar}, {"BTC": 1.0}, bar_index=bar)
        context["open_interest_usdt"] *= 1.03
        opened = actor.act({"BTC": 111.0}, {"BTC": 1.0}, bar_index=11)
        assert opened["BTC"] == 7

        context.update(
            {
                "funding_rate": 0.0,
                "long_ratio": 0.50,
                "short_ratio": 0.50,
                "mark_price": 99.99,
            }
        )
        held = actor.act({"BTC": 111.0}, {"BTC": 1.0}, bar_index=12)
        closed = actor.act({"BTC": 111.0}, {"BTC": 1.0}, bar_index=13)

        assert held["BTC"] == 0
        assert closed["BTC"] == 8
        assert actor.last_signal_diagnostics["BTC"]["normalized"] is True
        assert (
            actor.last_signal_diagnostics["BTC"]["close_trigger"]
            == "normalization"
        )
    finally:
        panteon_agents.set_fetcher(None)
