from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from panteon_v2.app.startup import _close_paper_canary_positions_on_shutdown
from panteon_v2.attribution import AttributionLedger, EventLog
from panteon_v2.domain.types import Action, Regime, Signal
from panteon_v2.execution import (
    FakeExchange,
    PositionTracker,
    RiskLimits,
    SymbolHealthMonitor,
    TradeExecutor,
)
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory


def _pipeline():
    exchange = FakeExchange(name="PAPER")
    tracker = PositionTracker()
    event_log = EventLog()
    ledger = AttributionLedger()
    executor = TradeExecutor(
        exchange=exchange,
        health=SymbolHealthMonitor(),
        risk_limits=RiskLimits(),
        position_tracker=tracker,
        perf=PerformanceMemory(trade_fraction=1.0),
        event_log=event_log,
    )
    pipeline = SimpleNamespace(
        executor=executor,
        current_balance=1000.0,
        event_log=event_log,
        ledger=ledger,
        exchange_name="MEXC",
        mode="paper_live_feed",
    )
    return pipeline, tracker, exchange


def _open_short(pipeline, *, price: float = 100.0) -> None:
    signal = Signal(
        id=1,
        bar=10,
        sym="ETH",
        action=Action.FUT_SHORT_FULL,
        price=price,
        regime=Regime.NEUTRAL,
        by_player="LiveVolCompress",
        by_agent="",
    )
    result = pipeline.executor.execute(signal, balance_usd=1000.0)
    assert result.is_success


def test_paper_canary_shutdown_flatten_closes_owned_virtual_positions():
    pipeline, tracker, exchange = _pipeline()
    _open_short(pipeline, price=100.0)

    summary = _close_paper_canary_positions_on_shutdown(
        pipeline,
        mode="paper_live_feed",
        enabled=True,
        latest_prices={"ETH": 90.0},
        bar=11,
    )

    assert summary["enabled"] is True
    assert summary["attempted"] == 1
    assert summary["closed"] == 1
    assert summary["failed"] == 0
    assert summary["skipped_external"] == 0
    assert tracker.open_count == 0
    assert exchange.get_position("ETH") is None
    assert pipeline.ledger.closed_count == 1
    assert pipeline.ledger.total_realized_pnl > 0.0


def test_paper_canary_shutdown_flatten_is_noop_without_flag():
    pipeline, tracker, _exchange = _pipeline()
    _open_short(pipeline, price=100.0)

    summary = _close_paper_canary_positions_on_shutdown(
        pipeline,
        mode="paper_live_feed",
        enabled=False,
        latest_prices={"ETH": 90.0},
        bar=11,
    )

    assert summary["enabled"] is False
    assert summary["skipped_reason"] == "disabled"
    assert summary["attempted"] == 0
    assert tracker.open_count == 1


def test_paper_canary_shutdown_flatten_is_noop_for_live_mode_even_with_flag():
    pipeline, tracker, _exchange = _pipeline()
    _open_short(pipeline, price=100.0)

    summary = _close_paper_canary_positions_on_shutdown(
        pipeline,
        mode="live_futures",
        enabled=True,
        latest_prices={"ETH": 90.0},
        bar=11,
    )

    assert summary["enabled"] is False
    assert summary["skipped_reason"] == "non_virtual_mode"
    assert summary["attempted"] == 0
    assert tracker.open_count == 1


def test_paper_canary_shutdown_flatten_skips_external_positions():
    pipeline, tracker, _exchange = _pipeline()
    tracker.force_set(
        TrackedPosition(
            open_signal_id=1,
            sym="ETH",
            side="short",
            entry_price=100.0,
            qty=1.0,
            fee_open=0.0,
            by_player="RecoveredExchangePosition",
            by_agent="",
            opened_at=datetime.now(timezone.utc),
            opened_bar=10,
            open_action=Action.FUT_SHORT_FULL.name,
            open_regime=Regime.NEUTRAL.label,
        )
    )

    summary = _close_paper_canary_positions_on_shutdown(
        pipeline,
        mode="paper_live_feed",
        enabled=True,
        latest_prices={"ETH": 90.0},
        bar=11,
    )

    assert summary["attempted"] == 0
    assert summary["closed"] == 0
    assert summary["skipped_external"] == 1
    assert tracker.open_count == 1
