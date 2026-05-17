"""Tests for live-state guards before real execution."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from panteon_v2.app.live_state import (
    filter_real_signals_against_tracker,
    reconcile_tracker_with_exchange,
    sync_player_agents_to_real_positions,
)
from panteon_v2.attribution import EventLog, PositionClosed
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import ExchangePosition, PositionTracker
from panteon_v2.execution.position_tracker import TrackedPosition


def _signal(sid: int, action: Action, *, price: float = 100.0) -> Signal:
    return Signal(
        id=sid,
        bar=1,
        sym="BTC",
        action=action,
        price=price,
        regime=Regime.BULLISH,
        by_player="Leader",
        by_agent="Agent",
    )


def _external_position(sym: str = "BTC", side: str = "long") -> TrackedPosition:
    return TrackedPosition(
        open_signal_id=0,
        sym=sym,
        side=side,
        entry_price=100.0,
        qty=0.1,
        fee_open=0.0,
        by_player="RecoveredExchangePosition",
        by_agent="",
        opened_at=datetime.now(timezone.utc),
    )


def _pipeline_for_tracker(tracker: PositionTracker):
    class Executor:
        _tracker = tracker

    class Pipeline:
        executor = Executor()

    return Pipeline()


class _Player:
    agents = []


class TestLiveSignalGuard(unittest.TestCase):
    def test_drops_new_open_when_position_capacity_is_full(self):
        tracker = PositionTracker()
        for sid, sym in ((1, "BTC"), (2, "ETH")):
            open_sig = _signal(sid, Action.FUT_LONG_FULL, price=100.0)
            object.__setattr__(open_sig, "sym", sym)
            tracker.on_open(
                signal=open_sig,
                trade=Trade(
                    signal_id=sid,
                    bar=1,
                    sym=sym,
                    side="long",
                    qty=0.1,
                    fill_price=100.0,
                    fee=0.01,
                ),
            )
        new_open = _signal(3, Action.FUT_LONG_FULL, price=20.0)
        object.__setattr__(new_open, "sym", "SOL")

        result = filter_real_signals_against_tracker(
            [new_open],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
            max_open_positions=2,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.max_position_saturated_opens, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn("max_position_saturated:SOL:Agent", result.details)

    def test_opposite_open_becomes_close_instead_of_duplicate_drop(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_LONG_FULL, price=100.0)
        open_trade = Trade(
            signal_id=1,
            bar=1,
            sym="BTC",
            side="long",
            qty=0.1,
            fill_price=100.0,
            fee=0.01,
        )
        tracker.on_open(signal=open_sig, trade=open_trade)

        class Executor:
            _tracker = tracker

        class Pipeline:
            executor = Executor()

        class Player:
            agents = []

        result = filter_real_signals_against_tracker(
            [_signal(2, Action.FUT_SHORT_FULL, price=95.0)],
            player=Player(),
            pipeline=Pipeline(),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.duplicate_opens, 0)
        self.assertEqual(len(result.signals), 1)
        self.assertEqual(result.signals[0].action, Action.FUT_CLOSE_ALL)
        self.assertEqual(result.signals[0].id, 2)
        self.assertIn("reverse_close:BTC:Agent", result.details)

    def test_drops_close_for_external_recovered_position(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position())

        result = filter_real_signals_against_tracker(
            [_signal(2, Action.FUT_CLOSE_ALL, price=95.0)],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.external_position_signals, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn("external_position_signal:BTC:Agent", result.details)

    def test_drops_opposite_open_on_external_instead_of_reverse_close(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position(side="long"))

        result = filter_real_signals_against_tracker(
            [_signal(2, Action.FUT_SHORT_FULL, price=95.0)],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.external_position_signals, 1)
        self.assertNotIn("reverse_close:BTC:Agent", result.details)

    def test_sync_skips_external_recovered_positions_for_agents(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position())

        class Agent:
            pos = {"BTC": "short"}
            ep = {"BTC": 90.0}
            et = {"BTC": 1}

        class Player:
            agents = [Agent()]

        summary = sync_player_agents_to_real_positions(
            Player(),
            _pipeline_for_tracker(tracker),
            bar_index=2,
            market_symbols=["BTC"],
        )

        self.assertEqual(summary["real_positions"], 1)
        self.assertEqual(summary["agent_positions"], 0)
        self.assertEqual(summary["external_positions_skipped"], 1)
        self.assertIsNone(Agent.pos["BTC"])
        self.assertEqual(Agent.ep["BTC"], 0.0)
        self.assertEqual(Agent.et["BTC"], 0)


class TestLiveExchangeReconcile(unittest.TestCase):
    def test_reconcile_adds_exchange_position_missing_from_tracker(self):
        tracker = PositionTracker()

        class Exchange:
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(
                        sym="BTC",
                        side="long",
                        qty=0.25,
                        entry=101.0,
                    )
                }

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()

        summary = reconcile_tracker_with_exchange(Pipeline(), bar_index=7)

        self.assertEqual(summary["added"], 1)
        pos = tracker.get("BTC")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.by_player, "RecoveredExchangePosition")
        self.assertAlmostEqual(pos.qty, 0.25)
        self.assertAlmostEqual(pos.entry_price, 101.0)

    def test_reconcile_external_close_keeps_forensic_context(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position())
        event_log = EventLog()

        class Exchange:
            def get_all_positions(self):
                return {}

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()
            exchange_name = "BITGET"
            timeframe = "bridge_poll"
            mode = "live_futures"
            run_id = "run-7"
            session_id = "session-7"

        pipeline = Pipeline()
        pipeline.event_log = event_log
        summary = reconcile_tracker_with_exchange(pipeline, bar_index=7)

        self.assertEqual(summary["removed"], 1)
        closed = list(event_log.query(event_types=[PositionClosed]))
        self.assertEqual(len(closed), 1)
        ev = closed[0]
        self.assertEqual(ev.by_player, "RecoveredExchangePosition")
        self.assertEqual(ev.decision_id, "exchange-reconcile-7-BTC")
        self.assertEqual(ev.exchange, "BITGET")
        self.assertEqual(ev.symbol, "BTC")
        self.assertEqual(ev.timeframe, "bridge_poll")
        self.assertEqual(ev.mode, "live_futures")
        self.assertEqual(ev.run_id, "run-7")
        self.assertEqual(ev.session_id, "session-7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
