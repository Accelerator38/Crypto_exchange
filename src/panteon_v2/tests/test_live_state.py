"""Tests for live-state guards before real execution."""

from __future__ import annotations

import unittest

from panteon_v2.app.live_state import (
    filter_real_signals_against_tracker,
    reconcile_tracker_with_exchange,
)
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import ExchangePosition, PositionTracker


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


class TestLiveSignalGuard(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
