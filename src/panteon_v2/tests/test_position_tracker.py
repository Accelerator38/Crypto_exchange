"""Тесты PositionTracker."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from panteon_v2.attribution.events import PositionClosed, PositionOpened
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import PositionTracker, TrackedPosition


def _open_signal(sid: int = 1, sym: str = "BTC",
                 action: Action = Action.FUT_LONG_FULL) -> Signal:
    return Signal(
        id=sid, bar=1, sym=sym, action=action, price=100.0,
        regime=Regime.BULLISH, by_player="Player", by_agent="Agent",
    )


def _close_signal(sid: int = 2, sym: str = "BTC") -> Signal:
    return Signal(
        id=sid, bar=2, sym=sym, action=Action.FUT_CLOSE_ALL, price=110.0,
        regime=Regime.BULLISH, by_player="Player", by_agent="Agent",
    )


def _trade(sig: Signal, fill_price: float = None, qty: float = 1.0,
           fee: float = 0.0) -> Trade:
    side = sig.action.side or "long"
    return Trade(
        signal_id=sig.id, bar=sig.bar, sym=sig.sym, side=side,
        qty=qty, fill_price=fill_price or sig.price, fee=fee,
    )


class TestPositionTracker(unittest.TestCase):
    def test_initial_empty(self):
        tracker = PositionTracker()
        self.assertEqual(tracker.open_count, 0)
        self.assertFalse(tracker.has("BTC"))
        self.assertEqual(tracker.all_open(), {})

    def test_on_open_emits_event(self):
        tracker = PositionTracker()
        sig = _open_signal()
        events = tracker.on_open(signal=sig, trade=_trade(sig))
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], PositionOpened)
        self.assertEqual(events[0].sym, "BTC")
        self.assertEqual(events[0].signal_id, 1)
        self.assertTrue(tracker.has("BTC"))
        self.assertEqual(tracker.open_count, 1)
        self.assertEqual(tracker.get("BTC").opened_bar, sig.bar)

    def test_on_open_ignores_close_action(self):
        tracker = PositionTracker()
        sig = _close_signal(sid=1)
        events = tracker.on_open(signal=sig, trade=_trade(sig))
        self.assertEqual(events, [])

    def test_on_close_emits_pnl(self):
        tracker = PositionTracker()
        op_sig = _open_signal(sid=1)
        cl_sig = _close_signal(sid=2)
        tracker.on_open(signal=op_sig, trade=_trade(op_sig, fill_price=100.0))
        events = tracker.on_close(
            signal=cl_sig,
            trade=_trade(cl_sig, fill_price=110.0),
        )
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertIsInstance(ev, PositionClosed)
        self.assertEqual(ev.open_signal_id, 1)
        self.assertEqual(ev.close_signal_id, 2)
        self.assertEqual(ev.entry, 100.0)
        self.assertEqual(ev.exit, 110.0)
        # qty=1, long: pnl = (110 - 100) × 1 = 10
        self.assertAlmostEqual(ev.realized_pnl, 10.0, places=5)
        # Атрибуция к открывающему
        self.assertEqual(ev.by_player, "Player")
        self.assertEqual(ev.by_agent, "Agent")
        # Position очистилась
        self.assertFalse(tracker.has("BTC"))

    def test_on_close_short(self):
        tracker = PositionTracker()
        op_sig = _open_signal(sid=1, action=Action.FUT_SHORT_FULL)
        tracker.on_open(signal=op_sig, trade=_trade(op_sig, fill_price=100.0))
        cl_sig = _close_signal(sid=2)
        events = tracker.on_close(
            signal=cl_sig, trade=_trade(cl_sig, fill_price=90.0),
        )
        # short от 100 до 90 = +10 на 1 qty
        self.assertAlmostEqual(events[0].realized_pnl, 10.0, places=5)

    def test_on_close_with_fees(self):
        tracker = PositionTracker()
        op_sig = _open_signal(sid=1)
        tracker.on_open(
            signal=op_sig,
            trade=_trade(op_sig, fill_price=100.0, fee=0.5),
        )
        cl_sig = _close_signal(sid=2)
        events = tracker.on_close(
            signal=cl_sig,
            trade=_trade(cl_sig, fill_price=110.0, fee=0.55),
        )
        # Gross pnl = 10, net = 10 - 0.5 - 0.55 = 8.95
        self.assertAlmostEqual(events[0].realized_pnl, 8.95, places=5)

    def test_close_without_open_no_event(self):
        tracker = PositionTracker()
        cl_sig = _close_signal(sid=1)
        events = tracker.on_close(signal=cl_sig, trade=_trade(cl_sig))
        self.assertEqual(events, [])

    def test_double_open_ignored(self):
        tracker = PositionTracker()
        sig1 = _open_signal(sid=1)
        sig2 = _open_signal(sid=2)  # тот же sym
        tracker.on_open(signal=sig1, trade=_trade(sig1))
        # Второй open не должен заменить первый
        events = tracker.on_open(signal=sig2, trade=_trade(sig2))
        self.assertEqual(events, [])
        # Первый по-прежнему trackится
        pos = tracker.get("BTC")
        self.assertEqual(pos.open_signal_id, 1)

    def test_force_set_and_remove(self):
        tracker = PositionTracker()
        pos = TrackedPosition(
            open_signal_id=99, sym="ETH", side="long",
            entry_price=200.0, qty=0.5, fee_open=0.0,
            by_player="X", by_agent="Y",
            opened_at=datetime.now(timezone.utc),
        )
        tracker.force_set(pos)
        self.assertTrue(tracker.has("ETH"))
        removed = tracker.force_remove("ETH")
        self.assertEqual(removed.open_signal_id, 99)

    def test_clear(self):
        tracker = PositionTracker()
        sig = _open_signal()
        tracker.on_open(signal=sig, trade=_trade(sig))
        tracker.clear()
        self.assertEqual(tracker.open_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
