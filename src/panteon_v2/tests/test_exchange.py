"""Тесты Exchange Protocol и FakeExchange."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import (
    Exchange,
    FakeExchange,
    OrderResult,
    OrderStatus,
)


def _make_signal(sid: int = 1, sym: str = "BTC",
                 action: Action = Action.FUT_LONG_FULL,
                 price: float = 100.0,
                 close_fraction: float = 1.0) -> Signal:
    return Signal(
        id=sid, bar=1, sym=sym, action=action,
        price=price, regime=Regime.BULLISH,
        by_player="P", by_agent="A",
        close_fraction=close_fraction,
    )


class TestFakeExchange(unittest.TestCase):
    def test_protocol_satisfied(self):
        ex = FakeExchange()
        self.assertIsInstance(ex, Exchange)

    def test_default_fill(self):
        ex = FakeExchange()
        sig = _make_signal()
        result = ex.send_order(sig, qty=0.1)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertIsNotNone(result.trade)
        self.assertEqual(result.trade.signal_id, 1)
        # Position должна появиться
        self.assertIsNotNone(ex.get_position("BTC"))

    def test_close_after_open(self):
        ex = FakeExchange()
        op_sig = _make_signal(sid=1, action=Action.FUT_LONG_FULL)
        ex.send_order(op_sig, qty=0.1)
        cl_sig = _make_signal(sid=2, action=Action.FUT_CLOSE_ALL, price=110.0)
        result = ex.send_order(cl_sig, qty=0.1)
        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertIsNone(ex.get_position("BTC"))

    def test_spot_sell_all_does_not_close_futures_position_when_family_strict(self):
        ex = FakeExchange(strict_close_action_family=True)
        ex.send_order(_make_signal(sid=1, action=Action.FUT_LONG_FULL), qty=0.1)
        cl_sig = _make_signal(sid=2, action=Action.SPOT_SELL_ALL, price=110.0)

        result = ex.send_order(cl_sig, qty=0.1)

        self.assertEqual(result.status, OrderStatus.FILLED)
        position = ex.get_position("BTC")
        self.assertIsNotNone(position)
        self.assertEqual(position.side, "long")

    def test_default_close_family_keeps_legacy_shadow_exchange_semantics(self):
        ex = FakeExchange()
        ex.send_order(_make_signal(sid=1, action=Action.FUT_LONG_FULL), qty=0.1)
        cl_sig = _make_signal(sid=2, action=Action.SPOT_SELL_ALL, price=110.0)

        result = ex.send_order(cl_sig, qty=0.1)

        self.assertEqual(result.status, OrderStatus.FILLED)
        self.assertIsNone(ex.get_position("BTC"))

    def test_close_without_open_rejected(self):
        ex = FakeExchange()
        cl_sig = _make_signal(sid=1, action=Action.FUT_CLOSE_ALL)
        result = ex.send_order(cl_sig, qty=0.1)
        self.assertEqual(result.status, OrderStatus.REJECTED)

    def test_configure_reject(self):
        ex = FakeExchange()
        ex.configure_reject("BTC", count=2)
        sig1 = _make_signal(sid=1)
        sig2 = _make_signal(sid=2)
        sig3 = _make_signal(sid=3)
        self.assertEqual(ex.send_order(sig1, qty=0.1).status, OrderStatus.REJECTED)
        self.assertEqual(ex.send_order(sig2, qty=0.1).status, OrderStatus.REJECTED)
        # 3-й уже без forced — должен пройти
        self.assertEqual(ex.send_order(sig3, qty=0.1).status, OrderStatus.FILLED)

    def test_configure_pending(self):
        ex = FakeExchange()
        ex.configure_pending("BTC")
        result = ex.send_order(_make_signal(), qty=0.1)
        self.assertEqual(result.status, OrderStatus.PENDING)
        self.assertIn("FAKE-PENDING", result.exchange_order_id)
        self.assertIsNone(result.trade)

    def test_pending_can_fill_on_poll(self):
        ex = FakeExchange()
        ex.configure_pending_fill("BTC", after_polls=1)
        sig = _make_signal()

        pending = ex.send_order(sig, qty=0.1)
        filled = ex.poll_order(pending.exchange_order_id, sig)

        self.assertEqual(pending.status, OrderStatus.PENDING)
        self.assertEqual(filled.status, OrderStatus.FILLED)
        self.assertIsNotNone(ex.get_position("BTC"))

    def test_pending_partial_close_keeps_remaining_position(self):
        ex = FakeExchange()
        ex.send_order(_make_signal(sid=1, action=Action.FUT_LONG_FULL), qty=0.1)
        ex.configure_pending_fill("BTC", after_polls=1)
        sig = _make_signal(
            sid=2,
            action=Action.FUT_CLOSE_ALL,
            price=110.0,
            close_fraction=0.4,
        )

        pending = ex.send_order(sig, qty=0.04)
        filled = ex.poll_order(pending.exchange_order_id, sig)

        self.assertEqual(filled.status, OrderStatus.FILLED)
        position = ex.get_position("BTC")
        self.assertIsNotNone(position)
        self.assertAlmostEqual(position.qty, 0.06)

    def test_pending_default_close_all_clears_position_even_when_qty_is_smaller(self):
        ex = FakeExchange()
        ex.send_order(_make_signal(sid=1, action=Action.FUT_LONG_FULL), qty=0.1)
        ex.configure_pending_fill("BTC", after_polls=1)
        sig = _make_signal(sid=2, action=Action.FUT_CLOSE_ALL, price=110.0)

        pending = ex.send_order(sig, qty=0.04)
        filled = ex.poll_order(pending.exchange_order_id, sig)

        self.assertEqual(filled.status, OrderStatus.FILLED)
        self.assertIsNone(ex.get_position("BTC"))

    def test_spot_sell_all_keeps_remaining_position_when_qty_is_smaller(self):
        ex = FakeExchange()
        ex.send_order(_make_signal(sid=1, action=Action.FUT_LONG_FULL), qty=0.1)
        sig = _make_signal(sid=2, action=Action.SPOT_SELL_ALL, price=110.0)

        result = ex.send_order(sig, qty=0.04)

        self.assertEqual(result.status, OrderStatus.FILLED)
        position = ex.get_position("BTC")
        self.assertIsNotNone(position)
        self.assertAlmostEqual(position.qty, 0.06)

    def test_pending_can_reject_on_poll(self):
        ex = FakeExchange()
        ex.configure_pending_reject("BTC", after_polls=1)
        sig = _make_signal()

        pending = ex.send_order(sig, qty=0.1)
        rejected = ex.poll_order(pending.exchange_order_id, sig)

        self.assertEqual(pending.status, OrderStatus.PENDING)
        self.assertEqual(rejected.status, OrderStatus.REJECTED)
        self.assertIsNone(rejected.trade)

    def test_slippage_long(self):
        ex = FakeExchange()
        ex.set_slippage_pct(0.001)  # 0.1%
        sig = _make_signal(price=100.0, action=Action.FUT_LONG_FULL)
        result = ex.send_order(sig, qty=0.1)
        self.assertGreater(result.trade.fill_price, 100.0)
        self.assertAlmostEqual(result.trade.fill_price, 100.1, places=2)

    def test_slippage_short(self):
        ex = FakeExchange()
        ex.set_slippage_pct(0.001)
        sig = _make_signal(price=100.0, action=Action.FUT_SHORT_FULL)
        result = ex.send_order(sig, qty=0.1)
        self.assertLess(result.trade.fill_price, 100.0)

    def test_orders_log(self):
        ex = FakeExchange()
        ex.send_order(_make_signal(sid=1), qty=0.1)
        ex.configure_reject("ETH")
        ex.send_order(_make_signal(sid=2, sym="ETH"), qty=0.1)
        log = ex.orders_log
        self.assertEqual(len(log), 2)


class TestOrderResultValidation(unittest.TestCase):
    def test_filled_requires_trade(self):
        with self.assertRaises(ValueError):
            OrderResult(status=OrderStatus.FILLED, signal_id=1, sym="BTC")

    def test_signal_id_mismatch_rejected(self):
        from panteon_v2.domain.types import Trade
        bad_trade = Trade(signal_id=999, bar=1, sym="BTC", side="long",
                          qty=0.1, fill_price=100.0, fee=0.0)
        with self.assertRaises(ValueError):
            OrderResult(status=OrderStatus.FILLED, signal_id=1, sym="BTC",
                        trade=bad_trade)


if __name__ == "__main__":
    unittest.main(verbosity=2)
