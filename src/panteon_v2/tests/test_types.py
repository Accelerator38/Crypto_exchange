"""Тесты core types: Action, Regime, Signal, Trade, Metrics, MarketSnapshot.

Цель: зафиксировать гарантии immutability и валидации, чтобы любой
ошибочный конструктор падал в тестах, а не в production.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from panteon_v2.domain.types import (
    Action,
    Regime,
    Signal,
    Trade,
    Metrics,
    MarketSnapshot,
)


def test_market_snapshot_accepts_immutable_technical_indicators():
    from panteon_v2.domain.types import TechnicalIndicators

    snapshot = MarketSnapshot(
        bar=1,
        timestamp=datetime.now(timezone.utc),
        regime=Regime.BULLISH,
        prices={"BTC/USDT": 100.0},
        volumes={"BTC/USDT": 1000.0},
        technicals_by_symbol={
            "BTC/USDT": TechnicalIndicators(
                rsi_14=61.0,
                macd_line_pct=0.25,
                macd_signal_pct=0.15,
                macd_histogram_pct=0.10,
                atr_14_pct=1.75,
            )
        },
    )

    assert snapshot.technicals_by_symbol["BTC/USDT"].rsi_14 == 61.0
    assert snapshot.technicals_by_symbol["BTC/USDT"].atr_14_pct == 1.75


class TestAction(unittest.TestCase):
    def test_open_close_predicates(self):
        opens = [
            Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL,
            Action.FUT_LONG_HALF, Action.FUT_LONG_FULL,
            Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL,
        ]
        closes = [Action.SPOT_SELL_ALL, Action.FUT_CLOSE_ALL]
        for a in opens:
            self.assertTrue(a.is_open, f"{a} should be open")
            self.assertFalse(a.is_close, f"{a} should NOT be close")
        for a in closes:
            self.assertFalse(a.is_open, f"{a} should NOT be open")
            self.assertTrue(a.is_close, f"{a} should be close")
        self.assertFalse(Action.HOLD.is_open)
        self.assertFalse(Action.HOLD.is_close)
        self.assertTrue(Action.HOLD.is_hold)

    def test_side(self):
        for a in (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL,
                  Action.FUT_LONG_HALF, Action.FUT_LONG_FULL):
            self.assertEqual(a.side, "long")
        for a in (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL):
            self.assertEqual(a.side, "short")
        self.assertEqual(Action.HOLD.side, "")
        self.assertEqual(Action.SPOT_SELL_ALL.side, "")
        self.assertEqual(Action.FUT_CLOSE_ALL.side, "")

    def test_fraction(self):
        self.assertEqual(Action.SPOT_BUY_HALF.fraction, 0.5)
        self.assertEqual(Action.SPOT_BUY_FULL.fraction, 1.0)
        self.assertEqual(Action.FUT_LONG_HALF.fraction, 0.5)
        self.assertEqual(Action.FUT_SHORT_FULL.fraction, 1.0)
        self.assertEqual(Action.HOLD.fraction, 0.0)
        self.assertEqual(Action.SPOT_SELL_ALL.fraction, 0.0)

    def test_int_compatibility(self):
        # Совместимость с v1 (где actions — int)
        self.assertEqual(int(Action.HOLD), 0)
        self.assertEqual(int(Action.FUT_LONG_FULL), 5)
        self.assertEqual(int(Action.FUT_CLOSE_ALL), 8)


class TestRegime(unittest.TestCase):
    def test_canonical_strings(self):
        for r in Regime:
            self.assertEqual(Regime.from_string(r.label), r)

    def test_aliases(self):
        self.assertEqual(Regime.from_string("bull"), Regime.BULLISH)
        self.assertEqual(Regime.from_string("UP"), Regime.BULLISH)
        self.assertEqual(Regime.from_string("uptrend"), Regime.BULLISH)
        self.assertEqual(Regime.from_string("bear"), Regime.BEARISH)
        self.assertEqual(Regime.from_string("down"), Regime.BEARISH)
        self.assertEqual(Regime.from_string("sideways"), Regime.NEUTRAL)
        self.assertEqual(Regime.from_string("range"), Regime.NEUTRAL)
        self.assertEqual(Regime.from_string("flash_crash"), Regime.CRASH)
        self.assertEqual(Regime.from_string(""), Regime.NEUTRAL)
        self.assertEqual(Regime.from_string("garbage"), Regime.NEUTRAL)


class TestSignal(unittest.TestCase):
    def test_valid(self):
        sig = Signal(
            id=1, bar=100, sym="BTC", action=Action.FUT_LONG_FULL,
            price=50000.0, regime=Regime.BULLISH, by_player="Test",
        )
        self.assertEqual(sig.sym, "BTC")
        self.assertTrue(sig.action.is_open)

    def test_negative_id_rejected(self):
        with self.assertRaises(ValueError):
            Signal(id=-1, bar=100, sym="BTC", action=Action.HOLD,
                   price=1.0, regime=Regime.NEUTRAL, by_player="X")

    def test_empty_sym_rejected(self):
        with self.assertRaises(ValueError):
            Signal(id=1, bar=100, sym="", action=Action.HOLD,
                   price=1.0, regime=Regime.NEUTRAL, by_player="X")

    def test_negative_price_rejected(self):
        with self.assertRaises(ValueError):
            Signal(id=1, bar=100, sym="BTC", action=Action.HOLD,
                   price=-1.0, regime=Regime.NEUTRAL, by_player="X")

    def test_empty_player_rejected(self):
        with self.assertRaises(ValueError):
            Signal(id=1, bar=100, sym="BTC", action=Action.HOLD,
                   price=1.0, regime=Regime.NEUTRAL, by_player="")

    def test_immutable(self):
        sig = Signal(id=1, bar=100, sym="BTC", action=Action.HOLD,
                     price=1.0, regime=Regime.NEUTRAL, by_player="X")
        with self.assertRaises(Exception):  # FrozenInstanceError или AttributeError
            sig.sym = "ETH"  # type: ignore[misc]


class TestTrade(unittest.TestCase):
    def test_valid(self):
        t = Trade(signal_id=1, bar=100, sym="BTC", side="long",
                  qty=0.001, fill_price=50000.0, fee=0.05)
        self.assertEqual(t.notional, 50.0)

    def test_signal_id_required(self):
        # signal_id обязателен — ключевая гарантия v2
        with self.assertRaises(ValueError):
            Trade(signal_id=-1, bar=100, sym="BTC", side="long",
                  qty=0.001, fill_price=50000.0, fee=0.05)

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            Trade(signal_id=1, bar=100, sym="BTC", side="invalid",
                  qty=0.001, fill_price=50000.0, fee=0.05)

    def test_zero_qty_rejected(self):
        with self.assertRaises(ValueError):
            Trade(signal_id=1, bar=100, sym="BTC", side="long",
                  qty=0.0, fill_price=50000.0, fee=0.05)

    def test_zero_price_rejected(self):
        with self.assertRaises(ValueError):
            Trade(signal_id=1, bar=100, sym="BTC", side="long",
                  qty=0.001, fill_price=0.0, fee=0.05)


class TestMetrics(unittest.TestCase):
    def test_empty(self):
        m = Metrics.empty()
        self.assertFalse(m.has_data)
        self.assertEqual(m.win_rate, 0.0)
        self.assertEqual(m.pnl_per_trade, 0.0)

    def test_win_rate(self):
        m = Metrics(closed_trades=10, wins=7, losses=3, pnl_pct=2.5)
        self.assertEqual(m.win_rate, 70.0)
        self.assertAlmostEqual(m.pnl_per_trade, 0.25)

    def test_invalid_negative_counts(self):
        with self.assertRaises(ValueError):
            Metrics(closed_trades=-1)

    def test_invalid_wins_plus_losses(self):
        with self.assertRaises(ValueError):
            # 6 + 5 = 11 > 10 closed
            Metrics(closed_trades=10, wins=6, losses=5)

    def test_has_data_predicate(self):
        self.assertTrue(Metrics(signals=1).has_data)
        self.assertTrue(Metrics(entries=1).has_data)
        self.assertTrue(Metrics(closed_trades=1).has_data)
        self.assertFalse(Metrics().has_data)


class TestMarketSnapshot(unittest.TestCase):
    def test_construction(self):
        snap = MarketSnapshot(
            bar=100,
            timestamp=datetime.now(timezone.utc),
            regime=Regime.BULLISH,
            prices={"BTC": 50000.0, "ETH": 3000.0},
            volumes={"BTC": 100.0, "ETH": 500.0},
        )
        self.assertTrue(snap.has_price("BTC"))
        self.assertTrue(snap.has_price("ETH"))
        self.assertFalse(snap.has_price("UNKNOWN"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
