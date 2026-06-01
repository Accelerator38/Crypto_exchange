"""Тесты RiskLimits."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from panteon_v2.domain.types import Action, Regime, Signal
from panteon_v2.execution import RiskLimits, RiskLimitsConfig


def _make_signal(action: Action = Action.FUT_LONG_FULL,
                 sym: str = "BTC", price: float = 100.0,
                 risk_mult: float = 1.0) -> Signal:
    return Signal(
        id=1, bar=1, sym=sym, action=action, price=price,
        regime=Regime.BULLISH, by_player="P", by_agent="A",
        risk_mult=risk_mult,
    )


class TestRiskLimits(unittest.TestCase):
    def setUp(self):
        self.rl = RiskLimits()

    def test_hold_blocked(self):
        sig = _make_signal(action=Action.HOLD)
        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions={})
        self.assertFalse(result.allowed)
        self.assertIn("hold", result.reason)

    def test_open_within_limits(self):
        sig = _make_signal()
        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions={})
        self.assertTrue(result.allowed)
        self.assertGreater(result.qty, 0)
        # 1000 × 0.10 (capital_fraction) × 1.0 (full) = 100
        self.assertAlmostEqual(result.notional, 100.0, places=5)
        self.assertAlmostEqual(result.qty, 1.0, places=5)

    def test_half_action_smaller_qty(self):
        sig_full = _make_signal(action=Action.FUT_LONG_FULL)
        sig_half = _make_signal(action=Action.FUT_LONG_HALF)
        full = self.rl.evaluate(sig_full, balance_usd=1000.0, open_positions={})
        half = self.rl.evaluate(sig_half, balance_usd=1000.0, open_positions={})
        self.assertAlmostEqual(half.qty, full.qty * 0.5, places=5)

    def test_leverage_multiplier_is_opt_in(self):
        sig = _make_signal()
        default = RiskLimits(config=RiskLimitsConfig(max_leverage=3))
        leveraged = RiskLimits(
            config=RiskLimitsConfig(
                max_leverage=3,
                apply_leverage_to_notional=True,
            )
        )

        default_result = default.evaluate(sig, balance_usd=1000.0, open_positions={})
        leveraged_result = leveraged.evaluate(sig, balance_usd=1000.0, open_positions={})

        self.assertAlmostEqual(default_result.notional, 100.0, places=5)
        self.assertAlmostEqual(leveraged_result.notional, 300.0, places=5)

    def test_max_open_positions(self):
        cfg = RiskLimitsConfig(max_open_positions=2)
        rl = RiskLimits(config=cfg)
        positions = {"ETH": object(), "DOGE": object()}
        sig = _make_signal(sym="BTC")
        result = rl.evaluate(sig, balance_usd=1000.0, open_positions=positions)
        self.assertFalse(result.allowed)
        self.assertIn("max_open_positions", result.reason)

    def test_external_recovered_positions_do_not_consume_panteon_open_limit(self):
        cfg = RiskLimitsConfig(max_open_positions=2)
        rl = RiskLimits(config=cfg)
        positions = {
            "UNI": SimpleNamespace(by_player="RecoveredExchangePosition"),
            "NEAR": SimpleNamespace(by_player="RecoveredExchangePosition"),
            "DOGE": SimpleNamespace(by_player="DefaultEnsemble"),
        }
        sig = _make_signal(sym="BTC")

        result = rl.evaluate(sig, balance_usd=1000.0, open_positions=positions)

        self.assertTrue(result.allowed, result.reason)

    def test_external_recovered_position_still_blocks_same_symbol_open(self):
        positions = {
            "BTC": SimpleNamespace(by_player="RecoveredExchangePosition"),
        }
        sig = _make_signal(sym="BTC")

        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions=positions)

        self.assertFalse(result.allowed)
        self.assertIn("already open", result.reason)

    def test_external_recovered_position_cannot_be_closed_by_panteon_signal(self):
        positions = {
            "BTC": SimpleNamespace(by_player="RecoveredExchangePosition"),
        }
        sig = _make_signal(sym="BTC", action=Action.FUT_CLOSE_ALL)

        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions=positions)

        self.assertFalse(result.allowed)
        self.assertIn("external", result.reason)

    def test_position_already_open(self):
        sig = _make_signal()
        positions = {"BTC": object()}
        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions=positions)
        self.assertFalse(result.allowed)
        self.assertIn("already open", result.reason)

    def test_min_notional_global(self):
        cfg = RiskLimitsConfig(min_notional_usd=50.0)
        rl = RiskLimits(config=cfg)
        sig = _make_signal()
        # balance × fraction = 100 × 0.10 = 10 — ниже 50
        result = rl.evaluate(sig, balance_usd=100.0, open_positions={})
        self.assertFalse(result.allowed)
        self.assertIn("min", result.reason)

    def test_min_notional_per_symbol_overrides(self):
        sig = _make_signal()
        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions={},
                                  min_notional_for_sym=200.0)
        # Notional = 100, < 200 (per-symbol min)
        self.assertFalse(result.allowed)

    def test_exchange_min_notional_can_floor_small_live_orders(self):
        cfg = RiskLimitsConfig(
            capital_fraction=0.04,
            floor_to_exchange_min_notional=True,
            max_min_notional_upscale=4.0,
        )
        rl = RiskLimits(config=cfg)
        sig = _make_signal(sym="ETH", price=2000.0)

        result = rl.evaluate(
            sig,
            balance_usd=43.61146697,
            open_positions={},
            min_notional_for_sym=5.10,
        )

        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.notional, 5.10, places=5)
        self.assertAlmostEqual(result.qty, 5.10 / 2000.0, places=8)

    def test_max_notional_caps(self):
        cfg = RiskLimitsConfig(max_notional_usd=50.0, capital_fraction=1.0)
        rl = RiskLimits(config=cfg)
        sig = _make_signal()
        result = rl.evaluate(sig, balance_usd=1000.0, open_positions={})
        # 1000 × 1.0 = 1000, capped к 50
        self.assertTrue(result.allowed)
        self.assertAlmostEqual(result.notional, 50.0, places=5)

    def test_close_with_position(self):
        sig = _make_signal(action=Action.FUT_CLOSE_ALL)
        positions = {"BTC": object()}
        result = self.rl.evaluate(sig, balance_usd=1000.0,
                                  open_positions=positions)
        self.assertTrue(result.allowed)

    def test_close_without_position(self):
        sig = _make_signal(action=Action.FUT_CLOSE_ALL)
        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions={})
        self.assertFalse(result.allowed)
        self.assertIn("no position", result.reason)

    def test_invalid_price(self):
        # Цена = 0 — не должна пройти
        from panteon_v2.domain.types import Signal
        sig = Signal(
            id=1, bar=1, sym="BTC", action=Action.FUT_LONG_FULL, price=0.0,
            regime=Regime.BULLISH, by_player="P", by_agent="A",
        )
        # Signal сам отвергнет негативную, но 0 разрешён
        result = self.rl.evaluate(sig, balance_usd=1000.0, open_positions={})
        # invalid_price проверка в RiskLimits отвергает qty
        self.assertFalse(result.allowed)


class TestRiskMultiplier(unittest.TestCase):
    def test_risk_mult_scales_notional(self):
        sig1 = _make_signal(risk_mult=1.0)
        sig2 = _make_signal(risk_mult=0.5)
        rl = RiskLimits()
        r1 = rl.evaluate(sig1, balance_usd=1000.0, open_positions={})
        r2 = rl.evaluate(sig2, balance_usd=1000.0, open_positions={})
        self.assertAlmostEqual(r2.notional, r1.notional * 0.5, places=5)


class TestConfigValidation(unittest.TestCase):
    def test_invalid_capital_fraction(self):
        with self.assertRaises(ValueError):
            RiskLimitsConfig(capital_fraction=0)
        with self.assertRaises(ValueError):
            RiskLimitsConfig(capital_fraction=1.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
