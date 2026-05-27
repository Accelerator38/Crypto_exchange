"""Tests for Flash partial profit-lock signal generation."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from panteon_v2.app.main_loop import _flash_partial_profit_lock_close_signals
from panteon_v2.domain.types import Action, Regime
from panteon_v2.execution import PositionTracker, TrackedPosition
from panteon_v2.tests._helpers import make_market


class TestFlashPartialProfitLock(unittest.TestCase):
    def test_emits_fractional_close_for_unlocked_profitable_position(self):
        tracker = PositionTracker()
        tracker.force_set(
            TrackedPosition(
                open_signal_id=1,
                sym="BTC/USDT",
                side="long",
                entry_price=100.0,
                qty=2.0,
                fee_open=0.0,
                by_player="Panteon_Flash",
                by_agent="MomentumScalper",
                opened_at=datetime.now(timezone.utc),
                opened_bar=10,
            )
        )
        pipeline = SimpleNamespace(
            executor=SimpleNamespace(_tracker=tracker),
            flash_partial_profit_lock_enabled=True,
            flash_partial_profit_lock_trigger_pnl_pct=1.5,
            flash_partial_profit_lock_close_fraction=0.5,
            flash_partial_profit_lock_min_age_bars=2,
        )

        signals = _flash_partial_profit_lock_close_signals(
            pipeline,
            make_market(
                bar=13,
                regime=Regime.BULLISH,
                prices={"BTC/USDT": 102.0},
            ),
            signal_id_start=20,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, Action.FUT_CLOSE_ALL)
        self.assertEqual(signals[0].by_agent, "PartialProfitLock")
        self.assertAlmostEqual(signals[0].close_fraction, 0.5)

    def test_skips_position_that_already_locked_profit(self):
        tracker = PositionTracker()
        tracker.force_set(
            TrackedPosition(
                open_signal_id=1,
                sym="BTC/USDT",
                side="long",
                entry_price=100.0,
                qty=1.0,
                fee_open=0.0,
                by_player="Panteon_Flash",
                by_agent="MomentumScalper",
                opened_at=datetime.now(timezone.utc),
                opened_bar=10,
                partial_profit_locked=True,
            )
        )
        pipeline = SimpleNamespace(
            executor=SimpleNamespace(_tracker=tracker),
            flash_partial_profit_lock_enabled=True,
            flash_partial_profit_lock_trigger_pnl_pct=1.0,
        )

        signals = _flash_partial_profit_lock_close_signals(
            pipeline,
            make_market(bar=20, prices={"BTC/USDT": 103.0}),
            signal_id_start=20,
        )

        self.assertEqual(signals, [])

    def test_skips_selected_subset_protected_open_signal_key_by_default(self):
        tracker = PositionTracker()
        tracker.force_set(
            TrackedPosition(
                open_signal_id=1,
                sym="BTC/USDT",
                side="long",
                entry_price=100.0,
                qty=1.0,
                fee_open=0.0,
                by_player="Solo_MomentumScalper",
                by_agent="MomentumScalper",
                opened_at=datetime.now(timezone.utc),
                opened_bar=10,
                open_action="FUT_LONG_FULL",
            )
        )
        pipeline = SimpleNamespace(
            executor=SimpleNamespace(_tracker=tracker),
            flash_partial_profit_lock_enabled=True,
            flash_partial_profit_lock_trigger_pnl_pct=1.0,
            flash_selected_subset_do_not_demote_signal_keys=(
                "ensemble:Solo_MomentumScalper|BTC/USDT|FUT_LONG_FULL",
            ),
        )

        signals = _flash_partial_profit_lock_close_signals(
            pipeline,
            make_market(bar=20, prices={"BTC/USDT": 105.0}),
            signal_id_start=20,
        )

        self.assertEqual(signals, [])

    def test_can_allow_partial_lock_for_protected_open_signal_key(self):
        tracker = PositionTracker()
        tracker.force_set(
            TrackedPosition(
                open_signal_id=1,
                sym="BTC/USDT",
                side="long",
                entry_price=100.0,
                qty=1.0,
                fee_open=0.0,
                by_player="Solo_MomentumScalper",
                by_agent="MomentumScalper",
                opened_at=datetime.now(timezone.utc),
                opened_bar=10,
                open_action="FUT_LONG_FULL",
            )
        )
        pipeline = SimpleNamespace(
            executor=SimpleNamespace(_tracker=tracker),
            flash_partial_profit_lock_enabled=True,
            flash_partial_profit_lock_trigger_pnl_pct=1.0,
            flash_partial_profit_lock_skip_protected_signal_keys=False,
            flash_selected_subset_do_not_demote_signal_keys=(
                "ensemble:Solo_MomentumScalper|BTC/USDT|FUT_LONG_FULL",
            ),
        )

        signals = _flash_partial_profit_lock_close_signals(
            pipeline,
            make_market(bar=20, prices={"BTC/USDT": 105.0}),
            signal_id_start=20,
        )

        self.assertEqual(len(signals), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
