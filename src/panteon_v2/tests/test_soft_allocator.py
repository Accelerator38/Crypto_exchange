"""Tests for offline soft capital allocation analysis."""

from __future__ import annotations

import unittest

from panteon_v2.analysis.soft_allocator import (
    ShadowPnLEvent,
    SoftAllocatorPolicy,
    default_soft_allocator_policies,
    simulate_perfect_monthly_panteon,
    simulate_soft_allocator_policies,
)


class TestSoftAllocator(unittest.TestCase):
    def test_decayed_policy_allocates_before_current_bar_and_reports_best(self):
        events = [
            ShadowPnLEvent(bar=1, label="Alpha", pnl_usd=10.0, closed_trades=2, wins=2),
            ShadowPnLEvent(bar=1, label="Beta", pnl_usd=-4.0, closed_trades=2, wins=0),
            ShadowPnLEvent(bar=2, label="Alpha", pnl_usd=6.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=2, label="Beta", pnl_usd=30.0, closed_trades=2, wins=2),
            ShadowPnLEvent(bar=3, label="Alpha", pnl_usd=-8.0, closed_trades=1, wins=0),
            ShadowPnLEvent(bar=3, label="Beta", pnl_usd=-10.0, closed_trades=1, wins=0),
            ShadowPnLEvent(bar=4, label="Alpha", pnl_usd=12.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=4, label="Beta", pnl_usd=1.0, closed_trades=1, wins=1),
        ]

        report = simulate_soft_allocator_policies(
            events,
            initial_capital=100.0,
            policies=[
                SoftAllocatorPolicy(
                    name="top1",
                    top_k=1,
                    cash_reserve_weight=0.0,
                    max_weight_per_leader=1.0,
                    min_closed_trades=1,
                    half_life_bars=10_000,
                ),
                SoftAllocatorPolicy(
                    name="top2_cash",
                    top_k=2,
                    cash_reserve_weight=0.25,
                    max_weight_per_leader=0.60,
                    min_closed_trades=1,
                    half_life_bars=10_000,
                ),
            ],
        )

        by_name = {item.policy_name: item for item in report.policy_results}
        self.assertEqual(report.best_policy_name, "top2_cash")
        self.assertEqual(report.best_single_label, "Alpha")
        self.assertAlmostEqual(report.best_single_pnl_usd, 20.0)
        self.assertAlmostEqual(by_name["top1"].pnl_usd, -3.0)
        self.assertAlmostEqual(by_name["top1"].average_cash_weight_pct, 25.0)
        self.assertGreater(by_name["top2_cash"].average_leader_count, 0.0)
        self.assertGreater(by_name["top1"].max_drawdown_pct, 0.0)

    def test_default_policies_include_alltime_top1_baseline(self):
        names = {policy.name for policy in default_soft_allocator_policies()}

        self.assertIn("soft_top1_alltime", names)

    def test_regime_rolling_contrarian_policy_uses_recent_regime_loss(self):
        events = [
            ShadowPnLEvent(bar=1, label="Alpha", regime="bullish", pnl_usd=-10.0, closed_trades=1),
            ShadowPnLEvent(bar=1, label="Beta", regime="bullish", pnl_usd=5.0, closed_trades=1),
            ShadowPnLEvent(bar=2, label="Alpha", regime="bullish", pnl_usd=20.0, closed_trades=1),
            ShadowPnLEvent(bar=2, label="Beta", regime="bullish", pnl_usd=-5.0, closed_trades=1),
            ShadowPnLEvent(bar=3, label="Alpha", regime="bearish", pnl_usd=-50.0, closed_trades=1),
            ShadowPnLEvent(bar=3, label="Beta", regime="bearish", pnl_usd=8.0, closed_trades=1),
        ]

        report = simulate_soft_allocator_policies(
            events,
            initial_capital=100.0,
            policies=[
                SoftAllocatorPolicy(
                    name="regime_bottom",
                    top_k=1,
                    cash_reserve_weight=0.0,
                    max_weight_per_leader=1.0,
                    min_closed_trades=1,
                    score_scope="regime",
                    score_mode="bottom",
                    rolling_window_bars=24,
                )
            ],
        )

        result = report.policy_results[0]
        self.assertAlmostEqual(result.pnl_usd, 20.0)
        self.assertEqual(report.best_single_label, "Beta")
        self.assertTrue(report.beats_best_single)

    def test_perfect_monthly_panteon_selects_best_monthly_actor_and_cash(self):
        events = [
            ShadowPnLEvent(
                bar=1,
                label="Alpha",
                timestamp="2025-01-01T00:00:00+00:00",
                pnl_usd=10.0,
                closed_trades=1,
                wins=1,
            ),
            ShadowPnLEvent(
                bar=2,
                label="Beta",
                timestamp="2025-01-02T00:00:00+00:00",
                pnl_usd=30.0,
                closed_trades=2,
                wins=2,
            ),
            ShadowPnLEvent(
                bar=3,
                label="Alpha",
                timestamp="2025-02-01T00:00:00+00:00",
                pnl_usd=-5.0,
                closed_trades=1,
                wins=0,
            ),
            ShadowPnLEvent(
                bar=4,
                label="Beta",
                timestamp="2025-02-02T00:00:00+00:00",
                pnl_usd=-7.0,
                closed_trades=1,
                wins=0,
            ),
            ShadowPnLEvent(
                bar=5,
                label="Alpha",
                timestamp="2025-03-01T00:00:00+00:00",
                pnl_usd=12.0,
                closed_trades=1,
                wins=1,
            ),
            ShadowPnLEvent(
                bar=6,
                label="Beta",
                timestamp="2025-03-02T00:00:00+00:00",
                pnl_usd=1.0,
                closed_trades=1,
                wins=1,
            ),
        ]

        report = simulate_perfect_monthly_panteon(events, initial_capital=100.0)

        self.assertEqual(report.month_count, 3)
        self.assertAlmostEqual(report.pnl_usd, 42.0)
        self.assertAlmostEqual(report.pnl_pct, 42.0)
        self.assertEqual(report.profitable_months, 2)
        self.assertEqual(report.cash_months, 1)
        self.assertEqual([row.label for row in report.months], ["Beta", "CASH", "Alpha"])
        self.assertAlmostEqual(report.months[1].pnl_usd, 0.0)
        self.assertAlmostEqual(report.closed_trades, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
