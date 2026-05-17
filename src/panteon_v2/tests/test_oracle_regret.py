"""Tests for oracle/regret analysis helpers."""

from __future__ import annotations

import unittest

from panteon_v2.analysis.oracle_regret import LeaderPnL, compute_leader_regret


class TestOracleRegret(unittest.TestCase):
    def test_empty_input_returns_zeroed_report(self):
        report = compute_leader_regret([])

        self.assertEqual(report.selected_label, "Panteon")
        self.assertEqual(report.best_label, "")
        self.assertEqual(report.leader_count, 0)
        self.assertAlmostEqual(report.selected_pnl_usd, 0.0)
        self.assertAlmostEqual(report.best_pnl_usd, 0.0)
        self.assertAlmostEqual(report.regret_usd, 0.0)
        self.assertAlmostEqual(report.no_trade_share_pct, 0.0)
        self.assertAlmostEqual(report.profitable_leader_share_pct, 0.0)

    def test_compares_selected_against_best_realized_leader(self):
        report = compute_leader_regret(
            [
                LeaderPnL(label="Panteon", pnl_usd=-10.0, bars=100, trades=5),
                LeaderPnL(label="NoTrade", pnl_usd=0.0, bars=50, trades=0),
                LeaderPnL(label="Solo_A", pnl_usd=25.0, bars=25, trades=8),
                LeaderPnL(label="Solo_B", pnl_usd=-5.0, bars=25, trades=3),
            ]
        )

        self.assertEqual(report.selected_label, "Panteon")
        self.assertAlmostEqual(report.selected_pnl_usd, -10.0)
        self.assertEqual(report.best_label, "Solo_A")
        self.assertAlmostEqual(report.best_pnl_usd, 25.0)
        self.assertAlmostEqual(report.regret_usd, 35.0)
        self.assertEqual(report.leader_count, 4)
        self.assertAlmostEqual(report.no_trade_share_pct, 25.0)
        self.assertAlmostEqual(report.profitable_leader_share_pct, 25.0)

    def test_custom_selected_label_and_no_trade_share(self):
        report = compute_leader_regret(
            [
                LeaderPnL(label="AllocatorV3", pnl_usd=12.0, bars=80, trades=4),
                LeaderPnL(label="NoTrade", pnl_usd=0.0, bars=20, trades=0),
                LeaderPnL(label="Solo_A", pnl_usd=10.0, bars=0, trades=2),
            ],
            selected_label="AllocatorV3",
        )

        self.assertEqual(report.selected_label, "AllocatorV3")
        self.assertEqual(report.best_label, "AllocatorV3")
        self.assertAlmostEqual(report.regret_usd, 0.0)
        self.assertAlmostEqual(report.no_trade_share_pct, 20.0)
        self.assertAlmostEqual(report.profitable_leader_share_pct, 66.6666667, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
