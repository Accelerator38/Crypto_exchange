"""Tests for walk-forward report generation from stored Results/logs."""

from __future__ import annotations

import json
import os
import tempfile
import unittest


class TestWalkForwardReport(unittest.TestCase):
    def test_report_parses_events_and_computes_core_metrics(self):
        from panteon_v2.analysis.walk_forward import build_walk_forward_report

        with tempfile.TemporaryDirectory() as tmp:
            results = os.path.join(tmp, "Results")
            session = os.path.join(results, "MEXC", "session_v2")
            logs = os.path.join(tmp, "logs")
            os.makedirs(session)
            os.makedirs(logs)

            with open(os.path.join(session, "status.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "exchange": "MEXC",
                    "initial_capital": 1000.0,
                    "current_balance": 1008.0,
                    "assets_curve": [1000.0, 1012.0, 1008.0],
                }, f)

            event_log = os.path.join(logs, "events.jsonl")
            rows = [
                {"_type": "RegimeDetected", "bar": 1, "regime": 1},
                {"_type": "OrderFilled", "bar": 1, "trade": {"sym": "BTC", "notional": 100.0, "fee": 0.02}},
                {"_type": "PositionClosed", "bar": 2, "sym": "BTC", "realized_pnl": 10.0, "by_player": "P", "by_agent": "A"},
                {"_type": "RegimeDetected", "bar": 3, "regime": 2},
                {"_type": "OrderFilled", "bar": 3, "trade": {"sym": "ETH", "notional": 50.0, "fee": 0.01}},
                {"_type": "PositionClosed", "bar": 4, "sym": "ETH", "realized_pnl": -4.0, "by_player": "P", "by_agent": "B"},
            ]
            with open(event_log, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

            report = build_walk_forward_report(results_root=results, event_logs=[event_log])

        totals = report["totals"]
        self.assertEqual(totals["closed_trades"], 2)
        self.assertEqual(totals["wins"], 1)
        self.assertEqual(totals["losses"], 1)
        self.assertAlmostEqual(totals["winrate_pct"], 50.0)
        self.assertAlmostEqual(totals["profit_factor"], 2.5)
        self.assertAlmostEqual(totals["expectancy"], 3.0)
        self.assertAlmostEqual(totals["turnover_notional"], 150.0)
        self.assertIn("bullish", report["by_regime"])
        self.assertEqual(report["sessions"]["count"], 1)

    def test_regime_stats_use_entry_regime_when_close_happens_after_regime_change(self):
        from panteon_v2.analysis.walk_forward import build_walk_forward_report_from_events

        events = [
            {"_type": "RegimeDetected", "bar": 1, "regime": "bullish"},
            {
                "_type": "PositionOpened",
                "bar": 1,
                "signal_id": 101,
                "sym": "BTC/USDT",
            },
            {"_type": "RegimeDetected", "bar": 2, "regime": "neutral"},
            {
                "_type": "PositionClosed",
                "bar": 2,
                "open_signal_id": 101,
                "sym": "BTC/USDT",
                "realized_pnl": -5.0,
                "by_player": "P",
                "by_agent": "A",
            },
        ]

        report = build_walk_forward_report_from_events(
            results_root="Results",
            events=events,
        )

        self.assertIn("bullish", report["by_regime"])
        self.assertNotIn("neutral", report["by_regime"])
        self.assertEqual(report["by_regime"]["bullish"]["closed_trades"], 1)
        self.assertAlmostEqual(report["by_regime"]["bullish"]["net_pnl"], -5.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
