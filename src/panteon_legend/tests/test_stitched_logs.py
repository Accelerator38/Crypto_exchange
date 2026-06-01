from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class TestStitchedLiveLogs(unittest.TestCase):
    def test_build_stitched_log_report_counts_events_and_walk_forward(self) -> None:
        from panteon_legend.stitched_logs import build_stitched_log_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bitget = root / "bitget.jsonl"
            mexc = root / "mexc.jsonl"
            bitget.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"type": "BarStarted", "bar": 1},
                        {
                            "_type": "CandidateRejected",
                            "bar": 1,
                            "reason": "flash:inactive",
                        },
                        {
                            "event_type": "SignalEmitted",
                            "bar": 1,
                            "sym": "BTC/USDT",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            mexc.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {
                            "type": "OrderFilled",
                            "bar": 1,
                            "trade": {"sym": "BTC/USDT", "notional": 100.0, "fee": 0.02},
                        },
                        {
                            "type": "PositionClosed",
                            "bar": 2,
                            "sym": "BTC/USDT",
                            "realized_pnl": 1.25,
                            "by_player": "Legend",
                            "by_agent": "LiveOIBreakout",
                        },
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            report = build_stitched_log_report(
                [bitget, mexc],
                output_dir=root / "out",
            )

            self.assertEqual(report["total_events"], 5)
            self.assertEqual(report["event_type_counts"]["CandidateRejected"], 1)
            self.assertEqual(report["event_type_counts"]["SignalEmitted"], 1)
            self.assertEqual(report["reject_reasons"]["flash:inactive"], 1)
            self.assertEqual(report["walk_forward"]["totals"]["closed_trades"], 1)
            self.assertEqual(report["walk_forward"]["totals"]["net_pnl"], 1.25)
            self.assertTrue((root / "out" / "stitched_live_log_report.json").exists())
            self.assertTrue((root / "out" / "stitched_live_log_report.md").exists())


if __name__ == "__main__":
    unittest.main()
