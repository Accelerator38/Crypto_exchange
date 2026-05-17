from __future__ import annotations

import unittest

from panteon_v2.analysis.genetics_degradation_monitoring import (
    build_genetics_degradation_report,
)


class TestGeneticsDegradationMonitoring(unittest.TestCase):
    def test_report_flags_preexisting_quarantine_and_gate_decision(self) -> None:
        status = {
            "degradation_gate": {
                "config": {"max_execution_failure_rate": 0.5},
                "last_decisions": [
                    {
                        "label": "GeneticsBullish",
                        "should_disable": True,
                        "reasons": ["execution_failure_rate"],
                        "session_signals": 12,
                        "session_execution_failures": 11,
                        "session_rejected_signals": 11,
                        "session_blocked_signals": 0,
                        "session_pnl_pct": 0.0,
                        "session_closed_trades": 0,
                        "session_max_drawdown_pct": 0.0,
                        "execution_failure_rate": 11 / 12,
                        "blocked_signal_rate": 0.0,
                    },
                    {
                        "label": "LiveTrendFollow",
                        "should_disable": True,
                        "reasons": ["execution_failure_rate"],
                        "session_signals": 20,
                        "session_execution_failures": 20,
                    },
                ],
            },
            "quarantine_records": {
                "GeneticsBullish": {
                    "state": "quarantined",
                    "reason": "hopeless_in_all_regimes",
                }
            },
        }
        leaderboard = {
            "agents": {
                "V_GeneticsBullish": {
                    "is_quarantined": True,
                    "quarantine_reason": "hopeless_in_all_regimes",
                }
            }
        }

        report = build_genetics_degradation_report(status, leaderboard)

        self.assertEqual(report["summary"]["monitored_labels"], 1)
        self.assertEqual(report["summary"]["gate_disable_count"], 1)
        self.assertEqual(report["summary"]["preexisting_quarantine_count"], 1)
        self.assertEqual(report["summary"]["max_blocked_signal_rate"], 0.0)
        self.assertIn("old-memory", report["warnings"][0])
        row = report["labels"][0]
        self.assertEqual(row["label"], "GeneticsBullish")
        self.assertTrue(row["gate_should_disable"])
        self.assertTrue(row["preexisting_quarantine"])
        self.assertAlmostEqual(row["execution_failure_rate"], 11 / 12)
        self.assertEqual(row["session_unexecuted_signals"], 11)

    def test_report_uses_leaderboard_when_status_has_no_decision(self) -> None:
        status = {
            "degradation_gate": {
                "last_decisions": [],
            },
            "quarantine_records": {},
        }
        leaderboard = {
            "agents": {
                "V_GeneticsNeutral": {
                    "session_pnl_pct": -0.25,
                    "session_closed_trades": 3,
                    "session_signals": 9,
                    "is_quarantined": False,
                    "quarantine_reason": "",
                },
                "V_LiveMeanRev": {
                    "session_pnl_pct": -9.0,
                    "session_signals": 99,
                },
            }
        }

        report = build_genetics_degradation_report(status, leaderboard)

        self.assertEqual(report["summary"]["monitored_labels"], 1)
        self.assertEqual(report["labels"][0]["label"], "GeneticsNeutral")
        self.assertEqual(report["labels"][0]["session_signals"], 9)
        self.assertFalse(report["labels"][0]["gate_should_disable"])

    def test_report_enriches_genetics_rows_with_shadow_blocked_reasons(self) -> None:
        status = {
            "degradation_gate": {
                "last_decisions": [
                    {
                        "label": "GeneticsBullish",
                        "should_disable": True,
                        "reasons": ["blocked_signal_rate"],
                        "session_signals": 4,
                        "session_unexecuted_signals": 4,
                        "session_blocked_signals": 4,
                        "session_rejected_signals": 0,
                        "session_execution_failures": 0,
                        "blocked_signal_rate": 1.0,
                    }
                ],
            },
            "quarantine_records": {
                "GeneticsBullish": {
                    "state": "quarantined",
                    "reason": "degradation_gate:blocked_signal_rate",
                }
            },
        }
        leaderboard = {"agents": {}}
        shadow_events = [
            {
                "_type": "ShadowActorUpdated",
                "actor_type": "agent",
                "actor_label": "GeneticsBullish",
                "signals": 2,
                "blocked": 2,
                "agent_outcomes": [["GeneticsBullish", 2, 0, 0, 2]],
                "blocked_reasons": [["risk_limits: position already open on BTC", 2]],
                "agent_blocked_reasons": [
                    ["GeneticsBullish", "risk_limits: position already open on BTC", 2],
                ],
            },
            {
                "_type": "ShadowActorUpdated",
                "actor_type": "player",
                "actor_label": "TrendResearch",
                "signals": 2,
                "blocked": 2,
                "agent_outcomes": [["GeneticsBullish", 2, 0, 0, 2]],
                "blocked_reasons": [["risk_limits: no position to close", 2]],
                "agent_blocked_reasons": [
                    ["GeneticsBullish", "risk_limits: no position to close", 1],
                    ["OtherAgent", "risk_limits: no position to close", 1],
                ],
            },
            {
                "_type": "ShadowActorUpdated",
                "actor_type": "agent",
                "actor_label": "LiveTrendFollow",
                "signals": 3,
                "blocked": 3,
                "blocked_reasons": [["risk_limits: position already open on ETH", 3]],
            },
        ]

        report = build_genetics_degradation_report(
            status,
            leaderboard,
            shadow_events=shadow_events,
        )

        row = report["labels"][0]
        self.assertEqual(row["shadow_event_signals"], 4)
        self.assertEqual(row["shadow_event_blocked"], 4)
        self.assertEqual(row["session_execution_failures"], 0)
        self.assertEqual(row["session_unexecuted_signals"], 4)
        self.assertEqual(
            row["shadow_blocked_reasons"],
            [
                {"reason": "risk_limits: position already open on BTC", "count": 2},
                {"reason": "risk_limits: no position to close", "count": 1},
            ],
        )
        self.assertEqual(
            report["summary"]["top_shadow_blocked_reasons"],
            [
                {
                    "label": "GeneticsBullish",
                    "reason": "risk_limits: position already open on BTC",
                    "count": 2,
                },
                {
                    "label": "GeneticsBullish",
                    "reason": "risk_limits: no position to close",
                    "count": 1,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
