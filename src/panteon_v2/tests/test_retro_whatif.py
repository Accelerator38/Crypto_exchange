"""Tests for offline Retrodate what-if selector analysis."""

from __future__ import annotations

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from panteon_v2.analysis.retro_whatif import (
    RetroWhatIfConfig,
    build_recomposed_candidates,
    run_retrodate_whatif,
    select_whatif_candidate,
)


@contextmanager
def _workspace_temp_dir():
    root = Path.cwd() / ".tmp_retro_whatif_tests"
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class TestSessionAwareWhatIfScoring(unittest.TestCase):
    def test_session_overlay_promotes_current_session_winner_over_stale_memory(self):
        config = RetroWhatIfConfig(
            exchange="MEXC",
            event_log_path=Path("unused.jsonl"),
            results_dir=Path("unused"),
            output_root=Path("unused"),
            overlay_weight=1.0,
            stale_penalty=0.25,
            player_underperformance_weight=1.0,
            agent_underperformance_weight=0.0,
            include_png=False,
        )
        candidates = [
            {
                "player_label": "DefaultEnsemble",
                "score": 1.0,
                "selected_by_pantheon": True,
                "agent_labels": ["StaleAgent"],
            },
            {
                "player_label": "TrendResearch",
                "score": 0.85,
                "selected_by_pantheon": False,
                "agent_labels": ["HotAgent"],
            },
        ]
        agents = {
            "V_StaleAgent": {
                "session_pnl_pct": 0.0,
                "session_closed_trades": 0,
                "session_signals": 0,
            },
            "V_HotAgent": {
                "session_pnl_pct": 0.60,
                "session_closed_trades": 3,
                "session_signals": 4,
            },
        }
        players = {
            "V_DefaultEnsemble": {
                "session_pnl_pct": 0.0,
                "session_closed_trades": 0,
                "session_signals": 0,
            },
            "V_TrendResearch": {
                "session_pnl_pct": 0.0,
                "session_closed_trades": 0,
                "session_signals": 0,
            },
        }

        selected, scored = select_whatif_candidate(
            candidates,
            agent_metrics=agents,
            player_metrics=players,
            config=config,
        )

        self.assertEqual(selected["player_label"], "TrendResearch")
        by_label = {row["player_label"]: row for row in scored}
        self.assertLess(
            by_label["DefaultEnsemble"]["whatif_score"],
            by_label["TrendResearch"]["whatif_score"],
        )
        self.assertGreater(by_label["DefaultEnsemble"]["stale_penalty"], 0.0)
        self.assertGreater(by_label["TrendResearch"]["agent_session_overlay"], 0.0)

    def test_recomposed_candidates_use_full_agent_leaderboard(self):
        config = RetroWhatIfConfig(
            exchange="BITGET",
            event_log_path=Path("unused.jsonl"),
            results_dir=Path("unused"),
            output_root=Path("unused"),
            overlay_weight=1.5,
            stale_penalty=0.25,
            include_png=False,
        )
        agents = {
            "V_StaleMemory": {
                "pnl_pct": 0.30,
                "closed_trades": 8,
                "signals": 8,
                "wins": 5,
                "losses": 3,
                "session_pnl_pct": 0.0,
                "session_closed_trades": 0,
                "session_signals": 0,
            },
            "V_LiveTrendFollow": {
                "pnl_pct": -0.20,
                "closed_trades": 8,
                "signals": 8,
                "wins": 4,
                "losses": 4,
                "session_pnl_pct": 1.20,
                "session_closed_trades": 5,
                "session_signals": 4,
            },
            "V_CarryFlowAgentV2": {
                "pnl_pct": -0.10,
                "closed_trades": 7,
                "signals": 7,
                "wins": 4,
                "losses": 3,
                "session_pnl_pct": 0.80,
                "session_closed_trades": 3,
                "session_signals": 2,
            },
        }

        recomposed = build_recomposed_candidates(
            agent_metrics=agents,
            config=config,
            regime="bullish",
        )

        self.assertTrue(recomposed)
        first = recomposed[0]
        self.assertEqual(first["candidate_origin"], "recomposed")
        self.assertIn("LiveTrendFollow", first["agent_labels"])
        self.assertIn("CarryFlowAgentV2", first["agent_labels"])
        self.assertNotIn("StaleMemory", first["agent_labels"][:2])


class TestRetrodateWhatIfArtifacts(unittest.TestCase):
    def test_runner_writes_retrodate_artifacts_and_compares_actual_vs_whatif(self):
        with _workspace_temp_dir() as tmp:
            results_dir = tmp / "Results" / "MEXC" / "2026-05-15_08-00-00_v2"
            results_dir.mkdir(parents=True)
            event_log = tmp / "logs" / "v2_mexc_events.jsonl"
            event_log.parent.mkdir(parents=True)
            output_root = tmp / "Results" / "Retrodate"

            self._write_json(
                results_dir / "status.json",
                {
                    "exchange": "MEXC",
                    "timestamp": "2026-05-15T08:30:00+00:00",
                    "run_state": "finished",
                    "feed_status": "offline",
                    "regime": "neutral",
                    "bar_count": 11,
                    "current_leader": "DefaultEnsemble",
                    "initial_capital": 100.0,
                    "current_balance": 100.0,
                },
            )
            self._write_json(
                results_dir / "leaderboard_agents.json",
                {
                    "metadata": {"exchange": "MEXC"},
                    "agents": {
                        "V_StaleAgent": {
                            "pnl_pct": 1.0,
                            "session_pnl_pct": 0.0,
                            "session_closed_trades": 0,
                            "session_signals": 0,
                        },
                        "V_HotAgent": {
                            "pnl_pct": -0.2,
                            "session_pnl_pct": 0.75,
                            "session_closed_trades": 5,
                            "session_signals": 6,
                        },
                    },
                },
            )
            self._write_json(
                results_dir / "leaderboard_players.json",
                {
                    "metadata": {"exchange": "MEXC"},
                    "players": {
                        "V_DefaultEnsemble": {
                            "pnl_pct": 1.2,
                            "session_pnl_pct": 0.0,
                            "session_closed_trades": 0,
                            "session_signals": 0,
                        },
                        "V_TrendResearch": {
                            "pnl_pct": -0.1,
                            "session_pnl_pct": 0.0,
                            "session_closed_trades": 0,
                            "session_signals": 0,
                        },
                    },
                },
            )
            self._write_jsonl(
                event_log,
                [
                    {
                        "_type": "DecisionStarted",
                        "timestamp": "2026-05-14T08:00:00+00:00",
                        "decision_id": "old",
                        "bar": 1,
                        "exchange": "MEXC",
                    },
                    {
                        "_type": "DecisionStarted",
                        "timestamp": "2026-05-15T08:01:00+00:00",
                        "decision_id": "d1",
                        "bar": 10,
                        "exchange": "MEXC",
                    },
                    {
                        "_type": "CandidateScored",
                        "timestamp": "2026-05-15T08:01:00+00:00",
                        "decision_id": "d1",
                        "bar": 10,
                        "player_label": "DefaultEnsemble",
                        "score": 1.0,
                        "rank": 1,
                        "selected_by_pantheon": True,
                        "agent_labels": ["StaleAgent"],
                    },
                    {
                        "_type": "CandidateScored",
                        "timestamp": "2026-05-15T08:01:00+00:00",
                        "decision_id": "d1",
                        "bar": 10,
                        "player_label": "TrendResearch",
                        "score": 0.80,
                        "rank": 2,
                        "selected_by_pantheon": False,
                        "agent_labels": ["HotAgent"],
                    },
                ],
            )

            summary = run_retrodate_whatif(
                RetroWhatIfConfig(
                    exchange="MEXC",
                    event_log_path=event_log,
                    results_dir=results_dir,
                    output_root=output_root,
                    session_start=datetime(2026, 5, 15, 8, 0, tzinfo=timezone.utc),
                    overlay_weight=1.0,
                    stale_penalty=0.25,
                    include_png=False,
                )
            )

            self.assertEqual(summary["decisions"]["changed_count"], 1)
            out = Path(summary["output_dir"])
            for name in (
                "status.json",
                "leaderboard_agents.json",
                "leaderboard_players.json",
                "dashboard.txt",
                "trading.log",
                "whatif_summary.json",
                "whatif_decisions.jsonl",
                "whatif_candidates.csv",
            ):
                self.assertTrue((out / name).exists(), name)
            decision = json.loads((out / "whatif_decisions.jsonl").read_text().splitlines()[0])
            self.assertEqual(decision["actual_selected"], "DefaultEnsemble")
            self.assertEqual(decision["whatif_selected"], "TrendResearch")
            self.assertEqual(decision["classification"], "choice_session_overlay")

    def test_runner_writes_recomposed_whatif_from_full_agent_leaderboard(self):
        with _workspace_temp_dir() as tmp:
            results_dir = tmp / "Results" / "BITGET" / "2026-05-15_08-00-00_v2"
            results_dir.mkdir(parents=True)
            event_log = tmp / "logs" / "v2_bitget_events.jsonl"
            event_log.parent.mkdir(parents=True)
            output_root = tmp / "Results" / "Retrodate"

            self._write_json(
                results_dir / "status.json",
                {
                    "exchange": "BITGET",
                    "timestamp": "2026-05-15T08:30:00+00:00",
                    "run_state": "finished",
                    "feed_status": "offline",
                    "regime": "bullish",
                    "bar_count": 11,
                    "current_leader": "DefaultEnsemble",
                    "initial_capital": 100.0,
                    "current_balance": 100.0,
                },
            )
            self._write_json(
                results_dir / "leaderboard_agents.json",
                {
                    "metadata": {"exchange": "BITGET"},
                    "agents": {
                        "V_StaleMemory": {
                            "pnl_pct": 0.30,
                            "closed_trades": 8,
                            "signals": 8,
                            "wins": 5,
                            "losses": 3,
                            "session_pnl_pct": 0.0,
                            "session_closed_trades": 0,
                            "session_signals": 0,
                        },
                        "V_LiveTrendFollow": {
                            "pnl_pct": -0.20,
                            "closed_trades": 8,
                            "signals": 8,
                            "wins": 4,
                            "losses": 4,
                            "session_pnl_pct": 1.20,
                            "session_closed_trades": 5,
                            "session_signals": 4,
                        },
                        "V_CarryFlowAgentV2": {
                            "pnl_pct": -0.10,
                            "closed_trades": 7,
                            "signals": 7,
                            "wins": 4,
                            "losses": 3,
                            "session_pnl_pct": 0.80,
                            "session_closed_trades": 3,
                            "session_signals": 2,
                        },
                    },
                },
            )
            self._write_json(
                results_dir / "leaderboard_players.json",
                {
                    "metadata": {"exchange": "BITGET"},
                    "players": {
                        "V_DefaultEnsemble": {
                            "pnl_pct": 1.0,
                            "session_pnl_pct": 0.0,
                            "session_closed_trades": 0,
                            "session_signals": 0,
                        },
                        "V_TrendResearch": {
                            "pnl_pct": 0.0,
                            "session_pnl_pct": 0.0,
                            "session_closed_trades": 0,
                            "session_signals": 0,
                        },
                    },
                },
            )
            self._write_jsonl(
                event_log,
                [
                    {
                        "_type": "DecisionStarted",
                        "timestamp": "2026-05-15T08:01:00+00:00",
                        "decision_id": "d1",
                        "bar": 10,
                        "exchange": "BITGET",
                        "regime": "bullish",
                    },
                    {
                        "_type": "CandidateScored",
                        "timestamp": "2026-05-15T08:01:00+00:00",
                        "decision_id": "d1",
                        "bar": 10,
                        "player_label": "DefaultEnsemble",
                        "score": 1.0,
                        "rank": 1,
                        "selected_by_pantheon": True,
                        "agent_labels": ["StaleMemory"],
                    },
                ],
            )

            summary = run_retrodate_whatif(
                RetroWhatIfConfig(
                    exchange="BITGET",
                    event_log_path=event_log,
                    results_dir=results_dir,
                    output_root=output_root,
                    session_start=datetime(2026, 5, 15, 8, 0, tzinfo=timezone.utc),
                    overlay_weight=1.5,
                    stale_penalty=0.25,
                    include_png=False,
                )
            )

            self.assertEqual(summary["recomposed_decisions"]["changed_count"], 1)
            out = Path(summary["output_dir"])
            self.assertTrue((out / "whatif_recomposed_decisions.jsonl").exists())
            decision = json.loads(
                (out / "whatif_recomposed_decisions.jsonl").read_text().splitlines()[0]
            )
            self.assertEqual(decision["actual_selected"], "DefaultEnsemble")
            self.assertEqual(decision["recomposed_selected"], "TrendResearch")
            self.assertEqual(summary["actual_profile_recomposition"]["agent_set_changed_count"], 1)
            self.assertTrue(decision["actual_profile_agent_set_changed"])
            self.assertEqual(decision["actual_logged_agent_labels"], ["StaleMemory"])
            self.assertIn("LiveTrendFollow", decision["actual_recomposed_agent_labels"])
            selected = next(
                c for c in decision["recomposed_candidates"]
                if c["player_label"] == "TrendResearch"
            )
            self.assertIn("LiveTrendFollow", selected["agent_labels"])
            self.assertIn("CarryFlowAgentV2", selected["agent_labels"])

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
