"""Tests for the local Results/dashboard.html builder."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_build_dashboard_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_dashboard.py"
    spec = importlib.util.spec_from_file_location("panteon_build_dashboard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestBuildDashboardTool(unittest.TestCase):
    def test_collect_sessions_exposes_latest_visual_chart_payloads(self):
        build_dashboard = _load_build_dashboard_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_dir = root / "MEXC" / "2026-05-01_00-00-00_v2"
            new_dir = root / "MEXC" / "2026-05-02_00-00-00_v2"
            for sdir in (old_dir, new_dir):
                sdir.mkdir(parents=True)
                (sdir / "leaderboard_players.json").write_text(
                    json.dumps({"players": {}}, ensure_ascii=False),
                    encoding="utf-8",
                )
                (sdir / "leaderboard_agents.json").write_text(
                    json.dumps({"agents": {}}, ensure_ascii=False),
                    encoding="utf-8",
                )
            (old_dir / "status.json").write_text(
                json.dumps({"pnl_pct": 0.0}, ensure_ascii=False),
                encoding="utf-8",
            )
            (new_dir / "status.json").write_text(
                json.dumps({
                    "panteon_equity_usd": 101.0,
                    "total_assets_usd": 102.0,
                    "price_history": [
                        {"bar": 1, "regime": "bullish", "prices": {"BTC": 100.0}},
                    ],
                }, ensure_ascii=False),
                encoding="utf-8",
            )

            sessions = build_dashboard._collect_sessions(
                "MEXC",
                results_dir=root,
                sessions_limit=10,
            )

        self.assertEqual(sessions[0]["name"], "2026-05-02_00-00-00_v2")
        self.assertEqual(sessions[0]["summary"]["panteon_equity_usd"], 101.0)
        self.assertEqual(sessions[0]["price_history"][0]["prices"]["BTC"], 100.0)
        self.assertEqual(
            sessions[0]["visuals"]["dashboard_latest"],
            "MEXC/2026-05-02_00-00-00_v2/dashboard_latest.png",
        )

    def test_collect_sessions_sorts_assets_and_adds_configured_players_from_dashboard(self):
        build_dashboard = _load_build_dashboard_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sdir = root / "MEXC" / "2026-05-02_00-00-00_v2"
            sdir.mkdir(parents=True)
            (sdir / "status.json").write_text(
                json.dumps({
                    "initial_capital": 100.0,
                    "current_balance": 100.0,
                    "panteon_equity_usd": 99.0,
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            (sdir / "leaderboard_players.json").write_text(
                json.dumps({
                    "players": {
                        "V_DefaultEnsemble": {"session_pnl_pct": 1.0},
                    },
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            (sdir / "leaderboard_agents.json").write_text(
                json.dumps({
                    "agents": {
                        "V_LiveAfterShock": {"session_pnl_pct": 2.0},
                    },
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            (sdir / "dashboard.txt").write_text(
                "\n".join([
                    "Configured actor pool",
                    "  Players:",
                    "    Antonius_conservative            regime_switch  neutral                  neutral:ResearchValidatorAgent",
                    "  Agents:",
                ]),
                encoding="utf-8",
            )

            sessions = build_dashboard._collect_sessions(
                "MEXC",
                results_dir=root,
                sessions_limit=10,
            )

        self.assertEqual(
            [row["name"] for row in sessions[0]["asset_rows"][:4]],
            [
                "A:LiveAfterShock",
                "P:DefaultEnsemble",
                "P:Antonius_conservative",
                "PANTEON",
            ],
        )
        self.assertIn("V_Antonius_conservative", sessions[0]["players"])
        self.assertTrue(sessions[0]["players"]["V_Antonius_conservative"]["configured_pool_only"])


if __name__ == "__main__":
    unittest.main()
