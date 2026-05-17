"""Tests for the matplotlib operator PNG backend."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch


class TestPngRenderer(unittest.TestCase):
    def test_status_panel_shows_real_trade_outcomes(self):
        from panteon_v2.dashboards import png_renderer

        class FakeAxis:
            transAxes = object()

            def __init__(self):
                self.texts = []

            def set_axis_off(self):
                pass

            def set_facecolor(self, color):
                pass

            def text(self, *args, **kwargs):
                if len(args) >= 3:
                    self.texts.append(str(args[2]))

        axis = FakeAxis()

        png_renderer._draw_status_panel(axis, {
            "exchange": "BITGET",
            "run_state": "running",
            "feed_status": "active",
            "live_session": {
                "panteon_owned_total_pnl_usd": 0.0,
                "panteon_owned_pnl_pct": 0.0,
            },
            "real_trades": {
                "total": 4,
                "closed": 2,
                "successful": 1,
                "unsuccessful": 1,
                "unresolved": 2,
            },
        })

        self.assertIn("Real trades", axis.texts)
        self.assertIn("4 (closed 2, open 2)", axis.texts)
        self.assertIn("Real W/L/Open", axis.texts)
        self.assertIn("1 / 1 / 2", axis.texts)

    def test_write_operator_pngs_uses_combined_visual_dashboards(self):
        from panteon_v2.dashboards import png_renderer

        captured = []

        def fake_operator(path, status, agents, players, *, panteon_pnl_pct):
            captured.append(("operator", path.name, panteon_pnl_pct))

        def fake_shadow(path, agents, players, *, panteon_pnl_pct):
            captured.append(("shadow", path.name, len(agents), len(players), panteon_pnl_pct))

        def fake_regime(path, agents, players):
            captured.append(("regime", path.name, len(agents), len(players)))

        def fake_memory(path, agents, players):
            captured.append(("memory", path.name, len(agents), len(players)))

        with tempfile.TemporaryDirectory() as td, \
                patch.object(png_renderer, "_render_operator_dashboard",
                             side_effect=fake_operator), \
                patch.object(png_renderer, "_render_combined_shadow_dashboard",
                             side_effect=fake_shadow), \
                patch.object(png_renderer, "_render_combined_regime_dashboard",
                             side_effect=fake_regime), \
                patch.object(png_renderer, "_render_memory_dashboard",
                             side_effect=fake_memory):
            paths = png_renderer.write_operator_pngs(
                td,
                status={"pnl_pct": 2.75},
                agents_payload={"agents": {"V_AgentA": {"pnl_pct": 1.0}}},
                players_payload={"players": {"V_PlayerA": {"pnl_pct": 3.0}}},
            )

        self.assertEqual(
            [path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for path in paths],
            [
                "dashboard_latest.png",
                "shadow_dashboard.png",
                "regime_dashboard.png",
                "memory_dashboard.png",
            ],
        )
        self.assertIn(("operator", "dashboard_latest.png", 2.75), captured)
        self.assertIn(("shadow", "shadow_dashboard.png", 1, 1, 2.75), captured)
        self.assertIn(("regime", "regime_dashboard.png", 1, 1), captured)
        self.assertIn(("memory", "memory_dashboard.png", 1, 1), captured)

    def test_entries_prefer_session_pnl_for_operator_comparison(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._entries({
            "players": {
                "V_LongMemoryLeader": {"pnl_pct": 50.0, "session_pnl_pct": -1.0},
                "V_CurrentLeader": {"pnl_pct": 2.0, "session_pnl_pct": 3.0},
            },
        }, "players")

        self.assertEqual(rows[0][0], "CurrentLeader")
        self.assertEqual(rows[0][1]["display_pnl_pct"], 3.0)
        self.assertEqual(rows[1][1]["display_pnl_pct"], -1.0)

    def test_memory_dashboard_uses_cumulative_memory_pnl(self):
        from panteon_v2.dashboards import png_renderer

        captured = {}

        def fake_operator(path, status, agents, players, *, panteon_pnl_pct):
            pass

        def fake_shadow(path, agents, players, *, panteon_pnl_pct):
            pass

        def fake_regime(path, agents, players):
            pass

        def fake_memory(path, agents, players):
            captured["players"] = players
            captured["agents"] = agents

        with tempfile.TemporaryDirectory() as td, \
                patch.object(png_renderer, "_render_operator_dashboard",
                             side_effect=fake_operator), \
                patch.object(png_renderer, "_render_combined_shadow_dashboard",
                             side_effect=fake_shadow), \
                patch.object(png_renderer, "_render_combined_regime_dashboard",
                             side_effect=fake_regime), \
                patch.object(png_renderer, "_render_memory_dashboard",
                             side_effect=fake_memory):
            png_renderer.write_operator_pngs(
                td,
                status={"pnl_pct": 0.0},
                agents_payload={
                    "agents": {
                        "V_LongMemoryAgent": {
                            "pnl_pct": 12.0,
                            "session_pnl_pct": 0.0,
                        },
                        "V_CurrentFlatAgent": {
                            "pnl_pct": 0.1,
                            "session_pnl_pct": 2.0,
                        },
                    },
                },
                players_payload={
                    "players": {
                        "V_LongMemoryLeader": {
                            "pnl_pct": 50.0,
                            "session_pnl_pct": 0.0,
                        },
                        "V_CurrentFlatLeader": {
                            "pnl_pct": 2.0,
                            "session_pnl_pct": 3.0,
                        },
                    },
                },
            )

        self.assertEqual(captured["players"][0][0], "LongMemoryLeader")
        self.assertEqual(captured["players"][0][1]["display_pnl_pct"], 50.0)
        self.assertEqual(captured["agents"][0][0], "LongMemoryAgent")
        self.assertEqual(captured["agents"][0][1]["display_pnl_pct"], 12.0)

    def test_memory_regime_rows_keep_individual_player_and_agent_labels(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._memory_regime_rows(
            players=[
                ("PlayerA", {
                    "per_regime": {
                        "bullish": {"pnl_pct": 1.5, "closed_trades": 4},
                        "neutral": {"pnl_pct": -0.5, "closed_trades": 2},
                    },
                }),
                ("PlayerB", {
                    "per_regime": {
                        "bearish": {"pnl_pct": 0.7, "closed_trades": 3},
                    },
                }),
            ],
            agents=[
                ("AgentA", {
                    "per_regime": {
                        "bullish": {"pnl_pct": 2.0, "closed_trades": 5},
                    },
                }),
            ],
        )

        labels = [row[0] for row in rows]
        self.assertIn("P:PlayerA", labels)
        self.assertIn("P:PlayerB", labels)
        self.assertIn("A:AgentA", labels)
        self.assertNotEqual(labels, ["Players", "Agents"])

    def test_bar_rows_gray_inactive_statuses_and_split_virtual_panteon(self):
        from panteon_v2.dashboards import png_renderer

        rows = [
            ("GoodAgent", {"display_pnl_pct": 1.2, "status": "live"}),
            ("BadAgent", {"display_pnl_pct": -2.4, "status": "quarantine"}),
            ("Virtual_Panteon", {"display_pnl_pct": 0.8, "status": "shadow_only"}),
        ]

        bar_rows, virtual_panteon = png_renderer._split_virtual_panteon(rows)
        colors = png_renderer._bar_colors_for_rows(bar_rows, fallback_color=png_renderer.BLUE)

        self.assertEqual([name for name, _ in bar_rows], ["GoodAgent", "BadAgent"])
        self.assertEqual(virtual_panteon, 0.8)
        self.assertEqual(colors, [png_renderer.GREEN, png_renderer.MUTED])

    def test_panteon_benchmark_draws_virtual_panteon_gray_dashed(self):
        from panteon_v2.dashboards import png_renderer

        class FakeAxis:
            def __init__(self):
                self.lines = []
                self.texts = []
                self.xlim = None

            def set_xlim(self, left, right):
                self.xlim = (left, right)

            def axvline(self, x, **kwargs):
                self.lines.append((x, kwargs))

            def text(self, x, y, text, **kwargs):
                self.texts.append((x, y, text, kwargs))

            def get_xaxis_transform(self):
                return "x-transform"

            def legend(self, **kwargs):
                pass

        ax = FakeAxis()
        png_renderer._draw_panteon_benchmark(
            ax,
            [1.0, -0.5],
            panteon_pnl_pct=2.0,
            virtual_panteon_pnl_pct=0.75,
        )

        self.assertEqual(ax.lines[0][1]["color"], png_renderer.GOLD)
        self.assertEqual(ax.lines[1][0], 0.75)
        self.assertEqual(ax.lines[1][1]["color"], png_renderer.MUTED)
        self.assertEqual(ax.lines[1][1]["linestyle"], (0, (2, 3)))


if __name__ == "__main__":
    unittest.main()
