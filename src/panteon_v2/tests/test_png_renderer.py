"""Tests for the matplotlib operator PNG backend."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch


class TestPngRenderer(unittest.TestCase):
    def test_write_operator_pngs_passes_panteon_benchmark_to_shadow_charts(self):
        from panteon_v2.dashboards import png_renderer

        captured = []

        def fake_operator(path, status, agents, players, *, panteon_pnl_pct):
            captured.append(("operator", panteon_pnl_pct))

        def fake_leaderboard(path, title, rows, *, panteon_pnl_pct):
            captured.append((title, panteon_pnl_pct))

        def fake_heatmap(path, title, rows):
            captured.append((title, None))

        with tempfile.TemporaryDirectory() as td, \
                patch.object(png_renderer, "_render_operator_dashboard",
                             side_effect=fake_operator), \
                patch.object(png_renderer, "_render_leaderboard",
                             side_effect=fake_leaderboard), \
                patch.object(png_renderer, "_render_regime_heatmap",
                             side_effect=fake_heatmap):
            png_renderer.write_operator_pngs(
                td,
                status={"pnl_pct": 2.75},
                agents_payload={"agents": {"V_AgentA": {"pnl_pct": 1.0}}},
                players_payload={"players": {"V_PlayerA": {"pnl_pct": 3.0}}},
            )

        self.assertIn(("operator", 2.75), captured)
        self.assertIn(("SHADOW AGENTS DASHBOARD", 2.75), captured)
        self.assertIn(("SHADOW PLAYERS DASHBOARD", 2.75), captured)


if __name__ == "__main__":
    unittest.main()
