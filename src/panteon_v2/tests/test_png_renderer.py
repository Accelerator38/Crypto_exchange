"""Tests for the matplotlib operator PNG backend."""

from __future__ import annotations

from datetime import datetime, timezone
import tempfile
import unittest
from unittest.mock import patch


class TestPngRenderer(unittest.TestCase):
    def test_operator_dashboard_title_uses_global_version_and_build(self):
        from panteon_v2.dashboards import png_renderer

        with patch.object(png_renderer, "_git_build_id", return_value="6ea1c25b"):
            self.assertEqual(
                png_renderer._operator_dashboard_title({}),
                "Panteon v3 - сборка 6ea1c25b",
            )

        self.assertEqual(
            png_renderer._operator_dashboard_title({
                "global_version": "Panteon v3",
                "build_id": "1234567890abcdef",
            }),
            "Panteon v3 - сборка 12345678",
        )

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

    def test_panteon_pnl_prefers_clean_live_scope_for_dashboards(self):
        from panteon_v2.dashboards import png_renderer

        value = png_renderer._panteon_pnl_pct({
            "live_session": {
                "clean_panteon_pnl_pct": -0.17,
                "panteon_owned_pnl_pct": -1.34,
            },
        })

        self.assertAlmostEqual(value, -0.17)

    def test_write_operator_pngs_uses_combined_visual_dashboards(self):
        from panteon_v2.dashboards import png_renderer

        captured = []

        def fake_operator(path, status, agents, players, *, panteon_pnl_pct):
            captured.append(("operator", path.name, panteon_pnl_pct))

        def fake_shadow(path, agents, players, *, panteon_pnl_pct):
            captured.append(("shadow", path.name, len(agents), len(players), panteon_pnl_pct))

        def fake_regime(path, agents, players, *, status=None):
            captured.append(("regime", path.name, len(agents), len(players), status))

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
        self.assertIn(("regime", "regime_dashboard.png", 1, 1, {"pnl_pct": 2.75}), captured)
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

    def test_asset_rows_are_sorted_by_asset_size_with_panteon_highlight_only(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._asset_rows(
            {"initial_capital": 100.0, "panteon_equity_usd": 100.0},
            agents=[("LiveAfterShock", {"session_pnl_pct": 2.0})],
            players=[("Antonius_conservative", {"session_pnl_pct": 1.0})],
        )

        self.assertEqual(
            [row["label"] for row in rows],
            ["A:LiveAfterShock", "P:Antonius_conservative", "PANTEON"],
        )
        self.assertEqual(rows[-1]["kind"], "panteon")

    def test_flash_selected_rows_count_current_flash_actors(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._flash_selected_rows({
            "flash": {
                "selected_actors_by_symbol": {
                    "BTC": "NoTrade",
                    "ETH": "Antonius_conservative",
                    "SOL": "Antonius_conservative",
                },
            },
        })

        self.assertEqual(rows[0]["label"], "Antonius_conservative")
        self.assertEqual(rows[0]["count"], 2)
        self.assertEqual(rows[0]["symbols"], ("ETH", "SOL"))
        self.assertEqual(rows[1]["label"], "NoTrade")

    def test_flash_candidate_rows_include_actual_flash_candidates(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._flash_candidate_rows({
            "flash": {
                "decisions": [
                    {
                        "symbol": "BTC",
                        "selected_actor": "NoTrade",
                        "candidates": [
                            {
                                "label": "NoTrade",
                                "actor_type": "no_trade",
                                "score": 0.0,
                                "rejected": False,
                            },
                            {
                                "label": "Antonius_conservative",
                                "actor_type": "ensemble",
                                "score": 1.4,
                                "action": "FUT_LONG_FULL",
                                "rejected": False,
                            },
                            {
                                "label": "MomentumScalper",
                                "actor_type": "agent",
                                "score": 1.8,
                                "action": "HOLD",
                                "rejected": True,
                                "reason": "inactive",
                            },
                        ],
                    },
                    {
                        "symbol": "ETH",
                        "selected_actor": "Antonius_conservative",
                        "candidates": [
                            {
                                "label": "Antonius_conservative",
                                "actor_type": "ensemble",
                                "score": 1.9,
                                "action": "FUT_SHORT_FULL",
                                "rejected": False,
                            },
                        ],
                    },
                ],
            },
        })

        self.assertEqual(rows[0]["label"], "Antonius_conservative")
        self.assertEqual(rows[0]["actor_type"], "ensemble")
        self.assertEqual(rows[0]["symbols"], ("BTC", "ETH"))
        self.assertEqual(rows[0]["actionable_count"], 2)
        self.assertEqual(rows[0]["best_score"], 1.9)
        self.assertEqual(rows[1]["label"], "MomentumScalper")
        self.assertEqual(rows[1]["rejected_count"], 1)

    def test_flash_candidate_rows_aggregate_technical_overlay_counts(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._flash_candidate_rows({
            "flash": {
                "decisions": [
                    {
                        "symbol": "BTC",
                        "selected_actor": "TechAgent",
                        "candidates": [
                            {
                                "label": "TechAgent",
                                "actor_type": "agent",
                                "effective_score": 1.2,
                                "action": "FUT_LONG_FULL",
                                "rejected": False,
                                "technical_alignment": "long_aligned",
                                "technical_score_adjustment": 0.10,
                            },
                        ],
                    },
                    {
                        "symbol": "ETH",
                        "selected_actor": "NoTrade",
                        "candidates": [
                            {
                                "label": "TechAgent",
                                "actor_type": "agent",
                                "effective_score": 0.7,
                                "action": "FUT_LONG_FULL",
                                "rejected": True,
                                "reason": "technical_long_misaligned",
                                "technical_alignment": "long_misaligned",
                                "technical_score_adjustment": -0.25,
                            },
                        ],
                    },
                ],
            },
        })

        row = next(item for item in rows if item["label"] == "TechAgent")

        self.assertEqual(row["technical_aligned_count"], 1)
        self.assertEqual(row["technical_misaligned_count"], 1)
        self.assertAlmostEqual(row["technical_score_adjustment"], -0.15)

    def test_memory_dashboard_uses_cumulative_memory_pnl(self):
        from panteon_v2.dashboards import png_renderer

        captured = {}

        def fake_operator(path, status, agents, players, *, panteon_pnl_pct):
            pass

        def fake_shadow(path, agents, players, *, panteon_pnl_pct):
            pass

        def fake_regime(path, agents, players, *, status=None):
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

    def test_visual_dashboards_use_strategy_players_as_agent_fallback(self):
        from panteon_v2.dashboards import png_renderer

        captured = {}

        def fake_operator(path, status, agents, players, *, panteon_pnl_pct):
            captured["operator_agents"] = agents
            captured["operator_players"] = players

        def fake_shadow(path, agents, players, *, panteon_pnl_pct):
            captured["shadow_agents"] = agents
            captured["shadow_players"] = players

        def fake_regime(path, agents, players, **kwargs):
            captured["regime_agents"] = agents
            captured["regime_players"] = players

        def fake_memory(path, agents, players):
            captured["memory_agents"] = agents
            captured["memory_players"] = players

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
                agents_payload={"agents": {}},
                players_payload={
                    "players": {
                        "V_DefaultEnsemble": {
                            "pnl_pct": 1.0,
                            "actor_pool_kind": "profile",
                        },
                        "V_LiveAfterShock": {
                            "pnl_pct": 2.0,
                            "actor_pool_kind": "strategy",
                        },
                    },
                },
            )

        self.assertEqual([name for name, _ in captured["shadow_agents"]], ["LiveAfterShock"])
        self.assertEqual([name for name, _ in captured["shadow_players"]], ["DefaultEnsemble"])
        self.assertTrue(captured["shadow_agents"][0][1]["visual_agent_fallback"])
        self.assertEqual([name for name, _ in captured["memory_agents"]], ["LiveAfterShock"])

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

    def test_regime_dashboards_use_all_canonical_regimes(self):
        from panteon_v2.dashboards import png_renderer
        from panteon_v2.domain.types import Regime

        self.assertEqual(
            png_renderer._regime_labels(),
            [regime.label for regime in Regime],
        )

    def test_memory_regime_rows_include_extended_regimes(self):
        from panteon_v2.dashboards import png_renderer
        from panteon_v2.domain.types import Regime

        rows = png_renderer._memory_regime_rows(
            players=[
                ("PlayerA", {
                    "per_regime": {
                        "range_low_vol": {"pnl_pct": 0.4, "closed_trades": 1},
                        "choppy_down": {"pnl_pct": -0.9, "closed_trades": 2},
                        "mixed_rotational": {"pnl_pct": 1.2, "closed_trades": 3},
                    },
                }),
            ],
            agents=[],
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0][1]), len(Regime))
        values = dict(zip(png_renderer._regime_labels(), rows[0][1]))
        self.assertEqual(values["range_low_vol"], 0.4)
        self.assertEqual(values["choppy_down"], -0.9)
        self.assertEqual(values["mixed_rotational"], 1.2)

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

    def test_bar_rows_gray_quarantine_flags_without_status(self):
        from panteon_v2.dashboards import png_renderer

        rows = [
            ("QuarantinedWinner", {"display_pnl_pct": 2.4, "quarantined": True}),
            ("QuarantinedLoser", {"display_pnl_pct": -1.1, "is_quarantined": True}),
            ("LiveLoser", {"display_pnl_pct": -0.7, "quarantined": False}),
        ]

        colors = png_renderer._bar_colors_for_rows(
            rows,
            fallback_color=png_renderer.BLUE,
        )

        self.assertEqual(
            colors,
            [png_renderer.MUTED, png_renderer.MUTED, png_renderer.RED],
        )

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

    def test_current_session_rows_exclude_inactive_pool_members(self):
        from panteon_v2.dashboards import png_renderer

        rows = [
            ("LiveParticipant", {"session_signals": 2, "session_pnl_pct": 0.1}),
            ("OpenPositionAgent", {"session_signals": 0, "session_pnl_pct": 0.0}),
            ("OldMemoryOnly", {"pnl_pct": 5.0, "session_signals": 0, "session_pnl_pct": 0.0}),
        ]
        status = {
            "real_trading_actors": {
                "open_positions": [{"agent": "OpenPositionAgent"}],
            },
        }

        filtered = png_renderer._current_session_rows(
            rows,
            status=status,
            actor_kind="agent",
        )

        self.assertEqual(
            [name for name, _ in filtered],
            ["LiveParticipant", "OpenPositionAgent"],
        )

    def test_flash_diagnostics_explain_no_trade_in_plain_terms(self):
        from panteon_v2.dashboards import png_renderer

        info = png_renderer._flash_diagnostics({
            "flash": {
                "selected_actors_by_symbol": {
                    "BTC": "NoTrade",
                    "ETH": "NoTrade",
                },
                "decisions": [
                    {
                        "symbol": "BTC",
                        "selected_actor": "NoTrade",
                        "candidates": [
                            {"label": "A", "actor_type": "agent", "rejected": True, "reason": "inactive"},
                            {"label": "B", "actor_type": "agent", "rejected": True, "reason": "raw_suppressed_by_solo"},
                        ],
                    },
                    {
                        "symbol": "ETH",
                        "selected_actor": "NoTrade",
                        "candidates": [
                            {"label": "A", "actor_type": "agent", "rejected": True, "reason": "inactive"},
                        ],
                    },
                ],
            },
        })

        self.assertTrue(info["all_no_trade"])
        self.assertEqual(info["symbols_total"], 2)
        self.assertEqual(info["top_reasons"][0], ("inactive", 2))

    def test_flash_diagnostics_use_compact_top_rejected_candidates(self):
        from panteon_v2.dashboards import png_renderer

        info = png_renderer._flash_diagnostics({
            "flash": {
                "selected_actors_by_symbol": {
                    "BTC": "NoTrade",
                },
                "decisions": [
                    {
                        "symbol": "BTC",
                        "selected_actor": "NoTrade",
                        "candidates": [],
                        "top_rejected_candidates": [
                            {
                                "label": "DefaultEnsemble",
                                "actor_type": "ensemble",
                                "rejected": True,
                                "reason": "expected_edge_below_cost",
                            },
                        ],
                    },
                ],
                "gate_funnel_by_symbol": {
                    "DOGE": {
                        "top_blocker": "min_executable_notional",
                    },
                },
            },
        })

        self.assertTrue(info["all_no_trade"])
        self.assertEqual(info["rejected_candidates"], 1)
        self.assertIn(("expected_edge_below_cost", 1), info["top_reasons"])
        self.assertIn(("min_executable_notional", 1), info["top_reasons"])

    def test_price_history_rows_normalize_status_payload(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._price_history_rows({
            "price_history": [
                {"bar": 1, "regime": "bullish", "prices": {"BTC": "100", "BAD": "x"}},
                {"bar": 2, "regime": "neutral", "prices": {"BTC": 101, "ETH": 50}},
            ],
        })

        self.assertEqual(rows[0]["bar"], 1)
        self.assertEqual(rows[0]["prices"], {"BTC": 100.0})
        self.assertEqual(rows[1]["prices"], {"BTC": 101.0, "ETH": 50.0})

    def test_price_history_rows_add_time_axis_from_status_timestamp(self):
        from panteon_v2.dashboards import png_renderer

        rows = png_renderer._price_history_rows({
            "timestamp": "2026-06-03T19:00:10+00:00",
            "timeframe": "bridge_poll",
            "bar_count": 12,
            "price_history": [
                {"bar": 10, "regime": "neutral", "prices": {"BTC": 100}},
                {"bar": 12, "regime": "neutral", "prices": {"BTC": 101}},
            ],
        })

        self.assertEqual(rows[0]["timestamp"].isoformat(), "2026-06-03T19:00:00+00:00")
        self.assertEqual(rows[1]["timestamp"].isoformat(), "2026-06-03T19:00:10+00:00")

    def test_equity_curve_does_not_fabricate_linear_history(self):
        from panteon_v2.dashboards import png_renderer

        self.assertEqual(png_renderer._equity_curve({"pnl_pct": 10.0}), [])
        self.assertEqual(
            png_renderer._equity_curve({"equity_curve": [100.0, 98.0, 101.0]}),
            [100.0, 98.0, 101.0],
        )

    def test_equity_curves_use_common_utc_timeline_and_panteon(self):
        from panteon_v2.dashboards import png_renderer

        status = {
            "timestamp": "2026-06-06T10:00:10+00:00",
            "timeframe": "bridge_poll",
            "clean_panteon_equity_curve": [200.0, 202.0, 204.0],
        }
        timeline = png_renderer._common_equity_timeline(
            status,
            [("ShortActor", {"session_equity_curve": [100.0, 103.0]})],
        )

        self.assertEqual(
            [item.isoformat() for item in timeline],
            [
                "2026-06-06T10:00:00+00:00",
                "2026-06-06T10:00:05+00:00",
                "2026-06-06T10:00:10+00:00",
            ],
        )
        x_values, y_values = png_renderer._dashboard_equity_curve_points(
            {"session_equity_curve": [100.0, 103.0]},
            timeline,
        )
        self.assertEqual(x_values, timeline)
        self.assertEqual(y_values, [100.0, 100.0, 103.0])

        p_x, p_y = png_renderer._panteon_equity_curve_points(status, timeline)
        self.assertEqual(p_x, timeline)
        self.assertEqual(p_y, [100.0, 101.0, 102.0])

    def test_timestamped_equity_curve_is_step_aligned_to_timeline(self):
        from panteon_v2.dashboards import png_renderer

        timeline = [
            datetime(2026, 6, 6, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 6, 6, 10, 0, 5, tzinfo=timezone.utc),
            datetime(2026, 6, 6, 10, 0, 10, tzinfo=timezone.utc),
        ]

        x_values, y_values = png_renderer._dashboard_equity_curve_points(
            {
                "session_equity_curve": [100.0, 110.0],
                "session_equity_curve_timestamps": [
                    "2026-06-06T10:00:05+00:00",
                    "2026-06-06T10:00:10+00:00",
                ],
            },
            timeline,
        )

        self.assertEqual(x_values, timeline)
        self.assertEqual(y_values, [100.0, 100.0, 110.0])


if __name__ == "__main__":
    unittest.main()
