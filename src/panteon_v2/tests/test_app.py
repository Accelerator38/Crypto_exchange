"""Тесты Phase 9 — production wiring."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from panteon_v2.app import (
    OutputWriter,
    PRODUCTION_PROFILES,
    ProductionPipeline,
    build_dryrun_pipeline,
    build_production_pipeline,
    load_v2_snapshot,
    main_loop,
    migrate_from_v1_memory_file,
    migrate_v1_regime_memory,
    save_v2_snapshot,
)
from panteon_v2.app.bootstrap import LiveExecutionConfig
from panteon_v2.app.main_loop import StepResult
from panteon_v2.attribution import PositionClosed
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import ExchangePosition, FakeExchange
from panteon_v2.execution import RiskLimitsConfig
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory
from panteon_v2.selection import (
    AgentRegistry,
    EnsemblePlayer,
    PlayerProfile,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.shadow.adapters import make_market_snapshot
from panteon_v2.shadow.feed import ReplayFeed
from panteon_v2.tests._helpers import FakeAgent


# ════════════════════════════════════════════════════════════════════
# Bootstrap
# ════════════════════════════════════════════════════════════════════


class TestBootstrap(unittest.TestCase):
    def test_empty_registry_rejected(self):
        with self.assertRaises(ValueError):
            build_dryrun_pipeline(registry=AgentRegistry())

    def test_negative_capital_rejected(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("X"))
        with self.assertRaises(ValueError):
            build_dryrun_pipeline(registry=reg, initial_capital=-1)

    def test_dryrun_pipeline_has_all_components(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("LiveTrendFollow"))
        reg.register(FakeAgent("LiveAfterShock"))
        reg.register(FakeAgent("LiveMeanRev"))
        p = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        self.assertIsInstance(p, ProductionPipeline)
        for attr in ("registry", "perf", "qm", "selector", "composer",
                     "strategist", "executor", "event_log", "ledger",
                     "renderer"):
            self.assertIsNotNone(getattr(p, attr))
        self.assertEqual(p.exchange_name, "DRY-RUN")
        self.assertEqual(p.initial_capital, 100.0)

    def test_production_pipeline_uses_real_exchange(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("A"))
        ex = FakeExchange(name="MEXC-CUSTOM")
        p = build_production_pipeline(
            registry=reg, exchange=ex, initial_capital=500.0,
        )
        self.assertEqual(p.exchange_name, "MEXC-CUSTOM")

    def test_production_profiles_present(self):
        self.assertGreaterEqual(len(PRODUCTION_PROFILES), 5)

    def test_live_startup_prefers_exchange_equity_when_capital_not_configured(self):
        from panteon_v2.app.startup import _resolve_initial_capital

        class LiveExchange(FakeExchange):
            def get_account_equity(self):
                return 40.25

        capital = _resolve_initial_capital(
            exchange_adapter=LiveExchange(name="BITGET"),
            mode="live_futures",
            requested_initial_capital=None,
            exchange_name="BITGET",
        )

        self.assertEqual(capital, 40.25)

    def test_live_startup_respects_explicit_initial_capital_override(self):
        from panteon_v2.app.startup import _resolve_initial_capital

        class LiveExchange(FakeExchange):
            def get_account_equity(self):
                return 40.25

        capital = _resolve_initial_capital(
            exchange_adapter=LiveExchange(name="BITGET"),
            mode="live_futures",
            requested_initial_capital=77.0,
            exchange_name="BITGET",
        )

        self.assertEqual(capital, 77.0)

    def test_sync_pipeline_balance_uses_live_exchange_equity(self):
        from panteon_v2.app.main_loop import sync_pipeline_balance

        class LiveExchange(FakeExchange):
            def get_account_equity(self):
                return 120.5

        reg = AgentRegistry()
        reg.register(FakeAgent("A"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(name="MEXC"),
            initial_capital=100.0,
        )

        self.assertEqual(sync_pipeline_balance(pipeline), 120.5)
        self.assertEqual(pipeline.current_balance, 120.5)

    def test_sync_pipeline_balance_stores_account_snapshot_total_assets(self):
        from panteon_v2.app.main_loop import sync_pipeline_balance

        class LiveExchange(FakeExchange):
            def get_account_snapshot(self):
                return {
                    "current_balance": 120.5,
                    "futures_equity": 120.5,
                    "available_balance": 101.25,
                    "spot_assets": 7.5,
                    "total_assets": 128.0,
                    "unrealized_pnl": 0.75,
                }

        reg = AgentRegistry()
        reg.register(FakeAgent("A"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(name="MEXC"),
            initial_capital=100.0,
        )

        self.assertEqual(sync_pipeline_balance(pipeline), 120.5)
        self.assertEqual(pipeline.current_balance, 120.5)
        self.assertEqual(pipeline.account_snapshot["total_assets"], 128.0)

    def test_runtime_risk_config_uses_settings_trade_fraction(self):
        from panteon_v2.app.startup import _risk_config_from_trade_fraction

        cfg = _risk_config_from_trade_fraction(0.035)

        self.assertEqual(cfg.capital_fraction, 0.035)
        self.assertTrue(cfg.floor_to_exchange_min_notional)


# ════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════


class TestMainLoop(unittest.TestCase):
    def _setup(self):
        reg = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock", "LiveMeanRev"]:
            reg.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))
        feed = ReplayFeed()
        for bar in range(1, 4):
            feed.append(make_market_snapshot(
                bar=bar, prices={"BTC": 100.0 + bar},
                regime="bullish",
            ))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=1000.0)
        return pipeline, feed

    def test_processes_all_bars(self):
        pipeline, feed = self._setup()
        steps = main_loop(pipeline, feed)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].bar, 1)
        self.assertEqual(steps[-1].bar, 3)

    def test_max_bars(self):
        pipeline, feed = self._setup()
        steps = main_loop(pipeline, feed, max_bars=2)
        self.assertEqual(len(steps), 2)

    def test_on_step_callback(self):
        pipeline, feed = self._setup()
        seen = []
        main_loop(pipeline, feed, on_step=lambda s: seen.append(s.bar))
        self.assertEqual(seen, [1, 2, 3])

    def test_on_idle_callback_and_limit(self):
        pipeline, feed = self._setup()
        seen = []
        steps = main_loop(
            pipeline,
            ReplayFeed(),
            sleep_between_polls_sec=0.0,
            max_idle_polls=2,
            on_idle=lambda n: seen.append(n),
        )
        self.assertEqual(steps, [])
        self.assertEqual(seen, [1, 2])

    def test_ledger_populated_after_loop(self):
        pipeline, feed = self._setup()
        main_loop(pipeline, feed)
        # После loop ledger содержит хоть какие-то events
        # (не обязательно closed — зависит от leader/voting)
        self.assertGreaterEqual(len(pipeline.event_log), 3)

    def test_quarantined_blocks_pipeline(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("BadAgent",
                              {"BTC": Action.FUT_LONG_FULL}))
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1, prices={"BTC": 100.0}, regime="bullish",
        ))
        pipeline = build_dryrun_pipeline(
            registry=reg, initial_capital=100.0,
            seed_quarantine={"BadAgent"},
        )
        steps = main_loop(pipeline, feed)
        # Карантинный → не выберется в leader
        self.assertEqual(len(steps), 1)
        self.assertIsNone(steps[0].leader)

    def test_signal_events_emitted_for_generated_signals(self):
        from panteon_v2.attribution import SignalEmitted

        pipeline, feed = self._setup()
        main_loop(pipeline, feed, max_bars=1)

        events = list(pipeline.event_log.query(event_types=[SignalEmitted]))
        self.assertGreaterEqual(len(events), 1)

    def test_shadow_tournament_runs_agents_without_real_leader(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        reg = AgentRegistry()
        reg.register(FakeAgent("ShadowAgent", {"BTC": Action.FUT_LONG_FULL}))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
        )
        pipeline.profiles = []
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1, prices={"BTC": 100.0}, regime="bullish",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(len(exchange.orders_log), 0)
        self.assertGreater(steps[0].n_shadow_filled, 0)
        self.assertTrue(perf_has_data := pipeline.perf.get("ShadowAgent", Regime.BULLISH).has_data)
        self.assertTrue(perf_has_data)

    def test_shadow_tournament_rates_players_and_agent_contribution(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        agent = FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        perf = PerformanceMemory(trade_fraction=1.0)
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        player = EnsemblePlayer(
            label="PlayerA",
            agents=[agent],
            weights={"AgentA": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(open_single=0.10, open_multi=0.10, open_floor=0.10),
        )

        summary = tournament.run_bar(
            make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish"),
            players=[player],
            balance_usd=1000.0,
        )

        self.assertEqual(summary.agent_filled, 1)
        self.assertEqual(summary.player_filled, 1)
        self.assertEqual(perf.get("AgentA", Regime.BULLISH).entries, 2)
        self.assertEqual(perf.get("PlayerA", Regime.BULLISH).entries, 1)

    def test_real_exchange_receives_only_selected_leader_not_virtual_actors(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("AgentB", {"ETH": Action.FUT_LONG_FULL}))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
        )
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(len(exchange.orders_log), steps[0].n_filled)
        self.assertGreater(steps[0].n_shadow_filled, steps[0].n_filled)

    def test_step_result_records_blocked_reasons(self):
        pipeline, feed = self._setup()
        pipeline.current_balance = 1.0
        pipeline.account_snapshot = {
            "current_balance": 1.0,
            "futures_equity": 1.0,
            "available_balance": 1.0,
            "spot_assets": 0.0,
            "total_assets": 1.0,
            "unrealized_pnl": 0.0,
        }

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].n_blocked, 1)
        self.assertEqual(
            steps[0].blocked_reasons,
            {"risk_limits: notional $0.10 < min $5.00": 1},
        )

    def test_shadow_does_not_consume_stateful_agent_before_real_vote(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        class OneShotAgent:
            label = "AgentA"

            def __init__(self):
                self.calls = 0

            def act(self, market):
                self.calls += 1
                if self.calls == 1:
                    return {"BTC": Action.FUT_LONG_FULL}
                return {}

        agent = OneShotAgent()
        reg = AgentRegistry()
        reg.register(agent)
        reg.register(FakeAgent("AgentB"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
        )
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(steps[0].leader, "DefaultEnsemble")
        self.assertGreater(steps[0].n_shadow_filled, 0)
        self.assertEqual(steps[0].n_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(len(exchange.orders_log), 1)
        self.assertEqual(agent.calls, 1)

    def test_signal_id_counter_continues_after_perf_snapshot_restore(self):
        from panteon_v2.app.main_loop import _max_existing_signal_id

        pipeline, _feed = self._setup()
        pipeline.perf.restore({
            "trade_fraction": 0.1,
            "state": {},
            "open": {},
            "seen_signal_ids": [17, 41],
        })

        self.assertEqual(_max_existing_signal_id(pipeline), 41)

    def test_real_vote_syncs_stateful_agent_to_tracker_before_act(self):
        class StatefulAgent:
            label = "Stateful"

            def __init__(self):
                self.pos = {"BTC": "long"}
                self.ep = {"BTC": 99.0}
                self.et = {"BTC": 0}
                self.seen_pos = []

            def act(self, market):
                self.seen_pos.append(self.pos.get("BTC"))
                if self.pos.get("BTC"):
                    return {"BTC": Action.FUT_CLOSE_ALL}
                return {"BTC": Action.FUT_LONG_FULL}

        agent = StatefulAgent()
        reg = AgentRegistry()
        reg.register(agent)
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[PlayerProfile(
                label="DefaultEnsemble",
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
                min_agents=2,
                max_agents=2,
            )],
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(agent.seen_pos, [None])
        self.assertEqual(steps[0].n_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(steps[0].n_blocked, 0)
        self.assertEqual(exchange.orders_log[0].trade.side, "long")
        self.assertIsNotNone(pipeline.executor._tracker.get("BTC"))

    def test_real_signal_firewall_drops_stale_close_before_events_and_exchange(self):
        from panteon_v2.attribution import SignalEmitted

        class StubbornCloseAgent:
            label = "StubbornClose"

            def __init__(self):
                self.pos = {"INJ": "long"}
                self.ep = {"INJ": 7.5}
                self.et = {"INJ": 10}

            def act(self, market):
                return {"INJ": Action.FUT_CLOSE_ALL}

        agent = StubbornCloseAgent()
        reg = AgentRegistry()
        reg.register(agent)
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[PlayerProfile(
                label="DefaultEnsemble",
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
                min_agents=2,
                max_agents=2,
            )],
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"INJ": 7.6},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(steps[0].n_blocked, 0)
        self.assertEqual(steps[0].n_filtered_real_signals, 1)
        self.assertEqual(len(exchange.orders_log), 0)
        self.assertEqual(
            list(pipeline.event_log.query(event_types=[SignalEmitted])),
            [],
        )
        self.assertIsNone(agent.pos["INJ"])

    def test_live_guard_caps_new_opens_per_bar(self):
        class MultiOpenAgent:
            label = "MultiOpen"

            def act(self, market):
                return {
                    "BTC": Action.FUT_LONG_FULL,
                    "ETH": Action.FUT_LONG_FULL,
                    "SOL": Action.FUT_LONG_FULL,
                }

        reg = AgentRegistry()
        reg.register(MultiOpenAgent())
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[PlayerProfile(
                label="DefaultEnsemble",
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(open_single=0.20, open_multi=0.20),
                min_agents=2,
                max_agents=2,
            )],
            live_execution_config=LiveExecutionConfig(max_new_opens_per_bar=1),
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0, "SOL": 25.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(steps[0].n_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(steps[0].n_filtered_real_signals, 2)
        self.assertEqual(steps[0].n_rate_limited_open_signals, 2)
        self.assertEqual(len(exchange.orders_log), 1)

    def test_live_guard_keeps_closes_when_open_cap_is_reached(self):
        class CloseAndOpenAgent:
            label = "CloseAndOpen"

            def act(self, market):
                return {
                    "BTC": Action.FUT_CLOSE_ALL,
                    "ETH": Action.FUT_CLOSE_ALL,
                    "SOL": Action.FUT_LONG_FULL,
                    "XRP": Action.FUT_LONG_FULL,
                }

        reg = AgentRegistry()
        reg.register(CloseAndOpenAgent())
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[PlayerProfile(
                label="DefaultEnsemble",
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(
                    open_single=0.20,
                    open_multi=0.20,
                    close_single=0.20,
                ),
                min_agents=2,
                max_agents=2,
            )],
            live_execution_config=LiveExecutionConfig(max_new_opens_per_bar=1),
        )
        tracker = pipeline.executor._tracker
        opened_at = datetime.now(timezone.utc)
        for sym, side, entry in (("BTC", "long", 100.0), ("ETH", "short", 50.0)):
            tracker.force_set(TrackedPosition(
                open_signal_id=1,
                sym=sym,
                side=side,
                entry_price=entry,
                qty=1.0,
                fee_open=0.0,
                by_player="DefaultEnsemble",
                by_agent="CloseAndOpen",
                opened_at=opened_at,
            ))
            exchange._positions[sym] = ExchangePosition(sym=sym, side=side, qty=1.0, entry=entry)

        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=2,
            prices={"BTC": 101.0, "ETH": 49.0, "SOL": 25.0, "XRP": 0.5},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(steps[0].n_signals, 3)
        self.assertEqual(steps[0].n_filled, 3)
        self.assertEqual(steps[0].n_rate_limited_open_signals, 1)
        self.assertFalse(pipeline.executor._tracker.has("BTC"))
        self.assertFalse(pipeline.executor._tracker.has("ETH"))
        self.assertEqual(len(exchange.orders_log), 3)


class TestOutputWriter(unittest.TestCase):
    def test_writer_creates_operator_files_before_first_bar(self):
        reg = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock", "LiveMeanRev"]:
            reg.register(FakeAgent(label))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            out = writer.output_dir
            for name in (
                "trading.log",
                "status.json",
                "leaderboard_agents.json",
                "leaderboard_players.json",
                "dashboard.txt",
                "dashboard.html",
                "dashboard_latest.png",
                "shadow_dashboard.png",
                "regime_dashboard.png",
            ):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            with open(os.path.join(out, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            self.assertEqual(status["version"], "v2")
            self.assertEqual(status["run_state"], "starting")
            self.assertEqual(status["bar_count"], 0)
            with open(os.path.join(out, "dashboard_latest.png"), "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
            writer.close()

    def test_status_and_trading_log_include_balance_and_position_count(self):
        from panteon_v2.execution.exchange import ExchangePosition

        class LiveExchange(FakeExchange):
            def get_account_snapshot(self):
                return {
                    "current_balance": 120.5,
                    "futures_equity": 120.5,
                    "available_balance": 101.25,
                    "spot_assets": 7.5,
                    "total_assets": 128.0,
                    "unrealized_pnl": 0.75,
                }

            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(sym="BTC", side="long", qty=0.01, entry=100.0),
                    "ETH": ExchangePosition(sym="ETH", side="short", qty=0.2, entry=50.0),
                }

        reg = AgentRegistry()
        reg.register(FakeAgent("LiveTrendFollow"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(name="MEXC"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.write(StepResult(
                bar=1,
                regime=Regime.NEUTRAL,
                leader="DefaultEnsemble",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=2,
                blocked_reasons={
                    "risk_limits: notional $1.74 < min $5.10": 2,
                },
            ))
            with open(os.path.join(writer.output_dir, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            with open(os.path.join(writer.output_dir, "trading.log"), "r", encoding="utf-8") as f:
                trading_log = f.read()
            writer.close()

        self.assertEqual(status["current_balance"], 120.5)
        self.assertEqual(status["futures_equity_usd"], 120.5)
        self.assertEqual(status["available_balance_usd"], 101.25)
        self.assertEqual(status["spot_assets_usd"], 7.5)
        self.assertEqual(status["total_assets_usd"], 128.0)
        self.assertEqual(status["exchange_positions_count"], 2)
        self.assertEqual(status["tracked_positions_count"], 0)
        self.assertEqual(
            status["blocked_reasons_bar"],
            {"risk_limits: notional $1.74 < min $5.10": 2},
        )
        self.assertIn("balance=$120.50", trading_log)
        self.assertIn("positions=2", trading_log)
        self.assertIn("blocked_reasons=", trading_log)
        self.assertIn("notional $1.74 < min $5.10 x2", trading_log)
        self.assertNotIn("assets=", trading_log)

    def test_json_writer_falls_back_when_atomic_replace_is_denied(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "status.json")
            with patch(
                "panteon_v2.app.output_writer.os.replace",
                side_effect=PermissionError("locked"),
            ):
                OutputWriter._write_json_atomic(path, {"ok": True})

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        self.assertEqual(data, {"ok": True})

    def test_recover_exchange_positions_seeds_tracker_from_perf_snapshot(self):
        from panteon_v2.app.startup import _recover_exchange_positions
        from panteon_v2.execution.exchange import ExchangePosition

        class LiveExchange(FakeExchange):
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(sym="BTC", side="long", qty=0.01, entry=100.0, leverage=2),
                }

        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(name="MEXC"),
            initial_capital=100.0,
        )
        pipeline.perf.restore({
            "trade_fraction": 0.1,
            "state": {},
            "open": {
                "real|DefaultEnsemble|BTC": {
                    "signal_id": 77,
                    "label": "DefaultEnsemble",
                    "regime": "bullish",
                    "side": "long",
                    "entry_price": 100.0,
                    "qty": 0.01,
                    "fee_open": 0.02,
                },
                "real|AgentA|BTC": {
                    "signal_id": 77,
                    "label": "AgentA",
                    "regime": "bullish",
                    "side": "long",
                    "entry_price": 100.0,
                    "qty": 0.01,
                    "fee_open": 0.02,
                },
            },
            "seen_signal_ids": [77],
        })

        recovered = _recover_exchange_positions(pipeline)

        tracker = pipeline.executor._tracker
        pos = tracker.get("BTC")
        self.assertEqual(recovered, 1)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.open_signal_id, 77)
        self.assertEqual(pos.by_player, "DefaultEnsemble")
        self.assertEqual(pos.by_agent, "AgentA")

    def test_recover_exchange_positions_ignores_shadow_open_memory(self):
        from panteon_v2.app.startup import _recover_exchange_positions
        from panteon_v2.execution.exchange import ExchangePosition

        class LiveExchange(FakeExchange):
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(sym="BTC", side="long", qty=0.01, entry=100.0, leverage=2),
                }

        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(name="MEXC"),
            initial_capital=100.0,
        )
        pipeline.perf.restore({
            "trade_fraction": 0.1,
            "state": {},
            "open": {
                "shadow:player:DefaultEnsemble|DefaultEnsemble|BTC": {
                    "signal_id": 91,
                    "label": "DefaultEnsemble",
                    "regime": "bullish",
                    "side": "long",
                    "entry_price": 100.0,
                    "qty": 0.01,
                    "fee_open": 0.02,
                },
                "shadow:agent:AgentA|AgentA|BTC": {
                    "signal_id": 92,
                    "label": "AgentA",
                    "regime": "bullish",
                    "side": "long",
                    "entry_price": 100.0,
                    "qty": 0.01,
                    "fee_open": 0.02,
                },
            },
            "seen_signal_ids": [91, 92],
        })

        recovered = _recover_exchange_positions(pipeline)

        pos = pipeline.executor._tracker.get("BTC")
        self.assertEqual(recovered, 1)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.open_signal_id, 0)
        self.assertEqual(pos.by_player, "RecoveredExchangePosition")
        self.assertEqual(pos.by_agent, "")

    def test_player_leaderboard_uses_performance_memory(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        migrate_v1_regime_memory({
            "bullish": {
                "DefaultEnsemble": {
                    "samples": 5,
                    "wins": 3,
                    "losses": 2,
                    "pnl_pct": 1.2,
                },
                "AgentA": {
                    "samples": 4,
                    "wins": 2,
                    "losses": 2,
                    "pnl_pct": 0.5,
                },
            },
        }, pipeline.perf)

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer._write_leaderboards()
            with open(os.path.join(writer.output_dir, "leaderboard_players.json"),
                      "r", encoding="utf-8") as f:
                players = json.load(f)["players"]
            writer.close()

        self.assertIn("V_DefaultEnsemble", players)
        self.assertAlmostEqual(players["V_DefaultEnsemble"]["pnl_pct"], 1.2)
        self.assertNotIn("V_AgentA", players)

    def test_leaderboard_and_status_include_session_local_virtual_leader_metrics(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        migrate_v1_regime_memory({
            "bullish": {
                "DefaultEnsemble": {
                    "samples": 5,
                    "wins": 3,
                    "losses": 2,
                    "pnl_pct": 1.2,
                },
            },
        }, pipeline.perf)

        open_sig = Signal(
            id=101,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_agent="AgentA",
            by_player="DefaultEnsemble",
            position_scope="shadow:player:DefaultEnsemble",
        )
        close_sig = Signal(
            id=102,
            bar=2,
            sym="BTC",
            action=Action.FUT_CLOSE_ALL,
            price=110.0,
            regime=Regime.BULLISH,
            by_agent="AgentA",
            by_player="DefaultEnsemble",
            position_scope="shadow:player:DefaultEnsemble",
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            pipeline.perf.update_from_trade(
                Trade(signal_id=101, bar=1, sym="BTC", side="long",
                      qty=1.0, fill_price=100.0, fee=0.0),
                open_sig,
            )
            pipeline.perf.update_from_trade(
                Trade(signal_id=102, bar=2, sym="BTC", side="long",
                      qty=1.0, fill_price=110.0, fee=0.0),
                close_sig,
            )
            writer.write(StepResult(
                bar=2,
                regime=Regime.BULLISH,
                leader="DefaultEnsemble",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
            ))
            writer._write_leaderboards()
            with open(os.path.join(writer.output_dir, "leaderboard_players.json"),
                      "r", encoding="utf-8") as f:
                players = json.load(f)["players"]
            with open(os.path.join(writer.output_dir, "status.json"),
                      "r", encoding="utf-8") as f:
                status = json.load(f)
            writer.close()

        row = players["V_DefaultEnsemble"]
        self.assertAlmostEqual(row["pnl_pct"], 2.2)
        self.assertAlmostEqual(row["session_pnl_pct"], 1.0)
        self.assertEqual(row["session_closed_trades"], 1)
        self.assertEqual(status["live_session"]["leader"], "DefaultEnsemble")
        self.assertAlmostEqual(
            status["live_session"]["leader_virtual_session_pnl_pct"],
            1.0,
        )
        self.assertEqual(
            status["live_session"]["leader_virtual_session_closed_trades"],
            1,
        )

    def test_status_separates_panteon_owned_from_external_recovered_positions(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=100.0,
        )
        opened_at = datetime.now(timezone.utc)
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=10,
            sym="BTC",
            side="long",
            entry_price=100.0,
            qty=1.0,
            fee_open=0.0,
            by_player="DefaultEnsemble",
            by_agent="AgentA",
            opened_at=opened_at,
        ))
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=0,
            sym="ETH",
            side="short",
            entry_price=50.0,
            qty=1.0,
            fee_open=0.0,
            by_player="RecoveredExchangePosition",
            by_agent="",
            opened_at=opened_at,
        ))
        exchange._positions["BTC"] = ExchangePosition(
            sym="BTC", side="long", qty=1.0, entry=100.0, unrealized_pnl=0.75,
        )
        exchange._positions["ETH"] = ExchangePosition(
            sym="ETH", side="short", qty=1.0, entry=50.0, unrealized_pnl=-0.25,
        )
        pipeline.event_log.emit(PositionClosed(
            bar=2,
            trace_id="owned-close",
            open_signal_id=1,
            close_signal_id=2,
            sym="SOL",
            side="long",
            entry=10.0,
            exit=9.9,
            qty=1.0,
            realized_pnl=-0.10,
            by_player="DefaultEnsemble",
            by_agent="AgentA",
        ))
        pipeline.event_log.emit(PositionClosed(
            bar=2,
            trace_id="external-close",
            open_signal_id=0,
            close_signal_id=3,
            sym="XRP",
            side="long",
            entry=1.0,
            exit=0.8,
            qty=1.0,
            realized_pnl=-0.20,
            by_player="RecoveredExchangePosition",
            by_agent="",
        ))

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.write(StepResult(
                bar=2,
                regime=Regime.NEUTRAL,
                leader="DefaultEnsemble",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
            ))
            with open(os.path.join(writer.output_dir, "status.json"),
                      "r", encoding="utf-8") as f:
                status = json.load(f)
            writer.close()

        live = status["live_session"]
        self.assertAlmostEqual(live["panteon_owned_realized_pnl_usd"], -0.10)
        self.assertAlmostEqual(live["external_realized_pnl_usd"], -0.20)
        self.assertAlmostEqual(live["panteon_owned_unrealized_pnl_usd"], 0.75)
        self.assertAlmostEqual(live["external_unrealized_pnl_usd"], -0.25)
        self.assertAlmostEqual(live["panteon_owned_pnl_pct"], 0.65)
        self.assertEqual(live["panteon_owned_positions_count"], 1)
        self.assertEqual(live["external_positions_count"], 1)
        self.assertEqual(live["comparison_scope"], "panteon_owned")


# ════════════════════════════════════════════════════════════════════
# Migration
# ════════════════════════════════════════════════════════════════════


class TestMigration(unittest.TestCase):
    def test_migrate_v1_dict(self):
        v1 = {
            "bullish": {
                "FundingArb": {"samples": 10, "wins": 3, "losses": 7,
                              "pnl_pct": -1.5},
                "LiveAfterShock": {"samples": 8, "wins": 5, "losses": 3,
                                   "pnl_pct": 2.0},
            },
            "neutral": {
                "LiveMeanRev": {"samples": 12, "wins": 6, "losses": 6,
                                "pnl_pct": 0.0},
            },
        }
        perf = PerformanceMemory()
        report = migrate_v1_regime_memory(v1, perf)
        self.assertEqual(report.n_labels, 3)
        self.assertEqual(report.n_regime_pairs, 3)
        # Check perf populated
        m = perf.get("FundingArb", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 10)
        self.assertAlmostEqual(m.pnl_pct, -1.5)

    def test_migrate_strips_v_prefix(self):
        v1 = {"bullish": {"V_FundingArb": {"samples": 5, "wins": 2, "losses": 3,
                                            "pnl_pct": -0.5}}}
        perf = PerformanceMemory()
        report = migrate_v1_regime_memory(v1, perf)
        self.assertEqual(report.n_labels, 1)
        self.assertTrue(perf.get("FundingArb").has_data)
        self.assertFalse(perf.get("V_FundingArb").has_data)

    def test_invalid_regime_warning(self):
        # Junk regime string должен попасть в neutral (благодаря Regime.from_string)
        v1 = {"junk_regime": {"X": {"samples": 5, "wins": 2, "losses": 3,
                                     "pnl_pct": 1.0}}}
        perf = PerformanceMemory()
        report = migrate_v1_regime_memory(v1, perf)
        # Junk → NEUTRAL
        self.assertTrue(perf.get("X", regime=Regime.NEUTRAL).has_data)

    def test_migrate_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v1_memory.json")
            with open(path, "w") as f:
                json.dump({"_regime_memory": {
                    "bullish": {"X": {"samples": 5, "wins": 3, "losses": 2,
                                       "pnl_pct": 1.0}},
                }}, f)
            perf = PerformanceMemory()
            report = migrate_from_v1_memory_file(path, perf)
            self.assertEqual(report.n_labels, 1)

    def test_migrate_player_regime_memory_maps_v1_player_labels(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v1_player_memory.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"player_regime_memory": {
                    "neutral": {
                        "V_PanteonTrendResearch": {
                            "samples": 11,
                            "wins": 7,
                            "losses": 4,
                            "avg_pnl": 1.25,
                        },
                    },
                }}, f)

            perf = PerformanceMemory()
            report = migrate_from_v1_memory_file(path, perf)

            self.assertEqual(report.n_labels, 1)
            self.assertTrue(perf.get("TrendResearch", Regime.NEUTRAL).has_data)
            self.assertFalse(perf.get("V_PanteonTrendResearch", Regime.NEUTRAL).has_data)

    def test_migrate_keeps_solo_player_labels_separate_from_agents(self):
        v1 = {"neutral": {
            "LiveTrendFollow": {
                "samples": 3,
                "wins": 2,
                "losses": 1,
                "pnl_pct": 3.0,
            },
            "V_SoloLiveTrendFollow": {
                "samples": 8,
                "wins": 1,
                "losses": 7,
                "pnl_pct": -4.0,
            },
        }}
        perf = PerformanceMemory()

        migrate_v1_regime_memory(v1, perf)

        agent = perf.get("LiveTrendFollow", Regime.NEUTRAL)
        solo_player = perf.get("SoloLiveTrendFollow", Regime.NEUTRAL)
        self.assertEqual(agent.closed_trades, 3)
        self.assertAlmostEqual(agent.pnl_pct, 3.0)
        self.assertEqual(solo_player.closed_trades, 8)
        self.assertAlmostEqual(solo_player.pnl_pct, -4.0)

    def test_startup_migrates_when_v2_snapshot_is_empty(self):
        from panteon_v2.app.startup import start_production

        with tempfile.TemporaryDirectory() as td:
            snapshot_path = os.path.join(td, "mexc_snapshot.json")
            v1_path = os.path.join(td, "v1_player_memory.json")
            with open(snapshot_path, "w", encoding="utf-8") as f:
                json.dump({
                    "trade_fraction": 0.1,
                    "state": {},
                    "open": {},
                    "seen_signal_ids": [],
                }, f)
            with open(v1_path, "w", encoding="utf-8") as f:
                json.dump({"player_regime_memory": {
                    "neutral": {
                        "V_PanteonTrendResearch": {
                            "samples": 9,
                            "wins": 5,
                            "losses": 4,
                            "avg_pnl": 0.75,
                        },
                    },
                }}, f)

            def register_one_agent(registry, **_kwargs):
                registry.register(FakeAgent("LiveTrendFollow"))
                return ["LiveTrendFollow"]

            with patch("panteon_v2.app.startup.resolve_exchange",
                       return_value=FakeExchange(name="MEXC")), \
                 patch("panteon_v2.app.startup.register_all_v1_agents",
                       side_effect=register_one_agent), \
                 patch("panteon_v2.app.startup._resolve_trade_fraction",
                       return_value=0.1):
                rc = start_production(
                    exchange="MEXC",
                    mode="paper",
                    snapshot_path=snapshot_path,
                    migrate_from_v1=v1_path,
                    use_v1_bridge=False,
                    max_bars=0,
                    results_root=td,
                    sleep_between_polls_sec=0.0,
                )

            self.assertEqual(rc, 0)
            with open(snapshot_path, "r", encoding="utf-8") as f:
                migrated = json.load(f)
            self.assertIn("TrendResearch|neutral", migrated["state"])

    def test_missing_file(self):
        perf = PerformanceMemory()
        report = migrate_from_v1_memory_file("/nonexistent/path.json", perf)
        self.assertEqual(report.n_labels, 0)
        self.assertTrue(any("not found" in w for w in report.warnings))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "snap.json")
            perf1 = PerformanceMemory()
            migrate_v1_regime_memory({"bullish": {
                "Z": {"samples": 7, "wins": 4, "losses": 3, "pnl_pct": 0.5}
            }}, perf1)
            save_v2_snapshot(perf1, path)
            perf2 = PerformanceMemory()
            ok = load_v2_snapshot(perf2, path)
            self.assertTrue(ok)
            self.assertAlmostEqual(
                perf1.get("Z", regime=Regime.BULLISH).pnl_pct,
                perf2.get("Z", regime=Regime.BULLISH).pnl_pct,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
