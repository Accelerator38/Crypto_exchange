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
from panteon_v2.attribution import (
    CandidateScored,
    DecisionStarted,
    EventLog,
    LeaderSelected,
    OrderFilled,
    OrderRejected,
    OrderSent,
    PositionClosed,
    PositionOpened,
    ShadowActorUpdated,
    SignalEmitted,
)
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import ExchangePosition, FakeExchange
from panteon_v2.execution import RiskLimitsConfig
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory
from panteon_v2.selection import (
    AgentRegistry,
    EnsemblePlayer,
    NoTradePlayer,
    PlayerProfile,
    SwitchDecision,
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

    def test_exchange_health_blocks_new_opens_but_keeps_audit_reason(self):
        class HealthExchange(FakeExchange):
            def get_account_snapshot(self):
                return {
                    "current_balance": 120.0,
                    "futures_equity": 120.0,
                    "available_balance": 120.0,
                    "total_assets": 120.0,
                    "data_health": {"recent_data_error": "ticker timeout"},
                }

        reg = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock", "LiveMeanRev"]:
            reg.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=HealthExchange(name="BITGET"),
            initial_capital=100.0,
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(steps[0].n_blocked, 1)
        self.assertEqual(len(pipeline.executor._exchange.orders_log), 0)
        rejected = list(pipeline.event_log.query(event_types=[OrderRejected]))
        self.assertEqual(len(rejected), 1)
        self.assertIn("exchange_health", rejected[0].reason)
        self.assertIn("ticker timeout", rejected[0].reason)

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

    def test_candidate_score_events_emit_full_candidate_list(self):
        pipeline, feed = self._setup()
        pipeline.mode = "paper"
        pipeline.timeframe = "1m"
        pipeline.run_id = "run-test"
        pipeline.session_id = "session-test"

        main_loop(pipeline, feed, max_bars=1)

        decisions = list(pipeline.event_log.query(event_types=[DecisionStarted]))
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].exchange, "DRY-RUN")
        self.assertEqual(decisions[0].symbol, "BTC")
        self.assertEqual(decisions[0].timeframe, "1m")
        self.assertEqual(decisions[0].mode, "paper")
        self.assertEqual(decisions[0].run_id, "run-test")
        self.assertEqual(decisions[0].session_id, "session-test")

        scored = list(pipeline.event_log.query(event_types=[CandidateScored]))
        self.assertGreaterEqual(len(scored), 1)
        self.assertTrue(any(ev.selected_by_pantheon for ev in scored))
        self.assertTrue(all(ev.decision_id for ev in scored))
        self.assertTrue(all(ev.memory_keys_read for ev in scored))
        for ev in scored:
            self.assertEqual(ev.exchange, "DRY-RUN")
            self.assertEqual(ev.symbol, "BTC")
            self.assertEqual(ev.timeframe, "1m")
            self.assertEqual(ev.mode, "paper")
            self.assertEqual(ev.run_id, "run-test")
            self.assertEqual(ev.session_id, "session-test")

        selected = list(pipeline.event_log.query(event_types=[LeaderSelected]))
        self.assertGreaterEqual(len(selected), 1)
        self.assertEqual(selected[0].exchange, "DRY-RUN")
        self.assertEqual(selected[0].session_id, "session-test")

    def test_real_selection_reuses_pre_shadow_candidate_snapshot(self):
        reg = AgentRegistry()
        agent = FakeAgent("A")
        reg.register(agent)
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        profile = PlayerProfile(
            label="Only",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=1,
            min_agents=1,
        )
        player = EnsemblePlayer(
            label="Only",
            agents=[agent],
            weights={"A": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class CountingComposer:
            def __init__(self):
                self.calls = 0

            def compose_from_profile_with_fallback(self, profile, regime):
                self.calls += 1
                return player

        composer = CountingComposer()
        pipeline.profiles = [profile]
        pipeline.composer = composer
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1, prices={"BTC": 100.0}, regime="neutral",
        ))

        main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(composer.calls, 1)

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
        event_log = EventLog()
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=event_log,
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
        shadow_events = list(event_log.query(event_types=[ShadowActorUpdated]))
        self.assertGreaterEqual(len(shadow_events), 2)
        self.assertTrue(all(ev.filled >= 1 for ev in shadow_events))
        player_event = next(ev for ev in shadow_events if ev.actor_type == "player")
        self.assertEqual(player_event.regime, "bullish")
        self.assertEqual(player_event.agent_outcomes, (("AgentA", 1, 1, 0, 0),))

    def test_shadow_actor_update_reports_realized_pnl_after_virtual_close(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        class BarAgent:
            label = "AgentA"

            def act(self, market):
                if market.bar == 1:
                    return {"BTC": Action.FUT_LONG_FULL}
                return {"BTC": Action.FUT_CLOSE_ALL}

        agent = BarAgent()
        reg = AgentRegistry()
        reg.register(agent)
        perf = PerformanceMemory(trade_fraction=1.0)
        event_log = EventLog()
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=event_log,
        )
        player = EnsemblePlayer(
            label="PlayerA",
            agents=[agent],
            weights={"AgentA": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(open_single=0.10, open_multi=0.10, open_floor=0.10),
        )

        tournament.run_bar(
            make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish"),
            players=[player],
            balance_usd=1000.0,
        )
        tournament.run_bar(
            make_market_snapshot(bar=2, prices={"BTC": 110.0}, regime="bullish"),
            players=[player],
            balance_usd=1000.0,
        )

        player_events = [
            ev for ev in event_log.query(event_types=[ShadowActorUpdated])
            if ev.actor_type == "player"
        ]
        self.assertEqual(player_events[-1].regime, "bullish")
        self.assertGreater(player_events[-1].realized_pnl_usd, 0.0)
        self.assertEqual(player_events[-1].closed_trades, 1)
        self.assertEqual(player_events[-1].winning_trades, 1)

    def test_shadow_actor_update_includes_blocked_reason_counts(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        agent = FakeAgent("GeneticsCloser", {"BTC": Action.FUT_CLOSE_ALL})
        reg = AgentRegistry()
        reg.register(agent)
        event_log = EventLog()
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=PerformanceMemory(trade_fraction=1.0),
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=event_log,
        )

        tournament.run_bar(
            make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish"),
            players=[],
            balance_usd=1000.0,
        )

        events = list(event_log.query(event_types=[ShadowActorUpdated]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].blocked, 1)
        self.assertEqual(
            events[0].blocked_reasons,
            (("risk_limits: no position to close", 1),),
        )
        self.assertEqual(
            events[0].agent_blocked_reasons,
            (("GeneticsCloser", "risk_limits: no position to close", 1),),
        )

    def test_shadow_tournament_does_not_generate_opens_after_position_capacity_is_full(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        reg = AgentRegistry()
        reg.register(FakeAgent(
            "GeneticsBurst",
            {
                "BTC": Action.FUT_LONG_FULL,
                "ETH": Action.FUT_LONG_FULL,
                "SOL": Action.FUT_LONG_FULL,
                "XRP": Action.FUT_LONG_FULL,
            },
        ))
        event_log = EventLog()
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=PerformanceMemory(trade_fraction=1.0),
            risk_config=RiskLimitsConfig(max_open_positions=2, capital_fraction=0.10),
            event_log=event_log,
        )

        summary = tournament.run_bar(
            make_market_snapshot(
                bar=1,
                prices={"BTC": 100.0, "ETH": 50.0, "SOL": 20.0, "XRP": 1.0},
                regime="bullish",
            ),
            players=[],
            balance_usd=1000.0,
        )

        self.assertEqual(summary.agent_signals, 2)
        self.assertEqual(summary.agent_filled, 2)
        self.assertEqual(summary.agent_blocked, 0)
        events = list(event_log.query(event_types=[ShadowActorUpdated]))
        self.assertEqual(events[0].signals, 2)
        self.assertEqual(events[0].blocked_reasons, ())

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

    def test_actionable_fallback_does_not_override_selected_leader_for_real_execution(self):
        selected_agent = FakeAgent("SelectedAgent")
        fallback_agent = FakeAgent("FallbackAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(selected_agent)
        reg.register(fallback_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="SelectedPlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
                PlayerProfile(
                    label="FallbackPlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        players = {
            "SelectedPlayer": EnsemblePlayer(
                label="SelectedPlayer",
                agents=[selected_agent],
                weights={"SelectedAgent": 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
            ),
            "FallbackPlayer": EnsemblePlayer(
                label="FallbackPlayer",
                agents=[fallback_agent],
                weights={"FallbackAgent": 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
            ),
        }

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return players[profile.label]

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=players["SelectedPlayer"],
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                )

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].leader, "SelectedPlayer")
        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(len(exchange.orders_log), 0)
        self.assertFalse(steps[0].fallback_used)
        self.assertTrue(steps[0].fallback_skipped)
        self.assertIn("disabled", steps[0].fallback_reason)

    def test_no_trade_leader_closes_existing_panteon_positions(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("IdleAgent"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="IdleProfile",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        opened_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=10,
            sym="BTC",
            side="long",
            entry_price=100.0,
            qty=1.0,
            fee_open=0.0,
            by_player="PreviousLeader",
            by_agent="PreviousAgent",
            opened_at=opened_at,
        ))
        exchange._positions["BTC"] = ExchangePosition(
            sym="BTC",
            side="long",
            qty=1.0,
            entry=100.0,
        )

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=NoTradePlayer(),
                    previous=None,
                    score=0.0,
                    margin=0.0,
                    is_urgent=True,
                    reason="cash flat",
                    switched=True,
                )

        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 110.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].leader, "NoTrade")
        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(pipeline.executor._tracker.open_count, 0)
        self.assertEqual(exchange.get_all_positions(), {})
        self.assertEqual(exchange.orders_log[-1].trade.sym, "BTC")
        emitted = list(pipeline.event_log.query(event_types=[SignalEmitted]))
        self.assertEqual(emitted[-1].signal.by_player, "NoTrade")
        self.assertEqual(emitted[-1].signal.action, Action.FUT_CLOSE_ALL)

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

    def test_positive_solo_shadow_agent_does_not_override_selected_player_as_fallback(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        reg = AgentRegistry()
        labels = ["AgentA", "AgentB", "AgentC", "AgentD", "AgentE"]
        reg.register(FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL}))
        for label in labels[1:]:
            reg.register(FakeAgent(label))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="ZZZGroup",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=5,
                    min_agents=5,
                ),
            ],
        )
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        for label in labels:
            open_sig = Signal(
                id=100 + labels.index(label) * 2,
                bar=1,
                sym=f"{label}USD",
                action=Action.FUT_LONG_FULL,
                price=100.0,
                regime=Regime.NEUTRAL,
                by_player=label,
                by_agent=label,
            )
            close_sig = Signal(
                id=open_sig.id + 1,
                bar=2,
                sym=open_sig.sym,
                action=Action.FUT_CLOSE_ALL,
                price=101.0,
                regime=Regime.NEUTRAL,
                by_player=label,
                by_agent=label,
            )
            pipeline.perf.update_from_trade(
                Trade(signal_id=open_sig.id, bar=1, sym=open_sig.sym,
                      side="long", qty=1.0, fill_price=100.0, fee=0.0),
                open_sig,
            )
            pipeline.perf.update_from_trade(
                Trade(signal_id=close_sig.id, bar=2, sym=close_sig.sym,
                      side="long", qty=1.0, fill_price=101.0, fee=0.0),
                close_sig,
            )
        pipeline.selector.capture_session_baseline()
        pipeline.strategist.capture_session_baseline()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=10,
            prices={"BTC": 100.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].leader, "ZZZGroup")
        self.assertEqual(steps[0].selected_leader, "ZZZGroup")
        self.assertEqual(steps[0].executed_leader, "ZZZGroup")
        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(len(exchange.orders_log), 0)
        self.assertTrue(steps[0].fallback_skipped)

    def test_external_exchange_positions_do_not_trip_desync_kill_switch(self):
        from panteon_v2.execution.exchange import ExchangePosition

        class ExternalPositionExchange(FakeExchange):
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(
                        sym="BTC",
                        side="long",
                        qty=0.01,
                        entry=100.0,
                    ),
                }

        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=ExternalPositionExchange(name="MEXC"),
            initial_capital=100.0,
            live_execution_config=LiveExecutionConfig(max_exchange_desync_events=1),
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"ETH": 50.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertNotIn("kill switch active", steps[0].error or "")
        self.assertEqual(pipeline.kill_switch.disabled_reason, "")

    def test_real_events_include_decision_context(self):
        class OneShotAgent:
            label = "OneShot"

            def act(self, market):
                return {"BTC": Action.FUT_LONG_FULL}

        reg = AgentRegistry()
        reg.register(OneShotAgent())
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="BITGET")
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
        pipeline.mode = "live"
        pipeline.timeframe = "1m"
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed)

        self.assertEqual(steps[0].n_filled, 1)
        signal_event = list(pipeline.event_log.query(event_types=[SignalEmitted]))[0]
        sent_event = list(pipeline.event_log.query(event_types=[OrderSent]))[0]
        filled_event = list(pipeline.event_log.query(event_types=[OrderFilled]))[0]
        opened_event = list(pipeline.event_log.query(event_types=[PositionOpened]))[0]
        for event in (signal_event, sent_event, filled_event, opened_event):
            self.assertTrue(event.decision_id)
            self.assertEqual(event.exchange, "BITGET")
            self.assertEqual(event.symbol, "BTC")
            self.assertEqual(event.timeframe, "1m")
            self.assertEqual(event.mode, "live")
            self.assertEqual(event.run_id, pipeline.run_id)
            self.assertEqual(event.session_id, pipeline.session_id)

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
                "memory_dashboard.png",
            ):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            with open(os.path.join(out, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            self.assertEqual(status["version"], "v2")
            self.assertEqual(status["run_state"], "starting")
            self.assertEqual(status["bar_count"], 0)
            self.assertTrue(status["session_id"].endswith("_v2"))
            self.assertEqual(status["run_id"], status["session_id"])
            self.assertEqual(pipeline.session_id, status["session_id"])
            with open(os.path.join(out, "dashboard_latest.png"), "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
            writer.close()

    def test_writer_publishes_latest_exchange_dashboards_to_results_root(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="MEXC"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.close()

            for name in (
                "dashboard_latest_MEXC.png",
                "regime_dashboard_MEXC.png",
                "memory_dashboard_MEXC.png",
            ):
                path = os.path.join(td, name)
                self.assertTrue(os.path.exists(path), name)
                with open(path, "rb") as f:
                    self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

    def test_writer_refreshes_dashboard_on_first_live_bar(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="BITGET"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(
                pipeline,
                results_root=td,
                full_snapshot_every=30,
            )
            with patch.object(writer, "_write_leaderboards") as leaderboards, \
                    patch.object(writer, "_write_dashboard") as dashboard:
                writer.write(StepResult(
                    bar=5734,
                    regime=Regime.NEUTRAL,
                    leader="DefaultEnsemble",
                    leader_changed=True,
                    n_signals=0,
                    n_filled=0,
                    n_rejected=0,
                    n_blocked=0,
                ))

            leaderboards.assert_called_once()
            dashboard.assert_called_once()
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

    def test_status_shadow_counts_include_session_totals_and_last_bar(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
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
                n_blocked=0,
                n_shadow_signals=3,
                n_shadow_filled=2,
            ))
            writer.write(StepResult(
                bar=2,
                regime=Regime.NEUTRAL,
                leader="DefaultEnsemble",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
                n_shadow_signals=0,
                n_shadow_filled=0,
            ))
            with open(os.path.join(writer.output_dir, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            writer.close()

        self.assertEqual(status["shadow"]["signals"], 3)
        self.assertEqual(status["shadow"]["filled"], 2)
        self.assertEqual(status["shadow"]["signals_bar"], 0)
        self.assertEqual(status["shadow"]["filled_bar"], 0)
        self.assertEqual(status["decision_debug"]["leader_raw_signals_bar"], 0)

    def test_status_and_trading_log_include_selected_executed_fallback_debug(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.write(StepResult(
                bar=1,
                regime=Regime.NEUTRAL,
                leader="ExecutedLeader",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
                selected_leader="SelectedLeader",
                executed_leader="ExecutedLeader",
                fallback_used=True,
                fallback_candidate="ExecutedLeader",
                fallback_reason="actionable fallback disabled in test",
            ))
            with open(os.path.join(writer.output_dir, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            with open(os.path.join(writer.output_dir, "trading.log"), "r", encoding="utf-8") as f:
                trading_log = f.read()
            writer.close()

        self.assertEqual(status["current_leader"], "SelectedLeader")
        self.assertEqual(status["selected_leader"], "SelectedLeader")
        self.assertEqual(status["executed_leader"], "ExecutedLeader")
        self.assertTrue(status["decision_debug"]["fallback_used"])
        self.assertEqual(status["decision_debug"]["fallback_candidate"], "ExecutedLeader")
        self.assertIn("selected_leader=SelectedLeader", trading_log)
        self.assertIn("executed_leader=ExecutedLeader", trading_log)
        self.assertIn("fallback_used=True", trading_log)

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

    def test_leaderboards_export_memory_activity_counts(self):
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
            with open(os.path.join(writer.output_dir, "leaderboard_agents.json"),
                      "r", encoding="utf-8") as f:
                agents = json.load(f)["agents"]
            writer.close()

        player = players["V_DefaultEnsemble"]
        agent = agents["V_AgentA"]
        self.assertEqual(player["signals"], 5)
        self.assertEqual(player["entries"], 5)
        self.assertEqual(player["wins"], 3)
        self.assertEqual(player["losses"], 2)
        self.assertEqual(player["per_regime"]["bullish"]["signals"], 5)
        self.assertEqual(agent["signals"], 4)
        self.assertEqual(agent["per_regime"]["bullish"]["entries"], 4)

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
        self.assertEqual(live["real_total_trades"], 2)
        self.assertEqual(live["real_closed_trades"], 1)
        self.assertEqual(live["real_successful_trades"], 0)
        self.assertEqual(live["real_unsuccessful_trades"], 1)
        self.assertEqual(live["real_unresolved_trades"], 1)
        self.assertEqual(live["external_closed_trades"], 1)
        self.assertEqual(live["external_unresolved_trades"], 1)
        self.assertEqual(status["real_trades"]["total"], 2)
        self.assertEqual(status["real_trades"]["successful"], 0)
        self.assertEqual(status["real_trades"]["unsuccessful"], 1)
        self.assertEqual(status["real_trades"]["unresolved"], 1)

    def test_status_exports_panteon_realized_equity_curve_and_drawdown(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            pipeline.event_log.emit(PositionClosed(
                bar=1,
                trace_id="win",
                open_signal_id=1,
                close_signal_id=2,
                sym="BTC",
                side="long",
                entry=100.0,
                exit=120.0,
                qty=1.0,
                realized_pnl=20.0,
                by_player="DefaultEnsemble",
                by_agent="AgentA",
            ))
            writer.write(StepResult(
                bar=1,
                regime=Regime.BULLISH,
                leader="DefaultEnsemble",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
            ))
            pipeline.event_log.emit(PositionClosed(
                bar=2,
                trace_id="loss",
                open_signal_id=3,
                close_signal_id=4,
                sym="ETH",
                side="long",
                entry=100.0,
                exit=85.0,
                qty=1.0,
                realized_pnl=-15.0,
                by_player="DefaultEnsemble",
                by_agent="AgentA",
            ))
            writer.write(StepResult(
                bar=2,
                regime=Regime.BEARISH,
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

        self.assertEqual(status["panteon_realized_equity_usd"], 105.0)
        self.assertEqual(status["panteon_equity_usd"], 105.0)
        self.assertEqual(status["panteon_realized_equity_curve"][-3:], [100.0, 120.0, 105.0])
        self.assertEqual(status["panteon_equity_curve"][-3:], [100.0, 120.0, 105.0])
        self.assertAlmostEqual(status["panteon_realized_max_drawdown_pct"], 12.5)
        self.assertAlmostEqual(status["panteon_max_drawdown_pct"], 12.5)
        self.assertAlmostEqual(
            status["live_session"]["panteon_realized_max_drawdown_pct"],
            12.5,
        )
        self.assertAlmostEqual(status["live_session"]["panteon_max_drawdown_pct"], 12.5)


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
