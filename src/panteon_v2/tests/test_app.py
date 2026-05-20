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
from panteon_v2.app.main_loop import _compose_candidates
from panteon_v2.app.main_loop import _compose_fixed_agent_candidates
from panteon_v2.app.main_loop import _compose_regime_switch_agent_candidates
from panteon_v2.app.main_loop import _compose_rotating_agent_candidates
from panteon_v2.app.main_loop import _compose_solo_agent_candidates
from panteon_v2.app.main_loop import _drop_quarantined_candidates
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
from panteon_v2.domain.types import Action, Metrics, Regime, Signal, Trade
from panteon_v2.execution import ExchangePosition, FakeExchange
from panteon_v2.execution import RiskLimitsConfig
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import (
    AgentRegistry,
    EnsemblePlayer,
    NoTradePlayer,
    PlayerProfile,
    RotatingAgentPlayer,
    ScoredAgent,
    SwitchDecision,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.selection.strategist import CandidateRejection, CandidateScore
from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter, make_market_snapshot
from panteon_v2.shadow.feed import ReplayFeed
from panteon_v2.tests._helpers import FakeAgent, make_market


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
            make_market_snapshot(
                bar=1,
                prices={"BTC": 100.0},
                regime="bullish",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
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
        self.assertEqual(player_event.timestamp.isoformat(), "2025-01-01T00:00:00+00:00")
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

    def test_shadow_tournament_syncs_genetics_fill_state_before_next_bar(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        class SyncAwareLegacyGenetics:
            def __init__(self):
                self.has_futures_position = False

            def act(self, prices, volumes, month=None, portfolio_value=None):
                return {"BTC": 5 if self.has_futures_position else 3}

            def update_from_exchange(self, symbol, spot_qty, spot_entry, fut_qty, fut_entry):
                if symbol == "BTC":
                    self.has_futures_position = bool(fut_qty)

        agent = GeneticsV2AgentAdapter(
            "GeneticsProbe",
            SyncAwareLegacyGenetics(),
        )
        reg = AgentRegistry()
        reg.register(agent)
        event_log = EventLog()
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=PerformanceMemory(trade_fraction=1.0),
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=event_log,
        )

        first = tournament.run_bar(
            make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish"),
            players=[],
            balance_usd=1000.0,
        )
        second = tournament.run_bar(
            make_market_snapshot(bar=2, prices={"BTC": 110.0}, regime="bullish"),
            players=[],
            balance_usd=1000.0,
        )

        self.assertEqual(first.agent_filled, 1)
        self.assertEqual(second.agent_filled, 1)
        agent_events = [
            ev for ev in event_log.query(event_types=[ShadowActorUpdated])
            if ev.actor_type == "agent" and ev.actor_label == "GeneticsProbe"
        ]
        self.assertEqual(agent_events[-1].closed_trades, 1)
        self.assertGreater(agent_events[-1].realized_pnl_usd, 0.0)

    def test_shadow_tournament_exposes_last_actor_updates_without_event_log(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        agent = FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=PerformanceMemory(trade_fraction=1.0),
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=None,
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

        updates = tournament.last_actor_updates()
        self.assertTrue(any(ev.actor_type == "player" and ev.actor_label == "PlayerA" for ev in updates))

    def test_shadow_tournament_exposes_player_open_positions_for_position_gate(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        agent = FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        tournament = ProductionShadowTournament(
            registry=reg,
            perf=PerformanceMemory(trade_fraction=1.0),
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
            event_log=None,
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

        positions = tournament.last_player_open_positions()
        self.assertEqual(
            positions,
            {"PlayerA": ({
                "sym": "BTC",
                "side": "long",
                "opened_bar": 1,
                "age_bars": 0,
                "entry_price": 100.0,
                "current_price": 100.0,
                "qty": 1.0,
                "unrealized_pnl_usd": 0.0,
                "stop_price": None,
                "take_profit_price": None,
                "fresh": True,
            },)},
        )

    def test_pending_shadow_updates_sync_into_strategist_once(self):
        from panteon_v2.app.main_loop import _apply_pending_shadow_updates_to_strategist

        class FakeStrategist:
            def __init__(self):
                self.calls = []
                self.position_calls = []

            def update_shadow_actor_updates(self, updates):
                self.calls.append(tuple(updates))

            def update_shadow_position_snapshot(self, *, player_positions, real_positions):
                self.position_calls.append((dict(player_positions), tuple(real_positions)))

        pipeline, _ = self._setup()
        fake = FakeStrategist()
        pipeline.strategist = fake
        pending = (
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="PlayerA",
                regime="bullish",
                realized_pnl_usd=8.0,
                closed_trades=2,
            ),
        )
        pipeline._pending_shadow_actor_updates = pending
        pipeline._pending_shadow_player_positions = {"PlayerA": ({"sym": "BTC", "side": "long"},)}

        _apply_pending_shadow_updates_to_strategist(pipeline)
        _apply_pending_shadow_updates_to_strategist(pipeline)

        self.assertEqual(fake.calls, [pending])
        self.assertEqual(fake.position_calls, [({"PlayerA": ({"sym": "BTC", "side": "long"},)}, ())])
        self.assertEqual(pipeline._pending_shadow_actor_updates, ())
        self.assertIsNone(pipeline._pending_shadow_player_positions)

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

    def test_actionable_fallback_executes_when_explicitly_enabled(self):
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
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
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

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].leader, "FallbackPlayer")
        self.assertEqual(steps[0].executed_leader, "FallbackPlayer")
        self.assertEqual(steps[0].fallback_candidate, "FallbackPlayer")
        self.assertTrue(steps[0].fallback_used)
        self.assertFalse(steps[0].fallback_skipped)
        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(len(exchange.orders_log), 1)

    def test_genetics_probation_execution_scales_open_risk_multiplier(self):
        genetics_agent = FakeAgent("GeneticsGenomeEnsemble", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(genetics_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            risk_config=RiskLimitsConfig(capital_fraction=0.10, min_notional_usd=0.0),
            profiles=[
                PlayerProfile(
                    label="GeneticsResearch",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        object.__setattr__(
            pipeline.live_execution,
            "genetics_probation_execution_enabled",
            True,
        )
        object.__setattr__(pipeline.live_execution, "genetics_probation_risk_mult", 0.25)
        object.__setattr__(
            pipeline.live_execution,
            "genetics_probation_require_shadow_confirmation",
            False,
        )
        player = EnsemblePlayer(
            label="GeneticsResearch",
            agents=[genetics_agent],
            weights={"GeneticsGenomeEnsemble": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return player

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, **kwargs):
                return SwitchDecision(
                    new_leader=player,
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
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "GeneticsResearch")
        self.assertEqual(steps[0].executed_leader, "GeneticsResearch")
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(len(exchange.orders_log), 1)
        filled_trade = exchange.orders_log[0].trade
        self.assertIsNotNone(filled_trade)
        self.assertAlmostEqual(filled_trade.qty, 0.25)
        emitted = list(pipeline.event_log.query(event_types=[SignalEmitted]))
        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0].signal.risk_mult, 0.25)

    def test_genetics_probation_execution_requires_current_shadow_confirmation(self):
        genetics_agent = FakeAgent("GeneticsGenomeEnsemble", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(genetics_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            risk_config=RiskLimitsConfig(capital_fraction=0.10, min_notional_usd=0.0),
            profiles=[
                PlayerProfile(
                    label="GeneticsResearch",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        object.__setattr__(
            pipeline.live_execution,
            "genetics_probation_execution_enabled",
            True,
        )
        object.__setattr__(pipeline.live_execution, "genetics_probation_risk_mult", 0.25)
        object.__setattr__(
            pipeline.live_execution,
            "genetics_probation_require_shadow_confirmation",
            True,
        )
        player = EnsemblePlayer(
            label="GeneticsResearch",
            agents=[genetics_agent],
            weights={"GeneticsGenomeEnsemble": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return player

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, **kwargs):
                return SwitchDecision(
                    new_leader=player,
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
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "GeneticsResearch")
        self.assertEqual(steps[0].executed_leader, "GeneticsResearch")
        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(steps[0].n_filtered_real_signals, 1)
        self.assertEqual(len(exchange.orders_log), 0)
        self.assertIn(
            "genetics_shadow_unconfirmed:BTC:GeneticsGenomeEnsemble",
            steps[0].signal_filter_details,
        )

    def test_actionable_fallback_reuses_current_shadow_registry_without_real_revote(self):
        class ShadowOnlyAgent(FakeAgent):
            def __init__(self):
                super().__init__("FallbackAgent")

            def clone_for_shadow(self):
                return FakeAgent("FallbackAgent", {"BTC": Action.FUT_LONG_FULL})

            def act(self, market):
                raise AssertionError("fallback must not re-run real vote")

        selected_agent = FakeAgent("SelectedAgent")
        fallback_agent = ShadowOnlyAgent()
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
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
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

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "FallbackPlayer")
        self.assertTrue(steps[0].fallback_used)
        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(len(exchange.orders_log), 1)

    def test_selected_leader_replays_fresh_shadow_position_when_real_vote_is_empty(self):
        selected_agent = FakeAgent("SelectedAgent")
        reg = AgentRegistry()
        reg.register(selected_agent)
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
            ],
        )
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1

            _config = Config()

            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def update_current_actionable_labels(self, labels):
                self.actionable_labels = set(labels)

            def update_shadow_actor_updates(self, updates):
                self.shadow_updates = tuple(updates)

            def update_shadow_position_snapshot(self, *, player_positions, real_positions):
                self.player_positions = dict(player_positions)
                self.real_positions = tuple(real_positions)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                )

        class PositionOnlyTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {}

            def last_agent_signals(self):
                return {}

            def last_player_open_positions(self):
                return {
                    "SelectedPlayer": ({
                        "sym": "BTC",
                        "side": "long",
                        "opened_bar": 1,
                        "age_bars": 0,
                        "entry_price": 100.0,
                        "current_price": 101.0,
                        "qty": 1.0,
                        "unrealized_pnl_usd": 1.0,
                        "stop_price": None,
                        "take_profit_price": None,
                        "fresh": True,
                    },),
                }

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = PositionOnlyTournament()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 101.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertFalse(steps[0].fallback_used)
        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(exchange.orders_log[0].trade.sym, "BTC")
        self.assertEqual(exchange.orders_log[0].trade.side, "long")
        causal = steps[0].causal_decision
        self.assertEqual(causal["bar"], 1)
        self.assertEqual(causal["selected_leader"], "SelectedPlayer")
        self.assertEqual(causal["executed_leader"], "SelectedPlayer")
        self.assertTrue(causal["shadow_position_replay_used"])
        self.assertEqual(
            causal["selected_shadow_positions"][0]["unrealized_pnl_usd"],
            1.0,
        )
        self.assertEqual(causal["raw_signals"][0]["by_agent"], "ShadowPositionReplay")

    def test_selected_leader_does_not_replay_flat_shadow_position(self):
        selected_agent = FakeAgent("SelectedAgent")
        reg = AgentRegistry()
        reg.register(selected_agent)
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
            ],
        )
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1

            _config = Config()

            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def update_current_actionable_labels(self, labels):
                self.actionable_labels = set(labels)

            def update_shadow_actor_updates(self, updates):
                self.shadow_updates = tuple(updates)

            def update_shadow_position_snapshot(self, *, player_positions, real_positions):
                self.player_positions = dict(player_positions)
                self.real_positions = tuple(real_positions)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                )

        class FlatPositionTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {}

            def last_agent_signals(self):
                return {}

            def last_player_open_positions(self):
                return {
                    "SelectedPlayer": ({
                        "sym": "BTC",
                        "side": "long",
                        "opened_bar": 1,
                        "age_bars": 0,
                        "entry_price": 100.0,
                        "current_price": 100.0,
                        "qty": 1.0,
                        "unrealized_pnl_usd": 0.0,
                        "stop_price": None,
                        "take_profit_price": None,
                        "fresh": True,
                    },),
                }

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = FlatPositionTournament()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(exchange.orders_log, [])

    def test_selected_leader_does_not_replay_losing_shadow_position(self):
        selected_agent = FakeAgent("SelectedAgent")
        reg = AgentRegistry()
        reg.register(selected_agent)
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
            ],
        )
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1

            _config = Config()

            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def update_current_actionable_labels(self, labels):
                self.actionable_labels = set(labels)

            def update_shadow_actor_updates(self, updates):
                self.shadow_updates = tuple(updates)

            def update_shadow_position_snapshot(self, *, player_positions, real_positions):
                self.player_positions = dict(player_positions)
                self.real_positions = tuple(real_positions)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                )

        class LosingPositionTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {}

            def last_agent_signals(self):
                return {}

            def last_player_open_positions(self):
                return {
                    "SelectedPlayer": ({
                        "sym": "BTC",
                        "side": "long",
                        "opened_bar": 1,
                        "age_bars": 0,
                        "entry_price": 100.0,
                        "current_price": 99.0,
                        "qty": 1.0,
                        "unrealized_pnl_usd": -1.0,
                        "stop_price": None,
                        "take_profit_price": None,
                        "fresh": True,
                    },),
                }

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = LosingPositionTournament()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 99.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(exchange.orders_log, [])

    def test_selected_leader_replays_positive_shadow_position_with_configured_age(self):
        selected_agent = FakeAgent("SelectedAgent")
        reg = AgentRegistry()
        reg.register(selected_agent)
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
            ],
        )
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 6

            _config = Config()

            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def update_current_actionable_labels(self, labels):
                self.actionable_labels = set(labels)

            def update_shadow_actor_updates(self, updates):
                self.shadow_updates = tuple(updates)

            def update_shadow_position_snapshot(self, *, player_positions, real_positions):
                self.player_positions = dict(player_positions)
                self.real_positions = tuple(real_positions)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                )

        class PositionOnlyTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {}

            def last_agent_signals(self):
                return {}

            def last_player_open_positions(self):
                return {
                    "SelectedPlayer": ({
                        "sym": "BTC",
                        "side": "short",
                        "opened_bar": 5,
                        "age_bars": 4,
                        "entry_price": 100.0,
                        "current_price": 96.0,
                        "qty": 1.0,
                        "unrealized_pnl_usd": 4.0,
                        "stop_price": None,
                        "take_profit_price": None,
                        "fresh": False,
                    },),
                }

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = PositionOnlyTournament()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=9,
            prices={"BTC": 96.0},
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertTrue(steps[0].causal_decision["shadow_position_replay_used"])
        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertEqual(exchange.orders_log[0].trade.sym, "BTC")
        self.assertEqual(exchange.orders_log[0].trade.side, "short")

    def test_current_actionable_labels_include_fresh_shadow_positions(self):
        from panteon_v2.app.main_loop import _current_actionable_player_labels

        labels = _current_actionable_player_labels(
            updates=(),
            signals_by_player={},
            signals_by_agent={},
            positions_by_player={
                "ShadowLeader": ({
                    "sym": "BTC",
                    "side": "long",
                    "opened_bar": 10,
                    "age_bars": 1,
                    "fresh": True,
                },),
                "StaleLeader": ({
                    "sym": "ETH",
                    "side": "short",
                    "opened_bar": 1,
                    "age_bars": 20,
                    "fresh": False,
                },),
                "LosingFreshLeader": ({
                    "sym": "SOL",
                    "side": "long",
                    "opened_bar": 10,
                    "age_bars": 0,
                    "unrealized_pnl_usd": -0.25,
                    "fresh": True,
                },),
                "FlatFreshLeader": ({
                    "sym": "XRP",
                    "side": "short",
                    "opened_bar": 10,
                    "age_bars": 0,
                    "unrealized_pnl_usd": 0.0,
                    "fresh": True,
                },),
            },
        )

        self.assertIn("ShadowLeader", labels)
        self.assertNotIn("StaleLeader", labels)
        self.assertNotIn("LosingFreshLeader", labels)
        self.assertNotIn("FlatFreshLeader", labels)

    def test_current_actionable_labels_respect_configured_shadow_position_age(self):
        from panteon_v2.app.main_loop import _current_actionable_player_labels

        positions_by_player = {
            "AgedWinner": ({
                "sym": "BTC",
                "side": "short",
                "opened_bar": 10,
                "age_bars": 4,
                "unrealized_pnl_usd": 1.25,
                "fresh": False,
            },),
        }

        default_labels = _current_actionable_player_labels(
            updates=(),
            signals_by_player={},
            signals_by_agent={},
            positions_by_player=positions_by_player,
        )
        extended_labels = _current_actionable_player_labels(
            updates=(),
            signals_by_player={},
            signals_by_agent={},
            positions_by_player=positions_by_player,
            max_shadow_position_age_bars=6,
        )

        self.assertNotIn("AgedWinner", default_labels)
        self.assertIn("AgedWinner", extended_labels)

    def test_shadow_state_fallback_skips_unconfirmed_antonius_raw_open(self):
        funding_agent = FakeAgent("FundingArb", {"BTC": Action.FUT_SHORT_FULL})
        reg = AgentRegistry()
        reg.register(funding_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            risk_config=RiskLimitsConfig(capital_fraction=0.10, min_notional_usd=0.0),
            profiles=[
                PlayerProfile(
                    label="Antonius_conservative",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        pipeline.actionable_fallback_enabled = True
        pipeline.actionable_fallback_min_score = None
        antonius = EnsemblePlayer(
            label="Antonius_conservative",
            agents=[funding_agent],
            weights={"FundingArb": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return antonius

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=NoTradePlayer(),
                    previous=None,
                    score=0.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected no trade",
                    switched=True,
                )

        class PositionOnlyTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {
                    "Antonius_conservative": (Signal(
                        id=900,
                        bar=1,
                        sym="BTC",
                        action=Action.FUT_SHORT_FULL,
                        price=100.0,
                        regime=Regime.BEARISH,
                        by_player="Antonius_conservative",
                        by_agent="FundingArb",
                    ),),
                }

            def last_agent_signals(self):
                return {}

            def last_player_open_positions(self):
                return {
                    "Antonius_conservative": ({
                        "sym": "BTC",
                        "side": "short",
                        "opened_bar": 1,
                        "age_bars": 0,
                        "entry_price": 100.0,
                        "current_price": 100.0,
                        "qty": 1.0,
                        "unrealized_pnl_usd": 0.0,
                        "stop_price": None,
                        "take_profit_price": None,
                        "fresh": True,
                    },),
                }

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = PositionOnlyTournament()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "NoTrade")
        self.assertEqual(steps[0].executed_leader, "NoTrade")
        self.assertFalse(steps[0].fallback_used)
        self.assertTrue(steps[0].fallback_skipped)
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(len(exchange.orders_log), 0)

    def test_shadow_state_selected_antonius_skips_unconfirmed_raw_open(self):
        funding_agent = FakeAgent("FundingArb", {"BTC": Action.FUT_SHORT_FULL})
        reg = AgentRegistry()
        reg.register(funding_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            risk_config=RiskLimitsConfig(capital_fraction=0.10, min_notional_usd=0.0),
            profiles=[
                PlayerProfile(
                    label="Antonius_conservative",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        antonius = EnsemblePlayer(
            label="Antonius_conservative",
            agents=[funding_agent],
            weights={"FundingArb": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return antonius

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=antonius,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected antonius",
                    switched=True,
                )

        class PositionOnlyTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {}

            def last_agent_signals(self):
                return {}

            def last_player_open_positions(self):
                return {
                    "Antonius_conservative": ({
                        "sym": "BTC",
                        "side": "short",
                        "opened_bar": 1,
                        "age_bars": 0,
                        "entry_price": 100.0,
                        "current_price": 100.0,
                        "qty": 1.0,
                        "unrealized_pnl_usd": 0.0,
                        "stop_price": None,
                        "take_profit_price": None,
                        "fresh": True,
                    },),
                }

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = PositionOnlyTournament()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "Antonius_conservative")
        self.assertEqual(steps[0].executed_leader, "Antonius_conservative")
        self.assertFalse(steps[0].fallback_used)
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(len(exchange.orders_log), 0)

    def test_actionable_fallback_can_execute_current_agent_signal_as_solo(self):
        selected_agent = FakeAgent("SelectedAgent")
        current_agent = FakeAgent("CurrentAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(selected_agent)
        reg.register(current_agent)
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
            ],
        )
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
        pipeline.actionable_fallback_min_score = None
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
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

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "Solo_CurrentAgent")
        self.assertEqual(steps[0].fallback_candidate, "Solo_CurrentAgent")
        self.assertTrue(steps[0].fallback_used)
        self.assertEqual(exchange.orders_log[0].trade.sym, "BTC")

    def test_actionable_fallback_agent_solo_respects_hard_policy(self):
        selected_agent = FakeAgent("SelectedAgent")
        denied_agent = FakeAgent("PlayerFunding", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(selected_agent)
        reg.register(denied_agent)
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
            ],
        )
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
        pipeline.actionable_fallback_min_score = None
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                    candidate_rejections=(
                        CandidateRejection(
                            label="Solo_PlayerFunding",
                            reason="hard policy denylist: Solo_PlayerFunding",
                        ),
                    ),
                )

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertFalse(steps[0].fallback_used)
        self.assertEqual(len(exchange.orders_log), 0)

    def test_actionable_fallback_does_not_bypass_panteon_equity_guard(self):
        fallback_agent = FakeAgent("FallbackAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(fallback_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="FallbackPlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        pipeline.actionable_fallback_enabled = True
        fallback_player = EnsemblePlayer(
            label="FallbackPlayer",
            agents=[fallback_agent],
            weights={"FallbackAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return fallback_player

        class GuardStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=NoTradePlayer(),
                    previous=fallback_player,
                    score=0.0,
                    margin=0.0,
                    is_urgent=True,
                    reason="v3 panteon equity guard: test",
                    switched=True,
                )

        pipeline.composer = FixedComposer()
        pipeline.strategist = GuardStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "NoTrade")
        self.assertEqual(steps[0].executed_leader, "NoTrade")
        self.assertFalse(steps[0].fallback_used)
        self.assertTrue(steps[0].fallback_skipped)
        self.assertIn("portfolio equity guard", steps[0].fallback_reason)
        self.assertEqual(len(exchange.orders_log), 0)

    def test_actionable_fallback_uses_current_registry_signal_without_has_data(self):
        selected_agent = FakeAgent("SelectedAgent")
        low_agent = FakeAgent("LowAgent", {"BTC": Action.FUT_LONG_FULL})
        high_agent = FakeAgent("HighAgent", {"ETH": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(selected_agent)
        reg.register(low_agent)
        reg.register(high_agent)
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
                    label="LowFallback",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
                PlayerProfile(
                    label="HighFallback",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
        pipeline.actionable_fallback_min_score = 0.0
        pipeline.actionable_fallback_require_has_data = True
        players = {
            "SelectedPlayer": EnsemblePlayer(
                label="SelectedPlayer",
                agents=[selected_agent],
                weights={"SelectedAgent": 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
            ),
            "LowFallback": EnsemblePlayer(
                label="LowFallback",
                agents=[low_agent],
                weights={"LowAgent": 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
            ),
            "HighFallback": EnsemblePlayer(
                label="HighFallback",
                agents=[high_agent],
                weights={"HighAgent": 1.0},
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
                    candidate_scores=(
                        CandidateScore(
                            label="LowFallback",
                            score=-1.0,
                            rank=1,
                            has_data=True,
                            closed_trades=20,
                            signals=1,
                            execution_failures=0,
                            score_source="test",
                            uncertainty_penalty=0.0,
                            agent_labels=("LowAgent",),
                            memory_keys_read=(),
                        ),
                        CandidateScore(
                            label="HighFallback",
                            score=1.0,
                            rank=2,
                            has_data=False,
                            closed_trades=20,
                            signals=1,
                            execution_failures=0,
                            score_source="test",
                            uncertainty_penalty=0.0,
                            agent_labels=("HighAgent",),
                            memory_keys_read=(),
                        ),
                    ),
                )

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertTrue(steps[0].fallback_used)
        self.assertFalse(steps[0].fallback_skipped)
        self.assertEqual(steps[0].fallback_candidate, "HighFallback")
        self.assertEqual(len(exchange.orders_log), 1)

        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL2"),
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
                    label="LowFallback",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
                PlayerProfile(
                    label="HighFallback",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
        pipeline.actionable_fallback_min_score = 0.0
        pipeline.actionable_fallback_require_has_data = False
        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertTrue(steps[0].fallback_used)
        self.assertEqual(steps[0].fallback_candidate, "HighFallback")
        self.assertEqual(steps[0].n_filled, 1)

    def test_actionable_fallback_skips_candidates_rejected_by_hard_policy(self):
        denied_agent = FakeAgent("PlayerFunding", {"BTC": Action.FUT_LONG_FULL})
        safe_agent = FakeAgent("SafeAgent", {"ETH": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(denied_agent)
        reg.register(safe_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="Solo_PlayerFunding",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
                PlayerProfile(
                    label="SafePlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
        players = {
            "Solo_PlayerFunding": EnsemblePlayer(
                label="Solo_PlayerFunding",
                agents=[denied_agent],
                weights={"PlayerFunding": 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
            ),
            "SafePlayer": EnsemblePlayer(
                label="SafePlayer",
                agents=[safe_agent],
                weights={"SafeAgent": 1.0},
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
                    new_leader=NoTradePlayer(),
                    previous=None,
                    score=0.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="all candidates rejected",
                    switched=True,
                    candidate_rejections=(
                        CandidateRejection(
                            label="Solo_PlayerFunding",
                            reason="hard policy denylist: Solo_PlayerFunding",
                        ),
                    ),
                )

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "NoTrade")
        self.assertEqual(steps[0].executed_leader, "SafePlayer")
        self.assertEqual(steps[0].fallback_candidate, "SafePlayer")
        self.assertTrue(steps[0].fallback_used)
        self.assertEqual(exchange.orders_log[0].trade.sym, "ETH")

    def test_actionable_fallback_skips_candidates_rejected_by_probation_score(self):
        risky_agent = FakeAgent("RiskyAgent", {"BTC": Action.FUT_LONG_FULL})
        safe_agent = FakeAgent("SafeAgent", {"ETH": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(risky_agent)
        reg.register(safe_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="RiskyPlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
                PlayerProfile(
                    label="SafePlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament
        pipeline.shadow_tournament = ProductionShadowTournament(
            registry=reg,
            perf=pipeline.perf,
            risk_config=RiskLimitsConfig(capital_fraction=0.10),
        )
        pipeline.actionable_fallback_enabled = True
        players = {
            "RiskyPlayer": EnsemblePlayer(
                label="RiskyPlayer",
                agents=[risky_agent],
                weights={"RiskyAgent": 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
            ),
            "SafePlayer": EnsemblePlayer(
                label="SafePlayer",
                agents=[safe_agent],
                weights={"SafeAgent": 1.0},
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
                    new_leader=NoTradePlayer(),
                    previous=None,
                    score=0.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="all candidates rejected",
                    switched=True,
                    candidate_rejections=(
                        CandidateRejection(
                            label="RiskyPlayer",
                            reason=(
                                "real promotion gate: probation score 0.0000 < "
                                "1.0000 with 1 real closed trades"
                            ),
                        ),
                    ),
                )

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="bearish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "NoTrade")
        self.assertEqual(steps[0].executed_leader, "SafePlayer")
        self.assertEqual(steps[0].fallback_candidate, "SafePlayer")
        self.assertTrue(steps[0].fallback_used)
        self.assertEqual(exchange.orders_log[0].trade.sym, "ETH")

    def test_actionable_fallback_restores_failed_probe_state(self):
        class MutatingAgent(FakeAgent):
            def __init__(self):
                super().__init__("FallbackAgent", {"BTC": Action.FUT_LONG_FULL})
                self.calls = 0

            def act(self, market):
                self.calls += 1
                return super().act(market)

        selected_agent = FakeAgent("SelectedAgent")
        fallback_agent = MutatingAgent()
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
        pipeline.actionable_fallback_enabled = True
        pipeline.live_execution = LiveExecutionConfig(max_new_opens_per_bar=0)
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

        self.assertFalse(steps[0].fallback_used)
        self.assertTrue(steps[0].fallback_skipped)
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(fallback_agent.calls, 0)

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

    def test_solo_agent_candidate_limit_is_runtime_configurable(self):
        rows = [
            ScoredAgent(
                agent=FakeAgent(f"Agent{i}"),
                score=1.0,
                metrics=Metrics(pnl_pct=1.0, closed_trades=1),
            )
            for i in range(5)
        ]

        class RecordingSelector:
            def __init__(self):
                self.calls = []

            def select(self, regime, *, k):
                self.calls.append(("select", k))
                return rows[:k]

        class Pipeline:
            solo_agent_candidate_limit = 5
            selector = RecordingSelector()

        solo = _compose_solo_agent_candidates(Pipeline(), Regime.NEUTRAL, [])

        self.assertEqual(Pipeline.selector.calls, [("select", 5)])
        self.assertEqual(len(solo), 5)
        self.assertEqual(solo[-1].label, "Solo_Agent4")

    def test_solo_agent_candidate_limit_applies_to_fallback_selector(self):
        row = ScoredAgent(
            agent=FakeAgent("FallbackAgent"),
            score=1.0,
            metrics=Metrics(pnl_pct=1.0, closed_trades=1),
        )

        class RecordingSelector:
            def __init__(self):
                self.calls = []

            def select(self, regime, *, k):
                self.calls.append(("select", k))
                return []

            def select_with_fallback(self, regime, *, k, fallback_threshold, min_count):
                self.calls.append(("fallback", k, fallback_threshold, min_count))
                return [row]

        class Pipeline:
            solo_agent_candidate_limit = 6
            selector = RecordingSelector()

        solo = _compose_solo_agent_candidates(Pipeline(), Regime.NEUTRAL, [])

        self.assertEqual(
            Pipeline.selector.calls,
            [("select", 6), ("fallback", 6, -10.0, 1)],
        )
        self.assertEqual([player.label for player in solo], ["Solo_FallbackAgent"])

    def test_fixed_agent_candidate_uses_exact_registered_agent_set(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("AgentB", {"ETH": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("AgentC", {"SOL": Action.FUT_LONG_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Pipeline:
            registry = reg
            qm = QM()
            fixed_agent_player_sets = (
                ("Fixed_AB", ("AgentA", "AgentB")),
            )

        fixed = _compose_fixed_agent_candidates(Pipeline(), [])

        self.assertEqual([player.label for player in fixed], ["Fixed_AB"])
        self.assertEqual(fixed[0].agent_labels, ["AgentA", "AgentB"])
        self.assertEqual(fixed[0].weights, {"AgentA": 0.5, "AgentB": 0.5})

    def test_rotating_agent_candidate_uses_first_actionable_agent_for_regime(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("QuietAgent", {"BTC": Action.HOLD}))
        reg.register(FakeAgent("BullAgent", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("FallbackAgent", {"BTC": Action.FUT_SHORT_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Pipeline:
            registry = reg
            qm = QM()
            rotating_agent_player_sets = (
                (
                    "Optimal_StaticRotator",
                    {"bullish": ("QuietAgent", "BullAgent")},
                    ("FallbackAgent",),
                ),
            )

        players = _compose_rotating_agent_candidates(Pipeline(), Regime.BULLISH, [])

        self.assertEqual([player.label for player in players], ["Optimal_StaticRotator"])
        self.assertIsInstance(players[0], RotatingAgentPlayer)
        self.assertEqual(
            players[0].agent_labels,
            ["BullAgent", "FallbackAgent", "QuietAgent"],
        )
        signals = players[0].vote(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            signal_id_start=10,
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].by_agent, "BullAgent")
        self.assertEqual(signals[0].action, Action.FUT_LONG_FULL)

    def test_regime_switch_candidate_orders_agent_pool_by_selector_score(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("OldDefaultAgent", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("BestAgent", {"ETH": Action.FUT_SHORT_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Selector:
            def score_registered(
                self,
                label,
                regime,
                *,
                include_quarantined=False,
                ensure_baseline=True,
            ):
                scores = {
                    "OldDefaultAgent": 0.10,
                    "BestAgent": 0.80,
                }
                agent = reg.get(label)
                if agent is None:
                    return None
                return ScoredAgent(
                    agent=agent,
                    score=scores[label],
                    metrics=Metrics(
                        pnl_pct=scores[label] * 10.0,
                        closed_trades=25,
                        entries=25,
                        signals=25,
                        wins=6,
                        losses=4,
                    ),
                )

        class Pipeline:
            registry = reg
            qm = QM()
            selector = Selector()
            regime_switch_player_sets = (
                (
                    "Antonius_dynamic",
                    {"bearish": ("OldDefaultAgent", "BestAgent")},
                ),
            )

        players = _compose_regime_switch_agent_candidates(
            Pipeline(),
            Regime.BEARISH,
            [],
        )

        self.assertEqual([player.label for player in players], ["Antonius_dynamic"])
        self.assertIsInstance(players[0], RotatingAgentPlayer)
        signals = players[0].vote(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            signal_id_start=10,
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].by_agent, "BestAgent")
        self.assertEqual(signals[0].sym, "ETH")

    def test_regime_switch_candidate_keeps_default_order_without_walk_forward_evidence(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("OldDefaultAgent", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("ThinBestAgent", {"ETH": Action.FUT_SHORT_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Selector:
            def score_registered(
                self,
                label,
                regime,
                *,
                include_quarantined=False,
                ensure_baseline=True,
            ):
                scores = {
                    "OldDefaultAgent": (0.10, 25),
                    "ThinBestAgent": (0.80, 1),
                }
                agent = reg.get(label)
                if agent is None:
                    return None
                score, trades = scores[label]
                return ScoredAgent(
                    agent=agent,
                    score=score,
                    metrics=Metrics(
                        pnl_pct=score * 10.0,
                        closed_trades=trades,
                        entries=trades,
                        signals=trades,
                        wins=min(trades, 1),
                        losses=max(0, trades - 1),
                    ),
                )

        class Pipeline:
            registry = reg
            qm = QM()
            selector = Selector()
            regime_switch_player_sets = (
                (
                    "Antonius_dynamic",
                    {"bearish": ("OldDefaultAgent", "ThinBestAgent")},
                ),
            )

        players = _compose_regime_switch_agent_candidates(
            Pipeline(),
            Regime.BEARISH,
            [],
        )

        signals = players[0].vote(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            signal_id_start=10,
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].by_agent, "OldDefaultAgent")
        self.assertEqual(signals[0].sym, "BTC")

    def test_regime_switch_candidate_prefers_causal_pnl_over_lower_profit_score(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("RiskAdjustedAgent", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("ProfitAgent", {"ETH": Action.FUT_SHORT_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Selector:
            def score_registered(
                self,
                label,
                regime,
                *,
                include_quarantined=False,
                ensure_baseline=True,
            ):
                rows = {
                    "RiskAdjustedAgent": (0.90, 1.0),
                    "ProfitAgent": (0.20, 12.0),
                }
                agent = reg.get(label)
                if agent is None:
                    return None
                score, pnl_pct = rows[label]
                return ScoredAgent(
                    agent=agent,
                    score=score,
                    metrics=Metrics(
                        pnl_pct=pnl_pct,
                        closed_trades=25,
                        entries=25,
                        signals=25,
                        wins=15,
                        losses=10,
                    ),
                )

        class Pipeline:
            registry = reg
            qm = QM()
            selector = Selector()
            regime_switch_player_sets = (
                (
                    "Antonius_dynamic",
                    {"bearish": ("RiskAdjustedAgent", "ProfitAgent")},
                ),
            )

        players = _compose_regime_switch_agent_candidates(
            Pipeline(),
            Regime.BEARISH,
            [],
        )

        signals = players[0].vote(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            signal_id_start=10,
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].by_agent, "ProfitAgent")
        self.assertEqual(signals[0].sym, "ETH")

    def test_rotating_agent_candidate_does_not_use_off_regime_agents(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("QuietBullAgent", {"BTC": Action.HOLD}))
        reg.register(FakeAgent("BearAgent", {"BTC": Action.FUT_SHORT_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Pipeline:
            registry = reg
            qm = QM()
            rotating_agent_player_sets = (
                (
                    "Optimal_StaticRotator",
                    {
                        "bullish": ("QuietBullAgent",),
                        "bearish": ("BearAgent",),
                    },
                    (),
                ),
            )

        players = _compose_rotating_agent_candidates(Pipeline(), Regime.BULLISH, [])

        self.assertEqual([player.label for player in players], ["Optimal_StaticRotator"])
        self.assertEqual(
            players[0].vote(
                make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
                signal_id_start=10,
            ),
            [],
        )

    def test_protected_composite_label_is_not_hopeless_quarantined_from_one_regime(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        for idx in range(5):
            open_signal = Signal(
                id=idx * 2 + 1,
                bar=idx * 2 + 1,
                sym="BTC",
                action=Action.FUT_LONG_FULL,
                price=100.0,
                regime=Regime.NEUTRAL,
                by_agent="LiveCrashHunter",
                by_player="Antonius_strategy",
            )
            close_signal = Signal(
                id=idx * 2 + 2,
                bar=idx * 2 + 2,
                sym="BTC",
                action=Action.FUT_CLOSE_ALL,
                price=90.0,
                regime=Regime.NEUTRAL,
                by_agent="LiveCrashHunter",
                by_player="Antonius_strategy",
            )
            perf.update_from_trade(
                Trade(
                    signal_id=open_signal.id,
                    bar=open_signal.bar,
                    sym=open_signal.sym,
                    side="long",
                    qty=1.0,
                    fill_price=100.0,
                    fee=0.0,
                ),
                open_signal,
            )
            perf.update_from_trade(
                Trade(
                    signal_id=close_signal.id,
                    bar=close_signal.bar,
                    sym=close_signal.sym,
                    side="long",
                    qty=1.0,
                    fill_price=90.0,
                    fee=0.0,
                ),
                close_signal,
            )

        unprotected = QuarantineManager(seed=set())
        unprotected.recompute(perf)
        self.assertTrue(unprotected.is_quarantined("Antonius_strategy"))

        protected = QuarantineManager(
            seed=set(),
            protected_labels={"Antonius_strategy"},
        )
        protected.recompute(perf)
        self.assertFalse(protected.is_quarantined("Antonius_strategy"))

    def test_drop_quarantine_keeps_genetics_player_with_quarantined_raw_genetics_agent(self):
        class Pipeline:
            qm = QuarantineManager(seed={"GeneticsGenomeEnsemble", "BadAgent"})

        genetics_player = EnsemblePlayer(
            label="GeneticsResearch",
            agents=[FakeAgent("GeneticsGenomeEnsemble")],
            weights={"GeneticsGenomeEnsemble": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        bad_player = EnsemblePlayer(
            label="BadResearch",
            agents=[FakeAgent("BadAgent")],
            weights={"BadAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        filtered = _drop_quarantined_candidates(
            Pipeline(),
            [genetics_player, bad_player],
        )

        self.assertEqual([player.label for player in filtered], ["GeneticsResearch"])

    def test_compose_candidates_includes_fixed_agent_players(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("AgentB", {"ETH": Action.FUT_LONG_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Selector:
            def select(self, regime, *, k):
                return []

        class Pipeline:
            profiles = []
            registry = reg
            qm = QM()
            selector = Selector()
            fixed_agent_player_sets = (
                ("Fixed_AB", ("AgentA", "AgentB")),
            )

        candidates = _compose_candidates(Pipeline(), Regime.NEUTRAL)

        self.assertEqual([player.label for player in candidates], ["Fixed_AB"])
        self.assertEqual(candidates[0].agent_labels, ["AgentA", "AgentB"])

    def test_compose_candidates_includes_rotating_agent_players(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA", {"BTC": Action.HOLD}))
        reg.register(FakeAgent("AgentB", {"BTC": Action.FUT_LONG_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Selector:
            def select(self, regime, *, k):
                return []

        class Pipeline:
            profiles = []
            registry = reg
            qm = QM()
            selector = Selector()
            fixed_agent_player_sets = ()
            regime_switch_player_sets = ()
            rotating_agent_player_sets = (
                ("Optimal_StaticRotator", {"bullish": ("AgentA", "AgentB")}, ()),
            )

        candidates = _compose_candidates(Pipeline(), Regime.BULLISH)

        self.assertEqual([player.label for player in candidates], ["Optimal_StaticRotator"])
        self.assertEqual(candidates[0].agent_labels, ["AgentA", "AgentB"])

    def test_compose_candidates_includes_antonius_strategy_by_regime(self):
        reg = AgentRegistry()
        for label in (
            "LiveAfterShock",
            "MomentumScalper",
            "ResearchValidatorAgent",
            "LiveCrashHunter",
        ):
            reg.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))

        class QM:
            def is_quarantined(self, label):
                return False

        class Selector:
            def select(self, regime, *, k):
                return []

        class Pipeline:
            profiles = []
            registry = reg
            qm = QM()
            selector = Selector()
            fixed_agent_player_sets = ()
            regime_switch_player_sets = (
                (
                    "Antonius_strategy",
                    {
                        "bullish": "LiveAfterShock",
                        "bearish": "MomentumScalper",
                        "neutral": "ResearchValidatorAgent",
                        "crash": "LiveCrashHunter",
                    },
                ),
            )

        expectations = {
            Regime.BULLISH: ["LiveAfterShock"],
            Regime.BEARISH: ["MomentumScalper"],
            Regime.NEUTRAL: ["ResearchValidatorAgent"],
            Regime.CRASH: ["LiveCrashHunter"],
        }
        for regime, agent_labels in expectations.items():
            with self.subTest(regime=regime):
                candidates = _compose_candidates(Pipeline(), regime)
                self.assertEqual([player.label for player in candidates], ["Antonius_strategy"])
                self.assertEqual(candidates[0].agent_labels, agent_labels)

    def test_production_pipeline_exposes_antonius_strategy_default(self):
        reg = AgentRegistry()
        for label in (
            "LiveAfterShock",
            "FundingArb",
            "LiveCrashHunter",
            "RichardDennis",
            "LiveOIBreakout",
            "ResearchValidatorAgent",
            "GeneticsCore",
            "LiveMeanRev",
            "MomentumScalper",
            "VolBreakoutHunter",
            "CrashPanicShortAgent",
            "LiveVolCompress",
            "BullRotationAgent",
            "NeutralLiquiditySweep",
            "NeutralRangeScalper",
            "LiveRegimePullback",
        ):
            reg.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))

        pipeline = build_dryrun_pipeline(registry=reg, profiles=[])

        expected_static = {
            "Perfect_OIBreakout": {
                "bullish": "LiveOIBreakout",
                "neutral": "LiveOIBreakout",
                "bearish": "LiveOIBreakout",
                "crash": "LiveOIBreakout",
            },
            "Perfect_CrashSwitch": {
                "bearish": "LiveCrashHunter",
                "crash": "LiveCrashHunter",
            },
            "Perfect_NeutralValidator": {
                "neutral": "ResearchValidatorAgent",
            },
            "Perfect_GeneticsBearCrash": {
                "bearish": "GeneticsCore",
                "crash": "GeneticsCore",
            },
            "Perfect_MeanRev": {
                "neutral": "LiveMeanRev",
            },
        }
        actual = dict(pipeline.regime_switch_player_sets)
        self.assertNotIn("Antonius_strategy", actual)
        self.assertEqual(
            actual["Antonius_conservative"],
            {
                "bullish": "VolBreakoutHunter",
                "bearish": "FundingArb",
                "neutral": "ResearchValidatorAgent",
                "crash": "CrashPanicShortAgent",
            },
        )
        for label, mapping in expected_static.items():
            self.assertEqual(actual.get(label), mapping)
        rotating = dict(
            (label, mapping)
            for label, mapping, _fallback in pipeline.rotating_agent_player_sets
        )
        self.assertIn("Optimal_StaticRotator", rotating)
        self.assertIn("ResearchValidatorAgent", rotating["Optimal_StaticRotator"]["neutral"])
        self.assertIn("LiveCrashHunter", rotating["Optimal_StaticRotator"]["crash"])
        self.assertIn("Optimal_StaticRotator", pipeline.qm.protected_labels)
        self.assertIn("Antonius_conservative", pipeline.qm.protected_labels)
        self.assertNotIn("Antonius_strategy", pipeline.qm.protected_labels)

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

    def test_writer_exports_causal_entry_decisions_jsonl(self):
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
                bar=7,
                regime=Regime.BEARISH,
                leader="ExecutedLeader",
                leader_changed=False,
                n_signals=1,
                n_filled=1,
                n_rejected=0,
                n_blocked=0,
                selected_leader="SelectedLeader",
                executed_leader="ExecutedLeader",
                causal_decision={
                    "bar": 7,
                    "selected_leader": "SelectedLeader",
                    "executed_leader": "ExecutedLeader",
                    "raw_signals": [{"sym": "BTC", "action": "FUT_LONG_FULL"}],
                    "selected_shadow_positions": [{"sym": "BTC", "age_bars": 1}],
                },
            ))
            path = os.path.join(writer.output_dir, "causal_entry_decisions.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            writer.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bar"], 7)
        self.assertEqual(rows[0]["selected_leader"], "SelectedLeader")
        self.assertEqual(rows[0]["raw_signals"][0]["sym"], "BTC")

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
