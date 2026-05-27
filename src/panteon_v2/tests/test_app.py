"""Тесты Phase 9 — production wiring."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
import threading
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
from panteon_v2.app.main_loop import _apply_genetics_probation_execution_overlay
from panteon_v2.app.main_loop import _flash_actor_key_from_decision
from panteon_v2.app.main_loop import _flash_causal_decision_payload
from panteon_v2.app.main_loop import _flash_previous_actor_key_from_decision
from panteon_v2.app.main_loop import _compose_candidates
from panteon_v2.app.main_loop import _compose_fixed_agent_candidates
from panteon_v2.app.main_loop import _compose_regime_switch_agent_candidates
from panteon_v2.app.main_loop import _compose_rotating_agent_candidates
from panteon_v2.app.main_loop import _compose_solo_agent_candidates
from panteon_v2.app.main_loop import _drop_quarantined_candidates
from panteon_v2.app.main_loop import _flash_degraded_actor_keys
from panteon_v2.app.main_loop import _flash_degraded_open_symbols
from panteon_v2.app.main_loop import _flash_degraded_signal_keys
from panteon_v2.app.main_loop import _flash_event_log_tail
from panteon_v2.app.main_loop import _fallback_candidate_safety_issue
from panteon_v2.app.main_loop import _flash_register_degradation_signal_keys
from panteon_v2.app.main_loop import _flash_real_agents
from panteon_v2.app.main_loop import _flash_shadow_confirmation_scores
from panteon_v2.app.main_loop import _flash_shadow_player_signals_with_position_replay
from panteon_v2.app.main_loop import _genetics_probation_regime_exit_close_signals
from panteon_v2.app.main_loop import _flash_stale_position_close_signals
from panteon_v2.app.main_loop import _flash_update_degradation_state_from_events
from panteon_v2.attribution import (
    CandidateScored,
    CandidateRejected,
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
from panteon_v2.app.live_state import RealSignalGuardResult
from panteon_v2.execution import ExchangePosition, FakeExchange
from panteon_v2.execution import RiskLimitsConfig
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import (
    AgentRegistry,
    EnsemblePlayer,
    FlashAllocatorConfig,
    FlashCandidateAudit,
    FlashDecision,
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
from panteon_v2.tests.test_selector import _add_perf


# ════════════════════════════════════════════════════════════════════
# Bootstrap
# ════════════════════════════════════════════════════════════════════


class TestBootstrap(unittest.TestCase):
    def test_audit_emission_module_emits_switch_candidate_events(self):
        from panteon_v2.app.audit_emission import emit_candidate_audit_events

        event_log = EventLog()
        pipeline = type("Pipeline", (), {"event_log": event_log})()
        winner = EnsemblePlayer(
            label="Winner",
            agents=[FakeAgent("WinnerAgent")],
            weights={"WinnerAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        decision = SwitchDecision(
            new_leader=winner,
            previous=None,
            score=1.0,
            margin=0.0,
            is_urgent=False,
            reason="test",
            switched=True,
            candidate_scores=(
                CandidateScore(
                    label="Winner",
                    score=1.25,
                    rank=1,
                    has_data=True,
                    closed_trades=12,
                    signals=4,
                    execution_failures=0,
                    score_source="test",
                    uncertainty_penalty=0.0,
                    agent_labels=("WinnerAgent",),
                    memory_keys_read=("Winner|bullish",),
                ),
            ),
            candidate_rejections=(
                CandidateRejection(label="Rejected", reason="score_below_threshold"),
            ),
        )

        emit_candidate_audit_events(
            pipeline,
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            decision,
            decision_id="decision-1",
            trace_id="trace-1",
            context={"exchange": "TEST", "symbol": "BTC"},
        )

        scored = list(event_log.query(event_types=[CandidateScored]))
        rejected = list(event_log.query(event_types=[CandidateRejected]))
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0].player_label, "Winner")
        self.assertTrue(scored[0].selected_by_pantheon)
        self.assertEqual(scored[0].memory_keys_read, ("Winner|bullish",))
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].player_label, "Rejected")
        self.assertEqual(rejected[0].reason, "score_below_threshold")

    def test_audit_emission_module_emits_flash_candidate_events(self):
        from panteon_v2.app.audit_emission import emit_flash_audit_events

        event_log = EventLog()
        pipeline = type("Pipeline", (), {"event_log": event_log})()
        decision = FlashDecision(
            symbol="BTC",
            selected_actor="FlashAgent",
            actor_type="agent",
            score=1.0,
            action=Action.FUT_LONG_FULL,
            reason="selected",
            signal=None,
            candidates=(
                FlashCandidateAudit(
                    symbol="BTC",
                    label="FlashAgent",
                    actor_type="agent",
                    actor_key="agent:FlashAgent",
                    score=1.0,
                    action=Action.FUT_LONG_FULL,
                    rejected=False,
                    reason="eligible",
                    has_data=True,
                    closed_trades=10,
                    agent_labels=("CoreAgent",),
                ),
                FlashCandidateAudit(
                    symbol="BTC",
                    label="WeakAgent",
                    actor_type="agent",
                    actor_key="agent:WeakAgent",
                    score=-0.5,
                    action=Action.FUT_SHORT_FULL,
                    rejected=True,
                    reason="score_below_threshold",
                    has_data=True,
                    closed_trades=7,
                    agent_labels=("WeakCore",),
                ),
            ),
        )

        emit_flash_audit_events(
            pipeline,
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            (decision,),
            decision_id="decision-2",
            trace_id="trace-2",
            context={"exchange": "TEST", "symbol": "ALL"},
        )

        scored = list(event_log.query(event_types=[CandidateScored]))
        rejected = list(event_log.query(event_types=[CandidateRejected]))
        self.assertEqual([event.player_label for event in scored], ["FlashAgent", "WeakAgent"])
        self.assertEqual({event.symbol for event in scored}, {"BTC"})
        self.assertTrue(scored[0].selected_by_pantheon)
        self.assertIn("CoreAgent|bullish", scored[0].memory_keys_read)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].player_label, "WeakAgent")
        self.assertEqual(rejected[0].reason, "flash:score_below_threshold")

    def test_flash_audit_callback_can_be_disabled_for_long_retro_runs(self):
        from panteon_v2.app.main_loop import _emit_flash_audit_events

        event_log = EventLog()
        pipeline = type(
            "Pipeline",
            (),
            {"event_log": event_log, "flash_audit_events_enabled": False},
        )()
        decision = FlashDecision(
            symbol="BTC",
            selected_actor="FlashAgent",
            actor_type="agent",
            score=1.0,
            action=Action.FUT_LONG_FULL,
            reason="selected",
            signal=None,
            candidates=(
                FlashCandidateAudit(
                    symbol="BTC",
                    label="FlashAgent",
                    actor_type="agent",
                    actor_key="agent:FlashAgent",
                    score=1.0,
                    action=Action.FUT_LONG_FULL,
                    rejected=False,
                    reason="eligible",
                    has_data=True,
                    closed_trades=10,
                ),
            ),
        )

        _emit_flash_audit_events(
            pipeline,
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            (decision,),
            decision_id="decision-2",
            trace_id="trace-2",
            context={"exchange": "TEST", "symbol": "ALL"},
        )

        self.assertEqual(list(event_log.query(event_types=[CandidateScored])), [])

    def test_flash_decision_path_module_exposes_explicit_callback_contract(self):
        from dataclasses import fields

        from panteon_v2.app.decision_paths.flash import (
            FlashDecisionPathCallbacks,
            run_flash_decision_path,
        )

        callback_fields = {field.name for field in fields(FlashDecisionPathCallbacks)}
        self.assertTrue(callable(run_flash_decision_path))
        self.assertIn("flash_real_agents", callback_fields)
        self.assertIn("emit_flash_audit_events", callback_fields)
        self.assertIn("step_result_type", callback_fields)

    def test_flash_actor_key_can_preserve_original_actor_after_cap(self):
        decision = FlashDecision(
            symbol="BTC",
            selected_actor="NoTrade",
            actor_type="no_trade",
            score=0.0,
            action=Action.HOLD,
            reason="actor_signal_cap",
            signal=None,
            candidates=(
                FlashCandidateAudit(
                    symbol="BTC",
                    label="MomentumScalper",
                    actor_type="agent",
                    actor_key="agent:MomentumScalper",
                    score=1.2,
                    base_score=1.2,
                    effective_score=1.2,
                    gate_score=1.2,
                    action=Action.FUT_LONG_FULL,
                    rejected=False,
                    reason="eligible",
                    has_data=True,
                    closed_trades=10,
                ),
            ),
            original_selected_actor="MomentumScalper",
            original_actor_type="agent",
        )

        self.assertEqual(
            _flash_actor_key_from_decision(decision, prefer_original=True),
            "agent:MomentumScalper",
        )

    def test_flash_previous_actor_key_uses_final_actor_when_signal_traded(self):
        signal = Signal(
            id=1,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="AnchorAgent",
            by_agent="AnchorAgent",
        )
        decision = FlashDecision(
            symbol="BTC",
            selected_actor="AnchorAgent",
            actor_type="agent",
            score=1.0,
            action=Action.FUT_LONG_FULL,
            reason="anchor_dominance_hold",
            signal=signal,
            candidates=(
                FlashCandidateAudit(
                    symbol="BTC",
                    label="GreedyAgent",
                    actor_type="agent",
                    actor_key="agent:GreedyAgent",
                    score=1.1,
                    action=Action.FUT_SHORT_FULL,
                    rejected=False,
                ),
                FlashCandidateAudit(
                    symbol="BTC",
                    label="AnchorAgent",
                    actor_type="agent",
                    actor_key="agent:AnchorAgent",
                    score=1.0,
                    action=Action.FUT_LONG_FULL,
                    rejected=False,
                ),
            ),
            original_selected_actor="GreedyAgent",
            original_actor_type="agent",
        )

        self.assertEqual(
            _flash_previous_actor_key_from_decision(decision),
            "agent:AnchorAgent",
        )

    def test_flash_previous_actor_key_uses_original_actor_when_signal_suppressed(self):
        decision = FlashDecision(
            symbol="BTC",
            selected_actor="NoTrade",
            actor_type="no_trade",
            score=0.0,
            action=Action.HOLD,
            reason="actor_signal_cap",
            signal=None,
            candidates=(
                FlashCandidateAudit(
                    symbol="BTC",
                    label="OriginalAgent",
                    actor_type="agent",
                    actor_key="agent:OriginalAgent",
                    score=1.0,
                    action=Action.FUT_LONG_FULL,
                    rejected=False,
                ),
            ),
            original_selected_actor="OriginalAgent",
            original_actor_type="agent",
        )

        self.assertEqual(
            _flash_previous_actor_key_from_decision(decision),
            "agent:OriginalAgent",
        )

    def test_flash_real_agents_does_not_drop_agent_on_solo_wrapper_safety(self):
        registry = AgentRegistry()
        registry.register(FakeAgent("HealthyInEnsemble"))
        pipeline = type("Pipeline", (), {"registry": registry})()
        market = make_market(prices={"BTC": 100.0})

        with patch(
            "panteon_v2.app.main_loop._fallback_candidate_safety_issue",
            return_value=True,
        ):
            agents = _flash_real_agents(pipeline, market)

        self.assertEqual([agent.label for agent in agents], ["HealthyInEnsemble"])

    def test_flash_real_agents_allows_explicit_genetics_probation_agent_by_regime(self):
        allowed = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_LONG_FULL})
        allowed.shadow_only = True
        allowed.live_trading_eligible = False
        blocked = FakeAgent("GeneticsRegimeRouter", {"BTC": Action.FUT_LONG_FULL})
        blocked.shadow_only = True
        blocked.live_trading_eligible = False
        registry = AgentRegistry()
        registry.register(allowed)
        registry.register(blocked)
        pipeline = type("Pipeline", (), {
            "registry": registry,
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsNeutral",),
                genetics_probation_allowed_regimes=("neutral",),
            ),
        })()

        neutral_agents = _flash_real_agents(
            pipeline,
            make_market(regime=Regime.NEUTRAL, prices={"BTC": 100.0}),
        )
        crash_agents = _flash_real_agents(
            pipeline,
            make_market(regime=Regime.CRASH, prices={"BTC": 100.0}),
        )

        self.assertEqual([agent.label for agent in neutral_agents], ["GeneticsNeutral"])
        self.assertEqual(crash_agents, [])
        self.assertFalse(getattr(allowed, "live_trading_eligible"))

    def test_flash_real_agents_probation_genetics_bypass_hard_policy_only(self):
        allowed = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_LONG_FULL})
        allowed.shadow_only = True
        allowed.live_trading_eligible = False
        registry = AgentRegistry()
        registry.register(allowed)

        class Strategist:
            def _validate(self, candidate):
                return None

            def _validate_hard_policy(self, candidate, regime, **kwargs):
                return object()

            def _validate_v3_real_loss(self, candidate, regime, **kwargs):
                return None

        pipeline = type("Pipeline", (), {
            "registry": registry,
            "strategist": Strategist(),
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsNeutral",),
                genetics_probation_allowed_regimes=("neutral",),
            ),
        })()

        agents = _flash_real_agents(
            pipeline,
            make_market(regime=Regime.NEUTRAL, prices={"BTC": 100.0}),
        )

        self.assertEqual([agent.label for agent in agents], ["GeneticsNeutral"])

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

    def test_equity_peak_drawdown_kill_switch_tracks_live_equity_peak(self):
        from panteon_v2.app.main_loop import _kill_switch_reason, sync_pipeline_balance

        class LiveExchange(FakeExchange):
            def __init__(self):
                super().__init__(name="BITGET")
                self._equities = iter((100.0, 120.0, 113.9))

            def get_account_equity(self):
                return next(self._equities)

        reg = AgentRegistry()
        reg.register(FakeAgent("A"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(),
            initial_capital=100.0,
            live_execution_config=LiveExecutionConfig(max_equity_peak_drawdown_pct=5.0),
        )

        self.assertEqual(sync_pipeline_balance(pipeline), 100.0)
        self.assertEqual(pipeline.kill_switch.peak_equity_usd, 100.0)
        self.assertEqual(sync_pipeline_balance(pipeline), 120.0)
        self.assertEqual(pipeline.kill_switch.peak_equity_usd, 120.0)
        self.assertEqual(sync_pipeline_balance(pipeline), 113.9)

        reason = _kill_switch_reason(pipeline)
        self.assertIn("equity peak drawdown exceeded", reason)
        self.assertIn("peak=120.00", reason)
        self.assertEqual(pipeline.kill_switch.disabled_reason, reason)

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

    def test_live_execution_settings_parse_existing_position_adoption(self):
        from panteon_v2.app.startup import _live_execution_config_from_settings

        cfg = _live_execution_config_from_settings({
            "v2_adopt_existing_positions_enabled": "true",
            "v2_adopt_existing_position_symbols": "btc, eth/usdt",
            "v2_adopt_existing_position_player": "RestartAdopted",
            "v2_adopt_existing_position_agent": "RestartBridge",
        })

        self.assertTrue(cfg.adopt_existing_positions_enabled)
        self.assertEqual(cfg.adopt_existing_position_symbols, ("btc", "eth/usdt"))
        self.assertEqual(cfg.adopt_existing_position_player, "RestartAdopted")
        self.assertEqual(cfg.adopt_existing_position_agent, "RestartBridge")

    def test_quarantine_recovery_settings_parse_exchange_scoped_override(self):
        from panteon_v2.app.startup import (
            _quarantine_override_labels_from_settings,
            _quarantine_recovery_config_from_settings,
        )

        settings = {
            "mexc_v2_quarantine_override_labels": "LiveVolCompress, OtherAgent",
            "bitget_v2_quarantine_override_labels": "BitgetOnly",
            "v2_quarantine_recovery_enabled": "on",
            "v2_quarantine_recovery_min_pnl_pct": "0.25",
            "v2_quarantine_recovery_min_closed_trades": "4",
        }

        self.assertEqual(
            _quarantine_override_labels_from_settings(settings, "MEXC"),
            ("LiveVolCompress", "OtherAgent"),
        )
        self.assertEqual(
            _quarantine_override_labels_from_settings(settings, "BITGET"),
            ("BitgetOnly",),
        )
        cfg = _quarantine_recovery_config_from_settings(settings, "MEXC")
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["min_pnl_pct"], 0.25)
        self.assertEqual(cfg["min_closed_trades"], 4)

    def test_flash_settings_parse_flag_and_allocator_knobs(self):
        from panteon_v2.app.startup import (
            _flash_allocator_config_from_settings,
            _flash_enabled_from_settings,
            _flash_stale_position_exit_config_from_settings,
        )

        settings = {
            "v2_flash_enabled": "yes",
            "v2_flash_min_score_to_trade": "0.15",
            "v2_flash_actionable_bonus": "0.35",
            "v2_flash_no_data_score": "-0.25",
            "v2_flash_min_closed_trades_to_trade": "8",
            "v2_flash_min_pnl_pct_to_trade": "0.2",
            "v2_flash_shadow_confirmation_enabled": "true",
            "v2_flash_symbol_shadow_confirmation_enabled": "true",
            "v2_flash_shadow_actor_fallback_confirmation_enabled": "true",
            "v2_flash_shadow_base_fallback_confirmation_enabled": "true",
            "v2_flash_shadow_signal_handoff_enabled": "true",
            "v2_flash_shadow_actor_fallback_min_base_score": "3.5",
            "v2_flash_shadow_base_fallback_actor_keys": "agent:StrongActor,ensemble:SafeSolo",
            "v2_flash_actor_switch_margin": "0.45",
            "v2_flash_anchor_actor_keys": "ensemble:Solo_MomentumScalper",
            "v2_flash_portfolio_actor_keys": "ensemble:Solo_MomentumScalper,ensemble:Solo_LiveCrashHunter",
            "v2_flash_portfolio_shadow_bootstrap_min_closed_enabled": "true",
            "v2_flash_anchor_min_score_to_trade": "1.5",
            "v2_flash_anchor_shadow_min_score": "-0.1",
            "v2_flash_anchor_min_score_advantage": "0.8",
            "v2_flash_prefer_solo_player_wrappers_enabled": "true",
            "v2_flash_prefer_proven_solo_player_wrappers_enabled": "true",
            "v2_flash_proven_solo_min_score_advantage": "0.75",
            "v2_flash_shadow_min_score": "0.1",
            "v2_flash_shadow_min_closed_trades": "55",
            "v2_flash_shadow_min_full_open_closed_trades": "2",
            "v2_flash_max_signals_per_actor": "2",
            "v2_flash_overextension_guard_enabled": "true",
            "v2_flash_overextension_lookback_bars": "12",
            "v2_flash_short_overextension_return_floor_pct": "-8.0",
            "v2_flash_long_overextension_return_ceiling_pct": "8.0",
            "v2_flash_overextension_volatility_normalized_enabled": "true",
            "v2_flash_short_overextension_z_floor": "-2.5",
            "v2_flash_long_overextension_z_ceiling": "2.5",
            "v2_flash_overextension_min_volatility_pct": "0.2",
            "v2_flash_shadow_quality_confirmation_enabled": "true",
            "v2_flash_shadow_min_win_rate_pct": "60.0",
            "v2_flash_shadow_max_recent_downside_usd": "5.0",
            "v2_flash_shadow_min_pnl_per_trade_lcb_usd": "0.05",
            "v2_flash_shadow_pnl_per_trade_lcb_z": "1.0",
            "v2_flash_shadow_pnl_per_trade_lcb_penalty_floor_usd": "0.0",
            "v2_flash_shadow_pnl_per_trade_lcb_penalty_weight": "20.0",
            "v2_flash_shadow_pnl_lcb_risk_sizing_enabled": "true",
            "v2_flash_shadow_pnl_lcb_risk_min_mult": "0.4",
            "v2_flash_shadow_pnl_lcb_risk_floor_usd": "0.0",
            "v2_flash_shadow_pnl_lcb_risk_scale_usd": "2.0",
            "v2_flash_shadow_symbol_health_enabled": "true",
            "v2_flash_shadow_symbol_health_min_closed_trades": "9",
            "v2_flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd": "-0.2",
            "v2_flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd": "0.0",
            "v2_flash_shadow_symbol_health_pnl_lcb_penalty_weight": "12.0",
            "v2_flash_actor_risk_sizing_enabled": "true",
            "v2_flash_actor_risk_min_mult": "0.3",
            "v2_flash_actor_risk_max_mult": "1.2",
            "v2_flash_actor_risk_edge_scale_pct": "0.75",
            "v2_flash_funding_score_weight": "0.5",
            "v2_flash_funding_risk_mult_weight": "0.4",
            "v2_flash_funding_risk_mult_cap": "0.2",
            "v2_flash_no_trade_fee_saving_score_enabled": "true",
            "v2_flash_no_trade_default_fee_bps": "4.5",
            "v2_flash_volatility_risk_sizing_enabled": "true",
            "v2_flash_volatility_risk_target_pct": "1.8",
            "v2_flash_volatility_risk_min_volatility_pct": "0.4",
            "v2_flash_volatility_risk_max_mult": "1.5",
            "v2_flash_technical_overlay_enabled": "true",
            "v2_flash_technical_hard_gate_enabled": "true",
            "v2_flash_technical_score_bonus": "0.12",
            "v2_flash_technical_score_penalty": "0.30",
            "v2_flash_technical_rsi_long_min": "46.0",
            "v2_flash_technical_rsi_long_max": "70.0",
            "v2_flash_technical_rsi_short_min": "30.0",
            "v2_flash_technical_rsi_short_max": "54.0",
            "v2_flash_technical_macd_histogram_min_abs_pct": "0.02",
            "v2_flash_technical_atr_risk_sizing_enabled": "true",
            "v2_flash_technical_atr_target_pct": "1.8",
            "v2_flash_technical_atr_min_pct": "0.3",
            "v2_flash_technical_atr_max_mult": "1.4",
            "v2_flash_denied_signal_keys": (
                "ensemble:Solo_A|btc/usdt|fut_long_full,"
                "agent:LiveB|ETH/USDT|SPOT_BUY_FULL"
            ),
            "v2_flash_terminal_denied_signal_keys": (
                "ensemble:Solo_C|atom/usdt|fut_short_full"
            ),
            "v2_flash_denied_open_symbols": "atom/usdt,fil/usdt",
            "v2_flash_denied_open_regimes": "neutral,crash",
            "v2_flash_degradation_guard_enabled": "true",
            "v2_flash_degradation_actor_guard_enabled": "true",
            "v2_flash_degradation_actor_scope": "actor_regime",
            "v2_flash_degradation_signal_cooldown_bars": "144",
            "v2_flash_degradation_actor_cooldown_bars": "72",
            "v2_flash_degradation_symbol_guard_enabled": "true",
            "v2_flash_degradation_symbol_cooldown_bars": "48",
            "v2_flash_degradation_symbol_lookback_bars": "2160",
            "v2_flash_degradation_symbol_window_closed_trades": "2",
            "v2_flash_degradation_symbol_min_closed_trades": "2",
            "v2_flash_degradation_symbol_max_recent_pnl_usd": "-10.0",
            "v2_flash_degradation_window_closed_trades": "3",
            "v2_flash_degradation_min_closed_trades": "3",
            "v2_flash_degradation_max_recent_pnl_usd": "-25.0",
            "v2_flash_degradation_signal_min_pnl_per_trade_lcb_usd": "-2.0",
            "v2_flash_degradation_pnl_per_trade_lcb_z": "1.5",
            "v2_flash_degradation_signal_risk_sizing_enabled": "true",
            "v2_flash_degradation_signal_risk_mult": "0.35",
            "v2_flash_degradation_reserve_actor_cap": "true",
            "v2_flash_degradation_recovery_enabled": "true",
            "v2_flash_degradation_recovery_min_closed_trades": "3",
            "v2_flash_degradation_recovery_min_recent_pnl_usd": "0.0",
            "v2_flash_promotion_manifest_enabled": "true",
            "v2_flash_promoted_actor_cap_overrides": "ensemble:Solo_MomentumScalper=10",
            "v2_flash_stale_position_exit_enabled": "true",
            "v2_flash_stale_position_exit_max_age_bars": "168",
            "v2_flash_stale_position_exit_require_nonpositive_unrealized": "true",
        }

        self.assertTrue(_flash_enabled_from_settings(settings))
        cfg = _flash_allocator_config_from_settings(settings)
        self.assertEqual(cfg.min_score_to_trade, 0.15)
        self.assertEqual(cfg.actionable_bonus, 0.35)
        self.assertEqual(cfg.no_data_score, -0.25)
        self.assertEqual(cfg.min_closed_trades_to_trade, 8)
        self.assertEqual(cfg.min_pnl_pct_to_trade, 0.2)
        self.assertTrue(cfg.shadow_confirmation_enabled)
        self.assertTrue(cfg.shadow_symbol_confirmation_enabled)
        self.assertTrue(cfg.shadow_actor_fallback_confirmation_enabled)
        self.assertTrue(cfg.shadow_base_fallback_confirmation_enabled)
        self.assertTrue(cfg.shadow_signal_handoff_enabled)
        self.assertEqual(cfg.shadow_actor_fallback_min_base_score, 3.5)
        self.assertEqual(
            cfg.shadow_base_fallback_actor_keys,
            ("agent:StrongActor", "ensemble:SafeSolo"),
        )
        self.assertEqual(cfg.actor_switch_margin, 0.45)
        self.assertEqual(cfg.anchor_actor_keys, ("ensemble:Solo_MomentumScalper",))
        self.assertEqual(
            cfg.portfolio_actor_keys,
            ("ensemble:Solo_MomentumScalper", "ensemble:Solo_LiveCrashHunter"),
        )
        self.assertTrue(cfg.portfolio_shadow_bootstrap_min_closed_enabled)
        self.assertEqual(cfg.anchor_min_score_to_trade, 1.5)
        self.assertEqual(cfg.anchor_shadow_min_score, -0.1)
        self.assertEqual(cfg.anchor_min_score_advantage, 0.8)
        self.assertTrue(cfg.prefer_solo_player_wrappers_enabled)
        self.assertTrue(cfg.prefer_proven_solo_player_wrappers_enabled)
        self.assertEqual(cfg.proven_solo_min_score_advantage, 0.75)
        self.assertEqual(cfg.shadow_confirmation_min_score, 0.1)
        self.assertEqual(cfg.shadow_confirmation_min_closed_trades, 55)
        self.assertEqual(cfg.shadow_confirmation_min_full_open_closed_trades, 2)
        self.assertEqual(cfg.max_signals_per_actor, 2)
        self.assertTrue(cfg.open_overextension_guard_enabled)
        self.assertEqual(cfg.overextension_lookback_bars, 12)
        self.assertEqual(cfg.short_overextension_return_floor_pct, -8.0)
        self.assertEqual(cfg.long_overextension_return_ceiling_pct, 8.0)
        self.assertTrue(cfg.overextension_volatility_normalized_enabled)
        self.assertEqual(cfg.short_overextension_z_floor, -2.5)
        self.assertEqual(cfg.long_overextension_z_ceiling, 2.5)
        self.assertEqual(cfg.overextension_min_volatility_pct, 0.2)
        self.assertTrue(cfg.shadow_quality_confirmation_enabled)
        self.assertEqual(cfg.shadow_confirmation_min_win_rate_pct, 60.0)
        self.assertEqual(cfg.shadow_confirmation_max_recent_downside_usd, 5.0)
        self.assertEqual(cfg.shadow_confirmation_min_pnl_per_trade_lcb_usd, 0.05)
        self.assertEqual(cfg.shadow_confirmation_pnl_per_trade_lcb_z, 1.0)
        self.assertEqual(cfg.shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd, 0.0)
        self.assertEqual(cfg.shadow_confirmation_pnl_per_trade_lcb_penalty_weight, 20.0)
        self.assertTrue(cfg.shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled)
        self.assertEqual(cfg.shadow_confirmation_pnl_per_trade_lcb_risk_min_mult, 0.4)
        self.assertEqual(cfg.shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd, 0.0)
        self.assertEqual(cfg.shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd, 2.0)
        self.assertTrue(cfg.shadow_symbol_health_enabled)
        self.assertEqual(cfg.shadow_symbol_health_min_closed_trades, 9)
        self.assertEqual(cfg.shadow_symbol_health_min_pnl_per_trade_lcb_usd, -0.2)
        self.assertEqual(
            cfg.shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd,
            0.0,
        )
        self.assertEqual(
            cfg.shadow_symbol_health_pnl_per_trade_lcb_penalty_weight,
            12.0,
        )
        self.assertTrue(cfg.actor_risk_sizing_enabled)
        self.assertEqual(cfg.actor_risk_min_mult, 0.3)
        self.assertEqual(cfg.actor_risk_max_mult, 1.2)
        self.assertEqual(cfg.actor_risk_edge_scale_pct, 0.75)
        self.assertEqual(cfg.funding_score_weight, 0.5)
        self.assertEqual(cfg.funding_risk_mult_weight, 0.4)
        self.assertEqual(cfg.funding_risk_mult_cap, 0.2)
        self.assertTrue(cfg.no_trade_fee_saving_score_enabled)
        self.assertEqual(cfg.no_trade_default_fee_bps, 4.5)
        self.assertTrue(cfg.volatility_risk_sizing_enabled)
        self.assertEqual(cfg.volatility_risk_target_pct, 1.8)
        self.assertEqual(cfg.volatility_risk_min_volatility_pct, 0.4)
        self.assertEqual(cfg.volatility_risk_max_mult, 1.5)
        self.assertTrue(cfg.technical_overlay_enabled)
        self.assertTrue(cfg.technical_hard_gate_enabled)
        self.assertEqual(cfg.technical_score_bonus, 0.12)
        self.assertEqual(cfg.technical_score_penalty, 0.30)
        self.assertEqual(cfg.technical_rsi_long_min, 46.0)
        self.assertEqual(cfg.technical_rsi_long_max, 70.0)
        self.assertEqual(cfg.technical_rsi_short_min, 30.0)
        self.assertEqual(cfg.technical_rsi_short_max, 54.0)
        self.assertEqual(cfg.technical_macd_histogram_min_abs_pct, 0.02)
        self.assertTrue(cfg.technical_atr_risk_sizing_enabled)
        self.assertEqual(cfg.technical_atr_target_pct, 1.8)
        self.assertEqual(cfg.technical_atr_min_pct, 0.3)
        self.assertEqual(cfg.technical_atr_max_mult, 1.4)
        self.assertEqual(
            cfg.denied_signal_keys,
            (
                "ensemble:Solo_A|BTC/USDT|FUT_LONG_FULL",
                "agent:LiveB|ETH/USDT|SPOT_BUY_FULL",
            ),
        )
        self.assertEqual(
            cfg.terminal_denied_signal_keys,
            ("ensemble:Solo_C|ATOM/USDT|FUT_SHORT_FULL",),
        )
        self.assertEqual(cfg.denied_open_symbols, ("ATOM/USDT", "FIL/USDT"))
        self.assertEqual(cfg.denied_open_regimes, (Regime.NEUTRAL, Regime.CRASH))
        self.assertTrue(cfg.degradation_guard_enabled)
        self.assertTrue(cfg.degradation_actor_guard_enabled)
        self.assertEqual(cfg.degradation_actor_scope, "actor_regime")
        self.assertEqual(cfg.degradation_signal_cooldown_bars, 144)
        self.assertEqual(cfg.degradation_actor_cooldown_bars, 72)
        self.assertTrue(cfg.degradation_symbol_guard_enabled)
        self.assertEqual(cfg.degradation_symbol_cooldown_bars, 48)
        self.assertEqual(cfg.degradation_symbol_lookback_bars, 2160)
        self.assertEqual(cfg.degradation_symbol_window_closed_trades, 2)
        self.assertEqual(cfg.degradation_symbol_min_closed_trades, 2)
        self.assertEqual(cfg.degradation_symbol_max_recent_pnl_usd, -10.0)
        self.assertEqual(cfg.degradation_window_closed_trades, 3)
        self.assertEqual(cfg.degradation_min_closed_trades, 3)
        self.assertEqual(cfg.degradation_max_recent_pnl_usd, -25.0)
        self.assertEqual(cfg.degradation_signal_min_pnl_per_trade_lcb_usd, -2.0)
        self.assertEqual(cfg.degradation_pnl_per_trade_lcb_z, 1.5)
        self.assertTrue(cfg.degradation_signal_risk_sizing_enabled)
        self.assertEqual(cfg.degradation_signal_risk_mult, 0.35)
        self.assertTrue(cfg.degradation_reserve_actor_cap)
        self.assertTrue(cfg.degradation_recovery_enabled)
        self.assertEqual(cfg.degradation_recovery_min_closed_trades, 3)
        self.assertEqual(cfg.degradation_recovery_min_recent_pnl_usd, 0.0)
        self.assertTrue(cfg.promotion_manifest_enabled)
        self.assertEqual(
            cfg.promoted_actor_cap_overrides,
            ("ensemble:Solo_MomentumScalper=10",),
        )
        stale_exit = _flash_stale_position_exit_config_from_settings(settings)
        self.assertTrue(stale_exit["enabled"])
        self.assertEqual(stale_exit["max_age_bars"], 168)
        self.assertTrue(stale_exit["require_nonpositive_unrealized"])


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

    def test_can_stream_steps_without_retaining_results(self):
        pipeline, feed = self._setup()
        seen = []
        steps = main_loop(
            pipeline,
            feed,
            on_step=lambda step: seen.append(step.bar),
            collect_results=False,
        )
        self.assertEqual(steps, [])
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
        self.assertEqual(player_events[-1].symbol_outcomes[0][0], "BTC")
        self.assertGreater(player_events[-1].symbol_outcomes[0][1], 0.0)
        self.assertEqual(player_events[-1].symbol_outcomes[0][2], 1)
        self.assertEqual(player_events[-1].symbol_outcomes[0][3], 1)
        self.assertEqual(player_events[-1].symbol_action_outcomes[0][0], "BTC")
        self.assertEqual(player_events[-1].symbol_action_outcomes[0][1], "FUT_LONG_FULL")
        self.assertGreater(player_events[-1].symbol_action_outcomes[0][2], 0.0)
        self.assertEqual(player_events[-1].symbol_action_outcomes[0][3], 1)
        self.assertEqual(player_events[-1].symbol_action_outcomes[0][4], 1)

    def test_shadow_actor_update_preserves_agent_open_action_in_symbol_action_outcomes(self):
        from panteon_v2.app.shadow_tournament import ProductionShadowTournament

        class SpotAgent:
            label = "SpotAgent"

            def act(self, market):
                if market.bar == 1:
                    return {"BTC": Action.SPOT_BUY_FULL}
                return {"BTC": Action.SPOT_SELL_ALL}

        reg = AgentRegistry()
        reg.register(SpotAgent())
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
        tournament.run_bar(
            make_market_snapshot(bar=2, prices={"BTC": 110.0}, regime="bullish"),
            players=[],
            balance_usd=1000.0,
        )

        agent_events = [
            ev for ev in event_log.query(event_types=[ShadowActorUpdated])
            if ev.actor_type == "agent" and ev.actor_label == "SpotAgent"
        ]
        self.assertGreater(agent_events[-1].realized_pnl_usd, 0.0)
        self.assertEqual(agent_events[-1].symbol_action_outcomes[0][0], "BTC")
        self.assertEqual(agent_events[-1].symbol_action_outcomes[0][1], "SPOT_BUY_FULL")
        self.assertEqual(agent_events[-1].symbol_action_outcomes[0][3], 1)
        self.assertEqual(agent_events[-1].symbol_action_outcomes[0][4], 1)

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
            runtime_event_logs_enabled=False,
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
        for runtime in tournament._runtimes.values():
            self.assertEqual(tuple(runtime._executor._log.query()), ())

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
            {
                "Solo_AgentA": ({
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
                "PlayerA": ({
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
            },
        )

    def test_shadow_tournament_exposes_agent_open_positions_as_solo_replay_labels(self):
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

        tournament.run_bar(
            make_market_snapshot(bar=1, prices={"BTC": 100.0}, regime="bullish"),
            players=[],
            balance_usd=1000.0,
        )

        positions = tournament.last_player_open_positions()
        self.assertEqual(
            positions,
            {
                "Solo_AgentA": ({
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
            },
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

    def test_flash_mode_selects_one_actor_per_symbol_without_leader_selector(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("BtcAgent", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("EthAgent", {"ETH": Action.FUT_SHORT_FULL}))
        reg.register(FakeAgent(
            "WeakAgent",
            {"BTC": Action.FUT_SHORT_FULL, "ETH": Action.FUT_LONG_FULL},
        ))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[],
            live_execution_config=LiveExecutionConfig(max_new_opens_per_bar=2),
            flash_enabled=True,
        )
        _add_perf(pipeline.perf, "BtcAgent", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(pipeline.perf, "EthAgent", Regime.BULLISH, 5, 1.2, start_id=200)
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 5, 0.1, start_id=300)
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].leader, "Panteon_Flash")
        self.assertEqual(steps[0].selected_leader, "Panteon_Flash")
        self.assertEqual(steps[0].executed_leader, "Panteon_Flash")
        self.assertEqual(steps[0].n_raw_signals, 2)
        self.assertEqual(steps[0].n_signals, 2)
        self.assertEqual(steps[0].n_filled, 2)
        self.assertEqual(len(exchange.orders_log), 2)
        self.assertTrue(steps[0].causal_decision["flash_enabled"])
        selected_by_symbol = steps[0].causal_decision[
            "flash_selected_actors_by_symbol"
        ]
        self.assertIn(selected_by_symbol["BTC"], {"BtcAgent", "Solo_BtcAgent"})
        self.assertIn(selected_by_symbol["ETH"], {"EthAgent", "Solo_EthAgent"})
        self.assertEqual(
            [row["symbol"] for row in steps[0].causal_decision["flash_decisions"]],
            ["BTC", "ETH"],
        )

    def test_flash_genetics_no_backfill_veto_consumes_new_open_slot(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("BtcAgent", {"BTC": Action.FUT_LONG_FULL}))
        reg.register(FakeAgent("EthAgent", {"ETH": Action.FUT_LONG_FULL}))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[],
            live_execution_config=LiveExecutionConfig(max_new_opens_per_bar=1),
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_static_enabled=True,
                genetics_confirmation_contra_no_backfill_enabled=True,
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        _add_perf(pipeline.perf, "BtcAgent", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(pipeline.perf, "EthAgent", Regime.BULLISH, 5, 0.9, start_id=200)
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0, "ETH": 50.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(steps[0].n_rate_limited_open_signals, 1)
        self.assertEqual(len(exchange.orders_log), 0)
        self.assertIn(
            "rate_limited_open:ETH:EthAgent",
            steps[0].signal_filter_details,
        )
        decisions = {
            row["symbol"]: row
            for row in steps[0].causal_decision["flash_decisions"]
        }
        self.assertEqual(decisions["BTC"]["reason"], "genetics_contra_no_backfill")
        self.assertIn(
            decisions["BTC"]["original_selected_actor"],
            {"BtcAgent", "Solo_BtcAgent"},
        )

    def test_flash_degradation_state_tracks_closed_pnl_by_actor_symbol_action(self):
        agent = FakeAgent("WeakAgent", {"BTC": Action.SPOT_BUY_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_actor_guard_enabled=True,
                degradation_window_closed_trades=3,
                degradation_min_closed_trades=3,
                degradation_max_recent_pnl_usd=-25.0,
            ),
        )
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 10, 3.0, start_id=100)
        for offset, pnl in enumerate((-8.0, -9.0, -10.0)):
            signal_id = 7 + offset
            decisions = pipeline.flash_allocator.decide(
                make_market(prices={"BTC": 100.0}),
                agents=[agent],
                players=[],
                signal_id_start=signal_id,
            )
            _flash_register_degradation_signal_keys(pipeline, decisions)
            pipeline.event_log.emit(PositionClosed(
                bar=offset + 1,
                trace_id=f"t-{offset + 1}",
                open_signal_id=signal_id,
                close_signal_id=100 + offset,
                sym="BTC",
                side="long",
                realized_pnl=pnl,
                by_player="WeakAgent",
                by_agent="WeakAgent",
            ))

        _flash_update_degradation_state_from_events(pipeline)

        self.assertEqual(
            _flash_degraded_signal_keys(pipeline),
            {"agent:WeakAgent|BTC|SPOT_BUY_FULL"},
        )
        self.assertEqual(_flash_degraded_actor_keys(pipeline), {"agent:WeakAgent"})

    def test_flash_degradation_state_recovers_after_positive_recent_window(self):
        agent = FakeAgent("WeakAgent", {"BTC": Action.SPOT_BUY_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_actor_guard_enabled=True,
                degradation_window_closed_trades=3,
                degradation_min_closed_trades=3,
                degradation_max_recent_pnl_usd=-25.0,
                degradation_recovery_enabled=True,
                degradation_recovery_min_closed_trades=3,
                degradation_recovery_min_recent_pnl_usd=0.0,
            ),
        )
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 10, 3.0, start_id=100)
        signal_key = "agent:WeakAgent|BTC|SPOT_BUY_FULL"
        actor_key = "agent:WeakAgent"
        for offset, pnl in enumerate((-8.0, -9.0, -10.0)):
            signal_id = 7 + offset
            decisions = pipeline.flash_allocator.decide(
                make_market(prices={"BTC": 100.0}),
                agents=[agent],
                players=[],
                signal_id_start=signal_id,
            )
            _flash_register_degradation_signal_keys(pipeline, decisions)
            pipeline.event_log.emit(PositionClosed(
                bar=offset + 1,
                trace_id=f"loss-{offset + 1}",
                open_signal_id=signal_id,
                close_signal_id=100 + offset,
                sym="BTC",
                side="long",
                realized_pnl=pnl,
                by_player="WeakAgent",
                by_agent="WeakAgent",
            ))

        _flash_update_degradation_state_from_events(pipeline)
        self.assertEqual(_flash_degraded_signal_keys(pipeline), {signal_key})
        self.assertEqual(_flash_degraded_actor_keys(pipeline), {actor_key})

        for offset, pnl in enumerate((12.0, 9.0, 7.0)):
            signal_id = 70 + offset
            pipeline._flash_signal_key_by_id[signal_id] = signal_key
            pipeline._flash_actor_key_by_id[signal_id] = actor_key
            pipeline.event_log.emit(PositionClosed(
                bar=10 + offset,
                trace_id=f"recovery-{offset + 1}",
                open_signal_id=signal_id,
                close_signal_id=200 + offset,
                sym="BTC",
                side="long",
                realized_pnl=pnl,
                by_player="WeakAgent",
                by_agent="WeakAgent",
            ))

        _flash_update_degradation_state_from_events(pipeline)

        self.assertEqual(_flash_degraded_signal_keys(pipeline), set())
        self.assertEqual(_flash_degraded_actor_keys(pipeline), set())

    def test_flash_degradation_signal_lcb_can_degrade_without_sum_threshold(self):
        agent = FakeAgent("WeakAgent", {"BTC": Action.SPOT_BUY_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_window_closed_trades=3,
                degradation_min_closed_trades=3,
                degradation_max_recent_pnl_usd=-999.0,
                degradation_signal_min_pnl_per_trade_lcb_usd=-2.0,
                degradation_pnl_per_trade_lcb_z=1.0,
            ),
        )
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 10, 3.0, start_id=100)
        for offset, pnl in enumerate((1.0, 1.0, -6.0)):
            signal_id = 7 + offset
            decisions = pipeline.flash_allocator.decide(
                make_market(prices={"BTC": 100.0}),
                agents=[agent],
                players=[],
                signal_id_start=signal_id,
            )
            _flash_register_degradation_signal_keys(pipeline, decisions)
            pipeline.event_log.emit(PositionClosed(
                bar=offset + 1,
                trace_id=f"lcb-{offset + 1}",
                open_signal_id=signal_id,
                close_signal_id=100 + offset,
                sym="BTC",
                side="long",
                realized_pnl=pnl,
                by_player="WeakAgent",
                by_agent="WeakAgent",
            ))

        _flash_update_degradation_state_from_events(pipeline)

        self.assertEqual(
            _flash_degraded_signal_keys(pipeline),
            {"agent:WeakAgent|BTC|SPOT_BUY_FULL"},
        )

    def test_flash_degradation_actor_cooldown_expires_actor_key(self):
        agent = FakeAgent("WeakAgent", {"BTC": Action.SPOT_BUY_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_actor_guard_enabled=True,
                degradation_actor_cooldown_bars=2,
                degradation_window_closed_trades=1,
                degradation_min_closed_trades=1,
                degradation_max_recent_pnl_usd=-1.0,
            ),
        )
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 10, 3.0, start_id=100)
        decisions = pipeline.flash_allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=7,
        )
        _flash_register_degradation_signal_keys(pipeline, decisions)
        pipeline.event_log.emit(PositionClosed(
            bar=5,
            trace_id="loss",
            open_signal_id=7,
            close_signal_id=100,
            sym="BTC",
            side="long",
            realized_pnl=-2.0,
            by_player="WeakAgent",
            by_agent="WeakAgent",
        ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=5)
        self.assertEqual(_flash_degraded_actor_keys(pipeline), {"agent:WeakAgent"})

        _flash_update_degradation_state_from_events(pipeline, current_bar=6)
        self.assertEqual(_flash_degraded_actor_keys(pipeline), {"agent:WeakAgent"})

        _flash_update_degradation_state_from_events(pipeline, current_bar=7)
        self.assertEqual(_flash_degraded_actor_keys(pipeline), set())

    def test_flash_degradation_signal_cooldown_expires_signal_key(self):
        agent = FakeAgent("WeakAgent", {"BTC": Action.SPOT_BUY_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_signal_cooldown_bars=2,
                degradation_window_closed_trades=1,
                degradation_min_closed_trades=1,
                degradation_max_recent_pnl_usd=-1.0,
            ),
        )
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 10, 3.0, start_id=100)
        decisions = pipeline.flash_allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=7,
        )
        _flash_register_degradation_signal_keys(pipeline, decisions)
        pipeline.event_log.emit(PositionClosed(
            bar=5,
            trace_id="loss",
            open_signal_id=7,
            close_signal_id=100,
            sym="BTC",
            side="long",
            realized_pnl=-2.0,
            by_player="WeakAgent",
            by_agent="WeakAgent",
        ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=5)
        self.assertEqual(
            _flash_degraded_signal_keys(pipeline),
            {"agent:WeakAgent|BTC|SPOT_BUY_FULL"},
        )

        _flash_update_degradation_state_from_events(pipeline, current_bar=6)
        self.assertEqual(
            _flash_degraded_signal_keys(pipeline),
            {"agent:WeakAgent|BTC|SPOT_BUY_FULL"},
        )

        _flash_update_degradation_state_from_events(pipeline, current_bar=7)
        self.assertEqual(_flash_degraded_signal_keys(pipeline), set())

    def test_flash_degradation_symbol_cooldown_expires_open_symbol(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("WeakAgent", {"ETH/USDT": Action.SPOT_BUY_FULL}))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_symbol_guard_enabled=True,
                degradation_symbol_cooldown_bars=2,
                degradation_symbol_window_closed_trades=2,
                degradation_symbol_min_closed_trades=2,
                degradation_symbol_max_recent_pnl_usd=-3.0,
                degradation_window_closed_trades=1,
                degradation_min_closed_trades=1,
                degradation_max_recent_pnl_usd=-1.0,
            ),
        )
        pipeline.event_log.emit(PositionClosed(
            bar=5,
            trace_id="loss",
            open_signal_id=7,
            close_signal_id=100,
            sym="ETH/USDT",
            side="long",
            realized_pnl=-2.0,
            by_player="WeakAgent",
            by_agent="WeakAgent",
        ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=5)
        self.assertEqual(_flash_degraded_open_symbols(pipeline), set())

        pipeline.event_log.emit(PositionClosed(
            bar=6,
            trace_id="loss-2",
            open_signal_id=8,
            close_signal_id=101,
            sym="ETH/USDT",
            side="long",
            realized_pnl=-2.0,
            by_player="WeakAgent",
            by_agent="WeakAgent",
        ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=6)
        self.assertEqual(_flash_degraded_open_symbols(pipeline), {"ETH/USDT"})

        _flash_update_degradation_state_from_events(pipeline, current_bar=7)
        self.assertEqual(_flash_degraded_open_symbols(pipeline), {"ETH/USDT"})

        _flash_update_degradation_state_from_events(pipeline, current_bar=8)
        self.assertEqual(_flash_degraded_open_symbols(pipeline), set())

    def test_flash_degradation_symbol_lookback_ignores_stale_losses(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("WeakAgent", {"ETH/USDT": Action.SPOT_BUY_FULL}))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_symbol_guard_enabled=True,
                degradation_symbol_lookback_bars=10,
                degradation_symbol_min_closed_trades=2,
                degradation_symbol_max_recent_pnl_usd=-3.0,
                degradation_window_closed_trades=1,
                degradation_min_closed_trades=1,
                degradation_max_recent_pnl_usd=-1.0,
            ),
        )
        for bar, pnl in ((1, -2.0), (2, -2.0)):
            pipeline.event_log.emit(PositionClosed(
                bar=bar,
                trace_id=f"stale-{bar}",
                open_signal_id=bar,
                close_signal_id=100 + bar,
                sym="ETH/USDT",
                side="long",
                realized_pnl=pnl,
                by_player="WeakAgent",
                by_agent="WeakAgent",
            ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=20)
        self.assertEqual(_flash_degraded_open_symbols(pipeline), set())

        for bar, pnl in ((19, -2.0), (20, -2.0)):
            pipeline.event_log.emit(PositionClosed(
                bar=bar,
                trace_id=f"fresh-{bar}",
                open_signal_id=bar,
                close_signal_id=200 + bar,
                sym="ETH/USDT",
                side="long",
                realized_pnl=pnl,
                by_player="WeakAgent",
                by_agent="WeakAgent",
            ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=20)
        self.assertEqual(_flash_degraded_open_symbols(pipeline), {"ETH/USDT"})

    def test_flash_degradation_actor_scope_can_isolate_regime(self):
        agent = FakeAgent("WeakAgent", {"BTC": Action.SPOT_BUY_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                degradation_guard_enabled=True,
                degradation_actor_guard_enabled=True,
                degradation_actor_scope="actor_regime",
                degradation_window_closed_trades=1,
                degradation_min_closed_trades=1,
                degradation_max_recent_pnl_usd=-1.0,
            ),
        )
        _add_perf(pipeline.perf, "WeakAgent", Regime.BULLISH, 10, 3.0, start_id=100)
        decisions = pipeline.flash_allocator.decide(
            make_market(prices={"BTC": 100.0}, regime="bullish"),
            agents=[agent],
            players=[],
            signal_id_start=7,
        )
        _flash_register_degradation_signal_keys(pipeline, decisions)
        pipeline.event_log.emit(PositionClosed(
            bar=5,
            trace_id="loss",
            open_signal_id=7,
            close_signal_id=100,
            sym="BTC",
            side="long",
            realized_pnl=-2.0,
            by_player="WeakAgent",
            by_agent="WeakAgent",
        ))

        _flash_update_degradation_state_from_events(pipeline, current_bar=5)

        self.assertEqual(
            _flash_degraded_actor_keys(pipeline, Regime.BULLISH),
            {"agent:WeakAgent"},
        )
        self.assertEqual(
            _flash_degraded_actor_keys(pipeline, Regime.BEARISH),
            set(),
        )

    def test_flash_degradation_tail_reader_avoids_full_event_log_copy(self):
        first = PositionClosed(bar=1, trace_id="a", open_signal_id=1, sym="BTC")
        second = PositionClosed(bar=2, trace_id="b", open_signal_id=2, sym="ETH")

        class TailOnlyLog:
            def __init__(self):
                self._events = [first, second]
                self._lock = threading.Lock()

            def all(self):
                raise AssertionError("full EventLog copy should not be used")

        tail, event_count = _flash_event_log_tail(TailOnlyLog(), 1)

        self.assertEqual(event_count, 2)
        self.assertEqual(tail, [second])

    def test_flash_mode_does_not_execute_agent_rejected_by_hard_policy(self):
        blocked_agent = FakeAgent("BlockedAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(blocked_agent)
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
        )
        _add_perf(pipeline.perf, "BlockedAgent", Regime.BULLISH, 10, 5.0, start_id=100)

        class BlockingStrategist:
            def _validate(self, player):
                return None

            def _validate_hard_policy(self, player, regime, *, current_bar, market_tags=()):
                if player.label in {"BlockedAgent", "Solo_BlockedAgent"}:
                    return CandidateRejection(
                        label=player.label,
                        reason="hard policy denylist: blocked test agent",
                    )
                return None

        pipeline.strategist = BlockingStrategist()
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].leader, "Panteon_Flash")
        self.assertEqual(steps[0].n_raw_signals, 0)
        self.assertEqual(steps[0].n_signals, 0)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(exchange.orders_log, [])
        self.assertEqual(
            steps[0].causal_decision["flash_selected_actors_by_symbol"],
            {"BTC": "NoTrade"},
        )

    def test_flash_shadow_confirmation_scores_can_be_symbol_scoped(self):
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(active)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="agent",
                actor_label="ActiveAgent",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=2,
                winning_trades=1,
                symbol_outcomes=(
                    ("BTC", 2.0, 2, 2),
                    ("ETH", -1.0, 2, 0),
                ),
                symbol_action_outcomes=(
                    ("BTC", "FUT_LONG_FULL", 2.0, 2, 2),
                    ("ETH", "FUT_LONG_FULL", -1.0, 2, 0),
                ),
            )
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0, "ETH": 50.0},
                regime="bullish",
            ),
            agents=[active],
            players=[],
        )

        self.assertNotIn("ActiveAgent", scores)
        self.assertEqual(scores[("ActiveAgent", "BTC", "FUT_LONG_FULL")], (2.0, 2))
        self.assertEqual(scores[("ActiveAgent", "ETH", "FUT_LONG_FULL")], (-1.0, 2))

    def test_symbol_scoped_shadow_scores_include_actor_fallback_when_enabled(self):
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(active)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="agent",
                actor_label="ActiveAgent",
                regime="bullish",
                realized_pnl_usd=5.0,
                closed_trades=80,
                winning_trades=48,
                symbol_outcomes=(("BTC", 2.0, 2, 2),),
                symbol_action_outcomes=(("BTC", "FUT_LONG_FULL", 2.0, 2, 2),),
            )
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0},
                regime="bullish",
            ),
            agents=[active],
            players=[],
        )

        self.assertIn("ActiveAgent", scores)
        self.assertEqual(scores["ActiveAgent"], (5.0, 80))
        self.assertEqual(scores[("ActiveAgent", "BTC", "FUT_LONG_FULL")], (2.0, 2))

    def test_portfolio_shadow_confirmation_uses_all_regime_actor_stats(self):
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(agent)
        player = EnsemblePlayer(
            label="Solo_MomentumScalper",
            agents=[agent],
            weights={"MomentumScalper": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                portfolio_actor_keys=("ensemble:Solo_MomentumScalper",),
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="player",
                actor_label="Solo_MomentumScalper",
                regime="bullish",
                realized_pnl_usd=-5.0,
                closed_trades=2,
                winning_trades=0,
            ),
            ShadowActorUpdated(
                bar=10,
                actor_type="player",
                actor_label="Solo_MomentumScalper",
                regime="neutral",
                realized_pnl_usd=8.0,
                closed_trades=2,
                winning_trades=2,
            ),
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0},
                regime="bullish",
            ),
            agents=[],
            players=[player],
        )

        self.assertEqual(scores["Solo_MomentumScalper"]["score"], 3.0)
        self.assertEqual(scores["Solo_MomentumScalper"]["closed_trades"], 4)
        self.assertEqual(scores["Solo_MomentumScalper"]["winning_trades"], 2)

    def test_flash_shadow_confirmation_scores_include_action_scoped_keys(self):
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(active)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="agent",
                actor_label="ActiveAgent",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=4,
                winning_trades=2,
                symbol_action_outcomes=(
                    ("BTC", "long", 2.0, 2, 2),
                    ("BTC", "short", -1.0, 2, 0),
                ),
            )
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0},
                regime="bullish",
            ),
            agents=[active],
            players=[],
        )

        self.assertEqual(scores[("ActiveAgent", "BTC", "FUT_LONG_FULL")], (2.0, 2))
        self.assertEqual(scores[("ActiveAgent", "BTC", "FUT_SHORT_FULL")], (-1.0, 2))

    def test_flash_shadow_confirmation_scores_include_quality_payload_when_enabled(self):
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(active)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="agent",
                actor_label="ActiveAgent",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=4,
                winning_trades=2,
                symbol_action_outcomes=(
                    ("BTC", "long", 2.0, 4, 2),
                ),
            )
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0},
                regime="bullish",
            ),
            agents=[active],
            players=[],
        )
        payload = scores[("ActiveAgent", "BTC", "FUT_LONG_FULL")]

        self.assertEqual(payload["score"], 2.0)
        self.assertEqual(payload["closed_trades"], 4)
        self.assertEqual(payload["winning_trades"], 2)
        self.assertEqual(payload["win_rate_pct"], 50.0)
        self.assertEqual(payload["recent_downside_usd"], 0.0)

    def test_flash_shadow_confirmation_scores_include_symbol_health_payload(self):
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        backup = FakeAgent("BackupAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(active)
        reg.register(backup)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_symbol_health_enabled=True,
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="agent",
                actor_label="ActiveAgent",
                regime="bullish",
                realized_pnl_usd=2.0,
                closed_trades=2,
                winning_trades=2,
                symbol_action_outcomes=(("BTC", "long", 2.0, 2, 2),),
            ),
            ShadowActorUpdated(
                bar=10,
                actor_type="agent",
                actor_label="BackupAgent",
                regime="bullish",
                realized_pnl_usd=-1.0,
                closed_trades=3,
                winning_trades=1,
                symbol_action_outcomes=(("BTC", "long", -1.0, 3, 1),),
            ),
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0},
                regime="bullish",
            ),
            agents=[active, backup],
            players=[],
        )
        payload = scores[("__symbol_health__", "BTC", "FUT_LONG_FULL")]

        self.assertEqual(payload["score"], 1.0)
        self.assertEqual(payload["closed_trades"], 5)
        self.assertEqual(payload["winning_trades"], 3)
        self.assertAlmostEqual(payload["pnl_per_trade_mean_usd"], 0.2)

    def test_flash_shadow_actor_fallback_scores_include_quality_payload_when_enabled(self):
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        reg = AgentRegistry()
        reg.register(active)
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            flash_allocator_config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
            ),
        )
        pipeline.strategist.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=10,
                actor_type="player",
                actor_label="Solo_ActiveAgent",
                regime="bullish",
                realized_pnl_usd=-3.0,
                closed_trades=4,
                winning_trades=1,
            )
        ])

        scores = _flash_shadow_confirmation_scores(
            pipeline,
            make_market_snapshot(
                bar=11,
                prices={"BTC": 100.0},
                regime="bullish",
            ),
            agents=[active],
            players=[],
        )
        payload = scores["ActiveAgent"]

        self.assertEqual(payload["score"], -3.0)
        self.assertEqual(payload["closed_trades"], 4)
        self.assertEqual(payload["winning_trades"], 1)
        self.assertEqual(payload["win_rate_pct"], 25.0)
        self.assertEqual(payload["recent_downside_usd"], 3.0)

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

    def test_genetics_probation_overlay_caps_flash_origin_genetics_signal(self):
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsRegimeRouter",),
                genetics_probation_allowed_regimes=("bearish", "neutral", "bullish"),
                genetics_probation_risk_mult=0.25,
                genetics_probation_require_shadow_confirmation=True,
            ),
            "_current_shadow_player_signals": {
                "GeneticsRegimeRouter": (
                    Signal(
                        id=99,
                        bar=1,
                        sym="BTC",
                        action=Action.FUT_SHORT_FULL,
                        price=100.0,
                        regime=Regime.BEARISH,
                        by_player="GeneticsRegimeRouter",
                        by_agent="GeneticsRegimeRouter",
                    ),
                ),
            },
        })()
        market = make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
        )
        leader = EnsemblePlayer(
            label="Panteon_Flash",
            agents=[],
            weights={},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        signal = Signal(
            id=1,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
            risk_mult=1.0,
        )

        guard = _apply_genetics_probation_execution_overlay(
            pipeline,
            market,
            leader,
            RealSignalGuardResult(signals=[signal]),
        )

        self.assertEqual(len(guard.signals), 1)
        self.assertAlmostEqual(guard.signals[0].risk_mult, 0.25)
        self.assertIn(
            "genetics_probation_sized:BTC:GeneticsRegimeRouter:risk_mult=0.2500",
            guard.details,
        )

    def test_genetics_probation_overlay_blocks_low_confidence_open(self):
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsRegimeRouter",),
                genetics_probation_allowed_regimes=("bearish",),
                genetics_probation_min_regime_confidence=0.80,
                genetics_probation_risk_mult=0.25,
                genetics_probation_require_shadow_confirmation=False,
            ),
        })()
        market = make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
            regime_confidence=0.55,
        )
        leader = EnsemblePlayer(
            label="Panteon_Flash",
            agents=[],
            weights={},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        signal = Signal(
            id=1,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
            risk_mult=1.0,
        )

        guard = _apply_genetics_probation_execution_overlay(
            pipeline,
            market,
            leader,
            RealSignalGuardResult(signals=[signal]),
        )

        self.assertEqual(guard.signals, [])
        self.assertEqual(guard.filtered, 1)
        self.assertIn(
            "genetics_regime_confidence_blocked:BTC:GeneticsRegimeRouter:0.5500<0.8000",
            guard.details,
        )

    def test_genetics_probation_overlay_blocks_unlisted_signal_key(self):
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsRegimeRouter",),
                genetics_probation_allowed_regimes=("bearish",),
                genetics_probation_allowed_signal_keys=(
                    "agent:GeneticsRegimeRouter|ETH|FUT_SHORT_FULL",
                ),
                genetics_probation_risk_mult=0.25,
                genetics_probation_require_shadow_confirmation=True,
            ),
            "_current_shadow_player_signals": {
                "GeneticsRegimeRouter": (
                    Signal(
                        id=99,
                        bar=1,
                        sym="BTC",
                        action=Action.FUT_SHORT_FULL,
                        price=100.0,
                        regime=Regime.BEARISH,
                        by_player="GeneticsRegimeRouter",
                        by_agent="GeneticsRegimeRouter",
                    ),
                ),
            },
        })()
        market = make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
        )
        leader = EnsemblePlayer(
            label="Panteon_Flash",
            agents=[],
            weights={},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        signal = Signal(
            id=1,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
            risk_mult=1.0,
        )

        guard = _apply_genetics_probation_execution_overlay(
            pipeline,
            market,
            leader,
            RealSignalGuardResult(signals=[signal]),
        )

        self.assertEqual(len(guard.signals), 0)
        self.assertEqual(guard.filtered, 1)
        self.assertIn(
            "genetics_signal_key_blocked:BTC:GeneticsRegimeRouter",
            guard.details,
        )

    def test_genetics_probation_overlay_allows_exact_signal_key(self):
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsRegimeRouter",),
                genetics_probation_allowed_regimes=("bearish",),
                genetics_probation_allowed_signal_keys=(
                    "agent:GeneticsRegimeRouter|BTC|FUT_SHORT_FULL",
                ),
                genetics_probation_risk_mult=0.25,
                genetics_probation_require_shadow_confirmation=True,
            ),
            "_current_shadow_player_signals": {
                "GeneticsRegimeRouter": (
                    Signal(
                        id=99,
                        bar=1,
                        sym="BTC",
                        action=Action.FUT_SHORT_FULL,
                        price=100.0,
                        regime=Regime.BEARISH,
                        by_player="GeneticsRegimeRouter",
                        by_agent="GeneticsRegimeRouter",
                    ),
                ),
            },
        })()
        market = make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="bearish",
        )
        leader = EnsemblePlayer(
            label="Panteon_Flash",
            agents=[],
            weights={},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        signal = Signal(
            id=1,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
            risk_mult=1.0,
        )

        guard = _apply_genetics_probation_execution_overlay(
            pipeline,
            market,
            leader,
            RealSignalGuardResult(signals=[signal]),
        )

        self.assertEqual(len(guard.signals), 1)
        self.assertAlmostEqual(guard.signals[0].risk_mult, 0.25)

    def test_genetics_probation_preselection_degrades_unlisted_shadow_open_keys(self):
        main_loop_module = importlib.import_module("panteon_v2.app.main_loop")
        helper = getattr(
            main_loop_module,
            "_genetics_probation_preselection_degraded_signal_keys",
        )
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsRegimeRouter",),
                genetics_probation_allowed_signal_keys=(
                    "agent:GeneticsRegimeRouter|DOGE/USDT|FUT_SHORT_FULL",
                ),
            ),
        })()
        blocked_signal = Signal(
            id=1,
            bar=1,
            sym="ATOM/USDT",
            action=Action.FUT_SHORT_FULL,
            price=10.0,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
        )
        allowed_signal = Signal(
            id=2,
            bar=1,
            sym="DOGE/USDT",
            action=Action.FUT_SHORT_FULL,
            price=0.2,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
        )
        close_signal = Signal(
            id=3,
            bar=1,
            sym="TRX/USDT",
            action=Action.FUT_CLOSE_ALL,
            price=0.1,
            regime=Regime.BEARISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
        )

        degraded = helper(
            pipeline,
            {"GeneticsRegimeRouter": (blocked_signal, allowed_signal, close_signal)},
            {},
        )

        self.assertIn(
            "agent:GeneticsRegimeRouter|ATOM/USDT|FUT_SHORT_FULL",
            degraded,
        )
        self.assertIn(
            "ensemble:GeneticsRegimeRouter|ATOM/USDT|FUT_SHORT_FULL",
            degraded,
        )
        self.assertNotIn(
            "agent:GeneticsRegimeRouter|DOGE/USDT|FUT_SHORT_FULL",
            degraded,
        )
        self.assertNotIn(
            "agent:GeneticsRegimeRouter|TRX/USDT|FUT_CLOSE_ALL",
            degraded,
        )

    def test_genetics_probation_preselection_includes_shadow_agent_signals(self):
        main_loop_module = importlib.import_module("panteon_v2.app.main_loop")
        helper = getattr(
            main_loop_module,
            "_genetics_probation_preselection_degraded_signal_keys",
        )
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsNeutral",),
                genetics_probation_allowed_signal_keys=(
                    "agent:GeneticsNeutral|DOGE/USDT|FUT_SHORT_FULL",
                ),
            ),
        })()
        blocked_signal = Signal(
            id=1,
            bar=1,
            sym="ATOM/USDT",
            action=Action.FUT_SHORT_FULL,
            price=10.0,
            regime=Regime.BEARISH,
            by_player="Solo_GeneticsNeutral",
            by_agent="GeneticsNeutral",
        )

        degraded = helper(
            pipeline,
            {},
            {"GeneticsNeutral": (blocked_signal,)},
        )

        self.assertIn(
            "agent:GeneticsNeutral|ATOM/USDT|FUT_SHORT_FULL",
            degraded,
        )

    def test_genetics_probation_preselection_ignores_disallowed_regime(self):
        main_loop_module = importlib.import_module("panteon_v2.app.main_loop")
        helper = getattr(
            main_loop_module,
            "_genetics_probation_preselection_degraded_signal_keys",
        )
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsBearish",),
                genetics_probation_allowed_regimes=("bearish",),
                genetics_probation_allowed_signal_keys=(
                    "agent:GeneticsBearish|XLM/USDT|FUT_SHORT_FULL",
                ),
            ),
        })()
        signal = Signal(
            id=1,
            bar=1,
            sym="ETC/USDT",
            action=Action.FUT_SHORT_FULL,
            price=20.0,
            regime=Regime.BULLISH,
            by_player="GeneticsBearish",
            by_agent="GeneticsBearish",
        )

        bullish_degraded = helper(
            pipeline,
            {"GeneticsBearish": (signal,)},
            {},
            regime=Regime.BULLISH,
        )
        bearish_degraded = helper(
            pipeline,
            {"GeneticsBearish": (signal,)},
            {},
            regime=Regime.BEARISH,
        )

        self.assertEqual(bullish_degraded, ())
        self.assertIn(
            "agent:GeneticsBearish|ETC/USDT|FUT_SHORT_FULL",
            bearish_degraded,
        )

    def test_genetics_probation_candidate_not_executable_outside_allowed_regime(self):
        main_loop_module = importlib.import_module("panteon_v2.app.main_loop")
        is_executable = getattr(main_loop_module, "_candidate_is_real_executable")
        genetics_agent = FakeAgent("GeneticsBearish")
        genetics_agent.shadow_only = True
        genetics_agent.live_trading_eligible = False
        candidate = EnsemblePlayer(
            label="GeneticsBearish",
            agents=[genetics_agent],
            weights={"GeneticsBearish": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        pipeline = type("Pipeline", (), {
            "live_execution": LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsBearish",),
                genetics_probation_allowed_regimes=("bearish",),
            ),
        })()

        self.assertTrue(
            is_executable(candidate, pipeline, regime=Regime.BEARISH),
        )
        self.assertFalse(
            is_executable(candidate, pipeline, regime=Regime.BULLISH),
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

    def test_flash_causal_payload_includes_configured_shadow_position_diagnostics(self):
        class Strategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1
                v3_shadow_fresh_handoff_require_positive_unrealized = True

            _config = Config()

        class Pipeline:
            strategist = Strategist()
            _current_actionable_player_labels = {"Optimal_StaticRotator"}
            _current_shadow_player_signals = {
                "Optimal_StaticRotator": (
                    Signal(
                        id=10,
                        bar=2983,
                        sym="TRX/USDT",
                        action=Action.FUT_LONG_FULL,
                        price=0.085,
                        regime=Regime.BULLISH,
                        by_player="Optimal_StaticRotator",
                        by_agent="ShadowPositionReplay",
                    ),
                )
            }
            _pending_shadow_player_positions = {
                "Optimal_StaticRotator": ({
                    "sym": "TRX/USDT",
                    "side": "long",
                    "opened_bar": 2982,
                    "age_bars": 1,
                    "entry_price": 0.08,
                    "current_price": 0.085,
                    "qty": 100.0,
                    "unrealized_pnl_usd": 0.5,
                    "fresh": True,
                },),
                "OtherPlayer": ({
                    "sym": "ETH/USDT",
                    "side": "short",
                    "opened_bar": 2983,
                    "age_bars": 0,
                    "entry_price": 2000.0,
                    "current_price": 1990.0,
                    "qty": 1.0,
                    "unrealized_pnl_usd": 10.0,
                    "fresh": True,
                },),
                "BlockedPlayer": ({
                    "sym": "XRP/USDT",
                    "side": "long",
                    "opened_bar": 2982,
                    "age_bars": 1,
                    "entry_price": 0.60,
                    "current_price": 0.61,
                    "qty": 100.0,
                    "unrealized_pnl_usd": 1.0,
                    "fresh": True,
                },),
            }
            shadow_position_diagnostic_bars = (2983,)
            shadow_position_diagnostic_labels = (
                "Optimal_StaticRotator",
                "BlockedPlayer",
                "MissingPlayer",
            )
            _flash_candidate_safety_reasons = {
                "BlockedPlayer": "uses quarantined agent 'AgentX'",
            }

        class Guard:
            filtered = 0
            details = ()

        payload = _flash_causal_decision_payload(
            market=make_market_snapshot(
                bar=2983,
                prices={"TRX/USDT": 0.085, "ETH/USDT": 1990.0, "XRP/USDT": 0.61},
                regime="bullish",
            ),
            decisions=(
                FlashDecision(
                    symbol="TRX/USDT",
                    selected_actor="NoTrade",
                    actor_type="no_trade",
                    score=0.0,
                    action=Action.HOLD,
                    reason="no_eligible_actor",
                    signal=None,
                    candidates=(
                        FlashCandidateAudit(
                            symbol="TRX/USDT",
                            label="Optimal_StaticRotator",
                            actor_type="ensemble",
                            actor_key="ensemble:Optimal_StaticRotator",
                            score=1.25,
                            base_score=1.0,
                            gate_score=1.25,
                            action=Action.FUT_LONG_FULL,
                            rank=2,
                            rejected=True,
                            reason="pnl_below_threshold",
                            closed_trades=12,
                            shadow_score=2.0,
                            shadow_closed_trades=7,
                            shadow_source="actor",
                            selected_subset_protected=True,
                            selected_subset_score_boost=0.25,
                        ),
                    ),
                ),
            ),
            raw_signals=(),
            executable_signals=(),
            signal_guard=Guard(),
            n_filled=0,
            n_rejected=0,
            n_blocked=0,
            blocked_reasons={},
            pipeline=Pipeline(),
        )

        diagnostics = payload["shadow_position_diagnostics"]
        by_label = {item["label"]: item for item in diagnostics["players"]}
        self.assertEqual(diagnostics["bar"], 2983)
        self.assertEqual(diagnostics["max_age_bars"], 1)
        self.assertTrue(diagnostics["require_positive_unrealized"])
        self.assertEqual(
            by_label["Optimal_StaticRotator"]["shadow_signal_count"],
            1,
        )
        self.assertEqual(
            by_label["Optimal_StaticRotator"]["positions"][0]["replay_action"],
            "FUT_LONG_FULL",
        )
        self.assertTrue(
            by_label["Optimal_StaticRotator"]["positions"][0]["replay_eligible"],
        )
        decision_context = by_label["Optimal_StaticRotator"]["positions"][0][
            "decision_context"
        ]
        self.assertTrue(decision_context["candidate_found"])
        self.assertEqual(decision_context["decision_reason"], "no_eligible_actor")
        self.assertEqual(decision_context["candidate_reason"], "pnl_below_threshold")
        self.assertTrue(decision_context["candidate_selected_subset_protected"])
        blocked_context = by_label["BlockedPlayer"]["positions"][0][
            "decision_context"
        ]
        self.assertFalse(blocked_context["candidate_found"])
        self.assertEqual(
            blocked_context["pre_allocator_safety_reason"],
            "uses quarantined agent 'AgentX'",
        )
        self.assertEqual(by_label["MissingPlayer"]["positions"], [])
        self.assertNotIn("OtherPlayer", by_label)

    def test_flash_candidate_safety_uses_v3_real_loss_only_with_v3_rolling_score(self):
        candidate = EnsemblePlayer(
            label="Optimal_StaticRotator",
            agents=[FakeAgent("AgentA")],
            weights={"AgentA": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class Issue:
            reason = "v3 real-loss kill: pnl_pct -1.74 <= -1.00"

        class Strategist:
            class Config:
                use_v3_rolling_score = False

            _config = Config()

            def _validate(self, player):
                return None

            def _validate_hard_policy(self, player, regime, **kwargs):
                return None

            def _validate_v3_real_loss(self, player, regime, **kwargs):
                return Issue()

        class Pipeline:
            strategist = Strategist()

        market = make_market_snapshot(
            bar=2983,
            prices={"TRX/USDT": 0.085},
            regime="bullish",
        )

        pipeline = Pipeline()
        self.assertFalse(
            _fallback_candidate_safety_issue(pipeline, candidate, market)
        )
        self.assertEqual(
            pipeline._flash_candidate_safety_reasons["Optimal_StaticRotator"],
            "",
        )

        Strategist.Config.use_v3_rolling_score = True
        pipeline = Pipeline()
        self.assertTrue(
            _fallback_candidate_safety_issue(pipeline, candidate, market)
        )
        self.assertIn(
            "v3 real-loss kill",
            pipeline._flash_candidate_safety_reasons["Optimal_StaticRotator"],
        )

    def test_flash_shadow_player_signals_include_fresh_positive_positions(self):
        agent = FakeAgent("PositionAgent")
        player = EnsemblePlayer(
            label="Solo_PositionAgent",
            agents=[agent],
            weights={"PositionAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class Strategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1

            _config = Config()

        class Pipeline:
            strategist = Strategist()
            _current_shadow_player_signals = {}
            _pending_shadow_player_positions = {
                "Solo_PositionAgent": ({
                    "sym": "BTC",
                    "side": "long",
                    "opened_bar": 10,
                    "age_bars": 0,
                    "entry_price": 100.0,
                    "current_price": 102.0,
                    "qty": 1.0,
                    "unrealized_pnl_usd": 2.0,
                    "fresh": True,
                },),
            }

        signals = _flash_shadow_player_signals_with_position_replay(
            Pipeline(),
            make_market_snapshot(
                bar=10,
                prices={"BTC": 102.0},
                regime="neutral",
            ),
            [player],
            signal_id_counter=7,
        )

        self.assertIn("Solo_PositionAgent", signals)
        self.assertEqual(len(signals["Solo_PositionAgent"]), 1)
        signal = signals["Solo_PositionAgent"][0]
        self.assertEqual(signal.id, 7)
        self.assertEqual(signal.sym, "BTC")
        self.assertEqual(signal.action, Action.FUT_LONG_FULL)
        self.assertEqual(signal.by_player, "Solo_PositionAgent")
        self.assertEqual(signal.by_agent, "ShadowPositionReplay")

    def test_flash_shadow_player_signals_replay_known_solo_agent_position_without_candidate(self):
        agent = FakeAgent("PositionAgent")

        class Strategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1

            _config = Config()

        class Pipeline:
            strategist = Strategist()
            _current_shadow_player_signals = {}
            _pending_shadow_player_positions = {
                "Solo_PositionAgent": ({
                    "sym": "BTC",
                    "side": "long",
                    "opened_bar": 10,
                    "age_bars": 0,
                    "entry_price": 100.0,
                    "current_price": 102.0,
                    "qty": 1.0,
                    "unrealized_pnl_usd": 2.0,
                    "fresh": True,
                },),
            }

        signals = _flash_shadow_player_signals_with_position_replay(
            Pipeline(),
            make_market_snapshot(
                bar=10,
                prices={"BTC": 102.0},
                regime="neutral",
            ),
            [],
            agents=[agent],
            signal_id_counter=7,
        )

        self.assertIn("Solo_PositionAgent", signals)
        signal = signals["Solo_PositionAgent"][0]
        self.assertEqual(signal.sym, "BTC")
        self.assertEqual(signal.by_player, "Solo_PositionAgent")
        self.assertEqual(signal.by_agent, "ShadowPositionReplay")

    def test_flash_shadow_player_signals_can_include_fresh_losing_positions_when_configured(self):
        agent = FakeAgent("PositionAgent")
        player = EnsemblePlayer(
            label="Solo_PositionAgent",
            agents=[agent],
            weights={"PositionAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class Strategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = True
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1
                v3_shadow_fresh_handoff_require_positive_unrealized = False

            _config = Config()

        class Pipeline:
            strategist = Strategist()
            _current_shadow_player_signals = {}
            _pending_shadow_player_positions = {
                "Solo_PositionAgent": ({
                    "sym": "BTC",
                    "side": "long",
                    "age_bars": 0,
                    "unrealized_pnl_usd": -1.5,
                    "fresh": True,
                },),
            }

        signals = _flash_shadow_player_signals_with_position_replay(
            Pipeline(),
            make_market_snapshot(
                bar=10,
                prices={"BTC": 98.0},
                regime="neutral",
            ),
            [player],
            signal_id_counter=7,
        )

        self.assertIn("Solo_PositionAgent", signals)
        self.assertEqual(signals["Solo_PositionAgent"][0].action, Action.FUT_LONG_FULL)

    def test_flash_shadow_player_signals_do_not_replay_positions_when_disabled(self):
        agent = FakeAgent("PositionAgent")
        player = EnsemblePlayer(
            label="Solo_PositionAgent",
            agents=[agent],
            weights={"PositionAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class Strategist:
            class Config:
                v3_shadow_fresh_handoff_enabled = False
                v3_shadow_flat_handoff_enabled = False
                v3_shadow_fresh_handoff_max_age_bars = 1

            _config = Config()

        class Pipeline:
            strategist = Strategist()
            _current_shadow_player_signals = {}
            _pending_shadow_player_positions = {
                "Solo_PositionAgent": ({
                    "sym": "BTC",
                    "side": "short",
                    "age_bars": 0,
                    "unrealized_pnl_usd": 2.0,
                    "fresh": True,
                },),
            }

        signals = _flash_shadow_player_signals_with_position_replay(
            Pipeline(),
            make_market_snapshot(
                bar=10,
                prices={"BTC": 98.0},
                regime="bearish",
            ),
            [player],
            signal_id_counter=7,
        )

        self.assertEqual(signals, {})

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

    def test_actionable_fallback_registry_score_penalizes_drawdown_over_raw_pnl(self):
        from panteon_v2.app.main_loop import _fallback_registry_score

        registry = AgentRegistry()
        registry.register(FakeAgent("RiskyAgent"))
        registry.register(FakeAgent("SafeAgent"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
        )
        pipeline.perf.restore({
            "trade_fraction": 0.10,
            "state": {
                "RiskyAgent|neutral": {
                    "closed_trades": 20,
                    "entries": 20,
                    "signals": 40,
                    "wins": 12,
                    "losses": 8,
                    "pnl_pct": 6.0,
                    "returns": [0.4, 0.2, -0.5, 0.6],
                    "max_dd_pct": 5.0,
                },
                "SafeAgent|neutral": {
                    "closed_trades": 20,
                    "entries": 20,
                    "signals": 40,
                    "wins": 12,
                    "losses": 8,
                    "pnl_pct": 5.0,
                    "returns": [0.2, 0.2, -0.1, 0.3],
                    "max_dd_pct": 1.0,
                },
            },
            "open": {},
            "seen_signal_ids": [],
        })

        risky = _fallback_registry_score(
            pipeline,
            "Solo_RiskyAgent",
            "RiskyAgent",
            Regime.NEUTRAL,
            current_bar=100,
        )
        safe = _fallback_registry_score(
            pipeline,
            "Solo_SafeAgent",
            "SafeAgent",
            Regime.NEUTRAL,
            current_bar=100,
        )

        self.assertGreater(safe, risky)

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

    def test_genetics_probation_fixed_candidate_bypasses_live_flag_only_when_explicitly_allowed(self):
        allowed_agent = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_LONG_FULL})
        allowed_agent.shadow_only = True
        allowed_agent.live_trading_eligible = False
        blocked_agent = FakeAgent("GeneticsRegimeRouter", {"BTC": Action.FUT_LONG_FULL})
        blocked_agent.shadow_only = True
        blocked_agent.live_trading_eligible = False
        reg = AgentRegistry()
        reg.register(allowed_agent)
        reg.register(blocked_agent)

        class Selector:
            def select(self, regime, *, k):
                return []

        class Composer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return None

        class QM:
            def is_quarantined(self, label):
                return False

        class Pipeline:
            profiles = ()
            composer = Composer()
            registry = reg
            qm = QM()
            selector = Selector()
            fixed_agent_player_sets = (
                ("GeneticsNeutral", ("GeneticsNeutral",)),
                ("GeneticsRegimeRouter", ("GeneticsRegimeRouter",)),
            )
            regime_switch_player_sets = ()
            rotating_agent_player_sets = ()
            solo_agent_candidate_limit = 0
            live_execution = LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsNeutral",),
                genetics_probation_allowed_regimes=("neutral",),
            )

        candidates = _compose_candidates(Pipeline(), Regime.NEUTRAL)

        self.assertEqual([player.label for player in candidates], ["GeneticsNeutral"])
        self.assertFalse(getattr(allowed_agent, "live_trading_eligible"))

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
        signals, errors = players[0].vote(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            signal_id_start=10,
        )
        self.assertEqual(errors, [])
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
        signals, errors = players[0].vote(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            signal_id_start=10,
        )
        self.assertEqual(errors, [])
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

        signals, errors = players[0].vote(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            signal_id_start=10,
        )
        self.assertEqual(errors, [])
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

        signals, errors = players[0].vote(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            signal_id_start=10,
        )
        self.assertEqual(errors, [])
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
        signals, errors = players[0].vote(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            signal_id_start=10,
        )
        self.assertEqual(signals, [])
        self.assertEqual(errors, [])

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

    def test_flash_stale_position_guard_closes_old_losing_position(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
        )
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=7,
            sym="BTC",
            side="long",
            entry_price=100.0,
            qty=1.0,
            fee_open=0.0,
            by_player="Panteon_Flash",
            by_agent="MomentumScalper",
            opened_at=datetime.now(timezone.utc),
            opened_bar=10,
            open_action="FUT_LONG_FULL",
        ))

        signals = _flash_stale_position_close_signals(
            pipeline,
            make_market_snapshot(
                bar=180,
                prices={"BTC": 95.0},
                regime="bullish",
            ),
            signal_id_start=100,
            max_age_bars=168,
            require_nonpositive_unrealized=True,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].action, Action.FUT_CLOSE_ALL)
        self.assertEqual(signals[0].sym, "BTC")
        self.assertEqual(signals[0].by_agent, "StalePositionGuard")

    def test_flash_stale_position_guard_keeps_old_profitable_position(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Idle"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
        )
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=7,
            sym="BTC",
            side="short",
            entry_price=100.0,
            qty=1.0,
            fee_open=0.0,
            by_player="Panteon_Flash",
            by_agent="LiveCrashHunter",
            opened_at=datetime.now(timezone.utc),
            opened_bar=10,
            open_action="FUT_SHORT_FULL",
        ))

        signals = _flash_stale_position_close_signals(
            pipeline,
            make_market_snapshot(
                bar=180,
                prices={"BTC": 95.0},
                regime="bearish",
            ),
            signal_id_start=100,
            max_age_bars=168,
            require_nonpositive_unrealized=True,
        )

        self.assertEqual(signals, [])

    def test_flash_path_executes_stale_position_guard_close(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Idle"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
        )
        pipeline.flash_stale_position_exit_enabled = True
        pipeline.flash_stale_position_exit_max_age_bars = 168
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=7,
            sym="BTC",
            side="long",
            entry_price=100.0,
            qty=1.0,
            fee_open=0.0,
            by_player="Panteon_Flash",
            by_agent="MomentumScalper",
            opened_at=datetime.now(timezone.utc),
            opened_bar=10,
            open_action="FUT_LONG_FULL",
        ))
        exchange._positions["BTC"] = ExchangePosition(
            sym="BTC",
            side="long",
            qty=1.0,
            entry=100.0,
        )
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=180,
            prices={"BTC": 95.0},
            regime="bullish",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].n_raw_signals, 1)
        self.assertEqual(steps[0].n_filled, 1)
        self.assertFalse(pipeline.executor._tracker.has("BTC"))
        emitted = list(pipeline.event_log.query(event_types=[SignalEmitted]))
        self.assertEqual(emitted[0].signal.by_agent, "StalePositionGuard")

    def test_genetics_probation_regime_exit_closes_position_outside_allowed_regime(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Idle"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            live_execution_config=LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsBearish",),
                genetics_probation_allowed_regimes=("bearish",),
            ),
        )
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=17,
            sym="XLM/USDT",
            side="short",
            entry_price=0.12,
            qty=1000.0,
            fee_open=0.0,
            by_player="Panteon_Flash",
            by_agent="GeneticsBearish",
            opened_at=datetime.now(timezone.utc),
            opened_bar=20,
            open_action="FUT_SHORT_FULL",
            open_regime="bearish",
        ))

        signals = _genetics_probation_regime_exit_close_signals(
            pipeline,
            make_market_snapshot(
                bar=25,
                prices={"XLM/USDT": 0.13},
                regime="neutral",
                regime_confidence=0.95,
            ),
            signal_id_start=200,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].id, 200)
        self.assertEqual(signals[0].action, Action.FUT_CLOSE_ALL)
        self.assertEqual(signals[0].sym, "XLM/USDT")
        self.assertEqual(signals[0].by_agent, "GeneticsProbationRegimeExit")

    def test_genetics_probation_regime_exit_closes_position_on_low_confidence(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Idle"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            live_execution_config=LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsBearish",),
                genetics_probation_allowed_regimes=("bearish",),
                genetics_probation_min_regime_confidence=0.80,
            ),
        )
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=18,
            sym="ETC/USDT",
            side="short",
            entry_price=20.0,
            qty=2.0,
            fee_open=0.0,
            by_player="Panteon_Flash",
            by_agent="GeneticsBearish",
            opened_at=datetime.now(timezone.utc),
            opened_bar=20,
            open_action="FUT_SHORT_FULL",
            open_regime="bearish",
        ))

        signals = _genetics_probation_regime_exit_close_signals(
            pipeline,
            make_market_snapshot(
                bar=25,
                prices={"ETC/USDT": 21.0},
                regime="bearish",
                regime_confidence=0.50,
            ),
            signal_id_start=210,
        )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].sym, "ETC/USDT")
        self.assertEqual(signals[0].by_player, "Panteon_Flash")

    def test_genetics_probation_regime_exit_ignores_valid_confident_regime(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Idle"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=1000.0,
            profiles=[],
            flash_enabled=True,
            live_execution_config=LiveExecutionConfig(
                genetics_probation_execution_enabled=True,
                genetics_probation_labels=("GeneticsBearish",),
                genetics_probation_allowed_regimes=("bearish",),
                genetics_probation_min_regime_confidence=0.80,
            ),
        )
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=19,
            sym="ETC/USDT",
            side="short",
            entry_price=20.0,
            qty=2.0,
            fee_open=0.0,
            by_player="Panteon_Flash",
            by_agent="GeneticsBearish",
            opened_at=datetime.now(timezone.utc),
            opened_bar=20,
            open_action="FUT_SHORT_FULL",
            open_regime="bearish",
        ))

        signals = _genetics_probation_regime_exit_close_signals(
            pipeline,
            make_market_snapshot(
                bar=25,
                prices={"ETC/USDT": 19.0},
                regime="bearish",
                regime_confidence=0.90,
            ),
            signal_id_start=220,
        )

        self.assertEqual(signals, [])


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
            with open(os.path.join(out, "dashboard.html"), "r", encoding="utf-8") as f:
                html_text = f.read()
            self.assertIn('rel="icon"', html_text)
            self.assertIn("№1", html_text)
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
                "shadow_dashboard_MEXC.png",
                "regime_dashboard_MEXC.png",
                "memory_dashboard_MEXC.png",
            ):
                path = os.path.join(td, name)
                self.assertTrue(os.path.exists(path), name)
                with open(path, "rb") as f:
                    self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

    def test_writer_close_creates_final_session_reports(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="BITGET"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.close()
            report_json = os.path.join(writer.output_dir, "final_session_report.json")
            report_md = os.path.join(writer.output_dir, "final_session_report.md")

            self.assertTrue(os.path.exists(report_json))
            self.assertTrue(os.path.exists(report_md))
            with open(report_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["exchange"], "BITGET")
            self.assertEqual(payload["run_state"], "stopped")
            self.assertIn("live_session", payload)
            with open(report_md, "r", encoding="utf-8") as f:
                markdown = f.read()
            self.assertIn("# Panteon v2 final session report", markdown)
            self.assertIn("BITGET", markdown)

    def test_writer_disables_latest_dashboard_publish_after_permission_error(self):
        from pathlib import Path

        from panteon_v2.app.output_writer import OutputWriterConfig

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td) / "session"
            latest_dir = Path(td) / "latest"
            output_dir.mkdir()
            latest_dir.mkdir()
            src = output_dir / "dashboard_latest.png"
            src.write_bytes(b"not-a-real-png")
            writer = object.__new__(OutputWriter)
            writer._config = OutputWriterConfig(
                output_dir=str(output_dir),
                latest_dir=str(latest_dir),
            )
            writer._pipeline = type("Pipeline", (), {"exchange_name": "MEXC"})()

            with patch(
                "panteon_v2.app.output_writer.os.replace",
                side_effect=PermissionError("locked"),
            ) as replace_mock, patch(
                "panteon_v2.app.output_writer.log.exception",
            ) as log_mock:
                writer._publish_latest_visual_dashboards([str(src)])
                writer._publish_latest_visual_dashboards([str(src)])

        self.assertEqual(replace_mock.call_count, 1)
        self.assertEqual(log_mock.call_count, 1)

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

    def test_status_and_trading_log_include_flash_actor_debug(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
            flash_enabled=True,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.write(StepResult(
                bar=1,
                regime=Regime.BULLISH,
                leader="Panteon_Flash",
                leader_changed=False,
                n_signals=2,
                n_filled=2,
                n_rejected=0,
                n_blocked=0,
                selected_leader="Panteon_Flash",
                executed_leader="Panteon_Flash",
                selected_actors_by_symbol={"BTC": "BtcAgent", "ETH": "EthAgent"},
                causal_decision={
                    "flash_enabled": True,
                    "flash_selected_actors_by_symbol": {
                        "BTC": "BtcAgent",
                        "ETH": "EthAgent",
                    },
                    "flash_actor_types_by_symbol": {
                        "BTC": "agent",
                        "ETH": "agent",
                    },
                    "flash_decisions": [
                        {"symbol": "BTC", "selected_actor": "BtcAgent"},
                        {"symbol": "ETH", "selected_actor": "EthAgent"},
                    ],
                },
            ))
            with open(os.path.join(writer.output_dir, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            with open(os.path.join(writer.output_dir, "trading.log"), "r", encoding="utf-8") as f:
                trading_log = f.read()
            writer.close()

        self.assertTrue(status["flash"]["enabled"])
        self.assertEqual(
            status["flash"]["selected_actors_by_symbol"],
            {"BTC": "BtcAgent", "ETH": "EthAgent"},
        )
        self.assertEqual(status["decision_debug"]["flash_decisions_count"], 2)
        self.assertIn("flash_actors=BTC:BtcAgent,ETH:EthAgent", trading_log)

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

    def test_writer_compacts_causal_entry_flash_candidates_when_requested(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(
                pipeline,
                results_root=td,
                compact_causal_entry_decisions=True,
            )
            writer.write(StepResult(
                bar=8,
                regime=Regime.BULLISH,
                leader="Panteon_Flash",
                leader_changed=False,
                n_signals=1,
                n_filled=1,
                n_rejected=0,
                n_blocked=0,
                selected_leader="Panteon_Flash",
                executed_leader="Panteon_Flash",
                causal_decision={
                    "bar": 8,
                    "executable_signals": [
                        {
                            "id": 101,
                            "sym": "BTC/USDT",
                            "action": "FUT_LONG_FULL",
                            "by_player": "Alpha",
                        }
                    ],
                    "flash_decisions": [
                        {
                            "symbol": "BTC/USDT",
                            "selected_actor": "Alpha",
                            "actor_type": "agent",
                            "score": 2.5,
                            "action": "FUT_LONG_FULL",
                            "signal": {
                                "id": 101,
                                "sym": "BTC/USDT",
                                "action": "FUT_LONG_FULL",
                                "by_player": "Alpha",
                                "by_agent": "Alpha",
                            },
                            "candidates": [
                                {
                                    "label": "Alpha",
                                    "actor_type": "agent",
                                    "actor_key": "agent:Alpha",
                                    "rank": 1,
                                    "score": 2.5,
                                    "shadow_score": 1.5,
                                    "shadow_closed_trades": 70,
                                    "heavy_debug_payload": "x" * 1000,
                                },
                                {
                                    "label": "Beta",
                                    "actor_type": "agent",
                                    "actor_key": "agent:Beta",
                                    "rank": 2,
                                    "score": 2.4,
                                    "rejected": True,
                                    "reason": "shadow_pnl_per_trade_lcb_below_threshold",
                                    "shadow_score": 1.4,
                                    "shadow_closed_trades": 65,
                                },
                                {
                                    "label": "Gamma",
                                    "actor_type": "agent",
                                    "actor_key": "agent:Gamma",
                                    "rank": 3,
                                    "score": 2.3,
                                    "rejected": True,
                                    "reason": "score_below_threshold",
                                    "shadow_score": 1.3,
                                    "shadow_closed_trades": 60,
                                },
                            ],
                        }
                    ],
                },
            ))
            path = os.path.join(writer.output_dir, "causal_entry_decisions.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            writer.close()

        decision = rows[0]["flash_decisions"][0]
        self.assertEqual(decision["selected_actor"], "Alpha")
        self.assertEqual(decision["signal"]["id"], 101)
        self.assertEqual(rows[0]["executable_signals"][0]["id"], 101)
        self.assertEqual(
            [candidate["label"] for candidate in decision["candidates"]],
            ["Alpha"],
        )
        self.assertNotIn("heavy_debug_payload", decision["candidates"][0])
        self.assertEqual(decision["candidate_count"], 3)
        self.assertEqual(decision["rejected_candidate_count"], 2)
        self.assertEqual(
            decision["candidate_rejection_counts"],
            {
                "score_below_threshold": 1,
                "shadow_pnl_per_trade_lcb_below_threshold": 1,
            },
        )
        self.assertEqual(
            [candidate["label"] for candidate in decision["top_rejected_candidates"]],
            ["Beta", "Gamma"],
        )

    def test_writer_compacts_notrade_flash_decisions_without_candidate_payload(self):
        decision = OutputWriter._compact_flash_decision(
            {
                "symbol": "BTC/USDT",
                "selected_actor": "NoTrade",
                "actor_type": "no_trade",
                "score": 0.0,
                "action": "HOLD",
                "signal": None,
                "candidates": [
                    {
                        "label": "Alpha",
                        "actor_type": "agent",
                        "rank": 1,
                        "rejected": True,
                        "reason": "score_below_threshold",
                        "heavy_debug_payload": "x" * 1000,
                    }
                ],
            }
        )

        self.assertEqual(decision["candidate_count"], 1)
        self.assertEqual(decision["candidate_rejection_counts"], {
            "score_below_threshold": 1
        })
        self.assertEqual(decision["candidates"], [])
        self.assertEqual(decision["top_rejected_candidates"], [])

    def test_writer_can_compact_causal_entry_to_selected_flash_decisions_only(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(
                pipeline,
                results_root=td,
                compact_causal_entry_decisions=True,
                compact_causal_entry_selected_only=True,
            )
            writer.write(StepResult(
                bar=9,
                regime=Regime.BULLISH,
                leader="Panteon_Flash",
                leader_changed=False,
                n_signals=1,
                n_filled=1,
                n_rejected=0,
                n_blocked=0,
                selected_leader="Panteon_Flash",
                executed_leader="Panteon_Flash",
                causal_decision={
                    "bar": 9,
                    "flash_decisions": [
                        {
                            "symbol": "BTC/USDT",
                            "selected_actor": "Alpha",
                            "actor_type": "agent",
                            "action": "FUT_LONG_FULL",
                            "signal": {
                                "id": 101,
                                "sym": "BTC/USDT",
                                "action": "FUT_LONG_FULL",
                            },
                            "candidates": [],
                        },
                        {
                            "symbol": "ETH/USDT",
                            "selected_actor": "NoTrade",
                            "actor_type": "no_trade",
                            "action": "HOLD",
                            "signal": None,
                            "candidates": [],
                        },
                    ],
                },
            ))
            path = os.path.join(writer.output_dir, "causal_entry_decisions.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            writer.close()

        decisions = rows[0]["flash_decisions"]
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["selected_actor"], "Alpha")
        self.assertEqual(decisions[0]["signal"]["id"], 101)

    def test_writer_skips_empty_selected_only_causal_rows_without_diagnostics(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(
                pipeline,
                results_root=td,
                compact_causal_entry_decisions=True,
                compact_causal_entry_selected_only=True,
            )
            writer.write(StepResult(
                bar=10,
                regime=Regime.NEUTRAL,
                leader="Panteon_Flash",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
                selected_leader="Panteon_Flash",
                executed_leader="Panteon_Flash",
                causal_decision={
                    "bar": 10,
                    "flash_decisions": [
                        {
                            "symbol": "BTC/USDT",
                            "selected_actor": "NoTrade",
                            "actor_type": "no_trade",
                            "action": "HOLD",
                            "signal": None,
                            "candidates": [],
                        },
                    ],
                },
            ))
            writer.write(StepResult(
                bar=11,
                regime=Regime.NEUTRAL,
                leader="Panteon_Flash",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
                selected_leader="Panteon_Flash",
                executed_leader="Panteon_Flash",
                causal_decision={
                    "bar": 11,
                    "flash_decisions": [],
                    "shadow_position_diagnostics": {"bar": 11, "players": []},
                },
            ))
            path = os.path.join(writer.output_dir, "causal_entry_decisions.jsonl")
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            writer.close()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bar"], 11)
        self.assertIn("shadow_position_diagnostics", rows[0])

    def test_writer_recovers_when_output_dir_disappears_mid_run(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=FakeExchange(name="REAL"),
            initial_capital=100.0,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            out = writer.output_dir
            shutil.rmtree(out)

            writer.write(StepResult(
                bar=9,
                regime=Regime.NEUTRAL,
                leader="DefaultEnsemble",
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
                causal_decision={"bar": 9, "raw_signals": []},
            ))

            self.assertTrue(os.path.exists(os.path.join(out, "status.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "trading.log")))
            self.assertTrue(os.path.exists(
                os.path.join(out, "causal_entry_decisions.jsonl"),
            ))
            writer.close()

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

    def test_recover_exchange_positions_adopts_unknown_when_enabled(self):
        from panteon_v2.app.startup import _recover_exchange_positions
        from panteon_v2.execution.exchange import ExchangePosition

        class LiveExchange(FakeExchange):
            def get_all_positions(self):
                return {
                    "ETH": ExchangePosition(sym="ETH", side="short", qty=0.2, entry=50.0, leverage=2),
                }

        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=LiveExchange(name="BITGET"),
            initial_capital=100.0,
            live_execution_config=LiveExecutionConfig(
                adopt_existing_positions_enabled=True,
            ),
        )

        recovered = _recover_exchange_positions(pipeline)

        pos = pipeline.executor._tracker.get("ETH")
        self.assertEqual(recovered, 1)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.by_player, "PanteonFlashAdopted")
        self.assertEqual(pos.by_agent, "AdoptedExchangePosition")
        self.assertEqual(pos.open_action, "FUT_SHORT_FULL")
        self.assertEqual(pos.open_regime, "neutral")
        opened = pipeline.perf.snapshot()["open"]
        self.assertIn("AdoptedExchangePosition|ETH", opened)
        self.assertIn("PanteonFlashAdopted|ETH", opened)

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

    def test_player_leaderboard_includes_regime_switch_player_memory(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("ResearchValidatorAgent"))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        pipeline.regime_switch_player_sets = (
            ("Antonius_conservative", {"neutral": "ResearchValidatorAgent"}),
        )
        migrate_v1_regime_memory({
            "neutral": {
                "Antonius_conservative": {
                    "samples": 7,
                    "wins": 5,
                    "losses": 2,
                    "pnl_pct": 2.4,
                },
                "ResearchValidatorAgent": {
                    "samples": 7,
                    "wins": 5,
                    "losses": 2,
                    "pnl_pct": 2.4,
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

        self.assertIn("V_Antonius_conservative", players)
        self.assertEqual(players["V_Antonius_conservative"]["signals"], 7)
        self.assertEqual(
            players["V_Antonius_conservative"]["per_regime"]["neutral"]["closed_trades"],
            7,
        )

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

    def test_agent_leaderboard_includes_shadow_only_agent_labels(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        pipeline.shadow_agent_labels = ("ShadowOnlyAgent",)
        migrate_v1_regime_memory({
            "bullish": {
                "ShadowOnlyAgent": {
                    "samples": 3,
                    "wins": 2,
                    "losses": 1,
                    "pnl_pct": 0.7,
                },
            },
        }, pipeline.perf)

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer._write_leaderboards()
            with open(os.path.join(writer.output_dir, "leaderboard_agents.json"),
                      "r", encoding="utf-8") as f:
                agents = json.load(f)["agents"]
            writer.close()

        self.assertIn("V_ShadowOnlyAgent", agents)
        self.assertEqual(agents["V_ShadowOnlyAgent"]["signals"], 3)

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
        self.assertAlmostEqual(live["clean_panteon_realized_pnl_usd"], -0.10)
        self.assertAlmostEqual(live["external_realized_pnl_usd"], -0.20)
        self.assertAlmostEqual(live["panteon_owned_unrealized_pnl_usd"], 0.75)
        self.assertAlmostEqual(live["clean_panteon_unrealized_pnl_usd"], 0.75)
        self.assertAlmostEqual(live["external_unrealized_pnl_usd"], -0.25)
        self.assertAlmostEqual(live["panteon_owned_pnl_pct"], 0.65)
        self.assertAlmostEqual(live["clean_panteon_pnl_pct"], 0.65)
        self.assertEqual(live["panteon_owned_positions_count"], 1)
        self.assertEqual(live["clean_panteon_positions_count"], 1)
        self.assertEqual(live["external_positions_count"], 1)
        self.assertEqual(live["comparison_scope"], "panteon_owned")
        self.assertEqual(live["profit_accounting_scope"], "clean_panteon_owned")
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

    def test_status_reports_adopted_positions_inside_panteon_owned_scope(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("AgentA"))
        exchange = FakeExchange(name="REAL")
        pipeline = build_production_pipeline(
            registry=reg,
            exchange=exchange,
            initial_capital=100.0,
            live_execution_config=LiveExecutionConfig(
                adopt_existing_positions_enabled=True,
            ),
        )
        opened_at = datetime.now(timezone.utc)
        pipeline.executor._tracker.force_set(TrackedPosition(
            open_signal_id=0,
            sym="ETH",
            side="short",
            entry_price=50.0,
            qty=1.0,
            fee_open=0.0,
            by_player="PanteonFlashAdopted",
            by_agent="AdoptedExchangePosition",
            opened_at=opened_at,
            open_action="FUT_SHORT_FULL",
            open_regime="neutral",
        ))
        exchange._positions["ETH"] = ExchangePosition(
            sym="ETH", side="short", qty=1.0, entry=50.0, unrealized_pnl=0.42,
        )

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            writer.write(StepResult(
                bar=2,
                regime=Regime.NEUTRAL,
                leader="NoTrade",
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
        self.assertEqual(live["panteon_owned_positions_count"], 1)
        self.assertEqual(live["clean_panteon_positions_count"], 0)
        self.assertEqual(live["external_positions_count"], 0)
        self.assertEqual(live["adopted_positions_count"], 1)
        self.assertAlmostEqual(live["adopted_unrealized_pnl_usd"], 0.42)
        self.assertAlmostEqual(live["panteon_owned_total_pnl_usd"], 0.42)
        self.assertAlmostEqual(live["clean_panteon_total_pnl_usd"], 0.0)
        self.assertEqual(live["clean_real_total_trades"], 0)
        self.assertEqual(live["adopted_unresolved_trades"], 1)

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
