"""Tests for Flash decision path wiring."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from panteon_v2.app.decision_paths.flash import (
    FlashDecisionPathCallbacks,
    run_flash_decision_path,
)
from panteon_v2.app.live_state import RealSignalGuardResult
from panteon_v2.domain.types import Action, Signal
from panteon_v2.execution import ExecutionResult, ExecutionStatus
from panteon_v2.selection import FlashDecision
from panteon_v2.shadow.adapters import make_market_snapshot


class _CapturingAllocator:
    def __init__(self) -> None:
        self.degraded_signal_keys = ()
        self.probation_signal_keys = ()
        self.kwargs = {}

    def decide(self, *args, **kwargs):
        self.kwargs = dict(kwargs)
        self.degraded_signal_keys = tuple(kwargs.get("degraded_signal_keys", ()))
        self.probation_signal_keys = tuple(kwargs.get("probation_signal_keys", ()))
        return ()


class _FixedAllocator:
    def __init__(self, decisions) -> None:
        self.decisions = tuple(decisions)

    def decide(self, *args, **kwargs):
        return self.decisions


class _EventLog:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class _FlashActor:
    def __init__(self, label, agents=(), agent_labels=()) -> None:
        self.label = label
        self.agents = tuple(agents)
        self.agent_labels = tuple(agent_labels)


class _StepResult:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _PerfRecorder:
    def __init__(self) -> None:
        self.signals = []

    def record_signal(self, signal) -> None:
        self.signals.append(signal)


class _Executor:
    def __init__(self) -> None:
        self.signals = []

    def execute(self, signal, *, balance_usd):
        self.signals.append(signal)
        return ExecutionResult(status=ExecutionStatus.FILLED, signal=signal)


class TestFlashDecisionPath(unittest.TestCase):
    def test_exchange_min_notional_and_balance_are_passed_to_allocator(self):
        allocator = _CapturingAllocator()

        class Exchange:
            def get_min_notional(self, symbol):
                return 5.0 if symbol == "BTC/USDT" else 0.0

        pipeline = SimpleNamespace(
            flash_allocator=allocator,
            event_log=_EventLog(),
            _current_actionable_player_labels=set(),
            _current_shadow_agent_signals={},
            live_execution=SimpleNamespace(max_new_opens_per_bar=1),
            risk_config=SimpleNamespace(max_open_positions=None),
            executor=SimpleNamespace(_exchange=Exchange()),
            current_balance=1234.0,
        )
        callbacks = FlashDecisionPathCallbacks(
            fallback_candidate_safety_issue=lambda *args, **kwargs: False,
            flash_real_agents=lambda *args, **kwargs: (),
            flash_execution_actor_type=_FlashActor,
            sync_player_agents_to_real_positions=lambda *args, **kwargs: None,
            flash_update_degradation_state_from_events=lambda *args, **kwargs: None,
            flash_shadow_player_signals_with_position_replay=(
                lambda *args, **kwargs: {}
            ),
            flash_shadow_confirmation_scores=lambda *args, **kwargs: {},
            flash_degraded_signal_keys=lambda *args, **kwargs: (),
            genetics_probation_preselection_degraded_signal_keys=(
                lambda *args, **kwargs: ()
            ),
            genetics_probation_preselection_admission_signal_keys=(
                lambda *args, **kwargs: ()
            ),
            flash_degraded_actor_keys=lambda *args, **kwargs: (),
            flash_degraded_open_symbols=lambda *args, **kwargs: (),
            flash_promoted_signal_keys=lambda *args, **kwargs: (),
            flash_open_position_sides_by_symbol=lambda *args, **kwargs: {},
            flash_open_position_actor_keys_by_symbol=lambda *args, **kwargs: {},
            flash_previous_actor_key_from_decision=lambda *args, **kwargs: "",
            flash_register_degradation_signal_keys=lambda *args, **kwargs: None,
            emit_flash_audit_events=lambda *args, **kwargs: None,
            flash_stale_position_close_signals=lambda *args, **kwargs: [],
            flash_stale_position_exit_max_age_bars=lambda *args, **kwargs: 0,
            flash_stale_position_exit_requires_loss=lambda *args, **kwargs: False,
            genetics_probation_regime_exit_close_signals=lambda *args, **kwargs: [],
            flash_partial_profit_lock_close_signals=lambda *args, **kwargs: [],
            flash_guard_actor=lambda *args, **kwargs: _FlashActor(
                label="Panteon_Flash",
                agents=(),
                agent_labels=(),
            ),
            filter_real_signals_against_tracker=(
                lambda *args, **kwargs: RealSignalGuardResult(signals=[])
            ),
            genetics_probation_execution_overlay=lambda *args, **kwargs: args[-1],
            signal_context=lambda *args, **kwargs: {},
            record_order_failure=lambda *args, **kwargs: None,
            record_order_success=lambda *args, **kwargs: None,
            record_realized_result_for_strategy=lambda *args, **kwargs: None,
            kill_switch_reason=lambda *args, **kwargs: "",
            flash_causal_decision_payload=lambda *args, **kwargs: {},
            step_result_type=_StepResult,
        )

        run_flash_decision_path(
            pipeline,
            make_market_snapshot(
                bar=1,
                prices={"BTC/USDT": 100.0, "ETH/USDT": 2000.0},
                regime="bearish",
            ),
            (),
            SimpleNamespace(
                total_signals=0,
                total_filled=0,
                total_rejected=0,
                total_blocked=0,
                actors=0,
            ),
            signal_id_counter=1,
            last_qm_bar=0,
            last_regime=None,
            trace="trace",
            decision_id="decision",
            context={},
            health_reason="",
            callbacks=callbacks,
        )

        self.assertEqual(
            allocator.kwargs["min_notional_by_symbol"],
            {"BTC/USDT": 5.0},
        )
        self.assertEqual(allocator.kwargs["account_equity_usd"], 1234.0)

    def test_preselection_genetics_probation_degraded_keys_are_passed_to_allocator(self):
        allocator = _CapturingAllocator()
        pipeline = SimpleNamespace(
            flash_allocator=allocator,
            event_log=_EventLog(),
            _current_actionable_player_labels=set(),
            _current_shadow_agent_signals={},
            live_execution=SimpleNamespace(max_new_opens_per_bar=1),
            risk_config=SimpleNamespace(max_open_positions=None),
        )
        base_key = "agent:Existing|BTC/USDT|FUT_LONG_FULL"
        genetics_key = "agent:GeneticsNeutral|ATOM/USDT|FUT_SHORT_FULL"
        callbacks = FlashDecisionPathCallbacks(
            fallback_candidate_safety_issue=lambda *args, **kwargs: False,
            flash_real_agents=lambda *args, **kwargs: (),
            flash_execution_actor_type=_FlashActor,
            sync_player_agents_to_real_positions=lambda *args, **kwargs: None,
            flash_update_degradation_state_from_events=lambda *args, **kwargs: None,
            flash_shadow_player_signals_with_position_replay=(
                lambda *args, **kwargs: {}
            ),
            flash_shadow_confirmation_scores=lambda *args, **kwargs: {},
            flash_degraded_signal_keys=lambda *args, **kwargs: (base_key,),
            genetics_probation_preselection_degraded_signal_keys=(
                lambda *args, **kwargs: (genetics_key,)
            ),
            genetics_probation_preselection_admission_signal_keys=(
                lambda *args, **kwargs: ("agent:GeneticsCore|BTC/USDT|FUT_SHORT_FULL",)
            ),
            flash_degraded_actor_keys=lambda *args, **kwargs: (),
            flash_degraded_open_symbols=lambda *args, **kwargs: (),
            flash_promoted_signal_keys=lambda *args, **kwargs: (),
            flash_open_position_sides_by_symbol=lambda *args, **kwargs: {},
            flash_open_position_actor_keys_by_symbol=lambda *args, **kwargs: {},
            flash_previous_actor_key_from_decision=lambda *args, **kwargs: "",
            flash_register_degradation_signal_keys=lambda *args, **kwargs: None,
            emit_flash_audit_events=lambda *args, **kwargs: None,
            flash_stale_position_close_signals=lambda *args, **kwargs: [],
            flash_stale_position_exit_max_age_bars=lambda *args, **kwargs: 0,
            flash_stale_position_exit_requires_loss=lambda *args, **kwargs: False,
            genetics_probation_regime_exit_close_signals=lambda *args, **kwargs: [],
            flash_partial_profit_lock_close_signals=lambda *args, **kwargs: [],
            flash_guard_actor=lambda *args, **kwargs: _FlashActor(
                label="Panteon_Flash",
                agents=(),
                agent_labels=(),
            ),
            filter_real_signals_against_tracker=(
                lambda *args, **kwargs: RealSignalGuardResult(signals=[])
            ),
            genetics_probation_execution_overlay=lambda *args, **kwargs: args[-1],
            signal_context=lambda *args, **kwargs: {},
            record_order_failure=lambda *args, **kwargs: None,
            record_order_success=lambda *args, **kwargs: None,
            record_realized_result_for_strategy=lambda *args, **kwargs: None,
            kill_switch_reason=lambda *args, **kwargs: "",
            flash_causal_decision_payload=lambda *args, **kwargs: {},
            step_result_type=_StepResult,
        )

        run_flash_decision_path(
            pipeline,
            make_market_snapshot(
                bar=1,
                prices={"BTC/USDT": 100.0},
                regime="bearish",
            ),
            (),
            SimpleNamespace(
                total_signals=0,
                total_filled=0,
                total_rejected=0,
                total_blocked=0,
                actors=0,
            ),
            signal_id_counter=1,
            last_qm_bar=0,
            last_regime=None,
            trace="trace",
            decision_id="decision",
            context={},
            health_reason="",
            callbacks=callbacks,
        )

        self.assertIn(base_key, allocator.degraded_signal_keys)
        self.assertIn(genetics_key, allocator.degraded_signal_keys)
        self.assertIn(
            "agent:GeneticsCore|BTC/USDT|FUT_SHORT_FULL",
            allocator.probation_signal_keys,
        )

    def test_genetics_probation_regime_exit_signals_are_guarded_before_opens(self):
        allocator = _CapturingAllocator()
        pipeline = SimpleNamespace(
            flash_allocator=allocator,
            event_log=_EventLog(),
            _current_actionable_player_labels=set(),
            _current_shadow_agent_signals={},
            live_execution=SimpleNamespace(max_new_opens_per_bar=1),
            risk_config=SimpleNamespace(max_open_positions=None),
        )
        close_signal = Signal(
            id=50,
            bar=10,
            sym="ETC/USDT",
            action=Action.FUT_CLOSE_ALL,
            price=21.0,
            regime=make_market_snapshot(
                bar=10,
                prices={"ETC/USDT": 21.0},
                regime="neutral",
            ).regime,
            by_player="Panteon_Flash",
            by_agent="GeneticsProbationRegimeExit",
        )
        captured = {}

        def capture_guard(raw_signals, *args, **kwargs):
            captured["raw_signals"] = tuple(raw_signals)
            return RealSignalGuardResult(signals=[])

        callbacks = FlashDecisionPathCallbacks(
            fallback_candidate_safety_issue=lambda *args, **kwargs: False,
            flash_real_agents=lambda *args, **kwargs: (),
            flash_execution_actor_type=_FlashActor,
            sync_player_agents_to_real_positions=lambda *args, **kwargs: None,
            flash_update_degradation_state_from_events=lambda *args, **kwargs: None,
            flash_shadow_player_signals_with_position_replay=(
                lambda *args, **kwargs: {}
            ),
            flash_shadow_confirmation_scores=lambda *args, **kwargs: {},
            flash_degraded_signal_keys=lambda *args, **kwargs: (),
            genetics_probation_preselection_degraded_signal_keys=(
                lambda *args, **kwargs: ()
            ),
            genetics_probation_preselection_admission_signal_keys=(
                lambda *args, **kwargs: ()
            ),
            flash_degraded_actor_keys=lambda *args, **kwargs: (),
            flash_degraded_open_symbols=lambda *args, **kwargs: (),
            flash_promoted_signal_keys=lambda *args, **kwargs: (),
            flash_open_position_sides_by_symbol=lambda *args, **kwargs: {},
            flash_open_position_actor_keys_by_symbol=lambda *args, **kwargs: {},
            flash_previous_actor_key_from_decision=lambda *args, **kwargs: "",
            flash_register_degradation_signal_keys=lambda *args, **kwargs: None,
            emit_flash_audit_events=lambda *args, **kwargs: None,
            flash_stale_position_close_signals=lambda *args, **kwargs: [],
            flash_stale_position_exit_max_age_bars=lambda *args, **kwargs: 0,
            flash_stale_position_exit_requires_loss=lambda *args, **kwargs: False,
            genetics_probation_regime_exit_close_signals=(
                lambda *args, **kwargs: [close_signal]
            ),
            flash_partial_profit_lock_close_signals=lambda *args, **kwargs: [],
            flash_guard_actor=lambda *args, **kwargs: _FlashActor(
                label="Panteon_Flash",
                agents=(),
                agent_labels=(),
            ),
            filter_real_signals_against_tracker=capture_guard,
            genetics_probation_execution_overlay=lambda *args, **kwargs: args[-1],
            signal_context=lambda *args, **kwargs: {},
            record_order_failure=lambda *args, **kwargs: None,
            record_order_success=lambda *args, **kwargs: None,
            record_realized_result_for_strategy=lambda *args, **kwargs: None,
            kill_switch_reason=lambda *args, **kwargs: "",
            flash_causal_decision_payload=lambda *args, **kwargs: {},
            step_result_type=_StepResult,
        )

        step, *_ = run_flash_decision_path(
            pipeline,
            make_market_snapshot(
                bar=10,
                prices={"ETC/USDT": 21.0},
                regime="neutral",
            ),
            (),
            SimpleNamespace(
                total_signals=0,
                total_filled=0,
                total_rejected=0,
                total_blocked=0,
                actors=0,
            ),
            signal_id_counter=1,
            last_qm_bar=0,
            last_regime=None,
            trace="trace",
            decision_id="decision",
            context={},
            health_reason="",
            callbacks=callbacks,
        )

        self.assertEqual(step.n_raw_signals, 1)
        self.assertEqual(captured["raw_signals"][0].by_agent, "GeneticsProbationRegimeExit")

    def test_genetics_probation_runs_before_flash_open_rate_limit(self):
        market = make_market_snapshot(
            bar=6434,
            prices={"BNB": 600.0, "ETH": 3000.0, "TRX": 0.28},
            regime="bearish",
        )

        def open_signal(signal_id, sym):
            return Signal(
                id=signal_id,
                bar=6434,
                sym=sym,
                action=Action.FUT_SHORT_FULL,
                price=float(market.prices[sym]),
                regime=market.regime,
                by_player="DefaultEnsemble",
                by_agent="GeneticsCore",
            )

        decisions = tuple(
            FlashDecision(
                symbol=signal.sym,
                selected_actor="DefaultEnsemble",
                actor_type="ensemble",
                score=1.0,
                action=signal.action,
                reason="selected",
                signal=signal,
                candidates=(),
            )
            for signal in (
                open_signal(1, "BNB"),
                open_signal(2, "ETH"),
                open_signal(3, "TRX"),
            )
        )
        allocator = _FixedAllocator(decisions)
        executor = _Executor()
        perf = _PerfRecorder()
        pipeline = SimpleNamespace(
            flash_allocator=allocator,
            event_log=_EventLog(),
            _current_actionable_player_labels=set(),
            _current_shadow_agent_signals={},
            live_execution=SimpleNamespace(max_new_opens_per_bar=1),
            risk_config=SimpleNamespace(max_open_positions=None),
            executor=executor,
            current_balance=1000.0,
            perf=perf,
            real_perf=perf,
        )
        guard_calls = []

        def rate_limit_guard(raw_signals, *args, **kwargs):
            max_new_opens_per_bar = kwargs.get("max_new_opens_per_bar")
            reserved_new_opens = max(0, int(kwargs.get("reserved_new_opens", 0) or 0))
            kept = []
            rate_limited = 0
            details = []
            kept_open_count = 0
            for signal in raw_signals:
                if (
                    signal.action.is_open
                    and max_new_opens_per_bar is not None
                    and reserved_new_opens + kept_open_count
                    >= int(max_new_opens_per_bar)
                ):
                    rate_limited += 1
                    details.append(
                        f"rate_limited_open:{signal.sym}:{signal.by_agent or '-'}"
                    )
                    continue
                kept.append(signal)
                if signal.action.is_open:
                    kept_open_count += 1
            guard_calls.append((
                max_new_opens_per_bar,
                tuple(signal.sym for signal in raw_signals),
                tuple(signal.sym for signal in kept),
            ))
            return RealSignalGuardResult(
                signals=kept,
                filtered=rate_limited,
                rate_limited_opens=rate_limited,
                details=details,
            )

        def probation_overlay(pipeline, market, leader, signal_guard):
            kept = [
                signal for signal in signal_guard.signals
                if str(signal.sym).upper() != "BNB"
            ]
            blocked = len(signal_guard.signals) - len(kept)
            details = list(signal_guard.details)
            if blocked:
                details.append("genetics_signal_key_blocked:BNB:GeneticsCore")
            return replace(
                signal_guard,
                signals=kept,
                filtered=signal_guard.filtered + blocked,
                details=details,
            )

        callbacks = FlashDecisionPathCallbacks(
            fallback_candidate_safety_issue=lambda *args, **kwargs: False,
            flash_real_agents=lambda *args, **kwargs: (),
            flash_execution_actor_type=_FlashActor,
            sync_player_agents_to_real_positions=lambda *args, **kwargs: None,
            flash_update_degradation_state_from_events=lambda *args, **kwargs: None,
            flash_shadow_player_signals_with_position_replay=(
                lambda *args, **kwargs: {}
            ),
            flash_shadow_confirmation_scores=lambda *args, **kwargs: {},
            flash_degraded_signal_keys=lambda *args, **kwargs: (),
            genetics_probation_preselection_degraded_signal_keys=(
                lambda *args, **kwargs: ()
            ),
            genetics_probation_preselection_admission_signal_keys=(
                lambda *args, **kwargs: ()
            ),
            flash_degraded_actor_keys=lambda *args, **kwargs: (),
            flash_degraded_open_symbols=lambda *args, **kwargs: (),
            flash_promoted_signal_keys=lambda *args, **kwargs: (),
            flash_open_position_sides_by_symbol=lambda *args, **kwargs: {},
            flash_open_position_actor_keys_by_symbol=lambda *args, **kwargs: {},
            flash_previous_actor_key_from_decision=(
                lambda decision: f"{decision.actor_type}:{decision.selected_actor}"
            ),
            flash_register_degradation_signal_keys=lambda *args, **kwargs: None,
            emit_flash_audit_events=lambda *args, **kwargs: None,
            flash_stale_position_close_signals=lambda *args, **kwargs: [],
            flash_stale_position_exit_max_age_bars=lambda *args, **kwargs: 0,
            flash_stale_position_exit_requires_loss=lambda *args, **kwargs: False,
            genetics_probation_regime_exit_close_signals=lambda *args, **kwargs: [],
            flash_partial_profit_lock_close_signals=lambda *args, **kwargs: [],
            flash_guard_actor=lambda *args, **kwargs: _FlashActor(
                label="Panteon_Flash",
                agents=(),
                agent_labels=(),
            ),
            filter_real_signals_against_tracker=rate_limit_guard,
            genetics_probation_execution_overlay=probation_overlay,
            signal_context=lambda *args, **kwargs: {},
            record_order_failure=lambda *args, **kwargs: None,
            record_order_success=lambda *args, **kwargs: None,
            record_realized_result_for_strategy=lambda *args, **kwargs: None,
            kill_switch_reason=lambda *args, **kwargs: "",
            flash_causal_decision_payload=lambda *args, **kwargs: {},
            step_result_type=_StepResult,
        )

        step, *_ = run_flash_decision_path(
            pipeline,
            market,
            (),
            SimpleNamespace(
                total_signals=0,
                total_filled=0,
                total_rejected=0,
                total_blocked=0,
                actors=0,
            ),
            signal_id_counter=1,
            last_qm_bar=0,
            last_regime=None,
            trace="trace",
            decision_id="decision",
            context={},
            health_reason="",
            callbacks=callbacks,
        )

        self.assertEqual(step.n_signals, 1)
        self.assertEqual([signal.sym for signal in executor.signals], ["ETH"])
        self.assertEqual(guard_calls[0][0], None)
        self.assertEqual(guard_calls[1][0], 1)
        self.assertIn(
            "genetics_signal_key_blocked:BNB:GeneticsCore",
            step.signal_filter_details,
        )
        self.assertIn(
            "rate_limited_open:TRX:GeneticsCore",
            step.signal_filter_details,
        )
        self.assertNotIn(
            "rate_limited_open:ETH:GeneticsCore",
            step.signal_filter_details,
        )


if __name__ == "__main__":
    unittest.main()
