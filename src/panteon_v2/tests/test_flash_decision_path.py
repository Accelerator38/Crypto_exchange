"""Tests for Flash decision path wiring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from panteon_v2.app.decision_paths.flash import (
    FlashDecisionPathCallbacks,
    run_flash_decision_path,
)
from panteon_v2.app.live_state import RealSignalGuardResult
from panteon_v2.domain.types import Action, Signal
from panteon_v2.shadow.adapters import make_market_snapshot


class _CapturingAllocator:
    def __init__(self) -> None:
        self.degraded_signal_keys = ()

    def decide(self, *args, **kwargs):
        self.degraded_signal_keys = tuple(kwargs.get("degraded_signal_keys", ()))
        return ()


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


class TestFlashDecisionPath(unittest.TestCase):
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
            flash_degraded_actor_keys=lambda *args, **kwargs: (),
            flash_degraded_open_symbols=lambda *args, **kwargs: (),
            flash_promoted_signal_keys=lambda *args, **kwargs: (),
            flash_open_position_sides_by_symbol=lambda *args, **kwargs: {},
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
            flash_degraded_actor_keys=lambda *args, **kwargs: (),
            flash_degraded_open_symbols=lambda *args, **kwargs: (),
            flash_promoted_signal_keys=lambda *args, **kwargs: (),
            flash_open_position_sides_by_symbol=lambda *args, **kwargs: {},
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


if __name__ == "__main__":
    unittest.main()
