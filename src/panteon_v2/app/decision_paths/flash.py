"""Panteon Flash per-symbol decision path orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence

from ...attribution import DecisionStarted, LeaderSelected, SignalEmitted
from ...domain.types import MarketSnapshot, Regime
from ...execution import ExecutionStatus
from ...selection import EnsemblePlayer


log = logging.getLogger(__name__)


GENETICS_CONTRA_NO_BACKFILL_REASON = "genetics_contra_no_backfill"


def _no_flash_close_signals(*args: Any, **kwargs: Any) -> list[Any]:
    return []


def _flash_exchange_from_pipeline(pipeline: object) -> object | None:
    exchange = getattr(pipeline, "exchange", None)
    if exchange is not None:
        return exchange
    executor = getattr(pipeline, "executor", None)
    return getattr(executor, "_exchange", None)


def _flash_min_notional_by_symbol(
    pipeline: object,
    symbols: Sequence[str],
) -> Dict[str, float]:
    exchange = _flash_exchange_from_pipeline(pipeline)
    getter = getattr(exchange, "get_min_notional", None)
    if not callable(getter):
        return {}
    values: Dict[str, float] = {}
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip()
        if not symbol:
            continue
        try:
            value = float(getter(symbol) or 0.0)
        except Exception:
            log.debug("failed to read min_notional for %s", symbol, exc_info=True)
            continue
        if value > 0:
            values[symbol.upper()] = value
    return values


def _flash_account_equity_usd(pipeline: object) -> float:
    try:
        return float(getattr(pipeline, "current_balance", 0.0) or 0.0)
    except Exception:
        return 0.0


@dataclass(frozen=True)
class FlashDecisionPathCallbacks:
    """Dependencies supplied by main_loop to keep this module cycle-free."""

    fallback_candidate_safety_issue: Callable[..., Any]
    flash_real_agents: Callable[..., Any]
    flash_execution_actor_type: type
    sync_player_agents_to_real_positions: Callable[..., Any]
    flash_update_degradation_state_from_events: Callable[..., Any]
    flash_shadow_player_signals_with_position_replay: Callable[..., Any]
    flash_shadow_confirmation_scores: Callable[..., Any]
    flash_degraded_signal_keys: Callable[..., Any]
    genetics_probation_preselection_degraded_signal_keys: Callable[..., Any]
    genetics_probation_preselection_admission_signal_keys: Callable[..., Any]
    flash_degraded_actor_keys: Callable[..., Any]
    flash_degraded_open_symbols: Callable[..., Any]
    flash_promoted_signal_keys: Callable[..., Any]
    flash_open_position_sides_by_symbol: Callable[..., Any]
    flash_open_position_actor_keys_by_symbol: Callable[..., Any]
    flash_previous_actor_key_from_decision: Callable[..., Any]
    flash_register_degradation_signal_keys: Callable[..., Any]
    emit_flash_audit_events: Callable[..., Any]
    flash_stale_position_close_signals: Callable[..., Any]
    flash_stale_position_exit_max_age_bars: Callable[..., Any]
    flash_stale_position_exit_requires_loss: Callable[..., Any]
    genetics_probation_regime_exit_close_signals: Callable[..., Any]
    flash_partial_profit_lock_close_signals: Callable[..., Any]
    flash_guard_actor: Callable[..., Any]
    filter_real_signals_against_tracker: Callable[..., Any]
    genetics_probation_execution_overlay: Callable[..., Any]
    signal_context: Callable[..., Any]
    record_order_failure: Callable[..., Any]
    record_order_success: Callable[..., Any]
    record_realized_result_for_strategy: Callable[..., Any]
    kill_switch_reason: Callable[..., Any]
    flash_causal_decision_payload: Callable[..., Any]
    step_result_type: type
    flash_stop_loss_close_signals: Callable[..., Any] = _no_flash_close_signals


def run_flash_decision_path(
    pipeline: object,
    market: MarketSnapshot,
    candidates: Sequence[EnsemblePlayer],
    shadow_summary: object,
    *,
    signal_id_counter: int,
    last_qm_bar: int,
    last_regime: Optional[Regime],
    trace: str,
    decision_id: str,
    context: Dict[str, str],
    health_reason: str,
    callbacks: FlashDecisionPathCallbacks,
):
    allocator = getattr(pipeline, "flash_allocator", None)
    if allocator is None:
        raise RuntimeError("flash_enabled=True but pipeline.flash_allocator is not configured")

    try:
        pipeline._flash_candidate_safety_reasons = {}
    except Exception:
        pass
    candidates = tuple(
        candidate
        for candidate in candidates
        if not callbacks.fallback_candidate_safety_issue(pipeline, candidate, market)
    )
    agents = callbacks.flash_real_agents(pipeline, market)
    candidate_labels = tuple(dict.fromkeys(
        [str(getattr(agent, "label", "") or "") for agent in agents]
        + [str(getattr(candidate, "label", "") or "") for candidate in candidates]
    ))
    pipeline.event_log.emit(DecisionStarted(
        bar=market.bar,
        trace_id=trace,
        decision_id=decision_id,
        **context,
        candidate_labels=tuple(label for label in candidate_labels if label),
        shadow_total_signals=int(getattr(shadow_summary, "total_signals", 0) or 0),
        shadow_total_filled=int(getattr(shadow_summary, "total_filled", 0) or 0),
        shadow_total_rejected=int(getattr(shadow_summary, "total_rejected", 0) or 0),
        shadow_total_blocked=int(getattr(shadow_summary, "total_blocked", 0) or 0),
    ))

    callbacks.sync_player_agents_to_real_positions(
        callbacks.flash_execution_actor_type(
            label="Panteon_Flash",
            agents=tuple(agents),
            agent_labels=tuple(str(getattr(agent, "label", "") or "") for agent in agents),
        ),
        pipeline,
        bar_index=market.bar,
        market_symbols=market.prices.keys(),
    )
    callbacks.flash_update_degradation_state_from_events(pipeline, current_bar=market.bar)
    previous_flash_actors = dict(
        getattr(pipeline, "_flash_previous_actor_by_symbol", {}) or {}
    )
    shadow_player_signals = callbacks.flash_shadow_player_signals_with_position_replay(
        pipeline,
        market,
        candidates,
        agents=agents,
        signal_id_counter=signal_id_counter,
    )
    shadow_agent_signals = getattr(
        pipeline,
        "_current_shadow_agent_signals",
        {},
    ) or {}
    degraded_signal_keys = tuple(callbacks.flash_degraded_signal_keys(pipeline))
    degraded_signal_keys += tuple(
        callbacks.genetics_probation_preselection_degraded_signal_keys(
            pipeline,
            shadow_player_signals,
            shadow_agent_signals,
            regime=market.regime,
        ) or ()
    )
    probation_signal_keys = tuple(
        callbacks.genetics_probation_preselection_admission_signal_keys(
            pipeline,
            shadow_player_signals,
            shadow_agent_signals,
            market=market,
        ) or ()
    )
    decisions = tuple(allocator.decide(
        market,
        agents=agents,
        players=tuple(candidates),
        signal_id_start=signal_id_counter,
        actionable_labels=getattr(pipeline, "_current_actionable_player_labels", set()) or set(),
        shadow_confirmation=callbacks.flash_shadow_confirmation_scores(
            pipeline,
            market,
            agents=agents,
            players=candidates,
        ),
        shadow_player_signals=shadow_player_signals,
        shadow_agent_signals=shadow_agent_signals,
        degraded_signal_keys=degraded_signal_keys,
        degraded_actor_keys=callbacks.flash_degraded_actor_keys(pipeline, market.regime),
        degraded_open_symbols=callbacks.flash_degraded_open_symbols(pipeline, market),
        promoted_signal_keys=callbacks.flash_promoted_signal_keys(pipeline),
        probation_signal_keys=probation_signal_keys,
        previous_actor_by_symbol=previous_flash_actors,
        open_position_sides_by_symbol=callbacks.flash_open_position_sides_by_symbol(pipeline),
        open_position_actor_keys_by_symbol=callbacks.flash_open_position_actor_keys_by_symbol(pipeline),
        min_notional_by_symbol=_flash_min_notional_by_symbol(
            pipeline,
            tuple(str(symbol) for symbol in market.prices.keys()),
        ),
        account_equity_usd=_flash_account_equity_usd(pipeline),
    ))
    pipeline._flash_previous_actor_by_symbol = {
        symbol: actor_key
        for decision in decisions
        for symbol, actor_key in (
            (
                str(getattr(decision, "symbol", "") or "").upper(),
                callbacks.flash_previous_actor_key_from_decision(decision),
            ),
        )
        if symbol and actor_key and actor_key != "NoTrade"
    }
    callbacks.flash_register_degradation_signal_keys(pipeline, decisions)
    callbacks.emit_flash_audit_events(
        pipeline,
        market,
        decisions,
        decision_id=decision_id,
        trace_id=trace,
        context=context,
    )
    best_score = max(
        (float(getattr(decision, "score", 0.0) or 0.0) for decision in decisions),
        default=0.0,
    )
    pipeline.event_log.emit(LeaderSelected(
        bar=market.bar,
        trace_id=trace,
        player_label="Panteon_Flash",
        previous_label="",
        score=best_score,
        margin=0.0,
        is_urgent=False,
        reason="flash per-symbol actor selection",
        decision_id=decision_id,
        **context,
    ))

    raw_signals = [decision.signal for decision in decisions if decision.signal is not None]
    if raw_signals:
        signal_id_counter = max(signal.id for signal in raw_signals) + 1
    stop_loss_close_signals = callbacks.flash_stop_loss_close_signals(
        pipeline,
        market,
        signal_id_start=signal_id_counter,
    )
    if stop_loss_close_signals:
        signal_id_counter = max(signal.id for signal in stop_loss_close_signals) + 1
        raw_signals = stop_loss_close_signals + raw_signals
    stop_loss_symbols = {
        str(getattr(signal, "sym", "") or "").upper()
        for signal in stop_loss_close_signals
    }
    stale_close_signals = callbacks.flash_stale_position_close_signals(
        pipeline,
        market,
        signal_id_start=signal_id_counter,
        max_age_bars=callbacks.flash_stale_position_exit_max_age_bars(pipeline),
        require_nonpositive_unrealized=callbacks.flash_stale_position_exit_requires_loss(pipeline),
        excluded_symbols=stop_loss_symbols,
    )
    if stale_close_signals:
        signal_id_counter = max(signal.id for signal in stale_close_signals) + 1
        raw_signals = stale_close_signals + raw_signals
    regime_exit_close_signals = callbacks.genetics_probation_regime_exit_close_signals(
        pipeline,
        market,
        signal_id_start=signal_id_counter,
        excluded_symbols=stop_loss_symbols | {
            str(getattr(signal, "sym", "") or "").upper()
            for signal in stale_close_signals
        },
    )
    if regime_exit_close_signals:
        signal_id_counter = max(signal.id for signal in regime_exit_close_signals) + 1
        raw_signals = regime_exit_close_signals + raw_signals
    partial_profit_lock_signals = callbacks.flash_partial_profit_lock_close_signals(
        pipeline,
        market,
        signal_id_start=signal_id_counter,
        excluded_symbols=stop_loss_symbols | {
            str(getattr(signal, "sym", "") or "").upper()
            for signal in tuple(stale_close_signals) + tuple(regime_exit_close_signals)
        },
    )
    if partial_profit_lock_signals:
        signal_id_counter = max(signal.id for signal in partial_profit_lock_signals) + 1
        raw_signals = partial_profit_lock_signals + raw_signals
    raw_signal_count = len(raw_signals)
    guard_actor = callbacks.flash_guard_actor(decisions, agents=agents, players=candidates)
    reserved_new_opens = _reserved_new_opens_from_flash_decisions(decisions)
    max_new_opens_per_bar = getattr(
        getattr(pipeline, "live_execution", None),
        "max_new_opens_per_bar",
        1,
    )
    signal_guard = callbacks.filter_real_signals_against_tracker(
        raw_signals,
        player=guard_actor,
        pipeline=pipeline,
        bar_index=market.bar,
        max_new_opens_per_bar=None,
        max_open_positions=getattr(
            getattr(pipeline, "risk_config", None),
            "max_open_positions",
            None,
        ),
        reserved_new_opens=reserved_new_opens,
    )
    signal_guard = callbacks.genetics_probation_execution_overlay(
        pipeline,
        market,
        guard_actor,
        signal_guard,
    )
    if signal_guard.signals:
        rate_limited_signal_guard = callbacks.filter_real_signals_against_tracker(
            signal_guard.signals,
            player=guard_actor,
            pipeline=pipeline,
            bar_index=market.bar,
            max_new_opens_per_bar=max_new_opens_per_bar,
            max_open_positions=None,
            reserved_new_opens=reserved_new_opens,
        )
        signal_guard = _merge_signal_guard_results(signal_guard, rate_limited_signal_guard)
    signals = signal_guard.signals
    for sig in signals:
        signal_context = callbacks.signal_context(
            context,
            decision_id=decision_id,
            signal=sig,
        )
        if sig.action.is_open and health_reason:
            signal_context["exchange_health_reason"] = health_reason
        pipeline.event_log.emit(SignalEmitted(
            bar=market.bar,
            trace_id=trace,
            signal=sig,
            **signal_context,
        ))
        getattr(pipeline, "real_perf", pipeline.perf).record_signal(sig)

    n_filled = n_rejected = n_blocked = 0
    blocked_reasons: Dict[str, int] = {}
    for sig in signals:
        try:
            set_event_context = getattr(pipeline.executor, "set_event_context", None)
            if callable(set_event_context):
                signal_context = callbacks.signal_context(
                    context,
                    decision_id=decision_id,
                    signal=sig,
                )
                if sig.action.is_open and health_reason:
                    signal_context["exchange_health_reason"] = health_reason
                set_event_context(signal_context)
            res = pipeline.executor.execute(
                sig,
                balance_usd=pipeline.current_balance,
            )
        except Exception:
            log.exception("execute failed for signal %d", sig.id)
            callbacks.record_order_failure(pipeline, "execute exception", signal=sig)
            n_rejected += 1
            continue
        if res.status == ExecutionStatus.FILLED:
            n_filled += 1
            callbacks.record_order_success(pipeline, sig, res)
            callbacks.record_realized_result_for_strategy(pipeline, res)
        elif res.status == ExecutionStatus.REJECTED:
            callbacks.record_order_failure(pipeline, res.reason or "rejected", signal=sig)
            n_rejected += 1
        elif res.status == ExecutionStatus.PENDING:
            callbacks.record_order_failure(pipeline, res.reason or "pending", signal=sig)
            n_rejected += 1
        elif res.status == ExecutionStatus.BLOCKED:
            n_blocked += 1
            reason = res.reason or "blocked"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        if callbacks.kill_switch_reason(pipeline):
            break

    callbacks.flash_update_degradation_state_from_events(pipeline, current_bar=market.bar)
    selected_by_symbol = {
        str(getattr(decision, "symbol", "") or ""): str(getattr(decision, "selected_actor", "") or "")
        for decision in decisions
    }
    kill_reason_after_execution = callbacks.kill_switch_reason(pipeline)
    return (
        callbacks.step_result_type(
            bar=market.bar,
            regime=market.regime,
            leader="Panteon_Flash",
            leader_changed=False,
            n_signals=len(signals),
            n_filled=n_filled,
            n_rejected=n_rejected,
            n_blocked=n_blocked,
            n_shadow_signals=int(getattr(shadow_summary, "total_signals", 0) or 0),
            n_shadow_filled=int(getattr(shadow_summary, "total_filled", 0) or 0),
            n_shadow_rejected=int(getattr(shadow_summary, "total_rejected", 0) or 0),
            n_shadow_blocked=int(getattr(shadow_summary, "total_blocked", 0) or 0),
            n_shadow_actors=int(getattr(shadow_summary, "actors", 0) or 0),
            n_filtered_real_signals=signal_guard.filtered,
            n_stale_close_signals=signal_guard.stale_closes,
            n_duplicate_open_signals=signal_guard.duplicate_opens,
            n_rate_limited_open_signals=signal_guard.rate_limited_opens,
            n_max_position_saturated_open_signals=signal_guard.max_position_saturated_opens,
            n_external_position_signals=signal_guard.external_position_signals,
            n_raw_signals=raw_signal_count,
            leader_vote_errors=0,
            leader_agent_labels=guard_actor.agent_labels,
            selected_actors_by_symbol=selected_by_symbol,
            signal_filter_details=tuple(signal_guard.details),
            selected_leader="Panteon_Flash",
            executed_leader="Panteon_Flash",
            blocked_reasons=blocked_reasons,
            causal_decision=callbacks.flash_causal_decision_payload(
                market=market,
                decisions=decisions,
                raw_signals=raw_signals,
                executable_signals=signals,
                signal_guard=signal_guard,
                n_filled=n_filled,
                n_rejected=n_rejected,
                n_blocked=n_blocked,
                blocked_reasons=blocked_reasons,
                pipeline=pipeline,
            ),
            error=(
                f"kill switch active: {kill_reason_after_execution}"
                if kill_reason_after_execution else None
            ),
        ),
        signal_id_counter,
        last_qm_bar,
        last_regime,
    )


def _merge_signal_guard_results(base, overlay):
    return replace(
        base,
        signals=list(getattr(overlay, "signals", ()) or ()),
        filtered=int(getattr(base, "filtered", 0) or 0)
        + int(getattr(overlay, "filtered", 0) or 0),
        stale_closes=int(getattr(base, "stale_closes", 0) or 0)
        + int(getattr(overlay, "stale_closes", 0) or 0),
        foreign_closes=int(getattr(base, "foreign_closes", 0) or 0)
        + int(getattr(overlay, "foreign_closes", 0) or 0),
        duplicate_opens=int(getattr(base, "duplicate_opens", 0) or 0)
        + int(getattr(overlay, "duplicate_opens", 0) or 0),
        rate_limited_opens=int(getattr(base, "rate_limited_opens", 0) or 0)
        + int(getattr(overlay, "rate_limited_opens", 0) or 0),
        max_position_saturated_opens=int(
            getattr(base, "max_position_saturated_opens", 0) or 0
        )
        + int(getattr(overlay, "max_position_saturated_opens", 0) or 0),
        external_position_signals=int(getattr(base, "external_position_signals", 0) or 0)
        + int(getattr(overlay, "external_position_signals", 0) or 0),
        details=list(getattr(base, "details", ()) or ())
        + list(getattr(overlay, "details", ()) or ()),
    )


def _reserved_new_opens_from_flash_decisions(decisions: Sequence[object]) -> int:
    reserved = 0
    for decision in decisions or ():
        reasons = set(getattr(decision, "selected_reasons", ()) or ())
        reason = str(getattr(decision, "reason", "") or "")
        if (
            GENETICS_CONTRA_NO_BACKFILL_REASON not in reasons
            and reason != GENETICS_CONTRA_NO_BACKFILL_REASON
        ):
            continue
        original_label = str(getattr(decision, "original_selected_actor", "") or "")
        original_type = str(getattr(decision, "original_actor_type", "") or "")
        original_action = None
        for candidate in getattr(decision, "candidates", ()) or ():
            if (
                str(getattr(candidate, "label", "") or "") == original_label
                and str(getattr(candidate, "actor_type", "") or "") == original_type
            ):
                original_action = getattr(candidate, "action", None)
                break
        if bool(getattr(original_action, "is_open", False)):
            reserved += 1
    return reserved
