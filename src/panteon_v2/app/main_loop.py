"""Production main loop — обновляет ProductionPipeline на каждом баре.

Это «финальный шаг» Phase 9. Использует те же абстракции, что и
ShadowRunner, но работает с реальным Exchange-адаптером.

Главные гарантии:
  • Один bar = один пересчёт state. Никаких race conditions.
  • Все decisions эмиттятся в EventLog → можно reconstruct позже.
  • Ошибки exchange/feed → REJECTED, не crash main loop.
  • После каждого bar — AttributionLedger.replay → актуальная картина.

Использование:
    pipeline = build_production_pipeline(...)
    feed = MyExchangeFeed(...)
    main_loop(pipeline, feed, on_step=my_logging_callback)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..attribution import (
    AgentVoteFailed,
    BarStarted,
    CandidateRejected,
    CandidateScored,
    DecisionStarted,
    LeaderSelected,
    PlayerVoteFailed,
    QuarantineRecomputed,
    RegimeDetected,
    SignalEmitted,
    SwitchGateEvaluated,
)
from ..domain.types import Action, MarketSnapshot, Regime, Signal
from ..execution import ExecutionResult, ExecutionStatus
from ..selection import EnsemblePlayer, SwitchDecision, ThresholdProfile, WeightedConsensus
from ..shadow.feed import MarketFeed
from .bootstrap import ProductionPipeline
from .live_state import (
    filter_real_signals_against_tracker,
    is_external_position,
    reconcile_tracker_with_exchange,
    sync_player_agents_to_real_positions,
    tracker_positions_for_agent_sync,
)


log = logging.getLogger(__name__)
_SOLO_AGENT_CANDIDATE_LIMIT = 3


def sync_pipeline_balance(pipeline: ProductionPipeline) -> Optional[float]:
    """Best-effort sync of v2 balance from the live exchange adapter."""
    try:
        exchange = getattr(pipeline.executor, "_exchange", None)
        snapshot_getter = getattr(exchange, "get_account_snapshot", None)
        if callable(snapshot_getter):
            snapshot = _normalize_account_snapshot(snapshot_getter())
            balance = _snapshot_balance(snapshot)
            if balance and balance > 0:
                pipeline.account_snapshot = snapshot
                pipeline.current_balance = balance
                return balance

        getter = getattr(exchange, "get_account_equity", None)
        if not callable(getter):
            return None
        equity = float(getter() or 0.0)
        if equity <= 0:
            return None
        pipeline.account_snapshot = {
            "current_balance": equity,
            "futures_equity": equity,
            "available_balance": equity,
            "spot_assets": 0.0,
            "total_assets": equity,
            "unrealized_pnl": 0.0,
        }
        pipeline.current_balance = equity
        return equity
    except Exception:
        log.debug("live balance sync failed", exc_info=True)
        return None


def _normalize_account_snapshot(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    aliases = {
        "current_balance": ("current_balance", "balance", "primary_capital"),
        "futures_equity": ("futures_equity", "equity", "futures_equity_usd"),
        "available_balance": ("available_balance", "available", "available_balance_usd"),
        "spot_assets": ("spot_assets", "spot_value", "spot_assets_usd"),
        "total_assets": ("total_assets", "total_equity", "total_assets_usd"),
        "unrealized_pnl": ("unrealized_pnl", "unrealized", "unrealized_pnl_usd"),
    }
    for canonical, keys in aliases.items():
        for key in keys:
            try:
                value = raw.get(key)
                if value is not None and value != "":
                    out[canonical] = float(value)
                    break
            except (TypeError, ValueError):
                continue
    if "futures_equity" not in out and "current_balance" in out:
        out["futures_equity"] = out["current_balance"]
    if "current_balance" not in out and "futures_equity" in out:
        out["current_balance"] = out["futures_equity"]
    if "spot_assets" not in out:
        out["spot_assets"] = 0.0
    if "total_assets" not in out:
        out["total_assets"] = out.get("futures_equity", 0.0) + out.get("spot_assets", 0.0)
    if "available_balance" not in out:
        out["available_balance"] = out.get("current_balance", 0.0)
    if "unrealized_pnl" not in out:
        out["unrealized_pnl"] = 0.0
    data_health = raw.get("data_health")
    if isinstance(data_health, dict):
        out["data_health"] = dict(data_health)
    return out


def _snapshot_balance(snapshot: Dict[str, float]) -> Optional[float]:
    for key in ("current_balance", "futures_equity", "total_assets"):
        value = float(snapshot.get(key, 0.0) or 0.0)
        if value > 0:
            return value
    return None


def _decision_context(pipeline: ProductionPipeline, market: MarketSnapshot) -> Dict[str, str]:
    return {
        "exchange": str(getattr(pipeline, "exchange_name", "") or ""),
        "symbol": ",".join(sorted(str(sym) for sym in market.prices.keys())),
        "timeframe": str(getattr(pipeline, "timeframe", "") or ""),
        "mode": str(getattr(pipeline, "mode", "") or ""),
        "run_id": str(getattr(pipeline, "run_id", "") or ""),
        "session_id": str(getattr(pipeline, "session_id", "") or ""),
    }


def _signal_context(context: Dict[str, str], *, decision_id: str, signal: Signal) -> Dict[str, str]:
    event_context = dict(context)
    event_context["decision_id"] = str(decision_id or "")
    event_context["symbol"] = str(signal.sym)
    return event_context


def _ensure_degradation_baseline(pipeline: ProductionPipeline) -> None:
    gate = getattr(pipeline, "degradation_gate", None)
    if gate is None or getattr(gate, "baseline_captured", False):
        return
    try:
        gate.capture_baseline(
            pipeline.perf,
            labels=pipeline.registry.all_labels(),
        )
    except Exception:
        log.exception("degradation baseline capture failed")


def _apply_degradation_gate(
    pipeline: ProductionPipeline,
    *,
    bar: int,
    trace_id: str,
):
    gate = getattr(pipeline, "degradation_gate", None)
    if gate is None:
        return None
    try:
        result = gate.apply(
            pipeline.perf,
            pipeline.qm,
            labels=pipeline.registry.all_labels(),
            bar=bar,
        )
    except Exception:
        log.exception("degradation gate failed on bar %d", bar)
        return None
    if result is not None and not result.is_no_op:
        pipeline.event_log.emit(QuarantineRecomputed(
            bar=bar,
            trace_id=trace_id,
            added=result.added,
            removed=result.removed,
            current=result.current,
        ))
    return result


def _drop_quarantined_candidates(
    pipeline: ProductionPipeline,
    candidates: List[EnsemblePlayer],
) -> List[EnsemblePlayer]:
    filtered: List[EnsemblePlayer] = []
    for player in candidates:
        if pipeline.qm.is_quarantined(getattr(player, "label", "")):
            continue
        agents = getattr(player, "agents", ()) or ()
        if any(pipeline.qm.is_quarantined(getattr(agent, "label", "")) for agent in agents):
            continue
        filtered.append(player)
    return filtered


# ────────────────────────────────────────────────────────────────────
# StepResult
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepResult:
    """Результат одного bar-а в production main_loop."""

    bar:           int
    regime:        Regime
    leader:        Optional[str]
    leader_changed: bool
    n_signals:     int
    n_filled:      int
    n_rejected:    int
    n_blocked:     int
    n_shadow_signals: int = 0
    n_shadow_filled: int = 0
    n_shadow_rejected: int = 0
    n_shadow_blocked: int = 0
    n_shadow_actors: int = 0
    n_filtered_real_signals: int = 0
    n_stale_close_signals: int = 0
    n_duplicate_open_signals: int = 0
    n_rate_limited_open_signals: int = 0
    n_max_position_saturated_open_signals: int = 0
    n_external_position_signals: int = 0
    n_raw_signals: int = 0
    leader_vote_errors: int = 0
    leader_agent_labels: tuple[str, ...] = ()
    signal_filter_details: tuple[str, ...] = ()
    selected_leader: Optional[str] = None
    executed_leader: Optional[str] = None
    fallback_used: bool = False
    fallback_skipped: bool = False
    fallback_candidate: str = ""
    fallback_reason: str = ""
    blocked_reasons: Dict[str, int] = field(default_factory=dict)
    error:         Optional[str] = None


# ────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────


def main_loop(
    pipeline: ProductionPipeline,
    feed: MarketFeed,
    *,
    max_bars:                    Optional[int] = None,
    recompute_quarantine_every:  int = 10,
    sleep_between_polls_sec:     float = 0.0,
    on_step:                     Optional[Callable[[StepResult], None]] = None,
    on_error:                    Optional[Callable[[Exception], None]] = None,
    on_idle:                     Optional[Callable[[int], None]] = None,
    max_idle_polls:              Optional[int] = None,
) -> List[StepResult]:
    """Основной цикл production-режима.

    Работает пока feed выдаёт snapshots. Когда feed возвращает None —
    либо ждёт sleep_between_polls_sec и повторяет (live), либо завершается
    (replay).

    Параметры:
      max_bars                  — ограничение на число bar-ов (для testing)
      recompute_quarantine_every — каждые N bar-ов пересчитываем QM
      sleep_between_polls_sec   — пауза при пустом feed.next_bar()
      on_step                   — callback после каждого успешного bar-а
      on_error                  — callback при необработанной ошибке (loop не падает)
      on_idle                   — callback при пустом feed.next_bar()
      max_idle_polls            — ограничение idle-опросов для healthcheck/tests
    """
    results: List[StepResult] = []
    signal_id_counter = _max_existing_signal_id(pipeline) + 1
    last_qm_bar = -10**9
    last_regime: Optional[Regime] = None
    bars_processed = 0
    idle_polls = 0

    while True:
        if max_bars is not None and bars_processed >= max_bars:
            break

        try:
            market = feed.next_bar()
        except Exception as exc:
            log.exception("feed error")
            if on_error: on_error(exc)
            if sleep_between_polls_sec > 0:
                time.sleep(sleep_between_polls_sec)
            continue

        if market is None:
            idle_polls += 1
            _record_stale_feed_poll(pipeline, idle_polls)
            if on_idle:
                try:
                    on_idle(idle_polls)
                except Exception:
                    log.exception("on_idle callback error")
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                break
            if sleep_between_polls_sec <= 0 and max_idle_polls is None:
                break  # replay-режим — выходим
            if sleep_between_polls_sec > 0:
                time.sleep(sleep_between_polls_sec)
            continue

        try:
            idle_polls = 0
            _reset_stale_feed(pipeline)
            step, signal_id_counter, last_qm_bar, last_regime = _run_one_bar(
                pipeline, market,
                signal_id_counter=signal_id_counter,
                last_qm_bar=last_qm_bar,
                last_regime=last_regime,
                recompute_every=recompute_quarantine_every,
            )
        except Exception as exc:
            log.exception("step error on bar %d", market.bar)
            if on_error: on_error(exc)
            step = StepResult(
                bar=market.bar, regime=market.regime,
                leader=None, leader_changed=False,
                n_signals=0, n_filled=0, n_rejected=0, n_blocked=0,
                error=str(exc),
            )

        results.append(step)
        bars_processed += 1
        if on_step:
            try:
                on_step(step)
            except Exception:
                log.exception("on_step callback error")

    # Финальный replay через ledger
    pipeline.ledger.replay_from_event_log(pipeline.event_log)
    return results


# ────────────────────────────────────────────────────────────────────
# Internal: обработка одного bar-а
# ────────────────────────────────────────────────────────────────────


def _run_one_bar(
    pipeline:   ProductionPipeline,
    market:     MarketSnapshot,
    *,
    signal_id_counter: int,
    last_qm_bar:       int,
    last_regime:       Optional[Regime],
    recompute_every:   int,
):
    trace = f"prod-{market.bar}"
    sync_pipeline_balance(pipeline)
    reconcile_summary = reconcile_tracker_with_exchange(pipeline, bar_index=market.bar)
    _record_exchange_desync(pipeline, reconcile_summary)
    _expire_stale_pending_orders(pipeline)
    _apply_pending_shadow_updates_to_strategist(pipeline)
    pipeline.event_log.emit(BarStarted(bar=market.bar, trace_id=trace))

    # 1. Регим
    is_change = (last_regime is not None and last_regime != market.regime)
    pipeline.event_log.emit(RegimeDetected(
        bar=market.bar, trace_id=trace,
        regime=market.regime,
        from_regime=last_regime or market.regime,
        is_change=is_change,
    ))
    last_regime = market.regime

    # 2. Compose once before shadow updates. Real selection reuses this snapshot,
    # so current-bar shadow results cannot leak into selector scores.
    _ensure_degradation_baseline(pipeline)
    candidates = _compose_candidates(pipeline, market.regime)
    shadow_summary = _run_shadow_tournament(pipeline, market, candidates)
    _capture_pending_shadow_updates(pipeline)

    # 3. Карантин (раз в N bar-ов) после shadow-обновлений perf
    if market.bar - last_qm_bar >= recompute_every:
        result = pipeline.qm.recompute(pipeline.perf)
        last_qm_bar = market.bar
        if not result.is_no_op:
            pipeline.event_log.emit(QuarantineRecomputed(
                bar=market.bar, trace_id=trace,
                added=result.added, removed=result.removed,
                current=result.current,
            ))

    # 4. Composer → candidates for real leader selection
    degradation_result = _apply_degradation_gate(
        pipeline,
        bar=market.bar,
        trace_id=trace,
    )
    if degradation_result is not None and not degradation_result.is_no_op:
        candidates = _drop_quarantined_candidates(pipeline, candidates)

    kill_reason = _kill_switch_reason(pipeline)
    if kill_reason:
        return (
            StepResult(
                bar=market.bar, regime=market.regime,
                leader=None, leader_changed=False,
                n_signals=0, n_filled=0, n_rejected=0, n_blocked=0,
                n_shadow_signals=shadow_summary.total_signals,
                n_shadow_filled=shadow_summary.total_filled,
                n_shadow_rejected=shadow_summary.total_rejected,
                n_shadow_blocked=shadow_summary.total_blocked,
                n_shadow_actors=shadow_summary.actors,
                error=f"kill switch active: {kill_reason}",
            ),
            signal_id_counter, last_qm_bar, last_regime,
        )

    decision_id = trace

    if not candidates:
        # Никаких eligible — пропускаем bar
        return (
            StepResult(
                bar=market.bar, regime=market.regime,
                leader=None, leader_changed=False,
                n_signals=0, n_filled=0, n_rejected=0, n_blocked=0,
                n_shadow_signals=shadow_summary.total_signals,
                n_shadow_filled=shadow_summary.total_filled,
                n_shadow_rejected=shadow_summary.total_rejected,
                n_shadow_blocked=shadow_summary.total_blocked,
                n_shadow_actors=shadow_summary.actors,
            ),
            signal_id_counter, last_qm_bar, last_regime,
        )

    # 5. Strategist
    context = _decision_context(pipeline, market)
    health_reason = _exchange_health_open_block_reason(pipeline)
    pipeline.event_log.emit(DecisionStarted(
        bar=market.bar,
        trace_id=trace,
        decision_id=decision_id,
        **context,
        candidate_labels=tuple(getattr(c, "label", "") for c in candidates),
        shadow_total_signals=shadow_summary.total_signals,
        shadow_total_filled=shadow_summary.total_filled,
        shadow_total_rejected=shadow_summary.total_rejected,
        shadow_total_blocked=shadow_summary.total_blocked,
    ))
    pipeline.strategist.update_candidates(candidates)
    _sync_strategist_realized_snapshot(pipeline)
    try:
        decision: SwitchDecision = pipeline.strategist.consider_switch(
            market.regime,
            current_bar=market.bar,
            regime_confidence=float(getattr(market, "regime_confidence", 1.0) or 1.0),
        )
    except ValueError:
        return (
            StepResult(
                bar=market.bar, regime=market.regime,
                leader=None, leader_changed=False,
                n_signals=0, n_filled=0, n_rejected=0, n_blocked=0,
                n_shadow_signals=shadow_summary.total_signals,
                n_shadow_filled=shadow_summary.total_filled,
                n_shadow_rejected=shadow_summary.total_rejected,
                n_shadow_blocked=shadow_summary.total_blocked,
                n_shadow_actors=shadow_summary.actors,
                error="all candidates disqualified",
            ),
            signal_id_counter, last_qm_bar, last_regime,
        )

    _emit_candidate_audit_events(
        pipeline,
        market,
        decision,
        decision_id=decision_id,
        trace_id=trace,
        context=context,
    )
    pipeline.event_log.emit(SwitchGateEvaluated(
        bar=market.bar,
        trace_id=trace,
        decision_id=decision_id,
        **context,
        previous_label=(decision.previous.label if decision.previous else ""),
        current_label=getattr(decision, "current_label", "") or (
            decision.previous.label if decision.previous else ""
        ),
        best_label=getattr(decision, "best_label", ""),
        selected_label=decision.new_leader.label,
        best_score=float(getattr(decision, "best_score", 0.0) or 0.0),
        current_score=float(getattr(decision, "current_score", 0.0) or 0.0),
        margin=decision.margin,
        required_margin=float(getattr(decision, "required_margin", 0.0) or 0.0),
        cooldown_passed=bool(getattr(decision, "cooldown_passed", True)),
        cooldown_blocked=bool(getattr(decision, "cooldown_blocked", False)),
        streak_count=int(getattr(decision, "streak_count", 0) or 0),
        streak_needed=int(getattr(decision, "streak_needed", 1) or 1),
        is_urgent=decision.is_urgent,
        switched=decision.switched,
        reason=getattr(decision, "switch_gate_reason", "") or decision.reason,
    ))

    if decision.switched:
        pipeline.event_log.emit(LeaderSelected(
            bar=market.bar, trace_id=trace,
            player_label=decision.new_leader.label,
            previous_label=(decision.previous.label if decision.previous else ""),
            score=decision.score,
            margin=decision.margin,
            is_urgent=decision.is_urgent,
            reason=decision.reason,
            decision_id=decision_id,
            **context,
        ))

    # 6. Vote → real signals from selected leader only
    leader = decision.new_leader
    selected_leader = leader
    executed_leader = selected_leader
    vote_attempt = _vote_candidate_for_real_signals(
        pipeline,
        market,
        selected_leader,
        signal_id_counter=signal_id_counter,
        trace_id=trace,
    )
    if selected_leader.label == "NoTrade" and vote_attempt["raw_signal_count"] == 0:
        cash_flat_signals = _cash_flat_close_signals(
            pipeline,
            market,
            signal_id_start=signal_id_counter,
        )
        if cash_flat_signals:
            signal_id_counter = max(signal.id for signal in cash_flat_signals) + 1
            signal_guard = filter_real_signals_against_tracker(
                cash_flat_signals,
                player=selected_leader,
                pipeline=pipeline,
                bar_index=market.bar,
                max_new_opens_per_bar=getattr(
                    getattr(pipeline, "live_execution", None),
                    "max_new_opens_per_bar",
                    1,
                ),
                max_open_positions=getattr(
                    getattr(pipeline, "risk_config", None),
                    "max_open_positions",
                    None,
                ),
            )
            vote_attempt = {
                "leader": selected_leader,
                "raw_signals": cash_flat_signals,
                "raw_signal_count": len(cash_flat_signals),
                "leader_vote_error_count": 0,
                "signal_guard": signal_guard,
                "signal_id_counter": signal_id_counter,
            }
    fallback_used = False
    fallback_skipped = False
    fallback_candidate = ""
    fallback_reason = ""
    if vote_attempt["raw_signal_count"] == 0:
        fallback_skipped = any(
            candidate.label != selected_leader.label for candidate in candidates
        )
        if fallback_skipped:
            fallback_reason = (
                "actionable fallback disabled: selected leader emitted no raw signals"
            )
    raw_signals = vote_attempt["raw_signals"]
    raw_signal_count = vote_attempt["raw_signal_count"]
    leader_vote_error_count = vote_attempt["leader_vote_error_count"]
    signal_guard = vote_attempt["signal_guard"]
    signal_id_counter = vote_attempt["signal_id_counter"]
    signals = signal_guard.signals
    if signals:
        for sig in signals:
            signal_context = _signal_context(context, decision_id=decision_id, signal=sig)
            if sig.action.is_open and health_reason:
                signal_context["exchange_health_reason"] = health_reason
            pipeline.event_log.emit(SignalEmitted(
                bar=market.bar,
                trace_id=trace,
                signal=sig,
                **signal_context,
            ))
            getattr(pipeline, "real_perf", pipeline.perf).record_signal(sig)

    # 7. Execute real signals
    n_filled = n_rejected = n_blocked = 0
    blocked_reasons: Dict[str, int] = {}
    for sig in signals:
        try:
            set_event_context = getattr(pipeline.executor, "set_event_context", None)
            if callable(set_event_context):
                signal_context = _signal_context(context, decision_id=decision_id, signal=sig)
                if sig.action.is_open and health_reason:
                    signal_context["exchange_health_reason"] = health_reason
                set_event_context(signal_context)
            res: ExecutionResult = pipeline.executor.execute(
                sig, balance_usd=pipeline.current_balance,
            )
        except Exception:
            log.exception("execute failed for signal %d", sig.id)
            _record_order_failure(pipeline, "execute exception")
            n_rejected += 1
            continue
        if res.status == ExecutionStatus.FILLED:
            n_filled += 1
            _record_order_success(pipeline, sig, res)
            _record_realized_result_for_strategy(pipeline, res)
            # Update balance (simple): добавляем realized PnL если был close
            # (детальнее — в AttributionLedger; здесь упрощённо)
        elif res.status == ExecutionStatus.REJECTED:
            _record_order_failure(pipeline, res.reason or "rejected")
            n_rejected += 1
        elif res.status == ExecutionStatus.PENDING:
            _record_order_failure(pipeline, res.reason or "pending")
            n_rejected += 1
        elif res.status == ExecutionStatus.BLOCKED:
            n_blocked += 1
            reason = res.reason or "blocked"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        if _kill_switch_reason(pipeline):
            break
    kill_reason_after_execution = _kill_switch_reason(pipeline)

    return (
        StepResult(
            bar=market.bar, regime=market.regime,
            leader=executed_leader.label,
            leader_changed=decision.switched,
            n_signals=len(signals),
            n_filled=n_filled,
            n_rejected=n_rejected,
            n_blocked=n_blocked,
            n_shadow_signals=shadow_summary.total_signals,
            n_shadow_filled=shadow_summary.total_filled,
            n_shadow_rejected=shadow_summary.total_rejected,
            n_shadow_blocked=shadow_summary.total_blocked,
            n_shadow_actors=shadow_summary.actors,
            n_filtered_real_signals=signal_guard.filtered,
            n_stale_close_signals=signal_guard.stale_closes,
            n_duplicate_open_signals=signal_guard.duplicate_opens,
            n_rate_limited_open_signals=signal_guard.rate_limited_opens,
            n_max_position_saturated_open_signals=signal_guard.max_position_saturated_opens,
            n_external_position_signals=signal_guard.external_position_signals,
            n_raw_signals=raw_signal_count,
            leader_vote_errors=leader_vote_error_count,
            leader_agent_labels=tuple(getattr(executed_leader, "agent_labels", ()) or ()),
            signal_filter_details=tuple(signal_guard.details),
            selected_leader=selected_leader.label,
            executed_leader=executed_leader.label,
            fallback_used=fallback_used,
            fallback_skipped=fallback_skipped,
            fallback_candidate=fallback_candidate,
            fallback_reason=fallback_reason,
            blocked_reasons=blocked_reasons,
            error=(
                f"kill switch active: {kill_reason_after_execution}"
                if kill_reason_after_execution else None
            ),
        ),
        signal_id_counter, last_qm_bar, last_regime,
    )


def _compose_candidates(
    pipeline: ProductionPipeline,
    regime: Regime,
) -> List[EnsemblePlayer]:
    candidates: List[EnsemblePlayer] = []
    for profile in pipeline.profiles:
        try:
            player = pipeline.composer.compose_from_profile_with_fallback(
                profile,
                regime,
            )
            if player is not None:
                candidates.append(player)
        except Exception:
            log.exception("compose failed for profile %s", profile.label)
    candidates.extend(_compose_solo_agent_candidates(pipeline, regime, candidates))
    return candidates


def _compose_solo_agent_candidates(
    pipeline: ProductionPipeline,
    regime: Regime,
    existing: List[EnsemblePlayer],
) -> List[EnsemblePlayer]:
    """Expose top solo shadow performers as real player candidates."""
    existing_labels = {player.label for player in existing}
    solo: List[EnsemblePlayer] = []
    try:
        scored = pipeline.selector.select(regime, k=_SOLO_AGENT_CANDIDATE_LIMIT)
        if not scored and hasattr(pipeline.selector, "select_with_fallback"):
            scored = pipeline.selector.select_with_fallback(
                regime,
                k=_SOLO_AGENT_CANDIDATE_LIMIT,
                fallback_threshold=-10.0,
                min_count=1,
            )
    except Exception:
        log.exception("solo agent candidate selection failed")
        return solo
    for row in scored:
        if float(getattr(row.metrics, "pnl_pct", 0.0) or 0.0) <= 0.0:
            continue
        label = f"Solo_{row.label}"
        if label in existing_labels:
            continue
        solo.append(EnsemblePlayer(
            label=label,
            agents=[row.agent],
            weights={row.label: 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            affinity=None,
        ))
    return solo


def _vote_candidate_for_real_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    leader: EnsemblePlayer,
    *,
    signal_id_counter: int,
    trace_id: str,
) -> Dict[str, Any]:
    sync_player_agents_to_real_positions(
        leader,
        pipeline,
        bar_index=market.bar,
        market_symbols=market.prices.keys(),
    )
    try:
        raw_signals = leader.vote(market, signal_id_start=signal_id_counter)
    except Exception as exc:
        _record_player_vote_failure(pipeline, market, leader.label, exc, trace_id=trace_id)
        raw_signals = []
    _record_agent_vote_failures(pipeline, market, leader, trace_id=trace_id)
    raw_signal_count = len(raw_signals)
    leader_vote_error_count = len(getattr(leader, "last_vote_errors", []) or [])
    next_signal_id = signal_id_counter
    if raw_signals:
        next_signal_id = max(s.id for s in raw_signals) + 1
    signal_guard = filter_real_signals_against_tracker(
        raw_signals,
        player=leader,
        pipeline=pipeline,
        bar_index=market.bar,
        max_new_opens_per_bar=getattr(
            getattr(pipeline, "live_execution", None),
            "max_new_opens_per_bar",
            1,
        ),
        max_open_positions=getattr(
            getattr(pipeline, "risk_config", None),
            "max_open_positions",
            None,
        ),
    )
    return {
        "leader": leader,
        "raw_signals": raw_signals,
        "raw_signal_count": raw_signal_count,
        "leader_vote_error_count": leader_vote_error_count,
        "signal_guard": signal_guard,
        "signal_id_counter": next_signal_id,
    }


def _cash_flat_close_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    signal_id_start: int,
) -> List[Signal]:
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is None or not callable(getattr(tracker, "all_open", None)):
        return []
    signals: List[Signal] = []
    sid = int(signal_id_start)
    for sym, pos in sorted((tracker.all_open() or {}).items()):
        if is_external_position(pos):
            continue
        symbol = str(getattr(pos, "sym", sym) or sym).upper()
        price = float(market.prices.get(symbol, 0.0) or 0.0)
        if price <= 0:
            continue
        signals.append(Signal(
            id=sid,
            bar=market.bar,
            sym=symbol,
            action=Action.FUT_CLOSE_ALL,
            price=price,
            regime=market.regime,
            by_player="NoTrade",
            by_agent="CashFlat",
            timestamp=market.timestamp,
        ))
        sid += 1
    return signals


def _find_actionable_candidate_vote(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    candidates: List[EnsemblePlayer],
    decision: SwitchDecision,
    *,
    skip_label: str,
    signal_id_counter: int,
    trace_id: str,
) -> Optional[Dict[str, Any]]:
    candidates_by_label = {candidate.label: candidate for candidate in candidates}
    ordered_labels = [
        row.label
        for row in getattr(decision, "candidate_scores", ()) or ()
        if row.label in candidates_by_label
    ]
    ordered_labels.extend(
        candidate.label
        for candidate in candidates
        if candidate.label not in ordered_labels
    )
    for label in ordered_labels:
        if label == skip_label:
            continue
        candidate = candidates_by_label.get(label)
        if candidate is None:
            continue
        attempt = _vote_candidate_for_real_signals(
            pipeline,
            market,
            candidate,
            signal_id_counter=signal_id_counter,
            trace_id=trace_id,
        )
        if attempt["raw_signal_count"] > 0:
            return attempt
    return None


def _run_shadow_tournament(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    players: List[EnsemblePlayer],
):
    from .shadow_tournament import ShadowStepSummary

    tournament = getattr(pipeline, "shadow_tournament", None)
    if tournament is None:
        summary = ShadowStepSummary()
    else:
        try:
            summary = tournament.run_bar(
                market,
                players=players,
                balance_usd=pipeline.current_balance,
            )
        except Exception:
            log.exception("shadow tournament failed on bar %d", market.bar)
            summary = ShadowStepSummary()
    try:
        pipeline.shadow_last_summary = summary.as_dict()
    except Exception:
        pass
    return summary


def _apply_pending_shadow_updates_to_strategist(pipeline: ProductionPipeline) -> None:
    pending = tuple(getattr(pipeline, "_pending_shadow_actor_updates", ()) or ())
    pending_positions = getattr(pipeline, "_pending_shadow_player_positions", None)
    update = getattr(getattr(pipeline, "strategist", None), "update_shadow_actor_updates", None)
    if pending and callable(update):
        try:
            update(pending)
        except Exception:
            log.debug("failed to sync shadow updates into strategist", exc_info=True)
    update_positions = getattr(
        getattr(pipeline, "strategist", None),
        "update_shadow_position_snapshot",
        None,
    )
    if pending_positions is not None and callable(update_positions):
        try:
            update_positions(
                player_positions=pending_positions,
                real_positions=tracker_positions_for_agent_sync(pipeline),
            )
        except Exception:
            log.debug("failed to sync shadow positions into strategist", exc_info=True)
    pipeline._pending_shadow_actor_updates = ()
    pipeline._pending_shadow_player_positions = None


def _capture_pending_shadow_updates(pipeline: ProductionPipeline) -> None:
    tournament = getattr(pipeline, "shadow_tournament", None)
    getter = getattr(tournament, "last_actor_updates", None)
    if not callable(getter):
        pipeline._pending_shadow_actor_updates = ()
        return
    try:
        pipeline._pending_shadow_actor_updates = tuple(getter())
    except Exception:
        log.debug("failed to capture shadow updates for strategist", exc_info=True)
        pipeline._pending_shadow_actor_updates = ()
    positions_getter = getattr(tournament, "last_player_open_positions", None)
    if not callable(positions_getter):
        pipeline._pending_shadow_player_positions = None
        return
    try:
        pipeline._pending_shadow_player_positions = dict(positions_getter())
    except Exception:
        log.debug("failed to capture shadow positions for strategist", exc_info=True)
        pipeline._pending_shadow_player_positions = None


def _emit_candidate_audit_events(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    decision: SwitchDecision,
    *,
    decision_id: str,
    trace_id: str,
    context: Dict[str, str],
) -> None:
    selected_label = getattr(decision.new_leader, "label", "")
    for row in getattr(decision, "candidate_scores", ()) or ():
        pipeline.event_log.emit(CandidateScored(
            bar=market.bar,
            trace_id=trace_id,
            decision_id=decision_id,
            **context,
            player_label=row.label,
            rank=row.rank,
            score=row.score,
            score_source=row.score_source,
            selected_by_pantheon=(row.label == selected_label),
            has_data=row.has_data,
            closed_trades=row.closed_trades,
            signals=row.signals,
            execution_failures=row.execution_failures,
            uncertainty_penalty=row.uncertainty_penalty,
            memory_keys_read=row.memory_keys_read,
            agent_labels=row.agent_labels,
            session_score_delta=getattr(row, "session_score_delta", 0.0),
            session_pnl_pct=getattr(row, "session_pnl_pct", 0.0),
            session_underperformance_penalty=getattr(
                row,
                "session_underperformance_penalty",
                0.0,
            ),
            session_stale_penalty=getattr(row, "session_stale_penalty", 0.0),
        ))
    for row in getattr(decision, "candidate_rejections", ()) or ():
        pipeline.event_log.emit(CandidateRejected(
            bar=market.bar,
            trace_id=trace_id,
            decision_id=decision_id,
            **context,
            player_label=row.label,
            reason=row.reason,
        ))


def _record_agent_vote_failures(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    player: EnsemblePlayer,
    *,
    trace_id: str,
) -> int:
    count = 0
    for err in getattr(player, "last_vote_errors", []) or []:
        label = str(getattr(err, "agent_label", "") or "")
        reason = str(getattr(err, "reason", "") or "")
        if not label:
            continue
        pipeline.event_log.emit(AgentVoteFailed(
            bar=market.bar,
            trace_id=trace_id,
            player_label=getattr(player, "label", ""),
            agent_label=label,
            reason=reason,
        ))
        getattr(pipeline, "real_perf", pipeline.perf).record_actor_failure(label, market.regime)
        count += 1
    return count


def _record_player_vote_failure(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    player_label: str,
    exc: Exception,
    *,
    trace_id: str,
) -> None:
    reason = f"{type(exc).__name__}: {exc}"
    pipeline.event_log.emit(PlayerVoteFailed(
        bar=market.bar,
        trace_id=trace_id,
        player_label=player_label,
        reason=reason,
    ))
    getattr(pipeline, "real_perf", pipeline.perf).record_actor_failure(player_label, market.regime)


def _live_config(pipeline: ProductionPipeline) -> Any:
    return getattr(pipeline, "live_execution", None)


def _kill_state(pipeline: ProductionPipeline) -> Any:
    return getattr(pipeline, "kill_switch", None)


def _set_kill_switch(pipeline: ProductionPipeline, reason: str) -> None:
    state = _kill_state(pipeline)
    if state is None or not reason:
        return
    if getattr(state, "disabled_reason", ""):
        return
    state.disabled_reason = reason
    log.error("real trading disabled by kill switch: %s", reason)


def _kill_switch_reason(pipeline: ProductionPipeline) -> str:
    state = _kill_state(pipeline)
    if state is not None and getattr(state, "disabled_reason", ""):
        return str(state.disabled_reason)

    cfg = _live_config(pipeline)
    max_daily_loss_pct = float(getattr(cfg, "max_daily_loss_pct", 0.0) or 0.0)
    if max_daily_loss_pct > 0:
        initial = float(getattr(pipeline, "initial_capital", 0.0) or 0.0)
        current = float(getattr(pipeline, "current_balance", 0.0) or 0.0)
        if initial > 0 and current > 0:
            floor = initial * (1.0 - max_daily_loss_pct / 100.0)
            if current <= floor:
                reason = (
                    f"max daily loss exceeded: current={current:.2f}, "
                    f"floor={floor:.2f}"
                )
                _set_kill_switch(pipeline, reason)
                return reason
    return ""


def _exchange_health_open_block_reason(pipeline: ProductionPipeline) -> str:
    snapshot = getattr(pipeline, "account_snapshot", None) or {}
    if not isinstance(snapshot, dict):
        return ""
    health = snapshot.get("data_health") or {}
    if not isinstance(health, dict):
        return ""

    reason = str(
        health.get("recent_data_error")
        or health.get("last_data_error_reason")
        or health.get("reason")
        or ""
    ).strip()
    if reason:
        return reason

    if bool(health.get("uses_cached_balance") or health.get("cached_equity")):
        return "cached exchange equity snapshot"

    try:
        if float(health.get("api_errors_10m", 0) or 0) > 0:
            return f"api_errors_10m={int(float(health.get('api_errors_10m') or 0))}"
    except (TypeError, ValueError):
        pass

    snapshot_healthy = health.get("snapshot_healthy")
    if snapshot_healthy is False:
        return "snapshot unhealthy"
    return ""


def _record_exchange_desync(pipeline: ProductionPipeline, summary: Any) -> None:
    cfg = _live_config(pipeline)
    limit = int(getattr(cfg, "max_exchange_desync_events", 0) or 0)
    if limit <= 0 or not isinstance(summary, dict):
        return
    if any(key.startswith("owned_") for key in summary):
        events = int(summary.get("owned_added", 0) or 0)
        events += int(summary.get("owned_updated", 0) or 0)
        events += int(summary.get("owned_removed", 0) or 0)
    else:
        events = int(summary.get("added", 0) or 0)
        events += int(summary.get("updated", 0) or 0)
        events += int(summary.get("removed", 0) or 0)
    if events <= 0:
        return
    state = _kill_state(pipeline)
    if state is None:
        return
    state.exchange_desync_events += events
    if state.exchange_desync_events >= limit:
        _set_kill_switch(
            pipeline,
            f"exchange desync events {state.exchange_desync_events} >= {limit}",
        )


def _expire_stale_pending_orders(pipeline: ProductionPipeline) -> int:
    cfg = _live_config(pipeline)
    timeout = float(getattr(cfg, "pending_order_timeout_sec", 0.0) or 0.0)
    if timeout <= 0:
        return 0
    expire = getattr(getattr(pipeline, "executor", None), "expire_stale_pending_orders", None)
    if not callable(expire):
        return 0
    try:
        return int(expire(max_age_sec=timeout) or 0)
    except Exception:
        log.exception("pending order timeout sweep failed")
        return 0


def _record_stale_feed_poll(pipeline: ProductionPipeline, idle_polls: int) -> None:
    cfg = _live_config(pipeline)
    limit = int(getattr(cfg, "max_stale_feed_polls", 0) or 0)
    state = _kill_state(pipeline)
    if state is not None:
        state.stale_feed_polls = int(idle_polls)
    if limit > 0 and idle_polls >= limit:
        _set_kill_switch(pipeline, f"stale feed polls {idle_polls} >= {limit}")


def _reset_stale_feed(pipeline: ProductionPipeline) -> None:
    state = _kill_state(pipeline)
    if state is not None:
        state.stale_feed_polls = 0


def _record_order_success(
    pipeline: ProductionPipeline,
    signal: Signal,
    result: ExecutionResult,
) -> None:
    state = _kill_state(pipeline)
    if state is not None:
        state.consecutive_failed_orders = 0
        state.api_error_streak = 0

    cfg = _live_config(pipeline)
    limit = float(getattr(cfg, "max_slippage_pct", 0.0) or 0.0)
    slippage_pct = _result_slippage_pct(signal, result)
    if limit > 0 and slippage_pct is not None and slippage_pct > limit:
        _set_kill_switch(
            pipeline,
            f"excessive slippage {slippage_pct:.4f}% > {limit:.4f}%",
        )


def _record_realized_result_for_strategy(
    pipeline: ProductionPipeline,
    result: ExecutionResult,
) -> None:
    pnl_by_player = dict(getattr(pipeline, "_allocator_realized_pnl_by_player", {}) or {})
    trades_by_player = dict(
        getattr(pipeline, "_allocator_realized_trade_counts_by_player", {}) or {}
    )
    wins_by_player = dict(getattr(pipeline, "_allocator_realized_win_counts_by_player", {}) or {})

    for label, pnl in getattr(result, "realized_pnl_by_player", ()) or ():
        key = str(label or "")
        if not key:
            continue
        pnl_by_player[key] = float(pnl_by_player.get(key, 0.0) or 0.0) + float(pnl or 0.0)
    for label, count in getattr(result, "closed_trade_counts_by_player", ()) or ():
        key = str(label or "")
        if not key:
            continue
        trades_by_player[key] = int(trades_by_player.get(key, 0) or 0) + int(count or 0)
    for label, count in getattr(result, "win_counts_by_player", ()) or ():
        key = str(label or "")
        if not key:
            continue
        wins_by_player[key] = int(wins_by_player.get(key, 0) or 0) + int(count or 0)

    pipeline._allocator_realized_pnl_by_player = pnl_by_player
    pipeline._allocator_realized_trade_counts_by_player = trades_by_player
    pipeline._allocator_realized_win_counts_by_player = wins_by_player


def _sync_strategist_realized_snapshot(pipeline: ProductionPipeline) -> None:
    update = getattr(getattr(pipeline, "strategist", None), "update_realized_pnl_snapshot", None)
    if not callable(update):
        return
    try:
        update(
            pnl_by_player=getattr(pipeline, "_allocator_realized_pnl_by_player", {}) or {},
            trade_counts_by_player=(
                getattr(pipeline, "_allocator_realized_trade_counts_by_player", {}) or {}
            ),
            win_counts_by_player=getattr(
                pipeline,
                "_allocator_realized_win_counts_by_player",
                {},
            ) or {},
            initial_capital=float(getattr(pipeline, "initial_capital", 0.0) or 0.0),
        )
    except Exception:
        log.debug("failed to sync realized PnL snapshot into strategist", exc_info=True)


def _record_order_failure(pipeline: ProductionPipeline, reason: str) -> None:
    state = _kill_state(pipeline)
    cfg = _live_config(pipeline)
    if state is None:
        return
    state.consecutive_failed_orders += 1
    if _looks_like_api_error(reason):
        state.api_error_streak += 1

    failed_limit = int(getattr(cfg, "max_consecutive_failed_orders", 0) or 0)
    api_limit = int(getattr(cfg, "max_api_error_streak", 0) or 0)
    if failed_limit > 0 and state.consecutive_failed_orders >= failed_limit:
        _set_kill_switch(
            pipeline,
            (
                f"consecutive failed orders "
                f"{state.consecutive_failed_orders} >= {failed_limit}"
            ),
        )
    if api_limit > 0 and state.api_error_streak >= api_limit:
        _set_kill_switch(
            pipeline,
            f"API error storm {state.api_error_streak} >= {api_limit}",
        )


def _result_slippage_pct(
    signal: Signal,
    result: ExecutionResult,
) -> Optional[float]:
    trade = getattr(result, "trade", None)
    if trade is None:
        return None
    try:
        expected = float(signal.price)
        filled = float(trade.fill_price)
    except (TypeError, ValueError):
        return None
    if expected <= 0 or filled <= 0:
        return None
    return abs(filled - expected) / expected * 100.0


def _looks_like_api_error(reason: str) -> bool:
    lowered = str(reason or "").lower()
    return any(
        marker in lowered
        for marker in (
            "api",
            "exchange error",
            "exchange exception",
            "exchange metadata",
            "timeout",
            "rate limit",
            "connection",
            "network",
        )
    )


def _max_existing_signal_id(pipeline: ProductionPipeline) -> int:
    """Если pipeline восстановлен из snapshot — продолжаем нумерацию signal_id."""
    max_id = -1
    try:
        from ..attribution.events import SignalEmitted
        for ev in pipeline.event_log.query(event_types=[SignalEmitted]):
            if ev.signal and ev.signal.id > max_id:
                max_id = ev.signal.id
    except Exception:
        pass
    try:
        for raw_id in (pipeline.perf.snapshot().get("seen_signal_ids") or []):
            max_id = max(max_id, int(raw_id))
    except Exception:
        pass
    return max(max_id, 0)
