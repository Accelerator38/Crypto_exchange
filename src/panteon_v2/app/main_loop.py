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
    BarStarted,
    LeaderSelected,
    QuarantineRecomputed,
    RegimeDetected,
    SignalEmitted,
)
from ..domain.types import MarketSnapshot, Regime, Signal
from ..execution import ExecutionResult, ExecutionStatus
from ..selection import EnsemblePlayer, SwitchDecision
from ..shadow.feed import MarketFeed
from .bootstrap import ProductionPipeline
from .live_state import (
    filter_real_signals_against_tracker,
    reconcile_tracker_with_exchange,
    sync_player_agents_to_real_positions,
)


log = logging.getLogger(__name__)


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


def _normalize_account_snapshot(raw: Any) -> Dict[str, float]:
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
    return out


def _snapshot_balance(snapshot: Dict[str, float]) -> Optional[float]:
    for key in ("current_balance", "futures_equity", "total_assets"):
        value = float(snapshot.get(key, 0.0) or 0.0)
        if value > 0:
            return value
    return None


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

    # 2. Shadow tournament: all agents/players learn virtually before real selection.
    shadow_candidates = _compose_candidates(pipeline, market.regime)
    shadow_summary = _run_shadow_tournament(pipeline, market, shadow_candidates)

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

    candidates = _compose_candidates(pipeline, market.regime)

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
    pipeline.strategist.update_candidates(candidates)
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

    if decision.switched:
        pipeline.event_log.emit(LeaderSelected(
            bar=market.bar, trace_id=trace,
            player_label=decision.new_leader.label,
            previous_label=(decision.previous.label if decision.previous else ""),
            score=decision.score,
            margin=decision.margin,
            is_urgent=decision.is_urgent,
            reason=decision.reason,
        ))

    # 6. Vote → real signals from selected leader only
    leader = decision.new_leader
    sync_player_agents_to_real_positions(
        leader,
        pipeline,
        bar_index=market.bar,
        market_symbols=market.prices.keys(),
    )
    raw_signals: List[Signal] = leader.vote(market, signal_id_start=signal_id_counter)
    if raw_signals:
        signal_id_counter = max(s.id for s in raw_signals) + 1
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
    )
    signals = signal_guard.signals
    if signals:
        for sig in signals:
            pipeline.event_log.emit(SignalEmitted(
                bar=market.bar,
                trace_id=trace,
                signal=sig,
            ))
            getattr(pipeline, "real_perf", pipeline.perf).record_signal(sig)

    # 7. Execute real signals
    n_filled = n_rejected = n_blocked = 0
    blocked_reasons: Dict[str, int] = {}
    for sig in signals:
        try:
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
            leader=leader.label,
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
    return candidates


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


def _record_exchange_desync(pipeline: ProductionPipeline, summary: Any) -> None:
    cfg = _live_config(pipeline)
    limit = int(getattr(cfg, "max_exchange_desync_events", 0) or 0)
    if limit <= 0 or not isinstance(summary, dict):
        return
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
