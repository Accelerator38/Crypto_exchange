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

import copy
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..attribution import (
    AgentVoteFailed,
    BarStarted,
    DecisionStarted,
    LeaderSelected,
    OrderFilled,
    PlayerVoteFailed,
    QuarantineRecomputed,
    RegimeDetected,
    SignalEmitted,
    PositionOpened,
    SwitchGateEvaluated,
)
from ..domain.types import Action, MarketSnapshot, Regime, Signal
from ..execution import ExecutionResult, ExecutionStatus
from ..memory.quarantine import RecomputeResult
from ..selection import (
    EnsemblePlayer,
    FlashDecision,
    RotatingAgentPlayer,
    StrategyPlayer,
    SwitchDecision,
    ThresholdProfile,
    WeightedConsensus,
)
from ..selection.player import normalize_vote_result
from ..shadow.feed import MarketFeed
from .bootstrap import ProductionPipeline
from .audit_emission import (
    emit_candidate_audit_events,
    emit_flash_audit_events,
)
from .degradation_tracking import (
    flash_event_log_tail,
    flash_update_degradation_state_from_events,
)
from .decision_paths.flash import (
    FlashDecisionPathCallbacks,
    run_flash_decision_path,
)
from .flash_state import (
    flash_degraded_actor_keys,
    flash_degraded_open_symbols,
    flash_degraded_signal_keys,
    flash_open_position_actor_keys_by_symbol,
    flash_open_position_sides_by_symbol,
    flash_regime_degradation_key,
)
from .live_state import (
    RealSignalGuardResult,
    filter_real_signals_against_tracker,
    is_external_position,
    reconcile_tracker_with_exchange,
    sync_player_agents_to_real_positions,
    tracker_positions_for_agent_sync,
)


log = logging.getLogger(__name__)
_DEFAULT_SOLO_AGENT_CANDIDATE_LIMIT = 3
_REGIME_SWITCH_ADAPTIVE_MIN_CLOSED_TRADES = 20
_SHADOW_STATE_PLAYER_PREFIXES = ("Antonius_",)
_SHADOW_POSITION_REPLAY_AGENT = "ShadowPositionReplay"


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
                _record_live_equity_peak(pipeline, balance)
                return balance
            if snapshot.get("data_health"):
                pipeline.account_snapshot = snapshot

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
        _record_live_equity_peak(pipeline, equity)
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


def _market_policy_tags(market: MarketSnapshot) -> Tuple[str, ...]:
    tags: List[str] = []
    for attr in ("market_tags", "policy_tags", "tags"):
        value = getattr(market, attr, ())
        if isinstance(value, str):
            tags.append(value)
        else:
            try:
                tags.extend(str(item) for item in (value or ()))
            except TypeError:
                continue
    for attr in (
        "noise_bucket",
        "range_bucket",
        "vol_bucket",
        "actionability_bucket",
        "market_noise",
        "market_volatility",
    ):
        value = getattr(market, attr, "")
        if value:
            tags.append(str(value))
    bool_tags = {
        "is_noisy": "noisy",
        "noisy": "noisy",
        "is_high_range": "high_range",
        "high_range": "high_range",
        "is_low_actionability": "low_actionability",
        "low_actionability": "low_actionability",
    }
    for attr, tag in bool_tags.items():
        if bool(getattr(market, attr, False)):
            tags.append(tag)
    normalized: List[str] = []
    seen = set()
    for raw in tags:
        tag = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        if tag and tag not in seen:
            seen.add(tag)
            normalized.append(tag)
    return tuple(normalized)


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


def _apply_quarantine_recovery_override(
    pipeline: ProductionPipeline,
    *,
    bar: int,
    trace_id: str,
) -> RecomputeResult:
    qm = getattr(pipeline, "qm", None)
    perf = getattr(pipeline, "perf", None)
    if qm is None:
        return RecomputeResult(
            added=frozenset(),
            removed=frozenset(),
            current=frozenset(),
        )

    before = qm.all_quarantined()
    manual_labels = _quarantine_override_labels(pipeline)
    for label in manual_labels:
        _release_old_memory_quarantine(
            qm,
            label,
            reason="manual_release_override",
            bar=bar,
        )

    if bool(getattr(pipeline, "quarantine_recovery_enabled", True)) and perf is not None:
        gate = getattr(pipeline, "degradation_gate", None)
        session_metrics = getattr(gate, "session_metrics", None)
        if callable(session_metrics):
            min_pnl = _quarantine_recovery_min_pnl_pct(pipeline)
            min_closed = _quarantine_recovery_min_closed_trades(pipeline)
            for label in tuple(before):
                record = qm.record_for(label)
                if (
                    record is None
                    or record.state != "quarantined"
                    or record.reason != "hopeless_in_all_regimes"
                ):
                    continue
                try:
                    metrics = session_metrics(perf, label)
                except Exception:
                    log.debug("quarantine recovery session metrics failed", exc_info=True)
                    continue
                if (
                    int(getattr(metrics, "closed_trades", 0) or 0) < min_closed
                    or float(getattr(metrics, "pnl_pct", 0.0) or 0.0) < min_pnl
                ):
                    continue
                _release_old_memory_quarantine(
                    qm,
                    label,
                    reason=(
                        "session_recovery:"
                        f"pnl_pct={float(getattr(metrics, 'pnl_pct', 0.0) or 0.0):.4f};"
                        f"closed={int(getattr(metrics, 'closed_trades', 0) or 0)}"
                    ),
                    bar=bar,
                )

    after = qm.all_quarantined()
    result = RecomputeResult(
        added=frozenset(),
        removed=frozenset(before - after),
        current=after,
    )
    if not result.is_no_op:
        pipeline.event_log.emit(QuarantineRecomputed(
            bar=bar,
            trace_id=trace_id,
            added=result.added,
            removed=result.removed,
            current=result.current,
        ))
    return result


def _quarantine_override_labels(pipeline: ProductionPipeline) -> Tuple[str, ...]:
    labels = []
    seen = set()
    for raw in (getattr(pipeline, "quarantine_override_labels", ()) or ()):
        label = str(raw or "").strip()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
    return tuple(labels)


def _quarantine_recovery_min_pnl_pct(pipeline: ProductionPipeline) -> float:
    raw = getattr(pipeline, "quarantine_recovery_min_pnl_pct", None)
    if raw is None:
        cfg = getattr(getattr(pipeline, "qm", None), "_config", None)
        raw = getattr(cfg, "quarantine_recovery_pnl_pct", 0.10)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.10


def _quarantine_recovery_min_closed_trades(pipeline: ProductionPipeline) -> int:
    raw = getattr(pipeline, "quarantine_recovery_min_closed_trades", None)
    if raw is None:
        cfg = getattr(getattr(pipeline, "qm", None), "_config", None)
        raw = getattr(cfg, "quarantine_recovery_closed", 3)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 3


def _release_old_memory_quarantine(
    qm: object,
    label: str,
    *,
    reason: str,
    bar: int,
) -> None:
    release = getattr(qm, "force_release_override", None)
    if callable(release):
        release(label, reason=reason, bar=bar)


def _drop_quarantined_candidates(
    pipeline: ProductionPipeline,
    candidates: List[EnsemblePlayer],
) -> List[EnsemblePlayer]:
    filtered: List[EnsemblePlayer] = []
    for player in candidates:
        if pipeline.qm.is_quarantined(getattr(player, "label", "")):
            continue
        agents = getattr(player, "agents", ()) or ()
        if any(
            pipeline.qm.is_quarantined(getattr(agent, "label", ""))
            and not _allows_quarantined_genetics_agent(player, agent)
            for agent in agents
        ):
            continue
        filtered.append(player)
    return filtered


# ────────────────────────────────────────────────────────────────────
# StepResult
# ────────────────────────────────────────────────────────────────────


def _allows_quarantined_genetics_agent(player: EnsemblePlayer, agent) -> bool:
    player_label = str(getattr(player, "label", "") or "").lower()
    agent_label = str(getattr(agent, "label", "") or "").lower()
    return "genetic" in player_label and "genetic" in agent_label


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
    selected_actors_by_symbol: Dict[str, str] = field(default_factory=dict)
    signal_filter_details: tuple[str, ...] = ()
    selected_leader: Optional[str] = None
    executed_leader: Optional[str] = None
    fallback_used: bool = False
    fallback_skipped: bool = False
    fallback_candidate: str = ""
    fallback_reason: str = ""
    blocked_reasons: Dict[str, int] = field(default_factory=dict)
    causal_decision: Dict[str, Any] = field(default_factory=dict)
    error:         Optional[str] = None


@dataclass(frozen=True)
class _FlashExecutionActor:
    label: str
    agents: Tuple[Any, ...] = ()
    agent_labels: Tuple[str, ...] = ()


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
    collect_results:             bool = True,
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
      collect_results           — сохранять StepResult в возвращаемом списке
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

        if collect_results:
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
    pipeline.event_log.emit(BarStarted(bar=market.bar, trace_id=trace))

    if bool(getattr(pipeline, "player_only_runtime", False)):
        return _run_player_only_decision_path(
            pipeline,
            market,
            signal_id_counter=signal_id_counter,
            last_qm_bar=last_qm_bar,
            last_regime=last_regime,
            trace=trace,
        )

    policy_runtime = getattr(pipeline, "policy_runtime_v1", None)
    if policy_runtime is not None:
        decision_id = trace
        context = _decision_context(pipeline, market)
        try:
            policy_risk_reason = _policy_daily_loss_kill_switch_reason(
                pipeline,
                policy_runtime,
                market,
            )
        except Exception as exc:
            log.exception("policy risk-state update failed")
            policy_risk_reason = (
                "policy risk-state failure: "
                f"{type(exc).__name__}: {exc}"
            )
        if policy_risk_reason:
            _set_kill_switch(
                pipeline,
                policy_risk_reason,
                kind=(
                    "policy_risk_state"
                    if policy_risk_reason.startswith("policy risk-state failure")
                    else "policy_daily_loss"
                ),
            )
        kill_reason = _kill_switch_reason(pipeline, market)
        if kill_reason:
            return _run_manage_only_kill_switch_step(
                pipeline,
                market,
                None,
                signal_id_counter=signal_id_counter,
                last_qm_bar=last_qm_bar,
                last_regime=last_regime,
                trace=trace,
                decision_id=decision_id,
                context=context,
                kill_reason=kill_reason,
            )
        return _run_policy_v1_decision_path(
            pipeline,
            market,
            policy_runtime,
            signal_id_counter=signal_id_counter,
            last_qm_bar=last_qm_bar,
            last_regime=last_regime,
            trace=trace,
            decision_id=decision_id,
        )

    _apply_pending_shadow_updates_to_strategist(pipeline)

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
    _apply_quarantine_recovery_override(
        pipeline,
        bar=market.bar,
        trace_id=trace,
    )
    candidates = _compose_candidates(pipeline, market.regime)
    shadow_summary = _run_shadow_tournament(pipeline, market, candidates)
    _capture_pending_shadow_updates(pipeline)
    _sync_current_actionable_labels_to_strategist(pipeline)

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
    _apply_quarantine_recovery_override(
        pipeline,
        bar=market.bar,
        trace_id=trace,
    )
    degradation_result = _apply_degradation_gate(
        pipeline,
        bar=market.bar,
        trace_id=trace,
    )
    _apply_quarantine_recovery_override(
        pipeline,
        bar=market.bar,
        trace_id=trace,
    )
    if degradation_result is not None and not degradation_result.is_no_op:
        candidates = _drop_quarantined_candidates(pipeline, candidates)

    decision_id = trace
    context = _decision_context(pipeline, market)
    health_reason = _exchange_health_open_block_reason(pipeline)
    kill_reason = _kill_switch_reason(pipeline, market)
    if kill_reason:
        return _run_manage_only_kill_switch_step(
            pipeline,
            market,
            shadow_summary,
            signal_id_counter=signal_id_counter,
            last_qm_bar=last_qm_bar,
            last_regime=last_regime,
            trace=trace,
            decision_id=decision_id,
            context=context,
            kill_reason=kill_reason,
        )

    use_flash = _hybrid_resolve_use_flash(pipeline, market)
    if use_flash:
        return _run_flash_decision_path(
            pipeline,
            market,
            candidates,
            shadow_summary,
            signal_id_counter=signal_id_counter,
            last_qm_bar=last_qm_bar,
            last_regime=last_regime,
            trace=trace,
            decision_id=decision_id,
            context=context,
            health_reason=health_reason,
        )

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
        switch_kwargs = {
            "current_bar": market.bar,
            "regime_confidence": float(getattr(market, "regime_confidence", 1.0) or 1.0),
        }
        market_tags = _market_policy_tags(market)
        if market_tags:
            switch_kwargs["market_tags"] = market_tags
        decision: SwitchDecision = pipeline.strategist.consider_switch(
            market.regime,
            **switch_kwargs,
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
    soft_allocator_payload: Dict[str, Any] = {}
    soft_allocator_soft_only = _soft_allocator_execution_soft_only(pipeline)
    vote_attempt = _soft_allocator_execution_attempt(
        pipeline,
        market,
        candidates,
        signal_id_counter=signal_id_counter,
    )
    if vote_attempt is not None:
        executed_leader = vote_attempt["leader"]
        soft_allocator_payload = dict(vote_attempt.get("soft_allocator", {}) or {})
    elif soft_allocator_soft_only:
        vote_attempt = _soft_allocator_idle_attempt(
            pipeline,
            signal_id_counter=signal_id_counter,
            reason="soft_only_no_executable_soft_signal",
        )
        executed_leader = vote_attempt["leader"]
        soft_allocator_payload = dict(vote_attempt.get("soft_allocator", {}) or {})
    else:
        vote_attempt = _vote_candidate_for_real_signals(
            pipeline,
            market,
            selected_leader,
            signal_id_counter=signal_id_counter,
            trace_id=trace,
        )
    if (
        selected_leader.label == "NoTrade"
        and vote_attempt["raw_signal_count"] == 0
        and not _soft_allocator_execution_prevents_notrade_cash_flat(pipeline)
    ):
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
    shadow_position_replay_reason = ""
    if (
        selected_leader.label != "NoTrade"
        and vote_attempt["raw_signal_count"] == 0
        and not soft_allocator_soft_only
    ):
        replay_attempt = _find_selected_shadow_position_replay(
            pipeline,
            market,
            selected_leader,
            signal_id_counter=signal_id_counter,
        )
        if replay_attempt is not None:
            vote_attempt = replay_attempt
            shadow_position_replay_reason = (
                "selected leader emitted no raw signals; fresh shadow position replay used"
            )
    fallback_used = False
    fallback_skipped = False
    fallback_candidate = ""
    fallback_reason = shadow_position_replay_reason
    if vote_attempt["raw_signal_count"] == 0:
        forced_no_trade_reason = str(getattr(decision, "reason", "") or "")
        forced_no_trade = forced_no_trade_reason.startswith("v3 panteon equity guard")
        fallback_enabled = (
            bool(getattr(pipeline, "actionable_fallback_enabled", False))
            and not forced_no_trade
            and not soft_allocator_soft_only
        )
        fallback_attempt = None
        if fallback_enabled:
            fallback_attempt = _find_actionable_candidate_from_shadow_registry(
                pipeline,
                market,
                candidates,
                decision,
                skip_label=selected_leader.label,
                signal_id_counter=signal_id_counter,
                trace_id=trace,
            )
        if fallback_attempt is not None:
            fallback_used = True
            executed_leader = fallback_attempt["leader"]
            fallback_candidate = executed_leader.label
            fallback_reason = (
                "selected leader emitted no raw signals; actionable fallback used"
            )
            vote_attempt = fallback_attempt
        else:
            fallback_skipped = any(
                candidate.label != selected_leader.label for candidate in candidates
            )
            if soft_allocator_soft_only:
                fallback_reason = (
                    "soft allocator soft-only mode: normal leader execution suppressed"
                )
            elif fallback_skipped:
                fallback_reason = (
                    "portfolio equity guard active; fallback disabled"
                    if forced_no_trade
                    else
                    "actionable fallback disabled: selected leader emitted no raw signals"
                    if not fallback_enabled
                    else "no actionable fallback candidate passed signal guard"
                )
    raw_signals = vote_attempt["raw_signals"]
    raw_signal_count = vote_attempt["raw_signal_count"]
    leader_vote_error_count = vote_attempt["leader_vote_error_count"]
    signal_guard = vote_attempt["signal_guard"]
    signal_guard = _apply_genetics_probation_execution_overlay(
        pipeline,
        market,
        executed_leader,
        signal_guard,
    )
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
            _record_order_failure(pipeline, "execute exception", signal=sig)
            n_rejected += 1
            continue
        if res.status == ExecutionStatus.FILLED:
            n_filled += 1
            _record_order_success(pipeline, sig, res)
            _record_realized_result_for_strategy(pipeline, res)
            # Update balance (simple): добавляем realized PnL если был close
            # (детальнее — в AttributionLedger; здесь упрощённо)
        elif res.status == ExecutionStatus.REJECTED:
            _record_order_failure(pipeline, res.reason or "rejected", signal=sig)
            n_rejected += 1
        elif res.status == ExecutionStatus.PENDING:
            _record_order_failure(pipeline, res.reason or "pending", signal=sig)
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
            causal_decision=_causal_decision_payload(
                market=market,
                selected_leader=selected_leader,
                executed_leader=executed_leader,
                decision=decision,
                raw_signals=raw_signals,
                executable_signals=signals,
                signal_guard=signal_guard,
                leader_vote_error_count=leader_vote_error_count,
                fallback_used=fallback_used,
                fallback_skipped=fallback_skipped,
                fallback_candidate=fallback_candidate,
                fallback_reason=fallback_reason,
                shadow_position_replay_used=bool(shadow_position_replay_reason),
                n_filled=n_filled,
                n_rejected=n_rejected,
                n_blocked=n_blocked,
                blocked_reasons=blocked_reasons,
                pipeline=pipeline,
                soft_allocator_payload=soft_allocator_payload,
            ),
            error=(
                f"kill switch active: {kill_reason_after_execution}"
                if kill_reason_after_execution else None
            ),
        ),
        signal_id_counter, last_qm_bar, last_regime,
    )


def _run_player_only_decision_path(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    signal_id_counter: int,
    last_qm_bar: int,
    last_regime: Optional[Regime],
    trace: str,
):
    """Short active path: regime -> one player -> execution -> shadow memory."""
    decision_id = trace
    context = _decision_context(pipeline, market)
    is_change = last_regime is not None and last_regime != market.regime
    pipeline.event_log.emit(RegimeDetected(
        bar=market.bar,
        trace_id=trace,
        regime=market.regime,
        from_regime=last_regime or market.regime,
        is_change=is_change,
    ))
    last_regime = market.regime

    all_players = _compose_strategy_players(pipeline)
    real_candidates = [
        player for player in all_players
        if _strategy_player_is_real_executable(player, pipeline)
    ]
    pipeline.event_log.emit(DecisionStarted(
        bar=market.bar,
        trace_id=trace,
        decision_id=decision_id,
        **context,
        candidate_labels=tuple(player.label for player in real_candidates),
        shadow_total_signals=0,
        shadow_total_filled=0,
        shadow_total_rejected=0,
        shadow_total_blocked=0,
    ))
    pipeline.strategist.update_candidates(real_candidates)
    try:
        decision: SwitchDecision = pipeline.strategist.consider_switch(
            market.regime,
            current_bar=market.bar,
            regime_confidence=float(getattr(market, "regime_confidence", 1.0) or 1.0),
            portfolio_flat=not _player_only_has_owned_positions(pipeline),
        )
    except Exception as exc:
        log.exception("player-only selection failed on bar %d", market.bar)
        shadow_summary = _run_shadow_tournament(pipeline, market, all_players)
        return (
            StepResult(
                bar=market.bar,
                regime=market.regime,
                leader=None,
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=0,
                n_shadow_signals=shadow_summary.total_signals,
                n_shadow_filled=shadow_summary.total_filled,
                n_shadow_rejected=shadow_summary.total_rejected,
                n_shadow_blocked=shadow_summary.total_blocked,
                n_shadow_actors=shadow_summary.actors,
                error=f"player-only selection failed: {type(exc).__name__}: {exc}",
            ),
            signal_id_counter,
            last_qm_bar,
            last_regime,
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
        current_label=decision.current_label,
        best_label=decision.best_label,
        selected_label=decision.new_leader.label,
        best_score=decision.best_score,
        current_score=decision.current_score,
        margin=decision.margin,
        required_margin=decision.required_margin,
        cooldown_passed=decision.cooldown_passed,
        cooldown_blocked=decision.cooldown_blocked,
        streak_count=decision.streak_count,
        streak_needed=decision.streak_needed,
        is_urgent=decision.is_urgent,
        switched=decision.switched,
        reason=decision.switch_gate_reason or decision.reason,
    ))
    if decision.switched:
        pipeline.event_log.emit(LeaderSelected(
            bar=market.bar,
            trace_id=trace,
            player_label=decision.new_leader.label,
            previous_label=(decision.previous.label if decision.previous else ""),
            score=decision.score,
            margin=decision.margin,
            is_urgent=decision.is_urgent,
            reason=decision.reason,
            decision_id=decision_id,
            **context,
        ))

    # Selection is intentionally complete before this bar mutates shadow
    # memory.  These outcomes become eligible only on the next bar.
    shadow_summary = _run_shadow_tournament(pipeline, market, all_players)
    kill_reason = _kill_switch_reason(pipeline, market)
    if kill_reason:
        return _run_manage_only_kill_switch_step(
            pipeline,
            market,
            shadow_summary,
            signal_id_counter=signal_id_counter,
            last_qm_bar=last_qm_bar,
            last_regime=last_regime,
            trace=trace,
            decision_id=decision_id,
            context=context,
            kill_reason=kill_reason,
        )

    leader = decision.new_leader
    player_manage_only = bool(getattr(decision, "manage_only", False))
    if leader.label == "NoTrade":
        raw_signals = _cash_flat_close_signals(
            pipeline,
            market,
            signal_id_start=signal_id_counter,
        )
        if raw_signals:
            signal_id_counter = max(signal.id for signal in raw_signals) + 1
        signal_guard = filter_real_signals_against_tracker(
            raw_signals,
            player=leader,
            pipeline=pipeline,
            bar_index=market.bar,
            max_new_opens_per_bar=0,
            max_open_positions=getattr(pipeline.risk_config, "max_open_positions", None),
        )
        vote_attempt = {
            "leader": leader,
            "raw_signals": raw_signals,
            "raw_signal_count": len(raw_signals),
            "leader_vote_error_count": 0,
            "signal_guard": signal_guard,
            "signal_id_counter": signal_id_counter,
        }
    else:
        shadow_signals = tuple(
            pipeline.shadow_tournament.last_player_signals().get(
                leader.label,
                (),
            )
        )
        raw_signals, signal_id_counter = _real_signals_from_shadow_registry(
            shadow_signals,
            market,
            leader,
            signal_id_counter=signal_id_counter,
        )
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
            manage_only=player_manage_only,
        )
        vote_attempt = {
            "leader": leader,
            "raw_signals": raw_signals,
            "raw_signal_count": len(raw_signals),
            "leader_vote_error_count": 0,
            "signal_guard": signal_guard,
            "signal_id_counter": signal_id_counter,
        }

    raw_signals = vote_attempt["raw_signals"]
    raw_signal_count = vote_attempt["raw_signal_count"]
    leader_vote_error_count = vote_attempt["leader_vote_error_count"]
    signal_guard = vote_attempt["signal_guard"]
    signal_id_counter = vote_attempt["signal_id_counter"]
    signals = list(signal_guard.signals)
    health_reason = _exchange_health_open_block_reason(pipeline)
    for signal in signals:
        signal_context = _signal_context(context, decision_id=decision_id, signal=signal)
        if signal.action.is_open and health_reason:
            signal_context["exchange_health_reason"] = health_reason
        pipeline.event_log.emit(SignalEmitted(
            bar=market.bar,
            trace_id=trace,
            signal=signal,
            **signal_context,
        ))
        pipeline.real_perf.record_signal(signal)

    n_filled = n_rejected = n_blocked = 0
    blocked_reasons: Dict[str, int] = {}
    for signal in signals:
        try:
            set_event_context = getattr(pipeline.executor, "set_event_context", None)
            if callable(set_event_context):
                set_event_context(_signal_context(
                    context,
                    decision_id=decision_id,
                    signal=signal,
                ))
            result: ExecutionResult = pipeline.executor.execute(
                signal,
                balance_usd=pipeline.current_balance,
            )
        except Exception:
            log.exception("execute failed for signal %d", signal.id)
            _record_order_failure(pipeline, "execute exception", signal=signal)
            n_rejected += 1
            continue
        if result.status == ExecutionStatus.FILLED:
            n_filled += 1
            _record_order_success(pipeline, signal, result)
            _record_realized_result_for_strategy(pipeline, result)
        elif result.status in {ExecutionStatus.REJECTED, ExecutionStatus.PENDING}:
            _record_order_failure(pipeline, result.reason or result.status.value, signal=signal)
            n_rejected += 1
        elif result.status == ExecutionStatus.BLOCKED:
            n_blocked += 1
            reason = result.reason or "blocked"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        if _kill_switch_reason(pipeline):
            break

    kill_reason_after_execution = _kill_switch_reason(pipeline)
    transition_reason = (
        decision.reason
        if player_manage_only
        else "selected player's fresh shadow fills routed to real guard"
    )
    return (
        StepResult(
            bar=market.bar,
            regime=market.regime,
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
            n_max_position_saturated_open_signals=signal_guard.max_position_saturated_opens,
            n_external_position_signals=signal_guard.external_position_signals,
            n_raw_signals=raw_signal_count,
            leader_vote_errors=leader_vote_error_count,
            leader_agent_labels=(),
            signal_filter_details=tuple(signal_guard.details),
            selected_leader=leader.label,
            executed_leader=leader.label,
            fallback_used=False,
            fallback_skipped=False,
            fallback_candidate="",
            fallback_reason=transition_reason,
            blocked_reasons=blocked_reasons,
            causal_decision=_causal_decision_payload(
                market=market,
                selected_leader=leader,
                executed_leader=leader,
                decision=decision,
                raw_signals=raw_signals,
                executable_signals=signals,
                signal_guard=signal_guard,
                leader_vote_error_count=leader_vote_error_count,
                fallback_used=False,
                fallback_skipped=False,
                fallback_candidate="",
                fallback_reason=transition_reason,
                shadow_position_replay_used=False,
                n_filled=n_filled,
                n_rejected=n_rejected,
                n_blocked=n_blocked,
                blocked_reasons=blocked_reasons,
                pipeline=pipeline,
                soft_allocator_payload={},
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


def _compose_strategy_players(pipeline: ProductionPipeline) -> List[StrategyPlayer]:
    players = [
        StrategyPlayer(label=agent.label, strategy=agent)
        for agent in sorted(
            pipeline.registry.all_agents(),
            key=lambda item: str(getattr(item, "label", "")),
        )
        if str(getattr(agent, "label", "") or "").strip()
    ]
    pipeline._player_catalog_labels = tuple(player.label for player in players)
    return players


def _strategy_player_is_real_executable(
    player: StrategyPlayer,
    pipeline: ProductionPipeline,
) -> bool:
    if (
        str(getattr(pipeline, "trade_mode", "multi")) == "singleton"
        and player.label == str(getattr(pipeline, "fixed_player_label", ""))
    ):
        return True
    strategy = player.strategy
    return (
        not bool(getattr(strategy, "shadow_only", False))
        and getattr(strategy, "live_trading_eligible", True) is not False
    )


def _player_only_has_owned_positions(pipeline: ProductionPipeline) -> bool:
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is None or not callable(getattr(tracker, "all_open", None)):
        return False
    return any(
        not is_external_position(position)
        for position in (tracker.all_open() or {}).values()
    )


def _policy_daily_loss_kill_switch_reason(
    pipeline: ProductionPipeline,
    policy_runtime: Any,
    market: MarketSnapshot,
) -> str:
    manifest = getattr(policy_runtime, "manifest", None)
    risk = getattr(manifest, "risk", None)
    limit = float(getattr(risk, "max_daily_loss_usd", 0.0) or 0.0)
    if limit <= 0.0:
        return ""
    initial = float(getattr(pipeline, "initial_capital", 0.0) or 0.0)
    current = float(getattr(pipeline, "current_balance", 0.0) or 0.0)
    if initial <= 0.0 or current <= 0.0:
        return ""
    update_guard = getattr(policy_runtime, "update_equity_guard", None)
    if callable(update_guard):
        return str(
            update_guard(
                now=market.timestamp,
                current_equity_usd=current,
                initial_equity_usd=initial,
            )
            or ""
        )
    loss = max(0.0, initial - current)
    if loss < limit:
        return ""
    return (
        "policy max daily loss exceeded: "
        f"loss=${loss:.2f}, limit=${limit:.2f}, current=${current:.2f}"
    )


def _run_policy_v1_decision_path(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    policy_runtime: Any,
    *,
    signal_id_counter: int,
    last_qm_bar: int,
    last_regime: Optional[Regime],
    trace: str,
    decision_id: str,
):
    """Run the sealed one-actor policy route without legacy selection state."""

    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    if tracker is None or exchange is None:
        return (
            StepResult(
                bar=market.bar,
                regime=market.regime,
                leader=getattr(policy_runtime, "label", "Policy"),
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=1,
                blocked_reasons={"policy_execution_stack_unavailable": 1},
                error="policy execution stack unavailable",
            ),
            signal_id_counter,
            last_qm_bar,
            last_regime,
        )

    health_reason = _exchange_health_open_block_reason(pipeline)
    fallback_daily_loss = max(
        0.0,
        float(getattr(pipeline, "initial_capital", 0.0) or 0.0)
        - float(getattr(pipeline, "current_balance", 0.0) or 0.0),
    )
    daily_loss_usd = float(
        getattr(
            policy_runtime,
            "current_daily_loss_usd",
            fallback_daily_loss,
        )
        or 0.0
    )
    try:
        policy_step = policy_runtime.evaluate(
            market,
            exchange=exchange,
            tracker=tracker,
            signal_id_start=signal_id_counter,
            exchange_healthy=not bool(health_reason),
            daily_loss_usd=daily_loss_usd,
        )
    except Exception as exc:
        log.exception("policy_v1 evaluation failed on poll bar %d", market.bar)
        reason = f"policy_runtime_error:{type(exc).__name__}:{exc}"
        return (
            StepResult(
                bar=market.bar,
                regime=market.regime,
                leader=policy_runtime.label,
                leader_changed=False,
                n_signals=0,
                n_filled=0,
                n_rejected=0,
                n_blocked=1,
                leader_agent_labels=(policy_runtime.manifest.actor,),
                selected_leader=policy_runtime.label,
                executed_leader=policy_runtime.label,
                blocked_reasons={reason: 1},
                causal_decision={
                    "decision_path": "policy_v1",
                    "policy_id": policy_runtime.manifest.policy_id,
                    "manifest_sha256": policy_runtime.manifest.manifest_sha256,
                    "status": "error",
                    "reason": reason,
                },
                error=reason,
            ),
            signal_id_counter,
            last_qm_bar,
            last_regime,
        )

    signal_id_counter = policy_step.next_signal_id
    policy_market = policy_step.policy_market
    effective_market = policy_market or market
    context = _decision_context(pipeline, effective_market)
    if policy_market is not None:
        is_change = last_regime is not None and last_regime != policy_market.regime
        pipeline.event_log.emit(RegimeDetected(
            bar=market.bar,
            trace_id=trace,
            regime=policy_market.regime,
            from_regime=last_regime or policy_market.regime,
            is_change=is_change,
        ))
        last_regime = policy_market.regime
        pipeline.event_log.emit(DecisionStarted(
            bar=market.bar,
            trace_id=trace,
            decision_id=decision_id,
            **context,
            candidate_labels=(policy_runtime.label,),
            shadow_total_signals=0,
            shadow_total_filled=0,
            shadow_total_rejected=0,
            shadow_total_blocked=0,
        ))

    actor = _FlashExecutionActor(
        label=policy_runtime.label,
        agents=(policy_runtime.actor,),
        agent_labels=(policy_runtime.manifest.actor,),
    )
    raw_signals = list(policy_step.signals)
    signal_guard = filter_real_signals_against_tracker(
        raw_signals,
        player=actor,
        pipeline=pipeline,
        bar_index=market.bar,
        max_new_opens_per_bar=policy_runtime.manifest.risk.max_open_positions,
        max_open_positions=policy_runtime.manifest.risk.max_open_positions,
        manage_only=False,
    )
    signals = signal_guard.signals
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

    policy_reasons = Counter(
        trace_item.primary_reason.value
        for trace_item in policy_step.decision_traces
        if trace_item.outcome.value == "NO_TRADE"
    )
    blocked_reasons: Dict[str, int] = dict(policy_reasons)
    n_filled = n_rejected = n_execution_blocked = 0
    for sig in signals:
        try:
            set_event_context = getattr(pipeline.executor, "set_event_context", None)
            if callable(set_event_context):
                signal_context = _signal_context(
                    context,
                    decision_id=decision_id,
                    signal=sig,
                )
                if sig.action.is_open and health_reason:
                    signal_context["exchange_health_reason"] = health_reason
                set_event_context(signal_context)
            result: ExecutionResult = pipeline.executor.execute(
                sig,
                balance_usd=pipeline.current_balance,
            )
        except Exception:
            log.exception("policy_v1 execute failed for signal %d", sig.id)
            _record_order_failure(pipeline, "policy_v1 execute exception", signal=sig)
            n_rejected += 1
            continue
        if result.status == ExecutionStatus.FILLED:
            n_filled += 1
            _record_order_success(pipeline, sig, result)
            _record_realized_result_for_strategy(pipeline, result)
        elif result.status in (ExecutionStatus.REJECTED, ExecutionStatus.PENDING):
            _record_order_failure(pipeline, result.reason or result.status.value, signal=sig)
            n_rejected += 1
        elif result.status == ExecutionStatus.BLOCKED:
            n_execution_blocked += 1
            reason = result.reason or "blocked"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        if _kill_switch_reason(pipeline):
            break

    kill_reason_after_execution = _kill_switch_reason(pipeline)
    quality_payload = [item.as_dict() for item in policy_step.quality]
    causal_decision = {
        "decision_path": "policy_v1",
        "policy_id": policy_runtime.manifest.policy_id,
        "manifest_sha256": policy_runtime.manifest.manifest_sha256,
        "manifest_path": policy_runtime.loaded_manifest.path,
        "target": policy_runtime.manifest.target.value,
        "actor": policy_runtime.manifest.actor,
        "status": policy_step.status,
        "reason": policy_step.reason,
        "poll_bar": market.bar,
        "policy_bar": (policy_market.bar if policy_market is not None else None),
        "timestamp": _timestamp_payload(effective_market.timestamp),
        "cadence_timestamp": _timestamp_payload(
            policy_market.cadence_timestamp if policy_market is not None else None
        ),
        "regime": effective_market.regime.label,
        "regime_confidence": _safe_float_or_zero(
            effective_market.regime_confidence
        ),
        "regimes_by_symbol": _regimes_by_symbol_payload(effective_market),
        "regime_features_by_symbol": _regime_features_by_symbol_payload(
            effective_market
        ),
        "prices": _float_mapping(effective_market.prices),
        "funding": _float_mapping(effective_market.funding),
        "candidate_count": policy_step.candidate_count,
        "policy_denied_count": policy_step.denied_count,
        "activation_traces": [item.as_dict() for item in policy_step.activation_traces],
        "policy_decisions": [item.as_dict() for item in policy_step.decision_traces],
        "market_quality": quality_payload,
        "raw_signal_count": len(raw_signals),
        "executable_signal_count": len(signals),
        "raw_signals": [_signal_payload(sig) for sig in raw_signals],
        "executable_signals": [_signal_payload(sig) for sig in signals],
        "signal_filter_details": tuple(signal_guard.details),
        "n_filled": n_filled,
        "n_rejected": n_rejected,
        "n_blocked": policy_step.denied_count + n_execution_blocked,
        "blocked_reasons": blocked_reasons,
        "kill_switch_reason": str(kill_reason_after_execution or ""),
    }
    selected_actors = {
        sig.sym: policy_runtime.manifest.actor
        for sig in signals
    }
    return (
        StepResult(
            bar=market.bar,
            regime=effective_market.regime,
            leader=policy_runtime.label,
            leader_changed=False,
            n_signals=len(signals),
            n_filled=n_filled,
            n_rejected=n_rejected,
            n_blocked=policy_step.denied_count + n_execution_blocked,
            n_filtered_real_signals=signal_guard.filtered,
            n_stale_close_signals=signal_guard.stale_closes,
            n_duplicate_open_signals=signal_guard.duplicate_opens,
            n_rate_limited_open_signals=signal_guard.rate_limited_opens,
            n_max_position_saturated_open_signals=(
                signal_guard.max_position_saturated_opens
            ),
            n_external_position_signals=signal_guard.external_position_signals,
            n_raw_signals=len(raw_signals),
            leader_agent_labels=(policy_runtime.manifest.actor,),
            selected_actors_by_symbol=selected_actors,
            signal_filter_details=tuple(signal_guard.details),
            selected_leader=policy_runtime.label,
            executed_leader=policy_runtime.label,
            blocked_reasons=blocked_reasons,
            causal_decision=causal_decision,
            error=(
                f"kill switch active: {kill_reason_after_execution}"
                if kill_reason_after_execution else None
            ),
        ),
        signal_id_counter,
        last_qm_bar,
        last_regime,
    )


def _run_manage_only_kill_switch_step(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    shadow_summary: object,
    *,
    signal_id_counter: int,
    last_qm_bar: int,
    last_regime: Optional[Regime],
    trace: str,
    decision_id: str,
    context: Dict[str, str],
    kill_reason: str,
):
    raw_signals = _cash_flat_close_signals(
        pipeline,
        market,
        signal_id_start=signal_id_counter,
    )
    if raw_signals:
        signal_id_counter = max(signal.id for signal in raw_signals) + 1
    actor = _FlashExecutionActor(
        label="ManageOnly",
        agents=(),
        agent_labels=("CashFlat",),
    )
    signal_guard = filter_real_signals_against_tracker(
        raw_signals,
        player=actor,
        pipeline=pipeline,
        bar_index=market.bar,
        max_new_opens_per_bar=0,
        max_open_positions=getattr(
            getattr(pipeline, "risk_config", None),
            "max_open_positions",
            None,
        ),
        manage_only=True,
    )
    signals = signal_guard.signals
    for sig in signals:
        signal_context = _signal_context(context, decision_id=decision_id, signal=sig)
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
                set_event_context(
                    _signal_context(context, decision_id=decision_id, signal=sig)
                )
            res: ExecutionResult = pipeline.executor.execute(
                sig,
                balance_usd=pipeline.current_balance,
            )
        except Exception:
            log.exception("manage-only execute failed for signal %d", sig.id)
            _record_order_failure(pipeline, "manage-only execute exception", signal=sig)
            n_rejected += 1
            continue
        if res.status == ExecutionStatus.FILLED:
            n_filled += 1
            _record_order_success(pipeline, sig, res)
            _record_realized_result_for_strategy(pipeline, res)
        elif res.status == ExecutionStatus.REJECTED:
            _record_order_failure(pipeline, res.reason or "rejected", signal=sig)
            n_rejected += 1
        elif res.status == ExecutionStatus.PENDING:
            _record_order_failure(pipeline, res.reason or "pending", signal=sig)
            n_rejected += 1
        elif res.status == ExecutionStatus.BLOCKED:
            n_blocked += 1
            reason = res.reason or "blocked"
            blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

    causal_decision = {
        "manage_only": True,
        "bar": int(getattr(market, "bar", 0) or 0),
        "timestamp": _timestamp_payload(getattr(market, "timestamp", None)),
        "regime": getattr(
            getattr(market, "regime", None),
            "label",
            str(getattr(market, "regime", "")),
        ),
        "selected_leader": "ManageOnly",
        "executed_leader": "ManageOnly",
        "decision_reason": f"manage-only kill switch active: {kill_reason}",
        "prices": _float_mapping(getattr(market, "prices", {}) or {}),
        "raw_signal_count": len(tuple(raw_signals or ())),
        "executable_signal_count": len(tuple(signals or ())),
        "raw_signals": [_signal_payload(sig) for sig in raw_signals or ()],
        "executable_signals": [_signal_payload(sig) for sig in signals or ()],
        "guarded_signals": [_signal_payload(sig) for sig in signals or ()],
        "filtered_real_signals": int(getattr(signal_guard, "filtered", 0) or 0),
        "signal_filter_details": tuple(
            str(item) for item in (getattr(signal_guard, "details", ()) or ())
        ),
        "n_filled": int(n_filled),
        "n_rejected": int(n_rejected),
        "n_blocked": int(n_blocked),
        "blocked_reasons": dict(blocked_reasons),
        "kill_switch_reason": str(kill_reason or ""),
    }
    return (
        StepResult(
            bar=market.bar,
            regime=market.regime,
            leader="ManageOnly",
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
            n_max_position_saturated_open_signals=(
                signal_guard.max_position_saturated_opens
            ),
            n_external_position_signals=signal_guard.external_position_signals,
            n_raw_signals=len(raw_signals),
            leader_agent_labels=actor.agent_labels,
            signal_filter_details=tuple(signal_guard.details),
            selected_leader="ManageOnly",
            executed_leader="ManageOnly",
            fallback_reason=f"manage-only kill switch active: {kill_reason}",
            blocked_reasons=blocked_reasons,
            causal_decision=causal_decision,
            error=f"kill switch active: {kill_reason}",
        ),
        signal_id_counter,
        last_qm_bar,
        last_regime,
    )


def _run_flash_decision_path(
    pipeline: ProductionPipeline,
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
):
    return run_flash_decision_path(
        pipeline,
        market,
        candidates,
        shadow_summary,
        signal_id_counter=signal_id_counter,
        last_qm_bar=last_qm_bar,
        last_regime=last_regime,
        trace=trace,
        decision_id=decision_id,
        context=context,
        health_reason=health_reason,
        callbacks=FlashDecisionPathCallbacks(
            fallback_candidate_safety_issue=_fallback_candidate_safety_issue,
            flash_real_agents=_flash_real_agents,
            flash_execution_actor_type=_FlashExecutionActor,
            sync_player_agents_to_real_positions=sync_player_agents_to_real_positions,
            flash_update_degradation_state_from_events=_flash_update_degradation_state_from_events,
            flash_shadow_player_signals_with_position_replay=(
                _flash_shadow_player_signals_with_position_replay
            ),
            flash_shadow_confirmation_scores=_flash_shadow_confirmation_scores,
            flash_degraded_signal_keys=_flash_degraded_signal_keys,
            genetics_probation_preselection_degraded_signal_keys=(
                _genetics_probation_preselection_degraded_signal_keys
            ),
            genetics_probation_preselection_admission_signal_keys=(
                _genetics_probation_preselection_admission_signal_keys
            ),
            flash_degraded_actor_keys=_flash_degraded_actor_keys,
            flash_degraded_open_symbols=_flash_degraded_open_symbols,
            flash_promoted_signal_keys=_flash_promoted_signal_keys,
            flash_open_position_sides_by_symbol=_flash_open_position_sides_by_symbol,
            flash_open_position_actor_keys_by_symbol=(
                _flash_open_position_actor_keys_by_symbol
            ),
            flash_previous_actor_key_from_decision=_flash_previous_actor_key_from_decision,
            flash_register_degradation_signal_keys=_flash_register_degradation_signal_keys,
            emit_flash_audit_events=_emit_flash_audit_events,
            flash_stop_loss_close_signals=_flash_stop_loss_close_signals,
            flash_stale_position_close_signals=_flash_stale_position_close_signals,
            flash_stale_position_exit_max_age_bars=_flash_stale_position_exit_max_age_bars,
            flash_stale_position_exit_requires_loss=_flash_stale_position_exit_requires_loss,
            genetics_probation_regime_exit_close_signals=(
                _genetics_probation_regime_exit_close_signals
            ),
            flash_partial_profit_lock_close_signals=(
                _flash_partial_profit_lock_close_signals
            ),
            flash_guard_actor=_flash_guard_actor,
            filter_real_signals_against_tracker=filter_real_signals_against_tracker,
            genetics_probation_execution_overlay=(
                _apply_genetics_probation_execution_overlay
            ),
            signal_context=_signal_context,
            record_order_failure=_record_order_failure,
            record_order_success=_record_order_success,
            record_realized_result_for_strategy=_record_realized_result_for_strategy,
            kill_switch_reason=_kill_switch_reason,
            flash_causal_decision_payload=_flash_causal_decision_payload,
            step_result_type=StepResult,
        ),
    )


def _flash_real_agents(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
) -> List[object]:
    registry = getattr(pipeline, "registry", None)
    all_agents = getattr(registry, "all_agents", None)
    if not callable(all_agents):
        return []
    out: List[object] = []
    for agent in all_agents():
        if (
            not _agent_is_real_executable(agent)
            and not _agent_is_genetics_probation_agent_executable(agent, pipeline, market)
            and not _agent_is_promotion_derived_agent_executable(agent, pipeline)
        ):
            continue
        if _flash_agent_safety_issue(pipeline, agent, market):
            continue
        out.append(agent)
    return out


def _agent_is_promotion_derived_agent_executable(
    agent: object,
    pipeline: ProductionPipeline | None,
) -> bool:
    label = str(getattr(agent, "label", "") or "").strip()
    if not label:
        return False
    allocator = getattr(pipeline, "flash_allocator", None)
    cfg = getattr(allocator, "config", None)
    if not bool(getattr(cfg, "promotion_derived_router_enabled", False)):
        return False
    allowed_aliases: set[str] = set()
    for raw_label in getattr(cfg, "promotion_derived_actor_labels", ()) or ():
        allowed_aliases.update(_flash_actor_label_family_aliases(raw_label))
    if not allowed_aliases:
        return False
    actor_aliases = _flash_actor_label_family_aliases(label)
    actor_aliases.update(_flash_actor_label_family_aliases(f"agent:{label}"))
    actor_aliases.update(_flash_actor_label_family_aliases(f"Solo_{label}"))
    return bool(actor_aliases & allowed_aliases)


def _flash_actor_label_family_aliases(raw: object) -> set[str]:
    text = str(raw or "").strip()
    if not text:
        return set()
    label = text.split(":", 1)[-1].strip()
    labels = {text, label}
    if label.startswith("V_"):
        labels.add(label[2:])
    if label.startswith("Solo_"):
        labels.add(label[len("Solo_"):])
    for item in tuple(labels):
        if item.startswith("V_"):
            labels.add(item[2:])
        if item.startswith("Solo_"):
            labels.add(item[len("Solo_"):])
    aliases: set[str] = set()
    for item in labels:
        clean = str(item or "").strip()
        if not clean:
            continue
        aliases.add(clean.lower())
        aliases.add(f"V_{clean}".lower())
        aliases.add(f"Solo_{clean}".lower())
        aliases.add(f"agent:{clean}".lower())
        aliases.add(f"ensemble:{clean}".lower())
        aliases.add(f"ensemble:Solo_{clean}".lower())
    return aliases


def _agent_is_genetics_probation_agent_executable(
    agent: object,
    pipeline: ProductionPipeline | None,
    market: MarketSnapshot | None = None,
) -> bool:
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return False
    label = str(getattr(agent, "label", "") or "")
    if not label or not label.startswith("Genetics"):
        return False
    labels = {
        str(item or "").strip()
        for item in (getattr(cfg, "genetics_probation_labels", ()) or ())
        if str(item or "").strip()
    }
    if label not in labels:
        return False
    allowed_regimes = {
        str(item or "").strip().lower()
        for item in (getattr(cfg, "genetics_probation_allowed_regimes", ()) or ())
        if str(item or "").strip()
    }
    if allowed_regimes and market is not None:
        regime_key = str(getattr(market.regime, "label", market.regime) or "").lower()
        if regime_key not in allowed_regimes:
            return False
    return True


def _flash_solo_candidate_for_agent(
    agent: object,
    regime: Regime,
) -> EnsemblePlayer:
    label = str(getattr(agent, "label", "") or "")
    return EnsemblePlayer(
        label=f"Solo_{label}",
        agents=[agent],
        weights={label: 1.0},
        voting=WeightedConsensus(),
        thresholds=ThresholdProfile(),
        affinity=regime,
    )


def _flash_agent_safety_issue(
    pipeline: ProductionPipeline,
    agent: object,
    market: MarketSnapshot,
) -> bool:
    label = str(getattr(agent, "label", "") or "")
    if not label:
        return True
    candidate = EnsemblePlayer(
        label=label,
        agents=[agent],
        weights={label: 1.0},
        voting=WeightedConsensus(),
        thresholds=ThresholdProfile(),
        affinity=market.regime,
    )
    strategist = getattr(pipeline, "strategist", None)
    validate = getattr(strategist, "_validate", None)
    if callable(validate):
        try:
            if validate(candidate) is not None:
                return True
        except Exception:
            log.debug("flash agent safety quarantine validation failed", exc_info=True)
            return True
    validate_hard_policy = getattr(strategist, "_validate_hard_policy", None)
    if callable(validate_hard_policy):
        try:
            issue = validate_hard_policy(
                candidate,
                market.regime,
                current_bar=market.bar,
                market_tags=_market_policy_tags(market),
            )
            if issue is not None and not _agent_is_genetics_probation_agent_executable(
                agent,
                pipeline,
                market,
            ):
                return True
        except Exception:
            log.debug("flash agent hard-policy validation failed", exc_info=True)
            return True
    validate_real_loss = getattr(strategist, "_validate_v3_real_loss", None)
    if callable(validate_real_loss):
        try:
            if validate_real_loss(
                candidate,
                market.regime,
                current_bar=market.bar,
            ) is not None:
                return True
        except Exception:
            log.debug("flash agent real-loss validation failed", exc_info=True)
            return True
    return False


def _flash_guard_actor(
    decisions: Sequence[FlashDecision],
    *,
    agents: Sequence[object],
    players: Sequence[EnsemblePlayer],
) -> _FlashExecutionActor:
    selected_labels = {
        str(getattr(decision, "selected_actor", "") or "")
        for decision in decisions
        if str(getattr(decision, "selected_actor", "") or "") != "NoTrade"
    }
    selected_agent_labels = {
        str(getattr(getattr(decision, "signal", None), "by_agent", "") or "")
        for decision in decisions
        if getattr(decision, "signal", None) is not None
    }
    selected_agent_labels.discard("")

    by_label = {str(getattr(agent, "label", "") or ""): agent for agent in agents}
    out: List[object] = []
    seen: set[str] = set()

    def add_agent(agent: object) -> None:
        label = str(getattr(agent, "label", "") or "")
        if not label or label in seen:
            return
        seen.add(label)
        out.append(agent)

    for label in sorted(selected_labels | selected_agent_labels):
        agent = by_label.get(label)
        if agent is not None:
            add_agent(agent)
    for player in players:
        if str(getattr(player, "label", "") or "") not in selected_labels:
            continue
        for agent in getattr(player, "agents", ()) or ():
            add_agent(agent)

    return _FlashExecutionActor(
        label="Panteon_Flash",
        agents=tuple(out),
        agent_labels=tuple(str(getattr(agent, "label", "") or "") for agent in out),
    )


def _flash_shadow_confirmation_scores(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    agents: Sequence[object],
    players: Sequence[EnsemblePlayer],
) -> Dict[object, tuple[float, int]]:
    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    if not bool(getattr(config, "shadow_confirmation_enabled", False)):
        return {}
    symbol_scoped = bool(getattr(config, "shadow_symbol_confirmation_enabled", False))
    if symbol_scoped:
        strategist = getattr(pipeline, "strategist", None)
        label_scorer = getattr(strategist, "shadow_soft_score_for_label", None)
        label_stats_scorer = getattr(strategist, "shadow_soft_stats_for_label", None)
        label_all_scorer = getattr(
            strategist,
            "shadow_soft_score_for_label_all_regimes",
            None,
        )
        label_all_stats_scorer = getattr(
            strategist,
            "shadow_soft_stats_for_label_all_regimes",
            None,
        )
        signal_scorer = getattr(strategist, "shadow_soft_score_for_signal", None)
        signal_all_scorer = getattr(
            strategist,
            "shadow_soft_score_for_signal_all_regimes",
            None,
        )
        quality_enabled = bool(getattr(config, "shadow_quality_confirmation_enabled", False))
        actor_fallback_enabled = bool(
            getattr(config, "shadow_actor_fallback_confirmation_enabled", False)
        )
        stats_scorer = getattr(strategist, "shadow_soft_stats_for_signal", None)
        all_stats_scorer = getattr(
            strategist,
            "shadow_soft_stats_for_signal_all_regimes",
            None,
        )
        if not callable(signal_scorer):
            return {}
        scores: Dict[object, tuple[float, int]] = {}
        symbols = tuple(sorted(str(symbol).upper() for symbol in market.prices))
        confirmation_actions = (
            Action.SPOT_BUY_HALF,
            Action.SPOT_BUY_FULL,
            Action.FUT_LONG_HALF,
            Action.FUT_LONG_FULL,
            Action.FUT_SHORT_HALF,
            Action.FUT_SHORT_FULL,
        )
        for agent in agents:
            label = str(getattr(agent, "label", "") or "")
            if not label:
                continue
            portfolio_scope = _flash_portfolio_shadow_actor(config, "agent", label)
            if actor_fallback_enabled and callable(label_scorer):
                try:
                    if quality_enabled and callable(label_stats_scorer):
                        if portfolio_scope and callable(label_all_stats_scorer):
                            scores[label] = label_all_stats_scorer(
                                f"Solo_{label}",
                                market.bar,
                            )  # type: ignore[assignment,arg-type]
                        else:
                            scores[label] = label_stats_scorer(
                                f"Solo_{label}",
                                market.regime,
                                market.bar,
                            )  # type: ignore[assignment,arg-type]
                    elif portfolio_scope and callable(label_all_scorer):
                        scores[label] = tuple(
                            label_all_scorer(f"Solo_{label}", market.bar)
                        )  # type: ignore[arg-type]
                    else:
                        scores[label] = tuple(
                            label_scorer(f"Solo_{label}", market.regime, market.bar)
                        )  # type: ignore[arg-type]
                except Exception:
                    log.debug(
                        "failed to score flash shadow agent fallback %s",
                        label,
                        exc_info=True,
                    )
            for symbol in symbols:
                try:
                    for action in confirmation_actions:
                        if quality_enabled and callable(stats_scorer):
                            if portfolio_scope and callable(all_stats_scorer):
                                scores[(label, symbol, action.name)] = all_stats_scorer(
                                    f"Solo_{label}",
                                    symbol,
                                    market.bar,
                                    action,
                                )  # type: ignore[assignment,arg-type]
                            else:
                                scores[(label, symbol, action.name)] = stats_scorer(
                                    f"Solo_{label}",
                                    market.regime,
                                    symbol,
                                    market.bar,
                                    action,
                                )  # type: ignore[assignment,arg-type]
                        elif portfolio_scope and callable(signal_all_scorer):
                            scores[(label, symbol, action.name)] = tuple(
                                signal_all_scorer(
                                    f"Solo_{label}",
                                    symbol,
                                    market.bar,
                                    action,
                                )
                            )  # type: ignore[arg-type]
                        else:
                            scores[(label, symbol, action.name)] = tuple(
                                signal_scorer(
                                    f"Solo_{label}",
                                    market.regime,
                                    symbol,
                                    market.bar,
                                    action,
                                )
                            )  # type: ignore[arg-type]
                except Exception:
                    log.debug(
                        "failed to score flash shadow agent %s/%s",
                        label,
                        symbol,
                        exc_info=True,
                    )
        for player in players:
            label = str(getattr(player, "label", "") or "")
            if not label:
                continue
            portfolio_scope = _flash_portfolio_shadow_actor(config, "ensemble", label)
            if actor_fallback_enabled and callable(label_scorer):
                try:
                    if quality_enabled and callable(label_stats_scorer):
                        if portfolio_scope and callable(label_all_stats_scorer):
                            scores[label] = label_all_stats_scorer(
                                label,
                                market.bar,
                            )  # type: ignore[assignment,arg-type]
                        else:
                            scores[label] = label_stats_scorer(
                                label,
                                market.regime,
                                market.bar,
                            )  # type: ignore[assignment,arg-type]
                    elif portfolio_scope and callable(label_all_scorer):
                        scores[label] = tuple(label_all_scorer(label, market.bar))  # type: ignore[arg-type]
                    else:
                        scores[label] = tuple(label_scorer(label, market.regime, market.bar))  # type: ignore[arg-type]
                except Exception:
                    log.debug(
                        "failed to score flash shadow player fallback %s",
                        label,
                        exc_info=True,
                    )
            for symbol in symbols:
                try:
                    for action in confirmation_actions:
                        if quality_enabled and callable(stats_scorer):
                            if portfolio_scope and callable(all_stats_scorer):
                                scores[(label, symbol, action.name)] = all_stats_scorer(
                                    label,
                                    symbol,
                                    market.bar,
                                    action,
                                )  # type: ignore[assignment,arg-type]
                            else:
                                scores[(label, symbol, action.name)] = stats_scorer(
                                    label,
                                    market.regime,
                                    symbol,
                                    market.bar,
                                    action,
                                )  # type: ignore[assignment,arg-type]
                        elif portfolio_scope and callable(signal_all_scorer):
                            scores[(label, symbol, action.name)] = tuple(
                                signal_all_scorer(label, symbol, market.bar, action)
                            )  # type: ignore[arg-type]
                        else:
                            scores[(label, symbol, action.name)] = tuple(
                                signal_scorer(
                                    label,
                                    market.regime,
                                    symbol,
                                    market.bar,
                                    action,
                                )
                            )  # type: ignore[arg-type]
                except Exception:
                    log.debug(
                        "failed to score flash shadow player %s/%s",
                        label,
                        symbol,
                        exc_info=True,
                    )
        if bool(getattr(config, "shadow_symbol_health_enabled", False)):
            _add_flash_shadow_symbol_health_scores(scores)
        return scores
    scorer = getattr(
        getattr(pipeline, "strategist", None),
        "shadow_soft_score_for_label",
        None,
    )
    all_scorer = getattr(
        getattr(pipeline, "strategist", None),
        "shadow_soft_score_for_label_all_regimes",
        None,
    )
    if not callable(scorer):
        return {}
    scores: Dict[object, tuple[float, int]] = {}
    for agent in agents:
        label = str(getattr(agent, "label", "") or "")
        if not label:
            continue
        try:
            if (
                _flash_portfolio_shadow_actor(config, "agent", label)
                and callable(all_scorer)
            ):
                scores[label] = tuple(all_scorer(f"Solo_{label}", market.bar))  # type: ignore[arg-type]
            else:
                scores[label] = tuple(scorer(f"Solo_{label}", market.regime, market.bar))  # type: ignore[arg-type]
        except Exception:
            log.debug("failed to score flash shadow agent %s", label, exc_info=True)
    for player in players:
        label = str(getattr(player, "label", "") or "")
        if not label:
            continue
        try:
            if (
                _flash_portfolio_shadow_actor(config, "ensemble", label)
                and callable(all_scorer)
            ):
                scores[label] = tuple(all_scorer(label, market.bar))  # type: ignore[arg-type]
            else:
                scores[label] = tuple(scorer(label, market.regime, market.bar))  # type: ignore[arg-type]
        except Exception:
            log.debug("failed to score flash shadow player %s", label, exc_info=True)
    return scores


def _add_flash_shadow_symbol_health_scores(scores: Dict[object, object]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    for key, raw in list(scores.items()):
        if not isinstance(key, tuple) or len(key) != 3:
            continue
        label, symbol, action = key
        if str(label) == "__symbol_health__":
            continue
        payload = _shadow_confirmation_payload(raw)
        if not payload:
            continue
        grouped.setdefault(
            (str(symbol or "").upper(), str(action or "").upper()),
            [],
        ).append(payload)

    for (symbol, action), payloads in grouped.items():
        aggregate = _aggregate_shadow_confirmation_payloads(payloads)
        if aggregate["closed_trades"] <= 0:
            continue
        scores[("__symbol_health__", symbol, action)] = aggregate


def _shadow_confirmation_payload(raw: object) -> dict[str, float]:
    if isinstance(raw, Mapping):
        score = _shadow_safe_float(raw.get("score"))
        closed = _shadow_safe_int(raw.get("closed_trades"))
        wins = _shadow_safe_int(raw.get("winning_trades"))
        mean_raw = raw.get("pnl_per_trade_mean_usd")
        mean = (
            _shadow_safe_float(mean_raw)
            if mean_raw is not None
            else score / float(closed)
            if closed > 0
            else 0.0
        )
        return {
            "score": score,
            "closed_trades": float(closed),
            "winning_trades": float(wins),
            "recent_downside_usd": _shadow_safe_float(raw.get("recent_downside_usd")),
            "pnl_per_trade_mean_usd": mean,
            "pnl_per_trade_std_usd": max(
                0.0,
                _shadow_safe_float(raw.get("pnl_per_trade_std_usd")),
            ),
        }
    try:
        score, closed = raw  # type: ignore[misc]
    except (TypeError, ValueError):
        return {}
    closed_int = _shadow_safe_int(closed)
    score_float = _shadow_safe_float(score)
    return {
        "score": score_float,
        "closed_trades": float(closed_int),
        "winning_trades": 0.0,
        "recent_downside_usd": abs(score_float) if score_float < 0.0 else 0.0,
        "pnl_per_trade_mean_usd": (
            score_float / float(closed_int) if closed_int > 0 else 0.0
        ),
        "pnl_per_trade_std_usd": 0.0,
    }


def _shadow_safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _shadow_safe_float(value: object) -> float:
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _aggregate_shadow_confirmation_payloads(
    payloads: Sequence[dict[str, float]],
) -> dict[str, object]:
    score = sum(float(item.get("score", 0.0) or 0.0) for item in payloads)
    closed = sum(max(0, int(item.get("closed_trades", 0) or 0)) for item in payloads)
    wins = sum(max(0, int(item.get("winning_trades", 0) or 0)) for item in payloads)
    recent_downside = sum(
        max(0.0, float(item.get("recent_downside_usd", 0.0) or 0.0))
        for item in payloads
    )
    if closed <= 0:
        mean = 0.0
        std = 0.0
    else:
        mean = (
            sum(
                float(item.get("pnl_per_trade_mean_usd", 0.0) or 0.0)
                * max(0, int(item.get("closed_trades", 0) or 0))
                for item in payloads
            )
            / float(closed)
        )
        if closed < 2:
            std = 0.0
        else:
            variance_numerator = 0.0
            for item in payloads:
                n = max(0, int(item.get("closed_trades", 0) or 0))
                if n <= 0:
                    continue
                item_mean = float(item.get("pnl_per_trade_mean_usd", 0.0) or 0.0)
                item_std = max(0.0, float(item.get("pnl_per_trade_std_usd", 0.0) or 0.0))
                variance_numerator += max(0, n - 1) * (item_std ** 2)
                variance_numerator += n * ((item_mean - mean) ** 2)
            std = math.sqrt(max(0.0, variance_numerator / float(closed - 1)))
    lcb = mean - (std / math.sqrt(float(closed))) if closed > 0 else 0.0
    return {
        "score": float(score),
        "closed_trades": int(closed),
        "winning_trades": int(min(wins, closed)),
        "losing_trades": int(max(0, closed - wins)),
        "win_rate_pct": (float(wins) / float(closed) * 100.0) if closed > 0 else 0.0,
        "recent_downside_usd": float(recent_downside),
        "pnl_per_trade_mean_usd": float(mean),
        "pnl_per_trade_std_usd": float(std),
        "pnl_per_trade_lcb_usd": float(lcb),
    }


def _flash_portfolio_shadow_actor(
    config: object,
    actor_type: str,
    label: str,
) -> bool:
    actors = {
        str(item).strip()
        for item in (getattr(config, "portfolio_actor_keys", ()) or ())
        if str(item or "").strip()
    }
    if not actors:
        return False
    clean_label = str(label or "").strip()
    return clean_label in actors or _flash_actor_key(actor_type, clean_label) in actors


def _flash_register_degradation_signal_keys(
    pipeline: ProductionPipeline,
    decisions: Sequence[FlashDecision],
) -> None:
    if not _flash_degradation_guard_enabled(pipeline):
        return
    signal_key_by_id = dict(getattr(pipeline, "_flash_signal_key_by_id", {}) or {})
    actor_key_by_id = dict(getattr(pipeline, "_flash_actor_key_by_id", {}) or {})
    for decision in decisions or ():
        signal = getattr(decision, "signal", None)
        if signal is None:
            continue
        key = _flash_signal_key_from_decision(decision)
        if not key:
            continue
        signal_id = int(getattr(signal, "id", 0) or 0)
        signal_key_by_id[signal_id] = key
        actor_key = _flash_actor_degradation_key(pipeline, decision, signal)
        if actor_key:
            actor_key_by_id[signal_id] = actor_key
    pipeline._flash_signal_key_by_id = signal_key_by_id
    pipeline._flash_actor_key_by_id = actor_key_by_id


def _flash_update_degradation_state_from_events(
    pipeline: ProductionPipeline,
    *,
    current_bar: int = 0,
) -> None:
    flash_update_degradation_state_from_events(
        pipeline,
        current_bar=current_bar,
    )


def _flash_actor_degradation_key(
    pipeline: ProductionPipeline,
    decision: FlashDecision,
    signal: Signal,
) -> str:
    actor_key = _flash_actor_key_from_decision(decision)
    if not actor_key:
        return ""
    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    if getattr(config, "degradation_actor_scope", "actor") != "actor_regime":
        return actor_key
    return f"{actor_key}|regime:{_flash_regime_degradation_key(signal.regime)}"


def _flash_regime_degradation_key(regime: Regime | str | int) -> str:
    return flash_regime_degradation_key(regime)


def _flash_open_position_sides_by_symbol(pipeline: ProductionPipeline) -> dict[str, str]:
    return flash_open_position_sides_by_symbol(
        tracker_positions_for_agent_sync(pipeline),
        is_external=is_external_position,
    )


def _flash_open_position_actor_keys_by_symbol(
    pipeline: ProductionPipeline,
) -> dict[str, tuple[str, ...]]:
    return flash_open_position_actor_keys_by_symbol(
        tracker_positions_for_agent_sync(pipeline),
        is_external=is_external_position,
    )


def _flash_event_log_tail(event_log: object, cursor: int) -> tuple[list[object], int]:
    return flash_event_log_tail(event_log, cursor)


def _flash_degraded_signal_keys(pipeline: ProductionPipeline) -> set[str]:
    return flash_degraded_signal_keys(
        pipeline,
        guard_enabled=_flash_degradation_guard_enabled(pipeline),
    )


def _flash_degraded_actor_keys(
    pipeline: ProductionPipeline,
    regime: Regime | None = None,
) -> set[str]:
    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    return flash_degraded_actor_keys(
        pipeline,
        guard_enabled=_flash_degradation_guard_enabled(pipeline),
        actor_guard_enabled=bool(
            getattr(config, "degradation_actor_guard_enabled", False)
        ),
        actor_scope=str(getattr(config, "degradation_actor_scope", "actor")),
        regime=regime,
    )


def _flash_degraded_open_symbols(
    pipeline: ProductionPipeline,
    market: MarketSnapshot | None = None,
) -> set[str]:
    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    degraded = set(flash_degraded_open_symbols(
        pipeline,
        guard_enabled=_flash_degradation_guard_enabled(pipeline),
        symbol_guard_enabled=bool(
            getattr(config, "degradation_symbol_guard_enabled", False)
        ),
    ))
    degraded.update(_real_universe_contract_sizing_degraded_symbols(pipeline, market))
    return degraded


def _real_universe_contract_sizing_degraded_symbols(
    pipeline: ProductionPipeline,
    market: MarketSnapshot | None,
) -> set[str]:
    if market is None:
        return set()
    cfg = getattr(pipeline, "live_execution", None)
    max_min_executable = float(
        getattr(cfg, "max_real_symbol_min_executable_notional_usd", 0.0) or 0.0
    )
    if max_min_executable <= 0.0:
        try:
            pipeline._real_universe_symbol_reject_reasons = {}
        except Exception:
            pass
        return set()

    risk_cfg = getattr(pipeline, "risk_config", None)
    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    quantize = getattr(exchange, "quantize_order_qty", None)
    if not callable(quantize):
        try:
            pipeline._real_universe_symbol_reject_reasons = {}
        except Exception:
            pass
        return set()

    balance = float(
        getattr(pipeline, "current_balance", 0.0)
        or getattr(pipeline, "initial_capital", 0.0)
        or 0.0
    )
    capital_fraction = float(getattr(risk_cfg, "capital_fraction", 0.0) or 0.0)
    max_upscale = float(getattr(risk_cfg, "max_min_notional_upscale", 1.0) or 1.0)
    max_notional = float(getattr(risk_cfg, "max_notional_usd", float("inf")) or float("inf"))
    cfg_min_notional = float(getattr(risk_cfg, "min_notional_usd", 0.0) or 0.0)
    floor_min_notional = bool(
        getattr(risk_cfg, "floor_to_exchange_min_notional", False)
    )
    base_notional = balance * capital_fraction
    blocked: set[str] = set()
    reasons: Dict[str, str] = {}
    for raw_symbol, raw_price in (getattr(market, "prices", {}) or {}).items():
        symbol = str(raw_symbol or "").upper()
        try:
            price = float(raw_price or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        if not symbol or price <= 0.0 or base_notional <= 0.0:
            continue
        exchange_min = 0.0
        getter = getattr(exchange, "get_min_notional", None)
        if callable(getter):
            try:
                exchange_min = float(getter(symbol) or 0.0)
            except Exception:
                exchange_min = 0.0
        effective_min = max(cfg_min_notional, exchange_min)
        approved_notional = base_notional
        if approved_notional < effective_min and floor_min_notional:
            upscale = effective_min / max(approved_notional, 1e-12)
            if upscale <= max_upscale and effective_min <= max_notional:
                approved_notional = effective_min
        signal = Signal(
            id=0,
            bar=int(getattr(market, "bar", 0) or 0),
            sym=symbol,
            action=Action.FUT_LONG_FULL,
            price=price,
            regime=getattr(market, "regime", Regime.NEUTRAL),
            by_player="Panteon_Flash",
            by_agent="contract_sizing_guard",
            timestamp=getattr(market, "timestamp", None),
        )
        try:
            quantized_qty = float(quantize(signal, approved_notional / price) or 0.0)
        except Exception as exc:
            blocked.add(symbol)
            reasons[symbol] = f"contract_metadata_unavailable:{type(exc).__name__}: {exc}"
            continue
        quantized_notional = quantized_qty * price
        if quantized_notional <= 0.0:
            blocked.add(symbol)
            reasons[symbol] = "min_executable_notional unavailable"
            continue
        upscale = quantized_notional / max(approved_notional, 1e-12)
        if (
            quantized_notional >= max_min_executable
            and quantized_notional > approved_notional + 1e-9
        ) or upscale > max_upscale:
            blocked.add(symbol)
            reasons[symbol] = (
                "min_executable_notional "
                f"${quantized_notional:.2f} exceeds approved "
                f"${approved_notional:.2f} ({upscale:.2f}x)"
            )
    try:
        pipeline._real_universe_symbol_reject_reasons = reasons
    except Exception:
        pass
    return blocked


def _flash_promoted_signal_keys(pipeline: ProductionPipeline) -> set[str]:
    return {
        str(key).strip()
        for key in (getattr(pipeline, "_flash_promoted_signal_keys", ()) or ())
        if str(key or "").strip()
    }


def _flash_degradation_guard_enabled(pipeline: ProductionPipeline) -> bool:
    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    return bool(getattr(config, "degradation_guard_enabled", False))


def _flash_signal_key_from_decision(decision: FlashDecision) -> str:
    signal = getattr(decision, "signal", None)
    if signal is None:
        return ""
    actor_key = _flash_actor_key_from_decision(decision)
    symbol = str(getattr(decision, "symbol", "") or getattr(signal, "sym", "") or "").upper()
    action = getattr(decision, "action", getattr(signal, "action", Action.HOLD))
    action_name = getattr(action, "name", str(action or "")).upper()
    if not actor_key or not symbol or not action_name:
        return ""
    return f"{actor_key}|{symbol}|{action_name}"


def _flash_actor_key_from_decision(
    decision: FlashDecision,
    *,
    prefer_original: bool = False,
) -> str:
    if prefer_original:
        selected_actor = str(
            getattr(decision, "original_selected_actor", "") or ""
        )
        actor_type = str(getattr(decision, "original_actor_type", "") or "")
        if not selected_actor:
            selected_actor = str(getattr(decision, "selected_actor", "") or "")
        if not actor_type:
            actor_type = str(getattr(decision, "actor_type", "") or "")
    else:
        selected_actor = str(getattr(decision, "selected_actor", "") or "")
        actor_type = str(getattr(decision, "actor_type", "") or "")
    for row in getattr(decision, "candidates", ()) or ():
        if getattr(row, "rejected", False):
            continue
        if str(getattr(row, "label", "") or "") != selected_actor:
            continue
        if str(getattr(row, "actor_type", "") or "") != actor_type:
            continue
        actor_key = str(getattr(row, "actor_key", "") or "")
        if actor_key:
            return actor_key
    return _flash_actor_key(actor_type, selected_actor)


def _flash_previous_actor_key_from_decision(decision: FlashDecision) -> str:
    return _flash_actor_key_from_decision(
        decision,
        prefer_original=(
            getattr(decision, "signal", None) is None
            and bool(str(getattr(decision, "original_selected_actor", "") or ""))
        ),
    )


def _flash_actor_key(actor_type: str, label: str) -> str:
    if label == "NoTrade":
        return "NoTrade"
    clean_type = str(actor_type or "unknown").strip() or "unknown"
    clean_label = str(label or "").strip()
    return f"{clean_type}:{clean_label}"


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
    candidates.extend(_compose_regime_switch_agent_candidates(pipeline, regime, candidates))
    candidates.extend(_compose_fixed_agent_candidates(pipeline, candidates))
    candidates.extend(_compose_rotating_agent_candidates(pipeline, regime, candidates))
    candidates.extend(_compose_solo_agent_candidates(pipeline, regime, candidates))
    return [
        candidate
        for candidate in candidates
        if _candidate_is_real_executable(candidate, pipeline, regime=regime)
    ]


def _compose_regime_switch_agent_candidates(
    pipeline: ProductionPipeline,
    regime: Regime,
    existing: Sequence[EnsemblePlayer],
) -> List[EnsemblePlayer]:
    specs = tuple(getattr(pipeline, "regime_switch_player_sets", ()) or ())
    if not specs:
        return []
    registry = getattr(pipeline, "registry", None)
    if registry is None:
        return []
    qm = getattr(pipeline, "qm", None)
    existing_labels = {player.label for player in existing}
    out: List[EnsemblePlayer] = []
    for raw_spec in specs:
        label, regime_agents = _regime_switch_player_spec_parts(raw_spec)
        agent_labels = regime_agents.get(regime.label, ())
        if not label or not agent_labels or label in existing_labels:
            continue
        if qm is not None and qm.is_quarantined(label):
            continue
        valid_agents = []
        valid_labels = []
        for agent_label in agent_labels:
            if qm is not None and qm.is_quarantined(agent_label):
                continue
            agent = registry.get(agent_label)
            if agent is None:
                continue
            valid_agents.append(agent)
            valid_labels.append(agent.label)
        if not valid_agents:
            continue
        if len(valid_agents) == 1:
            agent = valid_agents[0]
            out.append(EnsemblePlayer(
                label=label,
                agents=[agent],
                weights={agent.label: 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
                affinity=regime,
            ))
        else:
            ordered_labels = _rank_regime_switch_agent_labels(
                pipeline,
                regime,
                tuple(valid_labels),
            )
            out.append(RotatingAgentPlayer(
                label=label,
                agents=valid_agents,
                regime_agent_order={regime.label: ordered_labels},
                fallback_agent_order=(),
                affinity=regime,
            ))
        existing_labels.add(label)
    return out


def _regime_switch_player_spec_parts(raw_spec: object) -> Tuple[str, Dict[str, Tuple[str, ...]]]:
    if isinstance(raw_spec, dict):
        label = str(raw_spec.get("label") or "").strip()
        raw_mapping = raw_spec.get("regime_agents") or raw_spec.get("agents") or {}
    else:
        try:
            parts = tuple(raw_spec)  # type: ignore[arg-type]
        except TypeError:
            return "", {}
        if len(parts) < 2:
            return "", {}
        label = str(parts[0] or "").strip()
        raw_mapping = parts[1]
    if not isinstance(raw_mapping, dict):
        return label, {}
    aliases = {
        "bull": "bullish",
        "bear": "bearish",
        "beearish": "bearish",
        "downtrend": "bearish",
        "uptrend": "bullish",
        "flat": "neutral",
        "sideways": "neutral",
        "panic": "crash",
    }
    regime_agents: Dict[str, str] = {}
    for raw_regime, raw_agent in raw_mapping.items():
        key = str(raw_regime or "").strip().lower()
        key = aliases.get(key, key)
        agent_labels = _agent_label_tuple(raw_agent)
        if key and agent_labels:
            regime_agents[key] = agent_labels
    return label, regime_agents


def _agent_label_tuple(raw_agent: object) -> Tuple[str, ...]:
    if isinstance(raw_agent, str):
        values = (raw_agent,)
    else:
        try:
            values = tuple(raw_agent)  # type: ignore[arg-type]
        except TypeError:
            values = (raw_agent,)
    labels: List[str] = []
    for value in values:
        label = str(value or "").strip()
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _rank_regime_switch_agent_labels(
    pipeline: ProductionPipeline,
    regime: Regime,
    agent_labels: Tuple[str, ...],
) -> Tuple[str, ...]:
    selector = getattr(pipeline, "selector", None)
    scorer = getattr(selector, "score_registered", None)
    if not callable(scorer):
        return agent_labels
    ranked: List[Tuple[bool, float, float, int, int, int, str]] = []
    for index, agent_label in enumerate(agent_labels):
        row = None
        try:
            row = scorer(
                agent_label,
                regime,
                include_quarantined=False,
            )
        except TypeError:
            try:
                row = scorer(agent_label, regime)
            except Exception:
                row = None
        except Exception:
            row = None
        metrics = getattr(row, "metrics", None)
        closed = int(getattr(metrics, "closed_trades", 0) or 0)
        has_data = (
            bool(getattr(metrics, "has_data", False))
            and closed >= _REGIME_SWITCH_ADAPTIVE_MIN_CLOSED_TRADES
        )
        score = _safe_float_or_zero(getattr(row, "score", 0.0))
        pnl_pct = _safe_float_or_zero(getattr(metrics, "pnl_pct", 0.0))
        signals = int(getattr(metrics, "signals", 0) or 0)
        ranked.append((has_data, pnl_pct, score, closed, signals, index, agent_label))
    if not any(item[0] for item in ranked):
        return agent_labels
    ranked.sort(
        key=lambda item: (
            not item[0],
            -item[1],
            -item[2],
            -item[3],
            -item[4],
            item[5],
        )
    )
    return tuple(item[6] for item in ranked)


def _compose_fixed_agent_candidates(
    pipeline: ProductionPipeline,
    existing: Sequence[EnsemblePlayer],
) -> List[EnsemblePlayer]:
    specs = tuple(getattr(pipeline, "fixed_agent_player_sets", ()) or ())
    if not specs:
        return []
    registry = getattr(pipeline, "registry", None)
    if registry is None:
        return []
    qm = getattr(pipeline, "qm", None)
    existing_labels = {player.label for player in existing}
    fixed: List[EnsemblePlayer] = []
    for raw_spec in specs:
        label, agent_labels = _fixed_agent_player_spec_parts(raw_spec)
        if not label or not agent_labels or label in existing_labels:
            continue
        if qm is not None and qm.is_quarantined(label):
            continue
        agents = []
        invalid = False
        seen_agents = set()
        for agent_label in agent_labels:
            if agent_label in seen_agents:
                invalid = True
                break
            seen_agents.add(agent_label)
            if qm is not None and qm.is_quarantined(agent_label):
                invalid = True
                break
            agent = registry.get(agent_label)
            if agent is None:
                invalid = True
                break
            agents.append(agent)
        if invalid or not agents:
            continue
        weight = 1.0 / len(agents)
        fixed.append(EnsemblePlayer(
            label=label,
            agents=agents,
            weights={agent.label: weight for agent in agents},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            affinity=None,
        ))
        existing_labels.add(label)
    return fixed


def _compose_rotating_agent_candidates(
    pipeline: ProductionPipeline,
    regime: Regime,
    existing: Sequence[EnsemblePlayer],
) -> List[RotatingAgentPlayer]:
    specs = tuple(getattr(pipeline, "rotating_agent_player_sets", ()) or ())
    if not specs:
        return []
    registry = getattr(pipeline, "registry", None)
    if registry is None:
        return []
    qm = getattr(pipeline, "qm", None)
    existing_labels = {player.label for player in existing}
    out: List[RotatingAgentPlayer] = []
    for raw_spec in specs:
        label, regime_orders, fallback_order = _rotating_agent_player_spec_parts(raw_spec)
        if not label or label in existing_labels:
            continue
        if qm is not None and qm.is_quarantined(label):
            continue
        agent_labels = []
        for labels in regime_orders.values():
            agent_labels.extend(labels)
        agent_labels.extend(fallback_order)
        filtered_labels: List[str] = []
        seen = set()
        for agent_label in agent_labels:
            if agent_label in seen:
                continue
            seen.add(agent_label)
            if qm is not None and qm.is_quarantined(agent_label):
                continue
            if registry.get(agent_label) is None:
                continue
            filtered_labels.append(agent_label)
        if not filtered_labels:
            continue
        agents = [registry.get(agent_label) for agent_label in filtered_labels]
        valid_agents = [agent for agent in agents if agent is not None]
        if not valid_agents:
            continue
        valid_set = {agent.label for agent in valid_agents}
        filtered_orders = {
            key: tuple(agent_label for agent_label in labels if agent_label in valid_set)
            for key, labels in regime_orders.items()
        }
        filtered_fallback = tuple(
            agent_label for agent_label in fallback_order if agent_label in valid_set
        )
        if not filtered_orders.get(regime.label) and not filtered_fallback:
            continue
        out.append(RotatingAgentPlayer(
            label=label,
            agents=valid_agents,
            regime_agent_order=filtered_orders,
            fallback_agent_order=filtered_fallback,
            affinity=regime,
        ))
        existing_labels.add(label)
    return out


def _rotating_agent_player_spec_parts(
    raw_spec: object,
) -> Tuple[str, Dict[str, Tuple[str, ...]], Tuple[str, ...]]:
    if isinstance(raw_spec, dict):
        label = str(raw_spec.get("label") or "").strip()
        raw_mapping = raw_spec.get("regime_agent_order") or raw_spec.get("agents") or {}
        raw_fallback = raw_spec.get("fallback_agent_order") or raw_spec.get("fallback") or ()
    else:
        try:
            parts = tuple(raw_spec)  # type: ignore[arg-type]
        except TypeError:
            return "", {}, ()
        if len(parts) < 2:
            return "", {}, ()
        label = str(parts[0] or "").strip()
        raw_mapping = parts[1]
        raw_fallback = parts[2] if len(parts) >= 3 else ()
    if not isinstance(raw_mapping, dict):
        return label, {}, ()
    aliases = {
        "bull": "bullish",
        "bear": "bearish",
        "beearish": "bearish",
        "downtrend": "bearish",
        "uptrend": "bullish",
        "flat": "neutral",
        "sideways": "neutral",
        "panic": "crash",
    }
    orders: Dict[str, Tuple[str, ...]] = {}
    for raw_regime, raw_agents in raw_mapping.items():
        key = str(raw_regime or "").strip().lower()
        key = aliases.get(key, key)
        agents = _agent_label_sequence(raw_agents)
        if key and agents:
            orders[key] = agents
    return label, orders, _agent_label_sequence(raw_fallback)


def _agent_label_sequence(raw_agents: object) -> Tuple[str, ...]:
    if isinstance(raw_agents, str):
        return tuple(part.strip() for part in raw_agents.split(",") if part.strip())
    try:
        return tuple(str(part).strip() for part in raw_agents if str(part).strip())  # type: ignore[union-attr]
    except TypeError:
        return ()


def _fixed_agent_player_spec_parts(raw_spec: object) -> Tuple[str, Tuple[str, ...]]:
    if isinstance(raw_spec, dict):
        label = str(raw_spec.get("label") or "").strip()
        raw_agents = raw_spec.get("agent_labels") or raw_spec.get("agents") or ()
    else:
        try:
            parts = tuple(raw_spec)  # type: ignore[arg-type]
        except TypeError:
            return "", ()
        if len(parts) < 2:
            return "", ()
        label = str(parts[0] or "").strip()
        raw_agents = parts[1]
    if isinstance(raw_agents, str):
        agent_labels = tuple(
            part.strip() for part in raw_agents.split(",") if part.strip()
        )
    else:
        try:
            agent_labels = tuple(
                str(part).strip() for part in raw_agents if str(part).strip()
            )
        except TypeError:
            return label, ()
    return label, agent_labels


def _compose_solo_agent_candidates(
    pipeline: ProductionPipeline,
    regime: Regime,
    existing: List[EnsemblePlayer],
) -> List[EnsemblePlayer]:
    """Expose top solo shadow performers as real player candidates."""
    existing_labels = {player.label for player in existing}
    solo: List[EnsemblePlayer] = []
    try:
        limit = int(
            getattr(
                pipeline,
                "solo_agent_candidate_limit",
                _DEFAULT_SOLO_AGENT_CANDIDATE_LIMIT,
            )
            or _DEFAULT_SOLO_AGENT_CANDIDATE_LIMIT
        )
        select_k = limit
        registry = getattr(pipeline, "registry", None)
        all_agents = getattr(registry, "all_agents", None)
        if callable(all_agents):
            try:
                select_k = max(limit, len(tuple(all_agents())))
            except Exception:
                select_k = limit
        scored = pipeline.selector.select(regime, k=select_k)
        if not scored and hasattr(pipeline.selector, "select_with_fallback"):
            scored = pipeline.selector.select_with_fallback(
                regime,
                k=select_k,
                fallback_threshold=-10.0,
                min_count=1,
            )
    except Exception:
        log.exception("solo agent candidate selection failed")
        return solo
    for row in scored:
        if float(getattr(row.metrics, "pnl_pct", 0.0) or 0.0) <= 0.0:
            continue
        if not _agent_is_real_executable(row.agent):
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
        if len(solo) >= limit:
            break
    return solo


def _agent_is_real_executable(agent: object) -> bool:
    if bool(getattr(agent, "shadow_only", False)):
        return False
    if getattr(agent, "live_trading_eligible", True) is False:
        return False
    return True


def _candidate_is_real_executable(
    candidate: object,
    pipeline: ProductionPipeline | None = None,
    *,
    regime: Regime | str | None = None,
) -> bool:
    if all(
        _agent_is_real_executable(agent)
        for agent in (getattr(candidate, "agents", ()) or ())
    ):
        return True
    return _candidate_is_genetics_probation_executable(
        candidate,
        pipeline,
        regime=regime,
    )


def _candidate_is_genetics_probation_executable(
    candidate: object,
    pipeline: ProductionPipeline | None,
    *,
    regime: Regime | str | None = None,
) -> bool:
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return False
    label = str(getattr(candidate, "label", "") or "")
    labels = {
        str(item or "").strip()
        for item in (getattr(cfg, "genetics_probation_labels", ()) or ())
        if str(item or "").strip()
    }
    if not label or label not in labels or not label.startswith("Genetics"):
        return False
    agents = tuple(getattr(candidate, "agents", ()) or ())
    if len(agents) != 1:
        return False
    agent_label = str(getattr(agents[0], "label", "") or "")
    if agent_label != label:
        return False
    allowed_regimes = {
        item.lower()
        for item in _string_tuple(
            getattr(cfg, "genetics_probation_allowed_regimes", ()),
        )
    }
    if allowed_regimes and regime is not None:
        return _regime_allowed_by_keys(regime, allowed_regimes)
    return True


def _regime_key(regime: Regime | str | object) -> str:
    return str(getattr(regime, "label", regime) or "").strip().lower()


def _regime_allowed_by_keys(
    regime: Regime | str | object,
    allowed_regimes: set[str],
) -> bool:
    key = _regime_key(regime)
    if key in allowed_regimes:
        return True
    compatible = {
        "neutral": {"range_low_vol", "mixed_rotational"},
        "range_low_vol": {"neutral"},
        "mixed_rotational": {"neutral"},
        "bullish": {"choppy_up"},
        "choppy_up": {"bullish"},
        "bearish": {"choppy_down"},
        "choppy_down": {"bearish"},
    }
    return bool(compatible.get(key, set()) & set(allowed_regimes or set()))


def _genetics_probation_label_map_value(
    cfg: object,
    map_attr: str,
    scalar_attr: str,
    label: str,
    default: object,
    cast: Callable[[object], object],
) -> object:
    fallback = getattr(cfg, scalar_attr, default)
    raw = getattr(cfg, map_attr, {}) or {}
    value = None
    if isinstance(raw, Mapping):
        value = raw.get(str(label or "").strip())
    if value is None:
        value = fallback
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(default)


def _genetics_probation_int_for_label(
    cfg: object,
    map_attr: str,
    scalar_attr: str,
    label: str,
    default: int,
) -> int:
    return max(
        0,
        int(
            _genetics_probation_label_map_value(
                cfg,
                map_attr,
                scalar_attr,
                label,
                default,
                int,
            )
        ),
    )


def _genetics_probation_float_for_label(
    cfg: object,
    map_attr: str,
    scalar_attr: str,
    label: str,
    default: float,
) -> float:
    return float(
        _genetics_probation_label_map_value(
            cfg,
            map_attr,
            scalar_attr,
            label,
            default,
            float,
        )
    )


def _candidate_uses_shadow_state_entry_gate(candidate: object) -> bool:
    label = str(getattr(candidate, "label", "") or "")
    return label.startswith(_SHADOW_STATE_PLAYER_PREFIXES)


def _filter_shadow_state_entry_signals(
    pipeline: ProductionPipeline,
    player: EnsemblePlayer,
    raw_signals: Sequence[Signal],
) -> List[Signal]:
    signals = list(raw_signals or ())
    if not signals or not _candidate_uses_shadow_state_entry_gate(player):
        return signals
    player_label = str(getattr(player, "label", "") or "")
    kept: List[Signal] = []
    for signal in signals:
        action = getattr(signal, "action", None)
        if action is None or not getattr(action, "is_open", False):
            kept.append(signal)
            continue
        if str(getattr(signal, "by_agent", "") or "") == _SHADOW_POSITION_REPLAY_AGENT:
            kept.append(signal)
            continue
        if _shadow_position_confirms_positive_open(pipeline, player_label, signal):
            kept.append(signal)
    return kept


def _shadow_position_confirms_positive_open(
    pipeline: ProductionPipeline,
    player_label: str,
    signal: Signal,
) -> bool:
    side = str(getattr(getattr(signal, "action", None), "side", "") or "").lower()
    symbol = str(getattr(signal, "sym", "") or "").upper()
    if not symbol or side not in {"long", "short"}:
        return False
    positions_by_player = dict(
        getattr(pipeline, "_pending_shadow_player_positions", {}) or {}
    )
    max_age_bars = _shadow_position_replay_max_age(pipeline)
    for payload in positions_by_player.get(str(player_label), ()) or ():
        if not isinstance(payload, dict):
            continue
        if str(payload.get("sym") or "").upper() != symbol:
            continue
        if str(payload.get("side") or "").lower() != side:
            continue
        age_bars = _safe_nonnegative_int(payload.get("age_bars"))
        if age_bars is not None and age_bars > max_age_bars:
            continue
        unrealized = _safe_float_or_none(payload.get("unrealized_pnl_usd"))
        if unrealized is None or unrealized <= 0.0:
            continue
        return True
    return False


def _vote_candidate_for_real_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    leader: Any,
    *,
    signal_id_counter: int,
    trace_id: str,
    manage_only: bool = False,
) -> Dict[str, Any]:
    sync_player_agents_to_real_positions(
        leader,
        pipeline,
        bar_index=market.bar,
        market_symbols=market.prices.keys(),
    )
    try:
        raw_signals, vote_errors = normalize_vote_result(
            leader.vote(market, signal_id_start=signal_id_counter)
        )
    except Exception as exc:
        _record_player_vote_failure(pipeline, market, leader.label, exc, trace_id=trace_id)
        raw_signals = []
        vote_errors = []
    if bool(getattr(pipeline, "player_only_runtime", False)):
        for error in vote_errors:
            _record_player_vote_failure(
                pipeline,
                market,
                leader.label,
                RuntimeError(str(getattr(error, "reason", "player vote failed"))),
                trace_id=trace_id,
            )
    else:
        _record_agent_vote_failures(
            pipeline,
            market,
            leader,
            trace_id=trace_id,
            errors=vote_errors,
        )
    raw_signals = _filter_shadow_state_entry_signals(pipeline, leader, raw_signals)
    raw_signal_count = len(raw_signals)
    leader_vote_error_count = len(vote_errors)
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
        manage_only=manage_only,
    )
    return {
        "leader": leader,
        "raw_signals": raw_signals,
        "raw_signal_count": raw_signal_count,
        "leader_vote_error_count": leader_vote_error_count,
        "signal_guard": signal_guard,
        "signal_id_counter": next_signal_id,
    }


def _apply_genetics_probation_execution_overlay(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    leader: EnsemblePlayer,
    signal_guard,
):
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return signal_guard
    label = str(getattr(leader, "label", "") or "")
    labels = _string_tuple(
        getattr(cfg, "genetics_probation_labels", ("GeneticsResearch",)),
    )
    label_set = set(labels)
    if not label_set:
        return signal_guard
    allowed_regimes = {
        item.lower()
        for item in _string_tuple(
            getattr(cfg, "genetics_probation_allowed_regimes", ("bearish", "crash")),
        )
    }
    min_regime_confidence = max(
        0.0,
        min(
            1.0,
            float(getattr(cfg, "genetics_probation_min_regime_confidence", 0.0) or 0.0),
        ),
    )
    require_shadow_confirmation = bool(
        getattr(cfg, "genetics_probation_require_shadow_confirmation", True)
    )
    allowed_signal_keys = {
        str(item or "").strip()
        for item in getattr(cfg, "genetics_probation_allowed_signal_keys", ()) or ()
        if str(item or "").strip()
    }

    kept: List[Signal] = []
    dropped = 0
    details = list(getattr(signal_guard, "details", []) or [])
    changed = False
    for signal in getattr(signal_guard, "signals", []) or []:
        probation_label = _genetics_probation_signal_label(
            leader_label=label,
            signal=signal,
            labels=label_set,
        )
        if not probation_label:
            kept.append(signal)
            continue
        if not signal.action.is_open:
            kept.append(signal)
            continue
        sym = str(signal.sym).upper()
        agent_label = str(signal.by_agent or probation_label or "-")
        max_real_trades = _genetics_probation_int_for_label(
            cfg,
            "genetics_probation_max_real_trades_by_label",
            "genetics_probation_max_real_trades",
            probation_label,
            20,
        )
        max_daily_trades = _genetics_probation_int_for_label(
            cfg,
            "genetics_probation_max_daily_trades_by_label",
            "genetics_probation_max_daily_trades",
            probation_label,
            0,
        )
        max_risk_mult = max(
            0.0,
            min(
                1.0,
                _genetics_probation_float_for_label(
                    cfg,
                    "genetics_probation_risk_mult_by_label",
                    "genetics_probation_risk_mult",
                    probation_label,
                    0.25,
                ),
            ),
        )
        disabled_reason = _genetics_probation_disabled_reason(
            pipeline,
            probation_label,
        )
        try:
            symbol_regime = market.regime_for_symbol(sym)
        except Exception:
            symbol_regime = getattr(market, "regime", "")
        regime_key = _regime_key(symbol_regime)
        regime_allowed = (
            not allowed_regimes
            or _regime_allowed_by_keys(regime_key, allowed_regimes)
        )
        raw_regime_confidence: object = None
        try:
            features = market.regime_features_for_symbol(sym)
        except Exception:
            features = {}
        if isinstance(features, dict):
            raw_regime_confidence = features.get("regime_confidence")
        if raw_regime_confidence is None:
            raw_regime_confidence = getattr(market, "regime_confidence", 1.0)
        try:
            regime_confidence = float(raw_regime_confidence or 0.0)
        except (TypeError, ValueError):
            regime_confidence = 0.0
        regime_confident = regime_confidence >= min_regime_confidence
        if disabled_reason:
            dropped += 1
            changed = True
            details.append(
                f"genetics_probation_disabled:{sym}:{probation_label}:{disabled_reason}"
            )
            continue
        if (
            max_real_trades > 0
            and _real_trade_attempts_for_label(pipeline, probation_label)
            >= max_real_trades
        ):
            dropped += 1
            changed = True
            details.append(
                f"genetics_max_real_trades:{sym}:{probation_label}:{max_real_trades}"
            )
            continue
        if (
            max_daily_trades > 0
            and _real_trade_attempts_for_label_today(
                pipeline,
                probation_label,
                market,
            )
            >= max_daily_trades
        ):
            dropped += 1
            changed = True
            details.append(
                f"genetics_daily_trade_limit:{sym}:{probation_label}:{max_daily_trades}"
            )
            continue
        if not regime_allowed:
            dropped += 1
            changed = True
            details.append(f"genetics_regime_blocked:{sym}:{agent_label}:{regime_key}")
            continue
        if not regime_confident:
            dropped += 1
            changed = True
            details.append(
                "genetics_regime_confidence_blocked:"
                f"{sym}:{agent_label}:"
                f"{regime_confidence:.4f}<{min_regime_confidence:.4f}"
            )
            continue
        if allowed_signal_keys and not (
            _genetics_probation_signal_keys(probation_label, signal)
            & allowed_signal_keys
        ):
            dropped += 1
            changed = True
            details.append(f"genetics_signal_key_blocked:{sym}:{agent_label}")
            continue
        if require_shadow_confirmation:
            shadow_confirmed = _shadow_confirms_signal(
                pipeline,
                probation_label,
                signal,
            )
            primary_shadow_bypass = (
                not shadow_confirmed
                and bool(getattr(
                    cfg,
                    "flash_genetics_core_primary_bypass_shadow_confirmation_enabled",
                    False,
                ))
                and _is_flash_genetics_core_primary_signal(
                    cfg,
                    probation_label,
                    signal,
                )
            )
            if not shadow_confirmed and not primary_shadow_bypass:
                dropped += 1
                changed = True
                details.append(f"genetics_shadow_unconfirmed:{sym}:{agent_label}")
                continue
            if primary_shadow_bypass:
                changed = True
                details.append(
                    f"flash_genetics_core_primary_shadow_bypass:{sym}:{agent_label}"
                )
        old_risk_mult = max(0.0, float(signal.risk_mult or 0.0))
        new_risk_mult = min(old_risk_mult, max_risk_mult)
        min_executable_risk_mult = _genetics_probation_min_executable_risk_mult(
            pipeline,
            market,
            signal,
        )
        if min_executable_risk_mult > new_risk_mult:
            new_risk_mult = min(1.0, min_executable_risk_mult)
        if abs(new_risk_mult - old_risk_mult) > 1e-12:
            changed = True
            details.append(
                f"genetics_probation_sized:{sym}:{agent_label}:risk_mult={new_risk_mult:.4f}"
            )
            kept.append(replace(signal, risk_mult=new_risk_mult))
        else:
            kept.append(signal)
    if not changed:
        return signal_guard
    return replace(
        signal_guard,
        signals=kept,
        filtered=int(getattr(signal_guard, "filtered", 0) or 0) + dropped,
        details=details,
    )


def _genetics_probation_min_executable_risk_mult(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    signal: Signal,
) -> float:
    risk_cfg = getattr(pipeline, "risk_config", None)
    if risk_cfg is None:
        risk_obj = getattr(getattr(pipeline, "executor", None), "_risk", None)
        risk_cfg = getattr(risk_obj, "config", None)
    if risk_cfg is None or not bool(
        getattr(risk_cfg, "floor_to_exchange_min_notional", False)
    ):
        return 0.0

    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    get_min_notional = getattr(exchange, "get_min_notional", None)
    if not callable(get_min_notional):
        return 0.0
    try:
        exchange_min = float(get_min_notional(signal.sym) or 0.0)
    except Exception:
        return 0.0
    if exchange_min <= 0.0:
        return 0.0

    cfg_min = float(getattr(risk_cfg, "min_notional_usd", 0.0) or 0.0)
    effective_min = max(cfg_min, exchange_min)
    max_notional = float(
        getattr(risk_cfg, "max_notional_usd", float("inf")) or float("inf")
    )
    if effective_min <= 0.0 or effective_min > max_notional:
        return 0.0

    balance = float(
        getattr(pipeline, "current_balance", 0.0)
        or getattr(pipeline, "initial_capital", 0.0)
        or 0.0
    )
    capital_fraction = float(getattr(risk_cfg, "capital_fraction", 0.0) or 0.0)
    size_mult = float(getattr(signal.action, "fraction", 1.0) or 1.0)
    leverage_mult = (
        max(1.0, float(getattr(risk_cfg, "max_leverage", 1.0) or 1.0))
        if bool(getattr(risk_cfg, "apply_leverage_to_notional", False))
        else 1.0
    )
    max_upscale = float(getattr(risk_cfg, "max_min_notional_upscale", 1.0) or 1.0)
    base_notional = balance * capital_fraction * size_mult * leverage_mult
    if base_notional <= 0.0 or max_upscale <= 0.0:
        return 0.0

    required = effective_min / (base_notional * max_upscale)
    return max(0.0, min(1.0, required * 1.000001))


def _soft_allocator_execution_attempt(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    candidates: Sequence[EnsemblePlayer],
    *,
    signal_id_counter: int,
) -> Optional[Dict[str, Any]]:
    if not bool(getattr(pipeline, "soft_allocator_execution_enabled", False)):
        return None
    allocator = getattr(pipeline, "soft_allocator_execution", None)
    weights_for_bar = getattr(allocator, "weights_for_bar", None)
    if not callable(weights_for_bar):
        return None
    try:
        raw_weights = weights_for_bar(market.regime, market.bar) or {}
    except Exception:
        log.debug("soft allocator weight lookup failed", exc_info=True)
        return None
    weights = {
        str(label).strip(): max(0.0, min(1.0, float(weight or 0.0)))
        for label, weight in dict(raw_weights).items()
        if str(label).strip()
    }
    weights = {label: weight for label, weight in weights.items() if weight > 1e-12}
    if not weights:
        return None

    candidates_by_label = {
        str(getattr(candidate, "label", "") or ""): candidate
        for candidate in candidates or ()
        if str(getattr(candidate, "label", "") or "")
    }
    shadow_registry = {
        str(label): tuple(signals or ())
        for label, signals in dict(
            getattr(pipeline, "_current_shadow_player_signals", {}) or {}
        ).items()
    }
    raw_signals: List[Signal] = []
    selected_labels: List[str] = []
    skipped: List[str] = []
    agent_labels: List[str] = []
    agents: List[object] = []
    next_signal_id = int(signal_id_counter)
    allowed_labels = _soft_allocator_execution_allowed_labels(pipeline)
    denied_actor_symbols = _soft_allocator_execution_denied_actor_symbols(pipeline)
    for label, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        if allowed_labels and label not in allowed_labels:
            skipped.append(f"{label}:not_allowed")
            continue
        realized_gate_reason = _soft_allocator_realized_gate_reason(
            pipeline,
            label,
            current_bar=market.bar,
        )
        if realized_gate_reason:
            skipped.append(f"{label}:{realized_gate_reason}")
            continue
        candidate = candidates_by_label.get(label)
        if candidate is None:
            skipped.append(f"{label}:not_candidate")
            continue
        safety_reason = _fallback_candidate_safety_reason(pipeline, candidate, market)
        if safety_reason:
            skipped.append(f"{label}:{safety_reason}")
            continue
        shadow_signals = shadow_registry.get(label, ())
        if not shadow_signals:
            skipped.append(f"{label}:no_current_shadow_signal")
            continue
        converted, next_signal_id = _real_signals_from_shadow_registry(
            shadow_signals,
            market,
            candidate,
            signal_id_counter=next_signal_id,
        )
        converted = _filter_shadow_state_entry_signals(pipeline, candidate, converted)
        if not converted:
            skipped.append(f"{label}:no_real_signal_after_filter")
            continue
        if denied_actor_symbols:
            kept: list[Signal] = []
            for signal in converted:
                if _soft_allocator_actor_symbol_denied(
                    denied_actor_symbols,
                    label,
                    getattr(signal, "sym", ""),
                ):
                    skipped.append(
                        f"{label}:denied_symbol:{str(getattr(signal, 'sym', '') or '')}"
                    )
                    continue
                kept.append(signal)
            converted = tuple(kept)
            if not converted:
                skipped.append(f"{label}:no_real_signal_after_symbol_gate")
                continue
        for signal in converted:
            raw_signals.append(_soft_allocator_scaled_signal(signal, weight))
        selected_labels.append(label)
        for agent_label in getattr(candidate, "agent_labels", ()) or ():
            clean = str(agent_label or "").strip()
            if clean and clean not in agent_labels:
                agent_labels.append(clean)
        for agent in getattr(candidate, "agents", ()) or ():
            agents.append(agent)

    if not raw_signals:
        return None

    policy_name = _soft_allocator_policy_name(pipeline)
    actor = _soft_allocator_execution_actor(
        pipeline,
        agents=tuple(agents),
        agent_labels=tuple(agent_labels),
        selected_labels=tuple(selected_labels),
    )
    signal_guard = filter_real_signals_against_tracker(
        raw_signals,
        player=actor,
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
    executable = tuple(getattr(signal_guard, "signals", ()) or ())
    if not executable:
        return None
    details = list(getattr(signal_guard, "details", []) or [])
    details.append(f"soft_allocator_policy:{policy_name}")
    for label in selected_labels:
        details.append(f"soft_allocator_weight:{label}:{weights[label]:.4f}")
    if skipped:
        details.extend(f"soft_allocator_skip:{item}" for item in skipped[:8])
    signal_guard = replace(signal_guard, details=details)
    return {
        "leader": actor,
        "raw_signals": raw_signals,
        "raw_signal_count": len(raw_signals),
        "leader_vote_error_count": 0,
        "signal_guard": signal_guard,
        "signal_id_counter": next_signal_id,
        "soft_allocator": {
            "enabled": True,
            "policy_name": policy_name,
            "soft_only": _soft_allocator_execution_soft_only(pipeline),
            "realized_gate_enabled": _soft_allocator_realized_gate_enabled(pipeline),
            "weights": {label: weights[label] for label in sorted(weights)},
            "selected_labels": tuple(selected_labels),
            "skipped": tuple(skipped),
            "raw_signal_count": len(raw_signals),
            "executable_signal_count": len(executable),
            "single_position_model": True,
        },
    }


def _soft_allocator_scaled_signal(signal: Signal, weight: float) -> Signal:
    if not signal.action.is_open:
        return signal
    risk_mult = max(0.0, float(getattr(signal, "risk_mult", 1.0) or 1.0))
    return replace(signal, risk_mult=risk_mult * max(0.0, min(1.0, float(weight))))


def _soft_allocator_policy_name(pipeline: ProductionPipeline) -> str:
    allocator = getattr(pipeline, "soft_allocator_execution", None)
    policy = getattr(allocator, "policy", None)
    return str(getattr(policy, "name", "") or "soft_allocator")


def _soft_allocator_execution_actor(
    pipeline: ProductionPipeline,
    *,
    agents: Sequence[object] = (),
    agent_labels: Sequence[str] = (),
    selected_labels: Sequence[str] = (),
) -> _FlashExecutionActor:
    labels = tuple(selected_labels or agent_labels or ())
    return _FlashExecutionActor(
        label=f"PanteonSoft:{_soft_allocator_policy_name(pipeline)}",
        agents=tuple(agents or ()),
        agent_labels=labels,
    )


def _soft_allocator_idle_attempt(
    pipeline: ProductionPipeline,
    *,
    signal_id_counter: int,
    reason: str,
) -> Dict[str, Any]:
    policy_name = _soft_allocator_policy_name(pipeline)
    actor = _soft_allocator_execution_actor(pipeline)
    details = [
        f"soft_allocator_policy:{policy_name}",
        str(reason or "soft_allocator_idle"),
    ]
    return {
        "leader": actor,
        "raw_signals": [],
        "raw_signal_count": 0,
        "leader_vote_error_count": 0,
        "signal_guard": RealSignalGuardResult(signals=[], details=details),
        "signal_id_counter": signal_id_counter,
        "soft_allocator": {
            "enabled": True,
            "policy_name": policy_name,
            "soft_only": _soft_allocator_execution_soft_only(pipeline),
            "realized_gate_enabled": _soft_allocator_realized_gate_enabled(pipeline),
            "reason": str(reason or "soft_allocator_idle"),
            "weights": {},
            "selected_labels": tuple(),
            "skipped": tuple(),
            "raw_signal_count": 0,
            "executable_signal_count": 0,
            "single_position_model": True,
        },
    }


def _soft_allocator_execution_soft_only(pipeline: ProductionPipeline) -> bool:
    return (
        bool(getattr(pipeline, "soft_allocator_execution_enabled", False))
        and bool(getattr(pipeline, "soft_allocator_execution_soft_only", False))
        and getattr(pipeline, "soft_allocator_execution", None) is not None
    )


def _soft_allocator_execution_allowed_labels(
    pipeline: ProductionPipeline,
) -> set[str]:
    return {
        str(label or "").strip()
        for label in getattr(pipeline, "soft_allocator_execution_allow_labels", ()) or ()
        if str(label or "").strip()
    }


def _soft_allocator_execution_denied_actor_symbols(
    pipeline: ProductionPipeline,
) -> set[tuple[str, str]]:
    denied: set[tuple[str, str]] = set()
    for item in getattr(pipeline, "soft_allocator_execution_deny_actor_symbols", ()) or ():
        raw = str(item or "").strip()
        if not raw or "|" not in raw:
            continue
        label, symbol = (part.strip() for part in raw.split("|", 1))
        if not label or not symbol:
            continue
        for key in _soft_allocator_symbol_keys(symbol):
            denied.add((label, key))
    return denied


def _soft_allocator_actor_symbol_denied(
    denied_actor_symbols: set[tuple[str, str]],
    label: str,
    symbol: object,
) -> bool:
    actor_label = str(label or "").strip()
    if not actor_label:
        return False
    return any(
        (actor_label, key) in denied_actor_symbols
        for key in _soft_allocator_symbol_keys(symbol)
    )


def _soft_allocator_symbol_keys(symbol: object) -> set[str]:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return set()
    keys = {raw}
    if "/" in raw:
        base = raw.split("/", 1)[0].strip()
        if base:
            keys.add(base)
    else:
        keys.add(f"{raw}/USDT")
    return keys


def _soft_allocator_realized_gate_enabled(pipeline: ProductionPipeline) -> bool:
    return bool(getattr(pipeline, "soft_allocator_realized_gate_enabled", False))


def _soft_allocator_realized_gate_reason(
    pipeline: ProductionPipeline,
    label: str,
    *,
    current_bar: int,
) -> str:
    if not _soft_allocator_realized_gate_enabled(pipeline):
        return ""
    clean = str(label or "").strip()
    if not clean:
        return ""
    try:
        lookback_bars = int(
            getattr(pipeline, "soft_allocator_realized_gate_lookback_bars", 0) or 0
        )
    except (TypeError, ValueError):
        lookback_bars = 0
    try:
        min_closed = int(
            getattr(pipeline, "soft_allocator_realized_gate_min_closed_trades", 0) or 0
        )
    except (TypeError, ValueError):
        min_closed = 0
    try:
        max_recent_pnl = float(
            getattr(pipeline, "soft_allocator_realized_gate_max_recent_pnl_usd", 0.0)
            or 0.0
        )
    except (TypeError, ValueError):
        max_recent_pnl = 0.0
    if min_closed <= 0:
        return ""
    events = _soft_allocator_recent_realized_events(
        pipeline,
        clean,
        current_bar=current_bar,
        lookback_bars=lookback_bars,
    )
    closed = sum(max(0, int(item[2] or 0)) for item in events)
    if closed < min_closed:
        return ""
    pnl = sum(float(item[1] or 0.0) for item in events)
    wins = sum(max(0, int(item[3] or 0)) for item in events)
    if pnl <= max_recent_pnl:
        return (
            f"realized_gate_recent_pnl:{pnl:.2f}:closed={closed}:"
            f"wins={wins}:lookback={lookback_bars}"
        )
    return ""


def _soft_allocator_recent_realized_events(
    pipeline: ProductionPipeline,
    label: str,
    *,
    current_bar: int,
    lookback_bars: int,
) -> tuple[tuple[int, float, int, int], ...]:
    events_by_player = dict(
        getattr(pipeline, "_soft_allocator_realized_events_by_player", {}) or {}
    )
    events = tuple(events_by_player.get(str(label or "").strip(), ()) or ())
    if lookback_bars <= 0:
        return tuple(_normalize_soft_allocator_realized_event(item) for item in events)
    floor_bar = max(0, int(current_bar or 0) - int(lookback_bars))
    out = []
    for item in events:
        event = _normalize_soft_allocator_realized_event(item)
        if event[0] >= floor_bar:
            out.append(event)
    return tuple(out)


def _normalize_soft_allocator_realized_event(
    item: object,
) -> tuple[int, float, int, int]:
    try:
        values = tuple(item)  # type: ignore[arg-type]
    except TypeError:
        values = ()
    bar = int(values[0] if len(values) > 0 else 0 or 0)
    pnl = float(values[1] if len(values) > 1 else 0.0 or 0.0)
    closed = int(values[2] if len(values) > 2 else 0 or 0)
    wins = int(values[3] if len(values) > 3 else 0 or 0)
    return (bar, pnl, closed, wins)


def _soft_allocator_execution_prevents_notrade_cash_flat(
    pipeline: ProductionPipeline,
) -> bool:
    return (
        bool(getattr(pipeline, "soft_allocator_execution_enabled", False))
        and getattr(pipeline, "soft_allocator_execution", None) is not None
    )


def _genetics_probation_preselection_degraded_signal_keys(
    pipeline: ProductionPipeline,
    shadow_player_signals: Mapping[str, Sequence[Signal]] | None,
    shadow_agent_signals: Mapping[str, Sequence[Signal]] | None = None,
    *,
    regime: Regime | str | None = None,
) -> Tuple[str, ...]:
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return ()
    allowed_regimes = {
        item.lower()
        for item in _string_tuple(
            getattr(cfg, "genetics_probation_allowed_regimes", ()),
        )
    }
    if (
        allowed_regimes
        and regime is not None
        and not _regime_allowed_by_keys(regime, allowed_regimes)
    ):
        return ()
    allowed_signal_keys = {
        str(item or "").strip()
        for item in getattr(cfg, "genetics_probation_allowed_signal_keys", ()) or ()
        if str(item or "").strip()
    }
    if not allowed_signal_keys:
        return ()
    labels = set(_string_tuple(
        getattr(cfg, "genetics_probation_labels", ("GeneticsResearch",)),
    ))
    if not labels:
        return ()

    degraded: List[str] = []

    def add_unlisted_open_keys(raw_label: object, signals: Sequence[Signal] | None) -> None:
        label = str(raw_label or "").strip()
        if label not in labels:
            return
        for signal in signals or ():
            action = getattr(signal, "action", None)
            if not bool(getattr(action, "is_open", False)):
                continue
            keys = _genetics_probation_signal_keys(label, signal)
            if keys and not keys & allowed_signal_keys:
                degraded.extend(sorted(keys))

    for label, signals in dict(shadow_player_signals or {}).items():
        add_unlisted_open_keys(label, signals)
    for label, signals in dict(shadow_agent_signals or {}).items():
        add_unlisted_open_keys(label, signals)
    return tuple(dict.fromkeys(degraded))


def _genetics_probation_preselection_admission_signal_keys(
    pipeline: ProductionPipeline,
    shadow_player_signals: Mapping[str, Sequence[Signal]] | None,
    shadow_agent_signals: Mapping[str, Sequence[Signal]] | None = None,
    *,
    market: MarketSnapshot,
) -> Tuple[str, ...]:
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return ()
    labels = set(_string_tuple(
        getattr(cfg, "genetics_probation_labels", ("GeneticsResearch",)),
    ))
    if not labels:
        return ()
    allowed_regimes = {
        item.lower()
        for item in _string_tuple(
            getattr(cfg, "genetics_probation_allowed_regimes", ()),
        )
    }
    min_regime_confidence = max(
        0.0,
        min(
            1.0,
            float(getattr(cfg, "genetics_probation_min_regime_confidence", 0.0) or 0.0),
        ),
    )
    if _genetics_probation_disabled_reason(pipeline):
        return ()
    allowed_signal_keys = {
        str(item or "").strip()
        for item in getattr(cfg, "genetics_probation_allowed_signal_keys", ()) or ()
        if str(item or "").strip()
    }
    keys: List[str] = []

    def label_budget_available(label: str) -> bool:
        if _genetics_probation_disabled_reason(pipeline, label):
            return False
        max_real_trades = _genetics_probation_int_for_label(
            cfg,
            "genetics_probation_max_real_trades_by_label",
            "genetics_probation_max_real_trades",
            label,
            20,
        )
        max_daily_trades = _genetics_probation_int_for_label(
            cfg,
            "genetics_probation_max_daily_trades_by_label",
            "genetics_probation_max_daily_trades",
            label,
            0,
        )
        if (
            max_real_trades > 0
            and _real_trade_attempts_for_label(pipeline, label) >= max_real_trades
        ):
            return False
        if (
            max_daily_trades > 0
            and _real_trade_attempts_for_label_today(pipeline, label, market)
            >= max_daily_trades
        ):
            return False
        return True

    def symbol_regime_allowed(symbol: str) -> bool:
        if not allowed_regimes:
            return True
        try:
            regime = market.regime_for_symbol(symbol)
        except Exception:
            regime = getattr(market, "regime", "")
        return _regime_allowed_by_keys(regime, allowed_regimes)

    def symbol_regime_confident(symbol: str) -> bool:
        raw_confidence: object = None
        try:
            features = market.regime_features_for_symbol(symbol)
        except Exception:
            features = {}
        if isinstance(features, dict):
            raw_confidence = features.get("regime_confidence")
        if raw_confidence is None:
            raw_confidence = getattr(market, "regime_confidence", 1.0)
        try:
            confidence = float(raw_confidence or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return confidence >= min_regime_confidence

    for signal_key in sorted(allowed_signal_keys):
        key_label = _label_from_genetics_signal_key(signal_key)
        key_symbol = _symbol_from_genetics_signal_key(signal_key)
        if (
            key_label in labels
            and label_budget_available(key_label)
            and symbol_regime_allowed(key_symbol)
            and symbol_regime_confident(key_symbol)
        ):
            keys.append(signal_key)

    def add_current_shadow_open_keys(
        raw_label: object,
        signals: Sequence[Signal] | None,
    ) -> None:
        leader_label = str(raw_label or "").strip()
        for signal in signals or ():
            action = getattr(signal, "action", None)
            if not bool(getattr(action, "is_open", False)):
                continue
            probation_label = _genetics_probation_signal_label(
                leader_label=leader_label,
                signal=signal,
                labels=labels,
            )
            if not probation_label:
                continue
            if not label_budget_available(probation_label):
                continue
            sym = str(getattr(signal, "sym", "") or "").upper()
            if not symbol_regime_allowed(sym):
                continue
            if not symbol_regime_confident(sym):
                continue
            signal_keys = _genetics_probation_signal_keys(probation_label, signal)
            if allowed_signal_keys and not signal_keys & allowed_signal_keys:
                continue
            keys.extend(sorted(signal_keys))

    for label, signals in dict(shadow_player_signals or {}).items():
        add_current_shadow_open_keys(label, signals)
    for label, signals in dict(shadow_agent_signals or {}).items():
        add_current_shadow_open_keys(label, signals)
    return tuple(dict.fromkeys(keys))


def _label_from_genetics_signal_key(signal_key: object) -> str:
    raw = str(signal_key or "").strip()
    head, _, _tail = raw.partition("|")
    _kind, sep, label = head.partition(":")
    return label.strip() if sep else ""


def _symbol_from_genetics_signal_key(signal_key: object) -> str:
    parts = [part.strip() for part in str(signal_key or "").split("|")]
    if len(parts) != 3:
        return ""
    return parts[1].upper()


def _genetics_probation_symbol_variants(symbol: object) -> tuple[str, ...]:
    raw = str(symbol or "").strip().upper().replace("-", "/")
    if not raw:
        return tuple()
    variants = [raw]
    if raw.endswith("/USDT"):
        variants.append(raw[:-5])
    elif "/" not in raw:
        variants.append(f"{raw}/USDT")
    return tuple(dict.fromkeys(variants))


def _genetics_probation_signal_label(
    *,
    leader_label: str,
    signal: Signal,
    labels: set[str],
) -> str:
    if leader_label in labels:
        return leader_label
    player_label = str(getattr(signal, "by_player", "") or "")
    if player_label in labels:
        return player_label
    return ""


def _genetics_probation_signal_keys(
    probation_label: str,
    signal: Signal,
) -> set[str]:
    action = getattr(signal, "action", None)
    action_name = str(getattr(action, "name", action) or "")
    symbols = _genetics_probation_symbol_variants(getattr(signal, "sym", ""))
    labels = tuple(dict.fromkeys(
        str(item or "").strip()
        for item in (
            probation_label,
            getattr(signal, "by_agent", ""),
            getattr(signal, "by_player", ""),
        )
        if str(item or "").strip()
    ))
    return {
        f"{actor_type}:{label}|{sym}|{action_name}"
        for label in labels
        for sym in symbols
        for actor_type in ("agent", "ensemble")
    }


def _is_flash_genetics_core_primary_signal(
    cfg,
    probation_label: str,
    signal: Signal,
) -> bool:
    if not bool(getattr(cfg, "flash_genetics_core_primary_enabled", False)):
        return False
    primary_labels = set(_string_tuple(
        getattr(cfg, "flash_genetics_core_primary_labels", ("GeneticsCore",)),
    ))
    if not primary_labels:
        return False
    variants: set[str] = set()
    for raw in (
        probation_label,
        getattr(signal, "by_agent", ""),
        getattr(signal, "by_player", ""),
    ):
        clean = str(raw or "").strip()
        if not clean:
            continue
        variants.add(clean)
        suffix = clean.split(":", 1)[1] if ":" in clean else clean
        variants.add(suffix)
        variants.add(f"agent:{suffix}")
        variants.add(f"ensemble:{suffix}")
        if suffix.startswith("Solo_"):
            solo_base = suffix.removeprefix("Solo_")
            variants.add(solo_base)
            variants.add(f"agent:{solo_base}")
            variants.add(f"ensemble:{solo_base}")
    return bool(variants & primary_labels)


def _shadow_confirms_signal(
    pipeline: ProductionPipeline,
    player_label: str,
    signal: Signal,
) -> bool:
    clean_label = str(player_label)
    shadow_signals: List[Signal] = []
    for attr in ("_current_shadow_player_signals", "_current_shadow_agent_signals"):
        registry = dict(getattr(pipeline, attr, {}) or {})
        shadow_signals.extend(tuple(registry.get(clean_label, ()) or ()))
    sym = str(signal.sym).upper()
    for shadow_signal in shadow_signals:
        if str(getattr(shadow_signal, "sym", "") or "").upper() != sym:
            continue
        if getattr(shadow_signal, "action", None) == signal.action:
            return True
    return False


def _genetics_probation_disabled_reason(
    pipeline: ProductionPipeline,
    label: str = "",
) -> str:
    state = _kill_state(pipeline)
    if state is None:
        return ""
    clean_label = str(label or "").strip()
    if clean_label:
        reasons = getattr(state, "genetics_probation_disabled_reasons_by_label", {}) or {}
        if isinstance(reasons, Mapping):
            reason = str(reasons.get(clean_label, "") or "")
            if reason:
                return reason
    return str(getattr(state, "genetics_probation_disabled_reason", "") or "")


def _set_genetics_probation_disabled(
    pipeline: ProductionPipeline,
    reason: str,
    *,
    label: str = "",
) -> None:
    clean = str(reason or "").strip()
    if not clean:
        return
    state = _kill_state(pipeline)
    if state is None:
        return
    clean_label = str(label or "").strip()
    if clean_label:
        reasons = getattr(state, "genetics_probation_disabled_reasons_by_label", None)
        if not isinstance(reasons, dict):
            reasons = {}
            state.genetics_probation_disabled_reasons_by_label = reasons
        if reasons.get(clean_label):
            return
        reasons[clean_label] = clean
        if (
            len(_genetics_probation_labels(pipeline)) <= 1
            and not getattr(state, "genetics_probation_disabled_reason", "")
        ):
            state.genetics_probation_disabled_reason = clean
        log.error("genetics probation disabled for %s: %s", clean_label, clean)
        return
    if getattr(state, "genetics_probation_disabled_reason", ""):
        return
    state.genetics_probation_disabled_reason = clean
    log.error("genetics probation disabled: %s", clean)


def _real_trade_attempts_for_label(
    pipeline: ProductionPipeline,
    label: str,
) -> int:
    clean_label = str(label or "")
    if not clean_label:
        return 0
    actual_count = 0
    actual_source_available = False

    ledger = getattr(pipeline, "ledger", None)
    realized_attributions = getattr(ledger, "realized_attributions", None)
    if callable(realized_attributions):
        actual_source_available = True
        try:
            for attr in realized_attributions():
                if _label_matches_trade_source(clean_label, attr):
                    actual_count += 1
        except Exception:
            actual_count = 0

    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    all_open = getattr(tracker, "all_open", None)
    if callable(all_open):
        actual_source_available = True
        try:
            for position in (all_open() or {}).values():
                if _label_matches_trade_source(clean_label, position):
                    actual_count += 1
        except Exception:
            pass

    if actual_source_available:
        return max(0, actual_count)

    perf = getattr(pipeline, "real_perf", None)
    getter = getattr(perf, "get", None)
    if not callable(getter):
        return 0
    try:
        metrics = getter(label)
    except Exception:
        return 0
    return max(
        0,
        int(getattr(metrics, "entries", 0) or 0),
        int(getattr(metrics, "closed_trades", 0) or 0),
    )


def _real_trade_attempts_for_label_today(
    pipeline: ProductionPipeline,
    label: str,
    market: MarketSnapshot,
) -> int:
    target_date = _utc_date(getattr(market, "timestamp", None))
    if target_date is None:
        return 0
    clean_label = str(label or "")
    if not clean_label:
        return 0
    event_log = getattr(pipeline, "event_log", None)
    query = getattr(event_log, "query", None)
    actual_source_available = False
    if callable(query):
        actual_source_available = True
        try:
            events = query(event_types=[SignalEmitted, PositionOpened])
        except Exception:
            events = ()
        signal_ids_for_label: set[int] = set()
        opened_events = []
        for event in events:
            if isinstance(event, SignalEmitted):
                signal = getattr(event, "signal", None)
                if not _label_matches_signal_source(clean_label, signal):
                    continue
                try:
                    signal_ids_for_label.add(int(getattr(signal, "id")))
                except (TypeError, ValueError):
                    continue
            elif isinstance(event, PositionOpened):
                if _utc_date(getattr(event, "timestamp", None)) == target_date:
                    opened_events.append(event)
        if opened_events:
            count = 0
            for event in opened_events:
                try:
                    signal_id = int(getattr(event, "signal_id", -1))
                except (TypeError, ValueError):
                    signal_id = -1
                if signal_id in signal_ids_for_label:
                    count += 1
            return count

    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    all_open = getattr(tracker, "all_open", None)
    if callable(all_open):
        actual_source_available = True
        count = 0
        try:
            for position in (all_open() or {}).values():
                if _utc_date(getattr(position, "opened_at", None)) != target_date:
                    continue
                if _label_matches_trade_source(clean_label, position):
                    count += 1
            return count
        except Exception:
            return 0

    return 0


def _label_matches_signal_source(label: str, signal: object) -> bool:
    if signal is None:
        return False
    return label in {
        str(getattr(signal, "by_player", "") or ""),
        str(getattr(signal, "by_agent", "") or ""),
    }


def _label_matches_trade_source(label: str, trade_source: object) -> bool:
    if trade_source is None:
        return False
    return label in {
        str(getattr(trade_source, "by_player", "") or ""),
        str(getattr(trade_source, "by_agent", "") or ""),
    }


def _utc_date(value: object):
    if value is None:
        return None
    try:
        if getattr(value, "tzinfo", None) is None:
            replace = getattr(value, "replace", None)
            if callable(replace):
                return value.replace(tzinfo=timezone.utc).date()
        astimezone = getattr(value, "astimezone", None)
        if callable(astimezone):
            return value.astimezone(timezone.utc).date()
    except Exception:
        return None
    return None


def _string_tuple(value: object) -> Tuple[str, ...]:
    if isinstance(value, str):
        item = value.strip()
        return (item,) if item else ()
    try:
        return tuple(str(item).strip() for item in (value or ()) if str(item).strip())
    except TypeError:
        return ()


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


def _genetics_probation_regime_exit_close_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    signal_id_start: int,
    excluded_symbols: Optional[set[str]] = None,
) -> List[Signal]:
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return []
    labels = {
        str(item or "").strip()
        for item in _string_tuple(
            getattr(cfg, "genetics_probation_labels", ("GeneticsResearch",)),
        )
        if str(item or "").strip()
    }
    if not labels:
        return []
    allowed_regimes = {
        item.lower()
        for item in _string_tuple(
            getattr(cfg, "genetics_probation_allowed_regimes", ("bearish", "crash")),
        )
    }
    regime_allowed = (
        not allowed_regimes
        or _regime_allowed_by_keys(market.regime, allowed_regimes)
    )
    try:
        regime_confidence = float(getattr(market, "regime_confidence", 1.0) or 0.0)
    except (TypeError, ValueError):
        regime_confidence = 0.0
    try:
        min_confidence = float(
            getattr(cfg, "genetics_probation_min_regime_confidence", 0.0) or 0.0
        )
    except (TypeError, ValueError):
        min_confidence = 0.0
    min_confidence = max(0.0, min(1.0, min_confidence))
    confidence_allowed = regime_confidence >= min_confidence
    if regime_allowed and confidence_allowed:
        return []

    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is None or not callable(getattr(tracker, "all_open", None)):
        return []
    excluded = {str(symbol or "").upper() for symbol in (excluded_symbols or set())}

    try:
        open_positions = tracker.all_open() or {}
    except Exception:
        log.debug("genetics probation regime-exit guard failed to read tracker", exc_info=True)
        return []

    signals: List[Signal] = []
    sid = int(signal_id_start)
    for sym, pos in sorted(open_positions.items()):
        if is_external_position(pos):
            continue
        actor_labels = {
            str(getattr(pos, "by_player", "") or "").strip(),
            str(getattr(pos, "by_agent", "") or "").strip(),
        }
        if not actor_labels & labels:
            continue
        symbol = str(getattr(pos, "sym", sym) or sym).upper()
        if symbol in excluded:
            continue
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
            by_player="Panteon_Flash",
            by_agent="GeneticsProbationRegimeExit",
            timestamp=market.timestamp,
        ))
        sid += 1
    return signals


def _flash_stale_position_exit_enabled(pipeline: ProductionPipeline) -> bool:
    return bool(getattr(pipeline, "flash_stale_position_exit_enabled", False))


def _flash_stale_position_exit_max_age_bars(pipeline: ProductionPipeline) -> int:
    if not _flash_stale_position_exit_enabled(pipeline):
        return 0
    try:
        return max(0, int(getattr(pipeline, "flash_stale_position_exit_max_age_bars", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _flash_stale_position_exit_requires_loss(pipeline: ProductionPipeline) -> bool:
    return bool(
        getattr(
            pipeline,
            "flash_stale_position_exit_require_nonpositive_unrealized",
            True,
        )
    )


def _flash_partial_profit_lock_enabled(pipeline: ProductionPipeline) -> bool:
    return bool(getattr(pipeline, "flash_partial_profit_lock_enabled", False))


def _flash_partial_profit_lock_trigger_pnl_pct(
    pipeline: ProductionPipeline,
) -> float:
    try:
        return max(
            0.0,
            float(getattr(pipeline, "flash_partial_profit_lock_trigger_pnl_pct", 1.5)),
        )
    except (TypeError, ValueError):
        return 1.5


def _flash_partial_profit_lock_close_fraction(
    pipeline: ProductionPipeline,
) -> float:
    try:
        value = float(getattr(pipeline, "flash_partial_profit_lock_close_fraction", 0.5))
    except (TypeError, ValueError):
        return 0.5
    if value <= 0.0:
        return 0.5
    return min(1.0, value)


def _flash_partial_profit_lock_min_age_bars(pipeline: ProductionPipeline) -> int:
    try:
        return max(
            0,
            int(getattr(pipeline, "flash_partial_profit_lock_min_age_bars", 1) or 0),
        )
    except (TypeError, ValueError):
        return 1


def _flash_partial_profit_lock_skip_protected_signal_keys(
    pipeline: ProductionPipeline,
) -> bool:
    return bool(
        getattr(
            pipeline,
            "flash_partial_profit_lock_skip_protected_signal_keys",
            True,
        )
    )


def _flash_partial_profit_lock_skip_signal_keys(
    pipeline: ProductionPipeline,
) -> set[str]:
    raw_keys: list[object] = list(
        getattr(pipeline, "flash_partial_profit_lock_skip_signal_keys", ()) or ()
    )
    if _flash_partial_profit_lock_skip_protected_signal_keys(pipeline):
        raw_keys.extend(
            getattr(pipeline, "flash_selected_subset_do_not_demote_signal_keys", ())
            or ()
        )
        allocator_config = getattr(
            getattr(pipeline, "flash_allocator", None),
            "_config",
            None,
        )
        raw_keys.extend(
            getattr(
                allocator_config,
                "selected_subset_do_not_demote_signal_keys",
                (),
            )
            or ()
        )
    return {
        normalized
        for raw in raw_keys
        for normalized in [_normalize_partial_profit_lock_signal_key(raw)]
        if normalized
    }


def _normalize_partial_profit_lock_signal_key(raw: object) -> str:
    parts = [str(part or "").strip() for part in str(raw or "").split("|")]
    if len(parts) < 3:
        return ""
    actor_key, symbol, action = parts[:3]
    if not actor_key or not symbol or not action:
        return ""
    return "|".join((actor_key, symbol.upper(), action.upper()))


def _partial_profit_lock_open_signal_keys(pos: object) -> set[str]:
    symbol = str(getattr(pos, "sym", "") or "").upper()
    action = str(getattr(pos, "open_action", "") or "").upper()
    if not symbol or not action:
        return set()
    by_player = str(getattr(pos, "by_player", "") or "").strip()
    by_agent = str(getattr(pos, "by_agent", "") or "").strip()
    actor_keys: list[str] = []
    if by_player and by_player not in {"Panteon_Flash", by_agent}:
        actor_keys.append(f"ensemble:{by_player}")
    if by_agent:
        actor_keys.append(f"agent:{by_agent}")
    if by_player and by_player == by_agent:
        actor_keys.append(f"agent:{by_player}")
    return {
        _normalize_partial_profit_lock_signal_key(f"{actor}|{symbol}|{action}")
        for actor in actor_keys
        if actor
    }


def _flash_stop_loss_close_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    signal_id_start: int,
    excluded_symbols: Optional[set[str]] = None,
) -> List[Signal]:
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is None or not callable(getattr(tracker, "all_open", None)):
        return []
    excluded = {str(symbol or "").upper() for symbol in (excluded_symbols or set())}
    try:
        open_positions = tracker.all_open() or {}
    except Exception:
        log.debug("flash stop-loss guard failed to read tracker", exc_info=True)
        return []

    signals: List[Signal] = []
    sid = int(signal_id_start)
    for sym, pos in sorted(open_positions.items()):
        if is_external_position(pos):
            continue
        symbol = str(getattr(pos, "sym", sym) or sym).upper()
        if symbol in excluded:
            continue
        stop_price = _safe_float_or_none(getattr(pos, "stop_price", 0.0))
        if stop_price is None or stop_price <= 0.0:
            continue
        market_price = float(market.prices.get(symbol, 0.0) or 0.0)
        if market_price <= 0.0:
            continue
        side = str(getattr(pos, "side", "") or "").lower()
        if side == "long":
            triggered = market_price <= stop_price
        elif side == "short":
            triggered = market_price >= stop_price
        else:
            continue
        if not triggered:
            continue
        stop_loss_pct = _safe_float_or_none(getattr(pos, "stop_loss_pct", 0.0))
        open_action = str(getattr(pos, "open_action", "") or "").upper()
        close_action = (
            Action.SPOT_SELL_ALL
            if open_action.startswith("SPOT_")
            else Action.FUT_CLOSE_ALL
        )
        signals.append(Signal(
            id=sid,
            bar=market.bar,
            sym=symbol,
            action=close_action,
            price=float(stop_price),
            regime=market.regime,
            by_player="Panteon_Flash",
            by_agent="StopLossGuard",
            timestamp=market.timestamp,
            metadata={
                "stop_price": float(stop_price),
                "stop_loss_pct": float(stop_loss_pct or 0.0),
                "stop_trigger_price": float(market_price),
                "stop_source": "controlled_exploration",
            },
        ))
        sid += 1
    return signals


def _flash_partial_profit_lock_close_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    signal_id_start: int,
    excluded_symbols: Optional[set[str]] = None,
) -> List[Signal]:
    if not _flash_partial_profit_lock_enabled(pipeline):
        return []
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is None or not callable(getattr(tracker, "all_open", None)):
        return []
    trigger_pct = _flash_partial_profit_lock_trigger_pnl_pct(pipeline)
    if trigger_pct <= 0.0:
        return []
    close_fraction = _flash_partial_profit_lock_close_fraction(pipeline)
    min_age_bars = _flash_partial_profit_lock_min_age_bars(pipeline)
    skip_signal_keys = _flash_partial_profit_lock_skip_signal_keys(pipeline)
    excluded = {str(symbol or "").upper() for symbol in (excluded_symbols or set())}

    try:
        open_positions = tracker.all_open() or {}
    except Exception:
        log.debug("flash partial-profit lock failed to read tracker", exc_info=True)
        return []

    signals: List[Signal] = []
    sid = int(signal_id_start)
    for sym, pos in sorted(open_positions.items()):
        if is_external_position(pos):
            continue
        if bool(getattr(pos, "partial_profit_locked", False)):
            continue
        symbol = str(getattr(pos, "sym", sym) or sym).upper()
        if symbol in excluded:
            continue
        if skip_signal_keys and (
            _partial_profit_lock_open_signal_keys(pos) & skip_signal_keys
        ):
            continue
        price = float(market.prices.get(symbol, 0.0) or 0.0)
        entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
        qty = float(getattr(pos, "qty", 0.0) or 0.0)
        if price <= 0.0 or entry_price <= 0.0 or qty <= 0.0:
            continue
        opened_bar = _safe_nonnegative_int(getattr(pos, "opened_bar", 0))
        if opened_bar is not None and min_age_bars > 0:
            age_bars = max(0, int(market.bar) - opened_bar)
            if age_bars < min_age_bars:
                continue
        side = str(getattr(pos, "side", "") or "").lower()
        if side == "long":
            pnl_pct = (price - entry_price) / entry_price * 100.0
        elif side == "short":
            pnl_pct = (entry_price - price) / entry_price * 100.0
        else:
            continue
        if pnl_pct < trigger_pct:
            continue
        signals.append(Signal(
            id=sid,
            bar=market.bar,
            sym=symbol,
            action=Action.FUT_CLOSE_ALL,
            price=price,
            regime=market.regime,
            by_player="Panteon_Flash",
            by_agent="PartialProfitLock",
            close_fraction=close_fraction,
            timestamp=market.timestamp,
        ))
        sid += 1
    return signals


def _flash_stale_position_close_signals(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    signal_id_start: int,
    max_age_bars: int,
    require_nonpositive_unrealized: bool = True,
    excluded_symbols: Optional[set[str]] = None,
) -> List[Signal]:
    max_age = max(0, int(max_age_bars or 0))
    if max_age <= 0:
        return []
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is None or not callable(getattr(tracker, "all_open", None)):
        return []
    excluded = {str(symbol or "").upper() for symbol in (excluded_symbols or set())}

    signals: List[Signal] = []
    sid = int(signal_id_start)
    try:
        open_positions = tracker.all_open() or {}
    except Exception:
        log.debug("flash stale-position guard failed to read tracker", exc_info=True)
        return []
    for sym, pos in sorted(open_positions.items()):
        if is_external_position(pos):
            continue
        age_bars = _stale_position_age_bars(pos, market)
        if age_bars is None:
            continue
        if age_bars < max_age:
            continue
        side = str(getattr(pos, "side", "") or "").lower()
        entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
        qty = float(getattr(pos, "qty", 0.0) or 0.0)
        symbol = str(getattr(pos, "sym", sym) or sym).upper()
        if symbol in excluded:
            continue
        market_price = float(market.prices.get(symbol, 0.0) or 0.0)
        if require_nonpositive_unrealized:
            if market_price <= 0:
                continue
            if side == "long":
                unrealized = (market_price - entry_price) * qty
            elif side == "short":
                unrealized = (entry_price - market_price) * qty
            else:
                continue
            if unrealized > 0:
                continue
            price = market_price
        else:
            price = market_price if market_price > 0 else _stale_position_reference_price(pos)
            if price <= 0:
                continue
        signals.append(Signal(
            id=sid,
            bar=market.bar,
            sym=symbol,
            action=Action.FUT_CLOSE_ALL,
            price=price,
            regime=market.regime,
            by_player="Panteon_Flash",
            by_agent="StalePositionGuard",
            timestamp=market.timestamp,
        ))
        sid += 1
    return signals


def _stale_position_reference_price(pos: object) -> float:
    for key in ("current_price", "mark_price", "last_price", "price", "entry_price", "entry"):
        if isinstance(pos, Mapping):
            raw = pos.get(key)
        else:
            raw = getattr(pos, key, None)
        value = _safe_float_or_none(raw)
        if value is not None and value > 0.0:
            return float(value)
    return 0.0


def _stale_position_age_bars(
    pos: object,
    market: MarketSnapshot,
) -> Optional[int]:
    opened_bar = _safe_nonnegative_int(getattr(pos, "opened_bar", 0))
    if opened_bar is not None and opened_bar > 0:
        return max(0, int(market.bar) - opened_bar)

    opened_at = _coerce_utc_datetime(getattr(pos, "opened_at", None))
    if opened_at is None:
        return None
    market_time = _coerce_utc_datetime(getattr(market, "timestamp", None))
    if market_time is None:
        market_time = datetime.now(timezone.utc)
    age_seconds = max(0.0, (market_time - opened_at).total_seconds())
    return int(age_seconds // 60)


def _coerce_utc_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _find_actionable_candidate_from_shadow_registry(
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
    rejected_labels = _fallback_safety_rejected_labels(
        getattr(decision, "candidate_rejections", ()) or ()
    )
    score_rows = {
        str(getattr(row, "label", "") or ""): row
        for row in getattr(decision, "candidate_scores", ()) or ()
        if str(getattr(row, "label", "") or "")
    }
    min_score = getattr(pipeline, "actionable_fallback_min_score", None)
    registry = {
        str(label): tuple(signals or ())
        for label, signals in dict(
            getattr(pipeline, "_current_shadow_player_signals", {}) or {}
        ).items()
    }
    attempts: List[Tuple[Tuple[float, int, int], Dict[str, Any]]] = []
    ordered_labels = [
        label
        for label in registry.keys()
        if label in candidates_by_label
    ]
    ordered_labels.extend(
        row.label
        for row in getattr(decision, "candidate_scores", ()) or ()
        if row.label in candidates_by_label and row.label not in ordered_labels
    )
    ordered_labels.extend(
        candidate.label
        for candidate in candidates
        if candidate.label not in ordered_labels
    )
    for order, label in enumerate(ordered_labels):
        if label in rejected_labels:
            continue
        shadow_signals = registry.get(label, ())
        if not shadow_signals:
            continue
        row = score_rows.get(label)
        row_has_data = bool(getattr(row, "has_data", False)) if row is not None else False
        row_score: Optional[float] = None
        threshold: Optional[float] = None
        if min_score is not None:
            try:
                threshold = float(min_score)
            except (TypeError, ValueError):
                continue
        if row is not None:
            try:
                row_score = float(getattr(row, "score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if row_score < 0.0:
                continue
            if row_has_data and threshold is not None and row_score < threshold:
                continue
        candidate = candidates_by_label.get(label)
        if candidate is None:
            continue
        raw_signals, next_signal_id = _real_signals_from_shadow_registry(
            shadow_signals,
            market,
            candidate,
            signal_id_counter=signal_id_counter,
        )
        raw_signals = _filter_shadow_state_entry_signals(
            pipeline,
            candidate,
            raw_signals,
        )
        if raw_signals:
            next_signal_id = max(signal.id for signal in raw_signals) + 1
        if not raw_signals:
            continue
        signal_guard = filter_real_signals_against_tracker(
            raw_signals,
            player=candidate,
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
        if len(getattr(signal_guard, "signals", ()) or ()) <= 0:
            continue
        ranking_score = row_score if row_score is not None else 0.0
        executable_signals = len(getattr(signal_guard, "signals", ()) or ())
        raw_signal_count = len(raw_signals)
        attempt = {
            "leader": candidate,
            "raw_signals": raw_signals,
            "raw_signal_count": raw_signal_count,
            "leader_vote_error_count": 0,
            "signal_guard": signal_guard,
            "signal_id_counter": next_signal_id,
        }
        attempts.append((
            (ranking_score, executable_signals, raw_signal_count, -order),
            attempt,
        ))
    attempted_labels = {attempt["leader"].label for _rank, attempt in attempts}
    agent_registry = {
        str(label): tuple(signals or ())
        for label, signals in dict(
            getattr(pipeline, "_current_shadow_agent_signals", {}) or {}
        ).items()
    }
    for order, agent_label in enumerate(sorted(agent_registry.keys())):
        solo_label = f"Solo_{agent_label}"
        if solo_label == skip_label or solo_label in attempted_labels:
            continue
        if solo_label in rejected_labels:
            continue
        shadow_signals = agent_registry.get(agent_label, ())
        if not shadow_signals:
            continue
        candidate = candidates_by_label.get(solo_label)
        if candidate is None:
            source_agent = getattr(pipeline, "registry", None)
            agent = source_agent.get(agent_label) if source_agent is not None else None
            if agent is None:
                continue
            candidate = EnsemblePlayer(
                label=solo_label,
                agents=[agent],
                weights={agent.label: 1.0},
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
                affinity=market.regime,
            )
        safety_issue = _fallback_candidate_safety_issue(
            pipeline,
            candidate,
            market,
        )
        if safety_issue:
            continue
        ranking_score = _fallback_registry_score(
            pipeline,
            solo_label,
            agent_label,
            market.regime,
            current_bar=market.bar,
        )
        threshold = None
        if min_score is not None:
            try:
                threshold = float(min_score)
            except (TypeError, ValueError):
                continue
        if ranking_score < 0.0:
            continue
        if threshold is not None and ranking_score < threshold:
            continue
        raw_signals, next_signal_id = _real_signals_from_shadow_registry(
            shadow_signals,
            market,
            candidate,
            signal_id_counter=signal_id_counter,
        )
        if not raw_signals:
            continue
        signal_guard = filter_real_signals_against_tracker(
            raw_signals,
            player=candidate,
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
        if len(getattr(signal_guard, "signals", ()) or ()) <= 0:
            continue
        executable_signals = len(getattr(signal_guard, "signals", ()) or ())
        raw_signal_count = len(raw_signals)
        attempt = {
            "leader": candidate,
            "raw_signals": raw_signals,
            "raw_signal_count": raw_signal_count,
            "leader_vote_error_count": 0,
            "signal_guard": signal_guard,
            "signal_id_counter": next_signal_id,
        }
        attempts.append((
            (ranking_score, executable_signals, raw_signal_count, -len(ordered_labels) - order),
            attempt,
        ))
    if not attempts:
        return None
    attempts.sort(key=lambda item: item[0], reverse=True)
    return attempts[0][1]


def _fallback_safety_rejected_labels(rejections: Sequence[object]) -> set[str]:
    prefixes = (
        "hard policy denylist",
        "hard policy untrained genetics",
        "hard policy regime allowlist",
        "hard policy experimental maturity",
        "uses quarantined agent",
        "v3 real-loss kill",
        "v3 persistent real-loss kill",
        "v3 persistent realized-loss kill",
        "v3 probation realized-loss kill",
        "v3 realized profit-lock",
        "real promotion gate:",
        "promotion gate: max_dd_pct",
    )
    labels: set[str] = set()
    for rejection in rejections or ():
        label = str(getattr(rejection, "label", "") or "")
        reason = str(getattr(rejection, "reason", "") or "")
        if label and reason.startswith(prefixes):
            labels.add(label)
    return labels


def _fallback_candidate_safety_issue(
    pipeline: ProductionPipeline,
    candidate: EnsemblePlayer,
    market: MarketSnapshot,
) -> bool:
    reason = _fallback_candidate_safety_reason(pipeline, candidate, market)
    _record_flash_candidate_safety_reason(pipeline, candidate, reason)
    return bool(reason)


def _fallback_candidate_safety_reason(
    pipeline: ProductionPipeline,
    candidate: EnsemblePlayer,
    market: MarketSnapshot,
) -> str:
    if not _candidate_is_real_executable(candidate, pipeline, regime=market.regime):
        return "not real-executable candidate"
    strategist = getattr(pipeline, "strategist", None)
    validate = getattr(strategist, "_validate", None)
    if callable(validate):
        try:
            issue = validate(candidate)
            if issue is not None:
                return str(getattr(issue, "reason", "") or "strategist validation")
        except Exception:
            log.debug("fallback safety quarantine validation failed", exc_info=True)
            return "strategist validation error"
    validate_hard_policy = getattr(strategist, "_validate_hard_policy", None)
    if callable(validate_hard_policy):
        try:
            issue = validate_hard_policy(
                candidate,
                market.regime,
                current_bar=market.bar,
                market_tags=_market_policy_tags(market),
            )
            if issue is not None:
                return str(getattr(issue, "reason", "") or "hard policy validation")
        except Exception:
            log.debug("fallback hard-policy validation failed", exc_info=True)
            return "hard-policy validation error"
    validate_real_loss = getattr(strategist, "_validate_v3_real_loss", None)
    strategist_config = getattr(strategist, "_config", None)
    if (
        callable(validate_real_loss)
        and bool(getattr(strategist_config, "use_v3_rolling_score", False))
    ):
        try:
            issue = validate_real_loss(
                candidate,
                market.regime,
                current_bar=market.bar,
            )
            if issue is not None:
                return str(getattr(issue, "reason", "") or "v3 real-loss validation")
        except Exception:
            log.debug("fallback real-loss validation failed", exc_info=True)
            return "v3 real-loss validation error"
    return ""


def _record_flash_candidate_safety_reason(
    pipeline: ProductionPipeline,
    candidate: EnsemblePlayer,
    reason: str,
) -> None:
    label = str(getattr(candidate, "label", "") or "").strip()
    if not label:
        return
    reasons = getattr(pipeline, "_flash_candidate_safety_reasons", None)
    if not isinstance(reasons, dict):
        reasons = {}
    reasons[label] = str(reason or "")
    try:
        pipeline._flash_candidate_safety_reasons = reasons
    except Exception:
        pass


def _fallback_registry_score(
    pipeline: ProductionPipeline,
    solo_label: str,
    agent_label: str,
    regime: Regime,
    *,
    current_bar: int,
) -> float:
    entry_score = _fallback_entry_causal_score(
        pipeline,
        labels=(solo_label, agent_label),
        regime=regime,
        current_bar=current_bar,
    )
    if entry_score is not None:
        return entry_score
    perf = getattr(pipeline, "perf", None)
    getter = getattr(perf, "get", None)
    if not callable(getter):
        return 0.0
    try:
        regime_metrics = getter(agent_label, regime)
        aggregate_metrics = getter(agent_label)
    except Exception:
        log.debug("fallback registry score lookup failed", exc_info=True)
        return 0.0
    metrics = regime_metrics if regime_metrics.has_data else aggregate_metrics
    if not metrics.has_data:
        try:
            solo_metrics = getter(solo_label, regime)
            if not solo_metrics.has_data:
                solo_metrics = getter(solo_label)
            metrics = solo_metrics if solo_metrics.has_data else metrics
        except Exception:
            pass
    if not metrics.has_data:
        return 0.0
    win_component = min(100.0, max(0.0, float(metrics.win_rate or 0.0))) * 0.02
    trade_component = min(100, max(0, int(metrics.closed_trades or 0))) * 0.01
    dd_penalty = max(0.0, float(metrics.max_dd_pct or 0.0)) * 0.50
    return float(metrics.pnl_pct or 0.0) + win_component + trade_component - dd_penalty


def _fallback_entry_causal_score(
    pipeline: ProductionPipeline,
    *,
    labels: Sequence[str],
    regime: Regime,
    current_bar: int,
) -> Optional[float]:
    strategist = getattr(pipeline, "strategist", None)
    state = getattr(strategist, "_entry_causal_score", None)
    scorer = getattr(state, "score_with_stats", None)
    if not callable(scorer):
        return None
    min_filled = int(getattr(state, "min_filled", 0) or 0)
    for label in labels:
        if not label:
            continue
        try:
            stats = scorer(
                label=label,
                regime=regime.label,
                current_bar=current_bar,
            )
        except Exception:
            log.debug("fallback entry-causal score lookup failed", exc_info=True)
            continue
        if getattr(stats, "has_data", False):
            return float(getattr(stats, "score", 0.0) or 0.0)
        recent_filled = int(getattr(stats, "recent_filled", 0) or 0)
        recent_pnl = float(getattr(stats, "recent_pnl_usd", 0.0) or 0.0)
        if min_filled > 0 and recent_filled >= min_filled and recent_pnl < 0.0:
            return -1.0
    return None


def _real_signals_from_shadow_registry(
    shadow_signals: Sequence[Signal],
    market: MarketSnapshot,
    player: EnsemblePlayer,
    *,
    signal_id_counter: int,
) -> Tuple[List[Signal], int]:
    out: List[Signal] = []
    next_signal_id = int(signal_id_counter)
    for shadow_signal in shadow_signals or ():
        action = getattr(shadow_signal, "action", None)
        if action is None or getattr(action, "is_hold", False):
            continue
        symbol = str(getattr(shadow_signal, "sym", "") or "").upper()
        if symbol not in market.prices:
            continue
        price = float(market.prices.get(symbol, 0.0) or 0.0)
        if price <= 0:
            continue
        out.append(replace(
            shadow_signal,
            id=next_signal_id,
            bar=market.bar,
            sym=symbol,
            action=action,
            price=price,
            regime=market.regime,
            by_player=player.label,
            position_scope="",
            timestamp=market.timestamp,
        ))
        next_signal_id += 1
    return out, next_signal_id


def _find_selected_shadow_position_replay(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    selected_leader: EnsemblePlayer,
    *,
    signal_id_counter: int,
) -> Optional[Dict[str, Any]]:
    if not _shadow_position_replay_enabled(pipeline):
        return None
    positions_by_player = dict(
        getattr(pipeline, "_pending_shadow_player_positions", {}) or {}
    )
    shadow_positions = tuple(
        positions_by_player.get(str(selected_leader.label), ()) or ()
    )
    if not shadow_positions:
        return None
    raw_signals, next_signal_id = _real_signals_from_shadow_positions(
        shadow_positions,
        market,
        selected_leader,
        signal_id_counter=signal_id_counter,
        max_age_bars=_shadow_position_replay_max_age(pipeline),
        require_positive_unrealized=(
            _shadow_position_replay_requires_positive_unrealized(pipeline)
        ),
    )
    if not raw_signals:
        return None
    signal_guard = filter_real_signals_against_tracker(
        raw_signals,
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
    if len(getattr(signal_guard, "signals", ()) or ()) <= 0:
        return None
    return {
        "leader": selected_leader,
        "raw_signals": raw_signals,
        "raw_signal_count": len(raw_signals),
        "leader_vote_error_count": 0,
        "signal_guard": signal_guard,
        "signal_id_counter": next_signal_id,
    }


def _shadow_position_replay_enabled(pipeline: ProductionPipeline) -> bool:
    cfg = getattr(getattr(pipeline, "strategist", None), "_config", None)
    return bool(
        getattr(cfg, "v3_shadow_fresh_handoff_enabled", False)
        or getattr(cfg, "v3_shadow_flat_handoff_enabled", False)
    )


def _shadow_position_replay_max_age(pipeline: ProductionPipeline) -> int:
    cfg = getattr(getattr(pipeline, "strategist", None), "_config", None)
    try:
        return max(0, int(getattr(cfg, "v3_shadow_fresh_handoff_max_age_bars", 1) or 0))
    except (TypeError, ValueError):
        return 1


def _shadow_position_replay_requires_positive_unrealized(
    pipeline: ProductionPipeline,
) -> bool:
    cfg = getattr(getattr(pipeline, "strategist", None), "_config", None)
    return bool(
        getattr(cfg, "v3_shadow_fresh_handoff_require_positive_unrealized", True)
    )


def _real_signals_from_shadow_positions(
    shadow_positions: Sequence[object],
    market: MarketSnapshot,
    player: EnsemblePlayer,
    *,
    signal_id_counter: int,
    max_age_bars: int,
    require_positive_unrealized: bool = True,
) -> Tuple[List[Signal], int]:
    out: List[Signal] = []
    next_signal_id = int(signal_id_counter)
    for payload in shadow_positions or ():
        if not isinstance(payload, dict):
            continue
        symbol = str(payload.get("sym") or "").upper()
        if symbol not in market.prices:
            continue
        side = str(payload.get("side") or "").lower()
        if side == "long":
            action = Action.FUT_LONG_FULL
        elif side == "short":
            action = Action.FUT_SHORT_FULL
        else:
            continue
        age_bars = _safe_nonnegative_int(payload.get("age_bars"))
        if age_bars is None:
            opened_bar = _safe_nonnegative_int(payload.get("opened_bar"))
            age_bars = (
                max(0, int(market.bar) - opened_bar)
                if opened_bar is not None and opened_bar > 0
                else 0
            )
        if age_bars > max_age_bars:
            continue
        if (
            require_positive_unrealized
            and _shadow_position_has_nonpositive_unrealized(payload)
        ):
            continue
        price = float(market.prices.get(symbol, 0.0) or 0.0)
        if price <= 0:
            continue
        out.append(Signal(
            id=next_signal_id,
            bar=market.bar,
            sym=symbol,
            action=action,
            price=price,
            regime=market.regime,
            by_player=player.label,
            by_agent="ShadowPositionReplay",
            position_scope="",
            timestamp=market.timestamp,
        ))
        next_signal_id += 1
    return out, next_signal_id


def _safe_nonnegative_int(value: object) -> Optional[int]:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, out)


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
    rejected_labels = {
        str(getattr(rejection, "label", "") or "")
        for rejection in getattr(decision, "candidate_rejections", ()) or ()
    }
    score_rows = {
        str(getattr(row, "label", "") or ""): row
        for row in getattr(decision, "candidate_scores", ()) or ()
        if str(getattr(row, "label", "") or "")
    }
    min_score = getattr(pipeline, "actionable_fallback_min_score", None)
    require_has_data = bool(
        getattr(pipeline, "actionable_fallback_require_has_data", False)
    )
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
        if label in rejected_labels:
            continue
        row = score_rows.get(label)
        if require_has_data and (
            row is None or not bool(getattr(row, "has_data", False))
        ):
            continue
        if min_score is not None:
            if row is None:
                continue
            try:
                row_score = float(getattr(row, "score", 0.0) or 0.0)
                threshold = float(min_score)
            except (TypeError, ValueError):
                continue
            if row_score < threshold:
                continue
        candidate = candidates_by_label.get(label)
        if candidate is None:
            continue
        snapshot = _capture_vote_state(candidate)
        if snapshot is None:
            continue
        attempt = _vote_candidate_for_real_signals(
            pipeline,
            market,
            candidate,
            signal_id_counter=signal_id_counter,
            trace_id=trace_id,
        )
        if len(getattr(attempt["signal_guard"], "signals", ()) or ()) > 0:
            return attempt
        _restore_vote_state(snapshot)
    return None


def _capture_vote_state(
    player: EnsemblePlayer,
) -> Optional[Tuple[Tuple[object, Dict[str, Any]], ...]]:
    objects: List[object] = [player]
    objects.extend(getattr(player, "agents", ()) or ())
    captured: List[Tuple[object, Dict[str, Any]]] = []
    try:
        for obj in objects:
            state = getattr(obj, "__dict__", None)
            if not isinstance(state, dict):
                continue
            captured.append((obj, copy.deepcopy(state)))
    except Exception:
        return None
    return tuple(captured)


def _restore_vote_state(
    snapshot: Tuple[Tuple[object, Dict[str, Any]], ...],
) -> None:
    for obj, state in reversed(snapshot):
        target = getattr(obj, "__dict__", None)
        if not isinstance(target, dict):
            continue
        target.clear()
        target.update(copy.deepcopy(state))


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
    soft_update = getattr(
        getattr(pipeline, "soft_allocator_execution", None),
        "update_from_shadow_updates",
        None,
    )
    if pending and callable(soft_update):
        try:
            soft_update(pending)
        except Exception:
            log.debug("failed to sync shadow updates into soft allocator", exc_info=True)
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
        pipeline._current_shadow_agent_signals = {}
        pipeline._current_shadow_player_signals = {}
        pipeline._current_actionable_player_labels = set()
        return
    try:
        pipeline._pending_shadow_actor_updates = tuple(getter())
    except Exception:
        log.debug("failed to capture shadow updates for strategist", exc_info=True)
        pipeline._pending_shadow_actor_updates = ()
    agent_signals_getter = getattr(tournament, "last_agent_signals", None)
    if callable(agent_signals_getter):
        try:
            pipeline._current_shadow_agent_signals = dict(agent_signals_getter())
        except Exception:
            log.debug("failed to capture current shadow agent signals", exc_info=True)
            pipeline._current_shadow_agent_signals = {}
    else:
        pipeline._current_shadow_agent_signals = {}
    signals_getter = getattr(tournament, "last_player_signals", None)
    if callable(signals_getter):
        try:
            pipeline._current_shadow_player_signals = dict(signals_getter())
        except Exception:
            log.debug("failed to capture current shadow player signals", exc_info=True)
            pipeline._current_shadow_player_signals = {}
    else:
        pipeline._current_shadow_player_signals = {}
    positions_getter = getattr(tournament, "last_player_open_positions", None)
    if not callable(positions_getter):
        pipeline._pending_shadow_player_positions = None
    else:
        try:
            pipeline._pending_shadow_player_positions = dict(positions_getter())
        except Exception:
            log.debug("failed to capture shadow positions for strategist", exc_info=True)
            pipeline._pending_shadow_player_positions = None
    pipeline._current_actionable_player_labels = _current_actionable_player_labels(
        getattr(pipeline, "_pending_shadow_actor_updates", ()) or (),
        getattr(pipeline, "_current_shadow_player_signals", {}) or {},
        getattr(pipeline, "_current_shadow_agent_signals", {}) or {},
        positions_by_player=getattr(pipeline, "_pending_shadow_player_positions", {}) or {},
        quarantine_manager=getattr(pipeline, "qm", None),
        max_shadow_position_age_bars=_shadow_position_replay_max_age(pipeline),
        require_positive_unrealized=(
            _shadow_position_replay_requires_positive_unrealized(pipeline)
        ),
    )


def _current_actionable_player_labels(
    updates: Sequence[object],
    signals_by_player: Dict[str, Sequence[Signal]],
    signals_by_agent: Optional[Dict[str, Sequence[Signal]]] = None,
    positions_by_player: Optional[Dict[str, Sequence[object]]] = None,
    quarantine_manager: Optional[object] = None,
    max_shadow_position_age_bars: int = 1,
    require_positive_unrealized: bool = True,
) -> set[str]:
    labels = {
        str(label).strip()
        for label, signals in dict(signals_by_player or {}).items()
        if (
            str(label).strip()
            and len(tuple(signals or ())) > 0
            and not _actionable_label_is_quarantined(
                str(label).strip(),
                quarantine_manager,
            )
        )
    }
    for label, signals in dict(signals_by_agent or {}).items():
        clean = str(label).strip()
        if (
            clean
            and len(tuple(signals or ())) > 0
            and not _actionable_label_is_quarantined(clean, quarantine_manager)
        ):
            labels.add(f"Solo_{clean}")
    for event in updates or ():
        actor_type = str(getattr(event, "actor_type", "") or "")
        if actor_type not in {"agent", "player"}:
            continue
        label = str(getattr(event, "actor_label", "") or "").strip()
        if not label:
            continue
        if _actionable_label_is_quarantined(label, quarantine_manager):
            continue
        signals = int(getattr(event, "signals", 0) or 0)
        filled = int(getattr(event, "filled", 0) or 0)
        if signals > 0 or filled > 0:
            labels.add(label if actor_type == "player" else f"Solo_{label}")
    for label, positions in dict(positions_by_player or {}).items():
        clean = str(label).strip()
        if (
            clean
            and not _actionable_label_is_quarantined(clean, quarantine_manager)
            and any(
            _is_fresh_shadow_position_payload(
                item,
                max_age_bars=max_shadow_position_age_bars,
                require_positive_unrealized=require_positive_unrealized,
            )
            for item in positions or ()
            )
        ):
            labels.add(clean)
    return labels


def _actionable_label_is_quarantined(label: str, quarantine_manager: object) -> bool:
    is_quarantined = getattr(quarantine_manager, "is_quarantined", None)
    if not callable(is_quarantined):
        return False
    clean = str(label or "").strip()
    if not clean:
        return False
    try:
        if bool(is_quarantined(clean)):
            return True
        if clean.startswith("Solo_"):
            return bool(is_quarantined(clean[5:]))
    except Exception:
        return False
    return False


def _flash_shadow_player_signals_with_position_replay(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    candidates: Sequence[EnsemblePlayer],
    *,
    agents: Sequence[object] = (),
    signal_id_counter: int,
) -> Dict[str, Tuple[Signal, ...]]:
    signals_by_player: Dict[str, List[Signal]] = {
        str(label).strip(): list(signals or ())
        for label, signals in (
            getattr(pipeline, "_current_shadow_player_signals", {}) or {}
        ).items()
        if str(label).strip()
    }
    if not _shadow_position_replay_enabled(pipeline):
        return {
            label: tuple(signals)
            for label, signals in signals_by_player.items()
            if signals
        }

    positions_by_player = dict(
        getattr(pipeline, "_pending_shadow_player_positions", {}) or {}
    )
    if not positions_by_player:
        return {
            label: tuple(signals)
            for label, signals in signals_by_player.items()
            if signals
        }

    next_signal_id = int(signal_id_counter)
    max_age_bars = _shadow_position_replay_max_age(pipeline)
    require_positive_unrealized = (
        _shadow_position_replay_requires_positive_unrealized(pipeline)
    )
    seen_labels: set[str] = set()
    replay_players: List[EnsemblePlayer] = []

    def add_replay_player(player: EnsemblePlayer) -> None:
        label = str(getattr(player, "label", "") or "").strip()
        if not label or label in seen_labels:
            return
        seen_labels.add(label)
        replay_players.append(player)

    for player in candidates or ():
        add_replay_player(player)

    agents_by_solo_label = {
        f"Solo_{label}": agent
        for agent in agents or ()
        for label in (str(getattr(agent, "label", "") or "").strip(),)
        if label
    }
    for label in positions_by_player.keys():
        if label in seen_labels:
            continue
        agent = agents_by_solo_label.get(str(label).strip())
        if agent is None:
            continue
        add_replay_player(_flash_solo_candidate_for_agent(agent, market.regime))

    for player in replay_players:
        label = str(getattr(player, "label", "") or "").strip()
        shadow_positions = tuple(positions_by_player.get(label, ()) or ())
        if not shadow_positions:
            continue
        replay_signals, next_signal_id = _real_signals_from_shadow_positions(
            shadow_positions,
            market,
            player,
            signal_id_counter=next_signal_id,
            max_age_bars=max_age_bars,
            require_positive_unrealized=require_positive_unrealized,
        )
        if not replay_signals:
            continue
        player_signals = signals_by_player.setdefault(label, [])
        active_symbols = {
            str(getattr(signal, "sym", "") or "").upper()
            for signal in player_signals
            if not getattr(getattr(signal, "action", None), "is_hold", True)
        }
        for signal in replay_signals:
            symbol = str(getattr(signal, "sym", "") or "").upper()
            if not symbol or symbol in active_symbols:
                continue
            player_signals.append(signal)
            active_symbols.add(symbol)

    return {
        label: tuple(signals)
        for label, signals in signals_by_player.items()
        if signals
    }


def _is_fresh_shadow_position_payload(
    payload: object,
    *,
    max_age_bars: int = 1,
    require_positive_unrealized: bool = True,
) -> bool:
    if not isinstance(payload, dict):
        return False
    sym = str(payload.get("sym") or "").strip()
    side = str(payload.get("side") or "").lower()
    if not sym or side not in {"long", "short"}:
        return False
    if (
        require_positive_unrealized
        and _shadow_position_has_nonpositive_unrealized(payload)
    ):
        return False
    age_bars = _safe_nonnegative_int(payload.get("age_bars"))
    if age_bars is None:
        return bool(payload.get("fresh") is True)
    return age_bars <= max(0, int(max_age_bars))


def _shadow_position_has_nonpositive_unrealized(payload: Dict[str, object]) -> bool:
    unrealized = _safe_float_or_none(payload.get("unrealized_pnl_usd"))
    return unrealized is not None and unrealized <= 0.0


def _safe_float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _causal_decision_payload(
    *,
    market: MarketSnapshot,
    selected_leader: EnsemblePlayer,
    executed_leader: EnsemblePlayer,
    decision: SwitchDecision,
    raw_signals: Sequence[Signal],
    executable_signals: Sequence[Signal],
    signal_guard: object,
    leader_vote_error_count: int,
    fallback_used: bool,
    fallback_skipped: bool,
    fallback_candidate: str,
    fallback_reason: str,
    shadow_position_replay_used: bool,
    n_filled: int,
    n_rejected: int,
    n_blocked: int,
    blocked_reasons: Dict[str, int],
    pipeline: ProductionPipeline,
    soft_allocator_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, object]:
    selected_label = str(getattr(selected_leader, "label", "") or "")
    executed_label = str(getattr(executed_leader, "label", "") or "")
    payload = {
        "bar": int(getattr(market, "bar", 0) or 0),
        "timestamp": _timestamp_payload(getattr(market, "timestamp", None)),
        "regime": getattr(getattr(market, "regime", None), "label", str(getattr(market, "regime", ""))),
        "regime_confidence": _safe_float_or_zero(
            getattr(market, "regime_confidence", 0.0),
        ),
        "regimes_by_symbol": _regimes_by_symbol_payload(market),
        "regime_features_by_symbol": _regime_features_by_symbol_payload(market),
        "selected_leader": selected_label,
        "executed_leader": executed_label,
        "decision_reason": str(getattr(decision, "reason", "") or ""),
        "decision_score": _safe_float_or_zero(getattr(decision, "score", 0.0)),
        "decision_margin": _safe_float_or_zero(getattr(decision, "margin", 0.0)),
        "leader_changed": bool(getattr(decision, "switched", False)),
        "leader_vote_error_count": int(leader_vote_error_count),
        "switch_gate_reason": str(getattr(decision, "switch_gate_reason", "") or ""),
        "best_label": str(getattr(decision, "best_label", "") or ""),
        "best_score": _safe_float_or_zero(getattr(decision, "best_score", 0.0)),
        "current_label": str(getattr(decision, "current_label", "") or ""),
        "current_score": _safe_float_or_zero(getattr(decision, "current_score", 0.0)),
        "required_margin": _safe_float_or_zero(getattr(decision, "required_margin", 0.0)),
        "cooldown_passed": bool(getattr(decision, "cooldown_passed", True)),
        "cooldown_blocked": bool(getattr(decision, "cooldown_blocked", False)),
        "fallback_used": bool(fallback_used),
        "fallback_skipped": bool(fallback_skipped),
        "fallback_candidate": str(fallback_candidate or ""),
        "fallback_reason": str(fallback_reason or ""),
        "shadow_position_replay_used": bool(shadow_position_replay_used),
        "prices": _float_mapping(getattr(market, "prices", {}) or {}),
        "funding": _float_mapping(getattr(market, "funding", {}) or {}),
        "current_actionable_labels": sorted(
            str(label)
            for label in (getattr(pipeline, "_current_actionable_player_labels", set()) or set())
        ),
        "selected_shadow_positions": _shadow_position_payloads_for_label(
            pipeline,
            selected_label,
        ),
        "executed_shadow_positions": _shadow_position_payloads_for_label(
            pipeline,
            executed_label,
        ),
        "selected_shadow_signal_count": _shadow_signal_count_for_label(
            pipeline,
            selected_label,
        ),
        "executed_shadow_signal_count": _shadow_signal_count_for_label(
            pipeline,
            executed_label,
        ),
        "raw_signal_count": len(tuple(raw_signals or ())),
        "executable_signal_count": len(tuple(executable_signals or ())),
        "raw_signals": [_signal_payload(sig) for sig in raw_signals or ()],
        "executable_signals": [_signal_payload(sig) for sig in executable_signals or ()],
        "guarded_signals": [_signal_payload(sig) for sig in executable_signals or ()],
        "filtered_real_signals": int(getattr(signal_guard, "filtered", 0) or 0),
        "signal_filter_details": tuple(
            str(item) for item in (getattr(signal_guard, "details", ()) or ())
        ),
        "n_filled": int(n_filled),
        "n_rejected": int(n_rejected),
        "n_blocked": int(n_blocked),
        "blocked_reasons": {
            str(key): int(value)
            for key, value in dict(blocked_reasons or {}).items()
        },
        "candidate_scores": _candidate_score_payloads(decision),
        "candidate_rejections": _candidate_rejection_payloads(decision),
    }
    if soft_allocator_payload:
        payload["soft_allocator"] = dict(soft_allocator_payload)
    return payload


def _flash_causal_decision_payload(
    *,
    market: MarketSnapshot,
    decisions: Sequence[FlashDecision],
    raw_signals: Sequence[Signal],
    executable_signals: Sequence[Signal],
    signal_guard: object,
    n_filled: int,
    n_rejected: int,
    n_blocked: int,
    blocked_reasons: Dict[str, int],
    pipeline: ProductionPipeline,
) -> Dict[str, object]:
    selected_by_symbol = {
        str(getattr(decision, "symbol", "") or ""): str(getattr(decision, "selected_actor", "") or "")
        for decision in decisions or ()
    }
    payload: Dict[str, object] = {
        "flash_enabled": True,
        "bar": int(getattr(market, "bar", 0) or 0),
        "timestamp": _timestamp_payload(getattr(market, "timestamp", None)),
        "regime": getattr(getattr(market, "regime", None), "label", str(getattr(market, "regime", ""))),
        "regime_confidence": _safe_float_or_zero(
            getattr(market, "regime_confidence", 0.0),
        ),
        "regimes_by_symbol": _regimes_by_symbol_payload(market),
        "regime_features_by_symbol": _regime_features_by_symbol_payload(market),
        "selected_leader": "Panteon_Flash",
        "executed_leader": "Panteon_Flash",
        "decision_reason": "flash per-symbol actor selection",
        "leader_changed": False,
        "prices": _float_mapping(getattr(market, "prices", {}) or {}),
        "funding": _float_mapping(getattr(market, "funding", {}) or {}),
        "current_actionable_labels": sorted(
            str(label)
            for label in (getattr(pipeline, "_current_actionable_player_labels", set()) or set())
        ),
        "raw_signal_count": len(tuple(raw_signals or ())),
        "executable_signal_count": len(tuple(executable_signals or ())),
        "raw_signals": [_signal_payload(sig) for sig in raw_signals or ()],
        "executable_signals": [_signal_payload(sig) for sig in executable_signals or ()],
        "guarded_signals": [_signal_payload(sig) for sig in executable_signals or ()],
        "filtered_real_signals": int(getattr(signal_guard, "filtered", 0) or 0),
        "signal_filter_details": tuple(
            str(item) for item in (getattr(signal_guard, "details", ()) or ())
        ),
        "n_filled": int(n_filled),
        "n_rejected": int(n_rejected),
        "n_blocked": int(n_blocked),
        "blocked_reasons": {
            str(key): int(value)
            for key, value in dict(blocked_reasons or {}).items()
        },
        "flash_degraded_signal_keys": sorted(_flash_degraded_signal_keys(pipeline)),
        "flash_degraded_open_symbols": sorted(
            _flash_degraded_open_symbols(pipeline, market)
        ),
        "real_universe_symbol_reject_reasons": dict(
            getattr(pipeline, "_real_universe_symbol_reject_reasons", {}) or {}
        ),
        "flash_selected_actors_by_symbol": selected_by_symbol,
        "flash_actor_types_by_symbol": {
            str(getattr(decision, "symbol", "") or ""): str(getattr(decision, "actor_type", "") or "")
            for decision in decisions or ()
        },
        "flash_scores_by_symbol": {
            str(getattr(decision, "symbol", "") or ""): _safe_float_or_zero(
                getattr(decision, "score", 0.0),
            )
            for decision in decisions or ()
        },
        "flash_decisions": [
            decision.as_dict()
            for decision in decisions or ()
        ],
    }
    shadow_position_diagnostics = _flash_shadow_position_diagnostics(
        pipeline,
        market,
        decisions=decisions,
    )
    if shadow_position_diagnostics is not None:
        payload["shadow_position_diagnostics"] = shadow_position_diagnostics
    return payload


def _signal_payload(signal: Signal) -> Dict[str, object]:
    action = getattr(signal, "action", None)
    regime = getattr(signal, "regime", None)
    return {
        "id": int(getattr(signal, "id", 0) or 0),
        "bar": int(getattr(signal, "bar", 0) or 0),
        "sym": str(getattr(signal, "sym", "") or ""),
        "action": getattr(action, "name", str(action or "")),
        "side": str(getattr(action, "side", "") or ""),
        "price": _safe_float_or_zero(getattr(signal, "price", 0.0)),
        "regime": getattr(regime, "label", str(regime or "")),
        "by_player": str(getattr(signal, "by_player", "") or ""),
        "by_agent": str(getattr(signal, "by_agent", "") or ""),
        "position_scope": str(getattr(signal, "position_scope", "") or ""),
        "risk_mult": _safe_float_or_zero(getattr(signal, "risk_mult", 1.0)),
        "close_fraction": _safe_float_or_zero(
            getattr(signal, "close_fraction", 1.0)
        ),
        "timestamp": _timestamp_payload(getattr(signal, "timestamp", None)),
    }


def _regimes_by_symbol_payload(market: MarketSnapshot) -> Dict[str, str]:
    regimes = getattr(market, "regimes_by_symbol", {}) or {}
    if not isinstance(regimes, Mapping):
        return {}
    return {
        str(symbol).upper(): getattr(regime, "label", str(regime or "")).lower()
        for symbol, regime in regimes.items()
        if str(symbol or "").strip()
    }


def _regime_features_by_symbol_payload(market: MarketSnapshot) -> Dict[str, dict]:
    features = getattr(market, "regime_features_by_symbol", {}) or {}
    if not isinstance(features, Mapping):
        return {}
    out: Dict[str, dict] = {}
    for raw_symbol, raw_payload in features.items():
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol or not isinstance(raw_payload, Mapping):
            continue
        clean_payload: Dict[str, object] = {}
        for raw_key, raw_value in raw_payload.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if isinstance(raw_value, (str, bool)):
                clean_payload[key] = raw_value
                continue
            parsed = _safe_float_or_none(raw_value)
            if parsed is not None:
                clean_payload[key] = parsed
        out[symbol] = clean_payload
    return out


def _shadow_position_payloads_for_label(
    pipeline: ProductionPipeline,
    label: str,
) -> List[Dict[str, object]]:
    positions_by_player = getattr(pipeline, "_pending_shadow_player_positions", {}) or {}
    if not isinstance(positions_by_player, dict):
        return []
    out: List[Dict[str, object]] = []
    for payload in positions_by_player.get(label, ()) or ():
        if not isinstance(payload, dict):
            continue
        out.append({
            "sym": str(payload.get("sym") or ""),
            "side": str(payload.get("side") or ""),
            "opened_bar": _safe_nonnegative_int(payload.get("opened_bar")),
            "age_bars": _safe_nonnegative_int(payload.get("age_bars")),
            "entry_price": _safe_float_or_none(payload.get("entry_price")),
            "current_price": _safe_float_or_none(payload.get("current_price")),
            "qty": _safe_float_or_none(payload.get("qty")),
            "unrealized_pnl_usd": _safe_float_or_none(payload.get("unrealized_pnl_usd")),
            "stop_price": _safe_float_or_none(payload.get("stop_price")),
            "take_profit_price": _safe_float_or_none(payload.get("take_profit_price")),
            "fresh": bool(payload.get("fresh", False)),
        })
    return out


def _shadow_signal_count_for_label(pipeline: ProductionPipeline, label: str) -> int:
    signals_by_player = getattr(pipeline, "_current_shadow_player_signals", {}) or {}
    if not isinstance(signals_by_player, dict):
        return 0
    return len(tuple(signals_by_player.get(label, ()) or ()))


def _flash_shadow_position_diagnostics(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    *,
    decisions: Sequence[FlashDecision] = (),
) -> Optional[Dict[str, object]]:
    bars = {
        int(item)
        for item in (getattr(pipeline, "shadow_position_diagnostic_bars", ()) or ())
        if str(item).strip()
    }
    current_bar = int(getattr(market, "bar", 0) or 0)
    if not bars or current_bar not in bars:
        return None

    positions_by_player = getattr(pipeline, "_pending_shadow_player_positions", {}) or {}
    if not isinstance(positions_by_player, dict):
        positions_by_player = {}
    configured_labels = [
        str(label).strip()
        for label in (getattr(pipeline, "shadow_position_diagnostic_labels", ()) or ())
        if str(label).strip()
    ]
    labels = tuple(dict.fromkeys(configured_labels or sorted(positions_by_player.keys())))
    if not labels:
        return {
            "bar": current_bar,
            "max_age_bars": _shadow_position_replay_max_age(pipeline),
            "require_positive_unrealized": (
                _shadow_position_replay_requires_positive_unrealized(pipeline)
            ),
            "players": [],
        }

    max_age_bars = _shadow_position_replay_max_age(pipeline)
    require_positive_unrealized = (
        _shadow_position_replay_requires_positive_unrealized(pipeline)
    )
    players: List[Dict[str, object]] = []
    for label in labels:
        positions = [
            _shadow_position_diagnostic_payload(
                payload,
                market,
                max_age_bars=max_age_bars,
                require_positive_unrealized=require_positive_unrealized,
                decision_context=_shadow_position_decision_context(
                    decisions,
                    label=label,
                    symbol=str(payload.get("sym") or "").upper(),
                    action_name=(
                        "FUT_LONG_FULL"
                        if str(payload.get("side") or "").lower() == "long"
                        else "FUT_SHORT_FULL"
                        if str(payload.get("side") or "").lower() == "short"
                        else ""
                    ),
                    candidate_safety_reasons=getattr(
                        pipeline,
                        "_flash_candidate_safety_reasons",
                        {},
                    ),
                ),
            )
            for payload in tuple(positions_by_player.get(label, ()) or ())
            if isinstance(payload, dict)
        ]
        players.append({
            "label": label,
            "shadow_signal_count": _shadow_signal_count_for_label(pipeline, label),
            "positions": positions,
        })
    return {
        "bar": current_bar,
        "max_age_bars": max_age_bars,
        "require_positive_unrealized": require_positive_unrealized,
        "players": players,
    }


def _shadow_position_diagnostic_payload(
    payload: Dict[str, object],
    market: MarketSnapshot,
    *,
    max_age_bars: int,
    require_positive_unrealized: bool,
    decision_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    symbol = str(payload.get("sym") or "").upper()
    side = str(payload.get("side") or "").lower()
    if side == "long":
        replay_action = "FUT_LONG_FULL"
    elif side == "short":
        replay_action = "FUT_SHORT_FULL"
    else:
        replay_action = ""

    age_bars = _safe_nonnegative_int(payload.get("age_bars"))
    if age_bars is None:
        opened_bar = _safe_nonnegative_int(payload.get("opened_bar"))
        age_bars = (
            max(0, int(getattr(market, "bar", 0) or 0) - opened_bar)
            if opened_bar is not None and opened_bar > 0
            else 0
        )
    current_price = _safe_float_or_none(
        (getattr(market, "prices", {}) or {}).get(symbol)
    )
    if current_price is None:
        current_price = _safe_float_or_none(payload.get("current_price"))

    block_reason = ""
    if not symbol or symbol not in (getattr(market, "prices", {}) or {}):
        block_reason = "missing_market_price"
    elif not replay_action:
        block_reason = "unsupported_side"
    elif age_bars > max_age_bars:
        block_reason = "stale_position"
    elif require_positive_unrealized and _shadow_position_has_nonpositive_unrealized(payload):
        block_reason = "nonpositive_unrealized"
    elif current_price is None or current_price <= 0.0:
        block_reason = "invalid_market_price"

    out: Dict[str, object] = {
        "sym": symbol,
        "side": side,
        "opened_bar": _safe_nonnegative_int(payload.get("opened_bar")),
        "age_bars": age_bars,
        "entry_price": _safe_float_or_none(payload.get("entry_price")),
        "current_price": current_price,
        "qty": _safe_float_or_none(payload.get("qty")),
        "unrealized_pnl_usd": _safe_float_or_none(payload.get("unrealized_pnl_usd")),
        "fresh": bool(payload.get("fresh", False)),
        "replay_action": replay_action,
        "replay_eligible": not bool(block_reason),
        "replay_block_reason": block_reason,
    }
    if decision_context:
        out["decision_context"] = decision_context
    return out


def _shadow_position_decision_context(
    decisions: Sequence[FlashDecision],
    *,
    label: str,
    symbol: str,
    action_name: str,
    candidate_safety_reasons: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    clean_label = str(label or "").strip()
    clean_symbol = str(symbol or "").upper()
    clean_action = str(action_name or "").strip().upper()
    decision = next(
        (
            item
            for item in decisions or ()
            if str(getattr(item, "symbol", "") or "").upper() == clean_symbol
        ),
        None,
    )
    safety_reason = str(
        dict(candidate_safety_reasons or {}).get(clean_label, "") or ""
    )
    if decision is None:
        out = {"candidate_found": False}
        if safety_reason:
            out["pre_allocator_safety_reason"] = safety_reason
        return out

    out: Dict[str, object] = {
        "decision_selected_actor": str(getattr(decision, "selected_actor", "") or ""),
        "decision_actor_type": str(getattr(decision, "actor_type", "") or ""),
        "decision_action": getattr(
            getattr(decision, "action", None),
            "name",
            str(getattr(decision, "action", "") or ""),
        ),
        "decision_reason": str(getattr(decision, "reason", "") or ""),
        "candidate_found": False,
    }
    if safety_reason:
        out["pre_allocator_safety_reason"] = safety_reason
    candidate = next(
        (
            row
            for row in tuple(getattr(decision, "candidates", ()) or ())
            if str(getattr(row, "label", "") or "").strip() == clean_label
            and str(getattr(row, "actor_type", "") or "").strip() == "ensemble"
            and getattr(getattr(row, "action", None), "name", "") == clean_action
        ),
        None,
    )
    if candidate is None:
        return out

    out.update({
        "candidate_found": True,
        "candidate_actor_key": str(getattr(candidate, "actor_key", "") or ""),
        "candidate_rank": int(getattr(candidate, "rank", 0) or 0),
        "candidate_rejected": bool(getattr(candidate, "rejected", False)),
        "candidate_reason": str(getattr(candidate, "reason", "") or ""),
        "candidate_score": _safe_float_or_zero(getattr(candidate, "score", 0.0)),
        "candidate_base_score": _safe_float_or_zero(
            getattr(candidate, "base_score", 0.0)
        ),
        "candidate_gate_score": _safe_float_or_zero(
            getattr(candidate, "gate_score", 0.0)
        ),
        "candidate_closed_trades": int(
            getattr(candidate, "closed_trades", 0) or 0
        ),
        "candidate_shadow_score": _safe_float_or_zero(
            getattr(candidate, "shadow_score", 0.0)
        ),
        "candidate_shadow_closed_trades": int(
            getattr(candidate, "shadow_closed_trades", 0) or 0
        ),
        "candidate_shadow_source": str(
            getattr(candidate, "shadow_source", "") or ""
        ),
        "candidate_selected_subset_protected": bool(
            getattr(candidate, "selected_subset_protected", False)
        ),
        "candidate_selected_subset_score_boost": _safe_float_or_zero(
            getattr(candidate, "selected_subset_score_boost", 0.0)
        ),
        "candidate_selected_subset_risk_mult": _safe_float_or_zero(
            getattr(candidate, "selected_subset_risk_mult", 1.0)
        ),
    })
    return out


def _candidate_score_payloads(
    decision: SwitchDecision,
    *,
    limit: int = 20,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in tuple(getattr(decision, "candidate_scores", ()) or ())[:limit]:
        rows.append({
            "label": str(getattr(row, "label", "") or ""),
            "rank": int(getattr(row, "rank", 0) or 0),
            "score": _safe_float_or_zero(getattr(row, "score", 0.0)),
            "score_source": str(getattr(row, "score_source", "") or ""),
            "has_data": bool(getattr(row, "has_data", False)),
            "closed_trades": int(getattr(row, "closed_trades", 0) or 0),
            "signals": int(getattr(row, "signals", 0) or 0),
            "execution_failures": int(getattr(row, "execution_failures", 0) or 0),
            "uncertainty_penalty": _safe_float_or_zero(
                getattr(row, "uncertainty_penalty", 0.0),
            ),
            "agent_labels": tuple(
                str(item) for item in (getattr(row, "agent_labels", ()) or ())
            ),
            "session_score_delta": _safe_float_or_zero(
                getattr(row, "session_score_delta", 0.0),
            ),
            "session_pnl_pct": _safe_float_or_zero(
                getattr(row, "session_pnl_pct", 0.0),
            ),
            "recent_bars": int(getattr(row, "recent_bars", 0) or 0),
            "recent_actionable_bars": int(
                getattr(row, "recent_actionable_bars", 0) or 0,
            ),
            "actionable_share": _safe_float_or_zero(
                getattr(row, "actionable_share", 0.0),
            ),
            "recent_filled": int(getattr(row, "recent_filled", 0) or 0),
            "recent_pnl_usd": _safe_float_or_zero(
                getattr(row, "recent_pnl_usd", 0.0),
            ),
        })
    return rows


def _candidate_rejection_payloads(
    decision: SwitchDecision,
    *,
    limit: int = 50,
) -> List[Dict[str, object]]:
    return [
        {
            "label": str(getattr(row, "label", "") or ""),
            "reason": str(getattr(row, "reason", "") or ""),
        }
        for row in tuple(getattr(decision, "candidate_rejections", ()) or ())[:limit]
    ]


def _float_mapping(values: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in dict(values or {}).items():
        parsed = _safe_float_or_none(value)
        if parsed is not None:
            out[str(key)] = parsed
    return out


def _safe_float_or_zero(value: object) -> float:
    parsed = _safe_float_or_none(value)
    return parsed if parsed is not None else 0.0


def _timestamp_payload(value: object) -> str:
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return str(value or "")
    return str(value or "")


def _sync_current_actionable_labels_to_strategist(pipeline: ProductionPipeline) -> None:
    updater = getattr(
        getattr(pipeline, "strategist", None),
        "update_current_actionable_labels",
        None,
    )
    if not callable(updater):
        return
    try:
        updater(getattr(pipeline, "_current_actionable_player_labels", set()) or set())
    except Exception:
        log.debug("failed to sync current actionable labels into strategist", exc_info=True)


def _emit_candidate_audit_events(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    decision: SwitchDecision,
    *,
    decision_id: str,
    trace_id: str,
    context: Dict[str, str],
) -> None:
    emit_candidate_audit_events(
        pipeline,
        market,
        decision,
        decision_id=decision_id,
        trace_id=trace_id,
        context=context,
    )


def _emit_flash_audit_events(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    decisions: Sequence[FlashDecision],
    *,
    decision_id: str,
    trace_id: str,
    context: Dict[str, str],
) -> None:
    if not bool(getattr(pipeline, "flash_audit_events_enabled", True)):
        return
    emit_flash_audit_events(
        pipeline,
        market,
        decisions,
        decision_id=decision_id,
        trace_id=trace_id,
        context=context,
    )


def _record_agent_vote_failures(
    pipeline: ProductionPipeline,
    market: MarketSnapshot,
    player: EnsemblePlayer,
    *,
    trace_id: str,
    errors: Optional[Sequence[object]] = None,
) -> int:
    count = 0
    for err in (errors if errors is not None else getattr(player, "last_vote_errors", []) or []):
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


def _hybrid_open_position_count(pipeline: ProductionPipeline) -> int:
    tracker = getattr(pipeline, "position_tracker", None)
    if tracker is None:
        return 0
    try:
        oc = getattr(tracker, "open_count", None)
        if callable(oc):
            return int(oc())
        all_open = getattr(tracker, "all_open", None)
        if callable(all_open):
            return len(all_open() or {})
    except Exception:
        return 0
    return 0


def _hybrid_resolve_use_flash(pipeline: ProductionPipeline, market: MarketSnapshot) -> bool:
    """Решает, использовать ли flash-путь (vs агрессивный strategist).

    Режимы:
      • slow switch (#1, рекомендуемый): меняет путь ТОЛЬКО когда flat и после
        dwell — strategist работает непрерывно, без коррупции эджа.
      • legacy per-bar (по умолчанию выключен): мгновенный период/бар-триггер
        (доказанно коррумпирует strategist — оставлено для совместимости).
    """
    if not bool(getattr(pipeline, "flash_enabled", False)):
        return False
    trending = _hybrid_market_trending(pipeline, market)

    if bool(getattr(pipeline, "hybrid_slow_switch_enabled", False)):
        desired = "strategist" if trending else "flash"
        current = str(getattr(pipeline, "_hybrid_active_path", "flash") or "flash")
        if current != desired:
            flat = _hybrid_open_position_count(pipeline) == 0
            since = int(getattr(pipeline, "_hybrid_path_since_bar", 0) or 0)
            dwell = int(getattr(pipeline, "hybrid_min_dwell_bars", 720) or 0)
            if flat and (int(market.bar) - since) >= dwell:
                pipeline._hybrid_active_path = desired
                pipeline._hybrid_path_since_bar = int(market.bar)
        return str(getattr(pipeline, "_hybrid_active_path", "flash") or "flash") == "flash"

    # legacy per-bar (off by default)
    if trending:
        return False
    hybrid_regimes = getattr(pipeline, "hybrid_strategist_regimes", ()) or ()
    if hybrid_regimes and market.regime in hybrid_regimes:
        return False
    return True


def _hybrid_market_trending(pipeline: ProductionPipeline, market: MarketSnapshot) -> bool:
    """Период-уровневый триггер тренда: |трейлинг BTC-return| за окно ≥ порога.

    Преимущество strategist-пути — период-уровневое (он выигрывает в трендовые
    периоды целиком), поэтому переключаем путь по медленному трейлинг-тренду, а
    не по бар-режиму. window=0 → выключено (всегда False).
    """
    window = int(getattr(pipeline, "hybrid_trend_window_bars", 0) or 0)
    if window <= 0:
        return False
    prices = getattr(market, "prices", {}) or {}
    btc = prices.get("BTC/USDT") or prices.get("BTCUSDT") or prices.get("BTC")
    if btc is None:
        return False
    try:
        btc = float(btc)
    except (TypeError, ValueError):
        return False
    if btc <= 0:
        return False
    hist = getattr(pipeline, "_hybrid_trend_prices", None)
    if hist is None:
        from collections import deque as _deque
        hist = _deque(maxlen=window)
        pipeline._hybrid_trend_prices = hist
    elif hist.maxlen != window:
        from collections import deque as _deque
        hist = _deque(hist, maxlen=window)
        pipeline._hybrid_trend_prices = hist
    anchor = float(hist[0]) if len(hist) >= window and hist[0] else 0.0
    hist.append(btc)
    if anchor <= 0:
        return False
    ret_pct = (btc / anchor - 1.0) * 100.0
    threshold = float(getattr(pipeline, "hybrid_trend_threshold_pct", 15.0) or 15.0)
    return abs(ret_pct) >= threshold


def _live_config(pipeline: ProductionPipeline) -> Any:
    return getattr(pipeline, "live_execution", None)


def _kill_state(pipeline: ProductionPipeline) -> Any:
    return getattr(pipeline, "kill_switch", None)


def _set_kill_switch(
    pipeline: ProductionPipeline,
    reason: str,
    *,
    kind: str = "",
) -> None:
    state = _kill_state(pipeline)
    if state is None or not reason:
        return
    if getattr(state, "disabled_reason", ""):
        return
    state.disabled_reason = reason
    # Запоминаем категорию и бар защёлки для возможного авто-recovery (A7).
    if hasattr(state, "disabled_kind"):
        state.disabled_kind = str(kind or "")
    if hasattr(state, "disabled_bar"):
        state.disabled_bar = int(getattr(state, "last_seen_bar", 0) or 0)
    log.error("real trading disabled by kill switch: %s", reason)


def _maybe_recover_kill_switch(pipeline: ProductionPipeline) -> None:
    """Снять защёлку kill-switch, если включён авто-recovery и условия выполнены.

    Безопасно по умолчанию: при ``kill_switch_auto_recovery_enabled = False``
    (дефолт) ничего не делает — поведение прод сохраняется.

    Условия снятия:
      • прошёл cooldown в барах с момента защёлки;
      • для equity-категорий (daily_loss / peak_drawdown) equity восстановилась —
        остаточная просадка от пика ≤ recovery_max_drawdown_pct (0 = полное
        восстановление до пика);
      • операционные категории (desync / stale_feed / failed_orders / api / slippage)
        снимаются по одному cooldown, с обнулением соответствующих счётчиков.
    """
    state = _kill_state(pipeline)
    if state is None or not getattr(state, "disabled_reason", ""):
        return
    cfg = _live_config(pipeline)
    if not bool(getattr(cfg, "kill_switch_auto_recovery_enabled", False)):
        return

    cooldown = int(getattr(cfg, "kill_switch_recovery_cooldown_bars", 0) or 0)
    # ВНИМАНИЕ: не использовать `or` для дефолта — бар 0 ложный (0 or -1 == -1).
    disabled_bar_raw = getattr(state, "disabled_bar", -1)
    disabled_bar = int(disabled_bar_raw) if disabled_bar_raw is not None else -1
    last_bar_raw = getattr(state, "last_seen_bar", 0)
    last_bar = int(last_bar_raw) if last_bar_raw is not None else 0
    if disabled_bar < 0 or (last_bar - disabled_bar) < cooldown:
        return

    kind = str(getattr(state, "disabled_kind", "") or "")
    equity_kinds = {"daily_loss", "peak_drawdown"}
    if kind in equity_kinds:
        current = float(getattr(pipeline, "current_balance", 0.0) or 0.0)
        if current <= 0:
            return
        residual_dd_pct = float(
            getattr(cfg, "kill_switch_recovery_max_drawdown_pct", 0.0) or 0.0
        )
        if kind == "peak_drawdown":
            peak = float(getattr(state, "peak_equity_usd", 0.0) or 0.0)
            baseline = peak
        else:  # daily_loss — база восстановления = стартовый капитал
            baseline = float(getattr(pipeline, "initial_capital", 0.0) or 0.0)
        if baseline <= 0:
            return
        recovery_target = baseline * (1.0 - residual_dd_pct / 100.0)
        if current < recovery_target:
            return

    prev_reason = state.disabled_reason
    state.disabled_reason = ""
    if hasattr(state, "disabled_kind"):
        state.disabled_kind = ""
    if hasattr(state, "disabled_bar"):
        state.disabled_bar = -1
    # Сбрасываем транзиентные счётчики, чтобы не зациклить мгновенный повторный trip.
    for attr in (
        "consecutive_failed_orders",
        "api_error_streak",
        "exchange_desync_events",
        "stale_feed_polls",
    ):
        if hasattr(state, attr):
            setattr(state, attr, 0)
    log.warning(
        "kill switch auto-recovered (kind=%s) after cooldown; prior reason: %s",
        kind or "unknown",
        prev_reason,
    )


def _record_live_equity_peak(pipeline: ProductionPipeline, equity: float) -> None:
    state = _kill_state(pipeline)
    if state is None:
        return
    try:
        current = float(equity)
    except (TypeError, ValueError):
        return
    if current <= 0:
        return
    peak = float(getattr(state, "peak_equity_usd", 0.0) or 0.0)
    if current > peak:
        state.peak_equity_usd = current


def _kill_switch_reason(
    pipeline: ProductionPipeline,
    market: Optional[MarketSnapshot] = None,
) -> str:
    state = _kill_state(pipeline)
    if state is not None and market is not None and hasattr(state, "last_seen_bar"):
        try:
            state.last_seen_bar = int(market.bar)
        except (TypeError, ValueError):
            pass
    # Авто-recovery защёлки (no-op, если выключено в конфиге).
    _maybe_recover_kill_switch(pipeline)
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
                _set_kill_switch(pipeline, reason, kind="daily_loss")
                return reason
    max_peak_drawdown_pct = float(
        getattr(cfg, "max_equity_peak_drawdown_pct", 0.0) or 0.0
    )
    if max_peak_drawdown_pct > 0 and state is not None:
        current = float(getattr(pipeline, "current_balance", 0.0) or 0.0)
        if current > 0:
            _record_live_equity_peak(pipeline, current)
            peak = float(getattr(state, "peak_equity_usd", 0.0) or 0.0)
            if peak > 0:
                floor = peak * (1.0 - max_peak_drawdown_pct / 100.0)
                if current <= floor:
                    drawdown_pct = (peak - current) / peak * 100.0
                    reason = (
                        "equity peak drawdown exceeded: "
                        f"current={current:.2f}, peak={peak:.2f}, "
                        f"drawdown={drawdown_pct:.2f}%, floor={floor:.2f}"
                    )
                    _set_kill_switch(pipeline, reason, kind="peak_drawdown")
                    return reason
    return ""


def _is_exchange_open_capability_error(reason: str) -> bool:
    """True для биржевых кодов запрета открытий (MEXC 8950/6026 и эквиваленты)."""
    text = str(reason or "").lower()
    if "8950" in text or "6026" in text:
        return True
    return (
        ("open" in text)
        and (
            "unavailable" in text
            or "not allowed" in text
            or "risk control" in text
            or "region" in text
            or "country" in text
        )
    )


def _exchange_health_open_block_reason(pipeline: ProductionPipeline) -> str:
    if bool(getattr(pipeline, "exchange_open_capability_blocked", False)):
        return "exchange_open_capability_blocked (8950/6026): close-only"
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
        if _mexc_transient_assets_exception_is_recoverable(
            pipeline,
            snapshot=snapshot,
            health=health,
            reason=reason,
        ):
            return ""
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


def _mexc_transient_assets_exception_is_recoverable(
    pipeline: ProductionPipeline,
    *,
    snapshot: Dict[str, Any],
    health: Dict[str, Any],
    reason: str,
) -> bool:
    reason_l = str(reason or "").strip().lower()
    if not reason_l.startswith("futures_assets_exception:"):
        return False
    if not any(token in reason_l for token in ("connection", "timeout")):
        return False
    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    exchange_name = str(
        getattr(exchange, "name", "")
        or getattr(pipeline, "exchange_name", "")
        or ""
    ).strip().upper()
    if exchange_name != "MEXC":
        return False
    if bool(health.get("uses_cached_balance") or health.get("cached_equity")):
        return False
    if health.get("snapshot_healthy") is False:
        return False
    balance = _snapshot_balance(snapshot)
    if balance is None or balance <= 0.0:
        return False
    reliable = getattr(exchange, "positions_snapshot_reliable", None)
    if not callable(reliable):
        return False
    try:
        return bool(reliable())
    except Exception:
        return False


def _record_exchange_desync(pipeline: ProductionPipeline, summary: Any) -> None:
    cfg = _live_config(pipeline)
    limit = int(getattr(cfg, "max_exchange_desync_events", 0) or 0)
    if limit <= 0 or not isinstance(summary, dict):
        return
    if bool(summary.get("snapshot_unreliable")):
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
            kind="exchange_desync",
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
        _set_kill_switch(
            pipeline,
            f"stale feed polls {idle_polls} >= {limit}",
            kind="stale_feed",
        )


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
        probation_label = _genetics_probation_label_for_signal(pipeline, signal)
        if probation_label:
            state.genetics_probation_consecutive_failed_orders = 0
            by_label = getattr(
                state,
                "genetics_probation_consecutive_failed_orders_by_label",
                None,
            )
            if isinstance(by_label, dict):
                by_label[probation_label] = 0

    cfg = _live_config(pipeline)
    limit = float(getattr(cfg, "max_slippage_pct", 0.0) or 0.0)
    slippage_pct = _result_slippage_pct(signal, result)
    if limit > 0 and slippage_pct is not None and slippage_pct > limit:
        _set_kill_switch(
            pipeline,
            f"excessive slippage {slippage_pct:.4f}% > {limit:.4f}%",
            kind="slippage",
        )
    allocator = getattr(pipeline, "flash_allocator", None)
    recorder = getattr(allocator, "record_controlled_exploration_execution_result", None)
    if callable(recorder):
        try:
            recorder(result)
        except Exception:
            log.exception("controlled exploration outcome memory update failed")


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
    _record_soft_allocator_realized_gate_result(pipeline, result)
    _disable_genetics_probation_if_realized_loss_exceeded(pipeline)


def _record_soft_allocator_realized_gate_result(
    pipeline: ProductionPipeline,
    result: ExecutionResult,
) -> None:
    if not _soft_allocator_realized_gate_enabled(pipeline):
        return
    counts_by_player = {
        str(label or ""): int(count or 0)
        for label, count in getattr(result, "closed_trade_counts_by_player", ()) or ()
        if str(label or "")
    }
    if not counts_by_player:
        return
    wins_by_player = {
        str(label or ""): int(count or 0)
        for label, count in getattr(result, "win_counts_by_player", ()) or ()
        if str(label or "")
    }
    pnl_by_player = {
        str(label or ""): float(pnl or 0.0)
        for label, pnl in getattr(result, "realized_pnl_by_player", ()) or ()
        if str(label or "")
    }
    try:
        bar = int(getattr(getattr(result, "signal", None), "bar", 0) or 0)
    except (TypeError, ValueError):
        bar = 0
    events_by_player = {
        str(label): [
            _normalize_soft_allocator_realized_event(item)
            for item in (events or ())
        ]
        for label, events in dict(
            getattr(pipeline, "_soft_allocator_realized_events_by_player", {}) or {}
        ).items()
    }
    trim_lookback = int(
        max(0, getattr(pipeline, "soft_allocator_realized_gate_lookback_bars", 0) or 0)
    )
    trim_floor = max(0, bar - trim_lookback * 2) if trim_lookback > 0 else 0
    for label, closed in counts_by_player.items():
        if closed <= 0:
            continue
        events = events_by_player.setdefault(label, [])
        events.append((
            bar,
            float(pnl_by_player.get(label, 0.0) or 0.0),
            int(closed),
            int(wins_by_player.get(label, 0) or 0),
        ))
        if trim_floor > 0:
            events_by_player[label] = [
                event for event in events if int(event[0] or 0) >= trim_floor
            ]
    pipeline._soft_allocator_realized_events_by_player = events_by_player


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


def _record_order_failure(
    pipeline: ProductionPipeline,
    reason: str,
    *,
    signal: Optional[Signal] = None,
) -> None:
    state = _kill_state(pipeline)
    cfg = _live_config(pipeline)
    if state is None:
        return
    # ExchangeCapabilityGuard (#2): биржевые коды 8950/6026 = «открытия запрещены
    # (регион/risk-control)». Это НЕ деградация бота — переводим биржу в close-only
    # и НЕ копим consecutive_failed_orders (иначе доходим до kill switch впустую).
    if _is_exchange_open_capability_error(reason):
        pipeline.exchange_open_capability_blocked = True
        if not getattr(pipeline, "_exchange_capability_logged", False):
            log.error(
                "ExchangeCapabilityGuard: exchange returned open-capability error "
                "(8950/6026); switching to CLOSE-ONLY, suppressing open retries. reason=%s",
                reason,
            )
            pipeline._exchange_capability_logged = True
        return
    state.consecutive_failed_orders += 1
    if _looks_like_api_error(reason):
        state.api_error_streak += 1
    _record_genetics_probation_order_failure(pipeline, signal, reason)

    failed_limit = int(getattr(cfg, "max_consecutive_failed_orders", 0) or 0)
    api_limit = int(getattr(cfg, "max_api_error_streak", 0) or 0)
    if failed_limit > 0 and state.consecutive_failed_orders >= failed_limit:
        _set_kill_switch(
            pipeline,
            (
                f"consecutive failed orders "
                f"{state.consecutive_failed_orders} >= {failed_limit}"
            ),
            kind="failed_orders",
        )
    if api_limit > 0 and state.api_error_streak >= api_limit:
        _set_kill_switch(
            pipeline,
            f"API error storm {state.api_error_streak} >= {api_limit}",
            kind="api_storm",
        )


def _record_genetics_probation_order_failure(
    pipeline: ProductionPipeline,
    signal: Optional[Signal],
    reason: str,
) -> None:
    label = _genetics_probation_label_for_signal(pipeline, signal)
    if not label:
        return
    state = _kill_state(pipeline)
    cfg = _live_config(pipeline)
    if state is None:
        return
    by_label = getattr(
        state,
        "genetics_probation_consecutive_failed_orders_by_label",
        None,
    )
    if not isinstance(by_label, dict):
        by_label = {}
        state.genetics_probation_consecutive_failed_orders_by_label = by_label
    count = int(by_label.get(label, 0) or 0) + 1
    by_label[label] = count
    state.genetics_probation_consecutive_failed_orders = count
    limit = _genetics_probation_int_for_label(
        cfg,
        "genetics_probation_max_consecutive_failed_orders_by_label",
        "genetics_probation_max_consecutive_failed_orders",
        label,
        0,
    )
    if limit > 0 and count >= limit:
        _set_genetics_probation_disabled(
            pipeline,
            (
                f"{label} consecutive failed orders "
                f"{count} >= {limit}"
            ),
            label=label,
        )


def _disable_genetics_probation_if_realized_loss_exceeded(
    pipeline: ProductionPipeline,
) -> None:
    cfg = _live_config(pipeline)
    labels = _genetics_probation_labels(pipeline)
    if not labels:
        return
    perf = getattr(pipeline, "real_perf", None)
    getter = getattr(perf, "get", None)
    if not callable(getter):
        return
    for label in labels:
        try:
            metrics = getter(label)
        except Exception:
            continue
        if _genetics_probation_disabled_reason(pipeline, label):
            continue
        max_loss_pct = _genetics_probation_float_for_label(
            cfg,
            "genetics_probation_max_realized_loss_pct_by_label",
            "genetics_probation_max_realized_loss_pct",
            label,
            0.0,
        )
        if max_loss_pct <= 0:
            continue
        closed = int(getattr(metrics, "closed_trades", 0) or 0)
        pnl_pct = float(getattr(metrics, "pnl_pct", 0.0) or 0.0)
        if closed > 0 and pnl_pct <= -max_loss_pct:
            _set_genetics_probation_disabled(
                pipeline,
                (
                    f"{label} realized loss {pnl_pct:.4f}% "
                    f"<= -{max_loss_pct:.4f}% after {closed} closed trades"
                ),
                label=label,
            )
            return


def _genetics_probation_labels(pipeline: ProductionPipeline) -> set[str]:
    cfg = _live_config(pipeline)
    if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
        return set()
    return set(_string_tuple(
        getattr(cfg, "genetics_probation_labels", ("GeneticsResearch",)),
    ))


def _genetics_probation_label_for_signal(
    pipeline: ProductionPipeline,
    signal: Optional[Signal],
) -> str:
    if signal is None:
        return ""
    labels = _genetics_probation_labels(pipeline)
    if not labels:
        return ""
    label = _genetics_probation_signal_label(
        leader_label=str(getattr(signal, "by_player", "") or ""),
        signal=signal,
        labels=labels,
    )
    if label:
        return label
    player_label = str(getattr(signal, "by_player", "") or "")
    if player_label not in {"Panteon_Flash", "PanteonFlash", "PanteonFlashAdopted"}:
        return ""
    agent_label = str(getattr(signal, "by_agent", "") or "")
    if agent_label in labels:
        return agent_label
    return ""


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
