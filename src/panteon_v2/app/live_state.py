"""Live-state synchronization for the real Panteon execution path.

The live exchange/position tracker is the single source of truth for real
orders. V1 agents may keep their own ``pos``/``ep``/``et`` maps after warmup or
after a restart, so this module reconciles those maps before real voting and
filters impossible real signals before they reach EventLog/TradeExecutor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ..attribution.events import PositionClosed
from ..domain.types import Action, Regime, Signal
from ..execution.position_tracker import TrackedPosition


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealSignalGuardResult:
    signals: List[Signal]
    filtered: int = 0
    stale_closes: int = 0
    foreign_closes: int = 0
    duplicate_opens: int = 0
    rate_limited_opens: int = 0
    max_position_saturated_opens: int = 0
    external_position_signals: int = 0
    details: List[str] = field(default_factory=list)


EXTERNAL_POSITION_PLAYER_LABELS = {"", "RecoveredExchangePosition"}
ADOPTED_POSITION_PLAYER_LABEL = "PanteonFlashAdopted"
ADOPTED_POSITION_AGENT_LABEL = "AdoptedExchangePosition"
SYSTEM_CLOSE_AGENTS = {
    "CashFlat",
    "GeneticsProbationRegimeExit",
    "PartialProfitLock",
    "ShadowPositionReplay",
    "StalePositionGuard",
}


def is_external_position(pos: Dict[str, Any] | Any) -> bool:
    if isinstance(pos, dict):
        owner = pos.get("by_player", "")
    else:
        owner = getattr(pos, "by_player", "")
    return str(owner or "") in EXTERNAL_POSITION_PLAYER_LABELS


def is_adopted_position(pos: Dict[str, Any] | Any, pipeline: Any = None) -> bool:
    if isinstance(pos, dict):
        owner = pos.get("by_player", "")
    else:
        owner = getattr(pos, "by_player", "")
    adopted_player, _ = adopted_position_labels(pipeline)
    return str(owner or "") == adopted_player


def close_signal_matches_position_owner(signal: Signal, pos: Dict[str, Any] | Any) -> bool:
    """Return True when a real close is owned or emitted by a system guard."""
    if not signal.action.is_close:
        return True
    if is_external_position(pos):
        return False
    closer_agent = str(getattr(signal, "by_agent", "") or "").strip()
    if closer_agent in SYSTEM_CLOSE_AGENTS:
        return True
    closer_player = str(getattr(signal, "by_player", "") or "").strip()
    if isinstance(pos, dict):
        owner_player = str(pos.get("by_player", "") or "").strip()
        owner_agent = str(pos.get("by_agent", "") or "").strip()
    else:
        owner_player = str(getattr(pos, "by_player", "") or "").strip()
        owner_agent = str(getattr(pos, "by_agent", "") or "").strip()
    owners = {label for label in (owner_player, owner_agent) if label}
    closers = {label for label in (closer_player, closer_agent) if label}
    if owners & closers:
        return True
    return False


def adopted_position_labels(pipeline: Any = None) -> tuple[str, str]:
    cfg = getattr(pipeline, "live_execution", None)
    player = str(
        getattr(cfg, "adopt_existing_position_player", ADOPTED_POSITION_PLAYER_LABEL)
        or ADOPTED_POSITION_PLAYER_LABEL
    ).strip() or ADOPTED_POSITION_PLAYER_LABEL
    agent = str(
        getattr(cfg, "adopt_existing_position_agent", ADOPTED_POSITION_AGENT_LABEL)
        or ADOPTED_POSITION_AGENT_LABEL
    ).strip() or ADOPTED_POSITION_AGENT_LABEL
    return player, agent


def should_adopt_existing_position(pipeline: Any, symbol: str) -> bool:
    cfg = getattr(pipeline, "live_execution", None)
    if not bool(getattr(cfg, "adopt_existing_positions_enabled", False)):
        return False
    raw_symbols = tuple(getattr(cfg, "adopt_existing_position_symbols", ()) or ())
    allowed = {_normalize_symbol_key(item) for item in raw_symbols if str(item or "").strip()}
    if not allowed:
        return True
    if "*" in allowed or "ALL" in allowed:
        return True
    variants = _symbol_variants(symbol)
    return bool(allowed & variants)


def seed_perf_open_position_for_adoption(pipeline: Any, pos: TrackedPosition) -> None:
    perf = getattr(pipeline, "perf", None)
    seed = getattr(perf, "seed_open_position", None)
    if not callable(seed):
        return
    try:
        regime = Regime.from_string(str(getattr(pos, "open_regime", "") or "neutral"))
    except Exception:
        regime = Regime.NEUTRAL
    labels = [
        str(getattr(pos, "by_agent", "") or ""),
        str(getattr(pos, "by_player", "") or ""),
    ]
    seed(
        labels=labels,
        regime=regime,
        sym=str(getattr(pos, "sym", "") or ""),
        side=str(getattr(pos, "side", "") or ""),
        entry_price=float(getattr(pos, "entry_price", 0.0) or 0.0),
        qty=float(getattr(pos, "qty", 0.0) or 0.0),
        fee_open=float(getattr(pos, "fee_open", 0.0) or 0.0),
        funding_open=float(getattr(pos, "funding_open", 0.0) or 0.0),
        signal_id=int(getattr(pos, "open_signal_id", 0) or 0),
        position_scope="",
        count_entry=False,
    )


def _normalize_symbol_key(symbol: Any) -> str:
    return str(symbol or "").strip().upper().replace("-", "/")


def _symbol_variants(symbol: Any) -> set[str]:
    raw = _normalize_symbol_key(symbol)
    if not raw:
        return set()
    variants = {raw}
    if raw.endswith("/USDT"):
        variants.add(raw[:-5])
    elif "/" not in raw:
        variants.add(f"{raw}/USDT")
    return variants


def _open_action_for_side(side: Any) -> str:
    side_key = str(side or "").lower()
    if side_key == "short":
        return "FUT_SHORT_FULL"
    return "FUT_LONG_FULL"


def _agent_sync_positions(positions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [pos for pos in positions if not is_external_position(pos)]


def prepare_v2_agents_for_live_after_warmup(
    pipeline: Any,
    *,
    exchange_name: str,
    bar_index: int = 0,
) -> Dict[str, int]:
    """Clear warmup-only positions and seed registry agents with real positions."""
    registry = getattr(pipeline, "registry", None)
    agents = list(registry.all_agents()) if callable(getattr(registry, "all_agents", None)) else []
    real_positions = real_positions_for_agent_sync(pipeline)
    agent_positions = _agent_sync_positions(real_positions)

    for agent in agents:
        reset_actor_for_live(agent, bar_index=bar_index, seen=set())
        for pos in agent_positions:
            inject_actor_position(
                agent,
                sym=pos["sym"],
                side=pos["side"],
                entry_price=pos["entry_price"],
                qty=pos.get("qty", 0.0),
                bar_index=int(pos.get("opened_bar", 0) or bar_index),
                seen=set(),
            )

    summary = {
        "agents": len(agents),
        "real_positions": len(real_positions),
        "agent_positions": len(agent_positions),
        "external_positions_skipped": len(real_positions) - len(agent_positions),
        "bar_index": int(bar_index or 0),
    }
    _set_live_state_sync(pipeline, summary)
    log.info(
        "[%s] v2-agent live state synchronized: agents=%d real_positions=%d agent_positions=%d external_skipped=%d bar=%d",
        exchange_name,
        summary["agents"],
        summary["real_positions"],
        summary["agent_positions"],
        summary["external_positions_skipped"],
        summary["bar_index"],
    )
    return summary


def sync_player_agents_to_real_positions(
    player: Any,
    pipeline: Any,
    *,
    bar_index: int,
    market_symbols: Iterable[str],
) -> Dict[str, int]:
    """Reconcile selected live player agents with current tracker positions."""
    agents = _player_strategy_actors(player)
    real_positions = tracker_positions_for_agent_sync(pipeline)
    agent_positions = _agent_sync_positions(real_positions)
    agent_by_sym = {pos["sym"]: pos for pos in agent_positions}
    market_set = {str(sym).upper() for sym in market_symbols}
    stale_symbols = sorted(sym for sym in market_set if sym not in agent_by_sym)

    cleared = 0
    injected = 0
    for agent in agents:
        for sym in stale_symbols:
            cleared += clear_actor_position_symbols(agent, [sym], seen=set())
        for pos in agent_positions:
            injected += inject_actor_position(
                agent,
                sym=pos["sym"],
                side=pos["side"],
                entry_price=pos["entry_price"],
                qty=pos.get("qty", 0.0),
                bar_index=int(pos.get("opened_bar", 0) or bar_index),
                seen=set(),
            )

    summary = {
        "agents": len(agents),
        "real_positions": len(real_positions),
        "agent_positions": len(agent_positions),
        "external_positions_skipped": len(real_positions) - len(agent_positions),
        "bar_index": int(bar_index or 0),
        "cleared_symbols": len(stale_symbols),
        "cleared_fields": cleared,
        "injected_fields": injected,
    }
    _set_live_state_sync(pipeline, summary)
    return summary


def filter_real_signals_against_tracker(
    signals: Sequence[Signal],
    *,
    player: Any,
    pipeline: Any,
    bar_index: int,
    max_new_opens_per_bar: Optional[int] = None,
    max_open_positions: Optional[int] = None,
    reserved_new_opens: int = 0,
    manage_only: bool = False,
) -> RealSignalGuardResult:
    """Drop real signals that are impossible according to PositionTracker."""
    tracker_positions = {
        pos["sym"]: pos
        for pos in tracker_positions_for_agent_sync(pipeline)
    }
    agents = _player_strategy_actors(player)

    kept: List[Signal] = []
    stale_closes = 0
    foreign_closes = 0
    duplicate_opens = 0
    rate_limited_opens = 0
    max_position_saturated_opens = 0
    external_position_signals = 0
    reserved_new_opens = max(0, int(reserved_new_opens or 0))
    kept_new_opens = 0
    details: List[str] = []
    owned_open_count = sum(
        1 for pos in tracker_positions.values()
        if not is_external_position(pos)
    )
    planned_open_symbols = {
        str(sym).upper()
        for sym, pos in tracker_positions.items()
        if not is_external_position(pos)
    }

    for signal in signals:
        sym = str(signal.sym).upper()
        tracked = tracker_positions.get(sym)
        if manage_only and signal.action.is_open:
            rate_limited_opens += 1
            details.append(f"manage_only_open:{sym}:{signal.by_agent or '-'}")
            continue
        if tracked is not None and is_external_position(tracked):
            external_position_signals += 1
            details.append(f"external_position_signal:{sym}:{signal.by_agent or '-'}")
            for agent in agents:
                clear_actor_position_symbols(agent, [sym], seen=set())
            continue
        if signal.action.is_close and tracked is None:
            stale_closes += 1
            details.append(f"stale_close:{sym}:{signal.by_agent or '-'}")
            for agent in agents:
                clear_actor_position_symbols(agent, [sym], seen=set())
            continue
        if (
            signal.action.is_close
            and tracked is not None
            and not close_signal_matches_position_owner(signal, tracked)
        ):
            foreign_closes += 1
            owner = str(tracked.get("by_player") or tracked.get("by_agent") or "-")
            details.append(f"foreign_close:{sym}:{signal.by_agent or '-'}:owner={owner}")
            continue
        if signal.action.is_open and tracked is not None:
            tracked_side = str(tracked.get("side") or "").lower()
            if signal.action.side and tracked_side and signal.action.side != tracked_side:
                close_signal = replace(signal, action=Action.FUT_CLOSE_ALL)
                if not close_signal_matches_position_owner(close_signal, tracked):
                    foreign_closes += 1
                    owner = str(tracked.get("by_player") or tracked.get("by_agent") or "-")
                    details.append(
                        f"foreign_reverse_close:{sym}:{signal.by_agent or '-'}:owner={owner}"
                    )
                    continue
                kept.append(close_signal)
                details.append(f"reverse_close:{sym}:{signal.by_agent or '-'}")
                for agent in agents:
                    inject_actor_position(
                        agent,
                        sym=tracked["sym"],
                        side=tracked["side"],
                        entry_price=tracked["entry_price"],
                        qty=tracked.get("qty", 0.0),
                        bar_index=int(tracked.get("opened_bar", 0) or bar_index),
                        seen=set(),
                    )
                continue
            duplicate_opens += 1
            details.append(f"duplicate_open:{sym}:{signal.by_agent or '-'}")
            for agent in agents:
                inject_actor_position(
                    agent,
                    sym=tracked["sym"],
                    side=tracked["side"],
                    entry_price=tracked["entry_price"],
                    qty=tracked.get("qty", 0.0),
                    bar_index=int(tracked.get("opened_bar", 0) or bar_index),
                    seen=set(),
                )
            continue
        if signal.action.is_open and sym in planned_open_symbols:
            duplicate_opens += 1
            details.append(f"duplicate_open:{sym}:{signal.by_agent or '-'}")
            continue
        if (
            signal.action.is_open
            and max_open_positions is not None
            and owned_open_count + kept_new_opens >= max(0, int(max_open_positions))
        ):
            max_position_saturated_opens += 1
            details.append(f"max_position_saturated:{sym}:{signal.by_agent or '-'}")
            continue
        if (
            signal.action.is_open
            and max_new_opens_per_bar is not None
            and reserved_new_opens + kept_new_opens
            >= max(0, int(max_new_opens_per_bar))
        ):
            rate_limited_opens += 1
            details.append(f"rate_limited_open:{sym}:{signal.by_agent or '-'}")
            continue
        kept.append(signal)
        if signal.action.is_open:
            kept_new_opens += 1
            planned_open_symbols.add(sym)

    filtered = (
        stale_closes
        + foreign_closes
        + duplicate_opens
        + rate_limited_opens
        + max_position_saturated_opens
        + external_position_signals
    )
    if filtered:
        log.info(
            "real signal guard filtered=%d stale_closes=%d foreign_closes=%d duplicate_opens=%d rate_limited_opens=%d max_position_saturated_opens=%d external_position_signals=%d details=%s",
            filtered,
            stale_closes,
            foreign_closes,
            duplicate_opens,
            rate_limited_opens,
            max_position_saturated_opens,
            external_position_signals,
            ", ".join(details[:8]),
        )
    return RealSignalGuardResult(
        signals=kept,
        filtered=filtered,
        stale_closes=stale_closes,
        foreign_closes=foreign_closes,
        duplicate_opens=duplicate_opens,
        rate_limited_opens=rate_limited_opens,
        max_position_saturated_opens=max_position_saturated_opens,
        external_position_signals=external_position_signals,
        details=details,
    )


def _player_strategy_actors(player: Any) -> List[Any]:
    """Return mutable strategy implementations owned by a player."""
    actors = list(getattr(player, "agents", []) or [])
    strategy = getattr(player, "strategy", None)
    if strategy is not None and strategy not in actors:
        actors.append(strategy)
    return actors


def _exchange_position_snapshot_unreliable_reason(pipeline: Any, exchange: Any) -> str:
    reliable_probe = getattr(exchange, "positions_snapshot_reliable", None)
    if callable(reliable_probe):
        try:
            if not bool(reliable_probe()):
                error_probe = getattr(exchange, "positions_snapshot_error", None)
                if callable(error_probe):
                    reason = str(error_probe() or "").strip()
                    if reason:
                        return reason
                return "unreliable exchange position snapshot"
        except Exception:
            return "unreliable exchange position snapshot"

    account_snapshot = getattr(pipeline, "account_snapshot", None)
    if not isinstance(account_snapshot, dict):
        return ""
    health = account_snapshot.get("data_health")
    if not isinstance(health, dict):
        return ""
    reason = str(
        health.get("recent_data_error")
        or health.get("last_data_error_reason")
        or health.get("last_error_reason")
        or health.get("reason")
        or ""
    ).strip()
    if reason:
        return reason
    if bool(health.get("uses_cached_balance") or health.get("cached_equity")):
        return "cached exchange snapshot"
    if health.get("snapshot_healthy") is False:
        return "snapshot unhealthy"
    return ""


def reconcile_tracker_with_exchange(
    pipeline: Any,
    *,
    bar_index: int,
) -> Dict[str, Any]:
    """Conservatively add/update tracker positions from the live exchange."""
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    getter = getattr(exchange, "get_all_positions", None)
    if tracker is None or not callable(getter):
        return {"added": 0, "updated": 0, "removed": 0, "seen": 0, "bar_index": int(bar_index or 0)}

    try:
        exchange_positions = getter() or {}
    except Exception:
        log.debug("exchange position reconcile read failed", exc_info=True)
        return {"added": 0, "updated": 0, "removed": 0, "seen": 0, "bar_index": int(bar_index or 0)}

    unreliable_reason = ""
    if not exchange_positions:
        unreliable_reason = _exchange_position_snapshot_unreliable_reason(pipeline, exchange)

    added = 0
    updated = 0
    removed = 0
    owned_added = 0
    owned_updated = 0
    owned_removed = 0
    external_added = 0
    external_updated = 0
    external_removed = 0
    adopted_added = 0
    adopted_updated = 0
    adopted_removed = 0
    seen = 0
    exchange_seen_symbols: Set[str] = set()
    for key, raw_pos in exchange_positions.items():
        normalized = normalize_position_for_sync(key, raw_pos)
        if not normalized or normalized["qty"] <= 0 or normalized["entry_price"] <= 0:
            continue
        seen += 1
        exchange_seen_symbols.add(normalized["sym"])
        existing = tracker.get(normalized["sym"])
        adopt_existing = (
            (existing is None or is_external_position(existing))
            and should_adopt_existing_position(pipeline, normalized["sym"])
        )
        adopted_player, adopted_agent = adopted_position_labels(pipeline)
        by_player = (
            adopted_player
            if adopt_existing else (
                str(getattr(existing, "by_player", "") or "")
                if existing is not None else "RecoveredExchangePosition"
            )
        ) or "RecoveredExchangePosition"
        by_agent = (
            adopted_agent
            if adopt_existing else str(getattr(existing, "by_agent", "") or "")
        )
        open_action = (
            _open_action_for_side(normalized["side"])
            if adopt_existing else str(getattr(existing, "open_action", "") or "")
        )
        open_regime = (
            "neutral"
            if adopt_existing else str(getattr(existing, "open_regime", "") or "")
        )
        replacement = TrackedPosition(
            open_signal_id=(
                int(getattr(existing, "open_signal_id", 0) or 0)
                if existing is not None else 0
            ),
            sym=normalized["sym"],
            side=normalized["side"],
            entry_price=normalized["entry_price"],
            qty=normalized["qty"],
            fee_open=float(getattr(existing, "fee_open", 0.0) or 0.0),
            by_player=by_player,
            by_agent=by_agent,
            opened_at=(
                getattr(existing, "opened_at", None)
                if existing is not None else datetime.now(timezone.utc)
            ) or datetime.now(timezone.utc),
            opened_bar=(
                int(getattr(existing, "opened_bar", 0) or 0)
                if existing is not None else int(bar_index or 0)
            ),
            open_action=open_action,
            open_regime=open_regime,
            funding_open=float(getattr(existing, "funding_open", 0.0) or 0.0),
        )
        if existing is None:
            added += 1
            tracker.force_set(replacement)
            if is_external_position(replacement):
                external_added += 1
            elif is_adopted_position(replacement, pipeline):
                owned_added += 1
                adopted_added += 1
                seed_perf_open_position_for_adoption(pipeline, replacement)
            else:
                owned_added += 1
            continue
        changed = (
            existing.side != replacement.side
            or abs(existing.qty - replacement.qty) > 1e-12
            or abs(existing.entry_price - replacement.entry_price) > 1e-12
            or str(existing.by_player or "") != str(replacement.by_player or "")
            or str(existing.by_agent or "") != str(replacement.by_agent or "")
            or str(existing.open_action or "") != str(replacement.open_action or "")
            or str(existing.open_regime or "") != str(replacement.open_regime or "")
        )
        if changed:
            updated += 1
            tracker.force_set(replacement)
            if is_external_position(replacement):
                external_updated += 1
            elif is_adopted_position(replacement, pipeline):
                owned_updated += 1
                adopted_updated += 1
                seed_perf_open_position_for_adoption(pipeline, replacement)
            else:
                owned_updated += 1

    open_positions = list((tracker.all_open() or {}).items())
    removal_guarded = 0
    if seen == 0 and unreliable_reason:
        removal_guarded = len(open_positions)
    else:
        for sym, existing in open_positions:
            normalized_sym = str(getattr(existing, "sym", sym) or sym).upper()
            if normalized_sym in exchange_seen_symbols:
                continue
            removed_pos = tracker.force_remove(normalized_sym)
            if removed_pos is None and normalized_sym != sym:
                removed_pos = tracker.force_remove(sym)
            if removed_pos is None:
                continue
            removed += 1
            if is_external_position(removed_pos):
                external_removed += 1
            elif is_adopted_position(removed_pos, pipeline):
                owned_removed += 1
                adopted_removed += 1
            else:
                owned_removed += 1
            _emit_external_close_event(pipeline, removed_pos, bar_index=bar_index)

    summary = {
        "added": added,
        "updated": updated,
        "removed": removed,
        "owned_added": owned_added,
        "owned_updated": owned_updated,
        "owned_removed": owned_removed,
        "external_added": external_added,
        "external_updated": external_updated,
        "external_removed": external_removed,
        "adopted_added": adopted_added,
        "adopted_updated": adopted_updated,
        "adopted_removed": adopted_removed,
        "removal_guarded": removal_guarded,
        "snapshot_unreliable": bool(unreliable_reason),
        "guard_reason": unreliable_reason,
        "seen": seen,
        "bar_index": int(bar_index or 0),
    }
    try:
        previous = getattr(pipeline, "live_state_sync", None)
        merged = dict(previous) if isinstance(previous, dict) else {}
        merged["exchange_reconcile"] = summary
        setattr(pipeline, "live_state_sync", merged)
    except Exception:
        pass
    return summary


def _emit_external_close_event(pipeline: Any, pos: TrackedPosition, *, bar_index: int) -> None:
    event_log = getattr(pipeline, "event_log", None)
    emit = getattr(event_log, "emit", None)
    if not callable(emit):
        return
    trace_id = f"exchange-reconcile-{bar_index}-{pos.sym}"
    try:
        emit(PositionClosed(
            bar=int(bar_index or 0),
            trace_id=trace_id,
            open_signal_id=int(pos.open_signal_id or 0),
            close_signal_id=0,
            sym=pos.sym,
            side=pos.side,
            entry=pos.entry_price,
            exit=pos.entry_price,
            qty=pos.qty,
            realized_pnl=0.0,
            by_player=pos.by_player,
            by_agent=pos.by_agent,
            decision_id=trace_id,
            exchange=str(getattr(pipeline, "exchange_name", "") or ""),
            symbol=str(pos.sym or ""),
            timeframe=str(getattr(pipeline, "timeframe", "") or ""),
            mode=str(getattr(pipeline, "mode", "") or ""),
            run_id=str(getattr(pipeline, "run_id", "") or ""),
            session_id=str(getattr(pipeline, "session_id", "") or ""),
        ))
    except Exception:
        log.debug("external close event emit failed", exc_info=True)


def tracker_positions_for_agent_sync(pipeline: Any) -> List[Dict[str, Any]]:
    positions: Dict[str, Dict[str, Any]] = {}
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is not None and hasattr(tracker, "all_open"):
        try:
            for key, pos in (tracker.all_open() or {}).items():
                normalized = normalize_position_for_sync(
                    key,
                    pos,
                    default_by_player="UnknownTrackerPosition",
                )
                if normalized:
                    positions[normalized["sym"]] = normalized
        except Exception:
            log.debug("tracker position sync read failed", exc_info=True)
    return list(positions.values())


def real_positions_for_agent_sync(pipeline: Any) -> List[Dict[str, Any]]:
    """Return open real positions from tracker, with exchange as startup fallback."""
    positions: Dict[str, Dict[str, Any]] = {
        pos["sym"]: pos for pos in tracker_positions_for_agent_sync(pipeline)
    }

    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    getter = getattr(exchange, "get_all_positions", None)
    if callable(getter):
        try:
            for key, pos in (getter() or {}).items():
                normalized = normalize_position_for_sync(key, pos)
                if normalized and normalized["sym"] not in positions:
                    positions[normalized["sym"]] = normalized
        except Exception:
            log.debug("exchange position sync read failed", exc_info=True)

    return list(positions.values())


def normalize_position_for_sync(
    key: Any,
    pos: Any,
    *,
    default_by_player: str = "",
) -> Optional[Dict[str, Any]]:
    sym = str(getattr(pos, "sym", key) or "").upper()
    side = normalize_side(getattr(pos, "side", ""))
    entry = getattr(pos, "entry_price", None)
    if entry is None:
        entry = getattr(pos, "entry", None)
    if not sym or side not in ("long", "short"):
        return None
    return {
        "sym": sym,
        "side": side,
        "entry_price": float_or_zero(entry),
        "qty": float_or_zero(getattr(pos, "qty", 0.0)),
        "opened_bar": int(float_or_zero(getattr(pos, "opened_bar", 0))),
        "by_player": str(getattr(pos, "by_player", default_by_player) or default_by_player),
        "by_agent": str(getattr(pos, "by_agent", "") or ""),
    }


def normalize_side(side: Any) -> str:
    raw = str(side or "").strip().lower()
    if "short" in raw or raw in {"sell", "-1"}:
        return "short"
    if "long" in raw or raw in {"buy", "1"}:
        return "long"
    return raw


def reset_actor_for_live(actor: Any, *, bar_index: int, seen: Set[int], depth: int = 0) -> None:
    if actor is None or depth > 6:
        return
    ident = id(actor)
    if ident in seen:
        return
    seen.add(ident)

    reset = getattr(actor, "reset_for_live", None)
    if callable(reset):
        try:
            reset(bar_index)
        except TypeError:
            try:
                reset()
            except Exception:
                log.debug("agent reset_for_live failed", exc_info=True)
        except Exception:
            log.debug("agent reset_for_live failed", exc_info=True)
    else:
        clear_actor_position_maps(actor)
        reset_actor_check_timer_for_live(actor, bar_index=bar_index)

    for child in child_actors(actor):
        reset_actor_for_live(child, bar_index=bar_index, seen=seen, depth=depth + 1)


def reset_actor_check_timer_for_live(actor: Any, *, bar_index: int) -> bool:
    if not hasattr(actor, "_lc"):
        return False
    try:
        check_int = int(float(getattr(actor, "CHECK_INT", 1) or 1))
    except (TypeError, ValueError):
        check_int = 1
    check_int = max(1, check_int)
    try:
        setattr(actor, "_lc", int(bar_index or 0) - check_int)
        return True
    except Exception:
        log.debug("agent check timer reset failed", exc_info=True)
        return False


def clear_actor_position_maps(actor: Any) -> int:
    return clear_actor_position_symbols(actor, None, seen=set())


def clear_actor_position_symbols(
    actor: Any,
    symbols: Optional[Iterable[str]],
    *,
    seen: Set[int],
    depth: int = 0,
) -> int:
    if actor is None or depth > 6:
        return 0
    ident = id(actor)
    if ident in seen:
        return 0
    seen.add(ident)

    symbols_set = None if symbols is None else {str(sym).upper() for sym in symbols}
    changed = False

    for attr in ("pos", "_pos", "position"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            keys = list(mapping.keys()) if symbols_set is None else list(symbols_set)
            for sym in keys:
                if sym in mapping and mapping.get(sym) is not None:
                    mapping[sym] = None
                    changed = True
    for attr in ("ep", "entry_px", "entry_price"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            keys = list(mapping.keys()) if symbols_set is None else list(symbols_set)
            for sym in keys:
                if sym in mapping and float_or_zero(mapping.get(sym)) != 0.0:
                    mapping[sym] = 0.0
                    changed = True
    for attr in ("et", "entry_t", "entry_bar"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            keys = list(mapping.keys()) if symbols_set is None else list(symbols_set)
            for sym in keys:
                if sym in mapping and int(float_or_zero(mapping.get(sym))) != 0:
                    mapping[sym] = 0
                    changed = True
    for attr in ("spot_qty", "spot_entry", "fut_qty", "fut_entry"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            keys = list(mapping.keys()) if symbols_set is None else list(symbols_set)
            for sym in keys:
                if sym in mapping and float_or_zero(mapping.get(sym)) != 0.0:
                    mapping[sym] = 0.0
                    changed = True

    total = 1 if changed else 0
    for child in child_actors(actor):
        total += clear_actor_position_symbols(
            child,
            symbols_set,
            seen=seen,
            depth=depth + 1,
        )
    return total


def inject_actor_position(
    actor: Any,
    *,
    sym: str,
    side: str,
    entry_price: float,
    qty: float = 0.0,
    bar_index: int,
    seen: Set[int],
    depth: int = 0,
) -> int:
    if actor is None or depth > 6:
        return 0
    ident = id(actor)
    if ident in seen:
        return 0
    seen.add(ident)

    changed = False
    sync = getattr(actor, "sync_position", None)
    if callable(sync):
        try:
            sync(sym, side, entry_price, bar_index)
            changed = True
        except TypeError:
            try:
                sync(sym, side, entry_price)
                changed = True
            except Exception:
                log.debug("agent sync_position failed", exc_info=True)
        except Exception:
            log.debug("agent sync_position failed", exc_info=True)

    update = getattr(actor, "update_from_exchange", None)
    if callable(update):
        qty_abs = abs(float_or_zero(qty)) or 1.0
        fut_qty = qty_abs if side == "long" else -qty_abs
        try:
            update(sym, 0.0, 0.0, fut_qty, entry_price)
            changed = True
        except Exception:
            log.debug("agent update_from_exchange sync failed", exc_info=True)

    for attr in ("pos", "_pos", "position"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            mapping[sym] = side
            changed = True
    for attr in ("ep", "entry_px", "entry_price"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            mapping[sym] = entry_price
            changed = True
    for attr in ("et", "entry_t", "entry_bar"):
        mapping = getattr(actor, attr, None)
        if isinstance(mapping, dict):
            mapping[sym] = int(bar_index or 0)
            changed = True

    total = 1 if changed else 0
    for child in child_actors(actor):
        total += inject_actor_position(
            child,
            sym=sym,
            side=side,
            entry_price=entry_price,
            qty=qty,
            bar_index=bar_index,
            seen=seen,
            depth=depth + 1,
        )
    return total


def child_actors(actor: Any) -> List[Any]:
    children: List[Any] = []
    for attr in ("v1_agent", "_inner", "_a", "_b", "_b1", "_b2", "_agent", "agent"):
        child = getattr(actor, attr, None)
        if child is not None and child is not actor:
            children.append(child)
    for attr in ("agents", "_agents", "sub_agents", "_sub_agents"):
        raw = getattr(actor, attr, None)
        if isinstance(raw, dict):
            children.extend(v for v in raw.values() if v is not None and v is not actor)
        elif isinstance(raw, (list, tuple, set)):
            children.extend(v for v in raw if v is not None and v is not actor)
    return children


def float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _set_live_state_sync(pipeline: Any, summary: Dict[str, int]) -> None:
    try:
        previous = getattr(pipeline, "live_state_sync", None)
        if isinstance(previous, dict):
            merged = dict(previous)
            merged.update(summary)
            setattr(pipeline, "live_state_sync", merged)
        else:
            setattr(pipeline, "live_state_sync", summary)
    except Exception:
        pass
