"""Live-state synchronization for the real Panteon execution path.

The live exchange/position tracker is the single source of truth for real
orders. V1 agents may keep their own ``pos``/``ep``/``et`` maps after warmup or
after a restart, so this module reconciles those maps before real voting and
filters impossible real signals before they reach EventLog/TradeExecutor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ..domain.types import Signal


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealSignalGuardResult:
    signals: List[Signal]
    filtered: int = 0
    stale_closes: int = 0
    duplicate_opens: int = 0
    details: List[str] = field(default_factory=list)


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

    for agent in agents:
        reset_actor_for_live(agent, bar_index=bar_index, seen=set())
        for pos in real_positions:
            inject_actor_position(
                agent,
                sym=pos["sym"],
                side=pos["side"],
                entry_price=pos["entry_price"],
                bar_index=bar_index,
                seen=set(),
            )

    summary = {
        "agents": len(agents),
        "real_positions": len(real_positions),
        "bar_index": int(bar_index or 0),
    }
    _set_live_state_sync(pipeline, summary)
    log.info(
        "[%s] v2-agent live state synchronized: agents=%d real_positions=%d bar=%d",
        exchange_name,
        summary["agents"],
        summary["real_positions"],
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
    agents = list(getattr(player, "agents", []) or [])
    real_positions = tracker_positions_for_agent_sync(pipeline)
    real_by_sym = {pos["sym"]: pos for pos in real_positions}
    market_set = {str(sym).upper() for sym in market_symbols}
    stale_symbols = sorted(sym for sym in market_set if sym not in real_by_sym)

    cleared = 0
    injected = 0
    for agent in agents:
        for sym in stale_symbols:
            cleared += clear_actor_position_symbols(agent, [sym], seen=set())
        for pos in real_positions:
            injected += inject_actor_position(
                agent,
                sym=pos["sym"],
                side=pos["side"],
                entry_price=pos["entry_price"],
                bar_index=bar_index,
                seen=set(),
            )

    summary = {
        "agents": len(agents),
        "real_positions": len(real_positions),
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
) -> RealSignalGuardResult:
    """Drop real signals that are impossible according to PositionTracker."""
    tracker_positions = {
        pos["sym"]: pos
        for pos in tracker_positions_for_agent_sync(pipeline)
    }
    agents = list(getattr(player, "agents", []) or [])

    kept: List[Signal] = []
    stale_closes = 0
    duplicate_opens = 0
    details: List[str] = []

    for signal in signals:
        sym = str(signal.sym).upper()
        tracked = tracker_positions.get(sym)
        if signal.action.is_close and tracked is None:
            stale_closes += 1
            details.append(f"stale_close:{sym}:{signal.by_agent or '-'}")
            for agent in agents:
                clear_actor_position_symbols(agent, [sym], seen=set())
            continue
        if signal.action.is_open and tracked is not None:
            duplicate_opens += 1
            details.append(f"duplicate_open:{sym}:{signal.by_agent or '-'}")
            for agent in agents:
                inject_actor_position(
                    agent,
                    sym=tracked["sym"],
                    side=tracked["side"],
                    entry_price=tracked["entry_price"],
                    bar_index=bar_index,
                    seen=set(),
                )
            continue
        kept.append(signal)

    filtered = stale_closes + duplicate_opens
    if filtered:
        log.info(
            "real signal guard filtered=%d stale_closes=%d duplicate_opens=%d details=%s",
            filtered,
            stale_closes,
            duplicate_opens,
            ", ".join(details[:8]),
        )
    return RealSignalGuardResult(
        signals=kept,
        filtered=filtered,
        stale_closes=stale_closes,
        duplicate_opens=duplicate_opens,
        details=details,
    )


def tracker_positions_for_agent_sync(pipeline: Any) -> List[Dict[str, Any]]:
    positions: Dict[str, Dict[str, Any]] = {}
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if tracker is not None and hasattr(tracker, "all_open"):
        try:
            for key, pos in (tracker.all_open() or {}).items():
                normalized = normalize_position_for_sync(key, pos)
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


def normalize_position_for_sync(key: Any, pos: Any) -> Optional[Dict[str, Any]]:
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

    for child in child_actors(actor):
        reset_actor_for_live(child, bar_index=bar_index, seen=seen, depth=depth + 1)


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
