"""Flash degradation state tracking helpers.

This module owns the bookkeeping that turns closed-position outcomes into
temporary deny sets for signal keys, actors, and open symbols. Keeping it out
of ``main_loop.py`` makes the Flash decision path easier to audit: the main
loop decides when to refresh degradation state, while this module decides how
the rolling windows and cooldowns are updated.
"""

from __future__ import annotations

from collections import deque

from ..attribution import PositionClosed


def flash_update_degradation_state_from_events(
    pipeline: object,
    *,
    current_bar: int = 0,
) -> None:
    if not _flash_degradation_guard_enabled(pipeline):
        return

    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    window = max(1, int(getattr(config, "degradation_window_closed_trades", 3) or 3))
    min_closed = max(1, int(getattr(config, "degradation_min_closed_trades", 3) or 3))
    threshold = float(getattr(config, "degradation_max_recent_pnl_usd", -25.0) or 0.0)
    recovery_enabled = bool(getattr(config, "degradation_recovery_enabled", False))
    recovery_min_closed = max(
        1,
        int(getattr(config, "degradation_recovery_min_closed_trades", min_closed) or min_closed),
    )
    recovery_threshold = float(
        getattr(config, "degradation_recovery_min_recent_pnl_usd", 0.0) or 0.0
    )

    cursor = int(getattr(pipeline, "_flash_degradation_event_cursor", 0) or 0)
    events, event_count = flash_event_log_tail(pipeline.event_log, cursor)
    signal_key_by_id = dict(getattr(pipeline, "_flash_signal_key_by_id", {}) or {})
    actor_key_by_id = dict(getattr(pipeline, "_flash_actor_key_by_id", {}) or {})
    outcomes = dict(getattr(pipeline, "_flash_degradation_outcomes", {}) or {})
    degraded = set(getattr(pipeline, "_flash_degraded_signal_keys", set()) or set())
    signal_cooldown_bars = max(
        0,
        int(getattr(config, "degradation_signal_cooldown_bars", 0) or 0),
    )
    signal_degraded_until = {
        str(signal_key): int(until)
        for signal_key, until in (
            getattr(pipeline, "_flash_degraded_signal_until", {}) or {}
        ).items()
        if str(signal_key or "") and int(until or 0) > 0
    }
    actor_guard_enabled = bool(
        getattr(config, "degradation_actor_guard_enabled", False)
    )
    actor_cooldown_bars = max(
        0,
        int(getattr(config, "degradation_actor_cooldown_bars", 0) or 0),
    )
    now_bar = max(0, int(current_bar or 0))
    actor_outcomes = dict(
        getattr(pipeline, "_flash_degradation_actor_outcomes", {}) or {}
    )
    degraded_actors = set(
        getattr(pipeline, "_flash_degraded_actor_keys", set()) or set()
    )
    actor_degraded_until = {
        str(actor): int(until)
        for actor, until in (
            getattr(pipeline, "_flash_degraded_actor_until", {}) or {}
        ).items()
        if str(actor or "") and int(until or 0) > 0
    }
    symbol_guard_enabled = bool(
        getattr(config, "degradation_symbol_guard_enabled", False)
    )
    symbol_cooldown_bars = max(
        0,
        int(getattr(config, "degradation_symbol_cooldown_bars", 0) or 0),
    )
    symbol_lookback_bars = max(
        0,
        int(getattr(config, "degradation_symbol_lookback_bars", 0) or 0),
    )
    symbol_window = max(
        1,
        int(getattr(config, "degradation_symbol_window_closed_trades", 0) or window),
    )
    symbol_min_closed = max(
        1,
        int(getattr(config, "degradation_symbol_min_closed_trades", 0) or min_closed),
    )
    symbol_threshold_raw = getattr(
        config,
        "degradation_symbol_max_recent_pnl_usd",
        None,
    )
    symbol_threshold = (
        threshold
        if symbol_threshold_raw is None
        else float(symbol_threshold_raw)
    )
    symbol_outcomes = dict(
        getattr(pipeline, "_flash_degradation_symbol_outcomes", {}) or {}
    )
    symbol_time_outcomes = dict(
        getattr(pipeline, "_flash_degradation_symbol_time_outcomes", {}) or {}
    )
    degraded_symbols = {
        str(symbol).strip().upper()
        for symbol in (getattr(pipeline, "_flash_degraded_open_symbols", set()) or set())
        if str(symbol or "").strip()
    }
    symbol_degraded_until = {
        str(symbol).strip().upper(): int(until)
        for symbol, until in (
            getattr(pipeline, "_flash_degraded_open_symbol_until", {}) or {}
        ).items()
        if str(symbol or "").strip() and int(until or 0) > 0
    }

    _expire_degraded_items(degraded, signal_degraded_until, signal_cooldown_bars, now_bar)
    _expire_degraded_items(degraded_actors, actor_degraded_until, actor_cooldown_bars, now_bar)
    _expire_degraded_items(degraded_symbols, symbol_degraded_until, symbol_cooldown_bars, now_bar)

    for event in events:
        if not isinstance(event, PositionClosed):
            continue
        signal_id = int(getattr(event, "open_signal_id", -1) or -1)
        realized_pnl = float(getattr(event, "realized_pnl", 0.0) or 0.0)
        if symbol_guard_enabled:
            symbol = str(getattr(event, "sym", "") or "").strip().upper()
            if symbol:
                if symbol_lookback_bars > 0:
                    symbol_bucket = _time_bucket(
                        symbol_time_outcomes.get(symbol),
                        current_bar=now_bar,
                        lookback_bars=symbol_lookback_bars,
                    )
                    symbol_bucket.append((
                        max(0, int(getattr(event, "bar", 0) or now_bar or 0)),
                        realized_pnl,
                    ))
                    _prune_time_bucket(
                        symbol_bucket,
                        current_bar=now_bar,
                        lookback_bars=symbol_lookback_bars,
                    )
                    symbol_time_outcomes[symbol] = symbol_bucket
                    symbol_recent_pnl = sum(float(value) for _, value in symbol_bucket)
                    symbol_recent_closed = len(symbol_bucket)
                    symbol_recovery_pnl = symbol_recent_pnl
                else:
                    symbol_bucket = _rolling_bucket(
                        symbol_outcomes.get(symbol),
                        maxlen=symbol_window,
                    )
                    symbol_bucket.append(realized_pnl)
                    symbol_outcomes[symbol] = symbol_bucket
                    symbol_recent_pnl = sum(float(value) for value in symbol_bucket)
                    symbol_recent_closed = len(symbol_bucket)
                    symbol_recovery_pnl = flash_degradation_recent_pnl(
                        symbol_bucket,
                        recovery_min_closed,
                    )
                if (
                    symbol_recent_closed >= symbol_min_closed
                    and symbol_recent_pnl <= symbol_threshold
                ):
                    degraded_symbols.add(symbol)
                    _extend_cooldown(
                        symbol_degraded_until,
                        symbol,
                        event=event,
                        now_bar=now_bar,
                        cooldown_bars=symbol_cooldown_bars,
                    )
                elif (
                    recovery_enabled
                    and symbol in degraded_symbols
                    and symbol_recent_closed >= recovery_min_closed
                    and symbol_recovery_pnl >= recovery_threshold
                ):
                    degraded_symbols.discard(symbol)
                    symbol_degraded_until.pop(symbol, None)
        key = signal_key_by_id.get(signal_id)
        if key:
            bucket = _rolling_bucket(outcomes.get(key), maxlen=window)
            bucket.append(realized_pnl)
            outcomes[key] = bucket
            if len(bucket) >= min_closed and sum(float(value) for value in bucket) <= threshold:
                degraded.add(key)
                _extend_cooldown(
                    signal_degraded_until,
                    key,
                    event=event,
                    now_bar=now_bar,
                    cooldown_bars=signal_cooldown_bars,
                )
            elif (
                recovery_enabled
                and key in degraded
                and flash_degradation_recent_pnl(bucket, recovery_min_closed)
                >= recovery_threshold
            ):
                degraded.discard(key)
                signal_degraded_until.pop(key, None)
        if actor_guard_enabled:
            actor_key = actor_key_by_id.get(signal_id)
            if actor_key:
                actor_bucket = _rolling_bucket(
                    actor_outcomes.get(actor_key),
                    maxlen=window,
                )
                actor_bucket.append(realized_pnl)
                actor_outcomes[actor_key] = actor_bucket
                if (
                    len(actor_bucket) >= min_closed
                    and sum(float(value) for value in actor_bucket) <= threshold
                ):
                    degraded_actors.add(actor_key)
                    _extend_cooldown(
                        actor_degraded_until,
                        actor_key,
                        event=event,
                        now_bar=now_bar,
                        cooldown_bars=actor_cooldown_bars,
                    )
                elif (
                    recovery_enabled
                    and actor_key in degraded_actors
                    and flash_degradation_recent_pnl(
                        actor_bucket,
                        recovery_min_closed,
                    )
                    >= recovery_threshold
                ):
                    degraded_actors.discard(actor_key)
                    actor_degraded_until.pop(actor_key, None)

    _expire_degraded_items(degraded, signal_degraded_until, signal_cooldown_bars, now_bar)
    _expire_degraded_items(degraded_actors, actor_degraded_until, actor_cooldown_bars, now_bar)
    _expire_degraded_items(degraded_symbols, symbol_degraded_until, symbol_cooldown_bars, now_bar)

    pipeline._flash_degradation_event_cursor = event_count
    pipeline._flash_signal_key_by_id = signal_key_by_id
    pipeline._flash_actor_key_by_id = actor_key_by_id
    pipeline._flash_degradation_outcomes = outcomes
    pipeline._flash_degraded_signal_keys = degraded
    pipeline._flash_degraded_signal_until = signal_degraded_until
    pipeline._flash_degradation_actor_outcomes = actor_outcomes
    pipeline._flash_degraded_actor_keys = degraded_actors
    pipeline._flash_degraded_actor_until = actor_degraded_until
    pipeline._flash_degradation_symbol_outcomes = symbol_outcomes
    pipeline._flash_degradation_symbol_time_outcomes = symbol_time_outcomes
    pipeline._flash_degraded_open_symbols = degraded_symbols
    pipeline._flash_degraded_open_symbol_until = symbol_degraded_until


def flash_degradation_recent_pnl(bucket: object, closed_trades: int) -> float:
    values = [float(value) for value in list(bucket or ())]
    closed = max(1, int(closed_trades or 1))
    if len(values) < closed:
        return float("-inf")
    return sum(values[-closed:])


def flash_event_log_tail(event_log: object, cursor: int) -> tuple[list[object], int]:
    raw_events = getattr(event_log, "_events", None)
    if isinstance(raw_events, list):
        lock = getattr(event_log, "_lock", None)
        if lock is not None:
            with lock:
                event_count = len(raw_events)
                start = max(0, min(int(cursor or 0), event_count))
                return list(raw_events[start:]), event_count
        event_count = len(raw_events)
        start = max(0, min(int(cursor or 0), event_count))
        return list(raw_events[start:]), event_count

    all_events = list(getattr(event_log, "all")())
    event_count = len(all_events)
    start = max(0, min(int(cursor or 0), event_count))
    return all_events[start:], event_count


def _rolling_bucket(bucket: object, *, maxlen: int) -> deque[float]:
    if bucket is None or getattr(bucket, "maxlen", None) != maxlen:
        return deque(list(bucket or ())[-maxlen:], maxlen=maxlen)
    return bucket


def _time_bucket(
    bucket: object,
    *,
    current_bar: int,
    lookback_bars: int,
) -> deque[tuple[int, float]]:
    out: deque[tuple[int, float]] = deque()
    for raw_item in list(bucket or ()):
        try:
            bar, pnl = raw_item
        except (TypeError, ValueError):
            continue
        out.append((int(bar or 0), float(pnl or 0.0)))
    _prune_time_bucket(out, current_bar=current_bar, lookback_bars=lookback_bars)
    return out


def _prune_time_bucket(
    bucket: deque[tuple[int, float]],
    *,
    current_bar: int,
    lookback_bars: int,
) -> None:
    if lookback_bars <= 0 or current_bar <= 0:
        return
    min_bar = max(0, int(current_bar) - int(lookback_bars))
    while bucket and int(bucket[0][0]) < min_bar:
        bucket.popleft()


def _extend_cooldown(
    degraded_until: dict[str, int],
    key: str,
    *,
    event: PositionClosed,
    now_bar: int,
    cooldown_bars: int,
) -> None:
    if cooldown_bars <= 0:
        return
    event_bar = max(
        0,
        int(getattr(event, "bar", 0) or now_bar or 0),
    )
    until_bar = event_bar + cooldown_bars
    degraded_until[key] = max(int(degraded_until.get(key, 0) or 0), until_bar)


def _expire_degraded_items(
    degraded: set[str],
    degraded_until: dict[str, int],
    cooldown_bars: int,
    now_bar: int,
) -> None:
    if cooldown_bars <= 0 or now_bar <= 0:
        return
    expired = {
        key
        for key, until_bar in degraded_until.items()
        if until_bar <= now_bar
    }
    if not expired:
        return
    degraded.difference_update(expired)
    for key in expired:
        degraded_until.pop(key, None)


def _flash_degradation_guard_enabled(pipeline: object) -> bool:
    allocator = getattr(pipeline, "flash_allocator", None)
    config = getattr(allocator, "config", None)
    return bool(getattr(config, "degradation_guard_enabled", False))
