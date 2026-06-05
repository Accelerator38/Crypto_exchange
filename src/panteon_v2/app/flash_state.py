"""Small Flash state helpers kept outside the main loop orchestrator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from ..domain.types import Regime


def flash_regime_degradation_key(regime: Regime | str | int) -> str:
    if isinstance(regime, Regime):
        return regime.label
    if isinstance(regime, str):
        return Regime.from_string(regime).label
    try:
        return Regime(int(regime)).label
    except Exception:
        return Regime.from_string(str(regime or "")).label


def flash_open_position_sides_by_symbol(
    positions: Iterable[Mapping[str, object]],
    *,
    is_external: Callable[[Mapping[str, object]], bool],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for pos in positions:
        if is_external(pos):
            continue
        symbol = str(pos.get("sym") or "").upper()
        side = str(pos.get("side") or "").strip().lower()
        if symbol and side:
            out[symbol] = side
    return out


def flash_open_position_actor_keys_by_symbol(
    positions: Iterable[Mapping[str, object]],
    *,
    is_external: Callable[[Mapping[str, object]], bool],
) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for pos in positions:
        if is_external(pos):
            continue
        symbol = str(pos.get("sym") or "").upper()
        if not symbol:
            continue
        by_player = str(pos.get("by_player") or "").strip()
        by_agent = str(pos.get("by_agent") or "").strip()
        actor_keys: list[str] = []
        if by_player and by_player != "Panteon_Flash":
            actor_keys.append(f"ensemble:{by_player}")
            if not by_agent or by_player == by_agent:
                actor_keys.append(f"agent:{by_player}")
        if by_agent:
            actor_keys.append(f"agent:{by_agent}")
        clean_keys = tuple(dict.fromkeys(key for key in actor_keys if key))
        if clean_keys:
            out[symbol] = clean_keys
    return out


def flash_degraded_signal_keys(
    pipeline: object,
    *,
    guard_enabled: bool,
) -> set[str]:
    if not guard_enabled:
        return set()
    return set(getattr(pipeline, "_flash_degraded_signal_keys", set()) or set())


def flash_degraded_actor_keys(
    pipeline: object,
    *,
    guard_enabled: bool,
    actor_guard_enabled: bool,
    actor_scope: str,
    regime: Regime | None = None,
) -> set[str]:
    if not guard_enabled or not actor_guard_enabled:
        return set()
    degraded = set(getattr(pipeline, "_flash_degraded_actor_keys", set()) or set())
    if actor_scope != "actor_regime":
        return degraded
    if regime is None:
        return {
            str(key).split("|regime:", 1)[0]
            for key in degraded
            if str(key or "").strip()
        }
    suffix = f"|regime:{flash_regime_degradation_key(regime)}"
    return {
        str(key).split("|regime:", 1)[0]
        for key in degraded
        if str(key).endswith(suffix)
    }


def flash_degraded_open_symbols(
    pipeline: object,
    *,
    guard_enabled: bool,
    symbol_guard_enabled: bool,
) -> set[str]:
    if not guard_enabled or not symbol_guard_enabled:
        return set()
    return {
        str(symbol).strip().upper()
        for symbol in (getattr(pipeline, "_flash_degraded_open_symbols", set()) or set())
        if str(symbol or "").strip()
    }
