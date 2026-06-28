from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CausalActorRouterConfig:
    min_closed_trades: int = 5
    exploration_enabled: bool = False
    exploration_risk_mult: float = 0.05
    min_expectancy: float = 0.0
    min_pnl_lcb: float | None = None
    default_risk_mult: float = 1.0


@dataclass(frozen=True)
class RoutedActor:
    label: str
    actor_type: str
    score: float
    score_source: str
    sample_closed: int
    expectancy: float
    pnl_lcb: float | None
    risk_mult: float
    reason: str


def route_actor(
    candidates: Sequence[Any],
    memory: Mapping[str, Any],
    *,
    symbol: str,
    regime: str,
    bar: int,
    config: CausalActorRouterConfig | None = None,
) -> RoutedActor:
    cfg = config or CausalActorRouterConfig()
    usable = [candidate for candidate in candidates if _candidate_label(candidate)]
    if not usable:
        return RoutedActor(
            label="",
            actor_type="",
            score=0.0,
            score_source="none",
            sample_closed=0,
            expectancy=0.0,
            pnl_lcb=None,
            risk_mult=0.0,
            reason="no_candidates",
        )

    rows = []
    for candidate in usable:
        label = _candidate_label(candidate)
        stats = _prior_stats(
            memory.get(label),
            symbol=symbol,
            regime=regime,
            bar=bar,
        )
        rows.append((candidate, stats))

    confirmed = [
        (candidate, stats)
        for candidate, stats in rows
        if stats["closed_trades"] >= cfg.min_closed_trades
        and stats["expectancy"] > cfg.min_expectancy
        and (
            cfg.min_pnl_lcb is None
            or stats["pnl_lcb"] is None
            or stats["pnl_lcb"] >= cfg.min_pnl_lcb
        )
    ]
    if confirmed:
        candidate, stats = max(confirmed, key=_causal_rank_key)
        return _routed(
            candidate,
            stats,
            score=float(stats["expectancy"]),
            score_source="causal_memory",
            risk_mult=_candidate_risk_mult(candidate, cfg.default_risk_mult),
            reason="causal_memory",
        )

    if cfg.exploration_enabled:
        sparse = [
            (candidate, stats)
            for candidate, stats in rows
            if 0 < stats["closed_trades"] < cfg.min_closed_trades
            and stats["expectancy"] > cfg.min_expectancy
            and (
                cfg.min_pnl_lcb is None
                or stats["pnl_lcb"] is None
                or stats["pnl_lcb"] >= cfg.min_pnl_lcb
            )
        ]
        if sparse:
            candidate, stats = max(sparse, key=_causal_rank_key)
            return _routed(
                candidate,
                stats,
                score=float(stats["expectancy"]),
                score_source="causal_memory_sparse",
                risk_mult=min(
                    _candidate_risk_mult(candidate, cfg.default_risk_mult),
                    float(cfg.exploration_risk_mult),
                ),
                reason="exploration_floor",
            )

    candidate = max(usable, key=lambda item: _float_attr(item, "score", 0.0))
    return RoutedActor(
        label=_candidate_label(candidate),
        actor_type=str(_attr(candidate, "actor_type", "") or ""),
        score=_float_attr(candidate, "score", 0.0),
        score_source="candidate_score",
        sample_closed=0,
        expectancy=0.0,
        pnl_lcb=None,
        risk_mult=_candidate_risk_mult(candidate, cfg.default_risk_mult),
        reason="candidate_score_fallback",
    )


def _prior_stats(raw: Any, *, symbol: str, regime: str, bar: int) -> dict[str, Any]:
    best = {
        "closed_trades": 0,
        "expectancy": 0.0,
        "pnl_lcb": None,
    }
    for row in _memory_rows(raw):
        if not isinstance(row, Mapping):
            continue
        row_bar = _int_value(row.get("bar"), -1)
        if row_bar >= int(bar):
            continue
        row_symbol = str(row.get("symbol") or "").strip().upper()
        if row_symbol and row_symbol != str(symbol or "").strip().upper():
            continue
        row_regime = str(row.get("regime") or "").strip().lower()
        if row_regime and row_regime != str(regime or "").strip().lower():
            continue
        closed = _int_value(row.get("closed_trades"), 0)
        expectancy = _float_value(
            row.get("expectancy", row.get("pnl_per_trade", 0.0)),
            0.0,
        )
        pnl_lcb = _optional_float(row.get("pnl_lcb", row.get("pnl_per_trade_lcb")))
        rank = (
            closed,
            -1.0e18 if pnl_lcb is None else pnl_lcb,
            expectancy,
            row_bar,
        )
        best_rank = (
            int(best["closed_trades"]),
            -1.0e18 if best["pnl_lcb"] is None else float(best["pnl_lcb"]),
            float(best["expectancy"]),
            -1,
        )
        if rank > best_rank:
            best = {
                "closed_trades": closed,
                "expectancy": expectancy,
                "pnl_lcb": pnl_lcb,
            }
    return best


def _memory_rows(raw: Any) -> Iterable[Mapping[str, Any]]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("closed_trades", "expectancy", "pnl_lcb")):
            return (raw,)
        rows = []
        for value in raw.values():
            rows.extend(list(_memory_rows(value)))
        return tuple(rows)
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
        return tuple(item for item in raw if isinstance(item, Mapping))
    return ()


def _routed(
    candidate: Any,
    stats: Mapping[str, Any],
    *,
    score: float,
    score_source: str,
    risk_mult: float,
    reason: str,
) -> RoutedActor:
    return RoutedActor(
        label=_candidate_label(candidate),
        actor_type=str(_attr(candidate, "actor_type", "") or ""),
        score=float(score),
        score_source=score_source,
        sample_closed=int(stats.get("closed_trades", 0) or 0),
        expectancy=float(stats.get("expectancy", 0.0) or 0.0),
        pnl_lcb=_optional_float(stats.get("pnl_lcb")),
        risk_mult=max(0.0, float(risk_mult)),
        reason=reason,
    )


def _causal_rank_key(item: tuple[Any, Mapping[str, Any]]) -> tuple[float, float, int, float]:
    candidate, stats = item
    pnl_lcb = stats.get("pnl_lcb")
    return (
        -1.0e18 if pnl_lcb is None else float(pnl_lcb),
        float(stats.get("expectancy", 0.0) or 0.0),
        int(stats.get("closed_trades", 0) or 0),
        _float_attr(candidate, "score", 0.0),
    )


def _candidate_label(candidate: Any) -> str:
    return str(_attr(candidate, "label", "") or "").strip()


def _candidate_risk_mult(candidate: Any, default: float) -> float:
    return _float_attr(candidate, "risk_mult", default)


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_attr(obj: Any, key: str, default: float) -> float:
    return _float_value(_attr(obj, key, default), default)


def _float_value(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
