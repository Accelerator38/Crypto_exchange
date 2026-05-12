"""Builders — pure-функции, строящие data DTO из единых источников.

Принцип: ВСЕ дашборды читают через эти функции. Никакая панель не лезет
напрямую в state QM/perf/ledger — только через builders, которые
собирают данные согласованно.

Это устраняет в v1:
  • P19 (shadow != real) — `build_attribution_panel` использует AttributionLedger,
    `build_leaderboard` — PerformanceMemory, оба явно помечены как разные источники.
  • P27 (разные дашборды используют разные данные) — оба читают QM через
    единый `is_quarantined`.
  • P28 (карантинные не серые на api_trading) — каждый builder помечает
    `is_quarantined` явно.
"""

from __future__ import annotations

from typing import List, Optional

from ..attribution import AttributionLedger
from ..attribution.event_log import EventLog
from ..attribution.events import (
    BarStarted,
    LeaderSelected,
    QuarantineRecomputed,
    RegimeDetected,
    SymbolBlocked,
)
from ..domain.types import Regime
from ..execution import SymbolHealthMonitor
from ..memory import PerformanceMemory, QuarantineManager
from .data import (
    AttributionPanelData,
    AttributionRow,
    DecisionTimelineData,
    HeatmapCell,
    LeaderboardData,
    LeaderboardRow,
    QuarantinePanelData,
    RegimeHeatmapData,
    SymbolHealthPanelData,
    SymbolHealthRow,
    TimelineEvent,
)


# ────────────────────────────────────────────────────────────────────
# AttributionPanel (real PnL по делегатам)
# ────────────────────────────────────────────────────────────────────


def build_attribution_panel(
    ledger: AttributionLedger,
    qm:     QuarantineManager,
    *,
    by:     str = "player",   # "player" | "agent"
    title:  Optional[str] = None,
) -> AttributionPanelData:
    """Строит AttributionPanelData из ledger.

    by="player" — по делегатам (`by_player`); это ОСНОВНАЯ панель.
    by="agent"  — по агентам-инициаторам (`by_agent`).

    Карантинные помечаются is_quarantined=True. Сортировка — по
    realized_pnl убыванию.
    """
    if by not in ("player", "agent"):
        raise ValueError(f"by must be 'player' or 'agent', got {by!r}")
    if by == "player":
        pnl_map = ledger.total_pnl_by_player()
        title = title or "Реальный вклад делегатов в P&L Пантеона"
    else:
        pnl_map = ledger.total_pnl_by_agent()
        title = title or "Реальный вклад агентов-инициаторов"

    trade_counts = ledger.trade_counts_by_player()
    win_counts = ledger.win_counts_by_player()
    # for by=agent эти counts пусты, и win_rate=0 — ОК

    rows: List[AttributionRow] = []
    for label, pnl in pnl_map.items():
        rows.append(AttributionRow(
            label=label,
            realized_pnl=pnl,
            n_trades=trade_counts.get(label, 0) if by == "player" else 0,
            n_wins=win_counts.get(label, 0) if by == "player" else 0,
            is_quarantined=qm.is_quarantined(label),
        ))
    rows.sort(key=lambda r: -r.realized_pnl)

    return AttributionPanelData(
        title=title,
        rows=rows,
        total_realized_pnl=ledger.total_realized_pnl,
        consistency_check=ledger.consistency_check(),
        bar_range=(None, None),
    )


# ────────────────────────────────────────────────────────────────────
# Shadow leaderboard (virtual PnL по агентам/игрокам)
# ────────────────────────────────────────────────────────────────────


def build_leaderboard(
    perf:           PerformanceMemory,
    qm:             QuarantineManager,
    *,
    regime:         Optional[Regime] = None,
    title:          Optional[str] = None,
    only_with_data: bool = True,
) -> LeaderboardData:
    """Строит LeaderboardData из PerformanceMemory.

    Если regime указан — метрики per-regime; иначе агрегат.

    ВАЖНО: помечается как `is_virtual=True`. Это shadow PnL — «как если бы
    агент торговал в одиночку», не вклад в реальный Пантеон.
    """
    label_part = (
        f" (regime={regime.label})" if regime is not None else " (aggregate)"
    )
    title = title or f"Shadow Leaderboard{label_part}"

    rows: List[LeaderboardRow] = []
    for label in perf.all_labels():
        m = perf.get(label, regime=regime)
        if only_with_data and not m.has_data:
            continue
        rows.append(LeaderboardRow(
            label=label,
            pnl_pct=m.pnl_pct,
            closed_trades=m.closed_trades,
            win_rate=m.win_rate,
            sharpe=m.sharpe,
            max_dd_pct=m.max_dd_pct,
            is_quarantined=qm.is_quarantined(label),
        ))
    rows.sort(key=lambda r: -r.pnl_pct)
    return LeaderboardData(title=title, rows=rows, is_virtual=True)


# ────────────────────────────────────────────────────────────────────
# Regime heatmap
# ────────────────────────────────────────────────────────────────────


def build_regime_heatmap(
    perf:        PerformanceMemory,
    qm:          QuarantineManager,
    *,
    regimes:     Optional[List[Regime]] = None,
    only_with_data: bool = True,
    title:       str = "Agent × Regime PnL Heatmap",
) -> RegimeHeatmapData:
    """Строит heatmap: rows=labels, cols=regimes, value=pnl_pct."""
    regimes = regimes or [Regime.BULLISH, Regime.BEARISH, Regime.NEUTRAL, Regime.CRASH]

    # Собираем labels с хоть какой-то per-regime активностью
    labels: List[str] = []
    cells: List[List[HeatmapCell]] = []
    quarantined: List[str] = []

    for label in perf.all_labels():
        per_regime = perf.per_regime_for_label(label)
        # Skip полностью пустые
        if only_with_data and not any(m.has_data for m in per_regime.values()):
            continue
        is_q = qm.is_quarantined(label)
        if is_q:
            quarantined.append(label)
        row_cells = []
        for regime in regimes:
            m = per_regime.get(regime)
            if m is None:
                row_cells.append(HeatmapCell(
                    label=label, regime=regime,
                    pnl_pct=0.0, closed_trades=0,
                    is_quarantined=is_q,
                ))
            else:
                row_cells.append(HeatmapCell(
                    label=label, regime=regime,
                    pnl_pct=m.pnl_pct, closed_trades=m.closed_trades,
                    is_quarantined=is_q,
                ))
        labels.append(label)
        cells.append(row_cells)

    return RegimeHeatmapData(
        title=title,
        labels=labels,
        regimes=list(regimes),
        cells=cells,
        quarantined_labels=quarantined,
    )


# ────────────────────────────────────────────────────────────────────
# Quarantine panel
# ────────────────────────────────────────────────────────────────────


def build_quarantine_panel(
    qm:    QuarantineManager,
    *,
    auto_added:    Optional[List[str]] = None,
    auto_released: Optional[List[str]] = None,
    title:         str = "Карантин агентов/игроков",
) -> QuarantinePanelData:
    """Строит панель текущего состояния карантина."""
    current = sorted(qm.all_quarantined())
    seed = sorted(qm.seed)
    seed_in_q = [s for s in current if s in qm.seed]
    return QuarantinePanelData(
        title=title,
        quarantined_labels=current,
        seed_labels=seed_in_q,
        auto_added=sorted(auto_added or []),
        auto_released=sorted(auto_released or []),
    )


# ────────────────────────────────────────────────────────────────────
# Symbol health panel
# ────────────────────────────────────────────────────────────────────


def build_symbol_health_panel(
    health: SymbolHealthMonitor,
    *,
    syms:   Optional[List[str]] = None,
    title:  str = "Symbol health",
) -> SymbolHealthPanelData:
    """Строит панель здоровья символов.

    Если syms указан — только для них. Иначе — только заблокированные
    плюс те, у кого есть failures.
    """
    if syms is None:
        # все заблокированные
        rows = []
        for sym, remaining in health.all_blocked().items():
            st = health.status(sym)
            rows.append(SymbolHealthRow(
                sym=sym,
                is_blocked=True,
                failures_in_window=st.failures_in_window,
                seconds_remaining=remaining,
            ))
    else:
        rows = []
        for sym in syms:
            st = health.status(sym)
            rows.append(SymbolHealthRow(
                sym=sym,
                is_blocked=st.is_blocked,
                failures_in_window=st.failures_in_window,
                seconds_remaining=(
                    None if st.blocked_until_ts is None
                    else max(0.0, st.blocked_until_ts - health.now())
                ),
            ))
    rows.sort(key=lambda r: (-int(r.is_blocked), r.sym))
    return SymbolHealthPanelData(title=title, rows=rows)


# ────────────────────────────────────────────────────────────────────
# Decision timeline
# ────────────────────────────────────────────────────────────────────


def build_decision_timeline(
    event_log: EventLog,
    *,
    bar_from:  Optional[int] = None,
    bar_to:    Optional[int] = None,
    max_events: int = 200,
    title:     str = "Decision timeline",
) -> DecisionTimelineData:
    """Хронологическая лента ключевых решений.

    Включает: BarStarted, RegimeDetected, QuarantineRecomputed,
    LeaderSelected, SymbolBlocked. Не SignalEmitted (их слишком много).
    """
    interesting = (
        BarStarted, RegimeDetected, QuarantineRecomputed,
        LeaderSelected, SymbolBlocked,
    )
    raw_events = list(event_log.query(
        event_types=interesting,
        after_bar=bar_from,
        before_bar=bar_to,
    ))
    # ограничим max_events последними
    raw_events = raw_events[-max_events:]

    out: List[TimelineEvent] = []
    for ev in raw_events:
        out.append(TimelineEvent(
            bar=ev.bar,
            event_type=type(ev).__name__,
            trace_id=ev.trace_id,
            summary=_summarize(ev),
        ))
    actual_from = raw_events[0].bar if raw_events else (bar_from or 0)
    actual_to = raw_events[-1].bar if raw_events else (bar_to or 0)
    return DecisionTimelineData(
        title=title,
        events=out,
        bar_from=actual_from,
        bar_to=actual_to,
    )


def _summarize(ev) -> str:
    """Короткое описание события для timeline."""
    if isinstance(ev, BarStarted):
        return f"bar #{ev.bar} started"
    if isinstance(ev, RegimeDetected):
        if ev.is_change:
            return f"regime: {ev.from_regime.label} → {ev.regime.label}"
        return f"regime: {ev.regime.label}"
    if isinstance(ev, QuarantineRecomputed):
        adds = ",".join(sorted(ev.added)) if ev.added else "-"
        rems = ",".join(sorted(ev.removed)) if ev.removed else "-"
        return f"carantine +[{adds}] −[{rems}]"
    if isinstance(ev, LeaderSelected):
        prev = ev.previous_label or "(none)"
        return f"leader: {prev} → {ev.player_label} ({ev.reason})"
    if isinstance(ev, SymbolBlocked):
        return f"symbol blocked: {ev.sym} ({ev.failures_in_window} failures)"
    return type(ev).__name__
