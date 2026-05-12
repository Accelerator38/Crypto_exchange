"""Data DTOs для панелей дашбордов.

Все панели сначала строятся в data-form (immutable dataclasses), потом
рендерятся в любой backend. Это даёт:
  • Тестируемость: assert на конкретные значения, без рендеринга.
  • Backend-независимость: один и тот же data-объект → matplotlib PNG,
    HTML, mermaid, Jupyter-таблица.
  • Гарантия consistency: если у панели №1 у X показано −0.5, у панели №2
    у X тоже −0.5 — оба читают одно и то же data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..domain.types import Regime


# ────────────────────────────────────────────────────────────────────
# AttributionPanel — главная панель «вклад делегатов»
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AttributionRow:
    """Одна строка панели вклада."""

    label:           str
    realized_pnl:    float
    n_trades:        int
    n_wins:          int
    is_quarantined:  bool

    @property
    def win_rate(self) -> float:
        if self.n_trades <= 0:
            return 0.0
        return self.n_wins / self.n_trades * 100.0

    @property
    def display_label(self) -> str:
        suffix = " [Q]" if self.is_quarantined else ""
        return f"{self.label}{suffix}"


@dataclass(frozen=True)
class AttributionPanelData:
    """Панель «вклад делегатов в PnL Пантеона» (заменяет фейковый
    sub_agent_pvs из v1).
    """

    title:                 str
    rows:                  List[AttributionRow]
    total_realized_pnl:    float
    consistency_check:     bool   # True если суммы согласованы
    bar_range:             Tuple[Optional[int], Optional[int]] = (None, None)

    @property
    def positives(self) -> List[AttributionRow]:
        return [r for r in self.rows if r.realized_pnl > 0]

    @property
    def negatives(self) -> List[AttributionRow]:
        return [r for r in self.rows if r.realized_pnl < 0]


# ────────────────────────────────────────────────────────────────────
# Leaderboard — virtual PnL по агентам/игрокам
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LeaderboardRow:
    """Одна строка лидерборда."""

    label:           str
    pnl_pct:         float
    closed_trades:   int
    win_rate:        float
    sharpe:          float
    max_dd_pct:      float
    is_quarantined:  bool

    @property
    def display_label(self) -> str:
        suffix = " [Q]" if self.is_quarantined else ""
        return f"{self.label}{suffix}"


@dataclass(frozen=True)
class LeaderboardData:
    """Shadow leaderboard: «как если бы каждый агент торговал в одиночку».

    ВАЖНО: помечается как `is_virtual=True` (vs реальный AttributionPanel).
    Это явное разделение, чтобы пользователь не путал shadow PnL с real.
    """

    title:        str
    rows:         List[LeaderboardRow]    # отсортированы по pnl_pct↓
    is_virtual:   bool = True             # отображается в подзаголовке


# ────────────────────────────────────────────────────────────────────
# Regime heatmap — agent × regime PnL
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HeatmapCell:
    """Одна ячейка heatmap."""

    label:          str
    regime:         Regime
    pnl_pct:        float
    closed_trades:  int
    is_quarantined: bool


@dataclass(frozen=True)
class RegimeHeatmapData:
    """Heatmap: rows=labels (agents/players), cols=regimes."""

    title:    str
    labels:   List[str]               # отсортированы по affinity
    regimes:  List[Regime]
    cells:    List[List[HeatmapCell]] # cells[label_idx][regime_idx]
    quarantined_labels: List[str]


# ────────────────────────────────────────────────────────────────────
# Decision timeline — events с trace_id
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TimelineEvent:
    """Одно событие в timeline."""

    bar:        int
    event_type: str          # "BarStarted", "RegimeDetected", ...
    trace_id:   str
    summary:    str          # короткое описание


@dataclass(frozen=True)
class DecisionTimelineData:
    """Хронологическая лента событий — полезно для дебага."""

    title:    str
    events:   List[TimelineEvent]
    bar_from: int
    bar_to:   int


# ────────────────────────────────────────────────────────────────────
# Carantine status panel
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QuarantinePanelData:
    """Текущий состав карантина."""

    title:                str
    quarantined_labels:   List[str]    # отсортированы alpha
    seed_labels:          List[str]    # подмножество, было в seed
    auto_added:           List[str]    # появилось через recompute
    auto_released:        List[str]    # снято через recompute (положительный опыт)


# ────────────────────────────────────────────────────────────────────
# Symbol health panel
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SymbolHealthRow:
    sym:                str
    is_blocked:         bool
    failures_in_window: int
    seconds_remaining:  Optional[float]


@dataclass(frozen=True)
class SymbolHealthPanelData:
    title:  str
    rows:   List[SymbolHealthRow]


# ────────────────────────────────────────────────────────────────────
# Main dashboard — composite of all panels
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MainDashboardData:
    """Composite — все панели в одном объекте."""

    attribution:      AttributionPanelData
    leaderboard:      LeaderboardData
    regime_heatmap:   RegimeHeatmapData
    quarantine:       QuarantinePanelData
    symbol_health:    Optional[SymbolHealthPanelData] = None
    decision_timeline: Optional[DecisionTimelineData] = None
