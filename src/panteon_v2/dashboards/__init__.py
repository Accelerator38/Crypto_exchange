"""Dashboards layer (Phase 6).

Все панели читают из единых источников: AttributionLedger,
PerformanceMemory, QuarantineManager, EventLog, SymbolHealthMonitor.
Никаких ad-hoc формул и параллельных регистров — гарантированная
консистентность отображения.

Архитектура:
  colors        — единая палитра (карантин ВСЕГДА серый)
  data          — immutable DTO для каждой панели
  builders      — pure-функции, строящие data из источников
  renderer      — DashboardRenderer (фасад)
  text_renderer — простой ASCII backend для отладки и тестов
"""

from .colors import DEFAULT_PALETTE, DashboardPalette
from .data import (
    AttributionPanelData,
    AttributionRow,
    DecisionTimelineData,
    HeatmapCell,
    LeaderboardData,
    LeaderboardRow,
    MainDashboardData,
    QuarantinePanelData,
    RegimeHeatmapData,
    SymbolHealthPanelData,
    SymbolHealthRow,
    TimelineEvent,
)
from .builders import (
    build_attribution_panel,
    build_decision_timeline,
    build_leaderboard,
    build_quarantine_panel,
    build_regime_heatmap,
    build_symbol_health_panel,
)
from .renderer import DashboardRenderer
from .text_renderer import TextRenderer
from .png_renderer import write_operator_pngs

__all__ = [
    # colors
    "DEFAULT_PALETTE",
    "DashboardPalette",
    # data
    "AttributionPanelData",
    "AttributionRow",
    "DecisionTimelineData",
    "HeatmapCell",
    "LeaderboardData",
    "LeaderboardRow",
    "MainDashboardData",
    "QuarantinePanelData",
    "RegimeHeatmapData",
    "SymbolHealthPanelData",
    "SymbolHealthRow",
    "TimelineEvent",
    # builders
    "build_attribution_panel",
    "build_decision_timeline",
    "build_leaderboard",
    "build_quarantine_panel",
    "build_regime_heatmap",
    "build_symbol_health_panel",
    # renderer
    "DashboardRenderer",
    "TextRenderer",
    "write_operator_pngs",
]
