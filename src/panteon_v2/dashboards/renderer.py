"""DashboardRenderer — главный фасад дашбордов.

Гарантирует что все панели прочитают данные из ЕДИНЫХ источников
(AttributionLedger, PerformanceMemory, QuarantineManager, EventLog,
SymbolHealthMonitor) и используют ОДИН набор цветов / форматов.

Это устраняет в v1 P26-P29:
  • Разные функции вычисляли PnL по-разному (фейковая формула в одной,
    shadow PV в другой) → расхождение на дашбордах.
  • Карантинные где-то были серыми, где-то нет.
  • status_legend не обновлялся при изменении логики.

Использование:
    renderer = DashboardRenderer(
        ledger=attribution_ledger,
        perf=performance_memory,
        qm=quarantine_manager,
        event_log=event_log,
        health=symbol_health,
        palette=DEFAULT_PALETTE,
    )
    main_data = renderer.build_main()
    text_form = TextRenderer().render_main(main_data)
"""

from __future__ import annotations

from typing import List, Optional

from ..attribution import AttributionLedger
from ..attribution.event_log import EventLog
from ..domain.types import Regime
from ..execution import SymbolHealthMonitor
from ..memory import PerformanceMemory, QuarantineManager
from .builders import (
    build_attribution_panel,
    build_decision_timeline,
    build_leaderboard,
    build_quarantine_panel,
    build_regime_heatmap,
    build_symbol_health_panel,
)
from .colors import DEFAULT_PALETTE, DashboardPalette
from .data import (
    AttributionPanelData,
    DecisionTimelineData,
    LeaderboardData,
    MainDashboardData,
    QuarantinePanelData,
    RegimeHeatmapData,
    SymbolHealthPanelData,
)


class DashboardRenderer:
    """Фасад дашбордов: единая точка построения всех панелей.

    Не делает рендеринг — только сборка data DTO. Backend (text/HTML/
    matplotlib) — это отдельный слой.
    """

    def __init__(
        self,
        *,
        ledger:    AttributionLedger,
        perf:      PerformanceMemory,
        qm:        QuarantineManager,
        event_log: Optional[EventLog] = None,
        health:    Optional[SymbolHealthMonitor] = None,
        palette:   DashboardPalette = DEFAULT_PALETTE,
    ):
        self._ledger = ledger
        self._perf = perf
        self._qm = qm
        self._event_log = event_log
        self._health = health
        self._palette = palette

    # ── Per-panel builders ──────────────────────────────────────────

    def build_attribution(
        self,
        *,
        by: str = "player",
    ) -> AttributionPanelData:
        return build_attribution_panel(self._ledger, self._qm, by=by)

    def build_leaderboard(
        self,
        *,
        regime: Optional[Regime] = None,
    ) -> LeaderboardData:
        return build_leaderboard(self._perf, self._qm, regime=regime)

    def build_regime_heatmap(
        self,
        *,
        regimes: Optional[List[Regime]] = None,
    ) -> RegimeHeatmapData:
        return build_regime_heatmap(self._perf, self._qm, regimes=regimes)

    def build_quarantine(
        self,
        *,
        auto_added:    Optional[List[str]] = None,
        auto_released: Optional[List[str]] = None,
    ) -> QuarantinePanelData:
        return build_quarantine_panel(
            self._qm,
            auto_added=auto_added,
            auto_released=auto_released,
        )

    def build_symbol_health(
        self,
        *,
        syms: Optional[List[str]] = None,
    ) -> Optional[SymbolHealthPanelData]:
        if self._health is None:
            return None
        return build_symbol_health_panel(self._health, syms=syms)

    def build_decision_timeline(
        self,
        *,
        bar_from:   Optional[int] = None,
        bar_to:     Optional[int] = None,
        max_events: int = 200,
    ) -> Optional[DecisionTimelineData]:
        if self._event_log is None:
            return None
        return build_decision_timeline(
            self._event_log,
            bar_from=bar_from,
            bar_to=bar_to,
            max_events=max_events,
        )

    # ── Composite ───────────────────────────────────────────────────

    def build_main(
        self,
        *,
        regime:           Optional[Regime] = None,
        auto_added:       Optional[List[str]] = None,
        auto_released:    Optional[List[str]] = None,
        timeline_max:     int = 50,
    ) -> MainDashboardData:
        """Собирает все панели в MainDashboardData.

        Все панели гарантированно используют:
          • тот же `qm` для is_quarantined → консистентная серая раскраска
          • тот же `palette` для цветов
        """
        return MainDashboardData(
            attribution=self.build_attribution(),
            leaderboard=self.build_leaderboard(regime=regime),
            regime_heatmap=self.build_regime_heatmap(),
            quarantine=self.build_quarantine(
                auto_added=auto_added,
                auto_released=auto_released,
            ),
            symbol_health=self.build_symbol_health(),
            decision_timeline=self.build_decision_timeline(
                max_events=timeline_max,
            ),
        )

    @property
    def palette(self) -> DashboardPalette:
        return self._palette
