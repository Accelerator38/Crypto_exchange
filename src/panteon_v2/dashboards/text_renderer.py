"""Text backend для дашбордов.

Простой ASCII-renderer — для отладки, тестов, CLI. Все продакшн-бекенды
(matplotlib для PNG, jinja для HTML) — дополнительные слои; они принимают
тот же data DTO.

Цели:
  • Любую панель можно отрендерить и assert-ом проверить выход в тестах.
  • Visible source of truth для разработчика без запуска UI.
"""

from __future__ import annotations

from typing import List, Optional

from ..domain.types import Regime
from .data import (
    AttributionPanelData,
    DecisionTimelineData,
    LeaderboardData,
    MainDashboardData,
    QuarantinePanelData,
    RegimeHeatmapData,
    SymbolHealthPanelData,
)


class TextRenderer:
    """Renderer в plain ASCII.

    Использование:
        r = TextRenderer()
        text = r.render_attribution(panel_data)
        print(text)
    """

    # ── Attribution panel ───────────────────────────────────────────

    def render_attribution(self, data: AttributionPanelData) -> str:
        out = []
        out.append(_h(data.title))
        if not data.rows:
            out.append("  (нет данных — нет закрытых сделок)")
        else:
            out.append(f"  {'Делегат':<28s} {'PnL ($)':>10s}  {'Trades':>7s}  {'Win%':>6s}")
            out.append(f"  {'-'*28} {'-'*10}  {'-'*7}  {'-'*6}")
            for r in data.rows:
                marker = "[Q]" if r.is_quarantined else "   "
                out.append(
                    f"  {r.label:<25s} {marker} {r.realized_pnl:+10.4f}  "
                    f"{r.n_trades:>7d}  {r.win_rate:>5.1f}%"
                )
        out.append("")
        out.append(f"  ─── Σ realized PnL: {data.total_realized_pnl:+.4f}")
        if data.consistency_check:
            out.append("  ✓ consistency check: OK")
        else:
            out.append("  ✗ consistency check: FAILED")
        return "\n".join(out)

    # ── Leaderboard ─────────────────────────────────────────────────

    def render_leaderboard(self, data: LeaderboardData) -> str:
        out = []
        out.append(_h(data.title))
        if data.is_virtual:
            out.append("  (virtual — как если бы агент торговал в одиночку)")
        if not data.rows:
            out.append("  (нет данных)")
        else:
            out.append(
                f"  {'Label':<28s} {'PnL%':>8s}  "
                f"{'Closed':>6s}  {'Win%':>6s}  {'Sharpe':>7s}  {'MaxDD%':>7s}"
            )
            out.append(f"  {'-'*28} {'-'*8}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}")
            for r in data.rows:
                marker = "[Q]" if r.is_quarantined else "   "
                out.append(
                    f"  {r.label:<25s} {marker} {r.pnl_pct:+8.4f}  "
                    f"{r.closed_trades:>6d}  {r.win_rate:>5.1f}%  "
                    f"{r.sharpe:>+7.3f}  {r.max_dd_pct:>7.2f}"
                )
        return "\n".join(out)

    # ── Regime heatmap ──────────────────────────────────────────────

    def render_regime_heatmap(self, data: RegimeHeatmapData) -> str:
        out = []
        out.append(_h(data.title))
        if not data.labels:
            out.append("  (нет данных)")
            return "\n".join(out)
        # Header
        regime_headers = "  ".join(f"{r.label:>10s}" for r in data.regimes)
        out.append(f"  {'Label':<25s}     {regime_headers}")
        out.append(f"  {'-'*25}     " + "  ".join(["-" * 10 for _ in data.regimes]))
        for label_idx, label in enumerate(data.labels):
            row = data.cells[label_idx]
            marker = "[Q]" if (label in data.quarantined_labels) else "   "
            cells_str = "  ".join(
                _format_heatmap_cell(c) for c in row
            )
            out.append(f"  {label:<25s} {marker} {cells_str}")
        return "\n".join(out)

    # ── Quarantine panel ────────────────────────────────────────────

    def render_quarantine(self, data: QuarantinePanelData) -> str:
        out = []
        out.append(_h(data.title))
        if not data.quarantined_labels:
            out.append("  (карантин пуст)")
            return "\n".join(out)
        out.append(f"  В карантине ({len(data.quarantined_labels)}): "
                   f"{', '.join(data.quarantined_labels)}")
        if data.seed_labels:
            out.append(f"  Из них seed: {', '.join(data.seed_labels)}")
        if data.auto_added:
            out.append(f"  ↑ авто-добавлены: {', '.join(data.auto_added)}")
        if data.auto_released:
            out.append(f"  ↓ авто-выпущены: {', '.join(data.auto_released)}")
        return "\n".join(out)

    # ── Symbol health ──────────────────────────────────────────────

    def render_symbol_health(self, data: SymbolHealthPanelData) -> str:
        out = []
        out.append(_h(data.title))
        if not data.rows:
            out.append("  (все символы здоровы)")
            return "\n".join(out)
        out.append(f"  {'Symbol':<12s} {'Blocked':<8s} {'Failures':<10s} {'Remaining':<10s}")
        out.append(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
        for r in data.rows:
            blocked = "YES" if r.is_blocked else "no"
            remaining = (
                f"{r.seconds_remaining:.0f}s"
                if r.seconds_remaining is not None else "-"
            )
            out.append(
                f"  {r.sym:<12s} {blocked:<8s} {r.failures_in_window:<10d} {remaining:<10s}"
            )
        return "\n".join(out)

    # ── Decision timeline ──────────────────────────────────────────

    def render_timeline(self, data: DecisionTimelineData) -> str:
        out = []
        out.append(_h(data.title))
        out.append(f"  bars: [{data.bar_from} … {data.bar_to}], events: {len(data.events)}")
        if not data.events:
            return "\n".join(out)
        out.append("")
        for e in data.events:
            out.append(f"  bar={e.bar:<6d} {e.event_type:<22s} {e.summary}")
        return "\n".join(out)

    # ── Main dashboard (composite) ──────────────────────────────────

    def render_main(self, data: MainDashboardData) -> str:
        parts = [
            self.render_attribution(data.attribution),
            "",
            self.render_leaderboard(data.leaderboard),
            "",
            self.render_regime_heatmap(data.regime_heatmap),
            "",
            self.render_quarantine(data.quarantine),
        ]
        if data.symbol_health is not None:
            parts.extend(["", self.render_symbol_health(data.symbol_health)])
        if data.decision_timeline is not None:
            parts.extend(["", self.render_timeline(data.decision_timeline)])
        return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _h(title: str) -> str:
    line = "═" * max(len(title) + 4, 60)
    return f"\n{line}\n  {title}\n{line}"


def _format_heatmap_cell(cell) -> str:
    if cell.closed_trades == 0 and cell.pnl_pct == 0:
        return f"{'   -    ':>10s}"
    return f"{cell.pnl_pct:+8.3f}%/{cell.closed_trades:d}".rjust(10)
