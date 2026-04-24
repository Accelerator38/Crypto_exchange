"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  mexc_dashboards.py  —  Расширенные дашборды для MEXC trading suite        ║
║                                                                              ║
║  Три дашборда:                                                               ║
║  1. plot_compound_growth_enhanced()  — compound growth как в симуляции      ║
║       • Логарифмическая шкала Y                                              ║
║       • Топ-8 подсвечены, остальные прозрачны                               ║
║       • Таблица итоговой статистики по каждому агенту                       ║
║       • Тип рынка (режим) через цвет фона и метку                           ║
║                                                                              ║
║  2. plot_api_trading_dashboard()  — реальная торговля Panteon/live-agent    ║
║       • Equity curve с маркерами сделок                                      ║
║       • PnL по суб-агентам (waterfall bar chart)                             ║
║       • Вклад каждого суб-агента в прибыль/убыль                            ║
║       • Тип рынка: detected regime + funding rate indicator                 ║
║       • История ордеров + открытые позиции                                  ║
║                                                                              ║
║  3. plot_master_player_dashboard()  — бумажная торговля MasterPlayer        ║
║       • Compound growth по 6 суб-игрокам (логарит. шкала)                  ║
║       • Вклад каждого суб-игрока в итоговый PnL                             ║
║       • Активные агенты + позиции                                           ║
║       • Тип рынка: режим + funding bias                                     ║
║                                                                              ║
║  Использование (добавить в mexc_connector.py/_patched_cycle):               ║
║      from mexc_dashboards import (                                           ║
║          plot_compound_growth_enhanced,                                      ║
║          plot_api_trading_dashboard,                                         ║
║          plot_master_player_dashboard,                                       ║
║      )                                                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
import traceback
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import numpy as np

from project_paths import add_runtime_paths


def _bootstrap_project_paths():
    add_runtime_paths()


_bootstrap_project_paths()

log = logging.getLogger("mexc_dashboards")

# ── Цветовая схема (dark theme) ───────────────────────────────────────────────
DARK   = "#0D1117"
MID    = "#161B22"
CARD   = "#1C2128"
GRID   = "#21262D"
GRN    = "#3FB950"
RED    = "#F85149"
CYN    = "#58A6FF"
GOL    = "#F0C040"
PUR    = "#BC8CFF"
ORG    = "#FF8C00"
WHT    = "#E6EDF3"
GRY    = "#8B949E"
TEAL   = "#39D353"
PINK   = "#FF7B72"

# Цвета по суб-игрокам / агентам
PLAYER_COLORS = {
    'Alpha':  '#3498db', 'Beta':   '#27ae60', 'Gamma':  '#e74c3c',
    'P7Ways': '#9b59b6', 'Ultima': '#f39c12', 'Neuro':  '#1abc9c',
    'MasterPlayer': '#ffffff',
}

AGENT_PALETTE = [
    '#58A6FF', '#3FB950', '#F85149', '#F0C040', '#BC8CFF',
    '#FF8C00', '#39D353', '#FF7B72', '#79C0FF', '#56D364',
    '#FFA657', '#D2A8FF', '#FF9F7F', '#7EE787', '#A5D6FF',
]

REGIME_COLORS = {
    'bull':     ('#27ae60', '#0a2010', '[BULL]'),
    'bear':     ('#e74c3c', '#200a0a', '[BEAR]'),
    'sideways': ('#f39c12', '#1a1500', '[SIDE]'),
    'neutral':  ('#f39c12', '#1a1500', '[NEUTRAL]'),
    'crash':    ('#c0392b', '#300000', '[CRASH]'),
    'unknown':  ('#8B949E', '#111',    '[UNKNOWN]'),
}


def _style_ax(ax, title="", fs=9, xlab="", ylab=""):
    ax.set_facecolor(MID)
    ax.tick_params(colors=GRY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6)
    if title:
        ax.set_title(title, color=WHT, fontsize=fs, fontweight='bold', pad=4)
    if xlab:
        ax.set_xlabel(xlab, color=GRY, fontsize=7)
    if ylab:
        ax.set_ylabel(ylab, color=GRY, fontsize=7)


def _regime_info(regime: str):
    return REGIME_COLORS.get(regime.lower(), REGIME_COLORS['unknown'])


def _fmt_pct(v, digits=2):
    return f"{v:+.{digits}f}%"


def _resolve_sub_agent_pnl_and_initials(stats, sub_agent_pnl: Optional[Dict[str, float]] = None):
    sub_pvs = getattr(stats, 'sub_agent_pvs', {}) or {}
    sub_initials = getattr(stats, 'sub_agent_initials', {}) or {}
    fallback_base = float(getattr(stats, 'initial_capital', 0.0) or 0.0)

    pnl_data = {}
    if sub_agent_pnl:
        for name, value in sub_agent_pnl.items():
            try:
                pnl_data[str(name)] = float(value)
            except Exception:
                continue
    else:
        for name, pv_val in sub_pvs.items():
            try:
                pv = float(pv_val)
            except Exception:
                continue
            base = float(sub_initials.get(name, fallback_base) or fallback_base)
            pnl_data[str(name)] = pv - base
    return pnl_data, sub_initials


def _rolling_max_dd(series):
    """Максимальная просадка от пика."""
    arr = np.array(series)
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / np.where(peak > 0, peak, 1) * 100
    return float(np.min(dd))


def _sharpe(series, rf=0.0):
    arr = np.array(series)
    if len(arr) < 2:
        return 0.0
    rets = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1)
    mu   = float(np.mean(rets)) - rf
    sd   = float(np.std(rets))
    return round(mu / sd * np.sqrt(len(rets)), 2) if sd > 1e-12 else 0.0


def _repair_curve_spikes(values, floor=0.0):
    """Repairs isolated non-finite / zero / deep V-spike points in plotted curves."""
    if not values:
        return []
    raw = []
    for v in values:
        try:
            raw.append(float(v))
        except Exception:
            raw.append(float("nan"))

    cleaned = list(raw)
    n = len(raw)

    def _nearest_valid(idx, step):
        j = idx + step
        while 0 <= j < n:
            cand = raw[j]
            if np.isfinite(cand):
                return cand
            j += step
        return None

    for i, val in enumerate(raw):
        prev_v = _nearest_valid(i, -1)
        next_v = _nearest_valid(i, +1)

        if not np.isfinite(val):
            if prev_v is not None and next_v is not None:
                cleaned[i] = (prev_v + next_v) / 2.0
            elif prev_v is not None:
                cleaned[i] = prev_v
            elif next_v is not None:
                cleaned[i] = next_v
            continue

        if val <= floor:
            if prev_v is not None and prev_v > floor and next_v is not None and next_v > floor:
                cleaned[i] = (prev_v + next_v) / 2.0
            elif prev_v is not None and prev_v > floor:
                cleaned[i] = prev_v
            elif next_v is not None and next_v > floor:
                cleaned[i] = next_v
            continue

        if prev_v is None or next_v is None:
            continue
        ref_low = min(prev_v, next_v)
        ref_high = max(prev_v, next_v)
        if ref_low <= floor:
            continue
        close_neighbors = abs(prev_v - next_v) / max(ref_high, 1e-9) <= 0.08
        deep_drop = val < ref_low * 0.35
        if close_neighbors and deep_drop:
            cleaned[i] = (prev_v + next_v) / 2.0

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# 1. COMPOUND GROWTH ENHANCED (mexc_connector paper portfolios)
# ══════════════════════════════════════════════════════════════════════════════

def plot_compound_growth_enhanced(
    portfolios: Dict[str, Any],   # {name: PaperPortfolio}
    output_dir: str,
    warmup_end: int = 0,
    market_regime: str = 'unknown',
    funding_avg: float = 0.0,
    title_suffix: str = "",
) -> None:
    """
    Расширенная версия compound_growth.png:
      • Верх: кривые доходности (логарифм. шкала), топ-8 подсвечены
      • Тип рынка: цветной фон + метка режима
      • Funding rate индикатор
      • Низ: таблица статистики всех агентов (финальный PnL, MaxDD, Sharpe)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Собираем данные
        agent_data = {}
        for name, pf in portfolios.items():
            h_full = getattr(pf, 'history', [])
            if len(h_full) < 2:
                continue
            h_live = h_full[warmup_end:] if warmup_end < len(h_full) else h_full
            if len(h_live) < 2:
                continue
            ic   = getattr(pf, 'initial_capital', h_live[0])
            base = h_live[0] if h_live[0] > 0 else ic
            ret_arr = [(v / base - 1) * 100 for v in h_live]
            final_ret = ret_arr[-1]
            agent_data[name] = {
                'hist':      h_live,
                'ret':       ret_arr,
                'final':     final_ret,
                'sharpe':    _sharpe(h_live),
                'maxdd':     _rolling_max_dd(h_live),
                'ic':        ic,
                'n_trades':  len(getattr(pf, 'trades', [])),
            }

        if not agent_data:
            log.warning("[compound_growth_enhanced] нет данных")
            return

        # Сортируем по финальному PnL
        sorted_agents = sorted(agent_data.keys(),
                               key=lambda n: agent_data[n]['final'], reverse=True)
        top8 = sorted_agents[:8]
        n_agents = len(sorted_agents)

        fig = plt.figure(figsize=(22, 14), facecolor=DARK)
        rclr, rbg, rlbl = _regime_info(market_regime)
        fig.suptitle(
            f"Compound Portfolio Growth — MEXC Paper Trading  [{now_str}]{title_suffix}"
            f"  |  Режим рынка: {rlbl}  |  Funding avg: {funding_avg*100:+.4f}%",
            fontsize=12, fontweight='bold', color=WHT, y=0.99,
        )

        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2.5, 1],
                               hspace=0.35, left=0.06, right=0.98,
                               top=0.96, bottom=0.04)

        # ─── ROW 0: Growth curves ───────────────────────────────────────────
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor(DARK)

        # Фон по типу рынка
        ax.set_facecolor(rbg)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=GRY, labelsize=7)
        ax.grid(True, color=GRID, lw=0.4, alpha=0.5)

        palette = plt.cm.tab20(np.linspace(0, 1, max(n_agents, 1)))
        name_to_color = {}

        # Сначала рисуем неактивных (тонко, прозрачно)
        for idx, name in enumerate(sorted_agents):
            d   = agent_data[name]
            ret = d['ret']
            clr = PLAYER_COLORS.get(name, palette[idx % 20])
            name_to_color[name] = clr
            if name not in top8:
                ax.plot(range(len(ret)), ret, lw=0.7, color=clr,
                        alpha=0.18, zorder=1)

        # Топ-8 поверх
        legend_handles = []
        for idx, name in enumerate(sorted_agents):
            if name not in top8:
                continue
            d   = agent_data[name]
            ret = d['ret']
            clr = name_to_color[name]
            lw  = 2.0 if idx < 3 else 1.5
            ax.plot(range(len(ret)), ret, lw=lw, color=clr,
                    alpha=0.95, zorder=3)
            # Аннотация конечного значения
            ax.annotate(
                f"{name}\n{_fmt_pct(ret[-1])}",
                xy=(len(ret) - 1, ret[-1]),
                xytext=(8, 0), textcoords='offset points',
                color=clr, fontsize=7, fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='-', color=clr, lw=0.6, alpha=0.5),
            )
            legend_handles.append(
                Line2D([0], [0], color=clr, lw=lw,
                       label=f"{name}  {_fmt_pct(ret[-1])}  "
                             f"Sharpe={d['sharpe']:.2f}  MaxDD={d['maxdd']:.1f}%")
            )

        ax.axhline(0, color=GOL, lw=1.0, ls='--', alpha=0.5)

        # Метка режима
        ax.text(0.01, 0.97, rlbl, transform=ax.transAxes,
                color=rclr, fontsize=13, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor=rbg, edgecolor=rclr,
                          lw=1.5, alpha=0.9))

        # Funding rate как цветная полоса снизу
        if abs(funding_avg) > 0.0001:
            fclr = GRN if funding_avg >= 0 else RED
            ax.axhspan(ax.get_ylim()[0] if hasattr(ax, '_get_ylim') else -999,
                       -100, color=fclr, alpha=0.04, zorder=0)
            ax.text(0.01, 0.02,
                    f"Funding: {funding_avg*100:+.4f}%  "
                    f"({'лонги платят' if funding_avg > 0 else 'шорты платят'})",
                    transform=ax.transAxes, color=fclr, fontsize=7.5,
                    va='bottom', alpha=0.9)

        ax.legend(handles=legend_handles, loc='upper left', fontsize=7.5,
                  facecolor=CARD, edgecolor=GRID, labelcolor=WHT,
                  framealpha=0.92, ncol=2)
        ax.set_ylabel('Return %', color=GRY, fontsize=8)
        ax.set_xlabel('Bar (live)', color=GRY, fontsize=8)
        ax.set_title('Compound Portfolio Growth — топ-8 подсвечены',
                     color=WHT, fontsize=10, fontweight='bold')

        # ─── ROW 1: Stats table ──────────────────────────────────────────────
        ax_tbl = fig.add_subplot(gs[1])
        ax_tbl.set_facecolor(DARK)
        ax_tbl.axis('off')

        # Заголовок
        col_widths = [0.18, 0.09, 0.09, 0.09, 0.09, 0.09, 0.37]
        col_xs     = [0.01, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
        headers    = ['Агент', 'Final %', 'MaxDD %', 'Sharpe', 'IC $', 'Сделок', 'PnL bar']

        y_hdr = 0.95
        for i, (hdr, cx) in enumerate(zip(headers, col_xs)):
            ax_tbl.text(cx, y_hdr, hdr, transform=ax_tbl.transAxes,
                        color=GRY, fontsize=7.5, fontweight='bold', va='top')
        ax_tbl.plot([0.01, 0.99], [y_hdr - 0.02, y_hdr - 0.02],
                    color=GRID, lw=0.7, transform=ax_tbl.transAxes)

        y = y_hdr - 0.08
        max_rows = min(len(sorted_agents), 14)
        for rank, name in enumerate(sorted_agents[:max_rows]):
            d    = agent_data[name]
            fret = d['final']
            clr  = name_to_color.get(name, GRY)
            val_clr = GRN if fret >= 0 else RED
            bg_clr = '#1a2a1a' if fret >= 0 else '#2a1a1a'
            if rank < 3:
                ax_tbl.fill([0.01, 0.99, 0.99, 0.01],
                            [y - 0.005, y - 0.005, y + 0.058, y + 0.058],
                            transform=ax_tbl.transAxes,
                            color=bg_clr, alpha=0.4, zorder=0)

            n_bars = len(d['ret'])
            pnl_per_bar = fret / n_bars if n_bars > 0 else 0

            vals = [
                name,
                _fmt_pct(fret),
                _fmt_pct(d['maxdd'], 1),
                f"{d['sharpe']:+.2f}",
                f"${d['ic']:.1f}",
                str(d['n_trades']),
                f"{pnl_per_bar:+.4f}%/bar  ({n_bars} bars)",
            ]
            text_colors = [clr, val_clr, RED if d['maxdd'] < -10 else GOL,
                           GRN if d['sharpe'] > 0 else RED,
                           CYN, WHT, GRY]

            for i, (val, vc, cx) in enumerate(zip(vals, text_colors, col_xs)):
                ax_tbl.text(cx, y, val, transform=ax_tbl.transAxes,
                            color=vc, fontsize=7, va='top')
            y -= 0.065
            if y < 0.01:
                break

        out = os.path.join(output_dir, "compound_growth_enhanced.png")
        fig.savefig(out, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        fig.clf()
        plt.close(fig)
        log.info("  [compound_growth_enhanced] → %s", out)

    except Exception as e:
        log.warning("  [compound_growth_enhanced] failed: %s\n%s",
                    e, traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 2. API TRADING DASHBOARD (real money — Panteon / live-agent)
# ══════════════════════════════════════════════════════════════════════════════

def plot_api_trading_dashboard(
    stats,                          # TradingStats object from exchange_api_runtime.py
    output_dir: str,
    market_regime: str = 'unknown',
    funding_avg: float = 0.0,
    funding_history: Optional[List[float]] = None,
    sub_agent_pnl: Optional[Dict[str, float]] = None,  # {agent_name: pnl_usdt}
    sub_agent_trades: Optional[Dict[str, List[dict]]] = None,
    filename: str = "api_trading_enhanced.png",
) -> None:
    """
    Расширенный дашборд для реальной торговли Panteon:

    Макет (4 строки × 4 колонки):
      Row 0 [0:3]  — Equity curve + маркеры сделок + тип рынка
      Row 0 [3]    — Текущий статус + метрики
      Row 1 [0:2]  — Вклад суб-агентов: waterfall PnL chart
      Row 1 [2]    — Funding rate история + тип рынка
      Row 1 [3]    — Открытые позиции
      Row 2 [0:2]  — История ордеров + сделок
      Row 2 [2:4]  — Детальная таблица суб-агентов
      Row 3 [0:4]  — Последние 20 сигналов
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rclr, rbg, rlbl = _regime_info(market_regime)

        ic          = getattr(stats, 'initial_capital', 50.0)
        eq          = _repair_curve_spikes(getattr(stats, 'equity_curve', []))
        bal         = _repair_curve_spikes(getattr(stats, 'balance_curve', []))   # свободная маржа
        unreal_hist = _repair_curve_spikes(getattr(stats, 'unrealized_curve', []))  # нереализованный PnL
        signals     = getattr(stats, 'signals', [])
        trades      = getattr(stats, 'trades', [])
        positions   = getattr(stats, 'current_positions', {})
        order_hist  = getattr(stats, 'order_history', [])
        live_bars   = getattr(stats, 'live_bar_count', 0)
        sub_pvs     = getattr(stats, 'sub_agent_pvs', {})
        fund_hist   = funding_history or getattr(stats, 'funding_history', [])
        uptime      = getattr(stats, 'uptime_str', '?')

        fig = plt.figure(figsize=(24, 18), facecolor=DARK)
        _agent_name = getattr(stats, 'agent_name', 'Panteon v3')
        fig.suptitle(
            f"Live Dashboard — {_agent_name}  |  {now_str}"
            f"  |  Uptime {uptime}  |  live bars: {live_bars}",
            fontsize=11, fontweight='bold', color=WHT, y=0.995,
        )

        layout = gridspec.GridSpec(
            4, 4, figure=fig,
            hspace=0.52, wspace=0.30,
            left=0.05, right=0.98, top=0.975, bottom=0.03,
            height_ratios=[1.8, 1.2, 1.2, 0.9],
        )

        # ── ROW 0 left [0:3]: Equity curve + unrealPnL overlay ───────────────
        ax_eq = fig.add_subplot(layout[0, :3])
        ax_eq.set_facecolor(rbg)
        for sp in ax_eq.spines.values(): sp.set_color(rclr)
        ax_eq.tick_params(colors=GRY, labelsize=7)
        ax_eq.grid(True, color=GRID, lw=0.4, alpha=0.6)
        ax_eq.set_title(f"Equity (реальный фьючерсный баланс)",
                        color=WHT, fontsize=9, fontweight='bold')

        if len(eq) > 1:
            xs     = list(range(len(eq)))
            col_eq = GRN if eq[-1] >= ic else RED

            # Основная кривая Equity (available + unrealPnL)
            ax_eq.fill_between(xs, ic, eq, color=col_eq, alpha=0.15)
            ax_eq.plot(xs, eq, color=col_eq, lw=2.2,
                       label=f"Equity ${eq[-1]:.2f}", zorder=3)

            # Свободная маржа (available) — только если отличается от equity
            if len(bal) == len(eq) and abs(bal[-1] - eq[-1]) > 0.5:
                ax_eq.plot(xs, bal, color=CYN, lw=1.0, ls='--',
                           alpha=0.7, label=f"Свободно ${bal[-1]:.2f}")

            # Нереализованный PnL позиций — правая ось
            if len(unreal_hist) == len(eq):
                ax_ur = ax_eq.twinx()
                ax_ur.tick_params(colors=GRY, labelsize=6.5)
                ax_ur.set_ylabel('unrealPnL', color=PUR, fontsize=7)
                ur_col = [GRN if v >= 0 else RED for v in unreal_hist]
                # Рисуем как тонкие бары + сглаженную линию
                ax_ur.bar(xs, unreal_hist, color=ur_col, alpha=0.18, width=1.0)
                ax_ur.plot(xs, unreal_hist, color=PUR, lw=1.0, ls=':', alpha=0.8,
                           label=f"unrealPnL {unreal_hist[-1]:+.2f}")
                ax_ur.axhline(0, color=PUR, lw=0.5, alpha=0.3)
                ax_ur.legend(fontsize=6.5, loc='lower right',
                             facecolor=CARD, edgecolor=GRID, labelcolor=WHT)
                # Выделяем текущий unrealPnL цветом
                ur_now = unreal_hist[-1]
                ax_ur.text(0.99, 0.04,
                           f"unrealPnL: {ur_now:+.4f} USDT",
                           transform=ax_eq.transAxes, ha='right', va='bottom',
                           color=GRN if ur_now >= 0 else RED,
                           fontsize=7.5, fontweight='bold',
                           bbox=dict(fc=CARD, ec=GRID, lw=0.4, pad=2))

            ax_eq.axhline(ic, color=GOL, lw=1.0, ls='--', alpha=0.5,
                          label=f"Старт ${ic:.2f}")

            # Signal markers on the curve instead of full-height vertical lines.
            open_x, open_y = [], []
            close_x, close_y = [], []
            for sig in signals[-80:]:
                idx = sig.get('_eq_idx', len(eq) - 1)
                if 0 <= idx < len(eq):
                    if sig.get('action', 0) in (1, 2, 4, 5):
                        open_x.append(idx)
                        open_y.append(eq[idx])
                    else:
                        close_x.append(idx)
                        close_y.append(eq[idx])
            if open_x:
                ax_eq.scatter(open_x, open_y, s=24, marker='^', color=GRN,
                              alpha=0.78, edgecolors=DARK, linewidths=0.3,
                              zorder=4, label='Open/Long')
            if close_x:
                ax_eq.scatter(close_x, close_y, s=24, marker='v', color=RED,
                              alpha=0.78, edgecolors=DARK, linewidths=0.3,
                              zorder=4, label='Close/Short')

            pnl     = eq[-1] - ic
            pnl_pct = pnl / ic * 100 if ic > 0 else 0
            max_dd  = _rolling_max_dd(eq)
            ax_eq.text(
                0.99, 0.96,
                f"P&L: {pnl:+.4f} USDT ({pnl_pct:+.2f}%)\n"
                f"MaxDD: {max_dd:.2f}%  |  Сделок: {len(trades)}",
                transform=ax_eq.transAxes, ha='right', va='top',
                color=GRN if pnl >= 0 else RED, fontsize=8.5, fontweight='bold',
                bbox=dict(fc=CARD, ec=GRID, lw=0.5, pad=3),
            )
            ax_eq.legend(fontsize=7.5, loc='upper left',
                         facecolor=CARD, edgecolor=GRID, labelcolor=WHT)
        else:
            ax_eq.text(0.5, 0.5, f"Накопление данных...\nСтарт: ${ic:.2f}",
                       ha='center', va='center', color=GRY, fontsize=10,
                       transform=ax_eq.transAxes)
            ax_eq.axhline(ic, color=GOL, lw=1.5, ls='--', alpha=0.7)

        ax_eq.set_ylabel('USDT', color=GRY, fontsize=8)
        ax_eq.set_xlabel('Тик мониторинга', color=GRY, fontsize=8)

        # Метка режима на графике
        ax_eq.text(0.01, 0.97, rlbl, transform=ax_eq.transAxes,
                   color=rclr, fontsize=12, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', fc=rbg, ec=rclr, lw=1.5))

        # ── ROW 0 right [3]: Статус системы ─────────────────────────────────
        ax_st = fig.add_subplot(layout[0, 3])
        ax_st.set_facecolor(CARD)
        ax_st.axis('off')
        ax_st.text(0.5, 0.97, "Статус системы", ha='center', va='top',
                   color=WHT, fontsize=9, fontweight='bold',
                   transform=ax_st.transAxes)
        has_perm = getattr(stats, 'has_futures_perm', True)
        cur_bal  = eq[-1] if eq else getattr(stats, 'current_balance', ic)  # equity
        # Текущий unrealPnL и свободная маржа из кривых
        cur_unreal   = unreal_hist[-1] if unreal_hist else 0.0
        cur_avail    = bal[-1]         if bal         else cur_bal
        # Капитал = equity - unrealPnL (реализованный баланс)
        cur_realized = cur_bal - cur_unreal
        pnl_now      = cur_bal - ic

        # Считаем "Всего активов": equity уже включает unrealPnL
        # Дополнительно: спот (если есть)
        spot_val     = getattr(stats, 'spot_total', 0.0)
        total_assets = cur_bal + spot_val

        status_rows = [
            ("Статус",      "🟢 LIVE" if live_bars > 0 else "🟡 WARMUP",
             GRN if live_bars > 0 else GOL),
            ("Баланс",      f"${cur_bal:.4f}",      CYN),
            ("Капитал",     f"${cur_realized:.4f}",  WHT),
            ("Live баров",  str(live_bars),           WHT),
            ("Сигналов",    str(len(signals)),        PUR),
            ("Сделок",      str(len(trades)),         GRN if trades else GRY),
            ("Позиций",     str(len(positions)),      CYN if positions else GRY),
            ("Ордеров ок",  str(getattr(stats, 'orders_ok', 0)),
             GRN if getattr(stats, 'orders_ok', 0) > 0 else GRY),
            ("Ошибок API",  str(getattr(stats, 'orders_fail', 0)),
             RED if getattr(stats, 'orders_fail', 0) > 0 else GRY),
            ("API Futures", "ДА" if has_perm else "НЕТ",
             GRN if has_perm else RED),
        ]
        y0 = 0.88
        for lbl, val, vc in status_rows:
            ax_st.text(0.05, y0, lbl + ':', color=GRY, fontsize=7.5,
                       transform=ax_st.transAxes, va='top')
            ax_st.text(0.95, y0, val, color=vc, fontsize=7.5, fontweight='bold',
                       transform=ax_st.transAxes, va='top', ha='right')
            ax_st.plot([0.02, 0.98], [y0 - 0.01] * 2, color=GRID, lw=0.4,
                       transform=ax_st.transAxes)
            y0 -= 0.082

        # ── Блок «Все активы» внизу статус-панели ───────────────────────────
        y0 -= 0.02
        ax_st.text(0.5, y0, "─── Все активы ───", ha='center', color=GRY,
                   fontsize=7, transform=ax_st.transAxes, va='top')
        y0 -= 0.07

        assets_rows = [
            ("Equity (фьюч)", f"${cur_bal:.4f}",    CYN),
            ("  unrealPnL",   f"{cur_unreal:+.4f}",
             GRN if cur_unreal >= 0 else RED),
            ("  Свободно",    f"${cur_avail:.4f}",   WHT),
            ("  Залог",       f"${cur_bal - cur_avail - cur_unreal:.4f}", GRY),
            ("Спот",          f"${spot_val:.4f}",    WHT if spot_val > 0 else GRY),
            ("ИТОГО",         f"${total_assets:.4f}", GRN if total_assets >= ic else RED),
        ]
        for lbl, val, vc in assets_rows:
            ax_st.text(0.05, y0, lbl + ':', color=GRY, fontsize=7,
                       transform=ax_st.transAxes, va='top')
            ax_st.text(0.95, y0, val, color=vc, fontsize=7, fontweight='bold',
                       transform=ax_st.transAxes, va='top', ha='right')
            y0 -= 0.072

        # ── ROW 1 left [0:2]: Sub-agent PnL waterfall ───────────────────────
        ax_ag = fig.add_subplot(layout[1, :2])
        ax_ag.set_facecolor(MID)
        for sp in ax_ag.spines.values(): sp.set_color(GRID)
        ax_ag.tick_params(colors=GRY, labelsize=7)
        ax_ag.grid(True, color=GRID, lw=0.4, alpha=0.5, axis='x')
        ax_ag.set_title("Вклад суб-агентов в P&L (virtual PV)",
                        color=WHT, fontsize=9, fontweight='bold')

        # Строим из sub_pvs или sub_agent_pnl
        pnl_data, sub_initials = _resolve_sub_agent_pnl_and_initials(
            stats, sub_agent_pnl=sub_agent_pnl
        )

        if pnl_data:
            sorted_agents_pnl = sorted(pnl_data.items(),
                                        key=lambda kv: kv[1], reverse=True)
            names_a  = [k.replace('STP_', '') for k, _ in sorted_agents_pnl]
            values_a = [v for _, v in sorted_agents_pnl]
            colors_a = [GRN if v >= 0 else RED for v in values_a]
            y_pos    = np.arange(len(names_a))
            bars     = ax_ag.barh(y_pos, values_a, color=colors_a,
                                   alpha=0.85, height=0.65)
            ax_ag.set_yticks(y_pos)
            ax_ag.set_yticklabels(names_a, color=WHT, fontsize=8)
            ax_ag.axvline(0, color=WHT, lw=0.8, alpha=0.5)
            ax_ag.set_xlabel('P&L USDT', color=GRY, fontsize=8)
            for bar_i, (bar, val) in enumerate(zip(bars, values_a)):
                if abs(val) > 0.001:
                    ax_ag.text(
                        val + (0.01 if val >= 0 else -0.01),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.4f}",
                        va='center', ha='left' if val >= 0 else 'right',
                        fontsize=7, color=WHT, fontweight='bold',
                    )
        else:
            ax_ag.text(0.5, 0.5, "Нет данных\n(накапливается в процессе торговли)",
                       ha='center', va='center', color=GRY, fontsize=9,
                       transform=ax_ag.transAxes)

        # ── ROW 1 col [2]: Funding rate + режим ─────────────────────────────
        ax_fund = fig.add_subplot(layout[1, 2])
        _style_ax(ax_fund, "Funding rate + тип рынка")

        if len(fund_hist) > 1:
            fvals = [f * 100 for f in fund_hist[-80:]]
            fcols = [GRN if v < 0 else RED for v in fvals]
            # Отрицательный funding = шорты платят лонгам = bullish bias
            ax_fund.bar(range(len(fvals)), fvals, color=fcols,
                        width=0.8, alpha=0.82)
            ax_fund.axhline(0, color=WHT, lw=0.6, alpha=0.4)
            ax_fund.set_ylabel('%', color=GRY, fontsize=7)

            # Режим как цветная полоса
            ax_fund.fill_betweenx(
                [min(fvals) * 1.1, max(fvals) * 1.1],
                0, len(fvals),
                color=rclr, alpha=0.06,
            )
            ax_fund.text(
                0.02, 0.96,
                f"Avg: {funding_avg*100:+.4f}%\n{rlbl}",
                transform=ax_fund.transAxes, color=rclr, fontsize=8,
                va='top', fontweight='bold',
                bbox=dict(fc=rbg, ec=rclr, lw=1, pad=2),
            )
        else:
            ax_fund.text(0.5, 0.5, "Накопление данных...",
                         ha='center', va='center', color=GRY, fontsize=8,
                         transform=ax_fund.transAxes)
            status_text = "Funding: unsupported" if market_regime != 'unknown' else "Funding: waiting"
            ax_fund.text(0.5, 0.72, status_text,
                         ha='center', va='center', color=GRY, fontsize=8.5,
                         transform=ax_fund.transAxes)
            if abs(float(funding_avg or 0.0)) > 1e-12:
                ax_fund.text(0.5, 0.54, f"{funding_avg*100:+.4f}%",
                             ha='center', va='center', color=CYN, fontsize=10,
                             fontweight='bold', transform=ax_fund.transAxes)
            ax_fund.text(0.5, 0.16, rlbl, ha='center', va='center',
                         color=rclr, fontsize=16, fontweight='bold',
                         transform=ax_fund.transAxes)

        # ── ROW 1 col [3]: Открытые позиции ─────────────────────────────────
        ax_pos = fig.add_subplot(layout[1, 3])
        ax_pos.set_facecolor(CARD)
        ax_pos.axis('off')
        ax_pos.text(0.5, 0.97, "Открытые позиции",
                    ha='center', va='top', color=WHT, fontsize=9,
                    fontweight='bold', transform=ax_pos.transAxes)

        if positions:
            y0 = 0.86
            for sym, pos in list(positions.items())[:6]:
                pnl_v = pos.get('pnl', 0.0)
                vc    = GRN if pnl_v >= 0 else RED
                side  = pos.get('side', '?')
                entry = pos.get('entry', 0.0)
                ax_pos.text(0.04, y0, f"[{side}] {sym}",
                            color=GOL, fontsize=8, fontweight='bold',
                            transform=ax_pos.transAxes, va='top')
                ax_pos.text(0.96, y0, f"{pnl_v:+.4f}",
                            color=vc, fontsize=8, fontweight='bold',
                            transform=ax_pos.transAxes, va='top', ha='right')
                ax_pos.text(0.04, y0 - 0.06,
                            f"  entry: {entry:.5f}",
                            color=GRY, fontsize=7,
                            transform=ax_pos.transAxes, va='top')
                y0 -= 0.145
        else:
            ax_pos.text(0.5, 0.5, "Нет открытых позиций",
                        ha='center', va='center', color=GRY, fontsize=9,
                        transform=ax_pos.transAxes)

        # ── ROW 2 left [0:2]: История ордеров / Сигналы ───────────────────────
        ax_ord = fig.add_subplot(layout[2, :2])
        _style_ax(ax_ord, "Ордера (зелёный=успешно, красный=ошибка)")

        if order_hist:
            # Реальные ордера: зелёный=OK, красный=ошибка
            last_n = order_hist[-50:]
            cols_o = [o.get('color', GRY) for o in last_n]
            ax_ord.bar(range(len(last_n)), [1.0] * len(last_n),
                       color=cols_o, width=0.8, alpha=0.85)
            ax_ord.set_yticks([])
            ax_ord.set_xticks(range(len(last_n)))
            ax_ord.set_xticklabels(
                [o.get('sym', '?')[:5] for o in last_n],
                rotation=70, fontsize=5.5, color=GRY,
            )
            for i, o in enumerate(last_n):
                if o.get('color') == RED and o.get('code'):
                    ax_ord.text(i, 0.5, str(o.get('code', '')),
                                ha='center', va='center', fontsize=5, color=WHT)
            ok_n   = sum(1 for o in order_hist if o.get('color') == GRN)
            fail_n = sum(1 for o in order_hist if o.get('color') == RED)
            ax_ord.text(0.99, 0.96, f"OK {ok_n}  FAIL {fail_n}",
                        transform=ax_ord.transAxes, ha='right', va='top',
                        fontsize=8, color=WHT,
                        bbox=dict(fc=CARD, ec=GRID, pad=2))

        elif signals:
            # Нет ордеров, но есть сигналы — показываем их как timeline
            ax_ord.set_title("Сигналы суб-агентов (ордеров пока нет)",
                             color=WHT, fontsize=9, fontweight='bold')
            ACTION_COLORS = {
                "fl_half":   CYN,  "fl_full":   CYN,
                "fs_half":   ORG,  "fs_full":   ORG,
                "buy_half":  GRN,  "buy_full":  GRN,
                "sell_spot": RED,  "close_fut": RED,
                "hold":      GRY,
            }
            # Берём последние 60 не-HOLD сигналов
            active_sigs = [s for s in signals if s.get('action', 0) != 0][-60:]
            if active_sigs:
                sig_bars  = list(range(len(active_sigs)))
                sig_cols  = [ACTION_COLORS.get(s.get('name','hold'), GRY)
                             for s in active_sigs]
                sig_acts  = [s.get('name', 'hold') for s in active_sigs]
                sig_syms  = [s.get('sym', '?') for s in active_sigs]

                # Группируем по типу действия для понятной визуализации
                ax_ord.bar(sig_bars, [1.0] * len(active_sigs),
                           color=sig_cols, width=0.85, alpha=0.80)
                ax_ord.set_yticks([])

                # Подписи символов под каждым столбцом (только для крупных блоков)
                step = max(1, len(active_sigs) // 20)
                shown_ticks = list(range(0, len(active_sigs), step))
                ax_ord.set_xticks(shown_ticks)
                ax_ord.set_xticklabels(
                    [sig_syms[i][:4] for i in shown_ticks],
                    rotation=70, fontsize=5.5, color=GRY,
                )

                # Легенда по типам действий
                from collections import Counter
                act_counts = Counter(sig_acts)
                legend_parts = []
                for act, clr in ACTION_COLORS.items():
                    if act in act_counts and act != 'hold':
                        legend_parts.append(
                            f"{act.upper()}:{act_counts[act]}"
                        )
                if legend_parts:
                    ax_ord.text(0.01, 0.97, "  ".join(legend_parts),
                                transform=ax_ord.transAxes, ha='left', va='top',
                                fontsize=6.5, color=WHT,
                                bbox=dict(fc=CARD, ec=GRID, lw=0.4, pad=2))

                # Сводка по действиям справа
                long_n  = sum(1 for s in signals if s.get('name','') in ('fl_half','fl_full'))
                short_n = sum(1 for s in signals if s.get('name','') in ('fs_half','fs_full'))
                close_n = sum(1 for s in signals if s.get('name','') in ('close_fut','sell_spot'))
                ax_ord.text(0.99, 0.97,
                            f"LONG:{long_n}  SHORT:{short_n}  CLOSE:{close_n}",
                            transform=ax_ord.transAxes, ha='right', va='top',
                            fontsize=7, color=WHT,
                            bbox=dict(fc=CARD, ec=GRID, lw=0.4, pad=2))
            else:
                ax_ord.text(0.5, 0.5, "Сигналов пока нет\nЖдём следующего сигнала агента",
                            ha='center', va='center', color=GRY, fontsize=9,
                            transform=ax_ord.transAxes)
        elif trades:
            vals  = [t.get('value', 0) for t in trades[-40:]]
            sides = [t.get('side', '') for t in trades[-40:]]
            clrs  = [GRN if s in ('BUY', 'LONG', 'buy') else RED for s in sides]
            ax_ord.bar(range(len(vals)), vals, color=clrs, alpha=0.8, width=0.8)
            ax_ord.set_ylabel('USDT', color=GRY, fontsize=8)
        else:
            ax_ord.text(0.5, 0.5, "Сделок пока нет\nЖдём следующего сигнала агента",
                        ha='center', va='center', color=GRY, fontsize=9,
                        transform=ax_ord.transAxes)

        # ── ROW 2 right [2:4]: Детальная таблица суб-агентов ────────────────
        ax_tbl2 = fig.add_subplot(layout[2, 2:])
        ax_tbl2.set_facecolor(CARD)
        ax_tbl2.axis('off')
        ax_tbl2.text(0.5, 0.97, "Детальная статистика суб-агентов",
                     ha='center', va='top', color=WHT, fontsize=9,
                     fontweight='bold', transform=ax_tbl2.transAxes)

        hdr2_xs  = [0.02, 0.30, 0.48, 0.62, 0.78, 0.90]
        hdr2_lbl = ['Агент', 'Virtual PV', 'P&L', 'P&L %', 'Сделок', 'Статус']

        y0 = 0.88
        for cx, hl in zip(hdr2_xs, hdr2_lbl):
            ax_tbl2.text(cx, y0, hl, transform=ax_tbl2.transAxes,
                         color=GRY, fontsize=7.5, fontweight='bold', va='top')
        ax_tbl2.plot([0.02, 0.98], [y0 - 0.02] * 2, color=GRID, lw=0.6,
                     transform=ax_tbl2.transAxes)

        y0 -= 0.10
        display_agents = sorted(sub_pvs.items(),
                                 key=lambda kv: kv[1], reverse=True) if sub_pvs else []

        for ag_name, pv_val in display_agents[:8]:
            base = float(sub_initials.get(ag_name, ic) or 0.0)
            pnl_v = float(pnl_data.get(ag_name, float(pv_val) - base))
            pnl_p = pnl_v / base * 100 if base > 0 else 0
            vc    = GRN if pnl_v >= 0 else RED
            n_tr  = sub_agent_trades.get(ag_name, []) if sub_agent_trades else []
            short = ag_name.replace('STP_', '')
            ax_tbl2.text(0.02, y0, short, transform=ax_tbl2.transAxes,
                         color=CYN, fontsize=7.5, va='top')
            ax_tbl2.text(0.30, y0, f"${pv_val:.4f}", transform=ax_tbl2.transAxes,
                         color=WHT, fontsize=7.5, va='top')
            ax_tbl2.text(0.48, y0, f"{pnl_v:+.4f}", transform=ax_tbl2.transAxes,
                         color=vc, fontsize=7.5, va='top', fontweight='bold')
            ax_tbl2.text(0.62, y0, _fmt_pct(pnl_p), transform=ax_tbl2.transAxes,
                         color=vc, fontsize=7.5, va='top')
            ax_tbl2.text(0.78, y0, str(len(n_tr)), transform=ax_tbl2.transAxes,
                         color=WHT, fontsize=7.5, va='top')
            status_s = "+" if pnl_v >= 0 else "-"
            ax_tbl2.text(0.90, y0, status_s, transform=ax_tbl2.transAxes,
                         color=vc, fontsize=9, va='top')
            y0 -= 0.100

        if not display_agents:
            ax_tbl2.text(0.5, 0.5, "Нет данных суб-агентов\n(накапливается)",
                         ha='center', va='center', color=GRY, fontsize=9,
                         transform=ax_tbl2.transAxes)

        # ── ROW 3 [0:4]: Последние сигналы ──────────────────────────────────
        ax_sig = fig.add_subplot(layout[3, :])
        ax_sig.set_facecolor(CARD)
        ax_sig.axis('off')
        _agent_name = getattr(stats, 'agent_name', 'Panteon v3')
        ax_sig.text(0.5, 0.97, f"Последние сигналы — {_agent_name}",
                    ha='center', va='top', color=WHT, fontsize=9,
                    fontweight='bold', transform=ax_sig.transAxes)

        ACTION_NAMES = {
            0: "HOLD", 1: "BUY½", 2: "BUY", 3: "SELL",
            4: "FL_HALF", 5: "FL_FULL", 6: "FS_HALF", 7: "FS_FULL", 8: "CLOSE_FUT",
        }
        col_clrs = {
            "fl_half": CYN, "fl_full": CYN, "FL_HALF": CYN, "FL_FULL": CYN,
            "LONG½": CYN,   "LONG": CYN,
            "fs_half": ORG, "fs_full": ORG, "FS_HALF": ORG, "FS_FULL": ORG,
            "SHORT½": ORG,  "SHORT": ORG,
            "buy_half": GRN, "buy_full": GRN, "BUY½": GRN, "BUY": GRN,
            "sell_spot": RED, "SELL": RED, "close_fut": RED, "CLOSE_FUT": RED, "CLOSE": RED,
        }
        sig_xs  = [0.01, 0.09, 0.17, 0.28, 0.41, 0.54, 0.64, 0.76]
        sig_hdr = ["Время", "Bar", "Символ", "Действие", "Цена", "Статус", "Агент", "Режим"]

        y0 = 0.88
        for xi, hdr in zip(sig_xs, sig_hdr):
            ax_sig.text(xi, y0, hdr, transform=ax_sig.transAxes,
                        color=GRY, fontsize=7, fontweight='bold', va='top')
        ax_sig.plot([0.0, 1.0], [0.83] * 2, color=GRID, lw=0.5,
                    transform=ax_sig.transAxes)

        recent_sigs = list(reversed(signals[-20:]))
        y0 = 0.79
        for sig in recent_sigs:
            if y0 < 0.0:
                break
            act_name = sig.get('name', ACTION_NAMES.get(sig.get('action', 0), '?')).upper()
            vc       = col_clrs.get(act_name, col_clrs.get(sig.get('name',''), GRY))
            ok       = sig.get('order_ok')
            ok_s     = "OK" if ok is True else ("FAIL" if ok is False else "—")
            ok_c     = GRN if ok is True else (RED if ok is False else GRY)
            t_s      = sig.get('time', datetime.now()).strftime('%H:%M:%S')
            row_vals = [
                t_s, str(sig.get('bar', '?')), sig.get('sym', '?'),
                act_name, f"${sig.get('price', 0):.4f}",
                ok_s, sig.get('agent', '—'), sig.get('regime', '—'),
            ]
            row_colors = [WHT, GRY, GOL, vc, WHT, ok_c, CYN, rclr]
            for xi, rv, rc in zip(sig_xs, row_vals, row_colors):
                ax_sig.text(xi, y0, rv, transform=ax_sig.transAxes,
                            color=rc, fontsize=7, va='top')
            y0 -= 0.075

        if not signals:
            next_b = max(0, 60 - live_bars)
            ax_sig.text(0.5, 0.5,
                        f"Сигналов пока нет  |  Первая проверка через ~{next_b} мин",
                        ha='center', va='center', color=GRY, fontsize=9,
                        transform=ax_sig.transAxes)

        out = os.path.join(output_dir, filename)
        fig.savefig(out, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        fig.clf()
        plt.close(fig)
        log.info("  [api_trading_enhanced] → %s", out)

    except Exception as e:
        log.warning("  [api_trading_enhanced] failed: %s\n%s",
                    e, traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# 3. MASTER PLAYER DASHBOARD (paper — 6 sub-players)
# ══════════════════════════════════════════════════════════════════════════════

def plot_master_player_dashboard(
    players: Dict[str, Any],       # {name: _SubPlayerBase}
    prices: dict,
    output_dir: str,
    market_regime: str = 'unknown',
    funding_avg: float = 0.0,
    funding_history: Optional[List[float]] = None,
    bar: int = 0,
    warmup_end: int = 0,
    filename: str = "master_player_enhanced.png",
) -> None:
    """
    Расширенный дашборд MasterPlayer v7:

    Макет (3 строки):
      Row 0 [0:4]  — Compound growth кривые (логарит.) всех 6 суб-игроков
      Row 1 [0:2]  — Вклад каждого суб-игрока в итоговый PnL (waterfall)
      Row 1 [2]    — Тип рынка + funding rate
      Row 1 [3]    — Таблица суб-игроков: режим/стоп/позиции
      Row 2 [0:4]  — Детальные позиции по всем суб-игрокам
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.lines import Line2D

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rclr, rbg, rlbl = _regime_info(market_regime)
        live_b = max(0, bar - warmup_end)

        INITIAL_CAPITAL = 100.0
        try:
            sample = next(iter(players.values()))
            INITIAL_CAPITAL = getattr(sample, '_capital', 100.0)
        except Exception:
            pass

        fig = plt.figure(figsize=(24, 16), facecolor=DARK)
        fig.suptitle(
            f"MasterPlayer v7  —  Дашборд  [{now_str}]"
            f"  |  Live bars: {live_b}  |  Режим рынка: {rlbl}"
            f"  |  Funding: {funding_avg*100:+.4f}%",
            fontsize=11, fontweight='bold', color=WHT, y=0.995,
        )

        layout = gridspec.GridSpec(
            3, 4, figure=fig,
            hspace=0.48, wspace=0.28,
            left=0.05, right=0.98, top=0.975, bottom=0.03,
            height_ratios=[2.0, 1.4, 1.0],
        )

        # ── ROW 0 [0:4]: Compound growth curves ─────────────────────────────
        ax_gr = fig.add_subplot(layout[0, :])
        ax_gr.set_facecolor(rbg)
        for sp in ax_gr.spines.values(): sp.set_color(rclr)
        ax_gr.tick_params(colors=GRY, labelsize=7)
        ax_gr.grid(True, color=GRID, lw=0.4, alpha=0.5)
        ax_gr.set_title(
            f"Compound Growth — MasterPlayer суб-игроки | {rlbl}",
            color=WHT, fontsize=10, fontweight='bold',
        )

        legend_handles = []
        player_finals  = {}

        # MasterPlayer total line
        total_hist = []
        for name, player in players.items():
            pv_live = player._pv_history[player._live_start_idx:]
            if not total_hist:
                total_hist = list(pv_live)
            else:
                for i in range(min(len(total_hist), len(pv_live))):
                    pass  # будем суммировать ниже
        # Суммируем по всем игрокам
        total_hist_sum = None
        for name, player in players.items():
            pv_live = player._pv_history[player._live_start_idx:]
            if not pv_live:
                continue
            arr = np.array(pv_live)
            if total_hist_sum is None:
                total_hist_sum = arr.copy()
            else:
                n = min(len(total_hist_sum), len(arr))
                total_hist_sum = total_hist_sum[:n] + arr[:n]

        if total_hist_sum is not None and len(total_hist_sum) > 1:
            base_tot = total_hist_sum[0] if total_hist_sum[0] > 0 else 1.0
            ret_tot  = (total_hist_sum / base_tot - 1) * 100
            clr_tot  = GRN if ret_tot[-1] >= 0 else RED
            ax_gr.fill_between(range(len(ret_tot)), ret_tot, 0,
                               color=clr_tot, alpha=0.08)
            ax_gr.plot(range(len(ret_tot)), ret_tot, lw=2.5, color=clr_tot,
                       zorder=5, label=f"MasterPlayer TOTAL {ret_tot[-1]:+.2f}%")

        # Суб-игроки
        for idx, (name, player) in enumerate(players.items()):
            pv_live = player._pv_history[player._live_start_idx:]
            if len(pv_live) < 2:
                continue
            base_p = pv_live[0] if pv_live[0] > 0 else INITIAL_CAPITAL
            ret_p  = [(v / base_p - 1) * 100 for v in pv_live]
            clr    = PLAYER_COLORS.get(name, AGENT_PALETTE[idx % len(AGENT_PALETTE)])
            player_finals[name] = ret_p[-1]

            ax_gr.plot(range(len(ret_p)), ret_p, lw=1.6, color=clr,
                       alpha=0.9, zorder=3)
            # Аннотация конечного значения
            ax_gr.annotate(
                f"{name}  {ret_p[-1]:+.2f}%",
                xy=(len(ret_p) - 1, ret_p[-1]),
                xytext=(6, 0), textcoords='offset points',
                color=clr, fontsize=8, fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='-', color=clr, lw=0.5, alpha=0.4),
            )
            dd  = _rolling_max_dd(pv_live)
            shr = _sharpe(pv_live)
            legend_handles.append(
                Line2D([0], [0], color=clr, lw=1.6,
                       label=f"{name}  {ret_p[-1]:+.2f}%  "
                             f"DD:{dd:.1f}%  Sharpe:{shr:.2f}  "
                             f"pos={len(player._open_pos)}")
            )

        ax_gr.axhline(0, color=GOL, lw=0.8, ls='--', alpha=0.4)
        ax_gr.text(0.01, 0.97, rlbl, transform=ax_gr.transAxes,
                   color=rclr, fontsize=12, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', fc=rbg, ec=rclr, lw=1.5))

        if abs(funding_avg) > 0.0001:
            ax_gr.text(
                0.99, 0.97,
                f"Funding: {funding_avg*100:+.4f}%",
                transform=ax_gr.transAxes, ha='right', va='top',
                color=GRN if funding_avg < 0 else RED, fontsize=8,
                bbox=dict(fc=CARD, ec=GRID, pad=2),
            )

        ax_gr.legend(handles=legend_handles, loc='lower left', fontsize=7.5,
                     facecolor=CARD, edgecolor=GRID, labelcolor=WHT,
                     framealpha=0.92, ncol=3)
        ax_gr.set_ylabel('Return %', color=GRY, fontsize=8)
        ax_gr.set_xlabel('Bar (live)', color=GRY, fontsize=8)

        # ── ROW 1 left [0:2]: PnL по суб-игрокам (waterfall) ────────────────
        ax_wf = fig.add_subplot(layout[1, :2])
        ax_wf.set_facecolor(MID)
        for sp in ax_wf.spines.values(): sp.set_color(GRID)
        ax_wf.tick_params(colors=GRY, labelsize=7)
        ax_wf.grid(True, color=GRID, lw=0.4, alpha=0.5, axis='x')
        ax_wf.set_title("Вклад суб-игроков в P&L",
                        color=WHT, fontsize=9, fontweight='bold')

        if player_finals:
            sorted_pf = sorted(player_finals.items(),
                                key=lambda kv: kv[1], reverse=True)
            pf_names   = [k for k, _ in sorted_pf]
            pf_vals    = [v for _, v in sorted_pf]
            pf_colors  = [PLAYER_COLORS.get(n, GRY) for n in pf_names]
            pf_bar_clr = [GRN if v >= 0 else RED for v in pf_vals]

            y_pos = np.arange(len(pf_names))
            bars  = ax_wf.barh(y_pos, pf_vals, color=pf_bar_clr,
                                alpha=0.85, height=0.65)
            # Тонкая обводка цветом игрока
            for bar_i, (b, clr) in enumerate(zip(bars, pf_colors)):
                b.set_edgecolor(clr)
                b.set_linewidth(1.5)

            ax_wf.set_yticks(y_pos)
            ax_wf.set_yticklabels(pf_names, color=WHT, fontsize=9,
                                   fontweight='bold')
            ax_wf.axvline(0, color=WHT, lw=0.8, alpha=0.5)
            ax_wf.set_xlabel('Return %', color=GRY, fontsize=8)

            for b, val in zip(bars, pf_vals):
                if abs(val) > 0.01:
                    ax_wf.text(
                        val + (0.05 if val >= 0 else -0.05),
                        b.get_y() + b.get_height() / 2,
                        _fmt_pct(val),
                        va='center', ha='left' if val >= 0 else 'right',
                        fontsize=8, color=WHT, fontweight='bold',
                    )

        # ── ROW 1 col [2]: Funding + режим ──────────────────────────────────
        ax_fund = fig.add_subplot(layout[1, 2])
        _style_ax(ax_fund, "Funding rate + режим рынка")

        fund_h = funding_history or []
        if len(fund_h) > 1:
            fvals = [f * 100 for f in fund_h[-80:]]
            fcols = [GRN if v < 0 else RED for v in fvals]
            ax_fund.bar(range(len(fvals)), fvals, color=fcols,
                        width=0.8, alpha=0.82)
            ax_fund.axhline(0, color=WHT, lw=0.6, alpha=0.4)
            ax_fund.set_ylabel('%', color=GRY, fontsize=7)
        else:
            ax_fund.text(0.5, 0.55, f"Funding:\n{funding_avg*100:+.4f}%",
                         ha='center', va='center', color=CYN, fontsize=12,
                         transform=ax_fund.transAxes, fontweight='bold')

        ax_fund.text(0.5, 0.08, rlbl, ha='center', va='center',
                     color=rclr, fontsize=14, fontweight='bold',
                     transform=ax_fund.transAxes,
                     bbox=dict(boxstyle='round', fc=rbg, ec=rclr, lw=1.5))

        # ── ROW 1 col [3]: Таблица суб-игроков ──────────────────────────────
        ax_tab = fig.add_subplot(layout[1, 3])
        ax_tab.set_facecolor(CARD)
        ax_tab.axis('off')
        ax_tab.text(0.5, 0.97, "Состояние суб-игроков",
                    ha='center', va='top', color=WHT, fontsize=9,
                    fontweight='bold', transform=ax_tab.transAxes)

        y0 = 0.88
        for name, player in players.items():
            try:
                ps    = player.get_status(prices)
                regime_p = ps.get('regime', '?').upper()[:4]
                pv    = ps.get('pv', 0)
                dd    = ps.get('dd_pct', 0)
                npos  = ps.get('n_positions', 0)
                ntr   = ps.get('total_trades', 0)
                day_s = "⛔DAY" if ps.get('day_stopped') else f"d{ps.get('day_dd_pct',0):+.1f}%"
                clr   = PLAYER_COLORS.get(name, GRY)
                vc    = GRN if pv >= INITIAL_CAPITAL else RED
                ax_tab.text(0.02, y0, f"{name[:7]}", transform=ax_tab.transAxes,
                            color=clr, fontsize=8, fontweight='bold', va='top')
                ax_tab.text(0.98, y0,
                            f"${pv:.2f}  {regime_p}  pos={npos}  tr={ntr}\n"
                            f"  DD={dd:.1f}%  {day_s}",
                            transform=ax_tab.transAxes, ha='right',
                            color=vc, fontsize=7, va='top')
                ax_tab.plot([0.02, 0.98], [y0 - 0.01] * 2, color=GRID,
                            lw=0.4, transform=ax_tab.transAxes)
                y0 -= 0.145
            except Exception:
                y0 -= 0.145

        # ── ROW 2 [0:4]: Все открытые позиции ───────────────────────────────
        ax_pos = fig.add_subplot(layout[2, :])
        ax_pos.set_facecolor(CARD)
        ax_pos.axis('off')
        ax_pos.text(0.5, 0.97, "Все открытые позиции (все суб-игроки)",
                    ha='center', va='top', color=WHT, fontsize=9,
                    fontweight='bold', transform=ax_pos.transAxes)

        col_xs2  = [0.01, 0.10, 0.22, 0.33, 0.46, 0.57, 0.68, 0.79, 0.90]
        col_hdr2 = ['Игрок', 'Символ', 'Тип', 'Агент', 'Вход $', 'Тек. $', 'PnL %', 'Держим б.', 'Режим']

        y0 = 0.88
        for cx, hl in zip(col_xs2, col_hdr2):
            ax_pos.text(cx, y0, hl, transform=ax_pos.transAxes,
                        color=GRY, fontsize=7, fontweight='bold', va='top')
        ax_pos.plot([0.01, 0.99], [y0 - 0.025] * 2, color=GRID, lw=0.5,
                    transform=ax_pos.transAxes)

        y0 -= 0.12
        all_pos = []
        for pname, player in players.items():
            try:
                ps = player.get_status(prices)
                for pos in ps.get('positions', []):
                    pos['_player'] = pname
                    all_pos.append(pos)
            except Exception:
                pass

        if all_pos:
            for pos in all_pos[:10]:
                pnl_p = pos.get('pnl_pct', 0)
                vc    = GRN if pnl_p >= 0 else RED
                pname = pos.get('_player', '?')
                pclr  = PLAYER_COLORS.get(pname, GRY)
                tp    = pos.get('type', '?')
                ep    = pos.get('entry_price', 0)
                cp    = prices.get(pos.get('sym', ''), ep)
                held  = pos.get('held_bars', 0)
                rg    = player_finals.get(pname, 0)

                vals2 = [
                    pname[:6], pos.get('sym', '?')[:8],
                    tp[:8], pos.get('agent', '?')[:10],
                    f"${ep:.5f}", f"${cp:.5f}",
                    _fmt_pct(pnl_p), f"{held}б",
                    players.get(pname, None) and
                    getattr(players[pname], '_regime', '?') or '?',
                ]
                vcs2 = [pclr, GOL, CYN, GRY, WHT, WHT, vc, GRY, rclr]

                for cx, val, vc2 in zip(col_xs2, vals2, vcs2):
                    ax_pos.text(cx, y0, str(val), transform=ax_pos.transAxes,
                                color=vc2, fontsize=7, va='top')
                y0 -= 0.085
                if y0 < 0.02:
                    break
        else:
            ax_pos.text(0.5, 0.5, "Нет открытых позиций",
                        ha='center', va='center', color=GRY, fontsize=10,
                        transform=ax_pos.transAxes)

        out = os.path.join(output_dir, filename)
        fig.savefig(out, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        fig.clf()
        plt.close(fig)
        log.info("  [master_player_enhanced] → %s", out)

    except Exception as e:
        log.warning("  [master_player_enhanced] failed: %s\n%s",
                    e, traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# ИНТЕГРАЦИЯ: хелперы для вызова из bridge._patched_cycle
# ══════════════════════════════════════════════════════════════════════════════

def get_regime_from_bridge(bridge) -> str:
    """Извлекает текущий режим рынка из bridge или paper portfolios."""
    try:
        # Пробуем получить из истории цен
        from crypto_agents import _detect_regime_live, _r3
        ph = {}
        for snap in (bridge._price_hist or [])[-200:]:
            for s, p in snap.items():
                if s not in ph:
                    ph[s] = []
                ph[s].append(float(p))
        if ph:
            raw = _detect_regime_live(
                {s: __import__('collections').deque(v, maxlen=len(v))
                 for s, v in ph.items()})
            return {'bullish': 'bull', 'bearish': 'bear', 'neutral': 'sideways'}.get(
                _r3(raw), 'sideways')
    except Exception:
        pass
    return 'unknown'


def get_funding_from_bridge(bridge) -> tuple:
    """Возвращает (avg_rate, history_list) из bridge."""
    try:
        if bridge.funding:
            g = bridge.funding.get_global()
            if g:
                avg  = g.get('avg_funding', 0.0)
                hist = getattr(bridge, '_funding_hist', [])
                return avg, hist
    except Exception:
        pass
    return 0.0, []


def inject_enhanced_dashboards(bridge, players_dict=None) -> None:
    """
    Патч для bridge._patched_cycle.
    Вызывать внутри патча после стандартных дашбордов:

        from mexc_dashboards import inject_enhanced_dashboards
        inject_enhanced_dashboards(bridge, players_dict=players)
    """
    try:
        prices, _ = bridge._fetch_market()
        regime    = get_regime_from_bridge(bridge)
        favg, fh  = get_funding_from_bridge(bridge)

        # 1. Compound growth enhanced (paper portfolios)
        if hasattr(bridge, 'paper_pf') and bridge.paper_pf:
            plot_compound_growth_enhanced(
                portfolios=bridge.paper_pf,
                output_dir=bridge.output_dir,
                warmup_end=bridge._warmup_end,
                market_regime=regime,
                funding_avg=favg,
            )

        # 2. API trading dashboard (если есть stats)
        stats = getattr(bridge, '_stats_ref', None)
        if stats is None:
            # Пробуем через обёртку
            for attr in ('_agent_wrapper', '_wrapper'):
                wrapper = getattr(bridge, attr, None)
                if wrapper:
                    stats = getattr(wrapper, '_stats', None)
                    break
        if stats is not None:
            sub_pnl, _ = _resolve_sub_agent_pnl_and_initials(stats)
            plot_api_trading_dashboard(
                stats=stats,
                output_dir=bridge.output_dir,
                market_regime=regime,
                funding_avg=favg,
                funding_history=fh,
                sub_agent_pnl=sub_pnl,
            )

        # 3. MasterPlayer dashboard
        if players_dict:
            plot_master_player_dashboard(
                players=players_dict,
                prices=prices,
                output_dir=bridge.output_dir,
                market_regime=regime,
                funding_avg=favg,
                funding_history=fh,
                bar=bridge._bar,
                warmup_end=bridge._warmup_end,
            )

    except Exception as e:
        log.warning("  [inject_enhanced_dashboards] failed: %s\n%s",
                    e, traceback.format_exc())

