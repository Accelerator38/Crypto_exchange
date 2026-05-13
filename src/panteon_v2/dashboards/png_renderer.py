"""Matplotlib PNG backend for operator-facing Panteon v2 dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


DARK = "#0D1117"
MID = "#161B22"
GRID = "#30363D"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
GREEN = "#3FB950"
RED = "#F85149"
BLUE = "#58A6FF"
GOLD = "#F0C040"
PURPLE = "#BC8CFF"


def write_operator_pngs(
    output_dir: str,
    *,
    status: Mapping[str, object] | None,
    agents_payload: Mapping[str, object] | None,
    players_payload: Mapping[str, object] | None,
) -> List[str]:
    """Write all v2 visual dashboards and return their file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_map = dict(status or {})
    agents = _entries(agents_payload, "agents")
    players = _entries(players_payload, "players")
    panteon_pnl_pct = _panteon_pnl_pct(status_map)

    paths = [
        out / "dashboard_latest.png",
        out / "shadow_agents_dashboard.png",
        out / "shadow_player_dashboard.png",
        out / "agent_regime_dashboard.png",
        out / "player_regime_dashboard.png",
    ]
    _render_operator_dashboard(
        paths[0],
        status_map,
        agents,
        players,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _render_leaderboard(
        paths[1],
        "SHADOW AGENTS DASHBOARD",
        agents,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _render_leaderboard(
        paths[2],
        "SHADOW PLAYERS DASHBOARD",
        players,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _render_regime_heatmap(paths[3], "AGENT REGIME DASHBOARD", agents)
    _render_regime_heatmap(paths[4], "PLAYER REGIME DASHBOARD", players)
    return [str(p) for p in paths]


def _entries(payload: Mapping[str, object] | None, key: str) -> List[Tuple[str, dict]]:
    raw = (payload or {}).get(key) if isinstance(payload, Mapping) else {}
    if not isinstance(raw, Mapping):
        return []
    rows: List[Tuple[str, dict]] = []
    for name, values in raw.items():
        if isinstance(values, Mapping):
            rows.append((str(name).replace("V_", "", 1), dict(values)))
    rows.sort(key=lambda item: float(item[1].get("pnl_pct", 0.0) or 0.0), reverse=True)
    return rows


def _setup_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    return plt


def _style_ax(ax, title: str = "") -> None:
    ax.set_facecolor(MID)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.55)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)


def _render_operator_dashboard(
    path: Path,
    status: Mapping[str, object],
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
    *,
    panteon_pnl_pct: float,
) -> None:
    plt = _setup_pyplot()
    fig = plt.figure(figsize=(14, 8), facecolor=DARK)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], height_ratios=[1, 1])

    ax_status = fig.add_subplot(gs[:, 0])
    ax_agents = fig.add_subplot(gs[0, 1:])
    ax_players = fig.add_subplot(gs[1, 1:])
    for ax in (ax_status, ax_agents, ax_players):
        _style_ax(ax)

    _draw_status_panel(ax_status, status)
    _draw_barh(ax_agents, agents[:10], "Top Shadow Agents", BLUE,
               panteon_pnl_pct=panteon_pnl_pct)
    _draw_barh(ax_players, players[:10], "Top Shadow Players", PURPLE,
               panteon_pnl_pct=panteon_pnl_pct)

    fig.suptitle("Panteon v2 Operator Dashboard", color=TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_status_panel(ax, status: Mapping[str, object]) -> None:
    ax.set_axis_off()
    ax.set_facecolor(MID)
    lines = [
        ("Exchange", status.get("exchange", "-")),
        ("State", f"{status.get('run_state', '-')}/{status.get('feed_status', '-')}"),
        ("Regime", status.get("regime", "-")),
        ("Bar", status.get("bar_count", 0)),
        ("Leader", status.get("current_leader", "-") or "-"),
        ("Balance", _usd(status.get("current_balance", 0.0))),
        ("Futures equity", _usd(status.get("futures_equity_usd", status.get("current_balance", 0.0)))),
        ("Available", _usd(status.get("available_balance_usd", 0.0))),
        ("Spot assets", _usd(status.get("spot_assets_usd", 0.0))),
        ("Total assets", _usd(status.get("total_assets_usd", status.get("current_balance", 0.0)))),
        ("PnL", f"{_usd(status.get('pnl_usd', 0.0))} / {_pct(status.get('pnl_pct', 0.0))}"),
        ("Positions", status.get("n_positions", 0)),
    ]
    shadow = status.get("shadow") if isinstance(status.get("shadow"), Mapping) else {}
    if isinstance(shadow, Mapping):
        lines.extend([
            ("Shadow actors", shadow.get("actors", 0)),
            ("Shadow signals", shadow.get("signals", 0)),
            ("Shadow filled", shadow.get("filled", 0)),
        ])

    y = 0.95
    ax.text(0.03, y, "ACCOUNT / RUNTIME", transform=ax.transAxes,
            color=TEXT, fontsize=12, fontweight="bold", va="top")
    y -= 0.07
    for label, value in lines:
        ax.text(0.04, y, str(label), transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
        ax.text(0.54, y, str(value), transform=ax.transAxes, color=TEXT, fontsize=9, va="top")
        y -= 0.055


def _draw_barh(
    ax,
    rows: List[Tuple[str, dict]],
    title: str,
    color: str,
    *,
    panteon_pnl_pct: float,
) -> None:
    _style_ax(ax, title)
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    labels = [_short(name) for name, _ in rows]
    values = [float(data.get("pnl_pct", 0.0) or 0.0) for _, data in rows]
    colors = [GREEN if value >= 0 else RED for value in values]
    if all(abs(value) < 1e-12 for value in values):
        colors = [color for _ in values]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.86)
    ax.axvline(0, color=GRID, linewidth=0.8)
    _draw_panteon_benchmark(ax, values, panteon_pnl_pct)
    ax.set_xlabel("PnL %", color=MUTED, fontsize=8)
    for idx, value in enumerate(values[::-1]):
        ax.text(value, idx, f" {_pct(value)}", color=TEXT, va="center", fontsize=8)


def _render_leaderboard(
    path: Path,
    title: str,
    rows: List[Tuple[str, dict]],
    *,
    panteon_pnl_pct: float,
) -> None:
    plt = _setup_pyplot()
    height = max(5.0, min(12.0, 2.2 + len(rows[:18]) * 0.36))
    fig, ax = plt.subplots(figsize=(12, height), facecolor=DARK)
    _draw_barh(ax, rows[:18], title, BLUE, panteon_pnl_pct=panteon_pnl_pct)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_regime_heatmap(path: Path, title: str, rows: List[Tuple[str, dict]]) -> None:
    plt = _setup_pyplot()
    regimes = ["bullish", "bearish", "neutral", "crash"]
    usable = [(name, data) for name, data in rows if isinstance(data.get("per_regime"), Mapping)][:16]
    fig_h = max(4.5, 2.0 + len(usable) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor=DARK)
    ax.set_facecolor(MID)
    if not usable:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No regime data yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=12)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
        fig.tight_layout()
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    matrix = []
    labels = []
    for name, data in usable:
        labels.append(_short(name, 22))
        per_regime = data.get("per_regime") if isinstance(data.get("per_regime"), Mapping) else {}
        matrix.append([
            float((per_regime.get(regime) or {}).get("pnl_pct", 0.0) or 0.0)
            if isinstance(per_regime.get(regime), Mapping) else 0.0
            for regime in regimes
        ])

    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=5)
    ax.set_xticks(range(len(regimes)), labels=regimes, color=MUTED)
    ax.set_yticks(range(len(labels)), labels=labels, color=MUTED)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, _pct(value), ha="center", va="center", color="#111111", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _usd(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _panteon_pnl_pct(status: Mapping[str, object]) -> float:
    try:
        return float(status.get("pnl_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        pass
    try:
        initial = float(status.get("initial_capital", 0.0) or 0.0)
        total = float(status.get("total_assets_usd", status.get("current_balance", 0.0)) or 0.0)
        if initial > 0:
            return (total - initial) / initial * 100.0
    except (TypeError, ValueError):
        pass
    return 0.0


def _draw_panteon_benchmark(ax, shadow_values: Iterable[float], panteon_pnl_pct: float) -> None:
    values = list(shadow_values)
    all_values = values + [panteon_pnl_pct, 0.0]
    min_x = min(all_values)
    max_x = max(all_values)
    span = max(max_x - min_x, 1.0)
    pad = span * 0.16
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.axvline(
        panteon_pnl_pct,
        color=GOLD,
        linestyle=(0, (4, 3)),
        linewidth=2.2,
        label=f"Panteon {_pct(panteon_pnl_pct)}",
        zorder=5,
    )
    ax.text(
        panteon_pnl_pct,
        0.98,
        f" Panteon {_pct(panteon_pnl_pct)}",
        transform=ax.get_xaxis_transform(),
        color=GOLD,
        fontsize=8,
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT, fontsize=8)


def _pct(value: object) -> str:
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "+0.00%"


def _short(value: str, max_len: int = 18) -> str:
    value = str(value)
    return value if len(value) <= max_len else value[: max_len - 1] + "."
