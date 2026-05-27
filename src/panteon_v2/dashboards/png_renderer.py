"""Matplotlib PNG backend for operator-facing Panteon v2 dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


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
INACTIVE_STATUSES = frozenset({"quarantine", "shadow_only", "purgatory"})


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
    memory_agents = _entries(agents_payload, "agents", pnl_source="memory")
    memory_players = _entries(players_payload, "players", pnl_source="memory")
    panteon_pnl_pct = _panteon_pnl_pct(status_map)

    paths = [
        out / "dashboard_latest.png",
        out / "shadow_dashboard.png",
        out / "regime_dashboard.png",
        out / "memory_dashboard.png",
    ]
    _render_operator_dashboard(
        paths[0],
        status_map,
        agents,
        players,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _render_combined_shadow_dashboard(
        paths[1],
        agents,
        players,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _render_combined_regime_dashboard(paths[2], memory_agents, memory_players)
    _render_memory_dashboard(paths[3], memory_agents, memory_players)
    return [str(p) for p in paths]


def _entries(
    payload: Mapping[str, object] | None,
    key: str,
    *,
    pnl_source: str = "session",
) -> List[Tuple[str, dict]]:
    raw = (payload or {}).get(key) if isinstance(payload, Mapping) else {}
    if not isinstance(raw, Mapping):
        return []
    if pnl_source not in {"session", "memory"}:
        raise ValueError(f"unsupported pnl_source: {pnl_source!r}")
    rows: List[Tuple[str, dict]] = []
    for name, values in raw.items():
        if isinstance(values, Mapping):
            row = dict(values)
            row["memory_pnl_pct"] = _float_value(row.get("pnl_pct", 0.0))
            row["session_display_pnl_pct"] = _float_value(
                row.get("session_pnl_pct", row.get("pnl_pct", 0.0))
            )
            if pnl_source == "memory":
                row["display_pnl_pct"] = row["memory_pnl_pct"]
                row["display_pnl_source"] = "memory"
            else:
                row["display_pnl_pct"] = row["session_display_pnl_pct"]
                row["display_pnl_source"] = "session"
            rows.append((str(name).replace("V_", "", 1), row))
    rows.sort(key=lambda item: float(item[1].get("display_pnl_pct", 0.0) or 0.0), reverse=True)
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
    fig = plt.figure(figsize=(14, 9.5), facecolor=DARK)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.05, 1, 1],
        height_ratios=[1, 1, 0.82],
        hspace=0.30,
        wspace=0.20,
    )

    ax_status = fig.add_subplot(gs[:, 0])
    ax_agents = fig.add_subplot(gs[0, 1:])
    ax_players = fig.add_subplot(gs[1, 1:])
    ax_flash_selected = fig.add_subplot(gs[2, 1])
    ax_flash_candidates = fig.add_subplot(gs[2, 2])
    for ax in (ax_status, ax_agents, ax_players, ax_flash_selected, ax_flash_candidates):
        _style_ax(ax)

    _draw_status_panel(ax_status, status)
    _draw_barh(ax_agents, agents[:10], "Top Shadow Agents", BLUE,
               panteon_pnl_pct=panteon_pnl_pct)
    _draw_barh(ax_players, players[:10], "Top Shadow Players", PURPLE,
               panteon_pnl_pct=panteon_pnl_pct)
    _draw_flash_selected_panel(ax_flash_selected, _flash_selected_rows(status))
    _draw_flash_candidates_panel(ax_flash_candidates, _flash_candidate_rows(status))

    fig.suptitle("Panteon v2 Operator Dashboard", color=TEXT, fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.97)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_status_panel(ax, status: Mapping[str, object]) -> None:
    ax.set_axis_off()
    ax.set_facecolor(MID)
    live_session = status.get("live_session") if isinstance(status.get("live_session"), Mapping) else {}
    pnl_usd = (
        live_session.get("panteon_owned_total_pnl_usd")
        if isinstance(live_session, Mapping)
        else None
    )
    if pnl_usd is None:
        pnl_usd = status.get("pnl_usd", 0.0)
    panteon_pct = (
        live_session.get("panteon_owned_pnl_pct")
        if isinstance(live_session, Mapping)
        else status.get("pnl_pct", 0.0)
    )
    lines = [
        ("Exchange", status.get("exchange", "-")),
        ("State", f"{status.get('run_state', '-')}/{status.get('feed_status', '-')}"),
        ("Regime", status.get("regime", "-")),
        ("Bar", status.get("bar_count", 0)),
        ("Selected", status.get("selected_leader", status.get("current_leader", "-")) or "-"),
        ("Executed", status.get("executed_leader", status.get("current_leader", "-")) or "-"),
        ("Balance", _usd(status.get("current_balance", 0.0))),
        ("Futures equity", _usd(status.get("futures_equity_usd", status.get("current_balance", 0.0)))),
        ("Available", _usd(status.get("available_balance_usd", 0.0))),
        ("Spot assets", _usd(status.get("spot_assets_usd", 0.0))),
        ("Total assets", _usd(status.get("total_assets_usd", status.get("current_balance", 0.0)))),
        ("Panteon PnL", f"{_usd(pnl_usd)} / {_pct(panteon_pct)}"),
        ("Panteon DD", _pct(status.get("panteon_max_drawdown_pct", 0.0))),
        ("Account PnL", _pct(status.get("pnl_pct", 0.0))),
        ("Positions", status.get("n_positions", 0)),
    ]
    real_trades = status.get("real_trades") if isinstance(status.get("real_trades"), Mapping) else {}
    if isinstance(real_trades, Mapping):
        total = int(_float_value(real_trades.get("total", 0)))
        closed = int(_float_value(real_trades.get("closed", 0)))
        successful = int(_float_value(real_trades.get("successful", 0)))
        unsuccessful = int(_float_value(real_trades.get("unsuccessful", 0)))
        unresolved = int(_float_value(real_trades.get("unresolved", 0)))
        lines.extend([
            ("Real trades", f"{total} (closed {closed}, open {unresolved})"),
            ("Real W/L/Open", f"{successful} / {unsuccessful} / {unresolved}"),
        ])
    decision_debug = status.get("decision_debug") if isinstance(status.get("decision_debug"), Mapping) else {}
    if isinstance(decision_debug, Mapping):
        fallback_session = (
            decision_debug.get("fallback_session")
            if isinstance(decision_debug.get("fallback_session"), Mapping)
            else {}
        )
        if decision_debug.get("fallback_used") or decision_debug.get("fallback_skipped") or fallback_session:
            used = int(_float_value(fallback_session.get("used", 0))) if isinstance(fallback_session, Mapping) else 0
            skipped = int(_float_value(fallback_session.get("skipped", 0))) if isinstance(fallback_session, Mapping) else 0
            lines.append(("Fallback", f"used {used}, skipped {skipped}"))
    shadow = status.get("shadow") if isinstance(status.get("shadow"), Mapping) else {}
    if isinstance(shadow, Mapping):
        lines.extend([
            ("Shadow actors", shadow.get("actors", 0)),
            ("Shadow signals", _shadow_counter(shadow, "signals")),
            ("Shadow filled", _shadow_counter(shadow, "filled")),
        ])

    y = 0.95
    ax.text(0.03, y, "ACCOUNT / RUNTIME", transform=ax.transAxes,
            color=TEXT, fontsize=12, fontweight="bold", va="top")
    y -= 0.07
    line_step = min(0.055, 0.86 / max(len(lines), 1))
    for label, value in lines:
        ax.text(0.04, y, str(label), transform=ax.transAxes, color=MUTED, fontsize=9, va="top")
        ax.text(0.54, y, str(value), transform=ax.transAxes, color=TEXT, fontsize=9, va="top")
        y -= line_step


def _draw_barh(
    ax,
    rows: List[Tuple[str, dict]],
    title: str,
    color: str,
    *,
    panteon_pnl_pct: float,
    x_label: str | None = None,
    benchmark: bool = True,
) -> None:
    _style_ax(ax, title)
    rows, virtual_panteon_pnl_pct = _split_virtual_panteon(rows)
    if not rows and virtual_panteon_pnl_pct is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    labels = [
        _short(name) + (" [Q]" if _is_inactive_status(data.get("status")) else "")
        for name, data in rows
    ]
    values = [_row_pnl_value(data) for _, data in rows]
    colors = _bar_colors_for_rows(rows, fallback_color=color)
    if rows:
        ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.86)
    ax.axvline(0, color=GRID, linewidth=0.8)
    if benchmark:
        _draw_panteon_benchmark(
            ax,
            values,
            panteon_pnl_pct,
            virtual_panteon_pnl_pct=virtual_panteon_pnl_pct,
        )
    source = next(
        (str(data.get("display_pnl_source")) for _, data in rows
         if data.get("display_pnl_source")),
        "",
    )
    default_label = "Session PnL %" if source == "session" else "Memory PnL %"
    ax.set_xlabel(x_label or default_label, color=MUTED, fontsize=8)
    for idx, value in enumerate(values[::-1]):
        ax.text(value, idx, f" {_pct(value)}", color=TEXT, va="center", fontsize=8)


def _flash_selected_rows(status: Mapping[str, object]) -> List[dict]:
    flash = status.get("flash") if isinstance(status.get("flash"), Mapping) else {}
    selected = flash.get("selected_actors_by_symbol") if isinstance(flash, Mapping) else {}
    by_symbol: Dict[str, str] = {}
    if isinstance(selected, Mapping):
        for raw_symbol, raw_label in selected.items():
            symbol = str(raw_symbol or "").strip().upper()
            label = str(raw_label or "").strip()
            if symbol and label:
                by_symbol[symbol] = label
    if not by_symbol and isinstance(flash, Mapping):
        decisions = flash.get("decisions")
        if isinstance(decisions, (list, tuple)):
            for decision in decisions:
                if not isinstance(decision, Mapping):
                    continue
                symbol = str(decision.get("symbol") or "").strip().upper()
                label = str(decision.get("selected_actor") or "").strip()
                if symbol and label:
                    by_symbol[symbol] = label

    grouped: Dict[str, List[str]] = {}
    for symbol, label in by_symbol.items():
        grouped.setdefault(label, []).append(symbol)
    rows = [
        {
            "label": label,
            "count": len(symbols),
            "symbols": tuple(sorted(symbols)),
        }
        for label, symbols in grouped.items()
    ]
    rows.sort(key=lambda row: (
        row["label"] == "NoTrade",
        -int(row["count"]),
        str(row["label"]),
    ))
    return rows


def _flash_candidate_rows(status: Mapping[str, object]) -> List[dict]:
    flash = status.get("flash") if isinstance(status.get("flash"), Mapping) else {}
    decisions = flash.get("decisions") if isinstance(flash, Mapping) else ()
    if not isinstance(decisions, (list, tuple)):
        return []
    grouped: Dict[Tuple[str, str], dict] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        symbol = str(decision.get("symbol") or "").strip().upper()
        selected_actor = str(decision.get("selected_actor") or "").strip()
        candidates = decision.get("candidates")
        if not isinstance(candidates, (list, tuple)):
            candidates = ()
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            label = str(candidate.get("label") or "").strip()
            actor_type = str(candidate.get("actor_type") or "").strip() or "unknown"
            if not label or label == "NoTrade" or actor_type == "no_trade":
                continue
            key = (label, actor_type)
            row = grouped.setdefault(key, {
                "label": label,
                "actor_type": actor_type,
                "symbols": set(),
                "candidate_count": 0,
                "actionable_count": 0,
                "rejected_count": 0,
                "selected_count": 0,
                "best_score": 0.0,
                "technical_aligned_count": 0,
                "technical_misaligned_count": 0,
                "technical_missing_count": 0,
                "technical_score_adjustment": 0.0,
            })
            if symbol:
                row["symbols"].add(symbol)
            row["candidate_count"] += 1
            score = _float_value(candidate.get("effective_score", candidate.get("score", 0.0)))
            row["best_score"] = max(float(row["best_score"]), score)
            alignment = str(candidate.get("technical_alignment") or "").strip()
            if alignment in {"long_aligned", "short_aligned"}:
                row["technical_aligned_count"] += 1
            elif alignment in {"long_misaligned", "short_misaligned"}:
                row["technical_misaligned_count"] += 1
            elif alignment == "missing":
                row["technical_missing_count"] += 1
            row["technical_score_adjustment"] += _float_value(
                candidate.get("technical_score_adjustment", 0.0)
            )
            rejected = bool(candidate.get("rejected", False))
            if rejected:
                row["rejected_count"] += 1
            action = str(candidate.get("action") or "").strip().upper()
            is_hold = not action or action == "HOLD" or action.endswith(".HOLD")
            if not rejected and not is_hold:
                row["actionable_count"] += 1
            if selected_actor and selected_actor == label:
                row["selected_count"] += 1
    rows = []
    for row in grouped.values():
        out = dict(row)
        out["symbols"] = tuple(sorted(out["symbols"]))
        rows.append(out)
    rows.sort(key=lambda row: (
        -int(row["selected_count"]),
        -int(row["actionable_count"]),
        -float(row["best_score"]),
        -int(row["candidate_count"]),
        str(row["label"]),
    ))
    return rows


def _draw_flash_selected_panel(ax, rows: List[dict]) -> None:
    _style_ax(ax, "Flash Selected Actors")
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No Flash decisions yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
        return
    usable = rows[:8]
    labels = [_short(str(row["label"]), 20) for row in usable]
    values = [int(row["count"]) for row in usable]
    colors = [MUTED if row["label"] == "NoTrade" else GREEN for row in usable]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.86)
    ax.set_xlabel("Selected symbols", color=MUTED, fontsize=8)
    ax.set_xlim(0, max(values) * 1.25 if values else 1)
    for idx, row in enumerate(usable[::-1]):
        symbols = ", ".join(row["symbols"][:4])
        if len(row["symbols"]) > 4:
            symbols += ", ..."
        ax.text(
            int(row["count"]),
            idx,
            f" {int(row['count'])} {symbols}",
            color=TEXT,
            va="center",
            fontsize=7,
        )


def _draw_flash_candidates_panel(ax, rows: List[dict]) -> None:
    _style_ax(ax, "Flash Top Candidates")
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No Flash candidates yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
        return
    usable = rows[:8]
    values = [int(row["actionable_count"]) for row in usable]
    if not any(values):
        values = [int(row["candidate_count"]) for row in usable]
        x_label = "Candidate symbols"
    else:
        x_label = "Actionable symbols"
    labels = [_short(str(row["label"]), 20) for row in usable]
    colors = [
        GREEN if int(row["actionable_count"]) > 0 else
        RED if int(row["rejected_count"]) > 0 else
        BLUE
        for row in usable
    ]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], alpha=0.86)
    ax.set_xlabel(x_label, color=MUTED, fontsize=8)
    ax.set_xlim(0, max(values) * 1.30 if values else 1)
    for idx, row in enumerate(usable[::-1]):
        suffix = (
            f" a{int(row['actionable_count'])}/r{int(row['rejected_count'])}"
            f" score {float(row['best_score']):+.2f}"
        )
        tech_total = (
            int(row.get("technical_aligned_count", 0))
            + int(row.get("technical_misaligned_count", 0))
            + int(row.get("technical_missing_count", 0))
        )
        if tech_total:
            suffix += (
                f" tech {int(row.get('technical_aligned_count', 0))}/"
                f"{int(row.get('technical_misaligned_count', 0))}"
            )
        ax.text(values[::-1][idx], idx, suffix, color=TEXT, va="center", fontsize=7)


def _is_inactive_status(status: object) -> bool:
    return str(status or "").strip().lower() in INACTIVE_STATUSES


def _is_virtual_panteon_name(name: object) -> bool:
    text = str(name or "").strip()
    if text.startswith("V_"):
        text = text[2:]
    return text == "Virtual_Panteon"


def _row_pnl_value(data: Mapping[str, object]) -> float:
    return _float_value(data.get("display_pnl_pct", data.get("pnl_pct", 0.0)))


def _split_virtual_panteon(
    rows: List[Tuple[str, dict]],
) -> Tuple[List[Tuple[str, dict]], Optional[float]]:
    bar_rows: List[Tuple[str, dict]] = []
    virtual_panteon_pnl_pct: Optional[float] = None
    for name, data in rows:
        if _is_virtual_panteon_name(name):
            virtual_panteon_pnl_pct = _row_pnl_value(data)
            continue
        bar_rows.append((name, data))
    return bar_rows, virtual_panteon_pnl_pct


def _bar_colors_for_rows(rows: List[Tuple[str, dict]], *, fallback_color: str) -> List[str]:
    values = [_row_pnl_value(data) for _, data in rows]
    all_flat = bool(values) and all(abs(value) < 1e-12 for value in values)
    colors = []
    for value, (_, data) in zip(values, rows):
        if _is_inactive_status(data.get("status")):
            colors.append(MUTED)
        elif all_flat:
            colors.append(fallback_color)
        elif value > 0:
            colors.append(GREEN)
        elif value < 0:
            colors.append(RED)
        else:
            colors.append(MUTED)
    return colors


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


def _render_combined_shadow_dashboard(
    path: Path,
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
    *,
    panteon_pnl_pct: float,
) -> None:
    plt = _setup_pyplot()
    fig = plt.figure(figsize=(14, 9), facecolor=DARK)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.36)
    ax_players = fig.add_subplot(gs[0, 0])
    ax_agents = fig.add_subplot(gs[1, 0])
    _draw_barh(
        ax_players,
        players[:16],
        "Shadow Players",
        PURPLE,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _draw_barh(
        ax_agents,
        agents[:16],
        "Shadow Agents",
        BLUE,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    fig.suptitle("Shadow Players and Agents", color=TEXT, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.96)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_combined_regime_dashboard(
    path: Path,
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
) -> None:
    plt = _setup_pyplot()
    fig = plt.figure(figsize=(17, 10), facecolor=DARK)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
        height_ratios=[1, 1],
        hspace=0.34,
        wspace=0.22,
    )
    ax_player_heat = fig.add_subplot(gs[0, 0])
    ax_player_curve = fig.add_subplot(gs[0, 1])
    ax_agent_heat = fig.add_subplot(gs[1, 0])
    ax_agent_curve = fig.add_subplot(gs[1, 1])
    player_image = _draw_regime_heatmap(ax_player_heat, "Player Regime Efficiency", players)
    agent_image = _draw_regime_heatmap(ax_agent_heat, "Agent Regime Efficiency", agents)
    _draw_equity_curves(ax_player_curve, "Player Equity Curves", players, PURPLE)
    _draw_equity_curves(ax_agent_curve, "Agent Equity Curves", agents, BLUE)
    for ax, image in ((ax_player_heat, player_image), (ax_agent_heat, agent_image)):
        if image is None:
            continue
        cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Regime Dashboard", color=TEXT, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.96)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_memory_dashboard(
    path: Path,
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
) -> None:
    plt = _setup_pyplot()
    fig = plt.figure(figsize=(16, 10), facecolor=DARK)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.1, 1.2],
        height_ratios=[1, 1],
        hspace=0.34,
        wspace=0.24,
    )
    ax_player_score = fig.add_subplot(gs[0, 0])
    ax_agent_score = fig.add_subplot(gs[1, 0])
    ax_experience = fig.add_subplot(gs[0, 1])
    ax_regime = fig.add_subplot(gs[1, 1])

    _draw_barh(
        ax_player_score,
        players[:12],
        "Memory: Players PnL",
        PURPLE,
        panteon_pnl_pct=0.0,
        x_label="Cumulative Memory PnL %",
        benchmark=False,
    )
    _draw_barh(
        ax_agent_score,
        agents[:12],
        "Memory: Agents PnL",
        BLUE,
        panteon_pnl_pct=0.0,
        x_label="Cumulative Memory PnL %",
        benchmark=False,
    )
    _draw_memory_experience(ax_experience, players, agents)
    _draw_memory_regime_summary(ax_regime, players, agents)

    fig.suptitle("Panteon Memory Dashboard", color=TEXT, fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.96)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_memory_experience(
    ax,
    players: List[Tuple[str, dict]],
    agents: List[Tuple[str, dict]],
) -> None:
    _style_ax(ax, "Memory Experience")
    rows = _memory_experience_rows(players, agents)
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No memory experience yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    labels = [row[0] for row in rows]
    closed = [row[1] for row in rows]
    signals = [row[2] for row in rows]
    y = list(range(len(rows)))
    ax.barh(y, signals, color=GOLD, alpha=0.52, label="signals generated")
    ax.barh(y, closed, color=[row[3] for row in rows], alpha=0.86, label="closed trades")
    ax.set_yticks(y, labels=labels, color=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("Count", color=MUTED, fontsize=8)
    ax.legend(loc="lower right", frameon=False, labelcolor=TEXT, fontsize=8)


def _draw_memory_regime_summary(
    ax,
    players: List[Tuple[str, dict]],
    agents: List[Tuple[str, dict]],
) -> None:
    _style_ax(ax, "Memory By Regime")
    regimes = ["bullish", "bearish", "neutral", "crash"]
    rows = _memory_regime_rows(players=players, agents=agents)
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No regime memory yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    labels = [row[0] for row in rows]
    matrix = [row[1] for row in rows]
    vmin, vmax = _heatmap_limits(matrix)
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(regimes)), labels=regimes, color=MUTED)
    ax.set_yticks(range(len(labels)), labels=labels, color=MUTED)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            ax.text(x, y, _pct(value), ha="center", va="center", color="#111111", fontsize=9)
    try:
        fig = ax.figure
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
    except Exception:
        pass


def _memory_experience_rows(
    players: List[Tuple[str, dict]],
    agents: List[Tuple[str, dict]],
    *,
    limit: int = 16,
) -> List[Tuple[str, float, float, str]]:
    rows: List[Tuple[str, float, float, str]] = []
    for group, source, color in (("P", players, PURPLE), ("A", agents, BLUE)):
        for name, data in source:
            closed = float(data.get("closed_trades", 0.0) or 0.0)
            signals = float(data.get("signals", 0.0) or 0.0)
            if closed <= 0 and signals <= 0:
                continue
            rows.append((f"{group}:{_short(name, 18)}", closed, signals, color))
    rows.sort(key=lambda row: (row[2], row[1]), reverse=True)
    return rows[: max(0, int(limit))]


def _memory_regime_rows(
    *,
    players: List[Tuple[str, dict]],
    agents: List[Tuple[str, dict]],
    limit: int = 18,
) -> List[Tuple[str, List[float], int]]:
    regimes = ["bullish", "bearish", "neutral", "crash"]
    candidates: List[Tuple[str, List[float], int, float]] = []
    for prefix, source in (("P", players), ("A", agents)):
        for name, data in source:
            raw = data.get("per_regime")
            if not isinstance(raw, Mapping):
                continue
            values: List[float] = []
            closed_total = 0
            has_data = False
            for regime in regimes:
                item = raw.get(regime)
                if isinstance(item, Mapping):
                    pnl = float(item.get("pnl_pct", 0.0) or 0.0)
                    closed = int(item.get("closed_trades", 0) or 0)
                    has_data = has_data or closed > 0 or abs(pnl) > 1e-12
                else:
                    pnl = 0.0
                    closed = 0
                values.append(pnl)
                closed_total += closed
            if not has_data:
                continue
            best_regime_pnl = max(values) if values else 0.0
            label = f"{prefix}:{_short(name, 20)}"
            candidates.append((label, values, closed_total, best_regime_pnl))

    selected: List[Tuple[str, List[float], int, float]] = []
    seen = set()

    def add_rows(rows: List[Tuple[str, List[float], int, float]]) -> None:
        for row in rows:
            if len(selected) >= limit:
                return
            if row[0] in seen:
                continue
            seen.add(row[0])
            selected.append(row)

    for regime_idx in range(len(regimes)):
        add_rows(sorted(
            candidates,
            key=lambda row: (row[1][regime_idx], row[2]),
            reverse=True,
        )[:3])
    add_rows(sorted(
        candidates,
        key=lambda row: (row[3], row[2], sum(row[1])),
        reverse=True,
    ))
    return [(label, values, closed_total) for label, values, closed_total, _ in selected]


def _heatmap_limits(matrix: List[List[float]]) -> Tuple[float, float]:
    max_abs = 0.0
    for row in matrix:
        for value in row:
            max_abs = max(max_abs, abs(float(value)))
    limit = max(0.25, min(5.0, max_abs))
    return -limit, limit


def _render_regime_heatmap(path: Path, title: str, rows: List[Tuple[str, dict]]) -> None:
    plt = _setup_pyplot()
    regimes = ["bullish", "bearish", "neutral", "crash"]
    usable = [(name, data) for name, data in rows if isinstance(data.get("per_regime"), Mapping)][:16]
    fig_h = max(4.5, 2.0 + len(usable) * 0.42)
    fig, ax = plt.subplots(figsize=(11, fig_h), facecolor=DARK)
    image = _draw_regime_heatmap(ax, title, rows, regimes=regimes)
    if image is not None:
        cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_regime_heatmap(
    ax,
    title: str,
    rows: List[Tuple[str, dict]],
    *,
    regimes: List[str] | None = None,
):
    regimes = regimes or ["bullish", "bearish", "neutral", "crash"]
    usable = [(name, data) for name, data in rows if isinstance(data.get("per_regime"), Mapping)][:16]
    ax.set_facecolor(MID)
    if not usable:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No regime data yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=12)
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
        return None

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
    return image


def _draw_equity_curves(
    ax,
    title: str,
    rows: List[Tuple[str, dict]],
    fallback_color: str,
) -> None:
    _style_ax(ax, title)
    plotted = 0
    palette = None
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        palette = plt.cm.tab20
    except Exception:
        palette = None
    for idx, (name, data) in enumerate(rows[:12]):
        curve = _equity_curve(data)
        if len(curve) < 2:
            continue
        is_virtual_panteon = _is_virtual_panteon_name(name)
        color = MUTED if is_virtual_panteon else (
            palette(idx % 20) if palette is not None else fallback_color
        )
        lw = 2.1 if is_virtual_panteon else (1.7 if plotted < 4 else 0.9)
        alpha = 0.95 if is_virtual_panteon else (0.9 if plotted < 4 else 0.35)
        label = (
            f"Virtual_Panteon {_pct(data.get('display_pnl_pct', data.get('pnl_pct', 0.0)))}"
            if is_virtual_panteon
            else f"{_short(name, 16)} {_pct(data.get('display_pnl_pct', data.get('pnl_pct', 0.0)))}"
            if plotted < 6
            else None
        )
        linestyle = (0, (2, 3)) if is_virtual_panteon else "-"
        ax.plot(range(len(curve)), curve, color=color, linewidth=lw,
                alpha=alpha, linestyle=linestyle, label=label)
        plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No equity history yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    ax.set_xlabel("Bar", color=MUTED, fontsize=8)
    ax.set_ylabel("Assets", color=MUTED, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=TEXT)


def _equity_curve(data: Mapping[str, object]) -> List[float]:
    for key in ("equity_curve", "assets_curve", "total_assets_curve", "history"):
        raw = data.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        out: List[float] = []
        for value in raw:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                continue
        if len(out) >= 2:
            return out
    pnl = _float_value(data.get("display_pnl_pct", data.get("pnl_pct", 0.0)))
    return [100.0, 100.0 * (1.0 + pnl / 100.0)]


def _usd(value: object) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _shadow_counter(shadow: Mapping[str, object], key: str) -> str:
    total = int(_float_value(shadow.get(key, 0)))
    bar_key = f"{key}_bar"
    if bar_key not in shadow:
        return str(total)
    bar_value = int(_float_value(shadow.get(bar_key, 0)))
    return f"{total} (bar {bar_value})"


def _float_value(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _panteon_pnl_pct(status: Mapping[str, object]) -> float:
    live_session = status.get("live_session")
    if isinstance(live_session, Mapping):
        try:
            return float(live_session.get("panteon_owned_pnl_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
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


def _draw_panteon_benchmark(
    ax,
    shadow_values: Iterable[float],
    panteon_pnl_pct: float,
    *,
    virtual_panteon_pnl_pct: Optional[float] = None,
) -> None:
    values = list(shadow_values)
    all_values = values + [panteon_pnl_pct, 0.0]
    if virtual_panteon_pnl_pct is not None:
        all_values.append(virtual_panteon_pnl_pct)
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
    if virtual_panteon_pnl_pct is not None:
        ax.axvline(
            virtual_panteon_pnl_pct,
            color=MUTED,
            linestyle=(0, (2, 3)),
            linewidth=1.8,
            label=f"Virtual_Panteon {_pct(virtual_panteon_pnl_pct)}",
            zorder=4,
        )
        ax.text(
            virtual_panteon_pnl_pct,
            0.88,
            f" Virtual_Panteon {_pct(virtual_panteon_pnl_pct)}",
            transform=ax.get_xaxis_transform(),
            color=MUTED,
            fontsize=8,
            ha="left",
            va="top",
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
