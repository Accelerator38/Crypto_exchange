"""Matplotlib PNG backend for operator-facing Panteon v2 dashboards."""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from ..domain.types import Regime


DEFAULT_DASHBOARD_PRODUCT_VERSION = "Panteon v3"
PRODUCT_VERSION_STATUS_KEYS = (
    "product_version",
    "global_version",
    "display_version",
    "app_version",
)
PRODUCT_VERSION_ENV_KEYS = (
    "PANTEON_PRODUCT_VERSION",
    "PANTEON_GLOBAL_VERSION",
    "PANTEON_DISPLAY_VERSION",
    "PANTEON_APP_VERSION",
)
BUILD_STATUS_KEYS = (
    "build_id",
    "build",
    "build_label",
    "build_sha",
    "git_sha",
    "commit_sha",
    "commit",
)
BUILD_ENV_KEYS = (
    "PANTEON_BUILD_ID",
    "PANTEON_BUILD",
    "PANTEON_BUILD_SHA",
    "PANTEON_GIT_SHA",
    "PANTEON_COMMIT_SHA",
)
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
INACTIVE_STATUSES = frozenset({"quarantine", "quarantined", "shadow_only", "purgatory"})


def _regime_labels() -> List[str]:
    return [regime.label for regime in Regime]


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
    if bool(status_map.get("player_only_runtime")):
        players = _merge_actor_rows(players, agents)
        memory_players = _merge_actor_rows(memory_players, memory_agents)
        agents = []
        memory_agents = []
    else:
        agents, players = _split_visual_agents_and_players(agents, players)
        memory_agents, memory_players = _split_visual_agents_and_players(
            memory_agents,
            memory_players,
        )
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
    _render_combined_regime_dashboard(
        paths[2],
        memory_agents,
        memory_players,
        status=status_map,
    )
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


def _split_visual_agents_and_players(
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]]]:
    if agents:
        return agents, players
    fallback_agents: List[Tuple[str, dict]] = []
    visual_players: List[Tuple[str, dict]] = []
    for name, data in players:
        if _is_strategy_actor_row(data):
            row = dict(data)
            row["visual_agent_fallback"] = True
            fallback_agents.append((name, row))
        else:
            visual_players.append((name, data))
    if not fallback_agents:
        return agents, players
    return fallback_agents, visual_players


def _merge_actor_rows(
    primary: List[Tuple[str, dict]],
    secondary: List[Tuple[str, dict]],
) -> List[Tuple[str, dict]]:
    """Merge legacy-labelled rows into the player view without duplicates."""
    merged: dict[str, Tuple[str, dict]] = {}
    for name, data in [*secondary, *primary]:
        merged[_clean_actor_name(name)] = (name, data)
    return sorted(
        merged.values(),
        key=lambda item: _row_pnl_value(item[1]),
        reverse=True,
    )


def _is_strategy_actor_row(data: Mapping[str, object]) -> bool:
    kind = str(data.get("actor_pool_kind") or data.get("kind") or "").strip().lower()
    return kind == "strategy"


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
    player_only = bool(status.get("player_only_runtime"))
    session_agents = (
        []
        if player_only
        else _current_session_rows(agents, status=status, actor_kind="agent")
    )
    session_players = _current_session_rows(players, status=status, actor_kind="player")
    fig = plt.figure(figsize=(16, 12), facecolor=DARK)
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[1.05, 1, 1],
        height_ratios=[0.95, 0.95, 0.82, 0.78],
        hspace=0.30,
        wspace=0.20,
    )

    ax_status = fig.add_subplot(gs[:, 0])
    ax_assets = fig.add_subplot(gs[0, 1:])
    ax_prices = fig.add_subplot(gs[1, 1:])
    ax_agents = fig.add_subplot(gs[2, 1])
    ax_players = fig.add_subplot(gs[2, 2])
    ax_flash_selected = fig.add_subplot(gs[3, 1])
    ax_flash_candidates = fig.add_subplot(gs[3, 2])
    for ax in (
        ax_status,
        ax_assets,
        ax_prices,
        ax_agents,
        ax_players,
        ax_flash_selected,
        ax_flash_candidates,
    ):
        _style_ax(ax)

    _draw_status_panel(ax_status, status)
    _draw_assets_panel(
        ax_assets,
        status,
        session_agents,
        session_players,
        title=(
            "Current Session Assets: Panteon and Players"
            if player_only
            else "Current Session Assets: Panteon, Players, Agents"
        ),
    )
    _draw_price_panel(ax_prices, status)
    if player_only:
        _draw_barh(
            ax_agents,
            session_players[:12],
            "Current Session Players",
            PURPLE,
            panteon_pnl_pct=panteon_pnl_pct,
        )
        ax_players.set_axis_off()
        _draw_player_only_contract_panel(ax_flash_selected, status)
        _draw_player_only_shadow_panel(ax_flash_candidates, status)
    else:
        _draw_barh(ax_agents, session_agents[:12], "Current Session Agents", BLUE,
                   panteon_pnl_pct=panteon_pnl_pct)
        _draw_barh(ax_players, session_players[:12], "Current Session Players", PURPLE,
                   panteon_pnl_pct=panteon_pnl_pct)
        flash_info = _flash_diagnostics(status)
        _draw_flash_selected_panel(ax_flash_selected, _flash_selected_rows(status), flash_info)
        _draw_flash_candidates_panel(ax_flash_candidates, _flash_candidate_rows(status), flash_info)

    fig.suptitle(_operator_dashboard_title(status), color=TEXT, fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.97)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def _operator_dashboard_title(status: Mapping[str, object]) -> str:
    version = _dashboard_product_version(status)
    build = _dashboard_build_id(status)
    if build:
        return f"{version} - сборка {build}"
    return version


def _dashboard_product_version(status: Mapping[str, object]) -> str:
    value = _first_clean_value(status, PRODUCT_VERSION_STATUS_KEYS)
    if value:
        return value
    for key in PRODUCT_VERSION_ENV_KEYS:
        value = _clean_dashboard_text(os.getenv(key))
        if value:
            return value
    return DEFAULT_DASHBOARD_PRODUCT_VERSION


def _dashboard_build_id(status: Mapping[str, object]) -> str:
    value = _first_clean_value(status, BUILD_STATUS_KEYS)
    if value:
        return _short_build_id(value)
    for key in BUILD_ENV_KEYS:
        value = _clean_dashboard_text(os.getenv(key))
        if value:
            return _short_build_id(value)
    value = _git_build_id()
    if value:
        return value
    return _short_build_id(
        _first_clean_value(status, ("session_id", "run_id")) or "local"
    )


def _first_clean_value(
    source: Mapping[str, object],
    keys: Iterable[str],
) -> str:
    if not isinstance(source, Mapping):
        return ""
    for key in keys:
        value = _clean_dashboard_text(source.get(key))
        if value:
            return value
    return ""


def _clean_dashboard_text(value: object) -> str:
    text = str(value or "").strip()
    return text if text and text.lower() not in {"none", "null", "-"} else ""


def _short_build_id(value: object) -> str:
    text = _clean_dashboard_text(value)
    if len(text) >= 12 and all(char in "0123456789abcdefABCDEF" for char in text):
        return text[:8]
    return text


@lru_cache(maxsize=1)
def _git_build_id() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return _clean_dashboard_text(proc.stdout)


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
        shadow_label = (
            "Shadow players"
            if bool(status.get("player_only_runtime"))
            else "Shadow actors"
        )
        lines.extend([
            (shadow_label, shadow.get("actors", 0)),
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
        _short(name) + (" [Q]" if _is_inactive_row(data) else "")
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


def _current_session_rows(
    rows: List[Tuple[str, dict]],
    *,
    status: Mapping[str, object],
    actor_kind: str,
) -> List[Tuple[str, dict]]:
    explicit = _real_actor_names(status, actor_kind) | _selected_flash_actor_names(status, actor_kind)
    selected: List[Tuple[str, dict]] = []
    for name, data in rows:
        clean = _clean_actor_name(name)
        if clean in explicit or _has_current_session_activity(data):
            selected.append((name, data))
    selected.sort(
        key=lambda item: (
            _has_current_session_activity(item[1]),
            item[0] in explicit,
            _row_pnl_value(item[1]),
        ),
        reverse=True,
    )
    return selected


def _clean_actor_name(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("V_"):
        text = text[2:]
    return text


def _has_current_session_activity(data: Mapping[str, object]) -> bool:
    count_keys = (
        "session_signals",
        "session_entries",
        "session_closed_trades",
        "session_wins",
        "session_losses",
    )
    if any(int(_float_value(data.get(key, 0))) > 0 for key in count_keys):
        return True
    return abs(_float_value(data.get("session_pnl_pct", 0.0))) > 1e-12


def _real_actor_names(status: Mapping[str, object], actor_kind: str) -> set[str]:
    payload = status.get("real_trading_actors")
    if not isinstance(payload, Mapping):
        return set()
    names: set[str] = set()
    if actor_kind == "player":
        for row in payload.get("open_positions", []) or []:
            if isinstance(row, Mapping):
                name = _clean_actor_name(row.get("player"))
                if name:
                    names.add(name)
        for row in payload.get("closed_players", []) or []:
            if isinstance(row, Mapping):
                name = _clean_actor_name(row.get("player"))
                if name:
                    names.add(name)
    elif actor_kind == "agent":
        for row in payload.get("open_positions", []) or []:
            if isinstance(row, Mapping):
                name = _clean_actor_name(row.get("agent"))
                if name:
                    names.add(name)
        for row in payload.get("closed_agents", []) or []:
            if isinstance(row, Mapping):
                name = _clean_actor_name(row.get("agent"))
                if name:
                    names.add(name)
    return names


def _selected_flash_actor_names(status: Mapping[str, object], actor_kind: str) -> set[str]:
    flash = status.get("flash") if isinstance(status.get("flash"), Mapping) else {}
    selected = flash.get("selected_actors_by_symbol") if isinstance(flash, Mapping) else {}
    actor_types = flash.get("actor_types_by_symbol") if isinstance(flash, Mapping) else {}
    if not isinstance(selected, Mapping):
        return set()
    if not isinstance(actor_types, Mapping):
        actor_types = {}
    want = "agent" if actor_kind == "agent" else "ensemble"
    names: set[str] = set()
    for symbol, raw_label in selected.items():
        label = _clean_actor_name(raw_label)
        if not label or label == "NoTrade":
            continue
        raw_type = str(actor_types.get(symbol, "") or "").strip().lower()
        if raw_type and raw_type != want:
            continue
        names.add(label)
    return names


def _draw_assets_panel(
    ax,
    status: Mapping[str, object],
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
    *,
    title: str,
) -> None:
    _style_ax(ax, title)
    rows = _asset_rows(status, agents=agents, players=players)
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No actor asset data yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    usable = rows[:24]
    labels = [row["label"] for row in usable]
    values = [float(row["asset"]) for row in usable]
    colors = [
        GOLD if row["kind"] == "panteon"
        else PURPLE if row["kind"] == "player"
        else BLUE
        for row in usable
    ]
    y_pos = list(range(len(usable)))
    ax.barh(y_pos, values, color=colors, alpha=0.88)
    ax.set_yticks(y_pos, labels=[_short(label, 28) for label in labels], color=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel("Current assets / virtual equity, USD", color=MUTED, fontsize=8)
    for idx, row in enumerate(usable):
        weight = "bold" if row["kind"] == "panteon" else "normal"
        color = GOLD if row["kind"] == "panteon" else TEXT
        ax.text(float(row["asset"]), idx, f" {_usd(row['asset'])}",
                color=color, va="center", fontsize=8, fontweight=weight)


def _asset_rows(
    status: Mapping[str, object],
    *,
    agents: List[Tuple[str, dict]],
    players: List[Tuple[str, dict]],
) -> List[dict]:
    kind_order = {"panteon": 0, "player": 1, "agent": 2}
    initial = _float_value(status.get("initial_capital", 0.0))
    if initial <= 0.0:
        initial = _float_value(status.get("current_balance", 100.0)) or 100.0
    live_session = status.get("live_session") if isinstance(status.get("live_session"), Mapping) else {}
    panteon_asset = _float_value(
        status.get(
            "panteon_equity_usd",
            live_session.get("panteon_equity_usd", status.get("total_assets_usd", initial)),
        )
    )
    rows: List[dict] = [{
        "label": "PANTEON",
        "kind": "panteon",
        "asset": panteon_asset or initial,
    }]
    for kind, source in (("player", players), ("agent", agents)):
        for name, data in source:
            if _is_virtual_panteon_name(name):
                continue
            rows.append({
                "label": f"{'P' if kind == 'player' else 'A'}:{_clean_actor_name(name)}",
                "kind": kind,
                "asset": _actor_asset_value(data, initial),
            })
    return sorted(
        rows,
        key=lambda row: (
            -float(row["asset"]),
            kind_order.get(str(row["kind"]), 99),
            str(row["label"]),
        ),
    )


def _actor_asset_value(data: Mapping[str, object], initial: float) -> float:
    for key in ("equity", "asset", "assets", "total_assets_usd"):
        value = _float_value(data.get(key, 0.0))
        if value > 0.0:
            return value
    pnl = _row_pnl_value(data)
    return max(0.0, initial * (1.0 + pnl / 100.0))


def _price_history_rows(
    status: Mapping[str, object],
    *,
    limit: int = 240,
) -> List[dict]:
    raw = status.get("price_history")
    if not isinstance(raw, (list, tuple)):
        raw = []
    rows: List[dict] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        prices = item.get("prices")
        if not isinstance(prices, Mapping):
            prices = item
        clean_prices = _clean_price_mapping(prices)
        if not clean_prices:
            continue
        rows.append({
            "bar": int(_float_value(item.get("bar", idx))),
            "regime": str(item.get("regime", "") or ""),
            "market": str(item.get("market", "") or ""),
            "prices": clean_prices,
        })
        timestamp = _parse_datetime_value(
            item.get("timestamp")
            or item.get("timestamp_utc")
            or item.get("time")
            or item.get("datetime")
        )
        if timestamp is not None:
            rows[-1]["timestamp"] = timestamp
    if not rows:
        current = _clean_price_mapping(status.get("current_prices"))
        if current:
            rows.append({
                "bar": int(_float_value(status.get("bar_count", 0))),
                "regime": str(status.get("regime", "") or ""),
                "market": str(status.get("market", "") or ""),
                "prices": current,
            })
            timestamp = _parse_datetime_value(
                status.get("timestamp")
                or status.get("timestamp_utc")
                or status.get("time")
            )
            if timestamp is not None:
                rows[-1]["timestamp"] = timestamp
    rows = rows[-max(1, int(limit)):]
    _fill_missing_price_timestamps(rows, status)
    return rows


def _fill_missing_price_timestamps(
    rows: List[dict],
    status: Mapping[str, object],
) -> None:
    if not rows or all(isinstance(row.get("timestamp"), datetime) for row in rows):
        return
    last_timestamp = next(
        (
            row.get("timestamp")
            for row in reversed(rows)
            if isinstance(row.get("timestamp"), datetime)
        ),
        None,
    )
    if last_timestamp is None:
        last_timestamp = _parse_datetime_value(
            status.get("timestamp")
            or status.get("timestamp_utc")
            or status.get("time")
        )
    if last_timestamp is None:
        return
    last_bar = int(_float_value(status.get("bar_count", rows[-1].get("bar", 0))))
    if last_bar <= 0:
        last_bar = int(_float_value(rows[-1].get("bar", 0)))
    interval = _timeframe_seconds(status.get("timeframe"))
    for row in rows:
        if isinstance(row.get("timestamp"), datetime):
            continue
        bar = int(_float_value(row.get("bar", last_bar)))
        row["timestamp"] = last_timestamp - timedelta(seconds=max(0, last_bar - bar) * interval)


def _parse_datetime_value(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timeframe_seconds(value: object) -> int:
    raw = str(value or "").strip().lower()
    if raw in {"bridge_poll", "poll", "polling", ""}:
        return 5
    aliases = {
        "minute": 60,
        "min": 60,
        "hour": 3600,
        "day": 86400,
    }
    if raw in aliases:
        return aliases[raw]
    unit = raw[-1:] if raw else ""
    number = raw[:-1] if unit in {"s", "m", "h", "d"} else raw
    try:
        amount = max(1, int(float(number)))
    except ValueError:
        return 60
    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 3600
    if unit == "d":
        return amount * 86400
    return amount * 60


def _clean_price_mapping(raw: object) -> Dict[str, float]:
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, float] = {}
    for raw_symbol, raw_price in raw.items():
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            continue
        price = _float_value(raw_price)
        if price > 0.0:
            out[symbol] = price
    return out


def _draw_price_panel(ax, status: Mapping[str, object]) -> None:
    _style_ax(ax, "Currency Prices With Current Market Type")
    rows = _price_history_rows(status)
    market_view = status.get("market_view") if isinstance(status.get("market_view"), Mapping) else {}
    market = str(market_view.get("market", status.get("market", status.get("regime", "-"))) or "-")
    confidence = market_view.get("regime_confidence", status.get("regime_confidence", None))
    note = f"market={market}"
    if confidence is not None:
        note += f" confidence={_float_value(confidence):.2f}"
    ax.text(0.01, 0.97, note, transform=ax.transAxes,
            color=GOLD, fontsize=9, fontweight="bold", va="top",
            bbox={"facecolor": MID, "edgecolor": "none", "alpha": 0.82, "pad": 2.0})
    if len(rows) < 2:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Price history will appear after two live bars",
                transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=11)
        return
    symbols = _price_panel_symbols(rows, status)
    if not symbols:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No clean prices in history", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        palette = plt.cm.tab10
    except Exception:
        palette = None
    use_time_axis = all(isinstance(row.get("timestamp"), datetime) for row in rows)
    x_values = [
        row["timestamp"] if use_time_axis else row["bar"]
        for row in rows
    ]
    for idx, symbol in enumerate(symbols):
        series = []
        for row in rows:
            value = row["prices"].get(symbol)
            series.append(float(value) if value else None)
        first = next((value for value in series if value and value > 0.0), None)
        if not first:
            continue
        normalized = [
            (value / first * 100.0) if value and value > 0.0 else None
            for value in series
        ]
        xs = [x for x, value in zip(x_values, normalized) if value is not None]
        ys = [value for value in normalized if value is not None]
        if len(xs) < 2:
            continue
        color = palette(idx % 10) if palette is not None else BLUE
        ax.plot(xs, ys, label=symbol, color=color, linewidth=1.4, alpha=0.9)
    ax.axhline(100.0, color=GRID, linewidth=0.8)
    if use_time_axis:
        try:
            import matplotlib.dates as mdates  # noqa: WPS433

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m/%d\n%H:%M", tz=timezone.utc)
            )
        except Exception:
            pass
        ax.set_xlabel("Time (UTC)", color=MUTED, fontsize=8)
    else:
        ax.set_xlabel("Bar", color=MUTED, fontsize=8)
    ax.set_ylabel("Indexed price (first=100)", color=MUTED, fontsize=8)
    ax.legend(loc="lower left", frameon=False, labelcolor=TEXT, fontsize=7, ncol=3)


def _price_panel_symbols(
    rows: List[dict],
    status: Mapping[str, object],
    *,
    limit: int = 6,
) -> List[str]:
    preferred: List[str] = []
    open_positions = status.get("open_positions")
    if isinstance(open_positions, Mapping):
        preferred.extend(str(sym).upper() for sym in open_positions.keys())
    flash = status.get("flash") if isinstance(status.get("flash"), Mapping) else {}
    selected = flash.get("selected_actors_by_symbol") if isinstance(flash, Mapping) else {}
    if isinstance(selected, Mapping):
        preferred.extend(str(sym).upper() for sym in selected.keys())
    all_symbols = sorted({sym for row in rows for sym in row["prices"]})
    ranked: List[Tuple[float, str]] = []
    for symbol in all_symbols:
        values = [row["prices"].get(symbol) for row in rows if row["prices"].get(symbol)]
        if len(values) < 2:
            move = 0.0
        else:
            first = float(values[0])
            last = float(values[-1])
            move = abs(last / first - 1.0) if first > 0.0 else 0.0
        ranked.append((move, symbol))
    ranked.sort(reverse=True)
    out: List[str] = []
    for symbol in preferred + [symbol for _, symbol in ranked]:
        if symbol in all_symbols and symbol not in out:
            out.append(symbol)
        if len(out) >= limit:
            break
    return out


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


def _flash_diagnostics(status: Mapping[str, object]) -> dict:
    selected_rows = _flash_selected_rows(status)
    symbols_total = sum(int(row.get("count", 0) or 0) for row in selected_rows)
    no_trade_symbols = sum(
        int(row.get("count", 0) or 0)
        for row in selected_rows
        if str(row.get("label", "")) == "NoTrade"
    )
    flash = status.get("flash") if isinstance(status.get("flash"), Mapping) else {}
    decisions = flash.get("decisions") if isinstance(flash, Mapping) else ()
    reasons: Counter[str] = Counter()
    actionable = 0
    rejected = 0
    symbols_with_candidate_reasons: set[str] = set()
    if isinstance(decisions, (list, tuple)):
        for decision in decisions:
            if not isinstance(decision, Mapping):
                continue
            symbol = str(decision.get("symbol") or "").strip()
            candidates = decision.get("candidates", ()) or ()
            compact_rejected = decision.get("top_rejected_candidates", ()) or ()
            use_compact_rejected = not candidates and compact_rejected
            for candidate in compact_rejected if use_compact_rejected else candidates:
                if not isinstance(candidate, Mapping):
                    continue
                label = str(candidate.get("label") or "")
                actor_type = str(candidate.get("actor_type") or "")
                if label == "NoTrade" or actor_type == "no_trade":
                    continue
                action = str(candidate.get("action") or "").strip().upper()
                is_hold = not action or action == "HOLD" or action.endswith(".HOLD")
                if use_compact_rejected or bool(candidate.get("rejected", False)):
                    rejected += 1
                    reason = str(candidate.get("reason") or "rejected").strip() or "rejected"
                    reasons[reason] += 1
                    if symbol:
                        symbols_with_candidate_reasons.add(symbol)
                elif not is_hold:
                    actionable += 1
    gate_funnel = flash.get("gate_funnel_by_symbol") if isinstance(flash, Mapping) else {}
    if isinstance(gate_funnel, Mapping):
        for symbol, row in gate_funnel.items():
            clean_symbol = str(symbol or "").strip()
            if clean_symbol in symbols_with_candidate_reasons:
                continue
            if not isinstance(row, Mapping):
                continue
            reason = str(row.get("top_blocker") or "").strip()
            if reason:
                reasons[reason] += 1
    return {
        "symbols_total": symbols_total,
        "no_trade_symbols": no_trade_symbols,
        "selected_symbols": symbols_total - no_trade_symbols,
        "all_no_trade": symbols_total > 0 and symbols_total == no_trade_symbols,
        "actionable_candidates": actionable,
        "rejected_candidates": rejected,
        "top_reasons": reasons.most_common(5),
    }


def _draw_player_only_contract_panel(ax, status: Mapping[str, object]) -> None:
    ax.set_axis_off()
    pool = (
        status.get("configured_actor_pool")
        if isinstance(status.get("configured_actor_pool"), Mapping)
        else {}
    )
    players = pool.get("players", []) if isinstance(pool, Mapping) else []
    lines = (
        "PLAYER-ONLY RUNTIME",
        f"Mode: {status.get('trade_mode', '-')}",
        f"Fixed player: {status.get('fixed_player_label', '-') or '-'}",
        f"Registered players: {len(players) if isinstance(players, list) else 0}",
        "Registered agents: 0",
    )
    for index, line in enumerate(lines):
        ax.text(
            0.04,
            0.88 - index * 0.16,
            line,
            transform=ax.transAxes,
            color=GOLD if index == 0 else TEXT,
            fontsize=11 if index == 0 else 9,
            fontweight="bold" if index == 0 else "normal",
            va="top",
        )


def _draw_player_only_shadow_panel(ax, status: Mapping[str, object]) -> None:
    ax.set_axis_off()
    debug = (
        status.get("decision_debug")
        if isinstance(status.get("decision_debug"), Mapping)
        else {}
    )
    session = (
        debug.get("shadow_session")
        if isinstance(debug.get("shadow_session"), Mapping)
        else {}
    )
    shadow = status.get("shadow") if isinstance(status.get("shadow"), Mapping) else {}
    lines = (
        "SHADOW PLAYERS",
        f"Players: {shadow.get('actors', 0)}",
        f"Signals: {int(_float_value(session.get('signals', 0)))}",
        f"Filled: {int(_float_value(session.get('filled', 0)))}",
        f"Real positions: {int(_float_value(status.get('n_positions', 0)))}",
    )
    for index, line in enumerate(lines):
        ax.text(
            0.04,
            0.88 - index * 0.16,
            line,
            transform=ax.transAxes,
            color=GOLD if index == 0 else TEXT,
            fontsize=11 if index == 0 else 9,
            fontweight="bold" if index == 0 else "normal",
            va="top",
        )


def _draw_flash_selected_panel(ax, rows: List[dict], diagnostics: Mapping[str, object]) -> None:
    _style_ax(ax, "Flash Selected Actors")
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No Flash decisions yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
        return
    if diagnostics.get("all_no_trade"):
        ax.set_axis_off()
        top_reasons = diagnostics.get("top_reasons") or []
        reason_text = ", ".join(f"{name} x{count}" for name, count in top_reasons[:3])
        lines = [
            f"NoTrade selected on {int(diagnostics.get('symbols_total', 0))} symbols.",
            "Flash did not find an executable actor this bar.",
        ]
        if reason_text:
            lines.append(f"Top blocks: {reason_text}")
        ax.text(0.03, 0.76, lines[0], transform=ax.transAxes,
                color=GOLD, fontsize=10, fontweight="bold", va="top")
        ax.text(0.03, 0.56, "\n".join(lines[1:]), transform=ax.transAxes,
                color=TEXT, fontsize=8, va="top")
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


def _draw_flash_candidates_panel(
    ax,
    rows: List[dict],
    diagnostics: Mapping[str, object],
) -> None:
    _style_ax(ax, "Flash Candidate Diagnostics")
    if not rows:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No Flash candidates yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=10)
        return
    if diagnostics.get("all_no_trade") or int(diagnostics.get("actionable_candidates", 0) or 0) <= 0:
        reasons = list(diagnostics.get("top_reasons") or [])
        if not reasons:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No actionable candidates this bar",
                    transform=ax.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=10)
            return
        labels = [_short(str(reason), 24) for reason, _ in reasons]
        values = [int(count) for _, count in reasons]
        ax.barh(labels[::-1], values[::-1], color=RED, alpha=0.82)
        ax.set_xlabel("Rejected candidate count", color=MUTED, fontsize=8)
        ax.set_xlim(0, max(values) * 1.25 if values else 1)
        for idx, value in enumerate(values[::-1]):
            ax.text(value, idx, f" {value}", color=TEXT, va="center", fontsize=8)
        ax.text(0.01, 0.98, "No executable candidates",
                transform=ax.transAxes, color=GOLD, fontsize=8,
                fontweight="bold", va="top")
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
            f" actionable {int(row['actionable_count'])},"
            f" rejected {int(row['rejected_count'])},"
            f" best {float(row['best_score']):+.2f}"
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


def _is_truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_inactive_row(data: Mapping[str, object]) -> bool:
    return (
        _is_inactive_status(data.get("status"))
        or _is_truthy_flag(data.get("quarantined"))
        or _is_truthy_flag(data.get("is_quarantined"))
    )


def _is_virtual_panteon_name(name: object) -> bool:
    text = str(name or "").strip()
    if text.startswith("V_"):
        text = text[2:]
    return text == "Virtual_Panteon"


def _row_pnl_value(data: Mapping[str, object]) -> float:
    return _float_value(
        data.get(
            "display_pnl_pct",
            data.get("session_pnl_pct", data.get("pnl_pct", 0.0)),
        )
    )


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
        if _is_inactive_row(data):
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
    fig_h = max(9.0, min(22.0, 3.5 + (len(players) + len(agents)) * 0.23))
    fig = plt.figure(figsize=(14, fig_h), facecolor=DARK)
    if not agents:
        ax_players = fig.add_subplot(1, 1, 1)
        _draw_barh(
            ax_players,
            players,
            "Shadow Players - Full Pool",
            PURPLE,
            panteon_pnl_pct=panteon_pnl_pct,
        )
        fig.suptitle("Shadow Players", color=TEXT, fontsize=15, fontweight="bold")
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.96)
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[max(6, len(players)), max(6, len(agents))],
        hspace=0.36,
    )
    ax_players = fig.add_subplot(gs[0, 0])
    ax_agents = fig.add_subplot(gs[1, 0])
    _draw_barh(
        ax_players,
        players,
        "Shadow Players - Full Pool",
        PURPLE,
        panteon_pnl_pct=panteon_pnl_pct,
    )
    _draw_barh(
        ax_agents,
        agents,
        "Shadow Agents - Full Pool",
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
    *,
    status: Mapping[str, object] | None = None,
) -> None:
    plt = _setup_pyplot()
    fig_h = max(10.0, min(22.0, 4.0 + (len(players) + len(agents)) * 0.18))
    fig = plt.figure(figsize=(17, fig_h), facecolor=DARK)
    if not agents:
        ax_player_heat = fig.add_subplot(1, 2, 1)
        ax_player_curve = fig.add_subplot(1, 2, 2)
        player_image = _draw_regime_heatmap(
            ax_player_heat,
            "Player Regime Efficiency",
            players,
        )
        _draw_equity_curves(
            ax_player_curve,
            "Player Equity Curves",
            players,
            PURPLE,
            status=status,
            include_panteon=True,
        )
        if player_image is not None:
            cbar = fig.colorbar(player_image, ax=ax_player_heat, fraction=0.025, pad=0.02)
            cbar.ax.tick_params(colors=MUTED, labelsize=8)
        fig.suptitle("Player Regime Dashboard", color=TEXT, fontsize=15, fontweight="bold")
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.96)
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return
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
    _draw_equity_curves(
        ax_player_curve,
        "Player Equity Curves",
        players,
        PURPLE,
        status=status,
        include_panteon=True,
    )
    _draw_equity_curves(
        ax_agent_curve,
        "Agent Equity Curves",
        agents,
        BLUE,
        status=status,
        include_panteon=True,
    )
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
    fig_h = max(10.0, min(24.0, 4.0 + (len(players) + len(agents)) * 0.20))
    fig = plt.figure(figsize=(16, fig_h), facecolor=DARK)
    if not agents:
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.1, 1.2],
            height_ratios=[1, 1],
            hspace=0.34,
            wspace=0.24,
        )
        ax_player_score = fig.add_subplot(gs[0, 0])
        ax_experience = fig.add_subplot(gs[0, 1])
        ax_regime = fig.add_subplot(gs[1, :])
        _draw_barh(
            ax_player_score,
            players,
            "Memory: Players PnL - Full Pool",
            PURPLE,
            panteon_pnl_pct=0.0,
            x_label="Cumulative Memory PnL %",
            benchmark=False,
        )
        _draw_memory_experience(ax_experience, players, [])
        _draw_memory_regime_summary(ax_regime, players, [])
        fig.suptitle("Panteon Player Memory", color=TEXT, fontsize=15, fontweight="bold")
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.96)
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        return
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
        players,
        "Memory: Players PnL - Full Pool",
        PURPLE,
        panteon_pnl_pct=0.0,
        x_label="Cumulative Memory PnL %",
        benchmark=False,
    )
    _draw_barh(
        ax_agent_score,
        agents,
        "Memory: Agents PnL - Full Pool",
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
    regimes = _regime_labels()
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
    limit: Optional[int] = None,
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
    if limit is None:
        return rows
    return rows[: max(0, int(limit))]


def _memory_regime_rows(
    *,
    players: List[Tuple[str, dict]],
    agents: List[Tuple[str, dict]],
    limit: Optional[int] = None,
) -> List[Tuple[str, List[float], int]]:
    regimes = _regime_labels()
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
            if limit is not None and len(selected) >= limit:
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
    regimes = _regime_labels()
    usable = [(name, data) for name, data in rows if isinstance(data.get("per_regime"), Mapping)]
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
    regimes = regimes or _regime_labels()
    usable = [(name, data) for name, data in rows if isinstance(data.get("per_regime"), Mapping)]
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
    *,
    status: Mapping[str, object] | None = None,
    include_panteon: bool = False,
) -> None:
    _style_ax(ax, title)
    status_map = dict(status or {})
    timeline = _common_equity_timeline(status_map, rows)
    plotted = 0
    palette = None
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        palette = plt.cm.tab20
    except Exception:
        palette = None
    for idx, (name, data) in enumerate(rows):
        x_values, y_values = _dashboard_equity_curve_points(data, timeline)
        if len(y_values) < 2:
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
        ax.plot(x_values, y_values, color=color, linewidth=lw,
                alpha=alpha, linestyle=linestyle, label=label)
        plotted += 1
    if include_panteon:
        x_values, y_values = _panteon_equity_curve_points(status_map, timeline)
        if len(y_values) >= 2:
            ax.plot(
                x_values,
                y_values,
                color=GOLD,
                linewidth=2.3,
                alpha=0.98,
                linestyle="-",
                label=f"Panteon {_pct(_panteon_pnl_pct(status_map))}",
            )
            plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "No equity history yet", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=11)
        return
    if timeline:
        try:
            import matplotlib.dates as mdates  # noqa: WPS433

            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%m/%d\n%H:%M", tz=timezone.utc)
            )
        except Exception:
            pass
        if len(timeline) >= 2:
            ax.set_xlim(timeline[0], timeline[-1])
        ax.set_xlabel("Time (UTC)", color=MUTED, fontsize=8)
    else:
        ax.set_xlabel("Sample", color=MUTED, fontsize=8)
    ax.axhline(100.0, color=GRID, linewidth=0.8)
    ax.set_ylabel("Indexed equity (first=100)", color=MUTED, fontsize=8)
    ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=TEXT)


def _dashboard_equity_curve_points(
    data: Mapping[str, object],
    timeline: List[datetime],
) -> Tuple[List[object], List[float]]:
    curve, key, embedded_timestamps = _equity_curve_with_source(
        data,
        (
            "session_equity_curve",
            "equity_curve",
            "assets_curve",
            "total_assets_curve",
            "history",
        ),
    )
    if len(curve) < 2:
        return [], []
    timestamps = _curve_timestamps_for_source(data, key, embedded_timestamps)
    return _resample_curve(curve, timestamps, timeline)


def _panteon_equity_curve_points(
    status: Mapping[str, object],
    timeline: List[datetime],
) -> Tuple[List[object], List[float]]:
    curve, key, embedded_timestamps = _equity_curve_with_source(
        status,
        (
            "clean_panteon_equity_curve",
            "panteon_equity_curve",
            "equity_curve",
            "assets_curve",
            "total_assets_curve",
            "history",
        ),
    )
    if len(curve) < 2:
        return [], []
    timestamps = _curve_timestamps_for_source(status, key, embedded_timestamps)
    return _resample_curve(curve, timestamps, timeline)


def _common_equity_timeline(
    status: Mapping[str, object],
    rows: List[Tuple[str, dict]],
) -> List[datetime]:
    panteon_curve, panteon_key, embedded = _equity_curve_with_source(
        status,
        (
            "clean_panteon_equity_curve",
            "panteon_equity_curve",
            "equity_curve",
            "assets_curve",
            "total_assets_curve",
            "history",
        ),
    )
    panteon_timestamps = _curve_timestamps_for_source(status, panteon_key, embedded)
    interval = _timeframe_seconds(status.get("timeframe"))
    if len(panteon_timestamps) >= 2:
        return _complete_datetime_timeline(
            panteon_timestamps,
            max(len(panteon_curve), len(panteon_timestamps)),
            interval,
        )
    length = len(panteon_curve) if len(panteon_curve) >= 2 else 0
    if length <= 0:
        length = max(
            (
                len(_dashboard_equity_curve_values(data))
                for _name, data in rows
            ),
            default=0,
        )
    if length < 2:
        return []
    length = min(length, 720)
    last_timestamp = _parse_datetime_value(
        status.get("timestamp")
        or status.get("timestamp_utc")
        or status.get("time")
    )
    if last_timestamp is None:
        last_timestamp = datetime.now(timezone.utc)
    return [
        last_timestamp - timedelta(seconds=(length - idx - 1) * interval)
        for idx in range(length)
    ]


def _complete_datetime_timeline(
    timestamps: List[datetime],
    length: int,
    interval_seconds: int,
) -> List[datetime]:
    clean = sorted(timestamp for timestamp in timestamps if isinstance(timestamp, datetime))
    if not clean:
        return []
    length = max(2, min(720, int(length or len(clean))))
    if len(clean) >= length:
        return clean[-length:]
    first = clean[0]
    missing = length - len(clean)
    prefix = [
        first - timedelta(seconds=(missing - idx) * interval_seconds)
        for idx in range(missing)
    ]
    return prefix + clean


def _dashboard_equity_curve_values(data: Mapping[str, object]) -> List[float]:
    curve, _key, _embedded = _equity_curve_with_source(
        data,
        (
            "session_equity_curve",
            "equity_curve",
            "assets_curve",
            "total_assets_curve",
            "history",
        ),
    )
    return curve


def _resample_curve(
    curve: List[float],
    timestamps: List[datetime],
    timeline: List[datetime],
) -> Tuple[List[object], List[float]]:
    values = _normalize_curve_to_100(curve)
    if len(values) < 2:
        return [], []
    if not timeline:
        if len(timestamps) >= 2:
            aligned_values, aligned_timestamps = _align_values_and_timestamps(
                values,
                timestamps,
            )
            return aligned_timestamps, aligned_values
        return list(range(len(values))), values
    if len(timestamps) >= 2:
        aligned_values, aligned_timestamps = _align_values_and_timestamps(
            values,
            timestamps,
        )
        if len(aligned_values) >= 2:
            return timeline, _step_values_on_timeline(
                aligned_values,
                aligned_timestamps,
                timeline,
            )
    if len(values) > len(timeline):
        values = values[-len(timeline):]
    if len(values) < len(timeline):
        values = [values[0]] * (len(timeline) - len(values)) + values
    return timeline, values


def _align_values_and_timestamps(
    values: List[float],
    timestamps: List[datetime],
) -> Tuple[List[float], List[datetime]]:
    if len(timestamps) > len(values):
        timestamps = timestamps[-len(values):]
    elif len(timestamps) < len(values):
        values = values[-len(timestamps):]
    pairs = sorted(
        (timestamp, value)
        for timestamp, value in zip(timestamps, values)
        if isinstance(timestamp, datetime)
    )
    return (
        [value for _timestamp, value in pairs],
        [timestamp for timestamp, _value in pairs],
    )


def _step_values_on_timeline(
    values: List[float],
    timestamps: List[datetime],
    timeline: List[datetime],
) -> List[float]:
    if not values or not timestamps:
        return []
    out: List[float] = []
    idx = 0
    for point in timeline:
        while idx + 1 < len(timestamps) and timestamps[idx + 1] <= point:
            idx += 1
        out.append(values[idx])
    return out


def _normalize_curve_to_100(curve: List[float]) -> List[float]:
    first = next((float(item) for item in curve if float(item) > 0.0), None)
    if first is None or first <= 0.0:
        return []
    return [round(float(item) / first * 100.0, 10) for item in curve if float(item) > 0.0]


def _curve_timestamps_for_source(
    data: Mapping[str, object],
    source_key: str,
    embedded_timestamps: List[datetime],
) -> List[datetime]:
    if len(embedded_timestamps) >= 2:
        return embedded_timestamps
    candidates = []
    if source_key:
        candidates.append(f"{source_key}_timestamps")
    candidates.extend((
        "curve_timestamps",
        "equity_curve_timestamps",
        "session_equity_curve_timestamps",
        "timestamps",
    ))
    seen = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        parsed = _parse_curve_timestamps(data.get(key))
        if len(parsed) >= 2:
            return parsed
    return []


def _parse_curve_timestamps(raw: object) -> List[datetime]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[datetime] = []
    for item in raw:
        raw_timestamp = item
        if isinstance(item, Mapping):
            raw_timestamp = (
                item.get("timestamp")
                or item.get("timestamp_utc")
                or item.get("time")
            )
        timestamp = _parse_datetime_value(raw_timestamp)
        if timestamp is not None:
            out.append(timestamp)
    return out


def _equity_curve(data: Mapping[str, object]) -> List[float]:
    curve, _key, _timestamps = _equity_curve_with_source(
        data,
        ("equity_curve", "assets_curve", "total_assets_curve", "history"),
    )
    return curve


def _equity_curve_with_source(
    data: Mapping[str, object],
    keys: Tuple[str, ...],
) -> Tuple[List[float], str, List[datetime]]:
    for key in keys:
        raw = data.get(key)
        if not isinstance(raw, (list, tuple)):
            continue
        out: List[float] = []
        timestamps: List[datetime] = []
        timestamps_complete = True
        for value in raw:
            parsed = _equity_curve_value(value)
            if parsed is not None:
                out.append(parsed)
                if isinstance(value, Mapping):
                    timestamp = _parse_datetime_value(
                        value.get("timestamp")
                        or value.get("timestamp_utc")
                        or value.get("time")
                        or value.get("datetime")
                    )
                    if timestamp is None:
                        timestamps_complete = False
                    else:
                        timestamps.append(timestamp)
                else:
                    timestamps_complete = False
        if len(out) >= 2:
            return (
                out,
                key,
                timestamps if timestamps_complete and len(timestamps) == len(out) else [],
            )
    return [], "", []


def _equity_curve_value(value: object) -> Optional[float]:
    if isinstance(value, Mapping):
        for key in ("equity", "asset", "assets", "total_assets_usd", "value"):
            parsed = _safe_curve_float(value.get(key))
            if parsed is not None:
                return parsed
        return None
    return _safe_curve_float(value)


def _safe_curve_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


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
        for key in ("clean_panteon_pnl_pct", "panteon_owned_pnl_pct", "panteon_pnl_pct"):
            if key not in live_session:
                continue
            try:
                return float(live_session.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
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
