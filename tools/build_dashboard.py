"""
Сборщик интерактивного HTML-дашборда по всем сессиям BITGET / MEXC.

Запуск:
    python tools/build_dashboard.py
    # → пишет Results/dashboard.html  (открыть в браузере, всё локально)

Что внутри:
  • Селектор биржи и конкретной сессии
  • Сводка sessions.json (real_pnl_pct, regime, bars, n_signals)
  • Таблица игроков с колонкой status (live / shadow_only / quarantine / purgatory)
  • Таблица агентов с теми же фильтрами
  • Кросс-биржевая сводка по агентам
  • Топ-победители и топ-проигравшие
  • Цвет PnL: зелёный +, красный −
  • Никаких внешних зависимостей — один HTML с встроенными данными
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Пути относительно корня репозитория
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
OUT_HTML = RESULTS / "dashboard.html"

EXCHANGES = ("BITGET", "MEXC")
SESSIONS_LIMIT = int(os.getenv("PANTEON_DASHBOARD_SESSIONS_LIMIT", "40"))
SESSION_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_v\d+)?$")


def _safe_read(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _clean_price_history(raw, limit=240):
    if not isinstance(raw, list):
        return []
    out = []
    for idx, item in enumerate(raw[-limit:]):
        if not isinstance(item, dict):
            continue
        prices = item.get("prices")
        if not isinstance(prices, dict):
            continue
        clean_prices = {}
        for raw_symbol, raw_price in prices.items():
            try:
                price = float(raw_price)
            except (TypeError, ValueError):
                continue
            symbol = str(raw_symbol or "").strip().upper()
            if symbol and price > 0:
                clean_prices[symbol] = price
        if not clean_prices:
            continue
        row = {
            "bar": int(float(item.get("bar", idx) or idx)),
            "regime": str(item.get("regime", "") or ""),
            "market": str(item.get("market", "") or ""),
            "prices": clean_prices,
        }
        timestamp = (
            item.get("timestamp")
            or item.get("timestamp_utc")
            or item.get("time")
            or item.get("datetime")
        )
        if timestamp:
            row["timestamp"] = str(timestamp)
        out.append(row)
    return out


def _fill_price_history_timestamps(history: list, status: dict):
    if not history:
        return history
    if all(isinstance(row, dict) and row.get("timestamp") for row in history):
        return history
    last_timestamp = next(
        (
            _parse_timestamp(row.get("timestamp"))
            for row in reversed(history)
            if isinstance(row, dict) and row.get("timestamp")
        ),
        None,
    )
    if last_timestamp is None:
        last_timestamp = _parse_timestamp(
            status.get("timestamp")
            or status.get("timestamp_utc")
            or status.get("time")
        )
    if last_timestamp is None:
        return history
    last_bar = int(_float_value(status.get("bar_count"), history[-1].get("bar", 0)))
    if last_bar <= 0:
        last_bar = int(_float_value(history[-1].get("bar"), 0))
    interval = _timeframe_seconds(status.get("timeframe"))
    out = []
    for row in history:
        row = dict(row)
        if not row.get("timestamp"):
            bar = int(_float_value(row.get("bar"), last_bar))
            row["timestamp"] = (
                last_timestamp - timedelta(seconds=max(0, last_bar - bar) * interval)
            ).isoformat()
        out.append(row)
    return out


def _parse_timestamp(value):
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


def _timeframe_seconds(value):
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


def _session_visuals(exchange: str, session_name: str):
    prefix = f"{exchange}/{session_name}"
    return {
        "dashboard_latest": f"{prefix}/dashboard_latest.png",
        "shadow": f"{prefix}/shadow_dashboard.png",
        "regime": f"{prefix}/regime_dashboard.png",
        "memory": f"{prefix}/memory_dashboard.png",
    }


def _float_value(value, default=0.0):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed == parsed else default


def _configured_pool_from_dashboard(path: Path):
    rows = {"players": [], "agents": []}
    if not path.exists():
        return rows
    mode = ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        stripped = line.strip()
        if stripped == "Players:":
            mode = "players"
            continue
        if stripped == "Agents:":
            mode = "agents"
            continue
        if stripped.startswith("="):
            mode = ""
            continue
        if not mode or not stripped or stripped.startswith("-"):
            continue
        parts = stripped.split()
        if not parts or parts[0] in {"Label", "Pnl%", "none"}:
            continue
        if mode == "players" and len(parts) >= 2:
            rows["players"].append({
                "label": parts[0],
                "kind": parts[1],
                "scope": parts[2] if len(parts) >= 3 else "",
                "agents": " ".join(parts[3:]) if len(parts) >= 4 else "",
            })
        elif mode == "agents":
            rows["agents"].append({
                "label": parts[0],
                "flags": parts[1] if len(parts) >= 2 else "",
            })
    return rows


def _configured_leaderboard_row(pool_row: dict, kind: str):
    return {
        "pnl_pct": 0.0,
        "session_pnl_pct": 0.0,
        "closed_trades": 0,
        "session_closed_trades": 0,
        "entries": 0,
        "session_entries": 0,
        "signals": 0,
        "session_signals": 0,
        "wins": 0,
        "session_wins": 0,
        "losses": 0,
        "session_losses": 0,
        "win_rate": 0.0,
        "session_win_rate": 0.0,
        "sharpe": 0.0,
        "max_drawdown_pct": 0.0,
        "configured_pool": True,
        "configured_pool_only": True,
        "actor_pool_kind": str(pool_row.get("kind") or kind),
        "status_reason": "configured actor pool; no leaderboard activity yet",
        "per_regime": {},
    }


def _merge_configured_pool(status: dict, players: dict, agents: dict, dashboard_path: Path):
    players = dict(players or {})
    agents = dict(agents or {})
    configured = status.get("configured_actor_pool") if isinstance(status, dict) else {}
    if isinstance(configured, dict):
        pool_players = list(configured.get("players") or [])
        pool_agents = list(configured.get("agents") or [])
    else:
        pool_players = []
        pool_agents = []
    fallback = _configured_pool_from_dashboard(dashboard_path)
    pool_players.extend(fallback["players"])
    pool_agents.extend(fallback["agents"])
    for row in pool_players:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        if label:
            players.setdefault("V_" + label, _configured_leaderboard_row(row, "player"))
    for row in pool_agents:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        if label:
            agents.setdefault("V_" + label, _configured_leaderboard_row(row, "agent"))
    return players, agents


def _split_visual_agents_from_strategy_players(players: dict, agents: dict):
    players = dict(players or {})
    agents = dict(agents or {})
    if agents:
        return players, agents
    visual_players = {}
    visual_agents = {}
    for name, row in players.items():
        row_data = dict(row) if isinstance(row, dict) else {}
        kind = str(row_data.get("actor_pool_kind") or row_data.get("kind") or "").strip().lower()
        if kind == "strategy":
            row_data["visual_agent_fallback"] = True
            visual_agents[name] = row_data
        else:
            visual_players[name] = row_data
    if not visual_agents:
        return players, agents
    return visual_players, visual_agents


def _actor_asset_value(row: dict, initial: float):
    for key in ("equity", "asset", "assets", "total_assets_usd"):
        value = _float_value(row.get(key), 0.0)
        if value > 0.0:
            return value
    pnl = _float_value(row.get("session_pnl_pct", row.get("pnl_pct", 0.0)), 0.0)
    return max(0.0, initial * (1.0 + pnl / 100.0))


def _asset_rows(summary: dict, players: dict, agents: dict):
    kind_order = {"panteon": 0, "player": 1, "agent": 2}
    initial = _float_value(summary.get("initial_capital"), 0.0)
    if initial <= 0.0:
        initial = _float_value(summary.get("current_balance"), 100.0) or 100.0
    panteon_asset = _float_value(
        summary.get(
            "panteon_equity_usd",
            summary.get("total_assets_usd", summary.get("current_balance", initial)),
        ),
        initial,
    ) or initial
    rows = [{"name": "PANTEON", "kind": "panteon", "asset": panteon_asset}]
    for name, row in (players or {}).items():
        clean = str(name).replace("V_", "", 1)
        rows.append({
            "name": "P:" + clean,
            "kind": "player",
            "asset": _actor_asset_value(row if isinstance(row, dict) else {}, initial),
        })
    for name, row in (agents or {}).items():
        clean = str(name).replace("V_", "", 1)
        rows.append({
            "name": "A:" + clean,
            "kind": "agent",
            "asset": _actor_asset_value(row if isinstance(row, dict) else {}, initial),
        })
    return sorted(
        rows,
        key=lambda row: (
            -float(row["asset"]),
            kind_order.get(str(row["kind"]), 99),
            str(row["name"]),
        ),
    )


def _tail_lines(path: Path, max_lines: int = 120, max_bytes: int = 48 * 1024 * 1024):
    if not path.exists() or max_lines <= 0:
        return []
    data = b""
    chunk = 1024 * 1024
    read_total = 0
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            while pos > 0 and data.count(b"\n") <= max_lines and read_total < max_bytes:
                step = min(chunk, pos, max_bytes - read_total)
                pos -= step
                fh.seek(pos)
                data = fh.read(step) + data
                read_total += step
    except OSError:
        return []
    lines = data.splitlines()
    return [line.decode("utf-8", errors="replace") for line in lines[-max_lines:]]


def _price_history_from_causal_log(path: Path, limit: int = 120):
    rows = []
    for line in _tail_lines(path, max_lines=limit):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        prices = item.get("prices")
        if not isinstance(prices, dict):
            continue
        rows.append({
            "bar": item.get("bar", len(rows)),
            "timestamp": item.get("timestamp") or item.get("timestamp_utc") or item.get("time") or "",
            "regime": item.get("regime", ""),
            "market": item.get("market", item.get("regime", "")),
            "prices": prices,
        })
    return _clean_price_history(rows, limit=limit)


def _collect_sessions(exchange: str, results_dir: Path = RESULTS, sessions_limit: int = SESSIONS_LIMIT):
    base = Path(results_dir) / exchange
    if not base.exists():
        return []
    sessions = sorted(
        [p for p in base.iterdir() if p.is_dir() and SESSION_DIR_RE.match(p.name)]
    )
    sessions = sessions[-sessions_limit:]
    out = []
    for idx, sdir in enumerate(reversed(sessions)):  # newest first
        status = _safe_read(sdir / "status.json") or {}
        players = _safe_read(sdir / "leaderboard_players.json") or {}
        agents = _safe_read(sdir / "leaderboard_agents.json") or {}
        meta = (players.get("metadata") if isinstance(players, dict) else {}) or {}
        player_rows = (players.get("players") if isinstance(players, dict) else {}) or {}
        agent_rows = (agents.get("agents") if isinstance(agents, dict) else {}) or {}
        player_rows, agent_rows = _merge_configured_pool(
            status,
            player_rows,
            agent_rows,
            sdir / "dashboard.txt",
        )
        player_rows, agent_rows = _split_visual_agents_from_strategy_players(
            player_rows,
            agent_rows,
        )
        price_history = _clean_price_history(status.get("price_history", []))
        price_history = _fill_price_history_timestamps(price_history, status)
        if not price_history and idx == 0:
            price_history = _price_history_from_causal_log(
                sdir / "causal_entry_decisions.jsonl",
            )
            price_history = _fill_price_history_timestamps(price_history, status)
        summary = {
                "version": status.get("version", "v1"),
                "uptime": status.get("uptime", ""),
                "bar_count": status.get("bar_count", 0),
                "live_bar_count": status.get("live_bar_count", 0),
                "initial_capital": status.get("initial_capital", 0),
                "current_balance": status.get("current_balance", 0),
                "pnl_pct": status.get("pnl_pct", 0),
                "max_drawdown_pct": status.get("max_drawdown_pct", 0),
                "n_signals": status.get("n_signals", 0),
                "n_trades": status.get("n_trades", 0),
                "regime": meta.get("regime", "?"),
                "real_pnl_pct": meta.get("real_pnl_pct", status.get("pnl_pct", 0)),
                "panteon_equity_usd": status.get("panteon_equity_usd", status.get("current_balance", 0)),
                "panteon_realized_equity_usd": status.get("panteon_realized_equity_usd", status.get("current_balance", 0)),
                "total_assets_usd": status.get("total_assets_usd", status.get("current_balance", 0)),
                "market": status.get("market", status.get("regime", "?")),
                "regime_confidence": status.get("regime_confidence"),
                "timestamp": status.get("timestamp", ""),
        }
        out.append({
            "name": sdir.name,
            "summary": summary,
            "open_positions": status.get("open_positions", {}) or {},
            "players": player_rows,
            "agents": agent_rows,
            "regime_history": meta.get("regime_history", []) or [],
            "price_history": price_history,
            "asset_rows": _asset_rows(summary, player_rows, agent_rows),
            "assets_curve": status.get("assets_curve", []) or [],
            "panteon_equity_curve": status.get("panteon_equity_curve", []) or [],
            "market_view": status.get("market_view", {}) or {},
            "visuals": _session_visuals(exchange, sdir.name),
        })
    return out


def _payload(results_dir: Path = RESULTS, sessions_limit: int = SESSIONS_LIMIT):
    data = {
        ex: _collect_sessions(ex, results_dir=results_dir, sessions_limit=sessions_limit)
        for ex in EXCHANGES
    }
    return data


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Panteon — сессии и агенты</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
         Helvetica, Arial, sans-serif; margin: 0; padding: 20px;
         background: #0f1419; color: #d6deeb; }
  h1 { margin: 0 0 8px 0; font-size: 22px; color: #e6f1ff; }
  h2 { margin-top: 28px; font-size: 18px; color: #82aaff; border-bottom: 1px solid #1d2837; padding-bottom: 4px; }
  .controls { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin: 12px 0 18px; }
  select, button { background: #1d2937; color: #d6deeb; border: 1px solid #2c3e50;
                   padding: 6px 10px; border-radius: 6px; font-size: 14px; }
  select:hover, button:hover { background: #25324a; cursor: pointer; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
          font-size: 11px; font-weight: 600; }
  .pill.live        { background: #1f3a2b; color: #76e6a3; }
  .pill.shadow_only { background: #3a2f1f; color: #ffcb6b; }
  .pill.quarantine  { background: #3f1f25; color: #ff7986; }
  .pill.experimental { background: #2c2640; color: #c792ea; }
  .pill.purgatory   { background: #3d2620; color: #ff9974; }
  .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
             gap: 12px; margin-bottom: 18px; }
  .card { background: #131a25; border: 1px solid #1d2837; border-radius: 10px;
          padding: 12px 16px; }
  .card .label { font-size: 11px; color: #8c9bb1; text-transform: uppercase; letter-spacing: 0.04em; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .pos { color: #76e6a3; }
  .neg { color: #ff7986; }
  .muted { color: #8c9bb1; }
  tbody tr.inactive-row td,
  tbody tr.inactive-row .pos,
  tbody tr.inactive-row .neg { color: #8c9bb1; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
  th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #1d2837; }
  th:first-child, td:first-child { text-align: left; }
  th { color: #82aaff; font-weight: 500; cursor: pointer; user-select: none;
       background: #131a25; position: sticky; top: 0; }
  tbody tr:hover { background: #131a25; }
  .filters { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0; }
  .filter-btn { padding: 4px 10px; font-size: 12px; border-radius: 999px;
                background: #1d2937; color: #d6deeb; border: 1px solid #2c3e50; cursor: pointer; }
  .filter-btn.active { background: #2c456b; color: #fff; border-color: #82aaff; }
  .open-positions { margin-top: 12px; }
  .open-positions table { font-size: 12px; }
  .chart-panel { background: #111821; border: 1px solid #1d2837; border-radius: 10px;
                 padding: 12px; margin-top: 10px; }
  .asset-row { display: grid; grid-template-columns: minmax(170px, 260px) 1fr 90px;
               gap: 10px; align-items: center; margin: 5px 0; }
  .asset-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #d6deeb; }
  .asset-track { height: 14px; background: #0b1118; border-radius: 4px; overflow: hidden; }
  .asset-fill { height: 100%; background: #58a6ff; }
  .asset-row.player .asset-fill { background: #bc8cff; }
  .asset-row.panteon .asset-fill { background: #f0c040; }
  .asset-row.panteon .asset-label, .asset-row.panteon .asset-value { font-weight: 800; color: #f0c040; }
  .asset-value { text-align: right; color: #8c9bb1; font-variant-numeric: tabular-nums; }
  #price-chart { width: 100%; height: 260px; display: block; }
  .legend-chip { display: inline-flex; align-items: center; gap: 6px; margin: 4px 12px 0 0;
                 color: #d6deeb; font-size: 12px; }
  .legend-swatch { width: 18px; height: 3px; display: inline-block; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }
  @media (max-width: 980px) { .grid2 { grid-template-columns: 1fr; } }
  .reason { color: #8c9bb1; font-size: 11px; }
</style>
</head>
<body>
<h1>Panteon — анализ сессий</h1>
<div class="muted" id="dataset-info"></div>

<div class="controls">
  <label>Биржа:
    <select id="exchange"></select>
  </label>
  <label>Сессия:
    <select id="session"></select>
  </label>
  <button id="prev-session">‹ предыдущая</button>
  <button id="next-session">следующая ›</button>
</div>

<h2>Сессия — сводка</h2>
<div class="summary" id="summary"></div>

<h2>Открытые позиции</h2>
<div class="open-positions" id="open-positions"></div>

<h2>Assets: Panteon, Players, Agents</h2>
<div id="assets-chart" class="chart-panel"></div>

<h2>Currency Prices And Current Market Type</h2>
<div class="chart-panel">
  <div id="price-market-note" class="muted"></div>
  <canvas id="price-chart"></canvas>
  <div id="price-legend"></div>
</div>

<div class="grid2">
  <div>
    <h2>Игроки (ансамбли)</h2>
    <div class="filters" id="players-filters"></div>
    <table id="players-table"><thead></thead><tbody></tbody></table>
  </div>
  <div>
    <h2>Агенты</h2>
    <div class="filters" id="agents-filters"></div>
    <table id="agents-table"><thead></thead><tbody></tbody></table>
  </div>
</div>

<h2>Кросс-биржевая сводка по агентам (общие имена)</h2>
<table id="cross-table"><thead></thead><tbody></tbody></table>

<h2>История режимов</h2>
<div id="regime-history" class="muted"></div>

<script>
// Embedded data — подменяется build_dashboard.py
const DATA = __DATA_JSON__;

const $ = sel => document.querySelector(sel);
const escape = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

function pill(status) {
  const s = String(status || "live").toLowerCase();
  const cls = ["live","shadow_only","quarantine","experimental","purgatory"].includes(s) ? s : "live";
  return `<span class="pill ${cls}">${s}</span>`;
}

function isInactiveStatus(status) {
  const s = String(status || "").toLowerCase();
  return ["quarantine", "shadow_only", "purgatory"].includes(s);
}

function pnlCls(v, status) {
  if (isInactiveStatus(status)) return "muted";
  v = parseFloat(v) || 0;
  return v > 0.05 ? "pos" : v < -0.05 ? "neg" : "muted";
}

function fmt(n, digits=2) {
  if (n === null || n === undefined || Number.isNaN(parseFloat(n))) return "-";
  return parseFloat(n).toFixed(digits);
}

const TableState = { players: {sortBy: "pnl_pct", desc: true, filter: "all"},
                      agents:  {sortBy: "pnl_pct", desc: true, filter: "all"} };

function renderSummary(session) {
  const s = session.summary;
  const pnl = parseFloat(s.real_pnl_pct ?? s.pnl_pct ?? 0);
  const cards = [
    ["PnL %",            fmt(pnl, 3) + "%",                    pnlCls(pnl)],
    ["Баланс",            "$" + fmt(s.current_balance, 2),      ""],
    ["Стартовый капитал", "$" + fmt(s.initial_capital, 2),     "muted"],
    ["Просадка",          fmt(s.max_drawdown_pct, 2) + "%",     "neg"],
    ["Сигналы",           s.n_signals ?? 0,                     ""],
    ["Сделки",            s.n_trades ?? 0,                      ""],
    ["Live-баров",        s.live_bar_count ?? 0,                ""],
    ["Режим",             s.regime ?? "?",                      ""],
    ["Uptime",            s.uptime ?? "-",                       "muted"],
  ];
  $("#summary").innerHTML = cards.map(([l, v, cls]) =>
    `<div class="card"><div class="label">${l}</div>
       <div class="value ${cls}">${v}</div></div>`).join("");
}

function renderOpenPositions(session) {
  const op = session.open_positions || {};
  const syms = Object.keys(op);
  if (!syms.length) {
    $("#open-positions").innerHTML = `<div class="muted">нет открытых позиций</div>`;
    return;
  }
  let total_pnl = 0;
  const rows = syms.map(sym => {
    const p = op[sym];
    total_pnl += parseFloat(p.pnl ?? 0);
    return `<tr>
      <td>${escape(sym)}</td>
      <td>${escape(p.side)}</td>
      <td>${fmt(p.qty, 4)}</td>
      <td>${fmt(p.entry, 6)}</td>
      <td class="${pnlCls(p.pnl)}">${fmt(p.pnl, 4)} USDT</td>
      <td>${escape(p.type ?? "futures")}</td>
    </tr>`;
  }).join("");
  $("#open-positions").innerHTML = `
    <table>
      <thead><tr><th>Символ</th><th>Сторона</th><th>qty</th><th>entry</th><th>PnL</th><th>тип</th></tr></thead>
      <tbody>${rows}</tbody>
      <tfoot><tr><td colspan="4" class="muted">Σ unrealized PnL</td>
        <td class="${pnlCls(total_pnl)}"><b>${fmt(total_pnl, 4)} USDT</b></td><td></td></tr></tfoot>
    </table>`;
}

function actorAssetValue(row, initial) {
  const direct = parseFloat(row.equity ?? row.asset ?? row.assets ?? row.total_assets_usd);
  if (Number.isFinite(direct) && direct > 0) return direct;
  const pnl = parseFloat(row.session_pnl_pct ?? row.pnl_pct ?? 0) || 0;
  return Math.max(0, initial * (1 + pnl / 100));
}

function renderAssetsChart(session) {
  const s = session.summary || {};
  const initial = parseFloat(s.initial_capital || s.current_balance || 100) || 100;
  const rows = Array.isArray(session.asset_rows) && session.asset_rows.length ? session.asset_rows.slice() : [{
    name: "PANTEON",
    kind: "panteon",
    asset: parseFloat(s.panteon_equity_usd || s.total_assets_usd || s.current_balance || initial) || initial,
  }];
  if (!Array.isArray(session.asset_rows) || !session.asset_rows.length) {
    for (const [name, row] of Object.entries(session.players || {})) {
      rows.push({name: "P:" + String(name).replace(/^V_/, ""), kind: "player", asset: actorAssetValue(row, initial)});
    }
    for (const [name, row] of Object.entries(session.agents || {})) {
      rows.push({name: "A:" + String(name).replace(/^V_/, ""), kind: "agent", asset: actorAssetValue(row, initial)});
    }
  }
  const kindOrder = row => row.kind === "panteon" ? 0 : row.kind === "player" ? 1 : row.kind === "agent" ? 2 : 99;
  const ordered = rows.sort((a, b) =>
    (parseFloat(b.asset || 0) - parseFloat(a.asset || 0)) ||
    (kindOrder(a) - kindOrder(b)) ||
    String(a.name).localeCompare(String(b.name))
  );
  const maxAsset = Math.max(...ordered.map(r => r.asset), 1);
  $("#assets-chart").innerHTML = ordered.map(row => {
    const width = Math.max(1, Math.min(100, row.asset / maxAsset * 100));
    return `<div class="asset-row ${row.kind}">
      <div class="asset-label" title="${escape(row.name)}">${escape(row.name)}</div>
      <div class="asset-track"><div class="asset-fill" style="width:${width.toFixed(2)}%"></div></div>
      <div class="asset-value">$${fmt(row.asset, 2)}</div>
    </div>`;
  }).join("");
}

function priceSymbols(history, limit=6) {
  const symbols = new Set();
  for (const row of history) {
    for (const sym of Object.keys(row.prices || {})) symbols.add(sym);
  }
  const ranked = Array.from(symbols).map(sym => {
    const values = history.map(row => parseFloat((row.prices || {})[sym])).filter(v => Number.isFinite(v) && v > 0);
    const first = values[0] || 0, last = values[values.length - 1] || first;
    const move = first > 0 ? Math.abs(last / first - 1) : 0;
    return {sym, move};
  }).sort((a, b) => b.move - a.move || a.sym.localeCompare(b.sym));
  return ranked.slice(0, limit).map(x => x.sym);
}

function renderPriceChart(session) {
  const history = (session.price_history || []).filter(row => row && row.prices);
  const market = session.summary?.market || session.summary?.regime || "?";
  const confidence = session.summary?.regime_confidence;
  $("#price-market-note").textContent =
    `market=${market}` + (confidence === null || confidence === undefined ? "" : ` confidence=${fmt(confidence, 2)}`);
  const canvas = $("#price-chart");
  const legend = $("#price-legend");
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(900, Math.floor(rect.width || 900));
  canvas.height = 260;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (history.length < 2) {
    ctx.fillStyle = "#8c9bb1";
    ctx.font = "14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.fillText("Price history will appear after two live bars.", 24, 130);
    legend.innerHTML = "";
    return;
  }
  const symbols = priceSymbols(history);
  const colors = ["#58a6ff", "#bc8cff", "#3fb950", "#f0c040", "#ff7b72", "#79c0ff"];
  const pad = {l: 45, r: 20, t: 18, b: 32};
  const timeValues = history.map(row => Date.parse(row.timestamp || row.timestamp_utc || row.time || ""));
  const hasTimeAxis = timeValues.length > 1 && timeValues.every(v => Number.isFinite(v));
  const series = symbols.map((sym, idx) => {
    const raw = history.map(row => parseFloat((row.prices || {})[sym]));
    const first = raw.find(v => Number.isFinite(v) && v > 0);
    const points = raw.map(v => Number.isFinite(v) && v > 0 && first ? v / first * 100 : null);
    return {sym, color: colors[idx % colors.length], points};
  });
  const vals = series.flatMap(s => s.points).filter(v => Number.isFinite(v));
  const minY = Math.min(...vals, 99), maxY = Math.max(...vals, 101);
  const spanY = Math.max(1, maxY - minY);
  const minX = hasTimeAxis ? Math.min(...timeValues) : 0;
  const maxX = hasTimeAxis ? Math.max(...timeValues) : Math.max(1, history.length - 1);
  const spanX = Math.max(1, maxX - minX);
  const xAt = i => {
    if (hasTimeAxis) return pad.l + (canvas.width - pad.l - pad.r) * ((timeValues[i] - minX) / spanX);
    return pad.l + (canvas.width - pad.l - pad.r) * (i / Math.max(1, history.length - 1));
  };
  const yAt = v => pad.t + (canvas.height - pad.t - pad.b) * (1 - (v - minY) / spanY);
  ctx.strokeStyle = "#30363d";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (canvas.height - pad.t - pad.b) * (i / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(canvas.width - pad.r, y); ctx.stroke();
  }
  ctx.fillStyle = "#8c9bb1";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.fillText(minY.toFixed(1), 6, yAt(minY));
  ctx.fillText(maxY.toFixed(1), 6, yAt(maxY) + 4);
  const xLabelIndexes = [0, Math.floor((history.length - 1) / 2), history.length - 1]
    .filter((v, i, arr) => arr.indexOf(v) === i);
  for (const i of xLabelIndexes) {
    const label = hasTimeAxis
      ? new Date(timeValues[i]).toLocaleString("ru-RU", {month:"2-digit", day:"2-digit", hour:"2-digit", minute:"2-digit"})
      : `bar ${history[i].bar ?? i}`;
    ctx.fillText(label, Math.min(canvas.width - 112, Math.max(pad.l, xAt(i) - 34)), canvas.height - 10);
  }
  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    s.points.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      if (!started) { ctx.moveTo(xAt(i), yAt(v)); started = true; }
      else ctx.lineTo(xAt(i), yAt(v));
    });
    if (started) ctx.stroke();
  }
  legend.innerHTML = series.map(s =>
    `<span class="legend-chip"><span class="legend-swatch" style="background:${s.color}"></span>${escape(s.sym)}</span>`
  ).join("");
}

function buildTable(kind, session) {
  const data = session[kind] || {};
  const items = Object.entries(data).map(([name, d]) => ({name, ...d}));
  const state = TableState[kind];
  const filtered = state.filter === "all"
    ? items
    : items.filter(x => (x.status || "live") === state.filter);
  filtered.sort((a, b) => {
    const av = a[state.sortBy], bv = b[state.sortBy];
    if (typeof av === "number" || typeof bv === "number") {
      return state.desc ? (parseFloat(bv) || 0) - (parseFloat(av) || 0)
                         : (parseFloat(av) || 0) - (parseFloat(bv) || 0);
    }
    return state.desc ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv));
  });

  const cols = [
    ["name",             "Имя"],
    ["status",           "Status"],
    ["pnl_pct",          "PnL %"],
    ["win_rate",         "WR"],
    ["sharpe",           "Sharpe"],
    ["closed_trades",    "Closed"],
    ["signals",          "Sigs"],
    ["entries",          "Entr"],
    ["max_drawdown_pct", "MDD%"],
    ["equity",           "Equity"],
  ];
  const tbl = $("#" + kind + "-table");
  tbl.querySelector("thead").innerHTML =
    `<tr>${cols.map(([k, l]) => `<th data-col="${k}">${l}</th>`).join("")}</tr>`;
  tbl.querySelector("thead").querySelectorAll("th").forEach(th => {
    th.onclick = () => {
      const col = th.dataset.col;
      if (state.sortBy === col) state.desc = !state.desc;
      else { state.sortBy = col; state.desc = true; }
      buildTable(kind, session);
    };
  });
  tbl.querySelector("tbody").innerHTML = filtered.map(r => {
    return `<tr class="${isInactiveStatus(r.status) ? "inactive-row" : ""}">
      <td>${escape((r.name || "").replace("V_", ""))}
        ${r.status_reason ? `<div class="reason">${escape(r.status_reason)}</div>` : ""}
      </td>
      <td>${pill(r.status)}</td>
      <td class="${pnlCls(r.pnl_pct, r.status)}">${fmt(r.pnl_pct, 3)}%</td>
      <td>${fmt(r.win_rate, 1)}%</td>
      <td>${fmt(r.sharpe, 2)}</td>
      <td>${r.closed_trades ?? 0}</td>
      <td>${r.signals ?? 0}</td>
      <td>${r.entries ?? 0}</td>
      <td class="muted">${fmt(r.max_drawdown_pct, 2)}%</td>
      <td class="muted">${fmt(r.equity, 2)}</td>
    </tr>`;
  }).join("");
}

function renderFilters(kind, session) {
  const data = session[kind] || {};
  const statuses = Array.from(new Set(["all", ...Object.values(data).map(x => x.status || "live")]));
  const state = TableState[kind];
  $("#" + kind + "-filters").innerHTML = statuses.map(s =>
    `<button class="filter-btn ${state.filter === s ? "active" : ""}" data-status="${s}">${s}</button>`).join("");
  $("#" + kind + "-filters").querySelectorAll(".filter-btn").forEach(b => {
    b.onclick = () => { state.filter = b.dataset.status; renderFilters(kind, session); buildTable(kind, session); };
  });
}

function renderRegimeHistory(session) {
  const hist = session.regime_history || [];
  if (!hist.length) {
    $("#regime-history").innerHTML = "—";
    return;
  }
  $("#regime-history").innerHTML = hist.map(t =>
    `<span style="margin-right:14px;">bar ${t.bar}: <b>${escape(t.from)}</b> → <b>${escape(t.to)}</b>
      <span class="muted">(${escape(t.time || "")})</span></span>`).join("");
}

function renderCross() {
  const ex = Object.keys(DATA);
  if (ex.length < 2) { $("#cross-table").innerHTML = ""; return; }
  // Берём САМУЮ свежую сессию каждой биржи
  const latest = {};
  for (const e of ex) {
    if (DATA[e] && DATA[e].length) latest[e] = DATA[e][0];
  }
  const allAgents = new Set();
  for (const e of ex) {
    Object.keys(latest[e]?.agents || {}).forEach(n => allAgents.add(n));
  }
  const rows = Array.from(allAgents).map(name => {
    const cells = ex.map(e => latest[e]?.agents?.[name] || {});
    return {name, ...Object.fromEntries(ex.map((e, i) => [e, cells[i]]))};
  });
  rows.sort((a, b) => {
    const sumA = ex.reduce((s, e) => s + (parseFloat(a[e]?.pnl_pct) || 0), 0);
    const sumB = ex.reduce((s, e) => s + (parseFloat(b[e]?.pnl_pct) || 0), 0);
    return sumB - sumA;
  });
  const tbl = $("#cross-table");
  tbl.querySelector("thead").innerHTML =
    `<tr><th>Agent</th><th>Status</th>` +
    ex.map(e => `<th>${e} pnl%</th><th>${e} WR</th><th>${e} Sh</th><th>${e} closed</th>`).join("") + `</tr>`;
  tbl.querySelector("tbody").innerHTML = rows.map(r => {
    const status = ex.map(e => r[e]?.status).find(Boolean) || "live";
    return `<tr>
      <td>${escape((r.name || "").replace("V_", ""))}</td>
      <td>${pill(status)}</td>` +
      ex.map(e => {
        const c = r[e] || {};
        return `<td class="${pnlCls(c.pnl_pct, c.status)}">${fmt(c.pnl_pct, 3)}</td>
                <td>${fmt(c.win_rate, 1)}%</td>
                <td>${fmt(c.sharpe, 2)}</td>
                <td class="muted">${c.closed_trades ?? 0}</td>`;
      }).join("") + `</tr>`;
  }).join("");
}

function renderAll() {
  const exchange = $("#exchange").value;
  const sessions = DATA[exchange] || [];
  const sel = $("#session");
  const cur = sel.value;
  sel.innerHTML = sessions.map(s => `<option value="${s.name}">${s.name}</option>`).join("");
  if (cur && sessions.find(s => s.name === cur)) sel.value = cur;
  const session = sessions.find(s => s.name === sel.value) || sessions[0];
  if (!session) {
    $("#summary").innerHTML = `<div class="card"><div class="muted">нет сессий</div></div>`;
    return;
  }
  renderSummary(session);
  renderOpenPositions(session);
  renderAssetsChart(session);
  renderPriceChart(session);
  renderFilters("players", session);
  renderFilters("agents", session);
  buildTable("players", session);
  buildTable("agents", session);
  renderRegimeHistory(session);
  renderCross();
}

(function init() {
  $("#exchange").innerHTML = Object.keys(DATA).map(e => `<option>${e}</option>`).join("");
  $("#exchange").onchange = renderAll;
  $("#session").onchange  = renderAll;
  $("#prev-session").onclick = () => {
    const sel = $("#session"), idx = sel.selectedIndex;
    if (idx + 1 < sel.options.length) { sel.selectedIndex = idx + 1; renderAll(); }
  };
  $("#next-session").onclick = () => {
    const sel = $("#session"), idx = sel.selectedIndex;
    if (idx > 0) { sel.selectedIndex = idx - 1; renderAll(); }
  };
  const total = Object.values(DATA).reduce((s, arr) => s + arr.length, 0);
  $("#dataset-info").textContent = `Сессий загружено: ${total}, биржи: ${Object.keys(DATA).join(", ")}`;
  renderAll();
})();
</script>
</body>
</html>
"""


def build_dashboard(
    *,
    results_dir: Path = RESULTS,
    out_html: Path = OUT_HTML,
    sessions_limit: int = SESSIONS_LIMIT,
    quiet: bool = False,
):
    results_dir = Path(results_dir)
    out_html = Path(out_html)
    data = _payload(results_dir=results_dir, sessions_limit=sessions_limit)
    payload_json = json.dumps(data, ensure_ascii=False, default=str)
    html = HTML_TEMPLATE.replace("__DATA_JSON__", payload_json)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    tmp_html = out_html.with_name(f".{out_html.name}.tmp")
    tmp_html.write_text(html, encoding="utf-8")
    try:
        os.replace(tmp_html, out_html)
    except PermissionError:
        shutil.copyfile(tmp_html, out_html)
        try:
            tmp_html.unlink()
        except OSError:
            pass
    if not quiet:
        print(f"[ok] dashboard written: {out_html}")
        print(f"[ok] sessions: " + ", ".join(f"{ex}={len(sessions)}" for ex, sessions in data.items()))
    return data


def main():
    build_dashboard()


if __name__ == "__main__":
    sys.exit(main() or 0)
