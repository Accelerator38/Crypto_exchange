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
import sys
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


def _collect_sessions(exchange: str, results_dir: Path = RESULTS, sessions_limit: int = SESSIONS_LIMIT):
    base = Path(results_dir) / exchange
    if not base.exists():
        return []
    sessions = sorted(
        [p for p in base.iterdir() if p.is_dir() and SESSION_DIR_RE.match(p.name)]
    )
    sessions = sessions[-sessions_limit:]
    out = []
    for sdir in reversed(sessions):  # newest first
        status = _safe_read(sdir / "status.json") or {}
        players = _safe_read(sdir / "leaderboard_players.json") or {}
        agents = _safe_read(sdir / "leaderboard_agents.json") or {}
        meta = (players.get("metadata") if isinstance(players, dict) else {}) or {}
        out.append({
            "name": sdir.name,
            "summary": {
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
            },
            "open_positions": status.get("open_positions", {}) or {},
            "players": (players.get("players") if isinstance(players, dict) else {}) or {},
            "agents": (agents.get("agents") if isinstance(agents, dict) else {}) or {},
            "regime_history": meta.get("regime_history", []) or [],
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

function pnlCls(v) {
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
    return `<tr>
      <td>${escape((r.name || "").replace("V_", ""))}
        ${r.status_reason ? `<div class="reason">${escape(r.status_reason)}</div>` : ""}
      </td>
      <td>${pill(r.status)}</td>
      <td class="${pnlCls(r.pnl_pct)}">${fmt(r.pnl_pct, 3)}%</td>
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
        return `<td class="${pnlCls(c.pnl_pct)}">${fmt(c.pnl_pct, 3)}</td>
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
    os.replace(tmp_html, out_html)
    if not quiet:
        print(f"[ok] dashboard written: {out_html}")
        print(f"[ok] sessions: " + ", ".join(f"{ex}={len(sessions)}" for ex, sessions in data.items()))
    return data


def main():
    build_dashboard()


if __name__ == "__main__":
    sys.exit(main() or 0)
