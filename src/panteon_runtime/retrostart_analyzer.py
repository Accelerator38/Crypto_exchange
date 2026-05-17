"""
Retrostart Analyzer
═══════════════════

Ретроспективный анализ реальных торговых сессий Пантеона.

Зачем: live-сессии пишут в Results/{MEXC,BITGET}/<timestamp>/ полный набор
артефактов (trading.log, all_signals.csv, leaderboard_*.json,
final_combo_report.json, status.json). Этот модуль "прогоняет" эти данные
ретроспективно — то есть СВОДИТ И ДИАГНОСТИРУЕТ работу реального Пантеона
относительно его шэдоу-аналогов и составляющих агентов.

Что он определяет:
  1. Длительность сессии — фильтр >3 часов trading-времени.
  2. Реальный P&L против лучших/средних shadow-агентов и shadow-players.
  3. Дельта real vs V_Panteon_shadow (тот же код, но без слиппеджа/комиссий).
  4. Дубликаты open-сигналов (sym + одинаковая сторона в close-window <2 минут)
     — типичный симптом стэкинга позиций до фикса фильтра в _act_via_shadow_players.
  5. Сигналы "close" при отсутствии позиции.
  6. Согласованность regime между agent-leaderboard и player-leaderboard.
  7. Частота переключений регима (regime flapping).

Выдаёт:
  - Печать в stdout таблицы по каждой сессии + агрегированный вердикт.
  - JSON-файл с детальным разбором в Results/{EXCHANGE}/_retrostart_report.json.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


MIN_DURATION_HOURS = 3.0
COMBINE_MAX_GAP_HOURS = 2.0
DUP_OPEN_WINDOW_BARS = 5  # окно между сигналами одной стороны для подсчёта стэкинга
OPEN_ACTIONS = (1, 2, 4, 5)
CLOSE_ACTIONS = (3, 8)
SHORT_ACTIONS = (6, 7)
DEFAULT_TRADE_FRACTION = 0.10
DEFAULT_LEVERAGE = 3.0
DEFAULT_FUTURE_FEE = 0.0002
DEFAULT_SLIPPAGE = 0.0001


# ── Парсинг длительности ─────────────────────────────────────────────────────

def _parse_duration(s: Any) -> float:
    """Парсит строку "HH:MM:SS" или "Xh Ym" в часы float."""
    if s is None:
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, sec = parts
                return int(h) + int(m) / 60.0 + int(sec) / 3600.0
            if len(parts) == 2:
                h, m = parts
                return int(h) + int(m) / 60.0
        except ValueError:
            return 0.0
    # Формат "Xh Ym"
    h_match = re.search(r"(\d+)\s*h", s)
    m_match = re.search(r"(\d+)\s*m", s)
    if h_match or m_match:
        return (int(h_match.group(1)) if h_match else 0) + (int(m_match.group(1)) / 60.0 if m_match else 0)
    return 0.0


# ── Загрузка артефактов сессии ───────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _iter_signals(csv_path: Path):
    """Yield dicts из all_signals.csv, безопасно к ошибкам."""
    if not csv_path.exists():
        return
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row
    except (OSError, csv.Error):
        return


def _parse_log_times(log_path: Path) -> List[datetime]:
    """Return timestamps from trading.log lines."""
    if not log_path.exists():
        return []
    times: List[datetime] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if len(line) < 19:
                    continue
                try:
                    times.append(datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
    except OSError:
        return []
    return times


def _parse_iso_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _naive_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def _parse_session_name_time(name: str) -> Optional[datetime]:
    match = re.match(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", name or "")
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _infer_session_bounds(
    session_dir: Path,
    duration_hours: float,
    status: Optional[dict],
    log_path: Path,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Infer comparable session bounds for multi-period retro grouping."""
    started: Optional[datetime] = None
    ended: Optional[datetime] = None
    if status:
        started = _naive_datetime(_parse_iso_time(
            status.get("started_at")
            or status.get("start_time")
            or status.get("session_start")
        ))
        ended = _naive_datetime(_parse_iso_time(
            status.get("timestamp")
            or status.get("updated_at")
            or status.get("last_update")
        ))

    times = _parse_log_times(log_path)
    if len(times) >= 2:
        started = started or times[0]
        ended = ended or times[-1]

    started = started or _parse_session_name_time(session_dir.name)
    if started and not ended and duration_hours > 0:
        ended = started + timedelta(hours=duration_hours)
    if ended and not started and duration_hours > 0:
        started = ended - timedelta(hours=duration_hours)
    return started, ended


def _infer_status_duration_hours(status_path: Path) -> float:
    status = _load_json(status_path)
    if not status:
        return 0.0
    for key in ("uptime", "duration", "runtime"):
        duration = _parse_duration(status.get(key))
        if duration > 0:
            return duration

    started = _parse_iso_time(
        status.get("started_at")
        or status.get("start_time")
        or status.get("session_start")
    )
    ended = _parse_iso_time(
        status.get("timestamp")
        or status.get("updated_at")
        or status.get("last_update")
    )
    if not started or not ended:
        return 0.0
    if started.tzinfo is not None:
        started = started.replace(tzinfo=None)
    if ended.tzinfo is not None:
        ended = ended.replace(tzinfo=None)
    return max(0.0, (ended - started).total_seconds() / 3600.0)


def _parse_runtime_config(log_path: Path) -> Dict[str, float]:
    """Extract trade sizing and exchange drag settings from trading.log."""
    cfg = {
        "trade_fraction": DEFAULT_TRADE_FRACTION,
        "leverage": DEFAULT_LEVERAGE,
        "future_fee": DEFAULT_FUTURE_FEE,
        "slippage": DEFAULT_SLIPPAGE,
    }
    if not log_path.exists():
        return cfg
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return cfg

    tf = re.search(r"trade_fraction\s*=\s*([0-9.]+)", text)
    lev = re.search(r"leverage\s*=\s*([0-9.]+)x", text)
    fees = re.search(r"fees\s*=.*?fut\s*=\s*([0-9.]+).*?slip\s*=\s*([0-9.]+)", text, re.S)
    if tf:
        cfg["trade_fraction"] = float(tf.group(1))
    if lev:
        cfg["leverage"] = float(lev.group(1))
    if fees:
        cfg["future_fee"] = float(fees.group(1))
        cfg["slippage"] = float(fees.group(2))
    return cfg


def _infer_duration_hours(
    report: Optional[dict],
    csv_path: Path,
    log_path: Path,
    status_path: Optional[Path] = None,
) -> Tuple[float, Dict[str, float]]:
    """Infer live trading duration from report, real signal bars and log timestamps."""
    sources: Dict[str, float] = {}
    if report is not None:
        sources["report"] = _parse_duration(report.get("duration"))
    if status_path is not None:
        sources["status_uptime"] = _infer_status_duration_hours(status_path)

    bars: List[int] = []
    for row in _iter_signals(csv_path):
        if str(row.get("is_real", "")).strip().lower() not in ("true", "1"):
            continue
        raw_bar = row.get("live_bar") or row.get("bar")
        try:
            bars.append(int(raw_bar or 0))
        except ValueError:
            continue
    if bars:
        # One live bar is one minute in this runtime.
        sources["signals_live_bar"] = max(0.0, (max(bars) - min(bars)) / 60.0)

    times = _parse_log_times(log_path)
    if len(times) >= 2:
        sources["log_wall_clock"] = max(0.0, (times[-1] - times[0]).total_seconds() / 3600.0)

    for key in ("report", "status_uptime", "log_wall_clock", "signals_live_bar"):
        duration = float(sources.get(key, 0.0) or 0.0)
        if duration > 0:
            return duration, sources
    duration = 0.0
    return duration, sources


def _signal_side(action: int) -> Optional[str]:
    if action in OPEN_ACTIONS:
        return "long"
    if action in SHORT_ACTIONS:
        return "short"
    if action in CLOSE_ACTIONS:
        return "close"
    return None


def _action_size_multiplier(action: int, action_name: str) -> float:
    name = (action_name or "").lower()
    if "half" in name or action in (4, 6):
        return 0.5
    return 1.0


def _replay_real_trade_log(
    csv_path: Path,
    log_path: Path,
    report: Optional[dict],
    apply_risk_multiplier: bool = True,
) -> Dict[str, Any]:
    """
    Replay real trading signals from all_signals.csv as a deterministic retro trade log.

    This is intentionally a conservative one-position-per-symbol replay: duplicate opens are
    counted as rejected, because the Panteon live layer is expected to avoid position stacking.
    """
    if not csv_path.exists():
        return {"checked": False, "reason": "no all_signals.csv"}

    cfg = _parse_runtime_config(log_path)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(_iter_signals(csv_path)):
        if str(row.get("is_real", "")).strip().lower() not in ("true", "1"):
            continue
        try:
            action = int(row.get("action", 0) or 0)
            bar = int((row.get("live_bar") or row.get("bar") or 0))
            price = float(row.get("price", 0.0) or 0.0)
            risk_multiplier = float(row.get("risk_multiplier", 1.0) or 1.0)
        except ValueError:
            continue
        side = _signal_side(action)
        sym = str(row.get("symbol", "")).strip()
        if not side or not sym or price <= 0:
            continue
        rows.append({
            "idx": idx,
            "bar": bar,
            "symbol": sym,
            "action": action,
            "action_name": str(row.get("action_name", "")),
            "price": price,
            "risk_multiplier": max(0.0, risk_multiplier),
            "side": side,
        })
    rows.sort(key=lambda r: (r["bar"], r["idx"]))

    initial_equity = None
    if report is not None:
        try:
            pnl_pct = float(report.get("real_pnl_pct", 0.0) or 0.0)
            final_equity = float(report.get("final_equity", 0.0) or 0.0)
            if final_equity > 0 and pnl_pct > -99.0:
                initial_equity = final_equity / (1.0 + pnl_pct / 100.0)
        except (TypeError, ValueError, ZeroDivisionError):
            initial_equity = None
    if initial_equity is None:
        for row in _iter_signals(csv_path):
            if str(row.get("is_real", "")).strip().lower() in ("true", "1"):
                try:
                    pv = float(row.get("portfolio_value", 0.0) or 0.0)
                except ValueError:
                    pv = 0.0
                if pv > 0:
                    initial_equity = pv
                    break
    if initial_equity is None:
        initial_equity = 100.0

    equity = float(initial_equity)
    positions: Dict[str, Dict[str, float]] = {}
    last_price: Dict[str, float] = {}
    trades: List[Dict[str, Any]] = []
    rejected_duplicate_opens = 0
    close_without_position = 0
    flips_without_close = 0
    total_cost_rate = cfg["future_fee"] + cfg["slippage"]

    for row in rows:
        sym = row["symbol"]
        price = float(row["price"])
        last_price[sym] = price
        side = row["side"]

        if side in ("long", "short"):
            existing = positions.get(sym)
            if existing is not None:
                if existing["direction"] == (1.0 if side == "long" else -1.0):
                    rejected_duplicate_opens += 1
                    continue
                flips_without_close += 1
                # Close old side at the flip price, then open the new one.
                gross = existing["notional"] * existing["direction"] * ((price / existing["entry"]) - 1.0)
                cost = existing["notional"] * total_cost_rate
                equity += gross - cost
                trades.append({
                    "symbol": sym,
                    "side": "long" if existing["direction"] > 0 else "short",
                    "entry": existing["entry"],
                    "exit": price,
                    "entry_bar": int(existing["bar"]),
                    "exit_bar": row["bar"],
                    "pnl": gross - cost,
                    "reason": "flip",
                })
                positions.pop(sym, None)

            mult = _action_size_multiplier(int(row["action"]), row["action_name"])
            if apply_risk_multiplier:
                mult *= float(row.get("risk_multiplier", 1.0) or 1.0)
            notional = max(0.0, equity * cfg["trade_fraction"] * cfg["leverage"] * mult)
            open_cost = notional * total_cost_rate
            equity -= open_cost
            positions[sym] = {
                "direction": 1.0 if side == "long" else -1.0,
                "entry": price,
                "bar": float(row["bar"]),
                "notional": notional,
            }
            continue

        pos = positions.pop(sym, None)
        if pos is None:
            close_without_position += 1
            continue
        gross = pos["notional"] * pos["direction"] * ((price / pos["entry"]) - 1.0)
        cost = pos["notional"] * total_cost_rate
        equity += gross - cost
        trades.append({
            "symbol": sym,
            "side": "long" if pos["direction"] > 0 else "short",
            "entry": pos["entry"],
            "exit": price,
            "entry_bar": int(pos["bar"]),
            "exit_bar": row["bar"],
            "pnl": gross - cost,
            "reason": "close",
        })

    unrealized = 0.0
    for sym, pos in positions.items():
        price = last_price.get(sym)
        if price:
            unrealized += pos["notional"] * pos["direction"] * ((price / pos["entry"]) - 1.0)

    replay_equity = equity + unrealized
    replay_pnl_pct = ((replay_equity / initial_equity) - 1.0) * 100.0 if initial_equity else 0.0
    reported_pnl = float((report or {}).get("real_pnl_pct", 0.0) or 0.0)
    entry_risks = [
        float(row.get("risk_multiplier", 1.0) or 1.0)
        for row in rows
        if int(row.get("action", 0) or 0) in (1, 2, 4, 5, 6, 7)
    ]

    return {
        "checked": True,
        "config": cfg,
        "initial_equity": initial_equity,
        "replay_equity": replay_equity,
        "replay_pnl_pct": replay_pnl_pct,
        "reported_real_pnl_pct": reported_pnl,
        "delta_replay_vs_report": replay_pnl_pct - reported_pnl,
        "signals": len(rows),
        "closed_trades": len(trades),
        "open_positions_end": len(positions),
        "unrealized_pnl": unrealized,
        "rejected_duplicate_opens": rejected_duplicate_opens,
        "close_without_position": close_without_position,
        "flips_without_close": flips_without_close,
        "avg_entry_risk_multiplier": (sum(entry_risks) / len(entry_risks)) if entry_risks else 1.0,
        "max_entry_risk_multiplier": max(entry_risks) if entry_risks else 1.0,
        "last_trades": trades[-8:],
    }


def _source_agent_from_contributors(contributors: str) -> Optional[str]:
    first = str(contributors or "").split("+")[0].split(",")[0].strip()
    if not first:
        return None
    return first if first.startswith("V_") else f"V_{first}"


def _analyze_signal_alignment(csv_path: Path) -> Dict[str, Any]:
    """Compare real Panteon signals with the shadow source named in contributors."""
    if not csv_path.exists():
        return {"checked": False}
    rows = list(_iter_signals(csv_path) or [])
    source_index: Counter = Counter()
    for row in rows:
        if str(row.get("is_real", "")).strip().lower() in ("true", "1"):
            continue
        try:
            key = (
                int(row.get("bar", 0) or 0),
                str(row.get("agent", "")),
                str(row.get("symbol", "")),
                int(row.get("action", 0) or 0),
            )
        except ValueError:
            continue
        source_index[key] += 1

    total = 0
    matched = 0
    unmatched_open = []
    unmatched_close = []
    source_counts: Counter = Counter()
    for row in rows:
        if str(row.get("is_real", "")).strip().lower() not in ("true", "1"):
            continue
        total += 1
        try:
            action = int(row.get("action", 0) or 0)
            bar = int(row.get("bar", 0) or 0)
        except ValueError:
            continue
        source = _source_agent_from_contributors(str(row.get("contributors", "")))
        if source:
            source_counts[source] += 1
        key = (bar, source, str(row.get("symbol", "")), action)
        if source and source_index.get(key, 0) > 0:
            source_index[key] -= 1
            matched += 1
            continue
        sample = {
            "bar": bar,
            "symbol": str(row.get("symbol", "")),
            "action": action,
            "action_name": str(row.get("action_name", "")),
            "source": source,
        }
        if action in (3, 8):
            unmatched_close.append(sample)
        else:
            unmatched_open.append(sample)

    safety_adjusted = matched + len(unmatched_close)
    return {
        "checked": True,
        "real_signals": total,
        "matched_source_signals": matched,
        "match_ratio": matched / total if total else 1.0,
        "safety_adjusted_match_ratio": safety_adjusted / total if total else 1.0,
        "unmatched_open_signals": len(unmatched_open),
        "unmatched_close_signals": len(unmatched_close),
        "source_counts": dict(source_counts),
        "unmatched_open_samples": unmatched_open[:10],
        "unmatched_close_samples": unmatched_close[:10],
    }


# ── Диагностики ──────────────────────────────────────────────────────────────

def _detect_duplicate_opens(csv_path: Path) -> Dict[str, Any]:
    """
    Считает сигналы реального Пантеона: подряд идущие "open" одной стороны
    по тому же символу без промежуточного close. Это симптом стэкинга
    позиций (баг до фильтра в _act_via_shadow_players).
    """
    if not csv_path.exists():
        return {"checked": False}

    # Состояние позиции по символу для is_real==True
    pos_state: Dict[str, Optional[str]] = {}
    last_open_bar: Dict[Tuple[str, str], int] = {}
    dup_open_events: List[Tuple[str, int, str]] = []
    close_no_pos_events: List[Tuple[str, int]] = []
    n_real_opens = 0
    n_real_closes = 0

    for row in _iter_signals(csv_path):
        if str(row.get("is_real", "")).strip().lower() not in ("true", "1"):
            continue
        try:
            action = int(row.get("action", 0) or 0)
            bar = int(row.get("bar", 0) or 0)
        except ValueError:
            continue
        sym = str(row.get("symbol", ""))
        if not sym:
            continue

        side = None
        if action in OPEN_ACTIONS:
            side = "long"
        elif action in SHORT_ACTIONS:
            side = "short"
        elif action in CLOSE_ACTIONS:
            side = "close"
        else:
            continue

        if side in ("long", "short"):
            n_real_opens += 1
            current = pos_state.get(sym)
            key = (sym, side)
            prev_bar = last_open_bar.get(key, -10**9)
            if current == side and (bar - prev_bar) <= DUP_OPEN_WINDOW_BARS:
                dup_open_events.append((sym, bar, side))
            elif current is not None and current != side:
                # flip без close — игрок поменял сторону без явного close
                dup_open_events.append((sym, bar, f"flip {current}->{side}"))
            pos_state[sym] = side
            last_open_bar[key] = bar
        elif side == "close":
            n_real_closes += 1
            if pos_state.get(sym) is None:
                close_no_pos_events.append((sym, bar))
            pos_state[sym] = None

    return {
        "checked": True,
        "real_opens": n_real_opens,
        "real_closes": n_real_closes,
        "duplicate_open_events": len(dup_open_events),
        "duplicate_open_samples": dup_open_events[:8],
        "close_no_position_events": len(close_no_pos_events),
        "close_no_position_samples": close_no_pos_events[:8],
    }


def _analyze_shadow_perf(report: dict) -> Dict[str, Any]:
    """Сравнивает реальный P&L с шэдоу-агентами и игроками."""
    real_pnl = float(report.get("real_pnl_pct", 0.0) or 0.0)

    agents = report.get("shadow_agents", {}) or {}
    agent_pnls = [float((info or {}).get("pnl_pct", 0.0) or 0.0) for info in agents.values()]
    players = report.get("shadow_players", {}) or {}
    player_pnls = [float((info or {}).get("pnl_pct", 0.0) or 0.0) for info in players.values()]

    def _stat(name: str, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"name": name, "n": 0, "best": 0.0, "worst": 0.0, "avg": 0.0, "median": 0.0}
        sv = sorted(values)
        return {
            "name": name,
            "n": len(values),
            "best": sv[-1],
            "worst": sv[0],
            "avg": sum(values) / len(values),
            "median": sv[len(sv) // 2],
        }

    panteon_shadow_pnl = float((players.get("V_Panteon_shadow") or {}).get("pnl_pct", 0.0) or 0.0)

    # Лучший игрок и лучший агент по PnL
    best_player_name = max(players, key=lambda k: float((players[k] or {}).get("pnl_pct", 0.0) or 0.0), default=None)
    best_agent_name = max(agents, key=lambda k: float((agents[k] or {}).get("pnl_pct", 0.0) or 0.0), default=None)

    return {
        "real_pnl_pct": real_pnl,
        "agent_stats": _stat("agents", agent_pnls),
        "player_stats": _stat("players", player_pnls),
        "panteon_shadow_pnl_pct": panteon_shadow_pnl,
        "delta_real_vs_shadow_panteon": real_pnl - panteon_shadow_pnl,
        "best_shadow_player": best_player_name,
        "best_shadow_player_pnl": float((players.get(best_player_name or "") or {}).get("pnl_pct", 0.0) or 0.0),
        "best_shadow_agent": best_agent_name,
        "best_shadow_agent_pnl": float((agents.get(best_agent_name or "") or {}).get("pnl_pct", 0.0) or 0.0),
    }


def _status_real_pnl_pct(status: Optional[dict]) -> Optional[float]:
    if not isinstance(status, dict):
        return None
    live = status.get("live_session")
    if isinstance(live, dict):
        for key in (
            "panteon_owned_pnl_pct",
            "panteon_pnl_pct",
            "account_pnl_pct",
        ):
            try:
                value = live.get(key)
                if value is not None and value != "":
                    return float(value)
            except (TypeError, ValueError):
                continue
    for key in ("real_pnl_pct", "pnl_pct"):
        try:
            value = status.get(key)
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _analyze_live_consistency(status: Optional[dict]) -> Dict[str, Any]:
    if not isinstance(status, dict):
        return {"checked": False, "reason": "no status.json"}
    live = status.get("live_session")
    if not isinstance(live, dict):
        live = {}

    account_pnl = _float_or_none(live.get("account_pnl_pct"))
    if account_pnl is None:
        account_pnl = _float_or_none(status.get("account_pnl_pct"))
    if account_pnl is None:
        account_pnl = _float_or_none(status.get("pnl_pct"))

    panteon_pnl = _float_or_none(live.get("panteon_owned_pnl_pct"))
    if panteon_pnl is None:
        panteon_pnl = _float_or_none(live.get("panteon_pnl_pct"))
    if panteon_pnl is None:
        panteon_pnl = _float_or_none(status.get("panteon_owned_pnl_pct"))

    external_count = _int_or_zero(live.get("external_positions_count"))
    panteon_count = _int_or_zero(live.get("panteon_owned_positions_count"))
    external_unrealized = _float_or_none(live.get("external_unrealized_pnl_usd"))
    delta = None
    if account_pnl is not None and panteon_pnl is not None:
        delta = round(account_pnl - panteon_pnl, 3)

    return {
        "checked": True,
        "comparison_scope": "panteon_owned",
        "account_pnl_pct": account_pnl,
        "panteon_owned_pnl_pct": panteon_pnl,
        "account_vs_panteon_delta_pct": delta,
        "panteon_owned_positions_count": panteon_count,
        "external_positions_count": external_count,
        "external_unrealized_pnl_usd": external_unrealized,
        "has_external_positions": external_count > 0,
    }


def _analyze_regime_consistency(session_dir: Path) -> Dict[str, Any]:
    """
    Сравнивает заполненность per_regime в leaderboard_agents vs leaderboard_players.
    Если у агентов populated только NEUTRAL, а у игроков — все три бакета,
    это симптом отсутствующего set_dashboard_regime_classifier (исправлено
    в Panteon_Trade.py).
    """
    agents_lb = _load_json(session_dir / "leaderboard_agents.json") or {}
    players_lb = _load_json(session_dir / "leaderboard_players.json") or {}

    def _regime_buckets(lb: dict) -> Counter:
        c = Counter()
        for info in (lb.get("agents") or lb.get("players") or {}).values():
            for regime, bucket in (info or {}).get("per_regime", {}).items():
                ticks = int((bucket or {}).get("ticks", 0) or 0)
                if ticks > 0:
                    c[regime] += 1
        return c

    agent_buckets = _regime_buckets(agents_lb)
    player_buckets = _regime_buckets(players_lb)

    agent_keys = set(agent_buckets.keys())
    player_keys = set(player_buckets.keys())
    divergent = (agent_keys ^ player_keys) - {"unknown"}
    return {
        "agent_regime_buckets": dict(agent_buckets),
        "player_regime_buckets": dict(player_buckets),
        "divergent_buckets": sorted(divergent),
        "consistent": not divergent,
    }


def _analyze_regime_flapping(log_path: Path) -> Dict[str, Any]:
    """Считает количество "regime changed X -> Y" в trading.log за live-фазу."""
    if not log_path.exists():
        return {"checked": False}
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {"checked": False}

    pattern = re.compile(
        r"\[PLAYER meta\] regime changed\s+(\w+)\s*->\s*(\w+)"
    )
    transitions = pattern.findall(text)
    # Считаем только в LIVE фазе (после "Прогрев завершён")
    live_marker = "Прогрев завершён"
    if live_marker in text:
        live_text = text[text.index(live_marker):]
        live_transitions = pattern.findall(live_text)
    else:
        live_transitions = transitions

    pair_counts = Counter(f"{a}->{b}" for a, b in live_transitions)
    return {
        "checked": True,
        "total_transitions": len(transitions),
        "live_transitions": len(live_transitions),
        "live_pair_counts": dict(pair_counts),
    }


# ── Вердикт по сессии ────────────────────────────────────────────────────────

def _build_verdict(diag: Dict[str, Any]) -> List[str]:
    """Возвращает список диагностических замечаний (или ['OK'] если всё хорошо)."""
    notes: List[str] = []
    perf = diag.get("perf", {})
    real = float(perf.get("real_pnl_pct", 0.0))
    panteon_shadow = float(perf.get("panteon_shadow_pnl_pct", 0.0))
    best_player = float(perf.get("best_shadow_player_pnl", 0.0))
    best_agent = float(perf.get("best_shadow_agent_pnl", 0.0))

    # 1. Real Panteon vs V_Panteon_shadow (тот же код, без exchange-drag)
    delta = real - panteon_shadow
    if panteon_shadow != 0 or real != 0:
        if delta < -1.5:
            notes.append(
                f"⚠ real Panteon отстаёт от V_Panteon_shadow на {delta:+.2f}% — "
                "execution drag (slippage/fees/partial fills) выше нормы."
            )
        elif delta < -0.5:
            notes.append(
                f"ℹ real vs V_Panteon_shadow: {delta:+.2f}% (приемлемый exchange drag)."
            )

    # 2. Real vs best player
    if best_player - real > 2.0:
        notes.append(
            f"⚠ Лучший shadow-player ({perf.get('best_shadow_player')}) "
            f"делает {best_player:+.2f}% против реального {real:+.2f}% — "
            f"мета-селектор не выбирает оптимального игрока."
        )

    # 3. Real vs best agent
    if best_agent - real > 4.0:
        notes.append(
            f"⚠ Лучший shadow-agent ({perf.get('best_shadow_agent')}) "
            f"делает {best_agent:+.2f}% — рассмотри прямую делегацию агенту."
        )

    # 4. Дубликаты opens
    dup = diag.get("dup_opens", {})
    if dup.get("checked") and dup.get("duplicate_open_events", 0) > 0:
        notes.append(
            f"⚠ Найдено {dup['duplicate_open_events']} duplicate-open событий — "
            "симптом стэкинга позиций в hedge-mode (фикс: фильтр в "
            "_act_via_shadow_players)."
        )

    # 5. Close при отсутствии позиции
    if dup.get("checked") and dup.get("close_no_position_events", 0) > 5:
        notes.append(
            f"ℹ {dup['close_no_position_events']} close-сигналов при отсутствии "
            "позиции — игрок не видит реального состояния (норма после фикса)."
        )

    # 6. Regime консистентность
    regime = diag.get("regime_consistency", {})
    if not regime.get("consistent", True):
        notes.append(
            f"⚠ agent/player regime buckets расходятся: "
            f"divergent={regime.get('divergent_buckets')} — "
            "проверь set_dashboard_regime_classifier."
        )

    # 7. Regime flapping
    flap = diag.get("regime_flapping", {})
    live_trans = int(flap.get("live_transitions", 0) or 0)
    duration_h = float(diag.get("duration_hours", 0.0) or 0.0)
    if duration_h > 0 and live_trans / max(duration_h, 0.1) > 3.0:
        notes.append(
            f"⚠ Regime flapping: {live_trans} переключений за {duration_h:.1f}ч "
            f"({live_trans / duration_h:.1f}/ч) — увеличь REGIME_HYSTERESIS_BARS."
        )

    replay = diag.get("replay", {})
    if replay.get("checked"):
        replay_delta = float(replay.get("delta_replay_vs_report", 0.0) or 0.0)
        avg_risk = float(replay.get("avg_entry_risk_multiplier", 1.0) or 1.0)
        max_risk = float(replay.get("max_entry_risk_multiplier", 1.0) or 1.0)
        risk_impact = float(replay.get("risk_multiplier_pnl_impact", 0.0) or 0.0)
        if abs(replay_delta) > 3.0:
            notes.append(
                f"⚠ Retro-replay real signals diverges from final report by {replay_delta:+.2f}% "
                "(check execution accounting, sizing, duplicate-open handling)."
            )
        elif abs(replay_delta) > 1.0:
            notes.append(
                f"ℹ Retro-replay vs report delta {replay_delta:+.2f}% "
                "(acceptable only if partial fills/manual exchange state affected live trading)."
            )
        if int(replay.get("rejected_duplicate_opens", 0) or 0) > 0:
            notes.append(
                f"⚠ Retro-replay rejected {replay['rejected_duplicate_opens']} duplicate real opens "
                "under one-position-per-symbol rules."
            )
        if max_risk > 1.01:
            notes.append(
                f"⚠ Live-only risk multiplier was active (avg={avg_risk:.2f}x, max={max_risk:.2f}x); "
                "this makes real PnL non-comparable with shadow-player PnL and can amplify losses."
            )
        if risk_impact < -0.5:
            notes.append(
                f"WARN Risk multiplier worsened retro replay by {risk_impact:+.2f}% vs neutral 1.0 sizing."
            )

    alignment = diag.get("signal_alignment", {})
    if alignment.get("checked") and not alignment.get("legacy_unreliable"):
        total = int(alignment.get("real_signals", 0) or 0)
        unmatched_open = int(alignment.get("unmatched_open_signals", 0) or 0)
        adjusted = float(alignment.get("safety_adjusted_match_ratio", 1.0) or 1.0)
        if unmatched_open > max(2, int(total * 0.03)):
            notes.append(
                f"⚠ Real/source signal drift: {unmatched_open} real open signals were not present "
                "in the delegated shadow source on the same bar."
            )
        elif adjusted < 0.95:
            notes.append(
                f"ℹ Real/source alignment is {adjusted * 100:.1f}% after safety exits; "
                "review delegated player state if this persists."
            )

    live_consistency = diag.get("live_consistency", {})
    if live_consistency.get("checked"):
        delta = live_consistency.get("account_vs_panteon_delta_pct")
        if live_consistency.get("has_external_positions"):
            notes.append(
                f"WARN Live account includes {live_consistency.get('external_positions_count', 0)} "
                "external/recovered positions; compare retro to panteon_owned scope only."
            )
        if delta is not None and abs(float(delta)) > 0.5:
            notes.append(
                f"WARN live account vs panteon-owned PnL delta is {float(delta):+.3f}%."
            )

    return notes or ["OK"]


# ── Главный анализ ───────────────────────────────────────────────────────────

OUTLIER_PNL_THRESHOLD = 50.0  # |pnl_pct| выше этого — корраптная или paper-test сессия


def analyze_session(session_dir: Path) -> Optional[Dict[str, Any]]:
    """Возвращает diagnostic dict или None если сессия не содержит нужных артефактов."""
    report = _load_json(session_dir / "final_combo_report.json")
    had_report = report is not None
    csv_path = session_dir / "all_signals.csv"
    log_path = session_dir / "trading.log"
    status_path = session_dir / "status.json"
    status = _load_json(status_path)
    if report is None and not csv_path.exists() and not log_path.exists():
        return None
    duration_h, duration_sources = _infer_duration_hours(
        report,
        csv_path,
        log_path,
        status_path,
    )
    period_start, period_end = _infer_session_bounds(
        session_dir,
        duration_h,
        status,
        log_path,
    )
    short_duration = duration_h < MIN_DURATION_HOURS

    if report is None:
        report = {}

    perf = _analyze_shadow_perf(report)
    status_real_pnl = _status_real_pnl_pct(status)
    if status_real_pnl is not None and (
        not had_report or "real_pnl_pct" not in report
    ):
        perf["real_pnl_pct"] = status_real_pnl
        perf["delta_real_vs_shadow_panteon"] = (
            status_real_pnl
            - float(perf.get("panteon_shadow_pnl_pct", 0.0) or 0.0)
        )
    replay = _replay_real_trade_log(csv_path, log_path, report)
    risk_neutral_replay = _replay_real_trade_log(
        csv_path, log_path, report, apply_risk_multiplier=False
    )
    if replay.get("checked") and risk_neutral_replay.get("checked"):
        neutral_pnl = float(risk_neutral_replay.get("replay_pnl_pct", 0.0) or 0.0)
        replay["risk_neutral_pnl_pct"] = neutral_pnl
        replay["risk_multiplier_pnl_impact"] = (
            float(replay.get("replay_pnl_pct", 0.0) or 0.0) - neutral_pnl
        )
    if replay.get("checked") and status_real_pnl is None and not report.get("real_pnl_pct"):
        perf["real_pnl_pct"] = float(replay.get("replay_pnl_pct", 0.0) or 0.0)
        perf["delta_real_vs_shadow_panteon"] = (
            float(perf.get("real_pnl_pct", 0.0) or 0.0)
            - float(perf.get("panteon_shadow_pnl_pct", 0.0) or 0.0)
        )
    dup = _detect_duplicate_opens(csv_path)
    signal_alignment = _analyze_signal_alignment(csv_path)
    regime_consistency = _analyze_regime_consistency(session_dir)
    regime_flapping = _analyze_regime_flapping(log_path)
    live_consistency = _analyze_live_consistency(status)

    # Outlier-flag: если real PnL невозможно высокий или 0 сигналов с большим PnL —
    # это корраптный отчёт (paper-режим с битым equity, тестовые прогоны и т.п.).
    real_pnl = float(perf.get("real_pnl_pct", 0.0) or 0.0)
    real_signals = int(report.get("real_signals", 0) or replay.get("signals", 0) or 0)
    is_outlier = abs(real_pnl) > OUTLIER_PNL_THRESHOLD or (
        real_signals == 0 and abs(real_pnl) > 1.0
    )
    # Старый формат отчёта (до введения shadow_players)
    has_shadow_players = bool(report.get("shadow_players"))
    if signal_alignment.get("checked") and not has_shadow_players:
        signal_alignment["legacy_unreliable"] = True

    diag = {
        "session": session_dir.name,
        "duration": str(report.get("duration", "")),
        "duration_hours": duration_h,
        "duration_sources": duration_sources,
        "period_start": period_start.isoformat() if period_start else "",
        "period_end": period_end.isoformat() if period_end else "",
        "short_duration": short_duration,
        "real_player": report.get("real_player"),
        "real_signals": real_signals,
        "real_trades": report.get("real_trades"),
        "regime_summary": report.get("regime_summary"),
        "is_outlier": is_outlier,
        "has_shadow_players": has_shadow_players,
        "perf": perf,
        "replay": replay,
        "signal_alignment": signal_alignment,
        "dup_opens": dup,
        "regime_consistency": regime_consistency,
        "regime_flapping": regime_flapping,
        "live_consistency": live_consistency,
    }
    diag["verdict"] = _build_verdict(diag)
    if is_outlier:
        diag["verdict"].insert(
            0,
            f"⚠ OUTLIER: real_pnl={real_pnl:+.2f}% real_signals={real_signals} — "
            "сессия исключена из агрегата (paper-test или битый отчёт).",
        )
    elif not has_shadow_players:
        diag["verdict"].insert(
            0,
            "ℹ Старый формат отчёта (нет shadow_players) — сравнение неполное.",
        )
    return diag


def _print_session(diag: Dict[str, Any]) -> None:
    if diag.get("skipped"):
        print(f"  [skip] {diag['session']}  ({diag['reason']})")
        return

    perf = diag["perf"]
    duration_note = " short/combined-only" if diag.get("short_duration") else ""
    print(f"\n  📂 {diag['session']}  ({diag['duration']}, {diag['duration_hours']:.1f}h{duration_note})")
    print(
        f"     real {perf['real_pnl_pct']:+.2f}%  "
        f"panteon_shadow {perf['panteon_shadow_pnl_pct']:+.2f}%  "
        f"Δ {perf['delta_real_vs_shadow_panteon']:+.2f}%"
    )
    print(
        f"     best player: {perf['best_shadow_player']} {perf['best_shadow_player_pnl']:+.2f}%   "
        f"best agent: {perf['best_shadow_agent']} {perf['best_shadow_agent_pnl']:+.2f}%"
    )
    dup = diag["dup_opens"]
    if dup.get("checked"):
        print(
            f"     signals: real_opens={dup['real_opens']} closes={dup['real_closes']} "
            f"dup_open_events={dup['duplicate_open_events']} close_no_pos={dup['close_no_position_events']}"
        )
    replay = diag.get("replay", {})
    if replay.get("checked"):
        print(
            f"     retro-replay: {replay['replay_pnl_pct']:+.2f}%  "
            f"report_delta={replay['delta_replay_vs_report']:+.2f}%  "
            f"trades={replay['closed_trades']} open_end={replay['open_positions_end']} "
            f"dup_rejected={replay['rejected_duplicate_opens']} "
            f"risk_avg={replay.get('avg_entry_risk_multiplier', 1.0):.2f}x"
        )
        if "risk_multiplier_pnl_impact" in replay:
            print(
                f"     risk-neutral replay: {replay['risk_neutral_pnl_pct']:+.2f}%  "
                f"risk_impact={replay['risk_multiplier_pnl_impact']:+.2f}%"
            )
    alignment = diag.get("signal_alignment", {})
    if alignment.get("checked") and not alignment.get("legacy_unreliable"):
        print(
            f"     alignment: source_match={alignment['match_ratio'] * 100:.1f}%  "
            f"safety_adj={alignment['safety_adjusted_match_ratio'] * 100:.1f}%  "
            f"unmatched_open={alignment['unmatched_open_signals']} "
            f"unmatched_close={alignment['unmatched_close_signals']}"
        )
    flap = diag["regime_flapping"]
    if flap.get("checked"):
        print(f"     regime live_transitions={flap['live_transitions']}")
    for note in diag["verdict"]:
        print(f"     • {note}")


def _diag_time(diag: Dict[str, Any], key: str) -> Optional[datetime]:
    value = diag.get(key)
    if isinstance(value, datetime):
        return _naive_datetime(value)
    return _naive_datetime(_parse_iso_time(value))


def _round_float(value: float) -> float:
    return round(float(value), 10)


def _summarize_combined_period(rows: List[Dict[str, Any]], gaps: List[float]) -> Dict[str, Any]:
    total_hours = sum(float(d.get("duration_hours", 0.0) or 0.0) for d in rows)
    cumulative_real = sum(float(d["perf"].get("real_pnl_pct", 0.0) or 0.0) for d in rows)
    cumulative_shadow = sum(float(d["perf"].get("panteon_shadow_pnl_pct", 0.0) or 0.0) for d in rows)
    cumulative_delta = sum(float(d["perf"].get("delta_real_vs_shadow_panteon", 0.0) or 0.0) for d in rows)
    replay_rows = [d for d in rows if d.get("replay", {}).get("checked")]
    cumulative_replay = sum(
        float(d["replay"].get("replay_pnl_pct", 0.0) or 0.0)
        for d in replay_rows
    )
    first_start = _diag_time(rows[0], "period_start")
    last_end = _diag_time(rows[-1], "period_end")
    return {
        "sessions": [str(d.get("session", "")) for d in rows],
        "n_sessions": len(rows),
        "start": first_start.isoformat() if first_start else "",
        "end": last_end.isoformat() if last_end else "",
        "total_hours": _round_float(total_hours),
        "gap_hours": [_round_float(v) for v in gaps],
        "max_gap_hours": _round_float(max(gaps) if gaps else 0.0),
        "cumulative_real_pnl_pct": _round_float(cumulative_real),
        "cumulative_panteon_shadow_pnl_pct": _round_float(cumulative_shadow),
        "cumulative_delta_real_vs_shadow_pct": _round_float(cumulative_delta),
        "cumulative_replay_pnl_pct": _round_float(cumulative_replay),
        "replay_sessions": len(replay_rows),
        "replay_duplicate_opens_rejected": sum(
            int(d.get("replay", {}).get("rejected_duplicate_opens", 0) or 0)
            for d in replay_rows
        ),
        "sessions_with_duplicate_opens": sum(
            1
            for d in rows
            if d.get("dup_opens", {}).get("checked")
            and int(d.get("dup_opens", {}).get("duplicate_open_events", 0) or 0) > 0
        ),
        "sessions_with_inconsistent_regime": sum(
            1 for d in rows if not d.get("regime_consistency", {}).get("consistent", True)
        ),
    }


def _combined_periods(
    diags: List[Dict[str, Any]],
    *,
    max_gap_hours: float = COMBINE_MAX_GAP_HOURS,
    min_total_hours: float = MIN_DURATION_HOURS,
) -> Dict[str, Any]:
    candidates = []
    for diag in diags:
        if diag.get("skipped") or diag.get("is_outlier"):
            continue
        start = _diag_time(diag, "period_start")
        end = _diag_time(diag, "period_end")
        duration = float(diag.get("duration_hours", 0.0) or 0.0)
        if not start or not end or duration <= 0:
            continue
        candidates.append((start, end, diag))

    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("session", ""))))
    periods: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    gaps: List[float] = []
    current_end: Optional[datetime] = None

    def flush() -> None:
        if not current:
            return
        total_hours = sum(float(d.get("duration_hours", 0.0) or 0.0) for d in current)
        if total_hours >= min_total_hours:
            periods.append(_summarize_combined_period(current, gaps))

    for start, end, diag in candidates:
        if not current:
            current = [diag]
            gaps = []
            current_end = end
            continue
        assert current_end is not None
        gap = max(0.0, (start - current_end).total_seconds() / 3600.0)
        if gap <= max_gap_hours:
            current.append(diag)
            gaps.append(gap)
            current_end = max(current_end, end)
            continue
        flush()
        current = [diag]
        gaps = []
        current_end = end
    flush()

    total_sessions = sum(int(p["n_sessions"]) for p in periods)
    total_hours = sum(float(p["total_hours"]) for p in periods)
    cumulative_real = sum(float(p["cumulative_real_pnl_pct"]) for p in periods)
    cumulative_shadow = sum(float(p["cumulative_panteon_shadow_pnl_pct"]) for p in periods)
    cumulative_replay = sum(float(p["cumulative_replay_pnl_pct"]) for p in periods)
    return {
        "max_gap_hours": float(max_gap_hours),
        "min_total_hours": float(min_total_hours),
        "n_periods": len(periods),
        "n_sessions": total_sessions,
        "total_hours": _round_float(total_hours),
        "cumulative_real_pnl_pct": _round_float(cumulative_real),
        "cumulative_panteon_shadow_pnl_pct": _round_float(cumulative_shadow),
        "cumulative_delta_real_vs_shadow_pct": _round_float(cumulative_real - cumulative_shadow),
        "cumulative_replay_pnl_pct": _round_float(cumulative_replay),
        "periods": periods,
    }


def _aggregate(diags: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегирует по не-skipped и не-outlier сессиям."""
    combined = _combined_periods(diags)
    rows = [
        d for d in diags
        if not d.get("skipped")
        and not d.get("short_duration")
        and not d.get("is_outlier")
    ]
    n_outliers = sum(
        1 for d in diags if not d.get("skipped") and d.get("is_outlier")
    )
    if not rows:
        return {
            "n_sessions": 0,
            "n_outliers_excluded": n_outliers,
            "combined_periods": combined,
        }
    total_h = sum(d["duration_hours"] for d in rows)
    avg_real = sum(d["perf"]["real_pnl_pct"] for d in rows) / len(rows)

    # Сравнения с shadow players имеют смысл только для сессий с has_shadow_players=True
    rows_with_shadow = [d for d in rows if d.get("has_shadow_players")]
    if rows_with_shadow:
        avg_shadow = sum(d["perf"]["panteon_shadow_pnl_pct"] for d in rows_with_shadow) / len(rows_with_shadow)
        avg_delta = sum(d["perf"]["delta_real_vs_shadow_panteon"] for d in rows_with_shadow) / len(rows_with_shadow)
    else:
        avg_shadow = 0.0
        avg_delta = 0.0

    n_with_dup = sum(
        1 for d in rows
        if d["dup_opens"].get("checked") and d["dup_opens"].get("duplicate_open_events", 0) > 0
    )
    n_inconsistent_regime = sum(
        1 for d in rows if not d["regime_consistency"].get("consistent", True)
    )
    rows_with_replay = [d for d in rows if d.get("replay", {}).get("checked")]
    if rows_with_replay:
        avg_replay_pnl = sum(d["replay"]["replay_pnl_pct"] for d in rows_with_replay) / len(rows_with_replay)
        avg_replay_delta = sum(d["replay"]["delta_replay_vs_report"] for d in rows_with_replay) / len(rows_with_replay)
        replay_dup_rejected = sum(int(d["replay"].get("rejected_duplicate_opens", 0) or 0) for d in rows_with_replay)
        avg_entry_risk = sum(float(d["replay"].get("avg_entry_risk_multiplier", 1.0) or 1.0) for d in rows_with_replay) / len(rows_with_replay)
        avg_risk_impact = sum(float(d["replay"].get("risk_multiplier_pnl_impact", 0.0) or 0.0) for d in rows_with_replay) / len(rows_with_replay)
    else:
        avg_replay_pnl = 0.0
        avg_replay_delta = 0.0
        replay_dup_rejected = 0
        avg_entry_risk = 1.0
        avg_risk_impact = 0.0
    rows_with_alignment = [
        d for d in rows
        if d.get("has_shadow_players")
        and d.get("signal_alignment", {}).get("checked")
        and not d.get("signal_alignment", {}).get("legacy_unreliable")
    ]
    if rows_with_alignment:
        avg_source_match = sum(d["signal_alignment"]["match_ratio"] for d in rows_with_alignment) / len(rows_with_alignment)
        avg_safety_adjusted = sum(d["signal_alignment"]["safety_adjusted_match_ratio"] for d in rows_with_alignment) / len(rows_with_alignment)
        total_unmatched_open = sum(int(d["signal_alignment"].get("unmatched_open_signals", 0) or 0) for d in rows_with_alignment)
    else:
        avg_source_match = 1.0
        avg_safety_adjusted = 1.0
        total_unmatched_open = 0
    return {
        "n_sessions": len(rows),
        "n_sessions_with_shadow_players": len(rows_with_shadow),
        "n_sessions_with_replay": len(rows_with_replay),
        "n_sessions_with_alignment": len(rows_with_alignment),
        "n_outliers_excluded": n_outliers,
        "total_hours": total_h,
        "avg_real_pnl_pct": avg_real,
        "avg_replay_pnl_pct": avg_replay_pnl,
        "avg_replay_delta_vs_report": avg_replay_delta,
        "replay_duplicate_opens_rejected": replay_dup_rejected,
        "avg_entry_risk_multiplier": avg_entry_risk,
        "avg_risk_multiplier_pnl_impact": avg_risk_impact,
        "avg_source_signal_match": avg_source_match,
        "avg_safety_adjusted_signal_match": avg_safety_adjusted,
        "total_unmatched_open_signals": total_unmatched_open,
        "avg_panteon_shadow_pnl_pct": avg_shadow,
        "avg_delta_real_vs_shadow": avg_delta,
        "sessions_with_dup_opens": n_with_dup,
        "sessions_with_inconsistent_regime": n_inconsistent_regime,
        "combined_periods": combined,
    }


def run_retrostart(exchange: str, project_root: Optional[Path] = None) -> int:
    """
    Главная точка входа. Прогоняет анализ всех сессий >3ч для указанной биржи.
    Возвращает exit code (0 = OK, 1 = есть критические замечания).
    """
    project_root = project_root or Path(__file__).resolve().parents[2]
    results_dir = project_root / "Results" / exchange.upper()
    if not results_dir.exists():
        print(f"❌ Папка с результатами не найдена: {results_dir}")
        return 2

    print("═" * 78)
    print(f"  RETROSTART  ·  {exchange.upper()}  ·  фильтр: длительность ≥ {MIN_DURATION_HOURS}ч")
    print(f"  Источник: {results_dir}")
    print("═" * 78)

    sessions = sorted(p for p in results_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not sessions:
        print(f"  Нет сессий в {results_dir}")
        return 2

    diags: List[Dict[str, Any]] = []
    for s in sessions:
        diag = analyze_session(s)
        if diag is None:
            print(f"  [n/a] {s.name}  (нет final_combo_report.json)")
            continue
        diags.append(diag)
        _print_session(diag)

    print("\n" + "─" * 78)
    print("  📊 АГРЕГАТ")
    print("─" * 78)
    agg = _aggregate(diags)
    if agg.get("n_sessions", 0) == 0:
        print("  Нет сессий ≥ 3ч для агрегата.")
    else:
        print(f"  сессий: {agg['n_sessions']}   суммарно часов: {agg['total_hours']:.1f}")
        print(f"  outliers исключено:         {agg.get('n_outliers_excluded', 0)}")
        print(f"  с shadow_players в отчёте:  {agg.get('n_sessions_with_shadow_players', 0)}")
        print(f"  с retro-replay из логов:    {agg.get('n_sessions_with_replay', 0)}")
        print(f"  средний real PnL:           {agg['avg_real_pnl_pct']:+.2f}%")
        if agg.get("n_sessions_with_replay", 0) > 0:
            print(f"  средний replay PnL:         {agg['avg_replay_pnl_pct']:+.2f}%")
            print(f"  replay vs report:           {agg['avg_replay_delta_vs_report']:+.2f}%")
            print(f"  replay duplicate rejected:  {agg['replay_duplicate_opens_rejected']}")
            print(f"  risk multiplier impact:     {agg['avg_risk_multiplier_pnl_impact']:+.2f}%")
            print(f"  средний risk multiplier:    {agg['avg_entry_risk_multiplier']:.2f}x")
        if agg.get("n_sessions_with_alignment", 0) > 0:
            print(f"  real/source match:          {agg['avg_source_signal_match'] * 100:.1f}%")
            print(f"  safety-adjusted match:      {agg['avg_safety_adjusted_signal_match'] * 100:.1f}%")
            print(f"  unmatched real opens:       {agg['total_unmatched_open_signals']}")
        if agg.get("n_sessions_with_shadow_players", 0) > 0:
            print(f"  средний V_Panteon_shadow:   {agg['avg_panteon_shadow_pnl_pct']:+.2f}%")
            print(f"  средняя дельта (drag):      {agg['avg_delta_real_vs_shadow']:+.2f}%")
        print(f"  сессий с duplicate-opens:   {agg['sessions_with_dup_opens']}")
        print(f"  сессий с inconsistent regime: {agg['sessions_with_inconsistent_regime']}")
        combined = agg.get("combined_periods", {})
        if combined.get("n_periods", 0) > 0:
            print(
                f"  combined periods ≤{combined.get('max_gap_hours', COMBINE_MAX_GAP_HOURS):.1f}h gap: "
                f"{combined['n_periods']} periods / {combined['n_sessions']} sessions / "
                f"{combined['total_hours']:.1f}h"
            )
            print(
                f"  combined cumulative real/replay/shadow: "
                f"{combined['cumulative_real_pnl_pct']:+.2f}% / "
                f"{combined['cumulative_replay_pnl_pct']:+.2f}% / "
                f"{combined['cumulative_panteon_shadow_pnl_pct']:+.2f}%"
            )

        # Финальный вердикт
        print()
        avg_delta = agg.get("avg_delta_real_vs_shadow", 0.0)
        n_dup = agg.get("sessions_with_dup_opens", 0)
        n_regime = agg.get("sessions_with_inconsistent_regime", 0)
        if avg_delta < -2.0:
            print(f"  🔴 ВЕРДИКТ: Пантеон СИСТЕМАТИЧЕСКИ отстаёт от своей шэдоу-копии "
                  f"({avg_delta:+.2f}% средняя дельта). Проверь: фильтр в _act_via_shadow_players, "
                  f"мета-селектор, exchange-fees.")
        elif avg_delta < -0.8:
            print(f"  🟡 ВЕРДИКТ: умеренный exchange drag ({avg_delta:+.2f}%). "
                  f"В пределах нормы для live-торговли.")
        else:
            print(f"  🟢 ВЕРДИКТ: Пантеон работает в пределах ожидаемого drag "
                  f"({avg_delta:+.2f}% от shadow-копии).")
        if n_dup > 0:
            print(f"  ⚠ В {n_dup} сессии(ях) найдены duplicate-open события — "
                  "это симптом стэкинга позиций. Если фикс фильтра уже применён "
                  "(panteon_agents.py ~line 4860) — будущие сессии не должны их содержать.")
        if n_regime > 0:
            print(f"  ⚠ В {n_regime} сессии(ях) regime-buckets агентов и игроков "
                  "расходятся — set_dashboard_regime_classifier либо не зарегистрирован, "
                  "либо вызван слишком поздно.")

    # Сохраняем JSON-отчёт
    out_path = results_dir / "_retrostart_report.json"
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "exchange": exchange.upper(),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "min_duration_hours": MIN_DURATION_HOURS,
                    "aggregate": agg,
                    "sessions": diags,
                },
                fh, ensure_ascii=False, indent=2,
            )
        print(f"\n  💾 JSON-отчёт: {out_path}")
    except OSError as exc:
        print(f"\n  ⚠ не удалось сохранить отчёт: {exc}")

    print("═" * 78)
    # Exit code: 1 если в любой сессии критическое замечание, иначе 0
    has_critical = any(
        any(n.startswith("⚠") for n in d.get("verdict", []))
        for d in diags if not d.get("skipped")
    )
    return 1 if has_critical else 0


if __name__ == "__main__":
    import sys
    exch = (sys.argv[1] if len(sys.argv) > 1 else "MEXC").upper()
    raise SystemExit(run_retrostart(exch))
