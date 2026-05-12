"""Парсер v1-логов. Читает только данные (JSON/CSV/log) — никаких импортов
из panteon_runtime, чтобы v2 оставался изолированным.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ────────────────────────────────────────────────────────────────────
# DTO для распарсенных v1-данных
# ────────────────────────────────────────────────────────────────────


# v1 action имена (из exchange_api_runtime.TradingStats.record_signal)
V1_ACTION_NAMES = {
    "hold":          0,
    "spot_buy_half": 1,
    "buy_half":      1,
    "spot_buy_full": 2,
    "buy_full":      2,
    "sell_spot":     3,
    "spot_sell_all": 3,
    "fl_half":       4,
    "fl_full":       5,
    "fs_half":       6,
    "fs_full":       7,
    "close_fut":     8,
    "fut_close_all": 8,
    "fut_long_half": 4,
    "fut_long_full": 5,
    "fut_short_half": 6,
    "fut_short_full": 7,
}


@dataclass(frozen=True)
class V1Signal:
    """Сигнал из v1.

    Источник: status.json[recent_signals] / all_signals.csv (только real-сигналы).
    """

    bar:              int
    sym:              str
    action_code:      int          # 0..8
    action_name:      str
    price:            float
    timestamp:        Optional[datetime]
    selected_player:  str
    selected_agents:  str
    regime:           str          # raw — может быть "bullish"/"neutral"/"unknown"
    risk_multiplier:  float = 1.0
    is_real:          bool = True


@dataclass(frozen=True)
class V1Trade:
    """Trade из v1 (status.json[recent_trades])."""

    sym:        str
    side:       str          # "LONG" | "SHORT"
    qty:        float
    price:      float
    fee:        float
    funding:    float
    value:      float        # qty * price
    timestamp:  Optional[datetime]


@dataclass(frozen=True)
class V1AgentEntry:
    """Запись из leaderboard_*.json."""

    label:           str          # с префиксом V_
    pnl_pct:         float
    closed_trades:   int
    win_rate:        float
    sharpe:          float
    max_dd_pct:      float
    status:          str          # "live" | "quarantine" | "shadow_only" | ...
    status_reason:   str
    per_regime:      Dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class V1Session:
    """Полный снимок v1-сессии."""

    session_path:     str
    exchange:         str          # "MEXC" | "BITGET"
    initial_capital:  float
    current_balance:  float
    pnl_usd:          float
    pnl_pct:          float
    bar_count:        int
    live_bar_count:   int
    n_signals:        int
    n_trades:         int
    current_regime:   str
    real_signals:     List[V1Signal]
    recent_trades:    List[V1Trade]
    agents:           Dict[str, V1AgentEntry]   # label -> entry
    players:          Dict[str, V1AgentEntry]
    quarantined:      List[str]                  # из leaderboard статусов
    failed_orders:    int = 0


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _parse_timestamp(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    s = str(value)
    # "2026-05-05T09:23:36+00:00" или "2026-05-05 09:23:36+00:00"
    s = s.replace(" ", "T", 1) if " " in s and "T" not in s else s
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Попробуем без timezone
        try:
            return datetime.fromisoformat(s.split("+")[0]).replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _action_to_code(name_or_code) -> int:
    """v1 action → integer code 0..8."""
    if isinstance(name_or_code, int):
        return name_or_code
    if isinstance(name_or_code, str):
        # Может быть число в строке "5" или имя "fl_full"
        try:
            return int(name_or_code)
        except ValueError:
            pass
        return V1_ACTION_NAMES.get(name_or_code.strip().lower(), 0)
    return 0


# ────────────────────────────────────────────────────────────────────
# Parsers
# ────────────────────────────────────────────────────────────────────


def parse_status_json(path: str) -> dict:
    """Чтение status.json, ничего не интерпретируем — отдаём как есть."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"status.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_leaderboard_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"leaderboard not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_all_signals_csv(path: str, *, only_real: bool = True) -> List[V1Signal]:
    """all_signals.csv → list V1Signal. only_real=True → только real-сигналы."""
    out: List[V1Signal] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                is_real = str(row.get("is_real", "")).strip().lower() in (
                    "true", "1", "yes",
                )
                if only_real and not is_real:
                    continue
                action_code = _action_to_code(
                    row.get("action") or row.get("action_name", "")
                )
                out.append(V1Signal(
                    bar=int(float(row.get("bar") or 0)),
                    sym=str(row.get("symbol") or row.get("sym") or "").upper(),
                    action_code=action_code,
                    action_name=str(row.get("action_name") or "").lower(),
                    price=float(row.get("price") or 0.0),
                    timestamp=_parse_timestamp(row.get("timestamp")),
                    selected_player=str(row.get("selected_player") or row.get("agent") or "").strip(),
                    selected_agents=str(row.get("selected_agents") or row.get("contributors") or "").strip(),
                    regime=str(row.get("regime") or "").lower(),
                    risk_multiplier=float(row.get("risk_multiplier") or 1.0),
                    is_real=is_real,
                ))
            except (ValueError, TypeError):
                continue
    return out


def _parse_recent_signals(raw: list) -> List[V1Signal]:
    out = []
    for item in raw or []:
        try:
            out.append(V1Signal(
                bar=int(item.get("bar") or 0),
                sym=str(item.get("sym") or "").upper(),
                action_code=_action_to_code(item.get("action") or 0),
                action_name=str(item.get("action") or "").lower(),
                price=float(item.get("price") or 0.0),
                timestamp=_parse_timestamp(item.get("time")),
                selected_player=str(item.get("selected_player") or "").strip(),
                selected_agents=str(item.get("selected_agents") or "").strip(),
                regime="",  # status.json signals не несут regime; берём из meta
                risk_multiplier=float(item.get("risk_multiplier") or 1.0),
                is_real=True,
            ))
        except (ValueError, TypeError):
            continue
    return out


def _parse_recent_trades(raw: list) -> List[V1Trade]:
    out = []
    for item in raw or []:
        try:
            out.append(V1Trade(
                sym=str(item.get("sym") or "").upper(),
                side=str(item.get("side") or "long").upper(),
                qty=float(item.get("qty") or 0.0),
                price=float(item.get("price") or 0.0),
                fee=float(item.get("fee") or 0.0),
                funding=float(item.get("funding") or 0.0),
                value=float(item.get("value") or 0.0),
                timestamp=_parse_timestamp(item.get("time")),
            ))
        except (ValueError, TypeError):
            continue
    return out


def _parse_leaderboard_entries(data: dict) -> Dict[str, V1AgentEntry]:
    """data — содержимое leaderboard_*.json."""
    container = data.get("agents") or data.get("players") or {}
    out: Dict[str, V1AgentEntry] = {}
    for label, raw in container.items():
        try:
            out[label] = V1AgentEntry(
                label=label,
                pnl_pct=float(raw.get("pnl_pct") or 0.0),
                closed_trades=int(raw.get("closed_trades") or raw.get("total_trades") or 0),
                win_rate=float(raw.get("win_rate") or 0.0),
                sharpe=float(raw.get("sharpe") or 0.0),
                max_dd_pct=float(raw.get("max_drawdown_pct") or 0.0),
                status=str(raw.get("status") or "live").lower(),
                status_reason=str(raw.get("status_reason") or ""),
                per_regime=dict(raw.get("per_regime") or {}),
            )
        except (ValueError, TypeError):
            continue
    return out


def _detect_exchange(session_path: str) -> str:
    parts = os.path.normpath(session_path).split(os.sep)
    for p in reversed(parts):
        if p.upper() in ("MEXC", "BITGET", "BINANCE"):
            return p.upper()
    return "UNKNOWN"


def parse_session(session_path: str, *, prefer_csv: bool = True) -> V1Session:
    """Прочитать всю сессию: status + leaderboards + signals.

    prefer_csv=True → берёт сигналы из all_signals.csv (полный список),
                     иначе из status.json[recent_signals] (только последние 10).
    """
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session directory not found: {session_path}")

    status = parse_status_json(os.path.join(session_path, "status.json"))

    # Leaderboards
    agents_path = os.path.join(session_path, "leaderboard_agents.json")
    players_path = os.path.join(session_path, "leaderboard_players.json")
    agents = (
        _parse_leaderboard_entries(parse_leaderboard_json(agents_path))
        if os.path.exists(agents_path) else {}
    )
    players = (
        _parse_leaderboard_entries(parse_leaderboard_json(players_path))
        if os.path.exists(players_path) else {}
    )

    # Current regime — из leaderboard metadata
    current_regime = "neutral"
    if os.path.exists(agents_path):
        meta = parse_leaderboard_json(agents_path).get("metadata", {})
        current_regime = str(meta.get("regime") or "neutral").lower()

    # Quarantined: те, чей status в spec
    quarantined_set = set()
    for label, ent in {**agents, **players}.items():
        if ent.status in ("quarantine", "shadow_only", "purgatory"):
            # Чистим V_ префикс — внутренний QM работает без него
            clean = label.replace("V_", "")
            quarantined_set.add(clean)

    # Signals
    real_signals: List[V1Signal] = []
    csv_path = os.path.join(session_path, "all_signals.csv")
    if prefer_csv and os.path.exists(csv_path):
        real_signals = parse_all_signals_csv(csv_path, only_real=True)
    else:
        real_signals = _parse_recent_signals(status.get("recent_signals") or [])

    trades = _parse_recent_trades(status.get("recent_trades") or [])

    failed_orders = int(
        (status.get("data_health") or {}).get("failed_orders") or 0
    )

    return V1Session(
        session_path=session_path,
        exchange=_detect_exchange(session_path),
        initial_capital=float(status.get("initial_capital") or 0.0),
        current_balance=float(status.get("current_balance") or 0.0),
        pnl_usd=float(status.get("pnl_usd") or 0.0),
        pnl_pct=float(status.get("pnl_pct") or 0.0),
        bar_count=int(status.get("bar_count") or 0),
        live_bar_count=int(status.get("live_bar_count") or 0),
        n_signals=int(status.get("n_signals") or 0),
        n_trades=int(status.get("n_trades") or 0),
        current_regime=current_regime,
        real_signals=real_signals,
        recent_trades=trades,
        agents=agents,
        players=players,
        quarantined=sorted(quarantined_set),
        failed_orders=failed_orders,
    )
