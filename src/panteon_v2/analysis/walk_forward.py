"""Walk-forward/retro report over saved Results and JSONL event logs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from ..domain.types import Regime


def build_walk_forward_report(
    *,
    results_root: str = "Results",
    event_logs: Optional[Iterable[str]] = None,
    temporal_window_count: int = 5,
) -> dict:
    event_paths = [Path(p) for p in event_logs] if event_logs is not None else _default_event_logs()
    events = _load_events(event_paths)
    regime_by_bar = _regime_map(events)
    closed = _closed_trades(events, regime_by_bar)
    fills = _fills(events)
    sessions = _session_summary(Path(results_root))

    by_regime = {}
    for regime in (r.label for r in Regime):
        subset = [trade for trade in closed if trade.get("regime") == regime]
        if subset:
            by_regime[regime] = _trade_stats(subset, fills=[])

    totals = _trade_stats(closed, fills=fills)
    return {
        "results_root": str(Path(results_root)),
        "event_logs": [str(p) for p in event_paths if p.exists()],
        "sessions": sessions,
        "totals": totals,
        "by_regime": by_regime,
        "by_actor": _actor_stats(closed),
        "temporal_windows": _temporal_windows(
            events,
            closed,
            fills,
            window_count=temporal_window_count,
        ),
    }


def build_walk_forward_report_from_events(
    *,
    results_root: str = "Results",
    events: Iterable[Mapping],
    temporal_window_count: int = 5,
) -> dict:
    event_rows = [dict(event) for event in events if isinstance(event, Mapping)]
    regime_by_bar = _regime_map(event_rows)
    closed = _closed_trades(event_rows, regime_by_bar)
    fills = _fills(event_rows)
    sessions = _session_summary(Path(results_root))

    by_regime = {}
    for regime in (r.label for r in Regime):
        subset = [trade for trade in closed if trade.get("regime") == regime]
        if subset:
            by_regime[regime] = _trade_stats(subset, fills=[])

    totals = _trade_stats(closed, fills=fills)
    return {
        "results_root": str(Path(results_root)),
        "event_logs": [],
        "sessions": sessions,
        "totals": totals,
        "by_regime": by_regime,
        "by_actor": _actor_stats(closed),
        "temporal_windows": _temporal_windows(
            event_rows,
            closed,
            fills,
            window_count=temporal_window_count,
        ),
    }


def write_walk_forward_report(
    *,
    results_root: str = "Results",
    event_logs: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
    temporal_window_count: int = 5,
) -> str:
    report = build_walk_forward_report(
        results_root=results_root,
        event_logs=event_logs,
        temporal_window_count=temporal_window_count,
    )
    path = Path(output_path) if output_path else Path(results_root) / "walk_forward_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def write_walk_forward_report_from_events(
    *,
    results_root: str = "Results",
    events: Iterable[Mapping],
    output_path: Optional[str] = None,
    temporal_window_count: int = 5,
) -> str:
    report = build_walk_forward_report_from_events(
        results_root=results_root,
        events=events,
        temporal_window_count=temporal_window_count,
    )
    path = Path(output_path) if output_path else Path(results_root) / "walk_forward_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _default_event_logs() -> list[Path]:
    root = Path("logs")
    return sorted(root.glob("v2_*_events.jsonl")) if root.exists() else []


def _load_events(paths: Iterable[Path]) -> list[dict]:
    events: list[dict] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    payload["_source"] = str(path)
                    events.append(payload)
    return events


def _event_type(event: Mapping) -> str:
    return str(event.get("_type") or event.get("event_type") or event.get("type") or "")


def _regime_map(events: Iterable[Mapping]) -> dict[int, str]:
    out: dict[int, str] = {}
    current = Regime.NEUTRAL.label
    for event in events:
        try:
            bar = int(event.get("bar", 0) or 0)
        except (TypeError, ValueError):
            continue
        if _event_type(event) == "RegimeDetected":
            current = _normalize_regime(event.get("regime"))
        if bar > 0:
            out[bar] = current
    return out


def _closed_trades(events: Iterable[Mapping], regime_by_bar: Mapping[int, str]) -> list[dict]:
    out: list[dict] = []
    last_regime = Regime.NEUTRAL.label
    open_regime_by_signal_id: dict[int, str] = {}
    open_regime_by_sym: dict[str, str] = {}
    for event in events:
        try:
            bar = int(event.get("bar", 0) or 0)
        except (TypeError, ValueError):
            bar = 0
        if bar in regime_by_bar:
            last_regime = regime_by_bar[bar]
        event_type = _event_type(event)
        if event_type == "PositionOpened":
            open_regime = _normalize_regime(
                event.get("open_regime") or regime_by_bar.get(bar, last_regime)
            )
            signal_id = _int(event.get("signal_id"), default=-1)
            if signal_id >= 0:
                open_regime_by_signal_id[signal_id] = open_regime
            sym = str(event.get("sym") or "")
            if sym:
                open_regime_by_sym[sym] = open_regime
            continue
        if event_type != "PositionClosed":
            continue
        close_regime = _normalize_regime(regime_by_bar.get(bar, last_regime))
        open_signal_id = _int(event.get("open_signal_id"), default=-1)
        sym = str(event.get("sym") or "")
        entry_regime = _normalize_regime(
            event.get("open_regime")
            or event.get("entry_regime")
            or open_regime_by_signal_id.get(open_signal_id)
            or open_regime_by_sym.get(sym)
            or close_regime
        )
        pnl = _float(event.get("realized_pnl"))
        out.append({
            "bar": bar,
            "sym": sym,
            "realized_pnl": pnl,
            "by_player": str(event.get("by_player") or ""),
            "by_agent": str(event.get("by_agent") or ""),
            "side": str(event.get("side") or ""),
            "open_action": str(event.get("open_action") or event.get("action") or ""),
            "regime": entry_regime,
            "entry_regime": entry_regime,
            "exit_regime": close_regime,
        })
    return out


def _fills(events: Iterable[Mapping]) -> list[dict]:
    out: list[dict] = []
    for event in events:
        if _event_type(event) != "OrderFilled":
            continue
        trade = event.get("trade")
        if trade is None:
            continue
        notional = _float(_field(trade, "notional"))
        if notional <= 0:
            qty = _float(_field(trade, "qty"))
            price = _float(_field(trade, "fill_price") or _field(trade, "fillPrice"))
            notional = qty * price
        out.append({
            "bar": _int(event.get("bar"), default=0),
            "notional": notional,
            "fee": _float(_field(trade, "fee")),
            "funding": _float(_field(trade, "funding")),
            "sym": str(_field(trade, "sym") or ""),
        })
    return out


def _trade_stats(trades: list[Mapping], *, fills: list[Mapping]) -> dict:
    pnls = [_float(trade.get("realized_pnl")) for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    turnover = sum(_float(fill.get("notional")) for fill in fills)
    fees = sum(_float(fill.get("fee")) for fill in fills)
    funding = sum(_float(fill.get("funding")) for fill in fills)
    total_pnl = sum(pnls)
    return {
        "closed_trades": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "winrate_pct": (len(wins) / len(pnls) * 100.0) if pnls else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_pnl": total_pnl,
        "profit_factor": _profit_factor(gross_profit, gross_loss),
        "expectancy": (total_pnl / len(pnls)) if pnls else 0.0,
        "max_drawdown_pnl": _max_drawdown(pnls),
        "turnover_notional": turnover,
        "fees": fees,
        "funding": funding,
        "explicit_costs": fees + funding,
        "fee_per_turnover_pct": (fees / turnover * 100.0) if turnover > 0 else 0.0,
        "net_pnl_minus_5bps": total_pnl - turnover * 0.0005,
        "net_pnl_minus_10bps": total_pnl - turnover * 0.0010,
    }


def _actor_stats(trades: list[Mapping]) -> dict:
    buckets: dict[str, list[Mapping]] = {}
    for trade in trades:
        for key in ("by_player", "by_agent"):
            label = str(trade.get(key) or "")
            if label:
                buckets.setdefault(label, []).append(trade)
    return {
        label: _trade_stats(items, fills=[])
        for label, items in sorted(buckets.items())
    }


def _group_stats(
    trades: list[Mapping],
    key_fn: Callable[[Mapping], str],
) -> dict:
    buckets: dict[str, list[Mapping]] = {}
    for trade in trades:
        label = key_fn(trade)
        if label:
            buckets.setdefault(label, []).append(trade)
    return {
        label: _trade_stats(items, fills=[])
        for label, items in sorted(buckets.items())
    }


def _actor_labels(trade: Mapping) -> list[str]:
    labels = []
    player = str(trade.get("by_player") or "")
    agent = str(trade.get("by_agent") or "")
    if player:
        labels.append(f"player:{player}")
    if agent:
        labels.append(f"agent:{agent}")
    return labels


def _actor_keyed_stats(trades: list[Mapping]) -> dict:
    buckets: dict[str, list[Mapping]] = {}
    for trade in trades:
        for label in _actor_labels(trade):
            buckets.setdefault(label, []).append(trade)
    return {
        label: _trade_stats(items, fills=[])
        for label, items in sorted(buckets.items())
    }


def _actor_symbol_stats(trades: list[Mapping]) -> dict:
    buckets: dict[str, list[Mapping]] = {}
    for trade in trades:
        sym = str(trade.get("sym") or "")
        if not sym:
            continue
        for actor in _actor_labels(trade):
            buckets.setdefault(f"{actor}|{sym}", []).append(trade)
    return {
        label: _trade_stats(items, fills=[])
        for label, items in sorted(buckets.items())
    }


def _action_label(trade: Mapping) -> str:
    return str(trade.get("open_action") or trade.get("side") or "unknown")


def _ranked_stats_rows(stats: Mapping[str, Mapping], *, limit: int = 10) -> list[dict]:
    rows = []
    for key, value in stats.items():
        row = {"key": key}
        row.update(dict(value))
        rows.append(row)
    rows.sort(key=lambda row: (
        _float(row.get("net_pnl")),
        -_int(row.get("closed_trades"), default=0),
        str(row.get("key") or ""),
    ))
    return rows[:max(0, limit)]


def _temporal_windows(
    events: Iterable[Mapping],
    closed: list[Mapping],
    fills: list[Mapping],
    *,
    window_count: int,
) -> dict:
    event_rows = list(events)
    bars = [_int(event.get("bar"), default=0) for event in event_rows]
    bars = [bar for bar in bars if bar > 0]
    if not bars:
        return {
            "window_count": 0,
            "bar_start": 0,
            "bar_end": 0,
            "windows": [],
        }

    first_bar = min(bars)
    last_bar = max(bars)
    requested_windows = max(1, int(window_count or 1))
    available_span = max(1, last_bar - first_bar + 1)
    actual_windows = min(requested_windows, available_span)
    windows = []
    for index in range(actual_windows):
        start = first_bar + (index * available_span) // actual_windows
        end = first_bar + (((index + 1) * available_span) // actual_windows) - 1
        if index == actual_windows - 1:
            end = last_bar
        window_closed = [
            trade
            for trade in closed
            if start <= _int(trade.get("bar"), default=0) <= end
        ]
        window_fills = [
            fill
            for fill in fills
            if start <= _int(fill.get("bar"), default=0) <= end
        ]
        by_actor = _actor_keyed_stats(window_closed)
        by_symbol = _group_stats(
            window_closed,
            lambda trade: str(trade.get("sym") or ""),
        )
        by_action = _group_stats(window_closed, _action_label)
        by_actor_symbol = _actor_symbol_stats(window_closed)
        windows.append({
            "window": f"window_{index + 1:02d}",
            "bar_start": start,
            "bar_end": end,
            "stats": _trade_stats(window_closed, fills=window_fills),
            "by_actor": by_actor,
            "by_symbol": by_symbol,
            "by_action": by_action,
            "by_actor_symbol": by_actor_symbol,
            "worst_actors": _ranked_stats_rows(by_actor),
            "worst_symbols": _ranked_stats_rows(by_symbol),
            "worst_actions": _ranked_stats_rows(by_action),
            "worst_actor_symbols": _ranked_stats_rows(by_actor_symbol),
        })
    return {
        "window_count": len(windows),
        "bar_start": first_bar,
        "bar_end": last_bar,
        "windows": windows,
    }


def _session_summary(results_root: Path) -> dict:
    curves = []
    sessions = []
    if results_root.exists():
        for status_path in sorted(results_root.glob("*/*/status.json")):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            curve = _as_float_list(status.get("assets_curve") or status.get("equity_curve"))
            if curve:
                curves.append(curve)
            sessions.append({
                "path": str(status_path.parent),
                "exchange": str(status.get("exchange") or status_path.parents[1].name),
                "initial_capital": _float(status.get("initial_capital")),
                "current_balance": _float(status.get("current_balance")),
                "max_drawdown_pct": _curve_drawdown_pct(curve),
            })
    return {
        "count": len(sessions),
        "sessions": sessions,
        "max_drawdown_pct": max((_curve_drawdown_pct(c) for c in curves), default=0.0),
    }


def _normalize_regime(value) -> str:
    try:
        return Regime(int(value)).label
    except Exception:
        return Regime.from_string(str(value or "")).label


def _float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _field(value, key: str):
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _int(value, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float_list(value) -> list[float]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        val = _float(item)
        if math.isfinite(val):
            out.append(val)
    return out


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _curve_drawdown_pct(curve: list[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak * 100.0)
    return max_dd
