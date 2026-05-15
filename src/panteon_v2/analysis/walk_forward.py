"""Walk-forward/retro report over saved Results and JSONL event logs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional

from ..domain.types import Regime


def build_walk_forward_report(
    *,
    results_root: str = "Results",
    event_logs: Optional[Iterable[str]] = None,
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
    }


def write_walk_forward_report(
    *,
    results_root: str = "Results",
    event_logs: Optional[Iterable[str]] = None,
    output_path: Optional[str] = None,
) -> str:
    report = build_walk_forward_report(results_root=results_root, event_logs=event_logs)
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
    for event in events:
        try:
            bar = int(event.get("bar", 0) or 0)
        except (TypeError, ValueError):
            bar = 0
        if bar in regime_by_bar:
            last_regime = regime_by_bar[bar]
        if _event_type(event) != "PositionClosed":
            continue
        pnl = _float(event.get("realized_pnl"))
        out.append({
            "bar": bar,
            "sym": str(event.get("sym") or ""),
            "realized_pnl": pnl,
            "by_player": str(event.get("by_player") or ""),
            "by_agent": str(event.get("by_agent") or ""),
            "regime": regime_by_bar.get(bar, last_regime),
        })
    return out


def _fills(events: Iterable[Mapping]) -> list[dict]:
    out: list[dict] = []
    for event in events:
        if _event_type(event) != "OrderFilled":
            continue
        trade = event.get("trade")
        if not isinstance(trade, dict):
            continue
        notional = _float(trade.get("notional"))
        if notional <= 0:
            qty = _float(trade.get("qty"))
            price = _float(trade.get("fill_price") or trade.get("fillPrice"))
            notional = qty * price
        out.append({
            "notional": notional,
            "fee": _float(trade.get("fee")),
            "sym": str(trade.get("sym") or ""),
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
