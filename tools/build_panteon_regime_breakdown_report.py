from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.analysis.soft_allocator import (  # noqa: E402
    ShadowPnLEvent,
    SoftAllocatorPolicy,
    _ActorState,
    _apply_actor_event,
    _decay_states,
    _scope_for_event,
    _weights_for_policy,
)


REGIME_ORDER = (
    "bullish",
    "bearish",
    "neutral",
    "crash",
    "range_low_vol",
    "choppy_down",
    "choppy_up",
    "mixed_rotational",
    "unknown",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build market-regime breakdown report for a Panteon retrotest output dir."
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    build(args.output_dir)
    return 0


def build(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    run_summary = _load_json(output_dir / "run_summary.json")
    initial_capital = _safe_float(run_summary.get("initial_capital"), 1000.0)
    if initial_capital <= 0:
        initial_capital = 1000.0

    trading_rows = _parse_trading_log(output_dir / "trading.log")
    timestamps_by_bar = _timestamps_by_bar(
        output_dir / "shadow_agent_pnl_events.jsonl",
        output_dir / "shadow_player_pnl_events.jsonl",
    )
    leaderboard_regimes = {
        "agent": _load_leaderboard_regimes(output_dir, "agents"),
        "player": _load_leaderboard_regimes(output_dir, "players"),
    }

    market = _market_summary(
        trading_rows,
        initial_capital=initial_capital,
        timestamps_by_bar=timestamps_by_bar,
    )
    agents = _shadow_breakdown(
        output_dir / "shadow_agent_pnl_events.jsonl",
        initial_capital=initial_capital,
        actor_type="agent",
        leaderboard_regimes=leaderboard_regimes["agent"],
    )
    players = _shadow_breakdown(
        output_dir / "shadow_player_pnl_events.jsonl",
        initial_capital=initial_capital,
        actor_type="player",
        leaderboard_regimes=leaderboard_regimes["player"],
    )
    soft_policy = _soft_policy_from_run_summary(run_summary)
    soft_events = _read_shadow_events(output_dir / "shadow_player_pnl_events.jsonl")
    panteon_soft = _soft_allocator_by_regime(
        soft_events,
        initial_capital=initial_capital,
        policy=soft_policy,
    )

    data = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "run_summary": {
            "bars_processed": run_summary.get("bars_processed"),
            "first_timestamp": run_summary.get("first_timestamp"),
            "last_timestamp": run_summary.get("last_timestamp"),
            "executed_years": run_summary.get("executed_years"),
            "initial_capital": initial_capital,
            "include_optional_agents": run_summary.get("include_optional_agents"),
            "optional_agent_labels": run_summary.get("optional_agent_labels"),
            "registered_agents": run_summary.get("registered_agents"),
        },
        "market": market,
        "panteon_by_regime": market["panteon_by_regime"],
        "panteon_soft_policy": soft_policy.name,
        "panteon_soft_by_regime": panteon_soft,
        "agents_by_regime": agents,
        "players_by_regime": players,
        "top_by_regime": {
            "agents": _top_by_regime(agents),
            "players": _top_by_regime(players),
        },
    }

    _write_json(output_dir / "regime_breakdown_report.json", data)
    _write_csv(output_dir / "market_regime_summary.csv", market["regimes"])
    _write_csv(output_dir / "panteon_regime_breakdown.csv", market["panteon_by_regime"])
    _write_csv(output_dir / "panteon_soft_regime_breakdown.csv", panteon_soft)
    _write_csv(output_dir / "agent_regime_breakdown.csv", agents)
    _write_csv(output_dir / "player_regime_breakdown.csv", players)
    _write_markdown(output_dir / "regime_breakdown_report.md", data)
    return data


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


def _parse_trading_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields: dict[str, Any] = {}
        for part in line.split("  "):
            part = part.strip()
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            fields[key] = _coerce_trading_value(key, value)
        if fields.get("bar"):
            rows.append(fields)
    rows.sort(key=lambda item: int(item.get("bar", 0) or 0))
    return rows


def _coerce_trading_value(key: str, value: str) -> Any:
    if key in {
        "bar",
        "raw_signals",
        "signals",
        "filled",
        "rejected",
        "blocked",
        "shadow_signals",
        "shadow_filled",
        "shadow_session_signals",
        "shadow_session_filled",
        "positions",
    }:
        return _safe_int(value)
    if key in {"balance", "market_confidence"}:
        return _safe_float(value.replace("$", "").replace(",", ""), 0.0)
    return value


def _timestamps_by_bar(*paths: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                bar = _safe_int(row.get("bar"))
                if bar > 0 and bar not in out:
                    timestamp = str(row.get("timestamp") or "")
                    if timestamp:
                        out[bar] = timestamp
    return out


def _market_summary(
    rows: Sequence[dict[str, Any]],
    *,
    initial_capital: float,
    timestamps_by_bar: dict[int, str],
) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    year_regimes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    segments: list[dict[str, Any]] = []
    current_segment: dict[str, Any] | None = None
    prev_regime = ""
    prev_balance: float | None = None

    for row in rows:
        bar = _safe_int(row.get("bar"))
        regime = _regime_key(row.get("regime") or row.get("market"))
        timestamp = timestamps_by_bar.get(bar, "")
        bucket = buckets.setdefault(regime, _empty_market_bucket(regime))
        bucket["bars"] += 1
        bucket["raw_signals"] += _safe_int(row.get("raw_signals"))
        bucket["signals"] += _safe_int(row.get("signals"))
        bucket["filled"] += _safe_int(row.get("filled"))
        bucket["rejected"] += _safe_int(row.get("rejected"))
        bucket["blocked"] += _safe_int(row.get("blocked"))
        bucket["shadow_signals"] += _safe_int(row.get("shadow_signals"))
        bucket["shadow_filled"] += _safe_int(row.get("shadow_filled"))
        confidence = _safe_float(row.get("market_confidence"), math.nan)
        if math.isfinite(confidence):
            bucket["_confidence_sum"] += confidence
            bucket["_confidence_n"] += 1
        balance = _safe_float(row.get("balance"), math.nan)
        if math.isfinite(balance):
            if prev_balance is not None:
                bucket["panteon_pnl_usd"] += balance - prev_balance
            prev_balance = balance
            bucket["last_balance"] = balance
        if timestamp:
            bucket["first_timestamp"] = bucket["first_timestamp"] or timestamp
            bucket["last_timestamp"] = timestamp
            year_match = re.match(r"(\d{4})-", timestamp)
            if year_match:
                year_regimes[year_match.group(1)][regime] += 1

        if regime != prev_regime:
            if current_segment is not None:
                current_segment["end_bar"] = bar - 1
                current_segment["end_timestamp"] = timestamps_by_bar.get(bar - 1, "")
                current_segment["bars"] = (
                    int(current_segment["end_bar"]) - int(current_segment["start_bar"]) + 1
                )
                segments.append(current_segment)
            current_segment = {
                "regime": regime,
                "start_bar": bar,
                "start_timestamp": timestamp,
            }
            bucket["transitions_in"] += 1
            prev_regime = regime

    if current_segment is not None:
        last_bar = _safe_int(rows[-1].get("bar")) if rows else current_segment["start_bar"]
        current_segment["end_bar"] = last_bar
        current_segment["end_timestamp"] = timestamps_by_bar.get(last_bar, "")
        current_segment["bars"] = int(current_segment["end_bar"]) - int(current_segment["start_bar"]) + 1
        segments.append(current_segment)

    total_bars = sum(int(row["bars"]) for row in buckets.values())
    market_rows: list[dict[str, Any]] = []
    panteon_rows: list[dict[str, Any]] = []
    for regime, bucket in buckets.items():
        bars = int(bucket["bars"])
        avg_conf = (
            bucket["_confidence_sum"] / bucket["_confidence_n"]
            if bucket["_confidence_n"]
            else math.nan
        )
        segment_lengths = [int(item["bars"]) for item in segments if item["regime"] == regime]
        market_row = {
            "regime": regime,
            "bars": bars,
            "share_pct": _pct(bars, total_bars),
            "transitions_in": int(bucket["transitions_in"]),
            "avg_confidence": avg_conf,
            "avg_segment_bars": (
                sum(segment_lengths) / len(segment_lengths)
                if segment_lengths
                else 0.0
            ),
            "max_segment_bars": max(segment_lengths) if segment_lengths else 0,
            "first_timestamp": bucket["first_timestamp"],
            "last_timestamp": bucket["last_timestamp"],
            "raw_signals": int(bucket["raw_signals"]),
            "signals": int(bucket["signals"]),
            "filled": int(bucket["filled"]),
            "rejected": int(bucket["rejected"]),
            "blocked": int(bucket["blocked"]),
            "shadow_signals": int(bucket["shadow_signals"]),
            "shadow_filled": int(bucket["shadow_filled"]),
        }
        panteon_pnl = float(bucket["panteon_pnl_usd"])
        panteon_row = {
            "actor_type": "panteon",
            "label": "Panteon",
            "regime": regime,
            "pnl_usd": panteon_pnl,
            "pnl_pct": _pct(panteon_pnl, initial_capital),
            "signals": int(bucket["signals"]),
            "filled": int(bucket["filled"]),
            "rejected": int(bucket["rejected"]),
            "blocked": int(bucket["blocked"]),
            "bars": bars,
        }
        market_rows.append(market_row)
        panteon_rows.append(panteon_row)

    return {
        "total_bars": total_bars,
        "regimes": sorted(market_rows, key=lambda row: _regime_sort_key(row["regime"])),
        "segments": segments,
        "year_regime_bars": {
            year: dict(sorted(items.items(), key=lambda item: _regime_sort_key(item[0])))
            for year, items in sorted(year_regimes.items())
        },
        "panteon_by_regime": sorted(panteon_rows, key=lambda row: _regime_sort_key(row["regime"])),
    }


def _empty_market_bucket(regime: str) -> dict[str, Any]:
    return {
        "regime": regime,
        "bars": 0,
        "transitions_in": 0,
        "_confidence_sum": 0.0,
        "_confidence_n": 0,
        "first_timestamp": "",
        "last_timestamp": "",
        "raw_signals": 0,
        "signals": 0,
        "filled": 0,
        "rejected": 0,
        "blocked": 0,
        "shadow_signals": 0,
        "shadow_filled": 0,
        "panteon_pnl_usd": 0.0,
        "last_balance": math.nan,
    }


def _shadow_breakdown(
    path: Path,
    *,
    initial_capital: float,
    actor_type: str,
    leaderboard_regimes: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(row.get("label") or "").strip()
            if not label:
                continue
            regime = _regime_key(row.get("regime"))
            event_pnl = _safe_float(row.get("pnl_usd"), 0.0)
            event_closed = max(0, _safe_int(row.get("closed_trades")))
            event_wins = max(0, _safe_int(row.get("wins")))

            bucket = buckets.setdefault(
                (label, regime),
                {
                    "actor_type": actor_type,
                    "label": label,
                    "regime": regime,
                    "pnl_usd": 0.0,
                    "closed_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "bars_seen": 0,
                    "entries": 0,
                    "signals": 0,
                },
            )
            bucket["pnl_usd"] += event_pnl
            bucket["closed_trades"] += event_closed
            bucket["wins"] += event_wins
            bucket["losses"] += max(0, event_closed - event_wins)
            bucket["bars_seen"] += 1

    rows: list[dict[str, Any]] = []
    for (label, regime), bucket in buckets.items():
        lb = leaderboard_regimes.get((label, regime), {})
        entries = _safe_int(lb.get("entries"))
        signals = _safe_int(lb.get("signals"))
        closed = int(bucket["closed_trades"])
        wins = int(bucket["wins"])
        pnl = float(bucket["pnl_usd"])
        rows.append(
            {
                "actor_type": actor_type,
                "label": label,
                "regime": regime,
                "pnl_usd": pnl,
                "pnl_pct": _pct(pnl, initial_capital),
                "closed_trades": closed,
                "wins": wins,
                "losses": int(bucket["losses"]),
                "win_rate_pct": _pct(wins, closed),
                "pnl_per_trade_usd": pnl / closed if closed else 0.0,
                "entries": entries,
                "signals": signals,
                "bars_seen": int(bucket["bars_seen"]),
            }
        )
    rows.sort(key=lambda row: (_regime_sort_key(row["regime"]), row["label"]))
    return rows


def _read_shadow_events(path: Path) -> list[ShadowPnLEvent]:
    events: list[ShadowPnLEvent] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(row.get("label") or "").strip()
            bar = _safe_int(row.get("bar"))
            if not label or bar <= 0:
                continue
            events.append(
                ShadowPnLEvent(
                    bar=bar,
                    label=label,
                    timestamp=str(row.get("timestamp") or ""),
                    regime=_regime_key(row.get("regime")),
                    pnl_usd=_safe_float(row.get("pnl_usd"), 0.0),
                    closed_trades=max(0, _safe_int(row.get("closed_trades"))),
                    wins=max(0, _safe_int(row.get("wins"))),
                )
            )
    events.sort(key=lambda item: (item.bar, item.label))
    return events


def _soft_policy_from_run_summary(summary: dict[str, Any]) -> SoftAllocatorPolicy:
    name = str(summary.get("soft_allocator_execution_policy") or "").strip()
    if name == "soft_regime_top1_24_confirmed":
        return SoftAllocatorPolicy(
            name=name,
            top_k=1,
            cash_reserve_weight=0.0,
            max_weight_per_leader=1.0,
            min_closed_trades=50,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
            score_scope="regime",
            rolling_window_bars=24,
        )
    if name == "soft_regime_top1_24":
        return SoftAllocatorPolicy(
            name=name,
            top_k=1,
            cash_reserve_weight=0.0,
            max_weight_per_leader=1.0,
            min_closed_trades=20,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
            score_scope="regime",
            rolling_window_bars=24,
        )
    if "top3_w144_m20_cap45_cash15" in name:
        return SoftAllocatorPolicy(
            name=name or "regime_top3_w144_m20_cap45_cash15",
            top_k=3,
            cash_reserve_weight=0.15,
            max_weight_per_leader=0.45,
            min_closed_trades=20,
            half_life_bars=1_000_000,
            drawdown_penalty=0.0,
            persistent_loss_max_weight=0.0,
            score_scope="regime",
            rolling_window_bars=144,
        )
    return SoftAllocatorPolicy(
        name=name or "soft_regime_top1_24_confirmed",
        top_k=1,
        cash_reserve_weight=0.0,
        max_weight_per_leader=1.0,
        min_closed_trades=50,
        half_life_bars=1_000_000,
        drawdown_penalty=0.0,
        persistent_loss_max_weight=0.0,
        score_scope="regime",
        rolling_window_bars=24,
    )


def _soft_allocator_by_regime(
    events: Sequence[ShadowPnLEvent],
    *,
    initial_capital: float,
    policy: SoftAllocatorPolicy,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[ShadowPnLEvent]] = defaultdict(list)
    for event in events:
        if event.label and int(event.bar) > 0 and math.isfinite(float(event.pnl_usd)):
            grouped[int(event.bar)].append(event)

    states_by_scope: dict[str, dict[str, _ActorState]] = {}
    buckets: dict[str, dict[str, Any]] = {}
    last_bar = 0
    for bar in sorted(grouped):
        if last_bar:
            for states in states_by_scope.values():
                _decay_states(states.values(), elapsed_bars=max(0, bar - last_bar), policy=policy)
        last_bar = bar
        bar_events = grouped[bar]
        regime = _regime_key(bar_events[0].regime if bar_events else "")
        bucket = buckets.setdefault(
            regime,
            {
                "actor_type": "panteon_soft",
                "label": f"PanteonSoft:{policy.name}",
                "regime": regime,
                "pnl_usd": 0.0,
                "weighted_closed_trades": 0.0,
                "weighted_wins": 0.0,
                "active_bars": 0,
                "bars_seen": 0,
                "cash_weight_sum": 0.0,
                "leader_count_sum": 0.0,
            },
        )
        weights_by_scope = {
            scope: _weights_for_policy(states, policy, current_bar=bar)
            for scope, states in states_by_scope.items()
        }
        active_weights: dict[str, float] = {}
        for event in bar_events:
            scope = _scope_for_event(event, policy)
            weight = float(weights_by_scope.get(scope, {}).get(event.label, 0.0))
            if weight <= 0.0:
                continue
            active_weights[f"{scope}|{event.label}"] = weight
            bucket["pnl_usd"] += weight * float(event.pnl_usd)
            bucket["weighted_closed_trades"] += weight * max(0, int(event.closed_trades))
            bucket["weighted_wins"] += weight * max(0, int(event.wins))
        allocated = sum(active_weights.values())
        bucket["bars_seen"] += 1
        bucket["cash_weight_sum"] += max(0.0, 1.0 - allocated)
        bucket["leader_count_sum"] += sum(1 for value in active_weights.values() if value > 1e-9)
        if allocated > 1e-9:
            bucket["active_bars"] += 1

        for event in bar_events:
            scope = _scope_for_event(event, policy)
            states = states_by_scope.setdefault(scope, {})
            state = states.setdefault(
                event.label,
                _ActorState(equity_usd=initial_capital, peak_equity_usd=initial_capital),
            )
            _apply_actor_event(state, event, initial_capital=initial_capital)

    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        trades = float(bucket["weighted_closed_trades"])
        wins = float(bucket["weighted_wins"])
        bars_seen = int(bucket["bars_seen"])
        rows.append(
            {
                "actor_type": bucket["actor_type"],
                "label": bucket["label"],
                "regime": bucket["regime"],
                "pnl_usd": float(bucket["pnl_usd"]),
                "pnl_pct": _pct(float(bucket["pnl_usd"]), initial_capital),
                "weighted_closed_trades": trades,
                "weighted_wins": wins,
                "weighted_win_rate_pct": _pct(wins, trades),
                "active_bars": int(bucket["active_bars"]),
                "bars_seen": bars_seen,
                "average_cash_weight_pct": _pct(bucket["cash_weight_sum"], bars_seen),
                "average_leader_count": (
                    float(bucket["leader_count_sum"]) / bars_seen
                    if bars_seen
                    else 0.0
                ),
            }
        )
    rows.sort(key=lambda row: _regime_sort_key(row["regime"]))
    return rows


def _load_leaderboard_regimes(output_dir: Path, kind: str) -> dict[tuple[str, str], dict[str, Any]]:
    filenames = {
        "agents": ("leaderboard_agents.json", "leaderboard_shadow_only_agents.json"),
        "players": ("leaderboard_players.json", "leaderboard_shadow_only_players.json"),
    }[kind]
    section = "agents" if kind == "agents" else "players"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in filenames:
        data = _load_json(output_dir / filename)
        rows = data.get(section) or data.get("players") or data.get("agents") or {}
        if not isinstance(rows, dict):
            continue
        for raw_label, payload in rows.items():
            label = str(raw_label or "").strip()
            if label.startswith("V_"):
                label = label[2:]
            per_regime = payload.get("per_regime") if isinstance(payload, dict) else None
            if not isinstance(per_regime, dict):
                continue
            for regime, stats in per_regime.items():
                if isinstance(stats, dict):
                    out[(label, _regime_key(regime))] = stats
    return out


def _top_by_regime(rows: Sequence[dict[str, Any]], *, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("regime") or "unknown")].append(dict(row))
    return {
        regime: sorted(
            items,
            key=lambda row: (
                -_safe_float(row.get("pnl_usd"), 0.0),
                -_safe_int(row.get("closed_trades")),
                str(row.get("label") or ""),
            ),
        )[:limit]
        for regime, items in sorted(grouped.items(), key=lambda item: _regime_sort_key(item[0]))
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Panteon regime breakdown",
        "",
        "Scope: market regimes come from `trading.log`; `PanteonSoft` is the soft allocator replay over shadow player events; `Panteon execution log` is ordinary retrotest execution activity.",
        "",
        "## Market Regimes",
        "",
        "| Regime | Bars | Share % | Avg confidence | Signals | Filled | Blocked | Shadow filled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["market"]["regimes"]:
        lines.append(
            "| "
            f"{row['regime']} | {row['bars']} | {_fmt(row['share_pct'])} | "
            f"{_fmt(row['avg_confidence'])} | {row['signals']} | {row['filled']} | "
            f"{row['blocked']} | {row['shadow_filled']} |"
        )
    lines.extend(
        [
            "",
            f"## PanteonSoft By Regime ({data['panteon_soft_policy']})",
            "",
            "| Regime | PnL % | PnL USD | Weighted closed | Weighted win % | Active bars | Avg cash % | Avg leaders |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["panteon_soft_by_regime"]:
        lines.append(
            "| "
            f"{row['regime']} | {_fmt(row['pnl_pct'])} | {_fmt(row['pnl_usd'])} | "
            f"{_fmt(row['weighted_closed_trades'])} | {_fmt(row['weighted_win_rate_pct'])} | "
            f"{row['active_bars']} | {_fmt(row['average_cash_weight_pct'])} | "
            f"{_fmt(row['average_leader_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Panteon Execution Log By Regime",
            "",
            "| Regime | PnL % | PnL USD | Signals | Filled | Rejected | Blocked |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["panteon_by_regime"]:
        lines.append(
            "| "
            f"{row['regime']} | {_fmt(row['pnl_pct'])} | {_fmt(row['pnl_usd'])} | "
            f"{row['signals']} | {row['filled']} | {row['rejected']} | {row['blocked']} |"
        )

    lines.extend(_top_section("Top Agents By Regime", data["top_by_regime"]["agents"]))
    lines.extend(_top_section("Top Players By Regime", data["top_by_regime"]["players"]))
    lines.extend(_genetics_focus_section(data))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_section(title: str, grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines = ["", f"## {title}", ""]
    for regime, rows in grouped.items():
        lines.extend(
            [
                f"### {regime}",
                "",
                "| Label | PnL % | Closed | Win % | Entries | Signals |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows[:8]:
            lines.append(
                "| "
                f"{row['label']} | {_fmt(row['pnl_pct'])} | {row['closed_trades']} | "
                f"{_fmt(row['win_rate_pct'])} | {row.get('entries', 0)} | {row.get('signals', 0)} |"
            )
        lines.append("")
    return lines


def _genetics_focus_section(data: dict[str, Any]) -> list[str]:
    rows = [
        row
        for row in [*data["agents_by_regime"], *data["players_by_regime"]]
        if "Genetics" in str(row.get("label") or "")
        and (
            abs(_safe_float(row.get("pnl_usd"), 0.0)) > 1e-12
            or _safe_int(row.get("closed_trades")) > 0
            or _safe_int(row.get("signals")) > 0
        )
    ]
    rows.sort(
        key=lambda row: (
            row["actor_type"],
            row["label"],
            _regime_sort_key(row["regime"]),
        )
    )
    lines = [
        "",
        "## Genetics Focus",
        "",
        "| Type | Label | Regime | PnL % | Closed | Win % | Entries | Signals |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['actor_type']} | {row['label']} | {row['regime']} | {_fmt(row['pnl_pct'])} | "
            f"{row['closed_trades']} | {_fmt(row['win_rate_pct'])} | "
            f"{row.get('entries', 0)} | {row.get('signals', 0)} |"
        )
    if not rows:
        lines.append("| - | - | - | 0.00 | 0 | 0.00 | 0 | 0 |")
    return lines


def _regime_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def _regime_sort_key(regime: str) -> tuple[int, str]:
    key = _regime_key(regime)
    try:
        return (REGIME_ORDER.index(key), key)
    except ValueError:
        return (len(REGIME_ORDER), key)


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _pct(part: float | int, total: float | int) -> float:
    try:
        total_f = float(total)
        if total_f == 0.0:
            return 0.0
        return float(part) / total_f * 100.0
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: Any) -> str:
    number = _safe_float(value, math.nan)
    if not math.isfinite(number):
        return ""
    return f"{number:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
