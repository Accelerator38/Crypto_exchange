from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = ROOT / "tools"
for item in (SRC, TOOLS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from panteon_v2.app.regime_detector import PriceRegimeDetector  # noqa: E402

import build_panteon_regime_breakdown_report as base_report  # noqa: E402


CURRENT_REGIME_REPORT_BASENAME = "current_regime_breakdown_report"
STABLECOIN_SUFFIXES = ("USDT", "USDC", "USD", "BUSD", "PERP")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-bucket an existing Panteon retrotest by the current v2 "
            "PriceRegimeDetector labels."
        )
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--exchange-name", default="RETRODATE_MARKET")
    parser.add_argument("--source-update-minutes", type=int, default=1)
    parser.add_argument("--max-bars", type=int, default=None)
    args = parser.parse_args(argv)
    build(
        args.output_dir,
        exchange_name=args.exchange_name,
        source_update_minutes=args.source_update_minutes,
        max_bars=args.max_bars,
    )
    return 0


def build(
    output_dir: Path,
    *,
    exchange_name: str = "RETRODATE_MARKET",
    source_update_minutes: int = 1,
    max_bars: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)

    run_summary = base_report._load_json(output_dir / "run_summary.json")
    initial_capital = base_report._safe_float(run_summary.get("initial_capital"), 1000.0)
    if initial_capital <= 0:
        initial_capital = 1000.0
    stride_minutes = max(1, base_report._safe_int(run_summary.get("stride_minutes")) or 60)
    csv_paths = _validation_paths_from_run_summary(run_summary)
    if not csv_paths:
        raise ValueError("run_summary.json does not contain validation_files paths")

    series = build_current_regime_series(
        csv_paths,
        stride_minutes=stride_minutes,
        exchange_name=exchange_name,
        source_update_minutes=source_update_minutes,
        max_bars=max_bars,
    )
    regime_by_bar = {int(row["bar"]): str(row["regime"]) for row in series}
    confidence_by_bar = {
        int(row["bar"]): float(row["confidence"]) for row in series
    }
    timestamps_by_bar = {int(row["bar"]): str(row["timestamp"]) for row in series}

    trading_rows = _override_trading_rows(
        base_report._parse_trading_log(output_dir / "trading.log"),
        regime_by_bar=regime_by_bar,
        confidence_by_bar=confidence_by_bar,
    )
    market = base_report._market_summary(
        trading_rows,
        initial_capital=initial_capital,
        timestamps_by_bar=timestamps_by_bar,
    )
    agents = _shadow_breakdown_with_regime_override(
        output_dir / "shadow_agent_pnl_events.jsonl",
        initial_capital=initial_capital,
        actor_type="agent",
        regime_by_bar=regime_by_bar,
    )
    players = _shadow_breakdown_with_regime_override(
        output_dir / "shadow_player_pnl_events.jsonl",
        initial_capital=initial_capital,
        actor_type="player",
        regime_by_bar=regime_by_bar,
    )
    soft_policy = base_report._soft_policy_from_run_summary(run_summary)
    soft_events = _read_shadow_events_with_regime_override(
        output_dir / "shadow_player_pnl_events.jsonl",
        regime_by_bar=regime_by_bar,
    )
    panteon_soft = base_report._soft_allocator_by_regime(
        soft_events,
        initial_capital=initial_capital,
        policy=soft_policy,
    )
    translation = _translation_matrix(
        base_report._parse_trading_log(output_dir / "trading.log"),
        regime_by_bar=regime_by_bar,
    )

    data = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "source_report": str(output_dir / "regime_breakdown_report.json"),
        "method": {
            "description": (
                "Existing retrotest PnL events are re-bucketed by the current "
                "PriceRegimeDetector labels using the same hourly Retrodate "
                "snapshots. Trades and shadow PnL are not re-simulated."
            ),
            "detector": "panteon_v2.app.regime_detector.PriceRegimeDetector",
            "exchange_name": exchange_name,
            "stride_minutes": stride_minutes,
            "source_update_minutes": source_update_minutes,
            "symbol_normalization": "BTC/USDT -> BTC",
        },
        "run_summary": {
            "bars_processed": run_summary.get("bars_processed"),
            "first_timestamp": run_summary.get("first_timestamp"),
            "last_timestamp": run_summary.get("last_timestamp"),
            "executed_years": run_summary.get("executed_years"),
            "initial_capital": initial_capital,
            "optional_agent_labels": run_summary.get("optional_agent_labels"),
        },
        "market": market,
        "panteon_by_current_regime": market["panteon_by_regime"],
        "panteon_soft_policy": soft_policy.name,
        "panteon_soft_by_current_regime": panteon_soft,
        "agents_by_current_regime": agents,
        "players_by_current_regime": players,
        "top_by_current_regime": {
            "agents": base_report._top_by_regime(agents),
            "players": base_report._top_by_regime(players),
        },
        "old_to_current_regime_matrix": translation,
    }

    _write_outputs(output_dir, data, series, translation)
    return data


def build_current_regime_series(
    csv_paths: Sequence[str | Path],
    *,
    stride_minutes: int,
    exchange_name: str = "RETRODATE_MARKET",
    source_update_minutes: int = 1,
    detector_kwargs: Mapping[str, Any] | None = None,
    max_bars: int | None = None,
) -> list[dict[str, Any]]:
    source_update_minutes = max(1, int(source_update_minutes or 1))
    detector_config: dict[str, Any] = {
        "exchange_name": exchange_name,
        "poll_interval_sec": float(source_update_minutes * 60),
    }
    detector_config.update(dict(detector_kwargs or {}))
    detector = PriceRegimeDetector(**detector_config)
    rows: list[dict[str, Any]] = []
    bar = 1
    for csv_path in csv_paths:
        for snapshot in _iter_retrodate_stride_snapshots(
            Path(csv_path),
            source_update_minutes=source_update_minutes,
            stride_minutes=stride_minutes,
        ):
            regime = detector.update(
                snapshot["prices"],
                volumes=snapshot["volumes"],
            )
            if not snapshot["sample_output"]:
                continue
            rows.append(
                {
                    "bar": bar,
                    "timestamp_ms": snapshot["timestamp_ms"],
                    "timestamp": _iso_from_ms(snapshot["timestamp_ms"]),
                    "regime": regime.label,
                    "confidence": float(detector.confidence),
                    "symbol_count": len(snapshot["prices"]),
                    "symbol_regimes": {
                        symbol: regime.label
                        for symbol, regime in sorted(detector.symbol_regimes.items())
                    },
                }
            )
            bar += 1
            if max_bars is not None and len(rows) >= int(max_bars):
                return rows
    return rows


def _iter_retrodate_stride_snapshots(
    csv_path: Path,
    *,
    source_update_minutes: int,
    stride_minutes: int,
) -> Iterable[dict[str, Any]]:
    if source_update_minutes <= 0:
        raise ValueError("source_update_minutes must be > 0")
    if stride_minutes <= 0:
        raise ValueError("stride_minutes must be > 0")
    source_stride_ms = int(source_update_minutes) * 60 * 1000
    stride_ms = int(stride_minutes) * 60 * 1000
    prices_by_ts: dict[int, dict[str, float]] = {}
    volumes_by_ts: dict[int, dict[str, float]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                timestamp = int(str(row.get("timestamp") or "").strip())
            except ValueError:
                continue
            if timestamp % source_stride_ms != 0:
                continue
            symbol = _normalize_detector_symbol(row.get("symbol"))
            if not symbol:
                continue
            close = _safe_positive_float(row.get("close"))
            if close <= 0.0:
                continue
            volume = max(0.0, _safe_float(row.get("volume")))
            prices_by_ts.setdefault(timestamp, {})[symbol] = close
            volume_bucket = volumes_by_ts.setdefault(timestamp, {})
            volume_bucket[symbol] = volume_bucket.get(symbol, 0.0) + volume

    for timestamp in sorted(prices_by_ts):
        prices = prices_by_ts[timestamp]
        yield {
            "timestamp_ms": timestamp,
            "prices": dict(sorted(prices.items())),
            "volumes": {
                symbol: volumes_by_ts.get(timestamp, {}).get(symbol, 0.0)
                for symbol in prices
            },
            "sample_output": timestamp % stride_ms == 0,
        }


def _normalize_detector_symbol(raw: Any) -> str:
    symbol = str(raw or "").strip().upper()
    if not symbol:
        return ""
    symbol = symbol.split(":", 1)[-1]
    symbol = symbol.replace("-", "/").replace("_", "/")
    if "/" in symbol:
        return symbol.split("/", 1)[0].strip()
    for suffix in STABLECOIN_SUFFIXES:
        if symbol.endswith(suffix) and len(symbol) > len(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _shadow_breakdown_with_regime_override(
    path: Path,
    *,
    initial_capital: float,
    actor_type: str,
    regime_by_bar: Mapping[int, str],
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
            bar = base_report._safe_int(row.get("bar"))
            if not label or bar <= 0:
                continue
            regime = base_report._regime_key(
                regime_by_bar.get(bar) or row.get("regime")
            )
            event_pnl = base_report._safe_float(row.get("pnl_usd"), 0.0)
            event_closed = max(0, base_report._safe_int(row.get("closed_trades")))
            event_wins = max(0, base_report._safe_int(row.get("wins")))
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
                },
            )
            bucket["pnl_usd"] += event_pnl
            bucket["closed_trades"] += event_closed
            bucket["wins"] += event_wins
            bucket["losses"] += max(0, event_closed - event_wins)
            bucket["bars_seen"] += 1

    rows: list[dict[str, Any]] = []
    for (_label, _regime), bucket in buckets.items():
        closed = int(bucket["closed_trades"])
        wins = int(bucket["wins"])
        pnl = float(bucket["pnl_usd"])
        rows.append(
            {
                "actor_type": actor_type,
                "label": bucket["label"],
                "regime": bucket["regime"],
                "pnl_usd": pnl,
                "pnl_pct": base_report._pct(pnl, initial_capital),
                "closed_trades": closed,
                "wins": wins,
                "losses": int(bucket["losses"]),
                "win_rate_pct": base_report._pct(wins, closed),
                "pnl_per_trade_usd": pnl / closed if closed else 0.0,
                "entries": 0,
                "signals": 0,
                "bars_seen": int(bucket["bars_seen"]),
            }
        )
    rows.sort(key=lambda row: (base_report._regime_sort_key(row["regime"]), row["label"]))
    return rows


def _read_shadow_events_with_regime_override(
    path: Path,
    *,
    regime_by_bar: Mapping[int, str],
) -> list[base_report.ShadowPnLEvent]:
    events: list[base_report.ShadowPnLEvent] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(row.get("label") or "").strip()
            bar = base_report._safe_int(row.get("bar"))
            if not label or bar <= 0:
                continue
            events.append(
                base_report.ShadowPnLEvent(
                    bar=bar,
                    label=label,
                    timestamp=str(row.get("timestamp") or ""),
                    regime=base_report._regime_key(
                        regime_by_bar.get(bar) or row.get("regime")
                    ),
                    pnl_usd=base_report._safe_float(row.get("pnl_usd"), 0.0),
                    closed_trades=max(0, base_report._safe_int(row.get("closed_trades"))),
                    wins=max(0, base_report._safe_int(row.get("wins"))),
                )
            )
    events.sort(key=lambda item: (item.bar, item.label))
    return events


def _override_trading_rows(
    rows: Sequence[dict[str, Any]],
    *,
    regime_by_bar: Mapping[int, str],
    confidence_by_bar: Mapping[int, float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        bar = base_report._safe_int(copied.get("bar"))
        if bar in regime_by_bar:
            copied["old_regime"] = copied.get("regime") or copied.get("market")
            copied["regime"] = regime_by_bar[bar]
            copied["market_confidence"] = confidence_by_bar.get(bar, math.nan)
        out.append(copied)
    return out


def _translation_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    regime_by_bar: Mapping[int, str],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        bar = base_report._safe_int(row.get("bar"))
        old_regime = base_report._regime_key(row.get("regime") or row.get("market"))
        current_regime = base_report._regime_key(regime_by_bar.get(bar) or old_regime)
        bucket = buckets.setdefault(
            (old_regime, current_regime),
            {
                "old_regime": old_regime,
                "current_regime": current_regime,
                "bars": 0,
                "signals": 0,
                "filled": 0,
                "shadow_filled": 0,
            },
        )
        bucket["bars"] += 1
        bucket["signals"] += base_report._safe_int(row.get("signals"))
        bucket["filled"] += base_report._safe_int(row.get("filled"))
        bucket["shadow_filled"] += base_report._safe_int(row.get("shadow_filled"))
    total = sum(int(row["bars"]) for row in buckets.values())
    out = []
    for bucket in buckets.values():
        item = dict(bucket)
        item["share_pct"] = base_report._pct(item["bars"], total)
        out.append(item)
    out.sort(
        key=lambda row: (
            base_report._regime_sort_key(row["old_regime"]),
            base_report._regime_sort_key(row["current_regime"]),
        )
    )
    return out


def _validation_paths_from_run_summary(summary: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for item in summary.get("validation_files") or []:
        if not isinstance(item, Mapping):
            continue
        raw = str(item.get("path") or "").strip()
        if raw:
            paths.append(Path(raw))
    return paths


def _write_outputs(
    output_dir: Path,
    data: dict[str, Any],
    series: Sequence[dict[str, Any]],
    translation: Sequence[dict[str, Any]],
) -> None:
    base_report._write_json(output_dir / f"{CURRENT_REGIME_REPORT_BASENAME}.json", data)
    base_report._write_csv(output_dir / "current_market_regime_summary.csv", data["market"]["regimes"])
    base_report._write_csv(output_dir / "current_panteon_regime_breakdown.csv", data["panteon_by_current_regime"])
    base_report._write_csv(output_dir / "current_panteon_soft_regime_breakdown.csv", data["panteon_soft_by_current_regime"])
    base_report._write_csv(output_dir / "current_agent_regime_breakdown.csv", data["agents_by_current_regime"])
    base_report._write_csv(output_dir / "current_player_regime_breakdown.csv", data["players_by_current_regime"])
    base_report._write_csv(output_dir / "current_regime_translation_matrix.csv", translation)
    base_report._write_csv(
        output_dir / "current_regime_series.csv",
        [
            {
                "bar": row["bar"],
                "timestamp": row["timestamp"],
                "regime": row["regime"],
                "confidence": row["confidence"],
                "symbol_count": row["symbol_count"],
            }
            for row in series
        ],
    )
    _write_markdown(output_dir / f"{CURRENT_REGIME_REPORT_BASENAME}.md", data)


def _write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Panteon current-regime breakdown",
        "",
        "Scope: existing retrotest trades and shadow PnL are re-bucketed by the current v2 PriceRegimeDetector labels. This is a translation/what-if report, not a full re-simulation of agent decisions.",
        "",
        "## Current Market Regimes",
        "",
        "| Regime | Bars | Share % | Avg confidence | Signals | Filled | Shadow filled |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["market"]["regimes"]:
        lines.append(
            "| "
            f"{row['regime']} | {row['bars']} | {base_report._fmt(row['share_pct'])} | "
            f"{base_report._fmt(row['avg_confidence'])} | {row['signals']} | "
            f"{row['filled']} | {row['shadow_filled']} |"
        )
    lines.extend(
        [
            "",
            "## Old To Current Regime Matrix",
            "",
            "| Old regime | Current regime | Bars | Share % | Filled | Shadow filled |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["old_to_current_regime_matrix"]:
        lines.append(
            "| "
            f"{row['old_regime']} | {row['current_regime']} | {row['bars']} | "
            f"{base_report._fmt(row['share_pct'])} | {row['filled']} | "
            f"{row['shadow_filled']} |"
        )
    lines.extend(
        [
            "",
            f"## PanteonSoft By Current Regime ({data['panteon_soft_policy']})",
            "",
            "| Regime | PnL % | PnL USD | Weighted closed | Weighted win % | Active bars | Avg cash % | Avg leaders |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["panteon_soft_by_current_regime"]:
        lines.append(
            "| "
            f"{row['regime']} | {base_report._fmt(row['pnl_pct'])} | "
            f"{base_report._fmt(row['pnl_usd'])} | "
            f"{base_report._fmt(row['weighted_closed_trades'])} | "
            f"{base_report._fmt(row['weighted_win_rate_pct'])} | {row['active_bars']} | "
            f"{base_report._fmt(row['average_cash_weight_pct'])} | "
            f"{base_report._fmt(row['average_leader_count'])} |"
        )
    lines.extend(_top_section("Top Agents By Current Regime", data["top_by_current_regime"]["agents"]))
    lines.extend(_top_section("Top Players By Current Regime", data["top_by_current_regime"]["players"]))
    lines.extend(_genetics_focus_section(data))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _top_section(title: str, grouped: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[str]:
    lines = ["", f"## {title}", ""]
    for regime, rows in grouped.items():
        lines.extend(
            [
                f"### {regime}",
                "",
                "| Label | PnL % | Closed | Win % | PnL/trade USD |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows[:8]:
            lines.append(
                "| "
                f"{row['label']} | {base_report._fmt(row['pnl_pct'])} | "
                f"{row['closed_trades']} | {base_report._fmt(row['win_rate_pct'])} | "
                f"{base_report._fmt(row['pnl_per_trade_usd'])} |"
            )
        lines.append("")
    return lines


def _genetics_focus_section(data: Mapping[str, Any]) -> list[str]:
    rows = [
        row
        for row in [
            *data["agents_by_current_regime"],
            *data["players_by_current_regime"],
        ]
        if "Genetics" in str(row.get("label") or "")
        and (
            abs(base_report._safe_float(row.get("pnl_usd"), 0.0)) > 1e-12
            or base_report._safe_int(row.get("closed_trades")) > 0
        )
    ]
    rows.sort(
        key=lambda row: (
            row["actor_type"],
            row["label"],
            base_report._regime_sort_key(row["regime"]),
        )
    )
    lines = [
        "",
        "## Genetics Focus",
        "",
        "| Type | Label | Current regime | PnL % | Closed | Win % | PnL/trade USD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['actor_type']} | {row['label']} | {row['regime']} | "
            f"{base_report._fmt(row['pnl_pct'])} | {row['closed_trades']} | "
            f"{base_report._fmt(row['win_rate_pct'])} | "
            f"{base_report._fmt(row['pnl_per_trade_usd'])} |"
        )
    return lines


def _iso_from_ms(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000.0, timezone.utc).isoformat()


def _safe_positive_float(value: Any) -> float:
    parsed = _safe_float(value)
    return parsed if parsed > 0.0 else 0.0


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
