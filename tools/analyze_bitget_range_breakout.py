from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATA_CSV = (
    "Retrodate/bitget_futures_current_20260709/"
    "crypto_1m_2026_all_symbols.csv"
)
DEFAULT_OUT = "Reports/BitgetRangeBreakout/20260712_range_breakout_report.json"
DEFAULT_LOOKBACK_BARS = (30, 60, 120)
DEFAULT_HOLD_BARS = (15, 30, 60)
DEFAULT_MAX_RANGE_PCT = (0.6, 1.0, 1.5)
DEFAULT_VOLUME_MULT = (1.2, 1.5, 2.0)
DEFAULT_BREAKOUT_BUFFER_BPS = (2.0, 5.0, 10.0)
DEFAULT_ROUND_TRIP_COST_BPS = 8.0
DEFAULT_NOTIONAL_USD = 5.0
DEFAULT_MIN_CLOSED = 10
DEFAULT_MAX_DRAWDOWN_USD = 0.20
DEFAULT_OOS_FRACTION = 0.35
DEFAULT_STRESS_ROUND_TRIP_COST_BPS = (12.0,)
LCB_Z = 1.64


def load_bars_by_symbol(csv_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    bars_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                symbol = str(row["symbol"]).strip()
                bar = {
                    "timestamp": int(float(row["timestamp"])),
                    "datetime": row.get("datetime", ""),
                    "symbol": symbol,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if symbol and bar["close"] > 0.0:
                bars_by_symbol[symbol].append(bar)
    for symbol in list(bars_by_symbol):
        bars_by_symbol[symbol].sort(key=lambda item: (item["timestamp"], item["datetime"]))
    return dict(sorted(bars_by_symbol.items()))


def scan_breakouts(
    bars: Sequence[dict[str, Any]],
    *,
    symbol: str,
    lookback_bars: int,
    hold_bars: int,
    max_range_pct: float,
    volume_mult: float,
    breakout_buffer_bps: float,
    round_trip_cost_bps: float,
    notional_usd: float,
) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    if lookback_bars <= 1 or hold_bars <= 0:
        return trades
    next_allowed_index = lookback_bars
    buffer_mult = float(breakout_buffer_bps) / 10000.0
    cost_pct = float(round_trip_cost_bps) / 100.0
    last_entry_index = len(bars) - hold_bars - 1
    for index in range(lookback_bars, max(lookback_bars, last_entry_index + 1)):
        if index < next_allowed_index:
            continue
        window = bars[index - lookback_bars:index]
        highs = [float(item["high"]) for item in window]
        lows = [float(item["low"]) for item in window]
        volumes = [float(item["volume"]) for item in window]
        prior_high = max(highs)
        prior_low = min(lows)
        mid = (prior_high + prior_low) / 2.0
        if mid <= 0.0:
            continue
        range_pct = (prior_high - prior_low) / mid * 100.0
        if range_pct > max_range_pct:
            continue
        avg_volume = sum(volumes) / len(volumes)
        if avg_volume <= 0.0:
            continue
        bar = bars[index]
        if float(bar["volume"]) < avg_volume * volume_mult:
            continue
        close = float(bar["close"])
        direction = ""
        if close > prior_high * (1.0 + buffer_mult):
            direction = "LONG"
            direction_sign = 1.0
        elif close < prior_low * (1.0 - buffer_mult):
            direction = "SHORT"
            direction_sign = -1.0
        else:
            continue
        exit_index = index + hold_bars
        if exit_index >= len(bars):
            continue
        exit_close = float(bars[exit_index]["close"])
        gross_pct = direction_sign * ((exit_close - close) / close) * 100.0
        net_pct = gross_pct - cost_pct
        net_usd = notional_usd * net_pct / 100.0
        path = bars[index + 1:exit_index + 1]
        mfe_pct, mae_pct = _path_excursion_pct(path, entry=close, direction_sign=direction_sign)
        trades.append({
            "symbol": symbol,
            "direction": direction,
            "entry_index": index,
            "exit_index": exit_index,
            "entry_timestamp": bar["timestamp"],
            "exit_timestamp": bars[exit_index]["timestamp"],
            "entry_datetime": bar.get("datetime", ""),
            "exit_datetime": bars[exit_index].get("datetime", ""),
            "entry_price": close,
            "exit_price": exit_close,
            "prior_high": prior_high,
            "prior_low": prior_low,
            "range_pct": range_pct,
            "volume_ratio": float(bar["volume"]) / avg_volume,
            "gross_pct": gross_pct,
            "net_pct": net_pct,
            "net_usd": net_usd,
            "mfe_pct": mfe_pct,
            "mae_pct": mae_pct,
        })
        next_allowed_index = exit_index + 1
    return trades


def summarize_trades(
    trades: Sequence[dict[str, Any]],
    *,
    min_closed: int = DEFAULT_MIN_CLOSED,
    max_drawdown_usd: float = DEFAULT_MAX_DRAWDOWN_USD,
) -> dict[str, Any]:
    nets = [float(trade["net_usd"]) for trade in trades]
    closed = len(nets)
    net_pnl = sum(nets)
    expectancy = net_pnl / closed if closed else 0.0
    lcb = _lcb(nets)
    drawdown = _max_drawdown(nets)
    wins = [value for value in nets if value > 0.0]
    losses = [value for value in nets if value < 0.0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    fail_reasons: list[str] = []
    if closed < min_closed:
        fail_reasons.append("min_closed")
    if expectancy <= 0.0:
        fail_reasons.append("nonpositive_expectancy")
    if lcb <= 0.0:
        fail_reasons.append("nonpositive_lcb")
    if drawdown > max_drawdown_usd:
        fail_reasons.append("drawdown_limit")
    avg_net_pct = (
        sum(float(trade["net_pct"]) for trade in trades) / closed if closed else 0.0
    )
    profit_factor = None
    if gross_loss < 0.0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0.0:
        profit_factor = 999.0
    return {
        "closed_trades": closed,
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / closed if closed else 0.0,
        "net_pnl_usd": net_pnl,
        "expectancy_usd": expectancy,
        "lcb_usd": lcb,
        "max_drawdown_usd": drawdown,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": profit_factor,
        "avg_net_pct": avg_net_pct,
        "eligible": not fail_reasons,
        "fail_reasons": fail_reasons,
    }


def analyze_bars(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    lookback_grid: Sequence[int] = DEFAULT_LOOKBACK_BARS,
    hold_grid: Sequence[int] = DEFAULT_HOLD_BARS,
    max_range_grid: Sequence[float] = DEFAULT_MAX_RANGE_PCT,
    volume_mult_grid: Sequence[float] = DEFAULT_VOLUME_MULT,
    breakout_buffer_grid: Sequence[float] = DEFAULT_BREAKOUT_BUFFER_BPS,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    notional_usd: float = DEFAULT_NOTIONAL_USD,
    min_closed: int = DEFAULT_MIN_CLOSED,
    max_drawdown_usd: float = DEFAULT_MAX_DRAWDOWN_USD,
    oos_fraction: float = DEFAULT_OOS_FRACTION,
    min_oos_closed: int | None = None,
    stress_round_trip_cost_bps: Sequence[float] = DEFAULT_STRESS_ROUND_TRIP_COST_BPS,
    top_n: int = 30,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    aggregate_candidates: list[dict[str, Any]] = []
    scanned_configs = 0
    split_timestamp = _split_timestamp(bars_by_symbol, oos_fraction=oos_fraction)
    resolved_min_oos_closed = (
        int(min_oos_closed)
        if min_oos_closed is not None
        else max(1, int(math.ceil(min_closed * max(0.0, min(1.0, oos_fraction)))))
    )
    for lookback in lookback_grid:
        for hold in hold_grid:
            for max_range in max_range_grid:
                for volume_mult in volume_mult_grid:
                    for buffer_bps in breakout_buffer_grid:
                        scanned_configs += 1
                        aggregate_trades: dict[str, list[dict[str, Any]]] = {
                            "LONG": [],
                            "SHORT": [],
                        }
                        aggregate_counts: dict[str, dict[str, int]] = {
                            "LONG": {},
                            "SHORT": {},
                        }
                        for symbol, bars in bars_by_symbol.items():
                            trades = scan_breakouts(
                                bars,
                                symbol=symbol,
                                lookback_bars=int(lookback),
                                hold_bars=int(hold),
                                max_range_pct=float(max_range),
                                volume_mult=float(volume_mult),
                                breakout_buffer_bps=float(buffer_bps),
                                round_trip_cost_bps=round_trip_cost_bps,
                                notional_usd=notional_usd,
                            )
                            for direction in ("LONG", "SHORT"):
                                direction_trades = [
                                    trade for trade in trades if trade["direction"] == direction
                                ]
                                if direction_trades:
                                    clean_symbol = _clean_symbol(symbol)
                                    aggregate_trades[direction].extend(direction_trades)
                                    aggregate_counts[direction][clean_symbol] = len(direction_trades)
                                summary = summarize_trades(
                                    direction_trades,
                                    min_closed=min_closed,
                                    max_drawdown_usd=max_drawdown_usd,
                                )
                                validation = build_validation_summary(
                                    direction_trades,
                                    base_summary=summary,
                                    split_timestamp=split_timestamp,
                                    min_closed=min_closed,
                                    min_oos_closed=resolved_min_oos_closed,
                                    max_drawdown_usd=max_drawdown_usd,
                                    stress_round_trip_cost_bps=stress_round_trip_cost_bps,
                                    notional_usd=notional_usd,
                                )
                                key = (
                                    "RangeCompressionBreakout|"
                                    f"{_clean_symbol(symbol)}|range_low_vol_transition|"
                                    f"{direction}|lb{lookback}|hold{hold}|"
                                    f"range{_grid_value(max_range)}|vol{_grid_value(volume_mult)}|"
                                    f"buf{_grid_value(buffer_bps)}"
                                )
                                candidates.append({
                                    "key": key,
                                    "actor_label": "RangeCompressionBreakout",
                                    "symbol": _clean_symbol(symbol),
                                    "raw_symbol": symbol,
                                    "regime": "range_low_vol_transition",
                                    "direction": direction,
                                    "lookback_bars": int(lookback),
                                    "hold_bars": int(hold),
                                    "max_range_pct": float(max_range),
                                    "volume_mult": float(volume_mult),
                                    "breakout_buffer_bps": float(buffer_bps),
                                    "round_trip_cost_bps": float(round_trip_cost_bps),
                                    "notional_usd": float(notional_usd),
                                    **summary,
                                    **validation,
                                    "sample_trades": direction_trades[:5],
                                })
                        for direction in ("LONG", "SHORT"):
                            direction_trades = sorted(
                                aggregate_trades[direction],
                                key=lambda item: (item["entry_timestamp"], item["symbol"]),
                            )
                            summary = summarize_trades(
                                direction_trades,
                                min_closed=min_closed,
                                max_drawdown_usd=max_drawdown_usd,
                            )
                            validation = build_validation_summary(
                                direction_trades,
                                base_summary=summary,
                                split_timestamp=split_timestamp,
                                min_closed=min_closed,
                                min_oos_closed=resolved_min_oos_closed,
                                max_drawdown_usd=max_drawdown_usd,
                                stress_round_trip_cost_bps=stress_round_trip_cost_bps,
                                notional_usd=notional_usd,
                            )
                            key = (
                                "RangeCompressionBreakout|MULTI|range_low_vol_transition|"
                                f"{direction}|lb{lookback}|hold{hold}|"
                                f"range{_grid_value(max_range)}|vol{_grid_value(volume_mult)}|"
                                f"buf{_grid_value(buffer_bps)}"
                            )
                            aggregate_candidates.append({
                                "key": key,
                                "actor_label": "RangeCompressionBreakout",
                                "symbol": "MULTI",
                                "symbols": dict(sorted(aggregate_counts[direction].items())),
                                "regime": "range_low_vol_transition",
                                "direction": direction,
                                "lookback_bars": int(lookback),
                                "hold_bars": int(hold),
                                "max_range_pct": float(max_range),
                                "volume_mult": float(volume_mult),
                                "breakout_buffer_bps": float(buffer_bps),
                                "round_trip_cost_bps": float(round_trip_cost_bps),
                                "notional_usd": float(notional_usd),
                                **summary,
                                **validation,
                                "sample_trades": direction_trades[:8],
                            })
    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item["robust_eligible"]),
            bool(item["eligible"]),
            float(item["lcb_usd"]),
            float(item["expectancy_usd"]),
            int(item["closed_trades"]),
            -float(item["max_drawdown_usd"]),
        ),
        reverse=True,
    )
    aggregate_ranked = sorted(
        aggregate_candidates,
        key=lambda item: (
            bool(item["robust_eligible"]),
            bool(item["eligible"]),
            float(item["lcb_usd"]),
            float(item["expectancy_usd"]),
            int(item["closed_trades"]),
            -float(item["max_drawdown_usd"]),
        ),
        reverse=True,
    )
    eligible = [item for item in ranked if item["eligible"]]
    aggregate_eligible = [item for item in aggregate_ranked if item["eligible"]]
    robust_eligible = [item for item in ranked if item["robust_eligible"]]
    aggregate_robust_eligible = [
        item for item in aggregate_ranked if item["robust_eligible"]
    ]
    return {
        "metadata": {
            "data_symbols": sorted(_clean_symbol(symbol) for symbol in bars_by_symbol),
            "bars_by_symbol": {
                _clean_symbol(symbol): len(bars) for symbol, bars in bars_by_symbol.items()
            },
            "scanned_parameter_sets": scanned_configs,
            "scanned_symbol_direction_candidates": len(candidates),
            "scanned_aggregate_candidates": len(aggregate_candidates),
            "round_trip_cost_bps": float(round_trip_cost_bps),
            "notional_usd": float(notional_usd),
            "min_closed": int(min_closed),
            "min_oos_closed": int(resolved_min_oos_closed),
            "max_drawdown_usd": float(max_drawdown_usd),
            "oos_fraction": float(oos_fraction),
            "split_timestamp": split_timestamp,
            "stress_round_trip_cost_bps": [float(value) for value in stress_round_trip_cost_bps],
        },
        "eligible_count": len(eligible),
        "aggregate_eligible_count": len(aggregate_eligible),
        "robust_eligible_count": len(robust_eligible),
        "aggregate_robust_eligible_count": len(aggregate_robust_eligible),
        "hard_blocked": len(eligible) == 0,
        "aggregate_hard_blocked": len(aggregate_eligible) == 0,
        "robust_hard_blocked": len(robust_eligible) == 0,
        "aggregate_robust_hard_blocked": len(aggregate_robust_eligible) == 0,
        "top_candidates": ranked[:top_n],
        "eligible_candidates": eligible[:top_n],
        "robust_eligible_candidates": robust_eligible[:top_n],
        "aggregate_top_candidates": aggregate_ranked[:top_n],
        "aggregate_eligible_candidates": aggregate_eligible[:top_n],
        "aggregate_robust_eligible_candidates": aggregate_robust_eligible[:top_n],
        "notes": [
            "Research-only offline analyzer; this does not grant live trading approval.",
            "Candidates represent explicit range compression breakout confirmation, not plain range_low_vol trading.",
            "Net metrics subtract configured round-trip fee/slippage cost.",
            "Aggregate candidates combine the same parameter set and direction across symbols.",
            "robust_eligible additionally requires train/OOS pass and configured stress-cost pass.",
        ],
    }


def write_report(report: dict[str, Any], out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def parse_number_list(value: str, *, cast: type = float) -> list[Any]:
    result: list[Any] = []
    for item in value.split(","):
        stripped = item.strip()
        if stripped:
            result.append(cast(stripped))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Bitget range compression breakout candidates after costs."
    )
    parser.add_argument("--data-csv", default=DEFAULT_DATA_CSV)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--lookback-bars", default=",".join(map(str, DEFAULT_LOOKBACK_BARS)))
    parser.add_argument("--hold-bars", default=",".join(map(str, DEFAULT_HOLD_BARS)))
    parser.add_argument("--max-range-pct", default=",".join(map(str, DEFAULT_MAX_RANGE_PCT)))
    parser.add_argument("--volume-mult", default=",".join(map(str, DEFAULT_VOLUME_MULT)))
    parser.add_argument(
        "--breakout-buffer-bps",
        default=",".join(map(str, DEFAULT_BREAKOUT_BUFFER_BPS)),
    )
    parser.add_argument("--round-trip-cost-bps", type=float, default=DEFAULT_ROUND_TRIP_COST_BPS)
    parser.add_argument("--notional-usd", type=float, default=DEFAULT_NOTIONAL_USD)
    parser.add_argument("--min-closed", type=int, default=DEFAULT_MIN_CLOSED)
    parser.add_argument("--min-oos-closed", type=int, default=None)
    parser.add_argument("--max-drawdown-usd", type=float, default=DEFAULT_MAX_DRAWDOWN_USD)
    parser.add_argument("--oos-fraction", type=float, default=DEFAULT_OOS_FRACTION)
    parser.add_argument(
        "--stress-round-trip-cost-bps",
        default=",".join(map(str, DEFAULT_STRESS_ROUND_TRIP_COST_BPS)),
    )
    parser.add_argument("--top-n", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bars_by_symbol = load_bars_by_symbol(args.data_csv)
    report = analyze_bars(
        bars_by_symbol,
        lookback_grid=parse_number_list(args.lookback_bars, cast=int),
        hold_grid=parse_number_list(args.hold_bars, cast=int),
        max_range_grid=parse_number_list(args.max_range_pct, cast=float),
        volume_mult_grid=parse_number_list(args.volume_mult, cast=float),
        breakout_buffer_grid=parse_number_list(args.breakout_buffer_bps, cast=float),
        round_trip_cost_bps=args.round_trip_cost_bps,
        notional_usd=args.notional_usd,
        min_closed=args.min_closed,
        min_oos_closed=args.min_oos_closed,
        max_drawdown_usd=args.max_drawdown_usd,
        oos_fraction=args.oos_fraction,
        stress_round_trip_cost_bps=parse_number_list(
            args.stress_round_trip_cost_bps,
            cast=float,
        ),
        top_n=args.top_n,
    )
    out_path = write_report(report, args.out)
    print(json.dumps({
        "aggregate_eligible_count": report["aggregate_eligible_count"],
        "aggregate_hard_blocked": report["aggregate_hard_blocked"],
        "aggregate_robust_eligible_count": report["aggregate_robust_eligible_count"],
        "aggregate_robust_hard_blocked": report["aggregate_robust_hard_blocked"],
        "out": str(out_path),
        "eligible_count": report["eligible_count"],
        "hard_blocked": report["hard_blocked"],
        "robust_eligible_count": report["robust_eligible_count"],
        "robust_hard_blocked": report["robust_hard_blocked"],
        "top_key": (
            report["top_candidates"][0]["key"] if report["top_candidates"] else None
        ),
    }, sort_keys=True))
    return 0


def build_validation_summary(
    trades: Sequence[dict[str, Any]],
    *,
    base_summary: dict[str, Any],
    split_timestamp: int | None,
    min_closed: int,
    min_oos_closed: int,
    max_drawdown_usd: float,
    stress_round_trip_cost_bps: Sequence[float],
    notional_usd: float,
) -> dict[str, Any]:
    train_trades, oos_trades = split_trades(trades, split_timestamp=split_timestamp)
    train_min_closed = max(1, min_closed - min_oos_closed)
    train_summary = summarize_trades(
        train_trades,
        min_closed=train_min_closed,
        max_drawdown_usd=max_drawdown_usd,
    )
    oos_summary = summarize_trades(
        oos_trades,
        min_closed=min_oos_closed,
        max_drawdown_usd=max_drawdown_usd,
    )
    stress_summaries: dict[str, dict[str, Any]] = {}
    stress_oos_summaries: dict[str, dict[str, Any]] = {}
    for cost_bps in stress_round_trip_cost_bps:
        key = _grid_value(float(cost_bps))
        repriced = reprice_trades(
            trades,
            round_trip_cost_bps=float(cost_bps),
            notional_usd=notional_usd,
        )
        _, repriced_oos = split_trades(repriced, split_timestamp=split_timestamp)
        stress_summaries[key] = summarize_trades(
            repriced,
            min_closed=min_closed,
            max_drawdown_usd=max_drawdown_usd,
        )
        stress_oos_summaries[key] = summarize_trades(
            repriced_oos,
            min_closed=min_oos_closed,
            max_drawdown_usd=max_drawdown_usd,
        )
    robust_fail_reasons = _robust_fail_reasons(
        base_summary=base_summary,
        train_summary=train_summary,
        oos_summary=oos_summary,
        stress_summaries=stress_summaries,
        stress_oos_summaries=stress_oos_summaries,
    )
    return {
        "train_summary": train_summary,
        "oos_summary": oos_summary,
        "stress_summaries": stress_summaries,
        "stress_oos_summaries": stress_oos_summaries,
        "robust_eligible": not robust_fail_reasons,
        "robust_fail_reasons": robust_fail_reasons,
    }


def split_trades(
    trades: Sequence[dict[str, Any]],
    *,
    split_timestamp: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split_timestamp is None:
        return list(trades), list(trades)
    train: list[dict[str, Any]] = []
    oos: list[dict[str, Any]] = []
    for trade in trades:
        target = oos if int(trade["entry_timestamp"]) >= split_timestamp else train
        target.append(trade)
    return train, oos


def reprice_trades(
    trades: Sequence[dict[str, Any]],
    *,
    round_trip_cost_bps: float,
    notional_usd: float,
) -> list[dict[str, Any]]:
    cost_pct = float(round_trip_cost_bps) / 100.0
    repriced: list[dict[str, Any]] = []
    for trade in trades:
        copy = dict(trade)
        net_pct = float(copy["gross_pct"]) - cost_pct
        copy["net_pct"] = net_pct
        copy["net_usd"] = float(notional_usd) * net_pct / 100.0
        copy["round_trip_cost_bps"] = float(round_trip_cost_bps)
        repriced.append(copy)
    return repriced


def _path_excursion_pct(
    path: Sequence[dict[str, Any]],
    *,
    entry: float,
    direction_sign: float,
) -> tuple[float, float]:
    if not path or entry <= 0.0:
        return 0.0, 0.0
    favorable: list[float] = []
    adverse: list[float] = []
    for bar in path:
        high = float(bar["high"])
        low = float(bar["low"])
        if direction_sign > 0.0:
            favorable.append((high - entry) / entry * 100.0)
            adverse.append((low - entry) / entry * 100.0)
        else:
            favorable.append((entry - low) / entry * 100.0)
            adverse.append((entry - high) / entry * 100.0)
    return max(favorable), min(adverse)


def _lcb(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    stderr = math.sqrt(variance) / math.sqrt(len(values))
    return mean - LCB_Z * stderr


def _max_drawdown(values: Iterable[float]) -> float:
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for value in values:
        cumulative += float(value)
        peak = max(peak, cumulative)
        max_dd = max(max_dd, peak - cumulative)
    return max_dd


def _split_timestamp(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    *,
    oos_fraction: float,
) -> int | None:
    fraction = max(0.0, min(1.0, float(oos_fraction)))
    if fraction <= 0.0:
        return None
    timestamps = sorted({
        int(bar["timestamp"])
        for bars in bars_by_symbol.values()
        for bar in bars
    })
    if not timestamps:
        return None
    split_index = int(len(timestamps) * (1.0 - fraction))
    split_index = max(0, min(split_index, len(timestamps) - 1))
    return timestamps[split_index]


def _robust_fail_reasons(
    *,
    base_summary: dict[str, Any],
    train_summary: dict[str, Any],
    oos_summary: dict[str, Any],
    stress_summaries: dict[str, dict[str, Any]],
    stress_oos_summaries: dict[str, dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if not base_summary["eligible"]:
        reasons.extend(f"base_{reason}" for reason in base_summary["fail_reasons"])
    if not train_summary["eligible"]:
        reasons.extend(f"train_{reason}" for reason in train_summary["fail_reasons"])
    if not oos_summary["eligible"]:
        reasons.extend(f"oos_{reason}" for reason in oos_summary["fail_reasons"])
    for cost_key, summary in sorted(stress_summaries.items()):
        if not summary["eligible"]:
            reasons.extend(
                f"stress_{cost_key}_{reason}" for reason in summary["fail_reasons"]
            )
    for cost_key, summary in sorted(stress_oos_summaries.items()):
        if not summary["eligible"]:
            reasons.extend(
                f"stress_oos_{cost_key}_{reason}" for reason in summary["fail_reasons"]
            )
    return sorted(dict.fromkeys(reasons))


def _clean_symbol(symbol: str) -> str:
    return symbol.replace("/USDT", "").replace("USDT", "")


def _grid_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


if __name__ == "__main__":
    raise SystemExit(main())
