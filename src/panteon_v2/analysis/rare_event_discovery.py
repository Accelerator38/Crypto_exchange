"""Fixed low-turnover Bitget event hypotheses with anchored OOS evaluation."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cost_aware_directional import LCB_Z, _load_history


REPORT_SCHEMA_VERSION = "panteon.bitget_rare_event_discovery.v1"
DECISION_CADENCE_MINUTES = 5
HORIZON_MINUTES = 240
TAKE_PROFIT_GROSS_BPS = 60.0
STOP_LOSS_GROSS_BPS = 30.0
BASE_ROUNDTRIP_COST_BPS = 12.0
STRESS_ROUNDTRIP_COST_BPS = 16.0
MIN_TRAIN_DAYS = 14
VALIDATION_DAYS = 7
OOS_DAYS = 7
MIN_SPLIT_TRADES = 10
MAX_TRADES_PER_DAY = 4.0
MAX_DRAWDOWN_BPS = 2_000.0
MAX_SYMBOL_TRADE_SHARE = 0.50
MIN_DIRECTION_TRADE_SHARE = 0.10

SignalFunction = Callable[[Mapping[str, Any]], tuple[str, float] | None]


def evaluate_rare_event_discovery(dataset_dir: str | Path) -> dict[str, Any]:
    history, source = _load_history(dataset_dir)
    frame = _build_event_frame(history)
    fold_specs = _anchored_fold_specs(frame)
    if not fold_specs:
        raise ValueError("not enough complete days for anchored rare-event study")

    candidates = []
    for candidate_id, description, signal_function in _candidate_contracts():
        outcomes = _build_candidate_outcomes(frame, signal_function)
        fold_reports = []
        aggregate_oos_trades: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(fold_specs, start=1):
            development = _trades_for_dates(
                outcomes,
                fold["train_dates_all"],
                fold["validation_start_ms"],
            )
            validation = _trades_for_dates(
                outcomes,
                fold["validation_dates_all"],
                fold["oos_start_ms"],
            )
            oos = _trades_for_dates(
                outcomes,
                fold["oos_dates_all"],
                fold["oos_end_ms"],
            )
            development_metrics = _metrics(development)
            validation_metrics = _metrics(validation)
            oos_metrics = _metrics(oos)
            aggregate_oos_trades.extend(oos)
            fold_reports.append(
                {
                    "fold": fold_index,
                    "train_dates": fold["train_dates"],
                    "validation_dates": fold["validation_dates"],
                    "oos_dates": fold["oos_dates"],
                    "development_metrics": development_metrics,
                    "validation_metrics": validation_metrics,
                    "oos_metrics": oos_metrics,
                    "validation_gate_passed": _split_passed(validation_metrics),
                    "oos_gate_passed": _split_passed(oos_metrics),
                }
            )

        aggregate = _metrics(aggregate_oos_trades)
        failures = _candidate_failures(aggregate, fold_reports)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "description": description,
                "raw_signal_outcomes": len(outcomes),
                "folds": fold_reports,
                "aggregate_oos": aggregate,
                "failures": failures,
                "statistical_pass": not failures,
            }
        )

    candidates.sort(
        key=lambda row: (
            row["aggregate_oos"]["lcb_95_stress_net_bps"] is not None,
            row["aggregate_oos"]["lcb_95_stress_net_bps"] or -math.inf,
            row["aggregate_oos"]["mean_stress_net_bps"] or -math.inf,
        ),
        reverse=True,
    )
    passed = [row["candidate_id"] for row in candidates if row["statistical_pass"]]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "fixed_contract": {
            "candidate_count": len(_candidate_contracts()),
            "candidate_ids": [row[0] for row in _candidate_contracts()],
            "decision_cadence_minutes": DECISION_CADENCE_MINUTES,
            "horizon_minutes": HORIZON_MINUTES,
            "take_profit_gross_bps": TAKE_PROFIT_GROSS_BPS,
            "stop_loss_gross_bps": STOP_LOSS_GROSS_BPS,
            "same_bar_tp_sl_resolution": "stop_loss",
            "base_roundtrip_cost_bps": BASE_ROUNDTRIP_COST_BPS,
            "stress_roundtrip_cost_bps": STRESS_ROUNDTRIP_COST_BPS,
            "execution_assumption": "taker_entry_and_taker_exit",
            "minimum_split_trades": MIN_SPLIT_TRADES,
            "maximum_trades_per_day": MAX_TRADES_PER_DAY,
            "anchored_days": {
                "minimum_train": MIN_TRAIN_DAYS,
                "validation": VALIDATION_DAYS,
                "oos": OOS_DAYS,
            },
            "selection": "one_global_nonoverlapping_position_per_candidate",
            "oos_tuning": False,
        },
        "dataset": {
            "history_rows": len(history),
            "decision_rows": int(frame["decision_eligible"].sum()),
            "first_timestamp_ms": int(frame["timestamp"].min()),
            "last_timestamp_ms": int(frame["timestamp"].max()),
            "symbols": dict(sorted(Counter(frame["symbol"]).items())),
        },
        "candidates": candidates,
        "passed_candidates": passed,
        "verdict": (
            "eligible_for_prospective_microstructure_validation"
            if passed
            else "rejected"
        ),
        "research_only": True,
        "runtime_profile_created": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def render_rare_event_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bitget rare-event discovery v1",
        "",
        "## Verdict",
        "",
        f"**{str(report['verdict']).upper()}**",
        "",
        (
            f"Fixed TP/SL: +{TAKE_PROFIT_GROSS_BPS:.0f}/"
            f"-{STOP_LOSS_GROSS_BPS:.0f} gross bps; costs: "
            f"{BASE_ROUNDTRIP_COST_BPS:.0f} base and "
            f"{STRESS_ROUNDTRIP_COST_BPS:.0f} stress bps."
        ),
        "",
        "| Candidate | Raw events | OOS trades | Gross | Base net | "
        "Stress net | Stress LCB | DD | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        metrics = candidate["aggregate_oos"]
        lines.append(
            f"| {candidate['candidate_id']} | {candidate['raw_signal_outcomes']} | "
            f"{metrics['closed_trades']} | {_fmt(metrics['mean_gross_bps'])} | "
            f"{_fmt(metrics['mean_net_bps'])} | "
            f"{_fmt(metrics['mean_stress_net_bps'])} | "
            f"{_fmt(metrics['lcb_95_stress_net_bps'])} | "
            f"{metrics['max_drawdown_bps']:.1f} | "
            f"{str(candidate['statistical_pass']).lower()} |"
        )
    lines.extend(["", "## Fold results", ""])
    for candidate in report["candidates"]:
        lines.append(f"### {candidate['candidate_id']}")
        lines.append("")
        lines.append("| Fold | Validation trades/net/LCB | OOS trades/net/LCB |")
        lines.append("|---:|---:|---:|")
        for fold in candidate["folds"]:
            validation = fold["validation_metrics"]
            oos = fold["oos_metrics"]
            lines.append(
                f"| {fold['fold']} | {validation['closed_trades']} / "
                f"{_fmt(validation['mean_stress_net_bps'])} / "
                f"{_fmt(validation['lcb_95_stress_net_bps'])} | "
                f"{oos['closed_trades']} / "
                f"{_fmt(oos['mean_stress_net_bps'])} / "
                f"{_fmt(oos['lcb_95_stress_net_bps'])} |"
            )
        lines.append("")
        lines.append(
            "Failures: " + (", ".join(candidate["failures"]) or "none")
        )
        lines.append("")
    lines.extend(
        [
            "Signals use closed entry bars and trailing values only. OOS does not "
            "change thresholds or candidate definitions.",
            "",
            "No runtime policy, paper order or live order is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_event_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.sort_values(["symbol", "timestamp"]).reset_index(drop=True).copy()
    grouped = frame.groupby("symbol", sort=False)
    frame["bar_index"] = grouped.cumcount()
    for minutes in (1, 5, 15, 60):
        frame[f"return_{minutes}m_bps"] = (
            frame["close"] / grouped["close"].shift(minutes) - 1.0
        ) * 10_000.0
    returns_1m = frame["return_1m_bps"]
    for minutes in (5, 15, 60):
        frame[f"realized_vol_{minutes}m_bps"] = (
            returns_1m.groupby(frame["symbol"], sort=False)
            .rolling(minutes, min_periods=minutes)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
        )
    prior_volume = grouped["volume"].shift(1)
    frame["volume_ratio_60m"] = frame["volume"] / (
        prior_volume.groupby(frame["symbol"], sort=False)
        .rolling(60, min_periods=60)
        .mean()
        .reset_index(level=0, drop=True)
    )
    for minutes in (60, 120):
        prior_high = grouped["high"].shift(1)
        prior_low = grouped["low"].shift(1)
        frame[f"prior_high_{minutes}m"] = (
            prior_high.groupby(frame["symbol"], sort=False)
            .rolling(minutes, min_periods=minutes)
            .max()
            .reset_index(level=0, drop=True)
        )
        frame[f"prior_low_{minutes}m"] = (
            prior_low.groupby(frame["symbol"], sort=False)
            .rolling(minutes, min_periods=minutes)
            .min()
            .reset_index(level=0, drop=True)
        )
    frame["breakout_up_60_bps"] = (
        frame["close"] / frame["prior_high_60m"] - 1.0
    ) * 10_000.0
    frame["breakout_down_60_bps"] = (
        frame["prior_low_60m"] / frame["close"] - 1.0
    ) * 10_000.0
    frame["breakout_up_120_bps"] = (
        frame["close"] / frame["prior_high_120m"] - 1.0
    ) * 10_000.0
    frame["breakout_down_120_bps"] = (
        frame["prior_low_120m"] / frame["close"] - 1.0
    ) * 10_000.0
    frame["sweep_low_60_bps"] = (
        frame["prior_low_60m"] / frame["low"] - 1.0
    ) * 10_000.0
    frame["sweep_high_60_bps"] = (
        frame["high"] / frame["prior_high_60m"] - 1.0
    ) * 10_000.0
    frame["range_bps"] = (frame["high"] / frame["low"] - 1.0) * 10_000.0
    frame["body_bps"] = (frame["close"] / frame["open"] - 1.0) * 10_000.0
    frame["close_location"] = np.where(
        frame["high"] > frame["low"],
        (frame["close"] - frame["low"]) / (frame["high"] - frame["low"]),
        0.5,
    )
    lagged_rv_15 = grouped["realized_vol_15m_bps"].shift(1)
    lagged_rv_60 = grouped["realized_vol_60m_bps"].shift(1)
    frame["compression_ratio"] = lagged_rv_15 / lagged_rv_60
    breadth = (
        (frame["return_15m_bps"] > 0.0)
        .groupby(frame["timestamp"])
        .mean()
        .rename("market_breadth_15m")
    )
    frame = frame.join(breadth, on="timestamp")
    btc = (
        frame.loc[frame["symbol"] == "BTC/USDT", ["timestamp", "return_15m_bps"]]
        .set_index("timestamp")["return_15m_bps"]
        .rename("btc_return_15m_bps")
    )
    frame = frame.join(btc, on="timestamp")
    utc = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["date"] = utc.dt.strftime("%Y-%m-%d")
    required = [
        "return_5m_bps",
        "return_15m_bps",
        "return_60m_bps",
        "realized_vol_5m_bps",
        "realized_vol_15m_bps",
        "realized_vol_60m_bps",
        "volume_ratio_60m",
        "prior_high_60m",
        "prior_low_60m",
        "prior_high_120m",
        "prior_low_120m",
        "compression_ratio",
        "market_breadth_15m",
        "btc_return_15m_bps",
    ]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame["decision_eligible"] = (
        frame["timestamp"] % (DECISION_CADENCE_MINUTES * 60_000) == 0
    ) & frame[required].notna().all(axis=1)
    return frame.reset_index(drop=True)


def _candidate_contracts() -> tuple[tuple[str, str, SignalFunction], ...]:
    return (
        (
            "range_breakout_volume_v1",
            "120m close breakout with volume and market breadth quorum",
            _range_breakout_volume,
        ),
        (
            "compression_breakout_v1",
            "lagged volatility compression followed by a 60m close breakout",
            _compression_breakout,
        ),
        (
            "liquidity_sweep_reversal_v1",
            "60m high/low sweep that closes back inside with a rejection wick",
            _liquidity_sweep_reversal,
        ),
        (
            "cross_market_shock_continuation_v1",
            "15m directional shock confirmed by breadth, BTC and volume",
            _cross_market_shock_continuation,
        ),
        (
            "volatility_expansion_trend_v1",
            "short-horizon volatility expansion aligned with the 60m trend",
            _volatility_expansion_trend,
        ),
    )


def _range_breakout_volume(row: Mapping[str, Any]) -> tuple[str, float] | None:
    if (
        float(row["breakout_up_120_bps"]) > 0.0
        and float(row["return_5m_bps"]) >= 15.0
        and float(row["volume_ratio_60m"]) >= 1.5
        and float(row["market_breadth_15m"]) >= 0.625
        and float(row["close_location"]) >= 0.65
    ):
        return "LONG", float(row["return_5m_bps"]) + 10.0 * float(row["volume_ratio_60m"])
    if (
        float(row["breakout_down_120_bps"]) > 0.0
        and float(row["return_5m_bps"]) <= -15.0
        and float(row["volume_ratio_60m"]) >= 1.5
        and float(row["market_breadth_15m"]) <= 0.375
        and float(row["close_location"]) <= 0.35
    ):
        return "SHORT", -float(row["return_5m_bps"]) + 10.0 * float(row["volume_ratio_60m"])
    return None


def _compression_breakout(row: Mapping[str, Any]) -> tuple[str, float] | None:
    common = (
        float(row["compression_ratio"]) <= 0.55
        and float(row["volume_ratio_60m"]) >= 1.3
        and float(row["range_bps"]) >= 8.0
    )
    if common and float(row["breakout_up_60_bps"]) > 0.0 and float(row["return_5m_bps"]) >= 10.0:
        return "LONG", float(row["return_5m_bps"]) / max(float(row["compression_ratio"]), 0.05)
    if common and float(row["breakout_down_60_bps"]) > 0.0 and float(row["return_5m_bps"]) <= -10.0:
        return "SHORT", -float(row["return_5m_bps"]) / max(float(row["compression_ratio"]), 0.05)
    return None


def _liquidity_sweep_reversal(row: Mapping[str, Any]) -> tuple[str, float] | None:
    if (
        float(row["sweep_low_60_bps"]) >= 8.0
        and float(row["close"]) > float(row["prior_low_60m"])
        and float(row["close_location"]) >= 0.70
        and float(row["body_bps"]) > 0.0
        and float(row["volume_ratio_60m"]) >= 1.5
    ):
        return "LONG", float(row["sweep_low_60_bps"]) + 10.0 * float(row["volume_ratio_60m"])
    if (
        float(row["sweep_high_60_bps"]) >= 8.0
        and float(row["close"]) < float(row["prior_high_60m"])
        and float(row["close_location"]) <= 0.30
        and float(row["body_bps"]) < 0.0
        and float(row["volume_ratio_60m"]) >= 1.5
    ):
        return "SHORT", float(row["sweep_high_60_bps"]) + 10.0 * float(row["volume_ratio_60m"])
    return None


def _cross_market_shock_continuation(row: Mapping[str, Any]) -> tuple[str, float] | None:
    if (
        float(row["return_15m_bps"]) >= 35.0
        and float(row["market_breadth_15m"]) >= 0.75
        and float(row["btc_return_15m_bps"]) >= 20.0
        and float(row["volume_ratio_60m"]) >= 1.2
    ):
        return "LONG", float(row["return_15m_bps"]) + float(row["btc_return_15m_bps"])
    if (
        float(row["return_15m_bps"]) <= -35.0
        and float(row["market_breadth_15m"]) <= 0.25
        and float(row["btc_return_15m_bps"]) <= -20.0
        and float(row["volume_ratio_60m"]) >= 1.2
    ):
        return "SHORT", -float(row["return_15m_bps"]) - float(row["btc_return_15m_bps"])
    return None


def _volatility_expansion_trend(row: Mapping[str, Any]) -> tuple[str, float] | None:
    expansion = float(row["realized_vol_5m_bps"]) / max(float(row["realized_vol_60m_bps"]), 0.01)
    common = expansion >= 1.8 and float(row["volume_ratio_60m"]) >= 1.2
    if common and float(row["return_15m_bps"]) >= 30.0 and float(row["return_60m_bps"]) > 0.0:
        return "LONG", float(row["return_15m_bps"]) * expansion
    if common and float(row["return_15m_bps"]) <= -30.0 and float(row["return_60m_bps"]) < 0.0:
        return "SHORT", -float(row["return_15m_bps"]) * expansion
    return None


def _build_candidate_outcomes(
    frame: pd.DataFrame,
    signal_function: SignalFunction,
) -> list[dict[str, Any]]:
    bars = {
        symbol: rows.sort_values("bar_index").reset_index(drop=True)
        for symbol, rows in frame.groupby("symbol", sort=False)
    }
    outcomes = []
    decisions = frame.loc[frame["decision_eligible"]]
    for row in decisions.to_dict("records"):
        signal = signal_function(row)
        if signal is None:
            continue
        direction, score = signal
        outcome = _barrier_outcome(
            bars[str(row["symbol"])],
            entry_index=int(row["bar_index"]),
            direction=direction,
        )
        if outcome is not None:
            outcomes.append(
                {
                    **outcome,
                    "symbol": str(row["symbol"]),
                    "direction": direction,
                    "score": float(score),
                    "date": str(row["date"]),
                }
            )
    return outcomes


def _barrier_outcome(
    bars: pd.DataFrame,
    *,
    entry_index: int,
    direction: str,
) -> dict[str, Any] | None:
    exit_index = entry_index + HORIZON_MINUTES
    if exit_index >= len(bars):
        return None
    entry_timestamp = int(bars.iloc[entry_index]["timestamp"])
    if int(bars.iloc[exit_index]["timestamp"]) - entry_timestamp != HORIZON_MINUTES * 60_000:
        return None
    entry_price = float(bars.iloc[entry_index]["close"])
    take_price = entry_price * (
        1.0 + TAKE_PROFIT_GROSS_BPS / 10_000.0
        if direction == "LONG"
        else 1.0 - TAKE_PROFIT_GROSS_BPS / 10_000.0
    )
    stop_price = entry_price * (
        1.0 - STOP_LOSS_GROSS_BPS / 10_000.0
        if direction == "LONG"
        else 1.0 + STOP_LOSS_GROSS_BPS / 10_000.0
    )
    gross = None
    reason = "horizon"
    actual_exit = exit_index
    for index in range(entry_index + 1, exit_index + 1):
        high = float(bars.iloc[index]["high"])
        low = float(bars.iloc[index]["low"])
        hit_take = high >= take_price if direction == "LONG" else low <= take_price
        hit_stop = low <= stop_price if direction == "LONG" else high >= stop_price
        if hit_take and hit_stop:
            gross = -STOP_LOSS_GROSS_BPS
            reason = "ambiguous_stop"
            actual_exit = index
            break
        if hit_stop:
            gross = -STOP_LOSS_GROSS_BPS
            reason = "stop_loss"
            actual_exit = index
            break
        if hit_take:
            gross = TAKE_PROFIT_GROSS_BPS
            reason = "take_profit"
            actual_exit = index
            break
    if gross is None:
        exit_close = float(bars.iloc[exit_index]["close"])
        raw = (exit_close / entry_price - 1.0) * 10_000.0
        gross = raw if direction == "LONG" else -raw
    return {
        "timestamp": entry_timestamp,
        "exit_timestamp_ms": int(bars.iloc[actual_exit]["timestamp"]),
        "gross_bps": float(gross),
        "net_bps": float(gross - BASE_ROUNDTRIP_COST_BPS),
        "stress_net_bps": float(gross - STRESS_ROUNDTRIP_COST_BPS),
        "exit_reason": reason,
    }


def _anchored_fold_specs(frame: pd.DataFrame) -> list[dict[str, Any]]:
    dates = sorted(
        str(value)
        for value in frame.loc[frame["decision_eligible"], "date"].unique()
    )
    folds = []
    train_days = MIN_TRAIN_DAYS
    while train_days + VALIDATION_DAYS + OOS_DAYS <= len(dates):
        train = dates[:train_days]
        validation = dates[train_days : train_days + VALIDATION_DAYS]
        oos = dates[train_days + VALIDATION_DAYS : train_days + VALIDATION_DAYS + OOS_DAYS]
        folds.append(
            {
                "train_dates_all": set(train),
                "validation_dates_all": set(validation),
                "oos_dates_all": set(oos),
                "train_dates": [train[0], train[-1]],
                "validation_dates": [validation[0], validation[-1]],
                "oos_dates": [oos[0], oos[-1]],
                "validation_start_ms": _date_start_ms(validation[0]),
                "oos_start_ms": _date_start_ms(oos[0]),
                "oos_end_ms": _date_start_ms(_next_date(oos[-1])),
            }
        )
        train_days += OOS_DAYS
    return folds


def _trades_for_dates(
    outcomes: Sequence[Mapping[str, Any]],
    dates: set[str],
    end_exclusive_ms: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in outcomes
        if str(row["date"]) in dates
        and int(row["exit_timestamp_ms"]) < end_exclusive_ms
    ]
    return _one_global_position(eligible)


def _one_global_position(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_timestamp: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_timestamp.setdefault(int(row["timestamp"]), []).append(row)
    selected = []
    next_allowed = -1
    for timestamp in sorted(by_timestamp):
        if timestamp < next_allowed:
            continue
        best = max(
            by_timestamp[timestamp],
            key=lambda row: (float(row["score"]), str(row["symbol"])),
        )
        selected.append(dict(best))
        next_allowed = int(best["exit_timestamp_ms"])
    return selected


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gross = [float(row["gross_bps"]) for row in rows]
    net = [float(row["net_bps"]) for row in rows]
    stress = [float(row["stress_net_bps"]) for row in rows]
    symbols = dict(sorted(Counter(str(row["symbol"]) for row in rows).items()))
    directions = dict(sorted(Counter(str(row["direction"]) for row in rows).items()))
    mean_net = statistics.fmean(net) if net else None
    mean_stress = statistics.fmean(stress) if stress else None
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in net:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "fills": len(rows) * 2,
        "closed_trades": len(rows),
        "mean_gross_bps": statistics.fmean(gross) if gross else None,
        "mean_net_bps": mean_net,
        "lcb_95_net_bps": _lcb(net),
        "mean_stress_net_bps": mean_stress,
        "lcb_95_stress_net_bps": _lcb(stress),
        "net_sum_bps": sum(net),
        "max_drawdown_bps": drawdown,
        "win_rate": sum(value > 0.0 for value in net) / len(net) if net else 0.0,
        "trades_per_day": _trades_per_day(rows),
        "symbols": symbols,
        "directions": directions,
        "exit_reasons": dict(sorted(Counter(str(row["exit_reason"]) for row in rows).items())),
        "top_symbol_trade_share": max(symbols.values()) / len(rows) if rows else 0.0,
        "minimum_direction_trade_share": (
            min(directions.get("LONG", 0), directions.get("SHORT", 0)) / len(rows)
            if rows
            else 0.0
        ),
    }


def _lcb(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.fmean(values) - LCB_Z * statistics.stdev(values) / math.sqrt(len(values))


def _split_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics["closed_trades"]) >= MIN_SPLIT_TRADES
        and metrics["mean_net_bps"] is not None
        and float(metrics["mean_net_bps"]) > 0.0
        and metrics["lcb_95_net_bps"] is not None
        and float(metrics["lcb_95_net_bps"]) > 0.0
        and metrics["mean_stress_net_bps"] is not None
        and float(metrics["mean_stress_net_bps"]) > 0.0
        and metrics["lcb_95_stress_net_bps"] is not None
        and float(metrics["lcb_95_stress_net_bps"]) > 0.0
        and float(metrics["max_drawdown_bps"]) <= MAX_DRAWDOWN_BPS
        and float(metrics["top_symbol_trade_share"]) <= MAX_SYMBOL_TRADE_SHARE
        and float(metrics["minimum_direction_trade_share"]) >= MIN_DIRECTION_TRADE_SHARE
        and float(metrics["trades_per_day"]) <= MAX_TRADES_PER_DAY
    )


def _candidate_failures(
    aggregate: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
) -> list[str]:
    failures = []
    if int(aggregate["fills"]) < 20:
        failures.append("oos_fills_below_20")
    if int(aggregate["closed_trades"]) < 10:
        failures.append("oos_closed_trades_below_10")
    if aggregate["mean_net_bps"] is None or float(aggregate["mean_net_bps"]) <= 0.0:
        failures.append("oos_nonpositive_costed_expectancy")
    if aggregate["lcb_95_net_bps"] is None or float(aggregate["lcb_95_net_bps"]) <= 0.0:
        failures.append("oos_nonpositive_lcb")
    if aggregate["mean_stress_net_bps"] is None or float(aggregate["mean_stress_net_bps"]) <= 0.0:
        failures.append("oos_nonpositive_cost_stress_expectancy")
    if aggregate["lcb_95_stress_net_bps"] is None or float(aggregate["lcb_95_stress_net_bps"]) <= 0.0:
        failures.append("oos_nonpositive_cost_stress_lcb")
    if float(aggregate["max_drawdown_bps"]) > MAX_DRAWDOWN_BPS:
        failures.append("oos_drawdown_above_20pct")
    if float(aggregate["top_symbol_trade_share"]) > MAX_SYMBOL_TRADE_SHARE:
        failures.append("oos_symbol_concentration_above_50pct")
    if aggregate["closed_trades"] and float(aggregate["minimum_direction_trade_share"]) < MIN_DIRECTION_TRADE_SHARE:
        failures.append("oos_direction_collapse")
    if float(aggregate["trades_per_day"]) > MAX_TRADES_PER_DAY:
        failures.append("oos_turnover_above_4_trades_per_day")
    for fold in folds:
        if not bool(fold["validation_gate_passed"]):
            failures.append(f"validation_gate_failed:{fold['fold']}")
        if not bool(fold["oos_gate_passed"]):
            failures.append(f"oos_gate_failed:{fold['fold']}")
    return failures


def _trades_per_day(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    timestamps = [int(row["timestamp"]) for row in rows]
    span_days = max(1.0, (max(timestamps) - min(timestamps)) / 86_400_000.0 + 1.0)
    return len(rows) / span_days


def _date_start_ms(value: str) -> int:
    return int(pd.Timestamp(value, tz="UTC").timestamp() * 1000)


def _next_date(value: str) -> str:
    return str((pd.Timestamp(value) + pd.Timedelta(days=1)).date())


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
