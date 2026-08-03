"""Post-only maker-entry and taker-exit study using captured Bitget trades."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .event_study import (
    HORIZONS_MINUTES,
    LCB_Z,
    MIN_MOVE_BUDGET_BPS,
    STOP_LOSS_ADVERSE_MOVE_BPS,
    TAKE_PROFIT_NET_BPS,
    _candidate_contracts,
)
from .research_dataset import _execution_price, validate_research_dataset
from .segment_store import SegmentValidationError, validate_data_session


MAKER_EVENT_STUDY_SCHEMA_VERSION = "panteon.bitget_maker_event_study.v1"
ENTRY_NOTIONAL_USD = 25.0
MAKER_ORDER_TTL_MS = 60_000
MIN_REQUIRED_HOURS = 72.0


def evaluate_maker_event_study(dataset_dir: str | Path) -> dict[str, Any]:
    """Evaluate one fixed maker-entry scenario without promotion authority."""
    validation = validate_research_dataset(dataset_dir)
    source = Path(dataset_dir).resolve()
    research_manifest = _read_json(
        source / "microstructure_research_v1.manifest.json"
    )
    frame_dir = Path(str(research_manifest["source_frame_dataset_dir"]))
    frame_manifest = _read_json(frame_dir / "frame_1s_v1.manifest.json")
    session_validation = validate_data_session(
        frame_manifest["source_session_dir"]
    )
    database = source / str(research_manifest["database_file"])
    bars_by_symbol = _load_bars(database)
    maker_fee_bps = _load_maker_fee_bps(
        segment_rows=session_validation["segment_rows"],
        symbols=tuple(bars_by_symbol),
    )
    fill_index, raw_trade_counts = _build_fill_index(
        segment_rows=session_validation["segment_rows"],
        bars_by_symbol=bars_by_symbol,
    )

    attempts_by_horizon: dict[int, list[dict[str, Any]]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    for symbol, bars in bars_by_symbol.items():
        symbol_attempts = _build_symbol_attempts(
            symbol=symbol,
            bars=bars,
            fill_index=fill_index,
            maker_fee_bps=maker_fee_bps[symbol],
        )
        for horizon, rows in symbol_attempts.items():
            attempts_by_horizon[horizon].extend(rows)

    timestamps = sorted(
        {
            int(row["entry_bar_timestamp_ms"])
            for rows in attempts_by_horizon.values()
            for row in rows
        }
    )
    if not timestamps:
        raise ValueError("maker event-study attempts are empty")
    split_points = _split_points(timestamps)
    duration_hours = (
        (timestamps[-1] - timestamps[0]) / 3_600_000.0
        if len(timestamps) >= 2
        else 0.0
    )

    candidates = []
    for horizon, attempts in attempts_by_horizon.items():
        for candidate_id, signal in _candidate_contracts():
            selected_attempts, trades = _select_attempts(attempts, signal)
            split_metrics = {
                split: _metrics(
                    attempts=[
                        row
                        for row in selected_attempts
                        if _split_for_attempt(row, split_points) == split
                    ],
                    trades=[
                        row
                        for row in trades
                        if _split_for_trade(row, split_points) == split
                    ],
                )
                for split in ("development", "validation", "oos")
            }
            aggregate = _metrics(
                attempts=selected_attempts,
                trades=trades,
            )
            failures = _statistical_failures(split_metrics)
            candidates.append(
                {
                    "candidate_id": f"{candidate_id}@{horizon}m",
                    "signal_contract_id": candidate_id,
                    "horizon_minutes": horizon,
                    "aggregate": aggregate,
                    "splits": split_metrics,
                    "statistical_pass": not failures,
                    "failures": failures,
                    "symbols": dict(
                        sorted(
                            Counter(str(row["symbol"]) for row in trades).items()
                        )
                    ),
                    "directions": dict(
                        sorted(
                            Counter(
                                str(row["direction"]) for row in trades
                            ).items()
                        )
                    ),
                }
            )
    candidates.sort(
        key=lambda row: (
            row["splits"]["oos"]["mean_net_bps"] is not None,
            row["splits"]["oos"]["mean_net_bps"] or -math.inf,
            row["aggregate"]["mean_net_bps"] or -math.inf,
        ),
        reverse=True,
    )
    campaign_failures = ["maker_queue_position_unverified"]
    if duration_hours < MIN_REQUIRED_HOURS:
        campaign_failures.append("evidence_duration_below_72h")
    if not any(row["statistical_pass"] for row in candidates):
        campaign_failures.append("no_candidate_passed_validation_and_oos")
    return {
        "schema_version": MAKER_EVENT_STUDY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_dir": str(source),
            "research_manifest_sha256": validation["manifest_sha256"],
            "research_database_sha256": validation["database_sha256"],
            "source_session_id": research_manifest["source_session_id"],
            "source_head_manifest_sha256": session_validation[
                "head_manifest_sha256"
            ],
            "raw_trade_counts_by_symbol": raw_trade_counts,
        },
        "fixed_contract": {
            "entry_order_type": "post_only_limit",
            "entry_price": "captured_signal_minute_best_bid_or_ask",
            "entry_notional_usd": ENTRY_NOTIONAL_USD,
            "entry_ttl_ms": MAKER_ORDER_TTL_MS,
            "fill_rule": (
                "aggressive_opposite_side_trades_at_limit_or_better;"
                "cumulative_notional_at_least_order_notional"
            ),
            "queue_model": "front_of_queue_assumption_unverified",
            "maker_fee_bps_by_symbol": maker_fee_bps,
            "exit_order_type": "taker",
            "taker_fee_bps_by_symbol": research_manifest[
                "taker_fee_bps_by_symbol"
            ],
            "horizons_minutes": list(HORIZONS_MINUTES),
            "take_profit_net_bps": TAKE_PROFIT_NET_BPS,
            "stop_loss_adverse_move_bps": STOP_LOSS_ADVERSE_MOVE_BPS,
            "min_move_budget_bps": MIN_MOVE_BUDGET_BPS,
            "chronological_splits_pct": [60, 20, 20],
        },
        "evidence": {
            "duration_hours": duration_hours,
            "single_source_session": True,
        },
        "attempts": {
            str(horizon): _attempt_summary(rows)
            for horizon, rows in attempts_by_horizon.items()
        },
        "candidates": candidates,
        "campaign_failures": campaign_failures,
        "verdict": "research_only_not_promotion_evidence",
        "research_only": True,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def render_maker_event_study_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bitget maker-entry event study",
        "",
        "## Verdict",
        "",
        "**RESEARCH ONLY - NOT PROMOTION EVIDENCE**",
        "",
        (
            f"Evidence: {report['evidence']['duration_hours']:.2f} hours. "
            f"Entry: $25 post-only, TTL 60 seconds; exit: taker."
        ),
        "",
        "| Candidate | Attempts | Filled trades | Fill rate | Mean net, bps | "
        "OOS trades | OOS net, bps | OOS LCB | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        aggregate = candidate["aggregate"]
        oos = candidate["splits"]["oos"]
        lines.append(
            f"| {candidate['candidate_id']} | {aggregate['order_attempts']} | "
            f"{aggregate['closed_trades']} | "
            f"{aggregate['maker_fill_rate']:.3f} | "
            f"{_fmt(aggregate['mean_net_bps'])} | "
            f"{oos['closed_trades']} | {_fmt(oos['mean_net_bps'])} | "
            f"{_fmt(oos['lcb_95_net_bps'])} | "
            f"{str(candidate['statistical_pass']).lower()} |"
        )
    lines.extend(
        [
            "",
            "Campaign failures: "
            + ", ".join(report["campaign_failures"]),
            "",
            (
                "A trade-price touch and sufficient aggressive notional prove "
                "marketable volume, but do not prove queue position. Therefore "
                "this report cannot promote a policy."
            ),
            "",
            "No exchange order, paper order or runtime policy is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_bars(database: Path) -> dict[str, list[dict[str, Any]]]:
    query = """
        SELECT bar_timestamp_ms, symbol, continuity_id, eligible,
               bid_close, ask_close, mid_close, execution_curve_json,
               taker_fee_bps, return_5m_bps, realized_vol_5m_bps,
               flow_imbalance, book_imbalance_mean,
               microprice_edge_mean_bps, oi_change_5m_bps,
               basis_close_bps, funding_close
        FROM bars
        ORDER BY symbol, bar_timestamp_ms
    """
    result: dict[str, list[dict[str, Any]]] = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query):
            result.setdefault(str(row["symbol"]), []).append(dict(row))
    return result


def _build_fill_index(
    *,
    segment_rows: Sequence[Mapping[str, Any]],
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[tuple[str, int, str], dict[str, Any]], dict[str, int]]:
    entry_lookup = {
        symbol: {
            int(bar["bar_timestamp_ms"]): bar
            for bar in bars
            if _entry_eligible(bar)
        }
        for symbol, bars in bars_by_symbol.items()
    }
    states: dict[tuple[str, int, str], dict[str, Any]] = {}
    trade_counts: Counter[str] = Counter()
    for segment in segment_rows:
        database = Path(str(segment["database"]))
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for symbol, received_ns, payload_json in connection.execute(
                """
                SELECT symbol, received_timestamp_utc_ns, payload_json
                FROM events
                WHERE channel = 'trade'
                ORDER BY received_timestamp_utc_ns, id
                """
            ):
                payload = json.loads(str(payload_json))
                price = _positive_float(payload.get("price"))
                size = _positive_float(payload.get("size"))
                side = str(payload.get("side") or "").lower()
                if price <= 0.0 or size <= 0.0 or side not in {"buy", "sell"}:
                    continue
                clean_symbol = str(symbol)
                trade_counts[clean_symbol] += 1
                received_ms = int(received_ns) // 1_000_000
                signal_minute = received_ms - received_ms % 60_000
                entry = entry_lookup.get(clean_symbol, {}).get(signal_minute)
                if entry is None or received_ms <= signal_minute:
                    continue
                direction = "LONG" if side == "sell" else "SHORT"
                limit_price = float(
                    entry["bid_close"]
                    if direction == "LONG"
                    else entry["ask_close"]
                )
                price_matches = (
                    price <= limit_price
                    if direction == "LONG"
                    else price >= limit_price
                )
                if not price_matches:
                    continue
                key = (clean_symbol, signal_minute, direction)
                state = states.setdefault(
                    key,
                    {
                        "qualifying_notional_usd": 0.0,
                        "fill_timestamp_ms": None,
                    },
                )
                if state["fill_timestamp_ms"] is not None:
                    continue
                state["qualifying_notional_usd"] += price * size
                if state["qualifying_notional_usd"] >= ENTRY_NOTIONAL_USD:
                    state["fill_timestamp_ms"] = received_ms
    fills = {
        key: dict(value)
        for key, value in states.items()
        if value["fill_timestamp_ms"] is not None
    }
    return fills, dict(sorted(trade_counts.items()))


def _load_maker_fee_bps(
    *,
    segment_rows: Sequence[Mapping[str, Any]],
    symbols: Sequence[str],
) -> dict[str, float]:
    latest: dict[str, tuple[int, float]] = {}
    for segment in segment_rows:
        database = Path(str(segment["database"]))
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for symbol, received_ns, payload_json in connection.execute(
                """
                SELECT symbol, received_timestamp_utc_ns, payload_json
                FROM events
                WHERE channel = 'rest_instrument_rules'
                ORDER BY received_timestamp_utc_ns
                """
            ):
                payload = json.loads(str(payload_json))
                rate = _finite_float(payload.get("maker_fee_rate"))
                if rate is not None:
                    latest[str(symbol)] = (
                        int(received_ns),
                        rate * 10_000.0,
                    )
    missing = sorted(set(symbols) - set(latest))
    if missing:
        raise SegmentValidationError(
            "maker fee rules missing for: " + ", ".join(missing)
        )
    return {symbol: latest[symbol][1] for symbol in symbols}


def _build_symbol_attempts(
    *,
    symbol: str,
    bars: Sequence[Mapping[str, Any]],
    fill_index: Mapping[tuple[str, int, str], Mapping[str, Any]],
    maker_fee_bps: float,
) -> dict[int, list[dict[str, Any]]]:
    result = {horizon: [] for horizon in HORIZONS_MINUTES}
    for index, entry in enumerate(bars):
        if not _entry_eligible(entry):
            continue
        for direction in ("LONG", "SHORT"):
            entry_timestamp = int(entry["bar_timestamp_ms"])
            limit_price = float(
                entry["bid_close"] if direction == "LONG" else entry["ask_close"]
            )
            fill = fill_index.get((symbol, entry_timestamp, direction))
            features = _entry_features(entry)
            for horizon in HORIZONS_MINUTES:
                outcome = None
                if fill is not None:
                    outcome = _maker_outcome(
                        bars=bars,
                        entry_index=index,
                        direction=direction,
                        horizon_minutes=horizon,
                        entry_price=limit_price,
                        maker_fee_bps=maker_fee_bps,
                        fill_timestamp_ms=int(fill["fill_timestamp_ms"]),
                    )
                result[horizon].append(
                    {
                        **features,
                        **(outcome or {}),
                        "symbol": symbol,
                        "direction": direction,
                        "horizon_minutes": horizon,
                        "entry_bar_timestamp_ms": int(
                            entry["bar_timestamp_ms"]
                        ),
                        "order_expiry_timestamp_ms": int(
                            entry["bar_timestamp_ms"]
                        )
                        + MAKER_ORDER_TTL_MS,
                        "limit_price": limit_price,
                        "maker_filled": fill is not None,
                        "maker_fill_timestamp_ms": (
                            int(fill["fill_timestamp_ms"])
                            if fill is not None
                            else None
                        ),
                        "qualifying_notional_usd": (
                            float(fill["qualifying_notional_usd"])
                            if fill is not None
                            else 0.0
                        ),
                        "outcome_valid": outcome is not None,
                    }
                )
    return result


def _maker_outcome(
    *,
    bars: Sequence[Mapping[str, Any]],
    entry_index: int,
    direction: str,
    horizon_minutes: int,
    entry_price: float,
    maker_fee_bps: float,
    fill_timestamp_ms: int,
) -> dict[str, Any] | None:
    exit_limit = entry_index + int(horizon_minutes)
    if exit_limit >= len(bars):
        return None
    entry = bars[entry_index]
    entry_time = int(entry["bar_timestamp_ms"])
    continuity = int(entry["continuity_id"])
    entry_mid = float(entry["mid_close"])
    immediate_exit = _execution_price(
        entry["execution_curve_json"],
        side="sell" if direction == "LONG" else "buy",
    )
    if immediate_exit is None:
        return None
    initial_net = _mixed_fee_net_bps(
        direction=direction,
        entry_price=entry_price,
        exit_price=float(immediate_exit),
        maker_fee_bps=maker_fee_bps,
        taker_fee_bps=float(entry["taker_fee_bps"]),
    )
    stop_loss_net_bps = initial_net - STOP_LOSS_ADVERSE_MOVE_BPS
    for future_index in range(entry_index + 1, exit_limit + 1):
        future = bars[future_index]
        expected_time = entry_time + (future_index - entry_index) * 60_000
        if (
            int(future["eligible"]) != 1
            or int(future["continuity_id"]) != continuity
            or int(future["bar_timestamp_ms"]) != expected_time
            or future["mid_close"] is None
        ):
            return None
        exit_price = _execution_price(
            future["execution_curve_json"],
            side="sell" if direction == "LONG" else "buy",
        )
        if exit_price is None:
            return None
        net_bps = _mixed_fee_net_bps(
            direction=direction,
            entry_price=entry_price,
            exit_price=float(exit_price),
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=float(entry["taker_fee_bps"]),
        )
        exit_reason = None
        if net_bps >= TAKE_PROFIT_NET_BPS:
            exit_reason = "take_profit"
        elif net_bps <= stop_loss_net_bps:
            exit_reason = "stop_loss"
        elif future_index == exit_limit:
            exit_reason = "max_holding"
        if exit_reason is not None:
            sign = 1.0 if direction == "LONG" else -1.0
            gross_mid_bps = (
                sign
                * (float(future["mid_close"]) / entry_mid - 1.0)
                * 10_000.0
            )
            return {
                "exit_bar_timestamp_ms": int(future["bar_timestamp_ms"]),
                "fill_timestamp_ms": fill_timestamp_ms,
                "holding_minutes": (
                    int(future["bar_timestamp_ms"]) - fill_timestamp_ms
                )
                / 60_000.0,
                "exit_reason": exit_reason,
                "gross_mid_bps": gross_mid_bps,
                "net_execution_bps": net_bps,
                "total_cost_bps": gross_mid_bps - net_bps,
                "initial_roundtrip_net_bps": initial_net,
                "stop_loss_net_bps": stop_loss_net_bps,
            }
    return None


def _mixed_fee_net_bps(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
) -> float:
    quantity = ENTRY_NOTIONAL_USD / entry_price
    exit_notional = quantity * exit_price
    gross_pnl = (
        exit_notional - ENTRY_NOTIONAL_USD
        if direction == "LONG"
        else ENTRY_NOTIONAL_USD - exit_notional
    )
    fees = (
        ENTRY_NOTIONAL_USD * maker_fee_bps / 10_000.0
        + exit_notional * taker_fee_bps / 10_000.0
    )
    return (gross_pnl - fees) / ENTRY_NOTIONAL_USD * 10_000.0


def _entry_eligible(entry: Mapping[str, Any]) -> bool:
    return (
        int(entry["eligible"]) == 1
        and entry["bid_close"] is not None
        and entry["ask_close"] is not None
        and entry["mid_close"] is not None
        and entry["return_5m_bps"] is not None
        and entry["realized_vol_5m_bps"] is not None
        and entry["book_imbalance_mean"] is not None
        and entry["microprice_edge_mean_bps"] is not None
    )


def _entry_features(entry: Mapping[str, Any]) -> dict[str, float]:
    return {
        "return_5m_bps": float(entry["return_5m_bps"]),
        "realized_vol_5m_bps": float(entry["realized_vol_5m_bps"]),
        "move_budget_bps": (
            abs(float(entry["return_5m_bps"]))
            + 2.0 * float(entry["realized_vol_5m_bps"])
        ),
        "flow_imbalance": float(entry["flow_imbalance"]),
        "book_imbalance_mean": float(entry["book_imbalance_mean"]),
        "microprice_edge_mean_bps": float(
            entry["microprice_edge_mean_bps"]
        ),
        "oi_change_5m_bps": float(entry["oi_change_5m_bps"] or 0.0),
        "basis_close_bps": float(entry["basis_close_bps"] or 0.0),
        "funding_close": float(entry["funding_close"] or 0.0),
    }


def _select_attempts(
    rows: Sequence[Mapping[str, Any]],
    signal,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_timestamp: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        score = float(signal(row))
        if score <= 0.0:
            continue
        candidate = dict(row)
        candidate["score"] = score
        by_timestamp.setdefault(
            int(row["entry_bar_timestamp_ms"]),
            [],
        ).append(candidate)
    attempts = []
    trades = []
    next_allowed = -1
    for timestamp in sorted(by_timestamp):
        if timestamp < next_allowed:
            continue
        selected = max(
            by_timestamp[timestamp],
            key=lambda row: (float(row["score"]), str(row["symbol"])),
        )
        attempts.append(selected)
        if selected["maker_filled"] and selected["outcome_valid"]:
            trades.append(selected)
            next_allowed = int(selected["exit_bar_timestamp_ms"])
        else:
            next_allowed = int(selected["order_expiry_timestamp_ms"])
    return attempts, trades


def _split_points(timestamps: Sequence[int]) -> tuple[int, int]:
    start = int(timestamps[0])
    width = int(timestamps[-1]) - start
    return start + int(width * 0.60), start + int(width * 0.80)


def _split_for_attempt(
    row: Mapping[str, Any],
    points: tuple[int, int],
) -> str | None:
    entry = int(row["entry_bar_timestamp_ms"])
    expiry = int(row["order_expiry_timestamp_ms"])
    if entry <= points[0] and expiry <= points[0]:
        return "development"
    if entry > points[0] and entry <= points[1] and expiry <= points[1]:
        return "validation"
    if entry > points[1]:
        return "oos"
    return None


def _split_for_trade(
    row: Mapping[str, Any],
    points: tuple[int, int],
) -> str | None:
    entry = int(row["entry_bar_timestamp_ms"])
    exit_time = int(row["exit_bar_timestamp_ms"])
    if entry <= points[0] and exit_time <= points[0]:
        return "development"
    if entry > points[0] and entry <= points[1] and exit_time <= points[1]:
        return "validation"
    if entry > points[1]:
        return "oos"
    return None


def _metrics(
    *,
    attempts: Sequence[Mapping[str, Any]],
    trades: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    net = [float(row["net_execution_bps"]) for row in trades]
    gross = [float(row["gross_mid_bps"]) for row in trades]
    costs = [float(row["total_cost_bps"]) for row in trades]
    mean = statistics.fmean(net) if net else None
    lcb = (
        mean - LCB_Z * statistics.stdev(net) / math.sqrt(len(net))
        if len(net) >= 2 and mean is not None
        else None
    )
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in net:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    filled_attempts = sum(bool(row["maker_filled"]) for row in attempts)
    return {
        "order_attempts": len(attempts),
        "maker_filled_attempts": filled_attempts,
        "maker_fill_rate": (
            filled_attempts / len(attempts) if attempts else 0.0
        ),
        "fills": len(net) * 2,
        "closed_trades": len(net),
        "mean_gross_mid_bps": statistics.fmean(gross) if gross else None,
        "mean_total_cost_bps": statistics.fmean(costs) if costs else None,
        "mean_net_bps": mean,
        "lcb_95_net_bps": lcb,
        "net_sum_bps": sum(net),
        "win_rate": sum(value > 0.0 for value in net) / len(net) if net else 0.0,
        "max_drawdown_bps": drawdown,
        "mean_holding_minutes": (
            statistics.fmean(float(row["holding_minutes"]) for row in trades)
            if trades
            else None
        ),
        "exit_reasons": dict(
            sorted(Counter(str(row["exit_reason"]) for row in trades).items())
        ),
    }


def _statistical_failures(
    splits: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    failures = []
    for split in ("validation", "oos"):
        metrics = splits[split]
        if int(metrics["fills"]) < 20:
            failures.append(f"{split}_fills_below_20")
        if int(metrics["closed_trades"]) < 10:
            failures.append(f"{split}_closed_trades_below_10")
        if (
            metrics["mean_net_bps"] is None
            or float(metrics["mean_net_bps"]) <= 0.0
        ):
            failures.append(f"{split}_nonpositive_costed_expectancy")
    oos_lcb = splits["oos"]["lcb_95_net_bps"]
    if oos_lcb is None or float(oos_lcb) <= 0.0:
        failures.append("oos_nonpositive_lcb")
    return failures


def _attempt_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    filled = sum(bool(row["maker_filled"]) for row in rows)
    valid = sum(bool(row["outcome_valid"]) for row in rows)
    return {
        "attempt_rows": len(rows),
        "maker_filled_rows": filled,
        "maker_fill_rate": filled / len(rows) if rows else 0.0,
        "valid_outcomes": valid,
    }


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
