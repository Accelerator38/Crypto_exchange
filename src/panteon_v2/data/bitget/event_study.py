"""Costed event-driven triple-barrier study for Bitget microstructure data."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .research_dataset import (
    _execution_price,
    _net_execution_bps,
    validate_research_dataset,
)


EVENT_STUDY_SCHEMA_VERSION = "panteon.bitget_event_study.v1"
HORIZONS_MINUTES = (15, 30, 60)
TAKE_PROFIT_NET_BPS = 10.0
STOP_LOSS_ADVERSE_MOVE_BPS = 20.0
MIN_MOVE_BUDGET_BPS = 20.0
MIN_REQUIRED_HOURS = 72.0
LCB_Z = 1.6448536269514722


def evaluate_event_study(dataset_dir: str | Path) -> dict[str, Any]:
    """Evaluate fixed event candidates without producing a runtime policy."""
    validation = validate_research_dataset(dataset_dir)
    source = Path(dataset_dir).resolve()
    manifest = _read_json(source / "microstructure_research_v1.manifest.json")
    database = source / str(manifest["database_file"])
    bars_by_symbol = _load_bars(database)
    outcomes_by_horizon = {
        horizon: _build_outcomes(
            bars_by_symbol,
            horizon_minutes=horizon,
        )
        for horizon in HORIZONS_MINUTES
    }
    timestamps = sorted(
        {
            int(row["entry_bar_timestamp_ms"])
            for rows in outcomes_by_horizon.values()
            for row in rows
        }
    )
    if not timestamps:
        raise ValueError("event-study outcomes are empty")
    split_points = _split_points(timestamps)
    duration_hours = (
        (timestamps[-1] - timestamps[0]) / 3_600_000.0
        if len(timestamps) >= 2
        else 0.0
    )

    candidates = []
    for horizon, outcomes in outcomes_by_horizon.items():
        for candidate_id, signal in _candidate_contracts():
            trades = _select_nonoverlapping_trades(outcomes, signal)
            split_metrics = {
                split: _metrics(
                    [
                        row
                        for row in trades
                        if _split_for_trade(row, split_points) == split
                    ]
                )
                for split in ("development", "validation", "oos")
            }
            aggregate = _metrics(trades)
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
                    "exit_reasons": dict(
                        sorted(
                            Counter(
                                str(row["exit_reason"]) for row in trades
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

    campaign_failures = []
    if duration_hours < MIN_REQUIRED_HOURS:
        campaign_failures.append("evidence_duration_below_72h")
    if not any(row["statistical_pass"] for row in candidates):
        campaign_failures.append("no_candidate_passed_validation_and_oos")
    return {
        "schema_version": EVENT_STUDY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_dir": str(source),
            "manifest_sha256": validation["manifest_sha256"],
            "database_sha256": validation["database_sha256"],
            "source_session_id": manifest["source_session_id"],
        },
        "fixed_contract": {
            "horizons_minutes": list(HORIZONS_MINUTES),
            "take_profit_net_bps": TAKE_PROFIT_NET_BPS,
            "stop_loss_contract": (
                "entry_roundtrip_net_bps-20_adverse_move_bps"
            ),
            "stop_loss_adverse_move_bps": STOP_LOSS_ADVERSE_MOVE_BPS,
            "min_move_budget_bps": MIN_MOVE_BUDGET_BPS,
            "move_budget_formula": (
                "abs(return_5m_bps)+2*realized_vol_5m_bps"
            ),
            "barrier_observation": "minute_close_executable_book",
            "execution_notional_usd": manifest["execution_notional_usd"],
            "fees": "captured taker fee on entry and exit",
            "slippage": "captured books5 executable VWAP on entry and exit",
            "selection": (
                "highest_fixed_score_per_timestamp;"
                "one_global_position_until_actual_exit"
            ),
            "chronological_splits_pct": [60, 20, 20],
        },
        "outcomes": {
            str(horizon): _outcome_summary(rows)
            for horizon, rows in outcomes_by_horizon.items()
        },
        "evidence": {
            "duration_hours": duration_hours,
            "single_source_session": True,
        },
        "candidates": candidates,
        "campaign_failures": campaign_failures,
        "verdict": "research_only_not_promotion_evidence",
        "research_only": True,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def render_event_study_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bitget event-driven triple-barrier study",
        "",
        "## Verdict",
        "",
        "**RESEARCH ONLY - NOT PROMOTION EVIDENCE**",
        "",
        (
            f"Evidence: {report['evidence']['duration_hours']:.2f} hours. "
            f"Take-profit: +"
            f"{report['fixed_contract']['take_profit_net_bps']:.0f} net bps; "
            f"stop: initial cost - "
            f"{report['fixed_contract']['stop_loss_adverse_move_bps']:.0f} bps."
        ),
        "",
        "| Candidate | Trades | Mean net, bps | OOS trades | OOS mean, bps | "
        "OOS LCB, bps | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in report["candidates"]:
        aggregate = candidate["aggregate"]
        oos = candidate["splits"]["oos"]
        lines.append(
            f"| {candidate['candidate_id']} | {aggregate['closed_trades']} | "
            f"{_fmt(aggregate['mean_net_bps'])} | {oos['closed_trades']} | "
            f"{_fmt(oos['mean_net_bps'])} | {_fmt(oos['lcb_95_net_bps'])} | "
            f"{str(candidate['statistical_pass']).lower()} |"
        )
    lines.extend(
        [
            "",
            "Campaign failures: "
            + (
                ", ".join(report["campaign_failures"])
                if report["campaign_failures"]
                else "none"
            ),
            "",
            (
                "Signals use entry-time features only. Future executable books "
                "are used only for triple-barrier outcomes."
            ),
            "",
            "No runtime policy, paper order or live order is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_bars(database: Path) -> dict[str, list[dict[str, Any]]]:
    query = """
        SELECT bar_timestamp_ms, symbol, continuity_id, eligible,
               mid_close, execution_curve_json, taker_fee_bps,
               return_5m_bps, realized_vol_5m_bps, flow_imbalance,
               book_imbalance_mean, microprice_edge_mean_bps,
               oi_change_5m_bps, basis_close_bps, funding_close
        FROM bars
        ORDER BY symbol, bar_timestamp_ms
    """
    result: dict[str, list[dict[str, Any]]] = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for row in connection.execute(query):
            result.setdefault(str(row["symbol"]), []).append(dict(row))
    return result


def _build_outcomes(
    bars_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    horizon_minutes: int,
) -> list[dict[str, Any]]:
    outcomes = []
    for symbol, bars in bars_by_symbol.items():
        for index, entry in enumerate(bars):
            if not _entry_eligible(entry):
                continue
            for direction in ("LONG", "SHORT"):
                outcome = _triple_barrier_outcome(
                    bars=bars,
                    entry_index=index,
                    direction=direction,
                    horizon_minutes=horizon_minutes,
                )
                if outcome is not None:
                    outcomes.append(
                        {
                            **outcome,
                            "symbol": symbol,
                            "return_5m_bps": float(entry["return_5m_bps"]),
                            "realized_vol_5m_bps": float(
                                entry["realized_vol_5m_bps"]
                            ),
                            "move_budget_bps": (
                                abs(float(entry["return_5m_bps"]))
                                + 2.0
                                * float(entry["realized_vol_5m_bps"])
                            ),
                            "flow_imbalance": float(entry["flow_imbalance"]),
                            "book_imbalance_mean": float(
                                entry["book_imbalance_mean"]
                            ),
                            "microprice_edge_mean_bps": float(
                                entry["microprice_edge_mean_bps"]
                            ),
                            "oi_change_5m_bps": float(
                                entry["oi_change_5m_bps"] or 0.0
                            ),
                            "basis_close_bps": float(
                                entry["basis_close_bps"] or 0.0
                            ),
                            "funding_close": float(
                                entry["funding_close"] or 0.0
                            ),
                        }
                    )
    return outcomes


def _entry_eligible(entry: Mapping[str, Any]) -> bool:
    return (
        int(entry["eligible"]) == 1
        and entry["mid_close"] is not None
        and entry["return_5m_bps"] is not None
        and entry["realized_vol_5m_bps"] is not None
        and entry["book_imbalance_mean"] is not None
        and entry["microprice_edge_mean_bps"] is not None
    )


def _triple_barrier_outcome(
    *,
    bars: Sequence[Mapping[str, Any]],
    entry_index: int,
    direction: str,
    horizon_minutes: int,
) -> dict[str, Any] | None:
    exit_limit = entry_index + int(horizon_minutes)
    if exit_limit >= len(bars):
        return None
    entry = bars[entry_index]
    entry_time = int(entry["bar_timestamp_ms"])
    continuity = int(entry["continuity_id"])
    entry_mid = float(entry["mid_close"])
    entry_execution = _execution_price(
        entry["execution_curve_json"],
        side="buy" if direction == "LONG" else "sell",
    )
    immediate_exit_execution = _execution_price(
        entry["execution_curve_json"],
        side="sell" if direction == "LONG" else "buy",
    )
    if entry_execution is None or immediate_exit_execution is None:
        return None
    initial_roundtrip_net_bps = _net_execution_bps(
        direction=direction,
        entry_price=float(entry_execution),
        exit_price=float(immediate_exit_execution),
        notional_usd=25.0,
        taker_fee_bps=float(entry["taker_fee_bps"]),
    )
    stop_loss_net_bps = (
        initial_roundtrip_net_bps - STOP_LOSS_ADVERSE_MOVE_BPS
    )
    selected: dict[str, Any] | None = None
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
        exit_execution = _execution_price(
            future["execution_curve_json"],
            side="sell" if direction == "LONG" else "buy",
        )
        if exit_execution is None:
            return None
        net_bps = _net_execution_bps(
            direction=direction,
            entry_price=float(entry_execution),
            exit_price=float(exit_execution),
            notional_usd=25.0,
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
            selected = {
                "entry_bar_timestamp_ms": entry_time,
                "exit_bar_timestamp_ms": int(future["bar_timestamp_ms"]),
                "direction": direction,
                "holding_minutes": future_index - entry_index,
                "exit_reason": exit_reason,
                "gross_mid_bps": gross_mid_bps,
                "net_execution_bps": net_bps,
                "total_cost_bps": gross_mid_bps - net_bps,
                "initial_roundtrip_net_bps": initial_roundtrip_net_bps,
                "stop_loss_net_bps": stop_loss_net_bps,
            }
            break
    return selected


def _candidate_contracts() -> Sequence[
    tuple[str, Callable[[Mapping[str, Any]], float]]
]:
    return (
        ("cost_gate_trend_v1", _cost_gate_trend),
        ("breakout_flow_v1", _breakout_flow),
        ("breakout_book_v1", _breakout_book),
        ("breakout_quorum_v1", _breakout_quorum),
        ("oi_expansion_breakout_v1", _oi_expansion_breakout),
        ("microprice_breakout_v1", _microprice_breakout),
    )


def _cost_gate_trend(row: Mapping[str, Any]) -> float:
    if float(row["move_budget_bps"]) < MIN_MOVE_BUDGET_BPS:
        return 0.0
    trend = float(row["return_5m_bps"])
    if abs(trend) < 5.0:
        return 0.0
    direction = "LONG" if trend > 0.0 else "SHORT"
    return abs(trend) if row["direction"] == direction else 0.0


def _breakout_flow(row: Mapping[str, Any]) -> float:
    base = _cost_gate_trend(row)
    flow = float(row["flow_imbalance"])
    trend = float(row["return_5m_bps"])
    if base <= 0.0 or abs(flow) < 0.05 or trend * flow <= 0.0:
        return 0.0
    return base * abs(flow)


def _breakout_book(row: Mapping[str, Any]) -> float:
    base = _cost_gate_trend(row)
    book = float(row["book_imbalance_mean"])
    trend = float(row["return_5m_bps"])
    if base <= 0.0 or abs(book) < 0.05 or trend * book <= 0.0:
        return 0.0
    return base * abs(book)


def _breakout_quorum(row: Mapping[str, Any]) -> float:
    flow_score = _breakout_flow(row)
    book_score = _breakout_book(row)
    if flow_score <= 0.0 or book_score <= 0.0:
        return 0.0
    return flow_score + book_score


def _oi_expansion_breakout(row: Mapping[str, Any]) -> float:
    base = _breakout_flow(row)
    oi_change = float(row["oi_change_5m_bps"])
    if base <= 0.0 or oi_change < 2.0:
        return 0.0
    return base + oi_change


def _microprice_breakout(row: Mapping[str, Any]) -> float:
    base = _breakout_flow(row)
    edge = float(row["microprice_edge_mean_bps"])
    trend = float(row["return_5m_bps"])
    if base <= 0.0 or abs(edge) < 0.02 or trend * edge <= 0.0:
        return 0.0
    return base + abs(edge)


def _select_nonoverlapping_trades(
    rows: Sequence[Mapping[str, Any]],
    signal: Callable[[Mapping[str, Any]], float],
) -> list[dict[str, Any]]:
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
    selected = []
    next_allowed = -1
    for timestamp in sorted(by_timestamp):
        if timestamp < next_allowed:
            continue
        best = max(
            by_timestamp[timestamp],
            key=lambda row: (float(row["score"]), str(row["symbol"])),
        )
        selected.append(best)
        next_allowed = int(best["exit_bar_timestamp_ms"])
    return selected


def _split_points(timestamps: Sequence[int]) -> tuple[int, int]:
    start = int(timestamps[0])
    width = int(timestamps[-1]) - start
    return start + int(width * 0.60), start + int(width * 0.80)


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


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net = [float(row["net_execution_bps"]) for row in rows]
    gross = [float(row["gross_mid_bps"]) for row in rows]
    costs = [float(row["total_cost_bps"]) for row in rows]
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
    return {
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
            statistics.fmean(float(row["holding_minutes"]) for row in rows)
            if rows
            else None
        ),
        "exit_reasons": dict(
            sorted(Counter(str(row["exit_reason"]) for row in rows).items())
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


def _outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "outcomes": len(rows),
        "symbols": dict(
            sorted(Counter(str(row["symbol"]) for row in rows).items())
        ),
        "directions": dict(
            sorted(Counter(str(row["direction"]) for row in rows).items())
        ),
        "exit_reasons": dict(
            sorted(Counter(str(row["exit_reason"]) for row in rows).items())
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
