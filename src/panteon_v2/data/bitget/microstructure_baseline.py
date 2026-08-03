"""Fixed, research-only baselines for the Bitget microstructure dataset."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .research_dataset import validate_research_dataset


BASELINE_SCHEMA_VERSION = "panteon.bitget_microstructure_baselines.v1"
HORIZONS_MINUTES = (5, 15)
LCB_Z = 1.6448536269514722
MIN_REQUIRED_HOURS = 72.0


def evaluate_fixed_baselines(dataset_dir: str | Path) -> dict[str, Any]:
    validation = validate_research_dataset(dataset_dir)
    source = Path(dataset_dir).resolve()
    manifest = json.loads(
        (source / "microstructure_research_v1.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    database = source / str(manifest["database_file"])
    rows_by_horizon = {
        horizon: _load_rows(database, horizon_minutes=horizon)
        for horizon in HORIZONS_MINUTES
    }
    timestamps = sorted(
        {
            int(row["bar_timestamp_ms"])
            for rows in rows_by_horizon.values()
            for row in rows
        }
    )
    if not timestamps:
        raise ValueError("eligible baseline rows are empty")
    split_points = _split_points(timestamps)
    duration_hours = (
        (timestamps[-1] - timestamps[0]) / 3_600_000.0
        if len(timestamps) >= 2
        else 0.0
    )
    candidates = []
    for horizon, rows in rows_by_horizon.items():
        for candidate_id, signal in _candidate_contracts():
            trades = _select_nonoverlapping_trades(
                rows,
                signal,
                horizon_minutes=horizon,
            )
            split_metrics = {
                split: _metrics(
                    [
                        row
                        for row in trades
                        if _split_name(
                            int(row["bar_timestamp_ms"]),
                            split_points,
                        )
                        == split
                    ]
                )
                for split in ("development", "validation", "oos")
            }
            aggregate = _metrics(trades)
            statistical_failures = _statistical_failures(split_metrics)
            candidates.append(
                {
                    "candidate_id": f"{candidate_id}@{horizon}m",
                    "signal_contract_id": candidate_id,
                    "horizon_minutes": horizon,
                    "selection": (
                        "highest_absolute_fixed_score_per_timestamp;"
                        "single_global_nonoverlapping_position"
                    ),
                    "aggregate": aggregate,
                    "splits": split_metrics,
                    "statistical_pass": not statistical_failures,
                    "failures": statistical_failures,
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
    campaign_failures = []
    if duration_hours < MIN_REQUIRED_HOURS:
        campaign_failures.append("evidence_duration_below_72h")
    if (
        len(
            {
                str(row["source_session_id"])
                for horizon_rows in rows_by_horizon.values()
                for row in horizon_rows
            }
        )
        < 1
    ):
        campaign_failures.append("source_session_missing")
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_dir": str(source),
            "manifest_sha256": validation["manifest_sha256"],
            "database_sha256": validation["database_sha256"],
            "source_session_id": manifest["source_session_id"],
        },
        "fixed_contract": {
            "horizons_minutes": list(HORIZONS_MINUTES),
            "execution_notional_usd": manifest["execution_notional_usd"],
            "fees": "source instrument-rule taker fee on entry and exit",
            "slippage": "source books5 executable VWAP on entry and exit",
            "chronological_splits_pct": [60, 20, 20],
            "min_fills": 20,
            "min_closed_trades": 10,
            "required_positive_validation_expectancy": True,
            "required_positive_oos_expectancy": True,
            "required_positive_oos_lcb": True,
        },
        "evidence": {
            "duration_hours": duration_hours,
            "eligible_rows_by_horizon": {
                str(horizon): len(rows)
                for horizon, rows in rows_by_horizon.items()
            },
            "single_source_session": True,
        },
        "candidates": candidates,
        "campaign_failures": campaign_failures,
        "verdict": "research_only_not_promotion_evidence",
        "research_only": True,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def render_baseline_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Bitget microstructure fixed baselines",
        "",
        "## Verdict",
        "",
        "**RESEARCH ONLY - NOT PROMOTION EVIDENCE**",
        "",
        (
            f"Evidence: {report['evidence']['duration_hours']:.2f} hours; "
            "eligible direction rows: "
            f"{report['evidence']['eligible_rows_by_horizon']}."
        ),
        "",
        "| Candidate | Trades | Mean net, bps | OOS trades | OOS mean, bps | "
        "OOS LCB, bps | Statistical pass |",
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
                "All thresholds are fixed in code. Costs use the captured "
                "Bitget taker fee and executable books5 VWAP on both legs."
            ),
            "",
            "No policy, paper order, live order or promotion artifact is created.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_rows(
    database: Path,
    *,
    horizon_minutes: int,
) -> list[dict[str, Any]]:
    query = """
        SELECT b.bar_timestamp_ms, b.symbol, b.return_5m_bps,
               b.flow_imbalance, b.book_imbalance_mean,
               b.microprice_edge_mean_bps, b.basis_close_bps,
               b.funding_close, b.oi_change_5m_bps,
               l.direction, l.gross_mid_bps, l.net_execution_bps,
               l.total_cost_bps
        FROM bars b
        JOIN labels l
          ON l.entry_bar_timestamp_ms = b.bar_timestamp_ms
         AND l.symbol = b.symbol
        WHERE b.eligible = 1
          AND l.eligible = 1
          AND l.horizon_minutes = ?
          AND b.return_5m_bps IS NOT NULL
          AND b.book_imbalance_mean IS NOT NULL
          AND b.microprice_edge_mean_bps IS NOT NULL
        ORDER BY b.bar_timestamp_ms, b.symbol, l.direction
    """
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in connection.execute(query, (int(horizon_minutes),))
        ]
    source_session_id = _source_session_id(database.parent)
    for row in rows:
        row["source_session_id"] = source_session_id
    return rows


def _source_session_id(dataset_dir: Path) -> str:
    manifest = json.loads(
        (dataset_dir / "microstructure_research_v1.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return str(manifest["source_session_id"])


def _candidate_contracts() -> Sequence[
    tuple[str, Callable[[Mapping[str, Any]], float]]
]:
    return (
        ("control_always_long_v1", _always_long),
        ("control_always_short_v1", _always_short),
        ("trend_flow_two_sided_v1", _trend_flow),
        ("trend_book_two_sided_v1", _trend_book),
        ("flow_book_two_sided_v1", _flow_book),
        ("microprice_flow_two_sided_v1", _microprice_flow),
        ("carry_pressure_short_v1", _carry_pressure_short),
    )


def _always_long(row: Mapping[str, Any]) -> float:
    return 1.0 if row["direction"] == "LONG" else 0.0


def _always_short(row: Mapping[str, Any]) -> float:
    return 1.0 if row["direction"] == "SHORT" else 0.0


def _trend_flow(row: Mapping[str, Any]) -> float:
    trend = float(row["return_5m_bps"])
    flow = float(row["flow_imbalance"])
    if abs(trend) < 5.0 or abs(flow) < 0.03 or trend * flow <= 0.0:
        return 0.0
    direction = "LONG" if trend > 0.0 else "SHORT"
    return abs(trend) * abs(flow) if row["direction"] == direction else 0.0


def _trend_book(row: Mapping[str, Any]) -> float:
    trend = float(row["return_5m_bps"])
    book = float(row["book_imbalance_mean"])
    if abs(trend) < 5.0 or abs(book) < 0.03 or trend * book <= 0.0:
        return 0.0
    direction = "LONG" if trend > 0.0 else "SHORT"
    return abs(trend) * abs(book) if row["direction"] == direction else 0.0


def _flow_book(row: Mapping[str, Any]) -> float:
    flow = float(row["flow_imbalance"])
    book = float(row["book_imbalance_mean"])
    if abs(flow) < 0.05 or abs(book) < 0.05 or flow * book <= 0.0:
        return 0.0
    direction = "LONG" if flow > 0.0 else "SHORT"
    return abs(flow) * abs(book) * 100.0 if row["direction"] == direction else 0.0


def _microprice_flow(row: Mapping[str, Any]) -> float:
    edge = float(row["microprice_edge_mean_bps"])
    flow = float(row["flow_imbalance"])
    if abs(edge) < 0.02 or abs(flow) < 0.03 or edge * flow <= 0.0:
        return 0.0
    direction = "LONG" if edge > 0.0 else "SHORT"
    return abs(edge) * abs(flow) if row["direction"] == direction else 0.0


def _carry_pressure_short(row: Mapping[str, Any]) -> float:
    if row["direction"] != "SHORT":
        return 0.0
    funding_bps = float(row["funding_close"] or 0.0) * 10_000.0
    basis = float(row["basis_close_bps"] or 0.0)
    oi_change = float(row["oi_change_5m_bps"] or 0.0)
    flow = float(row["flow_imbalance"])
    if funding_bps <= 0.0 or basis <= 0.0 or oi_change <= 0.0 or flow >= -0.03:
        return 0.0
    return funding_bps + basis + oi_change * abs(flow)


def _select_nonoverlapping_trades(
    rows: Sequence[Mapping[str, Any]],
    signal: Callable[[Mapping[str, Any]], float],
    *,
    horizon_minutes: int,
) -> list[dict[str, Any]]:
    by_timestamp: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        score = float(signal(row))
        if score <= 0.0:
            continue
        candidate = dict(row)
        candidate["score"] = score
        by_timestamp.setdefault(int(row["bar_timestamp_ms"]), []).append(candidate)
    selected: list[dict[str, Any]] = []
    next_allowed = -1
    for timestamp in sorted(by_timestamp):
        if timestamp < next_allowed:
            continue
        best = max(
            by_timestamp[timestamp],
            key=lambda row: (float(row["score"]), str(row["symbol"])),
        )
        selected.append(best)
        next_allowed = timestamp + int(horizon_minutes) * 60_000
    return selected


def _split_points(timestamps: Sequence[int]) -> tuple[int, int]:
    start = int(timestamps[0])
    width = int(timestamps[-1]) - start
    return start + int(width * 0.60), start + int(width * 0.80)


def _split_name(timestamp: int, points: tuple[int, int]) -> str:
    if timestamp <= points[0]:
        return "development"
    if timestamp <= points[1]:
        return "validation"
    return "oos"


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
    }


def _statistical_failures(
    splits: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    failures = []
    validation = splits["validation"]
    oos = splits["oos"]
    if int(validation["fills"]) < 20:
        failures.append("validation_fills_below_20")
    if int(validation["closed_trades"]) < 10:
        failures.append("validation_closed_trades_below_10")
    if (
        validation["mean_net_bps"] is None
        or float(validation["mean_net_bps"]) <= 0.0
    ):
        failures.append("validation_nonpositive_costed_expectancy")
    if int(oos["fills"]) < 20:
        failures.append("oos_fills_below_20")
    if int(oos["closed_trades"]) < 10:
        failures.append("oos_closed_trades_below_10")
    if oos["mean_net_bps"] is None or float(oos["mean_net_bps"]) <= 0.0:
        failures.append("oos_nonpositive_costed_expectancy")
    if oos["lcb_95_net_bps"] is None or float(oos["lcb_95_net_bps"]) <= 0.0:
        failures.append("oos_nonpositive_lcb")
    return failures


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"
