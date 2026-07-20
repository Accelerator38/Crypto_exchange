"""Segment-aware CarryFlow screening that never joins discontinuous tapes."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..policy.evidence_tape import CarryFlowEvidenceTape


@dataclass(frozen=True)
class CarryFlowScreeningConfig:
    oi_spike: float = 0.02
    funding_entry: float = 0.00008
    crowd_ratio: float = 0.58
    current_short_basis_floor: float = 0.0004
    research_short_basis_floor: float = -0.0015
    stop_pct: float = 0.012
    target_pct: float = 0.024
    hold_bars: int = 6
    round_trip_cost_bps: float = 12.0

    def __post_init__(self) -> None:
        for name in (
            "oi_spike",
            "funding_entry",
            "crowd_ratio",
            "current_short_basis_floor",
            "research_short_basis_floor",
            "stop_pct",
            "target_pct",
            "round_trip_cost_bps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.oi_spike <= 0.0:
            raise ValueError("oi_spike must be positive")
        if not 0.0 < self.crowd_ratio < 1.0:
            raise ValueError("crowd_ratio must be in (0, 1)")
        if self.stop_pct <= 0.0 or self.target_pct <= 0.0:
            raise ValueError("stop_pct and target_pct must be positive")
        if self.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        if self.round_trip_cost_bps < 0.0:
            raise ValueError("round_trip_cost_bps must be non-negative")


@dataclass(frozen=True)
class ScreeningTrade:
    segment: str
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    gross_bps: float
    cost_bps: float
    net_bps: float
    exit_reason: str


def screen_carryflow_segments(
    tapes: Sequence[CarryFlowEvidenceTape],
    *,
    config: CarryFlowScreeningConfig | None = None,
) -> dict[str, Any]:
    if not tapes:
        raise ValueError("at least one evidence tape is required")
    cfg = config or CarryFlowScreeningConfig()
    symbol_sets = {tuple(tape.symbols) for tape in tapes}
    intervals = {int(tape.bar_interval_seconds) for tape in tapes}
    if len(symbol_sets) != 1:
        raise ValueError("screening tapes must have the same symbol set")
    if len(intervals) != 1:
        raise ValueError("screening tapes must have the same bar interval")

    funnel = Counter()
    basis_values: list[float] = []
    oi_values: list[float] = []
    current_events: list[dict[str, Any]] = []
    research_events: list[dict[str, Any]] = []
    current_trades: list[ScreeningTrade] = []
    research_trades: list[ScreeningTrade] = []
    segment_rows: list[dict[str, Any]] = []

    for tape in tapes:
        segment = tape.collector_run_id
        current_result = _simulate_segment(
            tape,
            cfg,
            basis_floor=cfg.current_short_basis_floor,
        )
        research_result = _simulate_segment(
            tape,
            cfg,
            basis_floor=cfg.research_short_basis_floor,
        )
        current_trades.extend(current_result["trades"])
        research_trades.extend(research_result["trades"])
        previous_oi: dict[str, float] = {}
        segment_funnel = Counter()
        for bar_index, sample in enumerate(tape.samples, start=1):
            for symbol in tape.symbols:
                row = sample["symbols"][symbol]
                if not bool(row["complete"]):
                    continue
                derivatives = row["derivatives"]
                mark = float(derivatives["mark_price"])
                index = float(derivatives["index_price"])
                oi_now = float(derivatives["open_interest_usdt"])
                basis = mark / index - 1.0 if mark > 0.0 and index > 0.0 else 0.0
                basis_values.append(basis)
                segment_funnel["complete_symbol_observations"] += 1
                prior = previous_oi.get(symbol)
                previous_oi[symbol] = oi_now
                if prior is None or prior <= 0.0 or oi_now <= 0.0:
                    continue
                oi_change = oi_now / prior - 1.0
                oi_values.append(oi_change)
                segment_funnel["adjacent_oi_observations"] += 1
                if oi_change < cfg.oi_spike:
                    continue
                segment_funnel["oi_pass"] += 1
                funding = float(derivatives["funding_rate"])
                if funding < cfg.funding_entry:
                    continue
                segment_funnel["funding_pass"] += 1
                long_ratio = float(derivatives["long_ratio"])
                if long_ratio < cfg.crowd_ratio:
                    continue
                segment_funnel["crowding_pass"] += 1
                event = {
                    "segment": segment,
                    "bar": bar_index,
                    "symbol": symbol,
                    "oi_change": oi_change,
                    "funding_rate": funding,
                    "long_ratio": long_ratio,
                    "basis": basis,
                }
                if basis >= cfg.research_short_basis_floor:
                    segment_funnel["research_basis_pass"] += 1
                    research_events.append(event)
                if basis >= cfg.current_short_basis_floor:
                    segment_funnel["current_basis_pass"] += 1
                    current_events.append(event)
        funnel.update(segment_funnel)
        segment_rows.append(
            {
                **tape.describe(),
                "gate_funnel": _funnel_dict(segment_funnel),
                "current_closed_trades": len(current_result["trades"]),
                "research_closed_trades": len(research_result["trades"]),
                "current_unclosed_entries": current_result["unclosed_entries"],
                "research_unclosed_entries": research_result["unclosed_entries"],
            }
        )

    basis_bps = [value * 10_000.0 for value in basis_values]
    current_metrics = _trade_metrics(current_trades)
    research_metrics = _trade_metrics(research_trades)
    samples = sum(len(tape.samples) for tape in tapes)
    interval_seconds = next(iter(intervals))
    observed_hours = samples * interval_seconds / 3600.0
    current_structural_block = bool(
        basis_values
        and max(basis_values) < cfg.current_short_basis_floor
    )
    current_event_rate = _safe_rate(
        len(current_events), funnel["adjacent_oi_observations"]
    )
    research_event_rate = _safe_rate(
        len(research_events), funnel["adjacent_oi_observations"]
    )
    return {
        "schema_version": "panteon.carryflow_segment_screening.v1",
        "research_only": True,
        "promotion_authority": False,
        "segments_concatenated": False,
        "segment_count": len(tapes),
        "symbols": list(next(iter(symbol_sets))),
        "bar_interval_seconds": interval_seconds,
        "total_segment_hours": observed_hours,
        "configuration": {
            key: getattr(cfg, key)
            for key in cfg.__dataclass_fields__
        },
        "segments": segment_rows,
        "gate_funnel": _funnel_dict(funnel),
        "basis_bps": _distribution(basis_bps),
        "oi_change_pct": _distribution([value * 100.0 for value in oi_values]),
        "current_policy": {
            "structural_activation_block": current_structural_block,
            "structural_block_reason": (
                "observed_max_basis_below_current_short_basis_floor"
                if current_structural_block
                else ""
            ),
            "precondition_events": len(current_events),
            "precondition_event_rate": current_event_rate,
            "zero_event_rate_upper_95": _zero_event_upper_bound(
                funnel["adjacent_oi_observations"]
            ) if not current_events else None,
            "trade_metrics": current_metrics,
        },
        "research_candidate": {
            "description": (
                "short-only OI/funding/crowding precondition with a "
                f"{cfg.research_short_basis_floor * 10_000.0:.1f} bps basis "
                f"sanity floor and {cfg.hold_bars}-bar maximum hold"
            ),
            "exact_actor_momentum_gate_included": False,
            "precondition_events": len(research_events),
            "precondition_event_rate": research_event_rate,
            "events_by_segment": dict(
                Counter(row["segment"] for row in research_events)
            ),
            "events_by_symbol": dict(
                Counter(row["symbol"] for row in research_events)
            ),
            "trade_metrics": research_metrics,
            "estimated_segment_hours_per_10_closed": (
                observed_hours / research_metrics["closed_trades"] * 10.0
                if research_metrics["closed_trades"]
                else None
            ),
        },
        "decision": _screening_decision(
            current_structural_block=current_structural_block,
            research_metrics=research_metrics,
        ),
    }


def _simulate_segment(
    tape: CarryFlowEvidenceTape,
    config: CarryFlowScreeningConfig,
    *,
    basis_floor: float,
) -> dict[str, Any]:
    previous_oi: dict[str, float] = {}
    position: dict[str, Any] | None = None
    trades: list[ScreeningTrade] = []
    unclosed_entries = 0
    for bar_index, sample in enumerate(tape.samples, start=1):
        prices = {
            symbol: float(sample["symbols"][symbol]["market"]["decision_price"])
            for symbol in tape.symbols
            if bool(sample["symbols"][symbol]["complete"])
        }
        if position is not None and position["symbol"] in prices:
            exit_price = prices[position["symbol"]]
            short_return = (
                position["entry_price"] - exit_price
            ) / position["entry_price"]
            held = bar_index - position["entry_bar"]
            exit_reason = ""
            if short_return <= -config.stop_pct:
                exit_reason = "stop"
            elif short_return >= config.target_pct:
                exit_reason = "target"
            elif held >= config.hold_bars:
                exit_reason = "hold"
            if exit_reason:
                gross_bps = short_return * 10_000.0
                trades.append(
                    ScreeningTrade(
                        segment=tape.collector_run_id,
                        symbol=position["symbol"],
                        entry_bar=position["entry_bar"],
                        exit_bar=bar_index,
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        gross_bps=gross_bps,
                        cost_bps=config.round_trip_cost_bps,
                        net_bps=gross_bps - config.round_trip_cost_bps,
                        exit_reason=exit_reason,
                    )
                )
                position = None

        candidates: list[tuple[float, str, float]] = []
        for symbol in tape.symbols:
            row = sample["symbols"][symbol]
            if not bool(row["complete"]):
                continue
            derivatives = row["derivatives"]
            oi_now = float(derivatives["open_interest_usdt"])
            prior = previous_oi.get(symbol)
            previous_oi[symbol] = oi_now
            if prior is None or prior <= 0.0 or oi_now <= 0.0:
                continue
            oi_change = oi_now / prior - 1.0
            funding = float(derivatives["funding_rate"])
            long_ratio = float(derivatives["long_ratio"])
            mark = float(derivatives["mark_price"])
            index = float(derivatives["index_price"])
            basis = mark / index - 1.0 if mark > 0.0 and index > 0.0 else 0.0
            if not (
                oi_change >= config.oi_spike
                and funding >= config.funding_entry
                and long_ratio >= config.crowd_ratio
                and basis >= basis_floor
            ):
                continue
            score = (
                (funding - config.funding_entry) * 10_000.0
                + (long_ratio - config.crowd_ratio) * 4.0
                + (basis - basis_floor) * 1_000.0
                + (oi_change - config.oi_spike) * 10.0
            )
            candidates.append(
                (
                    score,
                    symbol,
                    float(row["market"]["decision_price"]),
                )
            )
        if position is None and candidates:
            _, symbol, price = max(candidates, key=lambda item: (item[0], item[1]))
            position = {
                "symbol": symbol,
                "entry_bar": bar_index,
                "entry_price": price,
            }
    if position is not None:
        unclosed_entries = 1
    return {"trades": trades, "unclosed_entries": unclosed_entries}


def _trade_metrics(trades: Sequence[ScreeningTrade]) -> dict[str, Any]:
    values = [trade.net_bps for trade in trades]
    closed = len(values)
    mean = statistics.fmean(values) if values else 0.0
    lcb = (
        mean - 1.6448536269514722 * statistics.stdev(values) / math.sqrt(closed)
        if closed >= 2
        else None
    )
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    failures: list[str] = []
    if closed * 2 < 20:
        failures.append("fills_below_20")
    if closed < 10:
        failures.append("closed_trades_below_10")
    if mean <= 0.0:
        failures.append("nonpositive_costed_expectancy")
    if lcb is None or lcb <= 0.0:
        failures.append("nonpositive_lcb")
    symbol_counts = Counter(trade.symbol for trade in trades)
    segment_counts = Counter(trade.segment for trade in trades)
    return {
        "filled_orders": closed * 2,
        "closed_trades": closed,
        "mean_net_bps": mean,
        "median_net_bps": statistics.median(values) if values else 0.0,
        "expectancy_lcb_95_bps": lcb,
        "total_net_bps": sum(values),
        "positive_rate": _safe_rate(sum(value > 0.0 for value in values), closed),
        "max_drawdown_bps": drawdown,
        "mean_cost_bps": (
            statistics.fmean(trade.cost_bps for trade in trades)
            if trades
            else 0.0
        ),
        "exits": dict(Counter(trade.exit_reason for trade in trades)),
        "trades_by_symbol": dict(symbol_counts),
        "trades_by_segment": dict(segment_counts),
        "max_symbol_share": (
            max(symbol_counts.values()) / closed if closed else None
        ),
        "max_segment_share": (
            max(segment_counts.values()) / closed if closed else None
        ),
        "screening_gate_passed": not failures,
        "failures": failures,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _quantile(ordered: Sequence[float], probability: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _funnel_dict(counter: Mapping[str, int]) -> dict[str, int]:
    return {
        key: int(counter.get(key, 0))
        for key in (
            "complete_symbol_observations",
            "adjacent_oi_observations",
            "oi_pass",
            "funding_pass",
            "crowding_pass",
            "current_basis_pass",
            "research_basis_pass",
        )
    }


def _zero_event_upper_bound(observations: int) -> float | None:
    if observations < 1:
        return None
    return 1.0 - math.pow(0.05, 1.0 / observations)


def _screening_decision(
    *,
    current_structural_block: bool,
    research_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    actions = []
    if current_structural_block:
        actions.append("stop_current_positive_basis_candidate")
    actions.extend(
        (
            "use_discontinuous_segments_for_r_and_d_screening_only",
            "create_prospective_warmup_seed_before_new_tape",
            "run_exact_actor_replay_for_selected_research_candidate",
            "start_continuous_canary_only_after_positive_segment_screening",
        )
    )
    return {
        "current_candidate_continue_collecting": not current_structural_block,
        "research_candidate_ready_for_promotion": False,
        "research_candidate_screening_passed": bool(
            research_metrics["screening_gate_passed"]
        ),
        "actions": actions,
    }
