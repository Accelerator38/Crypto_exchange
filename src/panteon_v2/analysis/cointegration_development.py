from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from panteon_v2.policy.strategy_candidate_registry import (
    StrategyCandidateRegistry,
)

from .strategy_lab import (
    STRATEGY_LAB_SCHEMA_VERSION,
    MarketFrame,
    SymbolSeries,
    _build_symbol_series,
    _group_metrics,
    _metrics,
    _timestamp_ms,
    aggregate_market_frames,
    load_aligned_hourly_market,
    verify_registered_dataset,
)


class CointegrationDevelopmentError(ValueError):
    pass


@dataclass(frozen=True)
class CointegrationModel:
    y_symbol: str
    x_symbol: str
    formation_start_index: int
    formation_end_index: int
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    pvalue: float
    half_life_bars: float

    @property
    def pair_id(self) -> str:
        return f"{self.y_symbol}|{self.x_symbol}"


@dataclass(frozen=True)
class CointegrationSignal:
    model: CointegrationModel
    direction: str
    signal_index: int
    zscore: float
    estimated_reversion_bps: float


@dataclass
class CointegrationPosition:
    signal: CointegrationSignal
    entry_index: int
    entry_timestamp_ms: int
    y_entry_price: float
    x_entry_price: float
    y_notional_usd: float
    x_notional_usd: float
    pending_exit_reason: str = ""


def evaluate_cointegration_development(
    contract: Mapping[str, Any],
    registry: StrategyCandidateRegistry,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    contract_dataset = contract["dataset"]
    registered_dataset = registry.payload["dataset"]
    for key in (
        "dataset_id",
        "data_dir",
        "dataset_sha256",
        "integrity_manifest_sha256",
        "symbols",
    ):
        if contract_dataset[key] != registered_dataset[key]:
            raise CointegrationDevelopmentError(
                f"cointegration dataset mismatch: {key}"
            )
    provenance = verify_registered_dataset(registry, repository_root=root)
    hourly_frames, _ = load_aligned_hourly_market(
        registry,
        repository_root=root,
    )
    candidate = contract["candidate"]
    aggregate_bars = int(candidate["event_contract"]["aggregate_bars"])
    frames = aggregate_market_frames(
        hourly_frames,
        bars_per_frame=aggregate_bars,
    )
    symbols = tuple(str(item) for item in contract_dataset["symbols"])
    series = {
        symbol: _build_symbol_series(symbol, frames) for symbol in symbols
    }
    timestamps = [frame.timestamp_ms for frame in frames]
    protocol = contract["protocol"]
    start_timestamp = _timestamp_ms(protocol["development_start_at"])
    end_timestamp = _timestamp_ms(protocol["development_end_at"])
    start_index = int(np.searchsorted(timestamps, start_timestamp, side="left"))
    end_index = int(np.searchsorted(timestamps, end_timestamp, side="right")) - 1
    if end_index < start_index:
        raise CointegrationDevelopmentError(
            "cointegration development split has no 4h bars"
        )
    split = simulate_cointegration_split(
        candidate,
        split_name="development",
        frames=frames,
        series=series,
        start_index=start_index,
        end_index=end_index,
        costs=protocol["costs"],
        portfolio=protocol["portfolio"],
    )
    verdict = _development_verdict(
        candidate,
        split=split,
        gates=protocol["gates"],
    )
    return {
        "schema_version": STRATEGY_LAB_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screen_type": "walk_forward_cointegration_spread_4h_development",
        "stage": "development_only",
        "sealed_windows": list(protocol["sealed_windows"]),
        "candidate_id": candidate["candidate_id"],
        "profile_sha256": candidate["profile_sha256"],
        "contract_path": (
            "configs/strategy_candidate_p6_cointegration_spread_v1.json"
        ),
        "library": {
            "name": "statsmodels",
            "cointegration_test": candidate["event_contract"][
                "cointegration_test"
            ],
            "trend": candidate["event_contract"]["cointegration_trend"],
            "autolag": candidate["event_contract"]["adf_autolag"],
        },
        "dataset": provenance,
        "aggregation": {
            "source_timeframe": "1h",
            "bars_per_frame": aggregate_bars,
            "result_timeframe": "4h",
            "frame_count": len(frames),
        },
        "development_window": {
            "start_timestamp_ms": frames[start_index].timestamp_ms,
            "end_timestamp_ms": frames[end_index].timestamp_ms,
            "bars": end_index - start_index + 1,
        },
        "costs": protocol["costs"],
        "portfolio": protocol["portfolio"],
        "gates": protocol["gates"],
        "candidate": split,
        "baseline": _cash_baseline(),
        **verdict,
        "runtime_actor_created": False,
        "validation_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def simulate_cointegration_split(
    candidate: Mapping[str, Any],
    *,
    split_name: str,
    frames: Sequence[MarketFrame],
    series: Mapping[str, SymbolSeries],
    start_index: int,
    end_index: int,
    costs: Mapping[str, Any],
    portfolio: Mapping[str, Any],
) -> dict[str, Any]:
    event = candidate["event_contract"]
    exit_contract = candidate["exit_contract"]
    selection = candidate["selection_contract"]
    formation_bars = int(event["formation_bars"])
    refit_bars = int(event["refit_every_bars"])
    real_cost_bps = float(costs["round_trip_fee_bps"]) + float(
        costs["slippage_bps"]
    )
    stressed_cost_bps = max(
        real_cost_bps * float(costs["cost_stress_multiplier"]),
        real_cost_bps + float(costs.get("safety_buffer_bps", 0.0)),
    )
    gross_notional = float(portfolio["gross_notional_per_trade_usd"])

    current_model: CointegrationModel | None = None
    position: CointegrationPosition | None = None
    pending_entry: CointegrationSignal | None = None
    last_exit_index = -10**9
    trades: list[dict[str, Any]] = []
    leg_trades: list[dict[str, Any]] = []
    formation_refits = qualified_refits = tested_pairs = 0
    formation_leakage_violations = paired_fill_atomicity_violations = 0
    selected_pair_refits: dict[str, int] = {}
    candidate_signals = selected_signals = entry_fills = exit_fills = 0

    first_refit_index = max(start_index, formation_bars - 1)
    for index in range(start_index, end_index + 1):
        frame = frames[index]
        if pending_entry is not None and position is None:
            if pending_entry.model.formation_end_index > pending_entry.signal_index:
                formation_leakage_violations += 1
            position = _open_position(
                pending_entry,
                frame=frame,
                index=index,
                gross_notional_usd=gross_notional,
            )
            pending_entry = None
            entry_fills += 2

        if position is not None:
            exit_reason = ""
            if position.pending_exit_reason:
                exit_reason = position.pending_exit_reason
            elif index - position.entry_index >= int(
                exit_contract["max_holding_bars"]
            ):
                exit_reason = "max_holding"
            if exit_reason:
                trade, legs = _close_position(
                    position,
                    frame=frame,
                    exit_index=index,
                    exit_reason=exit_reason,
                    real_cost_bps=real_cost_bps,
                    stressed_cost_bps=stressed_cost_bps,
                )
                trades.append(trade)
                leg_trades.extend(legs)
                position = None
                exit_fills += 2
                last_exit_index = index

        refit_due = (
            index >= first_refit_index
            and (index - first_refit_index) % refit_bars == 0
        )
        if refit_due:
            formation_refits += 1
            current_model, audit = fit_best_cointegration_pair(
                candidate,
                index=index,
                series=series,
            )
            tested_pairs += int(audit["tested_pairs"])
            if current_model is not None:
                qualified_refits += 1
                selected_pair_refits[current_model.pair_id] = (
                    selected_pair_refits.get(current_model.pair_id, 0) + 1
                )
            if (
                position is not None
                and exit_contract["close_on_pair_reselection"] is True
                and (
                    current_model is None
                    or current_model.pair_id
                    != position.signal.model.pair_id
                )
                and index < end_index
            ):
                position.pending_exit_reason = "pair_reselected"

        if position is not None and not position.pending_exit_reason:
            zscore = model_zscore(
                position.signal.model,
                index=index,
                series=series,
            )
            if abs(zscore) <= float(exit_contract["exit_abs_zscore"]):
                if index < end_index:
                    position.pending_exit_reason = "mean_reversion"
            elif abs(zscore) >= float(exit_contract["stop_abs_zscore"]):
                if index < end_index:
                    position.pending_exit_reason = "zscore_stop"

        if (
            position is None
            and pending_entry is None
            and current_model is not None
            and index < end_index
        ):
            signal = model_entry_signal(
                candidate,
                model=current_model,
                index=index,
                series=series,
            )
            if signal is not None:
                candidate_signals += 1
                if (
                    index - last_exit_index
                    >= int(selection["cooldown_bars"])
                ):
                    pending_entry = signal
                    selected_signals += 1

    metrics = _cointegration_metrics(trades, net_field="net_bps")
    pair_counts: dict[str, int] = {}
    for trade in trades:
        pair_id = str(trade["pair_id"])
        pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
    max_pair_share = (
        max(pair_counts.values()) / len(trades) if trades else 0.0
    )
    return {
        "split": split_name,
        "start_timestamp_ms": frames[start_index].timestamp_ms,
        "end_timestamp_ms": frames[end_index].timestamp_ms,
        "bars": end_index - start_index + 1,
        "formation_refits": formation_refits,
        "qualified_refits": qualified_refits,
        "tested_pairs": tested_pairs,
        "selected_pair_refits": dict(sorted(selected_pair_refits.items())),
        "formation_leakage_violations": formation_leakage_violations,
        "paired_fill_atomicity_violations": paired_fill_atomicity_violations,
        "candidate_signals": candidate_signals,
        "selected_signals": selected_signals,
        "entry_fills": entry_fills,
        "exit_fills": exit_fills,
        "filled_orders": entry_fills + exit_fills,
        "closed_trades": len(trades),
        "remaining_open_positions": 2 if position is not None else 0,
        "right_censored_positions": 2 if position is not None else 0,
        "max_single_pair_trade_share": max_pair_share,
        "pair_trade_counts": dict(sorted(pair_counts.items())),
        "metrics": metrics,
        "cost_stress_metrics": _cointegration_metrics(
            trades,
            net_field="stressed_net_bps",
        ),
        "per_pair": _group_metrics(trades, "pair_id"),
        "per_direction": _group_metrics(trades, "direction"),
        "per_exit_reason": _group_metrics(trades, "exit_reason"),
        "per_symbol": _group_metrics(leg_trades, "symbol"),
        "trades": trades,
        "leg_trades": leg_trades,
    }


def fit_best_cointegration_pair(
    candidate: Mapping[str, Any],
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> tuple[CointegrationModel | None, dict[str, int]]:
    event = candidate["event_contract"]
    formation_bars = int(event["formation_bars"])
    start = index - formation_bars + 1
    symbols = sorted(series)
    tested = qualified = 0
    models: list[CointegrationModel] = []
    if start < 0:
        return None, {"tested_pairs": 0, "qualified_pairs": 0}
    for y_symbol, x_symbol in combinations(symbols, 2):
        tested += 1
        y = np.log(series[y_symbol].close[start:index + 1])
        x = np.log(series[x_symbol].close[start:index + 1])
        if (
            len(y) != formation_bars
            or np.any(~np.isfinite(y))
            or np.any(~np.isfinite(x))
        ):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pvalue, _ = coint(
                    y,
                    x,
                    trend=str(event["cointegration_trend"]),
                    autolag=str(event["adf_autolag"]),
                )
                regression = sm.OLS(y, sm.add_constant(x)).fit()
        except (ValueError, np.linalg.LinAlgError):
            continue
        alpha, beta = (float(value) for value in regression.params[:2])
        if (
            not math.isfinite(pvalue)
            or pvalue > float(event["max_cointegration_pvalue"])
            or not math.isfinite(beta)
            or beta <= 0.0
        ):
            continue
        spread = y - alpha - beta * x
        spread_std = float(np.std(spread, ddof=1))
        spread_mean = float(np.mean(spread))
        if not math.isfinite(spread_std) or spread_std <= 1e-8:
            continue
        half_life = _half_life_bars(spread)
        if (
            half_life is None
            or half_life < float(event["min_half_life_bars"])
            or half_life > float(event["max_half_life_bars"])
        ):
            continue
        qualified += 1
        models.append(
            CointegrationModel(
                y_symbol=y_symbol,
                x_symbol=x_symbol,
                formation_start_index=start,
                formation_end_index=index,
                alpha=alpha,
                beta=beta,
                spread_mean=spread_mean,
                spread_std=spread_std,
                pvalue=float(pvalue),
                half_life_bars=half_life,
            )
        )
    selected = (
        min(
            models,
            key=lambda row: (
                row.pvalue,
                row.half_life_bars,
                row.y_symbol,
                row.x_symbol,
            ),
        )
        if models
        else None
    )
    return selected, {"tested_pairs": tested, "qualified_pairs": qualified}


def model_zscore(
    model: CointegrationModel,
    *,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> float:
    y = math.log(float(series[model.y_symbol].close[index]))
    x = math.log(float(series[model.x_symbol].close[index]))
    spread = y - model.alpha - model.beta * x
    return (spread - model.spread_mean) / model.spread_std


def model_entry_signal(
    candidate: Mapping[str, Any],
    *,
    model: CointegrationModel,
    index: int,
    series: Mapping[str, SymbolSeries],
) -> CointegrationSignal | None:
    event = candidate["event_contract"]
    zscore = model_zscore(model, index=index, series=series)
    if abs(zscore) < float(event["entry_abs_zscore"]):
        return None
    exit_zscore = float(candidate["exit_contract"]["exit_abs_zscore"])
    estimated_reversion_bps = (
        max(0.0, abs(zscore) - exit_zscore)
        * model.spread_std
        * 10_000.0
    )
    if estimated_reversion_bps < float(
        event["min_estimated_reversion_bps"]
    ):
        return None
    return CointegrationSignal(
        model=model,
        direction="SHORT_SPREAD" if zscore > 0.0 else "LONG_SPREAD",
        signal_index=index,
        zscore=zscore,
        estimated_reversion_bps=estimated_reversion_bps,
    )


def write_cointegration_development_report(
    report: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "cointegration_spread_development.json"
    markdown_path = target / "cointegration_spread_development.md"
    compact = _compact_report(report)
    json_path.write_text(
        json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _open_position(
    signal: CointegrationSignal,
    *,
    frame: MarketFrame,
    index: int,
    gross_notional_usd: float,
) -> CointegrationPosition:
    beta = abs(signal.model.beta)
    y_notional = gross_notional_usd / (1.0 + beta)
    x_notional = gross_notional_usd - y_notional
    return CointegrationPosition(
        signal=signal,
        entry_index=index,
        entry_timestamp_ms=frame.timestamp_ms,
        y_entry_price=frame.bars[signal.model.y_symbol].open,
        x_entry_price=frame.bars[signal.model.x_symbol].open,
        y_notional_usd=y_notional,
        x_notional_usd=x_notional,
    )


def _close_position(
    position: CointegrationPosition,
    *,
    frame: MarketFrame,
    exit_index: int,
    exit_reason: str,
    real_cost_bps: float,
    stressed_cost_bps: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = position.signal.model
    y_direction = (
        "SHORT"
        if position.signal.direction == "SHORT_SPREAD"
        else "LONG"
    )
    x_direction = "LONG" if y_direction == "SHORT" else "SHORT"
    y_exit = frame.bars[model.y_symbol].open
    x_exit = frame.bars[model.x_symbol].open
    y_gross_bps = _leg_return_bps(
        direction=y_direction,
        entry_price=position.y_entry_price,
        exit_price=y_exit,
    )
    x_gross_bps = _leg_return_bps(
        direction=x_direction,
        entry_price=position.x_entry_price,
        exit_price=x_exit,
    )
    total_notional = position.y_notional_usd + position.x_notional_usd
    gross_pnl = (
        position.y_notional_usd * y_gross_bps
        + position.x_notional_usd * x_gross_bps
    ) / 10_000.0
    net_pnl = gross_pnl - total_notional * real_cost_bps / 10_000.0
    stressed_net_pnl = (
        gross_pnl - total_notional * stressed_cost_bps / 10_000.0
    )
    gross_bps = gross_pnl / total_notional * 10_000.0
    net_bps = net_pnl / total_notional * 10_000.0
    stressed_net_bps = stressed_net_pnl / total_notional * 10_000.0
    common = {
        "pair_id": model.pair_id,
        "direction": position.signal.direction,
        "regime": "cointegration_spread",
        "signal_index": position.signal.signal_index,
        "entry_index": position.entry_index,
        "exit_index": exit_index,
        "entry_timestamp_ms": position.entry_timestamp_ms,
        "exit_timestamp_ms": frame.timestamp_ms,
        "holding_bars": exit_index - position.entry_index,
        "exit_reason": exit_reason,
        "formation_start_index": model.formation_start_index,
        "formation_end_index": model.formation_end_index,
        "entry_zscore": position.signal.zscore,
        "estimated_reversion_bps": position.signal.estimated_reversion_bps,
        "cointegration_pvalue": model.pvalue,
        "half_life_bars": model.half_life_bars,
        "hedge_beta": model.beta,
        "funding_bps": 0.0,
    }
    trade = {
        **common,
        "symbol": model.pair_id,
        "gross_bps": gross_bps,
        "cost_bps": real_cost_bps,
        "stressed_cost_bps": stressed_cost_bps,
        "net_bps": net_bps,
        "stressed_net_bps": stressed_net_bps,
        "net_pnl_usd": net_pnl,
        "stressed_net_pnl_usd": stressed_net_pnl,
    }
    legs = [
        _leg_trade(
            common,
            symbol=model.y_symbol,
            direction=y_direction,
            gross_bps=y_gross_bps,
            notional_usd=position.y_notional_usd,
            real_cost_bps=real_cost_bps,
            stressed_cost_bps=stressed_cost_bps,
        ),
        _leg_trade(
            common,
            symbol=model.x_symbol,
            direction=x_direction,
            gross_bps=x_gross_bps,
            notional_usd=position.x_notional_usd,
            real_cost_bps=real_cost_bps,
            stressed_cost_bps=stressed_cost_bps,
        ),
    ]
    return trade, legs


def _leg_trade(
    common: Mapping[str, Any],
    *,
    symbol: str,
    direction: str,
    gross_bps: float,
    notional_usd: float,
    real_cost_bps: float,
    stressed_cost_bps: float,
) -> dict[str, Any]:
    net_bps = gross_bps - real_cost_bps
    stressed_net_bps = gross_bps - stressed_cost_bps
    return {
        **common,
        "symbol": symbol,
        "direction": direction,
        "gross_bps": gross_bps,
        "cost_bps": real_cost_bps,
        "stressed_cost_bps": stressed_cost_bps,
        "net_bps": net_bps,
        "stressed_net_bps": stressed_net_bps,
        "net_pnl_usd": notional_usd * net_bps / 10_000.0,
        "stressed_net_pnl_usd": (
            notional_usd * stressed_net_bps / 10_000.0
        ),
    }


def _leg_return_bps(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
) -> float:
    if direction == "LONG":
        return (exit_price / max(entry_price, 1e-12) - 1.0) * 10_000.0
    return (entry_price / max(exit_price, 1e-12) - 1.0) * 10_000.0


def _half_life_bars(spread: np.ndarray) -> float | None:
    lagged = spread[:-1]
    delta = np.diff(spread)
    try:
        regression = sm.OLS(delta, sm.add_constant(lagged)).fit()
    except (ValueError, np.linalg.LinAlgError):
        return None
    coefficient = float(regression.params[1])
    if not math.isfinite(coefficient) or coefficient >= 0.0:
        return None
    half_life = -math.log(2.0) / coefficient
    return half_life if math.isfinite(half_life) else None


def _cointegration_metrics(
    trades: Sequence[Mapping[str, Any]],
    *,
    net_field: str,
) -> dict[str, Any]:
    result = _metrics(trades, net_field=net_field)
    result["fills"] = len(trades) * 4
    return result


def _development_verdict(
    candidate: Mapping[str, Any],
    *,
    split: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = split["metrics"]
    stress = split["cost_stress_metrics"]
    failures: list[str] = []
    if int(split["filled_orders"]) < int(gates["min_filled_orders"]):
        failures.append("filled_orders_below_minimum")
    if int(split["closed_trades"]) < int(gates["min_closed_trades"]):
        failures.append("closed_trades_below_minimum")
    if int(split["formation_leakage_violations"]) != 0:
        failures.append("formation_leakage")
    if int(split["paired_fill_atomicity_violations"]) != 0:
        failures.append("paired_fill_atomicity_violation")
    if (metrics["mean_net_bps"] or -math.inf) <= float(
        gates["min_mean_net_bps"]
    ):
        failures.append("nonpositive_costed_expectancy")
    if (metrics["lcb_95_net_bps"] or -math.inf) <= float(
        gates["min_lcb_95_net_bps"]
    ):
        failures.append("nonpositive_lcb")
    if float(metrics["max_drawdown_usd"]) > float(
        gates["max_drawdown_usd"]
    ):
        failures.append("max_drawdown_exceeded")
    if (stress["mean_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_expectancy")
    if (stress["lcb_95_net_bps"] or -math.inf) <= 0.0:
        failures.append("cost_stress_nonpositive_lcb")
    for direction in candidate["hypothesis"]["directions"]:
        row = split["per_direction"].get(direction)
        if (
            row is None
            or int(row["trades"])
            < int(gates["min_spread_direction_trades"])
            or (row["mean_net_bps"] or -math.inf) <= 0.0
            or (row["lcb_95_net_bps"] or -math.inf) <= 0.0
        ):
            failures.append(f"direction_collapse:{direction}")
    if float(split["max_single_pair_trade_share"]) > float(
        gates["max_single_pair_trade_share"]
    ):
        failures.append("single_pair_concentration")
    beats_baseline = (
        (metrics["mean_net_bps"] or -math.inf) > 0.0
        and float(metrics["net_sum_bps"]) > 0.0
    )
    if not beats_baseline:
        failures.append("does_not_beat_cash_baseline")
    unique_failures = sorted(set(failures))
    passed = not unique_failures
    return {
        "verdict": (
            "accepted_for_sealed_validation"
            if passed
            else "terminal_rejected_development"
        ),
        "passed": passed,
        "failures": unique_failures,
        "baseline_comparison": {
            "baseline_id": gates["baseline_id"],
            "candidate_mean_net_bps": metrics["mean_net_bps"],
            "baseline_mean_net_bps": 0.0,
            "candidate_net_sum_bps": metrics["net_sum_bps"],
            "baseline_net_sum_bps": 0.0,
            "beats_baseline": beats_baseline,
        },
        "continuation_allowed": passed,
    }


def _cash_baseline() -> dict[str, Any]:
    return {
        "baseline_id": "cash_no_trade_v1",
        "metrics": {
            "trades": 0,
            "fills": 0,
            "mean_gross_bps": 0.0,
            "mean_net_bps": 0.0,
            "lcb_95_net_bps": 0.0,
            "gross_sum_bps": 0.0,
            "net_sum_bps": 0.0,
            "net_pnl_usd": 0.0,
            "max_drawdown_usd": 0.0,
        },
    }


def _compact_report(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _compact_report(item)
            for key, item in value.items()
            if not (
                key in {"trades", "leg_trades"}
                and isinstance(item, list)
            )
        }
    if isinstance(value, list):
        return [_compact_report(item) for item in value]
    return value


def _render_markdown(report: Mapping[str, Any]) -> str:
    candidate = report["candidate"]
    metrics = candidate["metrics"]
    stress = candidate["cost_stress_metrics"]
    lines = [
        "# Walk-forward cointegration spread 4h development screen",
        "",
        "## Safety",
        "",
        "- Stage: `development_only`",
        "- Development run budget: `1`",
        "- Parameter sweep allowed: `false`",
        "- Symbol allowlist: `none`",
        "- Validation/OOS/sanity opened: `false`",
        "- Runtime actor created: `false`",
        "- Paper/live allowed: `false`",
        "",
        "## Verdict",
        "",
        f"- Result: `{report['verdict']}`",
        f"- Continuation allowed: `{str(report['continuation_allowed']).lower()}`",
        "- Failures: "
        + (", ".join(f"`{item}`" for item in report["failures"]) or "none"),
        "",
        "## Metrics",
        "",
        "| Signals | Fills | Closed | Mean bps | LCB bps | Stress mean | DD USD |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {candidate['candidate_signals']} | "
        f"{candidate['filled_orders']} | {candidate['closed_trades']} | "
        f"{_fmt(metrics['mean_net_bps'])} | "
        f"{_fmt(metrics['lcb_95_net_bps'])} | "
        f"{_fmt(stress['mean_net_bps'])} | "
        f"{_fmt(metrics['max_drawdown_usd'])} |",
        "",
        "## Formation",
        "",
        f"- Refits: `{candidate['formation_refits']}`",
        f"- Qualified refits: `{candidate['qualified_refits']}`",
        f"- Tested pairs: `{candidate['tested_pairs']}`",
        f"- Leakage violations: `{candidate['formation_leakage_violations']}`",
        f"- Maximum single-pair trade share: "
        f"`{candidate['max_single_pair_trade_share']:.2%}`",
        "",
        "## Direction",
        "",
        "| Direction | Trades | Mean net bps | LCB bps |",
        "|---|---:|---:|---:|",
    ]
    for direction, row in candidate["per_direction"].items():
        lines.append(
            f"| {direction} | {row['trades']} | "
            f"{_fmt(row['mean_net_bps'])} | "
            f"{_fmt(row['lcb_95_net_bps'])} |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
