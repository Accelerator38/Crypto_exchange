from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .measurement import BootstrapBank
from .fitness_recency import score_recency_fitness
from .simulator import PrimusSimulation
from .unified_measurement_v1 import build_ranking_key


@dataclass(frozen=True)
class V4Score:
    fitness: float
    ranking_key: tuple[float, ...]
    metrics: Mapping[str, Any]
    folds: pd.DataFrame
    failed_gates: tuple[str, ...]


def _seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def contiguous_block_lcb(
    values: Sequence[float],
    *,
    samples: int,
    alpha: float,
    block_size: int,
    seed: int,
    resample_indices: np.ndarray | None = None,
) -> float | None:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if len(data) < max(2, block_size):
        return None
    if resample_indices is None:
        rng = np.random.default_rng(seed)
        starts = np.arange(0, len(data) - block_size + 1)
        blocks_needed = int(np.ceil(len(data) / block_size))
        selected = rng.choice(starts, size=(int(samples), blocks_needed), replace=True)
        offsets = np.arange(block_size)[None, None, :]
        indices = selected[:, :, None] + offsets
        indices = indices.reshape(int(samples), -1)[:, : len(data)]
    else:
        indices = np.asarray(resample_indices, dtype=np.int64)
        if indices.shape != (int(samples), len(data)):
            raise ValueError("bootstrap resample bank shape mismatch")
        if indices.min(initial=0) < 0 or indices.max(initial=0) >= len(data):
            raise ValueError("bootstrap resample bank index out of bounds")
    sampled = data[indices]
    return float(np.quantile(sampled.mean(axis=1), float(alpha)))


def daily_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    if portfolio.empty:
        return pd.DataFrame(columns=["timestamp", "net_return", "net_bps"])
    local = portfolio.loc[:, ["timestamp", "net_return"]].copy()
    local["day"] = pd.to_datetime(local["timestamp"], unit="ms", utc=True).dt.floor("1D")
    daily = (
        local.groupby("day", sort=True)["net_return"]
        .apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
        .rename("net_return")
        .reset_index()
    )
    # Pandas may store timezone-aware datetimes at ms, us or ns resolution.
    daily["timestamp"] = daily["day"].map(lambda value: int(value.timestamp() * 1000)).astype("int64")
    daily["net_bps"] = daily["net_return"] * 10_000.0
    return daily.loc[:, ["timestamp", "net_return", "net_bps"]]


def _max_share(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    counts = values.value_counts()
    return float(counts.max() / counts.sum())


def _opportunity_normalized_concentration(
    opportunities: Mapping[str, int],
    allowed: Mapping[str, int],
    *,
    minimum_bucket_opportunities: int,
) -> float:
    total_opportunities = int(sum(opportunities.values()))
    total_allowed = int(sum(allowed.values()))
    if total_opportunities <= 0 or total_allowed <= 0:
        return 0.0
    ratios = [
        (float(allowed.get(key, 0)) / total_allowed)
        / (float(count) / total_opportunities)
        for key, count in opportunities.items()
        if int(count) >= int(minimum_bucket_opportunities)
    ]
    return float(max(ratios, default=0.0))


def score_simulation_v4(
    simulation: PrimusSimulation,
    *,
    folds: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
    key: str,
    bootstrap_bank: BootstrapBank | None = None,
    complexity_metrics: Mapping[str, Any] | None = None,
    paired_reference_portfolios: Mapping[str, pd.DataFrame] | None = None,
) -> V4Score:
    if "search_objective" in config:
        from .profit_fitness_v1 import score_profit_simulation
        score = score_profit_simulation(
            simulation, config=config, folds=folds, bootstrap_bank=bootstrap_bank,
            complexity_metrics=complexity_metrics,
        )
        return V4Score(
            fitness=score["fitness"], ranking_key=score["ranking_key"],
            metrics=score["metrics"], folds=score["folds"], failed_gates=score["failed_gates"],
        )
    daily = daily_portfolio(simulation.portfolio)
    samples = int(config["bootstrap_samples"])
    alpha = float(config["bootstrap_alpha"])
    block_size = int(config["bootstrap_block_days"])
    if bootstrap_bank is not None and (
        bootstrap_bank.samples != samples or bootstrap_bank.block_days != block_size
    ):
        raise ValueError("bootstrap bank does not match fitness configuration")
    overall_values = daily["net_bps"].to_numpy(dtype=float)
    overall_mean = float(np.mean(overall_values)) if len(overall_values) else None
    overall_median = float(np.median(overall_values)) if len(overall_values) else None
    overall_lcb = contiguous_block_lcb(
        overall_values,
        samples=samples,
        alpha=alpha,
        block_size=block_size,
        seed=_seed(f"{key}:overall"),
        resample_indices=(
            bootstrap_bank.indices_for("overall", daily["timestamp"].to_numpy(np.int64))
            if bootstrap_bank is not None
            else None
        ),
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        start = int(pd.Timestamp(fold["start"]).timestamp() * 1000)
        end = int(pd.Timestamp(fold["end"]).timestamp() * 1000)
        subset = daily.loc[(daily["timestamp"] >= start) & (daily["timestamp"] < end)]
        values = subset["net_bps"].to_numpy(dtype=float)
        ledger = simulation.ledger.loc[
            (simulation.ledger["entry_timestamp"] >= start)
            & (simulation.ledger["entry_timestamp"] < end)
        ]
        fold_rows.append(
            {
                "fold": str(fold["name"]),
                "start": str(fold["start"]),
                "end": str(fold["end"]),
                "days": int(len(values)),
                "trades": int(len(ledger)),
                "mean_daily_bps": float(np.mean(values)) if len(values) else None,
                "median_daily_bps": float(np.median(values)) if len(values) else None,
                "lcb_daily_bps": contiguous_block_lcb(
                    values,
                    samples=samples,
                    alpha=alpha,
                    block_size=block_size,
                    seed=_seed(f"{key}:{fold['name']}"),
                    resample_indices=(
                        bootstrap_bank.indices_for(
                            str(fold["name"]), subset["timestamp"].to_numpy(np.int64)
                        )
                        if bootstrap_bank is not None
                        else None
                    ),
                ),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    finite_lcb = fold_frame["lcb_daily_bps"].dropna().astype(float)
    fold_means = fold_frame["mean_daily_bps"].dropna().astype(float)
    positive_fold_means = int((fold_means > 0.0).sum())
    worst_fold_mean = float(fold_means.min()) if len(fold_means) else -1.0e9
    median_fold_lcb = float(finite_lcb.median()) if len(finite_lcb) else -1.0e9
    worst_fold_lcb = float(finite_lcb.min()) if len(finite_lcb) else -1.0e9
    max_drawdown = float(-simulation.portfolio["drawdown"].min()) if not simulation.portfolio.empty else 0.0
    days = max(1, len(daily))
    fills_per_day = float(simulation.turnover_fills / days)
    ledger = simulation.ledger
    max_direction_share = _max_share(ledger["direction"]) if not ledger.empty else 0.0
    max_state_share = _max_share(ledger["market_state"]) if not ledger.empty else 0.0
    max_symbol_share = _max_share(ledger["symbol"]) if not ledger.empty else 0.0
    failures: list[str] = []
    if len(ledger) < int(config["minimum_trades"]):
        failures.append("minimum_trades")
    if max_drawdown > float(config["max_drawdown"]):
        failures.append("max_drawdown")
    if fills_per_day > float(config["max_turnover_fills_per_day"]):
        failures.append("turnover")
    if max_direction_share > float(config["max_direction_share"]):
        failures.append("direction_collapse")
    if max_state_share > float(config["max_state_share"]):
        failures.append("regime_collapse")
    if max_symbol_share > float(config["max_symbol_share"]):
        failures.append("symbol_collapse")
    if overall_mean is None or overall_mean <= 0.0:
        failures.append("nonpositive_overall_mean")
    if overall_lcb is None or overall_lcb <= 0.0:
        failures.append("nonpositive_overall_lcb")
    if positive_fold_means < int(config["minimum_positive_fold_means"]):
        failures.append("insufficient_positive_folds")
    if worst_fold_mean <= 0.0:
        failures.append("nonpositive_worst_fold_mean")
    if median_fold_lcb <= 0.0:
        failures.append("nonpositive_median_fold_lcb")
    temporal_pass = (
        positive_fold_means >= int(config["minimum_positive_fold_means"])
        and worst_fold_mean > 0.0
        and median_fold_lcb > 0.0
    )
    turnover_penalty = float(config["turnover_penalty_bps_per_fill_per_day"]) * fills_per_day
    fitness = min(worst_fold_mean, median_fold_lcb) - turnover_penalty
    metrics = {
        "trades": int(len(ledger)),
        "turnover_fills": int(simulation.turnover_fills),
        "fills_per_day": fills_per_day,
        "days": int(len(daily)),
        "mean_daily_bps": overall_mean,
        "median_daily_bps": overall_median,
        "lcb_daily_bps": overall_lcb,
        "max_drawdown": max_drawdown,
        "positive_fold_means": positive_fold_means,
        "fold_count": int(len(fold_frame)),
        "worst_fold_mean_bps": worst_fold_mean,
        "median_fold_lcb_bps": median_fold_lcb,
        "worst_fold_lcb_bps": worst_fold_lcb,
        "max_direction_share": max_direction_share,
        "max_state_share": max_state_share,
        "max_symbol_share": max_symbol_share,
        "temporal_gate_passed": temporal_pass,
        "fitness": fitness,
        "failed_gates": failures,
    }
    recency_config = config.get("recency")
    if recency_config is not None:
        if bootstrap_bank is None:
            raise ValueError("recency fitness requires a materialized common bootstrap bank")
        recency = score_recency_fitness(
            daily,
            simulation,
            fold_lcbs=fold_frame["lcb_daily_bps"].to_list(),
            config=recency_config,
            bootstrap_bank=bootstrap_bank,
            alpha=alpha,
            complexity_parameter_count=int(
                (complexity_metrics or {}).get(
                    "parameter_count", config.get("complexity_parameter_count", 0)
                )
            ),
            active_parameter_count=(complexity_metrics or {}).get("active_parameter_count"),
            active_feature_count=(complexity_metrics or {}).get("active_feature_count"),
            total_feature_count=(complexity_metrics or {}).get("feature_count"),
        )
        failures.extend(recency.failed_gates)
        fitness = recency.fitness
        metrics.update(recency.metrics)
        metrics["fitness"] = fitness
    constrained_overlay = False
    coverage_failure = 0
    allowed_fraction = 0.0
    coverage_penalty = 0.0
    abstention_config = config.get("abstention")
    if abstention_config is not None:
        opportunities = int(simulation.diagnostics.get("overlay_opportunities", 0))
        allowed = int(simulation.diagnostics.get("overlay_allowed", 0))
        allowed_fraction = float(allowed / opportunities) if opportunities else 0.0
        minimum_fraction = float(abstention_config.get("minimum_allowed_fraction", 0.0))
        maximum_fraction = float(abstention_config["maximum_allowed_fraction"])
        constrained_overlay = "minimum_allowed_fraction" in abstention_config
        shortfall = max(0.0, minimum_fraction - allowed_fraction)
        excess = max(0.0, allowed_fraction - maximum_fraction)
        penalty_rate = float(
            abstention_config.get(
                "coverage_penalty_bps",
                abstention_config.get("excess_coverage_penalty_bps", 0.0),
            )
        )
        coverage_penalty = (shortfall + excess) * penalty_rate
        coverage_failure = int(shortfall > 0.0 or excess > 0.0)
        if shortfall > 0.0:
            failures.append("overlay_action_coverage_below_minimum")
        if excess > 0.0:
            failures.append("overlay_action_coverage")
        fitness -= coverage_penalty
        metrics.update(
            {
                "overlay_opportunities": opportunities,
                "overlay_allowed": allowed,
                "overlay_abstained": int(
                    simulation.diagnostics.get("overlay_abstained", 0)
                ),
                "overlay_allowed_fraction": allowed_fraction,
                "overlay_minimum_allowed_fraction": minimum_fraction,
                "overlay_maximum_allowed_fraction": maximum_fraction,
                "overlay_coverage_penalty_bps": coverage_penalty,
            }
        )
        evidence_config = config.get("abstention_evidence", {})
        if evidence_config:
            if ledger.empty:
                trade_days = 0
            else:
                trade_days = int(
                    pd.to_datetime(ledger["entry_timestamp"], unit="ms", utc=True)
                    .dt.floor("D")
                    .nunique()
                )
            metrics.update(
                {
                    "overlay_trade_days": trade_days,
                    "minimum_overlay_opportunities": int(
                        evidence_config["minimum_opportunities"]
                    ),
                    "minimum_overlay_trade_days": int(
                        evidence_config["minimum_trade_days"]
                    ),
                }
            )
            if opportunities < int(evidence_config["minimum_opportunities"]):
                failures.append("minimum_opportunities")
            if trade_days < int(evidence_config["minimum_trade_days"]):
                failures.append("minimum_trade_days")
        concentration_config = config.get("opportunity_normalized_concentration", {})
        if concentration_config:
            maximum_lift = float(concentration_config["maximum_allocation_lift"])
            minimum_bucket = int(concentration_config["minimum_bucket_opportunities"])
            penalty_rate = float(concentration_config["penalty_bps_per_excess_lift"])
            concentration_penalty = 0.0
            for dimension in ("direction", "state", "symbol"):
                lift = _opportunity_normalized_concentration(
                    simulation.diagnostics.get(
                        f"overlay_opportunities_by_{dimension}", {}
                    ),
                    simulation.diagnostics.get(f"overlay_allowed_by_{dimension}", {}),
                    minimum_bucket_opportunities=minimum_bucket,
                )
                metrics[f"opportunity_normalized_{dimension}_lift"] = lift
                excess = max(0.0, lift - maximum_lift)
                concentration_penalty += excess * penalty_rate
                if excess > 0.0:
                    failures.append(f"opportunity_normalized_{dimension}_concentration")
            metrics["opportunity_normalized_concentration_penalty_bps"] = (
                concentration_penalty
            )
            fitness -= concentration_penalty
    paired_config = config.get("paired_incremental")
    paired_delta_standard_errors: list[float] = []
    if paired_config is not None:
        if bootstrap_bank is None:
            raise ValueError("paired incremental fitness requires a common bootstrap bank")
        references = paired_reference_portfolios or {}
        paired_lcbs: dict[str, float] = {}
        paired_penalty = 0.0
        minimum_lcb = float("inf")
        threshold = float(paired_config["minimum_lcb_bps"])
        materialized: dict[str, str] = {}
        reference_aliases: dict[str, str] = {}
        for reference_name in paired_config["references"]:
            if reference_name == "Incumbent" and reference_name not in references:
                continue
            if reference_name not in references:
                raise ValueError(f"missing paired fitness reference: {reference_name}")
            reference_daily = daily_portfolio(references[reference_name]).set_index("timestamp")
            reference_timestamps = reference_daily.index.to_numpy(np.int64)
            candidate_timestamps = daily["timestamp"].to_numpy(np.int64)
            if not np.array_equal(reference_timestamps, candidate_timestamps):
                raise ValueError(f"paired reference calendar mismatch: {reference_name}")
            reference_values = reference_daily["net_bps"].to_numpy(float)
            reference_sha = hashlib.sha256(
                candidate_timestamps.tobytes() + reference_values.astype(np.float64).tobytes()
            ).hexdigest()
            if reference_sha in materialized:
                reference_aliases[str(reference_name)] = materialized[reference_sha]
                continue
            materialized[reference_sha] = str(reference_name)
            delta = daily["net_bps"].to_numpy(float) - reference_values
            lcb = contiguous_block_lcb(
                delta,
                samples=samples,
                alpha=alpha,
                block_size=block_size,
                seed=_seed(f"{key}:paired:{reference_name}"),
                resample_indices=bootstrap_bank.indices_for(
                    "overall", daily["timestamp"].to_numpy(np.int64)
                ),
            )
            value = float(lcb if lcb is not None else -1.0e9)
            paired_lcbs[str(reference_name)] = value
            minimum_lcb = min(minimum_lcb, value)
            paired_delta_standard_errors.append(
                float(np.std(delta, ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
            )
            shortfall = max(0.0, threshold - value)
            paired_penalty += shortfall * float(
                paired_config.get("negative_lcb_penalty_multiplier", 0.0)
            )
            if value <= threshold:
                failures.append(f"nonpositive_paired_lcb:{reference_name}")
        if not paired_lcbs:
            raise ValueError("paired incremental fitness has no materialized references")
        fitness -= paired_penalty
        metrics["paired_incremental_lcb_bps"] = paired_lcbs
        metrics["paired_reference_aliases"] = reference_aliases
        metrics["paired_incremental_min_lcb_bps"] = minimum_lcb
        metrics["paired_incremental_penalty_bps"] = paired_penalty
    multiple_testing = config.get("adaptive_multiple_testing")
    if multiple_testing is not None:
        hypotheses = int(multiple_testing["max_hypotheses_per_origin"])
        if hypotheses <= 1:
            raise ValueError("adaptive hypothesis budget must exceed one")
        standard_error = (
            max(paired_delta_standard_errors)
            if paired_delta_standard_errors
            else float(np.std(overall_values, ddof=1) / np.sqrt(len(overall_values)))
            if len(overall_values) > 1
            else 0.0
        )
        selection_bias_penalty = (
            standard_error
            * float(np.sqrt(2.0 * np.log(hypotheses)))
            * float(multiple_testing["selection_bias_penalty_multiplier"])
        )
        if not constrained_overlay:
            fitness -= selection_bias_penalty
        metrics.update(
            {
                "adaptive_hypothesis_budget": hypotheses,
                "adaptive_standard_error_bps": standard_error,
                "adaptive_selection_bias_penalty_bps": selection_bias_penalty,
            }
        )
    else:
        selection_bias_penalty = 0.0
    if constrained_overlay and paired_config is not None:
        adjusted_lcbs = {
            name: float(value - selection_bias_penalty)
            for name, value in metrics["paired_incremental_lcb_bps"].items()
        }
        adjusted_minimum = min(adjusted_lcbs.values())
        absolute_name = str(paired_config["absolute_reference"])
        baseline_name = str(paired_config["baseline_reference"])
        absolute_lcb = adjusted_lcbs.get(absolute_name, -1.0e9)
        baseline_lcb = adjusted_lcbs.get(baseline_name, -1.0e9)
        fitness += adjusted_minimum
        metrics.update(
            {
                "paired_incremental_adjusted_lcb_bps": adjusted_lcbs,
                "paired_objective_lcb_bps": adjusted_minimum,
                "paired_absolute_lcb_bps": absolute_lcb,
                "paired_baseline_lcb_bps": baseline_lcb,
                "coverage_constraint_passed": coverage_failure == 0,
            }
        )
    ranking_key = build_ranking_key(
        fitness=fitness,
        failed_gates=failures,
        max_drawdown=max_drawdown,
        fills_per_day=fills_per_day,
        active_parameter_count=metrics.get("recency_active_parameter_count", 0.0),
    )
    metrics["ranking_semantics"] = "all_gates_then_penalized_fitness/1"
    metrics["fitness"] = fitness
    metrics["failed_gates"] = failures
    return V4Score(
        fitness=float(fitness),
        ranking_key=ranking_key,
        metrics=metrics,
        folds=fold_frame,
        failed_gates=tuple(failures),
    )
