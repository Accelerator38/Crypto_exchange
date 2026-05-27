"""Validation helpers for genetic-agent research runs.

These functions are intentionally pure and lightweight so tests can cover the
walk-forward contract without importing the heavy training module.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

FITNESS_V3_REGIMES = ("crash", "bearish", "neutral", "bullish")
FITNESS_V4_REGIMES = FITNESS_V3_REGIMES


def _period_year(period: str) -> int:
    return int(str(period)[:4])


def _period_month(period: str) -> int:
    parts = str(period).split("-", 1)
    if len(parts) < 2:
        return 1
    try:
        return int(parts[1][:2])
    except ValueError:
        return 1


def build_fixed_walk_forward_contract(
    periods: Sequence[str],
    *,
    train_years: tuple[int, int] = (2022, 2023),
    validation_year: int = 2024,
    oos_year: int = 2025,
    final_sanity_year: int = 2026,
    final_sanity_max_month: int = 6,
    required_regimes: Sequence[str] = FITNESS_V3_REGIMES,
) -> dict[str, Any]:
    """Return the fixed genetics v3 split contract.

    This is intentionally deterministic: training is 2022-2023, validation is
    2024, OOS is 2025, and final sanity is 2026 H1 by default.
    """

    unique_periods = sorted(dict.fromkeys(str(period) for period in periods))
    train_start, train_end = train_years
    if train_start > train_end:
        raise ValueError("train_years start must be <= end")
    if final_sanity_max_month <= 0:
        raise ValueError("final_sanity_max_month must be > 0")

    train_periods = [
        period for period in unique_periods
        if train_start <= _period_year(period) <= train_end
    ]
    validation_periods = [
        period for period in unique_periods if _period_year(period) == validation_year
    ]
    oos_periods = [
        period for period in unique_periods if _period_year(period) == oos_year
    ]
    final_sanity_periods = [
        period for period in unique_periods
        if _period_year(period) == final_sanity_year
        and _period_month(period) <= final_sanity_max_month
    ]
    return {
        "schema_version": 1,
        "train_years": [int(train_start), int(train_end)],
        "validation_year": int(validation_year),
        "oos_year": int(oos_year),
        "final_sanity_year": int(final_sanity_year),
        "final_sanity_max_month": int(final_sanity_max_month),
        "train_periods": train_periods,
        "validation_periods": validation_periods,
        "oos_periods": oos_periods,
        "final_sanity_periods": final_sanity_periods,
        "regime_balance_required": [str(regime) for regime in required_regimes],
    }


def build_rolling_year_folds(
    periods: Sequence[str],
    *,
    train_years: int = 4,
    validation_years: int = 1,
    final_test_year: int | None = None,
    embargo_months: int = 0,
) -> list[dict[str, Any]]:
    """Build rolling train/validation folds from YYYY-MM period labels."""
    unique_periods = sorted(dict.fromkeys(str(period) for period in periods))
    if train_years <= 0:
        raise ValueError("train_years must be > 0")
    if validation_years <= 0:
        raise ValueError("validation_years must be > 0")
    if embargo_months < 0:
        raise ValueError("embargo_months must be >= 0")
    if not unique_periods:
        return []

    min_year = _period_year(unique_periods[0])
    max_year = _period_year(unique_periods[-1])
    last_validation_year = max_year if final_test_year is None else final_test_year - 1
    folds: list[dict[str, Any]] = []

    train_start = min_year
    while True:
        train_end = train_start + train_years - 1
        validation_start = train_end + 1
        validation_end = validation_start + validation_years - 1
        if validation_end > last_validation_year:
            break

        train_periods = [
            period for period in unique_periods
            if train_start <= _period_year(period) <= train_end
        ]
        validation_periods = [
            period for period in unique_periods
            if validation_start <= _period_year(period) <= validation_end
        ]
        if embargo_months:
            embargo_periods = train_periods[-embargo_months:]
            train_periods = train_periods[:-embargo_months]
        else:
            embargo_periods = []

        if train_periods and validation_periods:
            validation_label = (
                str(validation_start)
                if validation_start == validation_end
                else f"{validation_start}_{validation_end}"
            )
            folds.append({
                "fold_id": (
                    f"train_{train_start}_{train_end}_"
                    f"validate_{validation_label}"
                ),
                "train_years": [train_start, train_end],
                "validation_years": [validation_start, validation_end],
                "train_periods": train_periods,
                "embargo_periods": embargo_periods,
                "validation_periods": validation_periods,
            })
        train_start += 1

    return folds


def split_precomp_by_periods(precomp: Sequence[Any], periods: Iterable[str]) -> list[Any]:
    wanted = {str(period) for period in periods}
    return [entry for entry in precomp if len(entry) > 4 and str(entry[4]) in wanted]


def robust_period_score(
    period_rets: Sequence[float],
    *,
    trim_fraction: float = 0.10,
    max_single_period_contribution_pct: float = 30.0,
    min_positive_period_pct: float = 55.0,
    min_median_ret: float = 0.0,
    min_trimmed_mean_ret: float = 0.0,
) -> dict[str, Any]:
    """Return robust diagnostics and default pass/fail gates for period returns."""
    arr = np.asarray(period_rets, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n_periods": 0,
            "passes_default_gates": False,
            "failed_gates": ["no_periods"],
        }

    sorted_arr = np.sort(arr)
    trim_n = int(math.floor(arr.size * max(0.0, min(trim_fraction, 0.45))))
    trimmed = sorted_arr[trim_n: arr.size - trim_n] if trim_n > 0 else sorted_arr
    if trimmed.size == 0:
        trimmed = sorted_arr

    positives = arr[arr > 0.0]
    positive_sum = float(positives.sum())
    max_positive_contribution = (
        float(positives.max() / positive_sum * 100.0)
        if positive_sum > 0.0 and positives.size
        else 0.0
    )
    tail_n = max(1, int(math.ceil(arr.size * 0.05)))
    cvar_5 = float(sorted_arr[:tail_n].mean())
    downside = np.minimum(arr, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))

    result: dict[str, Any] = {
        "n_periods": int(arr.size),
        "mean_ret": float(arr.mean()),
        "median_ret": float(np.median(arr)),
        "trimmed_mean_ret": float(trimmed.mean()),
        "std_ret": float(arr.std()),
        "min_ret": float(arr.min()),
        "max_ret": float(arr.max()),
        "cvar_5_ret": cvar_5,
        "downside_deviation": downside_deviation,
        "positive_period_pct": float((arr > 0.0).mean() * 100.0),
        "max_positive_contribution_pct": max_positive_contribution,
    }

    utility = (
        0.35 * result["median_ret"]
        + 0.35 * result["trimmed_mean_ret"]
        + 0.20 * result["mean_ret"]
        + 0.10 * result["cvar_5_ret"]
        - 0.10 * result["downside_deviation"]
    )
    result["robust_utility"] = float(utility)

    failed: list[str] = []
    if result["median_ret"] <= min_median_ret:
        failed.append("median_ret")
    if result["trimmed_mean_ret"] <= min_trimmed_mean_ret:
        failed.append("trimmed_mean_ret")
    if result["positive_period_pct"] < min_positive_period_pct:
        failed.append("positive_period_pct")
    if result["max_positive_contribution_pct"] > max_single_period_contribution_pct:
        failed.append("single_period_concentration")

    result["failed_gates"] = failed
    result["passes_default_gates"] = not failed
    return result


def fitness_v3_robust_score(
    period_rets: Sequence[float],
    *,
    period_max_drawdowns_pct: Sequence[float] | None = None,
    period_turnover_rates: Sequence[float] | None = None,
    period_effective_turnover_rates: Sequence[float] | None = None,
    period_saturation_rates: Sequence[float] | None = None,
    period_invalid_open_pressures: Sequence[float] | None = None,
    period_costs_pct: Sequence[float] | None = None,
    period_slippage_pct: Sequence[float] | None = None,
    period_regimes: Sequence[str] | None = None,
    required_regimes: Sequence[str] = FITNESS_V3_REGIMES,
    max_turnover_rate: float = 0.20,
    max_saturation_rate: float = 0.15,
    max_invalid_open_pressure: float = 0.25,
    min_positive_period_pct: float = 50.0,
    drawdown_penalty_weight: float = 0.25,
    downside_penalty_weight: float = 0.20,
    turnover_penalty_weight: float = 2.00,
    saturation_penalty_weight: float = 3.00,
    invalid_open_penalty_weight: float = 1.50,
    cost_penalty_weight: float = 1.00,
) -> dict[str, Any]:
    """Compute the v3 robust genetics objective.

    Returns are expected in percent points. Turnover, saturation and invalid
    pressure are rates. Costs/slippage should be percent points.
    """

    rets = _finite_array(period_rets)
    if rets.size == 0:
        return {
            "n_periods": 0,
            "fitness_v3_robust": float("-inf"),
            "passes_default_gates": False,
            "failed_gates": ["no_periods"],
        }

    n = int(rets.size)
    drawdowns = _aligned_array(period_max_drawdowns_pct, n)
    turnover = _aligned_array(period_turnover_rates, n)
    effective_turnover = _aligned_array(period_effective_turnover_rates, n)
    saturation = _aligned_array(period_saturation_rates, n)
    invalid_pressure = _aligned_array(period_invalid_open_pressures, n)
    costs = _aligned_array(period_costs_pct, n)
    slippage = _aligned_array(period_slippage_pct, n)
    regimes = _aligned_regimes(period_regimes, n)

    regime_means: dict[str, float] = {}
    missing_regimes: list[str] = []
    required = [str(regime) for regime in required_regimes]
    for regime in required:
        mask = np.asarray([item == regime for item in regimes], dtype=bool)
        if mask.any():
            regime_means[regime] = float(rets[mask].mean())
        else:
            regime_means[regime] = 0.0
            missing_regimes.append(regime)
    regime_balanced_mean = (
        float(np.mean([regime_means[regime] for regime in required]))
        if required
        else float(rets.mean())
    )

    downside = np.minimum(rets, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    mean_ret = float(rets.mean())
    min_ret = float(rets.min())
    positive_period_pct = float((rets > 0.0).mean() * 100.0)
    max_drawdown_pct = float(drawdowns.max())
    mean_turnover = float(turnover.mean())
    mean_effective_turnover = float(effective_turnover.mean())
    max_turnover = float(turnover.max())
    mean_saturation = float(saturation.mean())
    max_saturation = float(saturation.max())
    mean_invalid = float(invalid_pressure.mean())
    max_invalid = float(invalid_pressure.max())
    mean_cost = float(np.abs(costs).mean())
    mean_slippage = float(np.abs(slippage).mean())
    crash_floor_ret = float(regime_means.get("crash", 0.0))

    penalty = (
        max_drawdown_pct * float(drawdown_penalty_weight)
        + downside_deviation * float(downside_penalty_weight)
        + mean_turnover * float(turnover_penalty_weight)
        + mean_effective_turnover * float(turnover_penalty_weight) * 0.5
        + max_saturation * float(saturation_penalty_weight)
        + max_invalid * float(invalid_open_penalty_weight)
        + (mean_cost + mean_slippage) * float(cost_penalty_weight)
    )
    fitness = float(regime_balanced_mean - penalty)

    failed: list[str] = []
    failed.extend(f"missing_regime:{regime}" for regime in missing_regimes)
    if positive_period_pct < min_positive_period_pct:
        failed.append("positive_period_pct")
    if max_turnover > max_turnover_rate:
        failed.append("turnover")
    if max_saturation > max_saturation_rate:
        failed.append("saturation")
    if max_invalid > max_invalid_open_pressure:
        failed.append("invalid_open_pressure")

    return {
        "n_periods": n,
        "fitness_v3_robust": fitness,
        "mean_ret": mean_ret,
        "regime_balanced_mean_ret": regime_balanced_mean,
        "regime_mean_rets": regime_means,
        "min_ret": min_ret,
        "crash_floor_ret": crash_floor_ret,
        "positive_period_pct": positive_period_pct,
        "downside_deviation": downside_deviation,
        "max_drawdown_pct": max_drawdown_pct,
        "mean_turnover_rate": mean_turnover,
        "mean_effective_turnover_rate": mean_effective_turnover,
        "max_turnover_rate": max_turnover,
        "mean_saturation_rate": mean_saturation,
        "max_saturation_rate": max_saturation,
        "mean_invalid_open_pressure": mean_invalid,
        "max_invalid_open_pressure": max_invalid,
        "mean_cost_pct": mean_cost,
        "mean_slippage_pct": mean_slippage,
        "penalty": penalty,
        "failed_gates": failed,
        "passes_default_gates": not failed,
    }


def fitness_v4_robust_score(
    period_rets: Sequence[float],
    *,
    period_turnover_rates: Sequence[float] | None = None,
    period_effective_turnover_rates: Sequence[float] | None = None,
    period_saturation_rates: Sequence[float] | None = None,
    period_invalid_open_pressures: Sequence[float] | None = None,
    period_costs_pct: Sequence[float] | None = None,
    period_slippage_pct: Sequence[float] | None = None,
    period_regimes: Sequence[str] | None = None,
    required_regimes: Sequence[str] = FITNESS_V4_REGIMES,
    max_single_period_contribution_pct: float = 30.0,
    max_turnover_rate: float = 0.20,
    max_saturation_rate: float = 0.15,
    max_invalid_open_pressure: float = 0.25,
    min_positive_period_pct: float = 50.0,
    cvar_loss_penalty_weight: float = 0.50,
    downside_penalty_weight: float = 0.30,
    turnover_penalty_weight: float = 2.00,
    saturation_penalty_weight: float = 3.00,
    invalid_open_penalty_weight: float = 1.50,
    concentration_penalty_weight: float = 0.03,
    cost_penalty_weight: float = 1.00,
) -> dict[str, Any]:
    """Compute the v4 robust genetics utility.

    v4 is a promotion/selection utility, not a raw-return score: it starts from
    regime-balanced return and subtracts tail risk, downside, turnover,
    saturation, invalid-open pressure, costs, and single-period concentration.
    Returns are in percent points; rates are fractions.
    """

    rets = _finite_array(period_rets)
    if rets.size == 0:
        return {
            "n_periods": 0,
            "fitness_v4_robust": float("-inf"),
            "passes_default_gates": False,
            "failed_gates": ["no_periods"],
        }

    n = int(rets.size)
    turnover = _aligned_array(period_turnover_rates, n)
    effective_turnover = _aligned_array(period_effective_turnover_rates, n)
    saturation = _aligned_array(period_saturation_rates, n)
    invalid_pressure = _aligned_array(period_invalid_open_pressures, n)
    costs = _aligned_array(period_costs_pct, n)
    slippage = _aligned_array(period_slippage_pct, n)
    regimes = _aligned_regimes(period_regimes, n)

    regime_means: dict[str, float] = {}
    missing_regimes: list[str] = []
    required = [str(regime) for regime in required_regimes]
    for regime in required:
        mask = np.asarray([item == regime for item in regimes], dtype=bool)
        if mask.any():
            regime_means[regime] = float(rets[mask].mean())
        else:
            regime_means[regime] = 0.0
            missing_regimes.append(regime)
    regime_balanced_mean = (
        float(np.mean([regime_means[regime] for regime in required]))
        if required
        else float(rets.mean())
    )

    sorted_rets = np.sort(rets)
    tail_n = max(1, int(math.ceil(n * 0.05)))
    cvar_5_ret = float(sorted_rets[:tail_n].mean())
    cvar_loss = max(0.0, -cvar_5_ret)
    downside = np.minimum(rets, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    positives = rets[rets > 0.0]
    positive_sum = float(positives.sum())
    max_positive_contribution = (
        float(positives.max() / positive_sum * 100.0)
        if positive_sum > 0.0 and positives.size
        else 0.0
    )
    concentration_excess = max(
        0.0,
        max_positive_contribution - float(max_single_period_contribution_pct),
    )

    mean_ret = float(rets.mean())
    min_ret = float(rets.min())
    positive_period_pct = float((rets > 0.0).mean() * 100.0)
    mean_turnover = float(turnover.mean())
    mean_effective_turnover = float(effective_turnover.mean())
    max_turnover = float(turnover.max())
    mean_saturation = float(saturation.mean())
    max_saturation = float(saturation.max())
    mean_invalid = float(invalid_pressure.mean())
    max_invalid = float(invalid_pressure.max())
    mean_cost = float(np.abs(costs).mean())
    mean_slippage = float(np.abs(slippage).mean())
    crash_floor_ret = float(regime_means.get("crash", 0.0))

    penalty = (
        cvar_loss * float(cvar_loss_penalty_weight)
        + downside_deviation * float(downside_penalty_weight)
        + mean_turnover * float(turnover_penalty_weight)
        + mean_effective_turnover * float(turnover_penalty_weight) * 0.5
        + max_saturation * float(saturation_penalty_weight)
        + max_invalid * float(invalid_open_penalty_weight)
        + concentration_excess * float(concentration_penalty_weight)
        + (mean_cost + mean_slippage) * float(cost_penalty_weight)
    )
    fitness = float(regime_balanced_mean - penalty)

    failed: list[str] = []
    failed.extend(f"missing_regime:{regime}" for regime in missing_regimes)
    if positive_period_pct < min_positive_period_pct:
        failed.append("positive_period_pct")
    if max_positive_contribution > max_single_period_contribution_pct:
        failed.append("single_period_concentration")
    if max_turnover > max_turnover_rate:
        failed.append("turnover")
    if max_saturation > max_saturation_rate:
        failed.append("saturation")
    if max_invalid > max_invalid_open_pressure:
        failed.append("invalid_open_pressure")

    return {
        "n_periods": n,
        "fitness_v4_robust": fitness,
        "mean_ret": mean_ret,
        "regime_balanced_mean_ret": regime_balanced_mean,
        "regime_mean_rets": regime_means,
        "min_ret": min_ret,
        "crash_floor_ret": crash_floor_ret,
        "cvar_5_ret": cvar_5_ret,
        "cvar_loss": float(cvar_loss),
        "downside_deviation": downside_deviation,
        "positive_period_pct": positive_period_pct,
        "max_positive_contribution_pct": max_positive_contribution,
        "concentration_excess_pct": float(concentration_excess),
        "mean_turnover_rate": mean_turnover,
        "mean_effective_turnover_rate": mean_effective_turnover,
        "max_turnover_rate": max_turnover,
        "mean_saturation_rate": mean_saturation,
        "max_saturation_rate": max_saturation,
        "mean_invalid_open_pressure": mean_invalid,
        "max_invalid_open_pressure": max_invalid,
        "mean_cost_pct": mean_cost,
        "mean_slippage_pct": mean_slippage,
        "penalty": penalty,
        "failed_gates": failed,
        "passes_default_gates": not failed,
    }


def evaluate_fitness_v4_uplift_gate(
    *,
    baseline_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    baseline_oos: dict[str, Any] | None = None,
    candidate_oos: dict[str, Any] | None = None,
    baseline_final_sanity: dict[str, Any] | None = None,
    candidate_final_sanity: dict[str, Any] | None = None,
    max_turnover_rate: float = 0.20,
    max_saturation_rate: float = 0.15,
    max_invalid_open_pressure: float = 0.25,
    max_single_period_contribution_pct: float = 30.0,
    min_validation_fitness_delta: float | None = None,
) -> dict[str, Any]:
    """Gate candidates by Flash uplift against the matched baseline.

    The pass criteria are deliberately split-level: mean uplift >= 0, min-return
    uplift >= 0, positive-period rate not worse, and action-contract metrics
    within gates. Fitness v4 is reported as a diagnostic and can be made a hard
    validation requirement with ``min_validation_fitness_delta``.
    """

    failures: list[str] = []
    validation_payload = _fitness_v4_split_uplift(
        failures,
        prefix="validation",
        baseline=baseline_validation,
        candidate=candidate_validation,
        max_turnover_rate=max_turnover_rate,
        max_saturation_rate=max_saturation_rate,
        max_invalid_open_pressure=max_invalid_open_pressure,
        max_single_period_contribution_pct=max_single_period_contribution_pct,
    )
    if min_validation_fitness_delta is not None:
        if validation_payload["fitness_v4_delta"] < float(min_validation_fitness_delta):
            failures.append("validation_fitness_v4")

    oos_payload: dict[str, Any] | None = None
    if baseline_oos is not None and candidate_oos is not None:
        oos_payload = _fitness_v4_split_uplift(
            failures,
            prefix="oos",
            baseline=baseline_oos,
            candidate=candidate_oos,
            max_turnover_rate=max_turnover_rate,
            max_saturation_rate=max_saturation_rate,
            max_invalid_open_pressure=max_invalid_open_pressure,
            max_single_period_contribution_pct=max_single_period_contribution_pct,
        )

    final_payload: dict[str, Any] | None = None
    if baseline_final_sanity is not None and candidate_final_sanity is not None:
        final_payload = _fitness_v4_split_uplift(
            failures,
            prefix="final_sanity",
            baseline=baseline_final_sanity,
            candidate=candidate_final_sanity,
            max_turnover_rate=max_turnover_rate,
            max_saturation_rate=max_saturation_rate,
            max_invalid_open_pressure=max_invalid_open_pressure,
            max_single_period_contribution_pct=max_single_period_contribution_pct,
        )

    failures = list(dict.fromkeys(failures))
    return {
        "promotion_eligible": not failures,
        "promotion_failures": failures,
        "validation": validation_payload,
        "oos": oos_payload,
        "final_sanity": final_payload,
        "thresholds": {
            "max_turnover_rate": max_turnover_rate,
            "max_saturation_rate": max_saturation_rate,
            "max_invalid_open_pressure": max_invalid_open_pressure,
            "max_single_period_contribution_pct": max_single_period_contribution_pct,
            "min_validation_fitness_delta": min_validation_fitness_delta,
        },
    }


def evaluate_fitness_v3_promotion_gate(
    *,
    baseline_validation: dict[str, Any],
    candidate_validation: dict[str, Any],
    baseline_oos: dict[str, Any] | None = None,
    candidate_oos: dict[str, Any] | None = None,
    baseline_final_sanity: dict[str, Any] | None = None,
    candidate_final_sanity: dict[str, Any] | None = None,
    min_validation_fitness_delta: float = 0.0,
) -> dict[str, Any]:
    """Hard promotion gate for v3 robust genetics scoring."""

    failures: list[str] = []
    validation_mean_delta = _metric(candidate_validation, "mean_ret") - _metric(
        baseline_validation,
        "mean_ret",
    )
    validation_fitness_delta = _metric(
        candidate_validation,
        "fitness_v3_robust",
    ) - _metric(baseline_validation, "fitness_v3_robust")
    validation_min_delta = _metric(candidate_validation, "min_ret") - _metric(
        baseline_validation,
        "min_ret",
    )
    validation_positive_delta = _metric(
        candidate_validation,
        "positive_period_pct",
    ) - _metric(baseline_validation, "positive_period_pct")

    if validation_mean_delta <= 0.0:
        failures.append("validation_tie" if validation_mean_delta == 0.0 else "validation_mean_ret")
    if validation_fitness_delta <= min_validation_fitness_delta:
        failures.append("validation_fitness_v3")
    if validation_min_delta < 0.0:
        failures.append("validation_min_ret")
    if validation_positive_delta < 0.0:
        failures.append("validation_positive_period_pct")
    _extend_score_gate_failures(failures, candidate_validation, prefix="validation")

    oos_payload: dict[str, Any] | None = None
    if baseline_oos is not None and candidate_oos is not None:
        oos_mean_delta = _metric(candidate_oos, "mean_ret") - _metric(
            baseline_oos,
            "mean_ret",
        )
        oos_min_delta = _metric(candidate_oos, "min_ret") - _metric(
            baseline_oos,
            "min_ret",
        )
        oos_crash_delta = _metric(candidate_oos, "crash_floor_ret") - _metric(
            baseline_oos,
            "crash_floor_ret",
        )
        if oos_mean_delta < 0.0:
            failures.append("oos_mean_ret")
        if oos_min_delta < 0.0:
            failures.append("oos_min_ret")
        if oos_crash_delta < 0.0:
            failures.append("crash_floor")
        _extend_score_gate_failures(failures, candidate_oos, prefix="oos")
        oos_payload = {
            "mean_ret_delta": float(oos_mean_delta),
            "min_ret_delta": float(oos_min_delta),
            "crash_floor_delta": float(oos_crash_delta),
        }

    final_payload: dict[str, Any] | None = None
    if baseline_final_sanity is not None and candidate_final_sanity is not None:
        final_mean_delta = _metric(candidate_final_sanity, "mean_ret") - _metric(
            baseline_final_sanity,
            "mean_ret",
        )
        final_crash_delta = _metric(
            candidate_final_sanity,
            "crash_floor_ret",
        ) - _metric(baseline_final_sanity, "crash_floor_ret")
        if final_mean_delta < 0.0:
            failures.append("final_sanity_mean_ret")
        if final_crash_delta < 0.0:
            failures.append("final_sanity_crash_floor")
        _extend_score_gate_failures(
            failures,
            candidate_final_sanity,
            prefix="final_sanity",
        )
        final_payload = {
            "mean_ret_delta": float(final_mean_delta),
            "crash_floor_delta": float(final_crash_delta),
        }

    failures = list(dict.fromkeys(failures))
    return {
        "promotion_eligible": not failures,
        "promotion_failures": failures,
        "validation": {
            "mean_ret_delta": float(validation_mean_delta),
            "fitness_v3_delta": float(validation_fitness_delta),
            "min_ret_delta": float(validation_min_delta),
            "positive_period_pct_delta": float(validation_positive_delta),
        },
        "oos": oos_payload,
        "final_sanity": final_payload,
    }


def _finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _aligned_array(values: Sequence[float] | None, n: int) -> np.ndarray:
    if values is None:
        return np.zeros(n, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < n:
        out = np.zeros(n, dtype=np.float64)
        out[: arr.size] = arr
        arr = out
    else:
        arr = arr[:n]
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float64)


def _aligned_regimes(values: Sequence[str] | None, n: int) -> list[str]:
    if values is None:
        return ["unknown"] * n
    out = [str(value or "unknown") for value in values[:n]]
    if len(out) < n:
        out.extend(["unknown"] * (n - len(out)))
    return out


def _metric(payload: dict[str, Any], key: str) -> float:
    try:
        value = float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _extend_score_gate_failures(
    failures: list[str],
    score: dict[str, Any],
    *,
    prefix: str,
) -> None:
    for gate in score.get("failed_gates", []) or []:
        failures.append(f"{prefix}_{gate}")


def _fitness_v4_split_uplift(
    failures: list[str],
    *,
    prefix: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    max_turnover_rate: float,
    max_saturation_rate: float,
    max_invalid_open_pressure: float,
    max_single_period_contribution_pct: float,
) -> dict[str, Any]:
    mean_delta = _metric(candidate, "mean_ret") - _metric(baseline, "mean_ret")
    min_delta = _metric(candidate, "min_ret") - _metric(baseline, "min_ret")
    positive_delta = _metric(candidate, "positive_period_pct") - _metric(
        baseline,
        "positive_period_pct",
    )
    fitness_delta = _metric(candidate, "fitness_v4_robust") - _metric(
        baseline,
        "fitness_v4_robust",
    )
    crash_delta = _metric(candidate, "crash_floor_ret") - _metric(
        baseline,
        "crash_floor_ret",
    )

    if mean_delta < 0.0:
        failures.append(f"{prefix}_mean_ret")
    if min_delta < 0.0:
        failures.append(f"{prefix}_min_ret")
    if positive_delta < 0.0:
        failures.append(f"{prefix}_positive_period_pct")
    if _metric(candidate, "max_turnover_rate") > max_turnover_rate:
        failures.append(f"{prefix}_turnover")
    if _metric(candidate, "max_saturation_rate") > max_saturation_rate:
        failures.append(f"{prefix}_saturation")
    if _metric(candidate, "max_invalid_open_pressure") > max_invalid_open_pressure:
        failures.append(f"{prefix}_invalid_open_pressure")
    if (
        _metric(candidate, "max_positive_contribution_pct")
        > max_single_period_contribution_pct
    ):
        failures.append(f"{prefix}_single_period_concentration")
    _extend_score_gate_failures(failures, candidate, prefix=prefix)

    return {
        "mean_ret_delta": float(mean_delta),
        "min_ret_delta": float(min_delta),
        "positive_period_pct_delta": float(positive_delta),
        "fitness_v4_delta": float(fitness_delta),
        "crash_floor_delta": float(crash_delta),
        "baseline_mean_ret": _metric(baseline, "mean_ret"),
        "candidate_mean_ret": _metric(candidate, "mean_ret"),
        "baseline_min_ret": _metric(baseline, "min_ret"),
        "candidate_min_ret": _metric(candidate, "min_ret"),
        "candidate_max_turnover_rate": _metric(candidate, "max_turnover_rate"),
        "candidate_max_saturation_rate": _metric(candidate, "max_saturation_rate"),
        "candidate_max_invalid_open_pressure": _metric(
            candidate,
            "max_invalid_open_pressure",
        ),
        "candidate_max_positive_contribution_pct": _metric(
            candidate,
            "max_positive_contribution_pct",
        ),
    }
