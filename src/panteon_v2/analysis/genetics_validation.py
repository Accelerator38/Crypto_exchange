"""Validation helpers for genetic-agent research runs.

These functions are intentionally pure and lightweight so tests can cover the
walk-forward contract without importing the heavy training module.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np


def _period_year(period: str) -> int:
    return int(str(period)[:4])


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
