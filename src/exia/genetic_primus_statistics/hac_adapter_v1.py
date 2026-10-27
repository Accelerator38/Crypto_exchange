"""Frozen HAC small-sample t adapter, qualified only on n=30 and n=365 daily rows.

The formula is copied from the immutable CI experiment, not refactored out of it.
Before market-ledger use the runner requires parity with every saved qualifying
primary/audit array. This module neither selects models nor issues permissions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

import numpy as np
from scipy.stats import t as student_t

DAY_MS = 86_400_000
ALPHA = .025
EPS = 1e-14
QUALIFIED_HORIZONS = (30, 365)
QUALIFICATION_CONFIG_SHA = "560a84c371f512dc91671f5c56e6ffe8b8fda9cf71bb28f2a16f9f51ab317552"


def specification() -> dict:
    return {"schema": "exia.genetic_primus.hac_adapter/1", "method": "hac_small_sample_t",
            "alpha": ALPHA, "qualified_daily_counts": list(QUALIFIED_HORIZONS),
            "block_rule": "ceil_sqrt_n", "hac_lags": "block_minus_one", "kernel": "Bartlett",
            "df_rule": "max_2_floor_n_over_2block_minus_one", "zero_se": "insufficient_variation",
            "qualification_config_sha256": QUALIFICATION_CONFIG_SHA,
            "runtime_authority": False, "promotion_authority": False}


def _components(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or not len(paths) or paths.shape[1] not in QUALIFIED_HORIZONS:
        raise ValueError("unsupported daily horizon; qualified counts are 30 and 365")
    if not np.isfinite(paths).all():
        raise ValueError("nonfinite observations")
    n = paths.shape[1]
    block = math.ceil(math.sqrt(n))
    lags = block - 1
    mean = paths.mean(axis=1)
    centered = paths - paths.mean(axis=-1, keepdims=True)
    lrv = np.einsum("...i,...i->...", centered, centered) / n
    for lag in range(1, lags + 1):
        covariance = np.einsum("...i,...i->...", centered[..., lag:], centered[..., :-lag]) / n
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    se = np.sqrt(np.maximum(lrv, 0.0) / (n - 1.0))
    df = max(2, n // (2 * block) - 1)
    critical = float(student_t.ppf(1.0 - ALPHA, df))
    lcb = np.where(se > EPS, mean - critical * se, -np.inf)
    return mean, se, lcb, block, df, critical


def qualified_lcb_matrix(paths: np.ndarray) -> np.ndarray:
    """Parity interface for saved synthetic matrices; not a model-search API."""
    return _components(paths)[2]


@dataclass(frozen=True)
class HacAssessment:
    schema: str
    status: str
    daily_count: int
    mean_difference: float
    standard_error: float
    lcb: float | None
    block_length: int
    degrees_of_freedom: int
    critical_value: float
    observations_sha256: str
    promotion_authority: bool = False

    def mapping(self) -> dict:
        return asdict(self)


def assess_paired_daily(challenger: np.ndarray, incumbent: np.ndarray, timestamps: np.ndarray) -> HacAssessment:
    challenger, incumbent = np.asarray(challenger, dtype=float), np.asarray(incumbent, dtype=float)
    timestamps = np.asarray(timestamps)
    if challenger.ndim != 1 or challenger.shape != incumbent.shape or timestamps.shape != challenger.shape:
        raise ValueError("aligned one-dimensional paired observations required")
    if not np.issubdtype(timestamps.dtype, np.integer) or not np.all(np.diff(timestamps) == DAY_MS):
        raise ValueError("timestamps must be consecutive equal 24-hour observations")
    if not np.isfinite(challenger).all() or not np.isfinite(incumbent).all():
        raise ValueError("nonfinite daily returns")
    if (challenger <= -1).any() or (incumbent <= -1).any():
        raise ValueError("nonpositive equity return is not supported")
    mean, se, lcb, block, df, critical = _components((challenger - incumbent)[None, :])
    digest = sha256(json.dumps({"n": len(challenger), "columns": ["timestamp", "challenger", "incumbent"]}, sort_keys=True).encode())
    for values, dtype in ((timestamps, "<i8"), (challenger, "<f8"), (incumbent, "<f8")):
        digest.update(np.ascontiguousarray(values, dtype=dtype).tobytes())
    informative = bool(np.isfinite(lcb[0]))
    return HacAssessment("exia.genetic_primus.hac_assessment/1",
                         "informative" if informative else "insufficient_variation",
                         len(challenger), float(mean[0]), float(se[0]), float(lcb[0]) if informative else None,
                         block, df, critical, digest.hexdigest())


def complete_daily_returns(timestamps: np.ndarray, returns: np.ndarray, equity: np.ndarray, *,
                           start: int, end: int, timeframe_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact whole-day compounding from the frozen cash ledger, never UTC fragments."""
    timestamps, returns, equity = np.asarray(timestamps), np.asarray(returns, float), np.asarray(equity, float)
    if timeframe_ms <= 0 or DAY_MS % timeframe_ms or end <= start or (end - start) % DAY_MS:
        raise ValueError("complete equal-duration days and candles required")
    expected = np.arange(start + timeframe_ms, end + timeframe_ms, timeframe_ms, dtype=np.int64)
    if not np.issubdtype(timestamps.dtype, np.integer) or not np.array_equal(timestamps, expected):
        raise ValueError("missing, duplicate, reordered or shifted bar closes")
    if returns.shape != timestamps.shape or equity.shape != timestamps.shape:
        raise ValueError("unaligned cash portfolio")
    if not np.isfinite(returns).all() or not np.isfinite(equity).all() or (returns <= -1).any() or (equity <= 0).any():
        raise ValueError("invalid cash equity path")
    if not np.allclose(np.cumprod(1.0 + returns), equity, rtol=0, atol=1e-10):
        raise ValueError("cash equity and compounded returns disagree")
    per_day = DAY_MS // timeframe_ms
    values = np.prod(1.0 + returns.reshape(-1, per_day), axis=1) - 1.0
    closes = timestamps.reshape(-1, per_day)[:, -1]
    if len(values) not in QUALIFIED_HORIZONS:
        raise ValueError("daily horizon outside qualified domain")
    return values, closes


def diagnostic_gate(normal: HacAssessment, stress: HacAssessment, drawdown: float, *, limit: float = .10) -> dict:
    if not np.isfinite(drawdown) or not 0 <= drawdown <= 1 or not 0 < limit < 1:
        raise ValueError("invalid absolute drawdown risk budget")
    if normal.daily_count != stress.daily_count:
        raise ValueError("stress and normal horizons differ")
    reasons = []
    for label, result in (("normal", normal), ("stress", stress)):
        if result.status != "informative" or result.lcb is None:
            reasons.append(label + "_insufficient_variation")
        elif not np.isfinite(result.lcb) or not np.isfinite(result.standard_error) or result.standard_error <= EPS:
            reasons.append(label + "_invalid_assessment")
        elif result.lcb <= 0:
            reasons.append(label + "_lcb_not_positive")
    if drawdown > limit:
        reasons.append("absolute_drawdown_budget_exceeded")
    return {"passed": not reasons, "reasons": reasons, "drawdown_limit": limit,
            "diagnostic_only": True, "promotion_authority": False, "runtime_authority": False}
