from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


GENETIC_CORE_WALK_FORWARD_WINDOWS: Mapping[str, tuple[str, str]] = {
    "train": ("2022-01-01", "2023-12-31"),
    "validation": ("2024-01-01", "2024-12-31"),
    "oos": ("2025-01-01", "2025-12-31"),
    "sanity": ("2026-01-01", "2026-06-30"),
}

PROVEN_BC_SEED_AGENTS: tuple[str, ...] = (
    "LiveOIBreakout",
    "MomentumScalper",
    "ResearchValidatorAgent",
)
TEACHER_FEATURE_POLICY = "research_after_oos_stabilization"
TEACHER_RUNTIME_LOGITS_ENABLED = False


@dataclass(frozen=True)
class ExchangeCostProfile:
    exchange: str
    fees_bps: float
    funding_bps: float
    spread_bps: float
    slippage_bps: float
    min_notional_usd: float
    quantity_precision_step: float

    @property
    def round_trip_cost_pct(self) -> float:
        return (
            self.fees_bps
            + self.funding_bps
            + self.spread_bps
            + self.slippage_bps
        ) / 100.0

    def genetics_settings_overrides(self) -> dict[str, float | str]:
        return {
            "exchange_cost_profile": self.exchange,
            "train_fee": self.fees_bps / 10000.0,
            "train_funding_rate": self.funding_bps / 10000.0,
            "train_spread": self.spread_bps / 10000.0,
            "train_slippage": self.slippage_bps / 10000.0,
            "train_min_notional_usd": self.min_notional_usd,
            "train_quantity_precision_step": self.quantity_precision_step,
        }


@dataclass(frozen=True)
class GeneticCoreTrainingContract:
    windows: Mapping[str, tuple[str, str]]
    bc_seed_agents: tuple[str, ...]
    exchange_cost_profile: ExchangeCostProfile
    teacher_feature_policy: str
    teacher_runtime_logits_enabled: bool


@dataclass(frozen=True)
class GeneticCoreTrainingScore:
    score: float
    components: Mapping[str, float]
    penalties: Mapping[str, float]


_EXCHANGE_COST_PROFILES: Mapping[str, ExchangeCostProfile] = {
    "MEXC": ExchangeCostProfile(
        exchange="MEXC",
        fees_bps=4.0,
        funding_bps=1.0,
        spread_bps=2.0,
        slippage_bps=1.0,
        min_notional_usd=5.0,
        quantity_precision_step=0.001,
    ),
    "BITGET": ExchangeCostProfile(
        exchange="BITGET",
        fees_bps=6.0,
        funding_bps=1.0,
        spread_bps=2.0,
        slippage_bps=1.5,
        min_notional_usd=5.0,
        quantity_precision_step=0.001,
    ),
}


def exchange_cost_profile(exchange: str) -> ExchangeCostProfile:
    key = str(exchange or "").strip().upper()
    try:
        return _EXCHANGE_COST_PROFILES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_EXCHANGE_COST_PROFILES))
        raise ValueError(f"unsupported exchange for GeneticCore: {exchange!r}; supported: {supported}") from exc


def validate_genetic_core_training_contract(
    *,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    oos_start: str,
    oos_end: str,
    sanity_start: str,
    sanity_end: str,
    bc_seed_agents: Sequence[str],
    exchange: str,
) -> GeneticCoreTrainingContract:
    provided = {
        "train": (str(train_start), str(train_end)),
        "validation": (str(validation_start), str(validation_end)),
        "oos": (str(oos_start), str(oos_end)),
        "sanity": (str(sanity_start), str(sanity_end)),
    }
    for name, expected in GENETIC_CORE_WALK_FORWARD_WINDOWS.items():
        if provided[name] != expected:
            raise ValueError(
                f"{name} window must be {expected[0]}..{expected[1]}, got "
                f"{provided[name][0]}..{provided[name][1]}"
            )

    normalized_agents = tuple(str(agent).strip() for agent in bc_seed_agents if str(agent).strip())
    if normalized_agents != PROVEN_BC_SEED_AGENTS:
        raise ValueError(
            "BC seed agents must be exactly "
            + ",".join(PROVEN_BC_SEED_AGENTS)
            + f"; got {','.join(normalized_agents) or '<empty>'}"
        )

    return GeneticCoreTrainingContract(
        windows=provided,
        bc_seed_agents=normalized_agents,
        exchange_cost_profile=exchange_cost_profile(exchange),
        teacher_feature_policy=TEACHER_FEATURE_POLICY,
        teacher_runtime_logits_enabled=TEACHER_RUNTIME_LOGITS_ENABLED,
    )


def _finite_array(values: Sequence[float], *, fill: float | None = None, n: int | None = None) -> np.ndarray:
    arr = np.asarray(list(values or ()), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if n is None:
        return arr
    if arr.size == n:
        return arr
    if fill is None:
        fill = 0.0
    if arr.size == 0:
        return np.full(n, float(fill), dtype=np.float64)
    if arr.size > n:
        return arr[:n]
    return np.pad(arr, (0, n - arr.size), constant_values=float(fill))


def genetic_core_training_score(
    *,
    period_returns_pct: Sequence[float],
    period_lcb_pct: Sequence[float] | None = None,
    period_max_drawdown_pct: Sequence[float] | None = None,
    period_costs_pct: Sequence[float] | None = None,
    turnover_rates: Sequence[float] | None = None,
    invalid_open_pressures: Sequence[float] | None = None,
    bad_signal_key_rates: Sequence[float] | None = None,
    concentration_pct: float = 0.0,
) -> GeneticCoreTrainingScore:
    rets = _finite_array(period_returns_pct)
    if rets.size == 0:
        return GeneticCoreTrainingScore(
            score=float("-inf"),
            components={"n_periods": 0.0},
            penalties={"no_periods": float("inf")},
        )

    n = int(rets.size)
    lcb_values = _finite_array(period_lcb_pct or (), fill=0.0, n=n)
    if not np.any(lcb_values):
        lcb = float(rets.mean() - 1.645 * rets.std(ddof=0) / math.sqrt(max(n, 1)))
    else:
        lcb = float(lcb_values.mean())
    drawdowns = _finite_array(period_max_drawdown_pct or (), fill=5.0, n=n)
    costs = _finite_array(period_costs_pct or (), fill=0.0, n=n)
    turnover = _finite_array(turnover_rates or (), fill=0.0, n=n)
    invalid_open = _finite_array(invalid_open_pressures or (), fill=0.0, n=n)
    bad_signal = _finite_array(bad_signal_key_rates or (), fill=0.0, n=n)

    sorted_rets = np.sort(rets)
    tail_n = max(1, int(math.ceil(n * 0.05)))
    cvar_5 = float(sorted_rets[:tail_n].mean())
    mean_ret = float(rets.mean())
    mean_drawdown = float(np.maximum(drawdowns, 0.0).mean())
    calmar = float(mean_ret / (mean_drawdown + 5.0) * 5.0)
    positive_window_pct = float((rets > 0.0).mean() * 100.0)
    concentration_excess = max(0.0, float(concentration_pct) - 30.0)

    penalties = {
        "cost_penalty": float(np.abs(costs).mean()),
        "turnover_penalty": float(np.maximum(0.0, turnover - 0.20).mean() * 4.0),
        "invalid_open_penalty": float(np.maximum(0.0, invalid_open - 0.05).mean() * 3.0),
        "concentration_penalty": float(concentration_excess * 0.03),
        "bad_signal_key_penalty": float(np.maximum(0.0, bad_signal).mean() * 6.0),
    }
    components = {
        "n_periods": float(n),
        "mean_ret_pct": mean_ret,
        "lcb_pct": lcb,
        "calmar": calmar,
        "cvar_5_pct": cvar_5,
        "positive_window_pct": positive_window_pct,
    }
    score = (
        0.30 * lcb
        + 0.25 * calmar
        + 0.20 * cvar_5
        + 0.25 * (positive_window_pct / 100.0)
        - sum(penalties.values())
    )
    return GeneticCoreTrainingScore(
        score=float(score),
        components=components,
        penalties=penalties,
    )
