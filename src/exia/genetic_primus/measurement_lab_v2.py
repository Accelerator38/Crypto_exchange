"""Small, offline, versioned measurement lab; never a trading runtime.

The search objective is a single train-only daily-return LCB. A candidate chosen
on train is checked once on validation. Validation cannot select an alternative.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Callable

import numpy as np
import pandas as pd

from .accounting_v2 import CashSimulation, simulate_cash_targets

DAY_MS = 86_400_000
FEATURES = ("ema_gap_12_48", "ema_slope_12_3", "return_12", "efficiency_12")
Genome = tuple[int, int, int, int, int]
NO_TRADE: Genome = (0, 0, 0, 0, -1)
ALLOW_ALL: Genome = (0, 0, 0, 0, 1)


def object_sha(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode()).hexdigest()


def array_sha(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    prefix = json.dumps([array.dtype.str, array.shape]).encode()
    return sha256(prefix + array.tobytes()).hexdigest()


def symmetric_rank(values: pd.Series) -> pd.Series:
    """Centered average ranks: ties, odd universes and sign reflection are neutral."""
    count = int(values.notna().sum())
    if not count:
        return pd.Series(np.nan, index=values.index)
    return 2.0 * (values.rank(method="average") - 0.5) / count - 1.0


def build_lab_features(panel: pd.DataFrame) -> pd.DataFrame:
    from .features import build_feature_frame

    frame = build_feature_frame(panel)
    frame["relative_momentum_rank"] = frame.groupby("timestamp", observed=True)[
        "return_12"
    ].transform(symmetric_rank)
    return frame


@dataclass(frozen=True)
class LabScaler:
    median: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_rows: int
    fit_values_sha256: str

    @classmethod
    def fit(cls, values: np.ndarray) -> "LabScaler":
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(FEATURES):
            raise ValueError("four registered features required")
        finite = values[np.isfinite(values).all(axis=1)]
        if not len(finite):
            raise ValueError("no finite training features")
        median = np.median(finite, axis=0)
        q25, q75 = np.quantile(finite, [0.25, 0.75], axis=0)
        scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
        return cls(tuple(median), tuple(scale), len(finite), array_sha(finite))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.median) / self.scale, -8.0, 8.0)

    def mapping(self) -> dict:
        return {"median": self.median, "scale": self.scale,
                "fitted_rows": self.fitted_rows, "fit_values_sha256": self.fit_values_sha256}


def temporal_bounds(origin: int, training_days: int, validation_days: int, gap_bars: int,
                    timeframe_ms: int, deployment_days: int = 3) -> dict[str, int]:
    if min(training_days, validation_days, deployment_days, timeframe_ms) <= 0 or gap_bars < 1:
        raise ValueError("positive intervals and explicit boundary gap required")
    validation_end = origin - gap_bars * timeframe_ms
    validation_start = validation_end - validation_days * DAY_MS
    train_end = validation_start - gap_bars * timeframe_ms
    return {"train_start": train_end - training_days * DAY_MS, "train_end": train_end,
            "validation_start": validation_start, "validation_end": validation_end,
            "development_start": origin, "development_end": origin + deployment_days * DAY_MS}


@dataclass(frozen=True)
class PreparedWindow:
    bars: pd.DataFrame
    base_schedule: pd.DataFrame
    values: np.ndarray
    proposal: np.ndarray
    symbols: tuple[str, ...]
    start: int
    end: int
    timeframe_ms: int

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, symbols: tuple[str, ...], start: int,
                   end: int, timeframe_ms: int) -> "PreparedWindow":
        times = np.arange(start, end, timeframe_ms, dtype=np.int64)
        if end <= start or (end - start) % timeframe_ms:
            raise ValueError("whole candles required")
        index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
        relevant = frame[frame.symbol.isin(symbols) & frame.timestamp.between(start - timeframe_ms, end - 1)]
        if relevant.duplicated(["timestamp", "symbol"]).any():
            raise ValueError("duplicate market/feature row")
        bars = relevant.set_index(["timestamp", "symbol"])[["open", "close"]].reindex(index).reset_index()
        if not np.isfinite(bars[["open", "close"]].to_numpy(float)).all():
            raise ValueError("incomplete fixed-universe market grid")
        prior = relevant[["timestamp", "symbol", *FEATURES]].copy()
        prior["timestamp"] += timeframe_ms
        joined = prior.set_index(["timestamp", "symbol"]).reindex(index)
        values = joined[list(FEATURES)].to_numpy(float)
        finite = np.isfinite(values).all(axis=1)
        directions = np.sign(np.nan_to_num(values[:, 0]))
        consensus = finite & (values[:, 3] >= 0.30) & (np.sign(values[:, 1]) == directions)
        consensus &= np.sign(values[:, 2]) == directions
        proposal = np.where(consensus, directions, 0).astype(np.int8)
        base = index.to_frame(index=False).rename(columns={"timestamp": "execution_timestamp"})
        # Previous candle closes at this open. Zero latency is explicit and NOT live-ready.
        base["signal_timestamp"] = base.execution_timestamp
        return cls(bars, base, values, proposal, symbols, start, end, timeframe_ms)

    def model(self, genome: Genome, scaler: LabScaler) -> dict:
        return {"schema": "binary_sparse_lab/2", "genome": genome, "features": FEATURES,
                "scaler": scaler.mapping(), "proposal": "V7_EMA_TrendConsensus",
                "entry_rule": "logit_strictly_positive", "missing_features": "abstain"}

    def simulate(self, genome: Genome, scaler: LabScaler, cost_bps: float, *,
                 details: bool = False, force_flat_timestamps: tuple[int, ...] = ()) -> CashSimulation:
        schedule = self.schedule(genome, scaler)
        return simulate_cash_targets(
            self.bars, schedule, symbols=self.symbols, start_timestamp=self.start,
            end_timestamp=self.end, timeframe_ms=self.timeframe_ms,
            round_trip_cost_bps=cost_bps, record_details=details,
            force_flat_timestamps=force_flat_timestamps,
        )

    def schedule(self, genome: Genome, scaler: LabScaler) -> pd.DataFrame:
        """Return decisions without resetting cash at the deployment boundary."""
        if genome not in CATALOG:
            raise ValueError("unregistered model complexity")
        normalized = scaler.transform(self.values)
        allow = np.isfinite(normalized).all(axis=1)
        allow &= (np.nan_to_num(normalized) @ np.asarray(genome[:4]) + genome[4]) > 0.0
        schedule = self.base_schedule.copy()
        schedule["target"] = np.where(allow, self.proposal, 0)
        schedule["model_sha256"] = object_sha(self.model(genome, scaler))
        return schedule


def candidate_catalog() -> tuple[Genome, ...]:
    # Equivalent zero-weight always-abstain models are represented only once.
    values = [NO_TRADE, ALLOW_ALL]
    values.extend((*weights, bias) for weights in product((-1, 0, 1), repeat=4)
                  if 1 <= np.count_nonzero(weights) <= 2 for bias in (-1, 0, 1))
    return tuple(values)


CATALOG = candidate_catalog()


def daily_returns(simulation: CashSimulation) -> np.ndarray:
    frame = simulation.portfolio
    # Equal 24-hour blocks from window start, not unequal UTC fragments at a purge boundary.
    day = (frame.timestamp.to_numpy(np.int64) - int(frame.timestamp.iloc[0])) // DAY_MS
    return frame.net_return.groupby(day).apply(lambda x: float(np.prod(1.0 + x) - 1.0)).to_numpy(float)


def block_bank(days: int, samples: int, block_days: int, seed: int) -> np.ndarray:
    if min(days, samples, block_days) < 1:
        raise ValueError("positive bootstrap dimensions required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(samples, (days + block_days - 1) // block_days))
    return ((starts[..., None] + np.arange(block_days)) % days).reshape(samples, -1)[:, :days]


def paired_lcb(challenger: np.ndarray, incumbent: np.ndarray, bank: np.ndarray, alpha: float = 0.05) -> float:
    challenger, incumbent = np.asarray(challenger), np.asarray(incumbent)
    if challenger.ndim != 1 or challenger.shape != incumbent.shape or not len(challenger):
        raise ValueError("aligned nonempty daily observations required")
    difference = challenger - incumbent
    if not np.isfinite(difference).all() or bank.ndim != 2 or bank.shape[1] != len(difference):
        raise ValueError("invalid observations or shared bank")
    if bank.min() < 0 or bank.max() >= len(difference):
        raise ValueError("bank indices out of range")
    return float(np.quantile(difference[bank].mean(axis=1), alpha))


def rank_key(genome: Genome, score: float) -> tuple:
    return (score, -int(np.count_nonzero(genome[:4])), tuple(-x for x in genome))


def search(method: str, evaluate: Callable[[Genome], float], budget: int, seed: int,
           population_size: int = 6) -> dict:
    if not 2 <= budget <= len(CATALOG) or population_size < 2:
        raise ValueError("invalid unique evaluation budget")
    if method not in {"sparse_grid", "random_search", "genetic_search"}:
        raise ValueError("unknown method")
    rng = np.random.default_rng(seed)
    remaining = list(CATALOG[2:])
    if method == "sparse_grid":
        selected = [remaining[int(i)] for i in np.linspace(0, len(remaining) - 1, budget - 2)]
    else:
        selected = [remaining[int(i)] for i in rng.permutation(len(remaining))]
    records: dict[Genome, float] = {}
    pending = [NO_TRADE, ALLOW_ALL]
    generation_sizes: list[int] = []
    while len(records) < budget:
        if pending:
            genome = pending.pop(0)
        elif method != "genetic_search" or len(records) < population_size:
            genome = next(g for g in selected if g not in records)
        else:
            if not generation_sizes or len(records) >= sum(generation_sizes):
                parents = sorted(records, key=lambda g: rank_key(g, records[g]), reverse=True)[:population_size]
                batch_size = min(population_size, budget - len(records))
                generation_sizes = [len(records), batch_size]
                pending = []
                for _ in range(batch_size):
                    child = None
                    for _attempt in range(32):
                        first, second = (parents[int(rng.integers(len(parents)))] for _ in range(2))
                        raw = np.where(rng.random(5) < 0.5, first, second).astype(int)
                        raw[int(rng.integers(5))] = int(rng.choice([-1, 0, 1]))
                        trial = tuple(int(x) for x in raw)
                        if trial in CATALOG and trial not in records and trial not in pending:
                            child = trial
                            break
                    if child is None:
                        unseen = [g for g in CATALOG if g not in records and g not in pending]
                        child = unseen[int(rng.integers(len(unseen)))]
                    pending.append(child)
            genome = pending.pop(0)
        score = float(evaluate(genome))
        if genome in records or not np.isfinite(score):
            raise ValueError("duplicate evaluation or non-finite fitness")
        records[genome] = score
    winner = max(records, key=lambda g: rank_key(g, records[g]))
    return {"method": method, "winner": winner, "score": records[winner],
            "unique_evaluations": len(records), "seed": seed,
            "evaluations": [{"genome": g, "fitness_lcb": score} for g, score in records.items()]}


def validation_gate(normal: CashSimulation, stress: CashSimulation, bank: np.ndarray,
                    alpha: float = 0.05, maximum_drawdown: float = 0.10) -> dict:
    returns, stress_returns = daily_returns(normal), daily_returns(stress)
    normal_lcb = paired_lcb(returns, np.zeros_like(returns), bank, alpha)
    stress_lcb = paired_lcb(stress_returns, np.zeros_like(stress_returns), bank, alpha)
    reasons = []
    if not np.any(np.abs(returns) > 1e-14):
        reasons.append("no_nonzero_return_evidence")
    if normal_lcb <= 0:
        reasons.append("normal_lcb_not_positive")
    if stress_lcb <= 0:
        reasons.append("stress_lcb_not_positive")
    if normal.metrics["max_drawdown"] > maximum_drawdown:
        reasons.append("absolute_drawdown_budget_exceeded")
    return {"passed": not reasons, "reasons": reasons, "normal_lcb": normal_lcb,
            "stress_lcb": stress_lcb, "maximum_drawdown": maximum_drawdown,
            "selection_authority": False, "promotion_authority": False}
