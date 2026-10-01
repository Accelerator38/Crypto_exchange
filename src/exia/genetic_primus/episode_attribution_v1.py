"""Episode-preserving retrospective control for an archived GA entry mask."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .measurement_lab_v2 import object_sha


def episode_lengths(mask: np.ndarray) -> tuple[int, ...]:
    """Lengths of contiguous true episodes, including the final one."""
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return tuple((ends - starts + 1).tolist())


def reversed_episode_control(base: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    """Reverse the GA mask independently inside each complete EMA-long episode.

    Reversal preserves GA holding-episode lengths, counts and symbol exposure.
    It changes the chosen timestamps without reading any market return.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    required = {"execution_timestamp", "signal_timestamp", "symbol", "proposal",
                "ga_action", "ema_model_sha256"}
    if not required.issubset(base) or base.empty:
        raise ValueError("archived aligned decision rows required")
    if base.duplicated(["execution_timestamp", "symbol"]).any():
        raise ValueError("duplicate decision")
    if not np.array_equal(base.index.to_numpy(), np.arange(len(base))):
        raise ValueError("base must have a zero-based contiguous index")
    rng = np.random.default_rng(seed)
    target = np.zeros(len(base), dtype=np.int8)
    original = np.zeros(len(base), dtype=np.int8)
    reversal_count = 0
    opportunity_count = 0
    exposure_by_symbol: dict[str, dict] = {}
    for symbol, group in base.groupby("symbol", sort=True, observed=True):
        order = group.sort_values("execution_timestamp", kind="stable")
        indices = order.index.to_numpy(int)
        if not np.all(np.diff(order.execution_timestamp.to_numpy(np.int64)) == 14_400_000):
            raise ValueError("symbol 4h timeline must be complete")
        eligible = order.proposal.to_numpy(int) == 1
        ga = order.ga_action.to_numpy(int) == 1
        if (ga & ~eligible).any():
            raise ValueError("GA long action outside EMA long opportunity")
        original[indices] = ga.astype(np.int8)
        control = np.zeros(len(order), dtype=np.int8)
        starts = np.flatnonzero(eligible & ~np.r_[False, eligible[:-1]])
        ends = np.flatnonzero(eligible & ~np.r_[eligible[1:], False]) + 1
        for start, end in zip(starts, ends):
            opportunity_count += 1
            reverse = bool(rng.integers(0, 2))
            segment = ga[start:end].astype(np.int8)
            control[start:end] = segment[::-1] if reverse else segment
            reversal_count += reverse
        target[indices] = control
        if Counter(episode_lengths(ga)) != Counter(episode_lengths(control)):
            raise AssertionError("episode duration changed")
        exposure_by_symbol[str(symbol)] = {
            "ga_active_bars": int(ga.sum()),
            "control_active_bars": int(control.sum()),
            "ga_episode_lengths": sorted(episode_lengths(ga)),
            "control_episode_lengths": sorted(episode_lengths(control)),
        }
    schedule = base[["execution_timestamp", "signal_timestamp", "symbol"]].copy()
    schedule["target"] = target
    schedule["model_sha256"] = [object_sha({
        "policy": "EMA_long_episode_reversal", "seed": seed,
        "source_model_sha256": source}) for source in base.ema_model_sha256]
    return schedule, {
        "seed": seed, "opportunity_episodes": opportunity_count,
        "reversed_episodes": reversal_count,
        "active_bars_equal": all(
            row["ga_active_bars"] == row["control_active_bars"]
            for row in exposure_by_symbol.values()),
        "ga_active_bars": int(original.sum()),
        "control_active_bars": int(target.sum()),
        "by_symbol": exposure_by_symbol,
    }
