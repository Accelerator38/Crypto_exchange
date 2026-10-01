"""Outcome-blind placement of complete GA episodes into EMA opportunities."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .episode_attribution_v1 import episode_lengths
from .measurement_lab_v2 import object_sha


def _place(eligible: np.ndarray, lengths: tuple[int, ...],
           rng: np.random.Generator, max_nodes: int) -> np.ndarray:
    """Find an exact, nonadjacent placement; fail closed at the node budget."""
    eligible = np.asarray(eligible, dtype=bool)
    n = len(eligible)
    ordered = sorted(lengths, reverse=True)
    candidates = []
    for length in ordered:
        starts = np.array([start for start in range(n - length + 1)
                           if eligible[start:start + length].all()], dtype=int)
        candidates.append(rng.permutation(starts).tolist())
    chosen: list[tuple[int, int]] = []
    nodes = 0

    def search(depth: int) -> bool:
        nonlocal nodes
        if depth == len(ordered):
            return True
        length = ordered[depth]
        for start in candidates[depth]:
            nodes += 1
            if nodes > max_nodes:
                raise RuntimeError("episode placement search budget exhausted")
            end = start + length
            if any(start <= other_end and other_start <= end
                   for other_start, other_end in chosen):
                continue
            chosen.append((start, end))
            if search(depth + 1):
                return True
            chosen.pop()
        return False

    if not search(0):
        raise RuntimeError("no eligible exact episode placement")
    output = np.zeros(n, dtype=np.int8)
    for start, end in chosen:
        output[start:end] = 1
    if Counter(episode_lengths(output)) != Counter(lengths):
        raise AssertionError("episode multiset changed")
    return output


def placed_episode_control(base: pd.DataFrame, seed: int,
                           max_nodes: int = 50000) -> tuple[pd.DataFrame, dict]:
    if type(seed) is not int or seed < 0 or type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("invalid seed or node budget")
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
    by_symbol = {}
    for symbol, group in base.groupby("symbol", sort=True):
        ordered = group.sort_values("execution_timestamp", kind="stable")
        indices = ordered.index.to_numpy(int)
        eligible = ordered.proposal.to_numpy(int) == 1
        ga = ordered.ga_action.to_numpy(int) == 1
        if (ga & ~eligible).any():
            raise ValueError("GA action outside EMA opportunity")
        original[indices] = ga.astype(np.int8)
        lengths = episode_lengths(ga)
        control = _place(eligible, lengths, rng, max_nodes)
        target[indices] = control
        by_symbol[str(symbol)] = {
            "ga_active_bars": int(ga.sum()), "control_active_bars": int(control.sum()),
            "ga_episode_lengths": sorted(lengths),
            "control_episode_lengths": sorted(episode_lengths(control)),
            "changed_rows": int(np.count_nonzero(ga != control)),
        }
    schedule = base[["execution_timestamp", "signal_timestamp", "symbol"]].copy()
    schedule["target"] = target
    schedule["model_sha256"] = [object_sha({
        "policy": "episode_placement_in_EMA_long_opportunities", "seed": seed,
        "source_model_sha256": source}) for source in base.ema_model_sha256]
    return schedule, {
        "seed": seed, "changed_rows": int(np.count_nonzero(original != target)),
        "active_bars_equal": all(r["ga_active_bars"] == r["control_active_bars"]
                                 for r in by_symbol.values()),
        "ga_active_bars": int(original.sum()),
        "control_active_bars": int(target.sum()),
        "by_symbol": by_symbol,
    }
