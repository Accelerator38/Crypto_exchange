"""Outcome-blind exact placement of GA episodes within EMA opportunities."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .episode_attribution_v1 import episode_lengths
from .measurement_lab_v2 import object_sha


class EpisodePlacementProblem:
    """Binary placement with exact per-symbol episode lengths and bar exposure."""

    def __init__(self, eligible: np.ndarray, ga: np.ndarray, *, time_limit_seconds: float = 5.0):
        self.eligible = np.asarray(eligible, dtype=bool)
        self.ga = np.asarray(ga, dtype=bool)
        if (self.eligible.ndim != 1 or self.ga.shape != self.eligible.shape
                or (self.ga & ~self.eligible).any()
                or not np.isfinite(time_limit_seconds) or time_limit_seconds <= 0):
            raise ValueError("invalid eligible/GA masks or solve limit")
        self.lengths = tuple(sorted(episode_lengths(self.ga), reverse=True))
        self.time_limit_seconds = float(time_limit_seconds)
        self.active = int(self.ga.sum())
        if self.active == 0 or self.active == int(self.eligible.sum()):
            self.candidates = []
            self.minimum_overlap = self.active
            self.maximum_changed_rows = 0
            return
        self.candidates: list[tuple[int, int, int]] = []
        n = len(self.eligible)
        for episode, length in enumerate(self.lengths):
            for start in range(n - length + 1):
                if self.eligible[start:start + length].all():
                    self.candidates.append((episode, start, length))
        if not self.candidates:
            raise ValueError("no valid episode placement")
        k = len(self.candidates)
        rows = []
        lower = []
        upper = []
        for episode in range(len(self.lengths)):
            row = np.fromiter((int(item[0] == episode) for item in self.candidates), float)
            rows.append(row)
            lower.append(1.0)
            upper.append(1.0)
        # A two-bar window can contain at most one selected episode. A single
        # episode may cover both bars but contributes only one binary variable.
        for bar in range(n - 1):
            row = np.fromiter((int(start <= bar + 1 and start + length - 1 >= bar)
                               for _, start, length in self.candidates), float)
            if row.any():
                rows.append(row)
                lower.append(-np.inf)
                upper.append(1.0)
        # Equal-length episodes are indistinguishable. Order their starts to
        # remove pure solver symmetry without excluding a distinct mask.
        for episode in range(len(self.lengths) - 1):
            if self.lengths[episode] == self.lengths[episode + 1]:
                row = np.fromiter((
                    float(start) if index == episode else
                    -float(start) if index == episode + 1 else 0.0
                    for index, start, _ in self.candidates), float)
                rows.append(row)
                lower.append(-np.inf)
                upper.append(-1.0)
        self.constraints = [LinearConstraint(np.vstack(rows), lower, upper)]
        self.overlap = np.fromiter((
            int(self.ga[start:start + length].sum())
            for _, start, length in self.candidates), float)
        optimum = self._solve(self.overlap)
        self.minimum_overlap = int(round(float(self.overlap @ optimum)))
        self.maximum_changed_rows = 2 * (self.active - self.minimum_overlap)
        self.constraints.append(LinearConstraint(
            self.overlap.reshape(1, k), self.minimum_overlap, self.minimum_overlap))

    def _solve(self, objective: np.ndarray) -> np.ndarray:
        result = milp(objective, integrality=np.ones(len(objective), dtype=int),
                      bounds=Bounds(0, 1), constraints=self.constraints,
                      options={"time_limit": self.time_limit_seconds, "mip_rel_gap": 0.0})
        if result.status != 0 or result.x is None:
            raise RuntimeError(f"exact episode placement unresolved: status={result.status}")
        chosen = np.rint(result.x).astype(int)
        if not np.allclose(chosen, result.x, atol=1e-6):
            raise AssertionError("noninteger episode placement")
        return chosen

    def sample_maximally_changed(self, seed: int) -> np.ndarray:
        if type(seed) is not int or seed < 0:
            raise ValueError("nonnegative integer seed required")
        if self.maximum_changed_rows == 0:
            return self.ga.astype(np.int8)
        rng = np.random.default_rng(seed)
        chosen = self._solve(rng.uniform(-1.0, 1.0, len(self.candidates)))
        mask = np.zeros(len(self.eligible), dtype=np.int8)
        for selected, (_, start, length) in zip(chosen, self.candidates):
            if selected:
                mask[start:start + length] = 1
        if (Counter(episode_lengths(mask)) != Counter(self.lengths)
                or int(mask.sum()) != self.active
                or (mask.astype(bool) & ~self.eligible).any()
                or int(np.count_nonzero(mask != self.ga)) != self.maximum_changed_rows):
            raise AssertionError("exact placement conservation failed")
        return mask


def outcome_blind_bank(base: pd.DataFrame, seeds: tuple[int, ...], *,
                       time_limit_seconds: float = 5.0) -> tuple[list[tuple[pd.DataFrame, dict]], dict]:
    required = {"execution_timestamp", "signal_timestamp", "symbol", "proposal",
                "ga_action", "ema_model_sha256"}
    if (not required.issubset(base) or base.empty or not seeds
            or len(set(seeds)) != len(seeds)
            or base.duplicated(["execution_timestamp", "symbol"]).any()
            or not np.array_equal(base.index.to_numpy(), np.arange(len(base)))):
        raise ValueError("invalid aligned archived decisions or seed bank")
    records = []
    geometry = {}
    controls = {seed: np.zeros(len(base), dtype=np.int8) for seed in seeds}
    original = np.zeros(len(base), dtype=np.int8)
    for symbol_index, (symbol, group) in enumerate(base.groupby("symbol", sort=True)):
        ordered = group.sort_values("execution_timestamp", kind="stable")
        positions = ordered.index.to_numpy(int)
        ga = ordered.ga_action.to_numpy(int) == 1
        problem = EpisodePlacementProblem(
            ordered.proposal.to_numpy(int) == 1, ga,
            time_limit_seconds=time_limit_seconds)
        original[positions] = ga.astype(np.int8)
        geometry[str(symbol)] = {
            "eligible_bars": int(problem.eligible.sum()),
            "ga_active_bars": problem.active,
            "episode_lengths": sorted(problem.lengths),
            "maximum_changed_rows": problem.maximum_changed_rows,
            "minimum_overlap_bars": problem.minimum_overlap,
        }
        for seed in seeds:
            sample_seed = int(np.random.SeedSequence([seed, symbol_index]).generate_state(1)[0])
            controls[seed][positions] = problem.sample_maximally_changed(sample_seed)
    for seed in seeds:
        target = controls[seed]
        schedule = base[["execution_timestamp", "signal_timestamp", "symbol"]].copy()
        schedule["target"] = target
        schedule["model_sha256"] = [object_sha({
            "policy": "maximally_changed_episode_placement", "seed": seed,
            "source_model_sha256": source}) for source in base.ema_model_sha256]
        records.append((schedule, {
            "seed": seed, "changed_rows": int(np.count_nonzero(target != original)),
            "active_bars": int(target.sum()),
            "episode_lengths_by_symbol": {
                symbol: sorted(episode_lengths(target[group.sort_values(
                    "execution_timestamp", kind="stable").index.to_numpy(int)]))
                for symbol, group in base.groupby("symbol", sort=True)},
        }))
    return records, geometry
