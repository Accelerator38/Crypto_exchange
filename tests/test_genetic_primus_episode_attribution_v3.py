"""Pure offline checks of the constrained episode geometry."""

import unittest

import numpy as np
import pandas as pd

from exia.genetic_primus.episode_attribution_v1 import episode_lengths
from exia.genetic_primus.episode_attribution_v3 import (
    EpisodePlacementProblem, outcome_blind_bank,
)


class EpisodeGeometryTests(unittest.TestCase):
    def test_maximum_diversity_with_exact_duration_and_exposure(self):
        eligible = np.array([1, 1, 1, 0, 1, 1, 1], bool)
        ga = np.array([1, 1, 0, 0, 0, 1, 1], bool)
        problem = EpisodePlacementProblem(eligible, ga)
        self.assertEqual(problem.maximum_changed_rows, 4)
        for seed in (1, 9, 27):
            mask = problem.sample_maximally_changed(seed)
            self.assertEqual(int(mask.sum()), 4)
            self.assertEqual(sorted(episode_lengths(mask)), [2, 2])
            self.assertEqual(int(np.count_nonzero(mask != ga)), 4)
            self.assertFalse((mask.astype(bool) & ~eligible).any())

    def test_full_coverage_has_no_counterfactual(self):
        ga = np.array([1, 1, 0, 1, 1], bool)
        problem = EpisodePlacementProblem(ga, ga)
        self.assertEqual(problem.maximum_changed_rows, 0)
        self.assertTrue(np.array_equal(problem.sample_maximally_changed(11), ga))

    def test_schedule_bank_preserves_per_symbol_episode_multisets(self):
        base = pd.DataFrame({
            "execution_timestamp": np.arange(7),
            "signal_timestamp": np.arange(7) - 1,
            "symbol": ["BTC"] * 7,
            "proposal": [1, 1, 1, 0, 1, 1, 1],
            "ga_action": [1, 1, 0, 0, 0, 1, 1],
            "ema_model_sha256": ["a" * 64] * 7,
        })
        bank, geometry = outcome_blind_bank(base, (11, 23))
        self.assertEqual(geometry["BTC"]["maximum_changed_rows"], 4)
        self.assertEqual(len(bank), 2)
        for schedule, evidence in bank:
            self.assertEqual(evidence["changed_rows"], 4)
            self.assertEqual(evidence["episode_lengths_by_symbol"]["BTC"], [2, 2])
            self.assertEqual(len(schedule), 7)
            self.assertEqual(schedule.model_sha256.nunique(), 1)


if __name__ == "__main__":
    unittest.main()
