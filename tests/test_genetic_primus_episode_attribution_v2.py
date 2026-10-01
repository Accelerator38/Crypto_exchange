"""Pure offline controls: no market, network, collector or runtime."""

import unittest

import numpy as np
import pandas as pd

from exia.genetic_primus.episode_attribution_v1 import episode_lengths
from exia.genetic_primus.episode_attribution_v2 import _place, placed_episode_control


class EpisodePlacementTests(unittest.TestCase):
    def test_exact_lengths_and_opportunity_constraint(self):
        eligible = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], bool)
        placed = _place(eligible, (2, 3), np.random.default_rng(41), 50000)
        self.assertEqual(sorted(episode_lengths(placed)), [2, 3])
        self.assertFalse((placed.astype(bool) & ~eligible).any())

    def test_repeatable_and_conserved_by_symbol(self):
        n = 20
        base = pd.DataFrame({
            "execution_timestamp": np.tile(np.arange(n), 2),
            "signal_timestamp": np.tile(np.arange(n) - 1, 2),
            "symbol": ["A"] * n + ["B"] * n,
            "proposal": [1] * 40,
            "ga_action": ([0, 1, 1, 0, 0, 1, 1, 1] + [0] * 12) * 2,
            "ema_model_sha256": ["a" * 64] * 40,
        })
        first, evidence = placed_episode_control(base, 17)
        second, _ = placed_episode_control(base, 17)
        self.assertTrue(first.equals(second))
        self.assertTrue(evidence["active_bars_equal"])
        self.assertEqual(evidence["ga_active_bars"], 10)
        for row in evidence["by_symbol"].values():
            self.assertEqual(row["ga_episode_lengths"], row["control_episode_lengths"])


if __name__ == "__main__":
    unittest.main()
