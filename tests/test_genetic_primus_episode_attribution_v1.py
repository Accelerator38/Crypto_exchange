"""Conservation checks for a retrospective episode-matched control."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.episode_attribution_v1 import reversed_episode_control

DT = 14_400_000


def rows() -> pd.DataFrame:
    data = []
    masks = {"A": [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
             "B": [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0]}
    for index in range(12):
        for symbol in ("A", "B"):
            data.append({"execution_timestamp": index * DT,
                         "signal_timestamp": index * DT,
                         "symbol": symbol,
                         "proposal": 1 if index < 6 or index >= 7 else 0,
                         "ga_action": masks[symbol][index],
                         "ema_model_sha256": "a" * 64})
    return pd.DataFrame(data)


class EpisodeAttributionTests(unittest.TestCase):
    def test_reversal_keeps_symbol_exposure_and_holding_lengths(self) -> None:
        base = rows()
        for seed in (11, 23, 47):
            schedule, details = reversed_episode_control(base, seed)
            self.assertTrue(details["active_bars_equal"])
            self.assertEqual(details["ga_active_bars"], details["control_active_bars"])
            self.assertEqual(len(schedule), len(base))
            for symbol, metrics in details["by_symbol"].items():
                self.assertEqual(metrics["ga_episode_lengths"],
                                 metrics["control_episode_lengths"])
                group = schedule.loc[schedule.symbol == symbol]
                self.assertFalse(group.loc[base.loc[base.symbol == symbol, "proposal"].eq(0).to_numpy(), "target"].any())

    def test_invalid_ga_action_outside_opportunity_is_rejected(self) -> None:
        base = rows()
        base.loc[(base.symbol == "A") & (base.execution_timestamp == 6 * DT), "ga_action"] = 1
        with self.assertRaisesRegex(ValueError, "outside EMA long"):
            reversed_episode_control(base, 11)


if __name__ == "__main__":
    unittest.main()
