"""Pure local one-bar-delay regressions; no market snapshot or collector."""

import unittest

import pandas as pd

from exia.genetic_primus.latency_sensitivity_v1 import delay_target_schedule
from exia.genetic_primus.rolling_linear_retrain_v1 import DT


class DelayScheduleTests(unittest.TestCase):
    def setUp(self):
        self.start = 10 * DT
        self.end = self.start + 4 * DT
        self.symbols = ("A", "B")
        self.schedule = pd.DataFrame([
            {"execution_timestamp": self.start + index * DT, "symbol": symbol,
             "signal_timestamp": self.start if index < 2 else self.start + 2 * DT,
             "target": int((symbol == "A") == (index < 2)),
             "model_sha256": "old" if index < 2 else "new"}
            for index in range(4) for symbol in self.symbols])

    def test_one_bar_delay_carries_old_state_over_retrain_boundary(self):
        delayed = delay_target_schedule(
            self.schedule, self.symbols, self.start, self.end, 1)
        a = delayed.loc[delayed.symbol.eq("A")].reset_index(drop=True)
        b = delayed.loc[delayed.symbol.eq("B")].reset_index(drop=True)
        self.assertEqual(a.target.tolist(), [0, 1, 1, 0])
        self.assertEqual(b.target.tolist(), [0, 0, 0, 1])
        self.assertEqual(a.signal_timestamp.iloc[2], self.start)
        self.assertEqual(a.signal_timestamp.iloc[3], self.start + 2 * DT)
        self.assertEqual(a.model_sha256.iloc[2], "old")
        self.assertEqual(a.model_sha256.iloc[3], "new")

    def test_zero_delay_preserves_registered_targets(self):
        replay = delay_target_schedule(
            self.schedule, self.symbols, self.start, self.end, 0)
        self.assertEqual(replay.target.tolist(), self.schedule.target.tolist())
        self.assertEqual(replay.signal_timestamp.tolist(),
                         self.schedule.signal_timestamp.tolist())

    def test_incomplete_grid_fails_closed(self):
        with self.assertRaises(ValueError):
            delay_target_schedule(
                self.schedule.iloc[:-1], self.symbols, self.start, self.end, 1)


if __name__ == "__main__":
    unittest.main()
