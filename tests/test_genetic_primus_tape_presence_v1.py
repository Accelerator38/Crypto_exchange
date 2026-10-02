"""Pure local checks for frozen transition-to-tape requirements."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exia.genetic_primus.rolling_linear_retrain_v1 import DT, HORIZON_MS
from tools.check_genetic_primus_tape_presence_v1 import _events, _tape_name


class TapePresenceTests(unittest.TestCase):
    def test_entry_exit_dates_shift_without_creating_trades(self):
        start = 1780272000000
        ledger = []
        for index in range(20):
            target = int(index == 1)
            ledger.append({"origin_start": start + index * HORIZON_MS,
                           "decision": {"decisions": {"BTC/USDT": {
                               "target": target,
                               "predicted_gross_3d": 0.01 if target else None}}}})
        source = {"records": [
            {"feature_set": "two", "variant": "uniform", "origin_models": ledger},
            {"feature_set": "four", "variant": "uniform", "origin_models": ledger}]}
        zero = _events(source, "linear_four_uniform", 0)
        late = _events(source, "linear_four_uniform", 1)
        self.assertEqual([x["action"] for x in zero], ["enter", "exit"])
        self.assertEqual([x["action"] for x in late], ["enter", "exit"])
        self.assertEqual(len(zero), len(late))
        self.assertNotEqual(zero[0]["timestamp_utc"], late[0]["timestamp_utc"])
        self.assertEqual(_tape_name("book_ticker", zero[0]["day_utc"], "BTC/USDT"),
                         f"bitget-futures_book_ticker_{zero[0]['day_utc']}_BTCUSDT.csv.gz")
        self.assertEqual(DT, 14_400_000)

    def test_open_terminal_position_fails_closed(self):
        start = 1780272000000
        ledger = [{"origin_start": start + index * HORIZON_MS,
                   "decision": {"decisions": {"BTC/USDT": {
                       "target": 1, "predicted_gross_3d": 0.01}}}}
                  for index in range(20)]
        source = {"records": [{"feature_set": "four", "variant": "uniform",
                               "origin_models": ledger}]}
        with self.assertRaises(ValueError):
            _events(source, "linear_four_uniform", 0)


if __name__ == "__main__":
    unittest.main()
