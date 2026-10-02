"""Pure offline checks for the independent timing-audit arithmetic."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_genetic_primus_linear_timing_v1 import _manual_features, _require_close


class TimingAuditTests(unittest.TestCase):
    def test_manual_feature_is_unchanged_by_future_bars(self):
        data = pd.DataFrame({"timestamp": range(20),
                             "close": [100.0 + index for index in range(20)]})
        before = _manual_features(data)[12]
        mutated = data.copy()
        mutated.loc[mutated.timestamp.gt(12), "close"] += 1000.0
        after = _manual_features(mutated)[12]
        self.assertEqual(before, after)
        self.assertAlmostEqual(before["return_12"], 0.12)
        self.assertAlmostEqual(before["efficiency_12"], 1.0)

    def test_numeric_comparison_fails_closed(self):
        self.assertEqual(_require_close(1.0, 1.0, "test"), 0.0)
        with self.assertRaises(AssertionError):
            _require_close(1.0, 1.01, "test")


if __name__ == "__main__":
    unittest.main()
