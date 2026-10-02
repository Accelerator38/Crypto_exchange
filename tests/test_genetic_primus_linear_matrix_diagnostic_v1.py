"""Pure train-only diagnostic regressions; no snapshot, network or collector."""

import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.diagnose_genetic_primus_linear_matrix_v1 import (
    _decision_rows, _forecast_summary,
)


class FrozenDiagnosticTests(unittest.TestCase):
    def test_forecast_summary_keeps_abstained_losses_and_wins(self):
        rows = pd.DataFrame([
            {"origin_start": 0, "symbol": "A", "proposal_long": True,
             "forward_gross_3d": -0.03, "forecast": 0.02,
             "threshold": 0.0016, "target": 1, "position_before": 0},
            {"origin_start": 0, "symbol": "B", "proposal_long": True,
             "forward_gross_3d": 0.01, "forecast": -0.01,
             "threshold": 0.0016, "target": 0, "position_before": 0},
            {"origin_start": 0, "symbol": "C", "proposal_long": False,
             "forward_gross_3d": 0.02, "forecast": None,
             "threshold": 0.0016, "target": 0, "position_before": 0},
        ])
        report = _forecast_summary(rows)
        self.assertEqual(report["all_proposals"]["count"], 2)
        self.assertEqual(report["active"]["count"], 1)
        self.assertEqual(report["abstained"]["positive_count"], 1)
        self.assertAlmostEqual(report["active"]["mean_gross_3d"], -0.03)

    def test_decision_join_rejects_missing_symbol(self):
        opportunities = pd.DataFrame([
            {"origin_start": 0, "symbol": "A", "proposal_long": True,
             "forward_gross_3d": 0.01},
            {"origin_start": 0, "symbol": "B", "proposal_long": False,
             "forward_gross_3d": 0.02},
        ])
        ledger = [{"origin_start": 0, "decision": {"decisions": {
            "A": {"target": 1, "position_before": 0,
                  "predicted_gross_3d": 0.03, "incremental_threshold": 0.0016}}}}]
        with self.assertRaises(ValueError):
            _decision_rows(ledger, opportunities)


if __name__ == "__main__":
    unittest.main()
