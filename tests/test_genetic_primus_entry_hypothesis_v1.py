"""Causal feature and shared-exit checks without network or collector."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.entry_hypothesis_v1 import causal_entry_features, entry_schedule
from exia.genetic_primus.measurement_lab_v2 import build_lab_features

DT = 4 * 60 * 60 * 1000


class EntryHypothesisTests(unittest.TestCase):
    def test_features_match_existing_builder_and_are_prefix_causal(self) -> None:
        close = 100.0 + np.arange(80) * 0.2 + np.sin(np.arange(80) / 3.0)
        panel = pd.DataFrame({
            "timestamp": np.arange(80, dtype=np.int64) * DT,
            "symbol": ["BTC/USDT"] * 80,
            "open": close * 0.999, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": [1000.0] * 80,
        })
        full = causal_entry_features(panel, DT)
        prefix = causal_entry_features(panel.iloc[:60], DT)
        pd.testing.assert_frame_equal(prefix, full.iloc[:60].reset_index(drop=True))
        old = build_lab_features(panel)
        for new, previous in (("return_12", "return_12"),
                              ("volatility_6", "realized_vol_6"),
                              ("efficiency_12", "efficiency_12")):
            self.assertTrue(np.allclose(full[new], old[previous],
                                        equal_nan=True, rtol=1e-12, atol=1e-12))

    def test_entry_filter_does_not_force_exit(self) -> None:
        base = pd.DataFrame({
            "execution_timestamp": np.arange(4, dtype=np.int64) * DT,
            "signal_timestamp": np.arange(4, dtype=np.int64) * DT,
            "symbol": ["BTC/USDT"] * 4,
            "origin_id": ["origin_01"] * 4,
            "proposal": [1, 1, 1, 0],
            "ema_model_sha256": ["ema"] * 4,
        })
        features = pd.DataFrame({
            "execution_timestamp": np.arange(4, dtype=np.int64) * DT,
            "symbol": ["BTC/USDT"] * 4,
            "return_12": [0.1, 0.001, 0.001, 0.0],
            "volatility_6": [0.01] * 4,
            "efficiency_12": [0.7, 0.4, 0.4, 0.4],
        })
        schedule, diagnostics = entry_schedule(base, features)
        self.assertEqual(diagnostics["can_enter"].tolist(), [True, False, False, False])
        self.assertEqual(schedule["target"].tolist(), [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
