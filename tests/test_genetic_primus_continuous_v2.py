"""Deterministic offline regression for origin boundaries and scaler lineage."""

from __future__ import annotations

import json
import sys
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.prospective_continuous_v2 import (
    evaluate_committed_chain, validate_continuous_contract,
)
from exia.genetic_primus.prospective_manifest_v2 import SCHEMA


class ContinuousOriginTests(unittest.TestCase):
    def test_boundary_keeps_position_and_refits_scaler(self) -> None:
        contract = json.loads((ROOT / "configs/genetic_primus_prospective_3d_v1.json").read_text())
        contract["schema_version"] = "exia.genetic_primus.prospective_3d/2"
        contract["universe"] = ["BTC/USDT"]
        contract["training"]["initial_history_start_utc"] = "2026-08-15T00:00:00Z"
        contract["data_commitment"]["manifest_version"] = 2
        contract["model_selection"] = {
            "window": contract["training"]["window"],
            "scaler": contract["training"]["scaler"],
            "report_path": "Reports/Exia/Genetic_Primus/selection.json",
            "report_sha256": "0" * 64,
        }
        contract["accounting"]["terminal_exit"] = "completed_prefix_final_candle_close"
        contract["data_commitment"]["origin_snapshot_commit_rule"] = (
            "after_origin_complete_before_evaluation")
        contract["data_commitment"]["origin_commit_deadline_hours"] = 72
        wrong_window = deepcopy(contract)
        wrong_window["model_selection"]["window"] = "rolling"
        with self.assertRaisesRegex(ValueError, "selection and deployment"):
            validate_continuous_contract(wrong_window)
        start = datetime(2026, 8, 15, tzinfo=timezone.utc)
        timestamps = [int((start + timedelta(hours=4 * i)).timestamp() * 1000)
                      for i in range(23 * 6)]
        panel = pd.DataFrame({"timestamp": timestamps, "symbol": "BTC/USDT",
                              "open": 100.0, "high": 100.0, "low": 100.0,
                              "close": 100.0, "volume": 1.0})
        first_end = int(datetime(2026, 9, 4, tzinfo=timezone.utc).timestamp() * 1000)
        second_end = int(datetime(2026, 9, 7, tzinfo=timezone.utc).timestamp() * 1000)
        inputs = []
        for index, end_ms in enumerate((first_end, second_end)):
            prefix = panel.loc[panel.timestamp < end_ms].copy()
            manifest = {"schema_version": SCHEMA, "origin_index": index,
                        "metadata": {"coverage_start": timestamps[0],
                                     "coverage_end_exclusive": end_ms},
                        "snapshot_sha256": f"snapshot-{index}"}
            inputs.append((manifest, prefix))

        def simple_features(source: pd.DataFrame) -> pd.DataFrame:
            frame = source.copy()
            frame["ema_gap_12_48"] = 0.1
            frame["ema_slope_12_3"] = 0.1
            frame["return_12"] = 0.1
            frame["efficiency_12"] = 0.5
            return frame

        with patch("exia.genetic_primus.prospective_continuous_v2.build_lab_features",
                   side_effect=simple_features):
            result = evaluate_committed_chain(contract, inputs)
            shared_final_panel = [(manifest, inputs[-1][1]) for manifest, _ in inputs]
            shared_result = evaluate_committed_chain(contract, shared_final_panel)
        ema = result["evaluations"]["normal"]["candidates"]["EMA_TrendConsensus"]
        self.assertEqual(ema["ledger_sha256"], shared_result["evaluations"]["normal"]
                         ["candidates"]["EMA_TrendConsensus"]["ledger_sha256"])
        self.assertEqual(ema["daily_returns"], shared_result["evaluations"]["normal"]
                         ["candidates"]["EMA_TrendConsensus"]["daily_returns"])
        first_ms = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(ema["metrics"]["fills"], 2)
        self.assertEqual(ema["metrics"]["trades"], 1)
        self.assertEqual(ema["trades"][0]["entry_timestamp"], first_ms)
        self.assertEqual(ema["trades"][0]["exit_timestamp"], second_end)
        self.assertEqual(len(ema["daily_returns"]), 6)
        self.assertEqual(len(ema["ledger"]), 36)
        self.assertNotEqual(ema["model_sha256_by_origin"][0],
                            ema["model_sha256_by_origin"][1])
        boundary = [row for row in ema["ledger"]
                    if row["execution_timestamp"] == first_end]
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0]["position_before"], 1)
        self.assertEqual(boundary[0]["action"], 1)


if __name__ == "__main__":
    unittest.main()
