"""Bounded fault injection for final-only runner; synthetic I/O in temp dirs.

Never opens market datasets, starts a collector or evaluates trading returns.
The injected completed clock is test-only; the production CLI has no override.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools import run_genetic_primus_future_ema_viability_v1 as runner


class RunnerFaultTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.dest = self.root / "analysis"
        for name, value in {
            "ROOT": self.root, "DESTINATION": self.dest,
            "MANIFEST": self.dest / "data_manifest.json",
            "RESERVATION": self.dest / "analysis_reservation.json",
            "RESULT": self.dest / "result.json", "LOCK": self.root / "lock.json",
        }.items():
            self.stack.enter_context(patch.object(runner, name, value))
        runner._write_exclusive(runner.LOCK, {"test_fixture": True})
        runner._write_exclusive(self.root / "source.json", {"test_fixture": True})
        self.c = {
            "future_end_exclusive_utc": "2030-01-02T00:00:00Z",
            "offline_future_segment_start_utc": "2030-01-01T00:00:00Z",
            "universe": ["TEST/USDT"], "required_new_rows": 1,
            "objective_id": "synthetic_fault_injection_not_research",
            "known_revealed_snapshot_sha256": "synthetic-source",
            "known_revealed_manifest_path": "synthetic-source-manifest.json",
            "source_contract_path": "source.json",
            "safety": {"credentials_allowed": False, "orders_enabled": False,
                       "paper": False, "live": False, "promotion_authority": False},
        }
        self.now = datetime(2030, 1, 2, tzinfo=timezone.utc)
        self.frame = pd.DataFrame([{"synthetic": 1.0}])
        self.segment_path = self.root / "synthetic.parquet"
        self.manifest = {
            "schema_version": "exia.genetic_primus.future_ema_segment/1",
            "immutable": True, "objective_id": self.c["objective_id"],
            "config_sha256": runner.CONFIG_SHA,
            "implementation_lock_sha256": runner.file_sha256(runner.LOCK),
            "source_snapshot_sha256": self.c["known_revealed_snapshot_sha256"],
            "safety": self.c["safety"], "outcome_metrics_computed": False,
            "network_used": False,
            "segment": {"path": "synthetic.parquet", "sha256": "synthetic-bytes",
                        "size_bytes": 3, "row_count": 1,
                        "canonical_rows_sha256": runner.object_sha(self.frame.to_dict("records"))},
        }
        self.read = self.stack.enter_context(patch.object(
            runner, "_read_segment", return_value=(self.frame, "synthetic-bytes", 3)))
        self.git = self.stack.enter_context(patch.object(runner, "_require_git_commit"))
        self.history = self.stack.enter_context(patch.object(
            runner, "load_verified_snapshot", return_value=({}, self.frame)))
        self.evaluator = self.stack.enter_context(patch.object(
            runner, "evaluate_registered_panel", return_value={
                "research_status": "SYNTHETIC_ONLY", "gate": {"passed": False}}))

    def prepare(self, manifest=None):
        runner._write_exclusive(runner.MANIFEST, manifest or self.manifest)

    def test_commit_exclusive_and_does_not_evaluate(self):
        result = runner.commit_segment(self.c, self.segment_path, self.now)
        self.assertEqual(result["row_count"], 1)
        self.evaluator.assert_not_called()
        self.history.assert_not_called()
        with self.assertRaises(FileExistsError):
            runner.commit_segment(self.c, self.segment_path, self.now)
        self.assertEqual(self.read.call_count, 1)

    def test_header_mismatch_before_market_read(self):
        wrong = dict(self.manifest, objective_id="wrong")
        self.prepare(wrong)
        with self.assertRaisesRegex(ValueError, "commitment mismatch"):
            runner.evaluate_once(self.c, self.now)
        self.read.assert_not_called()
        self.evaluator.assert_not_called()
        self.assertFalse(runner.RESERVATION.exists())

    def test_changed_bytes_or_rows_never_reach_evaluator(self):
        self.prepare()
        for frame, digest, size in [
            (self.frame, "different-bytes", 3),
            (self.frame, "synthetic-bytes", 4),
            (pd.DataFrame([{"synthetic": 2.0}]), "synthetic-bytes", 3),
        ]:
            with self.subTest(digest=digest, size=size, row=frame.iloc[0, 0]):
                self.read.return_value = frame, digest, size
                with self.assertRaisesRegex(ValueError, "differ from commitment"):
                    runner.evaluate_once(self.c, self.now)
                self.assertFalse(runner.RESERVATION.exists())
        self.evaluator.assert_not_called()
        self.history.assert_not_called()

    def test_success_reserves_before_compute_and_rejects_repeat(self):
        self.prepare()

        def compute(*_):
            self.assertTrue(runner.RESERVATION.exists())
            return {"research_status": "SYNTHETIC_ONLY", "gate": {"passed": False}}

        self.evaluator.side_effect = compute
        result = runner.evaluate_once(self.c, self.now)
        self.assertFalse(result["promotion_authority"])
        self.assertTrue(runner.RESULT.exists())
        with self.assertRaises(FileExistsError):
            runner.evaluate_once(self.c, self.now)
        self.assertEqual(self.evaluator.call_count, 1)
        self.assertEqual(self.read.call_count, 1)

    def test_calculation_failure_keeps_reservation_and_blocks_retry(self):
        self.prepare()
        self.evaluator.side_effect = RuntimeError("injected bounded failure")
        with self.assertRaises(RuntimeError):
            runner.evaluate_once(self.c, self.now)
        failure = json.loads((self.dest / "analysis_failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertTrue(runner.RESERVATION.exists())
        self.assertFalse(runner.RESULT.exists())
        with self.assertRaises(FileExistsError):
            runner.evaluate_once(self.c, self.now)
        self.assertEqual(self.evaluator.call_count, 1)

    def test_nonfinite_result_does_not_leave_partial_result(self):
        self.prepare()
        self.evaluator.return_value["bad_metric"] = float("nan")
        with self.assertRaises(ValueError):
            runner.evaluate_once(self.c, self.now)
        self.assertTrue(runner.RESERVATION.exists())
        self.assertTrue((self.dest / "analysis_failure.json").exists())
        self.assertFalse(runner.RESULT.exists())

    def test_reservation_collision_never_computes(self):
        self.prepare()

        def reserve_during_load(*_, **__):
            runner._write_exclusive(runner.RESERVATION, {"competing_analysis": True})
            return {}, self.frame

        self.history.side_effect = reserve_during_load
        with self.assertRaises(FileExistsError):
            runner.evaluate_once(self.c, self.now)
        self.evaluator.assert_not_called()
        self.assertFalse(runner.RESULT.exists())

    def test_git_requires_exact_committed_manifest_bytes(self):
        self.prepare()
        original = runner._require_git_commit
        # Use the production function captured before the fixture's git mock.
        with patch.object(runner, "_require_git_commit", runner_git_check):
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=runner.MANIFEST.read_bytes())):
                runner._require_git_commit(runner.MANIFEST)
            with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(
                    returncode=0, stdout=b"changed")):
                with self.assertRaises(ValueError):
                    runner._require_git_commit(runner.MANIFEST)
        self.assertIs(runner._require_git_commit, original)

    def test_exclusive_writer_rejects_overwrite_and_nonfinite(self):
        path = self.root / "exclusive.json"
        runner._write_exclusive(path, {"value": 1})
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            runner._write_exclusive(path, {"value": 2})
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        bad = self.root / "not-created.json"
        with self.assertRaises(ValueError):
            runner._write_exclusive(bad, {"value": float("inf")})
        self.assertFalse(bad.exists())

    def test_segment_mutation_during_read_and_outside_path(self):
        # No parquet file: pandas and validator are controlled synthetic stubs.
        segment_reader = runner_segment_read
        with patch.object(runner, "_read_segment", segment_reader):
            with patch.object(runner, "file_sha256", side_effect=["before", "after"]), \
                    patch.object(Path, "stat", return_value=SimpleNamespace(st_size=3)), \
                    patch.object(runner.pd, "read_parquet", return_value=self.frame), \
                    patch.object(runner, "validate_market_grid", return_value=self.frame):
                with self.assertRaisesRegex(ValueError, "changed during validation"):
                    runner._read_segment(self.segment_path, self.c)
        with self.assertRaisesRegex(ValueError, "inside the workspace"):
            runner._inside(self.root.parent / "outside-synthetic.parquet")


# Capture functions before per-test patching; never execute CLI or its data loader.
runner_git_check = runner._require_git_commit
runner_segment_read = runner._read_segment

if __name__ == "__main__":
    unittest.main()
