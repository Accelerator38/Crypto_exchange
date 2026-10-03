"""Small stdlib-only synthetic operations tests; no market datasets or recovery."""

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.future_ema_operations_v1 import (
    MAX_ARTIFACT_BYTES, environment_differences, inspect_artifacts, read_json_artifact, sha_bytes,
)
from tools import check_genetic_primus_future_ema_operations_v1 as checker


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.safety = {"credentials_allowed": False, "orders_enabled": False,
                       "paper": False, "live": False, "promotion_authority": False}
        self.lineage = {"objective_id": "synthetic", "config_sha256": "config",
                        "implementation_lock_sha256": "implementation"}
        self.manifest = {**self.lineage,
                         "schema_version": "exia.genetic_primus.future_ema_segment/1",
                         "immutable": True, "safety": self.safety,
                         "outcome_metrics_computed": False, "network_used": False}

    def write(self, name, value):
        path = self.directory / name
        with path.open("xb") as stream:
            stream.write(json.dumps(value, sort_keys=True).encode("utf-8"))
        return path

    def manifest_and_reservation(self):
        path = self.write("data_manifest.json", self.manifest)
        reservation = {**self.lineage, "data_manifest_sha256": sha_bytes(path.read_bytes()),
                       "analysis_started_at_utc": "2030-01-02T00:00:00+00:00"}
        self.write("analysis_reservation.json", reservation)
        return reservation

    def inspect(self):
        return inspect_artifacts(self.directory, **self.lineage, safety=self.safety)

    def result(self, reservation):
        return {**reservation, "schema_version": "exia.genetic_primus.future_ema_result/1",
                "safety": self.safety, "final_policy": "NoTrade", "automatic_promotion": False,
                "promotion_authority": False, "market_readiness": False, "network_used": False,
                "scenarios": {"synthetic_private_metric": -123.456}}

    def test_environment_exact_match_and_drift(self):
        expected = {"python": {"version": "a"}, "packages": {"numpy": {"version": "1"}, "numexpr": None}}
        self.assertEqual(environment_differences(expected, copy.deepcopy(expected)), [])
        changed = copy.deepcopy(expected)
        changed["python"]["version"] = "b"
        changed["packages"]["numpy"]["version"] = "2"
        changed["packages"]["numexpr"] = {"version": "1"}
        self.assertEqual(environment_differences(expected, changed),
                         ["python.version", "packages.numexpr", "packages.numpy"])

    def test_cli_environment_drift_returns_nonzero_without_launch(self):
        expected = {"python": {"version": "a"}, "packages": {}}
        registration = {"schema_version": "exia.genetic_primus.future_ema_environment/1",
                        "config_sha256": checker.CONFIG_SHA,
                        "implementation_lock_sha256": checker.LOCK_SHA, "snapshot": expected}
        output = io.StringIO()
        with patch.object(sys, "argv", ["operations", "environment"]), \
                patch.object(checker, "read_json_artifact", side_effect=[
                    ({"frozen_code_sha256": {}}, checker.CONFIG_SHA),
                    ({"files_sha256": {}}, checker.LOCK_SHA),
                    (registration, checker.ENVIRONMENT_SHA)]), \
                patch.object(checker, "environment_snapshot", return_value={
                    "python": {"version": "b"}, "packages": {}}), redirect_stdout(output):
            self.assertTrue(checker.main())
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "ENVIRONMENT_DRIFT_BLOCKED")
        self.assertFalse(value["runner_launched"])

    def test_cli_changed_environment_registration_is_rejected(self):
        with patch.object(sys, "argv", ["operations", "environment"]), \
                patch.object(checker, "read_json_artifact", side_effect=[
                    ({"frozen_code_sha256": {}}, checker.CONFIG_SHA),
                    ({"files_sha256": {}}, checker.LOCK_SHA),
                    ({}, "changed-bytes")]), \
                patch.object(checker, "environment_snapshot") as snapshot:
            with self.assertRaisesRegex(ValueError, "registration lineage differs"):
                checker.main()
            snapshot.assert_not_called()

    def test_no_artifacts_is_read_only(self):
        result = self.inspect()
        self.assertEqual(result["status"], "AWAITING_INPUT_COMMIT")
        self.assertFalse(result["rerun_authorized"])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_manifest_only_does_not_claim_git_verified(self):
        self.write("data_manifest.json", self.manifest)
        result = self.inspect()
        self.assertEqual(result["status"], "INPUT_MANIFEST_PRESENT_COMMIT_NOT_VERIFIED")
        self.assertFalse(result["git_commit_verified"])

    def test_interrupted_analysis_is_blocked_without_mutation(self):
        self.manifest_and_reservation()
        before = {p.name: p.read_bytes() for p in self.directory.iterdir()}
        result = self.inspect()
        self.assertEqual(result["status"], "ANALYSIS_STARTED_OR_INTERRUPTED_NO_RETRY")
        self.assertFalse(result["rerun_authorized"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.directory.iterdir()})

    def test_failure_does_not_authorize_retry(self):
        reservation = self.manifest_and_reservation()
        self.write("analysis_failure.json", {**reservation, "exception_type": "RuntimeError",
                                              "promotion_authority": False})
        result = self.inspect()
        self.assertEqual(result["status"], "ANALYSIS_FAILED_BLOCKED")
        self.assertFalse(result["rerun_authorized"])

    def test_result_never_discloses_metrics(self):
        reservation = self.manifest_and_reservation()
        self.write("result.json", self.result(reservation))
        result = self.inspect()
        self.assertEqual(result["status"], "FINAL_RESULT_PRESENT_NO_REANALYSIS")
        self.assertNotIn("123.456", json.dumps(result))
        self.assertNotIn("scenarios", result)
        self.assertFalse(result["metrics_recomputed"])

    def test_result_and_failure_coexistence_is_blocked(self):
        reservation = self.manifest_and_reservation()
        self.write("result.json", self.result(reservation))
        self.write("analysis_failure.json", {**reservation, "exception_type": "RuntimeError",
                                              "promotion_authority": False})
        result = self.inspect()
        self.assertEqual(result["status"], "MANUAL_REVIEW_BLOCKED")
        self.assertIn("result_and_failure_coexist", result["issues"])

    def test_orphan_and_unsafe_result_are_blocked(self):
        result = self.result({**self.lineage, "data_manifest_sha256": "absent",
                              "analysis_started_at_utc": "2030-01-02T00:00:00Z"})
        result["promotion_authority"] = True
        self.write("result.json", result)
        summary = self.inspect()
        self.assertEqual(summary["status"], "MANUAL_REVIEW_BLOCKED")
        self.assertIn("missing_reservation", summary["issues"])
        self.assertIn("result.json:unsafe_authority", summary["issues"])

    def test_mutated_manifest_hash_is_blocked(self):
        reservation = self.manifest_and_reservation()
        self.write("analysis_failure.json", {**reservation, "exception_type": "RuntimeError",
                                              "promotion_authority": False})
        with (self.directory / "data_manifest.json").open("ab") as stream:
            stream.write(b"\n")
        result = self.inspect()
        self.assertEqual(result["status"], "MANUAL_REVIEW_BLOCKED")
        self.assertIn("analysis_reservation.json:manifest_sha_mismatch", result["issues"])

    def test_invalid_timestamp_is_blocked(self):
        path = self.write("data_manifest.json", self.manifest)
        self.write("analysis_reservation.json", {**self.lineage,
            "data_manifest_sha256": sha_bytes(path.read_bytes()),
            "analysis_started_at_utc": "2030-01-02T03:00:00+03:00"})
        self.assertIn("analysis_reservation.json:invalid_analysis_timestamp", self.inspect()["issues"])

    def test_json_rejects_truncation_duplicates_nonfinite_and_nonobject(self):
        for i, raw in enumerate((b'{"a":', b'{"a":1,"a":2}', b'{"a":NaN}',
                                 b'{"a":1e999}', b'[]')):
            with self.subTest(raw=raw):
                path = self.directory / f"bad-{i}.json"
                with path.open("xb") as stream:
                    stream.write(raw)
                with self.assertRaises(ValueError):
                    read_json_artifact(path)
        with (self.directory / "result.json").open("xb") as stream:
            stream.write(b'{"a":')
        self.assertEqual(self.inspect()["status"], "MANUAL_REVIEW_BLOCKED")

    def test_oversized_artifact_is_rejected_before_read(self):
        path = self.directory / "large.json"
        with patch.object(Path, "stat") as stat_mock, patch.object(Path, "open") as open_mock:
            stat_mock.return_value.st_size = MAX_ARTIFACT_BYTES + 1
            with self.assertRaisesRegex(ValueError, "oversized"):
                read_json_artifact(path)
            open_mock.assert_not_called()

    def test_artifact_symlink_is_rejected_without_read(self):
        with patch.object(Path, "is_symlink", return_value=True), \
                patch("exia.genetic_primus.future_ema_operations_v1.read_json_artifact") as read:
            result = self.inspect()
            self.assertEqual(result["status"], "MANUAL_REVIEW_BLOCKED")
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
