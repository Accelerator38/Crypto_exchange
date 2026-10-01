"""Deterministic admission checks; no collector, market data or network."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.prospective_admission_v1 import (
    REQUIRED_SAFETY_FLAGS, REQUIRED_VALIDATION_CHECKS, admission_reasons,
)

GENOME = [1, 0, 0, -1, 0]


def _passing_report() -> dict:
    return {
        "schema_version": "exia.genetic_primus.development_selection_result/1",
        "evidence_class": "revealed_historical_development_only",
        "final_policy": "selected_sparse_overlay_for_future_separate_audit_only",
        "selected_genome": GENOME,
        "candidate_budget": 24, "search_evaluations": 1,
        "candidate_records": [{"genome": GENOME, "training": {"rank_score": 0.001}}],
        "validation_result": {"genome": GENOME, "gate": {
            "passed": True, "checks": {key: True for key in REQUIRED_VALIDATION_CHECKS}}},
        "automatic_promotion": False,
        "safety": {key: False for key in REQUIRED_SAFETY_FLAGS},
    }


class ProspectiveAdmissionTests(unittest.TestCase):
    def test_exact_passed_selection_is_registration_ready_only(self) -> None:
        self.assertEqual(admission_reasons(_passing_report(), GENOME), [])

    def test_failed_validation_cannot_be_registered_despite_selected_genome(self) -> None:
        report = _passing_report()
        report["validation_result"]["gate"]["passed"] = False
        self.assertIn("VALIDATION_GATE_FAILED", admission_reasons(report, GENOME))
        report["validation_result"]["gate"]["passed"] = True
        del report["validation_result"]["gate"]["checks"]["stress_net_positive"]
        self.assertIn("VALIDATION_GATE_FAILED", admission_reasons(report, GENOME))

    def test_disclosed_no_trade_result_is_not_a_prospective_challenger(self) -> None:
        path = ROOT / "Reports/Exia/Genetic_Primus/selection_development_v1_20261001/result.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        reasons = admission_reasons(report, report["best_train_genome_even_if_ineligible"])
        self.assertIn("NO_VALIDATED_CHALLENGER", reasons)
        self.assertIn("GENOME_NOT_SELECTED", reasons)
        self.assertIn("VALIDATION_MISSING_OR_MISMATCHED", reasons)

    def test_mismatched_genome_or_nonpositive_training_is_rejected(self) -> None:
        report = _passing_report()
        self.assertIn("GENOME_NOT_SELECTED", admission_reasons(report, [0, 0, 0, 0, 0]))
        report = copy.deepcopy(report)
        report["candidate_records"][0]["training"]["rank_score"] = float("nan")
        self.assertIn("TRAIN_SELECTION_NOT_POSITIVE", admission_reasons(report, GENOME))


if __name__ == "__main__":
    unittest.main()
