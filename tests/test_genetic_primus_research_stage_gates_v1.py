"""A good-looking train record cannot bypass the inner validation gate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.research_stage_gates_v1 import (
    REQUIRED_CHECKS, REQUIRED_SAFETY, stage_gates,
)


class StageGateTests(unittest.TestCase):
    def test_disclosed_failed_results_keep_ga_and_outer_closed(self) -> None:
        import json
        p = ROOT / "Reports/Exia/Genetic_Primus"
        episode = json.loads((p / "episode_attribution_v1_20261001/result.json").read_text())
        linear = json.loads((p / "rolling_linear_development_v1_20261001/result.json").read_text())
        state = stage_gates(episode, linear)
        self.assertFalse(state["ga_search_allowed_under_current_plan"])
        self.assertFalse(state["linear_candidate_can_be_considered_for_future_contract"])

    def test_incomplete_validation_cannot_pass_a_nominal_success(self) -> None:
        episode = {"schema_version": "exia.genetic_primus.episode_attribution_result/1",
                   "next_search_gate_passed": True,
                   "paired_lcb": {"normal": 0.001, "stress": 0.001}}
        linear = {"schema_version": "exia.genetic_primus.rolling_linear_development_result/1",
                  "final_policy": "linear_overlay_for_future_separate_audit_only",
                  "selected_variant": "uniform",
                  "records": [{"variant": "uniform", "train_score": {"rank_score": 0.001}}],
                  "validation_result": {"variant": "uniform", "gate": {
                      "passed": True, "checks": {key: True for key in REQUIRED_CHECKS}}},
                  "automatic_promotion": False,
                  "safety": {key: False for key in REQUIRED_SAFETY}}
        self.assertTrue(stage_gates(episode, linear)["ga_search_allowed_under_current_plan"])
        del linear["validation_result"]["gate"]["checks"]["stress_net_positive"]
        self.assertFalse(stage_gates(episode, linear)["ga_search_allowed_under_current_plan"])


if __name__ == "__main__":
    unittest.main()
