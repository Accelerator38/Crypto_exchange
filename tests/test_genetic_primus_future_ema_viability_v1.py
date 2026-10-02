"""Pure fail-closed checks for the registered future-only research gate."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_genetic_primus_future_ema_viability_v1 import validate_contract


class FutureEMAGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/genetic_primus_future_ema_viability_v1.json")
                                .read_text(encoding="utf-8"))
        cls.manifest = json.loads((ROOT / cls.config["known_revealed_manifest_path"])
                                  .read_text(encoding="utf-8"))
        cls.contract = json.loads((ROOT / cls.config["source_contract_path"])
                                  .read_text(encoding="utf-8"))

    def test_registered_contract_is_pending_without_outcomes(self):
        status = validate_contract(self.config, self.manifest, self.contract)
        self.assertEqual(status["status"], "REGISTERED_AWAITING_NEW_OFFLINE_DATA")
        self.assertFalse(status["outcome_metrics_computed"])
        self.assertEqual(status["required_new_rows"], 8976)

    def test_safety_mutation_fails_closed(self):
        changed = copy.deepcopy(self.config)
        changed["safety"]["orders_enabled"] = True
        with self.assertRaises(ValueError):
            validate_contract(changed, self.manifest, self.contract)

    def test_reusing_revealed_date_fails_closed(self):
        changed = copy.deepcopy(self.config)
        changed["future_start_utc"] = "2026-09-27T00:00:00Z"
        with self.assertRaises(ValueError):
            validate_contract(changed, self.manifest, self.contract)


if __name__ == "__main__":
    unittest.main()
