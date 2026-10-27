"""Named, price-free unittest checks. No collector, pytest, IO tapes or sleeps."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from exia.genetic_primus import postclose_label_plan_v1 as plan

ROOT = Path(__file__).resolve().parents[1]


def contract():
    return json.loads((ROOT / "configs/genetic_primus_postclose_label_plan_v1.json").read_text(encoding="utf-8"))


def observation(base, t, sid, *, exchange=None, received=None):
    return {"snapshot_id": sid, "exchange_ms": base + (t - 10 if exchange is None else exchange),
            "received_ms": base + (t if received is None else received), "available_ms": base + t,
            "source_sha256": plan.identity({"synthetic": "source"}),
            "snapshot_sha256": plan.identity({"synthetic_snapshot": sid})}


def window(base, observations=None, end=7100, start=0):
    return {"coverage_start_ms": base + start, "coverage_end_ms_exclusive": base + end,
            "observations": observations if observations is not None else
            [observation(base, 200, "REF"), observation(base, 500, "END")]}


def pair(c):
    base = c["validation_start_ms"]
    return plan.clock_pair(base, base, window(base), window(base + plan.HORIZON_MS), c)


def declared_bundle(c):
    end = c["validation_end_ms_exclusive"] + 7100
    rows = {kind: [{"symbol": symbol, "coverage_start_ms": c["validation_start_ms"] - 49 * plan.BAR_MS,
                    "coverage_end_ms_exclusive": end, "evidence_kind": "observed",
                    "quality_status": "independently_verified", "source_manifest_sha256": "a" * 64,
                    "quality_receipt_sha256": "b" * 64} for symbol in c["symbols"]] for kind in plan.COMPONENTS}
    return {"schema": "postclose_execution_bundle_declaration/1", "target_clock_sha256": plan.identity(c), "components": rows}


class PostclosePlanTests(unittest.TestCase):
    def setUp(self):
        self.c = contract()
        self.base = self.c["validation_start_ms"]

    def resolve(self, w, c=None):
        return plan.resolve_endpoint(self.base + 100, w, c or self.c)

    def test_three_day_cadence_and_purge_contract(self):
        self.assertEqual(plan.validate_contract(self.c), plan.identity(self.c))
        for key, value in (("purge_bars", 6), ("retrain_cadence_ms", plan.BAR_MS),
                           ("order_ttl_ms", 250), ("ohlcv_to_l2_fallback", True)):
            c = deepcopy(self.c)
            c[key] = value
            with self.assertRaises(ValueError):
                plan.validate_contract(c)

    def test_authority_flags_cannot_be_enabled(self):
        for key in self.c["safety"]:
            c = deepcopy(self.c)
            c["safety"][key] = True
            with self.assertRaises(ValueError):
                plan.validate_contract(c)

    def test_training_cutoff_retains_extra_seven_bar_purge(self):
        ctx = plan.training_context(self.base, self.c)
        self.assertEqual(ctx["train_end_ms"], self.base - plan.BAR_MS)
        self.assertEqual(ctx["label_cutoff_ms"], self.base - 8 * plan.BAR_MS)
        for bad in (True, -1, self.base + 1):
            with self.assertRaises(ValueError):
                plan.training_context(bad, self.c)

    def test_immediate_signal_not_next_four_hour_open(self):
        requests = plan.window_requests(self.base, self.base, self.c)
        self.assertEqual(requests["entry"]["signal_available_ms"], self.base + 100)
        self.assertEqual(requests["scheduled_exit_decision_ms"], self.base + plan.HORIZON_MS)
        self.assertEqual(requests["exit"]["signal_available_ms"], self.base + plan.HORIZON_MS + 100)

    def test_delayed_feature_availability_does_not_shift_scheduled_exit(self):
        request = plan.window_requests(self.base, self.base + 800, self.c)
        self.assertEqual(request["entry"]["signal_available_ms"], self.base + 900)
        self.assertEqual(request["exit"]["signal_available_ms"], self.base + plan.HORIZON_MS + 100)
        for bad in (self.base - 1, self.base + plan.HORIZON_MS):
            with self.assertRaises(ValueError):
                plan.window_requests(self.base, bad, self.c)

    def test_reference_latency_and_first_eligible_endpoint(self):
        resolved = self.resolve(window(self.base))
        self.assertEqual(resolved["arrival_ms"], self.base + 450)
        self.assertEqual(resolved["endpoint"]["available_ms"], self.base + 500)
        self.assertFalse(resolved["actual_fill_verified"])

    def test_late_delivery_of_prearrival_quote_cannot_qualify(self):
        shots = [observation(self.base, 90, "PRE"), observation(self.base, 200, "REF"),
                 observation(self.base, 500, "LATE", exchange=200), observation(self.base, 700, "GOOD")]
        self.assertEqual(self.resolve(window(self.base, shots))["endpoint"]["snapshot_id"], "GOOD")

    def test_zero_latency_does_not_fill_reference_or_retransmitted_snapshot(self):
        c = deepcopy(self.c)
        c["order_latency_ms"] = 0
        shots = [observation(self.base, 200, "REF"), observation(self.base, 250, "REF", exchange=190),
                 observation(self.base, 300, "NEXT")]
        self.assertEqual(self.resolve(window(self.base, shots), c)["endpoint"]["snapshot_id"], "NEXT")

    def test_exclusive_ttl_does_not_accept_snapshot_at_deadline(self):
        shots = [observation(self.base, 200, "REF"), observation(self.base, 2200, "END")]
        self.assertEqual(self.resolve(window(self.base, shots))["status"], "EXPIRED_NO_POST_ARRIVAL")

    def test_incomplete_prefix_is_pending_not_no_trade_or_zero_return(self):
        prefix = window(self.base, [observation(self.base, 200, "REF")], end=300)
        self.assertEqual(self.resolve(prefix)["status"], "PENDING_POST_ARRIVAL")
        self.assertEqual(self.resolve(window(self.base, [], end=300))["status"], "PENDING_REFERENCE")
        self.assertEqual(self.resolve(window(self.base, []))["status"], "EXPIRED_NO_REFERENCE")

    def test_missing_signal_coverage_is_explicit(self):
        self.assertEqual(self.resolve(window(self.base, [], start=101))["status"], "MISSING_SIGNAL_COVERAGE")

    def test_stale_reference_ignored(self):
        shots = [observation(self.base, 700, "STALE", exchange=100),
                 observation(self.base, 800, "REF"), observation(self.base, 1100, "END")]
        self.assertEqual(self.resolve(window(self.base, shots))["reference"]["snapshot_id"], "REF")

    def test_future_and_regressed_clocks_rejected_without_input_mutation(self):
        bad = window(self.base, [observation(self.base, 200, "FUTURE", exchange=210)])
        original = deepcopy(bad)
        with self.assertRaises(ValueError):
            self.resolve(bad)
        self.assertEqual(bad, original)
        bad = window(self.base, [observation(self.base, 500, "END"), observation(self.base, 200, "REF")])
        with self.assertRaises(ValueError):
            self.resolve(bad)

    def test_changed_snapshot_identity_and_bad_source_hash_rejected(self):
        shots = [observation(self.base, 200, "REF"), observation(self.base, 500, "REF")]
        with self.assertRaises(ValueError):
            self.resolve(window(self.base, shots))
        shots = [observation(self.base, 200, "REF")]
        shots[0]["source_sha256"] = "unknown"
        with self.assertRaises(ValueError):
            self.resolve(window(self.base, shots))

    def test_hard_observation_bound_and_no_price_fallback(self):
        with self.assertRaises(ValueError):
            self.resolve(window(self.base, [observation(self.base, 200, "REF")] * 257))
        bad = window(self.base)
        bad["observations"][0]["ohlcv_open"] = "100"
        with self.assertRaises(ValueError):
            self.resolve(bad)

    def test_pair_provenance_without_financial_or_training_qualification(self):
        p = pair(self.c)
        self.assertTrue(p["clock_pair_resolved"])
        self.assertFalse(p["training_ready"])
        self.assertFalse(p["execution_qualified"])
        self.assertEqual(p["clock_pair_available_ms"], self.base + plan.HORIZON_MS + 500)
        self.assertEqual(p["sha256"], plan.identity({k: v for k, v in p.items() if k != "sha256"}))
        self.assertNotIn("return", p)

    def test_maturity_boundary_and_late_cost_source(self):
        origin = self.base + plan.HORIZON_MS + 9 * plan.BAR_MS
        cutoff = origin - 8 * plan.BAR_MS
        self.assertTrue(plan.dependency_maturity(pair(self.c), [cutoff], origin, self.c)["time_eligible"])
        self.assertFalse(plan.dependency_maturity(pair(self.c), [cutoff + 1], origin, self.c)["time_eligible"])
        self.assertFalse(plan.dependency_maturity(pair(self.c), [cutoff], origin, self.c)["training_ready"])

    def test_actual_exit_observation_maturity_overrides_nominal_three_days(self):
        origin = self.base + plan.HORIZON_MS + 8 * plan.BAR_MS
        # Cutoff equals nominal exit close; quote is available 500 ms later.
        self.assertFalse(plan.dependency_maturity(pair(self.c), [self.base], origin, self.c)["time_eligible"])

    def test_incomplete_pair_and_tampered_target_cannot_enter_maturity_check(self):
        p = plan.clock_pair(self.base, self.base, window(self.base, []), window(self.base + plan.HORIZON_MS), self.c)
        with self.assertRaises(ValueError):
            plan.dependency_maturity(p, [self.base], self.base, self.c)
        p = pair(self.c)
        p["clock_pair_available_ms"] -= 1
        with self.assertRaises(ValueError):
            plan.dependency_maturity(p, [self.base], self.base, self.c)

    def test_bundle_missing_and_old_target_hash_are_blockers(self):
        self.assertEqual(plan.bundle_declaration_issues(None, self.c), ["registered_execution_bundle_missing"])
        bundle = declared_bundle(self.c)
        bundle["target_clock_sha256"] = "0" * 64
        self.assertIn("target_clock_hash_mismatch", plan.bundle_declaration_issues(bundle, self.c))

    def test_bundle_full8_cost_coverage_and_provenance_are_separate(self):
        bundle = declared_bundle(self.c)
        self.assertEqual(plan.bundle_declaration_issues(bundle, self.c), [])
        bundle["components"]["marks"].pop()
        bundle["components"]["funding"][0]["coverage_end_ms_exclusive"] = self.c["validation_end_ms_exclusive"]
        bundle["components"]["fees"][0]["quality_receipt_sha256"] = None
        issues = plan.bundle_declaration_issues(bundle, self.c)
        self.assertTrue(any(x.startswith("marks:XRPUSDT:missing") for x in issues))
        self.assertTrue(any(x.startswith("funding:ADAUSDT:unverified") for x in issues))
        self.assertTrue(any(x.startswith("fees:ADAUSDT:missing_source") for x in issues))


if __name__ == "__main__":
    unittest.main()
