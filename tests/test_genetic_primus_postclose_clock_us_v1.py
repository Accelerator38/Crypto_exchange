"""Explicit, bounded price-free fixtures. No market/collector/test discovery."""
from copy import deepcopy
import unittest

from exia.genetic_primus.postclose_clock_us_v1 import (
    SAFETY, adapt_recorded_books, epoch_us, lift_contract, lossless_legacy_window,
    lossless_ms, resolve_endpoint_us, seal, validate_contract, verify, window_requests_us,
)
from exia.genetic_primus.postclose_label_plan_v1 import resolve_endpoint, window_requests

BASE = 1785542400000000
SOURCE = "a" * 64
PARENT_FILE = "9a25eb9114769b6d458a998858f08053d3c2eb3c8d27bfd578150a6bdb5fdb71"
PARENT = {
    "schema": "compact_postclose_label_plan/1", "safety": SAFETY,
    "ohlcv_to_l2_fallback": False, "old_next_4h_open_labels_compatible": False,
    "clock_resolution_is_execution_validation": False,
    "bar_ms": 14400000, "deployment_horizon_ms": 259200000, "retrain_cadence_ms": 259200000,
    "purge_bars": 7, "train_end_offset_bars": 1, "feature_min_history_bars": 49,
    "feature_state_requires_full_causal_prefix": True, "signal_compute_latency_ms": 100,
    "order_latency_ms": 250, "reference_wait_ms": 5000, "order_ttl_ms": 2000,
    "max_quote_age_ms": 500, "max_observations_per_endpoint": 256,
    "engine_contract_file_sha256": "b" * 64,
    "validation_start_ms": BASE // 1000, "validation_end_ms_exclusive": 1790812800000,
    "symbols": ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT"],
}


def contract():
    # This minimal parent is explicitly synthetic, not the registered file/body.
    return lift_contract(deepcopy(PARENT), PARENT_FILE)


def observation(name, exchange, received=None, available=None, **changes):
    received = exchange if received is None else received
    available = received if available is None else available
    body = {"snapshot_id": name, "exchange_us": exchange, "received_us": received,
            "available_us": available, "source_sha256": SOURCE,
            "snapshot_sha256": ("b" if name == "ref" else "c" if name == "end" else "d") * 64}
    body.update(changes)
    return body


def window(observations, ready=BASE, *, end=None, basis="synthetic_fixture"):
    return {"coverage_start_us": ready - 500000, "coverage_end_us_exclusive": end or ready + 7000000,
            "observations": observations, "clock_basis": basis}


def happy(*, ready=BASE, submillisecond=False):
    offset = 1001 if submillisecond else 1000
    ref = observation("ref", ready + offset, ready + offset + 1000)
    arrival = ref["available_us"] + 250000
    endpoint = observation("end", arrival + offset, arrival + offset + 1000)
    return window([ref, endpoint], ready)


def recorded_event(offset=1001, **changes):
    body = {"kind": "engineering_book_fragment", "symbol": "BTCUSDT",
            "exchange_ts_us": BASE + offset - 17, "received_ts_us": BASE + offset,
            "source_prefix_sha256": SOURCE, "source_message_rows_sha256": "b" * 64,
            "source_rows_end_exclusive": 30, "strategy_availability_verified": False,
            "bids": [["100", "1"]], "asks": [["101", "1"]]}
    body.update(changes)
    return seal(body)


class ClockUsTests(unittest.TestCase):
    def test_lossless_integer_conversion(self):
        self.assertEqual(lossless_ms(BASE), BASE // 1000)

    def test_submillisecond_conversion_rejected(self):
        with self.assertRaises(ValueError):
            lossless_ms(BASE + 1)

    def test_epoch_types_and_ranges(self):
        for value in (True, str(BASE), float(BASE), 0, 10**16):
            with self.assertRaises(ValueError):
                epoch_us(value)

    def test_parent_lift_all_parameters_exact(self):
        result = contract()
        self.assertEqual(result["order_latency_us"], PARENT["order_latency_ms"] * 1000)
        self.assertEqual(result["deployment_horizon_us"], 259200000000)
        validate_contract(result)

    def test_parent_invalid_safety_rejected(self):
        parent = deepcopy(PARENT)
        parent["safety"]["orders_enabled"] = True
        with self.assertRaises(ValueError):
            lift_contract(parent, PARENT_FILE)

    def test_native_safety_strict_false_not_zero(self):
        for key in ("financial_replay_ready", "real_market_coverage_verified"):
            c = contract()
            c[key] = 0
            with self.assertRaises(ValueError):
                validate_contract(c)

    def test_parent_hash_format_and_native_bounds(self):
        for change in ({"parent_file_sha256": "bad"}, {"order_latency_us": 1},
                       {"order_ttl_us": 250000}, {"max_observations_per_endpoint": 257},
                       {"purge_bars": 6}, {"deployment_horizon_us": True}):
            c = contract() | change
            with self.assertRaises(ValueError):
                validate_contract(c)

    def test_requests_preserve_submillisecond_feature_availability(self):
        result = window_requests_us(BASE, BASE + 17, contract())
        self.assertEqual(result["entry"]["signal_available_us"], BASE + 100017)
        self.assertEqual(result["scheduled_exit_decision_us"], BASE + 259200000000)
        self.assertEqual(result["exit"]["signal_available_us"], BASE + 259200100000)

    def test_requests_compare_legacy_away_from_boundaries(self):
        old = window_requests(BASE // 1000, BASE // 1000, PARENT)
        new = window_requests_us(BASE, BASE, contract())
        self.assertEqual(new["scheduled_exit_decision_us"], old["scheduled_exit_decision_ms"] * 1000)
        for side in ("entry", "exit"):
            for key in ("signal_available", "coverage_start"):
                self.assertEqual(new[side][key + "_us"], old[side][key + "_ms"] * 1000)
            self.assertEqual(new[side]["coverage_end_us_exclusive"], old[side]["coverage_end_ms_exclusive"] * 1000)

    def test_requests_reject_future_or_unaligned_close(self):
        for decision, available in ((BASE + 1, BASE + 1), (BASE, BASE - 1),
                                    (BASE, BASE + 259200000000)):
            with self.assertRaises(ValueError):
                window_requests_us(decision, available, contract())

    def test_recorded_adapter_preserves_precision_and_null(self):
        event = recorded_event()
        result = adapt_recorded_books([event])
        obs = result["observations"][0]
        self.assertEqual(obs["received_us"], BASE + 1001)
        self.assertIsNone(obs["available_us"])
        self.assertEqual(obs["snapshot_sha256"], event["sha256"])
        self.assertNotIn("bids", obs)
        self.assertEqual(result["timestamp_precision_loss_us"], 0)
        verify(result)

    def test_recorded_books_never_resolve_clock(self):
        adapted = adapt_recorded_books([recorded_event()])
        result = resolve_endpoint_us(BASE, window(adapted["observations"], basis="recorded_unknown"), contract())
        self.assertEqual(result["status"], "BLOCKED_UNKNOWN_STRATEGY_AVAILABILITY")
        self.assertIsNone(result["reference"])

    def test_provider_receipt_cannot_be_self_declared_available(self):
        obs = adapt_recorded_books([recorded_event()])["observations"][0]
        obs["available_us"] = obs["received_us"]
        with self.assertRaises(ValueError):
            resolve_endpoint_us(BASE, window([obs], basis="recorded_unknown"), contract())

    def test_recorded_availability_true_rejected(self):
        with self.assertRaises(ValueError):
            adapt_recorded_books([recorded_event(strategy_availability_verified=True)])

    def test_reported_funding_ticker_cannot_be_book_endpoint(self):
        with self.assertRaises(ValueError):
            adapt_recorded_books([recorded_event(kind="ticker")])

    def test_recorded_source_hash_and_order_tamper(self):
        event = recorded_event()
        event["received_ts_us"] += 1
        with self.assertRaises(ValueError):
            adapt_recorded_books([event])
        with self.assertRaises(ValueError):
            adapt_recorded_books([recorded_event(2000), recorded_event(1000)])

    def test_recorded_ordinal_bound(self):
        with self.assertRaises(ValueError):
            adapt_recorded_books([recorded_event(source_rows_end_exclusive=True)])

    def test_native_resolves_submillisecond_fixture_not_fill(self):
        result = resolve_endpoint_us(BASE, happy(submillisecond=True), contract())
        self.assertEqual(result["status"], "CLOCK_RESOLVED_NOT_FILL_QUALIFIED")
        self.assertEqual(result["reference"]["exchange_us"], BASE + 1001)
        self.assertFalse(result["actual_fill_verified"])
        self.assertFalse(result["strategy_availability_verified"])

    def test_bridge_non_aligned_fixture_rejected(self):
        with self.assertRaises(ValueError):
            lossless_legacy_window(BASE, happy(submillisecond=True), contract())

    def test_bridge_recorded_unknown_rejected(self):
        obs = adapt_recorded_books([recorded_event()])["observations"]
        with self.assertRaises(ValueError):
            lossless_legacy_window(BASE, window(obs, basis="recorded_unknown"), contract())

    def test_legacy_bridge_matches_aligned_resolved_fixture(self):
        native = resolve_endpoint_us(BASE, happy(), contract())
        bridge = lossless_legacy_window(BASE, happy(), contract())
        old = resolve_endpoint(bridge["signal_available_ms"], bridge["window"], PARENT)
        self.assertEqual(native["status"], old["status"])
        for side in ("reference", "endpoint"):
            self.assertEqual(native[side]["snapshot_id"], old[side]["snapshot_id"])
            self.assertEqual(native[side]["available_us"], old[side]["available_ms"] * 1000)

    def test_reference_signal_boundary_ambiguous(self):
        w = happy()
        w["observations"][0]["exchange_us"] = BASE
        result = resolve_endpoint_us(BASE, w, contract())
        self.assertEqual(result["status"], "AMBIGUOUS_REFERENCE_BOUNDARY")

    def test_reference_multiple_first_candidates_ambiguous(self):
        w = happy()
        ref = w["observations"][0]
        w["observations"].insert(1, observation("peer", ref["exchange_us"], ref["received_us"]))
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "AMBIGUOUS_REFERENCE_BATCH")

    def test_stale_same_clock_peer_still_ambiguous(self):
        w = happy()
        ref = w["observations"][0]
        w["observations"].insert(1, observation("peer", BASE - 1000000,
                                               ref["received_us"], ref["available_us"]))
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "AMBIGUOUS_REFERENCE_BATCH")

    def test_arrival_equal_boundary_ambiguous(self):
        w = happy()
        arrival = w["observations"][0]["available_us"] + 250000
        w["observations"][1] = observation("end", arrival)
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "AMBIGUOUS_POST_ARRIVAL_BOUNDARY")

    def test_one_us_after_arrival_is_eligible(self):
        w = happy()
        arrival = w["observations"][0]["available_us"] + 250000
        w["observations"][1] = observation("end", arrival + 1)
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "CLOCK_RESOLVED_NOT_FILL_QUALIFIED")

    def test_prearrival_exchange_late_receipt_cannot_fill(self):
        w = happy()
        arrival = w["observations"][0]["available_us"] + 250000
        w["observations"][1] = observation("end", arrival - 1, arrival + 1000)
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "EXPIRED_NO_POST_ARRIVAL")

    def test_one_us_before_expiry_is_eligible(self):
        w = happy()
        expires = w["observations"][0]["available_us"] + 2000000
        w["observations"][1] = observation("end", expires - 1)
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "CLOCK_RESOLVED_NOT_FILL_QUALIFIED")

    def test_at_expiry_excluded(self):
        w = happy()
        expires = w["observations"][0]["available_us"] + 2000000
        w["observations"][1] = observation("end", expires)
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "EXPIRED_NO_POST_ARRIVAL")

    def test_stale_age_one_us_over_rejected(self):
        w = happy()
        endpoint = w["observations"][1]
        endpoint["available_us"] = endpoint["exchange_us"] + 500001
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "EXPIRED_NO_POST_ARRIVAL")

    def test_stale_age_exact_bound_eligible(self):
        w = happy()
        endpoint = w["observations"][1]
        endpoint["available_us"] = endpoint["exchange_us"] + 500000
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "CLOCK_RESOLVED_NOT_FILL_QUALIFIED")

    def test_first_reference_at_deadline_excluded(self):
        self.assertEqual(resolve_endpoint_us(BASE, window([observation("ref", BASE + 5000000)]),
                                             contract())["status"], "EXPIRED_NO_REFERENCE")

    def test_missing_and_pending_reference(self):
        self.assertEqual(resolve_endpoint_us(BASE, window([]), contract())["status"], "EXPIRED_NO_REFERENCE")
        self.assertEqual(resolve_endpoint_us(BASE, window([], end=BASE + 1000000),
                                             contract())["status"], "PENDING_REFERENCE")

    def test_pending_postarrival_without_full_coverage(self):
        ref = observation("ref", BASE + 1000)
        w = window([ref], end=BASE + 300000)
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "PENDING_POST_ARRIVAL")

    def test_signal_outside_window(self):
        for ready in (BASE - 500001, BASE + 7000000):
            self.assertEqual(resolve_endpoint_us(ready, window([]), contract())["status"], "MISSING_SIGNAL_COVERAGE")

    def test_price_fallback_field_rejected(self):
        w = happy()
        w["observations"][0]["price"] = "100"
        with self.assertRaises(ValueError):
            resolve_endpoint_us(BASE, w, contract())

    def test_unknown_scope_and_time_regression_rejected(self):
        for w in (window([], basis="verified_real"), window(list(reversed(happy()["observations"])))):
            with self.assertRaises(ValueError):
                resolve_endpoint_us(BASE, w, contract())

    def test_duplicate_id_and_hash_alias_rejected(self):
        for field in ("snapshot_id", "snapshot_sha256"):
            w = happy()
            w["observations"][1][field] = w["observations"][0][field]
            with self.assertRaises(ValueError):
                resolve_endpoint_us(BASE, w, contract())

    def test_observation_and_adapter_budgets(self):
        with self.assertRaises(ValueError):
            resolve_endpoint_us(BASE, window([observation("ref", BASE + 1000)] * 257), contract())
        for events in ([], [recorded_event()] * 257):
            with self.assertRaises(ValueError):
                adapt_recorded_books(events)

    def test_exact_coverage_end_excluded(self):
        with self.assertRaises(ValueError):
            resolve_endpoint_us(BASE, window([observation("ref", BASE + 7000000)]), contract())

    def test_endpoint_batch_ambiguous(self):
        w = happy()
        endpoint = w["observations"][1]
        w["observations"].append(observation("peer", endpoint["exchange_us"], endpoint["received_us"]))
        self.assertEqual(resolve_endpoint_us(BASE, w, contract())["status"], "AMBIGUOUS_POST_ARRIVAL_BATCH")

    def test_bridge_ambiguous_clock_rejected(self):
        w = happy()
        w["observations"][0]["exchange_us"] = BASE
        with self.assertRaises(ValueError):
            lossless_legacy_window(BASE, w, contract())

    def test_no_mutation_and_deterministic_detached_results(self):
        w = happy()
        before = deepcopy(w)
        first = resolve_endpoint_us(BASE, w, contract())
        self.assertEqual(first, resolve_endpoint_us(BASE, w, contract()))
        self.assertEqual(w, before)
        first["reference"]["snapshot_id"] = "mutated"
        self.assertEqual(w, before)

    def test_all_authority_flags_false(self):
        for result in (contract(), window_requests_us(BASE, BASE, contract()),
                       resolve_endpoint_us(BASE, happy(), contract()), adapt_recorded_books([recorded_event()])):
            self.assertTrue(all(flag is False for flag in result["safety"].values()))
            self.assertFalse(result["financial_replay_ready"])

    def test_synthetic_available_before_receipt_rejected(self):
        w = happy()
        w["observations"][0]["available_us"] -= 1
        with self.assertRaises(ValueError):
            resolve_endpoint_us(BASE, w, contract())


if __name__ == "__main__":
    unittest.main()
