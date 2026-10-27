"""Explicit unittest only: no pytest, collector, sockets or wall-clock waits."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from exia.genetic_primus.compact_ioc_execution_v1 import CompactIOC, BAR_MS, identity, normalize_order

ROOT = Path(__file__).resolve().parents[1]
BASE = 100 * BAR_MS


def inputs(**overrides):
    contract = json.loads((ROOT / "configs/genetic_primus_compact_ioc_v1.json").read_text(encoding="utf-8"))
    contract.update(overrides)
    rule = {"price_tick": "0.1", "quantity_step": "0.001", "minimum_quantity": "0.01", "minimum_notional": "5",
            "source_sha256": identity({"evidence": "synthetic_rules"}), "valid_from_ms": BASE - 1000, "valid_until_exclusive_ms": BASE + 10000}
    fees = {"taker_fee": "0.0006", "source_sha256": identity({"evidence": "synthetic_fee"}),
            "valid_from_ms": BASE - 1000, "valid_until_exclusive_ms": BASE + 10000}
    binding = {"origin_closed_ms": BASE, "signal_available_ms": BASE + 100, "train_end_ms": BASE - BAR_MS,
        "label_cutoff_ms": BASE - 8 * BAR_MS, "feature_available_ms": BASE, "target_clock": "synthetic_execution_fixture_not_forecast",
        "new_clock_labels_qualified": False, **{name: identity({"synthetic": name}) for name in
            ("model_sha256", "scaler_sha256", "feature_prefix_sha256", "training_identity_sha256")}}
    binding["sha256"] = identity(binding)
    return {"contract": contract, "rules": {"X": rule}, "fee_schedule": fees, "bindings": [binding],
            "data_sha256": identity({"synthetic": "fixture_bank"}), "implementation_sha256": identity({"synthetic": "implementation_binding"})}


def mark(t=0, price="100", exchange=None):
    return {"kind": "mark", "ts": BASE + t, "exchange_ms": BASE + (t if exchange is None else exchange), "symbol": "X", "price": price}


def intent(args, oid="A", t=100, side="buy", quantity="0.2019"):
    return {"kind": "intent", "ts": BASE + t, "order_id": oid, "symbol": "X", "side": side, "quantity": quantity,
            "model_binding_sha256": args["bindings"][0]["sha256"]}


def book(t=200, exchange=None, sid="B1", qty="10", bid="99.9", ask="100.1"):
    return {"kind": "book", "ts": BASE + t, "exchange_ms": BASE + (t - 10 if exchange is None else exchange),
            "snapshot_id": sid, "symbol": "X", "bids": [[bid, qty]], "asks": [[ask, qty]]}


def funding(t, rate="0.0001", price="100", known=None):
    return {"kind": "funding", "ts": BASE + t, "settlement_ms": BASE + t, "known_at_ms": BASE + (t if known is None else known),
            "symbol": "X", "signed_rate": rate, "mark_price": price, "source_sha256": identity({"synthetic": "funding"})}


def run(events, args=None):
    args = inputs() if args is None else args
    engine = CompactIOC(**args)
    for event in events:
        engine.ingest(event)
    return engine


def round_trip_events(args):
    return [mark(), intent(args), book(), book(500, sid="B2"), funding(600), mark(700),
            intent(args, "EXIT", 800, "sell", "0.201"), book(900, sid="B3"), book(1200, sid="B4"), {"kind": "clock", "ts": BASE + 1300}]


class CompactIOCTests(unittest.TestCase):
    def submitted(self, **overrides):
        args = inputs(**overrides)
        return run([mark(), intent(args), book()], args), args

    def filled(self, **overrides):
        engine, args = self.submitted(**overrides)
        engine.ingest(book(500, sid="B2"))
        return engine, args

    def test_quantity_and_protected_price_grid(self):
        rule = inputs()["rules"]["X"]
        self.assertEqual(normalize_order("buy", ".12349", "100.39", rule)["quantity"], "0.123")
        self.assertEqual(normalize_order("buy", ".12349", "100.39", rule)["limit_price"], "100.3")
        self.assertEqual(normalize_order("sell", ".12349", "100.39", rule)["limit_price"], "100.4")

    def test_first_post_signal_reference_not_four_hours(self):
        engine, _ = self.submitted()
        row = engine.result()["orders"]["A"]
        self.assertEqual(row["submitted_ms"], BASE + 200)
        self.assertEqual(row["arrival_ms"], BASE + 450)
        self.assertEqual(row["status"], "SUBMITTED")
        self.assertFalse(any(r["kind"] == "fill" for r in engine.result()["ledger"]))

    def test_full_fill_and_all_provenance(self):
        engine, _ = self.filled()
        result = engine.result()
        self.assertEqual(result["orders"]["A"]["status"], "FILLED")
        self.assertEqual(result["positions"]["X"], "0.201")
        fill = next(r for r in result["ledger"] if r["kind"] == "fill")
        self.assertEqual(fill["ts"], BASE + 500)
        self.assertIn("model_sha256", fill)
        self.assertIn("scaler_sha256", fill)
        self.assertFalse(result["actual_fill_verified"])
        self.assertFalse(any(result["safety"].values()))

    def test_post_receive_but_prearrival_book_cannot_fill(self):
        engine, _ = self.submitted()
        engine.ingest(book(500, exchange=449, sid="OLD"))
        self.assertEqual(engine.result()["orders"]["A"]["status"], "SUBMITTED")
        engine.ingest(book(520, exchange=510, sid="NEW"))
        self.assertEqual(engine.result()["orders"]["A"]["status"], "FILLED")

    def test_zero_latency_still_requires_new_snapshot(self):
        engine, _ = self.submitted(order_latency_ms=0)
        duplicate = book(201, exchange=190, sid="B1")
        engine.ingest(duplicate)
        self.assertEqual(engine.result()["orders"]["A"]["status"], "SUBMITTED")
        engine.ingest(book(220, exchange=210, sid="B2"))
        self.assertEqual(engine.result()["orders"]["A"]["status"], "FILLED")

    def test_stale_snapshot_waits_for_fresh_quote(self):
        args = inputs()
        engine = run([mark(), intent(args), book(1000, exchange=100)], args)
        self.assertEqual(engine.result()["orders"]["A"]["status"], "WAIT_REFERENCE")
        engine.ingest(book(1200, sid="FRESH"))
        self.assertEqual(engine.result()["orders"]["A"]["status"], "SUBMITTED")

    def test_partial_fill_remainder_is_canceled_once(self):
        engine, _ = self.submitted()
        engine.ingest(book(500, sid="B2", qty="0.15"))
        self.assertEqual(engine.result()["orders"]["A"]["status"], "PARTIAL_CANCELED")
        self.assertEqual(engine.result()["positions"]["X"], "0.015")
        engine.ingest(book(600, sid="B3"))
        self.assertEqual(engine.result()["positions"]["X"], "0.015")

    def test_shared_snapshot_liquidity_is_not_double_spent(self):
        args = inputs()
        engine = run([mark(), intent(args, "A"), intent(args, "B"), book(), book(500, sid="B2", qty="0.15")], args)
        result = engine.result()
        self.assertEqual(result["positions"]["X"], "0.015")
        self.assertEqual(result["orders"]["B"]["status"], "UNFILLED_CANCELED")

    def test_changed_snapshot_identity_fails_atomically(self):
        engine, _ = self.submitted()
        before = engine.result()
        with self.assertRaises(ValueError):
            engine.ingest(book(210, exchange=190, sid="B1", qty="20"))
        self.assertEqual(before, engine.result())

    def test_crossed_offgrid_and_future_book_fail_atomically(self):
        for broken in (book(500, sid="B2", bid="100.2"), book(500, sid="B2", ask="100.15"), book(500, exchange=501, sid="B2")):
            engine, _ = self.submitted()
            before = engine.result()
            with self.assertRaises(ValueError):
                engine.ingest(broken)
            self.assertEqual(before, engine.result())

    def test_below_quantity_and_notional_rejected(self):
        for q, reason in (("0.0059", "below_minimum_quantity"), ("0.01", "below_minimum_notional")):
            args = inputs()
            result = run([mark(), intent(args, quantity=q), book()], args).result()
            self.assertEqual(result["orders"]["A"]["reason"], reason)
            self.assertEqual(result["cash_quote"], "1000")

    def test_protection_price_does_not_chase_market(self):
        engine, _ = self.submitted()
        engine.ingest(book(500, sid="B2", bid="100.9", ask="101.0"))
        self.assertEqual(engine.result()["orders"]["A"]["status"], "UNFILLED_CANCELED")
        self.assertEqual(engine.result()["positions"]["X"], "0")

    def test_timeout_and_exact_deadline_do_not_fill(self):
        for event in ({"kind": "clock", "ts": BASE + 2200}, book(2200, sid="LATE")):
            engine, _ = self.submitted()
            engine.ingest(event)
            self.assertEqual(engine.result()["orders"]["A"]["status"], "EXPIRED")
            self.assertEqual(engine.result()["positions"]["X"], "0")

    def test_reference_wait_timeout(self):
        args = inputs()
        result = run([mark(), intent(args), {"kind": "clock", "ts": BASE + 5100}], args).result()
        self.assertEqual(result["orders"]["A"]["status"], "EXPIRED")

    def test_reduce_only_oversell_rejected(self):
        engine, args = self.filled()
        engine.ingest(intent(args, "EXIT", 600, "sell", ".3"))
        engine.ingest(book(700, sid="SELL"))
        self.assertEqual(engine.result()["orders"]["EXIT"]["status"], "REJECTED")
        self.assertEqual(engine.result()["positions"]["X"], "0.201")

    def test_cash_reserved_across_simultaneous_intents(self):
        args = inputs()
        events = [mark()] + [intent(args, str(i), quantity="2") for i in range(5)] + [book()]
        result = run(events, args).result()
        self.assertEqual(result["orders"]["4"]["reason"], "cash_reservation_limit")
        self.assertEqual(sum(o["status"] == "SUBMITTED" for o in result["orders"].values()), 4)

    def test_missing_or_stale_mark_blocks_new_buy(self):
        args = inputs()
        result = run([intent(args), book()], args).result()
        self.assertEqual(result["orders"]["A"]["reason"], "missing_or_stale_mark")
        engine = run([mark(), intent(args), book(2500, sid="STALEMARK")], args)
        self.assertEqual(engine.result()["orders"]["A"]["reason"], "missing_or_stale_mark")

    def test_rule_and_fee_interval_enforced_at_processing_clock(self):
        for kind in ("rules", "fee_schedule"):
            args = inputs()
            container = args["rules"]["X"] if kind == "rules" else args["fee_schedule"]
            container["valid_until_exclusive_ms"] = BASE + 150
            result = run([mark(), intent(args), book()], args).result()
            self.assertEqual(result["orders"]["A"]["reason"], "rule_or_fee_interval")

    def test_rules_expire_between_submit_and_fill(self):
        args = inputs()
        args["rules"]["X"]["valid_until_exclusive_ms"] = BASE + 450
        result = run([mark(), intent(args), book(), book(500, sid="B2")], args).result()
        self.assertEqual(result["orders"]["A"]["reason"], "rule_or_fee_interval")

    def test_funding_both_signs_and_no_borrowing(self):
        engine, _ = self.filled()
        before = engine.result()["cash_quote"]
        engine.ingest(funding(600, rate="0.001"))
        self.assertEqual(Decimal(before) - Decimal(engine.result()["cash_quote"]), Decimal("0.0201"))
        engine.ingest(funding(700, rate="-0.001"))
        self.assertEqual(engine.result()["cash_quote"], before)
        state = engine.result()
        with self.assertRaises(ValueError):
            engine.ingest(funding(800, rate=".9", price="100000"))
        self.assertEqual(engine.result(), state)

    def test_funding_before_same_timestamp_buy_is_not_charged(self):
        engine, _ = self.submitted()
        engine.ingest(funding(500))
        engine.ingest(book(500, sid="B2"))
        self.assertEqual(Decimal(engine.result()["signed_funding_quote"]), 0)

    def test_late_unknown_funding_rejected_not_applied_to_wrong_inventory(self):
        engine, _ = self.filled()
        before = engine.result()
        with self.assertRaises(ValueError):
            engine.ingest(funding(600, known=601))
        self.assertEqual(engine.result(), before)

    def test_funding_priority_regression_rejected(self):
        engine, _ = self.filled()
        before = engine.result()
        with self.assertRaises(ValueError):
            engine.ingest(funding(500))
        self.assertEqual(engine.result(), before)

    def test_funding_debit_cancels_buy_cash_reservation(self):
        args = inputs()
        engine = run([mark(), intent(args, quantity="2"), book(), book(500, sid="B2", qty="100"),
                      intent(args, "PENDING", 600, quantity="2"),
                      intent(args, "PENDING2", 600, quantity="2"),
                      intent(args, "PENDING3", 600, quantity="2"), book(700, sid="B3")], args)
        # Synthetic fault stress: accepted reservations exceed cash after two debits.
        # No post-arrival book is delivered, so none of these reservations can fill.
        engine.ingest(funding(800, rate=".9", price="100"))
        engine.ingest(funding(900, rate=".9", price="100"))
        result = engine.result()
        self.assertTrue(any(o.get("reason") == "funding_cash_reservation" for o in result["orders"].values()))
        self.assertGreaterEqual(Decimal(result["cash_quote"]), 0)

    def test_funding_observed_mark_prevents_artificial_same_clock_DD(self):
        args = inputs()
        engine = run([mark(), intent(args, quantity="2"), book(), book(500, sid="B2", qty="100")], args)
        engine.ingest(funding(600, rate=".1", price="200"))
        self.assertEqual(engine.state["marks"]["X"]["price"], Decimal("200"))
        self.assertLess(Decimal(engine.result()["max_drawdown"]), Decimal(".001"))

    def test_drawdown_blocks_new_buys_but_allows_reduce_only_exit(self):
        args = inputs(research_drawdown_gate="0.01")
        events = [mark(), intent(args, quantity="2"), book(), book(500, sid="B2", qty="100"), mark(600, price="25"),
                  intent(args, "BLOCKED", 600, quantity=".25"), intent(args, "EXIT", 600, "sell", "2"),
                  book(700, sid="B3", bid="24.9", ask="25.1", qty="100"),
                  book(1000, sid="B4", bid="24.9", ask="25.1", qty="100")]
        result = run(events, args).result()
        self.assertEqual(result["orders"]["BLOCKED"]["reason"], "research_drawdown_breach_new_buy_blocked")
        self.assertEqual(result["orders"]["EXIT"]["status"], "FILLED")
        self.assertEqual(Decimal(result["positions"]["X"]), 0)

    def test_reduce_only_reservations_prevent_concurrent_oversell(self):
        engine, args = self.filled()
        for oid in ("E1", "E2"):
            engine.ingest(intent(args, oid, 600, "sell", ".101"))
        engine.ingest(book(700, sid="SELL"))
        self.assertEqual(engine.result()["orders"]["E2"]["reason"], "reduce_only_position_reservation")

    def test_funding_on_same_timestamp_close_charges_old_inventory(self):
        args = inputs()
        events = round_trip_events(args)
        events = [e for e in events if e["kind"] != "funding"]
        events.insert(-2, funding(1200))
        result = run(events, args).result()
        self.assertEqual(Decimal(result["signed_funding_quote"]), Decimal(".00201"))
        self.assertEqual(Decimal(result["positions"]["X"]), 0)

    def test_order_and_depth_bounds_fail_without_mutation(self):
        args = inputs(max_orders=1, max_levels=1)
        engine = run([mark(), intent(args)], args)
        before = engine.result()
        with self.assertRaises(ValueError):
            engine.ingest(intent(args, "SECOND"))
        bad = book()
        bad["asks"].append(["100.2", "10"])
        with self.assertRaises(ValueError):
            engine.ingest(bad)
        self.assertEqual(before, engine.result())

    def test_multilevel_partial_fill_uses_grid_at_each_level(self):
        args = inputs()
        engine = run([mark(), intent(args, quantity=".05"), book()], args)
        depth = book(500, sid="B2", qty=".05")
        depth["asks"].append(["100.2", ".05"])
        engine.ingest(depth)
        result = engine.result()
        self.assertEqual(result["orders"]["A"]["filled"], "0.010")
        self.assertEqual(result["orders"]["A"]["status"], "PARTIAL_CANCELED")
        self.assertEqual(sum(r["kind"] == "fill" for r in result["ledger"]), 2)

    def test_flat_round_trip_costs_and_terminal_flat(self):
        args = inputs()
        result = run(round_trip_events(args), args).result()
        self.assertEqual(result["positions"]["X"], "0.000")
        fees = sum(Decimal(r["fee_quote"]) for r in result["ledger"] if r["kind"] == "fill")
        expected = Decimal("1000") - Decimal(".201") * Decimal(".2") - fees - Decimal(".00201")
        self.assertEqual(Decimal(result["cash_quote"]), expected)
        self.assertLess(Decimal(result["equity_quote"]), 1000)

    def test_no_trade_zero_cost(self):
        result = run([mark(), {"kind": "clock", "ts": BASE + 1300}]).result()
        self.assertEqual(result["cash_quote"], "1000")
        self.assertEqual(Decimal(result["fees_quote"]), 0)

    def test_checkpoint_restart_exact_and_immutable_contract(self):
        engine, args = self.submitted()
        checkpoint = engine.checkpoint()
        restored = CompactIOC.restore(checkpoint, **args)
        for candidate in (engine, restored):
            candidate.ingest(book(500, sid="B2"))
        self.assertEqual(engine.result(), restored.result())
        broken = deepcopy(checkpoint)
        broken["events"][1]["quantity"] = "1"
        with self.assertRaises(ValueError):
            CompactIOC.restore(broken, **args)
        changed = deepcopy(args)
        changed["implementation_sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            CompactIOC.restore(checkpoint, **changed)

    def test_event_bounds_and_bad_clock_atomic(self):
        args = inputs(max_events=2)
        engine = run([mark(), intent(args)], args)
        before = engine.result()
        with self.assertRaises(ValueError):
            engine.ingest(book())
        self.assertEqual(before, engine.result())
        engine = CompactIOC(**inputs())
        with self.assertRaises(ValueError):
            engine.ingest({"kind": "clock", "ts": True})
        self.assertEqual(engine.result()["event_count"], 0)

    def test_future_features_purge_and_label_qualification_refused(self):
        for key, value in (("feature_available_ms", BASE + 1), ("label_cutoff_ms", BASE - 7 * BAR_MS), ("new_clock_labels_qualified", True)):
            args = inputs()
            b = args["bindings"][0]
            b[key] = value
            b["sha256"] = identity({k: v for k, v in b.items() if k != "sha256"})
            with self.assertRaises(ValueError):
                CompactIOC(**args)

    def test_pre_signal_intent_and_reused_id_refused(self):
        args = inputs()
        engine = run([mark()], args)
        with self.assertRaises(ValueError):
            engine.ingest(intent(args, t=99))
        engine.ingest(intent(args))
        with self.assertRaises(ValueError):
            engine.ingest(intent(args))

    def test_no_network_or_OHLCV_event_fallback(self):
        engine = CompactIOC(**inputs())
        with self.assertRaises(ValueError):
            engine.ingest({"kind": "ohlcv", "ts": BASE})
        args = inputs()
        args["contract"]["network_allowed"] = True
        args["contract"]["safety"]["network_allowed"] = True
        with self.assertRaises(ValueError):
            CompactIOC(**args)


if __name__ == "__main__":
    unittest.main()
