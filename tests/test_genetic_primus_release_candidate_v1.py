"""Tiny synthetic regressions. Run selected groups directly, never a collector.

All rule sizes, costs, timing and capital below are FIXTURES, not Bitget facts.
"""
import copy
import json
import runpy
import sys
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ROOT / "src" / "exia" / "genetic_primus"
if str(MODULES) not in sys.path:
    sys.path.insert(0, str(MODULES))
from release_contract_v1 import bind_model, object_sha, utc_ms, validate_contract
from execution_replay_v1 import Replay, normalize_order, overlay_target, transition_order

ORIGIN = utc_ms("2026-09-01T00:00:00Z")
RULE = {"price_tick": "0.1", "quantity_step": "0.1", "minimum_quantity": "0.1",
        "minimum_notional": "1", "source_sha256": object_sha("SYNTHETIC_RULES"),
        "valid_from_ms": ORIGIN, "valid_until_exclusive_ms": ORIGIN + 259200000}


def contract():
    return json.loads((ROOT / "configs/genetic_primus_release_candidate_v1.json").read_text(encoding="utf-8"))


def binding(c=None, origin=ORIGIN):
    c = contract() if c is None else c
    return bind_model(c, origin_start_ms=origin,
                      scaler={"median": ["0"] * 4, "scale": ["1"] * 4,
                              "fitted_rows": 12, "fit_values_sha256": object_sha("fixture_fit"),
                              "train_start_ms": utc_ms("2022-01-01T00:00:00Z"),
                              "train_end_exclusive_ms": origin - 7 * 14400000},
                      data_sha256=object_sha("fixture_training_data"),
                      train_rows_sha256=object_sha("fixture_rows"),
                      folds_sha256=object_sha("fixture_folds"),
                      resample_bank_sha256=object_sha("fixture_selection_bank"),
                      audit_bank_sha256=object_sha("fixture_audit_bank"),
                      implementation_sha256=object_sha("fixture_implementation_not_empirical"))


def fixture(**overrides):
    signed = binding()
    settings = {"contract_sha256": object_sha(contract()),
                "dataset_sha256": object_sha("SYNTHETIC_EXECUTION_EVENTS"),
                "fee_schedule_sha256": object_sha("SYNTHETIC_FEE_SCHEDULE"),
                "evidence_kind": "synthetic", "initial_cash": "1000", "taker_fee": "0.001",
                "latency_ms": 10, "max_quote_age_ms": 20, "max_mark_age_ms": 1000,
                "order_ttl_ms": 100, "participation": "1", "max_order_notional": "900",
                "max_gross_ratio": "1", "max_loss": "200", "max_drawdown": "0.2",
                "max_events": 128, "max_levels": 8,
                "fee_valid_from_ms": ORIGIN, "fee_valid_until_exclusive_ms": ORIGIN + 259200000}
    settings.update(overrides)
    return Replay(settings=settings, rules={"LINKUSDT": RULE}, models=[signed])


def must_fail(fn, fragment=None):
    try:
        fn()
    except ValueError as exc:
        if fragment:
            assert fragment in str(exc), str(exc)
    else:
        raise AssertionError("expected fail-closed ValueError")


def mark(r, offset=0, price="100", **extra):
    e = {"id": "m" + str(offset), "kind": "mark", "ts": ORIGIN + offset,
         "exchange_ms": ORIGIN + offset, "symbol": "LINKUSDT", "price": price, **extra}
    r.ingest(e)
    return e


def decision(r, offset=0, target=1, suffix="", **extra):
    e = {"id": "d" + str(offset) + suffix, "kind": "decision", "ts": ORIGIN + offset,
         "symbol": "LINKUSDT", "model_sha256": next(iter(r.models)),
         "feature_cutoff_ms": ORIGIN, "features": ["0", "0", "2", "0"],
         "proposal": target, **extra}
    r.ingest(e)
    return e["id"]


def intent(r, did, offset=0, oid="o1", side="buy", qty="1", limit="101", reduce=False, **extra):
    e = {"id": "i" + oid, "kind": "intent", "ts": ORIGIN + offset, "symbol": "LINKUSDT",
         "client_oid": oid, "decision_id": did, "model_sha256": next(iter(r.models)),
         "side": side, "quantity": qty, "limit_price": limit, "reduce_only": reduce, **extra}
    r.ingest(e)
    return e


def book(r, offset=10, asks=None, bids=None, **extra):
    e = {"id": "b" + str(offset), "kind": "book", "ts": ORIGIN + offset,
         "exchange_ms": ORIGIN + offset, "snapshot_id": "s" + str(offset),
         "symbol": "LINKUSDT", "asks": [["100.1", "10"]] if asks is None else asks,
         "bids": [["99.9", "10"]] if bids is None else bids, **extra}
    r.ingest(e)
    return e


def opened(side="buy", qty="1", **settings):
    r = fixture(**settings)
    mark(r)
    d = decision(r, target=1 if side == "buy" else -1)
    intent(r, d, side=side, qty=qty, limit="101" if side == "buy" else "99")
    book(r)
    return r


def fills(r):
    return [row for row in r.state["ledger"] if row["kind"] == "fill"]


def test_contract_fails_closed():
    result = validate_contract(contract(), root=ROOT)
    assert result["structurally_valid"], result["errors"]
    assert not result["release_ready"] and not result["trading_authorized"]
    assert len(result["qualification_blockers"]) == 14
    changed = contract()
    changed["safety"]["orders_enabled"] = True
    assert not validate_contract(changed)["structurally_valid"]
    changed = contract()
    changed["temporal"]["purge_bars"] = 0
    assert not validate_contract(changed)["structurally_valid"]


def test_contract_hashes_and_evidence_are_not_permission():
    c = contract()
    c["source_pins"]["src/exia/genetic_primus/accounting_v2.py"] = "a" * 64
    assert any(x.startswith("source_changed:") for x in validate_contract(c, root=ROOT)["errors"])
    c = contract()
    c["evidence"] = {k: {"sha256": object_sha(k), "synthetic": False} for k in c["evidence"]}
    c["risk_budget"].update(capital_usdt="1000", maximum_order_notional_usdt="100",
                          maximum_loss_usdt="10", maximum_drawdown_fraction="0.01")
    report = validate_contract(c, root=ROOT)
    assert report["structurally_valid"] and not report["release_ready"]
    assert any(x.startswith("external_evidence_review_required:") for x in report["qualification_blockers"])


def test_model_identity_binds_cutoff_and_scaler():
    a, b = binding(), binding(origin=ORIGIN + 259200000)
    assert a["model_sha256"] != b["model_sha256"]
    assert a["model_sha256"] != contract()["research_model_sha256"]
    body = a["model"]
    assert body["train_end_exclusive_ms"] == ORIGIN - 7 * 14400000
    assert object_sha(body) == a["model_sha256"]
    must_fail(lambda: binding(origin=ORIGIN + 1), "align")
    bad = copy.deepcopy(a)
    bad["model"]["scaler"]["train_end_exclusive_ms"] += 1
    bad["model_sha256"] = object_sha(bad["model"])
    r = fixture()
    must_fail(lambda: Replay(settings=r.settings, rules=r.rules, models=[bad]), "temporal")


def test_rounding_and_minimums():
    buy = normalize_order("buy", "1.29", "100.19", RULE)
    sell = normalize_order("sell", "1.29", "100.11", RULE)
    assert buy == {"quantity": "1.2", "limit_price": "100.1", "reason": None}
    assert sell["limit_price"] == "100.2"
    assert normalize_order("buy", "0.09", "100", RULE)["reason"] == "rounded_to_zero"
    assert normalize_order("buy", "0.1", "1", RULE)["reason"] == "below_minimum_notional"
    for value in ("NaN", "Infinity", 1.25):
        must_fail(lambda: normalize_order("buy", value, "100", RULE))


def test_overlay_and_transition_preserve_abstention():
    model = binding()["model"]
    assert overlay_target(["0", "0", "1", "0"], 1, model) == 0
    assert overlay_target(["0", "0", "2", "0"], -1, model) == -1
    assert overlay_target(["0", "NaN", "2", "0"], 1, model) == 0
    kwargs = dict(current_quantity="1", sleeve_equity="100", buy_limit="100.1",
                  sell_limit="99.9", taker_fee="0.001", rule=RULE)
    assert transition_order(target=1, **kwargs) is None
    exit_order = transition_order(target=0, **kwargs)
    assert exit_order["reduce_only"] and Decimal(exit_order["quantity"]) == 1
    reverse = transition_order(target=-1, **kwargs)
    assert reverse["side"] == "sell" and not reverse["reduce_only"]


def test_latency_and_prearrival_snapshot():
    r = fixture()
    mark(r)
    intent(r, decision(r))
    book(r, 9)
    book(r, 10, exchange_ms=ORIGIN + 9)
    assert not fills(r)
    book(r, 11)
    assert len(fills(r)) == 1 and fills(r)[0]["ts"] == ORIGIN + 11


def test_partial_depth_and_price_protection():
    r = fixture()
    mark(r)
    intent(r, decision(r), qty="2", limit="100.2")
    book(r, asks=[["100.1", "0.3"], ["100.2", "0.4"], ["100.3", "10"]])
    assert r.state["positions"]["LINKUSDT"]["quantity"] == Decimal("0.7")
    order = r.state["orders"]["o1"]
    assert order["status"] == "CANCELLED" and order["remaining"] == Decimal("1.3")
    assert sum(Decimal(f["fee"]) for f in fills(r)) == Decimal("0.07011")


def test_participation_cap_and_snapshot_redelivery():
    r = fixture(participation="0.2")
    mark(r)
    intent(r, decision(r), qty="1")
    first = book(r, asks=[["100.1", "1"]])
    assert r.state["positions"]["LINKUSDT"]["quantity"] == Decimal("0.2")
    d = decision(r, 11)
    intent(r, d, 11, oid="o2")
    repeated = dict(first, id="redelivery", ts=ORIGIN + 21)
    r.ingest(repeated)
    assert len(fills(r)) == 1
    assert r.state["orders"]["o2"]["status"] == "PENDING"


def test_stale_future_and_invalid_book_are_fail_closed():
    r = fixture()
    mark(r)
    intent(r, decision(r))
    book(r, 40, exchange_ms=ORIGIN + 10)
    assert not fills(r)
    before = r.result()["state_sha256"]
    must_fail(lambda: book(r, 41, exchange_ms=ORIGIN + 42), "future")
    must_fail(lambda: book(r, 42, asks=[["99", "1"]]), "crossed")
    must_fail(lambda: book(r, 43, asks=[["100.15", "1"]]), "off-grid")
    assert r.result()["state_sha256"] == before
    book(r, 44)
    assert len(fills(r)) == 1


def test_idempotent_order_and_event_conflicts():
    r = fixture()
    mark(r)
    event = intent(r, decision(r))
    r.ingest(event)
    r.ingest(dict(event, id="retry", ts=ORIGIN + 1))
    assert len(r.state["orders"]) == 1
    must_fail(lambda: r.ingest(dict(event, quantity="2")), "event ID")
    must_fail(lambda: r.ingest(dict(event, id="conflict", ts=ORIGIN + 2, quantity="2")), "client order ID")
    book(r)
    assert len(fills(r)) == 1


def test_pending_order_requires_reconciliation():
    r = fixture()
    mark(r)
    d = decision(r)
    intent(r, d)
    intent(r, d, oid="o2")
    assert r.state["orders"]["o2"]["reason"] == "pending_order_requires_reconciliation"
    book(r)
    assert len(fills(r)) == 1


def test_long_close_fees_and_futures_cash():
    r = opened()
    assert r.state["cash"] == Decimal("999.8999")  # No spot principal debit.
    mark(r, 20, "110")
    d = decision(r, 20, target=0)
    intent(r, d, 20, oid="close", side="sell", qty="1", limit="109", reduce=True)
    book(r, 30, bids=[["110", "10"]], asks=[["110.1", "10"]])
    result = r.result()
    assert Decimal(result["wallet_cash"]) == Decimal("1009.6899")
    assert result["flat_and_reconciled"]
    assert fills(r)[-1]["realized_pnl"] == "9.9"


def test_short_and_partial_reversal_accounting():
    r = opened(side="sell", qty="2")
    mark(r, 20, "90")
    d = decision(r, 20, target=1)
    intent(r, d, 20, oid="reverse", qty="3", limit="91")
    book(r, 30, asks=[["90", "2.5"]], bids=[["89.9", "10"]])
    p = r.state["positions"]["LINKUSDT"]
    assert p == {"quantity": Decimal("0.5"), "entry": Decimal("90")}
    assert r.state["cash"] == Decimal("1019.3752")
    assert r.state["orders"]["reverse"]["remaining"] == Decimal("0.5")


def test_reduce_only_cannot_reverse_and_dust_is_visible():
    r = opened()
    d = decision(r, 20, target=0)
    intent(r, d, 20, oid="close", side="sell", qty="2", limit="99", reduce=True)
    book(r, 30)
    assert r.state["positions"]["LINKUSDT"]["quantity"] == 0
    assert r.state["orders"]["close"]["remaining"] == 1
    d = decision(r, 40, target=0)
    intent(r, d, 40, oid="again", side="sell", reduce=True)
    assert r.state["orders"]["again"]["reason"] == "reduce_only_would_not_reduce"
    r = opened(qty="0.1")
    d = decision(r, 20, target=0)
    intent(r, d, 20, oid="dust", side="sell", qty="0.1", limit="1", reduce=True)
    assert r.state["orders"]["dust"]["reason"] == "below_minimum_notional"
    assert r.result()["unresolved_positions"] == ["LINKUSDT"]


def test_funding_long_short_and_duplicate_settlement():
    for side, payment in (("buy", Decimal("-0.1")), ("sell", Decimal("0.1"))):
        r = opened(side=side)
        before = r.state["cash"]
        e = {"id": "fund", "kind": "funding", "ts": ORIGIN + 20, "settlement_ms": ORIGIN + 20,
             "symbol": "LINKUSDT", "rate": "0.001", "settlement_mark": "100"}
        r.ingest(e)
        r.ingest(dict(e, id="fund_redelivery"))
        assert r.state["cash"] == before + payment
        must_fail(lambda: r.ingest(dict(e, id="bad_fund", rate="0.002")), "conflict")


def test_funding_before_same_time_fill_and_no_late_rewrite():
    r = fixture()
    mark(r)
    intent(r, decision(r))
    r.ingest({"id": "fund", "kind": "funding", "ts": ORIGIN + 10,
              "settlement_ms": ORIGIN + 10, "symbol": "LINKUSDT", "rate": "0.1", "settlement_mark": "100"})
    book(r)
    assert r.state["cash"] == Decimal("999.8999")
    before = r.result()["state_sha256"]
    must_fail(lambda: r.ingest({"id": "late", "kind": "funding", "ts": ORIGIN + 20,
                               "settlement_ms": ORIGIN + 10, "symbol": "LINKUSDT", "rate": "0.1", "settlement_mark": "100"}), "late funding")
    must_fail(lambda: r.ingest({"id": "reordered", "kind": "funding", "ts": ORIGIN + 10,
                               "settlement_ms": ORIGIN + 10, "symbol": "LINKUSDT", "rate": "0.1", "settlement_mark": "100"}), "out-of-order")
    assert r.result()["state_sha256"] == before


def test_expiry_boundary_cancel_and_end_state():
    r = fixture()
    mark(r)
    intent(r, decision(r))
    book(r, 100)
    assert not fills(r) and r.state["orders"]["o1"]["status"] == "EXPIRED"
    r = fixture()
    mark(r)
    intent(r, decision(r))
    assert r.result()["pending_orders"] == ["o1"]
    r.ingest({"id": "c", "kind": "cancel", "ts": ORIGIN + 1, "client_oid": "o1"})
    book(r)
    assert not fills(r) and r.state["orders"]["o1"]["reason"] == "explicit_cancel"


def test_gross_order_and_loss_budgets():
    r = fixture(max_order_notional="2000")
    mark(r)
    intent(r, decision(r), qty="10")
    book(r)
    assert not fills(r) and r.state["orders"]["o1"]["reason"] == "gross_exposure_budget"
    r = fixture(max_order_notional="50")
    mark(r)
    intent(r, decision(r))
    assert r.state["orders"]["o1"]["reason"] == "maximum_order_notional"
    r = opened(max_loss="5")
    mark(r, 20, "90")
    assert r.state["risk_latched"]
    d = decision(r, 20, target=1)
    intent(r, d, 20, oid="add")
    assert r.state["orders"]["add"]["reason"] == "risk_latched"
    d = decision(r, 21, target=0)
    intent(r, d, 21, oid="exit", side="sell", limit="89", reduce=True)
    book(r, 31, bids=[["90", "10"]], asks=[["90.1", "10"]])
    assert r.state["positions"]["LINKUSDT"]["quantity"] == 0


def test_stale_marks_and_no_future_decisions():
    r = fixture(max_mark_age_ms=5)
    mark(r)
    d = decision(r, 6)
    intent(r, d, 6)
    assert r.state["orders"]["o1"]["reason"] == "missing_or_stale_mark"
    must_fail(lambda: decision(r, 7, feature_cutoff_ms=ORIGIN + 14400000), "closed 4h")
    r = opened()
    r.ingest({"id": "end", "kind": "clock", "ts": ORIGIN + 2000})
    assert r.result()["stale_marks"] == ["LINKUSDT"]
    assert not r.result()["flat_and_reconciled"]


def test_checkpoint_resume_is_exact_and_bound():
    full = opened()
    events = copy.deepcopy(full.events)
    split = fixture()
    for e in events[:-1]:
        split.ingest(e)
    cp = split.checkpoint()
    resumed = Replay.restore(cp, settings=split.settings, rules=split.rules, models=[binding()])
    resumed.ingest(events[-1])
    assert resumed.result() == full.result()
    bad = copy.deepcopy(cp)
    bad["events"][0]["price"] = "999"
    must_fail(lambda: Replay.restore(bad, settings=split.settings, rules=split.rules, models=[binding()]), "integrity")
    settings = dict(split.settings, latency_ms=11)
    must_fail(lambda: Replay.restore(cp, settings=settings, rules=split.rules, models=[binding()]), "mismatch")


def test_decimal_context_and_input_mutation_cannot_change_replay():
    normal = opened().result()
    with localcontext() as ctx:
        ctx.prec = 7
        limited = opened().result()
    assert normal == limited
    r = fixture()
    signed = binding()
    other = Replay(settings=r.settings, rules=r.rules, models=[signed])
    r.settings["initial_cash"] = "1"
    signed["model"]["policy"]["genome"][0] = 1
    assert other.settings["initial_cash"] == "1000"
    assert next(iter(other.models.values()))["policy"]["genome"] == [0, 0, 1, 1, -1]


def test_bounded_replay_and_abstention_ledger():
    r = fixture(max_events=2)
    mark(r)
    d = decision(r, target=0)
    assert r.state["decisions"][d]["target"] == 0
    assert r.state["decisions"][d]["model_sha256"] == next(iter(r.models))
    must_fail(lambda: r.ingest({"id": "over", "kind": "clock", "ts": ORIGIN + 1}), "budget")
    assert r.result()["event_count"] == 2 and not r.result()["qualification_eligible"]


def bound_bundle(r):
    api = runpy.run_path(str(ROOT / "tools/replay_genetic_primus_release_candidate_v1.py"))
    signed = binding()
    sha = api["implementation_identity"]()["sha256"]
    signed["model"]["provenance"]["implementation_sha256"] = sha
    signed["model_sha256"] = object_sha(signed["model"])
    events = copy.deepcopy(r.events)
    for event in events:
        if "model_sha256" in event:
            event["model_sha256"] = signed["model_sha256"]
    settings = copy.deepcopy(r.settings)
    settings["dataset_sha256"] = object_sha({"rules": r.rules, "events": events})
    bundle = {"schema_version": "exia.genetic_primus.execution_bundle/1", "settings": settings,
              "rules": r.rules, "models": [signed], "events": events}
    return api, bundle, sha


def input_bundle():
    return bound_bundle(opened())


def demo_bundle():
    r = fixture()
    mark(r)
    intent(r, decision(r), qty="2", limit="100.2")
    book(r, asks=[["100.1", "0.3"], ["100.2", "0.4"], ["100.3", "10"]])
    r.ingest({"id": "demo_funding", "kind": "funding", "ts": ORIGIN + 20,
              "settlement_ms": ORIGIN + 20, "symbol": "LINKUSDT", "rate": "0.001", "settlement_mark": "102"})
    d = decision(r, 20, target=0)
    intent(r, d, 20, oid="close", side="sell", qty="0.7", limit="101", reduce=True)
    book(r, 30, bids=[["101.9", "10"]], asks=[["102.1", "10"]])
    assert r.result()["flat_and_reconciled"]
    assert r.state["cash"] == Decimal("1001.00716")
    return bound_bundle(r)[1]


def test_bundle_replays_and_pins_source_data_model():
    api, bundle, sha = input_bundle()
    result = api["replay_bundle"](bundle, contract(), sha)
    assert result["event_count"] == 4 and result["unresolved_positions"] == ["LINKUSDT"]
    assert not result["qualification_eligible"] and result["evidence_kind"] == "synthetic"
    assert result["state"]["decisions"]["d0"]["execution_dataset_sha256"] == bundle["settings"]["dataset_sha256"]
    tampered = copy.deepcopy(bundle)
    tampered["events"][-1]["asks"][0][0] = "100.2"
    must_fail(lambda: api["replay_bundle"](tampered, contract(), sha), "dataset hash")
    must_fail(lambda: api["replay_bundle"](bundle, contract(), "f" * 64), "implementation hash")


def test_bundle_model_cannot_claim_another_policy():
    api, bundle, sha = input_bundle()
    signed = bundle["models"][0]
    signed["model"]["policy"]["same_direction"] = "rebalance_every_tick"
    signed["model_sha256"] = object_sha(signed["model"])
    must_fail(lambda: api["replay_bundle"](bundle, contract(), sha), "manifest mismatch")


def test_absolute_drawdown_and_funding_latch():
    r = opened(max_drawdown="0.001", max_loss="999")
    mark(r, 20, "90")
    assert "maximum_drawdown" in r.result()["risk_reasons"]
    r = opened(max_loss="1")
    r.ingest({"id": "large_fund", "kind": "funding", "ts": ORIGIN + 20,
              "settlement_ms": ORIGIN + 20, "symbol": "LINKUSDT", "rate": "0.02", "settlement_mark": "100"})
    assert "maximum_loss" in r.result()["risk_reasons"]


def test_model_changes_only_in_nonoverlapping_origin():
    first, second = binding(), binding(origin=ORIGIN + 259200000)
    r = fixture()
    r = Replay(settings=r.settings, rules=r.rules, models=[first, second])
    decision(r)
    second_start = ORIGIN + 259200000
    d = {"id": "next_origin", "kind": "decision", "ts": second_start,
         "symbol": "LINKUSDT", "model_sha256": second["model_sha256"],
         "feature_cutoff_ms": second_start, "features": ["0", "0", "2", "0"], "proposal": 1}
    r.ingest(d)
    must_fail(lambda: r.ingest(dict(d, id="old_model", model_sha256=first["model_sha256"])), "outside")
    assert len({row["model_sha256"] for row in r.state["ledger"] if row["kind"] == "decision"}) == 2


def test_fee_and_instrument_validity_prevent_stale_execution():
    r = fixture(fee_valid_until_exclusive_ms=ORIGIN + 10)
    mark(r)
    intent(r, decision(r))
    book(r, 10)
    assert not fills(r) and r.state["orders"]["o1"]["status"] == "EXPIRED"
    r = fixture(fee_valid_from_ms=ORIGIN + 1)
    mark(r)
    intent(r, decision(r))
    assert r.state["orders"]["o1"]["reason"] == "rules_or_fees_outside_validity"
    r = fixture()
    rules = copy.deepcopy(r.rules)
    rules["LINKUSDT"]["valid_from_ms"] = ORIGIN + 1
    r = Replay(settings=r.settings, rules=rules, models=[binding()])
    mark(r)
    intent(r, decision(r))
    assert r.state["orders"]["o1"]["reason"] == "rules_or_fees_outside_validity"


def test_projected_loss_and_wrong_overlay_action_are_rejected():
    r = fixture(max_loss="0.15")
    mark(r)
    intent(r, decision(r))
    book(r)
    assert not fills(r) and r.state["orders"]["o1"]["reason"] == "projected_maximum_loss"
    r = fixture()
    mark(r)
    intent(r, decision(r, target=0))
    assert r.state["orders"]["o1"]["reason"] == "intent_contradicts_overlay_target"


def test_demo_has_partial_fills_funding_and_reconciled_close():
    bundle = demo_bundle()
    api, _, sha = input_bundle()
    result = api["replay_bundle"](bundle, contract(), sha)
    assert result["event_count"] == 8 and result["flat_and_reconciled"]
    assert Decimal(result["wallet_cash"]) == Decimal("1001.00716")


def test_multiple_symbols_share_the_account_exposure_budget():
    r = fixture(max_order_notional="2000")
    rules = {"LINKUSDT": RULE, "BTCUSDT": RULE}
    r = Replay(settings=r.settings, rules=rules, models=[binding()])
    mark(r)
    mark(r, id="m_btc", symbol="BTCUSDT")
    link = decision(r)
    btc = decision(r, suffix="btc", symbol="BTCUSDT")
    intent(r, link, qty="6")
    intent(r, btc, oid="btc", symbol="BTCUSDT", qty="6")
    book(r)
    book(r, id="b_btc", symbol="BTCUSDT")
    assert len(fills(r)) == 1
    assert r.state["orders"]["btc"]["reason"] == "gross_exposure_budget"


def test_mark_to_market_conservation_through_partial_round_trip():
    api, _, sha = input_bundle()
    result = api["replay_bundle"](demo_bundle(), contract(), sha)
    ledger = result["state"]["ledger"]
    fee = sum((Decimal(x["fee"]) for x in ledger if x["kind"] == "fill"), Decimal(0))
    pnl = sum((Decimal(x["realized_pnl"]) for x in ledger if x["kind"] == "fill"), Decimal(0))
    funding = sum((Decimal(x["payment"]) for x in ledger if x["kind"] == "funding"), Decimal(0))
    assert Decimal(result["wallet_cash"]) == Decimal(1000) + pnl - fee + funding
    assert Decimal(result["marked_equity"]) == Decimal(result["wallet_cash"])


GROUPS = {
    "contract": [test_contract_fails_closed, test_contract_hashes_and_evidence_are_not_permission,
                 test_model_identity_binds_cutoff_and_scaler, test_rounding_and_minimums,
                 test_overlay_and_transition_preserve_abstention],
    "execution": [test_latency_and_prearrival_snapshot, test_partial_depth_and_price_protection,
                  test_participation_cap_and_snapshot_redelivery, test_stale_future_and_invalid_book_are_fail_closed,
                  test_idempotent_order_and_event_conflicts, test_pending_order_requires_reconciliation],
    "accounting": [test_long_close_fees_and_futures_cash, test_short_and_partial_reversal_accounting,
                   test_reduce_only_cannot_reverse_and_dust_is_visible,
                   test_funding_long_short_and_duplicate_settlement,
                   test_funding_before_same_time_fill_and_no_late_rewrite,
                   test_mark_to_market_conservation_through_partial_round_trip],
    "safety": [test_expiry_boundary_cancel_and_end_state, test_gross_order_and_loss_budgets,
               test_stale_marks_and_no_future_decisions, test_checkpoint_resume_is_exact_and_bound,
               test_decimal_context_and_input_mutation_cannot_change_replay,
               test_bounded_replay_and_abstention_ledger,
               test_multiple_symbols_share_the_account_exposure_budget],
    "bundle": [test_bundle_replays_and_pins_source_data_model, test_bundle_model_cannot_claim_another_policy,
               test_absolute_drawdown_and_funding_latch, test_model_changes_only_in_nonoverlapping_origin,
               test_fee_and_instrument_validity_prevent_stale_execution,
               test_projected_loss_and_wrong_overlay_action_are_rejected,
               test_demo_has_partial_fills_funding_and_reconciled_close],
}
