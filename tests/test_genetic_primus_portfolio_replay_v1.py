"""Synthetic, deterministic and small. Never use these prices as market evidence."""
import copy
import runpy
import sys
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/exia/genetic_primus"))
from release_contract_v1 import bind_model, object_sha, utc_ms
from portfolio_replay_v1 import ROLES, TF, ema_proposal, run_comparison, sleeve_order, validate_portfolio_contract

START = utc_ms("2026-09-10T00:00:00Z")
UP = ["0.01", "0.01", "0.02", "0.8"]
WEAK = ["0.01", "0.01", "0.005", "0.3"]
DOWN = ["-0.01", "-0.01", "-0.02", "0.8"]
FLAT = ["0", "0", "0", "0"]


def context():
    api = runpy.run_path(str(ROOT / "tools/run_genetic_primus_portfolio_replay_v1.py"))
    return api["read_inputs"]()


def reseal(b):
    b["settings"]["dataset_sha256"] = object_sha({"rules": b["rules"], "events": b["events"]})
    return b


def fixture(*, offsets=(0,), features=None, prices=None, depth="10", active=None, **settings):
    c, release, identity = context()
    symbols = sorted(release["symbol_map"].values())
    rules = {s: {"price_tick": "0.1", "quantity_step": "0.1", "minimum_quantity": "0.1",
                 "minimum_notional": "1", "source_sha256": object_sha("SYNTHETIC_RULES:" + s),
                 "valid_from_ms": START, "valid_until_exclusive_ms": START + 10 * 86400000}
             for s in symbols}
    parameters = {"contract_sha256": object_sha(release), "dataset_sha256": "0" * 64,
                  "fee_schedule_sha256": object_sha("SYNTHETIC_FEES"), "evidence_kind": "synthetic",
                  "initial_cash": "1000", "taker_fee": "0.001", "latency_ms": 10,
                  "max_quote_age_ms": 20, "max_mark_age_ms": 1000, "order_ttl_ms": 100,
                  "participation": "1", "max_order_notional": "900", "max_gross_ratio": "1",
                  "max_loss": "200", "max_drawdown": "0.2", "max_events": 512, "max_levels": 8,
                  "fee_valid_from_ms": START, "fee_valid_until_exclusive_ms": START + 10 * 86400000}
    parameters.update(settings)
    origins = sorted({START + (offset // 259200000) * 259200000 for offset in offsets})
    models = []
    for origin in origins:
        scaler = {"median": ["0"] * 4, "scale": ["1", "1", "0.01", "1"], "fitted_rows": 16,
                  "fit_values_sha256": object_sha(["SYNTHETIC_FIT", origin]),
                  "train_start_ms": utc_ms("2022-01-01T00:00:00Z"), "train_end_exclusive_ms": origin - 7 * TF}
        models.append(bind_model(release, origin_start_ms=origin, scaler=scaler,
                                 data_sha256=object_sha(["SYNTHETIC_TRAIN", origin]),
                                 train_rows_sha256=object_sha(["SYNTHETIC_ROWS", origin]),
                                 folds_sha256=object_sha("SYNTHETIC_FOLDS"),
                                 resample_bank_sha256=object_sha("SYNTHETIC_SELECTION_BANK"),
                                 audit_bank_sha256=object_sha("SYNTHETIC_AUDIT_BANK"),
                                 implementation_sha256=identity["sha256"]))
    events = []
    for i, offset in enumerate(offsets):
        price = Decimal((prices or ["100"] * len(offsets))[i])
        raw = (features or [UP] * len(offsets))[i]
        for s in symbols:
            events.append({"id": f"m{i}:{s}", "kind": "mark", "ts": START + offset,
                           "exchange_ms": START + offset, "symbol": s, "price": str(price)})
        for s in symbols:
            values = raw if active is None or s in active[i] else FLAT
            events.append({"id": f"d{i}:{s}", "kind": "signal", "ts": START + offset,
                           "symbol": s, "feature_cutoff_ms": START + offset, "features": list(values)})
        for s in symbols:
            events.append({"id": f"b{i}:{s}", "kind": "book", "ts": START + offset + 10,
                           "exchange_ms": START + offset + 10, "snapshot_id": f"snap{i}:{s}", "symbol": s,
                           "bids": [[str(price - Decimal("0.1")), depth]],
                           "asks": [[str(price + Decimal("0.1")), depth]]})
    return reseal({"schema_version": "exia.genetic_primus.portfolio_bundle/1", "settings": parameters,
                   "rules": rules, "models": models, "events": events,
                   "profile": {"protection_fraction": "0.002", "max_input_events": 256,
                               "feature_dataset_sha256": object_sha("SYNTHETIC_RAW_FEATURES")},
                   "implementation_sha256": identity["sha256"]})


def run(b, **options):
    c, release, identity = context()
    return run_comparison(b, c, release, identity["sha256"], **options)


def state(result, role="GA"):
    return result["candidates"][role]["execution"]["state"]


def fills(result, role="GA"):
    return [x for x in state(result, role)["ledger"] if x["kind"] == "fill"]


def must_fail(fn, word=None):
    try:
        fn()
    except ValueError as exc:
        if word:
            assert word in str(exc), str(exc)
    else:
        raise AssertionError("expected fail-closed ValueError")


def test_raw_ema_rule_and_missing_features():
    assert ema_proposal(UP) == 1 and ema_proposal(DOWN) == -1
    assert ema_proposal(WEAK) == 1 and ema_proposal(FLAT) == 0
    assert ema_proposal(["1", "-1", "1", "0.8"]) == 0
    assert ema_proposal([None, "1", "1", "0.8"]) == 0
    assert ema_proposal(["1", "1", "1", "0.299"]) == 0
    must_fail(lambda: ema_proposal(["1"]), "four")


def test_zero_collateral_exit_and_same_direction_hold():
    rule = next(iter(fixture()["rules"].values()))
    args = dict(quantity="1", equity="125", mark="100", protection_fraction="0.002", taker_fee="0.001", rule=rule)
    assert sleeve_order(target=1, **args) is None
    for eq in ("0", "-1"):
        x = sleeve_order(target=0, **dict(args, equity=eq))
        assert x["reduce_only"] and x["side"] == "sell" and Decimal(x["quantity"]) == 1
    assert sleeve_order(target=1, **dict(args, quantity="0", equity="0")) is None


def test_frozen_contract_and_universe():
    c, release, _ = context()
    validate_portfolio_contract(c, release)
    changed = copy.deepcopy(c)
    changed["rules"]["capital"] = "reallocate_to_winners"
    must_fail(lambda: validate_portfolio_contract(changed, release), "frozen")
    changed = copy.deepcopy(c)
    changed["safety"]["orders_enabled"] = 0
    must_fail(lambda: validate_portfolio_contract(changed, release), "false")
    b = fixture()
    b["rules"].pop(next(iter(b["rules"])))
    must_fail(lambda: run(reseal(b)), "eight")


def test_manifest_tampering_and_model_roles():
    b = fixture()
    bad = copy.deepcopy(b)
    bad["models"][0]["model"]["train_end_exclusive_ms"] += TF
    bad["models"][0]["model_sha256"] = object_sha(bad["models"][0]["model"])
    must_fail(lambda: run(bad), "manifest")
    r = run(b)
    shas = [next(iter(r["candidates"][role]["model_manifests"])) for role in ROLES]
    assert len(set(shas)) == 3
    for role in ("EMA", "NoTrade"):
        m = next(iter(r["candidates"][role]["model_manifests"].values()))
        assert m["scaler"] is None and "genome" not in m["policy"]


def test_decimal_context_does_not_change_comparison():
    b = fixture(offsets=(0, TF), features=[UP, FLAT], prices=["100", "101"])
    a = run(b)
    with localcontext() as ctx:
        ctx.prec = 7
        other = run(b)
    assert a == other


def test_identical_targets_identical_economics():
    r = run(fixture(offsets=(0, TF), features=[UP, FLAT], prices=["100", "101"]))
    for key in ("ending_equity", "marked_return_fraction", "fees", "funding", "fill_notional"):
        assert r["candidates"]["GA"][key] == r["candidates"]["EMA"][key]
    assert r["candidates"]["GA"]["portfolio_path"] == r["candidates"]["EMA"]["portfolio_path"]
    assert all(Decimal(x["GA_minus_EMA_equity"]) == 0 for x in r["paired_equity_path"])


def test_notrade_remains_cash():
    r = run(demo_bundle())
    n = r["candidates"]["NoTrade"]
    assert Decimal(n["ending_equity"]) == 1000 and n["fills"] == 0
    assert Decimal(n["fees"]) == 0 and Decimal(n["funding"]) == 0
    assert len(state(r, "NoTrade")["decisions"]) == 16


def test_weak_momentum_is_ema_trade_but_ga_abstention():
    r = run(fixture(features=[WEAK]))
    assert r["candidates"]["EMA"]["fills"] == 8
    assert r["candidates"]["GA"]["fills"] == 0
    assert all(d["proposal"] == 1 and d["target"] == 0 for d in state(r)["decisions"].values())


def test_abstention_does_not_reallocate_capital():
    b = fixture()
    symbol = sorted(b["rules"])[0]
    all_active = run(b)
    one_active = run(fixture(active=[{symbol}]))
    qty = state(all_active)["positions"][symbol]["quantity"]
    assert qty == state(one_active)["positions"][symbol]["quantity"]
    assert Decimal(qty) == Decimal("1.2")
    assert all(Decimal(v) == 125 for s, v in state(one_active)["sleeve_cash"].items() if s != symbol)


def test_candidate_iteration_order_is_irrelevant():
    b = demo_bundle()
    assert run(b) == run(b, candidate_order=("NoTrade", "EMA", "GA"))


def test_no_future_quote_for_protection_or_sizing():
    b = fixture()
    for e in b["events"]:
        if e["kind"] == "book":
            e["asks"] = [["110", "10"]]
            e["bids"] = [["109.9", "10"]]
    r = run(reseal(b))
    assert r["candidates"]["EMA"]["fills"] == 0
    orders = state(r, "EMA")["orders"].values()
    assert all(Decimal(o["limit_price"]) == Decimal("100.2") for o in orders)


def test_partial_position_is_not_topped_up():
    r = run(fixture(offsets=(0, TF), depth="0.4"))
    assert r["candidates"]["GA"]["fills"] == 8
    assert all(Decimal(p["quantity"]) == Decimal("0.4") for p in state(r)["positions"].values())
    assert all(d["controller_action"] == "hold_or_flat" for d in state(r)["decisions"].values() if d["ts"] > START)


def test_abstention_partial_exit_leaves_visible_residual():
    b = fixture(offsets=(0, TF), features=[UP, WEAK], depth="0.4")
    for e in b["events"]:
        if e["kind"] == "book" and e["ts"] > START + TF:
            e["bids"][0][1] = "0.2"
    r = run(reseal(b))
    assert all(Decimal(p["quantity"]) == Decimal("0.2") for p in state(r)["positions"].values())
    assert not r["candidates"]["GA"]["execution"]["flat_and_reconciled"]
    assert all(Decimal(p["quantity"]) == Decimal("0.4") for p in state(r, "EMA")["positions"].values())


def test_origin_boundary_carries_positions_and_updates_model_sha():
    r = run(fixture(offsets=(0, 259200000)))
    assert r["candidates"]["GA"]["fills"] == 8
    assert len({d["model_sha256"] for d in state(r)["decisions"].values()}) == 2
    assert all(Decimal(p["quantity"]) == Decimal("1.2") for p in state(r)["positions"].values())


def test_ema_partial_reversal_is_real_position():
    b = fixture(offsets=(0, TF), features=[UP, DOWN], prices=["100", "105"])
    for e in b["events"]:
        if e["kind"] == "book" and e["ts"] > START + TF:
            e["bids"][0][1] = "1.5"
    r = run(reseal(b))
    assert all(Decimal(p["quantity"]) == Decimal("-0.3") for p in state(r, "EMA")["positions"].values())
    assert all(Decimal(p["quantity"]) == 0 for p in state(r)["positions"].values())


def demo_bundle():
    b = fixture(offsets=(0, TF), features=[UP, FLAT], prices=["100", "101"], depth="0.4")
    settlements = [{"id": "fund:" + s, "kind": "funding", "ts": START + TF,
                    "settlement_ms": START + TF, "symbol": s, "rate": "0.001", "settlement_mark": "101"}
                   for s in sorted(b["rules"])]
    at = next(i for i, e in enumerate(b["events"]) if e["ts"] >= START + TF)
    b["events"][at:at] = settlements
    return reseal(b)


def test_funding_cash_and_mark_reconcile_per_sleeve():
    r = run(demo_bundle())
    for role in ROLES:
        c = r["candidates"][role]
        st = state(r, role)
        assert sum(Decimal(v) for v in st["sleeve_cash"].values()) == Decimal(st["cash"])
        assert c["execution"]["flat_and_reconciled"]
    assert Decimal(r["candidates"]["GA"]["funding"]) == Decimal("-0.3232")
    assert Decimal(r["candidates"]["GA"]["ending_equity"]) == Decimal("1001.5936")


def test_profit_is_not_transferred_to_another_instrument():
    syms = sorted(fixture()["rules"])
    b = fixture(offsets=(0, TF), active=[{syms[0]}, {syms[0], syms[1]}], prices=["100", "110"])
    r = run(b)
    assert Decimal(state(r)["positions"][syms[1]]["quantity"]) == Decimal("1.1")
    assert Decimal(state(r)["positions"][syms[0]]["quantity"]) == Decimal("1.2")


def test_sleeve_budget_cannot_borrow_idle_capital():
    b = fixture(features=[DOWN], active=[{sorted(fixture()["rules"])[0]}])
    s = sorted(b["rules"])[0]
    # A new mark after the decision but before the short fill raises its risk.
    i = next(i for i, e in enumerate(b["events"]) if e["kind"] == "book")
    b["events"].insert(i, {"id": "risk_mark", "kind": "mark", "ts": START + 5,
                          "exchange_ms": START + 5, "symbol": s, "price": "120"})
    r = run(reseal(b))
    assert r["candidates"]["EMA"]["fills"] == 0
    assert next(iter(state(r, "EMA")["orders"].values()))["reason"] == "own_sleeve_exposure_budget"


def test_risk_latch_flattens_at_next_decision_without_forced_fill():
    r = run(fixture(offsets=(0, TF), prices=["100", "90"], max_loss="10"))
    assert r["candidates"]["EMA"]["execution"]["risk_latched"]
    assert r["candidates"]["EMA"]["execution"]["flat_and_reconciled"]
    assert all(d["target"] == 0 for d in state(r, "EMA")["decisions"].values() if d["ts"] > START)


def test_full8_missing_duplicate_and_future_signals_fail_closed():
    b = fixture()
    missing = copy.deepcopy(b)
    missing["events"] = [e for e in missing["events"] if e["id"] != "d0:" + sorted(b["rules"])[0]]
    must_fail(lambda: run(reseal(missing)), "full8")
    future = copy.deepcopy(b)
    next(e for e in future["events"] if e["kind"] == "signal")["feature_cutoff_ms"] += TF
    must_fail(lambda: run(reseal(future)), "4h")
    dup = copy.deepcopy(b)
    signal = next(e for e in dup["events"] if e["kind"] == "signal")
    i = next(i for i, e in enumerate(dup["events"]) if e["kind"] == "book")
    dup["events"].insert(i, dict(signal, id="another_same_symbol"))
    must_fail(lambda: run(reseal(dup)), "full8")


def test_tampered_dataset_and_supplied_orders_are_rejected():
    b = fixture()
    b["events"][-1]["asks"][0][0] = "100.2"
    must_fail(lambda: run(b), "dataset hash")
    b = fixture()
    b["events"][-1]["kind"] = "intent"
    must_fail(lambda: run(reseal(b)), "no supplied orders")


def test_stale_marks_do_not_create_orders():
    b = fixture(max_mark_age_ms=5)
    for e in b["events"]:
        if e["kind"] == "mark":
            e["ts"] -= 10
            e["exchange_ms"] -= 10
    r = run(reseal(b))
    assert all(c["fills"] == 0 for c in r["candidates"].values())
    assert all(d["controller_action"] == "missing_or_stale_mark" for d in state(r)["decisions"].values())


def test_restart_by_immutable_bundle_is_exact_and_terminal_is_open():
    b = fixture()
    before = copy.deepcopy(b)
    a, repeated = run(b), run(b)
    assert a == repeated and b == before
    assert len(a["candidates"]["GA"]["execution"]["unresolved_positions"]) == 8
    assert a["candidates"]["GA"]["fills"] == 8


def test_duplicate_delivery_does_not_repeat_decision():
    b = fixture()
    original = run(b)
    b["events"].append(copy.deepcopy(b["events"][0]))
    repeated = run(reseal(b))
    assert repeated["unique_input_events"] == original["unique_input_events"]
    assert len(state(repeated)["decisions"]) == 8
    assert repeated["candidates"]["GA"]["ending_equity"] == original["candidates"]["GA"]["ending_equity"]


def test_future_suffix_cannot_change_earlier_economics():
    a = fixture(offsets=(0, TF), features=[UP, FLAT])
    b = copy.deepcopy(a)
    for e in b["events"]:
        if e["kind"] == "book" and e["ts"] > START + TF:
            e["bids"] = [["99.8", "10"]]
    left, right = run(a), run(reseal(b))
    keys = ("ts", "symbol", "signed_quantity", "price", "fee", "wallet_cash")
    economic = lambda result: [{k: f[k] for k in keys} for f in fills(result) if f["ts"] < START + TF]
    assert economic(left) == economic(right)


def test_bounds_missing_origin_and_no_authority():
    b = fixture()
    bad = copy.deepcopy(b)
    bad["profile"]["max_input_events"] = 1025
    must_fail(lambda: run(bad), "1024")
    bad = fixture(offsets=(0, 259200000))
    bad["models"].pop()
    must_fail(lambda: run(bad), "origin")
    r = run(b)
    assert r["within_replay_candidates_comparable"] and not r["historical_research_results_comparable"]
    assert not r["qualification_eligible"] and not r["trading_authorized"]


def test_bad_feature_format_is_not_silently_notrade():
    must_fail(lambda: ema_proposal([0.01, "1", "1", "0.8"]), "decimal strings")
    must_fail(lambda: ema_proposal(["broken", "1", "1", "0.8"]), "invalid raw")
    assert ema_proposal(["NaN", "1", "1", "0.8"]) == 0
    b = fixture()
    for e in b["events"]:
        if e["kind"] == "signal":
            e["features"] = [None, "1", "1", "0.8"]
    r = run(reseal(b))
    assert r["common_nonzero_ema_opportunities"] == 0
    assert all(c["fills"] == 0 for c in r["candidates"].values())


def test_generated_intent_budget_and_freshness_are_bound():
    b = fixture(max_events=24)
    must_fail(lambda: run(b), "all possible generated intents")
    b = fixture()
    b["settings"]["order_ttl_ms"] = TF
    must_fail(lambda: run(b), "shorter")
    b = fixture()
    r = run(b)
    assert r["common_nonzero_ema_opportunities"] == 8
    assert r["candidates"]["NoTrade"]["nonzero_target_decisions"] == 0


GROUPS = {
    "policy": [test_raw_ema_rule_and_missing_features, test_zero_collateral_exit_and_same_direction_hold,
               test_frozen_contract_and_universe, test_manifest_tampering_and_model_roles,
               test_decimal_context_does_not_change_comparison],
    "comparison": [test_identical_targets_identical_economics, test_notrade_remains_cash,
                   test_weak_momentum_is_ema_trade_but_ga_abstention,
                   test_abstention_does_not_reallocate_capital, test_candidate_iteration_order_is_irrelevant],
    "execution": [test_no_future_quote_for_protection_or_sizing, test_partial_position_is_not_topped_up,
                  test_abstention_partial_exit_leaves_visible_residual,
                  test_origin_boundary_carries_positions_and_updates_model_sha,
                  test_ema_partial_reversal_is_real_position],
    "accounting": [test_funding_cash_and_mark_reconcile_per_sleeve, test_profit_is_not_transferred_to_another_instrument,
                   test_sleeve_budget_cannot_borrow_idle_capital,
                   test_risk_latch_flattens_at_next_decision_without_forced_fill],
    "safety": [test_full8_missing_duplicate_and_future_signals_fail_closed,
               test_tampered_dataset_and_supplied_orders_are_rejected, test_stale_marks_do_not_create_orders,
               test_restart_by_immutable_bundle_is_exact_and_terminal_is_open,
               test_duplicate_delivery_does_not_repeat_decision, test_future_suffix_cannot_change_earlier_economics,
               test_bounds_missing_origin_and_no_authority,
               test_bad_feature_format_is_not_silently_notrade,
               test_generated_intent_budget_and_freshness_are_bound],
}
