"""Deterministic regression checks; no pytest, bars, waits or original imports."""
from __future__ import annotations

import copy
import math
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"))
    count = 0
    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"regression {count + 1}")
        count += 1
    def rejects(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except (ValueError, TypeError, OverflowError):
            check(True)
        else:
            check(False)
    contract = api["expected_contract"]()
    check(api["validate_contract"](contract) == contract)
    for key in api["SAFETY"]:
        modified = copy.deepcopy(contract); modified["safety"][key] = True
        rejects(api["validate_contract"], modified)
    modified = copy.deepcopy(contract); modified["boundary"]["retrain_event_causes_trade"] = 0
    rejects(api["validate_contract"], modified)
    modified = copy.deepcopy(contract); modified["time"]["purge_bars"] = 6
    rejects(api["validate_contract"], modified)
    modified = copy.deepcopy(contract); modified["boundary"]["force_flat_timestamps"] = [1]
    rejects(api["validate_contract"], modified)
    day = api["DAY_MS"]; bar = api["BAR_MS"]
    plan = api["boundary_plan"](contract, 0, 9 * day)
    check(len(plan["origins"]) == 3)
    check(plan["forced_origin_exits"] == [])
    check(plan["explicit_accounting_parameters"]["force_flat_timestamps"] == [])
    check(plan["explicit_accounting_parameters"]["record_details"] is True)
    check(plan["terminal_close_ms"] == 9 * day)
    for origin in plan["origins"]:
        check(origin["deployment_end_exclusive_ms"] - origin["origin_open_ms"] == 3 * day)
        check(origin["training_feature_open_before_ms"] == origin["origin_open_ms"] - 7 * bar)
        check(origin["training_label_end_at_or_before_ms"] == origin["train_end_exclusive_ms"])
        check(origin["forced_trade"] is False and origin["capital_reset"] is False)
    for start, end in ((0, 0), (1, 3 * day), (0, 3 * day + 1), (0, 4 * day),
                       (-1, 3 * day), (False, 3 * day), (0, 3.0 * day), (0, 303 * day)):
        rejects(api["boundary_plan"], contract, start, end)
    shifted = api["boundary_plan"](contract, bar, 3 * day + bar)
    check(shifted["origins"][0]["origin_open_ms"] == bar)
    base = dict(position=1, target=1, entry_model_sha256="1" * 64,
                decision_model_sha256="2" * 64, origin_open_ms=3 * day,
                signal_close_ms=3 * day, model_available_ms=3 * day,
                interval_end_ms=9 * day)
    for position in (-1, 0, 1):
        for target in (-1, 0, 1):
            values = dict(base, position=position, target=target,
                          entry_model_sha256="1" * 64 if position else None)
            result = api["boundary_transition"](contract, **values)
            expected_events = (["hold"] if position else ["abstain"]) if position == target else (
                (["exit"] if position else []) + (["enter"] if target else []))
            expected_entry = values["entry_model_sha256"] if position == target else (
                "2" * 64 if target else None)
            check(result["semantic_events"] == expected_events)
            check(result["position_entry_model_sha256"] == expected_entry)
            check(result["decision_model_sha256"] == "2" * 64)
            check(result["capital_reset"] is False and result["orders_emitted"] is False)
    for key, value in (("position", True), ("target", 2), ("entry_model_sha256", None),
                       ("decision_model_sha256", "bad"), ("decision_model_sha256", "A" * 64),
                       ("signal_close_ms", 3 * day - bar), ("model_available_ms", 3 * day + 1),
                       ("origin_open_ms", 3 * day + 1), ("interval_end_ms", 3 * day),
                       ("interval_end_ms", -1), ("signal_close_ms", False)):
        rejects(api["boundary_transition"], contract, **dict(base, **{key: value}))
    rejects(api["boundary_transition"], contract, **dict(base, position=0))
    check(api["boundary_transition"](contract, **dict(base, model_available_ms=3 * day - 1))["semantic_events"] == ["hold"])
    for price in (90.0, 100.0, 110.0):
        for fee in (0.0, 16.0, 24.0):
            result = api["long_cash_identity_fixture"](1.0, 100.0, price, fee)
            check(math.isclose(result["gross_pnl"] - result["fees"], result["net_pnl"], abs_tol=1e-12))
            check(math.isclose(1 + result["net_pnl"], result["ending_equity"], abs_tol=1e-12))
            check(result["quantity"] * 100.0 + result["quantity"] * 100.0 * fee / 20000.0 <= 1.0 + 1e-12)
    zero = api["long_cash_identity_fixture"](1.0, 100.0, 110.0, 0.0)
    paid = api["long_cash_identity_fixture"](1.0, 100.0, 110.0, 16.0)
    check(abs(zero["gross_pnl"] - paid["gross_pnl"]) > 1e-6)
    check(not math.isclose(zero["gross_pnl"] - paid["fees"], paid["net_pnl"], abs_tol=1e-6))
    for values in ((False, 100, 110, 16), (0, 100, 110, 16), (1, 0, 110, 16),
                   (1, 100, -1, 16), (1, 100, 110, -1), (1, 100, 110, 101),
                   (1, 100, float("nan"), 16), (1, 100, 110, float("inf"))):
        rejects(api["long_cash_identity_fixture"], *values)
    rejects(api["long_cash_identity_fixture"], 1e308, 1e-308, 110.0, 16.0)
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_contract_v1.py"))
    rejects(builder["strict_json"], '{"x":0,"x":1}')
    rejects(builder["strict_json"], '{"x":NaN}')
    _, actual, artifact, summary, _, verify = builder["prepare"]()
    verify()
    check(api["digest"](actual) == api["digest"](contract))
    check(artifact["launch_completed"] is False)
    check(artifact["financial_result"] is False)
    check(summary["new_model_fits"] == summary["new_financial_evaluations"] == 0)
    check(summary["reviewed_source_reference_hashes"] == 3)
    check(all(value is False for value in summary["safety"].values()))
    print(f"boundary engineering regressions: {count} passed; pure synthetic/AST only")


if __name__ == "__main__":
    main()
