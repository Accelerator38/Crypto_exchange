"""Deterministic validation/attribution faults; no pytest or market data."""
from __future__ import annotations

import copy
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def main():
    api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"))
    adapter = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_stub_adapter_v1.py"))
    wrapper = runpy.run_path(str(ROOT / "src/exia/genetic_primus/strict_synthetic_wrapper_v1.py"))
    contract = api["expected_contract"]()
    count = 0
    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"wrapper assertion {count + 1}")
        count += 1
    def error(fn, *args):
        try:
            fn(*args)
        except (ValueError, TypeError, AttributeError, AssertionError) as exc:
            return type(exc).__name__, str(exc)
        raise AssertionError("invalid case accepted")
    callback_events = []
    def forbidden(**kwargs):
        callback_events.append(True)
        raise AssertionError("must not invoke an arbitrary callee")
    fake = ModuleType("untrusted_wrapper_callee")
    fake.simulate_cash_targets = forbidden
    for case in adapter["CASES"]:
        value = adapter["fixture"](api, contract, case)
        proof = wrapper["validate_inputs"](api, contract, value, "normal")
        check(proof["validated_before_callee"] is True)
        check(proof["input_semantic_sha256"] == wrapper["REGISTERED_INPUTS"][case])
        mutations = []
        for key, changed in (("synthetic_only", False), ("case", "real"), ("symbols", ["BTCUSDT"]),
                             ("start_ms", False), ("end_ms", value["end_ms"] + api["BAR_MS"])):
            modified = copy.deepcopy(value); modified[key] = changed; mutations.append(modified)
        for key, changed in (("target", True), ("target", 0.0), ("target", 2), ("model_sha256", "not-a-sha"),
                             ("model_sha256", "A" * 64), ("signal_timestamp", value["start_ms"] + 1)):
            modified = copy.deepcopy(value); modified["targets"][0][key] = changed; mutations.append(modified)
        for key, changed in (("open", True), ("open", float("nan")), ("close", 0), ("timestamp", True)):
            modified = copy.deepcopy(value); modified["bars"][0][key] = changed; mutations.append(modified)
        modified = copy.deepcopy(value); modified["targets"].pop(); mutations.append(modified)
        modified = copy.deepcopy(value); modified["bars"][1] = copy.deepcopy(modified["bars"][0]); mutations.append(modified)
        modified = copy.deepcopy(value); modified["models"][0]["available_ms"] += 1; mutations.append(modified)
        modified = copy.deepcopy(value); modified["models"][0]["payload"]["origin"] = 9; mutations.append(modified)
        modified = copy.deepcopy(value); modified["models"][0]["sha256"] = "9" * 64; mutations.append(modified)
        modified = copy.deepcopy(value); modified["bars"][0]["open"] += 0.01; mutations.append(modified)
        for modified in mutations:
            expected_error = error(wrapper["validate_inputs"], api, contract, modified, "normal")
            observed_error = error(wrapper["run_wrapper"], api, contract, modified, "normal", fake)
            check(expected_error == observed_error)
            check(callback_events == [])
        for scenario in ("gross_zero_cost", "live", None):
            check(error(wrapper["validate_inputs"], api, contract, value, scenario) ==
                  error(wrapper["run_wrapper"], api, contract, value, scenario, fake))
        check(error(wrapper["run_wrapper"], api, contract, value, "normal", fake)[0] == "ValueError")
    check(not any(name.split(".")[0] in {"numpy", "pandas", "exia"} for name in sys.modules))
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_strict_synthetic_wrapper_v1.py"))
    _, artifact, environment, summary, _, verify = builder["prepare"]()
    verify()
    namespace = sys.modules["_gp_synthetic_accounting_probe_v1"]
    predecessor = json.loads((builder["PREVIOUS"] / "fixtures.json").read_bytes())
    oracle = {(x["case"], x["scenario"]): x["oracle_decision_entry_sha_derived_not_original"] for x in predecessor["results"]}
    for result in artifact["results"]:
        ledger = result["entry_identity_sidecar"]["decision_ledger"]
        expected_entries = oracle[result["case"], result["scenario"]]
        trades = result["actual"]["trades"]
        before_by_symbol = {symbol: None for symbol in adapter["SYMBOLS"]}
        for row, decision, expected in zip(ledger, result["actual"]["decisions"], expected_entries):
            check(row["decision_model_sha256"] == decision["model_sha256"])
            check(row["position_entry_model_sha256_after"] == expected)
            check(row["position_entry_model_sha256_before"] == before_by_symbol[row["symbol"]])
            before_by_symbol[row["symbol"]] = expected
            check(row["quantity_after"] == decision["quantity_after"])
            check(row["field_origin"] == "wrapper_derived_bound_to_actual_closed_trade")
            for suffix in ("before", "after"):
                index = row["actual_closed_trade_index_" + suffix]
                model = row["position_entry_model_sha256_" + suffix]
                if index is None:
                    check(model is None)
                else:
                    check(trades[index]["entry_model_sha256"] == model)
                    check(trades[index]["entry_timestamp"] <= row["execution_timestamp"] <= trades[index]["exit_timestamp"])
        check(result["entry_identity_sidecar"]["entry_sha_is_original_decision_field"] is False)
        check(result["entry_identity_sidecar"]["all_entries_bound_to_actual_closed_trades"] is True)
    check(summary["sidecar_decisions"] == 648)
    check(summary["actual_closed_trades_attributed"] == 12)
    check(summary["terminal_sidecar_records"] == 12)
    check(summary["original_accounting_outputs_identical_to_predecessor"] is True)
    check(summary["full_contract_integration_verified"] is summary["runtime_dependency_graph_verified"] is False)
    check(all(flag is False for flag in summary["safety"].values()))
    check(all(x == "socket.gethostname" for x in environment["blocked_dependency_import_events"]))
    value = adapter["fixture"](api, contract, "carry")
    reference = next(x for x in artifact["results"] if x["case"] == "carry" and x["scenario"] == "normal")
    def holder():
        result = type("CopiedActualResult", (), {})()
        result.decisions = namespace.pd.DataFrame(copy.deepcopy(reference["actual"]["decisions"]))
        result.trades = namespace.pd.DataFrame(copy.deepcopy(reference["actual"]["trades"]))
        return result
    for kind in ("trade_sha", "trade_quantity", "trade_price", "trade_reason", "missing_trade",
                 "duplicate_trade", "decision_sha", "forced_reset", "entry_sign", "missing_decision"):
        damaged = holder()
        if kind == "trade_sha": damaged.trades.loc[0, "entry_model_sha256"] = "9" * 64
        if kind == "trade_quantity": damaged.trades.loc[0, "quantity"] += 0.001
        if kind == "trade_price": damaged.trades.loc[0, "exit_price"] += 1.0
        if kind == "trade_reason": damaged.trades.loc[0, "exit_reason"] = "unregistered"
        if kind == "missing_trade": damaged.trades = damaged.trades.iloc[1:].reset_index(drop=True)
        if kind == "duplicate_trade": damaged.trades = namespace.pd.concat([damaged.trades, damaged.trades.iloc[[0]]], ignore_index=True)
        if kind == "decision_sha": damaged.decisions.loc[0, "model_sha256"] = "9" * 64
        if kind == "forced_reset": damaged.decisions.loc[0, "forced_reset"] = True
        if kind == "entry_sign": damaged.decisions.loc[0, "quantity_after"] *= -1
        if kind == "missing_decision": damaged.decisions = damaged.decisions.iloc[1:].reset_index(drop=True)
        check(error(wrapper["entry_sidecar"], api, value, damaged)[0] == "ValueError")
    original_function = namespace.simulate_cash_targets
    namespace.simulate_cash_targets = forbidden
    try:
        check(error(wrapper["run_wrapper"], api, contract, value, "normal", namespace)[0] == "ValueError")
        check(callback_events == [])
    finally:
        namespace.simulate_cash_targets = original_function
    original_enter = namespace.CashSleeve.enter
    namespace.CashSleeve.enter = forbidden
    try:
        check(error(wrapper["checked_callee"], api, namespace)[0] == "ValueError")
    finally:
        namespace.CashSleeve.enter = original_enter
    check(wrapper["checked_callee"](api, namespace)["source_function_codes_matched"] is True)
    verify()
    print(f"strict synthetic wrapper: {count} deterministic assertions passed; 6 fixed combinations")
    print("Invalid inputs rejected before callee; original outputs unchanged; no market/model training/runtime")


if __name__ == "__main__":
    main()
