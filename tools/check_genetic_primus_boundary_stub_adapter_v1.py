"""Bounded deterministic protocol/identity regressions; no financial evaluator."""
from __future__ import annotations

import copy
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    modules_before = set(sys.modules)
    api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"))
    adapter = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_stub_adapter_v1.py"))
    contract = api["expected_contract"]()
    count = 0
    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"check {count + 1}")
        count += 1
    def rejects(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except (ValueError, TypeError, OverflowError):
            check(True)
        else:
            check(False)
    for case in adapter["CASES"]:
        value = adapter["fixture"](api, contract, case)
        before = api["digest"](value)
        check(len(value["bars"]) == len(value["targets"]) == 108)
        check(all(api["digest"](x["payload"]) == x["sha256"] for x in value["models"]))
        for scenario, fee in (("normal", 16.0), ("stress", 24.0)):
            stub = adapter["RecordingStub"](api)
            result = adapter["invoke_fixture"](api, contract, value, scenario, stub)
            capture = result["captured_call"]
            expected_parameters = {"symbols": list(adapter["SYMBOLS"]),
                                   "start_timestamp": value["start_ms"], "end_timestamp": value["end_ms"],
                                   "timeframe_ms": api["BAR_MS"], "round_trip_cost_bps": fee,
                                   "force_flat_timestamps": [], "record_details": True}
            check(capture["parameters"] == expected_parameters)
            check(capture["bars_semantic_sha256"] == api["digest"](value["bars"]))
            check(capture["targets_semantic_sha256"] == api["digest"](value["targets"]))
            check(capture["force_flat_argument_was_tuple"] is True)
            check(capture["symbols_argument_was_tuple"] is True)
            check(capture["financial_result"] is capture["fills_generated"] is capture["orders_emitted"] is False)
            check(result["stub_call_count"] == len(stub.calls) == 1)
            check(result["input_unchanged"] is True and api["digest"](value) == before)
            check(result["real_evaluator_adapter_verified"] is result["runtime_dependency_graph_verified"] is False)
            check(all(flag is False for flag in result["safety"].values()))
            # Separate elementary oracle: do not call the spec's transition function.
            directions = {symbol: 0 for symbol in adapter["SYMBOLS"]}
            entry = {symbol: None for symbol in adapter["SYMBOLS"]}
            for source, row in zip(value["targets"], result["decision_ledger"]):
                symbol = source["symbol"]
                previous, desired = directions[symbol], source["target"]
                events = (["hold"] if previous else ["abstain"]) if previous == desired else (
                    (["exit"] if previous else []) + (["enter"] if desired else []))
                if previous != desired:
                    entry[symbol] = source["model_sha256"] if desired else None
                directions[symbol] = desired
                check(row["semantic_events"] == events)
                check(row["decision_model_sha256"] == source["model_sha256"])
                check(row["position_entry_model_sha256"] == entry[symbol])
                check(row["position_after"] == desired)
                check(row["execution_timestamp"] == row["signal_timestamp"] == source["execution_timestamp"])
                check(row["capital_reset"] is row["quantity_resized_for_retrain"] is row["orders_emitted"] is False)
            check(len(result["decision_ledger"]) == len(value["targets"]))
            for terminal in result["terminal_expectations"]:
                symbol = terminal["symbol"]
                expected_price = next(x["close"] for x in value["bars"]
                                      if x["symbol"] == symbol and x["timestamp"] == value["end_ms"] - api["BAR_MS"])
                check(terminal["timestamp"] == value["end_ms"])
                check(terminal["expected_price"] == expected_price)
                check(terminal["position_before"] == directions[symbol])
                check(terminal["position_after_expected"] == 0)
                check(terminal["position_entry_model_sha256"] == entry[symbol])
                check(terminal["actual_fill_verified"] is False)
                check(terminal["expected_semantic_events"] == (["exit"] if directions[symbol] else []))
                check(terminal["fees_required_if_position"] is bool(directions[symbol]))
            rejects(adapter["invoke_fixture"], api, contract, value, scenario, stub)
            check(len(stub.calls) == 1)
            second = adapter["invoke_fixture"](api, contract, value, scenario, adapter["RecordingStub"](api))
            check(api["canonical"](result) == api["canonical"](second))
            capture["parameters"]["record_details"] = False
            check(stub.calls[0]["parameters"]["record_details"] is True)
        for origin in (1, 2):
            timestamp = value["start_ms"] + origin * 3 * api["DAY_MS"]
            previous_close = next(x["close"] for x in value["bars"]
                                  if x["symbol"] == "SYNTH_A" and x["timestamp"] == timestamp - api["BAR_MS"])
            new_open = next(x["open"] for x in value["bars"] if x["symbol"] == "SYNTH_A" and x["timestamp"] == timestamp)
            check(previous_close != new_open)
        if case == "carry":
            carry = adapter["invoke_fixture"](api, contract, value, "normal", adapter["RecordingStub"](api))
            later = [row for row in carry["decision_ledger"] if row["execution_timestamp"] >= value["start_ms"] + 3 * api["DAY_MS"]]
            check(all(row["semantic_events"] == ["hold"] for row in later))
            check(all(row["decision_model_sha256"] != row["position_entry_model_sha256"] for row in later))
    value = adapter["fixture"](api, contract, "carry")
    mutations = []
    for field, changed in (("synthetic_only", False), ("case", "real_market"),
                           ("start_ms", value["start_ms"] + 1), ("end_ms", value["end_ms"] + api["BAR_MS"]),
                           ("symbols", ["BTCUSDT"])):
        modified = copy.deepcopy(value); modified[field] = changed; mutations.append(modified)
    modified = copy.deepcopy(value); modified["bars"].pop(); mutations.append(modified)
    modified = copy.deepcopy(value); modified["targets"].append(copy.deepcopy(modified["targets"][0])); mutations.append(modified)
    for field, changed in (("open", float("nan")), ("close", 0), ("timestamp", value["end_ms"])):
        modified = copy.deepcopy(value); modified["bars"][0][field] = changed; mutations.append(modified)
    for field, changed in (("target", True), ("target", 2), ("model_sha256", "9" * 64),
                           ("signal_timestamp", value["start_ms"] + 1), ("symbol", "BTCUSDT")):
        modified = copy.deepcopy(value); modified["targets"][0][field] = changed; mutations.append(modified)
    modified = copy.deepcopy(value); modified["models"][0]["available_ms"] += 1; mutations.append(modified)
    modified = copy.deepcopy(value); modified["models"][0]["payload"]["origin"] = 7; mutations.append(modified)
    for modified in mutations:
        stub = adapter["RecordingStub"](api)
        rejects(adapter["invoke_fixture"], api, contract, modified, "normal", stub)
        check(stub.calls == [])
    for scenario in ("gross_zero_cost", "exchange_live", None):
        stub = adapter["RecordingStub"](api)
        rejects(adapter["invoke_fixture"], api, contract, value, scenario, stub)
        check(stub.calls == [])
    invoked = []
    def forbidden_callee(**kwargs):
        invoked.append(True)
        raise AssertionError("callee must never execute")
    rejects(adapter["invoke_fixture"], api, contract, value, "normal", forbidden_callee)
    check(invoked == [])
    class ForbiddenSubclass(adapter["RecordingStub"]):
        def __call__(self, **kwargs):
            return forbidden_callee(**kwargs)
    rejects(adapter["invoke_fixture"], api, contract, value, "normal", ForbiddenSubclass(api))
    check(invoked == [])
    rejects(adapter["invoke_fixture"], api, contract, value, "normal", adapter["RecordingStub"](dict(api)))
    changed_contract = copy.deepcopy(contract); changed_contract["safety"]["execution_allowed"] = True
    stub = adapter["RecordingStub"](api)
    rejects(adapter["invoke_fixture"], api, changed_contract, value, "normal", stub)
    check(stub.calls == [])
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_stub_adapter_v1.py"))
    _, artifact, sources, summary, _, verify = builder["prepare"]()
    verify()
    check(len(artifact["results"]) == 6)
    check(summary["synthetic_decision_records"] == 648)
    check(summary["terminal_expectations_not_actual_fills"] == 12)
    check(summary["call_parameter_names_matched"] == 9)
    check(summary["new_model_fits"] == summary["new_financial_evaluations"] == 0)
    check(sources["full_transitive_graph_verified"] is sources["actual_loaded_runtime_graph_verified"] is False)
    check(sum(x["role"] == "reviewed_reference_not_imported" for x in sources["sources"]) == 3)
    check(all(flag is False for flag in summary["safety"].values()))
    added_modules = set(sys.modules) - modules_before
    check(not any(name == "exia" or name.startswith("exia.") for name in added_modules))
    check(not any(name.split(".")[0] in {"numpy", "pandas", "websockets", "requests"} for name in added_modules))
    print(f"boundary stub adapter: {count} deterministic checks passed; 6 fixed cases, no financial evaluator")


if __name__ == "__main__":
    main()
