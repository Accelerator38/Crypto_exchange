"""Exclusive publication of artificial accounting checks, never model evaluation."""
from __future__ import annotations

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
OUTPUT = ROOT / (BASE + "synthetic_accounting_probe_v1_20261006")
PREVIOUS = ROOT / (BASE + "boundary_stub_adapter_v1_20261005")
PREVIOUS_RECEIPT_SHA = "32870a2832ce05708fcde16e48e3132c902e1d455417bd6c445d53204c101968"
MODULE = ROOT / "src/exia/genetic_primus/synthetic_accounting_probe_v1.py"


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def prepare():
    bank = {}
    def read(path, expected=None, external=False):
        path = Path(path).resolve()
        name = str(path) if external else path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        if len(raw) > 8_000_000 or (expected is not None and sha256(raw).hexdigest() != expected):
            raise ValueError("unreviewed/oversized input")
        bank[name] = {"path": name, "external_package_file": external,
                      "byte_sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw
    receipt = strict_json(read(PREVIOUS / "completion_receipt.json", PREVIOUS_RECEIPT_SHA))
    if receipt["receipt_kind"] != "synthetic_adapter_only_not_financial_launch":
        raise ValueError("unexpected predecessor")
    for item in receipt["inputs"]:
        raw = read(ROOT / item["path"], item["byte_sha256"])
        if len(raw) != item["bytes"]:
            raise ValueError("predecessor input size changed")
    previous_builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_stub_adapter_v1.py"))
    api, prev_artifact, prev_sources, prev_summary, prev_inputs, prev_verify = previous_builder["prepare"]()
    prev_verify()
    if api["canonical"](receipt["inputs"]) != api["canonical"](sorted(prev_inputs.values(), key=lambda x: x["path"])):
        raise ValueError("predecessor input bank changed")
    for name, expected in (("fixtures.json", prev_artifact), ("source_bank.json", prev_sources), ("summary.json", prev_summary)):
        raw = read(PREVIOUS / name, receipt["output_byte_sha256"][name])
        if api["canonical"](strict_json(raw)) != api["canonical"](expected):
            raise ValueError("predecessor output semantics changed")
    read(MODULE); read(Path(__file__))
    probe = runpy.run_path(str(MODULE))
    probe["checked_source"](ROOT)
    adapter = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_stub_adapter_v1.py"))
    contract = api["expected_contract"]()
    module, environment, denied = probe["load_verified_accounting"](ROOT, OUTPUT)
    for dep in environment["direct_packages_observed"]:
        read(Path(dep["package_init_path_observed"]), dep["package_init_file_byte_sha256"], external=True)
    read(Path(sys.executable), external=True)
    results = []
    for case in adapter["CASES"]:
        value = adapter["fixture"](api, contract, case)
        adapter["validate_fixture"](api, contract, value)
        for scenario, cost in (("normal", 16.0), ("stress", 24.0)):
            result = probe["run_artificial_case"](module, value, cost, api["BAR_MS"])
            result["scenario"] = scenario
            result["input_semantic_sha256"] = api["digest"](value)
            results.append(result)
    minimum = adapter["fixture"](api, contract, "abstain")
    minimum["end_ms"] = minimum["start_ms"] + 3 * api["DAY_MS"]
    minimum["symbols"] = ["SYNTH_A"]
    minimum["bars"] = [x for x in minimum["bars"] if x["symbol"] == "SYNTH_A" and x["timestamp"] < minimum["end_ms"]]
    minimum["targets"] = [x for x in minimum["targets"] if x["symbol"] == "SYNTH_A" and x["execution_timestamp"] < minimum["end_ms"]]
    sanity = probe["run_artificial_case"](module, minimum, 16.0, api["BAR_MS"])
    value = adapter["fixture"](api, contract, "carry")
    reset = value["start_ms"] + 3 * api["DAY_MS"]
    control = probe["run_artificial_case"](module, value, 16.0, api["BAR_MS"], (reset,))
    forced_exits = [x for x in control["actual"]["trades"] if x["exit_reason"] == "retrain_forced_exit"]
    if len(forced_exits) != 2:
        raise AssertionError("missing forced-open negative-control exits")
    comparisons = []
    for trade in forced_exits:
        symbol = trade["symbol"]
        opening = next(x["open"] for x in value["bars"] if x["timestamp"] == reset and x["symbol"] == symbol)
        prior_close = next(x["close"] for x in value["bars"] if x["timestamp"] == reset - api["BAR_MS"] and x["symbol"] == symbol)
        if trade["exit_price"] != opening or opening == prior_close:
            raise AssertionError("open/previous-close negative control failed")
        comparisons.append({"symbol": symbol, "actual_reset_exit_price": trade["exit_price"],
                            "origin_open": opening, "previous_close": prior_close,
                            "same_timestamp_different_price": True})
    if any(event != "socket.gethostname" for event in denied):
        raise PermissionError("unexpected blocked action in accounting probe")
    artifact = {"schema": "exia.genetic_primus.synthetic_accounting_probe/1", "results": results,
                "minimal_18_bar_notrade": sanity,
                "forced_open_reset_negative_control": control,
                "open_vs_previous_close": comparisons,
                "negative_control_is_not_registered_deployment_policy": True,
                "contract_semantic_sha256": api["digest"](contract),
                "synthetic_only": True, "financial_strategy_result": False,
                "per_decision_entry_sha_missing_in_actual_output": True,
                "safety": dict(api["SAFETY"])}
    all_results = results + [sanity, control]
    summary = {"schema": "exia.genetic_primus.synthetic_accounting_probe_summary/1",
               "registered_fixture_combinations": 6, "minimum_sanity_cases": 1,
               "forced_open_negative_controls": 1, "actual_synthetic_simulator_calls": 8,
               "oracle_comparison_checks": sum(x["checks"]["comparison_checks"] for x in all_results),
               "maximum_absolute_error": max(x["checks"]["maximum_absolute_error"] for x in all_results),
               "tolerance": probe["TOLERANCE"], "all_consistent": True,
               "current_accounting_source_identity_verified": True,
               "current_source_synthetic_behavior_verified": True,
               "full_contract_integration_verified": False,
               "per_decision_entry_sha_directly_available": False,
               "runtime_dependency_graph_verified": False, "historical_semantics_repaired": False,
               "existing_runners_modified": False, "original_inventory_files_verified": 1784,
               "new_model_fits": 0, "new_market_strategy_evaluations": 0,
               "blocked_import_metadata_events": list(denied), "safety": dict(api["SAFETY"])}
    def verify():
        prev_verify()
        for item in bank.values():
            path = Path(item["path"]) if item["external_package_file"] else ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
                raise ValueError("probe input identity changed")
    return api, artifact, environment, summary, bank, verify


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing probe snapshot must not be overwritten")
    api, artifact, environment, summary, bank, verify = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("fixtures.json", artifact), ("environment.json", environment), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        receipt = {"schema": "exia.genetic_primus.synthetic_accounting_probe_completion/1",
                   "receipt_kind": "synthetic_accounting_only_not_market_strategy_launch",
                   "previous_receipt_byte_sha256": PREVIOUS_RECEIPT_SHA,
                   "inputs": sorted(bank.values(), key=lambda x: x["path"]),
                   "output_byte_sha256": hashes, "fixture_semantic_sha256": api["digest"](artifact),
                   "environment_semantic_sha256": api["digest"](environment),
                   "inputs_verified_unchanged_before_and_after": True,
                   "runtime_dependency_graph_verified": False, "market_strategy_launch_completed": False,
                   "safety": dict(api["SAFETY"])}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
        verify()
    print(f"{'preflight' if dry else 'published'}: 8 bounded artificial accounting cases; {summary['oracle_comparison_checks']} oracle checks")
    print(f"max error {summary['maximum_absolute_error']:.3g}; 1784 originals unchanged; no market data/model fits/runtime")


if __name__ == "__main__":
    main()
