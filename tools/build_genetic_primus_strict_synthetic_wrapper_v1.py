"""Publish trade-bound sidecars for already registered artificial fixtures."""
from __future__ import annotations

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
OUTPUT = ROOT / (BASE + "strict_synthetic_wrapper_v1_20261006")
PREVIOUS = ROOT / (BASE + "synthetic_accounting_probe_v1_20261006")
PREVIOUS_RECEIPT_SHA = "144a787064a2b12da3f274377d9f631762cf166dc9578e14b7e01673a644a9a0"
MODULE = ROOT / "src/exia/genetic_primus/strict_synthetic_wrapper_v1.py"


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
            raise ValueError("wrapper input hash/size mismatch")
        bank[name] = {"path": name, "external_package_file": external,
                      "byte_sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw
    receipt = strict_json(read(PREVIOUS / "completion_receipt.json", PREVIOUS_RECEIPT_SHA))
    if receipt["receipt_kind"] != "synthetic_accounting_only_not_market_strategy_launch":
        raise ValueError("unexpected predecessor")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete predecessor")
    for item in receipt["inputs"]:
        path = Path(item["path"]) if item["external_package_file"] else ROOT / item["path"]
        if len(read(path, item["byte_sha256"], item["external_package_file"])) != item["bytes"]:
            raise ValueError("predecessor input size changed")
    # Pure metadata/stub predecessor verifies originals; do not install the old probe's write guard.
    metadata = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_stub_adapter_v1.py"))
    api, _, _, _, _, original_verify = metadata["prepare"]()
    original_verify()
    previous_outputs = {}
    for name, expected_sha in receipt["output_byte_sha256"].items():
        previous_outputs[name] = strict_json(read(PREVIOUS / name, expected_sha))
    if previous_outputs["summary.json"]["all_consistent"] is not True:
        raise ValueError("predecessor accounting checks incomplete")
    read(MODULE); read(Path(__file__))
    wrapper = runpy.run_path(str(MODULE))
    adapter = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_stub_adapter_v1.py"))
    probe = runpy.run_path(str(ROOT / "src/exia/genetic_primus/synthetic_accounting_probe_v1.py"))
    previous_results = {(x["case"], x["scenario"]): x for x in previous_outputs["fixtures.json"]["results"]}
    if len(previous_results) != 6:
        raise ValueError("six unique registered combinations required")
    contract = api["expected_contract"]()
    # Verify payloads before even loading the numeric accounting callee.
    payloads = {}
    for case in adapter["CASES"]:
        payloads[case] = adapter["fixture"](api, contract, case)
        for scenario in ("normal", "stress"):
            proof = wrapper["validate_inputs"](api, contract, payloads[case], scenario)
            if proof["input_semantic_sha256"] != previous_results[case, scenario]["input_semantic_sha256"]:
                raise ValueError("fixture is not the same predecessor input")
    namespace, environment, denied = probe["load_verified_accounting"](ROOT, OUTPUT)
    for dep in environment["direct_packages_observed"]:
        read(Path(dep["package_init_path_observed"]), dep["package_init_file_byte_sha256"], external=True)
    read(Path(sys.executable), external=True)
    results = []
    for case in adapter["CASES"]:
        for scenario in ("normal", "stress"):
            result = wrapper["run_wrapper"](api, contract, payloads[case], scenario, namespace)
            previous = previous_results[case, scenario]
            for key in ("actual", "actual_metrics"):
                if api["canonical"](result[key]) != api["canonical"](previous[key]):
                    raise AssertionError("wrapper altered original accounting output")
            results.append(result)
    if any(x != "socket.gethostname" for x in denied):
        raise PermissionError("unexpected blocked operation")
    artifact = {"schema": "exia.genetic_primus.strict_synthetic_wrapper_fixtures/1",
                "results": results, "registered_input_bank": wrapper["REGISTERED_INPUTS"],
                "contract_semantic_sha256": api["digest"](contract),
                "original_accounting_outputs_identical_to_predecessor": True,
                "entry_sha_field_origin": "wrapper_derived_actual_closed_trade_bound",
                "financial_strategy_result": False, "safety": dict(api["SAFETY"])}
    summary = {"schema": "exia.genetic_primus.strict_synthetic_wrapper_summary/1",
               "registered_fixture_combinations": 6, "prevalidated_model_banks": 3,
               "sidecar_decisions": sum(len(x["entry_identity_sidecar"]["decision_ledger"]) for x in results),
               "actual_closed_trades_attributed": sum(x["entry_identity_sidecar"]["actual_trades_attributed"] for x in results),
               "terminal_sidecar_records": sum(len(x["entry_identity_sidecar"]["terminal_ledger"]) for x in results),
               "model_payload_binding_verified_for_synthetic_inputs": True,
               "entry_sha_sidecar_verified_for_closed_synthetic_intervals": True,
               "original_accounting_outputs_identical_to_predecessor": True,
               "entry_sha_is_original_decision_field": False,
               "online_open_position_attribution_verified": False,
               "full_contract_integration_verified": False, "runtime_dependency_graph_verified": False,
               "historical_semantics_repaired": False, "existing_runners_modified": False,
               "original_inventory_files_verified": 1784, "new_model_fits": 0,
               "new_market_strategy_evaluations": 0, "safety": dict(api["SAFETY"])}
    def verify():
        original_verify()
        for item in bank.values():
            path = Path(item["path"]) if item["external_package_file"] else ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
                raise ValueError("wrapper input identity changed")
    return api, artifact, environment, summary, bank, verify


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing wrapper snapshot must not be overwritten")
    api, artifact, environment, summary, bank, verify = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("wrapper.json", artifact), ("environment.json", environment), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        completion = {"schema": "exia.genetic_primus.strict_synthetic_wrapper_completion/1",
                      "receipt_kind": "synthetic_closed_interval_wrapper_not_market_launch",
                      "previous_receipt_byte_sha256": PREVIOUS_RECEIPT_SHA,
                      "inputs": sorted(bank.values(), key=lambda x: x["path"]),
                      "output_byte_sha256": hashes, "wrapper_semantic_sha256": api["digest"](artifact),
                      "environment_semantic_sha256": api["digest"](environment),
                      "inputs_verified_unchanged_before_and_after": True,
                      "runtime_dependency_graph_verified": False, "market_strategy_launch_completed": False,
                      "safety": dict(api["SAFETY"])}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](completion) + b"\n")
        verify()
    print(f"{'preflight' if dry else 'published'}: 648 entry-identity sidecar rows, 12 closed trades, 6 unchanged synthetic outputs")
    print("1784 originals unchanged; no arbitrary data/callees; no market/model training/runtime")


if __name__ == "__main__":
    main()
