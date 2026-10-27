"""Verify a stored process-specific snapshot without equating fresh sys.modules."""
from __future__ import annotations

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/dependency_readiness_v1_20261006"


def main():
    receipt = json.loads((OUTPUT / "completion_receipt.json").read_bytes())
    if receipt["receipt_kind"] != "observed_files_and_engineering_readiness_not_market_launch":
        raise ValueError("unexpected receipt kind")
    builder_path = ROOT / "tools/build_genetic_primus_dependency_readiness_v1.py"
    module_path = ROOT / "src/exia/genetic_primus/observed_dependency_readiness_v1.py"
    bank = {x["path"]: x for x in receipt["inputs"]}
    if len(bank) != len(receipt["inputs"]):
        raise ValueError("duplicate receipt input")
    for path in (builder_path, module_path):
        if sha256(path.read_bytes()).hexdigest() != bank[str(path.resolve())]["byte_sha256"]:
            raise ValueError("fingerprint code changed")
    builder = runpy.run_path(str(builder_path))
    api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"))
    helper = runpy.run_path(str(module_path))
    values = {}
    for name, expected_sha in receipt["output_byte_sha256"].items():
        raw = (OUTPUT / name).read_bytes()
        if sha256(raw).hexdigest() != expected_sha:
            raise ValueError("readiness artifact bytes changed")
        values[name] = builder["strict_json"](raw)
    fingerprint, ready = values["dependency_fingerprint.json"], values["readiness.json"]
    roots = [Path(sys.prefix), Path(sys.base_prefix), ROOT / "tools", ROOT / "src/exia/genetic_primus"]
    helper["validate_fingerprint"](api, fingerprint, roots)
    helper["verify_files"](fingerprint)
    previous_raw = (builder["PREVIOUS"] / "completion_receipt.json").read_bytes()
    if sha256(previous_raw).hexdigest() != builder["PREVIOUS_RECEIPT_SHA"]:
        raise ValueError("readiness predecessor changed")
    previous = builder["strict_json"](previous_raw)
    summary_raw = (builder["PREVIOUS"] / "summary.json").read_bytes()
    if sha256(summary_raw).hexdigest() != previous["output_byte_sha256"]["summary.json"]:
        raise ValueError("predecessor summary changed")
    evidence = {"wrapper": {"path": str(builder["PREVIOUS"] / "wrapper.json"), "byte_sha256": previous["output_byte_sha256"]["wrapper.json"]},
                "fingerprint": {"path": str(OUTPUT / "dependency_fingerprint.json"), "semantic_sha256": api["digest"](fingerprint)}}
    expected_ready = helper["readiness"](api, fingerprint, builder["strict_json"](summary_raw), evidence)
    if api["canonical"](expected_ready) != api["canonical"](ready):
        raise ValueError("readiness derivation mismatch")
    expected_summary = {"schema": "exia.genetic_primus.dependency_readiness_summary/1",
                        **fingerprint["counts"], "synthetic_outputs_preserved": 6,
                        "scoped_engineering_gates_verified": 4, "unclosed_gates": 9,
                        "full_runtime_dependency_graph_verified": False, "market_research_launch_ready": False,
                        "real_money_ready": False, "original_inventory_files_verified": 1784,
                        "new_model_fits": 0, "new_market_strategy_evaluations": 0, "safety": dict(api["SAFETY"])}
    if api["canonical"](values["summary.json"]) != api["canonical"](expected_summary):
        raise ValueError("stored summary inconsistent with fingerprint/readiness")
    for item in receipt["inputs"]:
        actual = helper["hash_file"](item["path"])
        if actual["bytes"] != item["bytes"] or actual["byte_sha256"] != item["byte_sha256"]:
            raise ValueError("receipt input changed")
    if receipt["fingerprint_semantic_sha256"] != api["digest"](fingerprint) or receipt["readiness_semantic_sha256"] != api["digest"](ready):
        raise ValueError("readiness semantic digest mismatch")
    for flag in ("full_runtime_dependency_graph_verified", "market_research_launch_ready", "real_money_ready"):
        if receipt[flag] is not False or values["summary.json"][flag] is not False:
            raise ValueError("unearned readiness")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True or receipt["previous_receipt_byte_sha256"] != builder["PREVIOUS_RECEIPT_SHA"]:
        raise ValueError("incomplete provenance")
    if api["canonical"](receipt["safety"]) != api["canonical"](api["SAFETY"]):
        raise ValueError("readiness safety drift")
    # Only pure metadata/stub reconstruction: do not rerun the numeric environment.
    metadata = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_stub_adapter_v1.py"))
    _, _, _, _, _, original_verify = metadata["prepare"]()
    original_verify()
    print("stored dependency snapshot/receipt/readiness verified; 1784 originals unchanged")
    print("No fresh-process equivalence claim or numeric replay; market/real-money readiness remains false")


if __name__ == "__main__":
    main()
