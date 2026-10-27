"""Bounded metadata/fail-closed regressions, followed by the same toy fixtures."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def main():
    api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"))
    helper_path = ROOT / "src/exia/genetic_primus/observed_dependency_readiness_v1.py"
    helper = runpy.run_path(str(helper_path))
    count = 0
    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"dependency readiness assertion {count + 1}")
        count += 1
    def rejects(fn, *args):
        try:
            fn(*args)
        except (ValueError, TypeError):
            check(True)
        else:
            check(False)
    roots = [helper_path.parent]
    regular = ModuleType("synthetic_metadata_only"); regular.__file__ = str(helper_path)
    frozen = ModuleType("synthetic_frozen"); frozen.__spec__ = SimpleNamespace(origin="frozen")
    missing = ModuleType("synthetic_missing"); missing.__file__ = str(helper_path.parent / "absent_dependency_fixture.py")
    nofile = ModuleType("synthetic_nofile")
    sample = helper["snapshot"](api, {"regular": regular, "frozen": frozen, "missing": missing, "nofile": nofile, "placeholder": None}, roots, [])
    check(helper["validate_fingerprint"](api, sample, roots) is True)
    helper["verify_files"](sample)
    check(sample["counts"]["unique_files"] == 1)
    check(sample["counts"]["builtin_frozen_entries"] == 1)
    check(sample["counts"]["unresolved_module_entries"] == 3)
    check(sample["modules"][0]["loaded_memory_or_bytecode_identity_verified"] is False)
    for path, kind in (("x.py", "python_source"), ("x.pyc", "python_cache"), ("x.pyd", "extension_binary"),
                       ("x.dll", "native_candidate"), ("x.exe", "interpreter_binary"), ("x.env", "unsupported")):
        check(helper["file_kind"](path) == kind)
    rejects(helper["snapshot"], api, {str(i): None for i in range(helper["MAX_MODULES"] + 1)}, roots, [])
    rejects(helper["hash_file"], helper_path.parent)
    damages = []
    for key in ("full_transitive_graph_verified", "actual_loaded_code_or_memory_identity_verified", "native_dll_closure_verified", "future_dynamic_imports_verified"):
        value = copy.deepcopy(sample); value[key] = True; damages.append(value)
    value = copy.deepcopy(sample); value["counts"]["unique_files"] += 1; damages.append(value)
    value = copy.deepcopy(sample); value["files"][0]["bytes"] = True; damages.append(value)
    value = copy.deepcopy(sample); value["files"][0]["byte_sha256"] = "BAD"; damages.append(value)
    value = copy.deepcopy(sample); value["modules"][0]["loaded_memory_or_bytecode_identity_verified"] = True; damages.append(value)
    value = copy.deepcopy(sample); value["files"].append(copy.deepcopy(value["files"][0])); damages.append(value)
    value = copy.deepcopy(sample); value["scoped_unresolved_modules"] = []; damages.append(value)
    value = copy.deepcopy(sample); value["file_bank_semantic_sha256"] = "0" * 64; damages.append(value)
    value = copy.deepcopy(sample); value["allowed_roots"] = [str(ROOT)]; damages.append(value)
    value = copy.deepcopy(sample); value["unresolved_categories"] = []; damages.append(value)
    for damaged in damages:
        rejects(helper["validate_fingerprint"], api, damaged, roots)
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_dependency_readiness_v1.py"))
    _, fingerprint, ready, environment, summary, _, verify = builder["prepare"]()
    verify()
    check(helper["validate_fingerprint"](api, fingerprint, fingerprint["allowed_roots"]) is True)
    check(fingerprint["counts"]["file_kinds"].get("extension_binary", 0) > 0)
    check(summary["synthetic_outputs_preserved"] == 6)
    check(summary["file_bytes"] <= helper["MAX_TOTAL_BYTES"])
    check(summary["scoped_engineering_gates_verified"] == 4)
    check(summary["unclosed_gates"] == 9)
    check(sum(x["status"] == "verified_scoped" for x in ready["gates"]) == 4)
    check(ready["market_research_launch_ready"] is ready["real_money_ready"] is False)
    check(all(x is False for x in ready["safety"].values()))
    check(ready["new_model_fits"] == ready["new_market_strategy_evaluations"] == 0)
    check(ready["no_automatic_sealed_interval_reuse"] is True)
    check(environment["runtime_dependency_graph_verified"] is False)
    previous = builder["strict_json"]((builder["PREVIOUS"] / "summary.json").read_bytes())
    evidence = {"wrapper": "synthetic_test_reference", "fingerprint": "synthetic_test_reference"}
    for key in ("model_payload_binding_verified_for_synthetic_inputs", "entry_sha_sidecar_verified_for_closed_synthetic_intervals", "original_accounting_outputs_identical_to_predecessor"):
        changed = copy.deepcopy(previous); changed[key] = False
        rejects(helper["readiness"], api, fingerprint, changed, evidence)
    changed = copy.deepcopy(previous); changed["safety"]["orders_enabled"] = True
    rejects(helper["readiness"], api, fingerprint, changed, evidence)
    changed = copy.deepcopy(previous); changed["runtime_dependency_graph_verified"] = True
    rejects(helper["readiness"], api, fingerprint, changed, evidence)
    verify()
    print(f"dependency readiness: {count} deterministic assertions passed; same 6 artificial outputs")
    print("Disk-file evidence only; native/memory gaps retained; no new market/model training/runtime")


if __name__ == "__main__":
    main()
