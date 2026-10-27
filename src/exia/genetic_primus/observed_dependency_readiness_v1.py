"""Bounded loaded-module file observations, not a complete runtime attestation."""
from __future__ import annotations

import stat
import re
from hashlib import sha256
from pathlib import Path
from types import ModuleType

MAX_MODULES = 1600
MAX_FILES = 1200
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 384 * 1024 * 1024
UNRESOLVED_CATEGORIES = ["native_dependency_dlls_not_enumerated", "builtin_frozen_memory_not_attested",
                         "pyc_selection_and_loaded_bytecode_not_attested", "extension_memory_not_attested",
                         "ephemeral_runpy_namespaces_not_sys_modules", "future_dynamic_imports_not_observed"]


def file_kind(path):
    suffix = Path(path).suffix.lower()
    return {".py": "python_source", ".pyc": "python_cache", ".pyd": "extension_binary",
            ".so": "extension_binary", ".dll": "native_candidate", ".exe": "interpreter_binary"}.get(suffix, "unsupported")


def under(path, roots):
    return any(path == root or root in path.parents for root in roots)


def hash_file(path):
    path = Path(path).resolve()
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
        raise ValueError("nonregular/oversized dependency file")
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(512 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("dependency changed while hashing")
    return {"path": str(path), "bytes": after.st_size, "byte_sha256": digest.hexdigest(), "kind": file_kind(path)}


def snapshot(api, modules, allowed_roots, explicit_files):
    if len(modules) > MAX_MODULES:
        raise ValueError("loaded module cap exceeded")
    roots = tuple(Path(x).resolve() for x in allowed_roots)
    files, observations, unresolved = {}, [], []
    total = 0
    def add(path):
        nonlocal total
        path = Path(path).resolve()
        if not under(path, roots) or file_kind(path) == "unsupported":
            return None
        key = str(path)
        if key not in files:
            if len(files) >= MAX_FILES:
                raise ValueError("dependency file cap exceeded")
            size = path.stat().st_size
            if total + size > MAX_TOTAL_BYTES:
                raise ValueError("dependency total byte cap exceeded")
            item = hash_file(path)
            files[key] = item
            total += item["bytes"]
        return key
    for name, module in sorted(modules.items()):
        row = {"module": name, "disk_file": None, "cache_file": None,
               "loaded_memory_or_bytecode_identity_verified": False}
        if not isinstance(module, ModuleType):
            row["status"] = "nonmodule_or_placeholder"
            unresolved.append({"module": name, "reason": row["status"]})
            observations.append(row)
            continue
        values = vars(module)
        spec = values.get("__spec__")
        origin = vars(spec).get("origin") if spec is not None and hasattr(spec, "__dict__") else None
        if origin in ("built-in", "frozen"):
            row["status"] = "builtin_or_frozen_not_file_hashed"
            row["origin_category"] = origin
            observations.append(row)
            continue
        path = values.get("__file__")
        if not isinstance(path, str):
            row["status"] = "no_file_origin"
        else:
            candidate = Path(path).resolve()
            if not candidate.is_file():
                row["status"] = "missing_file_origin"
            elif not under(candidate, roots) or file_kind(candidate) == "unsupported":
                row["status"] = "out_of_scope_file_origin"
            else:
                row["disk_file"] = add(candidate)
                row["status"] = "observed_module_file_hashed"
                cached = values.get("__cached__")
                if isinstance(cached, str) and Path(cached).is_file():
                    row["cache_file"] = add(cached)
                row["cache_actually_used_verified"] = False
        if row["status"] != "observed_module_file_hashed":
            unresolved.append({"module": name, "reason": row["status"]})
        observations.append(row)
    candidates = []
    for role, path in explicit_files:
        path = Path(path).resolve()
        key = add(path) if path.is_file() else None
        candidates.append({"role": role, "disk_file": key, "loaded_memory_identity_verified": False})
    counts = {}
    for item in files.values():
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    result = {
        "schema": "exia.genetic_primus.observed_dependency_file_fingerprint/1",
        "modules": observations, "files": sorted(files.values(), key=lambda x: x["path"]),
        "allowed_roots": sorted(set(str(x) for x in roots)),
        "explicit_interpreter_candidates": candidates,
        "scoped_unresolved_modules": unresolved,
        "counts": {"module_entries": len(observations), "file_hashed_module_entries": sum(x["status"] == "observed_module_file_hashed" for x in observations),
                   "builtin_frozen_entries": sum(x["status"] == "builtin_or_frozen_not_file_hashed" for x in observations),
                   "unresolved_module_entries": len(unresolved), "unique_files": len(files),
                   "file_bytes": total, "file_kinds": counts},
        "limits": {"modules": MAX_MODULES, "files": MAX_FILES, "per_file_bytes": MAX_FILE_BYTES, "total_bytes": MAX_TOTAL_BYTES},
        "source_of_observation": "sys_modules_after_six_registered_artificial_wrapper_calls",
        "ephemeral_runpy_code_bound_separately_in_receipt": True,
        "full_transitive_graph_verified": False, "actual_loaded_code_or_memory_identity_verified": False,
        "native_dll_closure_verified": False, "future_dynamic_imports_verified": False,
        "unresolved_categories": list(UNRESOLVED_CATEGORIES),
    }
    result["file_bank_semantic_sha256"] = api["digest"](result["files"])
    result["module_observation_semantic_sha256"] = api["digest"](observations)
    return result


def validate_fingerprint(api, fingerprint, allowed_roots):
    roots = tuple(Path(x).resolve() for x in allowed_roots)
    if fingerprint["schema"] != "exia.genetic_primus.observed_dependency_file_fingerprint/1":
        raise ValueError("unknown fingerprint schema")
    if fingerprint["allowed_roots"] != sorted(set(str(x) for x in roots)):
        raise ValueError("fingerprint root scope changed")
    files, modules = fingerprint["files"], fingerprint["modules"]
    if not 0 < len(files) <= MAX_FILES or not 0 < len(modules) <= MAX_MODULES:
        raise ValueError("invalid fingerprint coverage size")
    if files != sorted(files, key=lambda x: x["path"]) or len({x["path"] for x in files}) != len(files):
        raise ValueError("duplicate/unordered fingerprint files")
    paths, total, kinds = set(), 0, {}
    for item in files:
        path = Path(item["path"]).resolve()
        if str(path) != item["path"] or not under(path, roots) or file_kind(path) != item["kind"] or item["kind"] == "unsupported":
            raise ValueError("out-of-scope/invalid file identity")
        if type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_FILE_BYTES:
            raise ValueError("invalid dependency byte count")
        if not isinstance(item["byte_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", item["byte_sha256"]) is None:
            raise ValueError("invalid dependency SHA")
        paths.add(item["path"]); total += item["bytes"]
        kinds[item["kind"]] = kinds.get(item["kind"], 0) + 1
    if total > MAX_TOTAL_BYTES or len({x["module"] for x in modules}) != len(modules):
        raise ValueError("excessive bytes/duplicate modules")
    statuses = {"nonmodule_or_placeholder", "builtin_or_frozen_not_file_hashed", "no_file_origin",
                "missing_file_origin", "out_of_scope_file_origin", "observed_module_file_hashed"}
    for row in modules:
        if row["status"] not in statuses or row["loaded_memory_or_bytecode_identity_verified"] is not False:
            raise ValueError("invalid/unearned module status")
        for key in ("disk_file", "cache_file"):
            if row[key] is not None and row[key] not in paths:
                raise ValueError("unbound module file reference")
        if row["status"] == "observed_module_file_hashed" and row["disk_file"] is None:
            raise ValueError("missing observed module file")
        if "cache_actually_used_verified" in row and row["cache_actually_used_verified"] is not False:
            raise ValueError("unearned cache-use claim")
    for item in fingerprint["explicit_interpreter_candidates"]:
        if item["loaded_memory_identity_verified"] is not False or (item["disk_file"] is not None and item["disk_file"] not in paths):
            raise ValueError("unbound/unearned interpreter claim")
    expected_limits = {"modules": MAX_MODULES, "files": MAX_FILES, "per_file_bytes": MAX_FILE_BYTES, "total_bytes": MAX_TOTAL_BYTES}
    if api["canonical"](fingerprint["limits"]) != api["canonical"](expected_limits) or fingerprint["unresolved_categories"] != UNRESOLVED_CATEGORIES:
        raise ValueError("fingerprint limits/gap declaration changed")
    unresolved = [{"module": row["module"], "reason": row["status"]} for row in modules
                  if row["status"] not in {"observed_module_file_hashed", "builtin_or_frozen_not_file_hashed"}]
    counts = {"module_entries": len(modules), "file_hashed_module_entries": sum(x["status"] == "observed_module_file_hashed" for x in modules),
              "builtin_frozen_entries": sum(x["status"] == "builtin_or_frozen_not_file_hashed" for x in modules),
              "unresolved_module_entries": len(unresolved), "unique_files": len(files), "file_bytes": total, "file_kinds": kinds}
    if api["canonical"](counts) != api["canonical"](fingerprint["counts"]) or unresolved != fingerprint["scoped_unresolved_modules"]:
        raise ValueError("fingerprint counters/unknowns inconsistent")
    if api["digest"](files) != fingerprint["file_bank_semantic_sha256"] or api["digest"](modules) != fingerprint["module_observation_semantic_sha256"]:
        raise ValueError("fingerprint semantic digest mismatch")
    for key in ("full_transitive_graph_verified", "actual_loaded_code_or_memory_identity_verified", "native_dll_closure_verified", "future_dynamic_imports_verified"):
        if fingerprint[key] is not False:
            raise ValueError("unearned full-runtime claim")
    return True


def verify_files(fingerprint):
    for expected in fingerprint["files"]:
        if hash_file(expected["path"]) != expected:
            raise ValueError("observed dependency file identity changed")


def readiness(api, fingerprint, predecessor_summary, evidence):
    if predecessor_summary["sidecar_decisions"] != 648 or predecessor_summary["actual_closed_trades_attributed"] != 12:
        raise ValueError("unexpected synthetic wrapper coverage")
    if predecessor_summary["original_accounting_outputs_identical_to_predecessor"] is not True:
        raise ValueError("synthetic accounting output drift")
    for key in ("model_payload_binding_verified_for_synthetic_inputs", "entry_sha_sidecar_verified_for_closed_synthetic_intervals"):
        if predecessor_summary[key] is not True:
            raise ValueError("missing scoped predecessor proof")
    if api["canonical"](predecessor_summary["safety"]) != api["canonical"](api["SAFETY"]):
        raise ValueError("predecessor safety drift")
    for key in ("full_contract_integration_verified", "runtime_dependency_graph_verified"):
        if predecessor_summary[key] is not False:
            raise ValueError("unearned predecessor integration claim")
    validate_fingerprint(api, fingerprint, fingerprint["allowed_roots"])
    gates = [
        {"gate": "synthetic_input_model_payload_binding", "status": "verified_scoped", "scope": "three_registered_artificial_model_banks", "evidence": evidence["wrapper"]},
        {"gate": "synthetic_closed_interval_entry_sidecar", "status": "verified_scoped", "scope": "648_rows_12_closed_trades_not_online", "evidence": evidence["wrapper"]},
        {"gate": "synthetic_accounting_output_preservation", "status": "verified_scoped", "scope": "six_previously_registered_combinations", "evidence": evidence["wrapper"]},
        {"gate": "observed_dependency_disk_file_identity", "status": "verified_scoped", "scope": "bounded_observed_files_not_loaded_memory", "evidence": evidence["fingerprint"]},
        {"gate": "full_loaded_runtime_native_closure", "status": "unverified", "scope": "native_dlls_bytecode_selection_generated_globals_future_imports"},
        {"gate": "historical_accounting_dependency_identity", "status": "unverified", "scope": "new_checks_do_not_attest_old_runs"},
        {"gate": "full_forward_market_contract_integration", "status": "unverified", "scope": "engineering_contract_does_not_supersede_frozen_contracts"},
        {"gate": "real_data_labels_scalers_folds_resamples_cutoff", "status": "not_evaluated", "scope": "no_market_data_loaded_this_stage"},
        {"gate": "causal_training_latency_online_open_positions", "status": "not_evaluated", "scope": "artificial_closed_intervals_only"},
        {"gate": "registered_new_objective_testing_budget", "status": "not_authorized", "scope": "no_new_search_or_training"},
        {"gate": "statistical_net_after_stress_no_loss", "status": "not_evaluated", "scope": "toy_pnl_not_model_profit_evidence"},
        {"gate": "realistic_fills_funding_borrow_intrabar_risk", "status": "not_evaluated", "scope": "offline_cash_ledger_only"},
        {"gate": "paper_shadow_fault_injection_canary", "status": "not_authorized", "scope": "no_runtime_or_network_authority"},
    ]
    return {"schema": "exia.genetic_primus.engineering_readiness_record/1", "gates": gates,
            "scoped_engineering_gates_verified": 4, "unclosed_gates": 9,
            "observed_dependency_files": fingerprint["counts"]["unique_files"],
            "full_contract_integration_verified": False, "runtime_dependency_graph_verified": False,
            "market_research_launch_ready": False, "real_money_ready": False,
            "new_model_fits": 0, "new_market_strategy_evaluations": 0,
            "safety": dict(api["SAFETY"]),
            "next_action": "separate_forward_research_contract_proposal_and_data_label_scaler_audit_before_any_new_training",
            "no_automatic_sealed_interval_reuse": True}
