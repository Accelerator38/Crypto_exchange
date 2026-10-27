"""Bounded, stdlib-only metadata regression and exclusive artifact publication."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "Reports/Exia/Genetic_Primus/pre_fit_admission_v1_20261007"
DESTINATION = "Reports/Exia/Genetic_Primus/research_execution_contract_v1_20261007"
MODULE = "src/exia/genetic_primus/research_execution_contract_v1.py"
CONFIG = "configs/genetic_primus_research_execution_contract_v1.json"
PROPOSAL = "configs/genetic_primus_forward_research_proposal_v1.json"
ASSERTIONS = 0


def check(condition):
    global ASSERTIONS
    if not condition:
        raise AssertionError("regression assertion " + str(ASSERTIONS + 1))
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except (ValueError, TypeError):
        check(True)
    else:
        raise AssertionError("invalid input was accepted")


def path(relative):
    if type(relative) is not str or "\\" in relative or ":" in relative:
        raise ValueError("repository relative path required")
    candidate = (ROOT / relative).resolve()
    candidate.relative_to(ROOT)
    if candidate.suffix not in (".json", ".py", ".md"):
        raise ValueError("only source/metadata/doc reads permitted")
    return candidate


def raw(relative):
    item = path(relative)
    if item.stat().st_size > 512_000:
        raise ValueError("metadata size bound exceeded")
    return item.read_bytes()


def identity(relative):
    content = raw(relative)
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def regress(api, contract):
    assessment = api["assess"](contract)
    check(len(assessment["missing_registration_fields"]) == 21)
    check(assessment["registration_complete"] is False)
    check(assessment["fit_allowed"] is False)
    check(assessment["actual_data_or_execution_verified"] is False)
    for key in api["SAFETY"]:
        for bad in (True, 0, "false"):
            mutated = copy.deepcopy(contract)
            mutated["safety"][key] = bad
            rejected(lambda: api["assess"](mutated))
    for bad in (True, -1, float("nan"), float("inf"), 1.5, "1000"):
        mutated = copy.deepcopy(contract)
        mutated["timing"]["execution_delay_ms"] = bad
        rejected(lambda: api["assess"](mutated))
    for bad in (0, True, -1):
        mutated = copy.deepcopy(contract)
        mutated["timing"]["training_budget_ms"] = bad
        rejected(lambda: api["assess"](mutated))
    for section, key, bad in (("policies", "purge_bars", 0),
                              ("data", "manifest_sha256", "A" * 64),
                              ("costs", "normal_fee_bps_per_side", -1)):
        mutated = copy.deepcopy(contract)
        mutated[section][key] = bad
        rejected(lambda: api["assess"](mutated))
    for key, bad in (("proposal_semantic_sha256", "0" * 64),
                     ("historical_results_requalified", True), ("status", "live")):
        mutated = copy.deepcopy(contract)
        mutated[key] = bad
        rejected(lambda: api["assess"](mutated))
    mutated = copy.deepcopy(contract)
    mutated["unexpected"] = False
    rejected(lambda: api["assess"](mutated))
    for invalid in ('{"x":0,"x":1}', '{"x":NaN}', '{"x":Infinity}'):
        rejected(lambda: api["strict_json"](invalid))
    b = api["BAR_MS"]
    h = api["HORIZON_MS"]
    for decision, delay, expected_entry in ((0, 0, b), (2500, 1000, b),
                                             (b - 1000, 1000, 2 * b), (b - 1, 1, 2 * b)):
        w = api["execution_window"](0, decision, delay)
        check(w["planned_entry_open_ms"] == expected_entry)
        check(w["planned_entry_open_ms"] > w["ready_at_ms"])
        check(w["planned_exit_open_ms"] - w["planned_entry_open_ms"] == h)
        following = api["execution_window"](h, decision + h, delay)
        check(following["planned_entry_open_ms"] == w["planned_exit_open_ms"])
        check(w["actual_fill_verified"] is False)
    for args in ((0, True, 0), (1, 1, 0), (0, b, 0), (0, 0, b), (0, 0, -1),
                 (0, 0.5, 0), (2**53 - 1, 2**53 - 1, 0)):
        rejected(lambda: api["execution_window"](*args))
    w = api["execution_window"](0, 2500, 1000)
    entry, exit_open = w["planned_entry_open_ms"], w["planned_exit_open_ms"]
    training_asof = exit_open + 7 * b + 1000
    passed = api["completed_label_gate"](w, entry + 1000, exit_open + 1000, training_asof)
    check(passed["completed_and_purged"] is True)
    check(passed["fit_allowed"] is False)
    late = api["completed_label_gate"](w, entry + 1000, exit_open + 1001, training_asof)
    check(late["completed_and_purged"] is False)
    late_entry = api["completed_label_gate"](w, exit_open + 1001, exit_open + 1000, training_asof)
    check(late_entry["completed_and_purged"] is False)
    rejected(lambda: api["completed_label_gate"](w, entry - 1, exit_open, training_asof))
    rejected(lambda: api["completed_label_gate"](w, entry, exit_open - 1, training_asof))
    tampered = dict(w, planned_exit_open_ms=exit_open - b)
    rejected(lambda: api["completed_label_gate"](tampered, entry, exit_open, training_asof))
    registered = copy.deepcopy(contract)  # Dummy metadata control, NEVER published as a registration.
    registered["status"] = "registered_research_only"
    registered["timing"] = {"training_budget_ms": 5000, "execution_delay_ms": 1000,
                             "budget_evidence_sha256": api["sha"]("dummy-budget")}
    for key in registered["data"]:
        registered["data"][key] = 100 * b if key.endswith("_ms") else api["sha"]("dummy-data-" + key)
    for key in registered["costs"]:
        registered["costs"][key] = api["sha"]("dummy-cost-" + key) if key.endswith("sha256") else (2 if key.startswith("stress") else 1)
    ready = api["assess"](registered)
    check(ready["registration_complete"] is True)
    check(ready["fit_allowed"] is False)
    check(ready["real_money_ready"] is False)
    for bad in (True, float("nan"), float("inf"), -0.1):
        mutated = copy.deepcopy(registered)
        mutated["costs"]["stress_fee_bps_per_side"] = bad
        rejected(lambda: api["assess"](mutated))
    mutated = copy.deepcopy(registered)
    mutated["costs"]["stress_slippage_bps_per_side"] = 0
    rejected(lambda: api["assess"](mutated))
    origin = 100 * b
    planned = api["execution_window"](origin, origin + 2500, 1000)
    evidence = {
        "contract_sha256": ready["contract_sha256"],
        "training_prefix_manifest_sha256": registered["data"]["training_prefix_manifest_sha256"],
        "training_rows_sha256": api["sha"]("dummy-label-rows"),
        "opportunity_feature_rows_sha256": api["sha"]("dummy-opportunities"),
        "scaler_fit_rows_sha256": api["sha"]("dummy-opportunities"),
        "scaler_payload_sha256": api["sha"]("dummy-scaler"),
        "model_payload_sha256": api["sha"]("dummy-model"),
        "decision_model_sha256": api["sha"]("dummy-model"),
        "folds_sha256": api["sha"]("dummy-folds"),
        "resample_bank_sha256": api["sha"]("dummy-search-bank"),
        "audit_bank_sha256": api["sha"]("dummy-audit-bank"),
        "origin_start_ms": origin, "training_asof_ms": origin - b,
        "training_cutoff_ms": origin - 8 * b, "fit_started_at_ms": origin - b + 1000,
        "fit_completed_at_ms": origin - b + 2000, "model_available_at_ms": origin - b + 2500,
        "decision_asof_ms": origin + 2500, "submitted_at_ms": origin + 3000,
        "planned_entry_open_ms": planned["planned_entry_open_ms"],
        "planned_exit_open_ms": planned["planned_exit_open_ms"],
    }
    binding = api["origin_binding_gate"](registered, evidence)
    check(binding["declared_bindings_consistent"] is True)
    check(binding["fit_allowed"] is False)
    check(binding["actual_fit_or_market_data_verified"] is False)
    check(binding["audit_seed_independence_verified"] is False)
    for key, bad in (("contract_sha256", "0" * 64), ("training_prefix_manifest_sha256", "0" * 64),
                     ("scaler_fit_rows_sha256", evidence["training_rows_sha256"]),
                     ("decision_model_sha256", "0" * 64),
                     ("audit_bank_sha256", evidence["resample_bank_sha256"]),
                     ("training_cutoff_ms", origin - 7 * b),
                     ("fit_started_at_ms", evidence["fit_completed_at_ms"] + 1),
                     ("fit_completed_at_ms", evidence["fit_started_at_ms"] + 5001),
                     ("model_available_at_ms", origin + 2501),
                     ("submitted_at_ms", origin + 3501),
                     ("planned_entry_open_ms", origin), ("planned_exit_open_ms", origin + h)):
        mutated = dict(evidence)
        mutated[key] = bad
        rejected(lambda: api["origin_binding_gate"](registered, mutated))
    unregistered = dict(evidence, contract_sha256=assessment["contract_sha256"])
    denied = api["origin_binding_gate"](contract, unregistered)
    check(denied["declared_bindings_consistent"] is False)
    check(denied["fit_allowed"] is False)
    return assessment


def main():
    if sys.executable.lower() != str(ROOT / ".venv/Scripts/python.exe").lower():
        raise ValueError("only workspace venv interpreter allowed")
    if sys.argv[1:] not in ([], ["--verify"]):
        raise ValueError("only optional --verify accepted")
    api = runpy.run_path(str(path(MODULE)))
    previous = api["strict_json"](raw(PREVIOUS + "/completion_receipt.json"))
    if not 1 <= len(previous["inputs"]) <= 128:
        raise ValueError("bounded predecessor input bank required")
    inputs = dict(previous["inputs"])
    for relative, expected in inputs.items():
        if identity(relative) != expected:
            raise ValueError("predecessor input changed: " + relative)
    for name in ("admission.json", "summary.json"):
        relative = PREVIOUS + "/" + name
        if identity(relative)["sha256"] != previous["output_byte_sha256"][name]:
            raise ValueError("predecessor output changed")
    for relative in (MODULE, CONFIG, PROPOSAL, PREVIOUS + "/completion_receipt.json",
                     PREVIOUS + "/admission.json", PREVIOUS + "/summary.json",
                     "tools/check_genetic_primus_research_execution_contract_v1.py",
                     "docs/GENETIC_PRIMUS_RESEARCH_EXECUTION_CONTRACT_V1.md",
                     "configs/genetic_primus_release_candidate_v1.json",
                     "configs/genetic_primus_linear_latency_sensitivity_v1.json"):
        item = identity(relative)
        if relative in inputs and inputs[relative] != item:
            raise ValueError("input bank collision")
        inputs[relative] = item
    check(api["sha"](api["strict_json"](raw(PROPOSAL))) == api["PROPOSAL_SHA"])
    contract = api["strict_json"](raw(CONFIG))
    assessment = regress(api, contract)
    check(not any(name == "exia" or name.startswith("exia.") or name.split(".")[0] in
                  ("numpy", "pandas", "websockets", "pytest") for name in sys.modules))
    summary = {"schema": "exia.genetic_primus.research_execution_check/1",
               "evidence_class": "synthetic_timing_and_metadata_only_not_profit_or_fit_evidence",
               "direct_assertions": ASSERTIONS, "input_files": len(inputs),
               "missing_registration_fields": len(assessment["missing_registration_fields"]),
               "registration_complete": False, "fit_count": 0, "market_evaluation_count": 0,
               "market_data_read": False, "market_research_launch_ready": False,
               "real_money_ready": False, "safety": contract["safety"]}
    encoded = {"assessment.json": api["canonical"](assessment) + b"\n",
               "summary.json": api["canonical"](summary) + b"\n"}
    for relative, expected in inputs.items():
        if identity(relative) != expected:
            raise ValueError("input changed during checks")
    output_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()}
    receipt = {"schema": "exia.genetic_primus.research_execution_receipt/1",
               "receipt_kind": "registration_draft_and_synthetic_metadata_checks_only",
               "inputs": inputs, "inputs_verified_unchanged_before_and_after": True,
               "output_byte_sha256": output_hashes, "contract_semantic_sha256": assessment["contract_sha256"],
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": contract["safety"]}
    destination = ROOT / DESTINATION
    if sys.argv[1:] == ["--verify"]:
        saved = api["strict_json"](raw(DESTINATION + "/completion_receipt.json"))
        if api["canonical"](saved) != api["canonical"](receipt):
            raise ValueError("receipt differs from verified inputs/results")
        for name, expected in encoded.items():
            if raw(DESTINATION + "/" + name) != expected:
                raise ValueError("artifact output substitution")
        print("PASS: " + str(ASSERTIONS) + " direct assertions; receipt + 2 outputs verified")
    else:
        destination.mkdir(exist_ok=False)
        for name, value in encoded.items():
            with (destination / name).open("xb") as stream:
                stream.write(value)
        with (destination / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
        print("PASS: " + str(ASSERTIONS) + " assertions; " + str(len(inputs)) +
              " inputs unchanged; 21 fields unregistered; fit=0, market=0")


if __name__ == "__main__":
    main()
