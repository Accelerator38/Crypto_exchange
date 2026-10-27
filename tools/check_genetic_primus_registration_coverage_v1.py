"""One bounded local metadata audit; no pytest, market data, fit or network."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "Reports/Exia/Genetic_Primus/research_execution_contract_v1_20261007"
DESTINATION = "Reports/Exia/Genetic_Primus/registration_coverage_v1_20261007"
MODULE = "src/exia/genetic_primus/registration_coverage_v1.py"
API_MODULE = "src/exia/genetic_primus/research_execution_contract_v1.py"
ASSERTIONS = 0


def check(condition):
    global ASSERTIONS
    if not condition:
        raise AssertionError("coverage assertion " + str(ASSERTIONS + 1))
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("invalid metadata was accepted")


def path(relative):
    if type(relative) is not str or ":" in relative or "\\" in relative:
        raise ValueError("relative repo source/metadata path required")
    result = (ROOT / relative).resolve()
    result.relative_to(ROOT)
    if result.suffix not in (".py", ".json", ".md"):
        raise ValueError("raw data reads forbidden")
    return result


def raw(relative):
    item = path(relative)
    if item.stat().st_size > 512_000:
        raise ValueError("bounded source/metadata only")
    return item.read_bytes()


def identity(relative):
    content = raw(relative)
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def regress(module, api, materials, identities):
    result = module["coverage"](materials, identities, api)
    check(len(result["rows"]) == 21)
    check(len({r["field"] for r in result["rows"]}) == 21)
    check(sum(result["status_counts"].values()) == 21)
    check(result["registered_fields"] == 0)
    check(result["remaining_registration_fields"] == 21)
    check(result["source_count"] == 13)
    check(result["status_counts"]["missing_registration"] == 6)
    for row in result["rows"]:
        check(row["registration_value"] is None and row["auto_registration_allowed"] is False)
        check(bool(row["references"]) and all(ref["source_file_sha256"] == identities[ref["source_role"]]["sha256"]
                                             for ref in row["references"]))
    for budgets in ([0], [16, 24, 40], [1.25, 2.5]):
        value = module["legacy_cost_evidence"](budgets)
        check(value["source_values"] == budgets and value["auto_mapping_allowed"] is False
              and value["per_side_fee_identified"] is False and value["per_side_slippage_identified"] is False)
    for budgets in ([], [True], [float("nan")], [float("inf")], [-1], ["16"], [1] * 9, (16, 24)):
        rejected(lambda: module["legacy_cost_evidence"](budgets))
    for role, key, value in (("ticker_integrity", "source_authenticity_verified", True),
                             ("ticker_coverage", "funding_settlements_verified", True),
                             ("l2_coverage", "complete", True),
                             ("recorded_manifest", "evidence_kind", "observed"),
                             ("latency", "no_model_refits", False),
                             ("latency", "source_manifest_sha256", "0" * 64)):
        mutated = copy.deepcopy(materials)
        mutated[role][key] = value
        rejected(lambda: module["coverage"](mutated, identities, api))
    changed_identity = copy.deepcopy(identities)
    changed_identity["ohlcv_manifest"]["sha256"] = "0" * 64
    rejected(lambda: module["coverage"](materials, changed_identity, api))
    changed_identity["ohlcv_manifest"]["sha256"] = "A" * 64
    rejected(lambda: module["coverage"](materials, changed_identity, api))
    for key in ("account_fee_schedule", "instrument_rules"):
        mutated = copy.deepcopy(materials)
        mutated["release"]["evidence"][key] = "claimed_verified"
        rejected(lambda: module["coverage"](mutated, identities, api))
    mutated = copy.deepcopy(materials)
    mutated["research_contract"]["safety"]["training_allowed"] = True
    rejected(lambda: module["coverage"](mutated, identities, api))
    mutated = copy.deepcopy(materials)
    mutated["research_contract"]["timing"]["training_budget_ms"] = 1000
    rejected(lambda: module["coverage"](mutated, identities, api))
    rejected(lambda: module["pointer"](materials["release"], "/missing_evidence"))
    check(result["observations"]["recorded_admission_kind"] == "synthetic")
    check(result["observations"]["last_inspected_l2_complete_claim"] is False)
    check(result["observations"]["funding_settlements_verified_claim"] is False)
    check(result["observations"]["ticker_authenticity_verified_claim"] is False)
    check(result["fit_allowed"] is False and result["actual_provenance_verified"] is False)
    check(result["auto_registration_allowed"] is False and result["raw_market_data_read"] is False)
    check(all(value is False for value in result["safety"].values()))
    return result


def main():
    if sys.executable.lower() != str(ROOT / ".venv/Scripts/python.exe").lower():
        raise ValueError("only workspace venv interpreter allowed")
    if sys.argv[1:] not in ([], ["--verify"]):
        raise ValueError("only optional --verify accepted")
    api = runpy.run_path(str(path(API_MODULE)))
    module = runpy.run_path(str(path(MODULE)))
    previous = api["strict_json"](raw(PREVIOUS + "/completion_receipt.json"))
    if not 1 <= len(previous["inputs"]) <= 128:
        raise ValueError("bounded predecessor bank required")
    inputs = dict(previous["inputs"])
    for relative, expected in inputs.items():
        if identity(relative) != expected:
            raise ValueError("predecessor input changed: " + relative)
    for name in ("assessment.json", "summary.json"):
        if identity(PREVIOUS + "/" + name)["sha256"] != previous["output_byte_sha256"][name]:
            raise ValueError("predecessor output changed")
    identities = {}
    materials = {}
    for role, relative in module["SOURCES"].items():
        identities[role] = identity(relative)
        content = raw(relative)
        materials[role] = content.decode("utf-8") if role.endswith("_source") else api["strict_json"](content)
    additional = list(module["SOURCES"].values()) + [MODULE, API_MODULE,
                  "tools/check_genetic_primus_registration_coverage_v1.py",
                  "docs/GENETIC_PRIMUS_REGISTRATION_COVERAGE_V1.md",
                  PREVIOUS + "/completion_receipt.json", PREVIOUS + "/assessment.json", PREVIOUS + "/summary.json"]
    for relative in additional:
        record = identity(relative)
        if relative in inputs and inputs[relative] != record:
            raise ValueError("input identity collision")
        inputs[relative] = record
    result = regress(module, api, materials, identities)
    check(not any(name == "exia" or name.startswith("exia.") or name.split(".")[0] in
                  ("numpy", "pandas", "websockets", "pytest") for name in sys.modules))
    summary = {"schema": "exia.genetic_primus.registration_coverage_check/1",
               "evidence_class": "metadata_coverage_not_actual_market_data_verification",
               "direct_assertions": ASSERTIONS, "input_files": len(inputs), "curated_sources": 13,
               "registered_fields": 0, "remaining_registration_fields": 21,
               "missing_registration": 6, "partial_or_incompatible_evidence": 15,
               "status_counts": result["status_counts"], "fit_count": 0, "market_evaluation_count": 0,
               "raw_market_data_read": False, "network_used": False,
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": result["safety"]}
    encoded = {"coverage.json": api["canonical"](result) + b"\n",
               "summary.json": api["canonical"](summary) + b"\n"}
    for relative, expected in inputs.items():
        if identity(relative) != expected:
            raise ValueError("input changed during checks")
    receipt = {"schema": "exia.genetic_primus.registration_coverage_receipt/1",
               "receipt_kind": "curated_metadata_and_code_anchor_audit_not_data_or_fit_admission",
               "inputs": inputs, "inputs_verified_unchanged_before_and_after": True,
               "output_byte_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()},
               "contract_semantic_sha256": result["contract_sha256"],
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": result["safety"]}
    if sys.argv[1:] == ["--verify"]:
        saved = api["strict_json"](raw(DESTINATION + "/completion_receipt.json"))
        if api["canonical"](saved) != api["canonical"](receipt):
            raise ValueError("receipt identity mismatch")
        for name, expected in encoded.items():
            if raw(DESTINATION + "/" + name) != expected:
                raise ValueError("artifact output mismatch")
        print("PASS: " + str(ASSERTIONS) + " assertions; receipt + 2 outputs verified")
    else:
        destination = ROOT / DESTINATION
        destination.mkdir(exist_ok=False)
        for name, value in encoded.items():
            with (destination / name).open("xb") as stream:
                stream.write(value)
        with (destination / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
        print("PASS: " + str(ASSERTIONS) + " assertions; " + str(len(inputs)) +
              " inputs unchanged; registered=0/21; fit=0, market=0")


if __name__ == "__main__":
    main()
