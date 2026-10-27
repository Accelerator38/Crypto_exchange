"""Freeze one research record; metadata-only, exclusive outputs, no fit/data execution."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "Reports/Exia/Genetic_Primus/registration_coverage_v1_20261007"
DESTINATION = "Reports/Exia/Genetic_Primus/forward_preregistration_v1_20261007"
MODULE = "src/exia/genetic_primus/forward_preregistration_v1.py"
API_MODULE = "src/exia/genetic_primus/research_execution_contract_v1.py"
ASSERTIONS = 0


def check(condition):
    global ASSERTIONS
    if not condition:
        raise AssertionError("preregistration assertion " + str(ASSERTIONS + 1))
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("invalid record accepted")


def path(relative):
    if type(relative) is not str or ":" in relative or "\\" in relative:
        raise ValueError("relative source/metadata path required")
    result = (ROOT / relative).resolve()
    result.relative_to(ROOT)
    if result.suffix not in (".py", ".json", ".md"):
        raise ValueError("market/raw data reads forbidden")
    return result


def raw(relative):
    item = path(relative)
    if item.stat().st_size > 512_000:
        raise ValueError("source/metadata size limit exceeded")
    return item.read_bytes()


def identity(relative):
    value = raw(relative)
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def regress(module, api, materials, identities):
    record = module["build"](materials, identities, api)
    verification = module["validate"](record, materials, identities, api)
    check(verification["record_consistent"] is True)
    check(verification["fit_allowed"] is False)
    body = record["record"]
    plan = body["engineering_plan"]
    check(plan["origin_start_utc"] == "2026-09-22T00:00:00Z")
    check(plan["nominal_deployment_end_exclusive_utc"] == "2026-09-25T00:00:00Z")
    check(plan["latest_required_exit_open_utc"] == "2026-09-25T08:00:00Z")
    check(plan["latest_required_exit_open_ms"] < plan["coverage_end_exclusive_ms"])
    check(plan["selection_uses_outcomes"] is False and plan["actual_availability_verified"] is False)
    check(plan["training_cutoff_ms"] == plan["training_asof_ms"] - 7 * api["BAR_MS"])
    check(plan["training_asof_ms"] == plan["origin_start_ms"] - api["BAR_MS"])
    check(plan["maximum_planned_engineering_pipeline_runs"] == 2)
    check(plan["authorized_market_runs"] == 0)
    check(len(body["unresolved"]) == 30 and all(row["value"] is None for row in body["unresolved"]))
    check(body["definitions"]["model"]["coefficient_count"] == 3)
    check(body["definitions"]["model"]["ga_enabled"] is False)
    check(body["definitions"]["recency"]["weight_all_outcomes_including_losses"] is True)
    check(body["definitions"]["variants"] == ["uniform", "half_life_90d_anchor_20pct"])
    check(body["definitions"]["economic_floor_bps_per_day"] == 0)
    check(body["definitions"]["decision_policy"]["registered_executable_threshold_rule"] is None)
    check(body["definitions"]["accounting_policies"]["normal_and_stress_cost_parameters"] is None)
    check(body["release_bridge"]["ohlcv_fill_fallback_for_release_allowed"] is False)
    check(body["historical_plan_is_not_prospective_preregistration"] is True)
    for key in api["SAFETY"]:
        check(body["safety"][key] is False)
    for path_keys, value in (
        (("gate", "fit_allowed"), True), (("gate", "market_data_read_authorized"), True),
        (("safety", "training_allowed"), True), (("safety", "network_allowed"), True),
        (("testing_budget", "authorized_market_runs"), 1),
        (("testing_budget", "hyperparameter_search_trials"), 1),
        (("testing_budget", "engineering_results_cannot_select_best_model_or_prove_edge"), False),
        (("definitions", "model", "coefficient_count"), 659),
        (("definitions", "scaler", "fit_scope"), "all_label_rows_including_deploy"),
        (("engineering_plan", "origin_start_ms"), plan["origin_start_ms"] + api["HORIZON_MS"]),
        (("release_bridge", "ohlcv_fill_fallback_for_release_allowed"), True),
        (("sealed", "peek_allowed"), True),
    ):
        mutated = copy.deepcopy(record)
        cursor = mutated["record"]
        for key in path_keys[:-1]:
            cursor = cursor[key]
        cursor[path_keys[-1]] = value
        # Valid self-hash is not permission to change bound declarations.
        mutated["record_semantic_sha256"] = api["sha"](mutated["record"])
        rejected(lambda: module["validate"](mutated, materials, identities, api))
    mutated = copy.deepcopy(record)
    mutated["record"]["unresolved"][0]["value"] = 0
    mutated["record_semantic_sha256"] = api["sha"](mutated["record"])
    rejected(lambda: module["validate"](mutated, materials, identities, api))
    mutated = copy.deepcopy(record)
    mutated["record_semantic_sha256"] = "0" * 64
    rejected(lambda: module["validate"](mutated, materials, identities, api))
    changed_identity = copy.deepcopy(identities)
    changed_identity["manifest"]["sha256"] = "0" * 64
    rejected(lambda: module["build"](materials, changed_identity, api))
    changed_material = copy.deepcopy(materials)
    changed_material["coverage"]["rows"][0]["registration_value"] = 0
    rejected(lambda: module["build"](changed_material, identities, api))
    changed_material = copy.deepcopy(materials)
    changed_material["manifest"]["metadata"]["timeframe_ms"] = 60_000
    rejected(lambda: module["build"](changed_material, identities, api))
    changed_material = copy.deepcopy(materials)
    changed_material["old_schedule"]["outer"]["retrain_cadence_days"] = 7
    rejected(lambda: module["build"](changed_material, identities, api))
    start = module["utc_ms"]("2022-01-01T00:00:00Z")
    anchor = module["utc_ms"]("2026-09-01T00:00:00Z")
    exact_end = module["utc_ms"]("2026-09-25T08:00:00Z")
    control = module["select_engineering_origin"](start, exact_end, anchor, api)
    check(control["origin_start_utc"] == "2026-09-19T00:00:00Z")
    control = module["select_engineering_origin"](start, exact_end + api["BAR_MS"], anchor, api)
    check(control["origin_start_utc"] == "2026-09-22T00:00:00Z")
    for args in ((True, exact_end, anchor), (start + 1, exact_end, anchor),
                 (start, start, anchor), (start, anchor, anchor),
                 (anchor, anchor + 4 * api["HORIZON_MS"], anchor), (start, exact_end, start - api["BAR_MS"])):
        rejected(lambda: module["select_engineering_origin"](*args, api))
    rejected(lambda: module["utc_ms"]("2026-09-01T03:00:00+03:00"))
    # Latest nominal old origin is invalid with the delayed label envelope.
    last_nominal = module["utc_ms"]("2026-09-25T00:00:00Z")
    check(last_nominal + api["HORIZON_MS"] + 2 * api["BAR_MS"] >= plan["coverage_end_exclusive_ms"])
    return record, verification


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
            raise ValueError("predecessor source changed: " + relative)
    for name in ("coverage.json", "summary.json"):
        if identity(PREVIOUS + "/" + name)["sha256"] != previous["output_byte_sha256"][name]:
            raise ValueError("predecessor output changed")
    materials, identities = {}, {}
    for role, relative in module["SOURCES"].items():
        identities[role] = identity(relative)
        content = raw(relative)
        materials[role] = content.decode("utf-8") if role == "linear_source" else api["strict_json"](content)
    for relative in list(module["SOURCES"].values()) + [MODULE, API_MODULE,
                     "tools/check_genetic_primus_forward_preregistration_v1.py",
                     "docs/GENETIC_PRIMUS_FORWARD_PREREGISTRATION_V1.md",
                     PREVIOUS + "/completion_receipt.json", PREVIOUS + "/summary.json"]:
        value = identity(relative)
        if relative in inputs and inputs[relative] != value:
            raise ValueError("input bank collision")
        inputs[relative] = value
    record, verification = regress(module, api, materials, identities)
    check(not any(name == "exia" or name.startswith("exia.") or name.split(".")[0] in
                  ("numpy", "pandas", "websockets", "pytest") for name in sys.modules))
    summary = {"schema": "exia.genetic_primus.forward_preregistration_check/1",
               "direct_assertions": ASSERTIONS, "input_files": len(inputs), "bound_source_roles": len(materials),
               "evidence_class": "frozen_definitions_and_proposed_engineering_calendar_not_model_results",
               "record_semantic_sha256": record["record_semantic_sha256"],
               "unresolved_registration_fields": 21, "additional_protocol_requirements": 9,
               "planned_engineering_origins": 1, "planned_candidate_pipelines": 2,
               "authorized_market_runs": 0, "fit_count": 0, "market_evaluation_count": 0,
               "raw_market_data_read": False, "network_used": False,
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": record["record"]["safety"]}
    encoded = {"record.json": api["canonical"](record) + b"\n",
               "verification.json": api["canonical"](verification) + b"\n",
               "summary.json": api["canonical"](summary) + b"\n"}
    for relative, expected in inputs.items():
        if identity(relative) != expected:
            raise ValueError("input changed during checks")
    receipt = {"schema": "exia.genetic_primus.forward_preregistration_receipt/1",
               "receipt_kind": "definitions_freeze_not_actual_data_fit_or_trading_qualification",
               "inputs": inputs, "inputs_verified_unchanged_before_and_after": True,
               "output_byte_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in encoded.items()},
               "record_semantic_sha256": record["record_semantic_sha256"],
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": summary["safety"]}
    if sys.argv[1:] == ["--verify"]:
        saved = api["strict_json"](raw(DESTINATION + "/completion_receipt.json"))
        if api["canonical"](saved) != api["canonical"](receipt):
            raise ValueError("receipt differs from bound sources")
        for name, expected in encoded.items():
            if raw(DESTINATION + "/" + name) != expected:
                raise ValueError("artifact differs from frozen record")
        print("PASS: " + str(ASSERTIONS) + " assertions; receipt + 3 outputs verified")
    else:
        destination = ROOT / DESTINATION
        destination.mkdir(exist_ok=False)
        for name, value in encoded.items():
            with (destination / name).open("xb") as stream:
                stream.write(value)
        with (destination / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
        print("PASS: " + str(ASSERTIONS) + " assertions; " + str(len(inputs)) +
              " inputs unchanged; proposed Sep22-25; authorized fits=0")


if __name__ == "__main__":
    main()
