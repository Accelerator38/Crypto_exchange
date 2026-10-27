"""Static audit of two disclosed result families. Does not import their producers."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
OUTPUT = ROOT / (BASE + "accounting_semantics_audit_v1_20261005")
INVENTORY = BASE + "development_review_v1_20261005/evidence_index_v2.json"
INVENTORY_SHA = "5e6192ccef065c9e0f488395bfd9456d160c071d706a9985447149e172d89bed"
CATALOG = BASE + "result_catalog_v1_20261005_r2/catalog.json"
CATALOG_SHA = "cafb0375dab4e0e608004c3fe9c216302ebc30331a745a0885d2c285ddbdb08b"
CONTEXT = BASE + "result_context_links_v1_20261005/context_links.json"
CONTEXT_SHA = "d2bec13b402512aa7d5ef08cef10eccd9789d7fb58909e52df1d855307f81c03"
REGISTRY = BASE + "observed_research_registry_v1_20261005/observed_registry.json"
REGISTRY_SHA = "69d90c05b66b75adc53f04215a5d8159d46b8646987c2b5bfa0e4dc105d7070e"
RECEIPT = BASE + "observed_research_registry_v1_20261005/completion_receipt.json"
RECEIPT_SHA = "c36c46868f714ff9ea1918426485e91c0e03372e1b407b7ebb83f7079eeb0ea4"
REVIEWED_CURRENT = {
    "src/exia/genetic_primus/accounting_v2.py": "f9294a2e4b2220aa7552ebed7d2480e7e7647744fe5a62330888acba569af5bb",
    "src/exia/genetic_primus/measurement_lab_v2.py": "ff9ec7f12412190426ebffe57d817fca9ac0d0643e628c15646e72ff84b3f648",
    "src/exia/genetic_primus/ablation_v1.py": "429082fda31e371e71e2b2420e613c0681b08357d2495985b4ba8a8b8e645312",
}


def prepare():
    module_path = ROOT / "src/exia/genetic_primus/accounting_semantics_audit_v1.py"
    module_sha = sha256(module_path.read_bytes()).hexdigest()
    builder_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    api = runpy.run_path(str(module_path))
    old_api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/result_catalog_v1.py"))
    inputs = {}

    def read(relative, expected):
        path = (ROOT / relative).resolve(strict=True)
        allowed = ((relative.startswith(BASE) and path.is_relative_to((ROOT / BASE).resolve()) and path.suffix == ".json")
                   or (relative.startswith("configs/genetic_primus_") and path.parent == (ROOT / "configs").resolve() and path.suffix == ".json")
                   or (relative.startswith("src/exia/genetic_primus/") and path.parent == (ROOT / "src/exia/genetic_primus").resolve() and path.suffix == ".py")
                   or (relative.startswith("tools/") and path.parent == (ROOT / "tools").resolve() and path.suffix == ".py"))
        api["require"](allowed, "input outside permitted metadata/static-source scope")
        cap = 40_000_000 if relative == INVENTORY else 8_000_000 if path.suffix == ".json" else 400_000
        api["require"](path.stat().st_size <= cap, "static input size cap exceeded")
        raw = path.read_bytes()
        api["require"](sha256(raw).hexdigest() == expected, "input fingerprint mismatch: " + relative)
        inputs[relative] = {"path": relative, "byte_sha256": expected, "size_bytes": len(raw)}
        return raw

    inventory = json.loads(read(INVENTORY, INVENTORY_SHA))["records"]
    original_by_path = {r["path"]: r for r in inventory}
    api["require"](len(inventory) == len(original_by_path) == 1784, "inventory coverage")
    catalog = json.loads(read(CATALOG, CATALOG_SHA))
    context = json.loads(read(CONTEXT, CONTEXT_SHA))
    registry = json.loads(read(REGISTRY, REGISTRY_SHA))
    receipt = json.loads(read(RECEIPT, RECEIPT_SHA))
    api["require"](receipt["output_byte_sha256"]["observed_registry.json"] == REGISTRY_SHA
                   and registry["context_links_byte_sha256"] == CONTEXT_SHA, "predecessor byte binding mismatch")
    envelopes = {e["source"]["path"]: e for e in catalog["records"]}
    api["require"](set(envelopes) == set(original_by_path), "catalog coverage")
    for envelope in envelopes.values():
        old_api["validate_serialized"](envelope)
    links = {(l["source_path"], l["policy"], l["scenario"]): l for f in context["families"] for l in f["evaluation_links"]}

    def original(path):
        return old_api["load_json_metadata"](read(path, original_by_path[path]["sha256"]))

    matrix_path = BASE + "rolling_linear_matrix_v1_20261002/result.json"
    baseline_path = BASE + "baseline_economics_audit_v1_20261002/result.json"
    matrix, baseline = original(matrix_path), original(baseline_path)
    contract_path = "configs/genetic_primus_prospective_3d_v1.json"
    contract_raw = read(contract_path, matrix["source_contract_sha256"])
    contract = old_api["load_json_metadata"](contract_raw)
    lab_path = "configs/genetic_primus_measurement_lab_v2.json"
    lab = old_api["load_json_metadata"](read(lab_path, contract["accounting"]["source_config_sha256"]))
    audit_config_path = "configs/genetic_primus_baseline_economics_audit_v1.json"
    audit_config = old_api["load_json_metadata"](read(audit_config_path, baseline["config_sha256"]))
    upstream_path = audit_config["source_ablation_result_path"]
    upstream = original(upstream_path)
    api["require"](original_by_path[upstream_path]["sha256"] == baseline["source_ablation_result_sha256"]
                   == audit_config["source_ablation_result_sha256"], "upstream audit-result binding mismatch")
    api["require"](upstream["contract_sha256"] == matrix["source_contract_sha256"], "selected contract declaration mismatch")
    code = {}
    for path, hashed in REVIEWED_CURRENT.items():
        code[path] = api["source"](path, read(path, hashed), hashed)
    matrix_runner_path = "tools/run_genetic_primus_rolling_linear_matrix_v1.py"
    baseline_runner_path = "tools/audit_genetic_primus_baseline_economics_v1.py"
    matrix_runner = api["source"](matrix_runner_path, read(matrix_runner_path, matrix["script_sha256"]),
                                   matrix["script_sha256"], historically_bound=True)
    baseline_runner = api["source"](baseline_runner_path, read(baseline_runner_path, baseline["script_sha256"]),
                                     baseline["script_sha256"], historically_bound=True)
    matrix_module_path = "src/exia/genetic_primus/rolling_linear_matrix_v1.py"
    read(matrix_module_path, matrix["matrix_module_sha256"])  # Byte-only; never import this model.
    current = api["current_simulator_profile"](code["src/exia/genetic_primus/accounting_v2.py"])
    current["daily_helper_observations"] = [
        api["current_daily_observation"](code["src/exia/genetic_primus/measurement_lab_v2.py"], "daily_returns"),
        api["current_daily_observation"](code["src/exia/genetic_primus/ablation_v1.py"], "summarize_simulation")]
    declarations = {"prospective_contract_accounting": contract["accounting"], "source_lab_accounting": lab["accounting"]}
    profiles = []
    for path, data, runner, function_name in ((matrix_path, matrix, matrix_runner, "_simulate"),
                                               (baseline_path, baseline, baseline_runner, "build")):
        profile = api["observation_profile"](path, original_by_path[path]["sha256"], contract_path,
                                              matrix["source_contract_sha256"], contract["accounting"],
                                              api["simulator_call"](runner, function_name), current)
        profile["declared_initial_equity_agrees_with_current_code"] = (
            contract["accounting"]["initial_equity"] == lab["accounting"]["initial_equity"]
            == current["observed_current_rules"]["initial_portfolio_equity"])
        profile["profile_sha256"] = api["digest"]({k: v for k, v in profile.items() if k != "profile_sha256"})
        profiles.append(profile)
    evaluations = []

    def audit_evaluation(path, policy, scenario, metrics, *, percent=False, economics=None):
        envelope = envelopes[path]
        base_eval = next(e for e in envelope["evaluations"] if e["policy"] == policy and e["scenario"] == scenario)
        link = links[path, policy, scenario]
        fields = {f["name"]: f["value"] for f in link["fields"]}
        duration = fields["interval_end_exclusive_ms"] - fields["interval_start_ms"]
        api["require"](duration > 0 and duration % 86_400_000 == 0, "selected interval not whole days")
        evaluations.append({"source_path": path, "source_byte_sha256": original_by_path[path]["sha256"],
                            "policy": policy, "scenario": scenario, "base_envelope_sha256": envelope["envelope_sha256"],
                            "base_metrics_semantic_sha256": api["digest"](base_eval["metrics"]),
                            "checks": api["stored_checks"](metrics, percent=percent, economics=economics,
                                                            expected_days=duration // 86_400_000),
                            "independent_validation_verified": False, "strict_historical_accounting_verified": False})

    for index, record in enumerate(matrix["records"]):
        policy = f"{record.get('feature_set','unknown')}:{record.get('variant','unknown')}:{index}"
        for scenario, metrics in sorted(record["evaluations"].items()):
            audit_evaluation(matrix_path, policy, scenario, metrics)
    for scenario, metrics in sorted(matrix["ema_training_reference"].items()):
        audit_evaluation(matrix_path, "EMA_training_reference", scenario, metrics)
    for policy, record in sorted(baseline["policies"].items()):
        for scenario, value in sorted(record["evaluations"].items()):
            audit_evaluation(baseline_path, policy, scenario, value["metrics"], percent=True, economics=value["economics"])
    statuses = Counter(check["status"] for e in evaluations for check in e["checks"])
    audit = {"schema": api["SCHEMA"], "base_catalog_byte_sha256": CATALOG_SHA, "context_links_byte_sha256": CONTEXT_SHA,
             "observed_registry_byte_sha256": REGISTRY_SHA, "current_simulator_profile": current,
             "contract_declarations": declarations, "producer_profiles": profiles, "evaluation_checks": evaluations,
             "source_contract_byte_binding": {"path": contract_path, "byte_sha256": matrix["source_contract_sha256"],
                                             "lab_config_path": lab_path, "lab_config_byte_sha256": contract["accounting"]["source_config_sha256"]},
             "new_strategy_evaluation_runs_started": 0, "new_model_fits": 0,
             "historical_dependency_graph_verified": False, "all_costs_execution_verified": False,
             "execution_allowed": False, "credentials_allowed": False, "orders_enabled": False,
             "paper": False, "live": False, "promotion_authority": False, "cross_source_comparison_authority": False}
    summary = {"schema": "exia.genetic_primus.accounting_semantics_audit_summary/1", "selected_sources": 2,
               "selected_evaluations": len(evaluations), "arithmetic_check_statuses": dict(sorted(statuses.items())),
               "producer_file_hashes_matched": 2, "model_module_file_hashes_matched": 1,
               "current_dependency_profiles_reviewed": 3, "strict_historical_accounting_verified_evaluations": 0,
               "potential_terminal_scope_mismatches": sum(p["findings"][0]["status"] == "potential_mismatch" for p in profiles),
               "historical_accounting_unknown_in_original_context_unchanged": sum(
                   e["context"]["accounting"] is None for record in catalog["records"] for e in record["evaluations"]),
               "new_strategy_evaluation_runs_started": 0, "new_model_fits": 0,
               "execution_allowed": False, "promotion_authority": False, "cross_source_comparison_authority": False}

    def verify():
        for previous in receipt["input_files"]:
            read(previous["path"], previous["byte_sha256"])
        for record in inventory:
            api["require"](len(read(record["path"], record["sha256"])) == record["size_bytes"], "original size mismatch")
        for row in list(inputs.values()):
            read(row["path"], row["byte_sha256"])
        api["require"](sha256(module_path.read_bytes()).hexdigest() == module_sha
                       and sha256(Path(__file__).read_bytes()).hexdigest() == builder_sha, "audit implementation changed")
    return api, audit, summary, inputs, verify, module_sha, builder_sha


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing audit snapshot must not be overwritten")
    api, audit, summary, inputs, verify, module_sha, builder_sha = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("accounting_audit.json", audit), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        completion = {"schema": "exia.genetic_primus.accounting_semantics_audit_completion/1", "input_files": list(inputs.values()),
                      "output_byte_sha256": hashes, "inputs_verified_unchanged_before_and_after": True,
                      "module_byte_sha256": module_sha, "builder_byte_sha256": builder_sha,
                      "execution_allowed": False, "credentials_allowed": False, "orders_enabled": False,
                      "paper": False, "live": False, "promotion_authority": False}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](completion) + b"\n")
    print(f"{'preflight' if dry else 'accounting audit'}: {len(audit['evaluation_checks'])} evaluations; arithmetic {summary['arithmetic_check_statuses']}")
    print("Historical accounting dependency identity unresolved; 1784 originals unchanged; no replay/training/runtime.")


if __name__ == "__main__":
    main()
