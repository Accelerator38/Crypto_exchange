"""Short deterministic metadata regressions; no pytest, model training or runtime."""

from copy import deepcopy
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    api = runpy.run_path(str(root / "src/exia/genetic_primus/research_identity_registry_v1.py"))
    checks = 0

    def check(condition):
        nonlocal checks
        assert condition
        checks += 1

    def rejects(call):
        try:
            call()
        except ValueError:
            return True
        return False

    for invalid in (True, False, -1, 0.5, "0", float("nan"), float("inf"), 10 ** 1000):
        check(api["count_claim"](invalid) is None)
    check(api["count_claim"](0) == 0)
    check(api["count_claim"](4.0) == 4)
    for invalid in (None, False, [], ""):
        check(api["objective_id"](invalid) is None)
    check(api["objective_id"]("EMA ") == "EMA ")
    check(api["objective_id"]("ema") != api["objective_id"]("EMA"))
    facts = {"r1.json": {"byte_sha256": "a" * 64}, "r1_copy.json": {"byte_sha256": "a" * 64},
             "r2.json": {"byte_sha256": "b" * 64},
             "semantic.json": {"byte_sha256": "c" * 64, "verified_semantic_identities": {"result_sha256": "d" * 64}}}
    source1 = {"path": "ledger1.json", "byte_sha256": "1" * 64}
    source2 = {"path": "ledger2.json", "byte_sha256": "2" * 64}
    item1 = {"objective_id": "A", "result_path": "r1.json", "result_sha256": "a" * 64, "candidate_count": 4}
    first = api["observation"](source1, "new_attempts", 0, item1, facts)
    second = api["observation"](source2, "known_later_attempts", 0,
                                {**item1, "result_path": "r1_copy.json"}, facts)
    check(first["result_reference"]["status"] == "verified_byte_reference")
    check(first["budget_claims"]["candidate_count"] == {"value": 4, "verification": "reported_only", "unknown_reason": None})
    groups = api["result_groups"]([first, second])
    check(len(groups) == 1 and len(groups[0]["observations"]) == 2)
    check(groups[0]["statistical_trial_count"] is None)
    check(len(groups[0]["roles"]) == 2 and len(groups[0]["result_paths"]) == 2)
    other = api["observation"](source2, "new_attempts", 1,
                               {**item1, "result_path": "r2.json", "result_sha256": "b" * 64}, facts)
    check(len(api["result_groups"]([first, other])) == 2)  # same objective != same result
    unknown1 = api["observation"](source1, "new_attempts", 1, {"name": "A"}, facts)
    unknown2 = api["observation"](source2, "new_attempts", 1, {"name": "A"}, facts)
    check(unknown1["objective_id"] is None)
    check(len(api["result_groups"]([unknown1, unknown2])) == 2)
    check(api["hash_reference"]({"result_path": "r1.json"}, "result_path", ("result_sha256",), facts)["status"] == "no_verified_byte_claim")
    check(api["hash_reference"]({"result_path": "not_present", "result_sha256": "a" * 64}, "result_path", ("result_sha256",), facts)["status"] == "not_in_verified_inventory")
    conflict = api["hash_reference"]({"result_path": "r1.json", "result_file_sha256": "a" * 64, "result_sha256": "b" * 64},
                                      "result_path", ("result_file_sha256", "result_sha256"), facts)
    check(conflict["status"] == "hash_claim_conflict" and conflict["byte_sha256"] is None)
    semantic_only = api["hash_reference"]({"result_path": "semantic.json", "result_sha256": "d" * 64},
                                           "result_path", ("result_file_sha256", "result_sha256"), facts)
    check(semantic_only["status"] == "no_verified_byte_claim")
    dual = api["hash_reference"]({"result_path": "semantic.json", "result_file_sha256": "c" * 64, "result_sha256": "d" * 64},
                                  "result_path", ("result_file_sha256", "result_sha256"), facts)
    check(dual["status"] == "verified_byte_reference" and not dual["conflicting_claim_fields"])
    engineering = api["observation"](source1, "engineering_checks", 0, item1, facts)
    check(engineering["role"] == "engineering_check")
    bad_budget = api["observation"](source1, "new_attempts", 2, {"candidate_count": {"registered": 4}}, facts)
    check(bad_budget["budget_claims"]["candidate_count"]["value"] is None)
    changed_id = api["observation"](source2, "new_attempts", 3, {**item1, "objective_id": "B"}, facts)
    check(api["result_groups"]([first, changed_id])[0]["identity_conflict"])
    proof = {"path": "synthetic.json", "byte_sha256": "e" * 64, "field": "objective_id"}
    objectives = api["objective_groups"]([
        {"objective_id": "A", "config_path": "c1", "config_sha256": "a" * 64, "provenance": proof},
        {"objective_id": "A", "config_path": "c2", "config_sha256": "b" * 64, "provenance": proof},
        {"objective_id": "a", "provenance": proof}, {"objective_id": "A ", "provenance": proof}], {"c1": "a" * 64})
    check(len(objectives) == 3)
    group_a = next(g for g in objectives if g["objective_id"] == "A")
    check(group_a["contract_identity_status"] == "multiple_contract_hash_claims_not_equated")
    check(group_a["contract_claims"][0]["verification"] == "byte_hash_verified_in_bound_context_inputs")
    check(group_a["contract_claims"][1]["verification"] == "reported_only")
    cycle_facts = {"l1": {"byte_sha256": "a" * 64}, "l2": {"byte_sha256": "b" * 64}}
    ledgers = [{"path": "l1", "payload": {"previous_ledger_path": "l2", "previous_ledger_sha256": "b" * 64}},
               {"path": "l2", "payload": {"previous_ledger_path": "l1", "previous_ledger_sha256": "a" * 64}}]
    cycle = api["ledger_chains"](ledgers, cycle_facts)
    check(cycle["cycle_nodes"] and not cycle["recorded_links_verified"])
    ledgers[1]["payload"] = {}
    chain = api["ledger_chains"](ledgers, cycle_facts)
    check(chain["recorded_links_verified"] and chain["global_history_complete"] is False)
    names = set(api["DEFINITIONS"])
    dictionary = api["metric_dictionary"]([], names)
    check(len(dictionary["definitions"]) == 14)
    check(all(d["unknown_value"] is None and not d["missing_is_zero"] for d in dictionary["definitions"]))
    check(rejects(lambda: api["metric_dictionary"]([], names - {"marked_net_return"})))
    record = {"evaluations": [{"adapter": "synthetic_pct", "metrics": [{"name": "total_net_return_reported",
               "normalized_unit": "fraction", "source_unit": "percent", "source_field": "x.net_return_pct"}]}]}
    observed = api["metric_dictionary"]([record], names)
    check(observed["observed_aliases"][0]["conversion"] == "divide_by_100")
    check([d["definition_sha256"] for d in observed["definitions"]] == [d["definition_sha256"] for d in dictionary["definitions"]])
    bad_record = deepcopy(record)
    bad_record["evaluations"][0]["metrics"][0]["source_unit"] = "count"
    check(rejects(lambda: api["metric_dictionary"]([bad_record], names)))

    builder = runpy.run_path(str(root / "tools/build_genetic_primus_observed_registry_v1.py"))
    _, actual_dictionary, registry, summary, inputs, _, _, _ = builder["prepare"]()
    check(summary["sources"] == 1784 and summary["metric_fields_covered"] == 233)
    check(summary["metric_definitions"] == 14 and summary["ledger_snapshots"] == 24)
    check(summary["registered_objective_snapshots"] == 9)
    check(registry["historical_statistical_trial_count"] is None)
    check(registry["unknown_historical_adaptive_decisions"] is True)
    check(registry["global_dsr_pbo_calibrated"] is registry["global_immutable_registry_verified"] is False)
    check(all(registry[k] is False for k in ("execution_allowed", "credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority", "cross_source_comparison_authority")))
    check(registry["new_training_or_financial_evaluation_runs_started"] == 0)
    check(registry["metric_dictionary_semantic_sha256"] == api["digest"](actual_dictionary))
    check(all(inputs[o["provenance"]["path"]]["byte_sha256"] == o["provenance"]["byte_sha256"] for o in registry["ledger_observations"]))
    check(all(g["statistical_trial_count"] is None for g in registry["observed_result_artifact_groups"]))
    print(f"observed registry regressions: {checks} passed; no training/evaluation/runtime")


if __name__ == "__main__":
    main()
