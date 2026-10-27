"""Observed identities and metric definitions; not a global trial-count estimator."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re


SCHEMA = "exia.genetic_primus.observed_research_registry/1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
COLLECTIONS = {
    "new_attempts": "declared_attempt", "known_later_attempts": "historically_reported_attempt",
    "new_diagnostics": "diagnostic_not_assumed_new_trial", "engineering_checks": "engineering_check",
    "engineering_updates": "engineering_update", "implementation_updates": "engineering_update",
    "new_registrations": "registration_not_execution", "registry_objectives": "objective_declaration",
    "research_reviews": "review_not_execution", "research_drafts": "draft_not_execution",
    "deferred_checks": "deferred_not_execution",
}
BUDGET_FIELDS = ("candidate_count", "candidate_budget", "recorded_policy_evaluations", "search_evaluations",
                 "new_model_fits", "candidate_count_added", "origins", "calendar_days", "decision_count")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and bool(SHA.fullmatch(value))


def objective_id(value):
    # Do not trim, case-fold or infer identity from a filename/name/family.
    return value if isinstance(value, str) and value and len(value) <= 256 else None


def count_claim(value):
    if type(value) in (int, float):
        try:
            if math.isfinite(value) and value >= 0 and int(value) == value:
                return int(value)
        except OverflowError:
            pass
    return None


@dataclass(frozen=True)
class Reference:
    path: str
    byte_sha256: str
    field: str


def hash_reference(item, path_field, hash_fields, facts):
    """Match bytes, never assume that a field named result_sha256 means bytes."""
    target_path = item.get(path_field)
    result = {"reported_path": target_path if isinstance(target_path, str) else None,
              "status": "missing_path", "identity_kind": None, "byte_sha256": None,
              "matching_claim_fields": [], "conflicting_claim_fields": []}
    if not isinstance(target_path, str) or not target_path:
        return result
    target = facts.get(target_path)
    if target is None:
        result["status"] = "not_in_verified_inventory"
        return result
    byte_sha = target["byte_sha256"]
    matched, conflicting = [], []
    for field in hash_fields:
        value = item.get(field)
        if value == byte_sha:
            matched.append(field)
        elif value is not None:
            # A separately declared semantic identity is not a byte-hash conflict.
            known_semantic = (field == "result_sha256" and valid_sha(value)
                              and target.get("verified_semantic_identities", {}).get("result_sha256") == value)
            if not known_semantic:
                conflicting.append(field)
    result.update(matching_claim_fields=matched, conflicting_claim_fields=conflicting)
    if conflicting:
        result["status"] = "hash_claim_conflict"
    elif matched:
        result.update(status="verified_byte_reference", identity_kind="bytes", byte_sha256=byte_sha)
    else:
        result["status"] = "no_verified_byte_claim"
    return result


def observation(source, collection, index, item, facts):
    role = COLLECTIONS[collection]
    ref = Reference(source["path"], source["byte_sha256"], f"{collection}[{index}]")
    if not isinstance(item, dict):
        return {"observation_id": digest(asdict(ref)), "provenance": asdict(ref), "role": role,
                "issue": "unsupported_record_shape", "objective_id": None,
                "result_reference": hash_reference({}, "result_path", (), facts), "budget_claims": {}}
    result = hash_reference(item, "result_path", ("result_file_sha256", "result_sha256"), facts)
    return {"observation_id": digest(asdict(ref)), "provenance": asdict(ref), "role": role,
            "objective_id": objective_id(item.get("objective_id")),
            "reported_source_objective_id": objective_id(item.get("source_attempt_objective_id")),
            "reported_attempt_id": objective_id(item.get("attempt_id")),
            "reported_name": item.get("name") if isinstance(item.get("name"), str) else None,
            "reported_status": item.get("status") if isinstance(item.get("status"), str) else None,
            "reported_evidence": item.get("evidence_class") if isinstance(item.get("evidence_class"), str) else None,
            "result_reference": result,
            "budget_claims": {k: {"value": count_claim(item[k]), "verification": "reported_only",
                                  "unknown_reason": None if count_claim(item[k]) is not None else "not_a_valid_scalar_count"}
                              for k in BUDGET_FIELDS if k in item},
            "config_claim": {"path": item.get("config_path") if isinstance(item.get("config_path"), str) else None,
                             "sha256": item.get("config_sha256") if valid_sha(item.get("config_sha256")) else None}}


def result_groups(observations):
    """Same verified result bytes => one observed artifact group, NOT one trial."""
    grouped = {}
    for observed in observations:
        reference = observed["result_reference"]
        if reference["status"] == "verified_byte_reference":
            basis = "verified_result_bytes"
            key = "result_bytes:" + reference["byte_sha256"]
        else:
            basis = "separate_unresolved_observation_no_deduplication"
            key = "observation:" + observed["observation_id"]
        group = grouped.setdefault(key, {"observed_group_id": digest({"key": key}), "basis": basis,
                                        "result_byte_sha256": reference["byte_sha256"],
                                        "observations": [], "roles": set(), "objective_ids": set(),
                                        "reported_attempt_ids": set(), "result_paths": set(),
                                        "statistical_trial_count": None})
        group["observations"].append(observed["observation_id"])
        group["roles"].add(observed["role"])
        if observed.get("objective_id"):
            group["objective_ids"].add(observed["objective_id"])
        if observed.get("reported_attempt_id"):
            group["reported_attempt_ids"].add(observed["reported_attempt_id"])
        if reference["reported_path"]:
            group["result_paths"].add(reference["reported_path"])
    output = []
    for key in sorted(grouped):
        group = grouped[key]
        for field in ("roles", "objective_ids", "reported_attempt_ids", "result_paths"):
            group[field] = sorted(group[field])
        group["identity_conflict"] = len(group["objective_ids"]) > 1 or len(group["reported_attempt_ids"]) > 1
        group["evidence_verified_independent"] = False
        output.append(group)
    return output


def objective_groups(claims, verified_inputs):
    grouped = {}
    for claim in claims:
        raw_id = objective_id(claim.get("objective_id"))
        if raw_id is None:
            continue
        group = grouped.setdefault(raw_id, {"objective_id": raw_id,
                                           "observed_identity_id": digest({"kind": "objective_id_claim", "value": raw_id}),
                                           "provenance": [], "contract_claims": {}})
        group["provenance"].append(claim["provenance"])
        path, hashed = claim.get("config_path"), claim.get("config_sha256")
        if isinstance(path, str) and valid_sha(hashed):
            group["contract_claims"][(path, hashed)] = {
                "path": path, "reported_byte_sha256": hashed,
                "verification": "byte_hash_verified_in_bound_context_inputs" if verified_inputs.get(path) == hashed else "reported_only"}
    output = []
    for raw_id in sorted(grouped):
        group = grouped[raw_id]
        contracts = group["contract_claims"]
        group["contract_claims"] = [contracts[key] for key in sorted(contracts)]
        group["contract_identity_status"] = ("multiple_contract_hash_claims_not_equated" if len({k[1] for k in contracts}) > 1
                                             else "one_reported_contract_hash" if contracts else "contract_unknown")
        group["global_objective_semantics_verified"] = False
        output.append(group)
    return output


def ledger_chains(ledgers, facts):
    references, edges = [], {}
    for ledger in ledgers:
        value = ledger["payload"]
        if value.get("previous_ledger_path") is None and value.get("previous_ledger_sha256") is None:
            continue
        reference = hash_reference(value, "previous_ledger_path", ("previous_ledger_sha256",), facts)
        references.append({"source_path": ledger["path"], **reference})
        if reference["status"] == "verified_byte_reference":
            if reference["reported_path"] in {x["path"] for x in ledgers}:
                edges[ledger["path"]] = reference["reported_path"]
            else:
                references[-1]["status"] = "target_not_a_ledger"
    cycles = set()
    for start in edges:
        seen, current = set(), start
        for _ in range(len(edges) + 1):
            if current in seen:
                cycles.add(current)
                break
            seen.add(current)
            if current not in edges:
                break
            current = edges[current]
    return {"references": references, "cycle_nodes": sorted(cycles),
            "recorded_links_verified": not cycles and all(r["status"] == "verified_byte_reference" for r in references),
            "global_history_complete": False}


# Definitions describe normalized reported quantities, not newly established semantics.
DEFINITIONS = {
    "total_net_return_reported": ("fraction", "reported_total", "higher", "aggregation/terminal/capital rules require source accounting"),
    "total_gross_return_reported": ("fraction", "reported_total", "higher", "before reported costs; actual gross calculation not re-audited"),
    "marked_net_return": ("fraction", "marked_NAV_total", "higher", "unrealized PnL included under source marked-NAV rules"),
    "net_after_hypothetical_terminal_liquidation": ("fraction", "hypothetical_terminal_NAV_total", "higher", "hypothetical terminal fee is not a real fill or all-cost verification"),
    "mean_net_return_per_trade": ("fraction", "reported_trade", "higher", "trade boundaries/weighting and cost scope require source definition"),
    "net_return_lcb_per_trade": ("fraction", "reported_trade", "higher", "confidence method/alpha/resample bank/audit bank require provenance"),
    "mean_net_return_per_calendar_day": ("fraction", "calendar_day", "higher", "complete zero-return days and averaging rule require source proof"),
    "net_return_lcb_per_calendar_day": ("fraction", "calendar_day", "higher", "not per-trade LCB; dependence/bootstrap method must be specified"),
    "max_drawdown_reported_sampling_unknown": ("fraction", "reported_NAV_path", "lower", "sampling/capital/valuation unknown; not asserted intrabar"),
    "max_drawdown_bar_close": ("fraction", "bar_close_NAV_path", "lower", "bar-close only; cannot bound intrabar risk"),
    "trades_reported": ("count", "reported_trade_count", "neither", "not asserted closed episodes; distinct from fills and transitions"),
    "fills_reported": ("count", "reported_fill_count", "neither", "simulated/reported fills are not verified real orders"),
    "position_transition_count": ("count", "position_transition", "neither", "not trade count or guaranteed exchange fill count"),
    "fees_fraction_initial_capital": ("fraction", "initial_capital_fee_sum", "lower", "not fees as percent of current NAV and not all execution costs"),
}


def metric_dictionary(catalog_records, allowed_names):
    if set(DEFINITIONS) != set(allowed_names):
        raise ValueError("metric definition coverage mismatch")
    aliases, observed_counts = {}, {name: 0 for name in DEFINITIONS}
    for record in catalog_records:
        for evaluation in record["evaluations"]:
            for metric in evaluation["metrics"]:
                name, unit = metric["name"], metric["source_unit"]
                if name not in DEFINITIONS or metric["normalized_unit"] != DEFINITIONS[name][0]:
                    raise ValueError("metric dictionary semantics mismatch")
                if (DEFINITIONS[name][0] == "count") != (unit == "count"):
                    raise ValueError("source unit/metric semantics mismatch")
                conversion = {"percent": "divide_by_100", "bps": "divide_by_10000", "fraction": "identity", "count": "identity_integer"}.get(unit)
                if conversion is None:
                    raise ValueError("unsupported source unit")
                tail = metric["source_field"].rsplit(".", 1)[-1]
                key = (name, evaluation["adapter"], tail, unit)
                aliases.setdefault(key, {"metric": name, "adapter": evaluation["adapter"],
                                         "source_field_tail": tail, "source_unit": unit,
                                         "conversion": conversion, "observed_fields": 0})["observed_fields"] += 1
                observed_counts[name] += 1
    definitions = []
    for name, (unit, horizon, direction, caveat) in DEFINITIONS.items():
        entry = {"name": name, "normalized_unit": unit, "aggregation_scope": horizon,
                 "preferred_direction": direction, "caveat": caveat,
                 "unknown_value": None, "missing_is_zero": False, "observed_fields": observed_counts[name],
                 "source_semantics_verified_by_normalization": False,
                 "required_context_for_cross_source_comparison": ["data_identity", "period", "accounting",
                                                                  "cost_funding_delay", "evidence_scope", "statistical_method_if_applicable"]}
        entry["definition_sha256"] = digest({k: v for k, v in entry.items() if k != "observed_fields"})
        definitions.append(entry)
    return {"schema": "exia.genetic_primus.metric_dictionary/1", "definitions": definitions,
            "observed_aliases": [aliases[key] for key in sorted(aliases)],
            "automatic_cross_source_comparison_authority": False}
