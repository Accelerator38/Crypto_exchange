"""Finite metadata consolidation; original artifacts/contracts remain untouched."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
INVENTORY = BASE + "development_review_v1_20261005/evidence_index_v2.json"
INVENTORY_SHA = "5e6192ccef065c9e0f488395bfd9456d160c071d706a9985447149e172d89bed"
CATALOG = BASE + "result_catalog_v1_20261005_r2/catalog.json"
CATALOG_SHA = "cafb0375dab4e0e608004c3fe9c216302ebc30331a745a0885d2c285ddbdb08b"
CONTEXT = BASE + "result_context_links_v1_20261005/context_links.json"
CONTEXT_SHA = "d2bec13b402512aa7d5ef08cef10eccd9789d7fb58909e52df1d855307f81c03"
CONTEXT_RECEIPT = BASE + "result_context_links_v1_20261005/completion_receipt.json"
CONTEXT_RECEIPT_SHA = "82160f126db5df8b67ec514e73a5cf08b3ace5ef7e1313cff067d40cdf737b9c"
OUTPUT = ROOT / (BASE + "observed_research_registry_v1_20261005")


def prepare():
    module_path = ROOT / "src/exia/genetic_primus/research_identity_registry_v1.py"
    module_sha = sha256(module_path.read_bytes()).hexdigest()
    builder_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    api = runpy.run_path(str(module_path))
    catalog_api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/result_catalog_v1.py"))
    inputs = {}

    def read(relative, expected):
        path = (ROOT / relative).resolve(strict=True)
        allowed = ((relative.startswith(BASE) and path.is_relative_to((ROOT / BASE).resolve()))
                   or (relative.startswith("configs/genetic_primus_") and path.parent == (ROOT / "configs").resolve()))
        if not allowed or path.suffix != ".json":
            raise ValueError("metadata path outside allowed scope")
        if path.stat().st_size > (40_000_000 if relative == INVENTORY else 8_000_000):
            raise ValueError("metadata size cap exceeded")
        raw = path.read_bytes()
        hashed = sha256(raw).hexdigest()
        if hashed != expected:
            raise ValueError("input byte hash mismatch: " + relative)
        inputs[relative] = {"path": relative, "byte_sha256": hashed, "size_bytes": len(raw)}
        return raw

    inventory = json.loads(read(INVENTORY, INVENTORY_SHA))["records"]
    catalog = json.loads(read(CATALOG, CATALOG_SHA))
    context = json.loads(read(CONTEXT, CONTEXT_SHA))
    receipt = json.loads(read(CONTEXT_RECEIPT, CONTEXT_RECEIPT_SHA))
    if receipt["output_byte_sha256"]["context_links.json"] != CONTEXT_SHA:
        raise ValueError("context receipt binding mismatch")
    if context["base_catalog_byte_sha256"] != CATALOG_SHA or context["inventory_byte_sha256"] != INVENTORY_SHA:
        raise ValueError("predecessor binding mismatch")
    records = {e["source"]["path"]: e for e in catalog["records"]}
    if len(inventory) != 1784 or len(records) != 1784 or set(records) != {r["path"] for r in inventory}:
        raise ValueError("inventory/catalog coverage mismatch")
    verified_context_inputs = {r["path"]: r["byte_sha256"] for r in receipt["input_files"]}
    facts, ledgers, claims, registry_refs = {}, [], [], []
    for original in inventory:
        path = original["path"]
        envelope = records[path]
        catalog_api["validate_serialized"](envelope)
        if envelope["source"]["byte_sha256"] != original["sha256"]:
            raise ValueError("catalog source identity mismatch")
        raw = read(path, original["sha256"])
        if len(raw) != original["size_bytes"]:
            raise ValueError("original size mismatch")
        payload = catalog_api["load_json_metadata"](raw)
        payload = payload if isinstance(payload, dict) else {}
        fact = {"path": path, "byte_sha256": original["sha256"], "artifact_kind": envelope["artifact_kind"],
                "reported_schema": envelope["source"]["reported_schema"],
                "reported_objective_id": api["objective_id"](payload.get("objective_id")),
                "verified_semantic_identities": {},
                "root_budget_claims": {k: {"value": api["count_claim"](payload[k]), "verification": "reported_only"}
                                       for k in api["BUDGET_FIELDS"] if k in payload}}
        # Only this exact semantic recipe is identified; no unchecked field-name inference.
        reported_semantic = payload.get("result_sha256")
        if api["valid_sha"](reported_semantic):
            try:
                calculated = api["digest"]({k: v for k, v in payload.items() if k != "result_sha256"})
            except (ValueError, TypeError, OverflowError):
                calculated = None
            if calculated == reported_semantic:
                fact["verified_semantic_identities"]["result_sha256"] = reported_semantic
        facts[path] = fact
        if fact["reported_objective_id"]:
            claims.append({"objective_id": fact["reported_objective_id"],
                           "provenance": asdict(api["Reference"](path, original["sha256"], "objective_id"))})
        if envelope["artifact_kind"] == "research_ledger":
            ledger = {"path": path, "byte_sha256": original["sha256"], "payload": payload}
            ledgers.append(ledger)
            for index, ref in enumerate(payload.get("registry_chain", [])):
                if isinstance(ref, dict):
                    registry_refs.append({"source": ledger, "index": index, "item": ref})
        # Deliberately do not retain models, arrays, returns or all parsed JSONs in memory.
    observations = []
    for ledger in ledgers:
        for collection in api["COLLECTIONS"]:
            values = ledger["payload"].get(collection, [])
            if not isinstance(values, list):
                raise ValueError("unsupported ledger collection shape")
            for index, item in enumerate(values):
                observed = api["observation"](ledger, collection, index, item, facts)
                observations.append(observed)
                if observed.get("objective_id"):
                    config_claim = observed.get("config_claim", {})
                    claims.append({"objective_id": observed["objective_id"], "provenance": observed["provenance"],
                                   "config_path": config_claim.get("path"), "config_sha256": config_claim.get("sha256")})
    external_registries, external_proofs, processed = [], [], set()
    for reference in registry_refs:
        item, source = reference["item"], reference["source"]
        path, expected = item.get("path"), item.get("sha256")
        if not isinstance(path, str) or not api["valid_sha"](expected):
            raise ValueError("invalid declared registry reference")
        raw = read(path, expected)
        external_proofs.append({"provenance": asdict(api["Reference"](source["path"], source["byte_sha256"],
                                                                      f"registry_chain[{reference['index']}]")),
                                "target_path": path, "target_byte_sha256": expected, "identity_kind": "bytes"})
        if (path, expected) in processed:
            continue
        processed.add((path, expected))
        value = catalog_api["load_json_metadata"](raw)
        if not isinstance(value, dict) or not isinstance(value.get("objectives"), list):
            raise ValueError("unsupported registered objective collection")
        external_registries.append({"path": path, "byte_sha256": expected, "registry_id": value.get("registry_id"),
                                    "reported_schema": value.get("schema_version"), "objective_records": len(value["objectives"]),
                                    "global_immutability_verified": False})
        for index, item in enumerate(value["objectives"]):
            if isinstance(item, dict):
                claims.append({"objective_id": api["objective_id"](item.get("objective_id")),
                               "config_path": item.get("config_path"), "config_sha256": item.get("config_sha256"),
                               "provenance": asdict(api["Reference"](path, expected, f"objectives[{index}]"))})
    objectives = api["objective_groups"](claims, verified_context_inputs)
    groups = api["result_groups"](observations)
    chains = api["ledger_chains"](ledgers, facts)
    dictionary = api["metric_dictionary"](catalog["records"], {str(name) for name in catalog_api["MetricName"]})
    linked_results = {o["result_reference"]["reported_path"] for o in observations
                      if o["result_reference"]["status"] == "verified_byte_reference"}
    registry = {"schema": api["SCHEMA"], "inventory_byte_sha256": INVENTORY_SHA,
                "base_catalog_byte_sha256": CATALOG_SHA, "context_links_byte_sha256": CONTEXT_SHA,
                "metric_dictionary_semantic_sha256": api["digest"](dictionary),
                "objective_claim_groups": objectives, "ledger_observations": observations,
                "observed_result_artifact_groups": groups, "ledger_chains": chains,
                "registered_objective_snapshots": external_registries, "registry_byte_bindings": external_proofs,
                "artifact_inventory": list(facts.values()),
                "unlinked_financial_artifacts": [path for path, f in facts.items()
                                                 if f["artifact_kind"] == "financial_result" and path not in linked_results],
                "historical_statistical_trial_count": None, "unknown_historical_adaptive_decisions": True,
                "global_immutable_registry_verified": False, "global_dsr_pbo_calibrated": False,
                "new_training_or_financial_evaluation_runs_started": 0,
                "execution_allowed": False, "credentials_allowed": False, "orders_enabled": False,
                "paper": False, "live": False, "promotion_authority": False, "cross_source_comparison_authority": False}
    summary = {"schema": "exia.genetic_primus.observed_research_registry_summary/1", "sources": len(facts),
               "metric_definitions": len(dictionary["definitions"]), "metric_aliases": len(dictionary["observed_aliases"]),
               "metric_fields_covered": sum(d["observed_fields"] for d in dictionary["definitions"]),
               "objective_id_claim_groups": len(objectives),
               "objective_multiple_contract_hash_claims": sum(g["contract_identity_status"] == "multiple_contract_hash_claims_not_equated" for g in objectives),
               "ledger_snapshots": len(ledgers), "ledger_reference_count": len(chains["references"]),
               "ledger_recorded_links_verified": chains["recorded_links_verified"],
               "ledger_observation_roles": dict(sorted(Counter(o["role"] for o in observations).items())),
               "result_reference_statuses": dict(sorted(Counter(o["result_reference"]["status"] for o in observations).items())),
               "observed_result_artifact_groups": len(groups),
               "verified_result_byte_groups": sum(g["basis"] == "verified_result_bytes" for g in groups),
               "result_group_identity_conflicts": sum(g["identity_conflict"] for g in groups),
               "registered_objective_snapshots": len(external_registries),
               "financial_artifacts_without_verified_ledger_result_reference": len(registry["unlinked_financial_artifacts"]),
               "sources_without_explicit_root_objective_id": sum(f["reported_objective_id"] is None for f in facts.values()),
               "historical_statistical_trial_count": None, "unknown_historical_adaptive_decisions": True,
               "global_immutable_registry_verified": False, "global_dsr_pbo_calibrated": False,
               "new_training_or_financial_evaluation_runs_started": 0,
               "execution_allowed": False, "promotion_authority": False, "cross_source_comparison_authority": False}

    def verify():
        # Includes all original files and bound context inputs; future config is bytes only.
        for previous in receipt["input_files"]:
            read(previous["path"], previous["byte_sha256"])
        for original in inventory:
            if len(read(original["path"], original["sha256"])) != original["size_bytes"]:
                raise ValueError("original size changed")
        for current in list(inputs.values()):
            read(current["path"], current["byte_sha256"])
        if sha256(module_path.read_bytes()).hexdigest() != module_sha or sha256(Path(__file__).read_bytes()).hexdigest() != builder_sha:
            raise ValueError("registry implementation changed")
    return api, dictionary, registry, summary, inputs, verify, module_sha, builder_sha


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing snapshot must not be overwritten")
    api, dictionary, registry, summary, inputs, verify, module_sha, builder_sha = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for filename, value in (("metric_dictionary.json", dictionary), ("observed_registry.json", registry), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / filename).open("xb") as stream:
                stream.write(raw)
            hashes[filename] = sha256(raw).hexdigest()
        verify()
        completion = {"schema": "exia.genetic_primus.observed_research_registry_completion/1",
                      "input_files": list(inputs.values()), "output_byte_sha256": hashes,
                      "inputs_verified_unchanged_before_and_after": True,
                      "module_byte_sha256": module_sha, "builder_byte_sha256": builder_sha,
                      "execution_allowed": False, "credentials_allowed": False, "orders_enabled": False,
                      "paper": False, "live": False, "promotion_authority": False}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](completion) + b"\n")
    print(f"{'preflight' if dry else 'observed registry'}: {summary['sources']} artifacts; {summary['metric_definitions']} metric definitions; {summary['objective_id_claim_groups']} objective claims")
    print("Originals/context unchanged; statistical trial count unknown; DSR/PBO and trading authority false.")


if __name__ == "__main__":
    main()
