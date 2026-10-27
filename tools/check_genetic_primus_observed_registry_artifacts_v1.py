"""Independently reconstruct metadata products and verify their bound source bytes."""

from hashlib import sha256
import json
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "Reports/Exia/Genetic_Primus/observed_research_registry_v1_20261005"
    builder_path = root / "tools/build_genetic_primus_observed_registry_v1.py"
    module_path = root / "src/exia/genetic_primus/research_identity_registry_v1.py"
    completion = json.loads((directory / "completion_receipt.json").read_bytes())
    assert completion["inputs_verified_unchanged_before_and_after"] is True
    assert sha256(builder_path.read_bytes()).hexdigest() == completion["builder_byte_sha256"]
    assert sha256(module_path.read_bytes()).hexdigest() == completion["module_byte_sha256"]
    builder = runpy.run_path(str(builder_path))
    api, dictionary, registry, summary, inputs, verify, _, _ = builder["prepare"]()
    verify()
    assert api["canonical"](completion["input_files"]) == api["canonical"](list(inputs.values()))
    for filename, expected in (("metric_dictionary.json", dictionary), ("observed_registry.json", registry), ("summary.json", summary)):
        raw = (directory / filename).read_bytes()
        assert sha256(raw).hexdigest() == completion["output_byte_sha256"][filename]
        assert api["canonical"](json.loads(raw)) == api["canonical"](expected)
    assert len(registry["artifact_inventory"]) == 1784
    assert sum(d["observed_fields"] for d in dictionary["definitions"]) == 233
    assert registry["metric_dictionary_semantic_sha256"] == api["digest"](dictionary)
    assert registry["historical_statistical_trial_count"] is None
    assert registry["unknown_historical_adaptive_decisions"] is True
    assert registry["global_dsr_pbo_calibrated"] is registry["global_immutable_registry_verified"] is False
    assert registry["new_training_or_financial_evaluation_runs_started"] == 0
    for flag in ("execution_allowed", "credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority"):
        assert registry[flag] is completion[flag] is False
    assert registry["cross_source_comparison_authority"] is dictionary["automatic_cross_source_comparison_authority"] is False
    observation_ids = {o["observation_id"] for o in registry["ledger_observations"]}
    assert len(observation_ids) == len(registry["ledger_observations"])
    grouped_ids = [identity for group in registry["observed_result_artifact_groups"] for identity in group["observations"]]
    assert len(grouped_ids) == len(set(grouped_ids)) and set(grouped_ids) == observation_ids
    for objective in registry["objective_claim_groups"]:
        for proof in objective["provenance"]:
            assert inputs[proof["path"]]["byte_sha256"] == proof["byte_sha256"]
    for binding in registry["registry_byte_bindings"]:
        assert inputs[binding["target_path"]]["byte_sha256"] == binding["target_byte_sha256"]
        proof = binding["provenance"]
        assert inputs[proof["path"]]["byte_sha256"] == proof["byte_sha256"]
    print("observed registry artifacts: exact outputs/receipt/provenance verified; 1784 originals unchanged")
    print("Predecessor catalogs/contracts unchanged; trial count unknown; no training or trading authority.")


if __name__ == "__main__":
    main()
