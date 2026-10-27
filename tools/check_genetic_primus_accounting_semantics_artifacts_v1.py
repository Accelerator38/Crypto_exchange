"""Reconstruct the metadata/number audit and verify the unchanged input bank."""

from hashlib import sha256
import json
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "Reports/Exia/Genetic_Primus/accounting_semantics_audit_v1_20261005"
    builder_path = root / "tools/build_genetic_primus_accounting_semantics_audit_v1.py"
    module_path = root / "src/exia/genetic_primus/accounting_semantics_audit_v1.py"
    receipt = json.loads((output / "completion_receipt.json").read_bytes())
    assert receipt["inputs_verified_unchanged_before_and_after"] is True
    assert sha256(builder_path.read_bytes()).hexdigest() == receipt["builder_byte_sha256"]
    assert sha256(module_path.read_bytes()).hexdigest() == receipt["module_byte_sha256"]
    builder = runpy.run_path(str(builder_path))
    api, audit, summary, inputs, verify, _, _ = builder["prepare"]()
    verify()
    assert api["canonical"](list(inputs.values())) == api["canonical"](receipt["input_files"])
    for filename, expected in (("accounting_audit.json", audit), ("summary.json", summary)):
        raw = (output / filename).read_bytes()
        assert sha256(raw).hexdigest() == receipt["output_byte_sha256"][filename]
        assert api["canonical"](json.loads(raw)) == api["canonical"](expected)
    assert len(audit["evaluation_checks"]) == 25
    catalog = json.loads((root / builder["CATALOG"]).read_bytes())
    envelopes = {e["source"]["path"]: e for e in catalog["records"]}
    for evaluation in audit["evaluation_checks"]:
        envelope = envelopes[evaluation["source_path"]]
        original = next(e for e in envelope["evaluations"] if (e["policy"], e["scenario"]) == (evaluation["policy"], evaluation["scenario"]))
        assert envelope["envelope_sha256"] == evaluation["base_envelope_sha256"]
        assert api["digest"](original["metrics"]) == evaluation["base_metrics_semantic_sha256"]
    for profile in audit["producer_profiles"]:
        assert profile["profile_sha256"] == api["digest"]({k: v for k, v in profile.items() if k != "profile_sha256"})
        assert profile["strict_historical_accounting_verified"] is False
        proof = profile["producer_callsite"]["proof"]
        assert inputs[proof["path"]]["byte_sha256"] == proof["byte_sha256"]
    for flag in ("execution_allowed", "credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority"):
        assert audit[flag] is receipt[flag] is False
    assert audit["historical_dependency_graph_verified"] is audit["all_costs_execution_verified"] is False
    assert audit["cross_source_comparison_authority"] is False
    print("accounting artifacts: receipt/provenance/base metrics verified; 1784 originals and contracts unchanged")
    print("No simulator/market replay/model fit; historical dependency identity and trading authority remain unverified.")


if __name__ == "__main__":
    main()
