"""Bounded verification of the completed catalog and untouched original JSONs."""

from hashlib import sha256
import json
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "Reports/Exia/Genetic_Primus"
    output = base / "result_catalog_v1_20261005_r2"
    api = runpy.run_path(str(root / "src/exia/genetic_primus/result_catalog_v1.py"))
    receipt = json.loads((output / "completion_receipt.json").read_bytes())
    assert receipt["originals_verified_unchanged_before_and_after"] is True
    for basename, key in (("catalog.json", "catalog_byte_sha256"), ("summary.json", "summary_byte_sha256")):
        assert sha256((output / basename).read_bytes()).hexdigest() == receipt[key]
    for relative, key in (("src/exia/genetic_primus/result_catalog_v1.py", "adapter_module_byte_sha256"),
                          ("tools/build_genetic_primus_result_catalog_v1.py", "builder_byte_sha256")):
        assert sha256((root / relative).read_bytes()).hexdigest() == receipt[key]
    inventory_raw = (base / "development_review_v1_20261005/evidence_index_v2.json").read_bytes()
    assert sha256(inventory_raw).hexdigest() == receipt["inventory_byte_sha256"]
    inventory = json.loads(inventory_raw)["records"]
    catalog = json.loads((output / "catalog.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    assert len(inventory) == len(catalog["records"]) == receipt["sources"] == summary["sources"] == 1784
    assert catalog["execution_allowed"] is catalog["promotion_authority"] is False
    bank = []
    for original, envelope in zip(inventory, catalog["records"]):
        api["validate_serialized"](envelope)
        source = envelope["source"]
        assert (source["path"], source["byte_sha256"], source["size_bytes"]) == (
            original["path"], original["sha256"], original["size_bytes"])
        path = (root / source["path"]).resolve(strict=True)
        assert path.is_relative_to(base.resolve()) and path.suffix == ".json"
        assert source["size_bytes"] <= 8_000_000
        raw = path.read_bytes()
        assert len(raw) == source["size_bytes"] and sha256(raw).hexdigest() == source["byte_sha256"]
        bank.append((source["path"], source["byte_sha256"], source["size_bytes"]))
    assert sha256(api["canonical_bytes"](bank)).hexdigest() == receipt["source_identity_bank_sha256"] == catalog["source_identity_bank_sha256"]
    evaluations = [e for r in catalog["records"] for e in r["evaluations"]]
    assert len(evaluations) == summary["evaluations"]
    assert sum(len(e["metrics"]) for e in evaluations) == summary["metric_fields"]
    assert sum(m["value"] is None for e in evaluations for m in e["metrics"]) == summary["unknown_metric_fields"]
    assert sum(bool(r["evaluations"]) for r in catalog["records"]) == summary["sources_with_supported_metrics"]
    assert len(summary["descriptive_groups"]) == 2
    audited = next(r for r in catalog["records"] if r["source"]["byte_sha256"] == api["AUDITED_BATCH_SHA"])
    for group in summary["descriptive_groups"]:
        actual = {e["policy"]: next(m["value"] for m in e["metrics"] if m["name"] == "total_net_return_reported")
                  for e in audited["evaluations"] if e["scenario"] == group["scenario"]}
        assert group["policies"] == [{"policy": policy, "value": value} for policy, value in
                                     sorted(actual.items(), key=lambda pair: pair[1], reverse=True)]
        assert group["policies"][0] == {"policy": "NoTrade", "value": 0.0}
    frozen_contract = root / "configs/genetic_primus_future_ema_viability_v1.json"
    assert sha256(frozen_contract.read_bytes()).hexdigest() == "11b83d2e3ecbdda572307dd902b73c9c2d8ae108499be85e22b988f193b227b6"
    print(f"artifact verification: {len(inventory)} originals unchanged; envelopes/receipt/rankings verified")
    print("Frozen future EMA contract unchanged; no training, collector, network or runtime.")


if __name__ == "__main__":
    main()
