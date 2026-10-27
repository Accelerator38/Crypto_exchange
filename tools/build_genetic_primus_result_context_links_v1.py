"""Finite static sidecar joins over a frozen catalog. No evaluation/training imports."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
CATALOG_DIR = BASE + "result_catalog_v1_20261005_r2/"
OUTPUT = ROOT / (BASE + "result_context_links_v1_20261005")
INVENTORY_PATH = BASE + "development_review_v1_20261005/evidence_index_v2.json"
INVENTORY_SHA = "5e6192ccef065c9e0f488395bfd9456d160c071d706a9985447149e172d89bed"
CATALOG_SHA = "cafb0375dab4e0e608004c3fe9c216302ebc30331a745a0885d2c285ddbdb08b"
RECEIPT_SHA = "631f02aa27f9e0d3bc9623e1823c5bfb6f8bd44989f665c589a26752689b2104"
FUTURE_PATH = "configs/genetic_primus_future_ema_viability_v1.json"
FUTURE_SHA = "11b83d2e3ecbdda572307dd902b73c9c2d8ae108499be85e22b988f193b227b6"


def prepare():
    module_path = ROOT / "src/exia/genetic_primus/result_context_links_v1.py"
    module_sha = sha256(module_path.read_bytes()).hexdigest()
    builder_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    api = runpy.run_path(str(module_path))
    old_api = runpy.run_path(str(ROOT / "src/exia/genetic_primus/result_catalog_v1.py"))
    inputs = {}

    def read(relative, expected=None):
        path = (ROOT / relative).resolve(strict=True)
        allowed = ((relative.startswith(BASE) and path.is_relative_to((ROOT / BASE).resolve()))
                   or (relative.startswith("configs/genetic_primus_") and path.parent == (ROOT / "configs").resolve()))
        api["require"](allowed and path.suffix == ".json", "sidecar outside allowed scope")
        api["require"](path.stat().st_size <= (40_000_000 if relative == INVENTORY_PATH else 8_000_000), "metadata size limit")
        raw = path.read_bytes()
        hashed = sha256(raw).hexdigest()
        if expected:
            api["require"](hashed == expected, "input byte hash mismatch: " + relative)
        if relative in inputs:
            api["require"](inputs[relative]["byte_sha256"] == hashed, "input changed while reading")
        inputs[relative] = {"path": relative, "byte_sha256": hashed, "size_bytes": len(raw)}
        return raw

    inventory = json.loads(read(INVENTORY_PATH, INVENTORY_SHA))["records"]
    api["require"](len(inventory) == 1784 and len({r['path'] for r in inventory}) == 1784, "inventory coverage")
    by_path = {r["path"]: r for r in inventory}
    receipt = json.loads(read(CATALOG_DIR + "completion_receipt.json", RECEIPT_SHA))
    catalog = json.loads(read(CATALOG_DIR + "catalog.json", CATALOG_SHA))
    api["require"](receipt["catalog_byte_sha256"] == CATALOG_SHA and receipt["inventory_byte_sha256"] == INVENTORY_SHA,
                   "base receipt binding mismatch")
    envelopes = {e["source"]["path"]: e for e in catalog["records"]}
    api["require"](len(envelopes) == 1784 and set(envelopes) == set(by_path), "catalog coverage mismatch")
    for path, envelope in envelopes.items():
        old_api["validate_serialized"](envelope)
        source, original = envelope["source"], by_path[path]
        api["require"](source["byte_sha256"] == original["sha256"] and source["size_bytes"] == original["size_bytes"], "base source mismatch")

    def doc(relative, expected=None):
        original = by_path.get(relative)
        if original:
            if expected:
                api["require"](expected == original["sha256"], "sidecar claim disagrees with inventory")
            expected = original["sha256"]
        api["require"](expected is not None, "unpinned sidecar prohibited")
        return api["document"](relative, read(relative, expected), expected)

    families = []
    reversal_dir = BASE + "reversal_calibration_pilot_v1_20261005/"
    final = doc(reversal_dir + "result.json")
    registration = doc(reversal_dir + "registration.json")
    reg = registration.payload
    pilot = doc(reg["pilot_draft_path"], reg["pilot_draft_sha256"])
    draft = doc(reg["model_draft_path"], reg["model_draft_sha256"])
    steps = [doc(reversal_dir + f"origin_{i:02d}.json") for i in range(1, 21)]
    families.append(api["reversal_family"](final, registration, pilot, draft, steps, envelopes))
    linear = doc(BASE + "rolling_linear_matrix_v1_20261002/result.json")
    linear_config = doc("configs/genetic_primus_rolling_linear_matrix_v1.json", linear.payload["config_sha256"])
    contract = doc(linear_config.payload["source_contract_path"], linear.payload["source_contract_sha256"])
    families.append(api["linear_family"](linear, linear_config, contract, envelopes[linear.path]))
    baseline = doc(BASE + "baseline_economics_audit_v1_20261002/result.json")
    baseline_config = doc("configs/genetic_primus_baseline_economics_audit_v1.json", baseline.payload["config_sha256"])
    manifest = doc(baseline_config.payload["source_manifest_path"], baseline.payload["source_manifest_sha256"])
    families.append(api["baseline_family"](baseline, baseline_config, manifest, envelopes[baseline.path]))
    for family, documents in ((families[0], (registration, pilot)), (families[1], (linear, linear_config))):
        for source in documents:
            if "source_manifest_path" in source.payload:
                api["require"](source.payload["source_manifest_path"] == manifest.path, "manifest path mismatch")
            family["bindings"].append(api["bind"](source, "source_manifest_sha256", manifest))
    # Byte-only check: do not open/decode future market data or evaluate its interval.
    read(FUTURE_PATH, FUTURE_SHA)
    links = [link for family in families for link in family["evaluation_links"]]
    base_evaluations = [e for envelope in envelopes.values() for e in envelope["evaluations"]]
    base_contexts = Counter({name: sum(e["context"].get(name) is None for e in base_evaluations)
                             for name in api["UNITS"] if name in base_evaluations[0]["context"]})
    linked_values = {(l["source_path"], l["policy"], l["scenario"]): {f["name"]: f["value"] for f in l["fields"]}
                     for l in links}
    after = {}
    for name in base_contexts:
        after[name] = sum(linked_values.get((path, e["policy"], e["scenario"]), {}).get(name,
                            e["context"].get(name)) is None for path, envelope in envelopes.items() for e in envelope["evaluations"])
    joined_sources = {link["source_path"] for link in links}
    output = {"schema": api["SCHEMA"], "base_catalog_byte_sha256": CATALOG_SHA,
              "inventory_byte_sha256": INVENTORY_SHA, "families": families,
              "sources": [{"path": path, "byte_sha256": e["source"]["byte_sha256"],
                           "base_envelope_sha256": e["envelope_sha256"],
                           "join_status": "linked" if path in joined_sources else "not_linked_no_context_inferred"}
                          for path, e in envelopes.items()],
              "credentials_allowed": False, "orders_enabled": False, "paper": False, "live": False,
              "promotion_authority": False, "execution_allowed": False, "cross_source_comparison_authority": False}
    summary = {"schema": "exia.genetic_primus.result_context_summary/1", "sources": 1784,
               "linked_sources": len(joined_sources), "base_evaluations": len(base_evaluations),
               "linked_evaluations": len(links), "families": {f["family"]: len(f["evaluation_links"]) for f in families},
               "unknown_context_before": dict(base_contexts), "unknown_context_after": after,
               "not_applicable_context_evaluations": {
                   name: sum(f["unknown_reason"] == "not_applicable_fixed_baseline" for link in links for f in link["fields"] if f["name"] == name)
                   for name in ("retrain_cadence_ms",)},
               "verified_model_payload_digests": sum(len(f["models"]) for f in families),
               "reversal_explicit_decision_model_hashes_matched": families[0]["decision_model_hashes_matched"],
               "linear_model_batch_hashes_matched": len(families[1]["models"]),
               "raw_market_data_hashes_recomputed": False, "comparison_authority": False,
               "promotion_authority": False, "execution_allowed": False}
    # Verify the entire original inventory, not only the linked subset.
    def verify_unchanged():
        for record in inventory:
            raw = read(record["path"], record["sha256"])
            api["require"](len(raw) == record["size_bytes"], "original size mismatch")
        for record in list(inputs.values()):
            read(record["path"], record["byte_sha256"])
        api["require"](sha256(module_path.read_bytes()).hexdigest() == module_sha, "link module changed")
        api["require"](sha256(Path(__file__).read_bytes()).hexdigest() == builder_sha, "builder changed")
    return api, output, summary, inputs, verify_unchanged, module_sha, builder_sha


def main():
    verify_only = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not verify_only:
        raise ValueError("only --verify-only is supported")
    if not verify_only and OUTPUT.exists():
        raise FileExistsError("output exists; overwrite prohibited")
    api, output, summary, inputs, verify, module_sha, builder_sha = prepare()
    verify()
    if not verify_only:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("context_links.json", output), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        receipt = {"schema": "exia.genetic_primus.result_context_completion/1", "input_files": list(inputs.values()),
                   "inputs_verified_unchanged_before_and_after": True, "output_byte_sha256": hashes,
                   "module_byte_sha256": module_sha, "builder_byte_sha256": builder_sha,
                   "credentials_allowed": False, "orders_enabled": False, "paper": False, "live": False,
                   "execution_allowed": False, "promotion_authority": False}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
    print(f"{'preflight' if verify_only else 'context links'}: {summary['linked_sources']} sources; {summary['linked_evaluations']}/90 evaluations; {summary['verified_model_payload_digests']} model digests")
    print("1784 originals/base catalog/future contract unchanged; no evaluation, training or runtime.")


if __name__ == "__main__":
    main()
