"""Normalize a pinned, finite JSON inventory; never scan/load market data or models."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "Reports/Exia/Genetic_Primus"
INVENTORY = BASE / "development_review_v1_20261005/evidence_index_v2.json"
INVENTORY_SHA = "5e6192ccef065c9e0f488395bfd9456d160c071d706a9985447149e172d89bed"
OUTPUT = BASE / "result_catalog_v1_20261005_r2"


def source_bytes(record):
    path = (ROOT / record["path"]).resolve(strict=True)
    if not path.is_relative_to(BASE.resolve()) or path.suffix != ".json":
        raise ValueError("source outside allowed JSON directory")
    if record["size_bytes"] > 8_000_000:
        raise ValueError("source exceeds static metadata limit")
    raw = path.read_bytes()
    if len(raw) != record["size_bytes"] or sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"original source changed: {record['path']}")
    return raw


def write_new(path, value, canonical):
    raw = canonical(value) + b"\n"
    with path.open("xb") as output:
        output.write(raw)
    return sha256(raw).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError("catalog output exists; do not overwrite")
    raw_inventory = INVENTORY.read_bytes()
    if sha256(raw_inventory).hexdigest() != INVENTORY_SHA:
        raise ValueError("frozen inventory hash mismatch")
    records = json.loads(raw_inventory)["records"]
    if len(records) != 1784 or len({r["path"] for r in records}) != len(records):
        raise ValueError("inventory coverage/uniqueness mismatch")
    module_path = ROOT / "src/exia/genetic_primus/result_catalog_v1.py"
    module_sha = sha256(module_path.read_bytes()).hexdigest()
    builder_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    api = runpy.run_path(str(module_path))
    canonical, as_dict = api["canonical_bytes"], api["envelope_dict"]
    envelopes = []
    for record in records:
        raw = source_bytes(record)
        payload = api["load_json_metadata"](raw)
        reported_schema = None
        if isinstance(payload, dict):
            reported_schema = payload.get("schema", payload.get("schema_version"))
            reported_schema = reported_schema if isinstance(reported_schema, str) else None
        source = api["Source"](record["path"], record["sha256"], len(raw), reported_schema)
        envelope = api["adapt"](payload, source)
        api["validate_serialized"](as_dict(envelope))
        envelopes.append(envelope)
    source_bank = [(r["path"], r["sha256"], r["size_bytes"]) for r in records]
    bank_sha = sha256(canonical(source_bank)).hexdigest()
    kinds = Counter(e.artifact_kind for e in envelopes)
    evidence = Counter(e.evidence_category for e in envelopes)
    adapters = Counter(v.adapter for e in envelopes for v in e.evaluations)
    issues = Counter(issue for e in envelopes for issue in e.issues)
    evaluations = [v for e in envelopes for v in e.evaluations]
    metric_count = sum(len(v.metrics) for v in evaluations)
    unknown_count = sum(m.value is None for v in evaluations for m in v.metrics)
    descriptive_groups = []
    for envelope in envelopes:
        if envelope.source.byte_sha256 != api["AUDITED_BATCH_SHA"]:
            continue
        for scenario in ("normal", "stress"):
            candidates = [e for e in envelope.evaluations if e.scenario == scenario]
            if len(candidates) != 5 or any(api["comparison_reasons"](
                    envelope, candidates[0], envelope, candidate, api["MetricName"].NET)
                    for candidate in candidates):
                raise ValueError("audited batch descriptive guard failed")
            ordered = sorted(candidates, key=lambda e: api["metric_value"](e, api["MetricName"].NET), reverse=True)
            descriptive_groups.append({
                "scope": "within_one_pinned_source_descriptive_only_not_promotion",
                "source_byte_sha256": envelope.source.byte_sha256,
                "scenario": scenario,
                "metric": str(api["MetricName"].NET),
                "unit": "fraction",
                "policies": [{"policy": e.policy, "value": api["metric_value"](e, api["MetricName"].NET)} for e in ordered],
                "warning": "Risk/exposure/search budgets are not asserted equal; unknown execution details remain unknown.",
            })
    if len(descriptive_groups) != 2:
        raise ValueError("pinned descriptive groups missing")
    catalog = {
        "schema": "exia.genetic_primus.result_catalog/1",
        "inventory_byte_sha256": INVENTORY_SHA,
        "source_identity_bank_sha256": bank_sha,
        "records": [as_dict(e) for e in envelopes],
        "execution_allowed": False,
        "promotion_authority": False,
    }
    summary = {
        "schema": "exia.genetic_primus.result_catalog_summary/1",
        "sources": len(envelopes), "source_bytes": sum(r["size_bytes"] for r in records),
        "artifact_kinds": dict(sorted(kinds.items())), "evidence_categories": dict(sorted(evidence.items())),
        "adapter_evaluations": dict(sorted(adapters.items())), "issues": dict(sorted(issues.items())),
        "evaluations": len(evaluations), "metric_fields": metric_count, "unknown_metric_fields": unknown_count,
        "sources_with_supported_metrics": sum(bool(e.evaluations) for e in envelopes),
        "context_unknown_evaluations": {
            field: sum(getattr(e.context, field) is None for e in evaluations)
            for field in ("interval_start_ms", "interval_end_exclusive_ms", "deployment_horizon_ms",
                          "retrain_cadence_ms", "round_trip_cost_bps", "delay_bars", "accounting")},
        "metadata_revision": "r2_add_gate_types_closed_origin_and_origin_step_adapters",
        "supersedes_catalog_directory": "result_catalog_v1_20261005",
        "descriptive_groups": descriptive_groups,
        "cross_source_comparison": "blocked_pending_verified_common_measurement_contract",
        "global_trial_registry": "not_created_or_calibrated",
        "independent_or_prospective_evidence_verified": False,
        "execution_allowed": False, "promotion_authority": False,
    }
    # Re-read originals immediately before publication, and again after it.
    for record in records:
        source_bytes(record)
    if sha256(module_path.read_bytes()).hexdigest() != module_sha or sha256(Path(__file__).read_bytes()).hexdigest() != builder_sha:
        raise ValueError("catalog implementation changed during build")
    OUTPUT.mkdir()
    catalog_sha = write_new(OUTPUT / "catalog.json", catalog, canonical)
    summary_sha = write_new(OUTPUT / "summary.json", summary, canonical)
    for record in records:
        source_bytes(record)
    if sha256(INVENTORY.read_bytes()).hexdigest() != INVENTORY_SHA:
        raise ValueError("inventory changed during build")
    receipt = {
        "schema": "exia.genetic_primus.result_catalog_completion/1",
        "originals_verified_unchanged_before_and_after": True,
        "sources": len(records), "inventory_byte_sha256": INVENTORY_SHA,
        "source_identity_bank_sha256": bank_sha,
        "catalog_byte_sha256": catalog_sha, "summary_byte_sha256": summary_sha,
        "adapter_module_byte_sha256": module_sha,
        "builder_byte_sha256": builder_sha,
        "execution_allowed": False, "promotion_authority": False,
    }
    write_new(OUTPUT / "completion_receipt.json", receipt, canonical)
    print(f"catalog: {len(records)} sources; {len(evaluations)} evaluations; {unknown_count}/{metric_count} unknown metrics")
    print("Original hashes verified unchanged before/after; no market data, fitting or runtime.")


if __name__ == "__main__":
    main()
