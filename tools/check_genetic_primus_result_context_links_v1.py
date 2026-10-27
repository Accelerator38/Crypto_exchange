"""Deterministic pure/metadata regressions; no pytest, collector, market data or fit."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    api = runpy.run_path(str(root / "src/exia/genetic_primus/result_context_links_v1.py"))
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

    def doc(path, payload):
        raw = api["canonical"](payload)
        return api["document"](path, raw, sha256(raw).hexdigest())

    check(api["utc_ms"]("2026-06-01T00:00:00Z") == 1780272000000)
    check(api["utc_ms"]("2026-06-01T00:00:00+00:00") == 1780272000000)
    check(rejects(lambda: api["utc_ms"]("2026-06-01T00:00:00")))
    check(rejects(lambda: api["utc_ms"]("2026-06-01T00:00:00+03:00")))
    for invalid in (True, -1, 0.5, "0"):
        check(rejects(lambda: api["integer"](invalid)))
    check(api["integer"](0) == 0)
    for invalid in (True, float("nan"), float("inf"), -1, "16", 10 ** 1000):
        check(rejects(lambda: api["number"](invalid)))
    check(api["number"](0) == 0)
    raw = b'{"a":1,"a":2}'
    check(rejects(lambda: api["document"]("synthetic.json", raw, sha256(raw).hexdigest())))
    check(rejects(lambda: api["document"]("synthetic.json", b'{}', "a" * 64)))
    check(rejects(lambda: api["document"]("synthetic.json", b'[]', sha256(b'[]').hexdigest())))
    check(rejects(lambda: doc("synthetic.json", {"schema": "a", "schema_version": "b"})))
    target_payload = {"schema": "synthetic/1", "start_ms": 0}
    target_payload["registration_sha256"] = api["digest"](target_payload)
    target = doc("synthetic/registration.json", target_payload)
    source = doc("synthetic/result.json", {"registration_sha256": target_payload["registration_sha256"]})
    binding = api["bind"](source, "registration_sha256", target, semantic_field="registration_sha256")
    check(binding["identity_kind"] == "semantic")
    check(binding["identity_sha256"] != binding["target_byte_sha256"])
    check(rejects(lambda: api["bind"](source, "registration_sha256", target)))
    changed = deepcopy(target_payload)
    changed["start_ms"] = 1
    check(rejects(lambda: api["semantic_identity"](changed, "registration_sha256")))
    wrong_source = doc("synthetic/wrong.json", {"registration_sha256": target.byte_sha256})
    check(rejects(lambda: api["bind"](wrong_source, "registration_sha256", target, semantic_field="registration_sha256")))
    empty_fields = api["complete"]([])
    check(all(f["value"] is None and f["unknown_reason"] for f in empty_fields))
    field = api["resolved"](target, "delay_bars", 0, "synthetic_zero")
    check(next(f for f in api["complete"]([field]) if f["name"] == "delay_bars")["value"] == 0)
    check(rejects(lambda: api["complete"]([field, field])))
    env = {"source": {"path": source.path, "byte_sha256": source.byte_sha256, "size_bytes": source.size_bytes},
           "envelope_sha256": "b" * 64, "evaluations": []}
    e = {"policy": "NoTrade", "scenario": "normal", "metrics": [], "context": {"delay_bars": 1}}
    env["evaluations"].append(e)
    check(rejects(lambda: api["evaluation_link"](env, e, [field])))
    bad_env = deepcopy(env)
    bad_env["source"]["byte_sha256"] = "c" * 64
    check(rejects(lambda: api["catalog_document"](source, bad_env)))
    fixed = api["complete"]([], fixed_baseline=True)
    check(next(f for f in fixed if f["name"] == "retrain_cadence_ms")["unknown_reason"] == "not_applicable_fixed_baseline")

    # End-to-end metadata-only resolution of the already-revealed artifact families.
    builder = runpy.run_path(str(root / "tools/build_genetic_primus_result_context_links_v1.py"))
    _, output, summary, inputs, _, _, _ = builder["prepare"]()
    check(summary["linked_sources"] == 23 and summary["linked_evaluations"] == 67)
    check(summary["verified_model_payload_digests"] == 100)
    check(summary["reversal_explicit_decision_model_hashes_matched"] == 160)
    check(summary["linear_model_batch_hashes_matched"] == 80)
    check(summary["unknown_context_before"]["interval_start_ms"] == 40)
    check(summary["unknown_context_after"]["interval_start_ms"] == 23)
    check(summary["unknown_context_after"]["deployment_horizon_ms"] == 33)
    check(summary["unknown_context_after"]["round_trip_cost_bps"] == 23)
    check(all(output[flag] is False for flag in ("credentials_allowed", "orders_enabled", "paper", "live",
                                                "execution_allowed", "promotion_authority", "cross_source_comparison_authority")))
    families = {f["family"]: f for f in output["families"]}
    final = next(l for l in families["reversal_calibration"]["evaluation_links"]
                 if l["source_path"].endswith("result.json") and l["scenario"] == "stress")
    values = {f["name"]: f["value"] for f in final["fields"]}
    check(values["interval_start_ms"] == 1780272000000 and values["interval_end_exclusive_ms"] == 1785456000000)
    check(values["deployment_horizon_ms"] == values["retrain_cadence_ms"] == 3 * api["DAY_MS"])
    check(values["round_trip_cost_bps"] == 40 and values["delay_bars"] == 1 and values["purge_bars"] == 7)
    check(values["bar_ms"] is None)  # Do not guess bar size from cutoff differences.
    bank = next(c for c in families["rolling_linear_matrix"]["identity_claims"] if c["field"] == "result.train_bank_sha256")
    check(bank["verification"] == "reported_only")
    ema = next(l for l in families["rolling_linear_matrix"]["evaluation_links"] if l["policy"] == "EMA_training_reference")
    check(next(f for f in ema["fields"] if f["name"] == "retrain_cadence_ms")["unknown_reason"] == "not_applicable_fixed_baseline")
    evidence_valid = all(inputs[proof["path"]]["byte_sha256"] == proof["byte_sha256"]
                         for family in output["families"] for link in family["evaluation_links"]
                         for field in link["fields"] for proof in field["evidence"])
    check(evidence_valid)
    check(all(link["comparison_authority"] is False and link["independent_validation_verified"] is False
              for family in output["families"] for link in family["evaluation_links"]))
    # Failures must arise from the intended guard, not a different broken fixture.
    import json
    raw_catalog = json.loads((root / builder["CATALOG_DIR"] / "catalog.json").read_bytes())
    envelopes = {e["source"]["path"]: e for e in raw_catalog["records"]}
    directory = builder["BASE"] + "reversal_calibration_pilot_v1_20261005/"

    def original(path):
        return api["document"](path, (root / path).read_bytes(), inputs[path]["byte_sha256"])

    final_doc = original(directory + "result.json")
    registration = original(directory + "registration.json")
    pilot = original(registration.payload["pilot_draft_path"])
    draft = original(registration.payload["model_draft_path"])
    steps = [original(directory + f"origin_{index:02d}.json") for index in range(1, 21)]

    def mutated_step_rejected(change, expected_reason):
        payload = deepcopy(steps[0].payload)
        change(payload)
        payload["step_sha256"] = api["digest"]({k: v for k, v in payload.items() if k != "step_sha256"})
        changed_doc = doc(steps[0].path, payload)
        changed_env = deepcopy(envelopes[steps[0].path])
        changed_env["source"]["byte_sha256"] = changed_doc.byte_sha256
        changed_env["source"]["size_bytes"] = changed_doc.size_bytes
        changed_envelopes = dict(envelopes)
        changed_envelopes[steps[0].path] = changed_env
        try:
            api["reversal_family"](final_doc, registration, pilot, draft, [changed_doc, *steps[1:]], changed_envelopes)
        except ValueError as error:
            return str(error) == expected_reason
        return False

    check(mutated_step_rejected(lambda p: p.update(previous_step_sha256="a" * 64), "broken step/state chain"))
    check(mutated_step_rejected(lambda p: p["result"]["model"].update(train_cutoff_exclusive_ms=0), "semantic hash mismatch"))
    check(mutated_step_rejected(lambda p: p["result"]["decisions"][0].update(model_sha256="a" * 64), "decision/model lineage mismatch"))
    check(mutated_step_rejected(lambda p: p["result"]["scenarios"]["stress"].update(delay_bars=0), "origin scenario mismatch"))
    def changed_cutoff(payload):
        model = payload["result"]["model"]
        model["train_cutoff_exclusive_ms"] -= 1
        model["model_sha256"] = api["digest"]({k: v for k, v in model.items() if k != "model_sha256"})
    check(mutated_step_rejected(changed_cutoff, "model cutoff mismatch"))
    print(f"context link regressions: {checks} passed; metadata only, no training/runtime")


if __name__ == "__main__":
    main()
