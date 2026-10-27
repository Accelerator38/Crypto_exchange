"""Repair v1 shared-source identity via independent synthetic train/deploy sources."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/causal_training_labels_v2_20261007"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/causal_training_labels_v1_20261007"
HELPER = ROOT / "src/exia/genetic_primus/causal_training_labels_v2.py"


def prepare():
    sys.dont_write_bytecode = True
    old = runpy.run_path(str(ROOT / "tools/build_genetic_primus_causal_training_labels_v1.py"))
    labels, features, guard, case, _, _, _, bank = old["prepare"]()
    api = runpy.run_path(str(HELPER))
    bank = dict(bank)

    def capture(path):
        data = old["read_bounded"](path)
        bank[path.relative_to(ROOT).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    for path in (HELPER, Path(__file__).resolve(), ROOT / "tools/check_genetic_primus_causal_training_labels_v2.py"):
        capture(path)
    previous = guard["strict_json"](capture(PREVIOUS / "completion_receipt.json").decode("utf-8"))
    if (previous["market_research_launch_ready"] is not False or previous["real_money_ready"] is not False
            or previous["safety"] != {k: False for k in guard["SAFETY_KEYS"]}):
        raise ValueError("predecessor authority")
    for name, expected in previous["output_byte_sha256"].items():
        if name not in {"synthetic_candles.json", "consumed_snapshot.json", "origin_inputs.json", "label_definition.json", "summary.json"}:
            raise ValueError("predecessor output scope")
        if hashlib.sha256(capture(PREVIOUS / name)).hexdigest() != expected:
            raise ValueError("predecessor artifact changed")
    if previous["inputs"] != {k: bank[k] for k in previous["inputs"]}:
        raise ValueError("predecessor source bank changed")
    train, deploy, result = api["build_origin_inputs"](case["candles"], case["request"], labels, features, guard)
    info = result["training_inputs"]
    summary = {"schema": "exia.genetic_primus.causal_training_label_summary/2",
               "supersedes": "v1_shared_training_deploy_snapshot_identity_defect",
               "training_source_candles": len(train["source"]), "deploy_source_candles": len(deploy["source"]),
               "completed_training_rows": info["training_guard"]["completed_rows"],
               "negative_labels": info["training_guard"]["negative_outcomes"],
               "positive_labels": info["training_guard"]["positive_outcomes"],
               "proposal_opportunities": info["scaler_candidate_input"]["opportunities"],
               "negative_opportunities": info["scaler_candidate_input"]["negative_opportunities"],
               "independent_training_deploy_bindings": True, "shared_feature_definition": True,
               "minimum_opportunities_64_passed": False, "ess_gate_verified": False, "actual_scaler_fitted": False,
               "model_ready": False, "new_model_fits": 0, "new_market_strategy_evaluations": 0,
               "market_data_read": False, "market_research_launch_ready": False, "real_money_ready": False,
               "safety": result["safety"], "next_action": "pre_fit_origin_admission_with_distinct_scaler_scope_and_sample_gates_no_training"}
    return api, labels, features, guard, old, case, train, deploy, result, summary, bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    api, _, _, _, old, case, train, deploy, result, summary, bank = prepare()
    old["verify_inputs"](bank)
    if not args.verify_only:
        OUTPUT.mkdir(exist_ok=False)
        hashes = {}
        for name, value in (("synthetic_candles.json", case), ("training_snapshot.json", train),
                            ("deploy_snapshot.json", deploy), ("origin_inputs.json", result), ("summary.json", summary)):
            data = api["canonical_bytes"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        old["verify_inputs"](bank)
        receipt = {"schema": "exia.genetic_primus.causal_training_label_receipt/2", "inputs": bank,
                   "inputs_verified_unchanged_before_and_after": True, "output_byte_sha256": hashes,
                   "origin_input_semantic_sha256": result["payload_sha256"],
                   "receipt_kind": "synthetic_independent_source_bindings_not_fitting_or_market_evidence",
                   "market_research_launch_ready": False, "real_money_ready": False, "safety": result["safety"]}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical_bytes"](receipt) + b"\n")
    print("v2 independent sources: 8 labels, 7 opportunities incl. 2 losses; training needs no deploy source; fits=0")


if __name__ == "__main__":
    main()
