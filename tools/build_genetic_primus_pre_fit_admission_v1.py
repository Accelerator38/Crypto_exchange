"""Read verified synthetic origin metadata, publish denial plans; never fit."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/pre_fit_admission_v1_20261007"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/causal_training_labels_v2_20261007"
HELPER = ROOT / "src/exia/genetic_primus/pre_fit_origin_admission_v1.py"
GUARD = ROOT / "src/exia/genetic_primus/causal_origin_payload_v1.py"
CAP = 1_048_576


def read_bounded(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("metadata path scope")
    before = path.stat()
    if before.st_size > CAP:
        raise ValueError("metadata file size cap")
    with path.open("rb") as stream:
        data = stream.read(CAP + 1)
    after = path.stat()
    if (len(data) != before.st_size or len(data) > CAP
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError("metadata changed during read")
    return data


def canonical_bytes(guard, value):
    return guard["canonical_bytes"](value) + b"\n"


def prepare():
    sys.dont_write_bytecode = True
    api, guard = runpy.run_path(str(HELPER)), runpy.run_path(str(GUARD))
    bank = {}

    def capture(path):
        data = read_bounded(path)
        bank[path.relative_to(ROOT).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    for path in (HELPER, GUARD, Path(__file__).resolve(), ROOT / "tools/check_genetic_primus_pre_fit_admission_v1.py"):
        capture(path)
    receipt = guard["strict_json"](capture(PREVIOUS / "completion_receipt.json").decode("utf-8"))
    if (set(receipt["output_byte_sha256"]) != {"synthetic_candles.json", "training_snapshot.json", "deploy_snapshot.json", "origin_inputs.json", "summary.json"}
            or receipt["market_research_launch_ready"] is not False or receipt["real_money_ready"] is not False
            or receipt["safety"] != {k: False for k in guard["SAFETY_KEYS"]}):
        raise ValueError("predecessor authority/scope")
    objects = {}
    for name, expected in receipt["output_byte_sha256"].items():
        data = capture(PREVIOUS / name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("predecessor output changed")
        objects[name] = guard["strict_json"](data.decode("utf-8"))
    for relative, expected in receipt["inputs"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("input path escape")
        data = capture(ROOT / relative)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("predecessor source/metadata changed")
    proposal = guard["strict_json"](capture(ROOT / "configs/genetic_primus_forward_research_proposal_v1.json").decode("utf-8"))
    origin = objects["origin_inputs.json"]
    training = origin["training_inputs"]["training"]
    if (training["binding"]["data_content_sha256"] != api["sha"](objects["training_snapshot.json"])
            or training["binding"]["data_file_sha256"] != receipt["output_byte_sha256"]["training_snapshot.json"]
            or origin["deploy_source_sha256"] != api["sha"](objects["deploy_snapshot.json"])
            or origin["deploy_source_file_sha256"] != receipt["output_byte_sha256"]["deploy_snapshot.json"]):
        raise ValueError("independent source snapshot binding")
    result = api["admit"](origin, proposal, guard)
    if origin["payload_sha256"] != receipt["origin_input_semantic_sha256"]:
        raise ValueError("origin semantic binding")
    plans = result["plans"]
    summary = {"schema": "exia.genetic_primus.pre_fit_admission_summary/1", "variants": len(plans),
               "fit_denied_variants": sum(p["fit_allowed"] is False for p in plans),
               "opportunities": plans[0]["opportunities"], "minimum_opportunities": plans[0]["minimum_opportunities"],
               "minimum_weight_ess": plans[0]["minimum_weight_ess"],
               "weight_ess_by_variant": {p["variant"]: p["kish_weight_ess"] for p in plans},
               "negative_outcome_weight_by_variant": {p["variant"]: p["weight_bank"]["negative_outcome_weight"] for p in plans},
               "scaler_input_scope": "opportunity_filtered_not_all_training_rows", "minimum_trade_count_forcing": False,
               "declared_timing_gate_passed": result["timing"]["declared_timing_gate_passed"],
               "new_model_fits": 0, "new_market_strategy_evaluations": 0,
               "market_data_read": False, "market_research_launch_ready": False, "real_money_ready": False,
               "safety": result["safety"],
               "next_action": "explicit_research_timing_execution_and_data_contract_before_any_fit_not_more_implicit_defaults"}
    return api, guard, origin, proposal, result, summary, bank


def verify_inputs(bank):
    for relative, expected in bank.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("input path escape")
        data = read_bounded(ROOT / relative)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("input changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    api, guard, _, _, result, summary, bank = prepare()
    verify_inputs(bank)
    if not args.verify_only:
        OUTPUT.mkdir(exist_ok=False)
        hashes = {}
        for name, value in (("admission.json", result), ("summary.json", summary)):
            data = canonical_bytes(guard, value)
            with (OUTPUT / name).open("xb") as stream:
                stream.write(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        verify_inputs(bank)
        receipt = {"schema": "exia.genetic_primus.pre_fit_admission_receipt/1", "inputs": bank,
                   "inputs_verified_unchanged_before_and_after": True, "output_byte_sha256": hashes,
                   "admission_semantic_sha256": result["payload_sha256"],
                   "receipt_kind": "synthetic_pre_fit_denial_plan_not_fit_or_market_evidence",
                   "market_research_launch_ready": False, "real_money_ready": False, "safety": result["safety"]}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(canonical_bytes(guard, receipt))
    print("pre-fit: 2 variants denied; 7 opportunities < 64; weight ESS < 40; unregistered timing; fits=0")


if __name__ == "__main__":
    main()
