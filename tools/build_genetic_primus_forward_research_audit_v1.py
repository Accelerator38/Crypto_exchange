"""Publish only a draft contract and selected-source static audit; no market run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/forward_research_audit_v1_20261006"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/dependency_readiness_v1_20261006"
HELPER = ROOT / "src/exia/genetic_primus/forward_research_audit_v1.py"
CONFIG = ROOT / "configs/genetic_primus_forward_research_proposal_v1.json"
LOCKED = (
    "configs/genetic_primus_boundary_engineering_v1.json",
    "configs/genetic_primus_prospective_3d_v1.json",
    "configs/genetic_primus_future_ema_viability_v1.json",
    "configs/genetic_primus_rolling_linear_semantics_v2.json",
)
MAX_INPUT_BYTES = 1_048_576


def read_bounded(path):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(ROOT) or not resolved.is_file():
        raise ValueError("input outside explicit repository scope")
    before = resolved.stat()
    if before.st_size > MAX_INPUT_BYTES:
        raise ValueError("metadata/source file size cap")
    with resolved.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    after = resolved.stat()
    if (len(data) > MAX_INPUT_BYTES or len(data) != before.st_size
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError("input changed during read")
    return data


def prepare():
    sys.dont_write_bytecode = True
    api = runpy.run_path(str(HELPER))  # Only our pure stdlib metadata helper.
    bank = {}

    def capture(path):
        relative = path.relative_to(ROOT).as_posix()
        data = read_bounded(path)
        bank[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data.decode("utf-8-sig")

    for path in (HELPER, Path(__file__).resolve(),
                 ROOT / "tools/check_genetic_primus_forward_research_audit_v1.py"):
        capture(path)
    proposal = api["strict_json"](capture(CONFIG))
    sources = {name: capture(ROOT / "src/exia/genetic_primus" / name)
               for name in api["SOURCE_NAMES"]}
    for relative in LOCKED:
        capture(ROOT / relative)  # Hash only: no sealed data/results read.
    previous_receipt = api["strict_json"](capture(PREVIOUS / "completion_receipt.json"))
    previous = {}
    for name in ("summary.json", "readiness.json"):
        previous[name] = api["strict_json"](capture(PREVIOUS / name))
        relative = (PREVIOUS / name).relative_to(ROOT).as_posix()
        if previous_receipt["output_byte_sha256"].get(name) != bank[relative]["sha256"]:
            raise ValueError("predecessor output binding changed")
    for item in (previous_receipt, *previous.values()):
        if item.get("market_research_launch_ready") is not False or item.get("real_money_ready") is not False:
            raise ValueError("predecessor readiness must remain closed")
        if item.get("safety") != proposal["safety"]:
            raise ValueError("predecessor authority mismatch")
    result = api["audit"](proposal, sources)
    result["source_byte_sha256"] = {
        name: bank["src/exia/genetic_primus/" + name]["sha256"] for name in sources}
    result["predecessor_receipt_byte_sha256"] = bank[
        (PREVIOUS / "completion_receipt.json").relative_to(ROOT).as_posix()]["sha256"]
    result["predecessor_full_dependency_graph_verified"] = False
    result["predecessor_scope"] = "summary_readiness_receipt_binding_not_full_dependency_reaudit"
    result["locked_contracts_byte_sha256"] = {path: bank[path]["sha256"] for path in LOCKED}
    summary = {
        "schema": "exia.genetic_primus.forward_research_audit_summary/1",
        "proposal_status": proposal["status"], "source_files_audited": len(sources),
        "static_anchors_observed": result["observation_count"],
        "locked_contracts_verified_unchanged": len(LOCKED),
        "planned_candidate_pipelines": proposal["testing_budget"]["planned_candidate_pipelines"],
        "authorized_market_runs": 0, "new_model_fits": 0,
        "new_market_strategy_evaluations": 0, "market_data_read": False,
        "audited_code_executed": False, "market_research_launch_ready": False,
        "real_money_ready": False, "blocking_requirements": proposal["readiness"]["blocking_requirements"],
        "safety": proposal["safety"], "next_action": result["next_action"],
    }
    return api, proposal, result, summary, bank


def verify_inputs(bank):
    for relative, item in bank.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("input path escape")
        data = read_bounded(ROOT / relative)
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("input changed: " + relative)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    _, proposal, result, summary, bank = prepare()
    verify_inputs(bank)
    if not args.verify_only:
        OUTPUT.mkdir(exist_ok=False)
        hashes = {}
        for name, value in (("proposal.json", proposal), ("audit.json", result), ("summary.json", summary)):
            data = canonical_bytes(value)
            with (OUTPUT / name).open("xb") as stream:
                stream.write(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        verify_inputs(bank)
        receipt = {"schema": "exia.genetic_primus.forward_research_audit_receipt/1",
                   "receipt_kind": "draft_and_static_source_observations_not_market_evidence",
                   "inputs": bank, "inputs_verified_unchanged_before_and_after": True,
                   "output_byte_sha256": hashes, "proposal_semantic_sha256": result["proposal_semantic_sha256"],
                   "market_research_launch_ready": False, "real_money_ready": False,
                   "safety": proposal["safety"]}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(canonical_bytes(receipt))
    print("static audit: 5 sources, 16 AST anchors; 4 contracts unchanged; draft only; fits=0, network/runtime=off")


if __name__ == "__main__":
    main()
