"""Append the disclosed baseline economics audit to the known-attempt chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v3_20261001/ledger.json"
PREVIOUS_SHA = "eab2110ec08ff5c7b6d15b72894507abce17fa75737594385ac54806363a5f10"
CONFIG = ROOT / "configs/genetic_primus_baseline_economics_audit_v1.json"
RESULT = ROOT / "Reports/Exia/Genetic_Primus/baseline_economics_audit_v1_20261002/result.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v4_20261002/ledger.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    if sha(PREVIOUS) != PREVIOUS_SHA:
        raise ValueError("previous research ledger bytes changed")
    prior = json.loads(PREVIOUS.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    if (result["objective_id"] != config["objective_id"]
            or result["config_sha256"] != sha(CONFIG)
            or result["source_ablation_result_sha256"] != config["source_ablation_result_sha256"]
            or result["source_manifest_sha256"] != config["source_manifest_sha256"]
            or set(result["policies"]) != set(config["policies"])
            or result["promotion_authority"]
            or any(result["safety"].values())):
        raise ValueError("audit lineage/policy count changed")
    for policy in result["policies"].values():
        if set(policy["evaluations"]) != set(config["costs_round_trip_bps"]):
            raise ValueError("fixed cost cases changed")
        for evaluation in policy["evaluations"].values():
            economics = evaluation["economics"]
            if (not evaluation["archived_metrics_replayed_exactly"]
                    or abs(economics["gross_pct_initial_capital"]
                           - economics["fees_pct_initial_capital"]
                           - economics["net_pct_initial_capital"]) > 1e-8):
                raise ValueError("baseline economics audit does not reconcile")
    if (result["data_integrity"]["missing_symbol_bars"] != 0
            or result["data_integrity"]["duplicates"] != 0
            or result["source_lineage"]["verified_recorded_raw_pages"] != 336):
        raise ValueError("source integrity audit changed")
    gates = dict(prior["stage_gates"])
    if any(gates.values()):
        raise ValueError("descriptive audit cannot open research gates")
    return {
        "schema_version": "exia.genetic_primus.research_ledger/4",
        "previous_ledger_path": PREVIOUS.relative_to(ROOT).as_posix(),
        "previous_ledger_sha256": PREVIOUS_SHA,
        "new_attempts": [{
            "objective_id": config["objective_id"],
            "config_path": CONFIG.relative_to(ROOT).as_posix(),
            "config_sha256": sha(CONFIG),
            "result_path": RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": sha(RESULT),
            "evidence_class": result["evidence_class"],
            "candidate_selected": False,
            "outcome": "historical_fixed_policy_descriptive_replay",
        }],
        "verified_recomputations": {
            "five_fixed_policy_two_cost_replay": True,
            "gross_fees_net_cash_reconciliation": True,
            "full8_4h_snapshot_grid": True,
            "recorded_source_page_hashes_and_product_metadata": True,
        },
        "stage_gates": gates,
        "unknown_historical_adaptive_decisions": prior["unknown_historical_adaptive_decisions"],
        "global_dsr_pbo_calibrated": prior["global_dsr_pbo_calibrated"],
        "promotion_authority": False,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError("immutable research ledger v4 exists")
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DESCRIPTIVE_AUDIT_APPENDED",
                      "attempts": len(result["new_attempts"]),
                      "promotion": result["promotion_authority"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
