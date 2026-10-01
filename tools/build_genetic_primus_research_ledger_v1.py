"""Freeze the known Genetic Primus research lineage without claiming completeness."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Reports/Exia/Genetic_Primus/research_ledger_v1_20261001/ledger.json"
PREFIX = "Reports/Exia/Genetic_Primus/"
RESULTS = {
    "disclosed_six_policy_ablation": PREFIX + "ablation_continuous_v1_20261001/result.json",
    "strength_efficiency_entry": PREFIX + "entry_hypothesis_v1_20261001/result.json",
    "sparse_catalog_24": PREFIX + "selection_development_v1_20261001/result.json",
    "relative_momentum_top2": PREFIX + "relative_momentum_development_v1_20261001/result.json",
}
METRICS = {
    "ablation_gross_return_pct": "100 * sum(realized_trade_gross_pnl) / starting_equity; realized quantities follow the cost-bearing cash simulation",
    "zero_cost_counterfactual_gross": "return from a separate simulation with round_trip_cost_bps=0; not identical to ablation_gross_return_pct",
    "net_return_pct": "100 * (terminal_equity / starting_equity - 1), after configured round-trip fees",
    "daily_return": "portfolio end-of-day equity / previous end-of-day equity - 1; geometrically compounded for the interval",
    "max_drawdown_pct": "100 * maximum peak-to-trough loss on recorded open/close equity events; intrabar extremes absent",
    "entry_notional_turnover": "sum(abs(quantity * entry_price)) / starting_equity over completed trades; one-way entry notional",
    "train_rank_score": "min(normal, stress) of the 5th percentile of 512 common 5-day block resamples of mean daily return vs zero NoTrade",
    "validation_gate": "positive normal/stress net, absolute and EMA-paired LCB, plus recorded 10% drawdown ceiling; only reached after positive train score",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", path.relative_to(ROOT).as_posix()],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def build() -> dict:
    registry_paths = sorted((ROOT / "configs").glob("genetic_primus_objective_registry_v*.json"))
    if [p.stem.rsplit("v", 1)[-1] for p in registry_paths] != [str(i) for i in range(1, 10)]:
        raise ValueError("expected complete known v1-v9 registry chain")
    registries = []
    objectives = []
    for index, path in enumerate(registry_paths):
        item = read(path)
        if index:
            prior = registry_paths[index - 1]
            if (item.get("previous_registry_path") != prior.relative_to(ROOT).as_posix()
                    or item.get("previous_registry_sha256") != sha(prior)):
                raise ValueError("objective registry ancestry changed")
        for objective in item["objectives"]:
            config = (ROOT / objective["config_path"]).resolve()
            if not config.is_relative_to(ROOT) or sha(config) != objective["config_sha256"]:
                raise ValueError("registered objective config changed")
            objectives.append({**objective, "registry": path.name,
                               "config_tracked_in_git": tracked(config)})
        registries.append({"path": path.relative_to(ROOT).as_posix(),
                           "sha256": sha(path), "tracked_in_git": tracked(path),
                           "entries": len(item["objectives"])})
    if len({x["objective_id"] for x in objectives}) != len(objectives):
        raise ValueError("duplicate objective IDs across registry chain")
    known = []
    for name, relative in RESULTS.items():
        path = ROOT / relative
        result = read(path)
        row = {"name": name, "result_path": relative,
               "result_sha256": sha(path),
               "evidence_class": result["evidence_class"],
               "selection_on_results": False}
        if "candidate_budget" in result:
            if result["search_evaluations"] > result["candidate_budget"]:
                raise ValueError("candidate budget exceeded")
            row["recorded_candidate_evaluations"] = result["search_evaluations"]
        if name == "disclosed_six_policy_ablation":
            row["recorded_policy_evaluations"] = sum(len(x) for x in result["policies"].values())
            row["policies"] = list(result["policies"])
        known.append(row)
    manifest_path = ROOT / (PREFIX + "prospective_3d_v1_202609/data_manifest_origin_09.json")
    manifest = read(manifest_path)
    return {
        "schema_version": "exia.genetic_primus.research_ledger/1",
        "status": "known_trials_snapshot_not_complete_global_trial_count",
        "promotion_authority": False,
        "registry_chain": registries, "registry_objectives": objectives,
        "known_later_attempts": known,
        "unknown_historical_adaptive_decisions": True,
        "global_dsr_pbo_calibrated": False,
        "market_data": {"manifest_path": manifest_path.relative_to(ROOT).as_posix(),
                        "manifest_sha256": sha(manifest_path),
                        "snapshot_sha256": manifest["snapshot_sha256"],
                        "metadata": manifest["metadata"],
                        "venue_product_provenance_verified": False},
        "metric_definitions": METRICS,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError("immutable research ledger snapshot exists")
    ledger = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(ledger, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": ledger["status"], "registries": len(ledger["registry_chain"]),
                      "objective_entries": len(ledger["registry_objectives"]),
                      "later_attempts": len(ledger["known_later_attempts"])}, separators=(",", ":")))


if __name__ == "__main__":
    main()
