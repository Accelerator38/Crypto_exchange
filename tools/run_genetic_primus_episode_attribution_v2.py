"""Disclosed, offline GA episode-placement control; no runtime or network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.ablation_v1 import RANDOM_SEEDS, summarize_simulation
from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.episode_attribution_v2 import placed_episode_control
from exia.genetic_primus.measurement_lab_v2 import array_sha, block_bank, paired_lcb
from exia.genetic_primus.prospective_contract_v1 import file_sha256, load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import _frame_sha
from run_genetic_primus_ablation_v1 import _archived_decisions, _market, CONFIG as SOURCE_CONFIG

CONFIG = ROOT / "configs/genetic_primus_episode_attribution_v2.json"
CONFIG_SHA = "7c975d3102779400f1eb6e1955d52afc8901de7ffbf098544d6b0a371e24a98c"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/episode_attribution_v2_20261001/result.json"


def _config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("pre-registered episode attribution changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["schema_version"] != "exia.genetic_primus.episode_attribution/2":
        raise ValueError("wrong episode contract")
    if config["evidence_class"] != "revealed_historical_development_only":
        raise ValueError("disclosed data status changed")
    if tuple(config["seeds"]) != RANDOM_SEEDS or any(config["safety"].values()):
        raise ValueError("seeds or offline safety changed")
    source = ROOT / config["source_ablation_result_path"]
    manifest = ROOT / config["source_manifest_path"]
    if (file_sha256(source) != config["source_ablation_result_sha256"]
            or file_sha256(manifest) != config["source_manifest_sha256"]):
        raise ValueError("source evidence bytes changed")
    return config


def _evaluate() -> dict:
    config = _config()
    source = json.loads((ROOT / config["source_ablation_result_path"]).read_text(encoding="utf-8"))
    contract, _ = load_and_validate(SOURCE_CONFIG)
    if (config["fixed_universe"] != contract["universe"]
            or config["cost_bps_round_trip"]["normal"] != contract["accounting"]["normal_round_trip_bps"]
            or config["cost_bps_round_trip"]["stress"] != contract["accounting"]["stress_round_trip_bps"]):
        raise ValueError("accounting/universe mismatch")
    base, lineage, origin_ends = _archived_decisions(contract)
    start = int(base.execution_timestamp.min())
    end = int(origin_ends[-1])
    if (start != source["interval_start"] or end != source["interval_end_exclusive"]
            or source["archive_lineage"] != lineage):
        raise ValueError("disclosed origin lineage mismatch")
    eligible_bars = int((base.proposal == 1).sum())
    ga_active_bars = int((base.ga_action == 1).sum())
    structural_hamming_max = 2 * (eligible_bars - ga_active_bars)
    result = {
        "schema_version": "exia.genetic_primus.episode_attribution.result/2",
        "objective_id": config["objective_id"], "evidence_class": config["evidence_class"],
        "config_sha256": file_sha256(CONFIG), "script_sha256": file_sha256(Path(__file__)),
        "market_rows_sha256": None, "archive_lineage": lineage,
        "interval_start": start, "interval_end_exclusive": end,
        "eligible_bars": eligible_bars, "ga_active_bars": ga_active_bars,
        "structural_hamming_max": structural_hamming_max,
        "seeds": list(RANDOM_SEEDS), "controls": [],
        "daily_median_paired_difference": {}, "paired_lcb": {},
        "all_controls_diverse": False, "next_search_gate_passed": False,
        "matched_pnl_evaluated": False,
        "promotion_authority": False, "safety": dict(config["safety"]),
    }
    if structural_hamming_max < config["minimum_changed_decision_rows_per_seed"]:
        result["inconclusive_reason"] = "exact_exposure_control_cannot_reach_preregistered_diversity"
        return result
    prepared = []
    for seed in RANDOM_SEEDS:
        schedule, conservation = placed_episode_control(
            base, seed, config["max_search_nodes_per_symbol"])
        if not conservation["active_bars_equal"]:
            raise AssertionError("per-symbol exposure changed")
        record = {"seed": seed, "schedule_sha256": _frame_sha(schedule),
                  "conservation": conservation, "evaluations": {}}
        prepared.append((schedule, record))
        result["controls"].append(record)
    result["all_controls_diverse"] = all(
        row["conservation"]["changed_rows"] >= config["minimum_changed_decision_rows_per_seed"]
        for row in result["controls"])
    if not result["all_controls_diverse"]:
        result["inconclusive_reason"] = "sampled_controls_below_preregistered_diversity"
        return result
    market, market_sha = _market(contract, start, end)
    result["market_rows_sha256"] = market_sha
    all_returns = {"normal": [], "stress": []}
    for schedule, record in prepared:
        for label, cost in config["cost_bps_round_trip"].items():
            simulation = simulate_cash_targets(
                market, schedule, symbols=tuple(contract["universe"]),
                start_timestamp=start, end_timestamp=end,
                timeframe_ms=14_400_000, round_trip_cost_bps=cost,
                record_details=True)
            metrics = summarize_simulation(simulation, schedule, base,
                                           tuple(origin_ends), cost)
            ga = source["policies"]["GA_target_gate"][0]["evaluations"][label]
            if metrics["trades"] != ga["trades"] or metrics["long_trades"] != ga["long_trades"]:
                raise AssertionError("holding episode/trade count changed")
            record["evaluations"][label] = metrics
            all_returns[label].append(np.asarray(metrics["daily_returns"], dtype=float))
    bank = block_bank(27, config["paired_bootstrap"]["samples"],
                      config["paired_bootstrap"]["block_days"],
                      config["paired_bootstrap"]["seed"])
    result["paired_bank_sha256"] = array_sha(bank)
    for label, matrix in all_returns.items():
        reference = np.asarray(source["policies"]["GA_target_gate"][0]["evaluations"][label]["daily_returns"], dtype=float)
        median_daily = np.median(np.stack(matrix), axis=0)
        result["daily_median_paired_difference"][label] = float(np.mean(reference - median_daily))
        result["paired_lcb"][label] = paired_lcb(reference, median_daily, bank)
    result["matched_pnl_evaluated"] = True
    result["next_search_gate_passed"] = (
        result["all_controls_diverse"]
        and all(value > 0.0 for value in result["paired_lcb"].values()))
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable episode result already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DISCLOSED_EPISODE_DIAGNOSTIC_ONLY",
                      "controls": len(result["controls"]),
                      "hamming_max": result["structural_hamming_max"],
                      "normal_lcb": (round(result["paired_lcb"]["normal"], 7)
                                     if result["paired_lcb"] else None),
                      "stress_lcb": (round(result["paired_lcb"]["stress"], 7)
                                     if result["paired_lcb"] else None),
                      "gate": result["next_search_gate_passed"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
