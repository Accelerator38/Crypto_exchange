"""Offline six-policy ablation on the disclosed first nine September origins."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.ablation_v1 import (
    POLICIES, RANDOM_SEEDS, summarize_simulation, targets_for_policy,
)
from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.prospective_contract_v1 import file_sha256, load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import _frame_sha
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot

CONFIG = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
ARCHIVE = ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609"
DEFAULT_OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/ablation_continuous_v1_20261001/result.json"
ORIGIN_COUNT = 9


def _archived_decisions(contract: dict) -> tuple[pd.DataFrame, list[dict], list[int]]:
    rows = []
    lineage = []
    ends = []
    prior_end = None
    for index in range(ORIGIN_COUNT):
        stem = f"origin_{index + 1:02d}"
        result_path = ARCHIVE / f"{stem}.json"
        manifest_path = ARCHIVE / f"data_manifest_{stem}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (result["origin_index"] != index or result["origin_id"] != stem
                or result["manifest_sha256"] != file_sha256(manifest_path)
                or result["contract_sha256"] != file_sha256(CONFIG)
                or result["data_snapshot_sha256"] != manifest["snapshot_sha256"]):
            raise ValueError("archived result/manifest lineage mismatch")
        if prior_end is not None and result["origin_start"] != prior_end:
            raise ValueError("nine origins are not contiguous")
        prior_end = result["origin_end_exclusive"]
        ends.append(prior_end)
        candidates = result["evaluations"]["normal"]["candidates"]
        ledgers = {}
        for name in ("NoTrade", "EMA_TrendConsensus", "GA_overlay_f7a4ae11"):
            candidate = candidates[name]
            ledger = pd.DataFrame(candidate["ledger"])[contract["ledger"]["required_fields"]]
            if (_frame_sha(ledger) != candidate["ledger_sha256"]
                    or len(ledger) != 18 * len(contract["universe"])
                    or ledger["origin_id"].nunique() != 1
                    or ledger["origin_id"].iloc[0] != stem
                    or ledger["model_sha256"].nunique() != 1
                    or ledger["signal_timestamp"].gt(ledger["execution_timestamp"]).any()):
                raise ValueError("archived decision ledger changed")
            ledgers[name] = ledger.sort_values(
                ["execution_timestamp", "symbol"], kind="stable").reset_index(drop=True)
        ema, ga, flat = (ledgers[name] for name in (
            "EMA_TrendConsensus", "GA_overlay_f7a4ae11", "NoTrade"))
        if (not ema[["execution_timestamp", "symbol", "proposal"]].equals(
                ga[["execution_timestamp", "symbol", "proposal"]])
                or not ema[["execution_timestamp", "symbol", "proposal"]].equals(
                    flat[["execution_timestamp", "symbol", "proposal"]])
                or not ema["action"].equals(ema["proposal"])
                or flat["action"].ne(0).any()):
            raise ValueError("archived candidate decisions are not aligned")
        frame = ema[["execution_timestamp", "signal_timestamp", "symbol", "origin_id",
                     "proposal", "model_sha256"]].rename(
                         columns={"model_sha256": "ema_model_sha256"})
        frame["ga_action"] = ga["action"].to_numpy(int)
        frame["ga_model_sha256"] = ga["model_sha256"].to_numpy(str)
        rows.append(frame)
        lineage.append({"origin_id": stem, "result_sha256": file_sha256(result_path),
                        "manifest_sha256": file_sha256(manifest_path),
                        "snapshot_sha256": manifest["snapshot_sha256"],
                        "ema_ledger_sha256": candidates["EMA_TrendConsensus"]["ledger_sha256"],
                        "ga_ledger_sha256": candidates["GA_overlay_f7a4ae11"]["ledger_sha256"]})
    return pd.concat(rows, ignore_index=True), lineage, ends


def _market(contract: dict, start: int, end: int) -> tuple[pd.DataFrame, str]:
    manifest, panel = load_verified_snapshot(
        ARCHIVE / "data_manifest_origin_09.json", root=ROOT,
        contract_path=CONFIG, contract=contract)
    if manifest["metadata"]["coverage_end_exclusive"] != end:
        raise ValueError("final committed snapshot ends outside the disclosed interval")
    market = panel.loc[panel["timestamp"].ge(start) & panel["timestamp"].lt(end),
                       ["timestamp", "symbol", "open", "close"]].copy()
    expected = ORIGIN_COUNT * 3 * 6 * len(contract["universe"])
    if len(market) != expected:
        raise ValueError("disclosed market grid is incomplete")
    return market, _frame_sha(market)


def _evaluate(contract: dict) -> dict:
    base, lineage, origin_ends = _archived_decisions(contract)
    start = int(base["execution_timestamp"].min())
    end = origin_ends[-1]
    market, market_sha = _market(contract, start, end)
    if len(base) != len(market):
        raise ValueError("market and decision grid counts differ")
    result = {"schema_version": "exia.genetic_primus.disclosed_ablation/1",
              "evidence_class": "retrospective_disclosed_engineering_attribution_only",
              "interval_start": start, "interval_end_exclusive": end,
              "origin_count": ORIGIN_COUNT, "contract_sha256": file_sha256(CONFIG),
              "script_sha256": file_sha256(Path(__file__)),
              "market_rows_sha256": market_sha, "archive_lineage": lineage,
              "random_seeds_fixed": list(RANDOM_SEEDS),
              "random_matching_rule": "exact_GA_long_target_count_per_symbol_on_EMA_long_opportunities",
              "policies": {}, "safety": dict(contract["safety"])}
    costs = {"normal": float(contract["accounting"]["normal_round_trip_bps"]),
             "stress": float(contract["accounting"]["stress_round_trip_bps"])}
    for policy in POLICIES:
        runs = []
        seeds = RANDOM_SEEDS if policy == "EMA_long_random_matched" else (None,)
        for seed in seeds:
            schedule = targets_for_policy(base, policy, seed=seed)
            schedule_sha = _frame_sha(schedule)
            metrics = {}
            for cost_label, cost_bps in costs.items():
                simulation = simulate_cash_targets(
                    market, schedule, symbols=tuple(contract["universe"]),
                    start_timestamp=start, end_timestamp=end,
                    timeframe_ms=240 * 60_000, round_trip_cost_bps=cost_bps,
                    record_details=True)
                metrics[cost_label] = summarize_simulation(
                    simulation, schedule, base, tuple(origin_ends), cost_bps)
            runs.append({"seed": seed, "schedule_sha256": schedule_sha,
                         "evaluations": metrics})
        result["policies"][policy] = runs
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    expected = (ROOT / ".venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError("only the project .venv Python is authorized")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "Reports/Exia/Genetic_Primus").resolve()):
        raise ValueError("output must stay inside Genetic Primus reports")
    if output.exists():
        raise FileExistsError("ablation result already exists")
    contract, _ = load_and_validate(CONFIG)
    result = _evaluate(contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    concise = {name: [round(run["evaluations"]["normal"]["net_return_pct"], 3)
                      for run in runs]
               for name, runs in result["policies"].items()}
    print(json.dumps({"status": "DISCLOSED_ABLATION_COMPLETE_NO_PROMOTION",
                      "output": output.relative_to(ROOT).as_posix(),
                      "normal_net_pct": concise}, separators=(",", ":")))


if __name__ == "__main__":
    main()
