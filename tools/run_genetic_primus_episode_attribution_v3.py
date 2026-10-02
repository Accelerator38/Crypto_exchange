"""Disclosed, maximum-diversity episode sensitivity with no trading authority."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.ablation_v1 import summarize_simulation
from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.episode_attribution_v3 import outcome_blind_bank
from exia.genetic_primus.measurement_lab_v2 import array_sha, block_bank, paired_lcb
from exia.genetic_primus.prospective_contract_v1 import file_sha256, load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import _frame_sha
from run_genetic_primus_ablation_v1 import _archived_decisions, _market, CONFIG as SOURCE_CONFIG

CONFIG = ROOT / "configs/genetic_primus_episode_attribution_v3.json"
CONFIG_SHA = "fb539ab2bf98dac464be061791d0ced6084da8e9ec5d8b74c5a318437039da0b"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/episode_attribution_v3_20261002/result.json"
DT = 14_400_000


def _source(path: str, expected_sha: str) -> dict:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or file_sha256(resolved) != expected_sha:
        raise ValueError("pinned source bytes changed")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("preregistered attribution contract changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["schema_version"] != "exia.genetic_primus.episode_attribution/3"
            or config["evidence_class"] != "revealed_historical_sensitivity_diagnostic_only"
            or any(config["safety"].values())
            or len(config["seeds"]) != 16
            or len(set(config["seeds"])) != 16
            or file_sha256(ROOT / config["geometry_module_path"])
            != config["geometry_module_sha256"]):
        raise ValueError("attribution method or offline safety changed")
    return config


def _mean_capital_exposure(simulation, market: pd.DataFrame) -> float:
    fills = simulation.events.loc[
        simulation.events.phase.eq("after_fills"), ["timestamp", "equity"]]
    frame = simulation.decisions.merge(
        market.rename(columns={"timestamp": "execution_timestamp"}),
        on=["execution_timestamp", "symbol"], validate="one_to_one"
    ).merge(fills.rename(columns={"timestamp": "execution_timestamp"}),
            on="execution_timestamp", validate="many_to_one")
    if len(frame) != len(simulation.decisions) or (frame.equity <= 0).any():
        raise ValueError("unreconciled portfolio exposure")
    frame["fraction"] = frame.quantity_after.abs() * frame.open / frame.equity
    return float(frame.groupby("execution_timestamp").fraction.sum().mean())


def _veto_episodes(base: pd.DataFrame, market: pd.DataFrame) -> list[dict]:
    prices = market.set_index(["timestamp", "symbol"])
    episodes = []
    for symbol, group in base.groupby("symbol", sort=True):
        ordered = group.sort_values("execution_timestamp", kind="stable").reset_index(drop=True)
        mask = (ordered.proposal.to_numpy(int) == 1) & (ordered.ga_action.to_numpy(int) == 0)
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
        for start, end in zip(starts, ends):
            first = int(ordered.execution_timestamp.iloc[start])
            last = int(ordered.execution_timestamp.iloc[end - 1])
            opening = float(prices.loc[(first, symbol), "open"])
            closing = float(prices.loc[(last, symbol), "close"])
            episodes.append({
                "symbol": str(symbol), "start_timestamp": first,
                "end_exclusive_timestamp": last + DT,
                "bars": int(end - start),
                "market_open_to_last_close_return_pct": float((closing / opening - 1.0) * 100),
                "origin_ids": sorted(set(ordered.origin_id.iloc[start:end].tolist())),
                "interpretation": "market_move_on_vetoed_bars_not_marginal_portfolio_PnL",
            })
    return episodes


def build() -> dict:
    config = _config()
    source = _source(config["source_ablation_result_path"],
                     config["source_ablation_result_sha256"])
    audit = _source(config["source_baseline_audit_path"],
                    config["source_baseline_audit_sha256"])
    _source(config["source_manifest_path"], config["source_manifest_sha256"])
    contract, _ = load_and_validate(SOURCE_CONFIG)
    if (config["fixed_universe"] != contract["universe"]
            or config["normal_round_trip_cost_bps"]
            != contract["accounting"]["normal_round_trip_bps"]
            or config["stress_round_trip_cost_bps"]
            != contract["accounting"]["stress_round_trip_bps"]
            or source["market_rows_sha256"] != config["market_rows_sha256"]
            or audit["market_rows_sha256"] != config["market_rows_sha256"]):
        raise ValueError("source universe/market/accounting mismatch")
    base, lineage, origin_ends = _archived_decisions(contract)
    start = int(base.execution_timestamp.min())
    end = int(origin_ends[-1])
    if (source["archive_lineage"] != lineage or audit["archive_lineage"] != lineage
            or source["interval_start"] != start or source["interval_end_exclusive"] != end):
        raise ValueError("nine-origin lineage changed")
    controls, geometry = outcome_blind_bank(
        base, tuple(config["seeds"]),
        time_limit_seconds=config["solver_seconds_per_problem"])
    maximum = sum(row["maximum_changed_rows"] for row in geometry.values())
    distinct = len({schedule.target.to_numpy(np.int8).tobytes()
                    for schedule, _ in controls})
    result = {
        "schema_version": "exia.genetic_primus.episode_attribution.result/3",
        "objective_id": config["objective_id"], "evidence_class": config["evidence_class"],
        "config_sha256": file_sha256(CONFIG), "script_sha256": file_sha256(Path(__file__)),
        "source_ablation_result_sha256": config["source_ablation_result_sha256"],
        "source_baseline_audit_sha256": config["source_baseline_audit_sha256"],
        "market_rows_sha256": config["market_rows_sha256"],
        "archive_lineage": lineage, "interval_start": start, "interval_end_exclusive": end,
        "geometry": geometry, "maximum_changed_decision_rows": maximum,
        "distinct_control_targets": distinct, "seeds": list(config["seeds"]),
        "controls": [], "veto_episodes": [], "comparisons": {},
        "matched_pnl_evaluated": False, "all_controls_match": False,
        "historical_support_gate": False, "promotion_authority": False,
        "safety": dict(config["safety"]),
    }
    if maximum != config["structural_maximum_changed_decision_rows"]:
        raise ValueError("outcome-blind geometry changed")
    if distinct < config["minimum_distinct_control_schedules"]:
        result["inconclusive_reason"] = "insufficient_distinct_outcome_blind_controls"
        return result
    market, market_sha = _market(contract, start, end)
    if market_sha != config["market_rows_sha256"]:
        raise ValueError("disclosed market rows changed")
    result["veto_episodes"] = _veto_episodes(base, market)
    costs = {"normal": config["normal_round_trip_cost_bps"],
             "stress": config["stress_round_trip_cost_bps"]}
    daily_controls = {label: [] for label in costs}
    for schedule, conservation in controls:
        if (conservation["changed_rows"] != maximum
                or conservation["active_bars"] != int((base.ga_action == 1).sum())
                or any(conservation["episode_lengths_by_symbol"][symbol]
                       != row["episode_lengths"] for symbol, row in geometry.items())):
            raise AssertionError("episode duration/exposure changed")
        record = {"seed": conservation["seed"], "schedule_sha256": _frame_sha(schedule),
                  "conservation": conservation, "evaluations": {}}
        for label, cost in costs.items():
            simulation = simulate_cash_targets(
                market, schedule, symbols=tuple(contract["universe"]),
                start_timestamp=start, end_timestamp=end, timeframe_ms=DT,
                round_trip_cost_bps=cost, record_details=True)
            metrics = summarize_simulation(simulation, schedule, base,
                                           tuple(origin_ends), cost)
            ga_metrics = source["policies"]["GA_target_gate"][0]["evaluations"][label]
            if metrics["trades"] != ga_metrics["trades"]:
                raise AssertionError("holding episode/trade count changed")
            exposure = _mean_capital_exposure(simulation, market)
            ga_exposure = audit["policies"]["GA_target_gate"]["evaluations"][label]["economics"]["mean_portfolio_capital_exposure_fraction"]
            turnover_difference = abs(metrics["entry_notional_turnover"]
                                      / ga_metrics["entry_notional_turnover"] - 1.0)
            exposure_difference = abs(exposure / ga_exposure - 1.0)
            matched = (turnover_difference <= config["maximum_relative_difference_in_entry_turnover"]
                       and exposure_difference <= config["maximum_relative_difference_in_mean_capital_exposure"])
            record["evaluations"][label] = {
                "metrics": metrics,
                "mean_portfolio_capital_exposure_fraction": exposure,
                "relative_turnover_difference_from_GA": float(turnover_difference),
                "relative_exposure_difference_from_GA": float(exposure_difference),
                "economic_match": bool(matched),
            }
            daily_controls[label].append(np.asarray(metrics["daily_returns"], float))
        result["controls"].append(record)
    result["all_controls_match"] = all(
        evaluation["economic_match"] for control in result["controls"]
        for evaluation in control["evaluations"].values())
    bank = block_bank(27, config["paired_bootstrap"]["samples"],
                      config["paired_bootstrap"]["block_days"],
                      config["paired_bootstrap"]["seed"])
    result["paired_bank_sha256"] = array_sha(bank)
    for label, series in daily_controls.items():
        ga_daily = np.asarray(
            source["policies"]["GA_target_gate"][0]["evaluations"][label]["daily_returns"], float)
        mean_daily = np.mean(np.stack(series), axis=0)
        ga_net = source["policies"]["GA_target_gate"][0]["evaluations"][label]["net_return_pct"]
        control_nets = [row["evaluations"][label]["metrics"]["net_return_pct"]
                        for row in result["controls"]]
        result["comparisons"][label] = {
            "ga_net_return_pct": ga_net,
            "control_net_return_pct_by_seed": control_nets,
            "control_net_median_pct": float(np.median(control_nets)),
            "GA_beats_control_count": int(sum(ga_net > value for value in control_nets)),
            "mean_daily_GA_minus_control": float(np.mean(ga_daily - mean_daily)),
            "paired_block_bootstrap_LCB_daily_GA_minus_control":
                paired_lcb(ga_daily, mean_daily, bank),
        }
    result["matched_pnl_evaluated"] = True
    result["historical_support_gate"] = (
        result["all_controls_match"]
        and distinct >= config["minimum_distinct_control_schedules"]
        and all(row["paired_block_bootstrap_LCB_daily_GA_minus_control"] > 0.0
                for row in result["comparisons"].values()))
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable episode attribution result already exists")
    result = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": "DISCLOSED_EPISODE_SENSITIVITY_ONLY",
        "controls": len(result["controls"]),
        "distinct": result["distinct_control_targets"],
        "changed": result["maximum_changed_decision_rows"],
        "matched": result["all_controls_match"],
        "historical_support": result["historical_support_gate"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
