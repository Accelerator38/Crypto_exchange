"""Bounded sparse overlay search on disclosed June-August development data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.measurement_lab_v2 import (
    FEATURES, LabScaler, PreparedWindow, array_sha, build_lab_features,
    daily_returns, object_sha,
)
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.selection_development_v1 import (
    absolute_score, choose_train_winner, common_bank, fixed_candidates,
    validation_gate,
)

CONFIG = ROOT / "configs/genetic_primus_selection_development_v1.json"
SOURCE_CONTRACT = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/selection_development_v1_20261001/result.json"


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("explicit UTC required")
    return parsed.astimezone(timezone.utc)


def _validate(config: dict) -> tuple[datetime, datetime]:
    if config["schema_version"] != "exia.genetic_primus.development_selection/1":
        raise ValueError("wrong selection schema")
    if config["evidence_class"] != "revealed_historical_development_only":
        raise ValueError("historical exposure must be disclosed")
    if any(config["safety"].values()) or config["september_outer_used_for_selection"]:
        raise ValueError("offline-only development selection required")
    if (config["training_window"] != "expanding" or config["purge_bars"] != 7
            or config["timeframe_minutes"] != 240
            or config["scaler"] != "refit_each_origin_train_only_median_iqr_clip_8"
            or tuple(config["features"]) != FEATURES):
        raise ValueError("selection preparation differs from prospective V2")
    if (config["candidate_budget"] != 24 or config["deployment_horizon_days"] != 3
            or config["selection_origin_count"] != 20
            or config["validation_origin_count"] != 10
            or config["normal_round_trip_bps"] != 16.0
            or config["stress_round_trip_bps"] != 24.0
            or config["bootstrap_samples"] != 512
            or config["bootstrap_block_days"] != 5
            or config["bootstrap_alpha"] != 0.05
            or config["validation_selects_alternative"] is not False):
        raise ValueError("bounded registered search design changed")
    start = _utc(config["selection_start_utc"])
    finish = start + timedelta(days=3 * (20 + 10))
    if finish != datetime(2026, 8, 30, tzinfo=timezone.utc):
        raise ValueError("development interval must end before September")
    if _utc(config["training_history_start_utc"]) != datetime(2022, 1, 1, tzinfo=timezone.utc):
        raise ValueError("expanding history start changed")
    if config["include_frozen_genome"] != [0, 0, 1, 1, -1]:
        raise ValueError("frozen incumbent comparison removed")
    expected_gate = [
        "normal_net_positive", "stress_net_positive",
        "normal_lcb_vs_NoTrade_positive", "stress_lcb_vs_NoTrade_positive",
        "normal_paired_lcb_vs_EMA_long_only_positive",
        "stress_paired_lcb_vs_EMA_long_only_positive",
        "drawdown_below_absolute_limit"]
    if config["validation_gate"] != expected_gate:
        raise ValueError("registered validation gate changed")
    if config["training_score"] != "minimum_normal_and_stress_paired_block_lcb_vs_NoTrade":
        raise ValueError("registered training score changed")
    return start, finish


def _windows(config: dict, panel: pd.DataFrame, start: datetime) -> list[tuple[PreparedWindow, LabScaler]]:
    end = start + timedelta(days=3 * 30)
    end_ms = int(end.timestamp() * 1000)
    causal_panel = panel.loc[panel["timestamp"] < end_ms].copy()
    features = build_lab_features(causal_panel)
    symbols = tuple(json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))["universe"])
    dt = 240 * 60_000
    history_ms = int(_utc(config["training_history_start_utc"]).timestamp() * 1000)
    result = []
    for index in range(30):
        origin_start = int((start + timedelta(days=3 * index)).timestamp() * 1000)
        origin_end = origin_start + 3 * 86_400_000
        train_end = origin_start - 7 * dt
        train = PreparedWindow.from_frame(
            features, symbols, history_ms + dt, train_end, dt)
        deploy = PreparedWindow.from_frame(
            features, symbols, origin_start, origin_end, dt)
        result.append((deploy, LabScaler.fit(train.values)))
    return result


def _simulate(windows: list[tuple[PreparedWindow, LabScaler]],
              genome: tuple[int, ...] | None, cost_bps: float):
    symbols = windows[0][0].symbols
    bars = pd.concat([window.bars for window, _ in windows], ignore_index=True)
    schedules = []
    for window, scaler in windows:
        if genome is None:
            schedule = window.base_schedule.copy()
            schedule["target"] = np.maximum(window.proposal, 0)
            schedule["model_sha256"] = object_sha({
                "policy": "EMA_long_only", "proposal": "V7_EMA_TrendConsensus",
                "scaler_fit_values_sha256": scaler.fit_values_sha256})
        else:
            schedule = window.schedule(genome, scaler)
        schedules.append(schedule)
    return simulate_cash_targets(
        bars, pd.concat(schedules, ignore_index=True), symbols=symbols,
        start_timestamp=windows[0][0].start, end_timestamp=windows[-1][0].end,
        timeframe_ms=240 * 60_000, round_trip_cost_bps=cost_bps,
        record_details=False)


def _summary(simulation) -> dict:
    return {"net_return": float(simulation.metrics["net_return"]),
            "max_drawdown": float(simulation.metrics["max_drawdown"]),
            "trades": int(simulation.metrics["trades"]),
            "fees": float(simulation.metrics["total_fees"]),
            "daily_returns": daily_returns(simulation).tolist()}


def _evaluate() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    start, finish = _validate(config)
    manifest_path = ROOT / config["source_manifest_path"]
    if file_sha256(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("development snapshot manifest changed")
    source_contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=SOURCE_CONTRACT,
        contract=source_contract)
    if manifest["metadata"]["coverage_start"] > int(_utc(
            config["training_history_start_utc"]).timestamp() * 1000):
        raise ValueError("expanding history is incomplete")
    windows = _windows(config, panel, start)
    train = windows[:20]
    validation = windows[20:]
    bank_train = common_bank(60, config, validation=False)
    bank_validation = common_bank(30, config, validation=True)
    candidates = fixed_candidates(config["candidate_budget"], config["candidate_seed"])
    if ([list(genome) for genome in candidates] != config["candidate_genomes"]
            or len(candidates) != config["candidate_budget"]):
        raise ValueError("sealed candidate set or budget changed")
    records = []
    for genome in candidates:
        normal = _summary(_simulate(train, genome, config["normal_round_trip_bps"]))
        stress = _summary(_simulate(train, genome, config["stress_round_trip_bps"]))
        score = absolute_score(np.asarray(normal["daily_returns"]),
                               np.asarray(stress["daily_returns"]), bank_train)
        records.append({"genome": list(genome), "training": {
            **score, "normal_net": normal["net_return"],
            "stress_net": stress["net_return"],
            "normal_trades": normal["trades"], "stress_trades": stress["trades"]}})
    selected = choose_train_winner(records)
    baseline = {label: _summary(_simulate(validation, None, cost))
                for label, cost in (("normal", config["normal_round_trip_bps"]),
                                    ("stress", config["stress_round_trip_bps"]))}
    validation_result = None
    final_policy = "NoTrade"
    if selected is not None:
        genome = tuple(selected["genome"])
        normal = _summary(_simulate(validation, genome, config["normal_round_trip_bps"]))
        stress = _summary(_simulate(validation, genome, config["stress_round_trip_bps"]))
        gate = validation_gate(
            np.asarray(normal["daily_returns"]), np.asarray(stress["daily_returns"]),
            np.asarray(baseline["normal"]["daily_returns"]),
            np.asarray(baseline["stress"]["daily_returns"]),
            bank_validation, max_drawdown_normal=normal["max_drawdown"],
            max_drawdown_stress=stress["max_drawdown"],
            drawdown_limit=float(config["maximum_drawdown"]))
        validation_result = {"genome": list(genome), "normal": normal,
                             "stress": stress, "gate": gate}
        if gate["passed"]:
            final_policy = "selected_sparse_overlay_for_future_separate_audit_only"
    top = max(records, key=lambda row: row["training"]["rank_score"])
    return {
        "schema_version": "exia.genetic_primus.development_selection_result/1",
        "evidence_class": "revealed_historical_development_only",
        "config_sha256": file_sha256(CONFIG),
        "script_sha256": file_sha256(Path(__file__)),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_snapshot_sha256": manifest["snapshot_sha256"],
        "selection_start_utc": start.isoformat().replace("+00:00", "Z"),
        "selection_cutoff_utc": finish.isoformat().replace("+00:00", "Z"),
        "train_origin_count": len(train), "validation_origin_count": len(validation),
        "candidate_budget": len(candidates),
        "candidate_set_sha256": object_sha([list(value) for value in candidates]),
        "train_bank_sha256": array_sha(bank_train),
        "validation_bank_sha256": array_sha(bank_validation),
        "candidate_records": records,
        "best_train_genome_even_if_ineligible": top["genome"],
        "best_train_score": top["training"]["rank_score"],
        "selected_genome": selected["genome"] if selected is not None else None,
        "window": config["training_window"], "scaler": config["scaler"],
        "validation_baseline_ema_long_only": baseline,
        "validation_result": validation_result,
        "final_policy": final_policy,
        "search_evaluations": len(records),
        "global_prior_trials_included_in_nominal_lcb": False,
        "automatic_promotion": False, "safety": dict(config["safety"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("only project .venv Python is authorized")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "Reports/Exia/Genetic_Primus").resolve()):
        raise ValueError("output must remain in Genetic Primus reports")
    if output.exists():
        raise FileExistsError("selection result already exists")
    result = _evaluate()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DEVELOPMENT_SELECTION_COMPLETE_NO_PROMOTION",
                      "evaluations": result["search_evaluations"],
                      "best_train_score": round(result["best_train_score"], 7),
                      "selected_genome": result["selected_genome"],
                      "final_policy": result["final_policy"]},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
