"""One sealed offline economic check of a fixed cross-sectional base signal."""

from __future__ import annotations

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
    PreparedWindow, array_sha, build_lab_features, daily_returns, object_sha,
)
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.relative_momentum_development_v1 import (
    MODEL_SHA256, MODEL_SPEC, origin_schedule, select_at_origin,
)
from exia.genetic_primus.selection_development_v1 import (
    absolute_score, common_bank, validation_gate,
)

CONFIG = ROOT / "configs/genetic_primus_relative_momentum_development_v1.json"
CONFIG_SHA256 = "279efe9b48fb5eb9b1ad7bf7e679edd1af180c93e96227d07098ba43d55e3c02"
SOURCE_CONTRACT = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/relative_momentum_development_v1_20261001/result.json"
DT = 240 * 60_000


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("explicit UTC required")
    return parsed.astimezone(timezone.utc)


def _validate(config: dict) -> tuple[datetime, datetime]:
    if config.get("schema_version") != "exia.genetic_primus.relative_momentum_development/1":
        raise ValueError("wrong development schema")
    if config.get("evidence_class") != "revealed_historical_development_only":
        raise ValueError("historical exposure must be disclosed")
    if any(config.get("safety", {}).values()) or not config.get("safety"):
        raise ValueError("offline-only safety contract required")
    if config.get("september_outer_used_for_selection") is not False:
        raise ValueError("September is already revealed and out of this check")
    expected = {
        "lookback_bars": MODEL_SPEC["lookback_bars"],
        "rank_top_k": MODEL_SPEC["rank_top_k"],
        "minimum_prior_return": MODEL_SPEC["minimum_prior_return"],
        "direction": MODEL_SPEC["direction"],
        "timeframe_minutes": 240, "deployment_horizon_days": 3,
        "selection_origin_count": 20, "validation_origin_count": 10,
        "candidate_budget": 1, "normal_round_trip_bps": 16.0,
        "stress_round_trip_bps": 24.0,
        "bootstrap_samples": 512, "bootstrap_block_days": 5,
        "bootstrap_alpha": 0.05, "maximum_drawdown": 0.10,
        "no_parameter_fitting": True, "validation_only_if_train_passes": True,
        "validation_selects_alternative": False,
        "training_score": "minimum_normal_and_stress_paired_block_lcb_vs_NoTrade_strictly_positive",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed hypothesis or budget changed")
    required_safety = {
        "credentials_allowed", "orders_enabled", "paper_allowed", "live_allowed",
        "promotion_authority", "runtime_authority", "network_allowed",
    }
    if not required_safety.issubset(config["safety"]) or any(
            value is not False for value in config["safety"].values()):
        raise ValueError("offline safety flags incomplete")
    expected_gate = [
        "normal_net_positive", "stress_net_positive",
        "normal_lcb_vs_NoTrade_positive", "stress_lcb_vs_NoTrade_positive",
        "normal_paired_lcb_vs_EMA_long_only_positive",
        "stress_paired_lcb_vs_EMA_long_only_positive",
        "drawdown_below_absolute_limit",
    ]
    if config.get("validation_gate") != expected_gate:
        raise ValueError("validation gate changed")
    start = _utc(config["selection_start_utc"])
    end = start + timedelta(days=3 * (20 + 10))
    if start.isoformat() != "2026-06-01T00:00:00+00:00" or end.isoformat() != "2026-08-30T00:00:00+00:00":
        raise ValueError("historical split changed")
    return start, end


def _schedules(panel: pd.DataFrame, symbols: tuple[str, ...],
               start: datetime, count: int) -> tuple[pd.DataFrame, list[dict]]:
    schedules = []
    decisions = []
    for index in range(count):
        origin = start + timedelta(days=3 * index)
        origin_ms = int(origin.timestamp() * 1000)
        selected = select_at_origin(panel, origin_ms, symbols, DT)
        schedules.append(origin_schedule(origin_ms, origin_ms + 3 * 86_400_000,
                                         DT, symbols, selected))
        decisions.append({"origin_start_utc": origin.isoformat().replace("+00:00", "Z"),
                          "selected_symbols": selected, "model_sha256": MODEL_SHA256})
    return pd.concat(schedules, ignore_index=True), decisions


def _simulate(panel: pd.DataFrame, schedule: pd.DataFrame,
              symbols: tuple[str, ...], cost: float):
    return simulate_cash_targets(
        panel, schedule, symbols=symbols,
        start_timestamp=int(schedule.execution_timestamp.min()),
        end_timestamp=int(schedule.execution_timestamp.max()) + DT,
        timeframe_ms=DT, round_trip_cost_bps=cost, record_details=False)


def _summary(simulation) -> dict:
    return {"net_return": float(simulation.metrics["net_return"]),
            "max_drawdown": float(simulation.metrics["max_drawdown"]),
            "trades": int(simulation.metrics["trades"]),
            "fees": float(simulation.metrics["total_fees"]),
            "daily_returns": daily_returns(simulation).tolist()}


def _ema_reference(panel: pd.DataFrame, symbols: tuple[str, ...],
                   start: datetime, count: int) -> pd.DataFrame:
    end_ms = int((start + timedelta(days=3 * count)).timestamp() * 1000)
    features = build_lab_features(panel.loc[panel.timestamp < end_ms].copy())
    schedules = []
    for index in range(count):
        origin_ms = int((start + timedelta(days=3 * index)).timestamp() * 1000)
        window = PreparedWindow.from_frame(features, symbols, origin_ms,
                                           origin_ms + 3 * 86_400_000, DT)
        schedule = window.base_schedule.copy()
        schedule["target"] = np.maximum(window.proposal, 0)
        schedule["model_sha256"] = object_sha({
            "policy": "EMA_long_only", "proposal": "V7_EMA_TrendConsensus"})
        schedules.append(schedule)
    return pd.concat(schedules, ignore_index=True)


def _evaluate() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("only the project .venv Python is authorized")
    if file_sha256(CONFIG) != CONFIG_SHA256:
        raise ValueError("pre-registered hypothesis bytes changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    start, _ = _validate(config)
    manifest_path = (ROOT / config["source_manifest_path"]).resolve()
    if not manifest_path.is_relative_to(ROOT.resolve()):
        raise ValueError("source manifest outside repository")
    if file_sha256(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("source manifest bytes changed")
    source_contract = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=SOURCE_CONTRACT,
        contract=source_contract)
    symbols = tuple(source_contract["universe"])
    train_schedule, train_decisions = _schedules(panel, symbols, start, 20)
    bank_train = common_bank(60, config, validation=False)
    train = {label: _summary(_simulate(panel, train_schedule, symbols, cost))
             for label, cost in (("gross", 0.0),
                                 ("normal", config["normal_round_trip_bps"]),
                                 ("stress", config["stress_round_trip_bps"]))}
    train_score = absolute_score(
        np.asarray(train["normal"]["daily_returns"]),
        np.asarray(train["stress"]["daily_returns"]), bank_train)
    result = {
        "schema_version": "exia.genetic_primus.relative_momentum_development_result/1",
        "evidence_class": "revealed_historical_development_only",
        "config_sha256": file_sha256(CONFIG),
        "script_sha256": file_sha256(Path(__file__)),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_snapshot_sha256": manifest["snapshot_sha256"],
        "model_sha256": MODEL_SHA256, "candidate_budget": 1,
        "search_evaluations": 1, "train_origin_count": 20,
        "validation_origin_count": 10, "train_bank_sha256": array_sha(bank_train),
        "train_decisions": train_decisions, "train": train,
        "train_score": train_score, "validation_result": None,
        "final_policy": "NoTrade", "automatic_promotion": False,
        "global_prior_trials_included_in_nominal_lcb": False,
        "safety": dict(config["safety"]),
    }
    if train_score["rank_score"] <= 0:
        return result
    validation_start = start + timedelta(days=60)
    schedule, decisions = _schedules(panel, symbols, validation_start, 10)
    ema_schedule = _ema_reference(panel, symbols, validation_start, 10)
    bank = common_bank(30, config, validation=True)
    validation = {label: _summary(_simulate(panel, schedule, symbols, cost))
                  for label, cost in (("gross", 0.0),
                                      ("normal", config["normal_round_trip_bps"]),
                                      ("stress", config["stress_round_trip_bps"]))}
    baseline = {label: _summary(_simulate(panel, ema_schedule, symbols, cost))
                for label, cost in (("normal", config["normal_round_trip_bps"]),
                                    ("stress", config["stress_round_trip_bps"]))}
    gate = validation_gate(
        np.asarray(validation["normal"]["daily_returns"]),
        np.asarray(validation["stress"]["daily_returns"]),
        np.asarray(baseline["normal"]["daily_returns"]),
        np.asarray(baseline["stress"]["daily_returns"]), bank,
        max_drawdown_normal=validation["normal"]["max_drawdown"],
        max_drawdown_stress=validation["stress"]["max_drawdown"],
        drawdown_limit=float(config["maximum_drawdown"]))
    result["validation_result"] = {
        "decisions": decisions, "candidate": validation,
        "baseline_ema_long_only": baseline, "gate": gate,
        "bank_sha256": array_sha(bank)}
    if gate["passed"]:
        result["final_policy"] = "base_signal_for_future_research_only"
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable development result already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DEVELOPMENT_ONLY_NO_PROMOTION",
                      "train_score": round(result["train_score"]["rank_score"], 7),
                      "validation_opened": result["validation_result"] is not None,
                      "final_policy": result["final_policy"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
