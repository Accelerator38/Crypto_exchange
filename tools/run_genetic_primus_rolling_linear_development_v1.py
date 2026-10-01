"""Two pre-registered rolling ridge policies on disclosed three-day origins."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.measurement_lab_v2 import array_sha, daily_returns, object_sha
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.rolling_linear_retrain_v1 import (
    DT, HORIZON_MS, VARIANTS, build_origin_examples, fit_at_origin, schedule_at_origin,
)
from exia.genetic_primus.selection_development_v1 import (
    absolute_score, common_bank, validation_gate,
)

CONFIG = ROOT / "configs/genetic_primus_rolling_linear_development_v1.json"
CONFIG_SHA = "020c7732026a23cfd97e2e19f1167a11e8d74552477bb2e62713359f6810379b"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/rolling_linear_development_v1_20261001/result.json"


def _ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC origin required")
    return int(parsed.timestamp() * 1000)


def _config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("pre-registered linear contract bytes changed")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["schema_version"] != "exia.genetic_primus.rolling_linear_development/1":
        raise ValueError("wrong rolling-linear contract")
    if config["evidence_class"] != "revealed_historical_development_only":
        raise ValueError("disclosed data status changed")
    if (config["variants"] != list(VARIANTS) or config["candidate_budget"] != 2
            or config["training_origin_count"] != 20
            or config["inner_validation_origin_count"] != 10
            or config["purge_bars"] != 7 or config["ridge_alpha"] != 0.5
            or config["features"] != ["return_12", "efficiency_12"]
            or config["bootstrap_samples"] != 512
            or config["bootstrap_block_days"] != 5
            or config["bootstrap_alpha"] != 0.05
            or config["normal_round_trip_bps"] != 16.0
            or config["stress_round_trip_bps"] != 24.0
            or any(config["safety"].values())
            or config["september_used_for_selection"] is not False):
        raise ValueError("fixed search budget or method changed")
    manifest = ROOT / config["source_manifest_path"]
    if file_sha256(manifest) != config["source_manifest_sha256"]:
        raise ValueError("source snapshot manifest changed")
    if _ms(config["development_start_utc"]) != 1_780_272_000_000:
        raise ValueError("registered June development split changed")
    return config


def _summarize(sim) -> dict:
    return {"net_return": float(sim.metrics["net_return"]),
            "max_drawdown": float(sim.metrics["max_drawdown"]),
            "trades": int(sim.metrics["trades"]),
            "fees": float(sim.metrics["total_fees"]),
            "daily_returns": daily_returns(sim).tolist()}


def _simulate(panel: pd.DataFrame, schedule: pd.DataFrame,
              symbols: tuple[str, ...], cost: float):
    return simulate_cash_targets(
        panel, schedule, symbols=symbols,
        start_timestamp=int(schedule.execution_timestamp.min()),
        end_timestamp=int(schedule.execution_timestamp.max()) + DT,
        timeframe_ms=DT, round_trip_cost_bps=cost, record_details=False)


def _schedule_chain(examples: pd.DataFrame, symbols: tuple[str, ...],
                    first: int, count: int, variant: str, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    schedules = []
    ledger = []
    for index in range(count):
        origin = first + index * HORIZON_MS
        model = fit_at_origin(
            examples, origin, variant, purge_bars=config["purge_bars"],
            ridge_alpha=config["ridge_alpha"],
            normal_round_trip_bps=config["normal_round_trip_bps"],
            minimum_opportunities=config["minimum_train_opportunities"],
            minimum_ess=config["minimum_weight_ess"])
        if index in (0, count - 1):
            prefix = examples.loc[examples.origin_start.le(origin)]
            checked = fit_at_origin(
                prefix, origin, variant, purge_bars=config["purge_bars"],
                ridge_alpha=config["ridge_alpha"],
                normal_round_trip_bps=config["normal_round_trip_bps"],
                minimum_opportunities=config["minimum_train_opportunities"],
                minimum_ess=config["minimum_weight_ess"])
            if checked["model_sha256"] != model["model_sha256"]:
                raise AssertionError("future rows changed a historical model")
        if (model["max_train_label_end_exclusive"] is not None
                and model["max_train_label_end_exclusive"] > origin - config["purge_bars"] * DT):
            raise AssertionError("purge/label availability violated")
        schedule, decision = schedule_at_origin(examples, symbols, origin, model)
        schedules.append(schedule)
        ledger.append({"origin_index": index, "origin_start": origin,
                       "model": model, "decision": decision})
    return pd.concat(schedules, ignore_index=True), ledger


def _ema_chain(examples: pd.DataFrame, symbols: tuple[str, ...],
               first: int, count: int) -> pd.DataFrame:
    schedules = []
    sha = object_sha({"policy": "EMA_long_at_origin_open_held_3d",
                      "source": "V7_EMA_TrendConsensus"})
    for index in range(count):
        origin = first + index * HORIZON_MS
        deploy = examples.loc[examples.origin_start.eq(origin)].set_index("symbol")
        if set(deploy.index) != set(symbols) or len(deploy) != len(symbols):
            raise ValueError("incomplete EMA reference universe")
        selected = tuple(symbol for symbol in symbols if bool(deploy.loc[symbol, "proposal_long"]))
        frame = pd.MultiIndex.from_product(
            [np.arange(origin, origin + HORIZON_MS, DT), symbols],
            names=["execution_timestamp", "symbol"]).to_frame(index=False)
        frame["signal_timestamp"] = origin
        frame["target"] = frame.symbol.isin(selected).astype(int)
        frame["model_sha256"] = sha
        schedules.append(frame)
    return pd.concat(schedules, ignore_index=True)


def _evaluate() -> dict:
    config = _config()
    manifest_path = ROOT / config["source_manifest_path"]
    source_contract_path = ROOT / config["source_contract_path"]
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=source_contract_path,
        contract=source_contract)
    start = _ms(config["development_start_utc"])
    finish = start + (config["training_origin_count"] + config["inner_validation_origin_count"]) * HORIZON_MS
    symbols = tuple(source_contract["universe"])
    examples = build_origin_examples(panel, symbols, _ms(config["history_start_utc"]),
                                     start, finish)
    train_bank = common_bank(60, config, validation=False)
    records = []
    for variant in VARIANTS:
        schedule, ledger = _schedule_chain(examples, symbols, start, 20, variant, config)
        evaluations = {name: _summarize(_simulate(panel, schedule, symbols, cost))
                       for name, cost in (("gross_zero_cost", 0.0),
                                          ("normal", 16.0), ("stress", 24.0))}
        score = absolute_score(np.asarray(evaluations["normal"]["daily_returns"]),
                               np.asarray(evaluations["stress"]["daily_returns"]), train_bank)
        records.append({"variant": variant, "origin_models": ledger,
                        "evaluations": evaluations, "train_score": score})
    winner = max(records, key=lambda item: item["train_score"]["rank_score"])
    selected = winner["variant"] if winner["train_score"]["rank_score"] > 0 else None
    result = {
        "schema_version": "exia.genetic_primus.rolling_linear_development_result/1",
        "evidence_class": "revealed_historical_development_only",
        "config_sha256": file_sha256(CONFIG),
        "script_sha256": file_sha256(Path(__file__)),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_snapshot_sha256": manifest["snapshot_sha256"],
        "examples_count": len(examples),
        "examples_sha256": object_sha(examples.to_dict("records")),
        "train_bank_sha256": array_sha(train_bank),
        "candidate_budget": 2, "search_evaluations": len(records),
        "records": records, "selected_variant": selected,
        "validation_result": None, "final_policy": "NoTrade",
        "automatic_promotion": False, "safety": dict(config["safety"]),
        "global_prior_trials_included_in_nominal_lcb": False,
    }
    if selected is None:
        return result
    validation_start = start + 20 * HORIZON_MS
    schedule, ledger = _schedule_chain(examples, symbols, validation_start, 10, selected, config)
    ema_schedule = _ema_chain(examples, symbols, validation_start, 10)
    candidate = {name: _summarize(_simulate(panel, schedule, symbols, cost))
                 for name, cost in (("gross_zero_cost", 0.0), ("normal", 16.0), ("stress", 24.0))}
    baseline = {name: _summarize(_simulate(panel, ema_schedule, symbols, cost))
                for name, cost in (("normal", 16.0), ("stress", 24.0))}
    bank = common_bank(30, config, validation=True)
    gate = validation_gate(
        np.asarray(candidate["normal"]["daily_returns"]),
        np.asarray(candidate["stress"]["daily_returns"]),
        np.asarray(baseline["normal"]["daily_returns"]),
        np.asarray(baseline["stress"]["daily_returns"]), bank,
        max_drawdown_normal=candidate["normal"]["max_drawdown"],
        max_drawdown_stress=candidate["stress"]["max_drawdown"],
        drawdown_limit=config["maximum_drawdown"])
    result["validation_result"] = {
        "variant": selected, "origin_models": ledger, "candidate": candidate,
        "baseline_ema_origin_held": baseline, "bank_sha256": array_sha(bank),
        "gate": gate}
    if gate["passed"]:
        result["final_policy"] = "linear_overlay_for_future_separate_audit_only"
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable rolling-linear result exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DISCLOSED_ROLLING_LINEAR_COMPLETE",
                      "train_scores": {r["variant"]: round(r["train_score"]["rank_score"], 7)
                                       for r in result["records"]},
                      "selected": result["selected_variant"],
                      "validation_opened": result["validation_result"] is not None,
                      "final_policy": result["final_policy"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
