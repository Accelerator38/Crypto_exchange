"""One-shot disclosed 2x2 linear matrix; no network or trading authority."""

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
from exia.genetic_primus.measurement_lab_v2 import (
    array_sha, block_bank, daily_returns, object_sha, paired_lcb,
)
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.rolling_linear_matrix_v1 import (
    FEATURE_SETS, build_matrix_examples, build_matrix_schedule, fit_matrix_at_origin,
)
from exia.genetic_primus.rolling_linear_retrain_v1 import DT, HORIZON_MS, VARIANTS

CONFIG = ROOT / "configs/genetic_primus_rolling_linear_matrix_v1.json"
CONFIG_SHA = "a75d48f73fea2f2f81daba128b7bac75aab78e9dd369b071284875152952a361"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/rolling_linear_matrix_v1_20261002/result.json"


def _ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC timestamp required")
    return int(parsed.timestamp() * 1000)


def _config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("preregistered config changed")
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if c["schema_version"] != "exia.genetic_primus.rolling_linear_matrix/1":
        raise ValueError("unexpected objective schema")
    if c["evidence_class"] != "revealed_historical_development_only":
        raise ValueError("unexpected evidence class")
    if file_sha256(ROOT / "src/exia/genetic_primus/rolling_linear_matrix_v1.py") != c["matrix_module_sha256"]:
        raise ValueError("model code changed after preregistration")
    if (c["feature_sets"] != {key: list(value) for key, value in FEATURE_SETS.items()}
            or c["variants"] != list(VARIANTS) or c["candidate_budget"] != 4
            or c["training_origin_count"] != 20 or c["inner_validation_origin_count"] != 10
            or c["origin_days"] != 3 or c["timeframe_minutes"] != 240
            or c["training_alpha_bonferroni"] != 0.05 / 4
            or c["safety"] != {key: False for key in (
                "credentials_allowed", "orders_enabled", "paper_allowed", "live_allowed",
                "promotion_authority", "runtime_authority", "network_allowed")}
            or c["september_used_for_selection"]):
        raise ValueError("bounded experiment/safety contract changed")
    return c


def _simulate(panel: pd.DataFrame, schedule: pd.DataFrame,
              symbols: tuple[str, ...], cost: float):
    return simulate_cash_targets(
        panel, schedule, symbols=symbols,
        start_timestamp=int(schedule.execution_timestamp.min()),
        end_timestamp=int(schedule.execution_timestamp.max()) + DT,
        timeframe_ms=DT, round_trip_cost_bps=cost, record_details=False)


def _summary(sim) -> dict:
    return {"net_return": float(sim.metrics["net_return"]),
            "max_drawdown": float(sim.metrics["max_drawdown"]),
            "trades": int(sim.metrics["trades"]),
            "fees": float(sim.metrics["total_fees"]),
            "daily_returns": daily_returns(sim).tolist()}


def _tail_summary(sim, boundary: int, days: int) -> dict:
    frame = sim.portfolio.loc[sim.portfolio.timestamp.gt(boundary)]
    returns = frame.net_return.to_numpy(float)
    if len(returns) != days * (86_400_000 // DT):
        raise ValueError("validation interval is not complete")
    daily = np.prod(1.0 + returns.reshape(days, -1), axis=1) - 1.0
    curve = np.r_[1.0, np.cumprod(1.0 + returns)]
    drawdown = float(np.max(1.0 - curve / np.maximum.accumulate(curve)))
    return {"net_return": float(curve[-1] - 1.0),
            "max_drawdown": drawdown, "daily_returns": daily.tolist(),
            "entry_state": "carried_from_training_simulation"}


def _ema_schedule(examples: pd.DataFrame, symbols: tuple[str, ...],
                  first: int, count: int) -> pd.DataFrame:
    schedules = []
    sha = object_sha({"policy": "EMA_long_at_origin_open_held_3d",
                     "source": "V7_EMA_TrendConsensus"})
    for index in range(count):
        origin = first + index * HORIZON_MS
        deploy = examples.loc[examples.origin_start.eq(origin)].set_index("symbol")
        if set(deploy.index) != set(symbols) or len(deploy) != len(symbols):
            raise ValueError("incomplete EMA reference universe")
        selected = tuple(s for s in symbols if bool(deploy.loc[s, "proposal_long"]))
        frame = pd.MultiIndex.from_product(
            [np.arange(origin, origin + HORIZON_MS, DT), symbols],
            names=["execution_timestamp", "symbol"]).to_frame(index=False)
        frame["signal_timestamp"] = origin
        frame["target"] = frame.symbol.isin(selected).astype(int)
        frame["model_sha256"] = sha
        schedules.append(frame)
    return pd.concat(schedules, ignore_index=True)


def _gate(evaluations: dict, bank: np.ndarray, alpha: float,
          baseline: dict | None = None, drawdown_limit: float | None = None) -> dict:
    conditions = {}
    lcbs = {}
    for cost_name in ("normal", "stress"):
        item = evaluations[cost_name]
        daily = np.asarray(item["daily_returns"], float)
        if daily.shape != (bank.shape[1],):
            raise ValueError("candidate/bank day mismatch")
        lcb_zero = paired_lcb(daily, np.zeros_like(daily), bank, alpha)
        lcbs[f"{cost_name}_vs_NoTrade"] = lcb_zero
        conditions[f"{cost_name}_net_positive"] = item["net_return"] > 0
        conditions[f"{cost_name}_lcb_vs_NoTrade_positive"] = lcb_zero > 0
        if baseline is not None:
            reference = np.asarray(baseline[cost_name]["daily_returns"], float)
            lcb_ema = paired_lcb(daily, reference, bank, alpha)
            lcbs[f"{cost_name}_vs_EMA"] = lcb_ema
            conditions[f"{cost_name}_lcb_vs_EMA_positive"] = lcb_ema > 0
        if drawdown_limit is not None:
            conditions[f"{cost_name}_drawdown_below_limit"] = item["max_drawdown"] <= drawdown_limit
    return {"alpha": alpha, "lcbs": lcbs, "conditions": conditions,
            "passed": all(conditions.values()),
            "rank_score": min(lcbs[k] for k in ("normal_vs_NoTrade", "stress_vs_NoTrade"))}


def _evaluate() -> dict:
    c = _config()
    manifest_path = ROOT / c["source_manifest_path"]
    if file_sha256(manifest_path) != c["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    contract_path = ROOT / c["source_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=contract_path, contract=contract)
    symbols = tuple(contract["universe"])
    start = _ms(c["development_start_utc"])
    train_end = start + c["training_origin_count"] * HORIZON_MS
    finish = train_end + c["inner_validation_origin_count"] * HORIZON_MS
    examples = build_matrix_examples(panel, symbols, _ms(c["history_start_utc"]), start, finish)
    fit_options = {"purge_bars": c["purge_bars"], "ridge_alpha": c["ridge_alpha"],
                   "round_trip_cost_bps": c["normal_round_trip_bps"],
                   "minimum_opportunities": c["minimum_train_opportunities"],
                   "minimum_ess": c["minimum_weight_ess"]}
    train_days = c["training_origin_count"] * c["origin_days"]
    bank = block_bank(train_days, c["bootstrap_samples"], c["bootstrap_block_days"],
                      c["training_bank_seed"])
    cost_map = {"gross_zero_cost": 0.0, "normal": c["normal_round_trip_bps"],
                "stress": c["stress_round_trip_bps"]}
    records = []
    for feature_set in c["feature_sets"]:
        for variant in c["variants"]:
            schedule, ledger = build_matrix_schedule(
                examples, symbols, start, train_end, feature_set, variant, **fit_options)
            for index in (0, len(ledger) - 1):
                origin = ledger[index]["origin_start"]
                prefix = examples.loc[examples.origin_start.le(origin)]
                checked = fit_matrix_at_origin(prefix, origin, feature_set, variant, **fit_options)
                if checked["model_sha256"] != ledger[index]["model"]["model_sha256"]:
                    raise AssertionError("future labels changed historical model")
            for item in ledger:
                model = item["model"]
                if (model["max_train_label_end_exclusive"] is not None
                        and model["max_train_label_end_exclusive"] > item["origin_start"] - c["purge_bars"] * DT):
                    raise AssertionError("purge violated")
            evaluated = {name: _summary(_simulate(panel, schedule, symbols, cost))
                         for name, cost in cost_map.items()}
            gate = _gate(evaluated, bank, c["training_alpha_bonferroni"])
            records.append({"feature_set": feature_set, "variant": variant,
                            "origin_models": ledger, "evaluations": evaluated,
                            "training_gate": gate})
    if len(records) != c["candidate_budget"]:
        raise AssertionError("unregistered candidate count")
    ema_train = _ema_schedule(examples, symbols, start, c["training_origin_count"])
    ema_reference = {name: _summary(_simulate(panel, ema_train, symbols, cost))
                     for name, cost in cost_map.items()}
    eligible = [r for r in records if r["training_gate"]["passed"]]
    winner = max(eligible, key=lambda r: r["training_gate"]["rank_score"]) if eligible else None
    result = {
        "schema_version": "exia.genetic_primus.rolling_linear_matrix_result/1",
        "evidence_class": c["evidence_class"], "config_sha256": file_sha256(CONFIG),
        "script_sha256": file_sha256(Path(__file__)),
        "matrix_module_sha256": c["matrix_module_sha256"],
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_snapshot_sha256": manifest["snapshot_sha256"],
        "source_contract_sha256": file_sha256(contract_path),
        "examples_count": len(examples), "examples_sha256": object_sha(examples.to_dict("records")),
        "train_bank_sha256": array_sha(bank), "candidate_budget": c["candidate_budget"],
        "search_evaluations": len(records), "records": records,
        "ema_training_reference": ema_reference,
        "selected_candidate": None if winner is None else {
            "feature_set": winner["feature_set"], "variant": winner["variant"]},
        "validation_result": None, "final_policy": "NoTrade",
        "automatic_promotion": False, "safety": dict(c["safety"]),
        "global_prior_trials_included_in_nominal_lcb": False,
    }
    if winner is None:
        return result
    feature_set, variant = winner["feature_set"], winner["variant"]
    full_schedule, full_ledger = build_matrix_schedule(
        examples, symbols, start, finish, feature_set, variant, **fit_options)
    train_schedule, _ = build_matrix_schedule(
        examples, symbols, start, train_end, feature_set, variant, **fit_options)
    if not full_schedule.iloc[:len(train_schedule)].reset_index(drop=True).equals(train_schedule):
        raise AssertionError("validation continuation changed training decisions")
    full_ema = _ema_schedule(
        examples, symbols, start, c["training_origin_count"] + c["inner_validation_origin_count"])
    val_days = c["inner_validation_origin_count"] * c["origin_days"]
    candidate = {name: _tail_summary(_simulate(panel, full_schedule, symbols, cost), train_end, val_days)
                 for name, cost in cost_map.items()}
    ema = {name: _tail_summary(_simulate(panel, full_ema, symbols, cost), train_end, val_days)
           for name, cost in cost_map.items()}
    val_bank = block_bank(val_days, c["bootstrap_samples"], c["bootstrap_block_days"],
                          c["validation_bank_seed"])
    gate = _gate(candidate, val_bank, c["validation_alpha"], ema, c["maximum_drawdown"])
    result["validation_result"] = {
        "candidate": candidate, "baseline_ema": ema,
        "origin_models": full_ledger[c["training_origin_count"]:],
        "bank_sha256": array_sha(val_bank), "gate": gate,
        "training_position_carried": True,
    }
    if gate["passed"]:
        result["final_policy"] = "linear_overlay_for_separate_future_audit_only"
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable matrix result already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "DISCLOSED_LINEAR_MATRIX_COMPLETE",
                      "train_lcbs": {f"{r['feature_set']}:{r['variant']}":
                                     round(r["training_gate"]["rank_score"], 8)
                                     for r in result["records"]},
                      "selected": result["selected_candidate"],
                      "validation_opened": result["validation_result"] is not None,
                      "final_policy": result["final_policy"]}, separators=(",", ":")))


if __name__ == "__main__":
    main()
