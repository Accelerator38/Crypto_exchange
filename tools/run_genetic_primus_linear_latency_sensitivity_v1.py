"""One-shot, train-only cash replay of frozen decisions at fixed delays/costs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.accounting_v2 import simulate_cash_targets
from exia.genetic_primus.latency_sensitivity_v1 import delay_target_schedule
from exia.genetic_primus.measurement_lab_v2 import object_sha
from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_manifest_v1 import load_verified_snapshot
from exia.genetic_primus.rolling_linear_retrain_v1 import DT, HORIZON_MS
from tools.diagnose_genetic_primus_linear_matrix_v1 import _schedule

CONFIG = ROOT / "configs/genetic_primus_linear_latency_sensitivity_v1.json"
CONFIG_SHA = "6f4f9d312aaf3e2c94ebc614abf5fdb31757454180ccb66e77c672f849ed7acf"
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/linear_latency_sensitivity_v1_20261002/result.json"


def _ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("UTC time required")
    return int(parsed.timestamp() * 1000)


def _config() -> dict:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("project Python required")
    if file_sha256(CONFIG) != CONFIG_SHA:
        raise ValueError("preregistered sensitivity config changed")
    c = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (c["schema_version"] != "exia.genetic_primus.linear_latency_sensitivity/1"
            or c["delay_bars"] != [0, 1]
            or c["round_trip_cost_bps"] != [16.0, 24.0, 40.0]
            or c["timeframe_minutes"] != 240 or c["origin_days"] != 3
            or not c["no_model_refits"] or not c["no_candidate_selection"]
            or c["validation_labels_used"]
            or c["safety"] != {key: False for key in (
                "credentials_allowed", "orders_enabled", "paper_allowed", "live_allowed",
                "promotion_authority", "runtime_authority", "network_allowed")}
            or file_sha256(ROOT / "src/exia/genetic_primus/latency_sensitivity_v1.py")
            != c["delay_module_sha256"]):
        raise ValueError("bounded sensitivity/safety contract changed")
    return c


def _ema_schedule(frozen: dict, symbols: tuple[str, ...], start: int,
                  end: int) -> pd.DataFrame:
    reference = next(r for r in frozen["records"]
                     if r["feature_set"] == "two" and r["variant"] == "uniform")
    if len(reference["origin_models"]) != (end - start) // HORIZON_MS:
        raise ValueError("missing frozen EMA origin records")
    sha = object_sha({"policy": "EMA_long_at_origin_open_held_3d",
                     "source": "V7_EMA_TrendConsensus"})
    frames = []
    for index, item in enumerate(reference["origin_models"]):
        origin = start + index * HORIZON_MS
        if item["origin_start"] != origin:
            raise ValueError("frozen origin order changed")
        decisions = item["decision"]["decisions"]
        if set(decisions) != set(symbols):
            raise ValueError("incomplete EMA origin")
        proposals = {symbol: int(decisions[symbol]["predicted_gross_3d"] is not None)
                     for symbol in symbols}
        frame = pd.MultiIndex.from_product(
            [np.arange(origin, origin + HORIZON_MS, DT), symbols],
            names=["execution_timestamp", "symbol"]).to_frame(index=False)
        frame["signal_timestamp"] = origin
        frame["target"] = frame.symbol.map(proposals).astype(int)
        frame["model_sha256"] = sha
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _no_trade_schedule(symbols: tuple[str, ...], start: int, end: int) -> pd.DataFrame:
    frame = pd.MultiIndex.from_product(
        [np.arange(start, end, DT), symbols],
        names=["execution_timestamp", "symbol"]).to_frame(index=False)
    frame["signal_timestamp"] = start
    frame["target"] = 0
    frame["model_sha256"] = object_sha({"policy": "NoTrade", "scope": "frozen_train"})
    return frame


def _summary(sim, schedule: pd.DataFrame) -> dict:
    return {"net_return": float(sim.metrics["net_return"]),
            "max_drawdown": float(sim.metrics["max_drawdown"]),
            "closed_trades": int(sim.metrics["trades"]),
            "total_fees": float(sim.metrics["total_fees"]),
            "target_exposure_fraction": float(schedule.target.mean())}


def _verify_frozen(policy: str, cost: float, summary: dict,
                   frozen: dict, record: dict | None) -> None:
    if cost not in (16.0, 24.0):
        return
    cost_name = "normal" if cost == 16.0 else "stress"
    if policy == "NoTrade":
        expected = {"net_return": 0.0, "trades": 0, "total_fees": 0.0}
    elif policy == "EMA_TrendConsensus":
        expected = frozen["ema_training_reference"][cost_name]
    else:
        if record is None:
            raise AssertionError("missing frozen candidate")
        expected = record["evaluations"][cost_name]
    for actual_name, frozen_name in (("net_return", "net_return"),
                                     ("closed_trades", "trades"),
                                     ("total_fees", "fees")):
        if policy == "NoTrade" and frozen_name == "fees":
            frozen_name = "total_fees"
        if abs(summary[actual_name] - expected[frozen_name]) > 1e-10:
            raise AssertionError(f"zero-delay {policy}/{cost} differs from frozen result")


def _evaluate() -> dict:
    c = _config()
    source_path = ROOT / c["source_result_path"]
    if file_sha256(source_path) != c["source_result_sha256"]:
        raise ValueError("frozen result changed")
    frozen = json.loads(source_path.read_text(encoding="utf-8"))
    if (frozen["config_sha256"] != file_sha256(ROOT / c["source_config_path"])
            or frozen["selected_candidate"] is not None
            or frozen["validation_result"] is not None
            or frozen["final_policy"] != "NoTrade"):
        raise ValueError("source is not the failed training attempt")
    manifest_path = ROOT / c["source_manifest_path"]
    if file_sha256(manifest_path) != c["source_manifest_sha256"]:
        raise ValueError("source manifest changed")
    contract_path = ROOT / c["source_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=contract_path, contract=contract)
    if manifest["snapshot_sha256"] != frozen["source_snapshot_sha256"]:
        raise ValueError("source snapshot changed")
    symbols = tuple(contract["universe"])
    start, end = _ms(c["train_start_utc"]), _ms(c["train_end_exclusive_utc"])
    if end - start != 20 * HORIZON_MS:
        raise ValueError("not the registered 20-origin train window")
    panel = panel.loc[panel.timestamp.ge(start) & panel.timestamp.lt(end)].copy()
    if panel.timestamp.max() != end - DT:
        raise ValueError("unexpected terminal OHLCV candle")
    candidate_records = {
        f"linear_{r['feature_set']}_{r['variant']}": r for r in frozen["records"]}
    base = {"EMA_TrendConsensus": _ema_schedule(frozen, symbols, start, end),
            **{name: _schedule(r["origin_models"], symbols)
               for name, r in candidate_records.items()},
            "NoTrade": _no_trade_schedule(symbols, start, end)}
    if list(base) != c["policies"]:
        raise ValueError("registered policy enumeration changed")
    reports = []
    for policy, original in base.items():
        delay_results = {}
        for bars in c["delay_bars"]:
            schedule = delay_target_schedule(original, symbols, start, end, bars)
            by_cost = {}
            for cost in c["round_trip_cost_bps"]:
                sim = simulate_cash_targets(
                    panel, schedule, symbols=symbols,
                    start_timestamp=start, end_timestamp=end,
                    timeframe_ms=DT, round_trip_cost_bps=cost, record_details=False)
                summary = _summary(sim, schedule)
                if bars == 0:
                    _verify_frozen(policy, cost, summary, frozen,
                                   candidate_records.get(policy))
                by_cost[str(int(cost))] = summary
            delay_results[str(bars)] = {
                "schedule_sha256": object_sha(schedule.to_dict("records")),
                "cost_results": by_cost}
        delta = {str(int(cost)): float(
            delay_results["1"]["cost_results"][str(int(cost))]["net_return"]
            - delay_results["0"]["cost_results"][str(int(cost))]["net_return"])
            for cost in c["round_trip_cost_bps"]}
        reports.append({"policy": policy, "delays": delay_results,
                        "one_bar_minus_zero_bar_net_return": delta})
    return {"schema_version": "exia.genetic_primus.linear_latency_sensitivity_result/1",
            "evidence_class": c["evidence_class"],
            "config_sha256": file_sha256(CONFIG),
            "script_sha256": file_sha256(Path(__file__)),
            "source_result_sha256": file_sha256(source_path),
            "source_snapshot_sha256": frozen["source_snapshot_sha256"],
            "train_start_utc": c["train_start_utc"],
            "train_end_exclusive_utc": c["train_end_exclusive_utc"],
            "policies": reports, "zero_delay_reconciled": True,
            "new_model_fits": 0, "candidate_count_added": 0,
            "validation_labels_used": False,
            "interpretation": c["interpretation"],
            "final_policy": "NoTrade", "promotion_authority": False,
            "network_allowed": False}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("immutable sensitivity result already exists")
    result = _evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "FROZEN_LATENCY_SENSITIVITY_COMPLETE",
                      "zero_delay_reconciled": result["zero_delay_reconciled"],
                      "policies": len(result["policies"]),
                      "validation_labels_used": False}, separators=(",", ":")))


if __name__ == "__main__":
    main()
