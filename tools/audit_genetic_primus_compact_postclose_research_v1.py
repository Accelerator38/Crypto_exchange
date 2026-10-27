"""Offline causal regressions and descriptive matched-ledger frozen-model controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import run_genetic_primus_compact_postclose_research_v1 as b
import numpy as np

LEGACY = b.BASE / "compact_forecast_v1_20261007_r2"


def paired_delta(left, right):
    a, c = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != c.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ValueError("finite matched daily vectors required")
    return a - c


def checks():
    reg, bank, _, _ = b.load()
    sample = {k[7:]: v for k, v in bank.items() if k.startswith("sample_")}
    passed = 0
    def check(value):
        nonlocal passed
        if not value:
            raise AssertionError("causal regression " + str(passed + 1))
        passed += 1
    start = reg["contract"]["report_start_ms"]
    p = int(np.searchsorted(bank["times"], start))
    # Full causal EMA prefix, not the last 49-bar reseeded window.
    prefix = b.d.feature_panel(bank["times"][:p + 1] - b.r.BAR_MS, bank["closes"][:p + 1])
    check(np.array_equal(prefix[-1], bank["features"][p]))
    changed = bank["closes"].copy()
    changed[p + 1:] *= np.linspace(2, 4, len(changed) - p - 1)[:, None]
    changed_features = b.d.feature_panel(bank["times"] - b.r.BAR_MS, changed)
    check(np.array_equal(changed_features[p], prefix[-1]))
    labels_entry = np.searchsorted(bank["times"], sample["origins"])
    labels_exit = np.searchsorted(bank["times"], sample["available"])
    check(np.array_equal(sample["available"] - sample["origins"], np.full(len(labels_entry), b.r.HORIZON_MS)))
    check(np.array_equal(sample["y"], bank["closes"][labels_exit] / bank["closes"][labels_entry] - 1))
    check(reg["report_origins_ms"][-1] + b.r.HORIZON_MS > reg["contract"]["report_end_ms"])
    check(reg["report_origins_ms"][-1] not in sample["origins"])
    for stage, origins in (("cal", reg["calibration_origins_ms"]), ("report", reg["report_origins_ms"])):
        for index, origin in enumerate(origins):
            record = b.verify_body(b.read(b.OUT / (stage + "_origin_" + str(index).zfill(2) + ".json")))
            temporal = b.r.split(sample, origin)
            t, inner, validation = (temporal[k] for k in ("train", "inner", "validation"))
            tx, ty = sample["x"][t].reshape(-1, 6), sample["y"][t].reshape(-1)
            check(np.all(sample["available"][t] <= origin - 8 * b.r.BAR_MS))
            check(np.all(sample["available"][inner] <= int(sample["origins"][validation[0]]) - 8 * b.r.BAR_MS))
            check(b.m.array_identity(sample["origins"][t], sample["available"][t], tx, ty) == record["training_identity"]["train_samples_sha256"])
            check(b.m.array_identity(sample["origins"][inner], sample["origins"][validation]) == record["training_identity"]["folds_sha256"])
            pos = int(np.searchsorted(bank["times"], origin))
            check(record["feature_prefix_sha256"] == b.m.array_identity(bank["times"][:pos + 1], bank["closes"][:pos + 1]))
            for arm in record["arms"].values():
                predictions = []
                for model in arm["models"]:
                    b.verify_body(model)
                    check(np.allclose(model["scaler"]["mean"], tx.mean(axis=0), rtol=0, atol=1e-12))
                    check(abs(model["scaler"]["target_mean"] - ty.mean()) < 1e-12)
                    predictions.append(b.m.predict(model, bank["features"][pos]))
                check(np.array_equal(np.mean(predictions, axis=0), np.asarray(arm["prediction"])))
                check(arm["positive_weight_mass"] > 0 and arm["negative_weight_mass"] > 0)
    b.write("causal_regressions.json", {"checks_passed": passed, "training_fits": 0,
                                        "registered_code_pin": b.pin(Path(__file__)), "future_perturbation_invariant": True,
                                        "all_33_origin_scalers_and_predictions_verified": True,
                                        "network_used": False, "safety": b.r.SAFETY})
    print(json.dumps({"causal_checks_passed": passed, "verified_origins": 33, "additional_fits": 0}))


def control_registration():
    reg, _, _, _ = b.load()
    path = b.OUT / "legacy_control_registration.json"
    if not path.exists():
        source_pins = {b.relative(LEGACY / ("origin_" + str(i).zfill(2) + ".json")):
                       b.pin(LEGACY / ("origin_" + str(i).zfill(2) + ".json")) for i in range(20)}
        value = {"schema": "frozen_models_matched_ledger_control/1", "source_pins": source_pins,
                 "code_pin": b.pin(Path(__file__)), "registration_sha256": reg["sha256"],
                 "matched_thresholds": "same_new_pre_august_threshold_for_each_architecture_and_weighting",
                 "last_origin": "Sept27_frozen_model_predicts_Sept30_features_no_new_legacy_fit",
                 "registered_after_new_results_known": True, "descriptive_only_not_new_validation": True,
                 "selection_or_promotion_authority": False, "safety": b.r.SAFETY}
        b.write(path.name, value)
    value = b.verify_body(b.read(path))
    if value["code_pin"] != b.pin(Path(__file__)) or value["registration_sha256"] != reg["sha256"]:
        raise ValueError("control registration changed")
    for name, expected in value["source_pins"].items():
        if b.pin(b.ROOT / name) != expected:
            raise ValueError("legacy model source changed")
    return value


def control(policy, scenario):
    reg, bank, _, rates = b.load()
    metadata = control_registration()
    cal = b.verify_body(b.read(b.OUT / "calibration.json"))
    threshold = cal["selected_thresholds_bps"][policy]
    decisions = []
    for index, origin in enumerate(reg["report_origins_ms"]):
        old = b.verify_body(b.read(LEGACY / ("origin_" + str(min(index, 19)).zfill(2) + ".json")))
        arm = old["arms"][policy]
        position = int(np.searchsorted(bank["times"], origin))
        predictions = np.mean([b.m.predict(v, bank["features"][position]) for v in arm["models"]], axis=0)
        if index < 20 and (old["origin_ms"] != origin or not np.allclose(predictions, arm["prediction"], rtol=0, atol=1e-12)):
            raise ValueError("legacy prefix/prediction compatibility failure")
        decisions.append({"origin_ms": origin, "targets": np.where(predictions * 10000 > threshold, 0.1, 0).tolist(),
                          "model_sha256": b.m.identity([v["sha256"] for v in arm["models"]]),
                          "scaler_sha256": old["scaler_sha256"], "feature_prefix_sha256": b.m.array_identity(bank["times"][:position + 1], bank["closes"][:position + 1]),
                          "training_identity_sha256": b.m.identity(old["training_identity"]), "threshold_bps": threshold,
                          "legacy_training_origin_ms": old["origin_ms"]})
    c = reg["contract"]
    result = b.r.replay(times=bank["times"], prices=bank["closes"], start=c["report_start_ms"], end=c["report_end_ms"],
                        symbols=b.SYMBOLS, decisions=decisions, rates=rates, fee_bps=c["taker_fee_bps"],
                        drag_bps=c["adverse_price_bps"][scenario])
    result.update(control_registration_sha256=metadata["sha256"], policy=policy, scenario=scenario,
                  threshold_bps=threshold, legacy_fits=0, descriptive_only=True)
    b.write("legacy_control_" + policy + "_" + scenario + ".json", result)
    print(json.dumps({"control": policy, "scenario": scenario, "net_pct": round(float(result["metrics"]["net_return"]) * 100, 4),
                      "DD_pct": round(float(result["metrics"]["max_drawdown"]) * 100, 4), "new_fits": 0}))


def finish():
    reg, _, banks, _ = b.load()
    control_reg = control_registration()
    old = b.read(b.OLD_RESULTS)
    rows = {}
    for policy in b.r.ARMS:
        row = {"old_report_reference_only": {scenario: {
                "net_return": float(old["policies"][policy]["metrics"][scenario]["net_return"]),
                "drawdown": float(old["policies"][policy]["metrics"][scenario]["max_drawdown"])}
                for scenario in ("normal", "stress")}, "matched_controls": {}}
        for scenario in ("normal", "stress"):
            incumbent = b.verify_body(b.read(b.OUT / ("legacy_control_" + policy + "_" + scenario + ".json")))
            new = b.verify_body(b.read(b.OUT / ("replay_" + policy + "_" + scenario + ".json")))
            delta = paired_delta(new["daily_log_returns"], incumbent["daily_log_returns"])
            paired = b.r.audit(delta, banks["report_selection"], banks["report_audit"])
            row["matched_controls"][scenario] = {"legacy_metrics": incumbent["metrics"], "new_metrics": new["metrics"],
                "net_return_difference_pp": (new["metrics"]["net_return"] - incumbent["metrics"]["net_return"]) * 100,
                "paired_daily_log_increment_audit": paired}
        rows[policy] = row
    b.write("legacy_comparison.json", {"rows": rows, "registration_sha256": reg["sha256"],
                                       "control_registration_sha256": control_reg["sha256"],
                                       "old_report_is_not_paired": True, "controls_are_descriptive_only": True,
                                       "legacy_control_replays": 8, "legacy_new_fits": 0,
                                       "successful_main_replays": 24, "repeated_calibration_replays_after_export_failure": 12,
                                       "total_replays_including_failed_export_and_controls": 44,
                                       "safety": b.r.SAFETY})
    print(json.dumps({k: {"delta_stress_pp": round(v["matched_controls"]["stress"]["net_return_difference_pp"], 4),
                         "paired_LCB_bps_day": round(v["matched_controls"]["stress"]["paired_daily_log_increment_audit"]["audit"]["LCB_mean_log_bps_per_day"], 4)} for k, v in rows.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("checks", "control", "finish"))
    parser.add_argument("--policy", choices=b.r.ARMS)
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    try:
        if args.step == "checks":
            checks()
        elif args.step == "control":
            control(args.policy, args.scenario)
        else:
            finish()
    except Exception as error:
        print((type(error).__name__ + ": " + str(error))[:600], file=sys.stderr)
        sys.exit(1)
