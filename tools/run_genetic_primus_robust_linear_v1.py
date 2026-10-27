"""Fixed robust-loss experiment; one bounded offline foreground command."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import run_genetic_primus_label_phase_v1 as phase
from exia.genetic_primus import robust_linear_v1 as h
c, base = h.c, phase.parent
OUT = base.BASE / "robust_linear_v1_20261012"
CONTRACT = ROOT / "configs/genetic_primus_robust_linear_v1.json"
CODE = (Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/robust_linear_v1.py")
POLICIES = ("B", "H", "A")


def sealed(path):
    return base.verify_body(base.read(path))


def write(name, value):
    value = c.plain(value)
    result = value | {"sha256": c.m.identity(value)}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return result


def verify_parents(phase_reg, old_reg):
    phase.full_parent_verification(old_reg)
    receipt = sealed(phase.OUT / "receipt.json")
    if receipt["registration_sha256"] != phase_reg["sha256"]:
        raise ValueError("diagnostic parent registration mismatch")
    for name, expected in receipt["artifact_pins"].items():
        if base.pin(phase.OUT / name) != expected:
            raise ValueError("diagnostic parent artifact changed: " + name)
    provenance = sealed(phase.OUT / "influence_diagnostic_provenance.json")
    if base.pin(ROOT / provenance["source_path"]) != provenance["source_pin"]:
        raise ValueError("diagnostic source changed")
    return receipt


def fit_name(period, index):
    return f"fit_{period}_index_{index:03}.json"


def prepare():
    if OUT.exists():
        raise ValueError("never overwrite or restart an existing experiment")
    phase_reg, old_reg, _, samples, _, _ = phase.load()
    receipt = verify_parents(phase_reg, old_reg)
    contract = base.read(CONTRACT)
    for key in ("architecture", "weighting", "l2", "threshold_bps", "fee_bps", "drag_bps"):
        if contract[key] != old_reg["contract"][key]:
            raise ValueError("control setting changed: " + key)
    expected = {"label_horizon_days": 3, "label_stride_days": 1, "decision_cadence_days": 3,
                "retrain_cadence_days": 3, "huber_delta_train_target_std": 1.345,
                "max_iterations": 100, "parameter_tolerance": 1e-10, "gradient_tolerance": 1e-8,
                "effective_weights_renormalized": False, "max_fits_per_command": 7,
                "new_model_fits": 50, "new_scaler_fits": 50, "new_financial_replays": 4}
    if any(contract[k] != v for k, v in expected.items()) or contract["safety"] != c.SAFETY:
        raise ValueError("fixed research contract required")
    plans = {}
    for period, settings in old_reg["periods"].items():
        grid = c.Clocks(3, 1, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        plans[period] = [phase.fold_plan(samples["B"], int(t)) for t in grid]
    if sum(map(len, plans.values())) != 50:
        raise ValueError("bounded fit budget mismatch")
    OUT.mkdir()
    write("folds.json", {"plans": plans, "safety": c.SAFETY})
    reg = write("registration.json", {"schema": "robust_linear_registration/1", "contract": contract,
        "periods": old_reg["periods"], "symbols": old_reg["symbols"], "before_model_fits": True,
        "parent_registration_sha256": old_reg["sha256"], "phase_registration_sha256": phase_reg["sha256"],
        "phase_registration_pin": base.pin(phase.OUT / "registration.json"),
        "phase_receipt_pin": base.pin(phase.OUT / "receipt.json"),
        "parent_code_and_sources_verified": True, "parent_artifacts_verified": 254,
        "phase_artifacts_verified": len(receipt["artifact_pins"]),
        "sample_bank_content_sha256": old_reg["sample_content_sha256"]["1"],
        "code_pins": {base.relative(f): base.pin(f) for f in CODE}, "folds_pin": base.pin(OUT / "folds.json"),
        "new_fit_budget": 50, "new_replay_budget": 4, "selection_or_promotion": False,
        "independent_validation_claim_allowed": False, "network_used": False, "safety": c.SAFETY})
    write("objective_registry_entry.json", {"registration_sha256": reg["sha256"],
        "new_candidate": "H_fixed_Huber_1.345_linear6", "direct_control": "B_daily_label_ridge",
        "context_not_direct_comparison": "A_3day_labels", "all_results_will_be_reported": True,
        "global_historical_registry_complete": False, "DSR_PBO_claim_allowed": False, "safety": c.SAFETY})
    print(json.dumps({"prepared": True, "new_fits": 50, "new_replays": 4, "phase_artifacts_verified": len(receipt["artifact_pins"])}))


def load():
    reg = sealed(OUT / "registration.json")
    for name, expected in reg["code_pins"].items():
        if base.pin(ROOT / name) != expected:
            raise ValueError("registered code changed: " + name)
    if base.pin(OUT / "folds.json") != reg["folds_pin"]:
        raise ValueError("registered folds changed")
    if (base.pin(phase.OUT / "registration.json") != reg["phase_registration_pin"]
            or base.pin(phase.OUT / "receipt.json") != reg["phase_receipt_pin"]):
        raise ValueError("diagnostic parent seal changed")
    phase_reg, old_reg, data, samples, banks, rates = phase.load()
    if phase_reg["sha256"] != reg["phase_registration_sha256"] or old_reg["sha256"] != reg["parent_registration_sha256"]:
        raise ValueError("parent registration changed")
    return reg, phase_reg, old_reg, data, samples["B"], banks, rates


def control_item(period, index):
    return phase.model_item(period, "B", index)[0]


def train(period, start_index, count):
    reg, _, _, data, samples, _, _ = load()
    if start_index < 0 or not 1 <= count <= 7:
        raise ValueError("bounded training slice required")
    plans = sealed(OUT / "folds.json")["plans"][period]
    selected = plans[start_index:start_index + count]
    if len(selected) != count or any((OUT / fit_name(period, j)).exists() for j in range(start_index, start_index + count)):
        raise ValueError("fit range absent or already complete; do not repeat")
    maximum_iterations = 0
    for index, plan in enumerate(selected, start_index):
        control = control_item(period, index)
        result = h.train_origin(samples, plan["fit_ms"], control, feature_prefix_sha=base.prefix(data, plan["fit_ms"])[1])
        for key in ("train_end_ms", "cutoff_ms", "inner_cutoff_ms", "train_samples_sha256", "folds_sha256"):
            if result["training_identity"][key] != plan[key]:
                raise ValueError("registered train/fold mismatch")
        result.update(registration_sha256=reg["sha256"], period=period, policy="H", control_model_sha256=control["model"]["sha256"])
        write(fit_name(period, index), result)
        maximum_iterations = max(maximum_iterations, result["solver"]["iterations"])
    print(json.dumps({"period": period, "start_index": start_index, "fits": count, "max_IRLS_iterations": maximum_iterations}))


def decisions(reg, data, period):
    settings = reg["periods"][period]
    rows = []
    for index, t in enumerate(c.Clocks(3, 1, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])):
        item = sealed(OUT / fit_name(period, index))
        model = base.verify_body(item["model"])
        if item["registration_sha256"] != reg["sha256"] or item["fit_ms"] != t or item["assumed_model_ready_us"] > t * 1000 + 100000:
            raise ValueError("causal model availability mismatch")
        pos, prefix_sha = base.prefix(data, t)
        pred = c.m.predict(model, data["features"][pos])
        rows.append({"origin_ms": t, "policy": "H", "targets": np.where(pred * 10000 > 20, .1, 0).tolist(),
            "fit_ms": t, "model_age_days": 0, "fit_artifact": fit_name(period, index), "model_sha256": model["sha256"],
            "scaler_sha256": item["scaler_sha256"], "training_identity_sha256": c.m.identity(item["training_identity"]),
            "cutoff_ms": item["training_identity"]["cutoff_ms"], "prediction": pred.tolist(),
            "feature_prefix_sha256": prefix_sha, "decision_feature_sha256": c.m.array_identity(data["features"][pos])})
    return rows


def replay_one(period, scenario):
    reg, _, _, data, _, _, rates = load()
    name = f"replay_{period}_H_{scenario}.json"
    if (OUT / name).exists():
        raise ValueError("replay already exists; do not repeat")
    settings = reg["periods"][period]
    result = c.replay(times=data["times"], prices=data["closes"], start=settings["start_ms"], end=settings["end_ms"],
        symbols=reg["symbols"], decisions=decisions(reg, data, period), rates=rates,
        fee_bps=6, drag_bps=reg["contract"]["drag_bps"][scenario], decision_cadence_days=3)
    result.update(registration_sha256=reg["sha256"], period=period, policy="H", scenario=scenario,
                  loss="fixed_Huber_1.345", execution_is_assumed=True)
    write(name, result)
    metric = result["metrics"]
    print(json.dumps({"period": period, "scenario": scenario, "net_pct": round(metric["net_return"] * 100, 4),
                      "DD_pct": round(metric["max_drawdown"] * 100, 4), "fills": metric["fills"]}))


def replay_item(period, policy, scenario):
    if policy == "H":
        return sealed(OUT / f"replay_{period}_H_{scenario}.json")
    return sealed(base.OUT / f"replay_{period}_{policy}_{scenario}.json")


def checks():
    load()
    result = h.direct_checks()
    write("direct_checks.json", result)
    print(json.dumps({"checks_passed": result["checks_passed"], "real_data_fits": 0, "collector": False}))


def audit():
    reg, _, _, data, samples, _, _ = load()
    total = 0
    def check(value, label):
        nonlocal total
        if not value:
            raise AssertionError(label)
        total += 1
    for period, settings in reg["periods"].items():
        grid = c.Clocks(3, 1, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        for j, t in enumerate(grid):
            item, control = sealed(OUT / fit_name(period, j)), control_item(period, j)
            split = c.old.split(samples, t)
            tx = samples["x"][split["train"]].reshape(-1, 6)
            ty = samples["y"][split["train"]].reshape(-1)
            model, scaler = item["model"], item["model"]["scaler"]
            check(item["fit_ms"] == t, "fixed fit schedule")
            check(item["training_identity"]["cutoff_ms"] == t - 8 * c.BAR_MS, "fixed purge")
            check(np.all(samples["available"][split["train"]] <= t - 8 * c.BAR_MS), "mature labels")
            check(item["model"]["scaler"] == control["model"]["scaler"], "same train-only scaler")
            check(np.array_equal(scaler["mean"], tx.mean(axis=0)), "causal mean")
            check(scaler["target_mean"] == float(ty.mean()), "all outcomes scaler")
            for key in control["training_identity"]:
                check(item["training_identity"][key] == control["training_identity"][key], "same control identity " + key)
            check(item["feature_prefix_sha256"] == base.prefix(data, t)[1], "causal feature prefix")
            base.verify_body(model)
            z = np.clip((tx - scaler["mean"]) / scaler["scale"], -scaler["clip"], scaler["clip"])
            target = (ty - scaler["target_mean"]) / scaler["target_scale"]
            design = np.column_stack((z, np.ones(len(z))))
            par = np.asarray(model["parameters"])
            residual = target - design @ par
            multipliers = h.residual_multipliers(residual, 1.345)
            gradient = -design.T @ np.clip(residual, -1.345, 1.345) / len(target) + np.r_[.01 * par[:6], 0]
            check(np.max(abs(gradient)) <= 1e-8, "Huber stationarity")
            check(c.m.array_identity(multipliers) == item["robust_weight_sha256"], "effective weight SHA")
            check(np.all(multipliers > 0) and np.all(multipliers <= 1), "no label excluded")
            check(np.any(ty[multipliers < 1] > 0) and np.any(ty[multipliers < 1] < 0), "both signs downweighted")
            check(item["solver"]["converged"] and item["solver"]["iterations"] <= 100, "bounded convergence")
            check(np.all(np.diff(item["solver"]["objective_history"]) <= 1e-12), "monotone objective")
            check(abs(h.objective(design, target, par, 1.345, .01) - item["solver"]["final_objective"]) < 1e-12, "stored objective")
            check(not item["solver"]["weights_renormalized"], "fixed regularization")
            check(item["safety"] == c.SAFETY, "no authority")
        for scenario in ("normal", "stress"):
            item, control = replay_item(period, "H", scenario), replay_item(period, "B", scenario)
            check([v["origin_ms"] for v in item["decisions"]] == grid, "fixed decisions")
            check(abs(phase.p.bar_components(item, data["times"], data["closes"], reg["symbols"])["reconciliation_error"]) < 1e-7, "cash decomposition")
            check(abs(phase.p.attribute(control, item, data["times"], data["closes"], reg["symbols"])["reconciliation_error"]) < 1e-7, "paired dollar attribution")
            for j, row in enumerate(item["decisions"]):
                model = sealed(OUT / fit_name(period, j))["model"]
                pos = int(np.searchsorted(data["times"], row["origin_ms"]))
                pred = c.m.predict(model, data["features"][pos])
                check(np.array_equal(pred, row["prediction"]), "stored causal prediction")
                check(np.array_equal(np.where(pred * 10000 > 20, .1, 0), row["targets"]), "same strict threshold")
                check(row["model_sha256"] == model["sha256"], "decision model SHA")
    check(len(list(OUT.glob("fit_*.json"))) == 50, "fit budget")
    check(len(list(OUT.glob("replay_*.json"))) == 4, "replay budget")
    write("audit_checks.json", {"checks_passed": total, "models_checked": 50, "additional_fits": 0,
                                 "same_scalers_and_labels_as_B": True, "safety": c.SAFETY})
    print(json.dumps({"audit_checks_passed": total, "models_checked": 50, "additional_fits": 0}))


def diagnostics():
    reg, _, _, data, _, _, _ = load()
    training, predictions, summaries = [], [], {}
    for period, settings in reg["periods"].items():
        grid = c.Clocks(3, 1, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        for j, t in enumerate(grid):
            robust, control = sealed(OUT / fit_name(period, j)), control_item(period, j)
            training.append({"period": period, "fit_ms": t, "solver": robust["solver"],
                "weight_diagnostics": robust["weight_diagnostics"], "huber_threshold_raw_bps": robust["huber_threshold_raw_bps"],
                "B_coefficients": phase.p.coefficient_view(control["model"]),
                "H_coefficients": phase.p.coefficient_view(robust["model"]), "same_scaler_sha256": robust["scaler_sha256"]})
            pos = int(np.searchsorted(data["times"], t))
            b, forecast = (c.m.predict(v["model"], data["features"][pos]) * 10000 for v in (control, robust))
            available = t + 3 * c.DAY_MS <= settings["end_ms"]
            end_pos = int(np.searchsorted(data["times"], t + 3 * c.DAY_MS)) if available else None
            for i, symbol in enumerate(reg["symbols"]):
                label = float((data["closes"][end_pos, i] / data["closes"][pos, i] - 1) * 10000) if available else None
                predictions.append({"period": period, "origin_ms": t, "symbol": symbol, "B_forecast_bps": float(b[i]),
                    "H_forecast_bps": float(forecast[i]), "B_margin_bps": float(b[i] - 20), "H_margin_bps": float(forecast[i] - 20),
                    "B_long": bool(b[i] > 20), "H_long": bool(forecast[i] > 20), "full_horizon_label_bps": label,
                    "B_model_sha256": control["model"]["sha256"], "H_model_sha256": robust["model"]["sha256"]})
        rows = [v for v in predictions if v["period"] == period]
        train = [v for v in training if v["period"] == period]
        complete = [v for v in rows if v["full_horizon_label_bps"] is not None]
        summaries[period] = {"decision_asset_cells": len(rows), "full_horizon_cells": len(complete),
            "action_disagreements": sum(v["B_long"] != v["H_long"] for v in rows),
            "mean_absolute_forecast_change_bps": float(np.mean([abs(v["H_forecast_bps"] - v["B_forecast_bps"]) for v in rows])),
            "zero_forecast_MAE_bps": float(np.mean([abs(v["full_horizon_label_bps"]) for v in complete])),
            "mean_downweighted_fraction": float(np.mean([v["weight_diagnostics"]["downweighted_fraction"] for v in train])),
            "mean_effective_weight_multiplier": float(np.mean([v["weight_diagnostics"]["mean_multiplier"] for v in train])),
            "min_effective_weight_multiplier": min(v["weight_diagnostics"]["min_multiplier"] for v in train),
            "mean_positive_mass": float(np.mean([v["weight_diagnostics"]["positive_outcome_normalized_mass"] for v in train])),
            "mean_negative_mass": float(np.mean([v["weight_diagnostics"]["negative_outcome_normalized_mass"] for v in train])),
            "max_IRLS_iterations": max(v["solver"]["iterations"] for v in train), "policies": {}}
        for policy in ("B", "H"):
            coeff = np.asarray([v[f"{policy}_coefficients"]["slopes_bps_per_one_train_std"] for v in train])
            margins = [v[f"{policy}_margin_bps"] for v in rows]
            summaries[period]["policies"][policy] = {
                "long_target_fraction": float(np.mean([v[f"{policy}_long"] for v in rows])),
                "near_threshold_10bps_fraction": float(np.mean(np.abs(margins) <= 10)),
                "forecast_MAE_bps": float(np.mean([abs(v[f"{policy}_forecast_bps"] - v["full_horizon_label_bps"]) for v in complete])),
                "forecast_bps": phase.p.describe([v[f"{policy}_forecast_bps"] for v in rows]),
                "coefficient_mean_bps_per_train_std": coeff.mean(axis=0).tolist(),
                "coefficient_std_across_origins": coeff.std(axis=0).tolist(),
                "coefficient_positive_fraction": (coeff > 0).mean(axis=0).tolist(),
                "coefficient_sign_changes_across_origins": (np.sign(coeff[1:]) != np.sign(coeff[:-1])).sum(axis=0).tolist()}
    write("diagnostics.json", {"training": training, "predictions": predictions, "period_summary": summaries,
        "multi_phase_Huber_stability_not_tested": True, "safety": c.SAFETY})
    print(json.dumps({"training_snapshots": len(training), "paired_prediction_cells": len(predictions), "new_fits": 0}))


def summarize():
    reg, _, _, data, _, all_banks, _ = load()
    result = {"periods": {}, "direct_comparison": "H_minus_B", "A_context_only": True,
              "new_fits": 50, "new_replays": 4, "default_deployment": "NoTrade",
              "promotion_authority": False, "independent_validation_claim_allowed": False,
              "registration_sha256": reg["sha256"], "safety": c.SAFETY}
    for period, settings in reg["periods"].items():
        banks = {k.removeprefix(period + "_"): v for k, v in all_banks.items() if k.startswith(period + "_")}
        policies = {}
        for scenario in ("normal", "stress"):
            control = replay_item(period, "B", scenario)
            for policy in POLICIES:
                item = replay_item(period, policy, scenario)
                bars = phase.p.bar_components(item, data["times"], data["closes"], reg["symbols"])
                daily = np.asarray(item["daily_log_returns"])
                windows = np.asarray([daily[j:j + 3].sum() for j in range(0, len(daily) - 2, 3)])
                best = int(np.argmax(windows))
                entry = {"metrics": item["metrics"], "bootstrap_vs_B": c.bootstrap(daily, banks, paired=control["daily_log_returns"]),
                    "net_return_difference_vs_B_pp": (item["metrics"]["net_return"] - control["metrics"]["net_return"]) * 100,
                    "best_full_3d_window_start_ms": settings["start_ms"] + best * 3 * c.DAY_MS,
                    "best_full_3d_window_return": float(np.expm1(windows[best])),
                    "without_best_full_3d_log_window_return": float(np.expm1(daily.sum() - windows[best])),
                    "positive_full_3d_windows": int(np.sum(windows > 0)), "full_3d_windows": len(windows),
                    "symbol_concentration": phase.p.concentration(reg["symbols"], bars["net"].sum(axis=0)),
                    "date_concentration": phase.p.concentration([phase.day_key(t, settings["end_ms"]) for t in bars["times"]], bars["net"].sum(axis=1))}
                if policy == "H":
                    attr = phase.p.attribute(control, item, data["times"], data["closes"], reg["symbols"])
                    rows = attr["rows"]
                    attr["by_symbol"] = phase.p.concentration([v["symbol"] for v in rows], [v["delta_usdt"] for v in rows])
                    attr["by_decision_origin"] = phase.p.concentration([v["origin_ms"] for v in rows], [v["delta_usdt"] for v in rows])
                    attr["largest_decision_asset_differences"] = sorted(rows, key=lambda v: abs(v["delta_usdt"]), reverse=True)[:12]
                    write(f"attribution_{period}_H_vs_B_{scenario}.json", attr | {"safety": c.SAFETY})
                    entry.update(attribution_categories=attr["categories"], difference_by_symbol=attr["by_symbol"],
                                 difference_by_decision_origin=attr["by_decision_origin"], attribution_reconciliation_error=attr["reconciliation_error"])
                policies.setdefault(policy, {})[scenario] = entry
        result["periods"][period] = {"policies": policies}
    write("comparison.json", result)
    print(json.dumps({"policies": 3, "periods": 2, "attribution_files": 4, "additional_fits": 0}))


def plot(period):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    colors = {"B": "#a93226", "H": "#27ae60", "A": "#1768ac"}
    for col, scenario in enumerate(("normal", "stress")):
        for policy in POLICIES:
            item = replay_item(period, policy, scenario)
            curve = [v for v in item["equity"] if v["phase"] == "calendar_bucket_end"]
            dates = [datetime.fromtimestamp(v["timestamp_ms"] / 1000, timezone.utc) for v in curve]
            ax[0, col].plot(dates, [v["nav"] for v in curve], color=colors[policy], label=policy,
                            linestyle="--" if policy == "A" else "-", lw=1.4)
            ax[1, col].plot(dates, [v["drawdown"] * 100 for v in curve], color=colors[policy],
                            linestyle="--" if policy == "A" else "-", lw=1.4)
        ax[0, col].axhline(10000, color="gray", linestyle=":", label="NoTrade", lw=1)
        ax[0, col].set_title(f"{period}: {scenario} costs | assumed OHLCV execution")
        ax[0, col].set_ylabel("Net liquidation capital, USDT")
        ax[1, col].set_ylabel("Drawdown, % (4h close marks)")
        ax[1, col].set_xlabel("UTC")
        for row in (0, 1):
            ax[row, col].grid(alpha=.25)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4,
               title="Fixed loss experiment | B=ridge daily labels; H=Huber daily labels; A=context only")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, .9))
    target = OUT / f"capital_drawdown_{period}.png"
    if target.exists():
        raise ValueError("plot already exists")
    fig.savefig(target, dpi=130)
    plt.close(fig)
    print(json.dumps({"plot": target.name, "bytes": target.stat().st_size}))


def verify():
    reg, phase_reg, old_reg, _, _, _, _ = load()
    verify_parents(phase_reg, old_reg)
    if len(list(OUT.glob("fit_*.json"))) != 50 or len(list(OUT.glob("replay_*.json"))) != 4:
        raise ValueError("incomplete experiment")
    if not (OUT / "REPORT_RU.md").exists():
        raise ValueError("results report missing")
    pins = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "receipt.json":
            if path.suffix == ".json":
                value = sealed(path)
                if "safety" in value and value["safety"] != c.SAFETY:
                    raise ValueError("authority changed")
            pins[path.name] = base.pin(path)
    write("receipt.json", {"registration_sha256": reg["sha256"], "artifact_pins": pins,
        "frozen_cadence_artifacts_reverified": 254, "frozen_phase_artifacts_reverified": reg["phase_artifacts_verified"],
        "all_9_authority_flags_false": True, "network_used": False, "safety": c.SAFETY})
    print(json.dumps({"verified_artifacts": len(pins), "parent_artifacts_reverified": 254 + reg["phase_artifacts_verified"], "all_flags_false": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "checks", "train", "replay", "audit", "diagnostics", "summarize", "plot", "verify"))
    parser.add_argument("--period", choices=("development", "report"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.step == "train":
        train(args.period, args.start_index, args.count)
    elif args.step == "replay":
        replay_one(args.period, args.scenario)
    elif args.step == "plot":
        plot(args.period)
    else:
        globals()[args.step]()
