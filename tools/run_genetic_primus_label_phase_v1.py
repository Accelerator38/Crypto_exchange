"""One bounded foreground research command; frozen parent files are read-only."""
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
from tools import run_genetic_primus_cadence_matrix_v1 as parent
from exia.genetic_primus import label_phase_v1 as p
c = p.c
OUT = parent.BASE / "label_phase_v1_20261012"
CONTRACT = ROOT / "configs/genetic_primus_label_phase_v1.json"
CODE = (Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/label_phase_v1.py")
POLICIES = ("P0", "P1", "P2", "B")


def read(path):
    return parent.read(path)


def sealed(path):
    return parent.verify_body(read(path))


def write(name, value):
    value = c.plain(value)
    result = value | {"sha256": c.m.identity(value)}
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return result


def full_parent_verification(reg):
    receipt = sealed(parent.OUT / "receipt.json")
    if receipt["registration_sha256"] != reg["sha256"]:
        raise ValueError("parent registration identity mismatch")
    for name, expected in reg["source_pins"].items():
        if parent.pin(ROOT / name) != expected:
            raise ValueError("parent source changed: " + name)
    for name, expected in reg["code_pins"].items():
        if parent.pin(ROOT / name) != expected:
            raise ValueError("parent code changed: " + name)
    for name, expected in receipt["artifact_pins"].items():
        if parent.pin(parent.OUT / name) != expected:
            raise ValueError("parent artifact changed: " + name)
    return receipt


def fit_name(period, phase, index):
    return f"fit_{period}_P{phase}_index_{index:03}.json"


def fold_plan(bank, fit_ms):
    split = c.old.split(bank, fit_ms)
    train, inner, validation = (split[k] for k in ("train", "inner", "validation"))
    tx, ty = bank["x"][train].reshape(-1, 6), bank["y"][train].reshape(-1)
    return {"fit_ms": fit_ms, "train_end_ms": split["train_end_ms"], "cutoff_ms": split["cutoff_ms"],
            "inner_cutoff_ms": split["inner_cutoff_ms"], "train_origins": len(train),
            "train_asset_rows": len(ty), "latest_mature_label_ms": int(bank["available"][train[-1]]),
            "train_samples_sha256": c.m.array_identity(bank["origins"][train], bank["available"][train], tx, ty),
            "folds_sha256": c.m.array_identity(bank["origins"][inner], bank["origins"][validation])}


def prepare():
    if OUT.exists():
        raise ValueError("experiment directory already exists; never overwrite or restart fits")
    reg, data, samples, _, _ = parent.load()
    receipt = full_parent_verification(reg)
    contract = read(CONTRACT)
    for name in ("architecture", "label_horizon_days", "threshold_bps", "fee_bps", "drag_bps", "l2", "weighting"):
        expected = reg["contract"].get(name, 3 if name == "label_horizon_days" else None)
        if contract[name] != expected:
            raise ValueError("fixed parent setting changed: " + name)
    if contract["safety"] != c.SAFETY:
        raise ValueError("research authority flags changed")
    anchor = parent.ms(reg["contract"]["label_anchor_utc"])
    phase_banks = {0: samples["3"]}
    for phase in (1, 2):
        phase_banks[phase] = c.make_samples(data["times"], data["closes"], data["features"],
                                          anchor=p.phase_anchor(anchor, phase), horizon_days=3, stride_days=3)
    all_origins = np.sort(np.concatenate([b["origins"] for b in phase_banks.values()]))
    if not np.array_equal(all_origins, samples["1"]["origins"]):
        raise ValueError("three phase grids must partition the complete daily grid")
    plans = {}
    total = 0
    for period, settings in reg["periods"].items():
        grid = c.Clocks(3, 3, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        plans[period] = {str(phase): [fold_plan(phase_banks[phase], int(t)) for t in grid] for phase in (1, 2)}
        total += 2 * len(grid)
    if total != contract["new_real_fits"]:
        raise ValueError("registered fit budget mismatch")
    OUT.mkdir()
    with (OUT / "phase_samples.npz").open("xb") as stream:
        np.savez_compressed(stream, **{f"p{phase}_{k}": v for phase in (1, 2) for k, v in phase_banks[phase].items()})
    write("folds.json", {"plans": plans, "safety": c.SAFETY})
    bank_shas = {str(phase): c.m.array_identity(*(phase_banks[phase][k] for k in ("origins", "available", "x", "y"))) for phase in (0, 1, 2)}
    registration = write("registration.json", {
        "schema": "label_phase_registration/1", "before_model_fits": True, "contract": contract,
        "periods": reg["periods"], "symbols": reg["symbols"], "anchor_ms": anchor,
        "phase_anchor_ms": {str(i): p.phase_anchor(anchor, i) for i in (0, 1, 2)},
        "parent_registration_sha256": reg["sha256"], "parent_receipt_sha256": receipt["sha256"],
        "parent_registration_pin": parent.pin(parent.OUT / "registration.json"),
        "parent_receipt_pin": parent.pin(parent.OUT / "receipt.json"),
        "parent_sources_verified": len(reg["source_pins"]), "parent_artifacts_verified": len(receipt["artifact_pins"]),
        "code_pins": {parent.relative(f): parent.pin(f) for f in CODE},
        "bank_pins": {name: parent.pin(OUT / name) for name in ("phase_samples.npz", "folds.json")},
        "phase_sample_content_sha256": bank_shas, "new_planned_fits": total,
        "selection_or_promotion": False, "network_used": False, "safety": c.SAFETY})
    write("objective_registry_entry.json", {
        "schema": "local_append_only_diagnostic_entry/1", "registration_sha256": registration["sha256"],
        "candidates": list(POLICIES), "new_candidates": ["P1", "P2"], "all_results_will_be_reported": True,
        "is_tuning": False, "global_historical_trial_registry_complete": False,
        "DSR_PBO_or_independent_validation_claim_allowed": False, "safety": c.SAFETY})
    print(json.dumps({"prepared": True, "new_fits": total, "new_replays": 8,
                      "verified_parent_artifacts": len(receipt["artifact_pins"]), "all_flags_false": True}))


def load():
    reg = sealed(OUT / "registration.json")
    for name, pin in reg["code_pins"].items():
        if parent.pin(ROOT / name) != pin:
            raise ValueError("registered code changed: " + name)
    for name, pin in reg["bank_pins"].items():
        if parent.pin(OUT / name) != pin:
            raise ValueError("registered bank changed: " + name)
    if (parent.pin(parent.OUT / "registration.json") != reg["parent_registration_pin"]
            or parent.pin(parent.OUT / "receipt.json") != reg["parent_receipt_pin"]):
        raise ValueError("parent seal changed")
    old_reg, data, old_samples, banks, rates = parent.load()
    if old_reg["sha256"] != reg["parent_registration_sha256"]:
        raise ValueError("parent identity changed")
    with np.load(OUT / "phase_samples.npz", allow_pickle=False) as stream:
        samples = {f"P{phase}": {k: stream[f"p{phase}_{k}"] for k in ("origins", "available", "x", "y")} for phase in (1, 2)}
    samples.update(P0=old_samples["3"], B=old_samples["1"])
    return reg, old_reg, data, samples, banks, rates


def train(period, phase, start_index, count):
    reg, _, data, samples, _, _ = load()
    if phase not in (1, 2) or type(count) is not int or not 1 <= count <= 7 or start_index < 0:
        raise ValueError("bounded new-phase training command required")
    plans = sealed(OUT / "folds.json")["plans"][period][str(phase)]
    selected = plans[start_index:start_index + count]
    if len(selected) != count:
        raise ValueError("requested fit range outside registered plan")
    for j in range(start_index, start_index + count):
        if (OUT / fit_name(period, phase, j)).exists():
            raise ValueError("fit already exists; do not repeat")
    for j, plan in enumerate(selected, start_index):
        fit_ms = plan["fit_ms"]
        _, prefix_sha = parent.prefix(data, fit_ms)
        result = c.fit_linear(samples[f"P{phase}"], fit_ms, horizon_days=3, stride_days=3,
                              feature_prefix_sha=prefix_sha, sample_sha=reg["phase_sample_content_sha256"][str(phase)])
        identity = result["training_identity"]
        for name in ("train_end_ms", "cutoff_ms", "inner_cutoff_ms", "train_samples_sha256", "folds_sha256"):
            if identity[name] != plan[name]:
                raise ValueError("pre-registered fold/cutoff mismatch")
        for name in ("train_origins", "train_asset_rows", "latest_mature_label_ms"):
            if result[name] != plan[name]:
                raise ValueError("pre-registered sample set mismatch")
        result.update(registration_sha256=reg["sha256"], period=period, policy=f"P{phase}",
                      phase_offset_days=phase, label_anchor_ms=reg["phase_anchor_ms"][str(phase)])
        write(fit_name(period, phase, j), result)
    print(json.dumps({"period": period, "phase": phase, "start_index": start_index, "fits_written": count}))


def model_item(period, policy, index):
    if policy in ("P0", "B"):
        name = parent.fit_name(period, 3 if policy == "P0" else 1, index * 3)
        return sealed(parent.OUT / name), "parent/" + name
    name = fit_name(period, int(policy[1]), index)
    return sealed(OUT / name), name


def decisions(reg, old_reg, data, period, policy):
    if policy in ("P0", "B"):
        return parent.decisions(old_reg, data, period, "A" if policy == "P0" else "B")[0]
    settings = reg["periods"][period]
    rows = []
    for index, t in enumerate(c.Clocks(3, 3, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])):
        item, name = model_item(period, policy, index)
        model = parent.verify_body(item["model"])
        if item["fit_ms"] != t or item["registration_sha256"] != reg["sha256"] or item["assumed_model_ready_us"] > t * 1000 + 100000:
            raise ValueError("causal phase model mismatch")
        position, prefix_sha = parent.prefix(data, t)
        prediction = c.m.predict(model, data["features"][position])
        rows.append({"origin_ms": t, "policy": policy, "targets": np.where(prediction * 10000 > 20, .1, 0).tolist(),
                     "fit_ms": t, "model_age_days": 0, "fit_artifact": name, "model_sha256": model["sha256"],
                     "scaler_sha256": item["scaler_sha256"], "training_identity_sha256": c.m.identity(item["training_identity"]),
                     "feature_prefix_sha256": prefix_sha, "decision_feature_sha256": c.m.array_identity(data["features"][position]),
                     "cutoff_ms": item["training_identity"]["cutoff_ms"], "prediction": prediction.tolist()})
    return rows


def replay_one(period, phase, scenario):
    reg, old_reg, data, _, _, rates = load()
    policy = f"P{phase}"
    if policy not in ("P1", "P2"):
        raise ValueError("P0/B replays must be reused, never rerun")
    name = f"replay_{period}_{policy}_{scenario}.json"
    if (OUT / name).exists():
        raise ValueError("replay already exists")
    settings, contract = reg["periods"][period], reg["contract"]
    rows = decisions(reg, old_reg, data, period, policy)
    result = c.replay(times=data["times"], prices=data["closes"], start=settings["start_ms"], end=settings["end_ms"],
                      symbols=reg["symbols"], decisions=rows, rates=rates, fee_bps=contract["fee_bps"],
                      drag_bps=contract["drag_bps"][scenario], decision_cadence_days=3)
    result.update(registration_sha256=reg["sha256"], period=period, policy=policy, scenario=scenario,
                  label_phase_offset_days=phase, threshold_bps=20)
    write(name, result)
    metric = result["metrics"]
    print(json.dumps({"period": period, "phase": phase, "scenario": scenario,
                      "net_pct": round(metric["net_return"] * 100, 4), "DD_pct": round(metric["max_drawdown"] * 100, 4)}))


def replay_item(period, policy, scenario):
    if policy in ("P0", "B"):
        return sealed(parent.OUT / f"replay_{period}_{'A' if policy == 'P0' else 'B'}_{scenario}.json")
    return sealed(OUT / f"replay_{period}_{policy}_{scenario}.json")


def checks():
    load()
    result = p.direct_checks()
    write("direct_checks.json", result)
    print(json.dumps({"checks_passed": result["checks_passed"], "real_fits": 0, "collector": False}))


def audit():
    reg, _, data, samples, _, _ = load()
    count = 0
    def check(condition, message):
        nonlocal count
        if not condition:
            raise AssertionError(message)
        count += 1
    for phase in (0, 1, 2):
        bank = samples[f"P{phase}"]
        check(np.all(((bank["origins"] - reg["anchor_ms"]) // c.DAY_MS) % 3 == phase), "label phase")
        check(np.all(bank["available"] - bank["origins"] == 3 * c.DAY_MS), "fixed outcome horizon")
        indices = np.searchsorted(samples["B"]["origins"], bank["origins"])
        for key in ("origins", "available", "x", "y"):
            check(np.array_equal(samples["B"][key][indices], bank[key]), "same daily-bank content")
    fits_checked = 0
    for period, settings in reg["periods"].items():
        grid = c.Clocks(3, 3, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        for j, t in enumerate(grid):
            union = []
            for policy in POLICIES:
                bank = samples[policy]
                split = c.old.split(bank, t)
                train = split["train"]
                tx, ty = bank["x"][train].reshape(-1, 6), bank["y"][train].reshape(-1)
                item, _ = model_item(period, policy, j)
                model = parent.verify_body(item["model"])
                identity, scaler = item["training_identity"], model["scaler"]
                check(item["fit_ms"] == t, "same training schedule")
                check(identity["cutoff_ms"] == t - 8 * c.BAR_MS, "same purge cutoff")
                check(np.all(bank["available"][train] <= identity["cutoff_ms"]), "mature labels only")
                check(identity["train_samples_sha256"] == c.m.array_identity(bank["origins"][train], bank["available"][train], tx, ty), "train hash")
                check(identity["folds_sha256"] == c.m.array_identity(bank["origins"][split["inner"]], bank["origins"][split["validation"]]), "fold hash")
                check(np.all(bank["available"][split["inner"]] <= split["inner_cutoff_ms"]), "purged inner fold")
                check(np.array_equal(np.asarray(scaler["mean"]), tx.mean(axis=0)), "train-only scaler mean")
                expected_scale = tx.std(axis=0)
                expected_scale = np.where(expected_scale < 1e-12, 1., expected_scale)
                check(np.array_equal(scaler["scale"], expected_scale), "train-only scaler scale")
                check(scaler["target_mean"] == float(ty.mean()), "all outcomes target mean")
                check(item["positive_weight_mass"] > 0 and item["negative_weight_mass"] > 0, "both signs weighted")
                check(item["feature_prefix_sha256"] == parent.prefix(data, t)[1], "causal prefix hash")
                check(item["latest_mature_label_ms"] == int(bank["available"][train[-1]]), "last eligible label")
                if policy != "B":
                    union.extend(bank["origins"][train].tolist())
                fits_checked += 1
            btrain = c.old.split(samples["B"], t)["train"]
            check(np.array_equal(np.sort(union), samples["B"]["origins"][btrain]), "phase union exactly equals B training set")
        for scenario in ("normal", "stress"):
            reference = replay_item(period, "P0", scenario)
            for policy in POLICIES:
                item = replay_item(period, policy, scenario)
                check([r["origin_ms"] for r in item["decisions"]] == grid, "same decision schedule")
                check(item["safety"] == c.SAFETY and not item["observed_execution_qualified"], "no authority")
                check(abs(p.bar_components(item, data["times"], data["closes"], reg["symbols"])["reconciliation_error"]) < 1e-7, "bar reconciliation")
                check(abs(p.attribute(reference, item, data["times"], data["closes"], reg["symbols"])["reconciliation_error"]) < 1e-7, "decision attribution reconciliation")
                for row in item["decisions"]:
                    j = (row["origin_ms"] - settings["start_ms"]) // (3 * c.DAY_MS)
                    model = model_item(period, policy, j)[0]["model"]
                    position = int(np.searchsorted(data["times"], row["origin_ms"]))
                    pred = c.m.predict(model, data["features"][position])
                    check(np.array_equal(pred, row["prediction"]), "stored prediction")
                    check(np.array_equal(np.where(pred * 10000 > 20, .1, 0), row["targets"]), "same strict threshold")
    check(len(list(OUT.glob("fit_*.json"))) == 100, "exact fit budget")
    check(len(list(OUT.glob("replay_*.json"))) == 8, "exact replay budget")
    write("audit_checks.json", {"checks_passed": count, "models_checked": fits_checked, "new_models": 100,
                                 "phase_union_equals_daily_training": True, "additional_fits": 0, "safety": c.SAFETY})
    print(json.dumps({"audit_checks_passed": count, "models_checked": fits_checked, "new_fits": 100}))


def diagnostics():
    reg, _, data, samples, _, _ = load()
    output = {"training": [], "predictions": [], "period_summary": {}, "safety": c.SAFETY}
    for period, settings in reg["periods"].items():
        grid = c.Clocks(3, 3, 3, 3).decision_times(settings["start_ms"], settings["end_ms"])
        for index, t in enumerate(grid):
            reference = model_item(period, "P0", index)[0]["model"]
            pos = int(np.searchsorted(data["times"], t))
            base_prediction = c.m.predict(reference, data["features"][pos]) * 10000
            for policy in POLICIES:
                bank = samples[policy]
                train = c.old.split(bank, t)["train"]
                y = bank["y"][train]
                item, source = model_item(period, policy, index)
                model, scaler = item["model"], item["model"]["scaler"]
                prediction = c.m.predict(model, data["features"][pos]) * 10000
                standardized = (data["features"][pos] - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
                output["training"].append({"period": period, "policy": policy, "fit_ms": t, "source": source,
                    "train_origins": len(train), "train_asset_rows": int(y.size),
                    "first_train_origin_ms": int(bank["origins"][train[0]]),
                    "last_train_origin_ms": int(bank["origins"][train[-1]]), "cutoff_ms": item["training_identity"]["cutoff_ms"],
                    "latest_mature_label_ms": item["latest_mature_label_ms"],
                    "last_mature_age_days": (t - item["latest_mature_label_ms"]) / c.DAY_MS,
                    "labels_bps": p.describe(y * 10000),
                    "labels_by_symbol_bps": {s: p.describe(y[:, i] * 10000) for i, s in enumerate(reg["symbols"])},
                    "scaler": scaler, "coefficients": p.coefficient_view(model),
                    "scaler_mean_delta_vs_P0": (np.asarray(scaler["mean"]) - reference["scaler"]["mean"]).tolist(),
                    "scaler_scale_ratio_vs_P0": (np.asarray(scaler["scale"]) / reference["scaler"]["scale"]).tolist(),
                    "target_mean_delta_vs_P0_bps": (scaler["target_mean"] - reference["scaler"]["target_mean"]) * 10000,
                    "clipped_decision_feature_cells": int(np.sum(np.abs(standardized) > scaler["clip"]))})
                available = t + 3 * c.DAY_MS <= settings["end_ms"]
                exit_pos = int(np.searchsorted(data["times"], t + 3 * c.DAY_MS)) if available else None
                for i, symbol in enumerate(reg["symbols"]):
                    label = float((data["closes"][exit_pos, i] / data["closes"][pos, i] - 1) * 10000) if available else None
                    output["predictions"].append({"period": period, "policy": policy, "origin_ms": t, "symbol": symbol,
                        "prediction_bps": float(prediction[i]), "margin_to_threshold_bps": float(prediction[i] - 20),
                        "prediction_delta_vs_P0_bps": float(prediction[i] - base_prediction[i]),
                        "target_long": bool(prediction[i] > 20), "P0_target_long": bool(base_prediction[i] > 20),
                        "full_horizon_label_bps": label, "model_sha256": model["sha256"]})
        output["period_summary"][period] = {}
        for policy in POLICIES:
            rows = [r for r in output["predictions"] if r["period"] == period and r["policy"] == policy]
            training = [r for r in output["training"] if r["period"] == period and r["policy"] == policy]
            margins = np.array([r["margin_to_threshold_bps"] for r in rows])
            complete = [r for r in rows if r["full_horizon_label_bps"] is not None]
            selected = [r for r in complete if r["target_long"]]
            coeff = np.array([r["coefficients"]["slopes_bps_per_one_train_std"] for r in training])
            base_coeff = np.array([r["coefficients"]["slopes_bps_per_one_train_std"] for r in output["training"] if r["period"] == period and r["policy"] == "P0"])
            output["period_summary"][period][policy] = {
                "decisions_assets": len(rows), "prediction_bps": p.describe([r["prediction_bps"] for r in rows]),
                "margin_bps": p.describe(margins), "near_threshold_fraction": {str(n): float(np.mean(abs(margins) <= n)) for n in (5, 10, 20)},
                "long_fraction": float(np.mean([r["target_long"] for r in rows])),
                "action_disagreements_vs_P0": sum(r["target_long"] != r["P0_target_long"] for r in rows),
                "mean_absolute_prediction_delta_vs_P0_bps": float(np.mean([abs(r["prediction_delta_vs_P0_bps"]) for r in rows])),
                "mean_target_mean_delta_vs_P0_bps": float(np.mean([r["target_mean_delta_vs_P0_bps"] for r in training])),
                "coefficient_mean_bps_per_train_std": coeff.mean(axis=0).tolist(),
                "coefficient_sign_disagreements_vs_P0": (np.sign(coeff) != np.sign(base_coeff)).sum(axis=0).tolist(),
                "last_mature_age_days": p.describe([r["last_mature_age_days"] for r in training]),
                "full_horizon_opportunities": len(complete), "selected_full_horizon_opportunities": len(selected),
                "forecast_MAE_bps": float(np.mean([abs(r["prediction_bps"] - r["full_horizon_label_bps"]) for r in complete])),
                "selected_gross_label_bps": p.describe([r["full_horizon_label_bps"] for r in selected]) if selected else None,
                "clipped_decision_feature_cells": sum(r["clipped_decision_feature_cells"] for r in training)}
    write("diagnostics.json", output)
    print(json.dumps({"training_snapshots": len(output["training"]), "prediction_cells": len(output["predictions"]), "new_fits": 0}))


def day_key(t, end):
    return datetime.fromtimestamp(min(int(t), end - 1) / 1000, timezone.utc).strftime("%Y-%m-%d")


def summarize():
    reg, _, data, _, all_banks, _ = load()
    result = {"periods": {}, "new_fits": 100, "new_replays": 8, "reused_financial_replays": 8,
              "primary_block_days": 7, "default_deployment": "NoTrade", "promotion_authority": False,
              "all_periods_already_open": True, "safety": c.SAFETY}
    for period, settings in reg["periods"].items():
        policies = {}
        banks = {k.removeprefix(period + "_"): v for k, v in all_banks.items() if k.startswith(period + "_")}
        for scenario in ("normal", "stress"):
            a, dense = (replay_item(period, policy, scenario) for policy in ("P0", "B"))
            for policy in POLICIES:
                item = replay_item(period, policy, scenario)
                bars = p.bar_components(item, data["times"], data["closes"], reg["symbols"])
                daily = np.asarray(item["daily_log_returns"])
                windows = np.array([daily[i:i + 3].sum() for i in range(0, len(daily) - 2, 3)])
                by_symbol = p.concentration(reg["symbols"], bars["net"].sum(axis=0))
                by_date = p.concentration([day_key(t, settings["end_ms"]) for t in bars["times"]], bars["net"].sum(axis=1))
                entry = {"metrics": item["metrics"],
                         "bootstrap_vs_P0": c.bootstrap(daily, banks, paired=a["daily_log_returns"]),
                         "bootstrap_vs_B": c.bootstrap(daily, banks, paired=dense["daily_log_returns"]),
                         "symbol_concentration": by_symbol, "date_concentration": by_date,
                         "without_best_full_3d_log_window_return": float(np.expm1(daily.sum() - windows.max())),
                         "full_3d_windows": len(windows), "positive_full_3d_windows": int(np.sum(windows > 0)),
                         "bar_reconciliation_error": bars["reconciliation_error"]}
                if policy != "P0":
                    attribution = p.attribute(a, item, data["times"], data["closes"], reg["symbols"])
                    rows = attribution["rows"]
                    attribution["by_symbol"] = p.concentration([r["symbol"] for r in rows], [r["delta_usdt"] for r in rows])
                    attribution["by_decision_origin"] = p.concentration([r["origin_ms"] for r in rows], [r["delta_usdt"] for r in rows])
                    attribution["largest_decision_asset_differences"] = sorted(rows, key=lambda r: abs(r["delta_usdt"]), reverse=True)[:12]
                    base_bars = p.bar_components(a, data["times"], data["closes"], reg["symbols"])
                    delta_bars = bars["net"] - base_bars["net"]
                    attribution["by_UTC_boundary_date"] = p.concentration([day_key(t, settings["end_ms"]) for t in bars["times"]], delta_bars.sum(axis=1))
                    positions = np.searchsorted(data["times"], bars["times"])
                    px = data["closes"][positions]
                    changes = px[1:] / px[:-1] - 1
                    largest = np.argsort(abs(changes).reshape(-1))[-10:][::-1]
                    shock_rows = []
                    for flat in largest:
                        j, i = np.unravel_index(int(flat), changes.shape)
                        shock_rows.append({"timestamp_ms": int(bars["times"][j + 1]), "symbol": reg["symbols"][i],
                                           "close_change_pct": float(changes[j, i] * 100),
                                           "delta_net_usdt_on_boundary": float(delta_bars[j + 1, i]),
                                           "reference_gross_usdt": float(base_bars["gross"][j + 1, i]),
                                           "challenger_gross_usdt": float(bars["gross"][j + 1, i])})
                    attribution["largest_market_close_moves"] = shock_rows
                    attribution["largest_10_market_moves_share_of_absolute_bar_differences"] = float(
                        sum(abs(r["delta_net_usdt_on_boundary"]) for r in shock_rows) / max(abs(delta_bars).sum(), 1e-30))
                    write(f"attribution_{period}_{policy}_vs_P0_{scenario}.json", attribution | {"safety": c.SAFETY})
                    entry["attribution_categories"] = attribution["categories"]
                    entry["attribution_delta_usdt"] = attribution["delta_usdt"]
                    entry["difference_by_symbol"] = attribution["by_symbol"]
                    entry["difference_by_decision_origin"] = attribution["by_decision_origin"]
                    entry["difference_by_date"] = attribution["by_UTC_boundary_date"]
                    entry["largest_10_market_moves_share_of_absolute_bar_differences"] = attribution["largest_10_market_moves_share_of_absolute_bar_differences"]
                policies.setdefault(policy, {})[scenario] = entry
        result["periods"][period] = {"policies": policies}
    result["registration_sha256"] = reg["sha256"]
    write("comparison.json", result)
    print(json.dumps({"policies": 4, "periods": 2, "attribution_files": 12, "additional_replays": 0}))


def plot(period):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    colors = {"P0": "#1768ac", "P1": "#e67e22", "P2": "#27ae60", "B": "#a93226"}
    for col, scenario in enumerate(("normal", "stress")):
        for policy in POLICIES:
            item = replay_item(period, policy, scenario)
            curve = [r for r in item["equity"] if r["phase"] == "calendar_bucket_end"]
            dates = [datetime.fromtimestamp(r["timestamp_ms"] / 1000, timezone.utc) for r in curve]
            axes[0, col].plot(dates, [r["nav"] for r in curve], label=policy, color=colors[policy], lw=1.4)
            axes[1, col].plot(dates, [r["drawdown"] * 100 for r in curve], color=colors[policy], lw=1.4)
        axes[0, col].axhline(10000, color="gray", linestyle="--", lw=1, label="NoTrade")
        axes[0, col].set_title(f"{period}: {scenario} costs | assumed OHLCV execution")
        axes[0, col].set_ylabel("Net liquidation capital, USDT")
        axes[1, col].set_ylabel("Drawdown, % (4h close marks)")
        axes[1, col].set_xlabel("UTC")
        for row in (0, 1):
            axes[row, col].grid(alpha=.25)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5,
               title="P0=A frozen phase; P1/P2 shifted labels only; B=daily labels")
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, .9))
    name = OUT / f"capital_drawdown_{period}.png"
    if name.exists():
        raise ValueError("plot already exists")
    fig.savefig(name, dpi=130)
    plt.close(fig)
    print(json.dumps({"plot": name.name, "bytes": name.stat().st_size}))


def verify():
    reg, old_reg, _, _, _, _ = load()
    full_parent_verification(old_reg)
    pins = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "receipt.json":
            if path.suffix == ".json":
                value = sealed(path)
                if "safety" in value and value["safety"] != c.SAFETY:
                    raise ValueError("authority flags changed")
            pins[path.name] = parent.pin(path)
    write("receipt.json", {"registration_sha256": reg["sha256"], "parent_artifacts_reverified": reg["parent_artifacts_verified"],
                            "artifact_pins": pins, "all_9_authority_flags_false": True,
                            "network_used": False, "safety": c.SAFETY})
    print(json.dumps({"verified_artifacts": len(pins), "parent_artifacts_verified": reg["parent_artifacts_verified"], "all_flags_false": True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "checks", "train", "replay", "audit", "diagnostics", "summarize", "plot", "verify"))
    parser.add_argument("--period", choices=("development", "report"))
    parser.add_argument("--phase", type=int, choices=(1, 2))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.step == "train":
        train(args.period, args.phase, args.start_index, args.count)
    elif args.step == "replay":
        replay_one(args.period, args.phase, args.scenario)
    elif args.step == "plot":
        plot(args.period)
    else:
        globals()[args.step]()


if __name__ == "__main__":
    main()
