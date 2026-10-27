"""One short foreground stage at a time; no collector, network or runtime."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tools import run_genetic_primus_causal_capital_controls_v1 as capital
from exia.genetic_primus import position_policy_v1 as p

c, base, robust = p.c, capital.base, capital.robust
OUT = base.BASE / "position_policy_v1_20261012"
CONTRACT = ROOT / "configs/genetic_primus_position_policy_v1.json"
CODE = (Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/position_policy_v1.py",
        ROOT / "docs/GENETIC_PRIMUS_POSITION_POLICY_V1.md")
SCENARIOS = ("normal", "stress")


def sealed(path):
    return base.verify_body(base.read(path))


def write(name, value):
    value = c.plain(value)
    result = value | {"sha256": c.m.identity(value)}
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def parents():
    reg = capital.load()
    receipt = sealed(capital.OUT / "receipt.json")
    if receipt["registration_sha256"] != reg["sha256"]:
        raise ValueError("capital control parent seal mismatch")
    for name, pin in receipt["artifact_pins"].items():
        if base.pin(capital.OUT / name) != pin:
            raise ValueError("parent artifact changed: " + name)
    old_h, phase_reg, old_reg, data, samples, banks, rates = robust.load()
    robust.verify_parents(phase_reg, old_reg)
    old_receipt = sealed(robust.OUT / "receipt.json")
    for name, pin in old_receipt["artifact_pins"].items():
        if base.pin(robust.OUT / name) != pin:
            raise ValueError("robust parent artifact changed")
    return old_reg, data, samples, banks, rates


def load(with_data=True):
    reg = sealed(OUT / "registration.json")
    for path, pin in (reg["code_pins"] | reg["input_pins"]).items():
        if base.pin(ROOT / path) != pin:
            raise ValueError("registered code/input changed: " + path)
    if base.pin(OUT / "folds.json") != reg["folds_pin"]:
        raise ValueError("registered folds changed")
    for name, pin in reg["bank_pins"].items():
        if base.pin(OUT / name) != pin:
            raise ValueError("resample bank changed")
    if with_data:
        _, _, _, data, samples, banks, rates = robust.load()
        return reg, data, samples, banks, rates
    return reg


def checks():
    OUT.mkdir(parents=True, exist_ok=True)
    result = p.direct_checks()
    write("direct_checks.json", result)
    print(json.dumps({"pure_checks_passed": result["checks_passed"], "fits": 0, "replays": 0}))


def ledger_checks():
    result = p.ledger_checks()
    write("ledger_checks.json", result)
    print(json.dumps({"synthetic_ledger_checks": result["checks_passed"], "collector": False, "real_fits": 0}))


def prepare():
    old_reg, data, samples, banks, rates = parents()
    contract = base.read(CONTRACT)
    if contract["safety"] != c.SAFETY or tuple(contract["policies"]) != p.POLICIES:
        raise ValueError("bounded offline contract required")
    periods = dict(old_reg["periods"])
    for name, item in contract["regimes"].items():
        start, end = base.ms(item["start"]), base.ms(item["end"])
        periods[name] = {"start_ms": start, "end_ms": end, "days": (end - start) // c.DAY_MS}
    folds = {}
    for name in contract["regimes"]:
        settings = periods[name]
        rows = []
        for t in range(settings["start_ms"], settings["end_ms"], 3 * c.DAY_MS):
            temporal = c.old.split(samples, int(t))
            train = temporal["train"]
            rows.append({"fit_ms": t, "day": (t - settings["start_ms"]) // c.DAY_MS,
                "cutoff_ms": temporal["cutoff_ms"], "train_origins": len(train),
                "latest_mature_label_ms": int(samples["available"][train[-1]]),
                "train_indices_sha256": c.m.array_identity(train),
                "training_arrays_sha256": c.m.array_identity(samples["origins"][train], samples["x"][train], samples["y"][train]),
                "feature_prefix_sha256": base.prefix(data, t)[1]})
        folds[name] = rows
    if sum(map(len, folds.values())) != contract["fit_budget_regimes"]:
        raise ValueError("fit budget mismatch")
    write("folds.json", {"regimes": folds, "sample_bank_sha256": c.m.array_identity(samples["origins"], samples["available"], samples["x"], samples["y"]), "safety": c.SAFETY})
    bank_pins = {}
    for number, name in enumerate(contract["regimes"]):
        days = periods[name]["days"]
        bank = {f"b{block}_{role}": c.resample_bank(days, block, contract["resamples"]["seed_base"] + number * 1000 + block * 10 + k)
                for block in (3, 7, 14) for k, role in enumerate(("selection", "audit"))}
        file = f"banks_{name}.npz"
        with (OUT / file).open("xb") as stream:
            np.savez(stream, **bank)
        bank_pins[file] = base.pin(OUT / file)
    paths = [capital.OUT / "registration.json", capital.OUT / "receipt.json", robust.OUT / "registration.json",
             robust.OUT / "receipt.json", base.OUT / "registration.json", base.OUT / "receipt.json"]
    paths += [base.OUT / f"replay_{period}_C_{scenario}.json" for period in old_reg["periods"] for scenario in SCENARIOS]
    reg = write("registration.json", {"schema": "position_policy_registration/1", "contract": contract,
        "before_new_fits_policy_replays_and_threshold_changes": True, "periods": periods, "symbols": old_reg["symbols"],
        "code_pins": {base.relative(x): base.pin(x) for x in CODE}, "input_pins": {base.relative(x): base.pin(x) for x in paths},
        "folds_pin": base.pin(OUT / "folds.json"), "bank_pins": bank_pins,
        "parent_bank_hashes": {k: c.m.array_identity(v) for k, v in banks.items()},
        "old_regime_funding_rows": {name: sum(periods[name]["start_ms"] <= int(x["settlement_ms"]) <= periods[name]["end_ms"] for x in rates)
                                    for name in contract["regimes"]},
        "safety": c.SAFETY})
    write("objective_registry_entry.json", {"registration_sha256": reg["sha256"],
        "primary_contrasts": contract["primary_contrasts"], "new_fit_budget": 123, "threshold_search_budget": 0,
        "adaptive_history_not_reset": True, "global_registry_completeness_not_claimed": True, "safety": c.SAFETY})
    print(json.dumps({"registered_regimes": 4, "new_fit_budget": 123, "policy_replays": 84, "funding_rows": reg["old_regime_funding_rows"]}))


def fit_batch(period, first, count):
    reg, data, samples, _, _ = load()
    if period not in reg["contract"]["regimes"] or not 1 <= count <= 8:
        raise ValueError("one registered regime and at most eight origins per foreground call")
    rows = sealed(OUT / "folds.json")["regimes"][period][first:first + count]
    sample_sha = sealed(OUT / "folds.json")["sample_bank_sha256"]
    for row in rows:
        item = c.fit_linear(samples, int(row["fit_ms"]), horizon_days=3, stride_days=1,
            feature_prefix_sha=row["feature_prefix_sha256"], sample_sha=sample_sha)
        if item["training_identity"]["cutoff_ms"] != row["cutoff_ms"] or item["latest_mature_label_ms"] > row["cutoff_ms"]:
            raise ValueError("mature/purged training identity mismatch")
        item.update(registration_sha256=reg["sha256"], period=period)
        write(f"fit_{period}_{row['day']:03d}.json", item)
    print(json.dumps({"period": period, "fit_indices": [first, first + len(rows)], "new_model_and_scaler_fits": len(rows)}))


def forecasts(period):
    reg, data, _, _, _ = load()
    settings = reg["periods"][period]
    if period in ("development", "report"):
        saved = sealed(base.OUT / f"replay_{period}_C_stress.json")
        normal = sealed(base.OUT / f"replay_{period}_C_normal.json")
        if saved["decisions"] != normal["decisions"]:
            raise ValueError("source forecasts differ by cost scenario")
        rows = saved["decisions"]
    else:
        rows = []
        for t in range(settings["start_ms"], settings["end_ms"], c.DAY_MS):
            day = (t - settings["start_ms"]) // c.DAY_MS
            fit_day = day // 3 * 3
            file = f"fit_{period}_{fit_day:03d}.json"
            item = sealed(OUT / file)
            model = base.verify_body(item["model"])
            index, prefix = base.prefix(data, t)
            prediction = c.m.predict(model, data["features"][index])
            rows.append({"origin_ms": t, "fit_ms": item["fit_ms"], "model_age_days": day - fit_day,
                "fit_artifact": file, "model_sha256": model["sha256"], "scaler_sha256": item["scaler_sha256"],
                "training_identity_sha256": c.m.identity(item["training_identity"]), "cutoff_ms": item["training_identity"]["cutoff_ms"],
                "prediction": prediction.tolist(), "targets": np.where(prediction * 10000 > 20, .1, 0).tolist(),
                "feature_prefix_sha256": prefix, "decision_feature_sha256": c.m.array_identity(data["features"][index])})
    policies = p.decision_policies(rows, data["times"], data["closes"], data["features"], reg["sha256"])
    write(f"decisions_{period}.json", {"period": period, "registration_sha256": reg["sha256"], "policies": policies, "safety": c.SAFETY})
    print(json.dumps({"period": period, "saved_forecasts": len(rows), "policy_streams": 7, "new_fits": 0}))


def funding(reg, period, scenario, source_rates):
    if period in ("development", "report"):
        return source_rates
    settings = reg["periods"][period]
    rate = reg["contract"]["costs"][f"regime_funding_{scenario}_bps_per_8h"] / 10000
    return [{"settlement_ms": t, "symbol": s.replace("/", ""), "rate_fraction": rate, "synthetic_assumption": True}
            for t in range(settings["start_ms"], settings["end_ms"] + 1, 2 * c.BAR_MS) for s in reg["symbols"]]


def replays(period):
    reg, data, _, _, rates = load()
    settings = reg["periods"][period]
    policies = sealed(OUT / f"decisions_{period}.json")
    for scenario in SCENARIOS:
        for name in p.POLICIES:
            item = p.replay(times=data["times"], prices=data["closes"], start=settings["start_ms"], end=settings["end_ms"],
                symbols=reg["symbols"], decisions=policies["policies"][name], rates=funding(reg, period, scenario, rates),
                fee_bps=6, drag_bps=2 if scenario == "normal" else 8, rebalance=name not in ("C", "C_H"))
            item.update(registration_sha256=reg["sha256"], period=period, policy=name, scenario=scenario,
                funding_source="frozen_reported_rates" if period in ("development", "report") else "registered_synthetic_carry",
                model_fits_in_this_replay=0, safety=c.SAFETY)
            write(f"replay_{period}_{name}_{scenario}.json", item)
    print(json.dumps({"period": period, "bounded_offline_replays": 14, "new_fits": 0, "network": False}))


def banks_for(reg, period, parent_banks):
    if period in ("development", "report"):
        banks = {k.removeprefix(period + "_"): v for k, v in parent_banks.items() if k.startswith(period + "_")}
        for key, value in banks.items():
            if c.m.array_identity(value) != reg["parent_bank_hashes"][period + "_" + key]:
                raise ValueError("parent resamples changed")
        return banks
    with np.load(OUT / f"banks_{period}.npz", allow_pickle=False) as file:
        return {k: file[k] for k in file.files}


def analyze(period):
    reg, data, samples, parent_banks, _ = load()
    banks = banks_for(reg, period, parent_banks)
    policies, ledgers, checks = {}, {}, 0
    training_checks = 0
    if period in reg["contract"]["regimes"]:
        for row in sealed(OUT / "folds.json")["regimes"][period]:
            fit = sealed(OUT / f"fit_{period}_{row['day']:03d}.json")
            model = base.verify_body(fit["model"])
            temporal = c.old.split(samples, row["fit_ms"])
            train = temporal["train"]
            tx, ty = samples["x"][train].reshape(-1, 6), samples["y"][train].reshape(-1)
            scaler = model["scaler"]
            scale = tx.std(axis=0)
            scale = np.where(scale < 1e-12, 1., scale)
            checks_here = [fit["registration_sha256"] == reg["sha256"],
                model["parameter_count"] == 7,
                c.m.array_identity(train) == row["train_indices_sha256"],
                c.m.array_identity(samples["origins"][train], samples["x"][train], samples["y"][train]) == row["training_arrays_sha256"],
                np.all(samples["available"][train] <= row["cutoff_ms"]),
                c.m.array_identity(tx, ty) == scaler["training_sha256"],
                np.allclose(scaler["mean"], tx.mean(axis=0), rtol=0, atol=1e-12),
                np.allclose(scaler["scale"], scale, rtol=0, atol=1e-12),
                abs(scaler["target_mean"] - ty.mean()) < 1e-12,
                abs(scaler["target_scale"] - (float(ty.std()) if ty.std() >= 1e-12 else 1.)) < 1e-12,
                fit["positive_weight_mass"] > 0 and fit["negative_weight_mass"] > 0]
            z = np.clip((tx - scaler["mean"]) / scaler["scale"], -scaler["clip"], scaler["clip"])
            design = np.column_stack((z, np.ones(len(z))))
            target = (ty - scaler["target_mean"]) / scaler["target_scale"]
            parameters = np.asarray(model["parameters"])
            gradient = design.T @ (design @ parameters - target) / len(target) + np.r_[.01 * parameters[:6], 0.]
            checks_here.append(np.max(abs(gradient)) < 1e-8)
            if not all(checks_here):
                raise ValueError("causal scaler / maturity / ridge normal equation audit")
            training_checks += len(checks_here)
    for scenario in SCENARIOS:
        for name in p.POLICIES:
            item = sealed(OUT / f"replay_{period}_{name}_{scenario}.json")
            if item["safety"] != c.SAFETY or abs(item["metrics"]["reconciliation_error"]) > 1e-7:
                raise ValueError("financial reconciliation/safety")
            if period in ("development", "report") and name == "C":
                original = sealed(base.OUT / f"replay_{period}_C_{scenario}.json")
                if any(abs(item["metrics"][k] - original["metrics"][k]) > 1e-8 for k in original["metrics"]):
                    raise ValueError("C baseline changed")
                if item["daily_log_returns"] != original["daily_log_returns"]:
                    raise ValueError("C daily return identity")
            for row in item["decisions"]:
                if row["risk_last_data_ms"] >= row["origin_ms"] or row["fit_ms"] > row["origin_ms"]:
                    raise ValueError("future information")
                if name not in ("C", "C_H") and row["ex_ante_daily_volatility"] > .005 + 1e-10:
                    raise ValueError("ex-ante risk cap")
                checks += 1
            detail = p.attribution(item, data["times"], data["closes"], reg["symbols"])
            write(f"attribution_{period}_{name}_{scenario}.json", detail | {"replay_sha256": item["sha256"]})
            daily = np.asarray(item["daily_log_returns"])
            policies[name + "_" + scenario] = {"metrics": item["metrics"], "daily_log_volatility": float(np.std(daily)),
                "mean_holding_days": detail["mean_holding_days"], "episodes": detail["episodes"],
                "attribution_USDT": detail["components_USDT"], "per_asset_net_USDT": detail["per_asset_net_USDT"],
                "calendar_months": detail["calendar_months"], "return_without_best_3day": detail["return_without_best_3day"],
                "bootstrap_absolute": c.bootstrap(daily, banks),
                "absolute_adjusted_LCB": {str(b): float(np.quantile(daily[banks[f"b{b}_audit"]].mean(axis=1) * 10000, .00625)) for b in (3, 7, 14)}}
            ledgers[name + "_" + scenario] = item
    contrasts = {}
    for a, b in reg["contract"]["primary_contrasts"] + [["C_V", "EMA_M"], ["C_V", "Passive_M"]]:
        for scenario in SCENARIOS:
            va, vb = ledgers[a + "_" + scenario], ledgers[b + "_" + scenario]
            delta = np.asarray(va["daily_log_returns"]) - vb["daily_log_returns"]
            contrasts[a + "_minus_" + b + "_" + scenario] = {
                "bootstrap": c.bootstrap(va["daily_log_returns"], banks, paired=vb["daily_log_returns"]),
                "adjusted_LCB": {str(block): float(np.quantile(delta[banks[f"b{block}_audit"]].mean(axis=1) * 10000, .00625)) for block in (3, 7, 14)},
                "mean_log_bps_day": float(delta.mean() * 10000),
                "exposure_ratio": va["metrics"]["mean_exposure"] / vb["metrics"]["mean_exposure"] if vb["metrics"]["mean_exposure"] else None,
                "volatility_ratio": float(np.std(va["daily_log_returns"]) / np.std(vb["daily_log_returns"])) if np.std(vb["daily_log_returns"]) else None,
                "matched_controls_secondary_only": b.endswith("_M")}
    write(f"analysis_{period}.json", {"registration_sha256": reg["sha256"], "period": period,
        "policies": policies, "contrasts": contrasts, "causal_decision_checks": checks,
        "training_checks": training_checks, "safety": c.SAFETY})
    print(json.dumps({"period": period, "stress_net_pct": {k: round(v["metrics"]["net_return"] * 100, 3)
        for k, v in policies.items() if k.endswith("_stress")}, "causal_checks": checks}))


def draw(period):
    reg = load(False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for scenario in SCENARIOS:
        target = OUT / f"capital_drawdown_{period}_{scenario}.png"
        if target.exists():
            raise FileExistsError(target)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for name in ("C", "C_H", "C_V", "EMA_V", "Passive_V"):
            item = sealed(OUT / f"replay_{period}_{name}_{scenario}.json")
            dates = [datetime.fromtimestamp(r["timestamp_ms"] / 1000, timezone.utc) for r in item["equity"]]
            line, = axes[0].plot(dates, [r["nav"] for r in item["equity"]], label=name, lw=1.3)
            axes[1].plot(dates, [-100 * r["drawdown"] for r in item["equity"]], color=line.get_color(), lw=1.1)
        axes[0].axhline(10000, color="black", ls=":", label="NoTrade")
        axes[0].set_title(f"{period} / {scenario} / " + ("synthetic funding" if period.startswith("q") else "frozen reported funding"))
        axes[0].set_ylabel("Capital, USDT")
        axes[1].set_ylabel("Drawdown, %")
        axes[1].set_xlabel("UTC; terminal liquidation included")
        axes[0].legend(ncol=3)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(target, dpi=140)
        plt.close(fig)
    print(json.dumps({"period": period, "plots": 2}))


def summarize():
    reg, _, _, parent_banks, _ = load()
    regimes = list(reg["contract"]["regimes"])
    analyses = {name: sealed(OUT / f"analysis_{name}.json") for name in reg["periods"]}
    banks = {name: banks_for(reg, name, parent_banks) for name in regimes}
    daily = {name: {policy: np.asarray(sealed(OUT / f"replay_{name}_{policy}_stress.json")["daily_log_returns"])
                    for policy in p.POLICIES} for name in regimes}
    total_days = sum(reg["periods"][name]["days"] for name in regimes)
    pooled = {}
    for policy in p.POLICIES:
        boot = sum(daily[name][policy][banks[name]["b7_audit"]].sum(axis=1) for name in regimes) / total_days * 10000
        logs = {name: float(daily[name][policy].sum()) for name in regimes}
        positive = sum(max(v, 0) for v in logs.values())
        pooled[policy] = {"sum_episode_log_returns": sum(logs.values()),
            "hypothetical_compounded_episode_return": math.expm1(sum(logs.values())),
            "not_continuously_traded_gap_CAGR": True, "per_regime_log_sums": logs,
            "absolute_LCB_5pct_bps_day": float(np.quantile(boot, .05)),
            "absolute_adjusted_LCB_bps_day": float(np.quantile(boot, .00625)),
            "maximum_positive_regime_share": max((max(v, 0) for v in logs.values())) / positive if positive else None,
            "leave_one_regime_out_log_sums": {name: sum(logs.values()) - value for name, value in logs.items()}}
    contrasts = {}
    for a, b in reg["contract"]["primary_contrasts"]:
        deltas = {name: daily[name][a] - daily[name][b] for name in regimes}
        boot = sum(deltas[name][banks[name]["b7_audit"]].sum(axis=1) for name in regimes) / total_days * 10000
        sums = {name: float(values.sum()) for name, values in deltas.items()}
        contrasts[a + "_minus_" + b] = {"LCB_5pct_bps_day": float(np.quantile(boot, .05)),
            "adjusted_LCB_bps_day": float(np.quantile(boot, .00625)), "per_regime_log_sums": sums,
            "positive_regimes": sum(v > 0 for v in sums.values()),
            "leave_one_regime_out_log_sums": {name: sum(sums.values()) - value for name, value in sums.items()}}
    report = analyses["report"]
    gates = {"report_absolute": report["policies"]["C_V_stress"]["absolute_adjusted_LCB"]["7"] > 0,
             "pooled_absolute": pooled["C_V"]["absolute_adjusted_LCB_bps_day"] > 0,
             "pooled_leave_one_out_absolute": min(pooled["C_V"]["leave_one_regime_out_log_sums"].values()) > 0,
             "concentration": (pooled["C_V"]["maximum_positive_regime_share"] or 1) <= .6}
    for agent in ("EMA_V", "Passive_V"):
        key = "C_V_minus_" + agent
        r, q = report["contrasts"][key + "_stress"], contrasts[key]
        gates[key + "_report"] = r["adjusted_LCB"]["7"] > 0
        gates[key + "_pooled"] = q["adjusted_LCB_bps_day"] > 0
        gates[key + "_regime_points"] = q["positive_regimes"] >= 3
        gates[key + "_leave_one_out"] = min(q["leave_one_regime_out_log_sums"].values()) > 0
        gates[key + "_risk_exposure_comparable"] = all(r[name] is not None and .8 <= r[name] <= 1.25 for name in ("exposure_ratio", "volatility_ratio"))
    result = write("summary.json", {"registration_sha256": reg["sha256"], "pooled_days": total_days,
        "stratified_bootstrap_no_cross_gap_blocks": True, "pooled_policies": pooled, "pooled_contrasts": contrasts,
        "step5_gate_checks": gates, "step5_gate_passed": all(gates.values()),
        "half_life_experiment_executed": False, "MLP_GA_executed": False, "safety": c.SAFETY})
    print(json.dumps({"step5_gate_passed": result["step5_gate_passed"], "failed_checks": [k for k, v in gates.items() if not v], "pooled_days": total_days}))


def prospective():
    reg = load(False)
    summary = sealed(OUT / "summary.json")
    if summary["step5_gate_passed"]:
        raise ValueError("gate passed: run separately registered paired weighting experiment before prospective activation")
    variances, details = [], {}
    for period in reg["periods"]:
        candidate = np.asarray(sealed(OUT / f"replay_{period}_C_V_stress.json")["daily_log_returns"]) * 10000
        ordinary = np.asarray(sealed(OUT / f"replay_{period}_EMA_V_stress.json")["daily_log_returns"]) * 10000
        for hypothesis, values in (("absolute", candidate), ("incremental", candidate - ordinary)):
            centered = values - values.mean()
            for lag in (3, 7, 14):
                lrv = float(centered @ centered / len(centered))
                lrv += 2 * sum((1 - k / (lag + 1)) * float(centered[k:] @ centered[:-k] / len(centered)) for k in range(1, lag + 1))
                lrv = max(lrv, 0.)
                variances.append(lrv)
                details[f"{period}_{hypothesis}_lag{lag}"] = lrv
    sigma2 = max(variances)
    z = NormalDist().inv_cdf(.975) + NormalDist().inv_cdf(.8)
    days = {str(effect): max(90, math.ceil(z * z * sigma2 / effect ** 2)) for effect in (1, 2, 5, 10)}
    write("prospective_design_NOT_ACTIVATED.json", {"registration_sha256": reg["sha256"],
        "status": "blocked_by_predeclared_signal_gate_not_a_running_task", "selected_promotable_model": None,
        "training_recipe_sha256": c.m.identity(reg["contract"]["clocks"] | {"architecture": "linear6", "l2": .01, "weighting": "uniform"}),
        "policy_contract_sha256": base.pin(CONTRACT)["sha256"], "no_start_or_end_date_selected": True,
        "maximum_dependent_variance_bps_squared": sigma2, "dependent_variance_diagnostics": details,
        "planning_effect_bps_per_day_to_calendar_days": days,
        "zero_true_effect_has_no_finite_80pct_power_duration": True,
        "alpha_each": .025, "power": .8, "normal_approximation_not_guaranteed_coverage": True,
        "not_a_duration_for_guaranteed_nonloss": True, "rolling_recipe_freeze_not_constant_future_weight_SHA": True,
        "single_final_analysis_no_interim_tuning": True, "new_market_data_required": True,
        "previously_revealed_periods_cannot_be_final_holdout": True, "safety": c.SAFETY})
    print(json.dumps({"prospective_activated": False, "planning_effect_bps_day_to_days": days, "new_training_variants": 0}))


def finalize():
    reg = load(False)
    parents()
    summary = sealed(OUT / "summary.json")
    if summary["step5_gate_passed"]:
        raise ValueError("paired weighting stage still required")
    required = ["direct_checks.json", "ledger_checks.json", "registration.json", "folds.json", "objective_registry_entry.json", "summary.json",
                "prospective_design_NOT_ACTIVATED.json", "REPORT_RU.md"]
    required += list(reg["bank_pins"])
    for period in reg["periods"]:
        required += [f"decisions_{period}.json", f"analysis_{period}.json"]
        required += [f"capital_drawdown_{period}_{scenario}.png" for scenario in SCENARIOS]
        required += [f"{kind}_{period}_{policy}_{scenario}.json" for kind in ("replay", "attribution") for policy in p.POLICIES for scenario in SCENARIOS]
    for period, rows in sealed(OUT / "folds.json")["regimes"].items():
        required += [f"fit_{period}_{row['day']:03d}.json" for row in rows]
    if any(not (OUT / name).is_file() for name in required):
        raise ValueError("required artifact missing")
    for name in required:
        if name.endswith(".json"):
            sealed(OUT / name)
    receipt = write("receipt.json", {"registration_sha256": reg["sha256"], "code_pins": reg["code_pins"],
        "artifact_pins": {name: base.pin(OUT / name) for name in required}, "new_model_fits": 123,
        "new_scaler_fits": 123, "offline_policy_replays": 84, "step5_gate_passed": False,
        "frozen_parent_chains_verified": 4, "network_market_requests": 0, "safety": c.SAFETY})
    print(json.dumps({"sealed_artifacts": len(receipt["artifact_pins"]), "new_fits": 123, "policy_replays": 84, "parents_unchanged": True, "live": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("checks", "ledger_checks", "prepare", "fit", "forecasts", "replays", "analyze", "draw", "summarize", "prospective", "finalize"))
    parser.add_argument("--period")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "fit":
        fit_batch(args.period, args.first, args.count)
    elif args.stage in ("forecasts", "replays", "analyze", "draw"):
        globals()[args.stage](args.period)
    else:
        globals()[args.stage]()
