"""Bounded foreground stages for one registered four-hour offline experiment."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import forecast4h_v1 as f

BASE = ROOT / "Reports/Exia/Genetic_Primus"
OUT = BASE / "forecast4h_v1_20261012"
CONFIG = ROOT / "configs/genetic_primus_forecast4h_v1.json"
PARENT = BASE / "cadence_matrix_v1_20261009/registration.json"
DATA = BASE / "compact_postclose_research_v1_20261008/data_bank.npz"
RATES = BASE / "compact_net_replay_v1_20261007/observed_rate_bank.json"
MODELED_RATES = BASE / "cadence_matrix_v1_20261009/modeled_rates.json"
PRIOR = BASE / "position_policy_v1_20261012"
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
CODE = (CONFIG, Path(__file__), ROOT / "src/exia/genetic_primus/forecast4h_v1.py",
        ROOT / "src/exia/genetic_primus/compact_forecast_v1.py",
        ROOT / "src/exia/genetic_primus/compact_forecast_data_v1.py",
        ROOT / "src/exia/genetic_primus/position_policy_v1.py",
        ROOT / "src/exia/genetic_primus/cadence_matrix_v1.py",
        ROOT / "src/exia/genetic_primus/compact_postclose_research_v1.py")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write(name, value):
    value = f.p.c.plain(value)
    value["sha256"] = f.m.identity(value)
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    return value


def sealed(path):
    value = read(path)
    if value.get("sha256") != f.m.identity({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("invalid JSON seal: " + str(path))
    return value


def stamp(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def registration():
    reg = sealed(OUT / "registration.json")
    effective_pins = reg["code_pins"] | reg["input_pins"]
    for amendment_path in sorted(OUT.glob("engineering_amendment_*.json")):
        amendment = sealed(amendment_path)
        if amendment["registration_sha256"] != reg["sha256"]:
            raise ValueError("engineering amendment registration mismatch")
        for name, change in amendment["code_changes"].items():
            allowed = (relative(Path(__file__)), "src/exia/genetic_primus/forecast4h_v1.py")
            if name not in allowed or change["original_pin"] != effective_pins[name]:
                raise ValueError("unsupported engineering amendment")
            if pin(OUT / change["archived_original"]) != change["original_pin"]:
                raise ValueError("original registered runner snapshot changed")
            effective_pins[name] = change["replacement_pin"]
        for name, expected in amendment.get("added_input_pins", {}).items():
            if name != relative(MODELED_RATES) or expected != sealed(PARENT)["bank_pins"]["modeled_rates.json"]:
                raise ValueError("unsupported funding source amendment")
            effective_pins[name] = expected
    for name, expected in effective_pins.items():
        if pin(ROOT / name) != expected:
            raise ValueError("registered code/input changed: " + name)
    for name, expected in reg["artifact_pins"].items():
        if pin(OUT / name) != expected:
            raise ValueError("registered folds/resample bank changed: " + name)
    return reg


def load_data():
    with np.load(DATA, allow_pickle=False) as stream:
        result = {k: stream[k] for k in ("times", "closes", "features")}
    result["symbols"] = list(SYMBOLS)
    return result


def checks():
    OUT.mkdir(parents=True, exist_ok=True)
    result = f.direct_checks(read(CONFIG))
    write("direct_checks.json", result)
    print(json.dumps(result))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    check = sealed(OUT / "direct_checks.json")
    contract, parent = read(CONFIG), sealed(PARENT)
    if any(contract["safety"].values()):
        raise ValueError("offline research authority must remain false")
    for path in (DATA, RATES):
        if pin(path) != parent["source_pins"][relative(path)]:
            raise ValueError("frozen source pin changed: " + relative(path))
    data = load_data()
    bank = f.samples(data["times"], data["closes"], data["features"])
    start, end = (stamp(contract["period"][k]) for k in ("start", "end_exclusive"))
    if start % f.DAY_MS or end % f.DAY_MS or start >= end:
        raise ValueError("complete UTC calendar interval required")
    if end > int(data["times"][-1]):
        raise ValueError("missing terminal price")
    with (OUT / "samples.npz").open("xb") as stream:
        np.savez_compressed(stream, **bank)
    days = (end - start) // f.DAY_MS
    cfg = contract["bootstrap"]
    bootstrap = {
        "primary": f.bank(days, cfg["repetitions"], cfg["primary_block_days"], cfg["primary_seed"]),
        "audit": f.bank(days, cfg["repetitions"], cfg["primary_block_days"], cfg["audit_seed"]),
        "sensitivity": f.bank(days, cfg["repetitions"], cfg["sensitivity_block_days"], cfg["primary_seed"] + 7)}
    with (OUT / "resample_bank.npz").open("xb") as stream:
        np.savez_compressed(stream, **bootstrap)
    folds = []
    for origin in range(start, end, 3 * f.DAY_MS):
        cutoff = origin - 8 * f.BAR_MS
        indices = np.flatnonzero(bank["available"] <= cutoff)
        folds.append({"origin_ms": origin, "cutoff_ms": cutoff, "train_origins": len(indices),
                      "train_start_ms": int(bank["origins"][indices[0]]),
                      "latest_label_available_ms": int(bank["available"][indices[-1]]),
                      "training_index_sha256": f.m.array_identity(indices),
                      "train_samples_sha256": f.m.array_identity(bank["origins"][indices], bank["available"][indices],
                                               bank["x"][indices].reshape(-1, 6), bank["y"][indices].reshape(-1))})
    write("folds.json", {"folds": folds, "no_random_folds": True, "inner_tuning_trials": 0})
    inputs = (PARENT, DATA, RATES, PRIOR / "replay_report_EMA_V_normal.json", PRIOR / "replay_report_EMA_V_stress.json",
              BASE / "research_review_20261012/release_manifest.json")
    reg = write("registration.json", {
        "schema": "forecast4h_registration/1", "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract, "contract_sha256": f.m.identity(contract), "start_ms": start, "end_ms": end,
        "symbols": list(SYMBOLS), "feature_definition_sha256": f.d.FEATURE_SHA256,
        "price_data_rows": len(data["times"]), "forecast_rows": (end - start) // f.BAR_MS,
        "fit_budget": len(folds), "replay_budget": 2 * len(f.POLICIES), "hyperparameter_trials": 1,
        "code_pins": {relative(path): pin(path) for path in CODE},
        "input_pins": {relative(path): pin(path) for path in inputs},
        "artifact_pins": {name: pin(OUT / name) for name in ("direct_checks.json", "samples.npz", "folds.json", "resample_bank.npz")},
        "direct_checks_passed": check["checks_passed"], "data_clock": "completed bar close timestamps, as frozen by parent",
        "selection_warning": "previously revealed data; within-pilot correction is not a global historical search correction",
        "safety": contract["safety"]})
    print(json.dumps({"registered": reg["sha256"], "fit_budget": len(folds), "forecast_rows": reg["forecast_rows"], "replay_budget": reg["replay_budget"]}))


def fit_batch(first, count):
    reg = registration()
    if not 1 <= count <= 3:
        raise ValueError("one to three foreground fits per invocation")
    folds = sealed(OUT / "folds.json")["folds"]
    if first < 0 or first + count > len(folds):
        raise ValueError("fit range out of bounds")
    with np.load(OUT / "samples.npz", allow_pickle=False) as stream:
        bank = {k: stream[k] for k in stream.files}
    elapsed = []
    for index in range(first, first + count):
        if (OUT / f"fit_{index:02d}.json").exists():
            raise ValueError("fit already exists; no overwrite")
        begin = time.perf_counter_ns()
        result = f.fit(bank, folds[index]["origin_ms"], reg["contract"])
        duration_us = (time.perf_counter_ns() - begin) // 1000
        if duration_us > reg["contract"]["model_ready_delay_us"]:
            raise ValueError("measured local fit exceeds registered model-ready allowance")
        if result["model"]["training_identity"]["train_samples_sha256"] != folds[index]["train_samples_sha256"]:
            raise ValueError("fit differs from frozen temporal fold")
        result["measured_local_fit_us"] = duration_us
        result["model_ready_is_assumed_not_observed_exchange_time"] = True
        write(f"fit_{index:02d}.json", result)
        elapsed.append(duration_us)
    print(json.dumps({"completed_fits": [first, first + count - 1], "max_fit_ms": max(elapsed) / 1000}))


def vectors():
    reg, data = registration(), load_data()
    folds = sealed(OUT / "folds.json")["folds"]
    fits = [sealed(OUT / f"fit_{i:02d}.json") for i in range(len(folds))]
    rows, residuals = [], []
    contract = reg["contract"]
    for origin in range(reg["start_ms"], reg["end_ms"], f.BAR_MS):
        fit_index = (origin - reg["start_ms"]) // (3 * f.DAY_MS)
        result = fits[fit_index]
        index = int(np.searchsorted(data["times"], origin))
        if data["times"][index] != origin or result["fit_ms"] > origin:
            raise ValueError("causal aligned forecast origin required")
        prediction = f.m.predict(result["model"], data["features"][index])
        # Only prior out-of-origin errors whose next close predates this close.
        if len(rows) >= 2:
            previous = rows[-2]
            previous_index = index - 2
            actual = data["closes"][previous_index + 1] / data["closes"][previous_index] - 1
            if previous["target_ms"] > origin - f.BAR_MS:
                raise ValueError("future residual in calibration bank")
            residuals.append(actual - np.asarray(previous["expected_return_4h"]))
        used = residuals[-contract["uncertainty"]["maximum_rows_per_asset"]:]
        error = np.sqrt(np.mean(np.square(used), axis=0)).tolist() if len(used) >= contract["uncertainty"]["minimum_rows_per_asset"] else None
        row = {"asof_ms": origin, "available_us": origin * 1000 + contract["model_ready_delay_us"],
               "target_ms": origin + f.BAR_MS, "expires_ms": origin + f.BAR_MS,
               "expected_return_4h": prediction.tolist(), "past_oof_error_rmse": error,
               "calibration_rows_per_asset": len(used), "calibration_latest_target_ms": origin - f.BAR_MS if used else None,
               "quality_status": "point_forecast_only" if error is None else "point_forecast_with_past_error_scale",
               "model_sha256": result["model"]["sha256"], "scaler_sha256": result["model"]["scaler_sha256"],
               "feature_definition_sha256": reg["feature_definition_sha256"], "fit_ms": result["fit_ms"],
               "training_cutoff_ms": result["cutoff_ms"],
               "controls": {"zero": [0.] * len(SYMBOLS), "pooled_train_mean": [result["pooled_train_mean"]] * len(SYMBOLS),
                            "asset_train_mean": result["asset_train_mean"],
                            "last_bar_return": (data["closes"][index] / data["closes"][index - 1] - 1).tolist()}}
        row["sha256"] = f.m.identity(row)
        rows.append(row)
    write("vectors.json", {"registration_sha256": reg["sha256"], "symbols": list(SYMBOLS), "vectors": rows,
                           "fit_pins": {f"fit_{i:02d}.json": pin(OUT / f"fit_{i:02d}.json") for i in range(len(fits))}})
    print(json.dumps({"forecast_times": len(rows), "asset_forecasts": len(rows) * len(SYMBOLS), "models": len(fits), "calibrated_times": sum(r["past_oof_error_rmse"] is not None for r in rows)}))


def vector_bank(reg):
    payload = sealed(OUT / "vectors.json")
    if payload["registration_sha256"] != reg["sha256"]:
        raise ValueError("vector registration mismatch")
    for name, expected in payload["fit_pins"].items():
        if pin(OUT / name) != expected:
            raise ValueError("fit changed after vector creation")
    for row in payload["vectors"]:
        if row["sha256"] != f.m.identity({k: v for k, v in row.items() if k != "sha256"}):
            raise ValueError("vector seal mismatch")
    return payload["vectors"]


def banks():
    with np.load(OUT / "resample_bank.npz", allow_pickle=False) as stream:
        return {k: stream[k] for k in stream.files}


def quality():
    reg, data = registration(), load_data()
    rows = vector_bank(reg)
    indices = np.searchsorted(data["times"], [r["asof_ms"] for r in rows])
    actual = data["closes"][indices + 1] / data["closes"][indices] - 1
    forecasts = {"linear4h": np.asarray([r["expected_return_4h"] for r in rows])}
    forecasts.update({name: np.asarray([r["controls"][name] for r in rows]) for name in reg["contract"]["forecast_controls"]})
    stats, daily_losses = {}, {}
    for name, prediction in forecasts.items():
        errors = prediction - actual
        daily_losses[name] = np.square(errors).reshape(-1, 6, 8).mean(axis=(1, 2))
        stats[name] = {"rmse_bps": float(np.sqrt(np.mean(errors ** 2)) * 10000),
                       "mae_bps": float(np.mean(abs(errors)) * 10000),
                       "direction_accuracy_nonzero_predictions": float(np.mean((prediction[abs(prediction) > 1e-15] > 0) == (actual[abs(prediction) > 1e-15] > 0))) if np.any(abs(prediction) > 1e-15) else None,
                       "per_asset_rmse_bps": np.sqrt(np.mean(errors ** 2, axis=0)) * 10000}
    alpha = reg["contract"]["bootstrap"]["family_alpha"] / (len(reg["contract"]["forecast_controls"]) + len(reg["contract"]["bootstrap"]["trade_contrasts"]))
    resamples, contrasts = banks(), {}
    for name in reg["contract"]["forecast_controls"]:
        improvement = daily_losses[name] - daily_losses["linear4h"]
        contrasts[name] = {key: f.lcb(improvement, indices, alpha) for key, indices in resamples.items()}
        contrasts[name]["relative_mse_skill"] = 1 - float(daily_losses["linear4h"].mean() / daily_losses[name].mean())
    prediction = forecasts["linear4h"]
    threshold = reg["contract"]["entry_strictly_above_bps"] / 10000
    opportunities = (data["features"][indices, :, 0] > 0) & (prediction > threshold)
    result = write("forecast_quality.json", {"metrics": stats, "contrasts_baseline_mse_minus_linear_mse": contrasts,
        "daily_mse": daily_losses, "nominal_alpha_per_contrast": alpha,
        "prediction_quantiles_bps": np.quantile(prediction * 10000, [0, .05, .5, .95, 1]),
        "actual_quantiles_bps": np.quantile(actual * 10000, [0, .05, .5, .95, 1]),
        "entry_opportunities_by_asset": dict(zip(SYMBOLS, opportunities.sum(axis=0).tolist())),
        "asset_forecast_rows": int(actual.size), "calendar_days": len(next(iter(daily_losses.values()))),
        "mean4h_prediction_bps": float(prediction.mean() * 10000),
        "forecast_gate": all(contrasts[name][key]["lcb"] > 0 for name in ("zero", "pooled_train_mean") for key in resamples),
        "gate_warning": "research only; all banks reuse the same revealed observations"})
    print(json.dumps({"rmse_bps": {k: round(v["rmse_bps"], 3) for k, v in stats.items()}, "forecast_gate": result["forecast_gate"]}))


def run_replay(policy, scenario):
    reg, data = registration(), load_data()
    rows = vector_bank(reg)
    covariances = {t: f.p.risk_covariance(data["times"], data["closes"], t)[0]
                   for t in range(reg["start_ms"], reg["end_ms"], f.DAY_MS)}
    result = f.replay(data, rows, covariances, sealed(MODELED_RATES)["events"], reg["contract"], policy, scenario)
    result["registration_sha256"] = reg["sha256"]
    result["vectors_pin"] = pin(OUT / "vectors.json")
    result["engineering_amendment_pins"] = {path.name: pin(path) for path in sorted(OUT.glob("engineering_amendment_*.json"))}
    if policy == "R0_EMA_daily":
        original = read(PRIOR / f"replay_report_EMA_V_{scenario}.json")
        for key in ("final_capital", "gross_pnl", "fees", "slippage", "funding", "max_drawdown", "mean_exposure"):
            if not np.isclose(result["metrics"][key], original["metrics"][key], atol=1e-7, rtol=1e-10):
                raise ValueError("daily EMA backward parity failed: " + key)
        if not np.allclose(result["daily_log_returns"], original["daily_log_returns"], atol=1e-11, rtol=0):
            raise ValueError("daily EMA path parity failed")
        result["saved_daily_ema_parity"] = True
    write(f"replay_{policy}_{scenario}.json", result)
    metric = result["metrics"]
    print(json.dumps({"policy": policy, "costs": scenario, "return_pct": round(metric["net_return"] * 100, 4), "dd_pct": round(metric["max_drawdown"] * 100, 4), "fills": metric["fills"]}))


def audit_replay(run, valid_models):
    cash, quantities, checks = float(run["metrics"]["initial_capital"]), {s: 0. for s in SYMBOLS}, 0
    for row in run["ledger"]:
        cash += row["cash_delta"]
        if row["kind"] == "modeled_fill":
            sign = 1 if row["side"] == "BUY" else -1
            expected = -sign * row["quantity"] * row["price"] - row["fee"]
            if abs(row["cash_delta"] - expected) > 1e-7:
                raise ValueError("fill cash formula failed")
            if abs(quantities[row["symbol"]] - row["quantity_before"]) > 1e-9:
                raise ValueError("inventory prefix failed")
            quantities[row["symbol"]] += sign * row["quantity"]
            if abs(quantities[row["symbol"]] - row["quantity_after"]) > 1e-9:
                raise ValueError("inventory delta failed")
            checks += 3
    if abs(cash - run["metrics"]["final_capital"]) > 1e-7 or max(abs(q) for q in quantities.values()) > 1e-8:
        raise ValueError("independent terminal cash/inventory check failed")
    if abs(sum(a["net_pnl"] for a in run["per_asset"].values()) - cash + 10000) > 1e-7:
        raise ValueError("asset contribution reconciliation failed")
    for row in run["decisions"]:
        if row["model_sha256"] not in valid_models or row["risk_last_data_ms"] >= row["asof_ms"]:
            raise ValueError("decision model or risk causality failed")
        checks += 2
    return checks + 2


def analyze():
    reg, sample_banks = registration(), banks()
    rows = vector_bank(reg)
    valid_models = {row["model_sha256"] for row in rows}
    runs, stats, audit_checks = {}, {}, 0
    for scenario in ("normal", "stress"):
        runs[scenario] = {policy: sealed(OUT / f"replay_{policy}_{scenario}.json") for policy in f.POLICIES}
        stats[scenario] = {}
        for policy, run in runs[scenario].items():
            if run["registration_sha256"] != reg["sha256"] or run["vectors_pin"] != pin(OUT / "vectors.json"):
                raise ValueError("replay provenance mismatch")
            audit_checks += audit_replay(run, valid_models)
            daily = np.asarray(run["daily_log_returns"])
            days = datetime.fromtimestamp(reg["start_ms"] / 1000, timezone.utc)
            split = 31  # The fixed August/September calendar split.
            stats[scenario][policy] = {"metrics": run["metrics"], "per_asset": run["per_asset"],
                "absolute_mean_daily_log_bps": {key: f.lcb(daily * 10000, indices, .05 / 9) for key, indices in sample_banks.items()},
                "monthly_returns": {"2026-08": float(np.expm1(daily[:split].sum())), "2026-09": float(np.expm1(daily[split:].sum()))},
                "top3_positive_day_share": float(np.sort(np.maximum(daily, 0))[-3:].sum() / np.maximum(daily, 0).sum()) if np.maximum(daily, 0).sum() else 0.}
    contrasts = {}
    for left, right in reg["contract"]["bootstrap"]["trade_contrasts"]:
        a, b = runs["stress"][left], runs["stress"][right]
        differences = (np.asarray(a["daily_log_returns"]) - np.asarray(b["daily_log_returns"])) * 10000
        contrasts[left + "-" + right] = {"total_return_difference_percentage_points": 100 * (a["metrics"]["net_return"] - b["metrics"]["net_return"]),
            "mean_daily_log_difference_bps": {key: f.lcb(differences, indices, .05 / 9) for key, indices in sample_banks.items()}}
    quality_result = sealed(OUT / "forecast_quality.json")
    gate = quality_result["forecast_gate"] and all(contrasts["R2_entry-R1_EMA_4h"]["mean_daily_log_difference_bps"][key]["lcb"] > 0 for key in sample_banks)
    result = write("analysis.json", {"registration_sha256": reg["sha256"], "policies": stats, "contrasts": contrasts,
          "independent_ledger_checks": audit_checks, "model_count": len(valid_models),
          "saved_daily_ema_parity": all(runs[s]["R0_EMA_daily"].get("saved_daily_ema_parity", False) for s in runs),
          "forecast_gate": quality_result["forecast_gate"], "further_model_search_gate": gate,
          "promote_to_real_trading": False, "independent_validation": False,
          "all_diagnostic_replays_retained": True})
    print(json.dumps({"ledger_checks": audit_checks, "daily_ema_parity": result["saved_daily_ema_parity"], "forecast_gate": quality_result["forecast_gate"], "further_model_search_gate": gate}))


def charts():
    registration()
    analysis = sealed(OUT / "analysis.json")
    import matplotlib.pyplot as plt
    colors = {"NoTrade": "#777777", "Passive_V": "#b58900", "R0_EMA_daily": "#1f77b4", "R1_EMA_4h": "#17becf",
              "R2_entry": "#2ca02c", "R3_exit": "#d62728", "R4_size": "#9467bd"}
    for scenario in ("normal", "stress"):
        target = OUT / f"capital_drawdown_{scenario}.png"
        if target.exists():
            raise ValueError("chart already exists")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for policy in f.POLICIES:
            run = sealed(OUT / f"replay_{policy}_{scenario}.json")
            points = [row for row in run["equity"] if row["phase"] == "calendar_bucket_end"]
            dates = [datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc) for row in points]
            capital = [row["nav"] for row in points]
            dd = [row["drawdown"] * 100 for row in points]
            metric = analysis["policies"][scenario][policy]["metrics"]
            label = f'{policy}: {metric["net_return"] * 100:+.2f}%'
            axes[0].plot(dates, capital, label=label, color=colors[policy], lw=1.4)
            axes[1].plot(dates, dd, color=colors[policy], lw=1.2)
        axes[0].set(title=f"4h forecast pilot | {scenario} costs | revealed Aug-Sep 2026", ylabel="Capital, USDT (initial 10,000)")
        axes[1].set(ylabel="Drawdown, %", xlabel="UTC date")
        axes[0].legend(loc="upper left", fontsize=8, ncol=2)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.text(.5, .01, "Research replay: modeled close fills and funding marks; not observed execution", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
        fig.savefig(target, dpi=150)
        plt.close(fig)
    quality_result = sealed(OUT / "forecast_quality.json")
    target = OUT / "forecast_error.png"
    if target.exists():
        raise ValueError("forecast chart already exists")
    fig, axis = plt.subplots(figsize=(10, 4))
    metrics = quality_result["metrics"]
    names = list(metrics)
    axis.bar(names, [metrics[name]["rmse_bps"] for name in names], color=["#2ca02c"] + ["#8899aa"] * 4)
    axis.set(ylabel="4h return RMSE, basis points", title="Causal out-of-origin forecast error | Aug-Sep 2026")
    axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(json.dumps({"charts": 3}))


def report():
    reg, analysis, quality_result = registration(), sealed(OUT / "analysis.json"), sealed(OUT / "forecast_quality.json")
    lines = ["# Четырёхчасовой прогноз и управление позицией", "", "12 октября 2026. Причинный исследовательский replay 1 августа — 30 сентября 2026; период уже раскрыт.", "",
        "## Контракт", "", "Одна pooled linear6 модель: 6 прежних ценовых признаков, 7 параметров, L2=0.01, uniform. Горизонт и шаг labels — 4 часа; решения — 4 часа; полное переобучение и отдельный scaler каждые 3 дня. Expanding train с 2022 года, только созревшие labels, purge 7 баров + один availability bar. Веса получают положительные и отрицательные исходы. Гиперпараметры, признаки и пороги по результату не менялись.", "",
        f"Зафиксированы registration `{reg['sha256']}`, hashes исходного price bank, rate bank, всех исходников, временных folds, training rows и общих bootstrap samples. Выполнено {reg['fit_budget']} основных обучений, {reg['forecast_rows']} временных прогнозов × 8 активов, {reg['replay_budget']} policy/cost replay. Дополнительные синтетические fit в engineering checks не являются поиском кандидатов.", "",
        "Каждый прогноз и каждое решение связаны с model SHA и scaler SHA. Ошибка прогноза оценивается только по созревшим прошлым out-of-origin residuals; в начале периода её ещё недостаточно. Вероятности и квантили не заявляются. Модель считается готовой через 1 секунду после close, сигнал через 2 секунды, условный fill через 2.25 секунды; локальное время fit проверено относительно 1 секунды. Цена close в этом допущении неизменна до fill.", "",
        "## Качество прогнозирования", "", "| Прогноз | RMSE, bps | MAE, bps |", "|---|---:|---:|"]
    for name, metric in quality_result["metrics"].items():
        lines.append(f"| {name} | {metric['rmse_bps']:.3f} | {metric['mae_bps']:.3f} |")
    lines += ["", f"Forecast gate относительно zero и pooled train mean: **{quality_result['forecast_gate']}**. Требование — положительный paired LCB улучшения MSE в primary, audit и 7-day sensitivity banks. Дневная агрегация учитывает одновременные активы и шесть прогнозов в день; независимых наблюдений здесь не 2928.", "",
        "![Ошибка прогнозов](forecast_error.png)", "", "## Торговые сравнения", "",
        "R0 — сохранённый EMA_V на дневной сетке; он повторно рассчитан новым ledger и сверён с сохранёнными результатами и всей дневной траекторией. R1 меняет сетку решений на 4 часа. R2 добавляет только entry filter: прогноз >30 bps = stress roundtrip 28 bps + фиксированный carry cushion 2 bps. Доли отклонённых активов не перераспределяются. Уже удерживаемая позиция не обязана снова проходить entry threshold.", "",
        "R3 добавляет выход, если прогноз <=−14 bps, либо прежний EMA exit. Это фиксированное исследовательское правило на границе бара, не оптимальный внутрисвечной выход. R4 умножает целевую долю R3 на clip(forecast/30bps, 0.25, 1). Доля может восстанавливаться до базовой EMA-доли; превышение её запрещено.", "",
        "Все варианты используют один lagged daily covariance estimator: 60 дневных доходностей строго до UTC дня решения, shrinkage 20%, individual floor 0.2%, дневная volatility target 0.5%, cap 10% на актив и 80% gross. После исключения активов риск при необходимости дополнительно снижается. На каждом решении выполняется rebalance по фактическому delta units: сохранение количества не создаёт новую entry fee; изменение количества создаёт соответствующие costs. Это может давать небольшие rebalance fills даже при прежнем направлении.", "",
        "Normal costs: fee 6 + drag 2 bps на сторону; stress: fee 6 + drag 8 bps. Funding — frozen historical rates × текущая позиция × reference close, без наблюдённых marks. В конце интервала позиции ликвидируются с costs. Ни плеча, ни short, ни заимствования.", ""]
    for scenario in ("normal", "stress"):
        lines += [f"### {scenario}", "", "| Вариант | Итог, % | Max DD, % | Экспозиция, % | Оборот / капитал | Fills | Удержание, ч |", "|---|---:|---:|---:|---:|---:|---:|"]
        for policy in f.POLICIES:
            metric = analysis["policies"][scenario][policy]["metrics"]
            lines.append(f"| {policy} | {metric['net_return']*100:+.3f} | {metric['max_drawdown']*100:.3f} | {metric['mean_exposure']*100:.2f} | {metric['turnover_initial_capital']:.2f} | {metric['fills']} | {metric['mean_holding_hours']:.1f} |")
        lines += ["", f"![Капитал и просадка, {scenario}](capital_drawdown_{scenario}.png)", ""]
    lines += ["## Приращение после stress-costs", "", "LCB — в bps средней дневной log-return разницы, не процент общей доходности. Общие circular block samples: 5000 повторов, 3 дня; независимый audit seed; sensitivity block 7 дней. Для 4 forecast + 5 trading contrasts α=0.05/9. Эта поправка относится только к текущему пилоту, не исправляет прошлую адаптивную настройку на раскрытом периоде.", "",
        "| Контраст | Разница итогов, п.п. | LCB primary, bps/day | LCB audit | LCB 7d |", "|---|---:|---:|---:|---:|"]
    for name, value in analysis["contrasts"].items():
        bounds = value["mean_daily_log_difference_bps"]
        lines.append(f"| {name} | {value['total_return_difference_percentage_points']:+.3f} | {bounds['primary']['lcb']:+.3f} | {bounds['audit']['lcb']:+.3f} | {bounds['sensitivity']['lcb']:+.3f} |")
    lines += ["", "## Атрибуция по активам и месяцам", "", "Stress PnL ниже дан в USDT и сходится с непрерывным капиталом. Непрерывный holding episode заканчивается только при закрытии позиции; partial rebalance не создаёт новый episode.", "",
        "| Вариант | Gross PnL | Fees | Slippage | Funding | Август, % | Сентябрь, % | Доля top-3 прибыльных дней |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, value in analysis["policies"]["stress"].items():
        metric = value["metrics"]
        lines.append(f"| {policy} | {metric['gross_pnl']:.2f} | {metric['fees']:.2f} | {metric['slippage']:.2f} | {metric['funding']:.2f} | {value['monthly_returns']['2026-08']*100:+.2f} | {value['monthly_returns']['2026-09']*100:+.2f} | {value['top3_positive_day_share']*100:.1f}% |")
    lines += ["", "| Актив | R1 net PnL | R2 net PnL | R3 net PnL | R4 net PnL | Entry opportunities |", "|---|---:|---:|---:|---:|---:|"]
    for symbol in SYMBOLS:
        values = [analysis["policies"]["stress"][policy]["per_asset"][symbol]["net_pnl"] for policy in ("R1_EMA_4h", "R2_entry", "R3_exit", "R4_size")]
        lines.append(f"| {symbol} | " + " | ".join(f"{value:+.2f}" for value in values) + f" | {quality_result['entry_opportunities_by_asset'][symbol]} |")
    best = max(f.POLICIES, key=lambda name: analysis["policies"]["stress"][name]["metrics"]["net_return"])
    lines += ["", "## Решение и дальнейшие работы", "",
        f"Наибольший итог среди зарегистрированных вариантов на этом раскрытом интервале: **{best}**. Это описательное сравнение, не независимый выбор будущего победителя. Gate для расширения поиска моделей: **{analysis['further_model_search_gate']}**; admission к реальным торгам: **False**.", ""]
    if analysis["further_model_search_gate"]:
        lines += ["Прежде чем расширять архитектуру, нужен один зафиксированный повтор выбранного контракта на других заранее выбранных режимах и затем новый prospective interval. Параметры текущего пилота повторно не подбирать.", ""]
    else:
        lines += ["Оснований сразу подключать MLP или GA этот пилот не даёт. Следующая полезная проверка — разобрать ошибки четырёхчасового прогноза по активам и режимам; затем зарегистрировать максимум один содержательно другой feature block. Если точность остаётся на уровне простых прогнозов, продолжать сравнение EMA/пассивного контроля с управлением риском, а не увеличивать сложность ради краткосрочного вектора.", ""]
    lines += ["Положительный net-итог при малом числе входов может означать удачную экспозицию в одном режиме. Снижение drawdown вместе со снижением экспозиции само по себе не подтверждает качество прогноза. Низкий RMSE среднего прогноза не гарантирует полезность редких прогнозов выше entry threshold; их хвостовой результат виден в отдельных policy replay.", "",
        "## Проверки и ограничения", "", f"Engineering checks: {reg['direct_checks_passed']}. Независимых арифметических ledger/causality checks: {analysis['independent_ledger_checks']}. Saved daily EMA parity: {analysis['saved_daily_ema_parity']}. Все 21 fit, forecasts и policies сохранены, неудачные варианты не удалялись.", "",
        "OHLCV replay не доказывает реализуемость fills, latency, liquidity, min lot и funding marks. Drawdown измерен по bar close и промежуточным модельным events; внутрисвечная просадка неизвестна. Nominal model-ready delay и fill price — допущения. Короткий период и bootstrap не заменяют новые данные. Успешная проверка четырёхчасового прогноза не доказывает оптимальное время закрытия внутри свечи. Данные стакана/сделок и вероятностные heads в этом пилоте не добавлялись.", "",
        "При первом replay исправлена инженерная ошибка адаптера risk_covariance: из пары (matrix, metadata) требуется matrix. Исходный зарегистрированный runner сохранён побайтно; изменение отражено в [engineering amendment](engineering_amendment_01.json). Обучение, прогнозы, параметры и пороги не пересматривались.", "",
        "Parity-проверка затем выявила 8 отсутствующих в раннем rate bank строк на terminal settlement. Подключён полный уже замороженный modeled rate bank прежнего evaluator; source pin подтверждён parent registration. [Второе инженерное исправление](engineering_amendment_02.json) сохраняет обе версии runner. Это reported rates с условной close valuation, не наблюдённые settlement cashflows.", "",
        "Credentials, orders, paper/live, runtime, promotion и network authority остаются false. Collector, WebSocket и торговый runtime не использовались.", "",
        "## Артефакты", "", "- [Регистрация](registration.json)", "- [Временные folds](folds.json)", "- [Прогнозный поток](vectors.json)", "- [Качество прогнозов](forecast_quality.json)", "- [Сводный анализ и атрибуция](analysis.json)", "- [Исходный план](../research_review_20261012_vectors_v2/REPORT_2_FORECAST_VECTOR_PLAN_RU.md)", ""]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    files = {path.name: pin(path) for path in OUT.iterdir() if path.is_file()}
    write("receipt.json", {"schema": "forecast4h_receipt/1", "registration_sha256": reg["sha256"],
          "artifact_pins": files, "financial_fits": reg["fit_budget"], "replays": reg["replay_budget"],
          "runtime_or_network_started": False})
    print(json.dumps({"report": relative(OUT / "REPORT_RU.md"), "best_revealed_stress_return": best, "files_pinned": len(files)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("checks", "prepare", "fit", "vectors", "quality", "replay", "analyze", "charts", "report"))
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--policy", choices=f.POLICIES)
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit_batch(args.first, args.count)
    elif args.stage == "replay":
        if args.policy is None or args.scenario is None:
            parser.error("replay requires --policy and --scenario")
        run_replay(args.policy, args.scenario)
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, FileExistsError, FileNotFoundError, KeyError) as error:
        print(f"ERROR {type(error).__name__}: {str(error)[:500]}", file=sys.stderr)
        raise SystemExit(1) from None
