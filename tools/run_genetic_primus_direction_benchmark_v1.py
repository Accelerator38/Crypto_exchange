"""One bounded foreground stage per call; no network, collector or trading runtime."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus import direction_benchmark_v1 as d
from exia.genetic_primus import compact_forecast_v1 as m
from exia.genetic_primus import forecast4h_v1 as f
from exia.genetic_primus import position_policy_v1 as p
import run_genetic_primus_forecast4h_v1 as parent

BASE = ROOT / "Reports/Exia/Genetic_Primus"
OUT = BASE / "direction_benchmark_v1_20261012"
CONFIG = ROOT / "configs/genetic_primus_direction_benchmark_v1.json"
PREP = BASE / "compact_forecast_v1_20261007_r2/preparation.json"
PREVIOUS_REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v10.json"
NEXT_REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v11.json"
START = 1785542400000
END = 1790812800000
CONTROL_NAMES = ["train_majority", "asset_train_majority", "persistence", "ema_direction"]
FINANCIAL_CONTROLS = ["NoTrade", "Passive_V", "R1_EMA_4h"]
SOURCE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_benchmark_v1.py",
          ROOT / "src/exia/genetic_primus/compact_forecast_v1.py",
          ROOT / "src/exia/genetic_primus/forecast4h_v1.py", ROOT / "src/exia/genetic_primus/position_policy_v1.py",
          ROOT / "tools/run_genetic_primus_forecast4h_v1.py"]


def read(path):
    if not 0 < path.stat().st_size < 20_000_000:
        raise ValueError("bounded JSON required")
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def pin(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    value = f.p.c.plain(value)
    if "sha256" in value:
        raise ValueError("write unsealed body only")
    body = {**value, "sha256": m.identity(value)}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return body


def sealed(path):
    value = read(path)
    if value.get("sha256") != m.identity({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("artifact content hash mismatch: " + str(path))
    return value


def registration():
    reg = sealed(OUT / "registration.json")
    for name, expected in reg["source_pins"].items():
        if pin(ROOT / name) != expected:
            raise ValueError("registered source changed: " + name)
    for name, expected in reg["bank_pins"].items():
        if pin(OUT / name) != expected:
            raise ValueError("registered bank changed: " + name)
    if any(read(CONFIG)["safety"].values()):
        raise ValueError("fail-closed safety changed")
    return reg


def load_bank():
    with np.load(OUT / "feature_bank.npz", allow_pickle=False) as stream:
        return {k: stream[k] for k in stream.files}


def split(times, origin):
    candidates = np.arange(48, len(times) - 1)
    cutoff_us = origin * 1000 - 7 * d.BAR * 1000
    train = candidates[times[candidates + 1] * 1000 + 2_250_000 <= cutoff_us]
    test = candidates[(times[candidates] >= origin) & (times[candidates] < min(origin + 3 * d.DAY, END))]
    if not len(train) or not len(test) or times[train[-1] + 1] * 1000 + 2_250_000 > cutoff_us:
        raise ValueError("invalid causal origin")
    return train, test, cutoff_us


def prepare():
    if OUT.exists() or NEXT_REGISTRY.exists():
        raise ValueError("exclusive experiment root and registry version required")
    config = read(CONFIG)
    if config["models"] != d.MODELS or config["budget"]["fits"] != 252 or any(config["safety"].values()):
        raise ValueError("fixed model budget and safety required")
    checks = d.direct_checks()
    import pandas as pd
    import pyarrow.parquet as pq
    sources = read(PREP)["sources"]
    frames = []
    for name, expected in sources.items():
        path = ROOT / name
        path.resolve().relative_to(ROOT / "Retrodate")
        if pin(path) != expected:
            raise ValueError("frozen OHLCV changed: " + name)
        frame = pq.ParquetFile(path, pre_buffer=False, memory_map=False).read(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"], use_threads=False).to_pandas(use_threads=False)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    if panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("duplicate OHLCV")
    original = parent.load_data()
    arrays = []
    for name in ("open", "high", "low", "close", "volume"):
        pivot = panel.pivot(index="timestamp", columns="symbol", values=name).reindex(columns=parent.SYMBOLS)
        if not np.array_equal(pivot.index.to_numpy(np.int64) + d.BAR, original["times"]):
            raise ValueError("completed-bar timestamps must exactly match frozen bank")
        arrays.append(pivot.to_numpy(float))
    ohlcv = np.stack(arrays, axis=2)
    features = d.feature_panel(original["closes"], original["features"], ohlcv)
    times, closes = original["times"], original["closes"]
    if np.any(np.diff(times) != d.BAR) or times[0] != 1641009600000 or times[-1] != END:
        raise ValueError("full continuous completed-bar history required")
    returns = closes[1:] / closes[:-1] - 1
    origins = list(range(START, END, 3 * d.DAY))
    folds = []
    for origin in origins:
        train, test, cutoff = split(times, origin)
        folds.append({"origin_ms": origin, "cutoff_us": cutoff, "train_first_asof_ms": int(times[train[0]]),
                      "train_last_asof_ms": int(times[train[-1]]), "latest_label_available_us": int(times[train[-1] + 1] * 1000 + 2_250_000),
                      "train_origins": len(train), "test_origins": len(test),
                      "train_index_sha256": m.array_identity(train), "test_index_sha256": m.array_identity(test),
                      "train_data_sha256": m.array_identity(features[train], returns[train])})
    OUT.mkdir()
    with (OUT / "feature_bank.npz").open("xb") as stream:
        np.savez_compressed(stream, times=times, closes=closes, features=features, returns=returns)
    with (OUT / "resample_bank.npz").open("xb") as stream:
        np.savez_compressed(stream, primary=f.bank(61, 5000, 3, 48101), audit=f.bank(61, 5000, 3, 48103))
    test_ix = np.flatnonzero((times[:-1] >= START) & (times[:-1] < END))
    audit = {"completed_bars": len(times), "assets": list(parent.SYMBOLS), "source_files": len(sources),
             "nonfinite": 0, "unresolved_duplicates": 0, "ohlcv_close_exact_match": True,
             "test_forecasts": int(len(test_ix) * 8), "test_flat_labels": int((returns[test_ix] == 0).sum()),
             "test_up_fraction_nonflat": float((returns[test_ix][returns[test_ix] != 0] > 0).mean()),
             "test_max_abs_return_bps": float(np.abs(returns[test_ix]).max() * 10000),
             "test_zero_volume_bars": int((ohlcv[test_ix, :, 4] == 0).sum()),
             "independent_exchange_confirmation": False, "observed_l2_or_marks_qualified": False,
             "feature_availability": "raw open timestamp + four hours; completed OHLCV only",
             "flat_bars_not_relabelled_or_removed_from_financial_replay": True}
    write(OUT / "data_audit.json", audit)
    source_paths = SOURCE + [PREP, parent.DATA, parent.MODELED_RATES, parent.CONFIG, PREVIOUS_REGISTRY]
    source_pins = {relative(path): pin(path) for path in source_paths}
    source_pins.update(sources)
    reg = write(OUT / "registration.json", {
        "schema": "direction_benchmark_registration/1", "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_pins": source_pins,
        "bank_pins": {name: pin(OUT / name) for name in ("feature_bank.npz", "resample_bank.npz", "data_audit.json")},
        "feature_definition": {"base": list(m.FEATURES), "added": d.EXTRA_FEATURES,
                               "volume": "log1p(current completed volume) minus mean log1p(volume) of preceding 42 bars"},
        "models": {name: ("ridge6: normalized return ridge L2=0.01; train residual Gaussian CDF" if name == "ridge6"
                           else d.parameters(name).get_params()) for name in d.MODELS},
        "folds": folds, "folds_sha256": m.identity(folds), "direct_checks": checks,
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas", "pyarrow", "joblib")},
        "maximum_fits": 252, "maximum_financial_replays": 30,
        "repeatedly_revealed_period": True, "search_budget_is_not_independent_trials": True,
        "safety": config["safety"]})
    previous = read(PREVIOUS_REGISTRY)
    body = {k: v for k, v in previous.items() if k != "sha256"}
    body["registry_id"] = "global_v11_direction_benchmark_disclosure"
    body["predecessor"] = {"path": relative(PREVIOUS_REGISTRY), **pin(PREVIOUS_REGISTRY)}
    body["objectives"] = list(previous["objectives"]) + [{
        "objective_id": config["experiment_id"], "family": "twelve_fixed_direction_models_revealed_interval",
        "config_path": relative(CONFIG), "config_sha256": pin(CONFIG)["sha256"],
        "registration_path": relative(OUT / "registration.json"), "registration_sha256": reg["sha256"],
        "fit_budget": 252, "financial_replay_budget": 30, "selection_bias_removed": False,
        "historical_adaptive_trial_count_complete": False, "promotion_authority": False}]
    write(NEXT_REGISTRY, body)
    print(json.dumps({"registered_models": 12, "origins": len(folds), "fits_cap": 252,
                      "test_rows": audit["test_forecasts"], "flat": audit["test_flat_labels"], "checks": checks["passed"]}))


def fit_origin(number):
    import joblib
    reg, bank = registration(), load_bank()
    if not 0 <= number < 21:
        raise ValueError("registered origin index required")
    fold = reg["folds"][number]
    train, test, cutoff = split(bank["times"], fold["origin_ms"])
    if m.array_identity(bank["features"][train], bank["returns"][train]) != fold["train_data_sha256"]:
        raise ValueError("training fold hash mismatch")
    x = bank["features"][train].reshape(-1, 12)
    y = bank["returns"][train].reshape(-1)
    xx = bank["features"][test].reshape(-1, 12)
    entries = {}
    for name in d.MODELS:
        prefix = f"origin_{number:02d}_{name}"
        meta_path, model_path, pred_path = OUT / (prefix + ".json"), OUT / (prefix + ".joblib"), OUT / (prefix + ".npz")
        if meta_path.exists():
            meta = sealed(meta_path)
            if pin(model_path) != meta["model_pin"] or pin(pred_path) != meta["prediction_pin"]:
                raise ValueError("partial origin has changed saved fit")
        else:
            if model_path.exists() or pred_path.exists():
                raise ValueError("incomplete fit artifact; do not silently refit")
            fitted = d.fit(name, x, y)
            prediction = d.predict(fitted, xx).reshape(len(test), 8)
            joblib.dump(fitted, model_path, compress=3)
            restored = joblib.load(model_path)
            if not np.array_equal(prediction, d.predict(restored, xx).reshape(len(test), 8)):
                raise ValueError("saved model prediction parity failed")
            with pred_path.open("xb") as stream:
                np.savez_compressed(stream, times=bank["times"][test], probability=prediction)
            info = fitted["info"]
            n = len(info["features"])
            selected = x[y != 0, :n]
            scale = selected.std(axis=0)
            scale[scale < 1e-12] = 1
            if not np.array_equal(selected.mean(axis=0), info["mean"]) or not np.array_equal(scale, info["scale"]):
                raise ValueError("train-only scaler check failed")
            meta = write(meta_path, {"schema": "saved_direction_fit/1", "registration_sha256": reg["sha256"],
                "origin_index": number, "origin_ms": fold["origin_ms"], "cutoff_us": cutoff,
                "latest_label_available_us": fold["latest_label_available_us"],
                "fold_sha256": m.identity(fold), "info": info, "model_pin": pin(model_path), "prediction_pin": pin(pred_path),
                "decision_model_sha256": pin(model_path)["sha256"], "prediction_parity": True, "train_only_scaler": True,
                "runtime_authority": False, "promotion_authority": False})
        entries[name] = {"metadata": meta_path.name, "sha256": meta["sha256"]}
    # Baseline probabilities are fitted only from the same mature labels.
    rr = bank["returns"][train]
    pooled = float((rr[rr != 0] > 0).mean())
    per_asset = np.array([(rr[:, a][rr[:, a] != 0] > 0).mean() for a in range(8)])
    prev = bank["closes"][test] / bank["closes"][test - 1] - 1
    control_path = OUT / f"origin_{number:02d}_controls.npz"
    if not control_path.exists():
        with control_path.open("xb") as stream:
            np.savez_compressed(stream, times=bank["times"][test],
                train_majority=np.full((len(test), 8), pooled), asset_train_majority=np.tile(per_asset, (len(test), 1)),
                persistence=np.where(prev >= 0, 0.75, 0.25), ema_direction=np.where(bank["features"][test, :, 0] > 0, 0.75, 0.25))
    receipt_path = OUT / f"origin_{number:02d}_receipt.json"
    if not receipt_path.exists():
        write(receipt_path, {"registration_sha256": reg["sha256"], "fits": entries, "controls_pin": pin(control_path),
                            "completed_models": 12, "new_fit_budget_used": 12, "test_origins": len(test)})
    print(json.dumps({"origin": number, "models": 12, "train_rows_nonflat": int((y != 0).sum()),
                      "test_forecasts": len(test) * 8, "latest_label_before_purge": True}))


def predictions(name):
    arrays, clocks, sha = [], [], []
    for i in range(21):
        receipt = sealed(OUT / f"origin_{i:02d}_receipt.json")
        if name in d.MODELS:
            meta = sealed(OUT / receipt["fits"][name]["metadata"])
            if meta["sha256"] != receipt["fits"][name]["sha256"]:
                raise ValueError("fit receipt mismatch")
            path = OUT / f"origin_{i:02d}_{name}.npz"
            if pin(path) != meta["prediction_pin"]:
                raise ValueError("prediction pin mismatch")
            key, model_sha = "probability", meta["decision_model_sha256"]
        else:
            path = OUT / f"origin_{i:02d}_controls.npz"
            if pin(path) != receipt["controls_pin"]:
                raise ValueError("control pin mismatch")
            key, model_sha = name, m.identity({"control": name, "receipt": receipt["sha256"]})
        with np.load(path, allow_pickle=False) as stream:
            arrays.append(stream[key]); clocks.append(stream["times"])
            sha += [model_sha] * len(stream["times"])
    times, result = np.concatenate(clocks), np.concatenate(arrays)
    if not np.array_equal(times, np.arange(START, END, d.BAR)) or result.shape != (366, 8):
        raise ValueError("complete common decision grid required")
    return times, result, sha


def analyze_forecasts():
    registration()
    bank = load_bank()
    ix = np.flatnonzero((bank["times"][:-1] >= START) & (bank["times"][:-1] < END))
    returns = bank["returns"][ix]
    _, control, _ = predictions("train_majority")
    valid = returns != 0
    base_correct = ((control >= 0.5) == (returns > 0)) & valid
    output = {}
    with np.load(OUT / "resample_bank.npz", allow_pickle=False) as stream:
        primary, audit = stream["primary"], stream["audit"]
    for name in d.MODELS + CONTROL_NAMES:
        times, probability, _ = predictions(name)
        summary = d.accuracy_summary(probability.flatten(), returns.flatten())
        correct = ((probability >= 0.5) == (returns > 0)) & valid
        count = valid.reshape(61, -1).sum(axis=1)
        correct_count = correct.reshape(61, -1).sum(axis=1)
        delta_count = (correct.astype(int) - base_correct.astype(int)).reshape(61, -1).sum(axis=1)
        for label, samples in (("primary", primary), ("audit", audit)):
            boot = correct_count[samples].sum(axis=1) / count[samples].sum(axis=1)
            delta = delta_count[samples].sum(axis=1) / count[samples].sum(axis=1)
            summary[label] = {"accuracy_interval_95": np.quantile(boot, [0.025, 0.975]).tolist(),
                              "paired_accuracy_delta": float(delta_count.sum() / count.sum()),
                              "paired_delta_lcb": float(np.quantile(delta, 0.05 / 12)),
                              "familywise_alpha": 0.05, "comparison_count": 12,
                              "simultaneous_accuracy_lcb": float(np.quantile(boot, 0.05 / 12))}
        summary["selective"] = {}
        for threshold in (0.6, 0.7, 0.8, 0.9):
            take = valid & (np.maximum(probability, 1 - probability) >= threshold)
            n = int(take.sum())
            summary["selective"][str(threshold)] = {"n": n, "coverage": n / int(valid.sum()),
                 "accuracy": float(correct[take].mean()) if n else None,
                 "up_fraction": float((returns[take] > 0).mean()) if n else None,
                 "calibrated_probability_claim": False}
        summary["by_asset"] = {symbol: d.accuracy_summary(probability[:, a], returns[:, a]) for a, symbol in enumerate(parent.SYMBOLS)}
        summary["by_month"] = {month: d.accuracy_summary(probability[mask].flatten(), returns[mask].flatten())
            for month, mask in (("2026-08", times < START + 31 * d.DAY), ("2026-09", times >= START + 31 * d.DAY))}
        bins = []
        for lower in np.arange(0, 1, 0.1):
            take = valid & (probability >= lower) & (probability < lower + 0.1)
            if take.any():
                bins.append({"lower": float(lower), "n": int(take.sum()), "mean_probability": float(probability[take].mean()),
                             "observed_up_fraction": float((returns[take] > 0).mean())})
        summary["reliability_bins"] = bins
        summary["goal_80_point_estimate_met"] = summary["accuracy"] >= 0.8
        summary["goal_80_lower_bound_met"] = min(summary[s]["simultaneous_accuracy_lcb"] for s in ("primary", "audit")) >= 0.8
        output[name] = summary
    ranked = sorted(d.MODELS, key=lambda name: output[name]["accuracy"], reverse=True)
    write(OUT / "forecast_comparison.json", {"models": output, "ranking_by_full_coverage_accuracy": ranked,
          "forecast_rows": int(valid.size), "nonflat": int(valid.sum()), "flat": int((~valid).sum()),
          "independent_validation": False, "selection_bias_removed": False})
    print(json.dumps({"models_scored": 12, "best": ranked[0], "accuracy": round(output[ranked[0]]["accuracy"], 5),
                      "any_80_full_coverage": any(output[n]["goal_80_point_estimate_met"] for n in d.MODELS)}))


def financial_replay(name):
    reg = registration()
    if name not in d.MODELS + FINANCIAL_CONTROLS:
        raise ValueError("registered financial policy required")
    bank = load_bank()
    times, probability, model_sha = predictions(name if name in d.MODELS else "train_majority")
    data = {"times": bank["times"], "closes": bank["closes"], "features": bank["features"][:, :, :6], "symbols": list(parent.SYMBOLS)}
    vectors = []
    for i, t in enumerate(times):
        vector = {"asof_ms": int(t), "target_ms": int(t + d.BAR), "available_us": int(t * 1000 + 1_000_000),
                  "model_sha256": model_sha[i], "probability_up": probability[i].tolist(),
                  "expected_return_4h": (probability[i] - 0.5).tolist(),
                  "transport_alias_not_return": True, "target_kind": "binary_probability_margin_not_price_or_return"}
        vectors.append({**vector, "sha256": m.identity(vector)})
    contract = read(parent.CONFIG)
    contract["direction_benchmark_registration_sha256"] = reg["sha256"]
    contract["direction_policy"] = "risk-bounded long if p_up >= 0.5; actual delta costs; no tuned threshold"
    covariances = {day: p.risk_covariance(data["times"], data["closes"], day)[0] for day in range(START, END, d.DAY)}
    rates = read(parent.MODELED_RATES)["events"]
    checks = {}
    for scenario in ("normal", "stress"):
        path = OUT / f"replay_{name}_{scenario}.json"
        if path.exists():
            raise ValueError("do not repeat registered financial replay")
        policy = "Passive_V" if name in d.MODELS else name
        rule = (lambda policy, margin, base, held, contract: base * (np.asarray(margin) >= 0)) if name in d.MODELS else None
        run = f.replay(data, vectors, covariances, rates, contract, policy, scenario, target_rule=rule)
        audit = parent.audit_replay(run, set(model_sha))
        write(path, {"model": name, "scenario": scenario, "registration_sha256": reg["sha256"],
                     "audit": audit, "run": run})
        checks[scenario] = round(run["metrics"]["net_return"] * 100, 5)
    print(json.dumps({"model": name, "return_percent": checks, "replays": 2, "live": False}))


def finalize():
    reg = registration()
    comparison = sealed(OUT / "forecast_comparison.json")
    results = {}
    with np.load(OUT / "resample_bank.npz", allow_pickle=False) as stream:
        primary, audit = stream["primary"], stream["audit"]
    for name in d.MODELS + FINANCIAL_CONTROLS:
        results[name] = {}
        for scenario in ("normal", "stress"):
            run = sealed(OUT / f"replay_{name}_{scenario}.json")["run"]
            base = sealed(OUT / f"replay_R1_EMA_4h_{scenario}.json")["run"]
            delta = np.asarray(run["daily_log_returns"]) - np.asarray(base["daily_log_returns"])
            results[name][scenario] = {"metrics": run["metrics"], "absolute_primary_lcb_bps_day": f.lcb(run["daily_log_returns"], primary, 0.05)["lcb"] * 10000,
                "paired_primary_lcb_bps_day": f.lcb(delta, primary, 0.05 / 12)["lcb"] * 10000,
                "paired_audit_lcb_bps_day": f.lcb(delta, audit, 0.05 / 12)["lcb"] * 10000}
    ranked = sorted(d.MODELS, key=lambda n: results[n]["stress"]["metrics"]["net_return"], reverse=True)
    write(OUT / "financial_comparison.json", {"models": results, "ranking_by_stress_return": ranked,
          "replays": 30, "fits": 252, "promotion_authority": False, "observed_execution_qualified": False})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    acc_names = comparison["ranking_by_full_coverage_accuracy"]
    accuracies = [comparison["models"][n]["accuracy"] * 100 for n in acc_names]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(acc_names[::-1], accuracies[::-1], color="#3f80b4")
    ax.axvline(50, color="gray", linestyle=":", label="50%")
    ax.axvline(80, color="#ba3a34", linestyle="--", label="Цель 80%")
    ax.set(xlim=(0, 100), xlabel="Верное направление, %; все ненулевые исходы", title="12 моделей: раскрытый август–сентябрь 2026")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "direction_accuracy.png", dpi=140); plt.close(fig)
    shown = ranked[:5] + FINANCIAL_CONTROLS
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name in shown:
        run = sealed(OUT / f"replay_{name}_stress.json")["run"]
        rows = run["equity"]
        # Use every ledger mark, including pre/post funding and actual fills.
        clocks = [datetime.fromtimestamp(row["timestamp_ms"] / 1000, timezone.utc) for row in rows]
        nav = np.array([row["nav"] for row in rows])
        axes[0].plot(clocks, nav, label=name, linewidth=1.5)
        axes[1].plot(clocks, (nav / np.maximum.accumulate(nav) - 1) * 100, linewidth=1.2)
    axes[0].set(ylabel="Капитал, USDT", title="Stress: пять лидеров по капиталу и общие контроли; модельные fills")
    axes[0].legend(ncol=3, fontsize=8); axes[1].set(ylabel="Просадка, %", xlabel="Дата UTC")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT / "stress_capital_drawdown.png", dpi=140); plt.close(fig)
    receipt = write(OUT / "verification.json", {"registration_sha256": reg["sha256"],
        "artifact_pins": {path.name: pin(path) for path in sorted(OUT.iterdir()) if path.is_file()},
        "completed_origins": 21, "completed_fits": 252, "completed_replays": 30,
        "per_fit_saved_prediction_parity": True, "per_fit_train_only_scaler": True,
        "per_replay_independent_cash_inventory_cost_funding_audit": True,
        "goal_80_full_coverage_met": any(comparison["models"][n]["goal_80_point_estimate_met"] for n in d.MODELS),
        "independent_validation": False, "safety": read(CONFIG)["safety"]})
    print(json.dumps({"fits": 252, "replays": 30, "best_stress": ranked[0], "pinned_artifacts": len(receipt["artifact_pins"]), "promotion": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "fit", "score", "replay", "finalize"])
    parser.add_argument("--origin", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "fit": fit_origin(args.origin)
    elif args.stage == "score": analyze_forecasts()
    elif args.stage == "replay": financial_replay(args.model)
    else: finalize()
