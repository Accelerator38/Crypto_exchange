"""Bounded direction-only stages. No execution, funding, runtime or network imports."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import direction_multiscale_v1 as d
from exia.genetic_primus import compact_forecast_v1 as m

OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_multiscale_v1_20261012"
OLD = ROOT / "Reports/Exia/Genetic_Primus/direction_benchmark_v1_20261012"
OLD_BANK = ROOT / "Reports/Exia/Genetic_Primus/compact_postclose_research_v1_20261008/data_bank.npz"
CONFIG = ROOT / "configs/genetic_primus_direction_multiscale_v1.json"
PREVIOUS_REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v11.json"
NEXT_REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v12.json"
START, END, RAW_START = 1785542400000, 1790812800000, 1781913600000
SYMBOLS = ["ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT"]
TIMEFRAMES = [1, 10, 30, 240]
CONTROLS = ["random_coin", "train_majority", "asset_train_majority", "persistence", "reverse_last_bar", "ema_direction"]
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_multiscale_v1.py",
        ROOT / "src/exia/genetic_primus/direction_benchmark_v1.py", ROOT / "src/exia/genetic_primus/compact_forecast_v1.py"]


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def read(path):
    if not 0 < path.stat().st_size < 30_000_000: raise ValueError("bounded JSON required")
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""): digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def relative(path): return path.resolve().relative_to(ROOT).as_posix()


def write(path, value):
    value = plain(value)
    if "sha256" in value: raise ValueError("unsealed body required")
    result = {**value, "sha256": m.identity(value)}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def sealed(path):
    value = read(path)
    if value.get("sha256") != m.identity({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("seal mismatch: " + str(path))
    return value


def save_array(path, array):
    with path.open("xb") as stream: np.save(stream, array, allow_pickle=False)


def sources():
    base = ROOT / "Retrodate"
    paths = [base / "bitget_full8_1m_panel_v1_20220101_20260714/parts" / (month + ".parquet") for month in ("2026-06", "2026-07")]
    paths += [base / "bitget_full8_1m_holdout_panel_v1_20260715_20260817/parts" / (month + ".parquet") for month in ("2026-07", "2026-08")]
    paths += [base / "bitget_full8_1m_prospective_panel_v1_20260818_20260903/parts" / (month + ".parquet") for month in ("2026-08", "2026-09")]
    paths += [base / "bitget_full8_1m_prospective_panel_v1_20260904_20260906/parts/2026-09.parquet"]
    paths += [base / "genetic_primus_v8_clean_ohlcv_backfill_20260907_20260921_v2/derived_v1" / f"origin_{i:02d}_1m.parquet" for i in range(3, 8)]
    paths += [base / "genetic_primus_v8_clean_ohlcv_tail_20260922_20260927/derived_v1" / f"origin_{i:02d}_1m.parquet" for i in (8, 9)]
    paths += [base / "genetic_primus_v8_clean_ohlcv_final_20260928_20260930/derived_v1/origin_10_1m.parquet"]
    return paths


def data_stage():
    import pyarrow.parquet as pq
    if OUT.exists(): raise ValueError("exclusive new experiment root required")
    count = (END - RAW_START) // d.MINUTE
    times = RAW_START + np.arange(count, dtype=np.int64) * d.MINUTE
    values = np.full((count, 8, 5), np.nan)
    seen = np.zeros((count, 8), bool)
    symbol_map = {symbol: i for i, symbol in enumerate(SYMBOLS)}
    pins, duplicates = {}, 0
    for path in sources():
        pins[relative(path)] = pin(path)
        for batch in pq.ParquetFile(path, memory_map=False, pre_buffer=False).iter_batches(
                batch_size=32768, columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"], use_threads=False):
            ts = batch.column(0).to_numpy(zero_copy_only=False)
            take = (ts >= RAW_START) & (ts < END)
            if not take.any(): continue
            if ts.dtype.kind not in "iu" or np.any(ts[take] % d.MINUTE): raise ValueError("integer minute open timestamps required")
            row = ((ts[take] - RAW_START) // d.MINUTE).astype(np.int64)
            sy = np.array([symbol_map[s] for s in batch.column(1).to_pylist()], dtype=np.int8)[take]
            a = np.column_stack([batch.column(i).to_numpy(zero_copy_only=False)[take] for i in range(2, 7)])
            key = row * 8 + sy
            if len(np.unique(key)) != len(key): raise ValueError("within-batch duplicate minute keys")
            duplicate = seen[row, sy]
            if duplicate.any():
                if not np.array_equal(values[row[duplicate], sy[duplicate]], a[duplicate]):
                    raise ValueError("conflicting source minute prices")
                duplicates += int(duplicate.sum())
            values[row, sy] = a; seen[row, sy] = True
    if not seen.all(): raise ValueError("missing minute bars: " + str(int((~seen).sum())))
    if (not np.isfinite(values).all() or np.any(values[:, :, :4] <= 0) or np.any(values[:, :, 4] < 0)
            or np.any(values[:, :, 1] < np.maximum(values[:, :, 0], values[:, :, 3]))
            or np.any(values[:, :, 2] > np.minimum(values[:, :, 0], values[:, :, 3]))):
        raise ValueError("invalid minute OHLCV")
    tf_times, tf_ohlcv = d.aggregate(times, values, 240)
    with np.load(OLD_BANK, allow_pickle=False) as stream:
        old_t, old_c = stream["times"], stream["closes"]
    ix = np.searchsorted(old_t, tf_times)
    if not np.array_equal(old_t[ix], tf_times) or not np.array_equal(old_c[ix], tf_ohlcv[:, :, 3]):
        raise ValueError("independent minute aggregation disagrees with previous 4h closes")
    OUT.mkdir()
    save_array(OUT / "minute_open_times.npy", times)
    save_array(OUT / "minute_ohlcv.npy", values)
    result = write(OUT / "minute_data_audit.json", {"source_pins": pins, "raw_start_ms": RAW_START,
        "end_exclusive_ms": END, "rows_per_asset": count, "assets": SYMBOLS, "missing": 0,
        "identical_duplicates": duplicates, "source_files": len(pins), "raw_ohlcv_checks": True,
        "previous_4h_closes_exactly_reproduced": int(tf_ohlcv.shape[0] * 8),
        "completed_time_rule": "open timestamp plus bar width", "new_downloads": 0,
        "independent_exchange_confirmation": False, "synthetic_checks": d.direct_checks(),
        "bank_pins": {name: pin(OUT / name) for name in ("minute_open_times.npy", "minute_ohlcv.npy")}})
    print({"minute_bars_per_asset": count, "sources": len(pins), "missing": 0, "duplicate_rows": duplicates,
           "old_4h_close_matches": result["previous_4h_closes_exactly_reproduced"], "fits": 0})


def tf_path(minutes):
    if minutes not in TIMEFRAMES: raise ValueError("registered timeframe required")
    return OUT / f"tf_{minutes}m"


def load_bank(minutes):
    return {key: np.load(tf_path(minutes) / (key + ".npy"), allow_pickle=False, mmap_mode="r")
            for key in ("times", "closes", "features", "returns")}


def prepare(minutes):
    audit = sealed(OUT / "minute_data_audit.json")
    for name, expected in audit["bank_pins"].items():
        if pin(OUT / name) != expected: raise ValueError("minute source bank changed")
    path = tf_path(minutes)
    if path.exists(): raise ValueError("exclusive new timeframe preparation required")
    raw_t = np.load(OUT / "minute_open_times.npy", mmap_mode="r", allow_pickle=False)
    raw = np.load(OUT / "minute_ohlcv.npy", mmap_mode="r", allow_pickle=False)
    times, ohlcv = d.aggregate(raw_t, raw, minutes)
    closes = ohlcv[:, :, 3].copy()
    x = d.features(closes, ohlcv)
    returns = closes[1:] / closes[:-1] - 1
    if not np.array_equal(np.sign(returns), np.sign(closes[1:] - closes[:-1])):
        raise ValueError("return sign and price difference must match")
    path.mkdir()
    for key, value in (("times", times), ("closes", closes), ("features", x), ("returns", returns)):
        save_array(path / (key + ".npy"), value)
    folds = []
    all_candidates = np.arange(48, len(times) - 1)
    for i, origin in enumerate(range(START, END, 3 * d.DAY)):
        cutoff = origin - 7 * minutes * d.MINUTE
        candidates = all_candidates[(times[all_candidates] >= origin - 30 * d.DAY) & (times[all_candidates + 1] <= cutoff)]
        seed = np.random.SeedSequence([20261012, minutes, i])
        selected = np.sort(np.random.default_rng(seed).choice(candidates, min(4000, len(candidates)), replace=False))
        test = all_candidates[(times[all_candidates] >= origin) & (times[all_candidates] < min(origin + 3 * d.DAY, END))]
        assert len(test) == (min(origin + 3 * d.DAY, END) - origin) // (minutes * d.MINUTE)
        with (path / f"fold_{i:02d}.npz").open("xb") as stream:
            np.savez_compressed(stream, train=selected, test=test)
        folds.append({"origin_ms": origin, "cutoff_ms": cutoff, "candidate_origins": len(candidates),
            "selected_origins": len(selected), "test_origins": len(test), "latest_label_ms": int(times[selected[-1] + 1]),
            "train_first_ms": int(times[selected[0]]), "train_last_ms": int(times[selected[-1]]),
            "fold_pin": pin(path / f"fold_{i:02d}.npz"),
            "train_data_sha256": m.array_identity(x[selected], returns[selected])})
    test = all_candidates[(times[all_candidates] >= START) & (times[all_candidates] < END)]
    coins = np.empty((len(test), 8), np.uint8)
    for a in range(8): coins[:, a] = np.random.default_rng(np.random.SeedSequence([50521, minutes, a])).integers(0, 2, len(test))
    save_array(path / "random_coins.npy", coins)
    quality = {}
    for a, symbol in enumerate(SYMBOLS):
        delta = closes[test + 1, a] - closes[test, a]
        quality[symbol] = {"intervals": len(test), "up": int((delta > 0).sum()), "down": int((delta < 0).sum()),
                          "flat": int((delta == 0).sum()), "zero_volume": int((ohlcv[test, a, 4] == 0).sum())}
    prep = write(path / "preparation.json", {"timeframe_minutes": minutes, "folds": folds,
        "folds_sha256": m.identity(folds), "quality": quality, "feature_names": d.models.FEATURES,
        "bank_pins": {p.name: pin(p) for p in sorted(path.iterdir()) if p.is_file()},
        "financial_replays": 0, "network": False})
    print({"timeframe_minutes": minutes, "test_intervals_per_asset": len(test), "total_flats": sum(q["flat"] for q in quality.values()),
           "first_train_time_origins": folds[0]["selected_origins"], "fits": 0})


def register():
    config = read(CONFIG)
    if any(config["safety"].values()) or config["models"] != d.models.MODELS or NEXT_REGISTRY.exists():
        raise ValueError("fixed offline contract and new registry required")
    audit = sealed(OUT / "minute_data_audit.json")
    preparations = {str(t): sealed(tf_path(t) / "preparation.json") for t in TIMEFRAMES}
    with (OUT / "resample_bank.npz").open("xb") as stream:
        np.savez_compressed(stream, primary=d.sample_bank(61, 5000, 3, 48101), audit=d.sample_bank(61, 5000, 3, 48103))
    old_pins = {relative(path): pin(path) for path in [OLD_BANK, OLD / "forecast_comparison.json", OLD / "registration.json"]}
    old_pins.update({relative(path): pin(path) for path in OLD.glob("origin_*.npz") if "controls" not in path.name})
    reg = write(OUT / "registration.json", {"schema": "direction_multiscale_registration/1", "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": config, "source_pins": {
            **{relative(p): pin(p) for p in CODE + [PREVIOUS_REGISTRY]}, **audit["source_pins"], **old_pins},
        "bank_pins": {**audit["bank_pins"], "minute_data_audit.json": pin(OUT / "minute_data_audit.json"),
                      "resample_bank.npz": pin(OUT / "resample_bank.npz")},
        "timeframe_preparation_pins": {str(t): pin(tf_path(t) / "preparation.json") for t in TIMEFRAMES},
        "fit_budget": 1008, "financial_replays": 0, "maximum_train_rows_before_nonflat_filter": 32000,
        "different_timeframes_not_pure_horizon_ablation": True, "training_uses_selected_times_only": True,
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "pyarrow", "joblib")},
        "models": {name: ("normalized return ridge6 L2=0.01" if name == "ridge6" else d.models.parameters(name).get_params()) for name in d.models.MODELS},
        "safety": config["safety"]})
    prev = sealed(PREVIOUS_REGISTRY)
    body = {k: v for k, v in prev.items() if k != "sha256"}
    body["registry_id"] = "global_v12_direction_multiscale_disclosure"
    body["predecessor"] = {"path": relative(PREVIOUS_REGISTRY), **pin(PREVIOUS_REGISTRY)}
    body["objectives"] = prev["objectives"] + [{"objective_id": config["experiment_id"], "family": "direction_only_fixed_multiscale",
        "config_path": relative(CONFIG), "config_sha256": pin(CONFIG)["sha256"], "registration_path": relative(OUT / "registration.json"),
        "registration_sha256": reg["sha256"], "historical_fit_budget": 1008, "financial_replays": 0,
        "historical_adaptive_trial_count_complete": False, "independent_validation": False, "promotion_authority": False}]
    write(NEXT_REGISTRY, body)
    print({"registered_timeframes": TIMEFRAMES, "models": 12, "fits_cap": 1008, "financial_replays": 0})


def registration(minutes=None):
    reg = sealed(OUT / "registration.json")
    for name, expected in reg["source_pins"].items():
        if pin(ROOT / name) != expected: raise ValueError("registered source changed: " + name)
    for name, expected in reg["bank_pins"].items():
        if pin(OUT / name) != expected: raise ValueError("registered bank changed: " + name)
    if minutes is not None:
        path = tf_path(minutes)
        if pin(path / "preparation.json") != reg["timeframe_preparation_pins"][str(minutes)]: raise ValueError("preparation changed")
        prep = sealed(path / "preparation.json")
        for name, expected in prep["bank_pins"].items():
            if pin(path / name) != expected: raise ValueError("feature/fold bank changed: " + name)
    if any(read(CONFIG)["safety"].values()): raise ValueError("offline safety changed")
    return reg


def fit_origin(minutes, number):
    import joblib
    reg, path = registration(minutes), tf_path(minutes)
    prep, bank = sealed(path / "preparation.json"), load_bank(minutes)
    if not 0 <= number < 21: raise ValueError("bounded origin required")
    fold = prep["folds"][number]
    with np.load(path / f"fold_{number:02d}.npz", allow_pickle=False) as stream:
        train, test = stream["train"], stream["test"]
    x, r = np.asarray(bank["features"][train]), np.asarray(bank["returns"][train])
    if m.array_identity(x, r) != fold["train_data_sha256"]: raise ValueError("training fold changed")
    x, r, xx = x.reshape(-1, 12), r.reshape(-1), np.asarray(bank["features"][test]).reshape(-1, 12)
    if fold["latest_label_ms"] > fold["cutoff_ms"]: raise ValueError("purge failed")
    entries = {}
    for name in d.models.MODELS:
        prefix = f"origin_{number:02d}_{name}"
        metadata, model_path, prediction_path = path / (prefix + ".json"), path / (prefix + ".joblib"), path / (prefix + ".npz")
        if metadata.exists():
            meta = sealed(metadata)
            if pin(model_path) != meta["model_pin"] or pin(prediction_path) != meta["prediction_pin"]:
                raise ValueError("saved partial origin changed")
        else:
            if model_path.exists() or prediction_path.exists(): raise ValueError("incomplete fit; no silent refit")
            fitted = d.models.fit(name, x, r)
            probability = d.models.predict(fitted, xx).reshape(len(test), 8)
            joblib.dump(fitted, model_path, compress=3)
            restored = joblib.load(model_path)
            if not np.array_equal(probability, d.models.predict(restored, xx).reshape(len(test), 8)):
                raise ValueError("saved model parity failed")
            info = fitted["info"]; n = len(info["features"])
            selected = x[r != 0, :n]; scale = selected.std(axis=0); scale[scale < 1e-12] = 1
            if not np.array_equal(selected.mean(axis=0), info["mean"]) or not np.array_equal(scale, info["scale"]):
                raise ValueError("causal scaler failed")
            with prediction_path.open("xb") as stream: np.savez_compressed(stream, times=bank["times"][test], probability=probability)
            meta = write(metadata, {"registration_sha256": reg["sha256"], "timeframe_minutes": minutes, "origin_index": number,
                "origin_ms": fold["origin_ms"], "cutoff_ms": fold["cutoff_ms"], "latest_label_ms": fold["latest_label_ms"],
                "fold_sha256": m.identity(fold), "info": info, "model_pin": pin(model_path), "prediction_pin": pin(prediction_path),
                "saved_prediction_parity": True, "train_only_scaler": True, "network": False})
        entries[name] = {"metadata": metadata.name, "sha256": meta["sha256"]}
    control_path = path / f"origin_{number:02d}_controls.npz"
    if not control_path.exists():
        rr = np.asarray(bank["returns"][train])
        pooled = float((rr[rr != 0] > 0).mean())
        per_asset = np.array([(rr[:, a][rr[:, a] != 0] > 0).mean() for a in range(8)])
        prev = bank["closes"][test] - bank["closes"][test - 1]
        test_positions = ((bank["times"][test] - START) // (minutes * d.MINUTE)).astype(np.int64)
        coins = np.load(path / "random_coins.npy", mmap_mode="r", allow_pickle=False)[test_positions]
        with control_path.open("xb") as stream:
            np.savez_compressed(stream, times=bank["times"][test], random_coin=coins,
                train_majority=np.full((len(test), 8), pooled), asset_train_majority=np.tile(per_asset, (len(test), 1)),
                persistence=np.where(prev >= 0, 0.75, 0.25), reverse_last_bar=np.where(prev >= 0, 0.25, 0.75),
                ema_direction=np.where(bank["features"][test, :, 0] > 0, 0.75, 0.25))
    receipt = path / f"origin_{number:02d}_receipt.json"
    if not receipt.exists(): write(receipt, {"registration_sha256": reg["sha256"], "fits": entries,
        "controls_pin": pin(control_path), "models": 12, "financial_replays": 0})
    print({"tf_min": minutes, "origin": number, "fits": 12, "train_nonflat_rows": int((r != 0).sum()), "forecasts_per_model": len(test) * 8})


def predictions(minutes, name):
    path = tf_path(minutes)
    values, times = [], []
    for i in range(21):
        receipt = sealed(path / f"origin_{i:02d}_receipt.json")
        if name in d.models.MODELS:
            meta = sealed(path / receipt["fits"][name]["metadata"])
            file = path / f"origin_{i:02d}_{name}.npz"
            if meta["sha256"] != receipt["fits"][name]["sha256"] or pin(file) != meta["prediction_pin"]: raise ValueError("fit pin failed")
            key = "probability"
        else:
            file, key = path / f"origin_{i:02d}_controls.npz", name
            if pin(file) != receipt["controls_pin"]: raise ValueError("control pin failed")
        with np.load(file, allow_pickle=False) as stream:
            values.append(stream[key]); times.append(stream["times"])
    clocks, result = np.concatenate(times), np.concatenate(values)
    if not np.array_equal(clocks, np.arange(START, END, minutes * d.MINUTE)) or result.shape != (len(clocks), 8):
        raise ValueError("full common prediction grid required")
    return clocks, result


def score(minutes):
    from scipy.stats import binom
    reg, bank = registration(minutes), load_bank(minutes)
    ix = np.flatnonzero((bank["times"][:-1] >= START) & (bank["times"][:-1] < END))
    current, future = np.asarray(bank["closes"][ix]), np.asarray(bank["closes"][ix + 1])
    diff = future - current
    valid = diff != 0
    _, coins = predictions(minutes, "random_coin")
    random_correct = ((coins >= 0.5) & (diff > 0)) | ((coins < 0.5) & (diff < 0))
    with np.load(OUT / "resample_bank.npz", allow_pickle=False) as stream:
        banks = {key: stream[key] for key in stream.files}
    rows, paired_draws = [], {}
    steps = 1440 // minutes
    for name in d.models.MODELS + CONTROLS:
        clocks, probability = predictions(minutes, name)
        up = probability >= 0.5
        correct = (up & (diff > 0)) | (~up & (diff < 0))
        for a, symbol in enumerate(SYMBOLS):
            result = {"timeframe_minutes": minutes, "model": name, "symbol": symbol,
                      **d.metrics(up[:, a], current[:, a], future[:, a])}
            daily_correct = correct[:, a].reshape(61, steps).sum(axis=1)
            daily_count = valid[:, a].reshape(61, steps).sum(axis=1)
            daily_delta = (correct[:, a].astype(int) - random_correct[:, a].astype(int)).reshape(61, steps).sum(axis=1)
            for key, samples in banks.items():
                denominator = daily_count[samples].sum(axis=1)
                bootstrap = daily_correct[samples].sum(axis=1) / denominator
                delta = daily_delta[samples].sum(axis=1) / denominator
                result[key] = {"accuracy_nonflat_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                    "accuracy_all_ci95": np.quantile(daily_correct[samples].sum(axis=1) / (61 * steps), [0.025, 0.975]).tolist(),
                    "paired_random_delta": float(daily_delta.sum() / daily_count.sum()), "paired_delta_std": float(delta.std(ddof=1))}
                if name in d.models.MODELS:
                    paired_draws[f"{name}|{symbol}|{key}"] = delta
            result["random_null_ci95_nonflat"] = (np.asarray(binom.interval(0.95, result["n_nonflat"], 0.5)) / result["n_nonflat"]).tolist()
            result["random_expected_accuracy_all"] = 0.5 * result["n_nonflat"] / result["n_all"]
            result["by_month"] = {month: d.metrics(up[mask, a], current[mask, a], future[mask, a])
                for month, mask in (("2026-08", clocks < START + 31 * d.DAY), ("2026-09", clocks >= START + 31 * d.DAY))}
            rows.append(result)
    with (tf_path(minutes) / "paired_bootstrap_draws.npz").open("xb") as stream:
        np.savez_compressed(stream, **paired_draws)
    write(tf_path(minutes) / "scores.json", {"registration_sha256": reg["sha256"], "rows": rows,
        "bootstrap_draws_pin": pin(tf_path(minutes) / "paired_bootstrap_draws.npz"), "per_asset_scoring": True,
        "flat_outcomes_never_counted_as_down": True, "financial_replays": 0})
    trained = [r for r in rows if r["model"] in d.models.MODELS]
    best = max(trained, key=lambda r: r["accuracy_nonflat"])
    print({"tf_min": minutes, "model_asset_rows": len(rows), "highest_single_cell": [best["model"], best["symbol"], round(best["accuracy_nonflat"] * 100, 3)],
           "financial_replays": 0})


def audit_previous():
    registration(240)
    bank = load_bank(240)
    ix = np.flatnonzero((bank["times"][:-1] >= START) & (bank["times"][:-1] < END))
    current, future = np.asarray(bank["closes"][ix]), np.asarray(bank["closes"][ix + 1])
    archived = sealed(OLD / "forecast_comparison.json")["models"]
    rows, checks = [], 0
    for name in d.models.MODELS:
        probabilities, clocks = [], []
        for i in range(21):
            with np.load(OLD / f"origin_{i:02d}_{name}.npz", allow_pickle=False) as stream:
                probabilities.append(stream["probability"]); clocks.append(stream["times"])
        p, t = np.concatenate(probabilities), np.concatenate(clocks)
        assert np.array_equal(t, bank["times"][ix])
        for a, symbol in enumerate(SYMBOLS):
            correct, nonflat = 0, 0
            # Separate scalar implementation, without the metric function or vectorized return sign.
            for j in range(len(t)):
                before, after = float(current[j, a]), float(future[j, a])
                if after == before: continue
                nonflat += 1
                predicted_up = float(p[j, a]) >= 0.5
                correct += int((after > before and predicted_up) or (after < before and not predicted_up))
            calculated = correct / nonflat
            reference = archived[name]["by_asset"][symbol]["accuracy"]
            assert abs(calculated - reference) < 1e-15
            rows.append({"model": name, "symbol": symbol, "correct": correct, "nonflat": nonflat,
                         "independent_accuracy": calculated, "archived_accuracy": reference, "difference": calculated - reference})
            checks += 1
    write(OUT / "previous_4h_independent_audit.json", {"comparisons": rows, "exact_metric_matches": checks,
        "price_basis": "independently aggregated actual minute OHLCV; asof completed close versus next completed close",
        "aggregation_parity": sealed(OUT / "minute_data_audit.json")["previous_4h_closes_exactly_reproduced"],
        "one_bar_price_shift_detected": False, "old_exact_zeros_excluded_not_down": True, "new_fits": 0, "financial_replays": 0})
    print({"independent_model_asset_metric_matches": checks, "misaligned_targets": 0, "new_fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["data", "prepare", "register", "fit", "score", "audit-previous"])
    parser.add_argument("--minutes", type=int)
    parser.add_argument("--origin", type=int)
    args = parser.parse_args()
    if args.stage == "data": data_stage()
    elif args.stage == "prepare": prepare(args.minutes)
    elif args.stage == "register": register()
    elif args.stage == "fit": fit_origin(args.minutes, args.origin)
    elif args.stage == "score": score(args.minutes)
    else: audit_previous()
