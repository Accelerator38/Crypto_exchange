"""Local one-day microstructure hypotheses, fixed models and causal receipt clock."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus import direction_selective_v1 as d
from exia.genetic_primus import direction_hypotheses_v1 as h
from run_genetic_primus_direction_selective_v1 import digest, identity, read, write

CONFIG = ROOT / "configs/genetic_primus_direction_micro_hypotheses_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_micro_hypotheses_v1_20261013"
PARENT = ROOT / "Reports/Exia/Genetic_Primus/direction_microstructure_v1_20261013"
DOWNLOADS = Path("C:/Users/anton/Downloads")
BOOK = DOWNLOADS / "bitget-futures_incremental_book_L2_2024-12-01_BTCUSDT.csv.gz"
DERIV = DOWNLOADS / "bitget-futures_derivative_ticker_2024-12-01_BTCUSDT.csv.gz"
START_MS = 1733011200000
MINUTE_US = 60_000_000
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_hypotheses_v1.py"]


def registration():
    reg = read(OUT / "registration.json")
    for p, sha in reg["code_pins"].items():
        if digest(ROOT / p) != sha: raise ValueError("registered micro hypothesis code changed")
    return reg


def init():
    if OUT.exists(): raise ValueError("exclusive engineering registration required")
    cfg = read(CONFIG)
    if any(cfg["safety"].values()) or len(cfg["models"])*3 != cfg["fit_budget"]: raise ValueError("fixed offline budget required")
    OUT.mkdir()
    reg = write(OUT / "registration.json", {"contract": cfg,
        "code_pins": {p.relative_to(ROOT).as_posix(): digest(p) for p in CODE},
        "source_pins": {str(p): digest(p) for p in [BOOK, DERIV]},
        "parent_bank_sha256": digest(PARENT / "minute_bank.npz"),
        "parent_registration_sha256": digest(PARENT / "registration.json"), "new_data_opened": False,
        "independent_validation": False, "safety": cfg["safety"]})
    predecessor = ROOT / "configs/genetic_primus_objective_registry_v16.json"
    previous = read(predecessor); registry = {k: v for k, v in previous.items() if k != "sha256"}
    entry = {"objective_id": cfg["objective_id"], "family": "direction_single_day_flow_book_derivatives_ablation",
        "registration_path": (OUT / "registration.json").relative_to(ROOT).as_posix(),
        "registration_sha256": reg["sha256"], "historical_fit_budget": 54, "financial_replays": 0,
        "historical_adaptive_trial_count_complete": False, "independent_validation": False, "promotion_authority": False}
    registry.update({"registry_id": "genetic_primus_objective_registry_v17",
        "previous_registry_path": predecessor.relative_to(ROOT).as_posix(), "previous_registry_sha256": digest(predecessor),
        "objectives": previous["objectives"]+[entry]})
    write(ROOT / "configs/genetic_primus_objective_registry_v17.json", registry)
    print(json.dumps({"registered": reg["sha256"], "fits": 54, "primary_rows": 216}))


def quotes():
    import pandas as pd
    registration()
    result = np.full((1440, 6), np.nan)
    stamps = np.full(1440, -1, np.int64)
    columns = ["local_timestamp", "side", "price", "amount"]
    carry = pd.DataFrame(columns=columns)
    count = 0
    def consume(frame):
        if frame.empty: return
        ts = frame["local_timestamp"].to_numpy(dtype=np.int64)
        if np.any(np.diff(ts) < 0): raise ValueError("chronological receipt rows required")
        frame = frame.copy()
        frame["minute"] = (ts-START_MS*1000)//MINUTE_US
        if ((frame["minute"] < 0) | (frame["minute"] >= 1440)).any(): raise ValueError("single registered day required")
        newest = frame.groupby("minute")["local_timestamp"].transform("max")
        tail_frames = frame[frame["local_timestamp"] == newest]
        for minute, packet in tail_frames.groupby("minute"):
            received = int(packet["local_timestamp"].iloc[0])
            bids = packet[(packet["side"] == "bid") & (packet["amount"] > 0)]
            asks = packet[(packet["side"] == "ask") & (packet["amount"] > 0)]
            if bids.empty or asks.empty: raise ValueError("complete two-sided atomic snapshot required")
            bid, ask = float(bids["price"].max()), float(asks["price"].min())
            if not 0 < bid < ask: raise ValueError("uncrossed quote required")
            bs = float(bids.loc[bids["price"] == bid, "amount"].sum())
            az = float(asks.loc[asks["price"] == ask, "amount"].sum())
            if received > stamps[minute]:
                stamps[minute] = received
                result[minute] = [bid, ask, bs, az, (bid+ask)/2,
                    (START_MS*1000+(minute+1)*MINUTE_US-received)/1000]
    for chunk in pd.read_csv(BOOK, usecols=columns, chunksize=1_000_000,
                             dtype={"local_timestamp": "int64", "side": "category", "price": "float64", "amount": "float64"}):
        count += len(chunk)
        if count > 24_000_000: raise ValueError("hard finite row cap exceeded")
        frame = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
        last = frame["local_timestamp"].iloc[-1]
        carry = frame[frame["local_timestamp"] == last].copy()
        consume(frame[frame["local_timestamp"] != last])
    consume(carry)
    with (OUT / "quotes.npz").open("xb") as f: np.savez_compressed(f, quotes=result, stamps=stamps)
    print(json.dumps({"source_rows": count, "quote_minutes": int(np.isfinite(result).all(axis=1).sum())}))


def derivative_bank(times):
    values = np.full((1440, 7), np.nan)
    fields = ["open_interest", "index_price", "mark_price", "last_price", "funding_rate", "funding_timestamp"]
    last = np.full(6, np.nan); last_us, next_minute, rows = None, 0, 0
    with gzip.open(DERIV, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["exchange"] != "bitget-futures" or row["symbol"] != "BTCUSDT": raise ValueError("registered derivative symbol required")
            received = int(row["local_timestamp"])
            if not START_MS*1000 <= received < (START_MS+d.DAY)*1000:
                raise ValueError("derivative receipt outside registered day")
            if last_us is not None and received < last_us: raise ValueError("derivative clock reversed")
            while next_minute < 1440 and times[next_minute]*1000 <= received:
                if last_us is not None: values[next_minute] = np.r_[last, (times[next_minute]*1000-last_us)/1000]
                next_minute += 1
            for j, field in enumerate(fields):
                if row[field] != "": last[j] = float(row[field])
            last_us = received; rows += 1
    while next_minute < 1440:
        values[next_minute] = np.r_[last, (times[next_minute]*1000-last_us)/1000]
        next_minute += 1
    return values, rows


def prepare():
    reg = registration()
    for p, sha in reg["source_pins"].items():
        if digest(p) != sha: raise ValueError("local market source changed")
    if digest(PARENT / "minute_bank.npz") != reg["parent_bank_sha256"]: raise ValueError("parent minute bank changed")
    with np.load(PARENT / "minute_bank.npz", allow_pickle=False) as a: old = {k: a[k] for k in a.files}
    with np.load(OUT / "quotes.npz", allow_pickle=False) as a: quote = a["quotes"].copy()
    deriv, count = derivative_bank(old["times"])
    full, price = old["price_and_microstructure"], old["price_only"]
    spread = (quote[:, 1]-quote[:, 0])/quote[:, 4]*10000
    if not np.allclose(spread, full[:, 9], atol=1e-8) or not np.array_equal(quote[:, 5], full[:, 15]):
        raise ValueError("quote extraction and original atomic book disagree")
    gap = (old["closes"]/quote[:, 4]-1)*10000
    imbalance1 = (quote[:, 2]-quote[:, 3])/(quote[:, 2]+quote[:, 3])
    quote_extra = np.column_stack([gap, imbalance1, np.r_[np.nan, np.diff(spread)]])
    oi = deriv[:, 0]
    oi1, oi5 = np.full(1440, np.nan), np.full(1440, np.nan)
    oi1[1:] = np.log(oi[1:]/oi[:-1]); oi5[5:] = np.log(oi[5:]/oi[:-5])
    extra = np.column_stack([oi1, oi5, (deriv[:, 2]/deriv[:, 1]-1)*10000,
        (deriv[:, 2]/deriv[:, 3]-1)*10000, deriv[:, 4]*10000,
        np.maximum(0, deriv[:, 5]/1000-old["times"])/(8*3600000)])
    lagged = []
    for lag in [1, 3, 6]:
        a = np.full((1440, 3), np.nan); a[lag:] = full[:-lag, [6, 12, 9]]; lagged.append(a)
    interactions = np.column_stack([full[:, 6]*full[:, 12], full[:, 6]*full[:, 14],
                                    full[:, 9]*full[:, 12], full[:, 13]*full[:, 12]])
    features = {"price": price, "full": full, "flow": np.column_stack([price, full[:, 6:9]]),
        "book": np.column_stack([price, full[:, 9:16]]), "lag_micro": np.column_stack([full, *lagged]),
        "derivatives": np.column_stack([full, extra]), "compact6": full[:, [0, 6, 12, 9, 14, 13]],
        "quotes": np.column_stack([full, quote_extra]), "interactions": np.column_stack([full, interactions])}
    eligible = (quote[:, 5] <= 5000) & (deriv[:, 6] <= 30000)
    for x in features.values(): eligible &= np.isfinite(x).all(axis=1)
    with (OUT / "feature_bank.npz").open("xb") as f:
        np.savez_compressed(f, **features, times=old["times"], closes=old["closes"], mid=quote[:, 4], eligible=eligible)
    write(OUT / "data_audit.json", {"bank_sha256": digest(OUT / "feature_bank.npz"),
        "quote_parity_with_original_book": True, "derivative_rows": count, "eligible_minutes": int(eligible.sum()),
        "feature_dimensions": {k: v.shape[1] for k, v in features.items()}, "receipt_clock": True,
        "predicted_funding_rate_unused": True, "funding_is_known_displayed_rate_not_realized_future_payment": True})
    print(json.dumps({"derivative_rows": count, "eligible_minutes": int(eligible.sum()), "feature_sets": len(features)}))


def fit(horizon):
    import joblib
    reg = registration(); cfg = reg["contract"]
    if horizon not in [1, 10, 30]: raise ValueError("registered horizon required")
    if digest(OUT / "feature_bank.npz") != read(OUT / "data_audit.json")["bank_sha256"]:
        raise ValueError("micro feature bank changed")
    with np.load(OUT / "feature_bank.npz", allow_pickle=False) as a: bank = {k: a[k] for k in a.files}
    ix = np.arange(1440-horizon); t = bank["times"]
    valid = bank["eligible"][ix] & np.isfinite(bank["closes"][ix+horizon]) & np.isfinite(bank["mid"][ix+horizon])
    train = ix[valid & (t[ix+horizon] <= START_MS+(864-7*horizon)*60000)]
    cal = ix[valid & (t[ix] >= START_MS+864*60000) & (t[ix+horizon] <= START_MS+(1152-7*horizon)*60000)]
    test = ix[valid & (t[ix] >= START_MS+1152*60000)]
    if min(len(train), len(cal), len(test)) < 16: raise ValueError("adequate mature temporal splits required")
    y = d.signs(bank["closes"][train], bank["closes"][train+horizon]); nonflat = y != 0
    truth = d.signs(bank["closes"][test], bank["closes"][test+horizon])
    mid_truth = d.signs(bank["mid"][test], bank["mid"][test+horizon])
    rows = []
    controls = {"random_coin": np.random.default_rng(20261013+horizon).integers(0, 2, len(test)).astype(bool),
        "train_majority": np.full(len(test), np.sum(y == 1) >= np.sum(y == -1)),
        "momentum": bank["closes"][test] >= bank["closes"][test-1]}
    for variant in cfg["models"]:
        name = f"h{horizon}_{variant['id']}"; model_path = OUT / (name+".joblib")
        if model_path.exists(): raise ValueError("no repeated single-day model fits")
        x = bank[variant["features"]]
        model = h.fit_binary(variant["estimator"], x[train][nonflat], y[nonflat] > 0, 20261013+horizon, minimum_leaf=16)
        pc, pt = h.predict(model, x[cal]), h.predict(model, x[test])
        thresholds = [d.calibrate(pc, t[cal], "BTCUSDT", coverage, 20261013) for coverage in cfg["coverage_targets"]]
        masks = np.stack([d.emit(pt, t[test], "BTCUSDT", threshold, 20261013) for threshold in thresholds])
        joblib.dump(model, model_path, compress=3)
        if not np.array_equal(pt, h.predict(joblib.load(model_path), x[test])): raise ValueError("micro hypothesis model parity failed")
        with (OUT / (name+".npz")).open("xb") as f:
            np.savez_compressed(f, times=t[test], probability=pt, masks=masks, labels=truth, mid_labels=mid_truth)
        write(OUT / (name+".json"), {"model_sha256": digest(model_path), "predictions_sha256": digest(OUT / (name+".npz")),
            "variant": variant, "info": model["info"], "train_sha256": identity(t[train]), "calibration_sha256": identity(t[cal]),
            "train_latest_label_ms": int(t[train[-1]+horizon]), "calibration_first_ms": int(t[cal[0]]),
            "calibration_latest_label_ms": int(t[cal[-1]+horizon]), "test_first_ms": int(t[test[0]]),
            "thresholds": thresholds, "n_train": len(train), "n_calibration": len(cal), "saved_parity": True})
        for k, coverage in enumerate(cfg["coverage_targets"]):
            rows.append({"horizon_minutes": horizon, "model": variant["id"], "features": variant["features"],
                "coverage_target": coverage, **d.evaluate(pt >= .5, truth, masks[k]),
                "midprice_diagnostic": d.evaluate(pt >= .5, mid_truth, masks[k]),
                "controls": {n: d.evaluate(p, truth, masks[k]) for n, p in controls.items()},
                "independent_improvement": False, "warnings": model["info"]["warnings"]})
    write(OUT / f"results_h{horizon}.json", {"rows": rows, "fits": len(cfg["models"]),
        "engineering_only": True, "single_day": True, "safety": cfg["safety"]})
    print(json.dumps({"horizon": horizon, "fits": len(cfg["models"]), "rows": len(rows)}))


def audit():
    import joblib
    reg = registration(); checked = 0
    with np.load(OUT / "feature_bank.npz", allow_pickle=False) as a: bank = {k: a[k] for k in a.files}
    for horizon in [1, 10, 30]:
        for variant in reg["contract"]["models"]:
            name = f"h{horizon}_{variant['id']}"; meta = read(OUT / (name+".json"))
            if digest(OUT / (name+".joblib")) != meta["model_sha256"]: raise ValueError("micro hypothesis hash failed")
            if meta["train_latest_label_ms"] > meta["calibration_first_ms"]-7*horizon*60000:
                raise ValueError("micro train/calibration purge failed")
            if meta["calibration_latest_label_ms"] > meta["test_first_ms"]-7*horizon*60000:
                raise ValueError("micro calibration/test purge failed")
            with np.load(OUT / (name+".npz"), allow_pickle=False) as saved:
                ix = np.searchsorted(bank["times"], saved["times"])
                truth = np.sign(bank["closes"][ix+horizon]-bank["closes"][ix]).astype(np.int8)
                if not np.array_equal(truth, saved["labels"]): raise ValueError("micro independent sign failed")
                model = joblib.load(OUT / (name+".joblib"))
                if not np.array_equal(h.predict(model, bank[variant["features"]][ix]), saved["probability"]):
                    raise ValueError("micro saved forecast audit failed")
            checked += 1
    write(OUT / "audit.json", {"models": checked, "independent_sign": True, "purge": True, "saved_parity": True,
                               "sequence_completeness_verified": False, "independent_validation": False})
    print(json.dumps({"models_audited": checked}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["init", "quotes", "prepare", "fit", "audit"])
    parser.add_argument("--horizon", type=int)
    args = parser.parse_args()
    if args.stage == "fit": fit(args.horizon)
    else: globals()[args.stage]()
