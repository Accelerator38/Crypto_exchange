"""Fixed hypothesis matrix. One horizon/origin per short offline fit command."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus import direction_selective_v1 as d
from exia.genetic_primus import direction_hypotheses_v1 as h
from run_genetic_primus_direction_selective_v1 import digest, identity, plain, read, write

CONFIG = ROOT / "configs/genetic_primus_direction_hypotheses_v1.json"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_hypotheses_v1_20261013"
PARENT = ROOT / "Reports/Exia/Genetic_Primus/direction_selective_v1_20261013"
DATA = ROOT / "Reports/Exia/Genetic_Primus/direction_multiscale_v1_20261012"
ARCHIVE = ROOT / "configs/research_archive/hypotheses_v1.json"
START, END = 1785542400000, 1790812800000
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_hypotheses_v1.py"]


def cases(cfg):
    return [(m, v) for m in cfg["timeframes_minutes"] for v in cfg["variants"] if m in v["horizons"]]


def registration():
    reg = read(OUT / "registration.json")
    if identity({k: v for k, v in reg.items() if k != "sha256"}) != reg["sha256"]:
        raise ValueError("hypothesis registration modified")
    for name, sha in reg["code_pins"].items():
        if digest(ROOT / name) != sha: raise ValueError("registered hypothesis code/config changed")
    if digest(PARENT / "registration.json") != reg["parent_registration_file_sha256"]:
        raise ValueError("parent registration modified")
    parent = read(PARENT / "registration.json")
    for name, sha in parent["code_pins"].items():
        if digest(ROOT / name) != sha: raise ValueError("parent code/config changed")
    return reg


def init():
    cfg, parent, archive = read(CONFIG), read(PARENT / "registration.json"), read(ARCHIVE)
    if OUT.exists() or any(cfg["safety"].values()): raise ValueError("exclusive offline registration required")
    expected = sum(0 if v["estimator"] == "ensemble" else 21 for _, v in cases(cfg))
    if expected != cfg["fit_budget"] or len(cases(cfg))*21 != cfg["forecast_artifacts"]:
        raise ValueError("fixed evaluation budget mismatch")
    OUT.mkdir()
    reg = write(OUT / "registration.json", {"contract": cfg, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_pins": {p.relative_to(ROOT).as_posix(): digest(p) for p in CODE},
        "parent_registration_file_sha256": digest(PARENT / "registration.json"),
        "parent_result_file_sha256": digest(PARENT / "results.json"), "data_pins": parent["data_pins"],
        "archive_sha256": digest(ARCHIVE), "archive_hypotheses": archive["hypotheses"],
        "archive_rules": archive["rules_for_new_work"], "symbols": parent["contract"]["symbols"],
        "historical_only": True, "new_data_opened": False, "safety": cfg["safety"]})
    predecessor = ROOT / "configs/genetic_primus_objective_registry_v15.json"
    previous = read(predecessor)
    registry = {k: v for k, v in previous.items() if k != "sha256"}
    entry = {"objective_id": cfg["objective_id"], "family": "direction_causal_features_regularization_and_selection",
        "config_path": CONFIG.relative_to(ROOT).as_posix(), "config_sha256": digest(CONFIG),
        "registration_path": (OUT / "registration.json").relative_to(ROOT).as_posix(),
        "registration_sha256": reg["sha256"], "historical_fit_budget": expected,
        "historical_adaptive_trial_count_complete": False, "financial_replays": 0,
        "independent_validation": False, "promotion_authority": False}
    registry.update({"registry_id": "genetic_primus_objective_registry_v16",
        "previous_registry_path": predecessor.relative_to(ROOT).as_posix(), "previous_registry_sha256": digest(predecessor),
        "registration_date_utc": reg["created_utc"], "objectives": previous["objectives"]+[entry]})
    write(ROOT / "configs/genetic_primus_objective_registry_v16.json", registry)
    print(json.dumps({"registered": reg["sha256"], "hypotheses": len(cfg["variants"]), "fits": expected}))


def bank_for(minutes, reg):
    for name, sha in reg["data_pins"].items():
        if f"tf_{minutes}m/" in name and digest(ROOT / name) != sha: raise ValueError("frozen data bank changed")
    return {name: np.load(DATA / f"tf_{minutes}m/{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ["times", "closes", "features"]}


def parent_artifact(minutes, number, kind):
    prefix = PARENT / f"tf_{minutes}m_origin_{number:02d}_{kind}"
    meta = read(str(prefix)+".json")
    model_path, path = Path(str(prefix)+".joblib"), Path(str(prefix)+".npz")
    if digest(model_path) != meta["model_sha256"] or digest(path) != meta["predictions_sha256"]:
        raise ValueError("parent model/prediction changed")
    with np.load(path, allow_pickle=False) as a: saved = {k: a[k].copy() for k in a.files}
    return meta, model_path, saved


def forecast(payload, x):
    kind = payload["kind"]
    if kind == "ensemble":
        p = np.mean([d.predict(m, x) for m in payload["members"]], axis=0)
        score = d.confidence(p)
    elif kind == "flat_gate":
        p = d.predict(payload["direction"], x)
        score = h.predict(payload["move"], x) * np.maximum(p, 1-p)
    else:
        p = h.predict(payload["model"], x)
        score = d.confidence(p)
    return p, score


def fit_origin(minutes, number):
    import joblib
    reg = registration(); cfg = reg["contract"]
    if minutes not in cfg["timeframes_minutes"] or not 0 <= number < 21:
        raise ValueError("registered horizon and single bounded origin required")
    bank = bank_for(minutes, reg); t = bank["times"]
    origin, width = START+number*3*d.DAY, minutes*60000
    seed = cfg["seed"]+minutes*1000+number
    original, cal = d.split(t, origin, width, seed=seed)
    test = np.flatnonzero((t >= origin) & (t < min(origin+3*d.DAY, END)))
    if test[-1]+1 >= len(t): raise ValueError("complete test labels required")
    trained, reused = 0, 0
    for v in cfg["variants"]:
        if minutes not in v["horizons"]: continue
        prefix = OUT / f"tf_{minutes}m_origin_{number:02d}_{v['id']}"
        meta_path, model_path, pred_path = [Path(str(prefix)+s) for s in [".json", ".joblib", ".npz"]]
        if meta_path.exists():
            meta = read(meta_path)
            if digest(model_path) != meta["model_sha256"] or digest(pred_path) != meta["prediction_sha256"]:
                raise ValueError("completed hypothesis artifact changed")
            reused += 1; continue
        if model_path.exists() or pred_path.exists(): raise ValueError("partial hypothesis artifact; no silent refit")
        parent_meta, parent_model_path, parent = parent_artifact(minutes, number, v["parent"])
        if not np.array_equal(parent["times"], t[test]) or identity(t[original]) != parent_meta["fold"]["train_times_sha256"]:
            raise ValueError("matched parent fold required")
        train = h.larger_train(t, original, origin, width, v["train_cap"], seed) if "train_cap" in v else original
        y = d.signs(bank["closes"][train], bank["closes"][train+1]).reshape(-1)
        x = h.feature_rows(bank, train, v["features"], minutes).reshape(len(train)*8, -1)
        xc = h.feature_rows(bank, cal, v["features"], minutes).reshape(len(cal)*8, -1)
        xt = h.feature_rows(bank, test, v["features"], minutes).reshape(len(test)*8, -1)
        references, fit_count = [parent_meta["model_sha256"]], 1
        if v["estimator"] == "ensemble":
            members, references = [], []
            for previous in range(max(0, number-2), number+1):
                mm, mp, _ = parent_artifact(minutes, previous, "logistic")
                if mm["fold"]["train_latest_label_ms"] > t[cal[0]]-7*width:
                    raise ValueError("ensemble member overlaps calibration")
                members.append(joblib.load(mp)); references.append(mm["model_sha256"])
            payload = {"kind": "ensemble", "members": members, "info": {"members": len(members), "new_fits": 0}}
            fit_count = 0
        elif v["estimator"] == "flat_gate":
            move = h.fit_binary("logistic", x, y != 0, seed)
            payload = {"kind": "flat_gate", "direction": joblib.load(parent_model_path), "move": move,
                       "info": move["info"]}
        else:
            nonflat = y != 0
            weights = None
            if "half_life_days" in v:
                weights = np.repeat(h.recency_weights(t[train], origin, v["half_life_days"]), 8)[nonflat]
            model = h.fit_binary(v["estimator"], x[nonflat], y[nonflat] > 0, seed, weights,
                                 robust=v.get("robust", False))
            payload = {"kind": "normal", "model": model, "info": model["info"]}
        pc, sc = [a.reshape(-1, 8) for a in forecast(payload, xc)]
        pt, st = [a.reshape(-1, 8) for a in forecast(payload, xt)]
        thresholds, masks = {}, np.empty((4, len(test), 8), bool)
        for j, symbol in enumerate(reg["symbols"]):
            thresholds[symbol] = []
            for k, coverage in enumerate(cfg["coverage_targets"]):
                threshold = h.score_threshold(sc[:, j], t[cal], symbol, coverage, cfg["seed"])
                thresholds[symbol].append(threshold)
                masks[k, :, j] = h.score_mask(st[:, j], t[test], symbol, threshold, cfg["seed"])
        if v["estimator"] == "flat_gate" and not np.array_equal(pt, parent["probability"]):
            raise ValueError("flat gate must preserve parent directions")
        joblib.dump(payload, model_path, compress=3)
        restored = forecast(joblib.load(model_path), xt)
        if not all(np.array_equal(a.reshape(-1, 8), b) for a, b in zip(restored, [pt, st])):
            raise ValueError("saved hypothesis model parity failed")
        with pred_path.open("xb") as f:
            np.savez_compressed(f, times=t[test], probability=pt, confidence_score=st, masks=masks, labels=parent["labels"],
                random_coin=parent["random_coin"], train_majority=parent["train_majority"], momentum=parent["momentum"],
                parent_probability=parent["probability"], parent_masks=parent["masks"])
        fold = {"origin_ms": origin, "train_sha256": identity(t[train]), "calibration_sha256": identity(t[cal]),
                "latest_train_label_ms": int(t[train[-1]+1]), "calibration_start_ms": int(t[cal[0]]),
                "latest_calibration_label_ms": int(t[cal[-1]+1]), "cutoff_ms": origin-7*width,
                "train_times": len(train), "calibration_times": len(cal), "test_times": len(test)}
        write(meta_path, {"registration_sha256": reg["sha256"], "variant": v, "minutes": minutes,
            "origin_index": number, "fold": fold, "fold_sha256": identity(fold), "thresholds": thresholds,
            "model_sha256": digest(model_path), "prediction_sha256": digest(pred_path), "source_models": references,
            "fit_count": fit_count, "info": payload["info"], "saved_parity": True})
        trained += fit_count
    print(json.dumps({"minutes": minutes, "origin": number, "new_fits": trained, "reused": reused}))


def score():
    reg = registration(); cfg = reg["contract"]
    banks = [d.block_bank(61, 1000, 3, seed) for seed in cfg["bootstrap"]["seeds"]]
    rows, contrast_meta, draws = [], [], [[], []]
    for minutes, variant in cases(cfg):
        chunks = []
        for number in range(21):
            prefix = OUT / f"tf_{minutes}m_origin_{number:02d}_{variant['id']}"
            path = Path(str(prefix)+".npz"); meta = read(str(prefix)+".json")
            if digest(path) != meta["prediction_sha256"]: raise ValueError("hypothesis predictions modified")
            with np.load(path, allow_pickle=False) as a: chunks.append({k: a[k].copy() for k in a.files})
        arrays = {k: np.concatenate([a[k] for a in chunks], axis=1 if k in ["masks", "parent_masks"] else 0) for k in chunks[0]}
        times = arrays["times"]
        if np.any(np.diff(times) != minutes*60000): raise ValueError("continuous test ledger required")
        for j, symbol in enumerate(reg["symbols"]):
            y, p, parent_p = arrays["labels"][:, j], arrays["probability"][:, j] >= .5, arrays["parent_probability"][:, j] >= .5
            for k, coverage in enumerate(cfg["coverage_targets"]):
                mask = arrays["masks"][k, :, j]
                row = {"minutes": minutes, "model": variant["id"], "parent": variant["parent"], "symbol": symbol,
                    "coverage_target": coverage, **d.evaluate(p, y, mask), "comparisons": {}, "months": {}}
                daily = d.daily_counts(times, y, p, mask, START, 61)
                accuracy_draws = []
                for b, bank in enumerate(banks):
                    info, draw = d.resample_statistics(daily, bank)
                    row["bootstrap" if b == 0 else "audit_bootstrap"] = info
                    accuracy_draws.append(draw)
                for month, select in [("2026-08", times < 1788220800000), ("2026-09", times >= 1788220800000)]:
                    row["months"][month] = d.evaluate(p[select], y[select], mask[select])
                competitors = {name: (arrays[name][:, j], mask) for name in ["random_coin", "train_majority", "momentum"]}
                competitors.update({"parent_same_mask": (parent_p, mask), "parent_policy": (parent_p, arrays["parent_masks"][k, :, j])})
                for name, (cp, cmask) in competitors.items():
                    baseline = d.evaluate(cp, y, cmask)
                    row["comparisons"][name] = {"metrics": baseline}
                    cd = d.daily_counts(times, y, cp, cmask, START, 61)
                    delta = row["accuracy_all"]-baseline["accuracy_all"] if row["accuracy_all"] is not None and baseline["accuracy_all"] is not None else None
                    contrast_meta.append({"row": len(rows), "name": name, "delta": delta})
                    for b, bank in enumerate(banks):
                        _, control_draw = d.resample_statistics(cd, bank)
                        draws[b].append(accuracy_draws[b]-control_draw)
                rows.append(row)
    critical_values = []
    for b, name in enumerate(["bootstrap", "audit_bootstrap"]):
        rep = np.asarray(draws[b]).T
        valid = np.isfinite(rep).all(axis=0) & np.array([c["delta"] is not None for c in contrast_meta])
        observed = np.array([c["delta"] if c["delta"] is not None else 0 for c in contrast_meta])
        lower = np.full(len(contrast_meta), np.nan)
        lower[valid], critical = d.simultaneous_lcb(observed[valid], rep[:, valid])
        critical_values.append(critical)
        for z, c in enumerate(contrast_meta):
            rows[c["row"]]["comparisons"][c["name"]][name] = {
                "delta": c["delta"], "pointwise_ci95": np.quantile(rep[:, z], [.025, .975]).tolist() if valid[z] else None,
                "simultaneous_lcb95": float(lower[z]) if valid[z] else None}
    for row in rows:
        for mode, parent in [("signal", "parent_same_mask"), ("policy", "parent_policy")]:
            row["research_positive_"+mode] = all(
                row["comparisons"][control][bank]["simultaneous_lcb95"] is not None
                and row["comparisons"][control][bank]["simultaneous_lcb95"] > 0
                for control in ["random_coin", "train_majority", "momentum", parent]
                for bank in ["bootstrap", "audit_bootstrap"])
        row["independent_improvement"] = False
    write(OUT / "results.json", {"registration_sha256": reg["sha256"], "rows": rows,
        "shared_bank_sha256": [identity(bank) for bank in banks], "family_contrasts": len(contrast_meta),
        "max_t_critical": critical_values, "historical_only": True, "independent_improvement": False, "safety": cfg["safety"]})
    print(json.dumps({"rows": len(rows), "contrasts": len(contrast_meta),
        "positive_signal": sum(r["research_positive_signal"] for r in rows),
        "positive_policy": sum(r["research_positive_policy"] for r in rows)}))


def audit(minutes_filter, origin_filter):
    import joblib
    reg = registration(); checked, thresholds, warnings = 0, 0, 0
    if minutes_filter not in [1, 10, 30] or origin_filter is None or not 0 <= origin_filter < 21:
        raise ValueError("audit one registered horizon/origin per command")
    for minutes, v in cases(reg["contract"]):
        if minutes != minutes_filter: continue
        bank = bank_for(minutes, reg)
        for number in [origin_filter]:
            prefix = OUT / f"tf_{minutes}m_origin_{number:02d}_{v['id']}"
            meta = read(str(prefix)+".json")
            if digest(str(prefix)+".joblib") != meta["model_sha256"]: raise ValueError("model hash audit failed")
            origin = START+number*3*d.DAY
            train, cal = d.split(bank["times"], origin, minutes*60000, seed=20261013+minutes*1000+number)
            if "train_cap" in v: train = h.larger_train(bank["times"], train, origin, minutes*60000, v["train_cap"], 20261013+minutes*1000+number)
            if identity(bank["times"][train]) != meta["fold"]["train_sha256"]: raise ValueError("training fold changed")
            if meta["fold"]["latest_train_label_ms"] > meta["fold"]["calibration_start_ms"]-7*minutes*60000:
                raise ValueError("train/calibration purge failed")
            if meta["fold"]["latest_calibration_label_ms"] > meta["fold"]["cutoff_ms"]: raise ValueError("calibration/test purge failed")
            payload = joblib.load(str(prefix)+".joblib")
            xc = h.feature_rows(bank, cal, v["features"], minutes).reshape(len(cal)*8, -1)
            _, calibration_score = forecast(payload, xc); calibration_score = calibration_score.reshape(-1, 8)
            with np.load(str(prefix)+".npz", allow_pickle=False) as saved:
                indices = np.searchsorted(bank["times"], saved["times"])
                truth = np.sign(bank["closes"][indices+1]-bank["closes"][indices]).astype(np.int8)
                if not np.array_equal(truth, saved["labels"]): raise ValueError("independent sign audit failed")
                for j, symbol in enumerate(reg["symbols"]):
                    for k, threshold in enumerate(meta["thresholds"][symbol]):
                        fresh = h.score_threshold(calibration_score[:, j], bank["times"][cal], symbol,
                                                  reg["contract"]["coverage_targets"][k], 20261013)
                        if fresh != threshold: raise ValueError("past calibration threshold audit failed")
                        emitted = h.score_mask(saved["confidence_score"][:, j], saved["times"], symbol, threshold, 20261013)
                        if not np.array_equal(emitted, saved["masks"][k, :, j]): raise ValueError("emission audit failed")
                        thresholds += 1
            warnings += bool(meta["info"].get("warnings"))
            checked += 1
    write(OUT / f"audit_tf_{minutes_filter}m_origin_{origin_filter:02d}.json", {"forecast_artifacts": checked, "thresholds": thresholds, "models_with_warnings": warnings,
        "independent_signs": True, "purge": True, "calibration_only_thresholds": True,
        "pure_function_checks": h.direct_checks(), "registration_sha256": reg["sha256"]})
    print(json.dumps({"artifacts": checked, "thresholds": thresholds, "models_with_warnings": warnings}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["init", "fit", "score", "audit", "checks"])
    parser.add_argument("--minutes", type=int); parser.add_argument("--origin", type=int)
    args = parser.parse_args()
    if args.stage == "fit": fit_origin(args.minutes, args.origin)
    elif args.stage == "audit": audit(args.minutes, args.origin)
    elif args.stage == "checks": print(json.dumps(h.direct_checks()))
    else: globals()[args.stage]()
