"""One bounded foreground offline step. No collector/network/runtime imports."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import cadence_matrix_v1 as c

BASE = ROOT / "Reports/Exia/Genetic_Primus"
OLD = BASE / "compact_postclose_research_v1_20261008"
OUT = BASE / "cadence_matrix_v1_20261009"
CONTRACT = ROOT / "configs/genetic_primus_cadence_matrix_v1.json"
OLD_RATES = BASE / "compact_net_replay_v1_20261007/observed_rate_bank.json"
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
CODE = [Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/cadence_matrix_v1.py",
        ROOT / "src/exia/genetic_primus/compact_forecast_v1.py",
        ROOT / "src/exia/genetic_primus/compact_forecast_data_v1.py",
        ROOT / "src/exia/genetic_primus/compact_postclose_research_v1.py"]


def read(path):
    if not 0 < path.stat().st_size <= 4_000_000:
        raise ValueError("bounded JSON required")
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path):
    if not 0 < path.stat().st_size <= 16_000_000:
        raise ValueError("bounded file pin required")
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def verify_body(value):
    if value.get("sha256") != c.m.identity({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("sealed body SHA mismatch")
    return value


def write(name, value):
    value = c.plain(value)
    result = value | {"sha256": c.m.identity(value)}
    # Serialize completely before creating an output; no partial JSON on encoding failures.
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
    return result


def ms(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


def prefix(bank, time):
    p = int(np.searchsorted(bank["times"], time))
    if p >= len(bank["times"]) or bank["times"][p] != time:
        raise ValueError("causal feature boundary absent")
    return p, c.m.array_identity(bank["times"][:p + 1], bank["closes"][:p + 1])


def fit_name(period, stride, day):
    return f"fit_{period}_s{stride}_day_{day:03d}.json"


def prepare():
    contract = read(CONTRACT)
    if OUT.exists():
        raise ValueError("fresh exclusive output root required")
    expected = {"A": (3, 3, 3, 3), "B": (3, 1, 3, 3), "C": (3, 1, 1, 3),
                "D": (3, 1, 1, 1), "E": (3, 1, 1, 2), "F": (3, 1, 1, 4), "G": (3, 1, 1, 5)}
    if (contract["safety"] != c.SAFETY or contract["parameter_count"] != 7
            or contract["architecture"] != "linear6" or contract["weighting"] != "uniform"
            or contract["threshold_bps"] != 20 or contract["l2"] != .01 or contract["purge_bars"] != 7
            or contract["observed_execution_qualified"] is not False or set(contract["configurations"]) != set(expected)):
        raise ValueError("fixed seven-way research contract required")
    for name, values in expected.items():
        clocks = c.Clocks(**contract["configurations"][name]).validate()
        if tuple(getattr(clocks, field) for field in clocks.__dataclass_fields__) != values:
            raise ValueError("registered clock configurations changed")
    old_reg = verify_body(read(OLD / "registration.json"))
    old_receipt = verify_body(read(OLD / "receipt.json"))
    for name in ("registration.json", "data_bank.npz", "modeled_rate_bank.json"):
        if pin(OLD / name) != old_receipt["artifact_pins"][name]:
            raise ValueError("frozen parent artifact changed")
    for name, expected_pin in old_reg["source_pins"].items():
        if pin(ROOT / name) != expected_pin:
            raise ValueError("parent market provenance changed")
    for path in CODE[3:]:
        if pin(path) != old_reg["code_pins"][relative(path)]:
            raise ValueError("frozen numerical/accounting code changed")
    inputs = dict(old_reg["source_pins"])
    for path in (OLD / "registration.json", OLD / "receipt.json", OLD / "data_bank.npz", OLD / "modeled_rate_bank.json"):
        inputs[relative(path)] = pin(path)
    for index in range(21):
        path = OLD / f"report_origin_{index:02d}.json"
        if pin(path) != old_receipt["artifact_pins"][path.name]:
            raise ValueError("frozen control model changed")
        inputs[relative(path)] = pin(path)
    for scenario in ("normal", "stress"):
        path = OLD / f"replay_linear6_uniform_{scenario}.json"
        if pin(path) != old_receipt["artifact_pins"][path.name]:
            raise ValueError("frozen control replay changed")
        inputs[relative(path)] = pin(path)
    with np.load(OLD / "data_bank.npz", allow_pickle=False) as stream:
        parent = {k: stream[k] for k in stream.files}
    sample_banks = {str(stride): c.make_samples(parent["times"], parent["closes"], parent["features"],
                    anchor=ms(contract["label_anchor_utc"]), horizon_days=3, stride_days=stride) for stride in (1, 3)}
    for key, value in sample_banks["3"].items():
        if not np.array_equal(value, parent["sample_" + key]):
            raise ValueError("A sample bank does not exactly reproduce original")
    rows = read(OLD_RATES)["events"]
    # Extend only the modeled rate bank; all original qualification flags remain untouched.
    rates_by_key = {(int(v["settlement_ms"]), v["symbol"]): v for v in rows}
    if len(rates_by_key) != len(rows):
        raise ValueError("duplicate source rate rows")
    for row in verify_body(read(OLD / "modeled_rate_bank.json"))["events"]:
        key = int(row["settlement_ms"]), row["symbol"]
        if key in rates_by_key:
            if Decimal(rates_by_key[key]["rate_fraction"]) != Decimal(row["rate_fraction"]):
                raise ValueError("conflicting reported modeled rates")
        else:
            rates_by_key[key] = row
    rates = [rates_by_key[k] for k in sorted(rates_by_key)]
    periods, folds, banks = {}, {}, {}
    sample_shas = {key: c.m.array_identity(*(v[k] for k in sorted(v))) for key, v in sample_banks.items()}
    for name, interval in contract["periods"].items():
        start, end = ms(interval["start_utc"]), ms(interval["end_exclusive_utc"])
        days = (end - start) // c.DAY_MS
        if days != {"development": 86, "report": 61}[name]:
            raise ValueError("fixed period lengths required")
        periods[name] = {"start_ms": start, "end_ms": end, "days": days}
        expected_rates = {(t, s.replace("/", "")) for t in range(start, end + 1, 2 * c.BAR_MS) for s in SYMBOLS}
        if {k for k in rates_by_key if start <= k[0] <= end} != expected_rates:
            raise ValueError("complete assumed 8-hour funding schedule required")
        for day in range(days):
            for stride in (1, 3):
                if stride == 3 and day % 3:
                    continue
                bank = sample_banks[str(stride)]
                temporal = c.old.split(bank, start + day * c.DAY_MS)
                train, inner, val = (temporal[k] for k in ("train", "inner", "validation"))
                folds[fit_name(name, stride, day)] = {"fit_ms": start + day * c.DAY_MS,
                    "cutoff_ms": temporal["cutoff_ms"], "label_stride_days": stride,
                    "training_origins": len(train), "sample_bank_sha256": sample_shas[str(stride)],
                    "folds_sha256": c.m.array_identity(bank["origins"][inner], bank["origins"][val]),
                    "eligible_origins_sha256": c.m.array_identity(bank["origins"][train], bank["available"][train])}
        for block in (3, 7, 14):
            for role, seed in (("selection", 3039), ("audit", 7079)):
                banks[f"{name}_b{block}_{role}"] = c.resample_bank(days, block, seed)
    if len(folds) != 197:
        raise ValueError("exact 197 unique linear fits expected")
    OUT.mkdir()
    with (OUT / "sample_banks.npz").open("xb") as stream:
        np.savez_compressed(stream, **{f"s{stride}_{k}": v for stride, bank in sample_banks.items() for k, v in bank.items()})
    with (OUT / "resample_banks.npz").open("xb") as stream:
        np.savez_compressed(stream, **banks)
    write("folds.json", {"fits": folds, "safety": c.SAFETY})
    write("modeled_rates.json", {"events": rates, "reported_rates_proxy_valuation": True,
                                 "observed_settlement_cashflows": False, "safety": c.SAFETY})
    saved = write("registration.json", {"schema": "cadence_matrix_registration/1", "before_model_fits": True,
        "contract": contract, "periods": periods, "symbols": SYMBOLS, "source_pins": inputs,
        "code_pins": {relative(p): pin(p) for p in CODE}, "sample_content_sha256": sample_shas,
        "resample_content_sha256": {k: c.m.array_identity(v) for k, v in banks.items()},
        "bank_pins": {n: pin(OUT / n) for n in ("sample_banks.npz", "resample_banks.npz", "folds.json", "modeled_rates.json")},
        "frozen_parent_registration_sha256": old_reg["sha256"], "unique_planned_fits": 197,
        "labels_overlap_days": {"1": 2, "3": 0}, "ESS_adjusted_for_overlap_claimed": False,
        "primary_bootstrap_block_days": 7, "independent_validation_claim_allowed": False,
        "network_used": False, "safety": c.SAFETY})
    write("objective_registry_entry.json", {"registration_sha256": saved["sha256"], "candidate_ids": list(expected),
        "candidate_count": 7, "periods_are_not_separate_hypotheses": True,
        "all_candidates_reported": True, "parameters_selected_on_report": False,
        "global_registry_complete": False, "DSR_PBO_claim_allowed": False, "safety": c.SAFETY})
    print(json.dumps({"prepared": True, "period_days": {k: v["days"] for k, v in periods.items()},
                      "configurations": 7, "unique_planned_fits": 197, "new_training_fits": 0}))


def load():
    reg = verify_body(read(OUT / "registration.json"))
    for name, expected in reg["code_pins"].items():
        if pin(ROOT / name) != expected:
            raise ValueError("registered code changed: " + name)
    for name, expected in reg["bank_pins"].items():
        if pin(OUT / name) != expected:
            raise ValueError("registered bank changed: " + name)
    if pin(OLD / "data_bank.npz") != reg["source_pins"][relative(OLD / "data_bank.npz")]:
        raise ValueError("frozen parent price/features bank changed")
    with np.load(OLD / "data_bank.npz", allow_pickle=False) as stream:
        parent = {k: stream[k] for k in ("times", "closes", "features")}
    with np.load(OUT / "sample_banks.npz", allow_pickle=False) as stream:
        samples = {str(s): {k: stream[f"s{s}_{k}"] for k in ("origins", "available", "x", "y")} for s in (1, 3)}
    with np.load(OUT / "resample_banks.npz", allow_pickle=False) as stream:
        banks = {k: stream[k] for k in stream.files}
    return reg, parent, samples, banks, verify_body(read(OUT / "modeled_rates.json"))["events"]


def train(period, start_day, days):
    reg, parent, samples, _, _ = load()
    p = reg["periods"][period]
    if (type(start_day) is not int or type(days) is not int or not 1 <= days <= 7
            or not 0 <= start_day < p["days"] or start_day + days > p["days"]):
        raise ValueError("one bounded nonoverrunning 1-to-7-day chunk required")
    folds = verify_body(read(OUT / "folds.json"))["fits"]
    written = cached = 0
    for day in range(start_day, start_day + days):
        fit_ms = p["start_ms"] + day * c.DAY_MS
        for stride in (1, 3):
            if stride == 3 and day % 3:
                continue
            name = fit_name(period, stride, day)
            if (OUT / name).exists():
                item = verify_body(read(OUT / name))
                if item["registration_sha256"] != reg["sha256"] or item["fit_ms"] != fit_ms:
                    raise ValueError("cached fit registration mismatch")
                cached += 1
                continue
            _, feature_sha = prefix(parent, fit_ms)
            item = c.fit_linear(samples[str(stride)], fit_ms, horizon_days=3, stride_days=stride,
                                 feature_prefix_sha=feature_sha, sample_sha=reg["sample_content_sha256"][str(stride)])
            planned = folds[name]
            if (item["training_identity"]["folds_sha256"] != planned["folds_sha256"]
                    or item["training_identity"]["cutoff_ms"] != planned["cutoff_ms"]
                    or item["train_origins"] != planned["training_origins"]):
                raise ValueError("pre-registered fold/cutoff mismatch")
            write(name, item | {"registration_sha256": reg["sha256"], "period": period, "day": day,
                                "label_stride_days": stride})
            written += 1
    print(json.dumps({"period": period, "first_day": start_day, "days": days,
                      "new_linear_fits": written, "cached": cached, "next_day": start_day + days}))


def smoke():
    reg, parent, _, _, _ = load()
    new = verify_body(read(OUT / fit_name("report", 3, 0)))
    old = verify_body(read(OLD / "report_origin_00.json"))
    model = old["arms"]["linear6_uniform"]["models"][0]
    if new["model"]["parameters"] != model["parameters"] or new["scaler_sha256"] != old["scaler_sha256"]:
        raise ValueError("A first-origin parameters/scaler differ from frozen control")
    pos, _ = prefix(parent, reg["periods"]["report"]["start_ms"])
    prediction = c.m.predict(new["model"], parent["features"][pos])
    if not np.array_equal(prediction, np.asarray(old["arms"]["linear6_uniform"]["prediction"])):
        raise ValueError("A first-origin prediction differs")
    write("first_origin_smoke.json", {"parameter_exact_match": True, "scaler_exact_match": True,
                                      "prediction_exact_match": True, "additional_fits": 0, "safety": c.SAFETY})
    print("A first-origin parameters/scaler/prediction exact match; no additional fit")


def decisions(reg, parent, period, policy):
    p = reg["periods"][period]
    start, end = p["start_ms"], p["end_ms"]
    configuration = reg["contract"]["configurations"].get(policy)
    clocks = c.Clocks(**configuration) if configuration else c.Clocks(3, 1, 3 if policy == "EMA3" else 1, 1)
    rows = []
    for t in clocks.decision_times(start, end):
        position, feature_sha = prefix(parent, t)
        if configuration:
            fit_ms = clocks.latest_fit(start, t)
            day = (fit_ms - start) // c.DAY_MS
            name = fit_name(period, clocks.label_stride_days, day)
            item = verify_body(read(OUT / name))
            model = verify_body(item["model"])
            if (item["registration_sha256"] != reg["sha256"] or item["fit_ms"] != fit_ms
                    or item["assumed_model_ready_us"] > t * 1000 + 100000):
                raise ValueError("causal model availability mismatch")
            prediction = c.m.predict(model, parent["features"][position])
            target = np.where(prediction * 10000 > reg["contract"]["threshold_bps"], .1, 0)
            metadata = {"fit_ms": fit_ms, "model_age_days": (t - fit_ms) // c.DAY_MS,
                        "fit_artifact": name, "model_sha256": model["sha256"],
                        "scaler_sha256": item["scaler_sha256"], "training_identity_sha256": c.m.identity(item["training_identity"]),
                        "cutoff_ms": item["training_identity"]["cutoff_ms"], "prediction": prediction.tolist()}
        else:
            if policy == "NoTrade":
                target = np.zeros(8)
            elif policy in ("EMA1", "EMA3"):
                target = np.where(parent["features"][position, :, 0] > 0, .1, 0)
            elif policy == "PassiveEW8":
                target = np.full(8, .1)
            else:
                raise ValueError("unregistered policy")
            metadata = {"fit_ms": start, "model_age_days": None, "fit_artifact": None,
                        "model_sha256": c.m.identity({"policy": policy, "clocks": c.plain(clocks.__dict__)}),
                        "scaler_sha256": None, "training_identity_sha256": None, "cutoff_ms": None}
        rows.append({"origin_ms": t, "policy": policy, "targets": target.tolist(),
                     "feature_prefix_sha256": feature_sha, "decision_feature_sha256": c.m.array_identity(parent["features"][position])} | metadata)
    return rows, clocks.decision_cadence_days


def replay_one(period, policy, scenario):
    reg, parent, _, _, rates = load()
    decision_rows, cadence = decisions(reg, parent, period, policy)
    p, contract = reg["periods"][period], reg["contract"]
    result = c.replay(times=parent["times"], prices=parent["closes"], start=p["start_ms"], end=p["end_ms"],
                      symbols=SYMBOLS, decisions=decision_rows, rates=rates, fee_bps=contract["fee_bps"],
                      drag_bps=contract["drag_bps"][scenario], decision_cadence_days=cadence)
    result.update(registration_sha256=reg["sha256"], period=period, policy=policy, scenario=scenario,
                  clocks=contract["configurations"].get(policy), threshold_bps=20 if policy in contract["configurations"] else None)
    write(f"replay_{period}_{policy}_{scenario}.json", result)
    metrics = result["metrics"]
    print(json.dumps({"period": period, "policy": policy, "scenario": scenario,
                      "net_pct": round(metrics["net_return"] * 100, 4),
                      "DD_pct": round(metrics["max_drawdown"] * 100, 4), "fills": metrics["fills"]}))


def audit():
    reg, parent, samples, _, _ = load()
    folds = verify_body(read(OUT / "folds.json"))["fits"]
    passed = 0
    def check(value):
        nonlocal passed
        if not value:
            raise AssertionError("full audit check " + str(passed + 1))
        passed += 1
    for name, planned in folds.items():
        item = verify_body(read(OUT / name))
        model = verify_body(item["model"])
        bank = samples[str(planned["label_stride_days"])]
        temporal = c.old.split(bank, item["fit_ms"])
        train, inner, val = (temporal[k] for k in ("train", "inner", "validation"))
        tx, ty = bank["x"][train].reshape(-1, 6), bank["y"][train].reshape(-1)
        check(item["registration_sha256"] == reg["sha256"])
        check(np.all(bank["available"][train] <= item["fit_ms"] - 8 * c.BAR_MS))
        check(np.all(bank["available"][inner] <= temporal["inner_cutoff_ms"]))
        check(c.m.array_identity(bank["origins"][inner], bank["origins"][val]) == planned["folds_sha256"])
        check(c.m.array_identity(bank["origins"][train], bank["available"][train], tx, ty) == item["training_identity"]["train_samples_sha256"])
        check(np.allclose(model["scaler"]["mean"], tx.mean(axis=0), atol=1e-12, rtol=0))
        check(abs(model["scaler"]["target_mean"] - ty.mean()) < 1e-12)
        check(item["positive_weight_mass"] > 0 and item["negative_weight_mass"] > 0)
        _, feature_sha = prefix(parent, item["fit_ms"])
        check(item["feature_prefix_sha256"] == feature_sha)
    for index in range(21):
        item = verify_body(read(OUT / fit_name("report", 3, index * 3)))
        old = verify_body(read(OLD / f"report_origin_{index:02d}.json"))
        check(item["model"]["parameters"] == old["arms"]["linear6_uniform"]["models"][0]["parameters"])
        check(item["scaler_sha256"] == old["scaler_sha256"])
    for scenario in ("normal", "stress"):
        new = verify_body(read(OUT / f"replay_report_A_{scenario}.json"))
        old = verify_body(read(OLD / f"replay_linear6_uniform_{scenario}.json"))
        check(new["metrics"] == old["metrics"])
        check(new["daily_log_returns"] == old["daily_log_returns"])
        check([v["targets"] for v in new["decisions"]] == [v["targets"] for v in old["decisions"]])
    # Daily examples overlap, but labels retain the exact full 3-day clock.
    bank = samples["1"]
    entries, exits = np.searchsorted(parent["times"], bank["origins"]), np.searchsorted(parent["times"], bank["available"])
    check(np.all(np.diff(bank["origins"]) == c.DAY_MS))
    check(np.all(bank["available"] - bank["origins"] == 3 * c.DAY_MS))
    check(np.array_equal(bank["y"], parent["closes"][exits] / parent["closes"][entries] - 1))
    p, _ = prefix(parent, reg["periods"]["report"]["start_ms"])
    causal = c.d.feature_panel(parent["times"][:p + 1] - c.BAR_MS, parent["closes"][:p + 1])
    check(np.array_equal(causal[-1], parent["features"][p]))
    changed = parent["closes"].copy()
    changed[p + 1:] *= 2
    future_changed = c.d.feature_panel(parent["times"] - c.BAR_MS, changed)
    check(np.array_equal(causal[-1], future_changed[p]))
    write("audit_checks.json", {"checks_passed": passed, "verified_fits": len(folds),
                                "all_A_models_and_money_results_exact_match": True, "additional_training_fits": 0,
                                "daily_overlapping_labels_checked": True, "safety": c.SAFETY})
    print(json.dumps({"audit_checks_passed": passed, "fits": len(folds), "A_exact_match": True}))


def summarize():
    reg, parent, _, banks, _ = load()
    verify_body(read(OUT / "audit_checks.json"))
    policies = list(reg["contract"]["configurations"]) + reg["contract"]["baselines"]
    results = {}
    for period, interval in reg["periods"].items():
        shared = {k[len(period) + 1:]: v for k, v in banks.items() if k.startswith(period + "_")}
        rows = {}
        for scenario in ("normal", "stress"):
            baseline = verify_body(read(OUT / f"replay_{period}_A_{scenario}.json"))
            ema = verify_body(read(OUT / f"replay_{period}_EMA1_{scenario}.json"))
            for policy in policies:
                item = verify_body(read(OUT / f"replay_{period}_{policy}_{scenario}.json"))
                if item["registration_sha256"] != reg["sha256"]:
                    raise ValueError("replay registration mismatch")
                daily = np.asarray(item["daily_log_returns"])
                windows = np.convolve(daily, np.ones(3), mode="valid")
                full = daily[:len(daily) // 3 * 3].reshape(-1, 3).sum(axis=1)
                model_ages = [v["model_age_days"] for v in item["decisions"] if v["model_age_days"] is not None]
                target = np.asarray([v["targets"] for v in item["decisions"]])
                start_positions, holding = {}, []
                for event in item["ledger"]:
                    if event["kind"] != "modeled_fill":
                        continue
                    if event["side"] == "BUY":
                        start_positions[event["symbol"]] = event["timestamp_us"]
                    else:
                        holding.append((event["timestamp_us"] - start_positions.pop(event["symbol"])) / (1000 * c.DAY_MS))
                row = {"metrics": item["metrics"],
                       "bootstrap_vs_A": c.bootstrap(daily, shared, paired=baseline["daily_log_returns"]),
                       "bootstrap_vs_EMA1": c.bootstrap(daily, shared, paired=ema["daily_log_returns"]),
                       "positive_full_3d_slices": float(np.mean(full > 0)), "full_3d_slices": len(full),
                       "terminal_partial_slice_days": len(daily) % 3,
                       "without_best_3d_log_window_return": float(np.expm1(daily.sum() - windows.max())),
                       "best_3d_log_share": float(windows.max() / daily.sum()) if daily.sum() > 0 else None,
                       "action_changes_per_calendar_day": int(np.count_nonzero(target[1:] != target[:-1])) / interval["days"],
                       "model_age_days_max": max(model_ages) if model_ages else None,
                       "mean_holding_days": float(np.mean(holding)) if holding else None,
                       "net_return_difference_vs_A_pp": (item["metrics"]["net_return"] - baseline["metrics"]["net_return"]) * 100,
                       "risk_budget_passes": {str(b): item["metrics"]["max_drawdown"] <= b for b in (.1, .2, .3)}}
                if period == "report":
                    row["month_returns"] = {"August": float(np.expm1(daily[:31].sum())), "September": float(np.expm1(daily[31:].sum()))}
                else:
                    row["development_slice_returns"] = [float(np.expm1(daily[a:b].sum())) for a, b in ((0, 28), (28, 56), (56, 86))]
                rows.setdefault(policy, {})[scenario] = row
        rank = sorted(policies, key=lambda p: rows[p]["stress"]["metrics"]["net_return"], reverse=True)
        rankings = {str(b): [p for p in rank if rows[p]["stress"]["metrics"]["max_drawdown"] <= b] for b in (.1, .2, .3)}
        results[period] = {"policies": rows, "stress_net_growth_ranking": rank, "ranking_by_DD_budget": rankings}
    effects = {}
    for period, value in results.items():
        effects[period] = {}
        for left, right in (("B", "A"), ("C", "B"), ("D", "C"), ("E", "C"), ("F", "C"), ("G", "C")):
            a = verify_body(read(OUT / f"replay_{period}_{left}_stress.json"))
            b = verify_body(read(OUT / f"replay_{period}_{right}_stress.json"))
            shared = {k[len(period) + 1:]: v for k, v in banks.items() if k.startswith(period + "_")}
            effects[period][f"{left}_minus_{right}"] = {
                "net_return_difference_pp": (a["metrics"]["net_return"] - b["metrics"]["net_return"]) * 100,
                "paired_banks": c.bootstrap(a["daily_log_returns"], shared, paired=b["daily_log_returns"])}
    fold_count = len(verify_body(read(OUT / "folds.json"))["fits"])
    write("comparison.json", {"schema": "cadence_matrix_comparison/1", "periods": results, "controlled_effects": effects,
        "configurations": reg["contract"]["configurations"], "unique_training_fits": fold_count, "scaler_fits": fold_count,
        "financial_replays": 44, "primary_bootstrap_block_days": 7,
        "all_candidate_results_are_research": True, "default_deployment": "NoTrade",
        "observed_execution_qualified": False, "promotion_authority": False,
        "registration_sha256": reg["sha256"], "network_used": False, "safety": c.SAFETY})
    print(json.dumps({"unique_fits": fold_count, "replays": 44,
        "development_stress_ranking": results["development"]["stress_net_growth_ranking"],
        "report_stress_ranking": results["report"]["stress_net_growth_ranking"], "default": "NoTrade"}))


def plot(period):
    reg, _, _, _, _ = load()
    verify_body(read(OUT / "comparison.json"))
    path = OUT / f"capital_drawdown_{period}.png"
    if path.exists():
        raise ValueError("immutable plot already exists")
    import matplotlib.pyplot as plt
    labels = {"A": "A: original 3/3/3", "B": "B: daily labels", "C": "C: decisions daily, fit3d",
              "D": "D: fit1d", "E": "E: fit2d", "F": "F: fit4d", "G": "G: fit5d",
              "NoTrade": "NoTrade", "EMA1": "EMA daily", "EMA3": "EMA 3d", "PassiveEW8": "Passive full8"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for column, scenario in enumerate(("normal", "stress")):
        for policy, label in labels.items():
            value = verify_body(read(OUT / f"replay_{period}_{policy}_{scenario}.json"))
            points = [v for v in value["equity"] if v["phase"] == "calendar_bucket_end"]
            x = [datetime.fromtimestamp(v["timestamp_ms"] / 1000, timezone.utc) for v in points]
            line, = axes[0, column].plot(x, [v["nav"] for v in points], label=label, lw=1.2,
                                       ls="--" if policy in reg["contract"]["baselines"] else "-")
            axes[1, column].plot(x, [v["drawdown"] * 100 for v in points], color=line.get_color(), lw=1.1, ls=line.get_linestyle())
        axes[0, column].set_title(f"{period}: {scenario} costs | assumed OHLCV execution")
        axes[0, column].set_ylabel("Net liquidation capital, USDT")
        axes[1, column].set_ylabel("Drawdown, % (4h close marks)")
        axes[1, column].set_xlabel("UTC")
        for row in (0, 1):
            axes[row, column].grid(alpha=.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=4, loc="upper center", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path.name + " saved; normal/stress capital and drawdown")


def verify():
    reg, _, _, _, _ = load()
    verify_body(read(OUT / "comparison.json"))
    for name, expected in reg["source_pins"].items():
        if pin(ROOT / name) != expected:
            raise ValueError("source input changed: " + name)
    pins = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "receipt.json":
            if path.suffix == ".json":
                value = verify_body(read(path))
                if "safety" in value and value["safety"] != c.SAFETY:
                    raise ValueError("authority flags changed")
            pins[path.name] = pin(path)
    write("receipt.json", {"registration_sha256": reg["sha256"], "source_pins_verified": len(reg["source_pins"]),
                            "code_pins_verified": len(reg["code_pins"]), "artifact_pins": pins,
                            "all_9_authority_flags_false": True, "network_used": False, "safety": c.SAFETY})
    print(json.dumps({"verified_sources": len(reg["source_pins"]), "verified_code": len(reg["code_pins"]),
                      "verified_artifacts": len(pins), "all_9_authority_flags_false": True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "checks", "train", "smoke", "replay", "audit", "summarize", "plot", "verify"))
    parser.add_argument("--period", choices=("development", "report"))
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--policy", choices=tuple("ABCDEFG") + ("NoTrade", "EMA1", "EMA3", "PassiveEW8"))
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "checks":
        result = c.direct_checks()
        if OUT.exists():
            write("direct_checks.json", result)
        print(json.dumps(result))
    elif args.step == "train":
        train(args.period, args.start_day, args.days)
    elif args.step == "smoke":
        smoke()
    elif args.step == "replay":
        replay_one(args.period, args.policy, args.scenario)
    elif args.step == "audit":
        audit()
    elif args.step == "summarize":
        summarize()
    elif args.step == "plot":
        plot(args.period)
    else:
        verify()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print((type(error).__name__ + ": " + str(error))[:650], file=sys.stderr)
        sys.exit(1)
