"""One foreground offline step per call. Exclusive versioned research outputs."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import compact_postclose_research_v1 as r
from exia.genetic_primus import compact_forecast_v1 as m
from exia.genetic_primus import compact_forecast_data_v1 as d

BASE = ROOT / "Reports/Exia/Genetic_Primus"
OUT = BASE / "compact_postclose_research_v1_20261008"
CONTRACT = ROOT / "configs/genetic_primus_compact_postclose_research_v1.json"
OLD_PREP = BASE / "compact_forecast_v1_20261007_r2/preparation.json"
OLD_RATES = BASE / "compact_net_replay_v1_20261007/observed_rate_bank.json"
TERMINAL_RATES = BASE / "execution_input_audit_v1_20261008/terminal_funding_reports.json"
OLD_RESULTS = BASE / "compact_net_replay_v1_20261007/results.json"
SYMBOLS = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
CODE = [Path(__file__), CONTRACT, ROOT / "src/exia/genetic_primus/compact_postclose_research_v1.py",
        ROOT / "src/exia/genetic_primus/compact_forecast_v1.py", ROOT / "src/exia/genetic_primus/compact_forecast_data_v1.py"]


def read(path):
    if not 0 < path.stat().st_size <= 4_000_000:
        raise ValueError("bounded JSON input required")
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path):
    if not 0 < path.stat().st_size <= 16_000_000:
        raise ValueError("bounded input pin required")
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def write(name, value):
    sealed = value | {"sha256": m.identity(value)}
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(sealed, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return sealed


def verify_body(value):
    if value.get("sha256") != m.identity({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("artifact content SHA mismatch")
    return value


def config():
    c = read(CONTRACT)
    if (c["safety"] != r.SAFETY or c["bar_ms"] != r.BAR_MS or c["deployment_horizon_ms"] != r.HORIZON_MS
            or c["retrain_cadence_ms"] != r.HORIZON_MS or c["purge_bars"] != 7
            or c["observed_execution_qualified"] is not False or c["independent_validation_claim_allowed"] is not False
            or c["mlp_seeds"] != list(m.SEEDS) or c["weightings"] != list(m.WEIGHTINGS)):
        raise ValueError("fixed research contract or fail-closed safety changed")
    return c


def prepare():
    c = config()
    if OUT.exists():
        raise ValueError("new exclusive report root required")
    prep = verify_body(read(OLD_PREP))
    if len(prep["sources"]) != 12:
        raise ValueError("exact frozen 12-source bank required")
    import pyarrow.parquet as pq
    import pandas as pd
    frames, source_pins = [], {}
    for name, expected in prep["sources"].items():
        path = (ROOT / name).resolve()
        path.relative_to(ROOT / "Retrodate")
        if pin(path) != expected:
            raise ValueError("frozen source changed: " + name)
        reader = pq.ParquetFile(path, pre_buffer=False, memory_map=False,
                               thrift_string_size_limit=1_000_000, thrift_container_size_limit=100_000)
        if not 0 < reader.metadata.num_rows <= 200_000:
            raise ValueError("bounded OHLCV source required")
        frame = reader.read(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
                            use_threads=False).to_pandas(use_threads=False)
        values = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
        if (frame.timestamp.dtype.kind not in "iu" or not set(frame.symbol) <= set(SYMBOLS)
                or not np.isfinite(values).all() or np.any(values[:, :4] <= 0) or np.any(values[:, 4] < 0)
                or np.any(values[:, 1] < np.maximum(values[:, 0], values[:, 3]))
                or np.any(values[:, 2] > np.minimum(values[:, 0], values[:, 3]))):
            raise ValueError("OHLCV integrity failure")
        frames.append(frame)
        source_pins[name] = expected
    panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])
    if panel.duplicated(["timestamp", "symbol"]).any():
        raise ValueError("unresolved duplicate OHLCV bars")
    closing = panel.pivot(index="timestamp", columns="symbol", values="close").reindex(columns=SYMBOLS)
    times = closing.index.to_numpy(dtype=np.int64)
    closes = closing.to_numpy(float)
    if int(times[0]) != 1640995200000 or int(times[-1]) + r.BAR_MS != c["report_end_ms"]:
        raise ValueError("frozen Jan2022-through-Sep2026 bank required")
    samples, features = r.samples(times, closes, c["report_start_ms"])
    cal_origins = list(range(c["calibration_start_ms"], c["calibration_end_ms"], r.HORIZON_MS))
    origins = list(range(c["report_start_ms"], c["report_end_ms"], r.HORIZON_MS))
    if len(cal_origins) != 12 or len(origins) != 21:
        raise ValueError("fixed twelve calibration and twenty-one report origins required")
    for origin in cal_origins + origins:
        r.split(samples, origin)
    rate_bank = read(OLD_RATES)
    terminal = verify_body(read(TERMINAL_RATES))
    if len(terminal["records"]) != 8 or terminal["target_ms"] != c["report_end_ms"]:
        raise ValueError("eight reported terminal funding rates required as explicit proxies")
    rates = [v for v in rate_bank["events"] if c["calibration_start_ms"] <= int(v["settlement_ms"]) <= c["report_end_ms"]]
    terminal_provenance = []
    for row in terminal["records"]:
        if row["consumable_as_funding_cashflow"] is not False or row["strategy_available_us"] is not None:
            raise ValueError("frozen nonqualified terminal report status changed")
        rates.append({"symbol": row["symbol"], "settlement_ms": row["reported_funding_time_ms"],
                      "rate_fraction": row["rate_fraction"], "rate_is_reported_not_verified_settlement": True})
        terminal_provenance.append(row)
    rates.sort(key=lambda v: (int(v["settlement_ms"]), v["symbol"]))
    keys = [(int(v["settlement_ms"]), v["symbol"]) for v in rates]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate modeled funding reports")
    # For each evaluated interval require the complete explicitly assumed 8-hour schedule.
    for start, end in ((c["calibration_start_ms"], c["calibration_end_ms"]), (c["report_start_ms"], c["report_end_ms"])):
        expected = {(t, s.replace("/", "")) for t in range(start, end + 1, 2 * r.BAR_MS) for s in SYMBOLS}
        actual = {(t, s) for t, s in keys if start <= t <= end}
        if actual != expected:
            raise ValueError("incomplete assumed 8-hour modeled funding schedule")
    OUT.mkdir()
    with (OUT / "data_bank.npz").open("xb") as stream:
        np.savez_compressed(stream, times=times + r.BAR_MS, closes=closes, features=features,
                            **{"sample_" + k: v for k, v in samples.items()})
    banks = {}
    for prefix, start, end in (("cal", c["calibration_start_ms"], c["calibration_end_ms"]),
                               ("report", c["report_start_ms"], c["report_end_ms"])):
        days = (end - start) // r.DAY_MS
        for name, seed in (("selection", 3039), ("audit", 7079)):
            banks[prefix + "_" + name] = r.resample_bank(days, seed)
    with (OUT / "resample_banks.npz").open("xb") as stream:
        np.savez_compressed(stream, **banks)
    write("modeled_rate_bank.json", {"events": rates, "terminal_report_provenance": terminal_provenance,
                                    "settlement_and_marks_are_assumed": True, "observed_execution_qualified": False,
                                    "safety": r.SAFETY})
    input_pins = source_pins | {relative(path): pin(path) for path in (OLD_PREP, OLD_RATES, TERMINAL_RATES, OLD_RESULTS)}
    registration = {"schema": "compact_postclose_research_registration/1", "before_fits": True,
                    "contract": c, "contract_file_pin": pin(CONTRACT), "source_pins": input_pins,
                    "code_pins": {relative(p): pin(p) for p in CODE}, "symbols": list(SYMBOLS),
                    "bank_pins": {n: pin(OUT / n) for n in ("data_bank.npz", "resample_banks.npz", "modeled_rate_bank.json")},
                    "source_market_rows": len(panel), "source_timestamp_count": len(times),
                    "sample_content_sha256": m.array_identity(*(samples[k] for k in sorted(samples))),
                    "feature_definition": d.FEATURE_DEFINITION, "feature_definition_sha256": d.FEATURE_SHA256,
                    "calibration_origins_ms": cal_origins, "report_origins_ms": origins,
                    "report_cutoffs_ms": [v - 8 * r.BAR_MS for v in origins],
                    "fold_rule": "last_60days_inner_validation_with_separate_7bar_purge",
                    "resample_content_sha256": {k: m.array_identity(v) for k, v in banks.items()},
                    "parameters_selected_on_report": False, "data_already_revealed": True,
                    "safety": r.SAFETY, "network_used": False}
    saved = write("registration.json", registration)
    write("objective_registry_entry.json", {"schema": "research_objective_registry_entry/1",
                                            "registration_sha256": saved["sha256"],
                                            "new_forecast_arms": 4, "thresholds_per_arm": 3,
                                            "registered_threshold_candidates": 12,
                                            "global_registry_complete": False, "DSR_PBO_claim_allowed": False,
                                            "opened_test_not_independent": True, "safety": r.SAFETY})
    print(json.dumps({"prepared": True, "rows": len(panel), "calibration_origins": 12, "report_origins": 21,
                      "model_fits": 0, "network_used": False}))


def load():
    registration = verify_body(read(OUT / "registration.json"))
    for name, expected in registration["code_pins"].items():
        if pin(ROOT / name) != expected:
            raise ValueError("registered source code changed: " + name)
    for name, expected in registration["bank_pins"].items():
        if pin(OUT / name) != expected:
            raise ValueError("registered bank changed: " + name)
    with np.load(OUT / "data_bank.npz", allow_pickle=False) as stream:
        bank = {k: stream[k] for k in stream.files}
    with np.load(OUT / "resample_banks.npz", allow_pickle=False) as stream:
        resamples = {k: stream[k] for k in stream.files}
    rates = verify_body(read(OUT / "modeled_rate_bank.json"))["events"]
    return registration, bank, resamples, rates


def fit(stage, index):
    reg, bank, _, _ = load()
    origins = reg["calibration_origins_ms" if stage == "cal" else "report_origins_ms"]
    if not 0 <= index < len(origins):
        raise ValueError("bounded registered origin index required")
    name = stage + "_origin_" + str(index).zfill(2) + ".json"
    if (OUT / name).exists():
        raise ValueError("origin is immutable and already computed")
    origin = int(origins[index])
    samples = {k[7:]: v for k, v in bank.items() if k.startswith("sample_")}
    position = int(np.searchsorted(bank["times"], origin))
    if bank["times"][position] != origin:
        raise ValueError("completed causal feature bar absent")
    features = bank["features"][position]
    prefix_sha = m.array_identity(bank["times"][:position + 1], bank["closes"][:position + 1])
    result = r.fit_origin(samples, origin, features, prefix_sha)
    result.pop("sha256")
    result.update(registration_sha256=reg["sha256"], stage=stage, origin_index=index,
                  source_content_sha256=reg["sample_content_sha256"])
    saved = write(name, result)
    print(json.dumps({"stage": stage, "index": index, "origin_utc": datetime.fromtimestamp(origin / 1000, timezone.utc).date().isoformat(),
                      "train_origins": saved["train_origin_count"], "fits": saved["model_fit_count"],
                      "gradient_steps": saved["gradient_steps"], "saved": name}))


def decisions(reg, bank, stage, policy, threshold):
    origins = reg["calibration_origins_ms" if stage == "cal" else "report_origins_ms"]
    rows = []
    for index, origin in enumerate(origins):
        if policy in r.ARMS:
            record = verify_body(read(OUT / (stage + "_origin_" + str(index).zfill(2) + ".json")))
            if record["origin_ms"] != origin or record["registration_sha256"] != reg["sha256"]:
                raise ValueError("origin registration substitution")
            arm = record["arms"][policy]
            for item in arm["models"]:
                verify_body(item)
            targets = np.where(np.asarray(arm["prediction"]) * 10000 > threshold, 0.1, 0).tolist()
            model_sha = m.identity([v["sha256"] for v in arm["models"]])
            metadata = {"scaler_sha256": record["scaler_sha256"], "feature_prefix_sha256": record["feature_prefix_sha256"],
                        "training_identity_sha256": m.identity(record["training_identity"]),
                        "cutoff_ms": record["training_identity"]["cutoff_ms"], "threshold_bps": threshold}
        else:
            position = int(np.searchsorted(bank["times"], origin))
            if policy == "NoTrade":
                targets = [0.0] * len(SYMBOLS)
            elif policy == "EMA48_long_only":
                targets = np.where(bank["features"][position, :, 0] > 0, 0.1, 0).tolist()
            else:
                raise ValueError("unregistered policy")
            model_sha = m.identity({"policy": policy, "semantic": "fixed_causal_3d_close_decisions"})
            metadata = {"scaler_sha256": None, "feature_prefix_sha256": m.array_identity(bank["times"][:position + 1], bank["closes"][:position + 1]),
                        "training_identity_sha256": None, "cutoff_ms": None, "threshold_bps": None}
        rows.append({"origin_ms": origin, "targets": targets, "model_sha256": model_sha, "policy": policy} | metadata)
    return rows


def run_replay(reg, bank, rates, stage, policy, threshold, scenario):
    c = reg["contract"]
    start, end = (c["calibration_start_ms"], c["calibration_end_ms"]) if stage == "cal" else (c["report_start_ms"], c["report_end_ms"])
    return r.replay(times=bank["times"], prices=bank["closes"], start=start, end=end, symbols=SYMBOLS,
                    decisions=decisions(reg, bank, stage, policy, threshold), rates=rates,
                    fee_bps=c["taker_fee_bps"], drag_bps=c["adverse_price_bps"][scenario])


def calibrate():
    reg, bank, banks, rates = load()
    rows, selected, gates = [], {}, {}
    for policy in r.ARMS:
        candidates = []
        for threshold in reg["contract"]["threshold_candidates_bps"]:
            result = run_replay(reg, bank, rates, "cal", policy, threshold, "stress")
            metrics = result["metrics"]
            a = r.audit(result["daily_log_returns"], banks["cal_selection"], banks["cal_audit"])
            row = {"policy": policy, "threshold_bps": threshold, "stress_metrics": metrics, "stress_audit": a}
            rows.append(row)
            candidates.append(row)
        feasible = [v for v in candidates if v["stress_metrics"]["max_drawdown"] <= 0.1]
        if not feasible:
            raise ValueError("no threshold under registered research drawdown budget")
        chosen = max(feasible, key=lambda v: (v["stress_metrics"]["net_return"],
                                              -v["stress_metrics"]["turnover_initial_capital"], v["threshold_bps"]))
        selected[policy] = chosen["threshold_bps"]
        gates[policy] = (chosen["stress_metrics"]["net_return"] > 0
                         and chosen["stress_audit"]["selection"]["LCB_mean_log_bps_per_day"] > 0
                         and chosen["stress_audit"]["audit"]["LCB_mean_log_bps_per_day"] > 0
                         and chosen["stress_audit"]["positive_outer_slices_fraction"] >= 0.65)
    simple_gate = gates["linear6_uniform"] or gates["linear6_half_life_90_anchor20"]
    value = {"registration_sha256": reg["sha256"], "rows": rows, "selected_thresholds_bps": selected,
             "signal_gates": gates, "simple_model_signal_confirmed": bool(simple_gate),
             "default_deployment": "NoTrade", "genetic_random_search_fits": 0,
             "GA_RS_status": "eligible_but_requires_separate_registered_equal_budget" if simple_gate else "skipped_no_confirmed_simple_signal",
             "threshold_selection_used_report_period": False, "safety": r.SAFETY}
    write("calibration.json", value)
    print(json.dumps({"selected_thresholds_bps": selected, "simple_gate": bool(simple_gate), "GA_RS_fits": 0}))


def replay_one(policy, scenario):
    reg, bank, _, rates = load()
    calibration = verify_body(read(OUT / "calibration.json"))
    if calibration["registration_sha256"] != reg["sha256"]:
        raise ValueError("calibration registration mismatch")
    threshold = calibration["selected_thresholds_bps"].get(policy)
    result = run_replay(reg, bank, rates, "report", policy, threshold, scenario)
    result.update(policy=policy, scenario=scenario, registration_sha256=reg["sha256"],
                  calibration_sha256=calibration["sha256"], threshold_bps=threshold)
    write("replay_" + policy + "_" + scenario + ".json", result)
    print(json.dumps({"policy": policy, "scenario": scenario, "net_pct": round(result["metrics"]["net_return"] * 100, 4),
                      "DD_pct": round(result["metrics"]["max_drawdown"] * 100, 4), "fills": result["metrics"]["fills"]}))


def summarize():
    reg, bank, banks, _ = load()
    calibration = verify_body(read(OUT / "calibration.json"))
    policies = ("NoTrade", "EMA48_long_only") + r.ARMS
    rows = {}
    all_replays = {}
    for scenario in ("normal", "stress"):
        ema = verify_body(read(OUT / ("replay_EMA48_long_only_" + scenario + ".json")))
        for policy in policies:
            result = verify_body(read(OUT / ("replay_" + policy + "_" + scenario + ".json")))
            if result["registration_sha256"] != reg["sha256"] or result["calibration_sha256"] != calibration["sha256"]:
                raise ValueError("replay evidence substitution")
            all_replays[(policy, scenario)] = result
            audit = r.audit(result["daily_log_returns"], banks["report_selection"], banks["report_audit"],
                            paired_log=ema["daily_log_returns"])
            metrics = result["metrics"]
            evidence_gate = (metrics["net_return"] > 0 and audit["selection"]["LCB_mean_log_bps_per_day"] > 0
                             and audit["audit"]["LCB_mean_log_bps_per_day"] > 0
                             and audit["positive_outer_slices_fraction"] >= 0.65
                             and audit["return_excluding_best_three_day_log_window"] >= 0)
            rows.setdefault(policy, {})[scenario] = {"metrics": metrics, "audit": audit,
                                                     "research_robustness_gate": bool(evidence_gate),
                                                     "drawdown_budget_passes": {str(v): metrics["max_drawdown"] <= v for v in (0.1, 0.2, 0.3)}}
    # Audit model forecasts only where a FULL 3-day outcome exists; terminal partial is excluded.
    sample_lookup = {int(t): i for i, t in enumerate(bank["sample_origins"])}
    diagnostics, records = {}, []
    for stage, n in (("cal", 12), ("report", 21)):
        records += [verify_body(read(OUT / (stage + "_origin_" + str(i).zfill(2) + ".json"))) for i in range(n)]
    for policy in r.ARMS:
        errors, actuals, ess, drift, epochs, chosen = [], [], [], [], [], []
        for record in records[12:]:
            arm = record["arms"][policy]
            predicted = np.asarray(arm["prediction"])
            if record["origin_ms"] in sample_lookup:
                actual = bank["sample_y"][sample_lookup[record["origin_ms"]]]
                errors.extend((predicted - actual).tolist())
                actuals.extend(actual.tolist())
            ess.append(arm["origin_ess"])
            epochs.extend([v["selected_epoch"] for v in arm["epochs"]])
            position = int(np.searchsorted(bank["times"], record["origin_ms"]))
            chosen.append((arm, bank["features"][position]))
        # Function-space drift on the SAME current feature probe, not raw parameter distance.
        for i in range(1, len(chosen)):
            previous, _ = chosen[i - 1]
            current, probe = chosen[i]
            old_prediction = np.mean([m.predict(v, probe) for v in previous["models"]], axis=0)
            new_prediction = np.asarray(current["prediction"])
            drift.append(float(np.sqrt(np.mean((new_prediction - old_prediction) ** 2)) * 10000))
        thresholds = calibration["selected_thresholds_bps"][policy]
        targets = np.asarray([v["targets"] for v in all_replays[(policy, "normal")]["decisions"]])
        diagnostics[policy] = {"complete_forecast_origins": len(errors) // 8, "forecast_samples": len(errors),
                               "rmse_bps": float(np.sqrt(np.mean(np.square(errors))) * 10000),
                               "zero_forecast_rmse_bps": float(np.sqrt(np.mean(np.square(actuals))) * 10000),
                               "origin_ess_min": min(ess), "origin_ess_max": max(ess),
                               "functional_drift_rms_bps_mean": float(np.mean(drift)),
                               "functional_drift_rms_bps_max": max(drift),
                               "adjacent_action_change_fraction": float(np.mean(targets[1:] != targets[:-1])),
                               "selected_epochs_min": min(epochs) if epochs else None,
                               "selected_epochs_max": max(epochs) if epochs else None,
                               "threshold_bps": thresholds,
                               "model_sizes": [v["parameter_count"] for v in chosen[-1][0]["models"]]}
    ranking = sorted(policies, key=lambda p: (rows[p]["stress"]["metrics"]["net_return"],
                                             -rows[p]["stress"]["metrics"]["max_drawdown"]), reverse=True)
    result = {"schema": "compact_postclose_comparison/1", "rows": rows, "diagnostics": diagnostics,
              "ranking_by_stress_net_growth": ranking,
              "ranking_at_dd_budget": {str(v): [p for p in ranking if rows[p]["stress"]["metrics"]["max_drawdown"] <= v] for v in (0.1, 0.2, 0.3)},
              "registered_before_fits": True, "fits": sum(v["model_fit_count"] for v in records),
              "gradient_steps": sum(v["gradient_steps"] for v in records),
              "report_origins": 21, "complete_report_labels": 20, "terminal_partial_days": 1,
              "fresh_scalers": 2 * len(records), "network_used": False,
              "research_replay_count": 24 + 12, "genetic_random_search_fits": 0,
              "default_deployment": "NoTrade", "real_money_eligible": False,
              "independent_validation_claim_allowed": False, "observed_execution_qualified": False,
              "registration_sha256": reg["sha256"], "calibration_sha256": calibration["sha256"], "safety": r.SAFETY}
    write("comparison.json", result)
    print(json.dumps({"ranking_stress": ranking, "fits": result["fits"], "all_trading_robustness_gates_pass":
                      all(rows[p]["stress"]["research_robustness_gate"] for p in policies if p != "NoTrade"), "real_money_eligible": False}))


def plot():
    verify_body(read(OUT / "comparison.json"))
    if (OUT / "capital_and_drawdown.png").exists():
        raise ValueError("plot already exists")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    labels = {"NoTrade": "NoTrade", "EMA48_long_only": "EMA48", "linear6_uniform": "Linear uniform",
              "linear6_half_life_90_anchor20": "Linear HL90", "mlp8_uniform": "MLP uniform", "mlp8_half_life_90_anchor20": "MLP HL90"}
    for policy, label in labels.items():
        value = verify_body(read(OUT / ("replay_" + policy + "_normal.json")))
        points = [v for v in value["equity"] if v["phase"] == "calendar_bucket_end"]
        x = [datetime.fromtimestamp(v["timestamp_ms"] / 1000, timezone.utc) for v in points]
        line, = axes[0].plot(x, [v["nav"] for v in points], label=label, lw=1.5)
        axes[1].plot(x, [v["drawdown"] * 100 for v in points], color=line.get_color(), lw=1.2)
        stress = verify_body(read(OUT / ("replay_" + policy + "_stress.json")))
        points_s = [v for v in stress["equity"] if v["phase"] == "calendar_bucket_end"]
        axes[0].plot(x, [v["nav"] for v in points_s], color=line.get_color(), lw=0.8, ls="--", alpha=0.5)
    axes[0].set_title("Aug 1 - Sep 30, 2026 | modeled post-close execution, NOT real fills")
    axes[0].set_ylabel("Net liquidation capital, USDT")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].set_ylabel("Drawdown, % (4h-close marks)")
    axes[1].set_xlabel("UTC | dashed capital curves: stress-costs")
    for ax in axes:
        ax.grid(alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(OUT / "capital_and_drawdown.png", dpi=140)
    plt.close(figure)
    print("capital_and_drawdown.png saved; six policies; normal + stress capital; no UI process")


def verify():
    reg, _, _, _ = load()
    pins = {}
    for name, expected in reg["source_pins"].items():
        if pin(ROOT / name) != expected:
            raise ValueError("source evidence changed: " + name)
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "receipt.json":
            if path.suffix == ".json":
                verify_body(read(path))
            pins[path.name] = pin(path)
    comparison = verify_body(read(OUT / "comparison.json"))
    write("receipt.json", {"verified_sources": len(reg["source_pins"]), "verified_code": len(reg["code_pins"]),
                            "artifact_pins": pins, "comparison_sha256": comparison["sha256"],
                            "all_9_authority_flags_false": True, "network_used": False, "safety": r.SAFETY})
    print(json.dumps({"verified_sources": len(reg["source_pins"]), "verified_code": len(reg["code_pins"]),
                      "verified_artifacts": len(pins), "all_9_authority_flags_false": True}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "checks", "cal-origin", "origin", "calibrate", "replay", "summarize", "plot", "verify"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--policy", choices=("NoTrade", "EMA48_long_only") + r.ARMS)
    parser.add_argument("--scenario", choices=("normal", "stress"))
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "checks":
        result = r.direct_checks()
        if OUT.exists():
            write("direct_checks.json", result)
        print(json.dumps(result))
    elif args.step in ("cal-origin", "origin"):
        fit("cal" if args.step == "cal-origin" else "report", args.index)
    elif args.step == "calibrate":
        calibrate()
    elif args.step == "replay":
        replay_one(args.policy, args.scenario)
    elif args.step == "summarize":
        summarize()
    elif args.step == "plot":
        plot()
    elif args.step == "verify":
        verify()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print((type(error).__name__ + ": " + str(error))[:600], file=sys.stderr)
        sys.exit(1)
