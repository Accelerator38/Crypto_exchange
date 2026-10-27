"""Bounded sequential offline steps; each invocation does one named step."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from exia.genetic_primus import compact_net_replay_v1 as adapter
from exia.genetic_primus import unified_measurement_v1 as m

CONFIG = ROOT / "configs/genetic_primus_compact_net_replay_v1.json"
FORECASTS = ROOT / "Reports/Exia/Genetic_Primus/compact_forecast_v1_20261007_r2"
FUNDING = ROOT / "Retrodate/bitget_full8_funding_20260801_20260930_v1_20261007"
OLD_FUNDING = ROOT / "Retrodate/bitget_funding_history_v1_20260727"
DEFAULT = ROOT / "Reports/Exia/Genetic_Primus/compact_net_replay_v1_20261007"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def pin(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def ms(value):
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def iso(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()


def prepare(folder):
    import pyarrow.parquet as parquet
    config, old = read(CONFIG), read(FORECASTS / "preparation.json")
    if config["safety"] != m.SAFETY or config["parameter_selection_on_benchmark_allowed"] is not False:
        raise ValueError("safety or preregistration mismatch")
    if old["sha256"] != m.identity({k: v for k, v in old.items() if k != "sha256"}):
        raise ValueError("forecast preparation identity mismatch")
    sources = {}
    for path, expected in (old["sources"] | old["code_inputs"]).items():
        if pin(ROOT / path) != expected:
            raise ValueError("frozen source differs: " + path)
        sources[path] = expected
    for path, expected in old["outputs"].items():
        if pin(FORECASTS / path) != expected:
            raise ValueError("forecast prepared artifact changed")
    times, panels = [], []
    symbols = old["symbols"]
    for path, expected in old["sources"].items():
        if not path.endswith(".parquet"):
            continue
        source = parquet.ParquetFile(io.BytesIO((ROOT / path).read_bytes()), pre_buffer=False,
                                    thrift_string_size_limit=1_000_000, thrift_container_size_limit=100_000)
        if source.metadata.num_rows > 200_000:
            raise ValueError("bounded market read exceeded")
        frame = source.read(columns=["timestamp", "symbol", "open", "close"], use_threads=False).to_pandas(use_threads=False)
        if frame.duplicated(["timestamp", "symbol"]).any():
            raise ValueError("duplicate source candle")
        panel = frame.pivot(index="timestamp", columns="symbol", values="open").reindex(columns=symbols)
        if not np.isfinite(panel.to_numpy()).all() or (panel.to_numpy() <= 0).any():
            raise ValueError("missing or invalid reference open")
        times.extend(panel.index.tolist())
        panels.extend(panel.to_numpy().tolist())
    order = np.argsort(times)
    times, prices = np.asarray(times, dtype=np.int64)[order], np.asarray(panels)[order]
    if not np.all(np.diff(times) == m.BAR_MS):
        raise ValueError("market grid has overlap or gap")
    calibration_start, calibration_end = ms(config["calibration_start"]), ms(config["calibration_end_exclusive"])
    origins = [w["origin_start_ms"] for w in m.complete_origins(calibration_start, calibration_end)]
    if len(origins) != 12:
        raise ValueError("exactly twelve registered calibration origins required")
    # Preserve full observed funding rate inputs, including both signs.
    rates = list(read(FUNDING / "funding_events.json")["events"])
    for path in sorted(OLD_FUNDING.glob("funding_*.csv")):
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = pin(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                fraction = Decimal(row["funding_rate"])
                rates.append({"symbol": row["symbol"] + "USDT", "settlement_ms": int(row["funding_time_ms"]),
                              "rate_fraction": str(fraction), "rate_bps": str(fraction * 10000)})
    for path in [FUNDING / "funding_events.json", FUNDING / "completion_manifest.json",
                 FUNDING / "evidence_receipt.json", OLD_FUNDING / "integrity_manifest.json",
                 FORECASTS / "preparation.json", FORECASTS / "sample_bank.npz"]:
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = pin(path)
    for i, origin in enumerate(old["benchmark_origins_ms"]):
        path = FORECASTS / f"origin_{i:02d}.json"
        adapter.validate_origin(read(path), origin)
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = pin(path)
    for path in [CONFIG, Path(__file__), ROOT / "src/exia/genetic_primus/compact_net_replay_v1.py",
                 ROOT / "src/exia/genetic_primus/unified_measurement_v1.py",
                 ROOT / "src/exia/genetic_primus/profit_fitness_v1.py",
                 ROOT / "src/exia/genetic_primus/measurement.py",
                 ROOT / "tests/test_genetic_primus_compact_net_replay_v1.py"]:
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = pin(path)
    folder.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(folder / "reference_open_panel.npz", timestamps=times, prices=prices)
    write(folder / "observed_rate_bank.json", {"events": rates, "mark_prices_present": False})
    registry = {"schema": "compact_net_replay_registration/1", "registered_before_calibration_fits": True,
                "config": config, "calibration_origins_ms": origins, "benchmark_origins_ms": old["benchmark_origins_ms"],
                "symbols": symbols, "forecast_preparation_sha256": old["sha256"], "sources": sources,
                "calibration_search_size": 3, "selection_model": "linear6_uniform",
                "new_model_fit_budget": 12, "benchmark_model_fit_budget": 0,
                "prior_model_fits_including_superseded_attempt": 560,
                "full_global_registry_imported": False, "ga_allowed": False,
                "costs_are_registered_sensitivities": True, "funding_marks_are_proxy": True,
                "new_outputs": {name: pin(folder / name) for name in ("reference_open_panel.npz", "observed_rate_bank.json")},
                "safety": dict(m.SAFETY)}
    registry["sha256"] = m.identity(registry)
    write(folder / "registration.json", registry)
    print("prepared: 12 calibration origins; 20 frozen benchmark origins; 3 threshold candidates; fits=0; network=0")


def verify(folder):
    registration = read(folder / "registration.json")
    if registration["sha256"] != m.identity({k: v for k, v in registration.items() if k != "sha256"}):
        raise ValueError("registration identity changed")
    if registration["safety"] != m.SAFETY:
        raise ValueError("authority flags changed")
    for path, expected in registration["sources"].items():
        if pin(ROOT / path) != expected:
            raise ValueError("registered input changed: " + path)
    for name, expected in registration["new_outputs"].items():
        if pin(folder / name) != expected:
            raise ValueError("prepared bank changed")
    return registration


def fit(folder, index):
    registration = verify(folder)
    if index is None or not 0 <= index < 12:
        raise ValueError("one calibration index 0..11 required")
    output = folder / f"calibration_origin_{index:02d}.json"
    if output.exists():
        raise ValueError("immutable existing calibration fit refused")
    with np.load(FORECASTS / "sample_bank.npz", allow_pickle=False) as bank:
        samples = {k: bank[k] for k in bank.files}
    result = adapter.fit_calibration_linear(samples, registration["calibration_origins_ms"][index])
    result["preparation_sha256"] = registration["sha256"]
    result["symbols"] = registration["symbols"]
    result["sha256"] = m.identity({k: v for k, v in result.items() if k != "sha256"})
    adapter.validate_origin(result, result["origin_ms"], expected_arms=("linear6_uniform",))
    write(output, result)
    print(f"calibration origin={index:02d}; date={iso(result['origin_ms'])}; fits={result['model_fit_count']}; gradient_steps={result['gradient_steps']}; network=0")


def setup_replay(folder, registration, *, calibration):
    config = registration["config"]
    prefix = "calibration" if calibration else "benchmark"
    start, end = ms(config[prefix + "_start"]), ms(config[prefix + "_end_exclusive"])
    windows = m.complete_origins(start, end)
    terminal = windows[-1]["planned_exit_open_ms"]
    with np.load(folder / "reference_open_panel.npz", allow_pickle=False) as bank:
        times, prices = bank["timestamps"], bank["prices"]
    select = (times >= start) & (times <= terminal)
    symbols = registration["symbols"]
    bars = [{"timestamp": int(time), "prices": {s: str(float(p)) for s, p in zip(symbols, row)}}
            for time, row in zip(times[select], prices[select])]
    spec = m.MeasurementSpec(start, end, terminal, config["capital_quote"], tuple(symbols), m.identity(bars))
    rates_path = folder / "observed_rate_bank.json"
    funding = adapter.funding_input(read(rates_path)["events"], bars, start_ms=start, end_ms=end,
                                    symbols=symbols, evidence_sha256=pin(rates_path)["sha256"])
    folder_source = folder if calibration else FORECASTS
    file_prefix = "calibration_origin" if calibration else "origin"
    origins = []
    for i, window in enumerate(windows):
        origin = read(folder_source / f"{file_prefix}_{i:02d}.json")
        adapter.validate_origin(origin, window["origin_start_ms"],
                                expected_arms=("linear6_uniform",) if calibration else adapter.ARMS)
        origins.append(origin)
    with np.load(FORECASTS / "sample_bank.npz", allow_pickle=False) as bank:
        ema = {int(origin): values[:, 0].tolist() for origin, values in zip(bank["origins"], bank["x"])}
    costs = {name: m.CostScenario(name, values["fee_bps_per_fill"], values["fee_bps_per_fill"],
                                 values["adverse_bps_per_fill"], values["adverse_bps_per_fill"],
                                 "registered_assumption", m.identity(config["cost_scenarios"]))
             for name, values in config["cost_scenarios"].items()}
    return spec, bars, funding, origins, ema, costs


def replays(registration, inputs, policy, threshold):
    spec, bars, funding, origins, ema, costs = inputs
    decisions = adapter.decisions_for_policy(origins, policy=policy, symbols=registration["symbols"],
        threshold_bps=threshold, weight=registration["config"]["target_weight_per_asset"],
        ema_features=ema, registry_sha256=registration["sha256"])
    return {name: m.replay(spec, price_bars=bars, decisions=decisions, funding=funding, costs=scenario)
            for name, scenario in costs.items()}


def bank_pair(folder, result, config, prefix):
    import pandas as pd
    from exia.genetic_primus.measurement import build_bootstrap_bank
    start, end = result["spec"]["start_ms"], result["spec"]["end_ms"]
    folds = [{"name": "slice_" + str(i), "start": iso(t), "end": iso(min(t + 3 * adapter.DAY_MS, end))}
             for i, t in enumerate(range(start, end, 3 * adapter.DAY_MS))]
    frame = pd.DataFrame({"timestamp": [r["timestamp"] for r in result["equity"]], "symbol": "portfolio"})
    bootstrap = config["bootstrap"]
    banks = {name: build_bootstrap_bank(frame, start_timestamp=start, cutoff_timestamp=end, folds=folds,
             bank_id=prefix + "_" + name, seed=bootstrap[name + "_seed"],
             samples=bootstrap["samples"], block_days=bootstrap["block_days"])
             for name in ("selection", "audit")}
    np.savez_compressed(folder / (prefix + "_resample_banks.npz"), **{
        name + "_" + kind + "_" + scope: array for name, bank in banks.items()
        for kind, data in (("timestamps", bank.scope_timestamps), ("indices", bank.scope_indices))
        for scope, array in data.items()})
    write(folder / (prefix + "_bank_identities.json"), {"folds": folds, "banks": {k: v.metadata() for k, v in banks.items()}})
    return folds, banks


def calibrate(folder):
    from exia.genetic_primus.profit_fitness_v1 import check_profit_stability
    registration = verify(folder)
    if (folder / "selected_policy.json").exists():
        raise ValueError("immutable parameter selection refused")
    inputs = setup_replay(folder, registration, calibration=True)
    rows, candidate_results = [], {}
    for threshold in registration["config"]["entry_threshold_candidates_bps"]:
        # Only three comparisons affect selection, not twelve model/threshold combinations.
        result = replays(registration, inputs, "linear6_uniform", threshold)
        candidate_results[threshold] = result
        for scenario, replay in result.items():
            write(folder / f"calibration_threshold_{threshold}_{scenario}.json", replay)
        rows.append({"policy": "linear6_uniform", "threshold_bps": threshold,
                     "metrics": {k: v["metrics"] for k, v in result.items()}})
    selected = adapter.rank_calibration(rows)
    result = candidate_results[selected["threshold_bps"]]
    folds, banks = bank_pair(folder, result["normal"], registration["config"], "calibration")
    audits = {str(threshold): check_profit_stability(results["normal"], stress=results["stress"],
                folds=folds, selection_bank=banks["selection"], audit_bank=banks["audit"])
              for threshold, results in candidate_results.items()}
    selected.update({"registration_sha256": registration["sha256"], "stability_audits": audits,
        "ga_executed": False, "random_search_executed": False,
        "ga_gate": "blocked_until_verified_net_signal_and_complete_global_trial_registry",
        "model_fit_count": sum(read(folder / f"calibration_origin_{i:02d}.json")["model_fit_count"] for i in range(12)),
        "new_financial_replays": 6, "parameter_candidates": 3,
        "model_fit_count_including_known_prior_attempts": 572,
        "safety": dict(m.SAFETY), "not_prospective_or_release_evidence": True})
    selected["sha256"] = m.identity(selected)
    write(folder / "selected_policy.json", selected)
    print(f"calibration complete; threshold={selected['threshold_bps']}bps; worst_log_growth={selected['worst_log_growth']:.6f}; audit={audits[str(selected['threshold_bps'])]['status']}; fits=12; replays=6; GA=0")


def benchmark(folder, policy):
    registration = verify(folder)
    if policy not in adapter.ARMS + adapter.BASELINES:
        raise ValueError("one registered policy required")
    selected = read(folder / "selected_policy.json")
    if selected["sha256"] != m.identity({k: v for k, v in selected.items() if k != "sha256"}):
        raise ValueError("selection identity changed")
    outputs = [folder / f"benchmark_{policy}_{name}.json" for name in ("normal", "stress")]
    if any(path.exists() for path in outputs):
        raise ValueError("immutable existing financial replay refused")
    result = replays(registration, setup_replay(folder, registration, calibration=False), policy, selected["threshold_bps"])
    for scenario, replay in result.items():
        write(folder / f"benchmark_{policy}_{scenario}.json", replay)
    print(policy + ": " + "; ".join(name + " net=" + str(round(100 * float(r['metrics']['net_return']), 4)) + "% DD=" + str(round(100 * float(r['metrics']['max_drawdown']), 4)) + "%" for name, r in result.items()) + "; new_fits=0")


def report(folder):
    from exia.genetic_primus.profit_fitness_v1 import ProfitObjective, check_profit_stability, score_net_growth
    registration = verify(folder)
    if (folder / "results.json").exists():
        raise ValueError("immutable report refused")
    config = registration["config"]
    results = {policy: {scenario: read(folder / f"benchmark_{policy}_{scenario}.json")
                        for scenario in ("normal", "stress")}
               for policy in adapter.ARMS + adapter.BASELINES}
    folds, banks = bank_pair(folder, results["NoTrade"]["normal"], config, "benchmark")
    summaries = {}
    for policy, scenarios in results.items():
        normal, stress = scenarios["normal"], scenarios["stress"]
        parameters = 195 if policy.startswith("mlp") else 7 if policy.startswith("linear") else 0
        objectives = {str(budget): score_net_growth(normal, objective=ProfitObjective(budget),
                      stress_result=stress, complexity_metrics={"active_parameter_count": parameters, "active_feature_count": 6})
                      for budget in config["research_drawdown_budgets_percent"]}
        stability = check_profit_stability(normal, stress=stress, folds=folds,
                        selection_bank=banks["selection"], audit_bank=banks["audit"])
        # Descriptive, with the anomaly retained in the primary ledger.
        daily = m.daily_measurements(normal)
        jumps = [row for row in daily if ms("2026-08-19T00:00:00+00:00") <= row["timestamp"] < ms("2026-08-23T00:00:00+00:00")]
        event_log = sum(math.log1p(float(row["net_return"])) for row in jumps)
        total_log = float(normal["metrics"]["net_log_growth"])
        summaries[policy] = {"metrics": {k: v["metrics"] for k, v in scenarios.items()}, "objectives": objectives,
            "stability": stability, "stored_parameters": parameters,
            "August19_22_log_growth": event_log, "other_days_log_growth": total_log - event_log,
            "max_observed_exposure_fraction": max(float(row["exposure_fraction"]) for row in normal["equity"]),
            "outlier_removed": False}
    ranking = sorted(summaries, key=lambda p: (all(float(summaries[p]["metrics"][s]["max_drawdown"]) <= .1 for s in ("normal", "stress")),
                                               min(float(summaries[p]["metrics"][s]["net_log_growth"]) for s in ("normal", "stress"))), reverse=True)
    aggregate = {"registration_sha256": registration["sha256"], "selected_policy_sha256": read(folder / "selected_policy.json")["sha256"],
                 "period": [config["benchmark_start"], config["benchmark_end_exclusive"]],
                 "ranking_research_10pct_worst_scenario": ranking, "policies": summaries,
                 "new_model_fits": 12, "new_financial_replays": 18, "benchmark_model_fits": 0,
                 "known_total_forecast_fits": 572, "GA_executed": False,
                 "trial_registry_complete": False, "multiplicity_adjusted_claim_allowed": False,
                 "price_anomaly_independently_verified": False, "observed_execution_verified": False,
                 "funding_exchange_mark_verified": False, "safety": dict(m.SAFETY)}
    write(folder / "results.json", aggregate)
    os.environ["MPLCONFIGDIR"] = str(folder / "mpl_cache")
    import matplotlib.pyplot as plt
    fig, (ax, drawdown) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for policy in ranking:
        for scenario in ("normal", "stress"):
            rows = results[policy][scenario]["equity"]
            dates = [datetime.fromtimestamp(row["timestamp"] / 1000, timezone.utc) for row in rows]
            values = [float(row["equity_quote"]) for row in rows]
            label = policy + (" stress" if scenario == "stress" else "")
            ax.plot(dates, values, label=label, linestyle="--" if scenario == "stress" else "-", alpha=.8)
            if scenario == "normal":
                drawdown.plot(dates, [100 * float(row["drawdown"]) for row in rows], label=policy)
    ax.set_title("Fixed Aug-Sep 2026 reference replay; funding rates observed, marks proxy; costs assumed")
    ax.set_ylabel("Capital, USDT (initial 10,000)")
    ax.legend(fontsize=7, ncol=2)
    drawdown.set_ylabel("Drawdown, %")
    for axis in (ax, drawdown):
        axis.grid(alpha=.25)
        axis.axvspan(datetime(2026, 8, 19, tzinfo=timezone.utc), datetime(2026, 8, 23, tzinfo=timezone.utc), color="orange", alpha=.12)
    fig.tight_layout()
    fig.savefig(folder / "equity_and_drawdown.png", dpi=150)
    fig.savefig(folder / "equity_and_drawdown.svg")
    plt.close(fig)
    lines = ["# Compact net replay — 2026-10-07", "", "Offline research only. No live/paper/promotion authority.", "",
             "Calibration: June 20–July 26; 12 causal 3-day retrains. Benchmark: August 1–September 30 inclusive; 20 frozen origins.",
             "Parameters chosen BEFORE benchmark replay; frozen forecast fits reused. Costs: fee 6bps/fill; adverse 2bps normal / 8bps stress.",
             "Signed funding rates observed; OHLCV open used as registered mark proxy. Not observed exchange marks or executable fills.",
             "10,000 USDT shared capital; 10% per selected asset, no redistribution, no borrowing; unchanged targets carry units.",
             "Actual exposure may drift above the 80% target budget. DD 10/20/30% are research constraints, not live risk agreements.", "",
             "| Policy | Net normal % | Net stress % | DD normal % | Audit log LCB normal/stress bps/day |", "|---|---:|---:|---:|---:|"]
    for policy in ranking:
        row = summaries[policy]
        audit = row["stability"]["banks"]["audit"]
        lines.append("| " + policy + " | " + " | ".join(f"{100*float(row['metrics'][s]['net_return']):.4f}" for s in ("normal", "stress"))
                     + f" | {100*float(row['metrics']['normal']['max_drawdown']):.4f} | "
                     + "/".join(f"{audit[s]['net_log_lcb_daily_bps']:.3f}" for s in ("normal", "stress")) + " |")
    lines.extend(["", "The Aug19–22 exceptional return window remains included; independent source verification is pending.",
                  "Bootstrap banks: shared across policies, block=3 calendar days, 2048 samples, seeds 7301/9703; daily cash bins included.",
                  "LCBs are robustness diagnostics, not multiplicity-adjusted confirmation; global immutable objective registry is incomplete.",
                  "12 additional linear calibration fits; 0 new benchmark fits; 560 known previous fits including superseded attempt; 572 total.",
                  "GA/random search skipped: no complete global registry or authorized production evidence. No new prospective interval sealed.",
                  "Next: independently validate the common price jump and actual execution/mark assumptions; do not tune Aug-Sep thresholds."])
    with (folder / "REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    receipts = {str(path.relative_to(folder)).replace("\\", "/"): pin(path) for path in sorted(folder.iterdir()) if path.is_file()}
    write(folder / "evidence_receipt.json", {"schema": "compact_net_replay_receipt/1", "files": receipts, "registration_sha256": registration["sha256"], "safety": m.SAFETY})
    print("report complete; policies=6; cost_scenarios=2; new_fits=12; benchmark_fits=0; network=0; release_authority=0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("prepare", "fit", "calibrate", "benchmark", "report"))
    parser.add_argument("--folder", type=Path, default=DEFAULT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--policy", choices=adapter.ARMS + adapter.BASELINES)
    args = parser.parse_args()
    if args.step == "fit":
        fit(args.folder, args.index)
    elif args.step == "benchmark":
        benchmark(args.folder, args.policy)
    else:
        {"prepare": prepare, "calibrate": calibrate, "report": report}[args.step](args.folder)
