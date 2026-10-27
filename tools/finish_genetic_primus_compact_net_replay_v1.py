"""Immutable continuation after a trailing bootstrap fold was too short.

No retraining/replay retries; only merge the last short cash slice into the
preceding descriptive slice. Block length, seeds and economic objective stay fixed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import run_genetic_primus_compact_net_replay_v1 as runner

m, adapter = runner.m, runner.adapter


def folds_for_interval(start, end, block_days=3):
    block = block_days * adapter.DAY_MS
    if end - start < block:
        raise ValueError("report interval shorter than one bootstrap block")
    ranges = [(t, min(t + block, end)) for t in range(start, end, block)]
    if ranges[-1][1] - ranges[-1][0] < block:
        last = ranges.pop()
        ranges[-1] = (ranges[-1][0], last[1])
    return [{"name": "slice_" + str(i), "start": runner.iso(a), "end": runner.iso(b)}
            for i, (a, b) in enumerate(ranges)]


def register(folder):
    registration = runner.verify(folder)
    paths = [folder / f"calibration_threshold_{t}_{s}.json" for t in (20, 50, 100) for s in ("normal", "stress")]
    paths += [folder / f"calibration_origin_{i:02d}.json" for i in range(12)]
    continuation = {"schema": "compact_net_replay_engineering_continuation/1",
        "original_registration_sha256": registration["sha256"],
        "reason": "last 1-day cash fold shorter than 3-day bootstrap block; bank construction failed",
        "fix": "merge final short remainder into previous descriptive fold",
        "economic_parameters_changed": False, "block_length_or_seeds_changed": False,
        "training_or_replay_repeated": False, "parameter_selection_or_benchmark_done_before_fix": False,
        "registered_after_calibration_ledger_before_parameter_selection_and_benchmark": True,
        "existing_inputs": {path.name: runner.pin(path) for path in paths},
        "source": {str(Path(__file__).relative_to(runner.ROOT)).replace("\\", "/"): runner.pin(Path(__file__))},
        "calibration_folds": folds_for_interval(runner.ms(registration["config"]["calibration_start"]),
                                               runner.ms(registration["config"]["calibration_end_exclusive"])),
        "benchmark_folds": folds_for_interval(runner.ms(registration["config"]["benchmark_start"]),
                                             runner.ms(registration["config"]["benchmark_end_exclusive"])),
        "safety": dict(m.SAFETY)}
    continuation["sha256"] = m.identity(continuation)
    runner.write(folder / "continuation_registration.json", continuation)
    print("continuation registered; last short fold merged; existing 12 fits/6 replay ledgers preserved")


def verify_continuation(folder):
    registration = runner.verify(folder)
    continuation = runner.read(folder / "continuation_registration.json")
    if continuation["sha256"] != m.identity({k: v for k, v in continuation.items() if k != "sha256"}):
        raise ValueError("continuation registration changed")
    if continuation["original_registration_sha256"] != registration["sha256"]:
        raise ValueError("original registration mismatch")
    for name, expected in continuation["existing_inputs"].items():
        if runner.pin(folder / name) != expected:
            raise ValueError("existing result changed: " + name)
    for name, expected in continuation["source"].items():
        if runner.pin(runner.ROOT / name) != expected:
            raise ValueError("continuation source changed")
    return registration, continuation


def bank_pair(folder, result, config, prefix):
    import pandas as pd
    from exia.genetic_primus.measurement import build_bootstrap_bank
    _, continuation = verify_continuation(folder)
    start, end = result["spec"]["start_ms"], result["spec"]["end_ms"]
    folds = folds_for_interval(start, end)
    if folds != continuation[prefix + "_folds"]:
        raise ValueError("registered folded interval mismatch")
    frame = pd.DataFrame({"timestamp": [r["timestamp"] for r in result["equity"]], "symbol": "portfolio"})
    cfg = config["bootstrap"]
    banks = {name: build_bootstrap_bank(frame, start_timestamp=start, cutoff_timestamp=end, folds=folds,
              bank_id=prefix + "_" + name, seed=cfg[name + "_seed"], samples=cfg["samples"], block_days=cfg["block_days"])
             for name in ("selection", "audit")}
    runner.np.savez_compressed(folder / (prefix + "_resample_banks.npz"), **{
        name + "_" + kind + "_" + scope: array for name, bank in banks.items()
        for kind, data in (("timestamps", bank.scope_timestamps), ("indices", bank.scope_indices))
        for scope, array in data.items()})
    runner.write(folder / (prefix + "_bank_identities.json"), {"folds": folds, "banks": {k: v.metadata() for k, v in banks.items()}})
    return folds, banks


def calibrate(folder):
    from exia.genetic_primus.profit_fitness_v1 import check_profit_stability
    registration, continuation = verify_continuation(folder)
    if (folder / "selected_policy.json").exists():
        raise ValueError("immutable selection exists")
    results = {t: {s: runner.read(folder / f"calibration_threshold_{t}_{s}.json") for s in ("normal", "stress")}
               for t in (20, 50, 100)}
    rows = [{"policy": "linear6_uniform", "threshold_bps": t,
             "metrics": {s: r["metrics"] for s, r in scenarios.items()}} for t, scenarios in results.items()]
    selected = adapter.rank_calibration(rows)
    reference = results[selected["threshold_bps"]]
    folds, banks = bank_pair(folder, reference["normal"], registration["config"], "calibration")
    audits = {str(t): check_profit_stability(scenarios["normal"], stress=scenarios["stress"], folds=folds,
                selection_bank=banks["selection"], audit_bank=banks["audit"]) for t, scenarios in results.items()}
    selected.update({"registration_sha256": registration["sha256"], "continuation_registration_sha256": continuation["sha256"],
        "stability_audits": audits, "ga_executed": False, "random_search_executed": False,
        "ga_gate": "blocked; simple_model_net_signal_not_confirmed_and_global_registry_incomplete",
        "model_fit_count": 12, "new_financial_replays": 6, "parameter_candidates": 3,
        "model_fit_count_including_known_prior_attempts": 572, "repeated_model_fits": 0,
        "default_deployment_policy": "NoTrade", "selected_threshold_is_research_comparison_only": True,
        "safety": dict(m.SAFETY), "not_prospective_or_release_evidence": True})
    selected["sha256"] = m.identity(selected)
    runner.write(folder / "selected_policy.json", selected)
    print(f"threshold={selected['threshold_bps']}bps; calibration worst_log_growth={selected['worst_log_growth']:.6f}; audit={audits[str(selected['threshold_bps'])]['status']}; default=NoTrade; repeated_fits=0")


def report(folder):
    verify_continuation(folder)
    runner.bank_pair = bank_pair
    runner.report(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("register", "calibrate", "report"))
    parser.add_argument("--folder", type=Path, default=runner.DEFAULT)
    args = parser.parse_args()
    {"register": register, "calibrate": calibrate, "report": report}[args.step](args.folder)
