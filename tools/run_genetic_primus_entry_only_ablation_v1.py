"""One registered entry/exit-semantics ablation. Fixed policies, no training/network."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools import analyze_genetic_primus_attribution_august_v2 as diag
from exia.genetic_primus import entry_only_overlay_v1 as policy
from exia.genetic_primus.accounting_v2 import simulate_cash_targets

np, warm, cold, read, DAY = diag.np, diag.warm, diag.cold, diag.read, diag.DAY
CONFIG = ROOT / "configs/genetic_primus_entry_only_ablation_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v9.json"
TEST = ROOT / "tests/test_genetic_primus_entry_only_ablation_v1.py"


def entry_schedule(window, scaler, *, reset_times=()):
    normalized = scaler.transform(window.values)
    valid = np.isfinite(normalized).all(axis=1)
    allowed = valid & ((np.nan_to_num(normalized) @ np.asarray(cold.FIXED[:4]) + cold.FIXED[4]) > 0)
    ema = np.where(valid, window.proposal, 0).reshape(-1, len(window.symbols))
    times = np.arange(window.start, window.end, window.timeframe_ms, dtype=np.int64)
    if ema.shape[0] != len(times) or any(t not in times for t in reset_times):
        raise ValueError("invalid time grid or reset timestamp")
    targets = policy.entry_only_targets(ema, allowed.reshape(ema.shape), reset_before=np.isin(times, reset_times))
    model = {**window.model(cold.FIXED, scaler), "schema": "binary_sparse_entry_only/1",
             "execution_policy": policy.POLICY, "policy_source_sha256": warm.file_sha256(Path(policy.__file__))}
    schedule = window.base_schedule.copy()
    assert np.array_equal(schedule.execution_timestamp.to_numpy(), np.repeat(times, len(window.symbols)))
    assert list(schedule.symbol) == list(window.symbols) * len(times)
    schedule["target"] = targets.ravel()
    schedule["model_sha256"] = warm.object_sha(model)
    return schedule, model


def simulate_entry(window, scaler, cost, *, reset_times=()):
    schedule, model = entry_schedule(window, scaler, reset_times=reset_times)
    sim = simulate_cash_targets(window.bars, schedule, symbols=window.symbols,
        start_timestamp=window.start, end_timestamp=window.end, timeframe_ms=window.timeframe_ms,
        round_trip_cost_bps=cost, record_details=True, force_flat_timestamps=tuple(reset_times))
    return sim, model


def context():
    c = read(CONFIG)
    _, spec, prior_output, warm_output = diag.context()
    assert warm.file_sha256(ROOT / c["source_completion"]) == c["source_completion_sha256"]
    prior = read(prior_output / "completion.json")
    assert prior["status"] == "COMPLETE" and prior["fingerprint"] == diag.fingerprint()
    assert all(warm.file_sha256(prior_output / n) == sha for n, sha in prior["artifact_sha256"].items())
    assert warm.file_sha256(ROOT / c["previous_registry"]) == c["previous_registry_sha256"]
    registry = read(REGISTRY)
    assert registry["previous_registry_sha256"] == c["previous_registry_sha256"]
    assert registry["objectives"][0]["config_sha256"] == warm.file_sha256(CONFIG)
    assert registry["objectives"][0]["objective_id"] == c["objective_id"]
    assert c["genome"] == list(cold.FIXED) and c["costs_bps"] == {"normal": 16.0, "stress": 24.0}
    assert c["comparators"] == ["EMA", "Continuous", "EntryOnly"] and c["origins"] == 10
    assert c["search_evaluations"] == 0 and c["fixed_policy_simulations"] == 180
    assert c["phases"] == ["train", "validation", "test"] and not any(c["safety"].values())
    return c, spec, ROOT / c["output"], prior_output, warm_output


def fingerprint():
    return {"config": warm.file_sha256(CONFIG), "registry": warm.file_sha256(REGISTRY),
        "runner": warm.file_sha256(Path(__file__)), "policy": warm.file_sha256(Path(policy.__file__)),
        "tests": warm.file_sha256(TEST), "unchanged_sources": diag.fingerprint()}


def comparison(challenger, reference, attrs, sims, symbols, start, day_lo, day_hi, dt):
    a, b = attrs[challenger], attrs[reference]
    lo, hi = day_lo * (DAY // dt), day_hi * (DAY // dt)
    delta = (a["net"][lo:hi] - b["net"][lo:hi]).sum(axis=0)
    gross = float((a["gross"][lo:hi] - b["gross"][lo:hi]).sum())
    saved = float((b["fees"][lo:hi] - a["fees"][lo:hi]).sum())
    assert abs(float(delta.sum()) - gross - saved) < 1e-10
    daily_a = cold.daily_returns(sims[challenger])[day_lo:day_hi]
    daily_b = cold.daily_returns(sims[reference])[day_lo:day_hi]
    return {"challenger": challenger, "reference": reference, "days": [day_lo, day_hi],
        "delta_net_pnl": float(delta.sum()), "delta_gross_pnl": gross, "fees_saved": saved,
        "different_target_decisions": int(np.count_nonzero(a["target"][lo:hi] != b["target"][lo:hi])),
        "symbols_delta_net_pnl": {s: float(delta[i]) for i, s in enumerate(symbols)},
        "paired_hac": diag.diagnostic_hac(daily_a, daily_b, start + day_lo * DAY)}


def prepare(_):
    c, spec, output, _, _ = context()
    manifest, _ = cold.panel_and_manifest(spec)
    output.mkdir(parents=True, exist_ok=True)
    cold.save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint(),
        "snapshot_sha256": manifest["snapshot_sha256"], "source_files": manifest["snapshot_files"],
        "bounds": [cold.bounds(spec, i) for i in range(10)], "execution_policy": policy.POLICY,
        "search_evaluations": 0, "fixed_policy_simulations": 180, "safety": c["safety"],
        "python": sys.version, "numpy": np.__version__})
    print(json.dumps({"status": "PREPARED", "origins": 10, "fixed_policy_simulations": 180, "search_evaluations": 0}))


def evaluate(args):
    c, spec, output, prior, source = context()
    if not 0 <= args.origin < 10:
        raise ValueError("origin outside registered budget")
    if args.origin and not (output / "origin_01.json").exists():
        raise ValueError("first engineering smoke must complete")
    pre = read(output / "pre_run_receipt.json")
    assert pre["fingerprint"] == fingerprint()
    stem = f"origin_{args.origin + 1:02d}"
    if (output / (stem + ".json")).exists() or (output / (stem + "_started.json")).exists():
        raise FileExistsError("origin already attempted; preserve receipt")
    selected, old_test = read(source / (stem + "_selected.json")), read(source / (stem + ".json"))
    baseline = read(prior / (stem + ".json"))
    b = cold.bounds(spec, args.origin)
    assert b == selected["bounds"] == baseline["bounds"] == old_test["bounds"]
    manifest, panel = cold.panel_and_manifest(spec)
    assert manifest["snapshot_sha256"] == pre["snapshot_sha256"]
    cold.save_new(output / (stem + "_started.json"), {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "origin_index": args.origin,
        "pre_run_sha256": warm.file_sha256(output / "pre_run_receipt.json")})
    scaler = warm.LabScaler(**{**selected["scaler"], "median": tuple(selected["scaler"]["median"]),
                               "scale": tuple(selected["scaler"]["scale"])})
    symbols, dt = tuple(manifest["metadata"]["symbols"]), spec["timeframe_minutes"] * 60_000
    result = {"origin_index": args.origin, "bounds": b, "scaler": scaler.mapping(),
        "snapshot_sha256": manifest["snapshot_sha256"], "safety": c["safety"],
        "search_evaluations": 0, "fixed_policy_simulations": 18, "phases": {}, "evaluations": {}}
    for phase in c["phases"]:
        start, end = b[phase + "_start"], b[phase + "_end"]
        window = cold.window(panel, symbols, start, end, dt)
        if phase == "train":
            assert warm.array_sha(window.values) == selected["train_values_sha256"]
        days = (end - start) // DAY
        edges = c["train_block_edges_days"] if phase == "train" else [0, days]
        result["phases"][phase] = {}
        for cost, bps in c["costs_bps"].items():
            sims, attrs, models = {}, {}, {}
            if phase == "test":
                result["evaluations"][cost] = {}
            for name in c["comparators"]:
                if name == "EntryOnly":
                    sim, model = simulate_entry(window, scaler, bps)
                else:
                    genome = warm.ALLOW_ALL if name == "EMA" else cold.FIXED
                    sim, model = window.simulate(genome, scaler, bps, details=True), window.model(genome, scaler)
                a = diag.base.bar_attribution(window.bars, sim.decisions, sim.trades, sim.portfolio,
                                              symbols, start, end, dt)
                model_sha = warm.object_sha(model)
                assert all(sim.decisions.model_sha256 == model_sha)
                assert all(sim.decisions.signal_timestamp <= sim.decisions.execution_timestamp)
                proof = {"model": model, "model_sha256": model_sha, "metrics": sim.metrics,
                    "decisions_sha256": warm.object_sha(sim.decisions.to_dict("records")),
                    "portfolio_sha256": warm.object_sha(sim.portfolio.to_dict("records")),
                    "attribution_max_error": a["error"]}
                if name != "EntryOnly":
                    old = baseline["phases"][phase][cost]["proofs"]["EMA" if name == "EMA" else "Fixed"]
                    assert all(warm.object_sha(proof[k]) == warm.object_sha(old[k]) for k in old)
                durations = (sim.trades.exit_timestamp - sim.trades.entry_timestamp).to_numpy() / 3_600_000 if len(sim.trades) else np.array([])
                models[name] = {"proof": proof, "daily_returns": cold.daily_returns(sim).tolist(),
                    "summary": diag.base.describe(a, 0, len(a["equity"])),
                    "mean_trade_hours": float(durations.mean()) if len(durations) else None,
                    "exit_reasons": dict(Counter(sim.trades.exit_reason)) if len(sim.trades) else {}}
                sims[name], attrs[name] = sim, a
                if phase == "test":
                    payload = cold.payload(sim, model, b, manifest["snapshot_sha256"], scaler.fit_values_sha256)
                    if name != "EntryOnly":
                        old_name = "EMA" if name == "EMA" else "FixedOverlaySameScaler"
                        assert warm.object_sha(payload) == warm.object_sha(old_test["evaluations"][cost][old_name])
                    result["evaluations"][cost][name] = payload
            assert sims["EntryOnly"].metrics["trades"] <= sims["EMA"].metrics["trades"]
            assert sims["EntryOnly"].metrics["trades"] <= sims["Continuous"].metrics["trades"]
            assert np.all((attrs["EntryOnly"]["target"] == 0) | (attrs["EntryOnly"]["target"] == attrs["EMA"]["target"]))
            assert np.all((attrs["Continuous"]["target"] == 0) | (attrs["Continuous"]["target"] == attrs["EntryOnly"]["target"]))
            pairs = {}
            for ref in ("EMA", "Continuous"):
                full = comparison("EntryOnly", ref, attrs, sims, symbols, start, 0, days, dt)
                parts = [comparison("EntryOnly", ref, attrs, sims, symbols, start, lo, hi, dt)
                         for lo, hi in diag.base.blocks(days, edges)]
                assert abs(sum(x["delta_net_pnl"] for x in parts) - full["delta_net_pnl"]) < 1e-10
                pairs[ref] = {"full": full, "blocks": parts}
            result["phases"][phase][cost] = {"models": models, "comparisons": pairs}
    cold.save_new(output / (stem + ".json"), result)
    print(json.dumps({"origin": args.origin + 1, "status": "COMPLETE", "test_delta_pp": {
        ref: round(result["phases"]["test"]["normal"]["comparisons"][ref]["full"]["delta_net_pnl"] * 100, 4)
        for ref in ("EMA", "Continuous")}}))


def finalize(_):
    c, _, output, _, _ = context()
    assert read(output / "pre_run_receipt.json")["fingerprint"] == fingerprint()
    rs = [read(output / f"origin_{i + 1:02d}.json") for i in range(10)]
    assert all(r["origin_index"] == i and not any(r["safety"].values()) for i, r in enumerate(rs))
    summary = {"status": "REVEALED_ABLATION_COMPLETE_NO_PROMOTION", "origins": 10,
        "search_evaluations": 0, "fixed_policy_simulations": 180, "hypotheses_added": 1,
        "data_already_revealed": True, "promotion_authority": False, "costs": {}, "phases": {}}
    for cost in c["costs_bps"]:
        candidates = {k: cold.aggregate(rs, cost, k) for k in c["comparators"]}
        entry = candidates["EntryOnly"]["daily_returns"]
        summary["costs"][cost] = {"candidates": candidates, "entry_only_hac_vs": {
            k: diag.diagnostic_hac(entry, candidates[k]["daily_returns"], rs[0]["bounds"]["test_start"])
            for k in ("EMA", "Continuous")}}
    for phase in c["phases"]:
        summary["phases"][phase] = {}
        for cost in c["costs_bps"]:
            summary["phases"][phase][cost] = {}
            for ref in ("EMA", "Continuous"):
                rows = [r["phases"][phase][cost]["comparisons"][ref]["full"] for r in rs]
                lcbs = [x["paired_hac"]["lcb"] for x in rows]
                summary["phases"][phase][cost][ref] = {
                    "positive_origins": sum(x["delta_net_pnl"] > 0 for x in rows),
                    "positive_lcb_origins": sum(x is not None and x > 0 for x in lcbs),
                    "delta_net_pnl_range": [min(x["delta_net_pnl"] for x in rows), max(x["delta_net_pnl"] for x in rows)]}
    cold.save_new(output / "summary.json", summary)
    names = ["pre_run_receipt.json", "summary.json"]
    names += [f"origin_{i:02d}{suffix}.json" for i in range(1, 11) for suffix in ("", "_started")]
    cold.save_new(output / "completion.json", {"status": summary["status"], "fingerprint": fingerprint(),
        "artifact_sha256": {n: warm.file_sha256(output / n) for n in names}, "safety": c["safety"]})
    print(json.dumps({"status": summary["status"], "origins": 10, "artifacts": len(names)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)
    sub.add_parser("prepare").set_defaults(fn=prepare)
    run = sub.add_parser("origin")
    run.add_argument("--origin", type=int, required=True)
    run.set_defaults(fn=evaluate)
    sub.add_parser("finalize").set_defaults(fn=finalize)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
