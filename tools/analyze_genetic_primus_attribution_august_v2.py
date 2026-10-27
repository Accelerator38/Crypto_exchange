"""Versioned retry: descriptive blocks; unchanged qualified 30/365-day HAC only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import analyze_genetic_primus_attribution_august_v1 as base

np, warm, cold = base.np, base.warm, base.cold
read, DAY = base.read, base.DAY
CONFIG = ROOT / "configs/genetic_primus_attribution_august_v2.json"
TEST = ROOT / "tests/test_genetic_primus_attribution_august_v2.py"


def diagnostic_hac(fixed_daily, ema_daily, start):
    a, b = np.asarray(fixed_daily, dtype=float), np.asarray(ema_daily, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid paired daily inputs")
    if len(a) not in (30, 365):
        return {"status": "descriptive_only_unqualified_horizon", "daily_count": len(a),
                "mean_difference": float((a - b).mean()), "lcb": None, "promotion_authority": False}
    return warm.assess_paired_daily(a, b, start + np.arange(1, len(a) + 1) * DAY).mapping()


def paired_block(ema, fixed, sims, symbols, start, end, dt, day_lo, day_hi):
    lo, hi = day_lo * (DAY // dt), day_hi * (DAY // dt)
    models = {k: base.describe(a, lo, hi) for k, a in (("EMA", ema), ("Fixed", fixed))}
    daily = {k: cold.daily_returns(v)[day_lo:day_hi] for k, v in sims.items()}
    diff = (fixed["net"][lo:hi] - ema["net"][lo:hi]).sum(axis=0)
    opportunities = (ema["target"][lo:hi] != 0).sum(axis=0)
    total_abs, total_opp = float(np.abs(diff).sum()), int(opportunities.sum())
    by_symbol = {}
    for col, symbol in enumerate(symbols):
        share = float(abs(diff[col]) / total_abs) if total_abs else None
        opp_share = int(opportunities[col]) / total_opp if total_opp else None
        by_symbol[symbol] = {
            "delta_net_pnl": float(diff[col]),
            "delta_gross_pnl": float((fixed["gross"][lo:hi, col] - ema["gross"][lo:hi, col]).sum()),
            "fees_saved": float((ema["fees"][lo:hi, col] - fixed["fees"][lo:hi, col]).sum()),
            "ema_opportunities": int(opportunities[col]), "absolute_delta_share": share,
            "opportunity_normalized_absolute_delta_share": share / opp_share if share is not None and opp_share else None}
    net = float(diff.sum())
    gross = models["Fixed"]["gross_pnl"] - models["EMA"]["gross_pnl"]
    saved = models["EMA"]["fees"] - models["Fixed"]["fees"]
    if abs(net - gross - saved) > 1e-10:
        raise ArithmeticError("paired gross/fee decomposition failed")
    return {"days": [day_lo, day_hi], "start": start + day_lo * DAY,
            "end_exclusive": start + day_hi * DAY, "models": models,
            "delta_net_pnl": net, "delta_gross_pnl": gross, "fees_saved": saved,
            "paired_hac": diagnostic_hac(daily["Fixed"], daily["EMA"], start + day_lo * DAY),
            "symbols": by_symbol, "ema_trade_buckets_by_entry": base.trade_buckets(
                sims["EMA"].trades, fixed["target"], symbols, start, end, dt,
                start + day_lo * DAY, start + day_hi * DAY)}


def context():
    c2 = read(CONFIG)
    assert warm.file_sha256(base.CONFIG) == c2["base_config_sha256"]
    assert warm.file_sha256(Path(base.__file__)) == c2["base_runner_sha256"]
    c, spec, old_output, source = base.config_and_source()
    assert warm.file_sha256(old_output / "pre_run_receipt.json") == c2["failed_receipt_sha256"]
    assert not c2["promotion_authority"]
    return c, spec, ROOT / c2["output"], source


def fingerprint():
    return {"config": warm.file_sha256(CONFIG), "runner": warm.file_sha256(Path(__file__)),
            "tests": warm.file_sha256(TEST), "base": base.fingerprint()}


def prepare(_):
    c, spec, output, _ = context()
    manifest, _ = cold.panel_and_manifest(spec)
    output.mkdir(parents=True, exist_ok=True)
    cold.save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint(),
        "snapshot_sha256": manifest["snapshot_sha256"], "source_files": manifest["snapshot_files"],
        "blocks": base.blocks(365, c["train_block_edges_days"]), "safety": c["safety"],
        "search_evaluations": 0, "fixed_policy_simulations": 120, "earlier_failed_simulations": 2})
    print(json.dumps({"status": "PREPARED", "search_evaluations": 0, "fixed_policy_simulations": 120}))


def evaluate(args):
    c, spec, output, source = context()
    if not 0 <= args.origin < 10:
        raise ValueError("origin outside fixed budget")
    pre = read(output / "pre_run_receipt.json")
    assert pre["fingerprint"] == fingerprint()
    stem = f"origin_{args.origin + 1:02d}"
    if (output / (stem + ".json")).exists():
        raise FileExistsError("origin already complete")
    selected, original = read(source / (stem + "_selected.json")), read(source / (stem + ".json"))
    b = cold.bounds(spec, args.origin)
    assert b == selected["bounds"] == original["bounds"]
    manifest, panel = cold.panel_and_manifest(spec)
    assert manifest["snapshot_sha256"] == pre["snapshot_sha256"]
    cold.save_new(output / (stem + "_started.json"), {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "origin_index": args.origin,
        "receipt_sha256": warm.file_sha256(output / "pre_run_receipt.json")})
    scaler = warm.LabScaler(**{**selected["scaler"], "median": tuple(selected["scaler"]["median"]),
                               "scale": tuple(selected["scaler"]["scale"])})
    assert warm.object_sha(scaler.mapping()) == warm.object_sha(selected["scaler"])
    symbols, dt = tuple(manifest["metadata"]["symbols"]), spec["timeframe_minutes"] * 60_000
    result = {"origin_index": args.origin, "bounds": b, "safety": c["safety"],
              "snapshot_sha256": manifest["snapshot_sha256"], "scaler": scaler.mapping(),
              "search_evaluations": 0, "fixed_policy_simulations": 12, "phases": {}}
    for phase in c["phases"]:
        start, end = b[phase + "_start"], b[phase + "_end"]
        window = cold.window(panel, symbols, start, end, dt)
        if phase == "train":
            assert warm.array_sha(window.values) == selected["train_values_sha256"]
        days = (end - start) // DAY
        edges = c["train_block_edges_days"] if phase == "train" else [0, days]
        result["phases"][phase] = {}
        for cost, bps in c["costs_bps"].items():
            sims, attrs, proofs = {}, {}, {}
            for name, genome in c["models"].items():
                genome = tuple(genome)
                sim = window.simulate(genome, scaler, bps, details=True)
                sims[name] = sim
                if phase == "train" and cost == "normal":
                    assert warm.object_sha(sim.metrics) == warm.object_sha(selected["training"][str(genome)]["metrics"])
                if phase == "test":
                    old = original["evaluations"][cost]["EMA" if name == "EMA" else "FixedOverlaySameScaler"]
                    for key, value in (("metrics", sim.metrics), ("portfolio", sim.portfolio.to_dict("records")),
                                       ("trades", sim.trades.to_dict("records"))):
                        assert warm.object_sha(value) == warm.object_sha(old[key])
                a = base.bar_attribution(window.bars, sim.decisions, sim.trades, sim.portfolio,
                                         symbols, start, end, dt)
                model_sha = warm.object_sha(window.model(genome, scaler))
                assert all(sim.decisions.model_sha256 == model_sha)
                attrs[name] = a
                proofs[name] = {"model_sha256": model_sha, "metrics": sim.metrics,
                    "decisions_sha256": warm.object_sha(sim.decisions.to_dict("records")),
                    "portfolio_sha256": warm.object_sha(sim.portfolio.to_dict("records")),
                    "attribution_max_error": a["error"]}
            assert np.all((attrs["Fixed"]["target"] == 0) | (attrs["Fixed"]["target"] == attrs["EMA"]["target"]))
            rows = [paired_block(attrs["EMA"], attrs["Fixed"], sims, symbols, start, end, dt, lo, hi)
                    for lo, hi in base.blocks(days, edges)]
            full = paired_block(attrs["EMA"], attrs["Fixed"], sims, symbols, start, end, dt, 0, days)
            assert abs(sum(x["delta_net_pnl"] for x in rows) - full["delta_net_pnl"]) < 1e-10
            if phase == "train" and cost == "normal":
                assert warm.object_sha(full["paired_hac"]) == warm.object_sha(selected["training"][str(cold.FIXED)]["hac"])
            result["phases"][phase][cost] = {"proofs": proofs, "blocks": rows, "full": full}
    cold.save_new(output / (stem + ".json"), result)
    print(json.dumps({"origin": args.origin + 1, "status": "COMPLETE", **{
        phase + "_delta_pp": round(100 * result["phases"][phase]["normal"]["full"]["delta_net_pnl"], 4)
        for phase in c["phases"]}}))


def finalize(_):
    c, _, output, _ = context()
    assert read(output / "pre_run_receipt.json")["fingerprint"] == fingerprint()
    rs = [read(output / f"origin_{i + 1:02d}.json") for i in range(10)]
    assert all(r["origin_index"] == i and not any(r["safety"].values()) for i, r in enumerate(rs))
    summary = {"status": "COMPLETE", "origins": 10, "search_evaluations": 0,
               "fixed_policy_simulations": 120, "earlier_failed_simulations": 2,
               "promotion_authority": False, "overlapping_origins_must_not_be_pooled": True, "phases": {}}
    for phase in c["phases"]:
        summary["phases"][phase] = {}
        for cost in c["costs_bps"]:
            rows = [r["phases"][phase][cost]["full"] for r in rs]
            summary["phases"][phase][cost] = {
                "positive_origins": sum(x["delta_net_pnl"] > 0 for x in rows),
                "delta_net_pnl_range": [min(x["delta_net_pnl"] for x in rows), max(x["delta_net_pnl"] for x in rows)],
                "delta_net_pnl_by_origin": [x["delta_net_pnl"] for x in rows],
                "fees_saved_range": [min(x["fees_saved"] for x in rows), max(x["fees_saved"] for x in rows)],
                "positive_blocks_by_origin": [sum(x["delta_net_pnl"] > 0 for x in r["phases"][phase][cost]["blocks"]) for r in rs]}
    cold.save_new(output / "summary.json", summary)
    names = ["pre_run_receipt.json", "summary.json"]
    names += [f"origin_{i:02d}{suffix}.json" for i in range(1, 11) for suffix in ("", "_started")]
    cold.save_new(output / "completion.json", {"status": "COMPLETE", "fingerprint": fingerprint(),
        "artifact_sha256": {n: warm.file_sha256(output / n) for n in names}, "safety": c["safety"]})
    print(json.dumps({"status": "COMPLETE", "origins": 10, "artifacts": len(names), "search_evaluations": 0}))


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
