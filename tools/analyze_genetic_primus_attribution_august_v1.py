"""Fixed-policy attribution of revealed data. No model fitting or genetic search."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from tools import run_genetic_primus_warmstart_august_v1 as warm

cold = warm.cold
CONFIG = ROOT / "configs/genetic_primus_attribution_august_v1.json"
TEST = ROOT / "tests/test_genetic_primus_attribution_august_v1.py"
DAY = 86_400_000


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def blocks(days, edges):
    """Disjoint half-open blocks, including all days exactly once."""
    if (len(edges) < 2 or edges[0] != 0 or edges[-1] != days
            or any(not isinstance(x, int) for x in edges)
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError("invalid fixed block boundaries")
    return tuple(zip(edges, edges[1:]))


def bar_attribution(bars, decisions, trades, portfolio, symbols, start, end, dt):
    """Reconcile per-symbol open gaps, holding PnL and actual fill fees to equity."""
    times = np.arange(start, end, dt, dtype=np.int64)
    if end <= start or (end - start) % dt or not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("invalid market grid")
    index = pd.MultiIndex.from_product([times, symbols], names=["timestamp", "symbol"])
    b = bars.set_index(["timestamp", "symbol"]).reindex(index)
    d = decisions.rename(columns={"execution_timestamp": "timestamp"}).set_index(
        ["timestamp", "symbol"]).reindex(index)
    shape = (len(times), len(symbols))
    opens = b["open"].to_numpy(dtype=float).reshape(shape)
    closes = b["close"].to_numpy(dtype=float).reshape(shape)
    quantity = d["quantity_after"].to_numpy(dtype=float).reshape(shape)
    target = d["desired_position"].to_numpy(dtype=float).reshape(shape)
    if not all(np.isfinite(a).all() for a in (opens, closes, quantity, target)):
        raise ValueError("missing/nonfinite attribution input")
    previous_q = np.vstack([np.zeros((1, shape[1])), quantity[:-1]])
    previous_close = np.vstack([opens[:1], closes[:-1]])
    gross = previous_q * (opens - previous_close) + quantity * (closes - opens)
    fees, turnover, exits = (np.zeros(shape) for _ in range(3))
    for t in trades.to_dict("records"):
        col = symbols.index(t["symbol"])
        for side in ("entry", "exit"):
            timestamp = int(t[f"{side}_timestamp"])
            if not start <= timestamp <= end or (timestamp - start) % dt:
                raise ValueError("fill off grid")
            if side == "entry" and timestamp == end:
                raise ValueError("entry at exclusive end")
            row = min((timestamp - start) // dt, len(times) - 1)
            fees[row, col] += float(t[f"{side}_fee"])
            turnover[row, col] += abs(float(t["quantity"]) * float(t[f"{side}_price"]))
            if side == "exit":
                exits[row, col] += 1
    net = gross - fees
    equity = 1.0 + np.cumsum(net.sum(axis=1))
    if not np.array_equal(portfolio.timestamp.to_numpy(), times + dt):
        raise ValueError("portfolio timestamps do not match")
    error = float(np.max(np.abs(equity - portfolio.equity.to_numpy())))
    if error > 1e-10:
        raise ArithmeticError(f"attribution does not reconcile: {error}")
    trade_gross = float(trades.gross_pnl.sum()) if len(trades) else 0.0
    if abs(float(gross.sum()) - trade_gross) > 1e-10:
        raise ArithmeticError("gross trade PnL mismatch")
    return {"gross": gross, "fees": fees, "net": net, "turnover": turnover,
            "exits": exits, "target": target, "equity": equity, "error": error}


def trade_buckets(trades, target, symbols, start, end, dt, entry_start=None, entry_end=None):
    """Classify original EMA trades; bucket PnL is NOT an incremental-return estimate."""
    result = {name: {"count": 0, "ema_net_pnl": 0.0, "ema_gross_pnl": 0.0}
              for name in ("fully_blocked", "partially_held", "fully_held")}
    for t in trades.to_dict("records"):
        entry, leave = int(t["entry_timestamp"]), int(t["exit_timestamp"])
        if entry_start is not None and not entry_start <= entry < entry_end:
            continue
        if not start <= entry < leave <= end:
            raise ValueError("invalid EMA trade interval")
        values = target[(entry - start) // dt:(leave - start) // dt, symbols.index(t["symbol"])]
        active = values == t["direction"]
        name = "fully_held" if active.all() else "partially_held" if active.any() else "fully_blocked"
        result[name]["count"] += 1
        result[name]["ema_net_pnl"] += float(t["net_pnl"])
        result[name]["ema_gross_pnl"] += float(t["gross_pnl"])
    return result


def describe(a, lo, hi):
    previous = 1.0 if lo == 0 else float(a["equity"][lo - 1])
    path = np.r_[previous, a["equity"][lo:hi]]
    return {
        "net_pnl": float(a["net"][lo:hi].sum()),
        "gross_pnl": float(a["gross"][lo:hi].sum()),
        "fees": float(a["fees"][lo:hi].sum()),
        "gross_notional_turnover": float(a["turnover"][lo:hi].sum()),
        "closed_trades": int(a["exits"][lo:hi].sum()),
        "period_return": float(path[-1] / previous - 1),
        "close_4h_drawdown": float(np.max(1 - path / np.maximum.accumulate(path))),
    }


def paired_block(ema, fixed, sims, symbols, start, end, dt, day_lo, day_hi):
    per_day = DAY // dt
    lo, hi = day_lo * per_day, day_hi * per_day
    models = {name: describe(a, lo, hi) for name, a in (("EMA", ema), ("Fixed", fixed))}
    daily = {k: cold.daily_returns(v)[day_lo:day_hi] for k, v in sims.items()}
    timestamps = start + np.arange(day_lo + 1, day_hi + 1) * DAY
    hac = warm.assess_paired_daily(daily["Fixed"], daily["EMA"], timestamps).mapping()
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
            "ema_opportunities": int(opportunities[col]),
            "absolute_delta_share": share,
            "opportunity_normalized_absolute_delta_share": share / opp_share if share is not None and opp_share else None,
        }
    net = float(diff.sum())
    gross = models["Fixed"]["gross_pnl"] - models["EMA"]["gross_pnl"]
    saved = models["EMA"]["fees"] - models["Fixed"]["fees"]
    if abs(net - gross - saved) > 1e-10:
        raise ArithmeticError("paired gross/fee decomposition failed")
    return {"days": [day_lo, day_hi], "start": start + day_lo * DAY,
            "end_exclusive": start + day_hi * DAY, "models": models,
            "delta_net_pnl": net, "delta_gross_pnl": gross, "fees_saved": saved,
            "paired_hac": hac, "symbols": by_symbol,
            "ema_trade_buckets_by_entry": trade_buckets(sims["EMA"].trades, fixed["target"],
                symbols, start, end, dt, start + day_lo * DAY, start + day_hi * DAY)}


def config_and_source():
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise RuntimeError("project venv required")
    c = read(CONFIG)
    assert c["search_evaluations"] == 0 and c["maximum_fixed_policy_simulations"] == 120
    assert c["origins"] == 10 and c["phases"] == ["train", "validation", "test"]
    assert c["models"] == {"EMA": list(warm.ALLOW_ALL), "Fixed": list(cold.FIXED)}
    assert c["costs_bps"] == {"normal": 16.0, "stress": 24.0}
    assert not any(c["safety"].values()) and c["train_block_edges_days"] == [0, 91, 182, 273, 365]
    source_path = ROOT / c["warm_completion"]
    assert warm.file_sha256(source_path) == c["warm_completion_sha256"]
    assert warm.file_sha256(ROOT / c["registry"]) == c["registry_sha256"]
    source = read(source_path)
    assert source["status"] == "REVEALED_WARMSTART_COMPLETE_NO_PROMOTION"
    assert source["fingerprint"] == warm.fingerprint()
    for name, sha in source["artifact_sha256"].items():
        assert warm.file_sha256(source_path.parent / name) == sha
    base, _ = cold.load_config()
    return c, base, ROOT / c["output"], source_path.parent


def fingerprint():
    return {"config": warm.file_sha256(CONFIG), "runner": warm.file_sha256(Path(__file__)),
            "tests": warm.file_sha256(TEST), "frozen_warm_sources": warm.fingerprint()}


def prepare(_):
    c, base, output, _ = config_and_source()
    manifest, _ = cold.panel_and_manifest(base)
    output.mkdir(parents=True, exist_ok=True)
    cold.save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint(),
        "snapshot_sha256": manifest["snapshot_sha256"], "source_files": manifest["snapshot_files"],
        "blocks": blocks(365, c["train_block_edges_days"]), "safety": c["safety"],
        "search_evaluations": 0, "fixed_policy_simulations": 120})
    print(json.dumps({"status": "PREPARED", "search_evaluations": 0, "fixed_policy_simulations": 120}))


def evaluate(args):
    c, base, output, source = config_and_source()
    if not 0 <= args.origin < 10:
        raise ValueError("origin outside fixed budget")
    pre = read(output / "pre_run_receipt.json")
    assert pre["fingerprint"] == fingerprint()
    stem = f"origin_{args.origin + 1:02d}"
    if (output / (stem + ".json")).exists():
        raise FileExistsError("origin already complete")
    selected, original = read(source / (stem + "_selected.json")), read(source / (stem + ".json"))
    b = cold.bounds(base, args.origin)
    assert b == selected["bounds"] == original["bounds"]
    manifest, panel = cold.panel_and_manifest(base)
    assert manifest["snapshot_sha256"] == pre["snapshot_sha256"]
    cold.save_new(output / (stem + "_started.json"), {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "origin_index": args.origin,
        "receipt_sha256": warm.file_sha256(output / "pre_run_receipt.json")})
    scaler = warm.LabScaler(**{**selected["scaler"], "median": tuple(selected["scaler"]["median"]),
                               "scale": tuple(selected["scaler"]["scale"])})
    assert warm.object_sha(scaler.mapping()) == warm.object_sha(selected["scaler"])
    symbols, dt = tuple(manifest["metadata"]["symbols"]), base["timeframe_minutes"] * 60_000
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
                a = bar_attribution(window.bars, sim.decisions, sim.trades, sim.portfolio,
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
                    for lo, hi in blocks(days, edges)]
            full = paired_block(attrs["EMA"], attrs["Fixed"], sims, symbols, start, end, dt, 0, days)
            assert abs(sum(x["delta_net_pnl"] for x in rows) - full["delta_net_pnl"]) < 1e-10
            if phase == "train" and cost == "normal":
                assert warm.object_sha(full["paired_hac"]) == warm.object_sha(selected["training"][str(cold.FIXED)]["hac"])
            result["phases"][phase][cost] = {"proofs": proofs, "blocks": rows, "full": full}
    cold.save_new(output / (stem + ".json"), result)
    f = result["phases"]["train"]["normal"]["full"]
    print(json.dumps({"origin": args.origin + 1, "status": "COMPLETE", "train_delta_pp": round(100 * f["delta_net_pnl"], 4),
                      "validation_delta_pp": round(100 * result["phases"]["validation"]["normal"]["full"]["delta_net_pnl"], 4),
                      "test_delta_pp": round(100 * result["phases"]["test"]["normal"]["full"]["delta_net_pnl"], 4)}))


def finalize(_):
    c, _, output, _ = config_and_source()
    assert read(output / "pre_run_receipt.json")["fingerprint"] == fingerprint()
    rs = [read(output / f"origin_{i + 1:02d}.json") for i in range(10)]
    assert all(r["origin_index"] == i and not any(r["safety"].values()) for i, r in enumerate(rs))
    summary = {"status": "COMPLETE", "origins": 10, "search_evaluations": 0,
               "fixed_policy_simulations": 120, "promotion_authority": False,
               "overlapping_origins_must_not_be_pooled": True, "phases": {}}
    for phase in c["phases"]:
        summary["phases"][phase] = {}
        for cost in c["costs_bps"]:
            rows = [r["phases"][phase][cost]["full"] for r in rs]
            summary["phases"][phase][cost] = {
                "positive_origins": sum(x["delta_net_pnl"] > 0 for x in rows),
                "delta_net_pnl_range": [min(x["delta_net_pnl"] for x in rows), max(x["delta_net_pnl"] for x in rows)],
                "delta_net_pnl_by_origin": [x["delta_net_pnl"] for x in rows],
                "fees_saved_range": [min(x["fees_saved"] for x in rows), max(x["fees_saved"] for x in rows)],
                "positive_blocks_by_origin": [sum(x["delta_net_pnl"] > 0 for x in r["phases"][phase][cost]["blocks"]) for r in rs],
            }
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
