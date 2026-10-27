"""Independent cash/inventory audit of already computed offline ledgers."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_genetic_primus_position_policy_v1 as run


def audit(seal=False):
    reg, data, _, _, _ = run.load()
    passed, files = 0, 0
    def check(value, label):
        nonlocal passed
        if not value:
            raise AssertionError(label)
        passed += 1
    for period in reg["periods"]:
        for policy in run.p.POLICIES:
            for scenario in run.SCENARIOS:
                item = run.sealed(run.OUT / f"replay_{period}_{policy}_{scenario}.json")
                n = len(reg["symbols"])
                cash, qty, basis = 10000., np.zeros(n), np.zeros(n)
                sums = {s: dict(gross_pnl=0., fees=0., slippage=0., funding=0.) for s in reg["symbols"]}
                indices = {s: i for i, s in enumerate(reg["symbols"])}
                decisions = {r["origin_ms"]: r for r in item["decisions"]}
                events, turnover = {}, 0.
                for event in item["ledger"]:
                    t = event["timestamp_us"] // (run.c.BAR_MS * 1000) * run.c.BAR_MS
                    events.setdefault(t, []).append(event)
                active_model = None
                for row in (r for r in item["equity"] if r["phase"] == "calendar_bucket_end"):
                    t = row["timestamp_ms"]
                    old_model = active_model
                    if t in decisions:
                        active_model = decisions[t]["model_sha256"]
                    position = int(np.searchsorted(data["times"], t))
                    prices = data["closes"][position]
                    for event in events.get(t, []):
                        i, symbol = indices[event["symbol"]], event["symbol"]
                        if event["kind"] == "modeled_funding":
                            check(event["model_sha256"] == old_model, "funding uses previously active model")
                            check(abs(event["quantity"] - qty[i]) < 1e-8, "funding quantity")
                            sums[symbol]["funding"] += event["cash_delta"]
                        else:
                            q, px, ref = event["quantity"], event["price"], event["reference_close"]
                            check(event["model_sha256"] == active_model, "fill model SHA")
                            check(abs(ref - prices[i]) < 1e-10, "reference price")
                            check(abs(event["fee"] - q * px * .0006) < 1e-8, "actual delta commission")
                            check(abs(event["slippage"] - q * abs(px - ref)) < 1e-8, "actual delta slippage")
                            if event["side"] == "BUY":
                                expected_delta, gross = -q * px - event["fee"], 0.
                                qty[i] += q
                                basis[i] += q * ref
                            else:
                                check(q <= qty[i] + 1e-8, "no overselling")
                                removed = basis[i] * min(1., q / qty[i])
                                expected_delta, gross = q * px - event["fee"], q * ref - removed
                                qty[i] -= q
                                basis[i] -= removed
                                if qty[i] < 1e-10:
                                    qty[i], basis[i] = 0., 0.
                            check(abs(expected_delta - event["cash_delta"]) < 1e-8, "fill cash delta")
                            check(abs(gross - event["gross_pnl"]) < 1e-7, "reference-basis gross")
                            for key, value in (("gross_pnl", gross), ("fees", event["fee"]), ("slippage", event["slippage"])):
                                sums[symbol][key] += value
                            turnover += q * ref / 10000
                        cash += event["cash_delta"]
                    nav = cash + float(qty @ prices)
                    check(abs(row["cash"] - cash) < 1e-7, "calendar cash")
                    check(abs(row["nav"] - nav) < 1e-7, "calendar marked NAV")
                    check(abs(row["exposure"] - float(qty @ prices) / nav) < 1e-10, "marked exposure")
                    check(cash >= -1e-7 and np.all(qty >= -1e-10), "no borrowing/shorting")
                check(np.max(abs(qty)) < 1e-10, "terminal inventory flat")
                check(abs(turnover - item["metrics"]["turnover_initial_capital"]) < 1e-8, "actual traded turnover")
                for symbol, detail in sums.items():
                    for key, value in detail.items():
                        check(abs(value - item["per_symbol"][symbol][key]) < 1e-7, "per-asset cashflows")
                files += 1
    result = {"checks_passed": passed, "replays_audited": files, "new_fits": 0, "new_replays": 0,
              "audit_source_path": run.base.relative(Path(__file__)), "audit_source_pin": run.base.pin(Path(__file__)),
              "registration_sha256": reg["sha256"], "safety": run.c.SAFETY}
    if seal:
        run.write("independent_audit.json", result)
    print({"checks_passed": passed, "replays_audited": files, "writes": bool(seal), "new_fits": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    audit(args.seal)
