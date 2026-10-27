"""Read-only independent fit, cashflow, funding and risk checks; no refits."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import joblib
from pathlib import Path
import run_genetic_primus_direction_benchmark_v1 as b


def main():
    reg, bank = b.registration(), b.load_bank()
    checks = {"saved_models": 0, "scalers": 0, "predictions": 0, "folds": 0,
              "fill_costs": 0, "funding_rows": 0, "funding_completeness": 0, "completed_transition_risk": 0,
              "decision_model_lineage": 0, "parent_cash_checks": 0}
    intermediate = {"max_estimated_daily_volatility": 0.0, "max_asset_weight": 0.0,
                    "max_gross_weight": 0.0, "marks_above_completed_transition_risk_target": 0}
    for i, fold in enumerate(reg["folds"]):
        train, test, cutoff = b.split(bank["times"], fold["origin_ms"])
        assert cutoff == fold["cutoff_us"]
        assert b.m.array_identity(train) == fold["train_index_sha256"]
        assert b.m.array_identity(test) == fold["test_index_sha256"]
        assert fold["latest_label_available_us"] <= cutoff
        checks["folds"] += 4
        x, y = bank["features"][train].reshape(-1, 12), bank["returns"][train].reshape(-1)
        receipt = b.sealed(b.OUT / f"origin_{i:02d}_receipt.json")
        for name in b.d.MODELS:
            meta = b.sealed(b.OUT / f"origin_{i:02d}_{name}.json")
            path = b.OUT / f"origin_{i:02d}_{name}.joblib"
            prediction_path = b.OUT / f"origin_{i:02d}_{name}.npz"
            assert meta["sha256"] == receipt["fits"][name]["sha256"]
            assert meta["registration_sha256"] == reg["sha256"]
            assert meta["model_pin"] == b.pin(path) and meta["prediction_pin"] == b.pin(prediction_path)
            model = joblib.load(path)  # Only freshly built local artifacts after pin verification.
            info = model["info"]
            assert b.f.p.c.plain(info) == meta["info"]
            n = len(info["features"])
            selected, labels = x[y != 0, :n], y[y != 0]
            assert info["training_sha256"] == b.m.array_identity(selected, labels)
            scale = selected.std(axis=0); scale[scale < 1e-12] = 1
            assert np.array_equal(selected.mean(axis=0), info["mean"])
            assert np.array_equal(scale, info["scale"])
            checks["scalers"] += 2
            with np.load(prediction_path, allow_pickle=False) as stream:
                expected = b.d.predict(model, bank["features"][test].reshape(-1, 12)).reshape(len(test), 8)
                assert np.array_equal(expected, stream["probability"])
                checks["predictions"] += expected.size
            checks["saved_models"] += 1
    covariances = {day: b.p.risk_covariance(bank["times"], bank["closes"], day)[0]
                   for day in range(b.START, b.END, b.d.DAY)}
    rates = {(int(r["settlement_ms"]), r["symbol"].replace("/", "")): float(r["rate_fraction"])
             for r in b.read(b.parent.MODELED_RATES)["events"]}
    symbols = list(b.parent.SYMBOLS)
    contract = b.read(b.parent.CONFIG)
    for name in b.d.MODELS + b.FINANCIAL_CONTROLS:
        clocks, probabilities, models = b.predictions(name if name in b.d.MODELS else "train_majority")
        lineage = dict(zip(clocks.tolist(), models))
        for scenario in ("normal", "stress"):
            artifact = b.sealed(b.OUT / f"replay_{name}_{scenario}.json")
            assert artifact["registration_sha256"] == reg["sha256"]
            run = artifact["run"]
            checks["parent_cash_checks"] += b.parent.audit_replay(run, set(models))
            quantity = {symbol: 0.0 for symbol in symbols}
            funded = {}
            fee_rate, drag_rate = contract["costs"][scenario]["fee_bps"] / 10000, contract["costs"][scenario]["drag_bps"] / 10000
            for row in run["ledger"]:
                symbol, t = row["symbol"], int(row["timestamp_us"] // 1000)
                if row["kind"] == "modeled_funding":
                    key = (t, symbol.replace("/", ""))
                    assert key not in funded
                    assert row["rate"] == rates[key]
                    px = bank["closes"][np.searchsorted(bank["times"], t), symbols.index(symbol)]
                    assert abs(row["quantity"] - quantity[symbol]) < 1e-9
                    assert abs(row["cash_delta"] + quantity[symbol] * px * rates[key]) < 1e-7
                    funded[key] = row
                    checks["funding_rows"] += 1
                else:
                    assert row["kind"] == "modeled_fill"
                    sign = 1 if row["side"] == "BUY" else -1
                    assert np.isclose(row["price"], row["reference_close"] * (1 + sign * drag_rate), atol=1e-10, rtol=1e-12)
                    assert abs(row["fee"] - row["quantity"] * row["price"] * fee_rate) < 1e-7
                    assert abs(row["slippage"] - row["quantity"] * abs(row["price"] - row["reference_close"])) < 1e-7
                    quantity[symbol] += sign * row["quantity"]
                    checks["fill_costs"] += 3
            for row in run["equity"]:
                t = int(row["timestamp_ms"])
                if row["phase"] == "before_funding_and_decision" and t % (8 * 3_600_000) == 0:
                    for a, symbol in enumerate(symbols):
                        key = (t, symbol.replace("/", ""))
                        notional = row["asset_exposure"][a] * row["nav"]
                        expected = -notional * rates[key]
                        actual = funded.get(key, {}).get("cash_delta", 0.0)
                        assert abs(expected - actual) < 1e-7
                        checks["funding_completeness"] += 1
                if row["phase"] == "after_modeled_fill" and t < b.END:
                    w = np.asarray(row["asset_exposure"])
                    risk = float(np.sqrt(max(0, w @ covariances[t - t % b.d.DAY] @ w)))
                    intermediate["max_estimated_daily_volatility"] = max(intermediate["max_estimated_daily_volatility"], risk)
                    intermediate["max_asset_weight"] = max(intermediate["max_asset_weight"], float(w.max()))
                    intermediate["max_gross_weight"] = max(intermediate["max_gross_weight"], float(w.sum()))
                    intermediate["marks_above_completed_transition_risk_target"] += int(risk > 0.005 + 1e-9)
                if row["phase"] == "calendar_bucket_end":
                    w = np.asarray(row["asset_exposure"])
                    assert np.min(w) >= -1e-12 and np.max(w) <= 0.1 + 1e-9 and w.sum() <= 0.8 + 1e-9
                    if t < b.END:
                        cov = covariances[t - t % b.d.DAY]
                        assert np.sqrt(max(0, w @ cov @ w)) <= 0.005 + 1e-9
                    checks["completed_transition_risk"] += 1
            for row in run["decisions"]:
                assert row["model_sha256"] == lineage[row["asof_ms"]]
                assert row["risk_last_data_ms"] < row["asof_ms"]
                checks["decision_model_lineage"] += 1
    registry = b.sealed(b.NEXT_REGISTRY)
    previous = b.read(b.PREVIOUS_REGISTRY)
    assert registry["objectives"][:-1] == previous["objectives"]
    assert registry["predecessor"]["sha256"] == b.pin(b.PREVIOUS_REGISTRY)["sha256"]
    output = b.write(b.OUT / "independent_checks.json", {
        "checker_pin": b.pin(Path(__file__)), "registration_sha256": reg["sha256"], "checks": checks,
        "intermediate_fill_state_diagnostic": intermediate,
        "risk_semantics": "target enforced at completed transition, not between sequential legs; execution not qualified",
        "registry_immutable_predecessor_preserved": True, "refits": 0, "collector_executed": False,
        "safety": b.read(b.CONFIG)["safety"]})
    print({"models_verified": checks["saved_models"], "cash_checks": checks["parent_cash_checks"],
           "funding_checks": checks["funding_rows"] + checks["funding_completeness"],
           "refits": 0, "passed": True})


if __name__ == "__main__":
    main()
