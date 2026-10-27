"""Post-result descriptive attribution only: zero fits, zero financial replays."""
from __future__ import annotations
import run_genetic_primus_direction_benchmark_v1 as b
from datetime import datetime, timezone
from collections import Counter
import numpy as np
from pathlib import Path


def main():
    reg = b.registration()
    verification = b.sealed(b.OUT / "verification.json")
    for name, expected in verification["artifact_pins"].items():
        assert b.pin(b.OUT / name) == expected
    forecasts = b.sealed(b.OUT / "forecast_comparison.json")["models"]
    with np.load(b.OUT / "resample_bank.npz", allow_pickle=False) as stream:
        banks = {key: stream[key] for key in stream.files}
    warnings = Counter()
    attribution = {}
    for name in b.d.MODELS:
        for i in range(21):
            meta = b.sealed(b.OUT / f"origin_{i:02d}_{name}.json")
            for warning in meta["info"].get("warnings", []):
                warnings[name + "/" + warning.split(":")[0]] += 1
        run = b.sealed(b.OUT / f"replay_{name}_stress.json")["run"]
        daily = np.asarray(run["daily_log_returns"])
        capital = 10000 * np.exp(np.r_[0, daily.cumsum()])
        pnl = np.diff(capital)
        top = np.argsort(pnl)[-3:][::-1]
        attribution[name] = {
            "august_net_return": float(np.expm1(daily[:31].sum())), "september_net_return": float(np.expm1(daily[31:].sum())),
            "top_three_days_ending_utc": [{"day": datetime.fromtimestamp((b.START + (int(i) + 1) * b.d.DAY) / 1000, timezone.utc).date().isoformat(),
                                           "net_pnl": float(pnl[i])} for i in top],
            "top_three_days_total_pnl": float(pnl[top].sum()),
            "total_net_pnl": float(capital[-1] - 10000), "per_asset": run["per_asset"],
            "absolute_family_adjusted_lcb_bps_day": {key: b.f.lcb(daily, bank, 0.05 / 12)["lcb"] * 10000 for key, bank in banks.items()},
            "accuracy_months": {month: row["accuracy"] for month, row in forecasts[name]["by_month"].items()}}
    t, p, _ = b.predictions("sgd_logistic12")
    bank = b.load_bank()
    indices = np.searchsorted(bank["times"], t)
    take = (np.maximum(p, 1 - p) >= 0.8) & (bank["returns"][indices] != 0)
    result = b.write(b.OUT / "descriptive_attribution.json", {
        "registration_sha256": reg["sha256"], "analysis_source_pin": b.pin(Path(__file__)),
        "status": "post_result_descriptive_not_new_confirmatory_test", "fits": 0, "replays": 0,
        "model_attribution": attribution, "warnings": dict(warnings),
        "sgd_high_confidence_subset": {"n": int(take.sum()), "distinct_days": int(len(np.unique(t[np.any(take, axis=1)] // b.d.DAY))),
                                      "timestamps": int(np.any(take, axis=1).sum()), "independent_validation": False}})
    print({"descriptive_models": len(attribution), "new_fits": 0, "new_replays": 0,
           "logistic6_monthly_percent": [round(attribution["logistic6"][key] * 100, 4) for key in ("august_net_return", "september_net_return")],
           "warnings": dict(warnings)})


if __name__ == "__main__":
    main()
