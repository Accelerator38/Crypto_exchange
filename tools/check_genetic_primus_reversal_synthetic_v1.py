"""Short synthetic-only checks; no collector, market data, network or pytest."""

from dataclasses import replace
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "src/exia/genetic_primus/market_relative_reversal_linear_v1.py"
    api = runpy.run_path(str(source))
    Observation, fit = api["Observation"], api["fit_origin"]
    forecast, features = api["predict_gross"], api["market_relative_features"]
    universe, bar, horizon = api["UNIVERSE"], api["BAR_MS"], api["HORIZON_MS"]
    implementation_sha = sha256(source.read_bytes()).hexdigest()
    deploy = 1780272000000  # Synthetic fixture anchor: 2026-06-01 UTC, not quotes.
    x = [-0.07, -0.05, -0.03, -0.01, 0.01, 0.03, 0.05, 0.07]
    rows = [Observation(deploy - (k + 2) * horizon, deploy - (k + 2) * horizon - bar,
                        deploy - (k + 1) * horizon, symbol, x[i], -0.2 * x[i] - 0.002)
            for k in range(60) for i, symbol in enumerate(universe)]

    def fitted(data, origin=deploy):
        return fit(data, deploy_origin_ms=origin, implementation_sha256=implementation_sha)

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("unsafe input was accepted")

    checks = []
    relative = features(dict(zip(universe, x)))
    assert list(relative) == list(universe) and isclose(sum(relative.values()), 0, abs_tol=1e-12)
    rejects(lambda: features({universe[0]: 0.01}))
    checks.append("full8_feature_and_missing_asset_rejection")

    model = fitted(rows)
    assert model["fallback"] is None and model["negative_outcomes"] > 0
    assert model["distinct_training_origins"] == 60 and model["eligible_rows"] == 480
    assert isclose(model["origin_weight_ess"], 60) and model["row_weight_ess"] == 480
    # Independent linear-quantile calculation at ranks 119.75 and 359.25.
    expected_iqr = (0.03 + 0.25*0.02) - (-0.05 + 0.75*0.02)
    variance = sum((v / expected_iqr) ** 2 for v in x) / 8
    expected_slope = (-0.2 * expected_iqr) * variance / (variance + 0.5)
    assert isclose(model["scaler"]["iqr"], expected_iqr, abs_tol=1e-12)
    assert isclose(model["slope"], expected_slope, abs_tol=1e-12)
    assert isclose(model["intercept"], -0.002, abs_tol=1e-12)
    checks.append("analytic_ridge_all_outcomes_and_temporal_ESS")

    assert fitted(reversed(rows))["model_sha256"] == model["model_sha256"]
    assert fitted(rows)["model_sha256"] == model["model_sha256"]
    checks.append("determinism_and_order_invariance")

    future = [replace(r, origin_ms=deploy, feature_bar_start_ms=deploy-bar,
                      label_end_ms=deploy+horizon, relative_return=999.0, gross_return_3d=999.0)
              for r in rows[:8]]
    near = [replace(r, origin_ms=deploy-horizon, feature_bar_start_ms=deploy-horizon-bar,
                    label_end_ms=deploy, relative_return=999.0, gross_return_3d=999.0)
            for r in rows[:8]]
    assert fitted(rows + future + near)["model_sha256"] == model["model_sha256"]
    checks.append("future_and_unpurged_labels_cannot_change_scaler_or_model")

    # Later synthetic deployment with old training rows and reserved labels.
    later = api["RESERVED_END_MS"] + 3 * horizon
    old = [replace(r, origin_ms=later-(k+64)*horizon,
                   feature_bar_start_ms=later-(k+64)*horizon-bar,
                   label_end_ms=later-(k+63)*horizon)
           for k in range(60) for r in rows[:8]]
    reserved = [replace(r, origin_ms=api["RESERVED_START_MS"],
                        feature_bar_start_ms=api["RESERVED_START_MS"]-bar,
                        label_end_ms=api["RESERVED_START_MS"]+horizon,
                        relative_return=777.0, gross_return_3d=777.0) for r in rows[:8]]
    assert fitted(old, later)["fallback"] is None
    assert fitted(old + reserved, later)["model_sha256"] == fitted(old, later)["model_sha256"]
    checks.append("reserved_EMA_label_exclusion")

    momentum = fitted([replace(r, gross_return_3d=0.2*r.relative_return) for r in rows])
    assert momentum["slope"] == 0.0
    checks.append("nonpositive_slope_projection")

    constant = fitted([replace(r, relative_return=0.0) for r in rows])
    sparse = fitted(rows[:8*59])
    assert constant["fallback"] == "zero_IQR_NoTrade"
    assert sparse["fallback"] == "insufficient_temporal_support"
    assert forecast(constant, 0.0, feature_bar_start_ms=deploy-bar) is None
    assert forecast(sparse, 0.0, feature_bar_start_ms=deploy-bar) is None
    checks.append("NoTrade_for_zero_IQR_and_insufficient_origins")

    rejects(lambda: fitted(rows[1:]))
    rejects(lambda: fitted(rows + rows[:1]))
    rejects(lambda: fitted([replace(rows[0], relative_return=float("nan"))] + rows[1:]))
    rejects(lambda: fitted([replace(rows[0], label_end_ms=rows[0].label_end_ms+bar)] + rows[1:]))
    rejects(lambda: fitted([replace(rows[0], feature_bar_start_ms=rows[0].origin_ms)] + rows[1:]))
    checks.append("incomplete_duplicate_nonfinite_horizon_and_feature_time_rejection")

    assert forecast(model, -0.02, feature_bar_start_ms=deploy-bar) > forecast(
        model, 0.02, feature_bar_start_ms=deploy-bar)
    rejects(lambda: forecast(model, 0.0, feature_bar_start_ms=deploy))
    altered = dict(model, slope=-99.0)
    rejects(lambda: forecast(altered, 0.0, feature_bar_start_ms=deploy-bar))
    checks.append("forecast_direction_timestamp_and_SHA_guard")

    assert model["execution_allowed"] is False
    print(json.dumps({"passed": len(checks), "synthetic_only": True,
                      "rows": model["eligible_rows"], "origin_ess": model["origin_weight_ess"],
                      "model_sha256": model["model_sha256"],
                      "real_data_fits": 0, "execution_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
