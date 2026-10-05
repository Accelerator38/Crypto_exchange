"""Synthetic-only builder checks; no market files, collector, network or pytest."""

from dataclasses import replace
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import runpy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "src/exia/genetic_primus/market_relative_reversal_builder_v1.py"
    fitter_source = root / "src/exia/genetic_primus/market_relative_reversal_linear_v1.py"
    api, fitter = runpy.run_path(str(source)), runpy.run_path(str(fitter_source))
    Candle, bar, horizon, symbols = api["Candle"], api["BAR_MS"], api["HORIZON_MS"], api["UNIVERSE"]
    assert symbols == fitter["UNIVERSE"] and horizon == fitter["HORIZON_MS"]
    draft = json.loads((root / "configs/genetic_primus_market_relative_reversal_draft_v1.json").read_text())
    assert list(symbols) == draft["universe"] and api["LOOKBACK_BARS"] == draft["features"]["lookback_bars"]
    builder_sha = sha256(source.read_bytes()).hexdigest()
    origin = 1780272000000  # Fixture anchor only; no historical prices loaded.
    candles = []
    for k in range(-13, 18):
        for i, symbol in enumerate(symbols):
            close = 100 + 2*i + (k+13)*(i+1)
            o = 200 + 10*i if k == 0 else close - 0.5
            candles.append(Candle(symbol, origin+k*bar, o, max(o, close)+1,
                                  min(o, close)-1, close, 100+i))

    def features(data, as_of=origin):
        return api["build_features"](data, origin_ms=origin, as_of_ms=as_of,
                                     implementation_sha256=builder_sha)

    def training(data, as_of=origin+horizon, at=origin):
        return api["build_training_origin"](data, origin_ms=at, as_of_ms=as_of,
                                            implementation_sha256=builder_sha)

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("unsafe builder input accepted")

    checks = []
    bundle = features(candles)
    expected_returns = {s: (100+2*i+12*(i+1))/(100+2*i)-1 for i, s in enumerate(symbols)}
    expected_relative = fitter["market_relative_features"](expected_returns)
    assert bundle["source_bars"] == 104
    assert all(isclose(bundle["relative_features"][s], expected_relative[s], abs_tol=1e-14) for s in symbols)
    checks.append("manual_close_lag12_and_shared_feature_semantics")

    # Invalid future values must not even be inspected by decision features.
    changed_future = [replace(c, close=float("nan")) if c.open_ms >= origin else c for c in candles]
    past_only = [c for c in candles if c.open_ms < origin]
    assert features(changed_future) == bundle == features(past_only)
    assert features(candles, origin+horizon) == bundle
    checks.append("decision_features_ignore_future_prices_and_asof_extension")

    result = training(candles)
    assert result["features"] == bundle
    for i, row in enumerate(result["observations"]):
        expected = (100+2*i+30*(i+1))/(200+10*i)-1
        assert isclose(row["gross_return_3d"], expected, abs_tol=1e-14)
    assert result["label_end_ms"] == origin+horizon
    checks.append("manual_open_to_last_close_complete_three_day_label")

    outside = Candle(symbols[0], origin+horizon, 1, 1, 1, float("nan"), 1)
    assert training(reversed(candles))["training_origin_sha256"] == result["training_origin_sha256"]
    assert training(candles+[outside])["training_origin_sha256"] == result["training_origin_sha256"]
    altered = list(candles)
    altered[0] = replace(altered[0], volume=12345)
    assert training(altered)["input_ohlcv_sha256"] != result["input_ohlcv_sha256"]
    checks.append("order_invariance_outside_window_and_input_hash_change")

    rejects(lambda: training(candles, origin+horizon-1))
    rejects(lambda: features(candles, origin-1))
    reserved = api["RESERVED_START_MS"]
    rejects(lambda: training([], reserved+horizon, reserved))
    rejects(lambda: training([], reserved+bar, reserved-horizon+bar))
    checks.append("unavailable_and_overlapping_reserved_labels_rejected")

    rejects(lambda: features([c for c in candles if c.symbol != symbols[0]]))
    rejects(lambda: features([c for c in candles if c.open_ms != origin-7*bar]))
    rejects(lambda: training([c for c in candles if c.open_ms != origin+5*bar]))
    rejects(lambda: features(candles+[candles[0]]))
    rejects(lambda: features([replace(candles[0], high=0)]+candles[1:]))
    rejects(lambda: features([replace(candles[0], volume=-1)]+candles[1:]))
    rejects(lambda: features([replace(candles[0], close=float("inf"))]+candles[1:]))
    rejects(lambda: api["build_features"](candles, origin_ms=origin+1, as_of_ms=origin+1,
                                           implementation_sha256=builder_sha))
    checks.append("missing_asset_gap_duplicate_invalid_OHLCV_and_grid_rejected")

    changed_label = list(candles)
    last = next(i for i, c in enumerate(changed_label) if c.open_ms == origin+horizon-bar)
    changed_label[last] = replace(changed_label[last], close=changed_label[last].close+0.5)
    changed = training(changed_label)
    assert changed["features"] == result["features"]
    assert changed["observations_sha256"] != result["observations_sha256"]
    checks.append("label_change_cannot_change_decision_features")

    # Integration fixture: 60 timestamp-shifted synthetic copies, not market evidence.
    deploy = origin + 2*horizon
    observations = [fitter["Observation"](**dict(row, origin_ms=origin-k*horizon,
                    feature_bar_start_ms=origin-k*horizon-bar,
                    label_end_ms=origin-(k-1)*horizon))
                    for k in range(60) for row in result["observations"]]
    model = fitter["fit_origin"](observations, deploy_origin_ms=deploy,
                                 implementation_sha256=sha256(fitter_source.read_bytes()).hexdigest())
    assert model["eligible_rows"] == 480 and model["distinct_training_origins"] == 60
    assert model["fallback"] is None and model["execution_allowed"] is False
    checks.append("Observation_schema_and_fitter_integration_synthetic_only")
    print(json.dumps({"passed": len(checks), "synthetic_only": True,
                      "feature_source_bars": 104, "training_source_bars": 248,
                      "training_origin_sha256": result["training_origin_sha256"],
                      "market_files_read": 0, "execution_allowed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
