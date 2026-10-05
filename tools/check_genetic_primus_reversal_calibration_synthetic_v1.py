"""Pure hypothetical pilot-statistics checks; no OHLCV, fits or network."""

from copy import deepcopy
from math import isclose, sqrt
from pathlib import Path
import json
import runpy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    api = runpy.run_path(str(root/"src/exia/genetic_primus/reversal_calibration_v1.py"))
    folds = api["calibration_folds"]()
    origins = [f["origin_ms"] for f in folds["folds"]]
    summarize = api["summarize_calibration"]
    checks = []

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("invalid calibration input accepted")

    assert len(origins) == 20 and origins[-1]+api["HORIZON_MS"] == origins[0]+60*api["DAY_MS"]
    for f in folds["folds"]:
        assert f["train_cutoff_ms"] == f["origin_ms"]-7*api["BAR_MS"]
        assert f["last_train_origin_ms"]+api["HORIZON_MS"] <= f["train_cutoff_ms"]
        assert f["first_train_origin_ms"] >= f["origin_ms"]-730*api["DAY_MS"]
        assert (f["origin_ms"]-f["first_train_origin_ms"]) % api["HORIZON_MS"] == 0
    assert api["calibration_folds"]() == folds
    checks.append("fixed_origin_grid_purge_rolling_window_and_fold_digest")

    sigma = api["hac_long_run_sigma"]
    assert isclose(sigma([.01,-.01],lag=0),.01,abs_tol=1e-12)
    assert isclose(sigma([.01,-.01],lag=1),sqrt(.00005),abs_tol=1e-12)
    checks.append("independent_two_point_Bartlett_variance_calculation")

    x = [.002,-.001,0.0]*20
    args = dict(daily_returns={"normal":x,"stress":[v-.0001 for v in x]},
                origins=origins,signal_origins=[i%2==0 for i in range(20)])
    before = deepcopy(args)
    result = summarize(**args)
    assert args == before and summarize(**args) == result
    assert result["distinct_signal_origins"] == 10
    assert result["scenarios"]["normal"]["calendar_days"] == 60
    assert result["scenarios"]["normal"]["zero_return_days"] == 20
    for scene in result["scenarios"].values():
        assert scene["planning_sigma_envelope"] == max(scene["hac_sigma_by_lag"].values())
    assert not result["variance_envelope_is_confidence_bound"]
    assert not result["signal_fraction_is_IID_probability_or_guarantee"]
    checks.append("calendar_zeros_losses_temporal_signal_count_and_predeclared_envelope")

    flat = summarize(daily_returns={"normal":[0.0]*60,"stress":[0.0]*60},
                     origins=origins,signal_origins=[False]*20)
    assert flat["status"] == "NO_ACTIVE_SIGNAL_KEEP_NOTRADE"
    losing = summarize(**dict(args,daily_returns={"normal":[-.001]*60,"stress":[-.002]*60}))
    assert losing["status"] == "NONPOSITIVE_DEVELOPMENT_NET_KEEP_NOTRADE_NO_RETUNING"
    constant = summarize(**dict(args,daily_returns={"normal":[.001]*60,"stress":[.001]*60}))
    assert constant["status"] == "DEGENERATE_VARIANCE_NO_POWER_SIZE"
    checks.append("no_activity_negative_net_and_zero_variance_do_not_force_trading")

    rejects(lambda: summarize(**dict(args,daily_returns={"normal":x[:59],"stress":x})))
    rejects(lambda: summarize(**dict(args,signal_origins=[1]*20)))
    rejects(lambda: summarize(**dict(args,origins=[t+1 for t in origins])))
    rejects(lambda: summarize(**dict(args,daily_returns={"normal":[float('nan')]*60,"stress":x})))
    rejects(lambda: summarize(**dict(args,signal_origins=[False]*20)))
    rejects(lambda: sigma(x,lag=60))
    checks.append("missing_nonfinite_misaligned_and_inconsistent_inputs_rejected")
    assert not result["market_power_calibrated"] and not result["promotion_authority"]
    assert not result["registration_allowed"] and not result["execution_allowed"]
    print(json.dumps({"passed":len(checks),"hypothetical_only":True,
                      "folds_sha256":folds["folds_sha256"],"origins":20,
                      "market_files_decoded":0,"model_fits":0,"execution_allowed":False},sort_keys=True))


if __name__ == "__main__":
    main()
