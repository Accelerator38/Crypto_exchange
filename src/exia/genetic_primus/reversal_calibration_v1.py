"""Fixed revealed-pilot folds and descriptive statistics; no data loaders."""

from hashlib import sha256
import json
from math import fsum, isfinite, prod, sqrt
from typing import Mapping, Sequence

BAR_MS = 14_400_000
DAY_MS = 6 * BAR_MS
HORIZON_MS = 3 * DAY_MS
PILOT_START_MS = 1780272000000
ORIGIN_COUNT = 20
HAC_LAGS = (0, 4, 9, 14)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def calibration_folds() -> dict:
    folds = [{"origin_ms": PILOT_START_MS+i*HORIZON_MS,
              "deployment_end_exclusive_ms": PILOT_START_MS+(i+1)*HORIZON_MS,
              "train_cutoff_ms": PILOT_START_MS+i*HORIZON_MS-7*BAR_MS,
              "first_train_origin_ms": PILOT_START_MS+i*HORIZON_MS-243*HORIZON_MS,
              "last_train_origin_ms": PILOT_START_MS+i*HORIZON_MS-2*HORIZON_MS}
             for i in range(ORIGIN_COUNT)]
    return {"folds": folds, "folds_sha256": _digest(folds)}


def _returns(values: Sequence) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("daily return sequence required")
    if any(type(v) not in (int, float) or not isfinite(v) or v <= -1 for v in values):
        raise ValueError("finite calendar returns above -1 required")
    return [float(v) for v in values]


def hac_long_run_sigma(values: Sequence, *, lag: int) -> float:
    """Biased autocovariances and Bartlett weights. Descriptive, not a bound."""
    x = _returns(values)
    n = len(x)
    if n < 2 or type(lag) is not int or not 0 <= lag < n:
        raise ValueError("valid bounded HAC lag required")
    mean = fsum(x)/n
    centered = [v-mean for v in x]
    variance = fsum(v*v for v in centered)/n
    variance += 2*fsum((1-k/(lag+1))*fsum(centered[t]*centered[t-k]
                        for t in range(k,n))/n for k in range(1,lag+1))
    if not isfinite(variance) or variance < -1e-12:
        raise ValueError("invalid HAC variance")
    return sqrt(max(0.0, variance))


def summarize_calibration(*, daily_returns: Mapping[str, Sequence],
                          origins: Sequence[int], signal_origins: Sequence[bool]) -> dict:
    expected = [f["origin_ms"] for f in calibration_folds()["folds"]]
    if list(origins) != expected or len(signal_origins) != ORIGIN_COUNT:
        raise ValueError("fixed twenty-origin pilot required")
    if any(type(v) is not bool for v in signal_origins):
        raise ValueError("one Boolean signal flag per temporal origin required")
    if not isinstance(daily_returns, Mapping) or set(daily_returns) != {"normal", "stress"}:
        raise ValueError("normal and stress daily returns required")
    values = {name: _returns(v) for name,v in daily_returns.items()}
    if any(len(v) != 60 for v in values.values()):
        raise ValueError("sixty complete calendar days including inactive days required")
    active = sum(signal_origins)
    if not active and any(v != 0 for scene in values.values() for v in scene):
        raise ValueError("nonzero PnL inconsistent with all-cash initial state and no signals")
    scenes = {}
    for name,x in values.items():
        sigmas = {str(lag): hac_long_run_sigma(x,lag=lag) for lag in HAC_LAGS}
        net = prod(1+v for v in x)-1
        if not isfinite(net):
            raise ValueError("nonfinite compounded net")
        scenes[name] = {"calendar_days": len(x), "mean_net_daily": fsum(x)/len(x),
                       "compounded_net": net, "hac_sigma_by_lag": sigmas,
                       "planning_sigma_envelope": max(sigmas.values()),
                       "zero_return_days": sum(v == 0 for v in x)}
    if not active:
        status = "NO_ACTIVE_SIGNAL_KEEP_NOTRADE"
    elif any(s["compounded_net"] <= 0 for s in scenes.values()):
        status = "NONPOSITIVE_DEVELOPMENT_NET_KEEP_NOTRADE_NO_RETUNING"
    elif any(s["planning_sigma_envelope"] == 0 for s in scenes.values()):
        status = "DEGENERATE_VARIANCE_NO_POWER_SIZE"
    else:
        status = "DESCRIPTIVE_CALIBRATION_ONLY_NOT_VALIDATION"
    return {"schema": "exia.genetic_primus.reversal_calibration_summary/1",
            "evidence_class": "revealed_development_calibration_not_independent_validation",
            "status": status, "origins": ORIGIN_COUNT, "distinct_signal_origins": active,
            "descriptive_signal_fraction": active/ORIGIN_COUNT, "scenarios": scenes,
            "hac_lags_predeclared": list(HAC_LAGS),
            "variance_envelope_is_confidence_bound": False,
            "signal_fraction_is_IID_probability_or_guarantee": False,
            "folds_sha256": calibration_folds()["folds_sha256"],
            "input_sha256": _digest({"origins": list(origins), "signals": list(signal_origins),
                                    "daily_returns": values}),
            "market_power_calibrated": False, "registration_allowed": False,
            "promotion_authority": False, "execution_allowed": False}
