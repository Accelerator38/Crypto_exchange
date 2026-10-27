"""Offline reference-price research adapter. No exchange/runtime imports.

Observed funding *rates* are valued at a registered OHLCV open proxy, NOT an
observed exchange mark. Costs are sensitivities, not a release execution model.
"""
from __future__ import annotations

from decimal import Decimal
import math

from . import unified_measurement_v1 as measurement

DAY_MS = 86_400_000
ARMS = ("linear6_uniform", "linear6_half_life_90_anchor20",
        "mlp8_uniform", "mlp8_half_life_90_anchor20")
BASELINES = ("NoTrade", "EMA48_long_only")
MARK_KIND = "registered_ohlcv_open_reference_proxy_not_observed_exchange_mark"


def target_weights(predictions, symbols, threshold_bps, weight="0.10"):
    """Fixed sleeves: losses/weak predictions abstain; no winner redistribution."""
    if len(predictions) != len(symbols) or len(set(symbols)) != len(symbols):
        raise ValueError("ordered prediction universe mismatch")
    threshold = Decimal(str(threshold_bps)) / Decimal(10000)
    sleeve = Decimal(str(weight))
    if not threshold.is_finite() or threshold < 0 or not sleeve.is_finite() or sleeve <= 0:
        raise ValueError("finite positive sleeve and nonnegative threshold required")
    if sleeve * len(symbols) > Decimal("0.8"):
        raise ValueError("at least 20 percent target cash reserve required")
    result = {}
    for symbol, prediction in zip(symbols, predictions):
        value = Decimal(str(prediction))
        if not value.is_finite():
            raise ValueError("nonfinite forecast")
        result[symbol] = str(sleeve if value > threshold else Decimal(0))
    return result


def funding_input(rate_events, price_bars, *, start_ms, end_ms, symbols, evidence_sha256):
    """Fail closed on any missing rate/reference; never impute zero funding."""
    prices = {row["timestamp"]: row["prices"] for row in price_bars}
    expected = {(time, symbol) for time in range(start_ms, end_ms, 8 * 3_600_000)
                for symbol in symbols}
    selected = {}
    for rate in rate_events:
        time = int(rate["settlement_ms"])
        symbol = rate["symbol"].replace("USDT", "/USDT")
        if start_ms <= time < end_ms:
            key = (time, symbol)
            if key in selected:
                raise ValueError("duplicate observed funding settlement")
            fraction = Decimal(str(rate["rate_fraction"]))
            bps = Decimal(str(rate["rate_bps"]))
            if not fraction.is_finite() or not bps.is_finite() or fraction * 10000 != bps:
                raise ValueError("funding units or finiteness mismatch")
            selected[key] = bps
    if set(selected) != expected:
        raise ValueError("complete observed 8h rate coverage required")
    events = []
    for (time, symbol), bps in sorted(selected.items()):
        # All downloaded settlements align to the registered 4h reference grid.
        if time > price_bars[-1]["timestamp"]:
            # The portfolio is already flat; still retain full rate coverage.
            continue
        if time not in prices or symbol not in prices[time]:
            raise ValueError("settlement valuation reference unavailable")
        events.append({"timestamp": time, "symbol": symbol,
                       "mark_price": prices[time][symbol], "signed_rate_bps": str(bps)})
    return {"complete": True, "coverage_start_ms": start_ms, "coverage_end_ms": end_ms,
            "symbols": list(symbols), "provenance": "observed",
            "provenance_scope": "observed_funding_rates_only",
            "settlement_mark_kind": MARK_KIND, "observed_exchange_mark_verified": False,
            "historical_settlement_schedule_independently_verified": False,
            "post_terminal_rate_count": len(selected) - len(events),
            "post_terminal_funding_cashflow": "zero_because_flat_not_missing_rate_imputation",
            "rate_evidence_sha256": evidence_sha256, "events": events,
            "evidence_sha256": measurement.identity({"rates_sha256": evidence_sha256,
                "mark_kind": MARK_KIND, "price_sha256": measurement.identity(price_bars)})}


def decisions_for_policy(origins, *, policy, symbols, threshold_bps, weight="0.10",
                         ema_features=None, registry_sha256):
    """Use only origin forecasts/features; never outcome or future rates."""
    decisions = []
    for origin in origins:
        if policy in ARMS:
            arm = origin["arms"][policy]
            targets = target_weights(arm["prediction"], symbols, threshold_bps, weight)
            model_sha = measurement.identity([item["sha256"] for item in arm["models"]])
        elif policy == "EMA48_long_only":
            if ema_features is None:
                raise ValueError("causal EMA feature panel required")
            values = ema_features[int(origin["origin_ms"])]
            targets = target_weights([1 if value > 0 else 0 for value in values], symbols, 0, weight)
            model_sha = measurement.identity({"baseline": policy, "feature": "distance_ema48>0"})
        elif policy == "NoTrade":
            targets = {symbol: "0" for symbol in symbols}
            model_sha = measurement.identity({"baseline": policy})
        else:
            raise ValueError("unregistered policy")
        decisions.append(measurement.make_decision(
            origin_start_ms=origin["origin_ms"], decision_asof_ms=origin["origin_ms"],
            targets=targets, model_sha256=model_sha, scaler_sha256=origin["scaler_sha256"],
            feature_prefix_sha256=origin["feature_prefix_sha256"],
            training_identity_sha256=measurement.identity({
                "training_identity": origin["training_identity"], "registry_sha256": registry_sha256,
                "policy": policy, "threshold_bps": threshold_bps, "weight": weight})))
    return decisions


def rank_calibration(rows, primary_budget_percent=10):
    """One common threshold chosen on LINEAR UNIFORM only, before benchmark."""
    candidates = [row for row in rows if row["policy"] == "linear6_uniform"]
    if not candidates:
        raise ValueError("linear calibration candidates absent")
    limit = primary_budget_percent / 100
    for row in candidates:
        metrics = row["metrics"]
        row["risk_valid"] = all(float(metrics[k]["max_drawdown"]) <= limit for k in ("normal", "stress"))
        row["worst_log_growth"] = min(float(metrics[k]["net_log_growth"]) for k in ("normal", "stress"))
    ranked = sorted(candidates, key=lambda r: (r["risk_valid"], r["worst_log_growth"],
                                              -r["threshold_bps"]), reverse=True)
    winner = ranked[0]
    return {"threshold_bps": winner["threshold_bps"], "selected_on": "pre_august_linear6_uniform",
            "risk_valid": winner["risk_valid"], "worst_log_growth": winner["worst_log_growth"],
            "beats_NoTrade_point_estimate": winner["risk_valid"] and winner["worst_log_growth"] > 0,
            "ranked_candidates": ranked, "benchmark_used_for_selection": False}


def fit_calibration_linear(samples, origin_ms):
    """One fixed 7-parameter fit, rather than wasting MLP fits on a threshold."""
    import numpy as np
    from . import compact_forecast_v1 as model
    from .compact_forecast_data_v1 import eligible_split, FEATURE_SHA256
    split = eligible_split(samples, origin_ms)
    train = split["train"]
    tx = samples["x"][train].reshape(-1, 6)
    ty = samples["y"][train].reshape(-1)
    decision_x = samples["x"][split["decision"]]
    scaler = model.fit_scaler(tx, ty)
    ow = model.temporal_weights(samples["available"][train], split["cutoff_ms"], "uniform")
    weights = np.repeat(ow / samples["x"].shape[1], samples["x"].shape[1])
    weighting = {"origin_ess": model.origin_ess(ow), "weights_sha256": model.array_identity(ow),
                 "positive_weight_mass": float(weights[ty > 0].sum()),
                 "negative_weight_mass": float(weights[ty < 0].sum())}
    training = {"train_samples_sha256": model.array_identity(samples["origins"][train], samples["available"][train], tx, ty),
                "fold_sha256": model.array_identity(samples["origins"][split["inner_train"]],
                                                    samples["origins"][split["inner_validation"]]),
                "train_end_ms": split["train_end_ms"], "cutoff_ms": split["cutoff_ms"],
                "feature_definition_sha256": FEATURE_SHA256, "l2": .01}
    params = model.fit_linear(scaler.transform(tx), scaler.target(ty), weights)
    payload = model.model_payload("linear6", params, scaler, weighting="uniform", seed=None,
                                  epochs=None, training_identity=training | weighting)
    result = {"schema": "compact_calibration_linear_forecast/1", "origin_ms": origin_ms,
              "training_identity": training, "scaler_sha256": scaler.sha256,
              "feature_prefix_sha256": model.array_identity(samples["origins"][:split["decision"] + 1],
                                                            samples["x"][:split["decision"] + 1]),
              "train_origin_count": len(train), "model_fit_count": 1, "scaler_fit_count": 1,
              "gradient_steps": 0, "forecast_only": True,
              "arms": {"linear6_uniform": weighting | {"models": [payload],
                "prediction": model.predict(payload, decision_x).tolist(), "stored_parameter_count": 7}}}
    result["sha256"] = model.identity(result)
    return result


def validate_origin(origin, expected_ms, *, expected_arms=ARMS):
    if (origin["origin_ms"] != expected_ms
            or origin["training_identity"]["cutoff_ms"] != expected_ms - 8 * measurement.BAR_MS):
        raise ValueError("origin or seven-bar purge cutoff mismatch")
    unhashed = {k: v for k, v in origin.items() if k != "sha256"}
    if origin["sha256"] != measurement.identity(unhashed):
        raise ValueError("frozen origin content hash mismatch")
    if set(origin["arms"]) != set(expected_arms):
        raise ValueError("registered forecast arms required")
    for arm in origin["arms"].values():
        if not all(math.isfinite(float(x)) for x in arm["prediction"]):
            raise ValueError("nonfinite frozen prediction")
        for model in arm["models"]:
            if model["sha256"] != measurement.identity({k: v for k, v in model.items() if k != "sha256"}):
                raise ValueError("model identity mismatch")
