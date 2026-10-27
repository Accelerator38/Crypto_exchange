"""Four-hour causal forecasts and bounded cash replays; pure, no I/O or runtime."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from . import compact_forecast_v1 as m
from . import compact_forecast_data_v1 as d
from . import position_policy_v1 as p

BAR_MS = d.BAR_MS
DAY_MS = m.DAY_MS
POLICIES = ("NoTrade", "Passive_V", "R0_EMA_daily", "R1_EMA_4h", "R2_entry", "R3_exit", "R4_size")


def samples(times, prices, features):
    times, prices, features = np.asarray(times), np.asarray(prices), np.asarray(features)
    if (times.ndim != 1 or len(times) < 100 or times.dtype.kind not in "iu"
            or np.any(np.diff(times) != BAR_MS) or np.any(times % BAR_MS)
            or prices.ndim != 2 or len(prices) != len(times)
            or features.shape != prices.shape + (6,)
            or not np.isfinite(prices).all() or np.any(prices <= 0)
            or not np.isfinite(features[48:]).all()):
        raise ValueError("complete completed-close six-feature panel required")
    return {"origins": times[48:-1], "available": times[49:], "x": features[48:-1],
            "y": prices[49:] / prices[48:-1] - 1}


def fit(bank, origin, contract):
    if type(origin) is not int or origin % BAR_MS:
        raise ValueError("aligned fit origin required")
    gap = contract["purge_bars"] + contract["additional_availability_bar"]
    cutoff = origin - gap * BAR_MS
    indices = np.flatnonzero(bank["available"] <= cutoff)
    if len(indices) < 50:
        raise ValueError("at least fifty mature origin clusters required")
    x = bank["x"][indices].reshape(-1, 6)
    y = bank["y"][indices].reshape(-1)
    scaler = m.fit_scaler(x, y)
    parameters = m.fit_linear(scaler.transform(x), scaler.target(y), np.ones(len(y)),
                              l2=contract["model"]["l2"])
    identity = {"fit_ms": origin, "cutoff_ms": cutoff, "horizon_bars": 1, "stride_bars": 1,
                "purge_bars": contract["purge_bars"], "additional_availability_bar": 1,
                "feature_definition_sha256": d.FEATURE_SHA256,
                "train_samples_sha256": m.array_identity(bank["origins"][indices], bank["available"][indices], x, y),
                "weighting": "uniform", "l2": contract["model"]["l2"],
                "fold_rule": "all mature expanding rows; no tuning/inner-fold selection"}
    payload = m.model_payload("linear6", parameters, scaler, weighting="uniform", seed=None,
                              epochs=None, training_identity=identity)
    payload["target_kind"] = "gross_completed_close_to_next_completed_close_4h"
    payload["sha256"] = m.identity({k: v for k, v in payload.items() if k != "sha256"})
    return {"fit_ms": origin, "cutoff_ms": cutoff, "model_ready_us": origin * 1000 + contract["model_ready_delay_us"],
            "model": payload, "train_origins": len(indices), "train_asset_rows": len(y),
            "latest_label_available_ms": int(bank["available"][indices[-1]]),
            "train_start_ms": int(bank["origins"][indices[0]]),
            "pooled_train_mean": float(y.mean()), "asset_train_mean": bank["y"][indices].mean(axis=0).tolist(),
            "positive_target_rows": int(np.sum(y > 0)), "negative_target_rows": int(np.sum(y < 0)),
            "scaler_fits": 1, "training_fits": 1, "parameter_count": 7}


def targets(policy, forecast, ema_weights, held, contract):
    if policy not in POLICIES:
        raise ValueError("unregistered policy")
    forecast, base, held = np.asarray(forecast, float), np.asarray(ema_weights, float), np.asarray(held, bool)
    if forecast.shape != base.shape or held.shape != base.shape or not np.isfinite(forecast).all():
        raise ValueError("finite aligned forecasts and positions required")
    if policy == "NoTrade":
        return np.zeros_like(base)
    if policy in ("Passive_V", "R0_EMA_daily", "R1_EMA_4h"):
        return base.copy()
    allowed = held | (forecast > contract["entry_strictly_above_bps"] / 10000)
    result = base * allowed
    if policy in ("R3_exit", "R4_size"):
        result *= forecast > contract["exit_at_or_below_bps"] / 10000
    if policy == "R4_size":
        result *= np.clip(forecast / (contract["entry_strictly_above_bps"] / 10000),
                          contract["sizing_minimum_fraction"], 1.)
    return result


def replay(data, vectors, covariances, rates, contract, policy, scenario, *, target_rule=None):
    """Long-only cost-inclusive delta-unit rebalancing and terminal liquidation."""
    if policy not in POLICIES or scenario not in contract["costs"]:
        raise ValueError("registered policy and costs required")
    times, prices = np.asarray(data["times"]), np.asarray(data["closes"], float)
    symbols, features = data["symbols"], np.asarray(data["features"])
    start, end = int(vectors[0]["asof_ms"]), int(vectors[-1]["target_ms"])
    expected = list(range(start, end, BAR_MS))
    if [v["asof_ms"] for v in vectors] != expected:
        raise ValueError("complete four-hour vectors required")
    if prices.shape != (len(times), len(symbols)) or np.any(np.diff(times) != BAR_MS):
        raise ValueError("complete aligned cash prices required")
    fee_bps, drag_bps = (contract["costs"][scenario][k] for k in ("fee_bps", "drag_bps"))
    fee_rate, drag_rate = fee_bps / 10000, drag_bps / 10000
    policy_sha = m.identity({"policy": policy, "contract": contract})
    vector_map = {v["asof_ms"]: v for v in vectors}
    funding_map = defaultdict(list)
    symbol_map = {s.replace("/", ""): i for i, s in enumerate(symbols)}
    seen = set()
    for rate in rates:
        t, symbol = int(rate["settlement_ms"]), rate["symbol"].replace("/", "")
        value = float(rate["rate_fraction"])
        if symbol not in symbol_map or not start <= t <= end:
            continue
        if (t, symbol) in seen or t % BAR_MS or not math.isfinite(value):
            raise ValueError("duplicate/misaligned funding or nonfinite rate")
        seen.add((t, symbol))
        funding_map[t].append((symbol_map[symbol], value))
    segment = np.flatnonzero((times >= start) & (times <= end))
    if len(segment) != len(expected) + 1 or times[segment[0]] != start or times[segment[-1]] != end:
        raise ValueError("complete replay period required")
    cash = float(contract["initial_capital"])
    quantity, basis = np.zeros(len(symbols)), np.zeros(len(symbols))
    fees = slippage = funding = gross = notional = 0.
    peak, max_dd = cash, 0.
    ledger, equity, decisions, episodes = [], [], [], []
    entry_times = [None] * len(symbols)
    per_asset = {s: {"gross_pnl": 0., "fees": 0., "slippage": 0., "funding": 0.} for s in symbols}
    active_sha = m.identity({"initial_no_position": True})

    def mark(t, px, phase):
        nonlocal peak, max_dd
        nav = cash + float(quantity @ px)
        if nav <= 0 or not math.isfinite(nav) or cash < -1e-7 or np.any(quantity < -1e-12):
            raise ValueError("positive NAV, no borrowing or shorts required")
        peak = max(peak, nav)
        dd = 1 - nav / peak
        max_dd = max(max_dd, dd)
        equity.append({"timestamp_ms": t, "phase": phase, "nav": nav, "cash": cash,
                       "exposure": float(quantity @ px) / nav, "drawdown": dd,
                       "asset_exposure": (quantity * px / nav).tolist()})

    for index in segment:
        t, px = int(times[index]), prices[index]
        mark(t, px, "before_funding_and_decision")
        for asset, rate in sorted(funding_map[t]):
            amount = -float(quantity[asset] * px[asset]) * rate
            cash += amount
            funding += amount
            per_asset[symbols[asset]]["funding"] += amount
            if quantity[asset]:
                ledger.append({"kind": "modeled_funding", "timestamp_us": t * 1000,
                               "symbol": symbols[asset], "quantity": float(quantity[asset]),
                               "rate": rate, "cash_delta": amount, "model_sha256": active_sha,
                               "policy_sha256": policy_sha})
            mark(t, px, "after_modeled_funding")
        should_decide = t != end and (policy != "R0_EMA_daily" or t % DAY_MS == 0)
        if not should_decide and t != end:
            mark(t, px, "calendar_bucket_end")
            continue
        nav = cash + float(quantity @ px)
        if t == end:
            weights = np.zeros(len(symbols))
            vector_sha = None
        else:
            vector = vector_map[t]
            if vector["available_us"] > t * 1000 + contract["signal_delay_us"] or vector["target_ms"] != t + BAR_MS:
                raise ValueError("available nonexpired one-bar vector required")
            covariance = covariances[t - t % DAY_MS]
            mask = np.ones(len(symbols), bool) if policy == "Passive_V" else features[index, :, 0] > 0
            base = p.bounded_weights(mask, covariance, budget=contract["risk"]["daily_volatility_target"])
            rule = targets if target_rule is None else target_rule
            weights = rule(policy, vector["expected_return_4h"], base, quantity > 1e-12, contract)
            target_risk = float(np.sqrt(max(0., weights @ covariance @ weights)))
            if target_risk > contract["risk"]["daily_volatility_target"]:
                # Removing a negatively correlated asset can increase risk.
                # Only reduce the surviving weights; never redistribute a rejected entry.
                weights *= contract["risk"]["daily_volatility_target"] / target_risk
            if (weights.max() > .1 + 1e-12 or weights.sum() > .8 + 1e-12
                    or float(np.sqrt(weights @ covariance @ weights)) > .005 + 1e-10):
                raise ValueError("target risk constraint violated")
            active_sha, vector_sha = vector["model_sha256"], vector["sha256"]
            decisions.append({"asof_ms": t, "model_sha256": active_sha, "vector_sha256": vector_sha,
                              "policy_sha256": policy_sha, "targets": weights.tolist(),
                              "risk_last_data_ms": t - t % DAY_MS - DAY_MS,
                              "signal_us": t * 1000 + contract["signal_delay_us"],
                              "modeled_fill_us": t * 1000 + contract["fill_delay_us"]})
        desired = nav * weights / (px * (1 + drag_rate) * (1 + fee_rate))
        deltas = desired - quantity
        operations = [(i, "SELL") for i in range(len(symbols)) if deltas[i] < -1e-12]
        operations += [(i, "BUY") for i in range(len(symbols)) if deltas[i] > 1e-12]
        for asset, side in operations:
            q, previous = abs(float(deltas[asset])), float(quantity[asset])
            execution = float(px[asset]) * (1 + (1 if side == "BUY" else -1) * drag_rate)
            fee = q * execution * fee_rate
            if side == "BUY":
                amount, realized = -q * execution - fee, 0.
                quantity[asset] += q
                basis[asset] += q * px[asset]
                if entry_times[asset] is None:
                    entry_times[asset] = t
            else:
                amount = q * execution - fee
                removed = basis[asset] * min(1., q / quantity[asset])
                realized = q * px[asset] - removed
                quantity[asset] -= q
                basis[asset] -= removed
                if quantity[asset] < 1e-12:
                    quantity[asset], basis[asset] = 0., 0.
                    episodes.append({"symbol": symbols[asset], "entry_ms": entry_times[asset],
                                     "exit_ms": t, "duration_hours": (t - entry_times[asset]) / 3600000})
                    entry_times[asset] = None
            slip = q * abs(execution - px[asset])
            cash += amount
            fees += fee
            slippage += slip
            gross += realized
            notional += q * px[asset]
            for key, value in (("gross_pnl", realized), ("fees", fee), ("slippage", slip)):
                per_asset[symbols[asset]][key] += float(value)
            ledger.append({"kind": "modeled_fill", "timestamp_us": t * 1000 + contract["fill_delay_us"],
                           "symbol": symbols[asset], "side": side, "quantity": q,
                           "quantity_before": previous, "quantity_after": float(quantity[asset]),
                           "reference_close": float(px[asset]), "price": execution,
                           "fee": fee, "slippage": slip, "gross_pnl": float(realized), "cash_delta": amount,
                           "model_sha256": active_sha, "vector_sha256": vector_sha, "policy_sha256": policy_sha})
            mark(t, px, "after_modeled_fill")
        mark(t, px, "calendar_bucket_end")
    error = cash - contract["initial_capital"] - gross + fees + slippage - funding
    if np.any(quantity) or abs(error) > 1e-7:
        raise ValueError("terminal position/capital reconciliation failed")
    calendar = {row["timestamp_ms"]: row for row in equity if row["phase"] == "calendar_bucket_end"}
    daily_nav = [float(contract["initial_capital"])] + [calendar[t]["nav"] for t in range(start + DAY_MS, end + 1, DAY_MS)]
    exposure = [calendar[t]["exposure"] for t in range(start, end, BAR_MS)]
    durations = [episode["duration_hours"] for episode in episodes]
    for values in per_asset.values():
        values["net_pnl"] = values["gross_pnl"] - values["fees"] - values["slippage"] + values["funding"]
    result = {"schema": "forecast4h_cash_replay/1", "policy": policy, "scenario": scenario,
              "policy_sha256": policy_sha, "start_ms": start, "end_ms": end,
              "execution_is_assumed": True, "observed_execution_qualified": False,
              "metrics": {"initial_capital": contract["initial_capital"], "final_capital": cash,
                          "net_return": cash / contract["initial_capital"] - 1, "max_drawdown": max_dd,
                          "gross_pnl": gross, "fees": fees, "slippage": slippage, "funding": funding,
                          "turnover_initial_capital": notional / contract["initial_capital"],
                          "mean_exposure": float(np.mean(exposure)), "maximum_exposure": float(max(exposure)),
                          "fills": sum(row["kind"] == "modeled_fill" for row in ledger),
                          "position_episodes": len(episodes), "decisions": len(decisions),
                          "mean_holding_hours": float(np.mean(durations)) if durations else 0.,
                          "median_holding_hours": float(np.median(durations)) if durations else 0.,
                          "reconciliation_error": error},
              "daily_log_returns": np.diff(np.log(daily_nav)), "equity": equity, "ledger": ledger,
              "decisions": decisions, "episodes": episodes, "per_asset": per_asset}
    return p.c.plain(result)


def bank(days, repetitions, block, seed):
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(repetitions, math.ceil(days / block)))
    return ((starts[:, :, None] + np.arange(block)) % days).reshape(repetitions, -1)[:, :days]


def lcb(values, indices, alpha):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()), "lcb": float(np.quantile(values[indices].mean(axis=1), alpha)),
            "alpha": alpha, "days": len(values)}


def direct_checks(contract):
    """Independent analytical cash example and future-perturbation invariants."""
    count = 0

    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"forecast4h check {count + 1} failed")
        count += 1

    ts = (np.arange(850, dtype=np.int64) + 1) * BAR_MS
    close = 100 * np.exp(np.arange(850)[:, None] * np.linspace(.0001, .0003, 8))
    feature = d.feature_panel(ts - BAR_MS, close)
    data = samples(ts, close, feature)
    origin = int(ts[700])
    first = fit(data, origin, contract)
    check(first["latest_label_available_ms"] <= origin - 8 * BAR_MS)
    check(first["parameter_count"] == 7)
    changed = close.copy()
    changed[701:] *= 1.5
    second = fit(samples(ts, changed, d.feature_panel(ts - BAR_MS, changed)), origin, contract)
    check(first["model"]["sha256"] == second["model"]["sha256"])
    check(np.allclose(m.predict(first["model"], feature[700]), m.predict(second["model"], feature[700]), atol=0, rtol=0))
    check(np.array_equal(data["available"] - data["origins"], np.full(len(data["origins"]), BAR_MS)))
    base, held = np.full(8, .1), np.ones(8, bool)
    check(np.all(targets("R2_entry", np.zeros(8), base, ~held, contract) == 0))
    check(np.all(targets("R2_entry", np.zeros(8), base, held, contract) == base))
    check(np.all(targets("R3_exit", np.full(8, -.002), base, held, contract) == 0))
    check(np.allclose(targets("R4_size", np.full(8, .0015), base, held, contract), .05))
    t0 = 1000 * DAY_MS
    toy_times = t0 + np.arange(7, dtype=np.int64) * BAR_MS
    toy_close = np.full((7, 8), 100.)
    toy_close[1:, 0] = 110.
    toy_features = np.zeros((7, 8, 6))
    toy_features[0, 0, 0] = 1.
    vectors = []
    for t in toy_times[:-1]:
        vectors.append({"asof_ms": int(t), "target_ms": int(t + BAR_MS),
                        "available_us": int(t * 1000 + 1000000), "expected_return_4h": [0.] * 8,
                        "model_sha256": "test_model", "sha256": "test_vector"})
    rates = [{"symbol": "A0", "settlement_ms": int(toy_times[1]), "rate_fraction": .0001}]
    toy = {"times": toy_times, "closes": toy_close, "features": toy_features,
           "symbols": [f"A{i}" for i in range(8)]}
    cov = {t0: np.eye(8) * .0001}
    run = replay(toy, vectors, cov, rates, contract, "R1_EMA_4h", "stress")
    q = 1000 / (100 * 1.0008 * 1.0006)
    expected_cash = 9000 + q * 110 * .9992 * .9994 - q * 110 * .0001
    check(abs(run["metrics"]["final_capital"] - expected_cash) < 1e-8)
    check(run["metrics"]["fills"] == 2)
    check(run["metrics"]["position_episodes"] == 1)
    check(run["metrics"]["mean_holding_hours"] == 4)
    check(abs(sum(a["net_pnl"] for a in run["per_asset"].values()) - (expected_cash - 10000)) < 1e-8)
    empty = replay(toy, vectors, cov, rates, contract, "R2_entry", "stress")
    check(empty["metrics"]["final_capital"] == 10000 and empty["metrics"]["fills"] == 0)
    check(bank(61, 10, 3, 123).shape == (10, 61))
    return {"checks_passed": count, "collector_runs": 0, "network_calls": 0}
