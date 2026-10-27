"""Causal EMA unit retention with explicit risk trims and an offline cash ledger."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from . import compact_forecast_v1 as m
from . import compact_forecast_data_v1 as d
from . import position_policy_v1 as p

BAR_MS, DAY_MS = d.BAR_MS, m.DAY_MS


def modeled_cost(current, desired, prices, fee_bps, drag_bps):
    delta = np.asarray(desired) - np.asarray(current)
    fee, drag = fee_bps / 10000, drag_bps / 10000
    return float(np.sum(abs(delta) * prices * (drag + fee * (1 + np.sign(delta) * drag))))


def position_risk(quantity, prices, nav, covariance):
    if not math.isfinite(nav) or nav <= 0:
        raise ValueError("positive projected NAV required")
    weights = np.asarray(quantity) * prices / nav
    return {"asset_max": float(weights.max()), "gross": float(weights.sum()),
            "daily_volatility": float(np.sqrt(max(0., weights @ covariance @ weights)))}


def desired_units(current, prices, nav, ema, base_weights, covariance, contract, scenario):
    """No iterative solver, time waiting or implicit increase of retained units."""
    current, prices = np.asarray(current, float), np.asarray(prices, float)
    ema, base_weights = np.asarray(ema, bool), np.asarray(base_weights, float)
    covariance = np.asarray(covariance, float)
    n = len(current)
    if (prices.shape != (n,) or ema.shape != (n,) or base_weights.shape != (n,)
            or covariance.shape != (n, n) or not np.isfinite(current).all() or np.any(current < 0)
            or not np.isfinite(prices).all() or np.any(prices <= 0) or not np.isfinite(base_weights).all()
            or np.any(base_weights < 0) or not np.isfinite(covariance).all()):
        raise ValueError("aligned finite long-only units/prices/weights/covariance required")
    costs, risk = contract["costs"][scenario], contract["risk"]
    fee, drag = costs["fee_bps"] / 10000, costs["drag_bps"] / 10000
    held = current > 1e-12
    entries = ema & ~held
    desired = np.where(ema, current, 0.)
    desired[entries] = nav * base_weights[entries] / (prices[entries] * (1 + drag) * (1 + fee))
    projected = nav - modeled_cost(current, desired, prices, costs["fee_bps"], costs["drag_bps"])
    initial = position_risk(desired, prices, projected, covariance)
    tolerance = risk["numerical_tolerance"]
    limits = {"asset_max": risk["asset_cap"], "gross": risk["gross_cap"], "daily_volatility": risk["daily_volatility_target"]}
    triggers = [name for name, limit in limits.items() if initial[name] > limit + tolerance]
    lower_bound = None
    if triggers:
        # All subsequent steps only decrease the tentative units. This bound
        # therefore remains valid for the exact final delta costs, including sells.
        max_cost = drag + fee * (1 + drag)
        lower_bound = nav - max_cost * float((current + desired) @ prices)
        if lower_bound <= 0:
            raise ValueError("insufficient conservative post-cost NAV")
        desired = np.minimum(desired, risk["asset_cap"] * lower_bound / prices)
        value = float(desired @ prices)
        if value > risk["gross_cap"] * lower_bound:
            desired *= risk["gross_cap"] * lower_bound / value
        volatility = position_risk(desired, prices, lower_bound, covariance)["daily_volatility"]
        if volatility > risk["daily_volatility_target"]:
            desired *= risk["daily_volatility_target"] / volatility
    projected = nav - modeled_cost(current, desired, prices, costs["fee_bps"], costs["drag_bps"])
    final = position_risk(desired, prices, projected, covariance)
    if any(final[name] > limit + tolerance for name, limit in limits.items()):
        raise ValueError("post-cost risk guard failed")
    reasons = []
    for asset in range(n):
        if held[asset] and not ema[asset]:
            reason = "signal_exit"
        elif entries[asset] and desired[asset] > 1e-12:
            reason = "entry"
        elif desired[asset] < current[asset] - 1e-12:
            reason = "risk_trim"
        else:
            reason = "hold_units"
        reasons.append(reason)
    return {"quantity": desired, "reasons": reasons, "constraint_triggers": triggers,
            "projected_post_cost_nav": projected, "projected_post_cost_risk": final,
            "nav_lower_bound": lower_bound,
            "retained_assets": int(np.sum(held & ema & (abs(desired - current) <= 1e-12))),
            "risk_trim_assets": reasons.count("risk_trim")}


def replay(data, vectors, covariances, rates, contract, scenario, *, retain_units):
    """One cash engine for parity mode and the registered retain-units policy."""
    times, prices, features = (np.asarray(data[k]) for k in ("times", "closes", "features"))
    symbols = data["symbols"]
    start, end = int(vectors[0]["asof_ms"]), int(vectors[-1]["target_ms"])
    if [row["asof_ms"] for row in vectors] != list(range(start, end, BAR_MS)):
        raise ValueError("complete shared four-hour clock required")
    if prices.shape != (len(times), len(symbols)) or np.any(np.diff(times) != BAR_MS) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("complete positive cash prices required")
    segment = np.flatnonzero((times >= start) & (times <= end))
    if len(segment) != len(vectors) + 1 or times[segment[0]] != start or times[segment[-1]] != end:
        raise ValueError("complete evaluation endpoints required")
    costs = contract["costs"][scenario]
    fee_rate, drag_rate = costs["fee_bps"] / 10000, costs["drag_bps"] / 10000
    policy = "U1_hold_units" if retain_units else "R1_parity_control"
    policy_sha = m.identity({"policy": policy, "contract": contract})
    model_sha = m.identity({"agent": "EMA48", "feature_definition_sha256": d.FEATURE_SHA256, "trained_parameters": 0})
    vector_map = {row["asof_ms"]: row for row in vectors}
    symbol_map = {symbol.replace("/", ""): i for i, symbol in enumerate(symbols)}
    funding_map, seen = defaultdict(list), set()
    for row in rates:
        t, symbol, rate = int(row["settlement_ms"]), row["symbol"].replace("/", ""), float(row["rate_fraction"])
        if symbol not in symbol_map or not start <= t <= end:
            continue
        if (t, symbol) in seen or t % BAR_MS or not math.isfinite(rate):
            raise ValueError("invalid/duplicate funding row")
        seen.add((t, symbol))
        funding_map[t].append((symbol_map[symbol], rate))
    cash = float(contract["initial_capital"])
    quantity, basis = np.zeros(len(symbols)), np.zeros(len(symbols))
    fees = slippage = funding = gross = notional = 0.
    peak, maximum_dd = cash, 0.
    ledger, equity, decisions, episodes = [], [], [], []
    entry_times = [None] * len(symbols)
    per_asset = {symbol: {"gross_pnl": 0., "fees": 0., "slippage": 0., "funding": 0.} for symbol in symbols}
    reason_totals = defaultdict(lambda: {"fills": 0, "notional": 0., "fees": 0., "slippage": 0.})

    def mark(t, px, phase):
        nonlocal peak, maximum_dd
        nav = cash + float(quantity @ px)
        if nav <= 0 or not math.isfinite(nav) or cash < -1e-7 or np.any(quantity < -1e-12):
            raise ValueError("no borrowing/shorting and positive NAV required")
        peak = max(peak, nav)
        drawdown = 1 - nav / peak
        maximum_dd = max(maximum_dd, drawdown)
        equity.append({"timestamp_ms": t, "phase": phase, "nav": nav, "cash": cash,
                       "exposure": float(quantity @ px) / nav, "drawdown": drawdown,
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
                ledger.append({"kind": "modeled_funding", "timestamp_us": t * 1000, "symbol": symbols[asset],
                               "quantity": float(quantity[asset]), "rate": rate, "cash_delta": amount,
                               "model_sha256": model_sha, "policy_sha256": policy_sha})
            mark(t, px, "after_modeled_funding")
        nav = cash + float(quantity @ px)
        if t == end:
            desired, reasons, triggers = np.zeros(len(symbols)), ["terminal"] * len(symbols), []
            vector_sha = None
        else:
            vector = vector_map[t]
            if vector["available_us"] > t * 1000 + contract["signal_delay_us"]:
                raise ValueError("future source vector")
            covariance = covariances[t - t % DAY_MS]
            ema = features[index, :, 0] > 0
            weights = p.bounded_weights(ema, covariance, budget=contract["risk"]["daily_volatility_target"])
            volatility = float(np.sqrt(max(0., weights @ covariance @ weights)))
            if volatility > contract["risk"]["daily_volatility_target"]:
                weights *= contract["risk"]["daily_volatility_target"] / volatility
            if retain_units:
                proposal = desired_units(quantity, px, nav, ema, weights, covariance, contract, scenario)
                desired, reasons, triggers = proposal["quantity"], proposal["reasons"], proposal["constraint_triggers"]
            else:
                desired = nav * weights / (px * (1 + drag_rate) * (1 + fee_rate))
                reasons, triggers = ["target_rebalance"] * len(symbols), []
                projected_nav = nav - modeled_cost(quantity, desired, px, costs["fee_bps"], costs["drag_bps"])
                proposal = {"projected_post_cost_risk": position_risk(desired, px, projected_nav, covariance),
                            "projected_post_cost_nav": projected_nav, "retained_assets": 0, "risk_trim_assets": 0}
            vector_sha = vector["sha256"]
            decisions.append({"asof_ms": t, "model_sha256": model_sha, "forecast_model_sha256": vector["model_sha256"],
                "forecast_used_for_trading": False, "vector_sha256": vector_sha, "policy_sha256": policy_sha,
                "risk_last_data_ms": t - t % DAY_MS - DAY_MS, "risk_covariance_sha256": m.array_identity(covariance),
                "signal_us": t * 1000 + contract["signal_delay_us"], "modeled_fill_us": t * 1000 + contract["fill_delay_us"],
                "desired_units": desired.tolist(), "constraint_triggers": triggers, "reasons": reasons,
                "retained_assets": proposal["retained_assets"], "risk_trim_assets": proposal["risk_trim_assets"],
                "projected_post_cost_nav": proposal["projected_post_cost_nav"],
                "projected_post_cost_risk": proposal["projected_post_cost_risk"]})
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
                    episodes.append({"symbol": symbols[asset], "entry_ms": entry_times[asset], "exit_ms": t,
                                     "duration_hours": (t - entry_times[asset]) / 3600000})
                    entry_times[asset] = None
            slip = q * abs(execution - px[asset])
            cash += amount
            fees += fee
            slippage += slip
            gross += realized
            notional += q * px[asset]
            for key, value in (("gross_pnl", realized), ("fees", fee), ("slippage", slip)):
                per_asset[symbols[asset]][key] += float(value)
            reason = reasons[asset]
            reason_totals[reason]["fills"] += 1
            for key, value in (("notional", q * px[asset]), ("fees", fee), ("slippage", slip)):
                reason_totals[reason][key] += float(value)
            ledger.append({"kind": "modeled_fill", "timestamp_us": t * 1000 + contract["fill_delay_us"],
                "symbol": symbols[asset], "side": side, "quantity": q, "quantity_before": previous,
                "quantity_after": float(quantity[asset]), "reference_close": float(px[asset]), "price": execution,
                "fee": fee, "slippage": slip, "gross_pnl": float(realized), "cash_delta": amount,
                "model_sha256": model_sha, "vector_sha256": vector_sha, "policy_sha256": policy_sha,
                "reason": reason, "constraint_triggers": triggers})
            mark(t, px, "after_modeled_fill")
        mark(t, px, "calendar_bucket_end")
    error = cash - contract["initial_capital"] - gross + fees + slippage - funding
    if np.any(quantity) or abs(error) > 1e-7:
        raise ValueError("terminal cash/positions reconciliation failed")
    calendar = {row["timestamp_ms"]: row for row in equity if row["phase"] == "calendar_bucket_end"}
    daily_nav = [float(contract["initial_capital"])] + [calendar[t]["nav"] for t in range(start + DAY_MS, end + 1, DAY_MS)]
    exposure = [calendar[t]["exposure"] for t in range(start, end, BAR_MS)]
    durations = [row["duration_hours"] for row in episodes]
    for values in per_asset.values():
        values["net_pnl"] = values["gross_pnl"] - values["fees"] - values["slippage"] + values["funding"]
    return p.c.plain({"schema": "ema_units_cash_ledger/1", "policy": policy, "scenario": scenario,
        "policy_sha256": policy_sha, "ema_model_sha256": model_sha, "start_ms": start, "end_ms": end,
        "execution_is_assumed": True, "observed_execution_qualified": False,
        "metrics": {"initial_capital": contract["initial_capital"], "final_capital": cash,
            "net_return": cash / contract["initial_capital"] - 1, "max_drawdown": maximum_dd,
            "gross_pnl": gross, "fees": fees, "slippage": slippage, "funding": funding,
            "turnover_initial_capital": notional / contract["initial_capital"], "mean_exposure": float(np.mean(exposure)),
            "maximum_exposure": float(max(exposure)), "fills": sum(row["kind"] == "modeled_fill" for row in ledger),
            "position_episodes": len(episodes), "decisions": len(decisions),
            "mean_holding_hours": float(np.mean(durations)) if durations else 0.,
            "median_holding_hours": float(np.median(durations)) if durations else 0., "reconciliation_error": error},
        "daily_log_returns": np.diff(np.log(daily_nav)), "equity": equity, "ledger": ledger,
        "decisions": decisions, "episodes": episodes, "per_asset": per_asset, "fills_by_reason": dict(reason_totals)})


def direct_checks(contract):
    count = 0

    def check(condition):
        nonlocal count
        if not condition:
            raise AssertionError(f"unit-retention check {count + 1} failed")
        count += 1

    px, current = np.full(8, 100.), np.zeros(8)
    current[0] = 9.
    ema = np.zeros(8, bool)
    ema[0] = True
    base = np.where(ema, .1, 0.)
    covariance = np.eye(8) * .0001
    held = desired_units(current, px, 10000., ema, base, covariance, contract, "stress")
    check(np.array_equal(held["quantity"], current))
    check(modeled_cost(current, held["quantity"], px, 6, 8) == 0)
    check(held["retained_assets"] == 1 and not held["constraint_triggers"])
    entry = desired_units(np.zeros(8), px, 10000., ema, base, covariance, contract, "stress")
    check(abs(entry["quantity"][0] - 1000 / (100 * 1.0008 * 1.0006)) < 1e-12)
    exited = desired_units(current, px, 10000., np.zeros(8, bool), base, covariance, contract, "stress")
    check(not np.any(exited["quantity"]) and exited["reasons"][0] == "signal_exit")
    too_large = current * 2
    trimmed = desired_units(too_large, px, 10000., ema, base, covariance, contract, "stress")
    check(trimmed["quantity"][0] < too_large[0] and trimmed["reasons"][0] == "risk_trim")
    check(trimmed["projected_post_cost_risk"]["asset_max"] <= .1 + 1e-10)
    vol = desired_units(current, px, 10000., ema, base, np.eye(8) * .01, contract, "stress")
    check("daily_volatility" in vol["constraint_triggers"])
    check(vol["projected_post_cost_risk"]["daily_volatility"] <= .005 + 1e-10)
    # Cost estimate is independently obtained from one buy and one sell formula.
    old, new = np.asarray([1., 2.]), np.asarray([3., .5])
    reference = np.asarray([100., 200.])
    expected = 2 * 100 * .0008 + 2 * 100 * 1.0008 * .0006
    expected += 1.5 * 200 * .0008 + 1.5 * 200 * .9992 * .0006
    check(abs(modeled_cost(old, new, reference, 6, 8) - expected) < 1e-12)
    return {"checks_passed": count, "training_fits": 0, "network_calls": 0, "collector_runs": 0}
