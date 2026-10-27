"""Joint-portfolio counterfactuals and a typed hook into the frozen cash engine."""
from __future__ import annotations

import math
import numpy as np

from . import forecast4h_v1 as f


def costs_for_delta(delta, prices, fee_bps, drag_bps):
    fee, drag = fee_bps / 10000., drag_bps / 10000.
    executed = prices * (1. + np.sign(delta) * drag)
    return np.abs(delta) * (prices * drag + executed * fee)


def move(cash, quantity, desired, prices, fee_bps, drag_bps):
    """Same sells-first delta-unit arithmetic and tolerance as frozen f.replay."""
    quantity = np.asarray(quantity, float).copy()
    desired, prices = np.asarray(desired, float), np.asarray(prices, float)
    if (quantity.shape != desired.shape or prices.shape != quantity.shape
            or not np.isfinite(desired).all() or np.any(desired < 0)
            or not np.isfinite(prices).all() or np.any(prices <= 0)):
        raise ValueError("finite aligned long-only units/prices required")
    fees = slips = 0.
    delta = desired - quantity
    for asset in list(np.flatnonzero(delta < -1e-12)) + list(np.flatnonzero(delta > 1e-12)):
        sign = 1. if delta[asset] > 0 else -1.
        q = abs(float(delta[asset]))
        executed = float(prices[asset]) * (1. + sign * drag_bps / 10000.)
        fee = q * executed * fee_bps / 10000.
        cash += -sign * q * executed - fee
        quantity[asset] += sign * q
        if abs(quantity[asset]) < 1e-12:
            quantity[asset] = 0.
        fees += fee
        slips += q * abs(executed - prices[asset])
        if cash < -1e-7:
            raise ValueError("no borrowing: joint transition infeasible")
    return float(cash), quantity, float(fees + slips)


def desired(cash, quantity, prices, weights, fee_bps, drag_bps):
    nav = float(cash + quantity @ prices)
    return nav * weights / (prices * (1. + drag_bps / 10000.) * (1. + fee_bps / 10000.))


def remove_and_bound(base, rejected, covariance, budget=.005):
    weights = np.asarray(base, float).copy()
    if not np.any(rejected):
        # Preserve the frozen baseline's exact floating-point operation order.
        return weights
    weights[np.asarray(rejected, bool)] = 0.
    risk = math.sqrt(max(0., float(weights @ covariance @ weights)))
    if risk > budget:
        weights *= budget / risk
    return weights


def paired_value(cash, quantity, now, end, base_now, base_end, covariance_now,
                 funding_per_unit, asset, fee_bps=6., drag_bps=8.):
    """Baseline rebalance now vs rejecting one held asset; common endpoint.

    base_end is a FUTURE label input from the reference rule, not a feature.
    Shared cash is used ONCE per branch, never duplicated across assets.
    """
    quantity, now, end = map(lambda x: np.asarray(x, float), (quantity, now, end))
    if quantity[asset] <= 1e-12 or base_now[asset] <= 0:
        raise ValueError("held positive-EMA opportunity required")
    initial_nav = float(cash + quantity @ now)
    rejected = np.zeros(len(quantity), bool); rejected[asset] = True
    action_weights = remove_and_bound(base_now, rejected, covariance_now)
    starts = []
    for weights in (base_now, action_weights):
        target = desired(cash, quantity, now, weights, fee_bps, drag_bps)
        c, q, cost = move(cash, quantity, target, now, fee_bps, drag_bps)
        paid = float(q @ funding_per_unit)
        c -= paid
        if c < -1e-7:
            raise ValueError("funding cash deficit: whole stage abort, never drop label")
        starts.append((c, q, cost, paid))
    hold_cash, hold_q, _, _ = starts[0]
    requested = desired(hold_cash, hold_q, end, base_end, fee_bps, drag_bps)
    def remaining_cash(state, common):
        c, q, _, _ = state
        delta = common - q
        return float(c - delta @ end - costs_for_delta(delta, end, fee_bps, drag_bps).sum())
    scale = 1.
    if min(remaining_cash(s, requested) for s in starts) < -1e-9:
        lo, hi = 0., 1.
        for _ in range(60):
            mid = (lo + hi) / 2.
            if min(remaining_cash(s, requested * mid) for s in starts) >= 0.:
                lo = mid
            else:
                hi = mid
        scale = lo
    common = requested * scale
    navs, total_costs = [], []
    for c, q, cost, paid in starts:
        terminal_cash, terminal_q, boundary_cost = move(c, q, common, end, fee_bps, drag_bps)
        navs.append(float(terminal_cash + terminal_q @ end))
        total_costs.append(cost + boundary_cost)
    delta_q = starts[1][1] - starts[0][1]
    gross = float(delta_q @ (end - now))
    funding = starts[1][3] - starts[0][3]
    cost_delta = total_costs[1] - total_costs[0]
    delta = navs[1] - navs[0]
    if not math.isclose(delta, gross - cost_delta - funding, rel_tol=1e-9, abs_tol=1e-8):
        raise ArithmeticError("joint paired decomposition failed")
    notional = float(quantity[asset] * now[asset])
    return {"target_bps": 10000. * delta / notional, "delta_nav": delta,
            "delta_bps_nav": 10000. * delta / initial_nav,
            "delta_gross": gross, "delta_costs": cost_delta, "delta_funding_paid": funding,
            "cross_asset_gross": gross - float(delta_q[asset] * (end[asset] - now[asset])),
            "endpoint_scale": scale, "endpoint_capped": scale < 1. - 1e-10}


def action_replay(data, vectors, covariances, rates, contract, name, scenario):
    """No extra trades: the hook changes targets inside the frozen actual ledger."""
    index = 0
    rejected_by_time = {}
    def rule(policy, transport_values, base, held, frozen):
        nonlocal index
        vector = vectors[index]; index += 1
        # expected_return_4h is only a legacy transport alias, NOT a return model.
        prediction = np.asarray(vector["predicted_close_gain_bps"], float)
        if not np.array_equal(prediction, np.asarray(transport_values)):
            raise ValueError("typed action-value bridge mismatch")
        rejected = held & (base > 0) & (prediction > 0)
        rejected_by_time[vector["asof_ms"]] = np.flatnonzero(rejected).tolist()
        day = vector["asof_ms"] - vector["asof_ms"] % f.DAY_MS
        return remove_and_bound(base, rejected, covariances[day],
                                budget=contract["risk"]["daily_volatility_target"])
    replay_contract = dict(contract) | {"action_value_policy": name,
        "action_value_semantics": "incremental net benefit, not price return"}
    result = f.replay(data, vectors, covariances, rates, replay_contract,
                      "R1_EMA_4h", scenario, target_rule=rule)
    if index != len(vectors):
        raise ValueError("one action vector per decision required")
    result["policy"] = name
    result["target_kind"] = "paired_incremental_close_vs_baseline_rebalance_net_bps"
    result["label_boundary_trades_executed"] = False
    for row in result["decisions"]:
        row["overlay_rejected_assets"] = rejected_by_time[row["asof_ms"]]
    return result
