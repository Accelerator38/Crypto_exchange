"""Pure paired one-step action accounting; not a trading runtime or trainer."""
from __future__ import annotations

import math


def _number(value, name, *, lower=None):
    if isinstance(value, bool):
        raise ValueError(name + " must be a real number, not bool")
    value = float(value)
    if not math.isfinite(value) or (lower is not None and value < lower):
        raise ValueError("invalid " + name)
    return value


def transition(cash, units, target_units, price, fee_bps, drag_bps):
    """Cost-inclusive fractional-unit transition at an assumed reference price."""
    cash = _number(cash, "cash", lower=0.)
    units = _number(units, "units", lower=0.)
    target_units = _number(target_units, "target_units", lower=0.)
    price = _number(price, "price", lower=0.)
    fee = _number(fee_bps, "fee_bps", lower=0.) / 10000.
    drag = _number(drag_bps, "drag_bps", lower=0.) / 10000.
    if price == 0. or fee >= 1. or drag >= 1.:
        raise ValueError("positive price and fee/drag below 10000 bps required")
    delta = target_units - units
    sign = 1. if delta > 0. else -1. if delta < 0. else 0.
    fill_price = price * (1. + sign * drag)
    fee_paid = abs(delta) * fill_price * fee
    slippage = abs(delta) * price * drag
    new_cash = cash - delta * fill_price - fee_paid
    tolerance = 1e-10 * max(1., cash + units * price)
    if new_cash < -tolerance:
        raise ValueError("cash-infeasible transition; no borrowing permitted")
    return {"cash": max(0., new_cash), "units": target_units,
            "fees": fee_paid, "slippage": slippage,
            "costs": fee_paid + slippage,
            "reference_turnover": abs(delta) * price}


def paired_action_value(*, initial_cash, current_units, action_units,
                        price_now, price_end, reference_units_end,
                        fee_bps, drag_bps, funding_events=()):
    """Act vs hold for one interval, then equalize terminal units.

    Snapshot is AFTER funding at t and BEFORE the action. Each funding item is
    (reference_price, rate) in chronological order for (t, H], BEFORE terminal
    transitions. Positive rates are debits. Caller owns timestamp/lineage checks.
    reference_units_end is a FUTURE label input, never a prediction feature.
    Only reductions are allowed. Both branches use the same feasible terminal
    units, capped jointly if either lacks cash to restore the requested reference.
    A funding cash deficit aborts the calculation, never silently drops a row.
    """
    q0 = _number(current_units, "current_units", lower=0.)
    qa = _number(action_units, "action_units", lower=0.)
    reference = _number(reference_units_end, "reference_units_end", lower=0.)
    if q0 == 0. or qa > q0 or reference > q0:
        raise ValueError("positive existing long; reduction-only reference/action")
    # The common transition function validates cash, prices, and costs.
    hold = transition(initial_cash, q0, q0, price_now, fee_bps, drag_bps)
    action = transition(initial_cash, q0, qa, price_now, fee_bps, drag_bps)
    price_end = _number(price_end, "price_end", lower=0.)
    if price_end == 0.:
        raise ValueError("positive terminal price required")
    fee, drag = float(fee_bps) / 10000., float(drag_bps) / 10000.
    branches = {"hold": hold, "act": action}
    funding_paid = {name: 0. for name in branches}
    for event in funding_events:
        if len(event) != 2:
            raise ValueError("funding event must be (price, rate)")
        funding_price = _number(event[0], "funding_price", lower=0.)
        rate = _number(event[1], "funding_rate")
        if funding_price == 0.:
            raise ValueError("positive funding price required")
        for name, state in branches.items():
            debit = state["units"] * funding_price * rate
            state["cash"] -= debit
            funding_paid[name] += debit
            if state["cash"] < 0.:
                raise ValueError("funding exhausts cash: unsupported whole-stage failure")
    # Same terminal position, with no branch-specific silent cash clipping.
    buy_unit_cost = price_end * (1. + drag) * (1. + fee)
    common = min(reference, *(state["units"] + state["cash"] / buy_unit_cost
                              for state in branches.values()))
    terminal = {name: transition(state["cash"], state["units"], common,
                                 price_end, fee_bps, drag_bps)
                for name, state in branches.items()}
    costs = {name: branches[name]["costs"] + terminal[name]["costs"]
             for name in branches}
    nav0 = float(initial_cash) + q0 * float(price_now)
    navs = {name: state["cash"] + common * price_end for name, state in terminal.items()}
    delta_gross = (qa - q0) * (price_end - float(price_now))
    delta_costs = costs["act"] - costs["hold"]
    delta_funding = funding_paid["act"] - funding_paid["hold"]
    delta_nav = navs["act"] - navs["hold"]
    decomposition = delta_gross - delta_costs - delta_funding
    if not math.isclose(delta_nav, decomposition, rel_tol=1e-10,
                        abs_tol=1e-10 * max(1., nav0)):
        raise ArithmeticError("paired capital decomposition mismatch")
    return {"hold_nav": navs["hold"], "action_nav": navs["act"],
            "delta_nav": delta_nav,
            "delta_bps_initial_nav": 10000. * delta_nav / nav0,
            "delta_bps_initial_exposure": 10000. * delta_nav / (q0 * float(price_now)),
            "delta_gross": delta_gross, "delta_costs": delta_costs,
            "delta_funding_paid": delta_funding,
            "hold_costs": costs["hold"], "action_costs": costs["act"],
            "common_terminal_units": common,
            "terminal_reference_capped": common < reference,
            "boundary_trades_are_label_only": True}
