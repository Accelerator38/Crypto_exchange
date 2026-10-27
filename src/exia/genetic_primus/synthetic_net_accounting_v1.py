"""Exact-input decimal synthetic long/flat accounting. Never venue execution."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import math
import re

SAFETY = ("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
          "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")
FIXED = {
    "schema": "exia.genetic_primus.synthetic_net_accounting_contract/1",
    "status": "engineering_only_not_registered_market_objective",
    "instrument": "SYN_LINEAR_USDT",
    "reference_prices": "synthetic_mid_not_ohlcv_open_or_observed_fill",
    "sizing": "allocated_quote / (entry_fill * (1 + entry_fee_rate))",
    "price_adjustment": "entry_mid_plus_half_spread_and_slippage_exit_mid_minus_half_spread_and_slippage",
    "fees": "each_side_rate_times_that_side_fill_notional",
    "funding": "fixed_quantity_times_explicit_settlement_mark_times_signed_rate",
    "funding_boundaries": "strictly_inside_period_equal_boundaries_require_unsupported_sequence_evidence",
    "borrow": "explicit_quote_debit_no_default",
    "normalization": "same_explicit_quote_capital_for_paired_agents",
    "no_trade": "zero_allocation_no_prices_no_costs",
    "arithmetic": "decimal_precision_50_no_exchange_rounding_claim",
    "market_calibration": None, "market_fit_allowed": False,
}
TRADE_KEYS = {"schema", "instrument", "comparison_key", "period_start_ms", "period_end_ms",
              "normalization_quote", "allocation_fraction", "action", "reference_price_kind",
              "reference_entry", "reference_exit", "funding_events", "funding_events_complete", "borrow_quote"}
ZERO = Decimal(0)
ONE = Decimal(1)


def sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def keys(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("accounting schema mismatch")


def assess(contract):
    keys(contract, set(FIXED) | {"safety"})
    if sha({k: contract[k] for k in FIXED}) != sha(FIXED):
        raise ValueError("accounting policy substitution")
    keys(contract["safety"], SAFETY)
    if any(contract["safety"][k] is not False for k in SAFETY):
        raise ValueError("authority must stay false")
    return {"fit_allowed": False, "costs_calibrated": False, "fills_verified": False,
            "market_net_objective_registered": False}


def number(value):
    if type(value) not in (int, float, str):
        raise ValueError("explicit decimal required")
    if type(value) is str:
        if len(value) > 32 or re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]{1,12})?", value) is None:
            raise ValueError("canonical decimal string required")
    else:
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError("finite decimal required")
    result = Decimal(str(value))
    if not result.is_finite() or result.copy_abs() > Decimal("1e15") or result.as_tuple().exponent < -12:
        raise ValueError("decimal input bounds")
    return result


def bps_fraction(value):
    rate = number(value)
    if not ZERO <= rate < 10000:
        raise ValueError("nonnegative per-side rate below 10000bps required")
    with localcontext() as ctx:
        ctx.prec = 50
        return rate / Decimal(10000)


def text(value):
    if value == ZERO:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def timestamp(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("timestamp required")


def scenario_rates(scenario):
    keys(scenario, {"schema", "name", "entry", "exit"})
    if scenario["schema"] != "exia.genetic_primus.synthetic_cost_scenario/1" or (
        type(scenario["name"]) is not str or re.fullmatch(r"fixture_[a-z_]{1,32}", scenario["name"]) is None):
        raise ValueError("synthetic cost scenario required")
    rates = {}
    for side in ("entry", "exit"):
        keys(scenario[side], {"fee_bps", "half_spread_bps", "slippage_bps"})
        rates[side] = {name: bps_fraction(value) for name, value in scenario[side].items()}
        if rates[side]["half_spread_bps"] + rates[side]["slippage_bps"] >= ONE:
            raise ValueError("nonpositive exit fill scenario")
    return rates


def account(trade, scenario, contract):
    assessment = assess(contract)
    keys(trade, TRADE_KEYS)
    if (trade["schema"] != "exia.genetic_primus.synthetic_cost_trade/1"
        or trade["instrument"] != "SYN_LINEAR_USDT"
        or trade["reference_price_kind"] != "synthetic_mid"
        or type(trade["comparison_key"]) is not str or not 1 <= len(trade["comparison_key"]) <= 64):
        raise ValueError("synthetic reference/instrument required")
    for name in ("period_start_ms", "period_end_ms"):
        timestamp(trade[name])
    if trade["period_end_ms"] <= trade["period_start_ms"]:
        raise ValueError("nonempty period required")
    capital, fraction, borrow = (number(trade[k]) for k in ("normalization_quote", "allocation_fraction", "borrow_quote"))
    if capital <= ZERO or not ZERO <= fraction <= ONE or borrow < ZERO:
        raise ValueError("capital/allocation/borrow bounds")
    if type(trade["funding_events_complete"]) is not bool:
        raise ValueError("explicit funding completeness declaration required")
    events = trade["funding_events"]
    if type(events) is not list or len(events) > 128:
        raise ValueError("explicit bounded funding list required")
    with localcontext() as ctx:
        ctx.prec = 50
        rates = scenario_rates(scenario)
        qty = reference_gross = drag = fees = funding = allocated = ZERO
        if trade["action"] == "NoTrade":
            if (fraction != ZERO or borrow != ZERO or events
                or trade["reference_entry"] is not None or trade["reference_exit"] is not None):
                raise ValueError("NoTrade cannot have prices/exposure/costs")
            entry_fee = exit_fee = entry_drag = exit_drag = ZERO
        elif trade["action"] == "Long":
            if fraction == ZERO or not trade["funding_events_complete"]:
                raise ValueError("positive allocation and complete synthetic funding required")
            entry_mid, exit_mid = number(trade["reference_entry"]), number(trade["reference_exit"])
            if min(entry_mid, exit_mid) <= ZERO:
                raise ValueError("positive reference prices required")
            entry_fill = entry_mid * (ONE + rates["entry"]["half_spread_bps"] + rates["entry"]["slippage_bps"])
            exit_fill = exit_mid * (ONE - rates["exit"]["half_spread_bps"] - rates["exit"]["slippage_bps"])
            allocated = capital * fraction
            qty = allocated / (entry_fill * (ONE + rates["entry"]["fee_bps"]))
            entry_fee = qty * entry_fill * rates["entry"]["fee_bps"]
            exit_fee = qty * exit_fill * rates["exit"]["fee_bps"]
            fees = entry_fee + exit_fee
            entry_drag = qty * (entry_fill - entry_mid)
            exit_drag = qty * (exit_mid - exit_fill)
            drag = entry_drag + exit_drag
            reference_gross = qty * (exit_mid - entry_mid)
            previous = -1
            for event in events:
                keys(event, {"settled_at_ms", "mark_price", "signed_rate_bps"})
                timestamp(event["settled_at_ms"])
                settled = event["settled_at_ms"]
                if not trade["period_start_ms"] < settled < trade["period_end_ms"] or settled <= previous:
                    raise ValueError("funding interval/boundary/order ambiguity")
                previous = settled
                mark, signed = number(event["mark_price"]), number(event["signed_rate_bps"])
                if mark <= ZERO or not -10000 < signed < 10000:
                    raise ValueError("funding mark/rate bounds")
                funding += qty * mark * signed / Decimal(10000)
        else:
            raise ValueError("only Long and NoTrade supported")
        net = reference_gross - drag - fees - funding - borrow
        return {"schema": "exia.genetic_primus.synthetic_accounting_result/1",
                "comparison_key": trade["comparison_key"], "action": trade["action"],
                "contract_sha256": sha(contract), "scenario_sha256": sha(scenario), "request_sha256": sha(trade),
                "quantity": text(qty), "allocated_quote": text(allocated),
                "reference_gross_quote": text(reference_gross), "execution_drag_quote": text(drag),
                "entry_drag_quote": text(entry_drag), "exit_drag_quote": text(exit_drag),
                "entry_fee_quote": text(entry_fee), "exit_fee_quote": text(exit_fee),
                "fees_quote": text(fees), "funding_debit_quote": text(funding), "borrow_debit_quote": text(borrow),
                "net_quote": text(net), "net_return": text(net / capital), "assessment": assessment,
                "rounding_or_venue_execution_verified": False, "safety": contract["safety"]}


def paired(challenger, incumbent, scenario, contract):
    keys(challenger, TRADE_KEYS)
    keys(incumbent, TRADE_KEYS)
    for key in ("comparison_key", "instrument", "period_start_ms", "period_end_ms", "reference_price_kind"):
        if challenger.get(key) != incumbent.get(key):
            raise ValueError("paired comparison context mismatch")
    if number(challenger["normalization_quote"]) != number(incumbent["normalization_quote"]):
        raise ValueError("paired capital mismatch")
    if challenger.get("action") == incumbent.get("action") == "Long":
        for key in ("reference_entry", "reference_exit", "funding_events", "funding_events_complete"):
            if challenger.get(key) != incumbent.get(key):
                raise ValueError("paired source inputs differ")
    left, right = account(challenger, scenario, contract), account(incumbent, scenario, contract)
    with localcontext() as ctx:
        ctx.prec = 50
        difference = Decimal(left["net_return"]) - Decimal(right["net_return"])
    return {"schema": "exia.genetic_primus.synthetic_paired_accounting/1",
            "scenario_sha256": sha(scenario), "contract_sha256": sha(contract),
            "challenger_result_sha256": sha(left), "incumbent_result_sha256": sha(right),
            "delta_net_return": text(difference), "promotion_allowed": False}
