"""Reference-open arithmetic and blocked market requests; not a fill model."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import re

BAR_MS = 14_400_000
SAFETY = ("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
          "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")
FIXED = {
    "schema": "exia.genetic_primus.reference_open_net_contract/1",
    "status": "draft_costs_unregistered_no_market_net_or_fit",
    "reference_price_kind": "ohlcv_open_reference_not_mid_or_observed_fill",
    "price_adjustment": "registered_adverse_relative_drag_against_reference_open_not_inferred_half_spread",
    "sizing": "allocated_quote_includes_entry_fee_fixed_quantity_until_registered_exit",
    "funding": "explicit_complete_ordered_settlements_with_marks_signed_positive_debits_long",
    "terminal": "liquidate_at_registered_18_bar_exit_no_carry_or_mark_to_market_substitution",
    "normalization": "same_explicit_quote_capital_and_comparison_interval_for_paired_actions",
    "arithmetic": "decimal_precision_50_no_venue_rounding_claim",
}
PARAMETERS = {"normal_scenario", "stress_scenario", "normalization_quote", "allocation_rule",
              "funding_policy_and_evidence", "borrow_basis", "market_cost_evidence",
              "decision_threshold_rule", "net_training_objective"}
TRADE_KEYS = {"comparison_key", "symbol", "period_start_ms", "period_end_ms", "action",
              "normalization_quote", "allocation_fraction", "reference_price_kind", "reference_entry",
              "reference_exit", "funding_events", "funding_events_complete", "borrow_quote"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def keys(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("reference accounting schema")


def assess(contract):
    keys(contract, set(FIXED) | {"parameters", "safety"})
    if {k: contract[k] for k in FIXED} != FIXED:
        raise ValueError("reference policy substitution")
    keys(contract["parameters"], PARAMETERS)
    if any(value is not None for value in contract["parameters"].values()):
        raise ValueError("v1 is an immutable draft, not cost registration")
    keys(contract["safety"], SAFETY)
    if any(contract["safety"][name] is not False for name in SAFETY):
        raise ValueError("authority must stay false")
    return {"missing_registration": sorted(PARAMETERS), "market_net_computation_allowed": False,
            "fit_allowed": False, "fills_or_costs_calibrated": False}


def number(value):
    if type(value) not in (int, float, str):
        raise ValueError("explicit numeric input required; no missing-value default")
    if type(value) is str and (len(value) > 48 or re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]{1,24})?", value) is None):
        raise ValueError("bounded canonical decimal required")
    try:
        result = Decimal(str(value))
    except Exception:
        raise ValueError("invalid decimal input") from None
    if not result.is_finite() or result.copy_abs() > Decimal("1e15") or result.as_tuple().exponent < -24:
        raise ValueError("decimal input bounds")
    return result


def text(value):
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def timestamp(value):
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("integer timestamp required")


def rates(scenario):
    keys(scenario, {"entry", "exit"})
    result = {}
    with localcontext() as context:
        context.prec = 50
        for side in ("entry", "exit"):
            keys(scenario[side], {"fee_bps", "adverse_price_bps"})
            result[side] = {}
            for name, value in scenario[side].items():
                rate = number(value)
                if not 0 <= rate < 10000:
                    raise ValueError("explicit nonnegative per-side bps below 10000 required")
                result[side][name] = rate / Decimal(10000)
    return result


def account(trade, scenario):
    """Pure math only; callers must separately register any market use."""
    keys(trade, TRADE_KEYS)
    if (trade["reference_price_kind"] != FIXED["reference_price_kind"]
            or type(trade["comparison_key"]) is not str or not 1 <= len(trade["comparison_key"]) <= 64
            or type(trade["symbol"]) is not str or not 1 <= len(trade["symbol"]) <= 40):
        raise ValueError("reference/comparison context required")
    for field in ("period_start_ms", "period_end_ms"):
        timestamp(trade[field])
    if (trade["period_start_ms"] % BAR_MS or trade["period_end_ms"] - trade["period_start_ms"] != 18 * BAR_MS):
        raise ValueError("registered 18-bar entry/exit required")
    capital, fraction, borrow = (number(trade[name]) for name in ("normalization_quote", "allocation_fraction", "borrow_quote"))
    if capital <= 0 or not 0 <= fraction <= 1 or borrow < 0:
        raise ValueError("capital/allocation/borrow bounds")
    if type(trade["funding_events_complete"]) is not bool or type(trade["funding_events"]) is not list:
        raise ValueError("explicit funding completeness/list required")
    if len(trade["funding_events"]) > 128:
        raise ValueError("funding count cap")
    with localcontext() as context:
        context.prec = 50
        r = rates(scenario)
        quantity = allocated = entry_fee = exit_fee = drag = gross = funding = Decimal(0)
        cash_after_entry = final_cash = capital
        if trade["action"] == "NoTrade":
            if (fraction != 0 or borrow != 0 or trade["funding_events"]
                    or trade["reference_entry"] is not None or trade["reference_exit"] is not None):
                raise ValueError("NoTrade cannot carry prices, exposure or costs")
        elif trade["action"] == "Long":
            if fraction == 0 or trade["funding_events_complete"] is not True:
                raise ValueError("positive exposure and complete funding inputs required")
            entry, exit_open = number(trade["reference_entry"]), number(trade["reference_exit"])
            if min(entry, exit_open) <= 0:
                raise ValueError("positive reference opens required")
            entry_price = entry * (1 + r["entry"]["adverse_price_bps"])
            exit_price = exit_open * (1 - r["exit"]["adverse_price_bps"])
            allocated = capital * fraction
            quantity = allocated / (entry_price * (1 + r["entry"]["fee_bps"]))
            entry_fee = quantity * entry_price * r["entry"]["fee_bps"]
            exit_fee = quantity * exit_price * r["exit"]["fee_bps"]
            cash_after_entry = capital - allocated
            previous = -1
            for event in trade["funding_events"]:
                keys(event, {"settled_at_ms", "mark_price", "signed_rate_bps"})
                settled = event["settled_at_ms"]
                timestamp(settled)
                if not trade["period_start_ms"] < settled < trade["period_end_ms"] or settled <= previous:
                    raise ValueError("funding boundary/order requires explicit sequence evidence")
                previous = settled
                mark, rate = number(event["mark_price"]), number(event["signed_rate_bps"])
                if mark <= 0 or not -10000 < rate < 10000:
                    raise ValueError("funding mark/rate bounds")
                funding += quantity * mark * rate / Decimal(10000)
            final_cash = cash_after_entry + quantity * exit_price - exit_fee - funding - borrow
            gross = quantity * (exit_open - entry)
            drag = quantity * (entry_price - entry) + quantity * (exit_open - exit_price)
        else:
            raise ValueError("only Long and NoTrade supported")
        net = final_cash - capital
        algebra_net = gross - drag - entry_fee - exit_fee - funding - borrow
        # Independent cash-flow and attribution paths must agree to arithmetic tolerance.
        if abs(net - algebra_net) > max(capital, Decimal(1)) * Decimal("1e-45"):
            raise ValueError("cash/attribution reconciliation failed")
        return {"quantity": text(quantity), "allocated_quote": text(allocated),
                "cash_after_entry": text(cash_after_entry), "final_cash": text(final_cash),
                "entry_fee_quote": text(entry_fee), "exit_fee_quote": text(exit_fee),
                "reference_gross_quote": text(gross), "adverse_price_drag_quote": text(drag),
                "funding_debit_quote": text(funding), "borrow_debit_quote": text(borrow),
                "net_quote": text(net), "net_return": text(net / capital),
                "request_sha256": sha(trade), "scenario_sha256": sha(scenario),
                "venue_execution_verified": False, "fit_allowed": False}


def paired(challenger, incumbent, scenario):
    keys(challenger, TRADE_KEYS)
    keys(incumbent, TRADE_KEYS)
    for field in ("comparison_key", "symbol", "period_start_ms", "period_end_ms", "reference_price_kind"):
        if challenger[field] != incumbent[field]:
            raise ValueError("paired comparison context differs")
    if number(challenger["normalization_quote"]) != number(incumbent["normalization_quote"]):
        raise ValueError("paired capital differs")
    if challenger["action"] == incumbent["action"] == "Long":
        for field in ("reference_entry", "reference_exit", "funding_events", "funding_events_complete"):
            if challenger[field] != incumbent[field]:
                raise ValueError("paired observations differ")
    left, right = account(challenger, scenario), account(incumbent, scenario)
    with localcontext() as context:
        context.prec = 50
        delta = text(Decimal(left["net_return"]) - Decimal(right["net_return"]))
    return {"challenger": left, "incumbent": right, "incremental_net_return": delta, "fit_allowed": False}


def blocked_request(sample, label_reference, contract):
    gate = assess(contract)
    if (sample["target_kind"] != "gross_open_open_diagnostic_not_net_training_target"
            or sample["fit_allowed"] is not False or sample["net_objective_allowed"] is not False
            or sample["outcome_filter_applied"] is not False):
        raise ValueError("gross input role changed")
    symbol, asof = sample["symbol"], sample["decision_asof_ms"]
    if sample["sample_id"] != sha({"symbol": symbol, "decision_asof_ms": asof}):
        raise ValueError("sample identity changed")
    start, end = sample["entry_open_ms"], sample["exit_open_ms"]
    for t in (asof, start, end, sample["label_available_at_ms"], sample["training_cutoff_ms"]):
        timestamp(t)
    if (asof % BAR_MS or start != asof + BAR_MS or end - start != 18 * BAR_MS
            or sample["label_available_at_ms"] != end + BAR_MS
            or sample["label_available_at_ms"] > sample["training_cutoff_ms"]):
        raise ValueError("causal timing/purge changed")
    if (sample["entry_observation"]["timestamp"] != start or sample["exit_observation"]["timestamp"] != end
            or sample["entry_observation"]["symbol"] != symbol or sample["exit_observation"]["symbol"] != symbol):
        raise ValueError("reference-open dependency key differs")
    entry, exit_open = number(sample["entry_observation"]["open"]), number(sample["exit_observation"]["open"])
    if min(entry, exit_open) <= 0 or type(sample["proposal_long"]) is not bool:
        raise ValueError("reference/proposal bounds")
    with localcontext() as context:
        context.prec = 50
        if Decimal(sample["gross_open_return_decimal"]) != exit_open / entry - Decimal(1):
            raise ValueError("gross diagnostic label mismatch")
    request = {"comparison_key": sample["sample_id"], "symbol": symbol,
               "period_start_ms": start, "period_end_ms": end, "action": "Long",
               "normalization_quote": None, "allocation_fraction": None,
               "reference_price_kind": FIXED["reference_price_kind"], "reference_entry": text(entry),
               "reference_exit": text(exit_open), "funding_events": None,
               "funding_events_complete": None, "borrow_quote": None}
    return {"sample_id": sample["sample_id"], "symbol": symbol, "decision_asof_ms": asof,
            "gross_label_reference": label_reference, "gross_sample_sha256": sha(sample),
            "feature_row_sha256": sample["feature_row_sha256"], "proposal_long": sample["proposal_long"],
            "long_input_pending_registration": request, "net_target": None, "stress_net_target": None,
            "paired_incremental_target": None, "status": "blocked_unknown_costs_not_zero_costs",
            "missing_registration": gate["missing_registration"], "outcome_filter_applied": False,
            "market_net_computation_allowed": False, "fit_allowed": False}
