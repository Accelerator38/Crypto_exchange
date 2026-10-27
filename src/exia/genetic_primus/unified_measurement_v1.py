"""Pure offline measurement: one cash account, fixed quantities and 3-day clocks.

Reference opens and registered adverse price adjustments are sensitivity inputs,
not observed exchange fills. No I/O, fitter, collector or runtime dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from typing import Mapping, Sequence

from .research_execution_contract_v1 import BAR_MS, HORIZON_MS, execution_window

SCHEMA = "exia.genetic_primus.unified_measurement/1"
REFERENCE_KIND = "ohlcv_open_reference_not_observed_fill"
SAFETY = {name: False for name in (
    "credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
    "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed",
)}
SEMANTICS = {
    "bar_ms": BAR_MS, "deployment_horizon_ms": HORIZON_MS,
    "retrain_cadence_ms": HORIZON_MS, "purge_bars": 7,
    "execution": "first_open_strictly_after_decision_plus_delay",
    "label": "entry_open_to_open_exactly_18_bars_later",
    "capital": "one_shared_cash_account_no_borrow_no_leverage",
    "sizing": "new_weights_times_post_sell_nav_fee_inclusive_scaled_to_available_cash",
    "unchanged_target": "carry_fixed_quantity_no_refill",
    "changed_positive_target": "close_then_reopen_not_implicit_mark_rebalancing",
    "same_timestamp_order": "funding_then_sells_then_buys_then_mark",
    "funding_ownership": "entry_time_lt_settlement_time_le_exit_time",
    "funding_cash": "signed_positive_debits_long_require_available_cash",
    "terminal": "one_explicit_global_liquidation_then_cash_to_reporting_end",
    "return_bucket": "increment_over_previous_mark_to_current_mark_timestamp_current_minus_1ms",
    "drawdown": "mark_to_reference_open_including_costs_with_initial_capital_peak",
    "scope": "long_cash_reference_open_offline_sensitivity",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


SEMANTIC_SHA256 = identity(SEMANTICS)


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("finite decimal required")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError("finite decimal required") from exc
    if not result.is_finite():
        raise ValueError("finite decimal required")
    return result


def _time(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer epoch milliseconds required")
    return value


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("explicit SHA256 required")
    return value


def _text(value):
    return format(value, "f")


def complete_origins(start_ms, end_ms, *, decision_offset_ms=0, execution_delay_ms=0):
    """Keep complete 18-bar outcomes; do not shorten the last deployment."""
    _time(start_ms)
    _time(end_ms)
    _time(decision_offset_ms)
    _time(execution_delay_ms)
    if start_ms % BAR_MS or end_ms % BAR_MS or end_ms <= start_ms or decision_offset_ms >= BAR_MS:
        raise ValueError("aligned nonempty reporting interval and within-bar decision offset required")
    result = []
    origin = start_ms
    while origin < end_ms:
        window = execution_window(origin, origin + decision_offset_ms, execution_delay_ms)
        if window["planned_exit_open_ms"] > end_ms:
            break
        result.append(window)
        origin += HORIZON_MS
    return result


def label_binding(sample, *, origin_start_ms, execution_delay_ms=0, training_asof_ms):
    """Admit existing diagnostic open/open labels with a NEW origin's cutoff.

    Historical open/last-close targets cannot pass this adapter. Admission here
    verifies timing/arithmetic only and does not grant permission to fit.
    """
    _time(training_asof_ms)
    window = execution_window(origin_start_ms, sample["decision_asof_ms"], execution_delay_ms)
    if sample.get("target_kind") != "gross_open_open_diagnostic_not_net_training_target":
        raise ValueError("legacy or unknown economic horizon cannot enter unified measurement")
    entry, exit_open = sample["entry_observation"], sample["exit_observation"]
    for key, observed, planned in (
        ("entry_open_ms", entry, window["planned_entry_open_ms"]),
        ("exit_open_ms", exit_open, window["planned_exit_open_ms"]),
    ):
        if sample[key] != planned or observed["timestamp"] != planned or observed["symbol"] != sample["symbol"]:
            raise ValueError("label/decision/ledger horizon mismatch")
    available = _time(sample["label_available_at_ms"])
    if available < window["planned_exit_open_ms"] + BAR_MS:
        raise ValueError("completed OHLCV open observation is not available yet")
    with localcontext() as ctx:
        ctx.prec = 50
        a, b = _number(entry["open"]), _number(exit_open["open"])
        if min(a, b) <= 0:
            raise ValueError("positive reference opens required")
        gross = b / a - 1
        if abs(gross - _number(sample["gross_open_return_decimal"])) > Decimal("1e-35"):
            raise ValueError("gross target does not reconcile to reference opens")
    cutoff = training_asof_ms - 7 * BAR_MS
    return {"measurement_semantic_sha256": SEMANTIC_SHA256,
            "label_definition_sha256": window["gross_label_definition_sha256"],
            "training_cutoff_ms": cutoff, "label_available_at_ms": available,
            "timing_eligible": available <= cutoff, "fit_allowed": False,
            "gross_open_return_decimal": _text(gross)}


def build_ranking_key(*, fitness, failed_gates, max_drawdown=0.0,
                      fills_per_day=0.0, active_parameter_count=0.0):
    """One final builder: all gates survive every objective/overlay branch.

    Within equal feasibility, the penalized scalar fitness is primary. This
    makes complexity/concentration/stability penalties operative in search.
    """
    values = tuple(float(v) for v in (fitness, max_drawdown, fills_per_day, active_parameter_count))
    if not all(math.isfinite(v) for v in values) or min(values[1:]) < 0:
        raise ValueError("finite ranking inputs and nonnegative risk/complexity required")
    failures = set(failed_gates)
    if any(not isinstance(v, str) or not v for v in failures):
        raise ValueError("named failed gates required")
    return (float(not failures), -float(len(failures)), values[0], -values[1], -values[2], -values[3])


@dataclass(frozen=True)
class CostScenario:
    name: str
    buy_fee_bps: str
    sell_fee_bps: str
    buy_adverse_bps: str
    sell_adverse_bps: str
    provenance: str
    evidence_sha256: str

    def validate(self):
        if not self.name or self.provenance not in {"synthetic", "registered_assumption"}:
            raise ValueError("named synthetic/registered sensitivity costs required")
        _digest(self.evidence_sha256)
        for name in ("buy_fee_bps", "sell_fee_bps", "buy_adverse_bps", "sell_adverse_bps"):
            if not 0 <= _number(getattr(self, name)) < 10000:
                raise ValueError("per-fill cost bps out of bounds")
        return self


@dataclass(frozen=True)
class MeasurementSpec:
    start_ms: int
    end_ms: int
    terminal_ms: int
    initial_capital_quote: str
    symbols: tuple[str, ...]
    market_inputs_sha256: str

    def validate(self):
        for value in (self.start_ms, self.end_ms, self.terminal_ms):
            if _time(value) % BAR_MS:
                raise ValueError("4-hour grid required")
        if not self.start_ms < self.terminal_ms <= self.end_ms or _number(self.initial_capital_quote) <= 0:
            raise ValueError("positive capital and registered terminal within reporting period required")
        if (not self.symbols or len(set(self.symbols)) != len(self.symbols)
                or any(not isinstance(s, str) or not s for s in self.symbols)):
            raise ValueError("unique fixed universe required")
        _digest(self.market_inputs_sha256)
        return self


def make_decision(*, origin_start_ms, decision_asof_ms, targets, model_sha256,
                  scaler_sha256, feature_prefix_sha256, training_identity_sha256,
                  execution_delay_ms=0):
    window = execution_window(origin_start_ms, decision_asof_ms, execution_delay_ms)
    for value in (model_sha256, scaler_sha256, feature_prefix_sha256, training_identity_sha256):
        _digest(value)
    weights = {s: _text(_number(v)) for s, v in targets.items()}
    if any(_number(v) < 0 for v in weights.values()) or sum(map(_number, weights.values())) > 1:
        raise ValueError("long/cash allocations must total at most one")
    return {"window": window, "targets": weights, "model_sha256": model_sha256,
            "scaler_sha256": scaler_sha256, "feature_prefix_sha256": feature_prefix_sha256,
            "training_identity_sha256": training_identity_sha256,
            "measurement_semantic_sha256": SEMANTIC_SHA256}


def replay(spec: MeasurementSpec, *, price_bars: Sequence[Mapping],
           decisions: Sequence[Mapping], funding: Mapping, costs: CostScenario):
    """Deterministic event replay; inputs are supplied, never fetched.

    Every supplied bar must have the same complete universe. Funding evidence must cover
    the full reporting interval, even when its complete event set is empty.
    Insufficient funding cash or insolvency raises rather than inventing credit.
    """
    spec.validate()
    costs.validate()
    bars = list(price_bars)
    terminal_length = (spec.terminal_ms - spec.start_ms) // BAR_MS + 1
    report_length = (spec.end_ms - spec.start_ms) // BAR_MS + 1
    if not terminal_length <= len(bars) <= report_length:
        raise ValueError("complete contiguous prices through registered liquidation required")
    for i, row in enumerate(bars):
        if _time(row["timestamp"]) != spec.start_ms + i * BAR_MS or set(row["prices"]) != set(spec.symbols):
            raise ValueError("price grid or universe mismatch")
        if any(_number(p) <= 0 for p in row["prices"].values()):
            raise ValueError("positive reference prices required")
    if identity(bars) != spec.market_inputs_sha256:
        raise ValueError("canonical price input hash differs from registered measurement spec")
    if (funding.get("complete") is not True or funding.get("coverage_start_ms") != spec.start_ms
            or funding.get("coverage_end_ms") != spec.end_ms or funding.get("symbols") != list(spec.symbols)
            or funding.get("provenance") not in {"synthetic", "observed"}):
        raise ValueError("explicit complete signed funding evidence for the whole universe required")
    _digest(funding["evidence_sha256"])
    if (costs.provenance == "synthetic") != (funding["provenance"] == "synthetic"):
        raise ValueError("synthetic and market inputs cannot be silently mixed")
    funding_rows = list(funding["events"])
    previous_funding_key = None
    for event in funding_rows:
        key = (_time(event["timestamp"]), event["symbol"])
        if (key[1] not in spec.symbols or not spec.start_ms <= key[0] <= spec.end_ms
                or (previous_funding_key is not None and key <= previous_funding_key)
                or _number(event["mark_price"]) <= 0 or not -10000 < _number(event["signed_rate_bps"]) < 10000):
            raise ValueError("funding event bounds, ordering or duplicate settlement")
        previous_funding_key = key
    by_entry = {}
    previous_origin = None
    previous_window = None
    checked_decisions = []
    for decision in decisions:
        window = decision["window"]
        normalized = make_decision(
            origin_start_ms=window["origin_start_ms"], decision_asof_ms=window["decision_asof_ms"],
            execution_delay_ms=window["execution_delay_ms"], targets=decision["targets"],
            **{k: decision[k] for k in ("model_sha256", "scaler_sha256", "feature_prefix_sha256", "training_identity_sha256")},
        )
        if normalized != decision or set(decision["targets"]) != set(spec.symbols):
            raise ValueError("decision identity/window/universe substitution")
        entry = window["planned_entry_open_ms"]
        if previous_origin is None and window["origin_start_ms"] != spec.start_ms:
            raise ValueError("first decision must use the registered reporting origin")
        if (not spec.start_ms <= window["origin_start_ms"] < spec.terminal_ms
                or window["planned_exit_open_ms"] > spec.terminal_ms
                or entry in by_entry or (previous_origin is not None and
                                        window["origin_start_ms"] - previous_origin != HORIZON_MS)):
            raise ValueError("complete nonoverlapping 3-day deployments on one origin grid required")
        if previous_window is not None and (
                entry != previous_window["planned_exit_open_ms"]
                or window["gross_label_definition_sha256"] != previous_window["gross_label_definition_sha256"]):
            raise ValueError("varying delay or entry clock would change the 3-day economic horizon")
        previous_origin = window["origin_start_ms"]
        previous_window = window
        by_entry[entry] = normalized
        checked_decisions.append(normalized)
    if checked_decisions and checked_decisions[-1]["window"]["planned_exit_open_ms"] != spec.terminal_ms:
        raise ValueError("terminal must equal the last complete deployment exit")
    input_sha = identity({"spec": asdict(spec), "prices": bars, "decisions": checked_decisions,
                          "funding": funding, "costs": asdict(costs), "semantics": SEMANTICS})
    # Cash needs no future OHLCV observation after the one registered liquidation.
    steps = bars[:terminal_length] + [
        {"timestamp": spec.start_ms + i * BAR_MS, "prices": {}}
        for i in range(terminal_length, report_length)
    ]
    with localcontext() as ctx:
        ctx.prec = 50
        return _replay(spec, steps, by_entry, checked_decisions, funding_rows, funding, costs, input_sha)


def _replay(spec, bars, by_entry, decisions, funding_rows, funding, costs, input_sha):
    initial = cash = _number(spec.initial_capital_quote)
    tolerance = max(initial, Decimal(1)) * Decimal("1e-38")
    buy_fee, sell_fee, buy_drag, sell_drag = (
        _number(getattr(costs, n)) / 10000
        for n in ("buy_fee_bps", "sell_fee_bps", "buy_adverse_bps", "sell_adverse_bps")
    )
    positions, journal, trades, equity_rows, decision_rows = {}, [], [], [], []
    totals = {n: Decimal(0) for n in ("fees_quote", "adverse_drag_quote", "funding_quote", "gross_realized_quote", "turnover_quote")}
    peak = previous_nav = initial
    funding_index = 0
    minimum_cash, maximum_drawdown = initial, Decimal(0)

    def close(symbol, reference, timestamp, reason, decision=None):
        nonlocal cash
        position = positions.pop(symbol)
        q = position["quantity"]
        fill = reference * (1 - sell_drag)
        fee, drag = q * fill * sell_fee, q * (reference - fill)
        gross = q * (reference - position["reference_entry"])
        cash += q * fill - fee
        totals["fees_quote"] += fee
        totals["adverse_drag_quote"] += drag
        totals["gross_realized_quote"] += gross
        totals["turnover_quote"] += q * fill
        owner = decision["model_sha256"] if decision else position["last_model_sha256"]
        journal.append({"kind": "sell", "timestamp": timestamp, "symbol": symbol, "reason": reason,
                        "quantity": _text(q), "reference_price": _text(reference), "fill_price": _text(fill),
                        "fee_quote": _text(fee), "adverse_drag_quote": _text(drag),
                        "cash_after": _text(cash), "model_sha256": owner,
                        "entry_model_sha256": position["model_sha256"], "actual_fill_verified": False})
        net = gross - position["entry_fee"] - fee - position["entry_drag"] - drag - position["funding"]
        trades.append({"symbol": symbol, "entry_timestamp": position["entry_timestamp"],
                       "exit_timestamp": timestamp, "holding_bars": (timestamp - position["entry_timestamp"]) // BAR_MS,
                       "entry_model_sha256": position["model_sha256"], "exit_model_sha256": owner,
                       "quantity": _text(q), "gross_pnl_quote": _text(gross), "net_pnl_quote": _text(net),
                       "funding_quote": _text(position["funding"]), "reason": reason})

    for bar in bars:
        timestamp = bar["timestamp"]
        prices = {s: _number(p) for s, p in bar["prices"].items()}
        while funding_index < len(funding_rows) and funding_rows[funding_index]["timestamp"] <= timestamp:
            event = funding_rows[funding_index]
            funding_index += 1
            position = positions.get(event["symbol"])
            if position is not None and position["entry_timestamp"] < event["timestamp"]:
                debit = position["quantity"] * _number(event["mark_price"]) * _number(event["signed_rate_bps"]) / 10000
                cash -= debit
                if cash < -tolerance:
                    raise ValueError("funding exceeds cash reserve; unregistered borrowing is forbidden")
                cash = max(cash, Decimal(0))
                position["funding"] += debit
                totals["funding_quote"] += debit
                minimum_cash = min(minimum_cash, cash)
                journal.append({"kind": "funding", "timestamp": event["timestamp"], "symbol": event["symbol"],
                                "debit_quote": _text(debit), "cash_after": _text(cash),
                                "model_sha256": position["last_model_sha256"],
                                "entry_model_sha256": position["model_sha256"]})
        decision = by_entry.get(timestamp)
        if timestamp == spec.terminal_ms:
            for symbol in sorted(tuple(positions)):
                close(symbol, prices[symbol], timestamp, "global_terminal")
        elif decision is not None:
            weights = {s: _number(v) for s, v in decision["targets"].items()}
            for symbol in sorted(tuple(positions)):
                if weights[symbol] != positions[symbol]["target_weight"]:
                    close(symbol, prices[symbol], timestamp, "target_transition", decision)
            nav_after_sells = cash + sum(p["quantity"] * prices[s] for s, p in positions.items())
            requested = {s: weights[s] * nav_after_sells for s in spec.symbols if weights[s] > 0 and s not in positions}
            total_requested = sum(requested.values(), Decimal(0))
            scale = min(Decimal(1), cash / total_requested) if total_requested else Decimal(1)
            for symbol in sorted(requested):
                allocated = requested[symbol] * scale
                if allocated <= 0:
                    raise ValueError("positive target cannot open without cash")
                reference = prices[symbol]
                fill = reference * (1 + buy_drag)
                q = allocated / (fill * (1 + buy_fee))
                fee, drag = q * fill * buy_fee, q * (fill - reference)
                cash -= allocated
                cash = max(cash, Decimal(0)) if cash >= -tolerance else cash
                if cash < 0:
                    raise ValueError("cash borrowing is forbidden")
                totals["fees_quote"] += fee
                totals["adverse_drag_quote"] += drag
                totals["turnover_quote"] += q * fill
                positions[symbol] = {"quantity": q, "target_weight": weights[symbol],
                                     "reference_entry": reference, "entry_timestamp": timestamp,
                                     "entry_fee": fee, "entry_drag": drag, "funding": Decimal(0),
                                     "model_sha256": decision["model_sha256"],
                                     "last_model_sha256": decision["model_sha256"]}
                journal.append({"kind": "buy", "timestamp": timestamp, "symbol": symbol,
                                "quantity": _text(q), "allocated_quote": _text(allocated),
                                "reference_price": _text(reference), "fill_price": _text(fill),
                                "fee_quote": _text(fee), "adverse_drag_quote": _text(drag),
                                "cash_after": _text(cash), "model_sha256": decision["model_sha256"],
                                "actual_fill_verified": False})
            for position in positions.values():
                position["last_model_sha256"] = decision["model_sha256"]
            decision_rows.append({**decision, "entry_timestamp": timestamp,
                                  "requested_new_allocation_quote": _text(total_requested),
                                  "cash_scaling_factor": _text(scale), "cash_after": _text(cash),
                                  "quantities_after": {s: _text(positions[s]["quantity"]) if s in positions else "0" for s in spec.symbols}})
        exposure = sum((p["quantity"] * prices[s] for s, p in positions.items()), Decimal(0))
        nav = cash + exposure
        unrealized = sum((p["quantity"] * (prices[s] - p["reference_entry"]) for s, p in positions.items()), Decimal(0))
        attributed = initial + totals["gross_realized_quote"] + unrealized - totals["fees_quote"] - totals["adverse_drag_quote"] - totals["funding_quote"]
        if nav <= 0 or abs(nav - attributed) > tolerance:
            raise ValueError("insolvency or cash/gross/cost attribution mismatch")
        peak = max(peak, nav)
        drawdown = 1 - nav / peak
        maximum_drawdown = max(maximum_drawdown, drawdown)
        minimum_cash = min(minimum_cash, cash)
        equity_rows.append({"timestamp": timestamp, "return_bucket_timestamp": max(spec.start_ms, timestamp - 1),
                            "equity_quote": _text(nav), "cash_quote": _text(cash),
                            "exposure_quote": _text(exposure), "exposure_fraction": _text(exposure / nav),
                            "net_return": _text(nav / previous_nav - 1), "drawdown": _text(drawdown),
                            "unrealized_gross_quote": _text(unrealized),
                            "reconciliation_error_quote": _text(nav - attributed),
                            "quantities": {s: _text(positions[s]["quantity"]) if s in positions else "0" for s in spec.symbols}})
        previous_nav = nav
    if positions or funding_index != len(funding_rows):
        raise ValueError("terminal positions or unprocessed funding remain")
    final = _number(equity_rows[-1]["equity_quote"])
    if abs(sum((_number(t["net_pnl_quote"]) for t in trades), Decimal(0)) - (final - initial)) > tolerance:
        raise ValueError("closed-trade attribution does not match final shared capital")
    days = Decimal(spec.end_ms - spec.start_ms) / 86400000
    fills = sum(row["kind"] in {"buy", "sell"} for row in journal)
    result = {"schema": SCHEMA, "measurement_semantic_sha256": SEMANTIC_SHA256,
              "spec": asdict(spec), "cost_scenario": asdict(costs), "input_sha256": input_sha,
              "decisions_sha256": identity(decisions), "cost_scenario_sha256": identity(asdict(costs)),
              "funding_inputs_sha256": identity(funding),
              "funding_evidence_sha256": funding["evidence_sha256"], "reference_price_kind": REFERENCE_KIND,
              "actual_fill_verified": False, "safety": dict(SAFETY),
              "ledger": journal, "decisions": decision_rows, "closed_trades": trades, "equity": equity_rows,
              "metrics": {"initial_capital_quote": _text(initial), "final_capital_quote": _text(final),
                          "net_return": _text(final / initial - 1), "max_drawdown": _text(maximum_drawdown),
                          "net_log_growth": _text((final / initial).ln()),
                          "mean_exposure_fraction": _text(sum(
                              (_number(row["exposure_fraction"]) for row in equity_rows[:-1]), Decimal(0)
                          ) / Decimal(len(equity_rows) - 1)),
                          "minimum_cash_quote": _text(minimum_cash), "trades": len(trades), "fills": fills,
                          "fills_per_calendar_day": _text(Decimal(fills) / days),
                          "turnover_over_initial_capital": _text(totals["turnover_quote"] / initial),
                          **{k: _text(v) for k, v in totals.items()}}}
    result["output_sha256"] = identity(result)
    return result


def compare_policies(spec, *, price_bars, policies, funding, scenarios):
    """Same capital, prices, funding and reporting interval for every policy."""
    if not policies or not scenarios or len({s.name for s in scenarios}) != len(scenarios):
        raise ValueError("named policies and distinct scenarios required")
    # Materialize once so generators cannot give different data to competitors.
    price_bars = list(price_bars)
    policies = {name: list(decisions) for name, decisions in policies.items()}
    if any(not isinstance(name, str) or not name for name in policies):
        raise ValueError("policy names required")
    grids = [tuple(identity(d["window"]) for d in decisions) for decisions in policies.values()]
    if any(grid != grids[0] for grid in grids[1:]):
        raise ValueError("all competitors require the same origin/entry/label clock")
    return {scenario.name: {name: replay(spec, price_bars=price_bars, decisions=decisions,
                                        funding=funding, costs=scenario)
                            for name, decisions in policies.items()}
            for scenario in scenarios}


def _checked_result(result):
    if (result.get("schema") != SCHEMA or result.get("measurement_semantic_sha256") != SEMANTIC_SHA256
            or identity({k: v for k, v in result.items() if k != "output_sha256"}) != result.get("output_sha256")):
        raise ValueError("measurement result identity mismatch")


def daily_measurements(result):
    """Calendar-day returns over all reporting days, including cash days."""
    _checked_result(result)
    days = {}
    with localcontext() as ctx:
        ctx.prec = 50
        for row in result["equity"]:
            day = row["return_bucket_timestamp"] // 86400000 * 86400000
            days[day] = days.get(day, Decimal(1)) * (1 + _number(row["net_return"]))
        return [{"timestamp": t, "net_return": _text(value - 1), "net_bps": _text((value - 1) * 10000)}
                for t, value in sorted(days.items())]


def as_primus_simulation(result, *, market_state_by_entry=None, overlay_diagnostics=None, stress_result=None):
    """Bridge to existing fitness/bootstrap code with explicit shared-NAV units.

    Market states and proposal-opportunity diagnostics must come from their
    actual causal logs; missing states remain 'unknown', not invented regimes.
    """
    _checked_result(result)
    if stress_result is not None:
        _checked_result(stress_result)
    import pandas as pd
    from .simulator import PrimusSimulation
    initial = _number(result["metrics"]["initial_capital_quote"])
    portfolio = pd.DataFrame([{
        "timestamp": row["return_bucket_timestamp"], "net_return": float(row["net_return"]),
        "equity": float(_number(row["equity_quote"]) / initial),
        "drawdown": -float(row["drawdown"]),
    } for row in result["equity"]])
    states = market_state_by_entry or {}
    ledger = pd.DataFrame([{
        **trade, "direction": "LONG",
        "market_state": states.get((trade["entry_timestamp"], trade["symbol"]), "unknown"),
    } for trade in result["closed_trades"]], columns=(
        "entry_timestamp", "exit_timestamp", "symbol", "direction", "market_state",
        "holding_bars", "quantity", "gross_pnl_quote", "net_pnl_quote", "funding_quote",
        "entry_model_sha256", "exit_model_sha256", "reason",
    ))
    longs = sum(_number(weight) > 0 for decision in result["decisions"] for weight in decision["targets"].values())
    slots = len(result["decisions"]) * len(result["spec"]["symbols"])
    diagnostics = dict(overlay_diagnostics or {})
    diagnostics.update({"measurement_semantic_sha256": SEMANTIC_SHA256,
                        "measurement_output_sha256": result["output_sha256"],
                        "measurement_result": result,
                        "capital_accounting": "shared_cash_nav", "initial_capital_quote": float(initial),
                        "actual_fill_verified": False})
    if stress_result is not None:
        diagnostics["stress_measurement_result"] = stress_result
    return PrimusSimulation(ledger, portfolio, {"LONG": longs, "FLAT": slots - longs, "SHORT": 0},
                            result["metrics"]["fills"], diagnostics=diagnostics)
