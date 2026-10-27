"""Research search maximizes net log wealth; bootstrap is a separate audit.

Pure supplied-input calculations. No fitter, market I/O or runtime authority.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from typing import Mapping

from .unified_measurement_v1 import (
    BAR_MS, SAFETY, _checked_result, _number, build_ranking_key, daily_measurements, identity,
)

OBJECTIVE_NAME = "net_log_growth_constrained_drawdown/1"
DAY_MS = 86400000
RESEARCH_DRAWDOWN_BUDGETS = (10, 20, 30)


@dataclass(frozen=True)
class ProfitObjective:
    drawdown_budget_percent: int
    active_parameter_penalty_bps_per_100: str = "0"
    active_feature_penalty_bps: str = "0"
    max_fills_per_calendar_day: str | None = None
    scope: str = "research"
    name: str = OBJECTIVE_NAME

    def validate(self):
        if (self.scope != "research" or self.name != OBJECTIVE_NAME
                or type(self.drawdown_budget_percent) is not int
                or self.drawdown_budget_percent not in RESEARCH_DRAWDOWN_BUDGETS):
            raise ValueError("only registered research drawdown variants 10/20/30 percent supported")
        for name in ("active_parameter_penalty_bps_per_100", "active_feature_penalty_bps"):
            if _number(getattr(self, name)) < 0:
                raise ValueError("nonnegative explicitly registered parsimony penalties required")
        if self.max_fills_per_calendar_day is not None and _number(self.max_fills_per_calendar_day) < 0:
            raise ValueError("nonnegative optional operational fill limit required")
        return self

    @property
    def sha256(self):
        self.validate()
        payload = asdict(self)
        for name in ("active_parameter_penalty_bps_per_100", "active_feature_penalty_bps", "max_fills_per_calendar_day"):
            if payload[name] is not None:
                value = _number(payload[name])
                text = format(value, "f")
                payload[name] = "0" if value == 0 else (text.rstrip("0").rstrip(".") if "." in text else text)
        return identity(payload)


def research_objectives(**kwargs):
    """Evaluate all three fixed budgets on the SAME already-computed ledgers."""
    return tuple(ProfitObjective(budget, **kwargs).validate() for budget in RESEARCH_DRAWDOWN_BUDGETS)


def objectives_from_contract(contract):
    if (contract.get("schema") != "exia.genetic_primus.profit_fitness_contract/1"
            or contract.get("research_drawdown_budgets_percent") != list(RESEARCH_DRAWDOWN_BUDGETS)
            or contract.get("search_computes_lcb") is not False
            or contract.get("live_risk_budget_approved") is not False
            or set(contract.get("safety", {})) != set(SAFETY)
            or any(value is not False for value in contract["safety"].values())):
        raise ValueError("fixed research-only profit fitness contract required")
    return research_objectives(**dict(contract["objective_defaults"]))


def _capital_path(result):
    _checked_result(result)
    initial = _number(result["spec"]["initial_capital_quote"])
    rows = result["equity"]
    if len(rows) != (result["spec"]["end_ms"] - result["spec"]["start_ms"]) // BAR_MS + 1:
        raise ValueError("continuous shared-capital path required")
    if any(value is not False for value in result["safety"].values()):
        raise ValueError("research-only measurement required")
    peak = previous = initial
    maximum_drawdown = Decimal(0)
    tolerance = max(initial, Decimal(1)) * Decimal("1e-35")
    for i, row in enumerate(rows):
        nav = _number(row["equity_quote"])
        cash, exposure = _number(row["cash_quote"]), _number(row["exposure_quote"])
        if (row["timestamp"] != result["spec"]["start_ms"] + i * BAR_MS
                or nav <= 0 or min(cash, exposure) < 0 or abs(nav - cash - exposure) > tolerance
                or abs(_number(row["net_return"]) - (nav / previous - 1)) > Decimal("1e-35")):
            raise ValueError("invalid or mismatched shared-NAV return path")
        peak = max(peak, nav)
        maximum_drawdown = max(maximum_drawdown, 1 - nav / peak)
        previous = nav
    if (abs(previous - _number(result["metrics"]["final_capital_quote"])) > tolerance
            or abs(maximum_drawdown - _number(result["metrics"]["max_drawdown"])) > Decimal("1e-35")):
        raise ValueError("capital/drawdown summary does not match continuous path")
    return initial, previous, maximum_drawdown


def _same_inputs(first, second, *, same_decisions=False, same_costs=False):
    _checked_result(first)
    _checked_result(second)
    if (first["spec"] != second["spec"] or first["funding_inputs_sha256"] != second["funding_inputs_sha256"]
            or [d["window"] for d in first["decisions"]] != [d["window"] for d in second["decisions"]]
            or (same_decisions and first["decisions_sha256"] != second["decisions_sha256"])
            or (same_costs and first["cost_scenario_sha256"] != second["cost_scenario_sha256"])):
        raise ValueError("capital/prices/funding/clock or scenario comparison mismatch")


def _count(complexity, name, *, required=False, fallback=None):
    value = complexity.get(name, complexity.get(fallback) if fallback else None)
    if value is None:
        if required:
            raise ValueError("complexity evidence required for nonzero penalty: " + name)
        return 0
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer complexity required: " + name)
    return value


def score_net_growth(result, *, objective: ProfitObjective, complexity_metrics=None, stress_result=None):
    """No LCB, fold averages, gross returns or repeated commission penalties.

    Statistical validation NEVER changes this search ranking. Missing stress
    evidence is reported; it is not a zero-cost or zero-drawdown scenario.
    """
    objective.validate()
    complexity = dict(complexity_metrics or {})
    parameter_rate = _number(objective.active_parameter_penalty_bps_per_100)
    feature_rate = _number(objective.active_feature_penalty_bps)
    parameters = _count(complexity, "active_parameter_count", required=parameter_rate > 0, fallback="parameter_count")
    features = _count(complexity, "active_feature_count", required=feature_rate > 0)
    with localcontext() as ctx:
        ctx.prec = 50
        initial, final, normal_dd = _capital_path(result)
        days = Decimal(result["spec"]["end_ms"] - result["spec"]["start_ms"]) / DAY_MS
        raw_fitness = (final / initial).ln() * 10000 / days
        parameter_penalty = Decimal(parameters) / 100 * parameter_rate
        feature_penalty = Decimal(features) * feature_rate
        fitness = raw_fitness - parameter_penalty - feature_penalty
        fills = sum(event["kind"] in {"buy", "sell"} for event in result["ledger"])
        fills_per_day = Decimal(fills) / days
        if fills != result["metrics"]["fills"]:
            raise ValueError("fill count does not match ledger")
        budget = Decimal(objective.drawdown_budget_percent) / 100
        failures = []
        if normal_dd > budget:
            failures.append("normal_drawdown_budget")
        stress_dd = None
        if stress_result is not None:
            _same_inputs(result, stress_result, same_decisions=True)
            _, _, stress_dd = _capital_path(stress_result)
            if stress_dd > budget:
                failures.append("stress_drawdown_budget")
        if objective.max_fills_per_calendar_day is not None and fills_per_day > _number(objective.max_fills_per_calendar_day):
            failures.append("operational_fill_limit")
        maximum_dd = max(normal_dd, stress_dd if stress_dd is not None else normal_dd)
        metrics = {
            "search_objective": OBJECTIVE_NAME, "objective_sha256": objective.sha256,
            "objective_scope": "research", "drawdown_budget_percent": objective.drawdown_budget_percent,
            "initial_capital_quote": float(initial), "final_capital_quote": float(final),
            "net_return_pct": float((final / initial - 1) * 100), "calendar_days": float(days),
            "net_log_growth": float((final / initial).ln()), "raw_net_log_growth_bps_per_day": float(raw_fitness),
            "fitness": float(fitness), "fitness_units": "net_log_growth_bps_per_calendar_day",
            "parsimony_parameter_penalty_bps": float(parameter_penalty),
            "parsimony_feature_penalty_bps": float(feature_penalty), "cost_double_counting_penalty_bps": 0.0,
            "active_parameter_count": parameters, "active_feature_count": features,
            "trades": result["metrics"]["trades"], "turnover_fills": fills,
            "fills_per_day": float(fills_per_day), "max_drawdown": float(maximum_dd),
            "normal_max_drawdown": float(normal_dd), "stress_max_drawdown": float(stress_dd) if stress_dd is not None else None,
            "stress_evaluated": stress_result is not None, "failed_gates": list(failures),
            "search_failed_gates": list(failures), "search_risk_passed": not failures,
            "validation_failed_gates": ["stability_not_evaluated"], "validation_status": "not_evaluated",
            "temporal_gate_passed": False, "promotion_authority": False, "research_candidate_only": True,
            "measurement_output_sha256": result["output_sha256"],
            "ranking_semantics": "research_risk_gates_then_net_log_growth/1",
        }
        ranking = build_ranking_key(fitness=fitness, failed_gates=failures, max_drawdown=maximum_dd,
                                    fills_per_day=fills_per_day, active_parameter_count=parameters)
    return {"fitness": float(fitness), "ranking_key": ranking, "metrics": metrics,
            "failed_gates": tuple(failures)}


def _daily_arrays(result):
    import numpy as np
    if result["spec"]["start_ms"] % DAY_MS or result["spec"]["end_ms"] % DAY_MS:
        raise ValueError("bootstrap requires complete calendar days, not mixed partial-day samples")
    rows = daily_measurements(result)
    timestamps = np.array([r["timestamp"] for r in rows], dtype=np.int64)
    expected = np.arange(result["spec"]["start_ms"], result["spec"]["end_ms"], DAY_MS, dtype=np.int64)
    if not np.array_equal(timestamps, expected):
        raise ValueError("calendar days, including cash days, cannot be omitted")
    arithmetic = np.array([float(r["net_bps"]) for r in rows], dtype=float)
    log = np.log1p(arithmetic / 10000) * 10000
    if not np.all(np.isfinite(log)):
        raise ValueError("finite positive wealth increments required")
    return timestamps, arithmetic, log


def _check_bank(bank, result, folds, *, verify_content=True):
    from .measurement import _content_sha, folds_sha256
    if bank.cutoff_ms != result["spec"]["end_ms"] or bank.folds_sha256 != folds_sha256(folds):
        raise ValueError("fixed bootstrap cutoff/fold hash mismatch")
    if verify_content:
        base = {key: getattr(bank, key) for key in ("bank_id", "seed", "samples", "block_days", "cutoff_ms", "folds_sha256")}
        if _content_sha(base, bank.scope_timestamps, bank.scope_indices) != bank.content_sha256:
            raise ValueError("fixed resample bank content was modified")


def _lcb(values, timestamps, bank, scope, alpha):
    import numpy as np
    if not 0 < alpha < 0.5:
        raise ValueError("LCB alpha must lie between zero and one half")
    indices = bank.indices_for(scope, timestamps)
    if (len(values) == 0 or indices.shape != (bank.samples, len(values)) or bank.samples < 1
            or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(values))):
        raise ValueError("complete fixed resample bank required")
    return float(np.quantile(np.mean(values[indices], axis=1), alpha))


def growth_diagnostics(result, *, folds, bootstrap_bank=None, alpha=0.05):
    """Descriptive temporal folds are NOT outer validation or search gates."""
    import numpy as np
    import pandas as pd
    alpha_value = _number(alpha)
    if not 0 < alpha_value < Decimal("0.5"):
        raise ValueError("LCB alpha must lie between zero and one half")
    alpha = float(alpha_value)
    timestamps, arithmetic, log = _daily_arrays(result)
    if bootstrap_bank is not None:
        _check_bank(bootstrap_bank, result, folds)
    rows = []
    for fold in folds:
        start, end = (int(pd.Timestamp(fold[key]).timestamp() * 1000) for key in ("start", "end"))
        if start % DAY_MS or end % DAY_MS or not result["spec"]["start_ms"] <= start < end <= result["spec"]["end_ms"]:
            raise ValueError("nonempty complete-calendar diagnostic fold within reporting period required")
        selected = (timestamps >= start) & (timestamps < end)
        a, g, t = arithmetic[selected], log[selected], timestamps[selected]
        rows.append({"fold": str(fold["name"]), "start": fold["start"], "end": fold["end"],
                     "days": len(a), "trades": sum(start <= r["entry_timestamp"] < end for r in result["closed_trades"]),
                     "mean_daily_bps": float(np.mean(a)), "median_daily_bps": float(np.median(a)),
                     "mean_net_log_bps": float(np.mean(g)),
                     "lcb_daily_bps": _lcb(a, t, bootstrap_bank, str(fold["name"]), alpha) if bootstrap_bank is not None else None,
                     "net_log_lcb_daily_bps": _lcb(g, t, bootstrap_bank, str(fold["name"]), alpha) if bootstrap_bank is not None else None})
    lcbs = [r["lcb_daily_bps"] for r in rows if r["lcb_daily_bps"] is not None]
    return {"mean_daily_bps": float(np.mean(arithmetic)), "median_daily_bps": float(np.median(arithmetic)),
            "mean_net_log_bps": float(np.mean(log)),
            "lcb_daily_bps": _lcb(arithmetic, timestamps, bootstrap_bank, "overall", alpha) if bootstrap_bank is not None else None,
            "net_log_lcb_daily_bps": _lcb(log, timestamps, bootstrap_bank, "overall", alpha) if bootstrap_bank is not None else None,
            "positive_fold_means": sum(r["mean_daily_bps"] > 0 for r in rows),
            "worst_fold_mean_bps": min((r["mean_daily_bps"] for r in rows), default=0.0),
            "median_fold_lcb_bps": float(np.median(lcbs)) if lcbs else None,
            "worst_fold_lcb_bps": min(lcbs) if lcbs else None,
            "folds": rows, "fold_role": "diagnostic_not_outer_slices",
            "bootstrap_alpha": alpha, "measurement_output_sha256": result["output_sha256"],
            "stability_bootstrap_evaluated": bootstrap_bank is not None,
            "bootstrap_bank": bootstrap_bank.metadata() if bootstrap_bank is not None else None}


def check_profit_stability(normal, *, stress, folds, selection_bank, audit_bank,
                           incumbent=None, stress_incumbent=None, alpha=0.05):
    """Top-K statistical check with independent audit seed; not promotion.

    These LCBs are never inputs to the search score. Default paired reference
    is explicit NoTrade (zero log growth on the same calendar).
    """
    import numpy as np
    alpha = float(_number(alpha))
    if not 0 < alpha < 0.5:
        raise ValueError("LCB alpha must lie between zero and one half")
    if stress is None or selection_bank is None or audit_bank is None:
        raise ValueError("normal/stress and two fixed independent seed banks required")
    _same_inputs(normal, stress, same_decisions=True)
    if (selection_bank.seed == audit_bank.seed or selection_bank.content_sha256 == audit_bank.content_sha256
            or selection_bank.samples != audit_bank.samples or selection_bank.block_days != audit_bank.block_days):
        raise ValueError("independent audit seed bank on the same resampling protocol required")
    if (incumbent is None) != (stress_incumbent is None):
        raise ValueError("paired incumbent requires both cost scenarios")
    if incumbent is not None:
        _same_inputs(normal, incumbent, same_costs=True)
        _same_inputs(stress, stress_incumbent, same_costs=True)
    failures, banks = [], {}
    for name, bank in (("selection", selection_bank), ("audit", audit_bank)):
        n = growth_diagnostics(normal, folds=folds, bootstrap_bank=bank, alpha=alpha)
        s = growth_diagnostics(stress, folds=folds, bootstrap_bank=bank, alpha=alpha)
        paired = {}
        for scenario, challenger, reference in (("normal", normal, incumbent), ("stress", stress, stress_incumbent)):
            timestamps, _, values = _daily_arrays(challenger)
            if reference is not None:
                rt, _, reference_values = _daily_arrays(reference)
                if not np.array_equal(timestamps, rt):
                    raise ValueError("paired calendar mismatch")
            else:
                reference_values = np.zeros(len(values))
            paired[scenario] = _lcb(values - reference_values, timestamps, bank, "overall", alpha)
            if paired[scenario] <= 0:
                failures.append(name + "_" + scenario + "_paired_log_lcb")
        for scenario, diagnostic in (("normal", n), ("stress", s)):
            if diagnostic["net_log_lcb_daily_bps"] <= 0:
                failures.append(name + "_" + scenario + "_absolute_log_lcb")
        banks[name] = {"normal": n, "stress": s, "paired_log_lcb_bps": paired}
    return {"status": "passed" if not failures else "failed", "failed_gates": failures,
            "statistic": "mean_calendar_daily_net_log_growth_bps_and_paired_log_difference",
            "paired_reference": "incumbent" if incumbent is not None else "NoTrade",
            "bootstrap_alpha": float(_number(alpha)),
            "normal_measurement_output_sha256": normal["output_sha256"],
            "stress_measurement_output_sha256": stress["output_sha256"],
            "incumbent_measurement_output_sha256": incumbent["output_sha256"] if incumbent is not None else None,
            "stress_incumbent_measurement_output_sha256": stress_incumbent["output_sha256"] if stress_incumbent is not None else None,
            "banks": banks, "affects_search_ranking": False, "promotion_authority": False}


def score_profit_simulation(simulation, *, config, folds, bootstrap_bank, complexity_metrics=None):
    """Existing scorer integration; explicit new objective and verified ledger."""
    import pandas as pd
    from .unified_measurement_v1 import as_primus_simulation
    objective = ProfitObjective(**dict(config["search_objective"])).validate()
    result = simulation.diagnostics.get("measurement_result")
    if not isinstance(result, Mapping):
        raise ValueError("net-growth objective requires the verified unified ledger, not legacy return sums")
    expected = as_primus_simulation(result)
    required = ["timestamp", "net_return", "equity", "drawdown"]
    if (not simulation.portfolio.loc[:, required].equals(expected.portfolio.loc[:, required])
            or simulation.turnover_fills != expected.turnover_fills):
        raise ValueError("simulation differs from its verified shared-capital measurement")
    payload = score_net_growth(result, objective=objective, complexity_metrics=complexity_metrics,
                               stress_result=simulation.diagnostics.get("stress_measurement_result"))
    compute = config.get("compute_stability_diagnostics", False)
    if type(compute) is not bool:
        raise ValueError("explicit boolean compute_stability_diagnostics required")
    if compute and bootstrap_bank is None:
        raise ValueError("stability diagnostics require a common fixed bootstrap bank")
    if bootstrap_bank is not None:
        _check_bank(bootstrap_bank, result, folds, verify_content=compute)
        if (bootstrap_bank.samples != config["bootstrap_samples"]
                or bootstrap_bank.block_days != config["bootstrap_block_days"]):
            raise ValueError("bootstrap configuration mismatch")
    diagnostic = growth_diagnostics(result, folds=folds, bootstrap_bank=bootstrap_bank if compute else None,
                                    alpha=config.get("bootstrap_alpha", 0.05))
    payload["folds"] = pd.DataFrame(diagnostic.pop("folds"))
    payload["metrics"].update(diagnostic)
    payload["metrics"]["selection_bank_identity"] = bootstrap_bank.metadata() if bootstrap_bank is not None else None
    if compute:
        passed = diagnostic["net_log_lcb_daily_bps"] > 0
        payload["metrics"].update({"validation_status": "selection_diagnostic_only",
                                    "validation_failed_gates": [] if passed else ["selection_normal_log_lcb"],
                                    "independent_audit_complete": False})
    return payload
