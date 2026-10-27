"""Pure segregated-capital controls; no training, transport or runtime."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np
from . import cadence_matrix_v1 as c


def budget_metrics(source, fraction):
    """Recompute nonlinear risk/exposure from a fixed, unleveraged sleeve."""
    k = float(fraction)
    if not math.isfinite(k) or not 0 <= k <= 1:
        raise ValueError("capital fraction must be finite and in [0, 1]")
    initial = float(source["metrics"]["initial_capital"])
    reserve = (1 - k) * initial
    start, end = source["start_ms"], source["end_ms"]
    if initial <= 0 or start >= end or (end - start) % c.DAY_MS:
        raise ValueError("positive capital and whole calendar days required")
    calendar, peak, maximum_dd = {}, initial, 0.
    for row in source["equity"]:
        nav, cash = reserve + k * row["nav"], reserve + k * row["cash"]
        if not math.isfinite(nav) or not math.isfinite(cash) or nav <= 0 or cash < -1e-7:
            raise ValueError("invalid source capital/cash")
        peak = max(peak, nav)
        maximum_dd = max(maximum_dd, 1 - nav / peak)
        if row["phase"] == "calendar_bucket_end":
            if row["timestamp_ms"] in calendar:
                raise ValueError("duplicate calendar mark")
            calendar[row["timestamp_ms"]] = (nav, (nav - cash) / nav)
    expected = set(range(start, end + 1, c.BAR_MS))
    if set(calendar) != expected:
        raise ValueError("incomplete source calendar")
    navs = [initial] + [calendar[t][0] for t in range(start + c.DAY_MS, end + 1, c.DAY_MS)]
    daily = np.diff(np.log(navs))
    exposure = [calendar[t][1] for t in range(start, end, c.BAR_MS)]
    metrics = deepcopy(source["metrics"])
    for key in ("gross_pnl", "fees", "slippage", "funding", "turnover_initial_capital"):
        metrics[key] *= k
    metrics.update(final_capital=calendar[end][0], net_return=calendar[end][0] / initial - 1,
                   max_drawdown=maximum_dd, mean_exposure=float(np.mean(exposure)),
                   maximum_exposure=float(max(exposure)), daily_log_volatility=float(np.std(daily, ddof=0)),
                   reconciliation_error=calendar[end][0] - initial - metrics["gross_pnl"]
                   + metrics["fees"] + metrics["slippage"] - metrics["funding"],
                   fills=source["metrics"]["fills"] if k else 0)
    return c.plain({"metrics": metrics, "daily_log_returns": daily})


def calibrate(source, target, metric, evaluation_start_ms):
    """Solve a pre-evaluation budget constraint, never a return objective."""
    if source["end_ms"] >= evaluation_start_ms:
        raise ValueError("calibration must finish strictly before evaluation")
    if metric not in ("mean_exposure", "daily_log_volatility") or not math.isfinite(target) or target < 0:
        raise ValueError("invalid calibration constraint")
    full = budget_metrics(source, 1)["metrics"][metric]
    capped = target > full
    low, high = 0., 1.
    if target == 0:
        fraction = 0.
    elif capped:
        fraction = 1.
    else:
        for _ in range(64):
            middle = (low + high) / 2
            if budget_metrics(source, middle)["metrics"][metric] < target:
                low = middle
            else:
                high = middle
        fraction = (low + high) / 2
    achieved = budget_metrics(source, fraction)["metrics"][metric]
    if not capped and abs(achieved - target) > 1e-10:
        raise ValueError("budget calibration residual exceeds tolerance")
    return {"fraction": fraction, "metric": metric, "target": target, "achieved": achieved,
            "full_capital_value": full, "residual": achieved - target, "capped_unreachable": capped,
            "iterations": 0 if capped or target == 0 else 64,
            "calibration_end_ms": source["end_ms"], "evaluation_start_ms": evaluation_start_ms}


def transform(source, fraction, policy_sha256, source_pin):
    """Scale all actual modeled cashflows/units, retaining an untouched reserve."""
    computed = budget_metrics(source, fraction)
    k, initial = float(fraction), float(source["metrics"]["initial_capital"])
    reserve = (1 - k) * initial
    result = deepcopy(source)
    result.pop("sha256", None)
    result.update(schema="causal_capital_ledger/1", metrics=computed["metrics"],
                  daily_log_returns=computed["daily_log_returns"], source_pin=source_pin,
                  capital_fraction=k, cash_reserve=reserve, policy_sha256=policy_sha256,
                  no_new_model_fits=True, execution_is_assumed=True,
                  observed_execution_qualified=False,
                  target_semantics="new-entry total-NAV fraction; retain existing units, no rebalancing",
                  source_event_count=len(source["ledger"]), safety=c.SAFETY)
    peak, before_decision = initial, {}
    for old, row in zip(source["equity"], result["equity"]):
        row["nav"], row["cash"] = reserve + k * old["nav"], reserve + k * old["cash"]
        row["exposure"] = (row["nav"] - row["cash"]) / row["nav"]
        peak = max(peak, row["nav"])
        row["drawdown"] = 1 - row["nav"] / peak
        if old["phase"] in ("before_funding_and_decision", "after_modeled_funding"):
            before_decision[old["timestamp_ms"]] = old["nav"]
    for row in result["ledger"]:
        for key in ("quantity", "fee", "slippage", "cash_delta", "gross_pnl"):
            if key in row:
                row[key] *= k
        row["policy_sha256"] = policy_sha256
    for values in result["per_symbol"].values():
        for key in values:
            values[key] *= k
    for row in result["decisions"]:
        source_nav = before_decision[row["origin_ms"]]
        share = k * source_nav / (reserve + k * source_nav)
        row["source_targets_relative_to_active_sleeve"] = list(row["targets"])
        row["targets"] = [float(v * share) for v in row["targets"]]
        row["active_sleeve_share_before_decision"] = share
        row["policy_sha256"] = policy_sha256
    return c.plain(result)


def direct_checks():
    """Small synthetic pure checks; never call an execution engine."""
    checked = 0
    def check(condition):
        nonlocal checked
        if not condition:
            raise AssertionError("capital control pure regression")
        checked += 1
    source = {"start_ms": 0, "end_ms": 2 * c.DAY_MS,
              "metrics": {"initial_capital": 10000., "gross_pnl": 1000., "fees": 10.,
                          "slippage": 20., "funding": -10., "turnover_initial_capital": 2.,
                          "fills": 2, "decisions": 1},
              "equity": [], "decisions": [{"origin_ms": 0, "targets": [.5], "model_sha256": "a" * 64}],
              "ledger": [{"kind": "modeled_fill", "quantity": 1., "fee": 10., "slippage": 20.,
                          "cash_delta": -5030., "gross_pnl": 1000., "model_sha256": "a" * 64}],
              "per_symbol": {"X": {"gross_pnl": 1000., "fees": 10., "slippage": 20., "funding": -10.}}}
    source["equity"].append({"timestamp_ms": 0, "phase": "before_funding_and_decision",
                            "nav": 10000., "cash": 10000.})
    for t in range(0, source["end_ms"] + 1, c.BAR_MS):
        nav = 10000. if t < c.DAY_MS else (9000. if t < 2 * c.DAY_MS else 10960.)
        source["equity"].append({"timestamp_ms": t, "phase": "calendar_bucket_end", "nav": nav,
                                "cash": nav if t == 2 * c.DAY_MS else 5000.})
    before = c.m.identity(source)
    zero = transform(source, 0, "b" * 64, {})
    full = transform(source, 1, "b" * 64, {})
    half = transform(source, .5, "b" * 64, {})
    check(zero["metrics"]["net_return"] == zero["metrics"]["daily_log_volatility"] == 0)
    check(zero["metrics"]["mean_exposure"] == zero["metrics"]["max_drawdown"] == 0)
    check(full["metrics"]["final_capital"] == 10960.)
    check(half["metrics"]["final_capital"] == 10480.)
    check(abs(half["metrics"]["reconciliation_error"]) < 1e-9)
    check(half["ledger"][0]["quantity"] == .5 and half["ledger"][0]["fee"] == 5.)
    check(half["metrics"]["funding"] == -5.)
    check(all(r["cash"] >= half["cash_reserve"] for r in half["equity"]))
    check(half["decisions"][0]["targets"] == [.25])
    check(half["decisions"][0]["model_sha256"] == "a" * 64)
    check(c.m.identity(source) == before)
    for metric in ("mean_exposure", "daily_log_volatility"):
        target = budget_metrics(source, .37)["metrics"][metric]
        fitted = calibrate(source, target, metric, 3 * c.DAY_MS)
        check(abs(fitted["fraction"] - .37) < 1e-10)
        check(abs(fitted["residual"]) < 1e-10)
        check(calibrate(source, 10., metric, 3 * c.DAY_MS)["capped_unreachable"])
    for bad in (-.1, 1.1, float("nan")):
        try:
            budget_metrics(source, bad)
        except ValueError:
            check(True)
        else:
            check(False)
    try:
        calibrate(source, .1, "mean_exposure", source["end_ms"])
    except ValueError:
        check(True)
    else:
        check(False)
    # Volatility and drawdown must be recomputed, not multiplied post hoc.
    check(abs(half["daily_log_returns"][0] - math.log(.95)) < 1e-12)
    check(abs(half["metrics"]["daily_log_volatility"] - .5 * full["metrics"]["daily_log_volatility"]) > 1e-4)
    return {"checks_passed": checked, "real_data_fits": 0, "execution_replays": 0, "safety": c.SAFETY}
