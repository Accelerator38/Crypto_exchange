"""Deterministic forecast/position separation. No transport, async or runtime."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import math
import numpy as np
from . import cadence_matrix_v1 as c

POLICIES = ("C", "C_H", "C_V", "EMA_V", "Passive_V", "EMA_M", "Passive_M")


def hysteresis(prediction, held, entry_bps=48., exit_bps=0.):
    value, held = np.asarray(prediction, float), np.asarray(held, bool)
    if value.shape != held.shape or not np.isfinite(value).all() or entry_bps <= exit_bps:
        raise ValueError("finite predictions and separated thresholds required")
    return np.where(held, value * 10000 > exit_bps, value * 10000 > entry_bps)


def risk_covariance(times, prices, decision_ms, lookback=60):
    """Daily close returns ending strictly BEFORE the decision day."""
    grid = np.arange(decision_ms - (lookback + 1) * c.DAY_MS, decision_ms, c.DAY_MS)
    indices = np.searchsorted(times, grid)
    if (len(grid) != lookback + 1 or np.any(indices >= len(times))
            or not np.array_equal(times[indices], grid)):
        raise ValueError("complete lagged daily risk history required")
    panel = np.asarray(prices[indices], float)
    if not np.isfinite(panel).all() or np.any(panel <= 0):
        raise ValueError("finite positive risk prices required")
    returns = np.diff(np.log(panel), axis=0)
    sample = np.cov(returns, rowvar=False, ddof=1)
    covariance = .8 * sample + .2 * np.diag(np.diag(sample))
    covariance += np.diag(np.maximum(.002 ** 2 - np.diag(covariance), 0))
    return covariance, {"risk_last_data_ms": int(grid[-1]), "risk_observations": lookback,
        "risk_history_sha256": c.m.array_identity(grid, panel), "risk_covariance_sha256": c.m.array_identity(covariance)}


def bounded_weights(mask, covariance, budget=.005, requested_gross=None):
    mask, covariance = np.asarray(mask, bool), np.asarray(covariance, float)
    if covariance.shape != (len(mask), len(mask)) or not np.isfinite(covariance).all() or budget <= 0:
        raise ValueError("finite matched covariance and positive risk budget required")
    if np.max(abs(covariance - covariance.T)) > 1e-10 or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("positive-semidefinite risk covariance required")
    gross = min(.8, .1 * int(mask.sum())) if requested_gross is None else min(.8, float(requested_gross), .1 * int(mask.sum()))
    if not math.isfinite(gross) or gross < 0:
        raise ValueError("nonnegative requested gross required")
    weights = mask.astype(float) * (gross / mask.sum() if mask.any() else 0)
    risk = float(np.sqrt(max(0, weights @ covariance @ weights)))
    if risk > budget:
        weights *= budget / risk
    return weights


def decision_policies(saved_rows, times, prices, features, policy_registration_sha):
    result = {name: [] for name in POLICIES}
    held = np.zeros(prices.shape[1], dtype=bool)
    for saved in saved_rows:
        t = int(saved["origin_ms"])
        if saved["fit_ms"] > t or saved.get("cutoff_ms", t) > t:
            raise ValueError("future fitted forecast")
        prediction = np.asarray(saved["prediction"], float)
        mask = prediction * 10000 > 20
        held = hysteresis(prediction, held)
        index = np.searchsorted(times, t)
        if index >= len(times) or times[index] != t:
            raise ValueError("missing decision feature")
        ema = features[index, :, 0] > 0
        covariance, risk = risk_covariance(times, prices, t)
        cv = bounded_weights(mask, covariance)
        masks = {"C": mask, "C_H": held, "C_V": mask, "EMA_V": ema,
                 "Passive_V": np.ones(len(mask), bool), "EMA_M": ema, "Passive_M": np.ones(len(mask), bool)}
        weights = {"C": .1 * mask, "C_H": .1 * held, "C_V": cv,
                   "EMA_V": bounded_weights(ema, covariance),
                   "Passive_V": bounded_weights(masks["Passive_V"], covariance),
                   "EMA_M": bounded_weights(ema, covariance, requested_gross=float(cv.sum())),
                   "Passive_M": bounded_weights(masks["Passive_M"], covariance, requested_gross=float(cv.sum()))}
        for name in POLICIES:
            row = deepcopy(saved)
            if name.startswith(("EMA", "Passive")):
                row.update(model_sha256=c.m.identity({"simple_agent": name.split("_")[0],
                    "feature_definition": c.d.FEATURE_SHA256}), fit_ms=t, model_age_days=None,
                    fit_artifact=None, scaler_sha256=None, training_identity_sha256=None, cutoff_ms=None)
                row.pop("prediction", None)
            row.update(policy=name, targets=weights[name].tolist(), signal_mask=masks[name].tolist(),
                policy_sha256=c.m.identity({"registration_sha256": policy_registration_sha, "policy": name}),
                risk_requested_gross=float(cv.sum()) if name.endswith("_M") else None,
                ex_ante_daily_volatility=float(np.sqrt(max(0, weights[name] @ covariance @ weights[name]))),
                **risk)
            result[name].append(row)
    return c.plain(result)


def replay(*, times, prices, start, end, symbols, decisions, rates, fee_bps, drag_bps, rebalance):
    """Long-only cash ledger; charge costs on delta units, never on holding."""
    if not rebalance:
        return c.replay(times=times, prices=prices, start=start, end=end, symbols=symbols,
            decisions=decisions, rates=rates, fee_bps=fee_bps, drag_bps=drag_bps, decision_cadence_days=1)
    times, prices = np.asarray(times), np.asarray(prices, float)
    if (prices.shape != (len(times), len(symbols)) or np.any(np.diff(times) != c.BAR_MS)
            or not np.isfinite(prices).all() or np.any(prices <= 0) or not 0 <= fee_bps <= 100 or not 0 <= drag_bps <= 100):
        raise ValueError("complete finite price panel and bounded costs required")
    if [r["origin_ms"] for r in decisions] != list(range(start, end, c.DAY_MS)):
        raise ValueError("complete daily decision grid required")
    decision_map = {r["origin_ms"]: r for r in decisions}
    for row in decisions:
        w = np.asarray(row["targets"], float)
        if (w.shape != (len(symbols),) or not np.isfinite(w).all() or np.any(w < 0)
                or np.any(w > .1 + 1e-12) or w.sum() > .8 + 1e-12 or row["fit_ms"] > row["origin_ms"]
                or row["risk_last_data_ms"] >= row["origin_ms"] or row["ex_ante_daily_volatility"] > .005 + 1e-10):
            raise ValueError("causal long-only risk target required")
    symbol_map = {s.replace("/", ""): i for i, s in enumerate(symbols)}
    funding_map, seen = defaultdict(list), set()
    for row in rates:
        t, symbol = int(row["settlement_ms"]), row["symbol"].replace("/", "")
        if start <= t <= end and symbol in symbol_map:
            if (t, symbol) in seen or t % c.BAR_MS or not math.isfinite(float(row["rate_fraction"])):
                raise ValueError("duplicate or invalid funding")
            seen.add((t, symbol))
            funding_map[t].append(row)
    segment = np.flatnonzero((times >= start) & (times <= end))
    if len(segment) != (end - start) // c.BAR_MS + 1 or times[segment[0]] != start or times[segment[-1]] != end:
        raise ValueError("incomplete evaluation calendar")
    cash, quantity, basis = 10000., np.zeros(len(symbols)), np.zeros(len(symbols))
    fees = slippage = fund_total = gross = notional = 0.
    peak, maximum_dd = cash, 0.
    ledger, equity, decision_ledger = [], [], []
    active_model, active_policy = c.m.identity({"initial_no_position": True}), decisions[0]["policy_sha256"]
    per_symbol = {s: {"gross_pnl": 0., "fees": 0., "slippage": 0., "funding": 0.} for s in symbols}
    def mark(t, px, phase):
        nonlocal peak, maximum_dd
        nav = cash + float(quantity @ px)
        if not np.isfinite(nav) or nav <= 0 or cash < -1e-7 or np.any(quantity < -1e-12):
            raise ValueError("no borrowing/shorting or nonpositive capital allowed")
        peak = max(peak, nav)
        maximum_dd = max(maximum_dd, 1 - nav / peak)
        equity.append({"timestamp_ms": t, "phase": phase, "nav": nav, "cash": cash,
                       "exposure": float(quantity @ px) / nav, "drawdown": 1 - nav / peak})
    for index in segment:
        t, px = int(times[index]), prices[index]
        mark(t, px, "before_funding_and_decision")
        for rate in sorted(funding_map.get(t, []), key=lambda r: r["symbol"]):
            i = symbol_map[rate["symbol"].replace("/", "")]
            amount = -quantity[i] * px[i] * float(rate["rate_fraction"])
            cash += amount
            fund_total += amount
            per_symbol[symbols[i]]["funding"] += amount
            if quantity[i]:
                ledger.append({"timestamp_us": t * 1000, "kind": "modeled_funding", "symbol": symbols[i],
                    "cash_delta": amount, "quantity": float(quantity[i]), "model_sha256": active_model,
                    "policy_sha256": active_policy})
            mark(t, px, "after_modeled_funding")
        row = decision_map.get(t)
        if row is None and t != end:
            mark(t, px, "calendar_bucket_end")
            continue
        if row:
            targets = np.asarray(row["targets"], float)
            active_model, active_policy = row["model_sha256"], row["policy_sha256"]
            decision_ledger.append(row | {"signal_us": t * 1000 + 100000, "model_fill_us": t * 1000 + 350000})
        else:
            targets = np.zeros(len(symbols))
        nav = cash + float(quantity @ px)
        # Desired market units use cost-inclusive entry budgets, as in frozen C.
        desired = nav * targets / (px * (1 + drag_bps / 10000) * (1 + fee_bps / 10000))
        deltas = desired - quantity
        operations = [(i, "SELL") for i in range(len(symbols)) if deltas[i] < -1e-12]
        operations += [(i, "BUY") for i in range(len(symbols)) if deltas[i] > 1e-12]
        for i, side in operations:
            q = abs(float(deltas[i]))
            execution_price = px[i] * (1 + (1 if side == "BUY" else -1) * drag_bps / 10000)
            fee = q * execution_price * fee_bps / 10000
            if side == "BUY":
                delta, realized = -q * execution_price - fee, 0.
                quantity[i] += q
                basis[i] += q * px[i]
            else:
                delta = q * execution_price - fee
                removed = basis[i] * min(1., q / quantity[i])
                realized = q * px[i] - removed
                quantity[i] -= q
                basis[i] -= removed
                if quantity[i] < 1e-12:
                    quantity[i], basis[i] = 0., 0.
            slip = q * abs(execution_price - px[i])
            cash += delta
            fees += fee
            slippage += slip
            gross += realized
            notional += q * px[i]
            for key, value in (("gross_pnl", realized), ("fees", fee), ("slippage", slip)):
                per_symbol[symbols[i]][key] += value
            ledger.append({"timestamp_us": t * 1000 + 350000, "kind": "modeled_fill", "side": side,
                "symbol": symbols[i], "quantity": q, "reference_close": float(px[i]), "price": float(execution_price),
                "fee": fee, "slippage": slip, "cash_delta": delta, "gross_pnl": realized,
                "model_sha256": active_model, "policy_sha256": active_policy})
            mark(t, px, "after_modeled_fill")
        mark(t, px, "calendar_bucket_end")
    error = cash - 10000 - gross + fees + slippage - fund_total
    if np.any(quantity) or abs(error) > 1e-7:
        raise ValueError("terminal cash/units reconciliation")
    calendar = {r["timestamp_ms"]: r for r in equity if r["phase"] == "calendar_bucket_end"}
    daily_nav = [10000.] + [calendar[t]["nav"] for t in range(start + c.DAY_MS, end + 1, c.DAY_MS)]
    exposures = [calendar[t]["exposure"] for t in range(start, end, c.BAR_MS)]
    return c.plain({"schema": "position_policy_ledger/1", "start_ms": start, "end_ms": end,
        "execution_is_assumed": True, "observed_execution_qualified": False, "terminal_fill_us": end * 1000 + 350000,
        "metrics": {"initial_capital": 10000., "final_capital": cash, "net_return": cash / 10000 - 1,
            "max_drawdown": maximum_dd, "gross_pnl": gross, "fees": fees, "slippage": slippage, "funding": fund_total,
            "turnover_initial_capital": notional / 10000, "fills": sum(r["kind"] == "modeled_fill" for r in ledger),
            "mean_exposure": float(np.mean(exposures)), "maximum_exposure": float(max(exposures)),
            "decisions": len(decision_ledger), "reconciliation_error": error},
        "daily_log_returns": np.diff(np.log(daily_nav)), "equity": equity, "ledger": ledger,
        "decisions": decision_ledger, "per_symbol": per_symbol, "safety": c.SAFETY})


def attribution(item, times, prices, symbols):
    """Exact accounting, not causal counterfactual alpha attribution."""
    start, end = item["start_ms"], item["end_ms"]
    indices = np.flatnonzero((times >= start) & (times <= end))
    calendar = {r["timestamp_ms"]: r for r in item["equity"] if r["phase"] == "calendar_bucket_end"}
    events, decisions = defaultdict(list), {r["origin_ms"]: r for r in item["decisions"]}
    for row in item["ledger"]:
        t = row["timestamp_us"] // (c.BAR_MS * 1000) * c.BAR_MS
        events[t].append(row)
    symbol_map = {s: i for i, s in enumerate(symbols)}
    quantity, signal, last_price, last_nav = np.zeros(len(symbols)), np.zeros(len(symbols)), None, 10000.
    opened, durations = {}, []
    totals = dict(market_presence=0., timing=0., asset_selection=0., sizing=0., fees=0., slippage=0., funding=0., net=0.)
    monthly = defaultdict(lambda: {k: 0. for k in totals})
    for index in indices:
        t, px = int(times[index]), prices[index]
        flow = {k: 0. for k in totals}
        if last_price is not None:
            returns = px / last_price - 1
            market = float(np.mean(returns))
            base = .1 * signal
            flow.update(market_presence=last_nav * .8 * market,
                        timing=last_nav * (base.sum() - .8) * market,
                        asset_selection=last_nav * float((base - base.sum() / len(symbols)) @ returns),
                        sizing=float((quantity * last_price - last_nav * base) @ returns))
        for event in events[t]:
            i = symbol_map[event["symbol"]]
            if event["kind"] == "modeled_funding":
                flow["funding"] += event["cash_delta"]
            else:
                flow["fees"] -= event["fee"]
                flow["slippage"] -= event["slippage"]
                old_quantity = quantity[i]
                quantity[i] += event["quantity"] * (1 if event["side"] == "BUY" else -1)
                if old_quantity <= 1e-10 and quantity[i] > 1e-10:
                    opened[i] = t
                if old_quantity > 1e-10 and quantity[i] <= 1e-10:
                    durations.append({"symbol": symbols[i], "entry_ms": opened.pop(i), "exit_ms": t})
                    quantity[i] = 0.
        if t in decisions:
            signal = np.asarray(decisions[t]["signal_mask"], float)
        flow["net"] = sum(flow[k] for k in totals if k != "net")
        current_nav = calendar[t]["nav"]
        if abs(current_nav - last_nav - flow["net"]) > 1e-7:
            raise ValueError("bar attribution reconciliation")
        key = datetime.fromtimestamp(min(max(t - 1, start), end - 1) / 1000, timezone.utc).strftime("%Y-%m")
        for name in totals:
            totals[name] += flow[name]
            monthly[key][name] += flow[name]
        last_price, last_nav = px, current_nav
    if opened or abs(totals["net"] - item["metrics"]["final_capital"] + 10000) > 1e-7:
        raise ValueError("attribution terminal reconciliation")
    holding = [(d["exit_ms"] - d["entry_ms"]) / c.DAY_MS for d in durations]
    daily = np.asarray(item["daily_log_returns"])
    best = max((float(daily[i:i + 3].sum()) for i in range(len(daily) - 2)), default=0.)
    asset_net = {s: v["gross_pnl"] - v["fees"] - v["slippage"] + v["funding"] for s, v in item["per_symbol"].items()}
    return c.plain({"components_USDT": totals, "calendar_months": dict(monthly), "per_asset_net_USDT": asset_net,
        "holding_episodes": durations, "mean_holding_days": float(np.mean(holding)) if holding else 0.,
        "median_holding_days": float(np.median(holding)) if holding else 0., "episodes": len(holding),
        "best_3day_log_sum": best, "return_without_best_3day": math.expm1(float(daily.sum()) - best),
        "accounting_not_investable_counterfactual": True, "safety": c.SAFETY})


def direct_checks():
    count = 0
    def check(value):
        nonlocal count
        if not value:
            raise AssertionError("position policy regression")
        count += 1
    check(hysteresis([.0048, .00481, 0, .0001], [False, False, True, True]).tolist() == [False, True, False, True])
    covariance = np.eye(8) * .04 ** 2
    w = bounded_weights(np.ones(8, bool), covariance)
    check(abs(np.sqrt(w @ covariance @ w) - .005) < 1e-12)
    check(w.max() <= .1 and w.sum() <= .8)
    check(bounded_weights(np.zeros(8, bool), covariance).sum() == 0)
    check(bounded_weights(np.ones(8, bool), covariance, requested_gross=.2).sum() <= .2)
    check(abs((np.ones(8) * .1) @ np.eye(8) @ np.ones(8) * .1 - .08) < 1e-12)
    times = np.arange(0, 72 * c.DAY_MS, c.BAR_MS, dtype=np.int64)
    prices = np.tile(np.exp(np.arange(len(times))[:, None] * .0001), (1, 8))
    cov, identity = risk_covariance(times, prices, 65 * c.DAY_MS)
    altered = prices.copy()
    altered[times >= 65 * c.DAY_MS] *= 2
    other, other_identity = risk_covariance(times, altered, 65 * c.DAY_MS)
    check(np.array_equal(cov, other) and identity == other_identity)
    check(identity["risk_last_data_ms"] == 64 * c.DAY_MS)
    check(np.linalg.eigvalsh(cov).min() >= .002 ** 2 - 1e-12)
    for invalid in (np.eye(8) * -1, np.full((8, 8), np.nan)):
        try:
            bounded_weights(np.ones(8, bool), invalid)
        except ValueError:
            check(True)
        else:
            check(False)
    return {"checks_passed": count, "real_data_fits": 0, "execution_replays": 0, "safety": c.SAFETY}


def ledger_checks():
    """Tiny numerical cash-ledger regressions, never an exchange/collector."""
    times = np.arange(0, 4 * c.DAY_MS + c.BAR_MS, c.BAR_MS, dtype=np.int64)
    prices, symbols = np.full((len(times), 8), 100.), [f"T{i}/USDT" for i in range(8)]
    decisions = [{"origin_ms": t, "fit_ms": t, "targets": [.1] * 8, "signal_mask": [True] * 8,
        "model_sha256": "a" * 64, "policy_sha256": "b" * 64,
        "risk_last_data_ms": t - c.DAY_MS, "ex_ante_daily_volatility": .005}
        for t in range(0, 4 * c.DAY_MS, c.DAY_MS)]
    args = dict(times=times, prices=prices, start=0, end=4 * c.DAY_MS, symbols=symbols,
                decisions=decisions, rates=[], fee_bps=6, drag_bps=8)
    hold = replay(**args, rebalance=False)
    sized = replay(**args, rebalance=True)
    if hold["metrics"]["fills"] != 16:
        raise AssertionError("holding must not charge daily entry commissions")
    checks = 1
    for item in (hold, sized):
        for event in item["ledger"]:
            if abs(event["fee"] - event["quantity"] * event["price"] * .0006) > 1e-10:
                raise AssertionError("commission must apply to actual delta units")
            checks += 1
        detail = attribution(item, times, prices, symbols)
        if (abs(detail["components_USDT"]["net"] - item["metrics"]["final_capital"] + 10000) > 1e-8
                or detail["episodes"] != 8 or detail["mean_holding_days"] != 4):
            raise AssertionError("cash/holding attribution")
        checks += 3
    bad = deepcopy(decisions)
    bad[0]["risk_last_data_ms"] = 0
    try:
        replay(**(args | {"decisions": bad}), rebalance=True)
    except ValueError:
        checks += 1
    else:
        raise AssertionError("future risk history not rejected")
    return {"checks_passed": checks, "synthetic_ledger_replays": 2,
            "real_data_fits": 0, "collector": False, "safety": c.SAFETY}
