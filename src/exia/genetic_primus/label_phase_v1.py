"""Bounded offline diagnostics; no search, network, collector or runtime."""
from __future__ import annotations
import numpy as np
from . import cadence_matrix_v1 as c


def phase_anchor(anchor, phase):
    if type(anchor) is not int or anchor % c.DAY_MS or type(phase) is not int or phase not in (0, 1, 2):
        raise ValueError("aligned frozen anchor and phase 0/1/2 required")
    return anchor + phase * c.DAY_MS


def describe(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty diagnostic sample required")
    return {"count": len(values), "mean": float(values.mean()), "std_ddof0": float(values.std()),
            "quantiles": dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                                  np.quantile(values, (0, .05, .25, .5, .75, .95, 1)).tolist())),
            "positive_fraction": float(np.mean(values > 0))}


def coefficient_view(model):
    p, s = np.asarray(model["parameters"]), model["scaler"]
    if model["architecture"] != "linear6" or p.shape != (7,):
        raise ValueError("frozen seven-parameter model required")
    slopes = p[:6] * s["target_scale"] / np.asarray(s["scale"])
    intercept = s["target_mean"] + s["target_scale"] * p[6] - slopes @ np.asarray(s["mean"])
    return {"standardized_target_coefficients": p.tolist(), "unclipped_raw_slopes": slopes.tolist(),
            "unclipped_raw_intercept": float(intercept),
            "slopes_bps_per_one_train_std": (p[:6] * s["target_scale"] * 10000).tolist(),
            "raw_formula_valid_only_inside_scaler_clip": True}


def bar_components(replay, times, prices, symbols):
    """Exact additive USDT mark-to-market attribution, not a new backtest."""
    times, prices = np.asarray(times), np.asarray(prices)
    start, end = replay["start_ms"], replay["end_ms"]
    positions = np.flatnonzero((times >= start) & (times <= end))
    ts, px = times[positions], prices[positions]
    if not len(ts) or ts[0] != start or ts[-1] != end or np.any(np.diff(ts) != c.BAR_MS):
        raise ValueError("complete replay price clock required")
    symbol_index = {s: i for i, s in enumerate(symbols)}
    event_map = {}
    for row in replay["ledger"]:
        event_ms = row["timestamp_us"] // 1000
        boundary = event_ms // c.BAR_MS * c.BAR_MS
        if boundary < start or boundary > end:
            raise ValueError("out-of-period ledger event")
        event_map.setdefault(boundary, []).append(row)
    values = {name: np.zeros((len(ts), len(symbols))) for name in ("gross", "fees", "slippage", "funding")}
    q, previous = np.zeros(len(symbols)), px[0]
    for j, t in enumerate(ts):
        values["gross"][j] = q * (px[j] - previous)
        for row in event_map.get(int(t), []):
            i = symbol_index[row["symbol"]]
            if row["kind"] == "modeled_funding":
                if not np.isclose(q[i], row["quantity"], rtol=1e-12, atol=1e-12):
                    raise ValueError("funding must precede fills on its boundary")
                values["funding"][j, i] += row["cash_delta"]
            elif row["kind"] == "modeled_fill":
                values["fees"][j, i] -= row["fee"]
                values["slippage"][j, i] -= row["slippage"]
                if row["side"] == "BUY":
                    if q[i] != 0:
                        raise ValueError("unexpected duplicate buy")
                    q[i] = row["quantity"]
                elif row["side"] == "SELL":
                    if not np.isclose(q[i], row["quantity"], rtol=1e-12, atol=1e-12):
                        raise ValueError("sell quantity mismatch")
                    q[i] = 0
                else:
                    raise ValueError("unknown fill side")
            else:
                raise ValueError("unknown ledger event")
        previous = px[j]
    net = sum(values.values())
    error = float(net.sum() - (replay["metrics"]["final_capital"] - 10000))
    if np.any(q) or abs(error) > 1e-7:
        raise ValueError("bar contribution reconciliation failed")
    for i, symbol in enumerate(symbols):
        detail = replay["per_symbol"][symbol]
        expected = detail["gross_pnl"] - detail["fees"] - detail["slippage"] + detail["funding"]
        if abs(net[:, i].sum() - expected) > 1e-7:
            raise ValueError("symbol contribution reconciliation failed")
    return {"times": ts, "net": net, **values, "reconciliation_error": error}


def category(a, b, previous_a, previous_b, carry, reference_net):
    if a and not b:
        return ("missed_profit" if reference_net >= 0 else "avoided_loss"), None
    if b and not a:
        reason = "extra_entry" if not previous_b else ("late_exit" if previous_a else carry)
        if reason not in ("extra_entry", "late_exit"):
            raise ValueError("unaccounted unilateral position episode")
        return reason, reason
    return "same_target_path_and_sizing", None


def attribute(reference, challenger, times, prices, symbols):
    """Reconcile actual path differences; classifications are descriptive, not causal alpha."""
    a, b = (bar_components(r, times, prices, symbols) for r in (reference, challenger))
    decision_times = np.array([row["origin_ms"] for row in reference["decisions"]])
    if not np.array_equal(decision_times, [row["origin_ms"] for row in challenger["decisions"]]):
        raise ValueError("fixed matched decision clock required")
    ts, end = a["times"], reference["end_ms"]
    if end != challenger["end_ms"] or not np.array_equal(ts, b["times"]):
        raise ValueError("matched period required")
    names = ("gross", "fees", "slippage", "funding")
    previous_a, previous_b = np.zeros(len(symbols)), np.zeros(len(symbols))
    carries = [None] * len(symbols)
    rows = []
    for j, t in enumerate(decision_times):
        right = int(decision_times[j + 1]) if j + 1 < len(decision_times) else end
        holding = (ts > t) & (ts <= right)  # right-boundary funding belongs to the preceding position
        costs = (ts >= t) & ((ts < right) | ((ts == end) & (right == end)))
        av = {k: a[k][holding if k in ("gross", "funding") else costs].sum(axis=0) for k in names}
        bv = {k: b[k][holding if k in ("gross", "funding") else costs].sum(axis=0) for k in names}
        at, bt = (np.asarray(r["decisions"][j]["targets"]) for r in (reference, challenger))
        for i, symbol in enumerate(symbols):
            anet, bnet = sum(v[i] for v in av.values()), sum(v[i] for v in bv.values())
            label, carries[i] = category(at[i], bt[i], previous_a[i], previous_b[i], carries[i], anet)
            rows.append({"origin_ms": int(t), "end_ms": right, "symbol": symbol,
                         "reference_target": float(at[i]), "challenger_target": float(bt[i]),
                         "category": label, "reference_net_usdt": float(anet), "challenger_net_usdt": float(bnet),
                         "delta_usdt": float(bnet - anet),
                         "component_delta_usdt": {k: float(bv[k][i] - av[k][i]) for k in names}})
        previous_a, previous_b = at, bt
    category_totals = {}
    for row in rows:
        item = category_totals.setdefault(row["category"], {"cells": 0, "delta_usdt": 0., "reference_net_usdt": 0.})
        item["cells"] += 1
        item["delta_usdt"] += row["delta_usdt"]
        item["reference_net_usdt"] += row["reference_net_usdt"]
    expected = challenger["metrics"]["final_capital"] - reference["metrics"]["final_capital"]
    error = sum(row["delta_usdt"] for row in rows) - expected
    if abs(error) > 1e-7:
        raise ValueError("decision attribution reconciliation failed")
    return {"rows": rows, "categories": category_totals, "delta_usdt": expected, "reconciliation_error": error,
            "includes_path_dependent_position_sizing": True,
            "not_a_retrained_or_retraded_counterfactual": True,
            "interval_convention": "price/funding (left,right]; fees/drag [left,right), terminal closure included in final interval"}


def concentration(keys, amounts):
    groups = {}
    for key, value in zip(keys, amounts):
        key = str(key)
        groups[key] = groups.get(key, 0.) + float(value)
    ordered = sorted(groups.items(), key=lambda pair: abs(pair[1]), reverse=True)
    absolute = sum(abs(value) for _, value in ordered)
    return {"total_usdt": sum(groups.values()), "sum_absolute_group_usdt": absolute,
            "top_absolute": [{"key": k, "usdt": v} for k, v in ordered[:10]],
            "absolute_top_share": {str(n): (sum(abs(v) for _, v in ordered[:n]) / absolute if absolute else 0.) for n in (1, 3, 5)},
            "groups": groups}


def direct_checks():
    checks = []
    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    check(phase_anchor(0, 2) == 2 * c.DAY_MS, "only label anchor shifted")
    for bad in (-1, 3, True):
        try:
            phase_anchor(0, bad)
        except ValueError:
            check(True, "reject invalid phase " + str(bad))
        else:
            raise AssertionError("bad phase accepted")
    check(describe([-1, 1])["mean"] == 0, "diagnostic mean")
    check(describe([-1, 1])["positive_fraction"] == .5, "both signs retained")
    expected = [(1, 0, 0, 0, None, 1, "missed_profit"), (1, 0, 0, 0, None, -1, "avoided_loss"),
                (0, 1, 0, 0, None, 1, "extra_entry"), (0, 1, 1, 1, None, 1, "late_exit"),
                (0, 1, 0, 1, "extra_entry", 1, "extra_entry"),
                (0, 1, 0, 1, "late_exit", 1, "late_exit"), (1, 1, 0, 0, None, 1, "same_target_path_and_sizing")]
    for *args, label in expected:
        check(category(*args)[0] == label, label)
    n = 20 * 6 + 1
    times = np.arange(n, dtype=np.int64) * c.BAR_MS
    prices = np.column_stack((100 + np.arange(n) * .1, 100 - np.arange(n) * .05))
    features = np.zeros((n, 2, 6))
    phases = [c.make_samples(times, prices, features, anchor=phase_anchor(0, p), horizon_days=3, stride_days=3) for p in range(3)]
    daily = c.make_samples(times, prices, features, anchor=0, horizon_days=3, stride_days=1)
    check(np.array_equal(np.sort(np.concatenate([b["origins"] for b in phases])), daily["origins"]), "phase union equals daily grid")
    symbols = ("BTC/USDT", "ETH/USDT")
    grids = c.Clocks(3, 3, 3, 3).decision_times(0, int(times[-1]))
    def replay(pattern):
        rows = [{"origin_ms": t, "targets": pattern(j), "fit_ms": t, "model_sha256": "a" * 64} for j, t in enumerate(grids)]
        rates = [{"settlement_ms": int(times[-1]), "symbol": "BTCUSDT", "rate_fraction": "0.001"}]
        return c.replay(times=times, prices=prices, start=0, end=int(times[-1]), symbols=symbols, decisions=rows,
                        rates=rates, fee_bps=6, drag_bps=8, decision_cadence_days=3)
    a = replay(lambda j: [.1 if j in (0, 1, 4, 5, 6) else 0, 0])
    b = replay(lambda j: [.1 if j in (1, 2, 4, 5, 6) else 0, .1 if j in (2, 3) else 0])
    for result in (a, b):
        bars = bar_components(result, times, prices, symbols)
        check(abs(bars["reconciliation_error"]) < 1e-7, "bar-dollar reconciliation")
        check(bars["funding"][-1, 0] < 0, "terminal funding precedes close")
    attribution = attribute(a, b, times, prices, symbols)
    check(abs(attribution["reconciliation_error"]) < 1e-7, "exact decision difference")
    for name in ("missed_profit", "late_exit", "extra_entry", "same_target_path_and_sizing"):
        check(name in attribution["categories"], "synthetic category " + name)
    check(abs(attribute(a, a, times, prices, symbols)["delta_usdt"]) < 1e-10, "self comparison zero")
    check(sum(r["cells"] for r in attribution["categories"].values()) == len(grids) * 2, "complete nonoverlapping attribution cells")
    check(concentration(["x", "y"], [2, -1])["absolute_top_share"]["1"] == 2 / 3, "absolute concentration denominator")
    check(concentration(["x"], [0])["absolute_top_share"]["1"] == 0, "zero difference concentration")
    return {"checks_passed": len(checks), "labels": checks, "real_training_fits": 0, "network_used": False,
            "collector_used": False, "safety": c.SAFETY}
