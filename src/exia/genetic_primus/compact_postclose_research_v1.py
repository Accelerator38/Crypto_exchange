"""Pure bounded offline research; assumed post-close prices, NOT real fills."""
from __future__ import annotations

import math
import numpy as np
from . import compact_forecast_v1 as model
from . import compact_forecast_data_v1 as data

BAR_MS = data.BAR_MS
HORIZON_MS = data.HORIZON_MS
DAY_MS = 86_400_000
ARMS = tuple(a + "_" + w for a in model.PARAMETERS for w in model.WEIGHTINGS)
SAFETY = {k: False for k in (
    "credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
    "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed")}


def split(samples, origin):
    """Training eligibility never depends on a future deployment label."""
    if type(origin) is not int or origin % BAR_MS:
        raise ValueError("integer aligned origin required")
    cutoff = origin - 8 * BAR_MS
    train = np.flatnonzero(samples["available"] <= cutoff)
    if len(train) < 50:
        raise ValueError("insufficient mature training origins")
    validation = train[samples["origins"][train] >= samples["origins"][train[-1]] - 60 * DAY_MS]
    inner_cutoff = int(samples["origins"][validation[0]]) - 8 * BAR_MS
    inner = train[samples["available"][train] <= inner_cutoff]
    if len(inner) < 25 or len(validation) < 10:
        raise ValueError("insufficient purged temporal inner split")
    return {"train": train, "inner": inner, "validation": validation,
            "train_end_ms": origin - BAR_MS, "cutoff_ms": cutoff,
            "inner_cutoff_ms": inner_cutoff}


def samples(times, closes, anchor):
    features = data.feature_panel(times, closes)
    boundary = np.asarray(times, dtype=np.int64) + BAR_MS
    earliest = int(boundary[48])
    first = anchor + -((anchor - earliest) // HORIZON_MS) * HORIZON_MS
    origins = np.arange(first, int(boundary[-1]) - HORIZON_MS + 1, HORIZON_MS, dtype=np.int64)
    entries = np.searchsorted(boundary, origins)
    exits = np.searchsorted(boundary, origins + HORIZON_MS)
    if not len(origins) or np.any(boundary[entries] != origins) or np.any(boundary[exits] != origins + HORIZON_MS):
        raise ValueError("complete scheduled close labels required")
    bank = {"origins": origins, "available": origins + HORIZON_MS,
            "x": features[entries], "y": closes[exits] / closes[entries] - 1}
    return bank, features


def fit_origin(bank, origin, decision_x, feature_prefix_sha):
    temporal = split(bank, origin)
    train, inner, validation = (temporal[k] for k in ("train", "inner", "validation"))
    flatten = lambda indices: (bank["x"][indices].reshape(-1, 6), bank["y"][indices].reshape(-1))
    tx, ty = flatten(train)
    ix, iy = flatten(inner)
    vx, vy = flatten(validation)
    scaler, inner_scaler = model.fit_scaler(tx, ty), model.fit_scaler(ix, iy)
    identity = {k: temporal[k] for k in ("train_end_ms", "cutoff_ms", "inner_cutoff_ms")}
    identity.update(train_samples_sha256=model.array_identity(bank["origins"][train], bank["available"][train], tx, ty),
                    folds_sha256=model.array_identity(bank["origins"][inner], bank["origins"][validation]),
                    feature_definition_sha256=data.FEATURE_SHA256, target="scheduled_close_to_close_3d",
                    l2=0.01, learning_rate=0.001, max_epochs=120, patience=12)
    result = {"schema": "compact_postclose_origin/1", "origin_ms": origin,
              "scheduled_label_end_ms": origin + HORIZON_MS,
              "signal_us": origin * 1000 + 100000, "model_fill_us": origin * 1000 + 350000,
              "training_identity": identity, "feature_prefix_sha256": feature_prefix_sha,
              "decision_feature_sha256": model.array_identity(decision_x),
              "scaler_sha256": scaler.sha256, "inner_scaler_sha256": inner_scaler.sha256,
              "train_origin_count": len(train), "inner_origin_count": len(inner),
              "inner_validation_count": len(validation), "target_positive_count": int((ty > 0).sum()),
              "target_negative_count": int((ty < 0).sum()), "model_fit_count": 0,
              "gradient_steps": 0, "arms": {}, "safety": SAFETY}
    universe_size = bank["x"].shape[1]
    for weighting in model.WEIGHTINGS:
        ow = model.temporal_weights(bank["available"][train], temporal["cutoff_ms"], weighting)
        iw = model.temporal_weights(bank["available"][inner], temporal["inner_cutoff_ms"], weighting)
        weights, inner_weights = np.repeat(ow / universe_size, universe_size), np.repeat(iw / universe_size, universe_size)
        weight_info = {"origin_ess": model.origin_ess(ow), "weights_sha256": model.array_identity(ow),
                       "positive_weight_mass": float(weights[ty > 0].sum()),
                       "negative_weight_mass": float(weights[ty < 0].sum())}

        def payload(architecture, parameters, seed, epochs):
            value = model.model_payload(architecture, parameters, scaler, weighting=weighting,
                                        seed=seed, epochs=epochs, training_identity=identity | weight_info)
            value["target_kind"] = "gross_scheduled_close_close_3d_forecast_not_net_profit"
            value["sha256"] = model.identity({k: v for k, v in value.items() if k != "sha256"})
            return value

        parameters = model.fit_linear(scaler.transform(tx), scaler.target(ty), weights)
        linear = payload("linear6", parameters, None, None)
        result["model_fit_count"] += 1
        result["arms"]["linear6_" + weighting] = weight_info | {
            "models": [linear], "prediction": model.predict(linear, decision_x).tolist(), "epochs": []}
        models, predictions, selection = [], [], []
        for seed in model.SEEDS:
            _, trace = model.fit_mlp(inner_scaler.transform(ix), inner_scaler.target(iy), inner_weights,
                                     seed=seed, validation=(inner_scaler.transform(vx), inner_scaler.target(vy)))
            epochs = trace["selected_epoch"]
            parameters, refit = model.fit_mlp(scaler.transform(tx), scaler.target(ty), weights, seed=seed, epochs=epochs)
            value = payload("mlp8", parameters, seed, epochs)
            models.append(value)
            predictions.append(model.predict(value, decision_x))
            selection.append({"seed": seed, "selected_epoch": epochs, "inner_epochs_run": trace["epochs_run"]})
            result["model_fit_count"] += 2
            result["gradient_steps"] += trace["gradient_steps"] + refit["gradient_steps"]
        result["arms"]["mlp8_" + weighting] = weight_info | {
            "models": models, "prediction": np.mean(predictions, axis=0).tolist(),
            "seed_predictions": np.asarray(predictions).tolist(), "epochs": selection}
    result["sha256"] = model.identity(result)
    return result


def resample_bank(days, seed, draws=4096):
    if type(days) is not int or not 3 <= days <= 366 or draws != 4096:
        raise ValueError("bounded calendar bank required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, days, size=(draws, math.ceil(days / 3)))
    return ((starts[:, :, None] + np.arange(3)) % days).reshape(draws, -1)[:, :days].astype(np.int32)


def replay(*, times, prices, start, end, symbols, decisions, rates, fee_bps, drag_bps):
    """One continuous long/cash account. Every execution/mark is a declared proxy."""
    times, prices = np.asarray(times), np.asarray(prices, dtype=float)
    if (times.dtype.kind not in "iu" or times.ndim != 1 or prices.shape != (len(times), len(symbols))
            or np.any(np.diff(times) != BAR_MS) or not np.isfinite(prices).all() or np.any(prices <= 0)
            or start % DAY_MS or end % DAY_MS or start >= end):
        raise ValueError("finite complete close panel and calendar interval required")
    expected = list(range(start, end, HORIZON_MS))
    if [row["origin_ms"] for row in decisions] != expected:
        raise ValueError("all origins including terminal partial deployment required")
    for row in decisions:
        target = np.asarray(row["targets"], dtype=float)
        if (target.shape != (len(symbols),) or not np.isin(target, [0, 0.1]).all()
                or target.sum() > 0.8 + 1e-12 or len(row["model_sha256"]) != 64):
            raise ValueError("fixed binary 10-percent sleeves and model identity required")
    if not 0 <= fee_bps <= 100 or not 0 <= drag_bps <= 100:
        raise ValueError("bounded modeled costs required")
    decision_map = {r["origin_ms"]: r for r in decisions}
    symbol_map = {s.replace("/", ""): i for i, s in enumerate(symbols)}
    funding_map, seen = {}, set()
    for row in rates:
        t, symbol = int(row["settlement_ms"]), row["symbol"]
        if start <= t <= end:
            key = (t, symbol)
            if key in seen or symbol not in symbol_map or t % BAR_MS or abs(float(row["rate_fraction"])) >= 1:
                raise ValueError("unique aligned signed modeled funding events required")
            seen.add(key)
            funding_map.setdefault(t, []).append(row)
    segment = np.flatnonzero((times >= start) & (times <= end))
    if len(segment) != (end - start) // BAR_MS + 1 or times[segment[0]] != start or times[segment[-1]] != end:
        raise ValueError("terminal close price must be present")
    cash, quantity, basis = 10000.0, np.zeros(len(symbols)), np.zeros(len(symbols))
    previous_targets = np.zeros(len(symbols))
    fees = slippage = fund_total = gross = notional = 0.0
    peak, maximum_dd = cash, 0.0
    ledger, equity, decision_ledger = [], [], []
    active_model = model.identity({"policy": "initial_no_position"})
    per_symbol = {s: {"gross_pnl": 0.0, "fees": 0.0, "slippage": 0.0, "funding": 0.0} for s in symbols}

    def mark(t, px, phase):
        nonlocal peak, maximum_dd
        nav = cash + float(quantity @ px)
        peak = max(peak, nav)
        dd = 1 - nav / peak
        maximum_dd = max(maximum_dd, dd)
        equity.append({"timestamp_ms": t, "phase": phase, "nav": nav, "cash": cash,
                       "exposure": float(quantity @ px) / nav, "drawdown": dd})
        if not math.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise ValueError("no borrowing, insolvency, or nonfinite capital allowed")
        return nav

    for index in segment:
        t, px = int(times[index]), prices[index]
        mark(t, px, "before_funding_and_decision")
        for rate in sorted(funding_map.get(t, []), key=lambda r: r["symbol"]):
            i = symbol_map[rate["symbol"]]
            amount = -quantity[i] * px[i] * float(rate["rate_fraction"])
            cash += amount
            fund_total += amount
            per_symbol[symbols[i]]["funding"] += amount
            if quantity[i]:
                ledger.append({"timestamp_us": t * 1000, "kind": "modeled_funding", "symbol": symbols[i],
                               "cash_delta": amount, "quantity": float(quantity[i]), "model_sha256": active_model})
            mark(t, px, "after_modeled_funding")
        row = decision_map.get(t)
        if row is None and t != end:
            mark(t, px, "calendar_bucket_end")
            continue
        if row is not None:
            targets = np.asarray(row["targets"])
            active_model = row["model_sha256"]
            decision_ledger.append(row | {"signal_us": t * 1000 + 100000, "model_fill_us": t * 1000 + 350000})
        else:
            targets = np.zeros(len(symbols))
        nav = cash + float(quantity @ px)
        # Sell exits first. Surviving sleeves retain units, not continuously target weights.
        operations = [(i, "SELL") for i in range(len(symbols)) if quantity[i] and not targets[i]]
        operations += [(i, "BUY") for i in range(len(symbols)) if targets[i] and not previous_targets[i]]
        for i, side in operations:
            if side == "BUY":
                execution_price = px[i] * (1 + drag_bps / 10000)
                q = nav * 0.1 / (execution_price * (1 + fee_bps / 10000))
                fee = q * execution_price * fee_bps / 10000
                delta = -q * execution_price - fee
                quantity[i], basis[i] = q, q * px[i]
                realized = 0.0
            else:
                q = quantity[i]
                execution_price = px[i] * (1 - drag_bps / 10000)
                fee = q * execution_price * fee_bps / 10000
                delta = q * execution_price - fee
                realized = q * px[i] - basis[i]
                quantity[i], basis[i] = 0, 0
            slip = q * abs(execution_price - px[i])
            cash += delta
            fees += fee
            slippage += slip
            gross += realized
            notional += q * px[i]
            detail = per_symbol[symbols[i]]
            detail["gross_pnl"] += realized
            detail["fees"] += fee
            detail["slippage"] += slip
            ledger.append({"timestamp_us": t * 1000 + 350000, "kind": "modeled_fill", "side": side,
                           "symbol": symbols[i], "quantity": float(q), "reference_close": float(px[i]),
                           "price": float(execution_price), "fee": float(fee), "slippage": float(slip),
                           "cash_delta": float(delta), "gross_pnl": float(realized), "model_sha256": active_model})
            mark(t, px, "after_modeled_fill")
        previous_targets = targets
        mark(t, px, "calendar_bucket_end")
    if np.any(quantity) or abs((cash - 10000) - (gross - fees - slippage + fund_total)) > 1e-7:
        raise ValueError("terminal reconciliation failed")
    by_time = {row["timestamp_ms"]: row for row in equity if row["phase"] == "calendar_bucket_end"}
    # Include initial entry costs in first day, terminal costs/funding in last day.
    daily_nav = [10000.0] + [by_time[t]["nav"] for t in range(start + DAY_MS, end + 1, DAY_MS)]
    daily_log = np.diff(np.log(daily_nav))
    exposure = [by_time[t]["exposure"] for t in range(start, end, BAR_MS)]
    fills = [r for r in ledger if r["kind"] == "modeled_fill"]
    return {"schema": "compact_postclose_research_ledger/1", "start_ms": start, "end_ms": end,
            "terminal_fill_us": end * 1000 + 350000, "execution_is_assumed": True,
            "observed_execution_qualified": False, "metrics": {
                "initial_capital": 10000.0, "final_capital": cash, "net_return": cash / 10000 - 1,
                "max_drawdown": maximum_dd, "gross_pnl": gross, "fees": fees,
                "slippage": slippage, "funding": fund_total, "turnover_initial_capital": notional / 10000,
                "fills": len(fills), "mean_exposure": float(np.mean(exposure)),
                "maximum_exposure": float(max(exposure)), "decisions": len(decision_ledger),
                "reconciliation_error": cash - 10000 - gross + fees + slippage - fund_total},
            "daily_log_returns": daily_log.tolist(), "equity": equity, "ledger": ledger,
            "decisions": decision_ledger, "per_symbol": per_symbol, "safety": SAFETY}


def audit(daily_log, selection_bank, audit_bank, *, paired_log=None):
    values = np.asarray(daily_log, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite calendar log returns required")
    result = {}
    for name, bank in (("selection", selection_bank), ("audit", audit_bank)):
        if bank.shape != (4096, len(values)) or np.any(bank < 0) or np.any(bank >= len(values)):
            raise ValueError("common matching registered bank required")
        mean_bps = values[bank].mean(axis=1) * 10000
        item = {"mean_log_bps_per_day": float(values.mean() * 10000),
                "LCB_mean_log_bps_per_day": float(np.quantile(mean_bps, 0.05)),
                "bank_sha256": model.array_identity(bank)}
        if paired_log is not None:
            paired = values - np.asarray(paired_log)
            item["paired_LCB_vs_EMA_log_bps_per_day"] = float(np.quantile(paired[bank].mean(axis=1) * 10000, 0.05))
        result[name] = item
    windows = np.convolve(values, np.ones(3), mode="valid")
    best = float(windows.max())
    result["worst_three_day_return"] = float(np.expm1(windows.min()))
    result["return_excluding_best_three_day_log_window"] = float(np.expm1(values.sum() - best))
    result["best_three_day_share_of_total_log_growth"] = best / float(values.sum()) if values.sum() > 0 else None
    slices = np.asarray([values[i:i + 3].sum() for i in range(0, len(values), 3)])
    result["positive_outer_slices_fraction"] = float(np.mean(slices > 0))
    result["outer_slices"] = len(slices)
    result["full_three_day_outer_slices"] = len(values) // 3
    result["terminal_partial_slice_days"] = len(values) % 3
    return result


def direct_checks():
    count = 0
    def check(value):
        nonlocal count
        if not value:
            raise AssertionError("research regression failed at " + str(count + 1))
        count += 1
    o = np.arange(100, dtype=np.int64) * HORIZON_MS
    bank = {"origins": o, "available": o + HORIZON_MS}
    cutoff_origin = int(105 * HORIZON_MS)
    s = split(bank, cutoff_origin)
    check(np.all(bank["available"][s["train"]] <= cutoff_origin - 8 * BAR_MS))
    check(np.all(bank["available"][s["inner"]] <= s["inner_cutoff_ms"]))
    check(cutoff_origin not in o)  # A missing future target does not block deployment.
    check(np.array_equal(resample_bank(61, 3039), resample_bank(61, 3039)))
    check(not np.array_equal(resample_bank(61, 3039), resample_bank(61, 7079)))
    times = np.arange(0, 7 * DAY_MS + BAR_MS, BAR_MS, dtype=np.int64)
    prices = np.full((len(times), 1), 100.0)
    sha = model.identity({"test": "synthetic"})
    decisions = [{"origin_ms": t, "targets": [0.1], "model_sha256": sha} for t in range(0, 7 * DAY_MS, HORIZON_MS)]
    arguments = dict(times=times, prices=prices, start=0, end=7 * DAY_MS,
                     symbols=("BTC/USDT",), decisions=decisions, rates=[], fee_bps=0, drag_bps=0)
    zero = replay(**arguments)
    check(zero["metrics"]["final_capital"] == 10000)
    check(zero["metrics"]["fills"] == 2)
    check(len(zero["decisions"]) == 3)
    cost = replay(**(arguments | {"fee_bps": 6, "drag_bps": 8}))
    check(cost["metrics"]["final_capital"] < 10000)
    check(abs(cost["metrics"]["reconciliation_error"]) < 1e-7)
    rates = [{"symbol": "BTCUSDT", "settlement_ms": 7 * DAY_MS, "rate_fraction": "0.001"}]
    terminal = replay(**(arguments | {"rates": rates}))
    check(terminal["metrics"]["funding"] == -1)
    check(terminal["metrics"]["final_capital"] == 9999)
    no_trade = replay(**(arguments | {"decisions": [r | {"targets": [0]} for r in decisions], "fee_bps": 6, "drag_bps": 8}))
    check(no_trade["metrics"]["final_capital"] == 10000)
    check(no_trade["metrics"]["fills"] == 0)
    check(sum(zero["daily_log_returns"]) == 0)
    rising = prices.copy()
    rising[times >= HORIZON_MS] = 110
    gain = replay(**(arguments | {"prices": rising}))
    check(abs(gain["metrics"]["final_capital"] - 10100) < 1e-7)
    check(gain["metrics"]["fills"] == 2)
    check(len(gain["daily_log_returns"]) == 7)
    a = audit(no_trade["daily_log_returns"], resample_bank(7, 3039), resample_bank(7, 7079))
    check(a["audit"]["LCB_mean_log_bps_per_day"] == 0)
    return {"checks_passed": count, "collector_used": False, "network_used": False, "safety": SAFETY}
