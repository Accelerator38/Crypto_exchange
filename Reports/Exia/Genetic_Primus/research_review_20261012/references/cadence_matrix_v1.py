"""Four independent clocks, causal linear fits, continuous offline cash ledger."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
from . import compact_forecast_v1 as m
from . import compact_forecast_data_v1 as d
from . import compact_postclose_research_v1 as old

BAR_MS = d.BAR_MS
DAY_MS = m.DAY_MS
SAFETY = old.SAFETY


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Clocks:
    label_horizon_days: int
    label_stride_days: int
    decision_cadence_days: int
    retrain_cadence_days: int

    def validate(self):
        if any(type(v) is not int or not 1 <= v <= 30 for v in asdict(self).values()):
            raise ValueError("four separate bounded integer day clocks required")
        return self

    def decision_times(self, start, end):
        self.validate()
        if start % DAY_MS or end % DAY_MS or start >= end:
            raise ValueError("positive calendar interval required")
        return list(range(start, end, self.decision_cadence_days * DAY_MS))

    def latest_fit(self, start, decision):
        self.validate()
        if start % DAY_MS or decision % DAY_MS or decision < start:
            raise ValueError("causal calendar fit schedule required")
        return start + ((decision - start) // (self.retrain_cadence_days * DAY_MS)) * self.retrain_cadence_days * DAY_MS


def make_samples(times, closes, features, *, anchor, horizon_days, stride_days):
    Clocks(horizon_days, stride_days, 1, 1).validate()
    times, closes, features = np.asarray(times), np.asarray(closes), np.asarray(features)
    if (times.dtype.kind not in "iu" or times.ndim != 1 or len(times) < 49
            or np.any(np.diff(times) != BAR_MS) or np.any(times % BAR_MS)
            or closes.ndim != 2 or closes.shape[0] != len(times)
            or features.shape != closes.shape + (6,) or not np.isfinite(closes).all() or np.any(closes <= 0)
            or not np.isfinite(features[48:]).all() or type(anchor) is not int or anchor % DAY_MS):
        raise ValueError("complete frozen causal panel and calendar anchor required")
    step, horizon = stride_days * DAY_MS, horizon_days * DAY_MS
    earliest = int(times[48])
    first = anchor + -((anchor - earliest) // step) * step
    origins = np.arange(first, int(times[-1]) - horizon + 1, step, dtype=np.int64)
    entry, exit_ = np.searchsorted(times, origins), np.searchsorted(times, origins + horizon)
    if not len(origins) or np.any(times[entry] != origins) or np.any(times[exit_] != origins + horizon):
        raise ValueError("complete outcome clocks required")
    return {"origins": origins, "available": origins + horizon,
            "x": features[entry], "y": closes[exit_] / closes[entry] - 1}


def fit_linear(bank, fit_ms, *, horizon_days, stride_days, feature_prefix_sha, sample_sha):
    temporal = old.split(bank, fit_ms)
    train, inner, validation = (temporal[k] for k in ("train", "inner", "validation"))
    tx, ty = bank["x"][train].reshape(-1, 6), bank["y"][train].reshape(-1)
    scaler = m.fit_scaler(tx, ty)
    ow = m.temporal_weights(bank["available"][train], temporal["cutoff_ms"], "uniform")
    weights = np.repeat(ow / bank["x"].shape[1], bank["x"].shape[1])
    parameters = m.fit_linear(scaler.transform(tx), scaler.target(ty), weights, l2=0.01)
    identity = {k: temporal[k] for k in ("train_end_ms", "cutoff_ms", "inner_cutoff_ms")}
    identity.update(label_horizon_days=horizon_days, label_stride_days=stride_days,
                    sample_bank_sha256=sample_sha, feature_definition_sha256=d.FEATURE_SHA256,
                    train_samples_sha256=m.array_identity(bank["origins"][train], bank["available"][train], tx, ty),
                    folds_sha256=m.array_identity(bank["origins"][inner], bank["origins"][validation]),
                    inner_folds_used_for_tuning=False, l2=0.01, weights_sha256=m.array_identity(ow))
    payload = m.model_payload("linear6", parameters, scaler, weighting="uniform", seed=None,
                              epochs=None, training_identity=identity)
    payload["target_kind"] = "gross_scheduled_close_close_forecast_not_net_profit"
    payload["sha256"] = m.identity({k: v for k, v in payload.items() if k != "sha256"})
    return {"schema": "cadence_matrix_linear_fit/1", "fit_ms": fit_ms,
            "assumed_model_ready_us": fit_ms * 1000 + 100000,
            "model": payload, "scaler_sha256": scaler.sha256, "training_identity": identity,
            "feature_prefix_sha256": feature_prefix_sha, "train_origins": len(train),
            "train_asset_rows": len(ty), "latest_mature_label_ms": int(bank["available"][train[-1]]),
            "inner_origins": len(inner), "inner_validation_origins": len(validation),
            "origin_weight_kish_ess": m.origin_ess(ow), "overlap_adjusted_ess_claimed": False,
            "positive_weight_mass": float(weights[ty > 0].sum()), "negative_weight_mass": float(weights[ty < 0].sum()),
            "training_fits": 1, "scaler_fits": 1, "safety": SAFETY}


def resample_bank(days, block_days, seed):
    if type(days) is not int or not 14 <= days <= 366 or block_days not in (3, 7, 14):
        raise ValueError("bounded shared calendar resampling required")
    starts = np.random.default_rng(seed).integers(0, days, size=(4096, math.ceil(days / block_days)))
    return ((starts[:, :, None] + np.arange(block_days)) % days).reshape(4096, -1)[:, :days].astype(np.int32)


def bootstrap(daily_log, banks, *, paired=None):
    values = np.asarray(daily_log, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("finite daily portfolio returns required")
    if paired is not None:
        other = np.asarray(paired, dtype=float)
        if other.shape != values.shape or not np.isfinite(other).all():
            raise ValueError("matched calendar comparator required")
    result = {}
    for block in (3, 7, 14):
        item = {}
        for role in ("selection", "audit"):
            bank = banks[f"b{block}_{role}"]
            if bank.shape != (4096, len(values)) or np.any(bank < 0) or np.any(bank >= len(values)):
                raise ValueError("shared bank identity/interval mismatch")
            entry = {"LCB_log_bps_day": float(np.quantile(values[bank].mean(axis=1) * 10000, .05)),
                     "mean_log_bps_day": float(values.mean() * 10000), "bank_sha256": m.array_identity(bank)}
            if paired is not None:
                entry["paired_LCB_log_bps_day"] = float(np.quantile((values - other)[bank].mean(axis=1) * 10000, .05))
            item[role] = entry
        result[str(block)] = item
    return result


def replay(*, times, prices, start, end, symbols, decisions, rates, fee_bps, drag_bps, decision_cadence_days):
    """Generalized clock validation; frozen v1 cash/quantity accounting semantics."""
    times, prices = np.asarray(times), np.asarray(prices, dtype=float)
    clocks = Clocks(3, 1, decision_cadence_days, 1).validate()
    if (times.dtype.kind not in "iu" or times.ndim != 1 or prices.shape != (len(times), len(symbols))
            or np.any(np.diff(times) != BAR_MS) or not np.isfinite(prices).all() or np.any(prices <= 0)):
        raise ValueError("complete finite close panel required")
    if [row["origin_ms"] for row in decisions] != clocks.decision_times(start, end):
        raise ValueError("complete separate decision grid required")
    for row in decisions:
        target = np.asarray(row["targets"], dtype=float)
        if (target.shape != (len(symbols),) or not np.isin(target, [0, .1]).all()
                or target.sum() > .8 + 1e-12 or len(row["model_sha256"]) != 64):
            raise ValueError("registered binary sleeves/model identity required")
        if row.get("fit_ms", row["origin_ms"]) > row["origin_ms"]:
            raise ValueError("future model cannot make current decision")
    if not 0 <= fee_bps <= 100 or not 0 <= drag_bps <= 100:
        raise ValueError("bounded costs required")
    decision_map = {r["origin_ms"]: r for r in decisions}
    symbol_map = {s.replace("/", ""): i for i, s in enumerate(symbols)}
    funding_map, seen = {}, set()
    for row in rates:
        t, symbol = int(row["settlement_ms"]), row["symbol"]
        if start <= t <= end:
            key = (t, symbol)
            if key in seen or symbol not in symbol_map or t % BAR_MS or abs(float(row["rate_fraction"])) >= 1:
                raise ValueError("unique aligned modeled funding required")
            seen.add(key)
            funding_map.setdefault(t, []).append(row)
    segment = np.flatnonzero((times >= start) & (times <= end))
    if len(segment) != (end - start) // BAR_MS + 1 or times[segment[0]] != start or times[segment[-1]] != end:
        raise ValueError("terminal reference close required")
    cash, quantity, basis = 10000.0, np.zeros(len(symbols)), np.zeros(len(symbols))
    previous_targets = np.zeros(len(symbols))
    fees = slippage = fund_total = gross = notional = 0.0
    peak, maximum_dd = cash, 0.0
    ledger, equity, decision_ledger = [], [], []
    active_model = m.identity({"policy": "initial_no_position"})
    per_symbol = {s: {"gross_pnl": 0., "fees": 0., "slippage": 0., "funding": 0.} for s in symbols}

    def mark(t, px, phase):
        nonlocal peak, maximum_dd
        nav = cash + float(quantity @ px)
        peak = max(peak, nav)
        dd = 1 - nav / peak
        maximum_dd = max(maximum_dd, dd)
        equity.append({"timestamp_ms": t, "phase": phase, "nav": nav, "cash": cash,
                       "exposure": float(quantity @ px) / nav, "drawdown": dd})
        if not math.isfinite(nav) or nav <= 0 or cash < -1e-7:
            raise ValueError("no borrowing/nonfinite capital allowed")

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
        operations = [(i, "SELL") for i in range(len(symbols)) if quantity[i] and not targets[i]]
        operations += [(i, "BUY") for i in range(len(symbols)) if targets[i] and not previous_targets[i]]
        for i, side in operations:
            if side == "BUY":
                execution_price = px[i] * (1 + drag_bps / 10000)
                q = nav * .1 / (execution_price * (1 + fee_bps / 10000))
                fee = q * execution_price * fee_bps / 10000
                delta = -q * execution_price - fee
                quantity[i], basis[i] = q, q * px[i]
                realized = 0.
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
        raise ValueError("terminal cash reconciliation failed")
    by_time = {row["timestamp_ms"]: row for row in equity if row["phase"] == "calendar_bucket_end"}
    daily_nav = [10000.] + [by_time[t]["nav"] for t in range(start + DAY_MS, end + 1, DAY_MS)]
    daily_log = np.diff(np.log(daily_nav))
    exposure = [by_time[t]["exposure"] for t in range(start, end, BAR_MS)]
    fills = [r for r in ledger if r["kind"] == "modeled_fill"]
    return plain({"schema": "cadence_matrix_ledger/1", "start_ms": start, "end_ms": end,
                  "terminal_fill_us": end * 1000 + 350000, "execution_is_assumed": True,
                  "observed_execution_qualified": False, "metrics": {
                      "initial_capital": 10000., "final_capital": cash, "net_return": cash / 10000 - 1,
                      "max_drawdown": maximum_dd, "gross_pnl": gross, "fees": fees,
                      "slippage": slippage, "funding": fund_total, "turnover_initial_capital": notional / 10000,
                      "fills": len(fills), "mean_exposure": float(np.mean(exposure)),
                      "maximum_exposure": float(max(exposure)), "decisions": len(decision_ledger),
                      "reconciliation_error": cash - 10000 - gross + fees + slippage - fund_total},
                  "daily_log_returns": daily_log, "equity": equity, "ledger": ledger,
                  "decisions": decision_ledger, "per_symbol": per_symbol, "safety": SAFETY})


def direct_checks():
    count = 0
    def check(value):
        nonlocal count
        if not value:
            raise AssertionError("cadence regression " + str(count + 1))
        count += 1
    c = Clocks(3, 1, 1, 4)
    check(len(c.decision_times(0, 7 * DAY_MS)) == 7)
    check(c.latest_fit(0, 3 * DAY_MS) == 0)
    check(c.latest_fit(0, 5 * DAY_MS) == 4 * DAY_MS)
    check(Clocks(3, 1, 3, 2).latest_fit(0, 3 * DAY_MS) == 2 * DAY_MS)
    check(Clocks(2, 4, 5, 1).validate().label_horizon_days == 2)
    try:
        Clocks(True, 1, 1, 1).validate()
    except ValueError:
        check(True)
    else:
        check(False)
    times = np.arange(0, 30 * DAY_MS + BAR_MS, BAR_MS, dtype=np.int64)
    close = np.full((len(times), 1), 100.)
    features = np.zeros((len(times), 1, 6))
    s = make_samples(times, close, features, anchor=0, horizon_days=2, stride_days=1)
    check(len(s["origins"]) > 10)
    check(np.all(np.diff(s["origins"]) == DAY_MS))
    check(np.all(s["available"] - s["origins"] == 2 * DAY_MS))
    check(np.all(s["y"] == 0))
    sha = m.identity({"synthetic": True})
    daily = [{"origin_ms": t, "targets": [.1], "model_sha256": sha} for t in range(0, 7 * DAY_MS, DAY_MS)]
    args = dict(times=times, prices=close, start=0, end=7 * DAY_MS, symbols=("BTC/USDT",),
                decisions=daily, rates=[], fee_bps=6, drag_bps=8, decision_cadence_days=1)
    result = replay(**args)
    check(result["metrics"]["fills"] == 2)
    check(result["metrics"]["decisions"] == 7)
    check(result["metrics"]["net_return"] < 0)
    check(abs(result["metrics"]["reconciliation_error"]) < 1e-7)
    terminal = [{"symbol": "BTCUSDT", "settlement_ms": 7 * DAY_MS, "rate_fraction": ".001"}]
    funded = replay(**(args | {"rates": terminal}))
    check(funded["metrics"]["funding"] < 0)
    check(funded["metrics"]["final_capital"] < result["metrics"]["final_capital"])
    cash = replay(**(args | {"decisions": [r | {"targets": [0]} for r in daily]}))
    check(cash["metrics"]["final_capital"] == 10000)
    check(cash["metrics"]["fills"] == 0)
    sparse = [r for r in daily if r["origin_ms"] % (3 * DAY_MS) == 0]
    args3 = args | {"decisions": sparse, "decision_cadence_days": 3}
    new3 = replay(**args3)
    old3 = old.replay(**{k: v for k, v in args3.items() if k != "decision_cadence_days"})
    check(new3["metrics"] == plain(old3["metrics"]))
    check(new3["daily_log_returns"] == old3["daily_log_returns"])
    check(new3["ledger"] == old3["ledger"])
    bad = [r | {"fit_ms": r["origin_ms"] + DAY_MS} for r in daily]
    try:
        replay(**(args | {"decisions": bad}))
    except ValueError:
        check(True)
    else:
        check(False)
    check(np.array_equal(resample_bank(61, 7, 3039), resample_bank(61, 7, 3039)))
    check(not np.array_equal(resample_bank(61, 7, 3039), resample_bank(61, 7, 7079)))
    return {"checks_passed": count, "real_training_fits": 0, "collector_used": False,
            "network_used": False, "safety": SAFETY}
