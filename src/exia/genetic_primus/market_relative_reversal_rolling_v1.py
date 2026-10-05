"""One-origin synthetic research evaluator; no I/O, clocks or trading runtime."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from typing import Callable, Iterable, Mapping

BAR_MS = 14_400_000
DAY_MS = 6 * BAR_MS
HORIZON_MS = 3 * DAY_MS
WINDOW_MS = 730 * DAY_MS
RESERVED_START_MS = 1791158400000
RESERVED_END_MS = 1806710400000
UNIVERSE = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT",
            "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
SCENARIOS = {"normal": (0, 16.0), "stress": (1, 40.0)}


@dataclass(frozen=True)
class Primitives:
    build_features: Callable
    build_training_origin: Callable
    observation_type: Callable
    fit_origin: Callable
    predict_gross: Callable
    builder_sha256: str
    fitter_sha256: str


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("source SHA256 required")


def _cache(candles: Iterable) -> dict:
    result = {}
    for c in candles:
        if type(c.open_ms) is not int or c.open_ms % BAR_MS or c.symbol not in UNIVERSE:
            raise ValueError("aligned full8 candle required")
        key = (c.open_ms, c.symbol)
        if key in result:
            raise ValueError("duplicate candle")
        result[key] = c
    return result


def _window(cache: dict, origin_ms: int, end_ms: int) -> list:
    return [cache[(t, s)] for t in range(origin_ms-13*BAR_MS, end_ms, BAR_MS)
            for s in UNIVERSE if (t, s) in cache]


def _flat_scenario() -> dict:
    return {"positions": {s: {"cash": 1.0, "quantity": 0.0, "target": 0,
                              "last_mark_price": None} for s in UNIVERSE},
            "last_nav": 1.0, "peak_nav": 1.0, "max_drawdown_close": 0.0,
            "cumulative_fees": 0.0}


def _scenario(cache: dict, *, origin_ms: int, targets: Mapping[str, int],
              previous: dict, delay_bars: int, round_trip_bps: float,
              model_sha256: str) -> dict:
    """Validated synthetic bars in, atomic sleeve accounting out. No orders."""
    state = deepcopy(previous)
    events, navs, daily = [], [], []
    fee_rate = round_trip_bps / 20_000
    daily_base = state["last_nav"]
    interval_base = daily_base
    fees = 0.0
    for k in range(18):
        t = origin_ms + k*BAR_MS
        for symbol in UNIVERSE:
            p = state["positions"][symbol]
            candle = cache[(t, symbol)]
            if k == delay_bars and p["target"] != targets[symbol]:
                before = p["cash"] + p["quantity"]*candle.open
                old_target = p["target"]
                if targets[symbol] == 1:
                    quantity = p["cash"] / (candle.open*(1+fee_rate))
                    fee = quantity*candle.open*fee_rate
                    p.update(cash=0.0, quantity=quantity, target=1)
                else:
                    proceeds = p["quantity"]*candle.open
                    fee = proceeds*fee_rate
                    p.update(cash=proceeds-fee, quantity=0.0, target=0)
                fees += fee / len(UNIVERSE)
                events.append({"symbol": symbol, "decision_time_ms": origin_ms,
                               "execution_time_ms": t, "previous_target": old_target,
                               "target": targets[symbol], "price": float(candle.open),
                               "fee": fee, "equity_before": before,
                               "equity_after": p["cash"]+p["quantity"]*candle.open,
                               "model_sha256": model_sha256})
            p["last_mark_price"] = float(candle.close)
        nav = fsum(p["cash"]+p["quantity"]*p["last_mark_price"]
                   for p in state["positions"].values()) / len(UNIVERSE)
        if not isfinite(nav) or nav <= 0:
            raise ValueError("invalid synthetic portfolio NAV")
        state["peak_nav"] = max(state["peak_nav"], nav)
        state["max_drawdown_close"] = max(state["max_drawdown_close"], 1-nav/state["peak_nav"])
        navs.append({"close_ms": t+BAR_MS, "nav": nav})
        if (k+1) % 6 == 0:
            daily.append(nav/daily_base-1)
            daily_base = nav
    state["last_nav"] = navs[-1]["nav"]
    state["cumulative_fees"] += fees
    return {"state": state, "events": events, "bar_close_nav": navs,
            "daily_returns": daily, "origin_net_return": state["last_nav"]/interval_base-1,
            "fees": fees, "delay_bars": delay_bars, "round_trip_cost_bps": round_trip_bps}


def evaluate_origin(
    candles: Iterable, *, origin_ms: int, train_start_origin_ms: int,
    previous_state: dict | None, primitives: Primitives,
    evaluator_sha256: str, draft_sha256: str, data_kind: str,
) -> dict:
    """Explicit three-day step, allowed only as synthetic engineering work.

    Source hashes are caller-declared lineage, not signed authentication.
    No bootstrap, outer analysis, funding or real-execution certification.
    """
    if data_kind != "synthetic":
        raise ValueError("unregistered design: market-data evaluation forbidden")
    if (type(origin_ms) is not int or origin_ms % DAY_MS
            or type(train_start_origin_ms) is not int or train_start_origin_ms % DAY_MS
            or train_start_origin_ms >= origin_ms
            or (origin_ms-train_start_origin_ms) % HORIZON_MS):
        raise ValueError("fixed midnight UTC three-day origin grid required")
    if origin_ms < RESERVED_END_MS and origin_ms+HORIZON_MS > RESERVED_START_MS:
        raise ValueError("reserved EMA evaluation interval forbidden")
    sources = {"builder": primitives.builder_sha256, "fitter": primitives.fitter_sha256,
               "evaluator": evaluator_sha256, "draft": draft_sha256}
    for h in sources.values():
        _sha(h)
    cache = _cache(candles)
    features = primitives.build_features(_window(cache, origin_ms, origin_ms),
        origin_ms=origin_ms, as_of_ms=origin_ms, implementation_sha256=primitives.builder_sha256)
    if previous_state is None:
        state = {"schema": "exia.genetic_primus.reversal_rolling_state/1",
                 "sources_sha256": sources, "train_start_origin_ms": train_start_origin_ms,
                 "next_origin_ms": origin_ms, "policy_targets": {s: 0 for s in UNIVERSE},
                 "scenarios": {name: _flat_scenario() for name in SCENARIOS},
                 "evidence_class": "synthetic_engineering_only", "execution_allowed": False}
    else:
        payload = {k: v for k, v in previous_state.items() if k != "state_sha256"}
        if _digest(payload) != previous_state.get("state_sha256"):
            raise ValueError("state SHA mismatch")
        if (previous_state.get("schema") != "exia.genetic_primus.reversal_rolling_state/1"
                or previous_state.get("execution_allowed") is not False
                or previous_state.get("sources_sha256") != sources
                or previous_state.get("train_start_origin_ms") != train_start_origin_ms
                or previous_state.get("next_origin_ms") != origin_ms):
            raise ValueError("state/contract/cadence mismatch")
        state = deepcopy(payload)
        for scenario in state["scenarios"].values():
            for s, position in scenario["positions"].items():
                if not isclose(position["last_mark_price"], cache[(origin_ms-BAR_MS, s)].close,
                               rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("previous close revised: explicit replay required")
    cutoff = origin_ms-7*BAR_MS
    earliest = max(train_start_origin_ms, origin_ms-WINDOW_MS)
    first = train_start_origin_ms + max(0, (earliest-train_start_origin_ms+HORIZON_MS-1)//HORIZON_MS)*HORIZON_MS
    observations, bundles = [], []
    for train_origin in range(first, cutoff-HORIZON_MS+1, HORIZON_MS):
        end = train_origin+HORIZON_MS
        if train_origin < RESERVED_END_MS and end > RESERVED_START_MS:
            continue
        bundle = primitives.build_training_origin(_window(cache, train_origin, end),
            origin_ms=train_origin, as_of_ms=cutoff, implementation_sha256=primitives.builder_sha256)
        bundles.append(bundle["training_origin_sha256"])
        observations.extend(primitives.observation_type(**row) for row in bundle["observations"])
    model = primitives.fit_origin(observations, deploy_origin_ms=origin_ms,
                                  implementation_sha256=primitives.fitter_sha256)
    data_sha = _digest({"training_origins": bundles, "decision_features": features["features_sha256"]})
    decisions, targets = [], {}
    entry_fee = SCENARIOS["normal"][1]/20_000
    entry_threshold = (1+entry_fee)/(1-entry_fee)-1
    for symbol in UNIVERSE:
        previous_target = state["policy_targets"][symbol]
        forecast = primitives.predict_gross(model, features["relative_features"][symbol],
                                             feature_bar_start_ms=origin_ms-BAR_MS)
        threshold = 0.0 if previous_target else entry_threshold
        target = int(forecast is not None and forecast > threshold)
        targets[symbol] = target
        decisions.append({"origin": origin_ms, "decision_time": origin_ms, "symbol": symbol,
                          "model_sha256": model["model_sha256"], "train_cutoff": cutoff,
                          "data_sha256": data_sha, "folds_sha256": model["folds_sha256"],
                          "weights_sha256": model["weights_sha256"], "scaler_sha256": model["scaler_sha256"],
                          "previous_target": previous_target, "forecast": forecast,
                          "threshold": threshold, "target": target, "fallback": model["fallback"]})
    # Validate all future valuation candles only after the decision calculation.
    path = primitives.build_training_origin(_window(cache, origin_ms, origin_ms+HORIZON_MS),
        origin_ms=origin_ms, as_of_ms=origin_ms+HORIZON_MS,
        implementation_sha256=primitives.builder_sha256)
    scenarios = {name: _scenario(cache, origin_ms=origin_ms, targets=targets,
                 previous=state["scenarios"][name], delay_bars=delay, round_trip_bps=bps,
                 model_sha256=model["model_sha256"])
                 for name, (delay, bps) in SCENARIOS.items()}
    state.update(next_origin_ms=origin_ms+HORIZON_MS, policy_targets=targets,
                 scenarios={name: result["state"] for name, result in scenarios.items()})
    state["state_sha256"] = _digest(state)
    return {"schema": "exia.genetic_primus.reversal_rolling_step/1", "origin_ms": origin_ms,
            "evidence_class": "synthetic_engineering_only", "execution_allowed": False,
            "sources_sha256": sources, "model": model, "decisions": decisions,
            "scenarios": scenarios, "state": state,
            "valuation_source_sha256": path["input_ohlcv_sha256"],
            "risk_scope": "close_only_drawdown_not_intrabar_funding_or_fill_certification"}
