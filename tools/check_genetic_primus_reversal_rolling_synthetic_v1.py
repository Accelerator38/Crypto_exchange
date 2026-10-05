"""Bounded synthetic rolling checks without pytest, package init or collectors."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from math import sin, pi, isclose, prod
from pathlib import Path
import runpy
from statistics import median


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = {name: root / f"src/exia/genetic_primus/market_relative_reversal_{name}_v1.py"
             for name in ("builder", "linear", "rolling")}
    api = {name: runpy.run_path(str(path)) for name, path in paths.items()}
    b, f, r = api["builder"], api["linear"], api["rolling"]
    assert b["UNIVERSE"] == f["UNIVERSE"] == r["UNIVERSE"]
    primitives = r["Primitives"](b["build_features"], b["build_training_origin"], f["Observation"],
                    f["fit_origin"], f["predict_gross"], sha256(paths["builder"].read_bytes()).hexdigest(),
                    sha256(paths["linear"].read_bytes()).hexdigest())
    draft_path = root / "configs/genetic_primus_market_relative_reversal_draft_v1.json"
    draft_sha = sha256(draft_path.read_bytes()).hexdigest()
    draft = json.loads(draft_path.read_text())
    assert draft["rolling"]["purge_bars"] == 7
    bar, horizon, symbols = r["BAR_MS"], r["HORIZON_MS"], r["UNIVERSE"]
    origin, first = 1780272000000, 1780272000000-61*horizon
    candles = []
    for k in range(-61*18-13, 3*18):
        for i, symbol in enumerate(symbols):
            def price(t):
                return (100+5*i)*(1+(0.02+0.006*i)*sin(2*pi*t/36+pi/4))
            o, close = price(k), price(k+1)
            candles.append(b["Candle"](symbol, origin+k*bar, o, max(o, close)*1.001,
                                        min(o, close)*0.999, close, 100.0))
    evaluator_sha = sha256(paths["rolling"].read_bytes()).hexdigest()

    def step(data=candles, at=origin, state=None, kind="synthetic"):
        return r["evaluate_origin"](data, origin_ms=at, train_start_origin_ms=first,
                    previous_state=state, primitives=primitives, evaluator_sha256=evaluator_sha,
                    draft_sha256=draft_sha, data_kind=kind)

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("unsafe step accepted")

    checks = []
    one = step()
    required = set(draft["decisions"]["ledger_required"])
    assert len(one["decisions"]) == 8 and all(required <= set(d) for d in one["decisions"])
    assert all(d["model_sha256"] == one["model"]["model_sha256"] for d in one["decisions"])
    assert one["model"]["distinct_training_origins"] == 60
    assert one["model"]["train_cutoff_exclusive_ms"] == origin-7*bar
    assert any(d["target"] for d in one["decisions"])
    checks.append("causal_fit_purge_and_per_decision_model_SHA")

    changed = [replace(c, open=c.open*1.1, high=c.high*1.1, low=c.low*1.1, close=c.close*1.1)
               if c.open_ms >= origin else c for c in candles]
    replay = step(changed)
    assert replay["decisions"] == one["decisions"]
    assert replay["model"]["model_sha256"] == one["model"]["model_sha256"]
    assert step(reversed(candles))["state"]["state_sha256"] == one["state"]["state_sha256"]
    checks.append("future_prices_do_not_change_fit_or_decisions_and_order_invariance")

    state_before = deepcopy(one["state"])
    two = step(at=origin+horizon, state=one["state"])
    three = step(at=origin+2*horizon, state=two["state"])
    assert one["state"] == state_before
    assert [x["model"]["distinct_training_origins"] for x in (one, two, three)] == [60, 61, 62]
    assert len({x["model"]["model_sha256"] for x in (one, two, three)}) == 3
    # Refit may legitimately yield identical parameters on a stationary fixture.
    # Independently reconstruct each origin's eligible feature sample instead.
    lookup = {(c.open_ms, c.symbol): c for c in candles}
    for x in (one, two, three):
        samples = []
        for train_origin in range(first, x["origin_ms"]-2*horizon+1, horizon):
            returns = [lookup[(train_origin-bar,s)].close/lookup[(train_origin-13*bar,s)].close-1
                       for s in symbols]
            center = median(returns)
            samples.extend(value-center for value in returns)
        ordered = sorted(samples)
        def quantile(q):
            position = (len(ordered)-1)*q
            lower = int(position)
            return ordered[lower]*(1-(position-lower))+ordered[min(lower+1,len(ordered)-1)]*(position-lower)
        assert isclose(x["model"]["scaler"]["median"], median(samples), abs_tol=1e-12)
        assert isclose(x["model"]["scaler"]["iqr"], quantile(.75)-quantile(.25), abs_tol=1e-12)
    assert one["model"]["scaler"] is not two["model"]["scaler"]
    for x in (one, two, three):
        for name, scene in x["scenarios"].items():
            assert len(scene["daily_returns"]) == 3
            assert isclose(prod(1+v for v in scene["daily_returns"])-1, scene["origin_net_return"], abs_tol=1e-12)
            assert all(e["execution_time_ms"] == x["origin_ms"]+scene["delay_bars"]*bar for e in scene["events"])
    checks.append("three_day_refit_scaler_refresh_atomic_state_and_daily_reconciliation")

    # Controlled target schedule tests accounting, not a second model candidate.
    cache = r["_cache"](candles)
    for delay, bps in ((0, 16.0), (1, 40.0)):
        args = dict(delay_bars=delay, round_trip_bps=bps, model_sha256=one["model"]["model_sha256"])
        buy = r["_scenario"](cache, origin_ms=origin, targets={s:1 for s in symbols},
                              previous=r["_flat_scenario"](), **args)
        hold = r["_scenario"](cache, origin_ms=origin+horizon, targets={s:1 for s in symbols},
                               previous=buy["state"], **args)
        sell = r["_scenario"](cache, origin_ms=origin+2*horizon, targets={s:0 for s in symbols},
                               previous=hold["state"], **args)
        assert len(buy["events"]) == len(sell["events"]) == 8 and not hold["events"] and hold["fees"] == 0
        for event in buy["events"]:
            assert isclose(event["equity_after"], event["equity_before"]/(1+bps/20_000), abs_tol=1e-12)
        for event in sell["events"]:
            assert isclose(event["equity_after"], event["equity_before"]*(1-bps/20_000), abs_tol=1e-12)
        for s in symbols:
            assert buy["state"]["positions"][s]["quantity"] == hold["state"]["positions"][s]["quantity"]
        if delay == 1:
            assert buy["bar_close_nav"][0]["nav"] == 1.0
            expected = sum(p["quantity"]*cache[(origin+2*horizon, s)].close
                           for s,p in hold["state"]["positions"].items())/8
            assert isclose(sell["bar_close_nav"][0]["nav"], expected, abs_tol=1e-12)
    checks.append("transition_only_fees_carry_inventory_and_old_target_during_delay")

    rejects(lambda: step(kind="market"))
    rejects(lambda: step(at=origin+1))
    rejects(lambda: step(at=origin+2*horizon, state=one["state"]))
    bad = deepcopy(one["state"])
    bad["policy_targets"][symbols[0]] = 1-bad["policy_targets"][symbols[0]]
    rejects(lambda: step(at=origin+horizon, state=bad))
    rejects(lambda: step(at=r["RESERVED_START_MS"]))
    rejects(lambda: step([c for c in candles if not (c.open_ms == origin+bar and c.symbol == symbols[0])]))
    assert one["state"] == state_before
    checks.append("market_mode_cadence_altered_state_reserved_interval_and_missing_path_rejected")
    print(json.dumps({"passed":len(checks), "synthetic_only":True, "origins":3,
                      "decisions":24, "real_data_fits":0, "execution_allowed":False,
                      "last_state_sha256":three["state"]["state_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
