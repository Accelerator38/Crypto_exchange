"""Small direct safety regressions; synthetic in-memory candles only."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import runpy


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return
    raise AssertionError("unsafe input accepted")


def main():
    root = Path(__file__).resolve().parents[1]
    p = root / "src/exia/genetic_primus"
    reader = runpy.run_path(str(p / "reversal_calibration_reader_v1.py"))
    b = runpy.run_path(str(p / "market_relative_reversal_builder_v1.py"))
    f = runpy.run_path(str(p / "market_relative_reversal_linear_v1.py"))
    e = runpy.run_path(str(p / "market_relative_reversal_rolling_v1.py"))
    digest = reader["digest"]
    start, bar = reader["START_MS"], reader["BAR_MS"]
    assert reader["bounds"](reader["LOWER_MS"], reader["END_MS"]) == 37976
    for lower, upper in ((reader["LOWER_MS"]-bar, start), (start, reader["END_MS"]+bar),
                         (start+1, start+bar), (start, start)):
        rejected(lambda: reader["bounds"](lower, upper))
    rows = [dict(timestamp=start, symbol=s, open=1., high=1., low=1., close=1., volume=1.)
            for s in reader["UNIVERSE"]]
    kwargs = dict(start_ms=start, end_ms=start+bar, candle_type=b["Candle"])
    assert len(reader["validate_rows"](rows, **kwargs)) == 8
    rejected(lambda: reader["validate_rows"](rows[:-1], **kwargs))
    rejected(lambda: reader["validate_rows"](rows[:-1]+[rows[0]], **kwargs))
    rejected(lambda: reader["local_path"](root, "../outside.parquet"))
    sources = {name: sha256((p / path).read_bytes()).hexdigest() for name, path in
               {"builder": "market_relative_reversal_builder_v1.py",
                "fitter": "market_relative_reversal_linear_v1.py",
                "evaluator": "market_relative_reversal_rolling_v1.py"}.items()}
    primitive = e["Primitives"](b["build_features"], b["build_training_origin"], f["Observation"],
        f["fit_origin"], f["predict_gross"], sources["builder"], sources["fitter"])
    draft_sha = sha256((root / "configs/genetic_primus_market_relative_reversal_draft_v1.json").read_bytes()).hexdigest()
    registration = {"schema": "exia.genetic_primus.reversal_calibration_registration/1",
        "user_authorized_one_offline_pilot": True, "candidate_budget": 1, "origin_count": 20,
        "start_ms": start, "end_ms": reader["END_MS"], "sources_sha256": sources,
        "model_draft_sha256": draft_sha, **{k: False for k in
          ("credentials_allowed", "orders_enabled", "paper", "live", "promotion_authority")}}
    registration["registration_sha256"] = digest(registration)
    args = dict(origin_ms=start, train_start_origin_ms=reader["TRAIN_ANCHOR_MS"],
        previous_state=None, primitives=primitive, evaluator_sha256=sources["evaluator"],
        draft_sha256=draft_sha, data_kind="revealed_calibration", calibration_registration=registration)
    rejected(lambda: e["evaluate_origin"]([], **{**args, "calibration_registration": None}))
    for key, value in (("live", True), ("origin_count", 21), ("candidate_budget", 2),
                       ("user_authorized_one_offline_pilot", False)):
        bad = deepcopy(registration)
        bad[key] = value
        bad["registration_sha256"] = digest({k: v for k, v in bad.items() if k != "registration_sha256"})
        rejected(lambda: e["evaluate_origin"]([], **{**args, "calibration_registration": bad}))
    rejected(lambda: e["evaluate_origin"]([], **{**args, "origin_ms": start+reader["DAY_MS"]}))
    rejected(lambda: e["evaluate_origin"]([], **{**args, "data_kind": "market"}))
    candles = [b["Candle"](s, t, 1., 1., 1., 1., 1.)
        for t in range(reader["LOWER_MS"], start+6*reader["DAY_MS"], bar)
        for s in reader["UNIVERSE"]]
    first = e["evaluate_origin"](candles, **args)
    assert not first["execution_allowed"]
    assert first["evidence_class"] == "revealed_development_calibration_not_independent_validation"
    assert all(d["target"] == 0 for d in first["decisions"])
    second = e["evaluate_origin"](candles, **{**args, "origin_ms": start+3*reader["DAY_MS"],
                                             "previous_state": first["state"]})
    assert all(second["scenarios"][s]["daily_returns"] == [0., 0., 0.] for s in ("normal", "stress"))
    bad_state = deepcopy(first["state"])
    bad_state["next_origin_ms"] += bar
    rejected(lambda: e["evaluate_origin"](candles, **{**args, "origin_ms": start+3*reader["DAY_MS"],
                                                     "previous_state": bad_state}))
    print("PASS: bounds, full8, paths, registration negatives, two synthetic origins, state-chain guard")


if __name__ == "__main__":
    main()
