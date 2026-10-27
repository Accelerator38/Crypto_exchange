"""User-requested 0.1% sensitivity, separate from the frozen main experiment."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_ternary_v2 as base
run = base.run
BASE_CONFIG = run.CONFIG
SUP_CONFIG = run.ROOT / "configs/genetic_primus_direction_ternary_01pct_v1.json"
run.CONFIG = SUP_CONFIG
run.OUT = run.ROOT / "Reports/Exia/Genetic_Primus/direction_ternary_01pct_v1_20261013"
run.REGISTRY = run.ROOT / "configs/genetic_primus_objective_registry_v21.json"
run.CODE = [Path(__file__), Path(base.__file__), Path(run.__file__), SUP_CONFIG, BASE_CONFIG,
            run.ROOT / "src/exia/genetic_primus/direction_ternary_v1.py",
            run.ROOT / "src/exia/genetic_primus/direction_selective_v1.py", Path(run.parent.__file__)]


def configuration():
    c = dict(run.read(BASE_CONFIG), **run.read(SUP_CONFIG))
    if any(c[k] is not False for k in run.SAFETY): raise ValueError("unsafe supplement")
    return c


run.config = configuration


def score():
    run.registered(); c = configuration()
    rows, records, inputs = [], [], {}
    days = (run.parent.END-run.parent.START)//run.core.DAY
    for minutes in c["timeframes_minutes"]:
        daily = run.np.zeros((8, len(run.METHODS), days, 3, 3), run.np.int64)
        for number in range(21):
            prefix = run.OUT / "origin_{:02d}".format(number) / "tf_{}m".format(minutes)
            meta = run.read(prefix.with_suffix(".json")); path = prefix.with_suffix(".npz")
            if meta["code_pins"] != run.pins() or run.parent.digest(path) != meta["ledger_sha256"]:
                raise ValueError("supplement ledger changed")
            for model in meta["models"]:
                if run.parent.digest(prefix.parent/model["file"]) != model["model_sha256"]: raise ValueError("supplement model changed")
            records.extend(meta["models"])
            inputs[str(path.relative_to(run.ROOT)).replace("\\", "/")] = run.parent.digest(path)
            with run.np.load(path, allow_pickle=False) as archive: ledger = {k: archive[k].copy() for k in archive.files}
            assert run.np.array_equal(run.core.labels(ledger["return_pct"], .1), ledger["labels"][0])
            predictions = run.predictions(ledger, 0)
            for asset in range(8):
                for k, pred in enumerate(predictions):
                    daily[asset, k] += run.core.daily_confusion(ledger["times"], ledger["labels"][0, :, asset],
                                                              pred[:, asset], run.parent.START, days)
        for asset, symbol in enumerate(c["symbols"]):
            for k, method in enumerate(run.METHODS):
                rows.append(dict(minutes=minutes, symbol=symbol, scheme="fixed_0.1pct", method=method,
                    band_pct_min=.1, band_pct_median=.1, band_pct_max=.1,
                    **run.core.from_confusion(daily[asset, k].sum(axis=0))))
    run.parent.write(run.OUT / "results.json", {"rows": rows, "fit_attempts": len(records),
        "fitted_estimators": sum(m["kind"] == "three_class_balanced_logit" for m in records),
        "prior_only": sum(m["kind"] == "prior_only_sparse_classes" for m in records),
        "estimator_warning_count": sum(bool(m["warnings"]) for m in records), "input_pins": inputs,
        "descriptive_user_requested_supplement": True, "inferential_family_cells": 0,
        "independent_validation": False, **{k: False for k in run.SAFETY}})
    print(run.json.dumps({"supplement_rows": len(rows), "fitted_estimators": sum(m["kind"] == "three_class_balanced_logit" for m in records)}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        run.parent.write(run.OUT / "supplement_reason.json", {"reason": "explicit user request for +/-0.1%",
            "main_completed_horizon_ledgers_at_registration": len(list((run.ROOT/"Reports/Exia/Genetic_Primus/direction_ternary_v2_20261013").glob("origin_*/*.npz"))),
            "main_accuracy_results_examined_for_supplement": False,
            "main_fit_repetitions": 0, "independent_validation": False})
        with redirect_stdout(io.StringIO()): run.init()
        print(run.json.dumps({"registry": "v21", "supplement_attempt_budget": 84, "band_pct": .1}))
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        run.main()


if __name__ == "__main__": main()
