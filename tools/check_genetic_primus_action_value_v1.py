"""Bounded offline pure checks and immutable semantic-contract registration."""
from __future__ import annotations

import os
for _thread_var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"
os.environ["MPLBACKEND"] = "Agg"

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus.action_value_v1 import paired_action_value, transition

OUT = ROOT / "Reports/Exia/Genetic_Primus/action_value_v1_20261012"
CONFIG = ROOT / "configs/genetic_primus_action_value_v1.json"


def pin(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9):
        raise AssertionError((a, b))


def checks():
    base = dict(initial_cash=1000., current_units=1., action_units=0.,
                price_now=100., price_end=100., reference_units_end=1.,
                fee_bps=6., drag_bps=8.)
    def value(**changes):
        return paired_action_value(**(base | changes))
    cases = []
    close(value(action_units=1.)["delta_nav"], 0.); cases.append("hold_is_zero")
    close(value()["delta_nav"], -.28); cases.append("flat_roundtrip_28bps")
    close(value(price_end=90.)["delta_nav"], 10. - .139952 - .1260432)
    cases.append("falling_price_positive_after_costs")
    close(value(price_end=110.)["delta_nav"], -10. - .139952 - .1540528)
    cases.append("rising_price_negative_after_costs")
    close(value(reference_units_end=0.)["delta_nav"], 0.)
    cases.append("natural_exit_no_fictitious_reentry")
    close(value(funding_events=((100., .01),))["delta_nav"], .72)
    cases.append("positive_funding_saved")
    close(value(funding_events=((100., -.01),))["delta_nav"], -1.28)
    cases.append("negative_funding_missed")
    close(value(price_end=90., action_units=.5)["delta_nav"],
          .5 * value(price_end=90.)["delta_nav"])
    cases.append("partial_is_scaled_full_action")
    close(value(initial_cash=3000., current_units=3., reference_units_end=3.)["delta_nav"], -.84)
    cases.append("notional_scaling")
    capped = value(initial_cash=0., price_end=110.)
    if not capped["terminal_reference_capped"] or not 0. < capped["common_terminal_units"] < 1.:
        raise AssertionError("common endpoint cap")
    cases.append("common_cash_feasible_endpoint")
    kept = transition(1000., 1., 1., 100., 6., 8.)
    close(kept["fees"], 0.); close(kept["slippage"], 0.)
    cases.append("hold_no_initial_fee")

    rejected = 0
    for change in (dict(price_now=0.), dict(price_end=float("nan")),
                   dict(initial_cash=-1.), dict(current_units=0.),
                   dict(action_units=2.), dict(reference_units_end=2.),
                   dict(fee_bps=10000.), dict(drag_bps=-1.),
                   dict(action_units=True),
                   dict(initial_cash=0., action_units=1., funding_events=((100., .01),))):
        try:
            value(**change)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("invalid input accepted: " + str(change))
    try:
        transition(0., 0., 1., 100., 6., 8.)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("cash borrowing accepted")

    # Independent closed-form counterfactual accounting, finite 648-case grid.
    grid_count = 0
    for pend, qa, ref, cash, fee_bps, drag_bps, rate in itertools.product(
            (80., 100., 120.), (0., .5, 1.), (0., .5, 1.),
            (100., 1000.), (0., 6.), (0., 8.), (-.01, 0., .01)):
        fee, drag = fee_bps / 10000., drag_bps / 10000.
        def cost(before, after, price):
            change = after - before
            executed = price * (1. + drag if change > 0. else 1. - drag)
            return abs(change) * price * drag + abs(change) * executed * fee
        funding_per_unit = 100. * rate - pend * rate / 2.
        expected = ((qa - 1.) * (pend - 100.) - cost(1., qa, 100.)
                    - cost(qa, ref, pend) + cost(1., ref, pend)
                    - (qa - 1.) * funding_per_unit)
        result = value(initial_cash=cash, action_units=qa, price_end=pend,
                       reference_units_end=ref, fee_bps=fee_bps, drag_bps=drag_bps,
                       funding_events=((100., rate), (pend, -rate / 2.)))
        close(result["delta_nav"], expected)
        close(result["common_terminal_units"], ref)
        close(result["delta_bps_initial_nav"], 10000. * expected / (cash + 100.))
        grid_count += 1
    write("direct_checks.json", {"schema": "action_value_checks/1", "passed": True,
          "named_cases": cases, "invalid_cases_rejected": rejected,
          "independent_grid_cases": grid_count, "new_training_fits": 0,
          "financial_replays": 0, "network_calls": 0, "collector_calls": 0})
    print(json.dumps({"passed": True, "named_checks": len(cases),
                      "rejected_invalid": rejected, "grid_cases": grid_count,
                      "new_training_fits": 0, "network_calls": 0}))


def seal():
    # Read-only registered parent validation; no model/collector/replay call.
    from tools import run_genetic_primus_ema_units_v1 as parent
    parent_reg = parent.registration()
    contract = json.loads(CONFIG.read_text(encoding="utf-8"))
    if any(value is not False for value in contract["safety"].values()):
        raise ValueError("all safety authorities must remain false")
    result = json.loads((OUT / "direct_checks.json").read_text(encoding="utf-8"))
    if result["passed"] is not True or result["independent_grid_cases"] != 648:
        raise ValueError("pure checks missing")
    parents, count = {}, 0
    for folder in (parent.b.OUT, parent.previous.OUT, parent.OUT):
        receipt_path = folder / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        parents[str(receipt_path.relative_to(ROOT)).replace("\\", "/")] = pin(receipt_path)
        for name, expected in receipt["artifact_pins"].items():
            path = folder / name
            if pin(path) != expected:
                raise ValueError("parent artifact changed: " + str(path))
            parents[str(path.relative_to(ROOT)).replace("\\", "/")] = expected
            count += 1
    paths = [CONFIG, ROOT / "src/exia/genetic_primus/action_value_v1.py",
             ROOT / "tools/check_genetic_primus_action_value_v1.py",
             ROOT / "docs/GENETIC_PRIMUS_ACTION_VALUE_V1.md", OUT / "direct_checks.json"]
    registration = {"schema": "action_value_semantic_registration/1", "contract": contract,
                    "pins": {str(path.relative_to(ROOT)).replace("\\", "/"): pin(path) for path in paths},
                    "parent_pins": parents, "parent_registration_sha256": parent_reg["sha256"],
                    "parent_artifacts_verified": count,
                    "new_training_fits": 0, "new_financial_replays": 0,
                    "not_a_training_dataset_registration": True}
    encoded = json.dumps(registration, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    registration["sha256"] = hashlib.sha256(encoded).hexdigest()
    write("registration.json", registration)
    write("receipt.json", {"schema": "action_value_semantic_receipt/1",
          "registration_sha256": registration["sha256"],
          "artifact_pins": {path.name: pin(path) for path in OUT.iterdir() if path.is_file()}})
    print(json.dumps({"registered": registration["sha256"], "parent_artifacts_verified": count,
                      "new_training_fits": 0, "new_financial_replays": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("checks", "seal"))
    args = parser.parse_args()
    globals()[args.stage]()
