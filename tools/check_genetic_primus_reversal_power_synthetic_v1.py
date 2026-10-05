"""Deterministic hypothetical inputs only; no data fitting, bootstrap or network."""

from math import isclose
from pathlib import Path
import json
import runpy


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    api = runpy.run_path(str(root / "src/exia/genetic_primus/reversal_power_design_v1.py"))
    args = dict(design_effect_daily=.0003, long_run_sigma_daily={"normal":.0005,"stress":.0015},
                signal_probability=.35, minimum_signal_origins=20, alpha=.05,
                target_joint_power=.8, maximum_calendar_days=720)
    plan = api["plan_duration"]
    checks = []

    def rejects(call):
        try:
            call()
        except ValueError:
            return
        raise AssertionError("invalid power input accepted")

    zero = plan(**dict(args, design_effect_daily=0))
    assert zero["calendar_days"] is None and not zero["feasible_under_assumptions"]
    for n in (3, 300, 30000):
        assert isclose(api["gaussian_mean_power"](calendar_days=n, design_effect_daily=0,
                       long_run_sigma_daily=.0015, alpha=.05), .05, abs_tol=1e-12)
    checks.append("zero_effect_has_null_power_and_no_finite_design")

    tail = api["iid_signal_support_probability"]
    assert isclose(tail(origins=3, signal_probability=.5, minimum_signal_origins=2), .5, abs_tol=1e-12)
    assert isclose(tail(origins=3, signal_probability=.5, minimum_signal_origins=1), .875, abs_tol=1e-12)
    assert tail(origins=3, signal_probability=1, minimum_signal_origins=4) == 0
    assert tail(origins=3, signal_probability=1, minimum_signal_origins=3) == 1
    checks.append("analytic_binomial_tail_and_support_floor")

    base = plan(**args)
    assert base["feasible_under_assumptions"] and base["calendar_days"] % 3 == 0
    assert base["joint_power_lower_bound_under_assumptions"] >= .8
    assert min(base["normal_approx_mean_powers"].values()) >= base["per_gate_power_target"]
    assert base["iid_support_probability"] >= base["per_gate_power_target"]
    prior = plan(**dict(args, maximum_calendar_days=base["calendar_days"]-3))
    assert not prior["feasible_under_assumptions"]
    checks.append("three_gate_failure_budget_grid_and_minimum_duration")

    smaller = plan(**dict(args, design_effect_daily=.0002, maximum_calendar_days=1800))
    rare = plan(**dict(args, signal_probability=.10, maximum_calendar_days=1800))
    assert smaller["calendar_days"] >= base["calendar_days"]
    assert rare["calendar_days"] >= base["calendar_days"]
    assert not plan(**dict(args, signal_probability=0))["feasible_under_assumptions"]
    checks.append("effect_and_activity_sensitivity_no_forced_trades")

    lr = api["ar1_long_run_sigma"](daily_sigma=.0015, rho=.6)
    assert isclose(lr, .003, abs_tol=1e-12)
    inflated = plan(**dict(args, long_run_sigma_daily={"normal":.001,"stress":lr},
                           maximum_calendar_days=1800))
    assert inflated["calendar_days"] >= 3.5*base["calendar_days"]
    checks.append("positive_autocorrelation_increases_required_calendar_time")

    rejects(lambda: plan(**dict(args, design_effect_daily=-.01)))
    rejects(lambda: plan(**dict(args, signal_probability=True)))
    rejects(lambda: plan(**dict(args, long_run_sigma_daily={"normal":.001,"stress":float('nan')})))
    rejects(lambda: plan(**dict(args, long_run_sigma_daily={"normal":.001})))
    rejects(lambda: api["ar1_long_run_sigma"](daily_sigma=.001, rho=1))
    checks.append("invalid_nonfinite_and_nonstationary_assumptions_rejected")

    assert plan(**args) == base
    assert not base["registration_allowed"] and not base["power_design_complete"]
    assert not base["bootstrap_and_full_promotion_gate_power_verified"]
    checks.append("determinism_and_no_automatic_registration_or_power_claim")
    print(json.dumps({"passed":len(checks), "hypothetical_only":True,
                      "illustrative_days":base["calendar_days"],
                      "zero_effect_days":zero["calendar_days"],
                      "market_files_read":0, "registration_allowed":False}, sort_keys=True))


if __name__ == "__main__":
    main()
