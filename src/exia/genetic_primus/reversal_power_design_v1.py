"""Conditional planning calculations, not calibrated power or registration."""

from math import ceil, exp, expm1, fsum, isfinite, lgamma, log, log1p, sqrt
from statistics import NormalDist
from typing import Mapping


def _number(value: object) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("finite numeric input required")
    return float(value)


def _alpha(value: float) -> float:
    value = _number(value)
    if not 0 < value < 0.5 or not 0 < 1-value < 1:
        raise ValueError("representable one-sided alpha in (0,0.5) required")
    return value


def gaussian_mean_power(*, calendar_days: int, design_effect_daily: float,
                        long_run_sigma_daily: float, alpha: float) -> float:
    """Gaussian mean approximation with known long-run variance, not row ESS."""
    effect, sigma, alpha = _number(design_effect_daily), _number(long_run_sigma_daily), _alpha(alpha)
    if type(calendar_days) is not int or calendar_days <= 0 or effect < 0 or sigma <= 0:
        raise ValueError("positive days/sigma and nonnegative effect required")
    critical = NormalDist().inv_cdf(1-alpha)
    return NormalDist().cdf(effect*sqrt(calendar_days)/sigma-critical)


def ar1_long_run_sigma(*, daily_sigma: float, rho: float) -> float:
    """Illustrative stationary AR(1) formula; not an estimate from market data."""
    sigma, rho = _number(daily_sigma), _number(rho)
    if sigma <= 0 or not -1 < rho < 1:
        raise ValueError("positive sigma and stationary rho required")
    return _number(sigma*sqrt((1+rho)/(1-rho)))


def iid_signal_support_probability(*, origins: int, signal_probability: float,
                                   minimum_signal_origins: int) -> float:
    """Binomial tail under an explicit IID origin assumption, not a guarantee."""
    p = _number(signal_probability)
    if (type(origins) is not int or origins < 0 or not 0 <= p <= 1
            or type(minimum_signal_origins) is not int or minimum_signal_origins <= 0):
        raise ValueError("valid origin count, probability and support floor required")
    if origins < minimum_signal_origins or p == 0:
        return 0.0
    if p == 1:
        return 1.0
    log_masses = [lgamma(origins+1)-lgamma(k+1)-lgamma(origins-k+1)
                  + k*log(p)+(origins-k)*log1p(-p)
                  for k in range(minimum_signal_origins)]
    highest = max(log_masses)
    log_cdf = highest+log(fsum(exp(v-highest) for v in log_masses))
    return max(0.0, min(1.0, -expm1(min(0.0, log_cdf))))


def plan_duration(*, design_effect_daily: float,
                  long_run_sigma_daily: Mapping[str, float],
                  signal_probability: float, minimum_signal_origins: int,
                  alpha: float, target_joint_power: float,
                  maximum_calendar_days: int) -> dict:
    """Size a hypothetical mean/support check, on a fixed three-day grid.

    Failure probability is allocated to normal mean, stress mean and signal
    support. A union bound does not require these gates to be independent.
    Their marginal models STILL require known long-run sigmas, a Gaussian
    mean approximation, and IID signal origins. None is calibrated here.
    Bootstrap-bank/risk gates and adaptive-testing history are not covered.
    """
    effect, p = _number(design_effect_daily), _number(signal_probability)
    alpha, target = _alpha(alpha), _number(target_joint_power)
    if (effect < 0 or not 0 <= p <= 1 or not alpha < target < 1
            or type(minimum_signal_origins) is not int or minimum_signal_origins <= 0
            or type(maximum_calendar_days) is not int or not 3 <= maximum_calendar_days <= 36600):
        raise ValueError("invalid bounded power design")
    if not isinstance(long_run_sigma_daily, Mapping) or set(long_run_sigma_daily) != {"normal", "stress"}:
        raise ValueError("normal and stress long-run sigmas required")
    sigmas = {key: _number(value) for key, value in long_run_sigma_daily.items()}
    if any(value <= 0 for value in sigmas.values()):
        raise ValueError("positive long-run sigmas required")
    per_gate_target = 1-(1-target)/3
    if per_gate_target == 1:
        raise ValueError("unrepresentable per-gate power")
    result = {
        "schema": "exia.genetic_primus.conditional_power_plan/1",
        "feasible_under_assumptions": False, "calendar_days": None, "origins": None,
        "design_effect_daily": effect, "alpha": alpha, "target_joint_power": target,
        "per_gate_power_target": per_gate_target,
        "status": "not_feasible_within_cap",
        "assumptions": ["known_long_run_daily_sigmas", "Gaussian_mean_approximation",
                        "IID_signal_origins", "same_positive_effect_floor_in_both_cost_scenarios"],
        "power_scope": "two_mean_gates_and_origin_support_only",
        "audit_banks_are_independent_market_samples": False,
        "bootstrap_and_full_promotion_gate_power_verified": False,
        "power_design_complete": False, "registration_allowed": False,
        "execution_allowed": False,
    }
    if effect == 0:
        result.update(status="no_positive_design_effect_no_finite_power_size",
                      mean_gate_power_at_null=alpha)
        return result
    if p == 0:
        result["status"] = "no_signal_opportunities_under_assumption"
        return result
    z = NormalDist().inv_cdf(1-alpha)+NormalDist().inv_cdf(per_gate_target)
    scales = [z*sigma/effect for sigma in sigmas.values()]
    sizes = [value*value for value in scales]
    if not all(isfinite(value) for value in sizes) or max(sizes) > maximum_calendar_days:
        return result
    first_day = max(3, 3*ceil(max(sizes)/3), 3*minimum_signal_origins)
    for days in range(first_day, maximum_calendar_days+1, 3):
        origins = days//3
        support = iid_signal_support_probability(origins=origins, signal_probability=p,
                                                 minimum_signal_origins=minimum_signal_origins)
        if support < per_gate_target:
            continue
        powers = {name: gaussian_mean_power(calendar_days=days, design_effect_daily=effect,
                    long_run_sigma_daily=sigma, alpha=alpha) for name, sigma in sigmas.items()}
        if min(powers.values()) < per_gate_target:
            continue
        result.update(feasible_under_assumptions=True, calendar_days=days, origins=origins,
                      status="conditional_size_not_registration",
                      normal_approx_mean_powers=powers, iid_support_probability=support,
                      expected_signal_origins=origins*p,
                      joint_power_lower_bound_under_assumptions=max(0.0, fsum(powers.values())+support-2))
        return result
    return result
