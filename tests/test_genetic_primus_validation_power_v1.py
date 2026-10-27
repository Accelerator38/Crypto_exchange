"""Small deterministic regressions; invoke directly, not through pytest."""
from pathlib import Path
import runpy

import numpy as np

POWER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_genetic_primus_validation_power_v1.py"))
from exia.genetic_primus.measurement_lab_v2 import block_bank, paired_lcb


def test_exact_gaussian_variance_oracle():
    variance = POWER["mean_variance"]
    assert variance(1, .6, 2.0) == 4.0
    assert abs(variance(2, .6) - .8) < 1e-14
    assert abs(variance(30, 0) - 1 / 30) < 1e-14
    for rho in (0.0, .3, .6):
        indices = np.arange(11)
        covariance = rho ** np.abs(indices[:, None] - indices[None, :])
        assert abs(variance(11, rho) - covariance.sum() / 121) < 1e-14


def test_oracle_null_level_and_power_monotonicity():
    power = POWER["oracle_power"]
    for rho in (0.0, .3, .6):
        assert abs(power(30, rho, .01, 0.0, .05) - .05) < 1e-14
        assert power(180, rho, .01, .001, .05) > power(30, rho, .01, .001, .05)
        assert power(30, rho, .01, .001, .05) > power(30, rho, .01, .0008, .05)


def test_stationary_initialization_and_reproducibility():
    generate = POWER["stationary_ar1"]
    seed = 193
    paths = generate(7, 30, .6, seed)
    innovations = np.random.default_rng(seed).standard_normal((7, 30))
    np.testing.assert_array_equal(paths[:, 0], innovations[:, 0])
    np.testing.assert_allclose(paths[:, 1], .6 * paths[:, 0] + .8 * innovations[:, 1])
    np.testing.assert_array_equal(paths, generate(7, 30, .6, seed))
    assert not np.array_equal(paths, generate(7, 30, .6, seed + 1))


def test_batched_lcb_matches_existing_measurement_function():
    paths = POWER["stationary_ar1"](7, 30, .3, 17)
    bank = block_bank(30, 256, 5, 21)
    expected = np.array([paired_lcb(row, np.zeros(30), bank, .05) for row in paths])
    for batch in (1, 3, 8):
        actual = POWER["batched_lcbs"](paths, bank, .05, batch)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    scaled = .001 + .02 * paths
    scaled_lcb = POWER["batched_lcbs"](scaled, bank, .05)
    np.testing.assert_allclose(scaled_lcb, .001 + .02 * expected, rtol=0, atol=1e-14)


def test_zero_negative_and_positive_controls():
    bank = block_bank(30, 256, 5, 22)
    for value in (-.001, 0., .001):
        observed = POWER["batched_lcbs"](np.full((5, 30), value), bank, .05)
        np.testing.assert_allclose(observed, value, rtol=0, atol=1e-14)
        assert bool(np.all(observed > 0)) == (value > 0)


def test_drawdown_includes_initial_capital():
    dd = POWER["daily_drawdown"](np.array([[-.10, 0.0], [.10, -.10], [0., 0.]]))
    np.testing.assert_allclose(dd, [.10, .10, 0.0])
    try:
        POWER["daily_drawdown"](np.array([[-1., 0.]]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid equity path was accepted")


def test_monte_carlo_intervals_and_inflation_diagnosis():
    interval = POWER["rate_interval"]
    classify = POWER["classify_null"]
    assert interval(0, 1024)["upper"] > 0
    assert interval(1024, 1024)["lower"] < 1
    confidence = 1 - .05 / 45
    assert classify(interval(120, 1024, confidence), .05) == "FALSE_POSITIVE_INFLATION_DETECTED"
    assert classify(interval(51, 1024, confidence), .05) == "INCONCLUSIVE_NOT_PROOF_OF_CALIBRATION"
    assert classify(interval(0, 1024, confidence), .05) == "CONSERVATIVE_IN_THIS_SYNTHETIC_CELL"


def test_invalid_inputs_fail_closed():
    for invalid in (np.array([[1., np.nan]]), np.array([[1., np.inf]])):
        try:
            POWER["batched_lcbs"](invalid, np.array([[0, 1]]), .05)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite input accepted")
    try:
        POWER["stationary_ar1"](5, 30, 1.0, 7)
    except ValueError:
        pass
    else:
        raise AssertionError("nonstationary process accepted")


def test_contract_has_no_search_or_trading_authority():
    config, _ = POWER["load_contract"]()
    assert len(POWER["cells"](config)) == 15
    assert config["new_strategy_objectives"] == 0
    assert not any(config["safety"].values())
    assert not config["automatic_validation_horizon_selection"]
    assert not config["automatic_threshold_or_block_length_changes"]
