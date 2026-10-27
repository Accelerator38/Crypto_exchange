"""Direct deterministic tests: no pytest, collector, network or market data."""
import math
from pathlib import Path
import runpy

import numpy as np

CI = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/run_genetic_primus_validation_ci_v1.py"))
from exia.genetic_primus.measurement_lab_v2 import block_bank

METHODS = ["hac_small_sample_t", "studentized_block_bootstrap"]


def test_hac_matches_independent_and_dense_covariance_oracles():
    values = np.array([[1., 2., 3., 4., 7., 2., -1., 5.]])
    se = CI["hac_standard_error"](values, 0)
    np.testing.assert_allclose(se, values.std(axis=1, ddof=1) / np.sqrt(8))
    lags = 3
    distance = np.abs(np.arange(8)[:, None] - np.arange(8)[None, :])
    kernel = np.maximum(1.0 - distance / (lags + 1), 0.0)
    centered = values[0] - values[0].mean()
    expected = np.sqrt(centered @ kernel @ centered / (8 * 7))
    np.testing.assert_allclose(CI["hac_standard_error"](values, lags), expected)


def test_studentized_lcb_matches_scalar_oracle():
    paths = np.random.default_rng(16).normal(0, .01, (3, 12))
    bank = block_bank(12, 32, 4, 15)
    actual, _ = CI["ci_bounds"](paths, [METHODS[1]], bank, .025)
    for index, path in enumerate(paths):
        roots = []
        for indices in bank:
            sample = path[indices]
            se_star = float(CI["hac_standard_error"](sample, 3))
            roots.append((sample.mean() - path.mean()) / se_star if se_star > 1e-14 else np.inf)
        critical = max(float(np.quantile(roots, .975, method="higher")), CI["NORMAL"].inv_cdf(.975))
        expected = path.mean() - critical * float(CI["hac_standard_error"](path, 3))
        np.testing.assert_allclose(actual[METHODS[1]][index], expected, rtol=0, atol=1e-14)


def test_affine_equivariance_and_batch_invariance():
    paths = np.random.default_rng(21).normal(0, .01, (5, 30))
    bank = block_bank(30, 64, 6, 17)
    first, _ = CI["ci_bounds"](paths, METHODS, bank, .025, 1)
    second, _ = CI["ci_bounds"](paths * 2 + .003, METHODS, bank, .025, 2)
    for method in METHODS:
        np.testing.assert_allclose(second[method], first[method] * 2 + .003, rtol=0, atol=1e-14)


def test_zero_variance_and_degenerate_bootstrap_fail_closed():
    bank = block_bank(30, 64, 6, 9)
    for constant in (0.0, .001):
        bounds, details = CI["ci_bounds"](np.full((3, 30), constant), METHODS, bank, .025)
        assert details["uninformative_original_count"] == 3
        assert all(np.isneginf(value).all() for value in bounds.values())
    paths = np.arange(30, dtype=float)[None, :] * .001
    bounds, _ = CI["ci_bounds"](paths, [METHODS[1]], np.zeros((64, 30), dtype=np.int64), .025)
    assert np.isneginf(bounds[METHODS[1]]).all()


def test_generators_deterministic_and_audit_noise_separate():
    names = ["iid_gaussian", "ar1_gaussian_06", "ar1_t5_06", "sparse_markov25_ar1_06", "volatility_shift_ar1_06"]
    for name in names:
        first = CI["generate_paths"](name, 30, 4, 57)
        np.testing.assert_array_equal(first, CI["generate_paths"](name, 30, 4, 57))
        assert not np.array_equal(first, CI["generate_paths"](name, 30, 4, 59))
        assert first.shape == (4, 30) and np.isfinite(first).all()
    sparse = CI["generate_paths"](names[3], 365, 4, 57)
    assert (sparse == 0).any() and (sparse != 0).any()


def test_selection_rejects_false_positives_and_degenerate_no_trade():
    def evidence(null_upper, power, score):
        return {"null_upper_bounds": [null_upper] * 10,
                "positive_control": {"lower": power}, "power10": {"lower": score}}
    bad = {METHODS[0]: evidence(.06, .95, .5), METHODS[1]: evidence(.01, .1, .01)}
    winner, checks = CI["select_candidate"](bad, METHODS, .05, .8)
    assert winner is None and not any(value["passed"] for value in checks.values())
    good = {METHODS[0]: evidence(.04, .9, .3), METHODS[1]: evidence(.04, .9, .3)}
    assert CI["select_candidate"](good, METHODS, .05, .8)[0] == METHODS[0]
    good[METHODS[1]]["power10"]["lower"] = .4
    assert CI["select_candidate"](good, METHODS, .05, .8)[0] == METHODS[1]


def test_audit_cannot_open_without_selection():
    missing = Path(__file__).resolve().parents[1] / "Reports/Exia/Genetic_Primus/nonexistent_ci_audit_guard"
    assert not missing.exists()
    try:
        CI["phase_methods"]({}, missing, "audit")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("audit opened without primary selection")


def test_invalid_inputs_and_work_bounds():
    bank = block_bank(30, 64, 6, 6)
    try:
        CI["ci_bounds"](np.full((3, 30), np.nan), METHODS, bank, .025)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite path accepted")
    try:
        CI["generate_paths"]("iid_gaussian", 366, 2, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("workload bound ignored")
    config, _ = CI["load_contract"]()
    assert len(CI["cases"](config)) == 10
    assert not any(config["safety"].values())
    assert not config["automatic_runtime_integration"]
