"""Small direct unittest regressions; never collector, network, pytest discovery."""
from __future__ import annotations

import unittest
import numpy as np

from exia.genetic_primus import compact_forecast_v1 as m
from exia.genetic_primus import compact_forecast_data_v1 as d


class ModelTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(80, 6))
        y = 0.03 * x[:, 0] - 0.02 * x[:, 1] + 0.01
        return x, y, np.linspace(1, 2, 80)

    def test_counts(self):
        self.assertEqual(len(m.initialize_mlp(11)), 65)
        self.assertEqual(m.PARAMETERS, {"linear6": 7, "mlp8": 65})

    def test_uniform_weights(self):
        self.assertTrue(np.allclose(m.temporal_weights(np.array([0, m.DAY_MS]), 2*m.DAY_MS, "uniform"), [.5, .5]))

    def test_half_life_anchor(self):
        self.assertTrue(np.allclose(m.temporal_weights(np.array([0, 90*m.DAY_MS]), 90*m.DAY_MS, "half_life_90_anchor20"), [11/30, 19/30]))

    def test_anchor_is_normalized_twenty_percent_mass(self):
        times = np.arange(500, dtype=np.int64)*3*m.DAY_MS
        cutoff = int(times[-1])+m.DAY_MS
        w = m.temporal_weights(times, cutoff, "half_life_90_anchor20")
        recency = np.exp2(-(cutoff-times)/(90*m.DAY_MS))
        self.assertTrue(np.allclose((w-.2/len(times))/.8, recency/recency.sum()))
        self.assertAlmostEqual(float(w.sum()), 1)

    def test_weights_reject_future(self):
        with self.assertRaises(ValueError):
            m.temporal_weights(np.array([2]), 1, "uniform")

    def test_weights_reject_duplicate_or_unknown(self):
        with self.assertRaises(ValueError):
            m.temporal_weights(np.array([1, 1]), 2, "uniform")
        with self.assertRaises(ValueError):
            m.temporal_weights(np.array([1]), 2, "profits_only")

    def test_origin_ess_not_eightfold(self):
        self.assertAlmostEqual(m.origin_ess(np.ones(10)), 10)

    def test_loss_includes_losses(self):
        loss, _ = m.loss_gradient(np.zeros(7), np.zeros((2, 6)), np.array([-2., 1.]), np.array([.6, .4]), 0, "linear6")
        self.assertAlmostEqual(loss, 2.8)

    def test_linear_gradient(self):
        self.gradient_check("linear6", np.arange(7)*.01)

    def test_mlp_gradient(self):
        self.gradient_check("mlp8", m.initialize_mlp(11))

    def gradient_check(self, arch, p):
        x, y, w = self.fixture()
        _, grad = m.loss_gradient(p, x, y, w, .02, arch)
        numerical = []
        for j in range(len(p)):
            plus, minus = p.copy(), p.copy()
            plus[j] += 1e-6
            minus[j] -= 1e-6
            numerical.append((m.loss_gradient(plus, x, y, w, .02, arch)[0] - m.loss_gradient(minus, x, y, w, .02, arch)[0])/2e-6)
        self.assertTrue(np.allclose(grad, numerical, atol=1e-7, rtol=1e-5))

    def test_bias_not_regularized(self):
        p = np.zeros(65)
        p[48:56], p[64] = 2, 3
        x = np.ones((5, 6))
        zero = m.loss_gradient(p, x, np.zeros(5), np.ones(5), 0, "mlp8")
        penalty = m.loss_gradient(p, x, np.zeros(5), np.ones(5), 1, "mlp8")
        self.assertEqual(zero[0], penalty[0])
        self.assertTrue(np.array_equal(zero[1], penalty[1]))

    def test_scaler_train_only_constant_and_clipping(self):
        x, y, _ = self.fixture()
        x[:, 5] = 1
        scaler = m.fit_scaler(x, y)
        before = scaler.sha256
        self.assertEqual(scaler.scale[5], 1)
        self.assertTrue(np.all(np.abs(scaler.transform(np.ones((1, 6))*1e6)) <= 8))
        self.assertEqual(before, scaler.sha256)

    def test_weighted_ridge_stationary(self):
        x, y, w = self.fixture()
        p = m.fit_linear(x, y, w)
        _, grad = m.loss_gradient(p, x, y, w, .01, "linear6")
        self.assertLess(np.linalg.norm(grad), 1e-10)

    def test_mlp_deterministic_bounded(self):
        x, y, w = self.fixture()
        p, trace = m.fit_mlp(x, y, w, seed=11, epochs=8)
        q, other = m.fit_mlp(x, y, w, seed=11, epochs=8)
        self.assertTrue(np.array_equal(p, q))
        self.assertEqual(trace, other)
        self.assertEqual(trace["gradient_steps"], 8)

    def test_model_sha_and_reload(self):
        x, y, w = self.fixture()
        scaler = m.fit_scaler(x, y)
        p = m.fit_linear(scaler.transform(x), scaler.target(y), w)
        payload = m.model_payload("linear6", p, scaler, weighting="uniform", seed=None, epochs=None, training_identity={})
        self.assertEqual(m.predict(payload, x).shape, (80,))
        payload["parameters"][0] += 1
        with self.assertRaises(ValueError):
            m.predict(payload, x)

    def test_invalid_fit_rejected(self):
        x, y, w = self.fixture()
        with self.assertRaises(ValueError):
            m.fit_linear(x, y, -w)
        with self.assertRaises(ValueError):
            m.fit_mlp(x, y, w, seed=11, epochs=121)
        with self.assertRaises(ValueError):
            m.fit_scaler(x[:, :5], y)


class FeatureTests(unittest.TestCase):
    def panel(self, count=160):
        times = np.arange(count, dtype=np.int64)*d.BAR_MS
        rng = np.random.default_rng(39)
        close = 100*np.exp(np.cumsum(rng.normal(0, .01, size=(count, 8)), axis=0))
        return times, close

    def test_prefix_invariance(self):
        t, c = self.panel()
        prefix = d.feature_panel(t[:100], c[:100])
        c[100:] *= 3
        full = d.feature_panel(t, c)
        self.assertTrue(np.array_equal(prefix[48:], full[48:100]))

    def test_flat_policy(self):
        t, _ = self.panel()
        value = d.feature_panel(t, np.ones((len(t), 8))*100)
        self.assertTrue(np.array_equal(value[48:], np.zeros_like(value[48:])))

    def test_six_features_full_universe(self):
        t, c = self.panel()
        value = d.feature_panel(t, c)
        self.assertEqual(value.shape, (160, 8, 6))
        self.assertTrue(np.allclose(value[48:, :, 3].mean(axis=1), 0))
        self.assertTrue(np.all((value[48:, :, 2] >= 0) & (value[48:, :, 2] <= 1+1e-12)))
        self.assertTrue(np.all(value[48:, :, 5] == value[48:, :1, 5]))

    def test_reject_missing_grid_or_price(self):
        t, c = self.panel()
        with self.assertRaises(ValueError):
            d.feature_panel(t[1:][::2], c[1:][::2])
        c[0, 0] = 0
        with self.assertRaises(ValueError):
            d.feature_panel(t, c)

    def test_exact_open_open_labels(self):
        t, c = self.panel(1400)
        samples = d.make_samples(t, c, c, 60*d.BAR_MS)
        origin = int(samples["origins"][0])
        self.assertEqual(samples["entry"][0], origin+d.BAR_MS)
        self.assertEqual(samples["exit"][0], origin+19*d.BAR_MS)
        self.assertEqual(samples["available"][0], origin+20*d.BAR_MS)
        self.assertTrue(np.allclose(samples["y"][0], c[origin//d.BAR_MS+19]/c[origin//d.BAR_MS+1]-1))

    def test_purged_inner_and_outer(self):
        t, c = self.panel(1400)
        samples = d.make_samples(t, c, c, 60*d.BAR_MS)
        origin = int(samples["origins"][-1])
        split = d.eligible_split(samples, origin)
        self.assertEqual(split["cutoff_ms"], origin-8*d.BAR_MS)
        self.assertTrue(np.all(samples["available"][split["train"]] <= split["cutoff_ms"]))
        self.assertTrue(np.all(samples["available"][split["inner_train"]] <= split["inner_cutoff_ms"]))
        self.assertFalse(set(split["inner_train"]) & set(split["inner_validation"]))


class RollingTests(unittest.TestCase):
    def test_fixed_arms_common_scaler_all_losses_and_future_independence(self):
        t, c = FeatureTests().panel(1400)
        samples = d.make_samples(t, c, c, 60*d.BAR_MS)
        origin = int(samples["origins"][-1])
        first = d.fit_origin(samples, origin)
        samples["y"][-1] = 999
        second = d.fit_origin(samples, origin)
        self.assertEqual(first["arms"], second["arms"])
        self.assertEqual(first["model_fit_count"], 14)
        self.assertEqual(len(first["arms"]), 4)
        self.assertGreater(first["target_positive_count"], 0)
        self.assertGreater(first["target_negative_count"], 0)
        for arm in first["arms"].values():
            self.assertGreater(arm["negative_weight_mass"], 0)
            self.assertGreater(arm["positive_weight_mass"], 0)
            for fitted in arm["models"]:
                self.assertEqual(fitted["scaler_sha256"], first["scaler_sha256"])


if __name__ == "__main__":
    unittest.main()
