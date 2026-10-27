"""Small deterministic adapter checks; explicitly no pytest/collector."""
from copy import deepcopy
import unittest

from exia.genetic_primus.compact_net_replay_v1 import (
    funding_input, rank_calibration, target_weights, validate_origin, fit_calibration_linear,
)
from exia.genetic_primus.unified_measurement_v1 import identity, BAR_MS
from exia.genetic_primus import unified_measurement_v1 as m


class AdapterTests(unittest.TestCase):
    def test_threshold_strict_and_negative_abstention(self):
        self.assertEqual(target_weights([.005, -.1, .006], ["A", "B", "C"], 50),
                         {"A": "0", "B": "0", "C": "0.10"})

    def test_reserve_no_redistribution(self):
        result = target_weights([1, 0] * 4, list("ABCDEFGH"), 20)
        self.assertEqual(sum(float(x) for x in result.values()), .4)
        with self.assertRaises(ValueError):
            target_weights([1] * 8, list("ABCDEFGH"), 20, ".11")

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                target_weights([value], ["X"], 20)

    def fixture(self):
        bars = [{"timestamp": t, "prices": {"BTC/USDT": "100"}} for t in range(0, 4 * BAR_MS, BAR_MS)]
        rates = [{"settlement_ms": t, "symbol": "BTCUSDT", "rate_fraction": sign,
                  "rate_bps": bps} for t, sign, bps in ((0, ".0001", "1"), (2 * BAR_MS, "-.0002", "-2"))]
        return rates, bars

    def build(self, rates, bars):
        return funding_input(rates, bars, start_ms=0, end_ms=4 * BAR_MS,
                             symbols=["BTC/USDT"], evidence_sha256="a" * 64)

    def test_funding_sign_units_and_proxy(self):
        rates, bars = self.fixture()
        result = self.build(rates, bars)
        self.assertEqual([x["signed_rate_bps"] for x in result["events"]], ["1", "-2"])
        self.assertFalse(result["observed_exchange_mark_verified"])
        self.assertEqual(result["provenance_scope"], "observed_funding_rates_only")

    def test_missing_duplicate_or_wrong_units_fail(self):
        rates, bars = self.fixture()
        for modified in (rates[:1], rates + rates[:1], rates[:1] + [dict(rates[1], rate_bps="2")]):
            with self.assertRaises(ValueError):
                self.build(modified, bars)

    def test_postterminal_cash_still_requires_rate(self):
        rates, bars = self.fixture()
        result = self.build(rates, bars[:2])
        self.assertEqual(result["post_terminal_rate_count"], 1)
        self.assertEqual(len(result["events"]), 1)
        with self.assertRaises(ValueError):
            self.build(rates[:1], bars[:2])

    def test_selection_worst_cost_and_drawdown(self):
        def row(threshold, normal, stress, dd=.05):
            return {"policy": "linear6_uniform", "threshold_bps": threshold,
                    "metrics": {s: {"net_log_growth": str(v), "max_drawdown": str(dd)}
                                for s, v in (("normal", normal), ("stress", stress))}}
        picked = rank_calibration([row(20, .2, -.1), row(50, .1, .08), row(100, .3, .2, .11)])
        self.assertEqual(picked["threshold_bps"], 50)
        self.assertFalse(picked["benchmark_used_for_selection"])

    def test_negative_selection_is_not_signal(self):
        rows = [{"policy": "linear6_uniform", "threshold_bps": 50,
                 "metrics": {s: {"net_log_growth": "-.01", "max_drawdown": ".01"} for s in ("normal", "stress")}}]
        self.assertFalse(rank_calibration(rows)["beats_NoTrade_point_estimate"])

    def integration(self, target):
        end, terminal = 24 * BAR_MS, 19 * BAR_MS
        bars = [{"timestamp": t, "prices": {"BTC/USDT": "100"}}
                for t in range(0, terminal + BAR_MS, BAR_MS)]
        rates = [{"symbol": "BTCUSDT", "settlement_ms": t, "rate_fraction": ".0001", "rate_bps": "1"}
                 for t in range(0, end, 2 * BAR_MS)]
        funding = funding_input(rates, bars, start_ms=0, end_ms=end,
                                symbols=["BTC/USDT"], evidence_sha256="a" * 64)
        decision = m.make_decision(origin_start_ms=0, decision_asof_ms=0,
            targets={"BTC/USDT": target}, model_sha256="a" * 64, scaler_sha256="b" * 64,
            feature_prefix_sha256="c" * 64, training_identity_sha256="d" * 64)
        costs = m.CostScenario("normal", "6", "6", "2", "2", "registered_assumption", "e" * 64)
        return m.replay(m.MeasurementSpec(0, end, terminal, "10000", ("BTC/USDT",), identity(bars)),
                        price_bars=bars, decisions=[decision], funding=funding, costs=costs)

    def test_NoTrade_has_zero_cost_and_funding(self):
        result = self.integration("0")
        self.assertEqual(float(result["metrics"]["final_capital_quote"]), 10000)
        self.assertEqual(float(result["metrics"]["funding_quote"]), 0)

    def test_flat_price_trade_loses_costs_and_funding(self):
        result = self.integration(".1")
        self.assertLess(float(result["metrics"]["net_return"]), 0)
        self.assertGreater(float(result["metrics"]["funding_quote"]), 0)
        self.assertEqual(len(m.daily_measurements(result)), 4)

    def test_calibration_fit_does_not_read_unmatured_labels(self):
        import numpy as np
        rng = np.random.default_rng(91)
        origins = np.arange(120, dtype=np.int64) * 18 * BAR_MS
        samples = {"origins": origins, "available": origins + 20 * BAR_MS,
                   "x": rng.normal(size=(120, 8, 6)), "y": rng.normal(size=(120, 8)) / 100}
        first = fit_calibration_linear(samples, int(origins[100]))
        changed = {k: v.copy() for k, v in samples.items()}
        changed["y"][changed["available"] > first["training_identity"]["cutoff_ms"]] = 99
        second = fit_calibration_linear(changed, int(origins[100]))
        self.assertEqual(first, second)
        validate_origin(first, int(origins[100]), expected_arms=("linear6_uniform",))
        self.assertEqual(first["model_fit_count"], 1)


if __name__ == "__main__":
    unittest.main()
