"""Tests for v2 market regime detection."""

from __future__ import annotations

import unittest

from panteon_v2.app.regime_detector import PriceRegimeDetector
from panteon_v2.domain.types import Regime


class TestPriceRegimeDetector(unittest.TestCase):
    def test_symbol_exchange_threshold_profiles_calibrate_same_move_differently(self):
        detector = PriceRegimeDetector(
            exchange_name="MEXC",
            poll_interval_sec=5,
            horizons_sec=(60, 180),
            hysteresis_bars=1,
            min_history_bars=12,
            regime_thresholds={
                "MEXC:BTC": {
                    "trend_threshold": 0.0003,
                    "low_volatility": 0.0030,
                    "choppy_volatility": 0.0060,
                },
                "MEXC:LAB": {
                    "trend_threshold": 0.0200,
                    "low_volatility": 0.0200,
                    "choppy_volatility": 0.0400,
                },
            },
        )

        for idx in range(180):
            price = 100.0 * (1.00004 ** idx)
            detector.update({"BTC": price, "LAB": price})

        self.assertEqual(detector.symbol_regimes["BTC"], Regime.BULLISH)
        self.assertEqual(detector.symbol_regimes["LAB"], Regime.RANGE_LOW_VOL)
        self.assertAlmostEqual(
            detector.symbol_stats["BTC"]["trend_threshold"],
            0.0003,
        )
        self.assertAlmostEqual(
            detector.symbol_stats["LAB"]["trend_threshold"],
            0.0200,
        )

    def test_symbol_hysteresis_requires_repeated_confirmation_before_flip(self):
        detector = PriceRegimeDetector(
            poll_interval_sec=5,
            horizons_sec=(10,),
            hysteresis_bars=3,
            min_history_bars=2,
            bullish_return=0.003,
            bearish_return=-0.003,
        )

        detector.update({"BTC": 100.0})
        detector.update({"BTC": 100.0})
        self.assertEqual(detector.symbol_regimes["BTC"], Regime.RANGE_LOW_VOL)

        detector.update({"BTC": 99.6})
        self.assertEqual(detector.symbol_regimes["BTC"], Regime.RANGE_LOW_VOL)
        detector.update({"BTC": 99.2})
        self.assertEqual(detector.symbol_regimes["BTC"], Regime.RANGE_LOW_VOL)
        detector.update({"BTC": 98.8})
        self.assertEqual(detector.symbol_regimes["BTC"], Regime.BEARISH)

    def test_symbol_stats_include_volatility_percentile_funding_and_liquidity(self):
        detector = PriceRegimeDetector(
            poll_interval_sec=5,
            horizons_sec=(60, 180),
            hysteresis_bars=1,
            min_history_bars=12,
            bullish_return=0.003,
            bearish_return=-0.003,
        )

        for idx in range(120):
            detector.update(
                {"BTC": 100.0 + (0.02 if idx % 2 else -0.02)},
                volumes={"BTC": 10_000.0},
                funding={"BTC": 0.0},
            )
        for idx in range(30):
            detector.update(
                {"BTC": 100.0 - idx * 0.08 + (0.85 if idx % 2 else -0.85)},
                volumes={"BTC": 100.0},
                funding={"BTC": -0.04},
            )

        stats = detector.symbol_stats["BTC"]
        self.assertEqual(detector.symbol_regimes["BTC"], Regime.CHOPPY_DOWN)
        self.assertGreater(stats["volatility_percentile"], 0.80)
        self.assertLess(stats["liquidity_percentile"], 0.30)
        self.assertAlmostEqual(stats["funding_rate"], -0.04)
        self.assertEqual(stats["carry_risk"], "short_pays_funding")
        self.assertGreater(stats["risk_pressure"], 1.0)

    def test_micro_macro_context_separates_short_trend_from_hour_bias(self):
        detector = PriceRegimeDetector(
            poll_interval_sec=5,
            horizons_sec=(60, 180, 720, 3600),
            hysteresis_bars=1,
            min_history_bars=12,
        )

        for idx in range(684):
            detector.update({"BTC": 100.0 + idx * 0.03})
        peak = 100.0 + 683 * 0.03
        for idx in range(36):
            detector.update({"BTC": peak - idx * 0.12})

        stats = detector.symbol_stats["BTC"]
        self.assertEqual(stats["micro_direction"], "down")
        self.assertEqual(stats["macro_direction"], "up")
        self.assertEqual(stats["micro_macro_alignment"], "conflict")

    def test_classifies_symbol_regimes_with_short_crypto_windows(self):
        detector = PriceRegimeDetector(
            poll_interval_sec=5,
            horizons_sec=(60, 180, 720, 3600),
            hysteresis_bars=1,
            min_history_bars=12,
        )

        for idx in range(720):
            detector.update({
                "BTC": 100.0 * (1.00002 ** idx),
                "ETH": 100.0 * (0.99990 ** idx),
                "SOL": 100.0 + (0.02 if idx % 2 else -0.02),
                "XRP": 100.0 * (1.00090 ** idx),
            })

        self.assertEqual(detector.symbol_regimes["BTC"], Regime.RANGE_LOW_VOL)
        self.assertEqual(detector.symbol_regimes["ETH"], Regime.BEARISH)
        self.assertEqual(detector.symbol_regimes["SOL"], Regime.RANGE_LOW_VOL)
        self.assertEqual(detector.symbol_regimes["XRP"], Regime.BULLISH)
        self.assertEqual(detector.current, Regime.MIXED_ROTATIONAL)
        self.assertGreater(detector.confidence, 0.0)

    def test_choppy_direction_uses_volatility_adjusted_return(self):
        detector = PriceRegimeDetector(
            poll_interval_sec=5,
            horizons_sec=(60, 180, 720, 3600),
            hysteresis_bars=1,
            min_history_bars=12,
        )

        for idx in range(240):
            detector.update({
                "BTC": 100.0 - idx * 0.025 + (0.75 if idx % 2 else -0.75),
                "ETH": 100.0 + idx * 0.025 + (-0.75 if idx % 2 else 0.75),
            })

        self.assertEqual(detector.symbol_regimes["BTC"], Regime.CHOPPY_DOWN)
        self.assertEqual(detector.symbol_regimes["ETH"], Regime.CHOPPY_UP)
        self.assertEqual(detector.current, Regime.MIXED_ROTATIONAL)


if __name__ == "__main__":
    unittest.main()
