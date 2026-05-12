"""Тесты pure scoring functions.

Цель: зафиксировать гарантии monotonicity, determinism, граничных
случаев. Скоринг — самая критичная часть, на ней держится весь Selector.
"""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Metrics, Regime
from panteon_v2.scoring import (
    DEFAULT_SCORING,
    ScoringConfig,
    confidence_from_sample,
    is_hopeless_in_all_regimes,
    is_locally_proven,
    regime_score,
)


class TestConfidence(unittest.TestCase):
    def test_zero_closed(self):
        self.assertEqual(confidence_from_sample(0), 0.0)

    def test_full_at_threshold(self):
        # min_closed_for_full_confidence = 6 (default)
        self.assertEqual(confidence_from_sample(6), 1.0)
        self.assertEqual(confidence_from_sample(60), 1.0)  # capped

    def test_partial(self):
        c3 = confidence_from_sample(3)
        c4 = confidence_from_sample(4)
        self.assertGreater(c3, 0.0)
        self.assertGreater(c4, c3)
        self.assertLess(c4, 1.0)


class TestRegimeScore(unittest.TestCase):
    def test_empty_metrics_zero(self):
        self.assertEqual(regime_score(Metrics.empty(), Regime.NEUTRAL), 0.0)

    def test_monotonic_in_pnl(self):
        """Score монотонно возрастает по pnl_pct при прочих равных."""
        base_kwargs = dict(closed_trades=5, entries=5, signals=10,
                           wins=3, losses=2, sharpe=0.5, max_dd_pct=2.0)
        scores = []
        for pnl in [-2.0, -1.0, -0.1, 0.0, 0.1, 1.0, 2.0]:
            m = Metrics(pnl_pct=pnl, **base_kwargs)
            scores.append(regime_score(m, Regime.NEUTRAL))
        # Должны быть отсортированы по возрастанию
        for i in range(len(scores) - 1):
            self.assertLessEqual(
                scores[i], scores[i + 1],
                f"score not monotonic: {scores}",
            )

    def test_deterministic(self):
        m = Metrics(pnl_pct=0.5, closed_trades=5, entries=5,
                    signals=10, wins=3, losses=2)
        s1 = regime_score(m, Regime.BULLISH)
        s2 = regime_score(m, Regime.BULLISH)
        self.assertEqual(s1, s2)

    def test_few_closed_dampened(self):
        """При closed < 2 даём ослабленный сигнал, не игнорируем полностью."""
        m = Metrics(pnl_pct=2.0, closed_trades=1, entries=1, signals=2,
                    wins=1, losses=0)
        score = regime_score(m, Regime.BULLISH)
        # Должен быть положительным, но скромнее чем при closed >= 6
        self.assertGreater(score, 0.0)
        m_full = Metrics(pnl_pct=2.0, closed_trades=10, entries=10, signals=10,
                         wins=6, losses=4, sharpe=0.5)
        score_full = regime_score(m_full, Regime.BULLISH)
        self.assertGreater(score_full, score)

    def test_max_dd_penalty(self):
        m_low_dd = Metrics(pnl_pct=1.0, closed_trades=5, entries=5, signals=8,
                           wins=3, losses=2, max_dd_pct=0.5)
        m_high_dd = Metrics(pnl_pct=1.0, closed_trades=5, entries=5, signals=8,
                            wins=3, losses=2, max_dd_pct=10.0)
        s_low = regime_score(m_low_dd, Regime.NEUTRAL)
        s_high = regime_score(m_high_dd, Regime.NEUTRAL)
        self.assertGreater(s_low, s_high)

    def test_inactivity_penalty(self):
        """Полностью неактивный → отрицательный score (даже без других факторов)."""
        m = Metrics()  # signals=0, entries=0
        # has_data = False → возвращаем 0, не применяем penalty
        self.assertEqual(regime_score(m, Regime.NEUTRAL), 0.0)

    def test_custom_config(self):
        cfg = ScoringConfig(pnl_weight=2.0)  # вдвое больше веса на pnl
        m = Metrics(pnl_pct=1.0, closed_trades=10, entries=10, signals=10,
                    wins=5, losses=5)
        s_default = regime_score(m, Regime.NEUTRAL)
        s_strong = regime_score(m, Regime.NEUTRAL, config=cfg)
        # При большем pnl_weight pnl-component вносит вдвое больше
        self.assertGreater(s_strong, s_default)

    def test_explicit_confidence_overrides(self):
        m = Metrics(pnl_pct=1.0, closed_trades=10, entries=10, signals=10,
                    wins=5, losses=5)
        s_full = regime_score(m, Regime.NEUTRAL)
        s_half = regime_score(m, Regime.NEUTRAL, confidence=0.5)
        self.assertGreater(s_full, s_half)


class TestIsLocallyProven(unittest.TestCase):
    def test_no_data(self):
        self.assertFalse(is_locally_proven({}))

    def test_negative_only(self):
        per = {Regime.BULLISH: Metrics(pnl_pct=-0.5, closed_trades=10, wins=4, losses=6)}
        self.assertFalse(is_locally_proven(per))

    def test_one_positive_enough(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=0.5, closed_trades=5, wins=3, losses=2),
            Regime.BEARISH: Metrics(pnl_pct=-0.3, closed_trades=10, wins=3, losses=7),
        }
        self.assertTrue(is_locally_proven(per))

    def test_positive_but_low_sample(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=2.0, closed_trades=1, wins=1, losses=0)
        }
        # closed < recovery_closed (3) → не считается
        self.assertFalse(is_locally_proven(per))

    def test_positive_below_threshold(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=0.05, closed_trades=10, wins=5, losses=5)
        }
        # 0.05 < default 0.10 → не считается
        self.assertFalse(is_locally_proven(per))


class TestIsHopelessInAllRegimes(unittest.TestCase):
    def test_no_data(self):
        self.assertFalse(is_hopeless_in_all_regimes({}))

    def test_one_positive_saves(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=0.5, closed_trades=3, wins=2, losses=1),
            Regime.BEARISH: Metrics(pnl_pct=-1.0, closed_trades=3, wins=0, losses=3),
        }
        self.assertFalse(is_hopeless_in_all_regimes(per))

    def test_all_negative_enough_sample(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=-0.5, closed_trades=3, wins=0, losses=3),
            Regime.BEARISH: Metrics(pnl_pct=-0.6, closed_trades=2, wins=0, losses=2),
            Regime.NEUTRAL: Metrics(pnl_pct=-0.4, closed_trades=2, wins=0, losses=2),
        }
        # 3+2+2=7 >= 5; worst=-0.6 <= -0.30 → hopeless
        self.assertTrue(is_hopeless_in_all_regimes(per))

    def test_all_negative_but_small_sample(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=-0.5, closed_trades=2, wins=0, losses=2),
            Regime.BEARISH: Metrics(pnl_pct=-0.5, closed_trades=2, wins=0, losses=2),
        }
        # Total closed=4 < 5 → не hopeless (нужно больше данных)
        self.assertFalse(is_hopeless_in_all_regimes(per))

    def test_all_negative_but_above_hard_threshold(self):
        per = {
            Regime.BULLISH: Metrics(pnl_pct=-0.10, closed_trades=5, wins=0, losses=5),
            Regime.BEARISH: Metrics(pnl_pct=-0.05, closed_trades=5, wins=0, losses=5),
        }
        # worst=-0.10 > -0.30 (hard_neg threshold) → не hopeless
        self.assertFalse(is_hopeless_in_all_regimes(per))


if __name__ == "__main__":
    unittest.main(verbosity=2)
