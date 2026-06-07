"""Тесты VotingPolicy: WeightedConsensus, StrongConsensus, RiskParity."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action, Regime
from panteon_v2.selection.voting import (
    RiskParity,
    StrongConsensus,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.tests._helpers import make_market


class TestThresholdProfile(unittest.TestCase):
    def test_default_valid(self):
        tp = ThresholdProfile()
        self.assertGreater(tp.open_single, tp.open_floor)

    def test_invalid_open_floor_higher(self):
        with self.assertRaises(ValueError):
            ThresholdProfile(open_floor=0.5, open_single=0.4)

    def test_invalid_open_multi_higher(self):
        with self.assertRaises(ValueError):
            ThresholdProfile(open_single=0.3, open_multi=0.5)


class TestWeightedConsensus(unittest.TestCase):
    def setUp(self):
        self.policy = WeightedConsensus()
        self.thresholds = ThresholdProfile()
        self.market = make_market(prices={"BTC": 100.0, "ETH": 50.0, "DOGE": 1.0})

    def test_empty_votes(self):
        out = self.policy.aggregate({}, {}, self.thresholds, self.market)
        self.assertEqual(out, {})

    def test_single_strong_long(self):
        # Один агент с весом 1.0 голосует за full long → ожидаем full long
        votes = {"A": {"BTC": Action.FUT_LONG_FULL}}
        weights = {"A": 1.0}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertEqual(out.get("BTC"), Action.FUT_LONG_FULL)

    def test_consensus_long(self):
        # Два агента согласны на long
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_LONG_HALF},
        }
        weights = {"A": 0.6, "B": 0.4}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        # Net = 0.6 + 0.4 = 1.0, выше open_single → full
        self.assertIn(out.get("BTC"), (Action.FUT_LONG_FULL, Action.FUT_LONG_HALF))

    def test_conflict_kills_open(self):
        # Голоса противоположные с равным весом → net=0 → не открываем
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_SHORT_FULL},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", out)

    def test_close_signal(self):
        # Закрытие — отдельный канал
        votes = {
            "A": {"BTC": Action.FUT_CLOSE_ALL},
            "B": {"BTC": Action.FUT_CLOSE_ALL},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertEqual(out.get("BTC"), Action.FUT_CLOSE_ALL)

    def test_close_signal_requires_close_share_not_just_absolute_weight(self):
        votes = {
            "A": {"BTC": Action.FUT_CLOSE_ALL},
            "B": {"BTC": Action.HOLD},
        }
        weights = {"A": 0.25, "B": 0.75}
        thresholds = ThresholdProfile(close_multi=0.22)

        out = self.policy.aggregate(votes, weights, thresholds, self.market)

        self.assertNotIn("BTC", out)

    def test_zero_weight_ignored(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_LONG_FULL},
        }
        weights = {"A": 1.0, "B": 0.0}  # B с нулевым весом
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        # Должен использовать только A
        self.assertEqual(out.get("BTC"), Action.FUT_LONG_FULL)

    def test_below_threshold_no_open(self):
        # Слабый сигнал — не открываемся
        votes = {"A": {"BTC": Action.FUT_LONG_HALF}}
        weights = {"A": 0.05}  # ниже open_floor=0.20
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", out)


class TestWeightedConsensusDirectional(unittest.TestCase):
    """Phase 2 / A1+B5: направленное голосование."""

    def setUp(self):
        self.policy = WeightedConsensus(directional=True)
        self.thresholds = ThresholdProfile()
        self.market = make_market(prices={"BTC": 100.0})

    def test_dominant_side_opens_despite_minority_opposition(self):
        # long=0.30, short=0.05. agents_voting=2 → open_thr=open_multi=0.26.
        # legacy: net=0.25 < 0.26 → НЕ открыл бы. directional: 0.30≥0.26 и >0.05 → long.
        votes = {
            "A": {"BTC": Action.FUT_LONG_HALF},
            "B": {"BTC": Action.FUT_SHORT_HALF},
        }
        weights = {"A": 0.30, "B": 0.05}
        legacy = WeightedConsensus().aggregate(votes, weights, self.thresholds, self.market)
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", legacy)
        self.assertIn(out.get("BTC"), (Action.FUT_LONG_FULL, Action.FUT_LONG_HALF))

    def test_tie_still_does_not_open(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_SHORT_FULL},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", out)

    def test_single_strong_long_opens_full(self):
        votes = {"A": {"BTC": Action.FUT_LONG_FULL}}
        weights = {"A": 1.0}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertEqual(out.get("BTC"), Action.FUT_LONG_FULL)

    def test_weak_signal_below_floor_no_open(self):
        votes = {"A": {"BTC": Action.FUT_LONG_HALF}}
        weights = {"A": 0.05}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", out)

    def test_close_not_blocked_by_passive_hold(self):
        # Позицию закрываем: один агент за close (0.30), другой HOLD (0.70).
        # legacy: score_keep=0.70 > score_close=0.30 → close ЗАБЛОКИРОВАН.
        # directional: нет противоположных open → close проходит.
        votes = {
            "A": {"BTC": Action.FUT_CLOSE_ALL},
            "B": {"BTC": Action.HOLD},
        }
        weights = {"A": 0.30, "B": 0.70}
        thresholds = ThresholdProfile(close_multi=0.22)
        legacy = WeightedConsensus().aggregate(votes, weights, thresholds, self.market)
        out = self.policy.aggregate(votes, weights, thresholds, self.market)
        self.assertNotIn("BTC", legacy)
        self.assertEqual(out.get("BTC"), Action.FUT_CLOSE_ALL)


class TestStrongConsensus(unittest.TestCase):
    def setUp(self):
        self.policy = StrongConsensus()
        self.thresholds = ThresholdProfile()
        self.market = make_market(prices={"BTC": 100.0})

    def test_unanimous_long_full(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_LONG_FULL},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertEqual(out.get("BTC"), Action.FUT_LONG_FULL)

    def test_disagreement_kills_open(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.HOLD},  # один HOLD → нет единогласия
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", out)

    def test_opposite_directions_kill_open(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_SHORT_FULL},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        self.assertNotIn("BTC", out)

    def test_close_priority(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_CLOSE_ALL},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        # Close имеет приоритет
        self.assertEqual(out.get("BTC"), Action.FUT_CLOSE_ALL)

    def test_mixed_full_half_returns_half(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_LONG_HALF},
        }
        weights = {"A": 0.5, "B": 0.5}
        out = self.policy.aggregate(votes, weights, self.thresholds, self.market)
        # Слабейшее общее → half
        self.assertEqual(out.get("BTC"), Action.FUT_LONG_HALF)


class TestRiskParity(unittest.TestCase):
    def test_default_falls_back_to_equal_weights(self):
        # Без volatilities → равные веса → должно работать как WeightedConsensus
        policy = RiskParity()
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL},
            "B": {"BTC": Action.FUT_LONG_FULL},
        }
        weights = {"A": 0.5, "B": 0.5}  # игнорируется
        market = make_market(prices={"BTC": 100.0})
        out = policy.aggregate(votes, weights, ThresholdProfile(), market)
        self.assertEqual(out.get("BTC"), Action.FUT_LONG_FULL)

    def test_uses_inverse_volatilities(self):
        # Высокая vol → меньший вес
        policy = RiskParity(volatilities={"A": 1.0, "B": 4.0})
        # A имеет vol=1, B имеет vol=4. inv: A=1, B=0.25, total=1.25
        # weights: A=0.8, B=0.2
        weights = policy._compute_weights(["A", "B"])
        self.assertAlmostEqual(weights["A"], 0.8, places=5)
        self.assertAlmostEqual(weights["B"], 0.2, places=5)


class TestVotingPolicyInvariants(unittest.TestCase):
    def test_risk_parity_with_equal_volatility_matches_weighted_consensus(self):
        votes = {
            "A": {"BTC": Action.FUT_LONG_FULL, "ETH": Action.FUT_CLOSE_ALL},
            "B": {"BTC": Action.FUT_LONG_HALF, "ETH": Action.FUT_CLOSE_ALL},
        }
        weights = {"A": 0.5, "B": 0.5}
        thresholds = ThresholdProfile()
        market = make_market(prices={"BTC": 100.0, "ETH": 50.0})

        weighted = WeightedConsensus().aggregate(votes, weights, thresholds, market)
        risk_parity = RiskParity(volatilities={"A": 2.0, "B": 2.0}).aggregate(
            votes,
            weights,
            thresholds,
            market,
        )

        self.assertEqual(risk_parity, weighted)

    def test_strong_open_consensus_is_subset_of_weighted_open_consensus(self):
        thresholds = ThresholdProfile()
        market = make_market(prices={"BTC": 100.0, "ETH": 50.0})
        cases = (
            {
                "A": {"BTC": Action.FUT_LONG_FULL},
                "B": {"BTC": Action.FUT_LONG_FULL},
            },
            {
                "A": {"ETH": Action.FUT_SHORT_FULL},
                "B": {"ETH": Action.HOLD},
            },
            {
                "A": {"BTC": Action.FUT_LONG_FULL},
                "B": {"BTC": Action.FUT_SHORT_FULL},
            },
        )
        weights = {"A": 0.5, "B": 0.5}

        for votes in cases:
            strong = StrongConsensus().aggregate(votes, weights, thresholds, market)
            weighted = WeightedConsensus().aggregate(votes, weights, thresholds, market)
            for symbol, action in strong.items():
                if action.is_open:
                    self.assertEqual(weighted.get(symbol), action)


if __name__ == "__main__":
    unittest.main(verbosity=2)
