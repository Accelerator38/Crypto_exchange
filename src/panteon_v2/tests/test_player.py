"""Тесты EnsemblePlayer."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action, Regime
from panteon_v2.selection.player import EnsemblePlayer, Player
from panteon_v2.selection.voting import (
    StrongConsensus,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.tests._helpers import FakeAgent, make_market


class TestEnsemblePlayerConstruction(unittest.TestCase):
    def test_valid(self):
        a1 = FakeAgent("A")
        a2 = FakeAgent("B")
        p = EnsemblePlayer(
            label="Test",
            agents=[a1, a2],
            weights={"A": 0.5, "B": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        self.assertEqual(p.label, "Test")
        self.assertEqual(p.agent_labels, ["A", "B"])

    def test_implements_player_protocol(self):
        # Pyright/mypy не проверяет в runtime, но visual sanity:
        a = FakeAgent("X")
        p = EnsemblePlayer(
            label="T", agents=[a], weights={"X": 1.0},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        self.assertTrue(hasattr(p, "label"))
        self.assertTrue(hasattr(p, "vote"))
        self.assertTrue(hasattr(p, "agent_labels"))

    def test_ensemble_actor_type_is_explicit(self):
        a = FakeAgent("X")
        p = EnsemblePlayer(
            label="T", agents=[a], weights={"X": 1.0},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )

        self.assertEqual(p.actor_type, "ensemble")

    def test_empty_label_rejected(self):
        with self.assertRaises(ValueError):
            EnsemblePlayer(
                label="", agents=[], weights={},
                voting=WeightedConsensus(), thresholds=ThresholdProfile(),
            )

    def test_weights_mismatch_rejected(self):
        a = FakeAgent("A")
        with self.assertRaises(ValueError):
            EnsemblePlayer(
                label="T", agents=[a], weights={"B": 1.0},
                voting=WeightedConsensus(), thresholds=ThresholdProfile(),
            )

    def test_weights_not_normalized_rejected(self):
        a = FakeAgent("A")
        with self.assertRaises(ValueError):
            EnsemblePlayer(
                label="T", agents=[a], weights={"A": 0.7},  # sum != 1
                voting=WeightedConsensus(), thresholds=ThresholdProfile(),
            )


class TestEnsemblePlayerVote(unittest.TestCase):
    def test_agents_receive_symbol_local_regime(self):
        class RegimeAwareAgent:
            label = "Local"

            def act(self, market):
                if market.regime == Regime.BULLISH:
                    return {sym: Action.FUT_LONG_FULL for sym in market.prices}
                if market.regime == Regime.BEARISH:
                    return {sym: Action.FUT_SHORT_FULL for sym in market.prices}
                return {sym: Action.HOLD for sym in market.prices}

        p = EnsemblePlayer(
            label="LocalTeam",
            agents=[RegimeAwareAgent()],
            weights={"Local": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        market = make_market(
            regime=Regime.NEUTRAL,
            prices={"BTC": 100.0, "ETH": 50.0},
            regimes_by_symbol={"BTC": Regime.BULLISH, "ETH": Regime.BEARISH},
        )

        signals, errors = p.vote(market, signal_id_start=10)
        by_symbol = {signal.sym: signal for signal in signals}

        self.assertEqual(errors, [])
        self.assertEqual(by_symbol["BTC"].action, Action.FUT_LONG_FULL)
        self.assertEqual(by_symbol["BTC"].regime, Regime.BULLISH)
        self.assertEqual(by_symbol["ETH"].action, Action.FUT_SHORT_FULL)
        self.assertEqual(by_symbol["ETH"].regime, Regime.BEARISH)

    def test_simple_long_signal(self):
        a1 = FakeAgent("A", {"BTC": Action.FUT_LONG_FULL})
        a2 = FakeAgent("B", {"BTC": Action.FUT_LONG_FULL})
        p = EnsemblePlayer(
            label="LongTeam",
            agents=[a1, a2],
            weights={"A": 0.5, "B": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})
        signals, errors = p.vote(market, signal_id_start=10)
        self.assertEqual(errors, [])
        self.assertEqual(len(signals), 1)
        s = signals[0]
        self.assertEqual(s.id, 10)
        self.assertEqual(s.sym, "BTC")
        self.assertTrue(s.action.is_long_open)
        self.assertEqual(s.by_player, "LongTeam")
        # Атрибуция: один из A/B
        self.assertIn(s.by_agent, ("A", "B"))

    def test_signal_ids_incremental(self):
        a1 = FakeAgent("A", {"BTC": Action.FUT_LONG_FULL,
                              "ETH": Action.FUT_LONG_FULL})
        p = EnsemblePlayer(
            label="T", agents=[a1], weights={"A": 1.0},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0, "ETH": 50.0})
        signals, errors = p.vote(market, signal_id_start=100)
        self.assertEqual(errors, [])
        self.assertEqual(len(signals), 2)
        ids = sorted([s.id for s in signals])
        # Должны быть последовательны
        self.assertEqual(ids[1] - ids[0], 1)
        self.assertEqual(ids[0], 100)

    def test_hold_no_signals(self):
        a = FakeAgent("A", {"BTC": Action.HOLD})
        p = EnsemblePlayer(
            label="T", agents=[a], weights={"A": 1.0},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})
        signals, errors = p.vote(market, signal_id_start=1)
        self.assertEqual(signals, [])
        self.assertEqual(errors, [])

    def test_unknown_sym_filtered(self):
        # Агент возвращает sym, которого нет в market.prices
        a = FakeAgent("A", {"UNKNOWN_SYM": Action.FUT_LONG_FULL})
        p = EnsemblePlayer(
            label="T", agents=[a], weights={"A": 1.0},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})
        signals, errors = p.vote(market, signal_id_start=1)
        self.assertEqual(signals, [])
        self.assertEqual(errors, [])

    def test_agent_exception_isolated(self):
        class CrashingAgent:
            label = "Crash"
            def act(self, market):
                raise RuntimeError("boom")
        a1 = CrashingAgent()
        a2 = FakeAgent("B", {"BTC": Action.FUT_LONG_FULL})
        p = EnsemblePlayer(
            label="T", agents=[a1, a2], weights={"Crash": 0.5, "B": 0.5},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})
        # B всё равно даёт сигнал, но weight 0.5 ниже open_floor (=0.20):
        # достаточен, потому что 0.5 > 0.34. Должен пройти.
        signals, errors = p.vote(market, signal_id_start=1)
        # Crash воспринимается как HOLD; B голосует FULL_LONG → score=0.5,
        # выше open_single. Открываем.
        self.assertEqual(len(signals), 1)
        self.assertTrue(signals[0].action.is_long_open)
        self.assertFalse(hasattr(p, "last_vote_errors"))
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].agent_label, "Crash")
        self.assertIn("boom", errors[0].reason)

    def test_close_signal(self):
        a = FakeAgent("A", {"BTC": Action.FUT_CLOSE_ALL})
        p = EnsemblePlayer(
            label="T", agents=[a], weights={"A": 1.0},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})
        signals, errors = p.vote(market, signal_id_start=1)
        self.assertEqual(len(signals), 1)
        self.assertTrue(signals[0].action.is_close)
        self.assertEqual(errors, [])

    def test_attribution_to_max_weight(self):
        a1 = FakeAgent("Heavy", {"BTC": Action.FUT_LONG_FULL})
        a2 = FakeAgent("Light", {"BTC": Action.FUT_LONG_FULL})
        p = EnsemblePlayer(
            label="T", agents=[a1, a2],
            weights={"Heavy": 0.7, "Light": 0.3},
            voting=WeightedConsensus(), thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})
        signals, errors = p.vote(market, signal_id_start=1)
        # by_agent должен быть Heavy (больший вес)
        self.assertEqual(signals[0].by_agent, "Heavy")
        self.assertEqual(errors, [])

    def test_signal_records_full_vote_weights_and_actions(self):
        a1 = FakeAgent("Heavy", {"BTC": Action.FUT_LONG_FULL})
        a2 = FakeAgent("Light", {"BTC": Action.FUT_LONG_HALF})
        a3 = FakeAgent("Hold", {"BTC": Action.HOLD})
        p = EnsemblePlayer(
            label="T",
            agents=[a1, a2, a3],
            weights={"Heavy": 0.5, "Light": 0.3, "Hold": 0.2},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        market = make_market(prices={"BTC": 100.0})

        signals, errors = p.vote(market, signal_id_start=1)
        signal = signals[0]

        self.assertEqual(signal.vote_weights, {"Heavy": 0.5, "Light": 0.3, "Hold": 0.2})
        self.assertEqual(signal.vote_actions["Heavy"], Action.FUT_LONG_FULL)
        self.assertEqual(signal.vote_actions["Light"], Action.FUT_LONG_HALF)
        self.assertEqual(signal.vote_actions["Hold"], Action.HOLD)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
