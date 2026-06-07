"""Тесты PlayerComposer + PlayerProfile."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action, Metrics, Regime
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import (
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_GENETICS_RESEARCH,
    PROFILE_TREND_RESEARCH,
    AgentRegistry,
    AgentSelector,
    ScoredAgent,
    PlayerComposer,
    PlayerProfile,
    WeightedConsensus,
    ThresholdProfile,
)
from panteon_v2.tests._helpers import FakeAgent
from panteon_v2.tests.test_selector import _add_perf


class _FixedScoreSelector:
    def __init__(self, normal=(), fallback=()):
        self._normal = tuple(normal)
        self._fallback = tuple(fallback)

    def select(self, regime, k, *args, **kwargs):
        return list(self._normal[:k])

    def select_with_fallback(self, regime, k, *args, **kwargs):
        return list(self._fallback[:k])


class TestPlayerProfile(unittest.TestCase):
    def test_default_valid(self):
        # Default profile — valid
        self.assertEqual(PROFILE_DEFAULT_ENSEMBLE.label, "DefaultEnsemble")

    def test_invalid_max_lt_min(self):
        with self.assertRaises(ValueError):
            PlayerProfile(
                label="X",
                voting=WeightedConsensus(),
                thresholds=ThresholdProfile(),
                max_agents=2, min_agents=5,
            )


class TestPlayerComposer(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock",
                      "LiveCrashHunter", "LiveOIBreakout",
                      "FundingArb"]:
            self.registry.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))
        self.perf = PerformanceMemory(trade_fraction=1.0)
        # Все 5 в плюсе
        for i, label in enumerate(self.registry.all_labels()):
            _add_perf(self.perf, label, Regime.BULLISH, 5, 0.5 + i * 0.1,
                      start_id=1000 + i*100)
        self.qm = QuarantineManager(seed={"FundingArb"})  # FundingArb в карантине
        self.selector = AgentSelector(self.registry, self.perf, self.qm)
        self.composer = PlayerComposer(self.selector)

    def test_compose_basic(self):
        player = self.composer.compose_from_profile(
            PROFILE_DEFAULT_ENSEMBLE, Regime.BULLISH,
        )
        self.assertIsNotNone(player)
        # FundingArb в карантине — не должен попасть
        self.assertNotIn("FundingArb", player.agent_labels)

    def test_compose_respects_min_agents(self):
        # min_agents=5, but only 4 eligible (FundingArb quarantined)
        profile = PlayerProfile(
            label="Strict5",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=5, min_agents=5,
        )
        player = self.composer.compose_from_profile(profile, Regime.BULLISH)
        self.assertIsNone(player, "Should fail min_agents=5 with 4 eligible")

    def test_compose_uses_max_agents_cap(self):
        profile = PlayerProfile(
            label="MaxThree",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=3, min_agents=2,
        )
        player = self.composer.compose_from_profile(profile, Regime.BULLISH)
        self.assertIsNotNone(player)
        self.assertLessEqual(len(player.agent_labels), 3)

    def test_shadow_only_agent_does_not_consume_profile_slot(self):
        registry = AgentRegistry()
        executable = FakeAgent("ExecutableAgent", {"BTC": Action.FUT_LONG_FULL})
        shadow_only = FakeAgent("ShadowOnlyAgent", {"BTC": Action.FUT_LONG_FULL})
        shadow_only.shadow_only = True
        shadow_only.paper_trading_eligible = True
        shadow_only.live_trading_eligible = False
        registry.register(executable)
        registry.register(shadow_only)
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "ExecutableAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "ShadowOnlyAgent", Regime.BULLISH, 10, 5.0, start_id=1000)
        composer = PlayerComposer(
            AgentSelector(registry, perf, QuarantineManager(seed=set()))
        )
        profile = PlayerProfile(
            label="OneSlot",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=1,
            min_agents=1,
        )

        player = composer.compose_from_profile_with_fallback(profile, Regime.BULLISH)

        self.assertIsNotNone(player)
        self.assertEqual(player.agent_labels, ["ExecutableAgent"])

    def test_compose_weights_normalized(self):
        player = self.composer.compose_from_profile(
            PROFILE_DEFAULT_ENSEMBLE, Regime.BULLISH,
        )
        self.assertIsNotNone(player)
        total = sum(player.weights.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_bias_increases_weight(self):
        # Profile с явным bias на LiveTrendFollow
        profile = PlayerProfile(
            label="BiasedTrend",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=4, min_agents=2,
            bias={"LiveTrendFollow": 5.0},  # огромный bias
        )
        player = self.composer.compose_from_profile(profile, Regime.BULLISH)
        if player is None:
            self.skipTest("Composer не дал игрока")
        # LiveTrendFollow должен быть с самым большим весом
        weights_sorted = sorted(player.weights.items(), key=lambda kv: -kv[1])
        self.assertEqual(weights_sorted[0][0], "LiveTrendFollow")

    def test_positive_score_weights_are_convex_to_promote_good_agents_faster(self):
        good = ScoredAgent(FakeAgent("Good"), score=4.0, metrics=Metrics.empty())
        okay = ScoredAgent(FakeAgent("Okay"), score=1.0, metrics=Metrics.empty())
        composer = PlayerComposer(_FixedScoreSelector(normal=(good, okay)))
        profile = PlayerProfile(
            label="DynamicWeights",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=2,
            min_agents=2,
        )

        player = composer.compose_from_profile(profile, Regime.NEUTRAL)

        self.assertIsNotNone(player)
        self.assertGreater(
            player.weights["Good"] / player.weights["Okay"],
            good.score / okay.score,
        )

    def test_weight_power_is_tunable(self):
        # Phase 5: при power=1.0 веса линейны (отношение = отношению скоров);
        # при power>1 — сильнее концентрируются на топ-агенте.
        good = ScoredAgent(FakeAgent("Good"), score=4.0, metrics=Metrics.empty())
        okay = ScoredAgent(FakeAgent("Okay"), score=1.0, metrics=Metrics.empty())
        profile = PlayerProfile(
            label="TunableWeights",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=2,
            min_agents=2,
        )
        linear = PlayerComposer(
            _FixedScoreSelector(normal=(good, okay)), weight_power=1.0
        ).compose_from_profile(profile, Regime.NEUTRAL)
        concentrated = PlayerComposer(
            _FixedScoreSelector(normal=(good, okay)), weight_power=3.0
        ).compose_from_profile(profile, Regime.NEUTRAL)

        # Линейный режим: отношение весов ≈ отношению скоров (4:1).
        self.assertAlmostEqual(
            linear.weights["Good"] / linear.weights["Okay"], 4.0, places=5
        )
        # Более высокая степень → большая концентрация на топ-агенте.
        self.assertGreater(
            concentrated.weights["Good"] / concentrated.weights["Okay"],
            linear.weights["Good"] / linear.weights["Okay"],
        )

    def test_invalid_weight_power_rejected(self):
        with self.assertRaises(ValueError):
            PlayerComposer(_FixedScoreSelector(), weight_power=0.0)

    def test_voting_directional_propagates_to_ensemble(self):
        # Phase 2/A1: composer(voting_directional=True) строит ансамбли с
        # направленным WeightedConsensus.
        good = ScoredAgent(FakeAgent("Good"), score=4.0, metrics=Metrics.empty())
        okay = ScoredAgent(FakeAgent("Okay"), score=1.0, metrics=Metrics.empty())
        profile = PlayerProfile(
            label="DirVote",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=2,
            min_agents=2,
        )
        default_player = PlayerComposer(
            _FixedScoreSelector(normal=(good, okay))
        ).compose_from_profile(profile, Regime.NEUTRAL)
        directional_player = PlayerComposer(
            _FixedScoreSelector(normal=(good, okay)), voting_directional=True
        ).compose_from_profile(profile, Regime.NEUTRAL)

        self.assertFalse(default_player.voting.directional)
        self.assertTrue(directional_player.voting.directional)

    def test_negative_fallback_weights_decay_by_score_severity(self):
        mild = ScoredAgent(FakeAgent("MildLoser"), score=-0.10, metrics=Metrics.empty())
        bad = ScoredAgent(FakeAgent("BadLoser"), score=-2.00, metrics=Metrics.empty())
        composer = PlayerComposer(_FixedScoreSelector(fallback=(mild, bad)))
        profile = PlayerProfile(
            label="DynamicFallbackWeights",
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
            max_agents=2,
            min_agents=2,
        )

        player = composer.compose_from_profile_with_fallback(profile, Regime.NEUTRAL)

        self.assertIsNotNone(player)
        self.assertGreater(player.weights["MildLoser"], player.weights["BadLoser"] * 2.0)

    def test_compose_with_fallback(self):
        # Если нет agents с positive scores, нормальный compose даст None,
        # fallback должен попробовать понизить порог.
        # Создаём ситуацию: все в минусе
        registry2 = AgentRegistry()
        for label in ["A", "B", "C"]:
            registry2.register(FakeAgent(label))
        perf2 = PerformanceMemory(trade_fraction=1.0)
        for label in ["A", "B", "C"]:
            _add_perf(perf2, label, Regime.NEUTRAL, 5, -1.0)
        qm2 = QuarantineManager(seed=set())
        sel2 = AgentSelector(registry2, perf2, qm2)
        comp2 = PlayerComposer(sel2)

        normal = comp2.compose_from_profile(PROFILE_DEFAULT_ENSEMBLE, Regime.NEUTRAL)
        # Ожидание: либо None, либо очень мало агентов
        if normal is None:
            fb = comp2.compose_from_profile_with_fallback(
                PROFILE_DEFAULT_ENSEMBLE, Regime.NEUTRAL,
            )
            self.assertIsNotNone(fb)


    def test_genetics_profile_only_composes_genetics_agents(self):
        registry = AgentRegistry()
        for label in [
            "LiveTrendFollow",
            "GeneticsCore",
            "GeneticsBullish",
            "GeneticsGenomeEnsemble",
        ]:
            registry.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "LiveTrendFollow", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "GeneticsCore", Regime.BULLISH, 10, 0.6, start_id=500)
        _add_perf(perf, "GeneticsBullish", Regime.BULLISH, 10, 0.5, start_id=900)
        _add_perf(perf, "GeneticsGenomeEnsemble", Regime.BULLISH, 10, 0.4, start_id=1300)
        composer = PlayerComposer(
            AgentSelector(registry, perf, QuarantineManager(seed=set()))
        )

        player = composer.compose_from_profile_with_fallback(
            PROFILE_GENETICS_RESEARCH,
            Regime.BEARISH,
        )

        self.assertIsNotNone(player)
        self.assertNotIn("LiveTrendFollow", player.agent_labels)
        self.assertTrue(
            all(label.startswith("Genetics") for label in player.agent_labels)
        )
        self.assertIn("GeneticsCore", player.agent_labels)
        self.assertNotIn("GeneticsGenomeEnsemble", player.agent_labels)
        self.assertNotIn("GeneticsBullish", player.agent_labels)
        self.assertLessEqual(
            len(player.agent_labels),
            PROFILE_GENETICS_RESEARCH.max_agents,
        )

    def test_genetics_profile_can_score_quarantined_core_from_player_context(self):
        registry = AgentRegistry()
        for label in ["GeneticsCore", "GeneticsGenomeEnsemble"]:
            registry.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "GeneticsCore", Regime.BEARISH, 20, 3.0, start_id=1)
        _add_perf(perf, "GeneticsGenomeEnsemble", Regime.BEARISH, 20, 0.2, start_id=500)
        composer = PlayerComposer(
            AgentSelector(
                registry,
                perf,
                QuarantineManager(seed={"GeneticsCore"}),
            )
        )

        player = composer.compose_from_profile_with_fallback(
            PROFILE_GENETICS_RESEARCH,
            Regime.BEARISH,
        )

        self.assertIsNotNone(player)
        self.assertIn("GeneticsCore", player.agent_labels)

    def test_genetics_profile_seeds_core_before_performance_history_exists(self):
        registry = AgentRegistry()
        registry.register(FakeAgent("GeneticsCore", {"BTC": Action.FUT_LONG_FULL}))
        registry.register(FakeAgent("LiveTrendFollow", {"BTC": Action.FUT_LONG_FULL}))
        composer = PlayerComposer(
            AgentSelector(
                registry,
                PerformanceMemory(trade_fraction=1.0),
                QuarantineManager(seed=set()),
            )
        )

        player = composer.compose_from_profile_with_fallback(
            PROFILE_GENETICS_RESEARCH,
            Regime.BEARISH,
        )

        self.assertIsNotNone(player)
        self.assertEqual(player.agent_labels, ["GeneticsCore"])

    def test_genetics_profile_is_limited_to_bearish_and_crash_regimes(self):
        registry = AgentRegistry()
        registry.register(FakeAgent("GeneticsCore", {"BTC": Action.FUT_LONG_FULL}))
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "GeneticsCore", Regime.NEUTRAL, 30, 1.0, start_id=1)
        _add_perf(perf, "GeneticsCore", Regime.BEARISH, 30, 1.0, start_id=1000)
        composer = PlayerComposer(
            AgentSelector(registry, perf, QuarantineManager(seed=set()))
        )

        neutral = composer.compose_from_profile_with_fallback(
            PROFILE_GENETICS_RESEARCH,
            Regime.NEUTRAL,
        )
        bearish = composer.compose_from_profile_with_fallback(
            PROFILE_GENETICS_RESEARCH,
            Regime.BEARISH,
        )

        self.assertIsNone(neutral)
        self.assertIsNotNone(bearish)


class TestComposerCarantineGuarantee(unittest.TestCase):
    """Property: композер НИКОГДА не возвращает игрока с карантинным агентом."""

    def test_quarantine_propagates_through_composer(self):
        registry = AgentRegistry()
        for label in ["A", "B", "C", "D", "BadOne"]:
            registry.register(FakeAgent(label))
        perf = PerformanceMemory(trade_fraction=1.0)
        # BadOne — самый прибыльный, но в карантине
        _add_perf(perf, "BadOne", Regime.BULLISH, 10, 10.0, start_id=1)
        for i, label in enumerate(["A", "B", "C", "D"]):
            _add_perf(perf, label, Regime.BULLISH, 5, 0.5,
                      start_id=1000 + i * 100)
        qm = QuarantineManager(seed={"BadOne"})
        composer = PlayerComposer(AgentSelector(registry, perf, qm))

        for _ in range(5):
            player = composer.compose_from_profile(
                PROFILE_DEFAULT_ENSEMBLE, Regime.BULLISH,
            )
            if player is not None:
                self.assertNotIn("BadOne", player.agent_labels,
                                 "Quarantined must NEVER appear in player.agents")


if __name__ == "__main__":
    unittest.main(verbosity=2)
