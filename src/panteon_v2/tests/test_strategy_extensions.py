"""Tests for live strategy extensions added after operator dashboard review."""

from __future__ import annotations

from datetime import datetime, timezone
import sys
import types
import unittest
from unittest.mock import patch

from panteon_v2.domain.types import Action, MarketSnapshot, Regime
from panteon_v2.app.agent_bootstrap import (
    _ensure_paths,
    experimental_flash_agent_labels,
    legacy_flash_real_agent_labels,
    known_labels,
    promote_legacy_flash_real_agents,
    register_all_v1_agents,
)
from panteon_v2.app.bootstrap import PRODUCTION_PROFILES
from panteon_v2.selection import AgentRegistry
from panteon_v2.selection.strategist import StrategistConfig


_ensure_paths()


def _flash_wrapper_market() -> MarketSnapshot:
    return MarketSnapshot(
        bar=1,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        regime=Regime.NEUTRAL,
        prices={
            "BTC/USDT": 100.0,
            "ETH/USDT": 10.0,
            "DOGE/USDT": 0.1,
            "SOL/USDT": 20.0,
        },
        volumes={},
    )


class TestNewStrategyAgents(unittest.TestCase):
    def test_neutral_liquidity_sweep_fades_failed_breakdown(self):
        from panteon_agents import NeutralLiquiditySweepAgent

        agent = NeutralLiquiditySweepAgent()
        agent.CHECK_INT = 1
        agent.LOOKBACK = 12
        agent.MARKET_LB = 12
        agent.VOL_WINDOW = 8
        agent.MAX_POS = 2

        prices = {"BTC": 100.0, "ETH": 50.0, "ALT": 10.0}
        volumes = {"BTC": 1000.0, "ETH": 800.0, "ALT": 100.0}
        for bar in range(1, 18):
            prices["ALT"] = 10.0 + (0.02 if bar % 2 else -0.02)
            agent.act(dict(prices), dict(volumes), bar_index=bar)

        prices["ALT"] = 9.86
        volumes["ALT"] = 260.0
        agent.act(dict(prices), dict(volumes), bar_index=18)
        prices["ALT"] = 9.99
        action = agent.act(dict(prices), dict(volumes), bar_index=19)["ALT"]

        self.assertIn(action, (Action.FUT_LONG_HALF, Action.FUT_LONG_FULL))

    def test_anchor_flow_momentum_trades_with_confirmed_anchor_breadth(self):
        from panteon_agents import AnchorFlowMomentumAgent

        agent = AnchorFlowMomentumAgent()
        agent.CHECK_INT = 1
        agent.FAST_LB = 4
        agent.SLOW_LB = 10
        agent.MARKET_LB = 10
        agent.VOL_WINDOW = 6
        agent.MAX_POS = 2

        prices = {"BTC": 100.0, "ETH": 50.0, "ALT": 10.0, "BETA": 20.0}
        volumes = {"BTC": 1000.0, "ETH": 800.0, "ALT": 100.0, "BETA": 110.0}
        for bar in range(1, 16):
            prices["BTC"] *= 1.0012
            prices["ETH"] *= 1.0010
            prices["ALT"] *= 1.0024
            prices["BETA"] *= 1.0008
            agent.act(dict(prices), dict(volumes), bar_index=bar)

        volumes["ALT"] = 260.0
        prices["ALT"] *= 1.003
        action = agent.act(dict(prices), dict(volumes), bar_index=16)["ALT"]

        self.assertIn(action, (Action.FUT_LONG_HALF, Action.FUT_LONG_FULL))

    def test_new_agents_and_neutral_edge_player_are_registered_for_production(self):
        labels = set(known_labels())
        self.assertIn("NeutralLiquiditySweep", labels)
        self.assertIn("AnchorFlowMomentum", labels)
        self.assertIn("CandlePatternAgent", labels)
        self.assertIn("CarryFlowAgentV2", labels)
        self.assertIn("Solo_CarryFlowAgentV2", StrategistConfig().hard_policy_deny_labels)

        profile_by_label = {profile.label: profile for profile in PRODUCTION_PROFILES}
        self.assertIn("NeutralEdgeResearch", profile_by_label)
        neutral_edge = profile_by_label["NeutralEdgeResearch"]
        self.assertIn("NeutralLiquiditySweep", neutral_edge.bias)
        self.assertIn("AnchorFlowMomentum", neutral_edge.bias)
        self.assertIn("ResearchValidatorAgent", neutral_edge.bias)
        defensive = profile_by_label["DefensiveResearch"]
        self.assertNotIn("CarryFlowAgentV2", defensive.bias)

    def test_restored_legacy_agents_register_shadow_only_by_default(self):
        registry = AgentRegistry()

        register_all_v1_agents(registry)

        for label in ("CandlePatternAgent", "CarryFlowAgentV2"):
            with self.subTest(label=label):
                agent = registry.get(label)
                self.assertIsNotNone(agent)
                self.assertTrue(getattr(agent, "shadow_only", False))
                self.assertFalse(getattr(agent, "live_trading_eligible", True))

    def test_flash_can_promote_selected_legacy_agents_to_real_execution(self):
        registry = AgentRegistry()

        register_all_v1_agents(registry)
        promoted = promote_legacy_flash_real_agents(
            registry,
            labels=("CarryFlowAgentV2", "CandlePatternAgent"),
        )

        self.assertEqual(
            tuple(promoted),
            ("CarryFlowAgentV2", "CandlePatternAgent"),
        )
        self.assertEqual(
            tuple(legacy_flash_real_agent_labels()),
            ("CarryFlowAgentV2", "CandlePatternAgent"),
        )
        for label in promoted:
            with self.subTest(label=label):
                agent = registry.get(label)
                self.assertIsNotNone(agent)
                self.assertFalse(getattr(agent, "shadow_only", False))
                self.assertTrue(getattr(agent, "live_trading_eligible", False))
                self.assertTrue(getattr(agent, "paper_trading_eligible", False))


class TestExperimentalFlashAgents(unittest.TestCase):
    def test_action_filter_agent_keeps_allowed_actions_and_symbols(self):
        from panteon_v2.app.agent_bootstrap import ActionFilterAgent

        class StaticAgent:
            label = "BaseAgent"

            def act(self, _market):
                return {
                    "BTC/USDT": Action.FUT_SHORT_FULL,
                    "ETH/USDT": Action.SPOT_BUY_FULL,
                    "DOGE/USDT": Action.SPOT_BUY_FULL,
                    "SOL/USDT": Action.SPOT_SELL_ALL,
                }

        short_only = ActionFilterAgent(
            label="BaseShortOnly",
            base_agent=StaticAgent(),
            allowed_actions=(
                Action.FUT_SHORT_HALF,
                Action.FUT_SHORT_FULL,
                Action.FUT_CLOSE_ALL,
            ),
        )
        self.assertEqual(
            short_only.act(_flash_wrapper_market()),
            {"BTC/USDT": Action.FUT_SHORT_FULL},
        )

        spot_quality = ActionFilterAgent(
            label="BaseSpotQuality",
            base_agent=StaticAgent(),
            allowed_actions=(
                Action.SPOT_BUY_HALF,
                Action.SPOT_BUY_FULL,
                Action.SPOT_SELL_ALL,
            ),
            allowed_symbols=("ETH/USDT", "SOL/USDT"),
        )
        self.assertEqual(
            spot_quality.act(_flash_wrapper_market()),
            {
                "ETH/USDT": Action.SPOT_BUY_FULL,
                "SOL/USDT": Action.SPOT_SELL_ALL,
            },
        )

    def test_register_experimental_flash_agents_adds_distinct_wrapper_labels(self):
        from panteon_v2.app.agent_bootstrap import register_experimental_flash_agents

        class StaticAgent:
            def __init__(self, label):
                self.label = label

            def act(self, market):
                return {symbol: Action.HOLD for symbol in market.prices}

        registry = AgentRegistry()
        registry.register(StaticAgent("MomentumScalper"))
        registry.register(StaticAgent("VolBreakoutHunter"))
        registry.register(StaticAgent("LiveCrashHunter"))
        registry.register(StaticAgent("LiveAfterShock"))
        registry.register(StaticAgent("LiveTrendFollow"))
        registry.register(StaticAgent("LiveMeanRev"))

        registered = register_experimental_flash_agents(registry)

        self.assertEqual(
            tuple(registered),
            (
                "MomentumScalperShortOnly",
                "MomentumScalperSpotQuality",
                "MomentumScalperUltimaQualityLongs",
                "MomentumScalperUltimaMajorShorts",
                "VolBreakoutSpotOnly",
                "MomentumScalperShortCrashOnly",
                "MomentumScalperShortBearOnly",
                "MomentumScalperSpotPullbackOnly",
                "CrashHunterStrict",
                "VolBreakoutFundingAware",
                "AfterShockRegimeOnly",
                "LiveTrendFollowBullOnly",
                "LiveMeanRevNeutralOnly",
            ),
        )
        for label in registered:
            self.assertTrue(registry.has(label), label)
            self.assertIsNot(registry.get(label), registry.get("MomentumScalper"))

    def test_experimental_flash_agent_catalog_includes_legacy_specialists(self):
        labels = set(experimental_flash_agent_labels())

        self.assertIn("CrashHunterStrict", labels)
        self.assertIn("VolBreakoutFundingAware", labels)
        self.assertIn("AfterShockRegimeOnly", labels)
        self.assertIn("LiveTrendFollowBullOnly", labels)
        self.assertIn("LiveMeanRevNeutralOnly", labels)


class TestGenomeEnsembleAgent(unittest.TestCase):
    def test_votes_from_legacy_genetics_agents_without_bar_index_kwarg(self):
        calls = []

        class LongVoter:
            def act(self, prices, volumes, month=None, portfolio_value=None):
                calls.append((type(self).__name__, month, portfolio_value))
                return {"BTC": 3}

        class HoldVoter:
            def act(self, prices, volumes, month=None, portfolio_value=None):
                calls.append((type(self).__name__, month, portfolio_value))
                return {"BTC": 0}

        fake_module = types.ModuleType("crypto_genetics")
        fake_module.GeneticsBullishAgent = LongVoter
        fake_module.GeneticsBearishAgent = LongVoter
        fake_module.GeneticsNeutralAgent = HoldVoter

        with patch.dict(sys.modules, {"crypto_genetics": fake_module}):
            from agents_v2 import GenomeEnsembleAgent

            agent = GenomeEnsembleAgent()
            result = agent.act(
                {"BTC": 100.0},
                {"BTC": 1000.0},
                month=5,
                portfolio_value=12345.0,
                bar_index=77,
            )

        self.assertEqual(result["BTC"], 3)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call[1:] == (5, 12345.0) for call in calls))

    def test_forwards_position_sync_to_loaded_regime_genetics_agents(self):
        sync_calls = []

        class SyncableVoter:
            def act(self, prices, volumes, month=None, portfolio_value=None):
                return {"BTC": 0}

            def update_from_exchange(self, symbol, spot_qty, spot_entry, fut_qty, fut_entry):
                sync_calls.append(
                    (type(self).__name__, symbol, spot_qty, spot_entry, fut_qty, fut_entry)
                )

        fake_module = types.ModuleType("crypto_genetics")
        fake_module.GeneticsBullishAgent = SyncableVoter
        fake_module.GeneticsBearishAgent = SyncableVoter
        fake_module.GeneticsNeutralAgent = SyncableVoter

        with patch.dict(sys.modules, {"crypto_genetics": fake_module}):
            from agents_v2 import GenomeEnsembleAgent

            agent = GenomeEnsembleAgent()
            agent.act({"BTC": 100.0}, {"BTC": 1000.0}, month=5, portfolio_value=1000.0)
            agent.update_from_exchange("BTC", 0.0, 0.0, 1.5, 101.0)

        self.assertEqual(len(sync_calls), 3)
        self.assertTrue(all(call[1:] == ("BTC", 0.0, 0.0, 1.5, 101.0) for call in sync_calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)
