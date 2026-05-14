"""Tests for live strategy extensions added after operator dashboard review."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action
from panteon_v2.app.agent_bootstrap import _ensure_paths, known_labels
from panteon_v2.app.bootstrap import PRODUCTION_PROFILES


_ensure_paths()


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

        profile_by_label = {profile.label: profile for profile in PRODUCTION_PROFILES}
        self.assertIn("NeutralEdgeResearch", profile_by_label)
        neutral_edge = profile_by_label["NeutralEdgeResearch"]
        self.assertIn("NeutralLiquiditySweep", neutral_edge.bias)
        self.assertIn("AnchorFlowMomentum", neutral_edge.bias)
        self.assertIn("ResearchValidatorAgent", neutral_edge.bias)


if __name__ == "__main__":
    unittest.main(verbosity=2)
