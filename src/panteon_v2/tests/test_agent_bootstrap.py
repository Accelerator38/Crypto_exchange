"""Tests for v1 agent bootstrap wrappers."""

from __future__ import annotations

import unittest

from panteon_v2.app.agent_bootstrap import (
    ActionFilterAgent,
    experimental_flash_agent_labels,
)
from panteon_v2.domain.types import Action
from panteon_v2.tests._helpers import FakeAgent, make_market


class TestAgentBootstrap(unittest.TestCase):
    def test_ultima_experimental_flash_wrappers_are_discoverable(self):
        labels = set(experimental_flash_agent_labels())
        self.assertIn("MomentumScalperUltimaQualityLongs", labels)
        self.assertIn("MomentumScalperUltimaMajorShorts", labels)
        self.assertIn("LiveOIBreakoutUltimaSpot", labels)

    def test_ultima_quality_longs_wrapper_keeps_only_target_long_cells(self):
        base = FakeAgent(
            "MomentumScalper",
            {
                "FIL/USDT": Action.FUT_LONG_FULL,
                "BNB/USDT": Action.FUT_LONG_FULL,
                "APT/USDT": Action.FUT_SHORT_FULL,
            },
        )
        wrapper = ActionFilterAgent(
            label="MomentumScalperUltimaQualityLongs",
            base_agent=base,
            allowed_actions=(Action.FUT_LONG_HALF, Action.FUT_LONG_FULL),
            allowed_symbols=("FIL/USDT", "APT/USDT", "ICP/USDT", "LTC/USDT"),
        )

        out = wrapper.act(
            make_market(
                prices={
                    "FIL/USDT": 5.0,
                    "BNB/USDT": 300.0,
                    "APT/USDT": 9.0,
                },
            )
        )

        self.assertEqual(out, {"FIL/USDT": Action.FUT_LONG_FULL})


if __name__ == "__main__":
    unittest.main(verbosity=2)
