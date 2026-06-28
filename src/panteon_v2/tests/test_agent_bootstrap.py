"""Tests for v1 agent bootstrap wrappers."""

from __future__ import annotations

import unittest

from panteon_v2.app.agent_bootstrap import (
    ActionFilterAgent,
    _apply_futures_replay_signal_fixes,
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

    def test_action_filter_agent_preserves_base_signal_diagnostics(self):
        class DiagnosticBase:
            label = "LiveVolCompress"

            def act(self, market):
                self.last_signal_diagnostics = {
                    "FIL/USDT": {"reason": "candidate_long", "bbw": 0.012},
                    "APT/USDT": {"reason": "candidate_short", "bbw": 0.011},
                }
                return {
                    "FIL/USDT": Action.FUT_LONG_FULL,
                    "APT/USDT": Action.FUT_SHORT_FULL,
                }

        wrapper = ActionFilterAgent(
            label="LiveVolCompressLongsOnly",
            base_agent=DiagnosticBase(),
            allowed_actions=(Action.FUT_LONG_FULL,),
        )

        out = wrapper.act(
            make_market(
                prices={
                    "FIL/USDT": 5.0,
                    "APT/USDT": 9.0,
                },
            )
        )

        self.assertEqual(out, {"FIL/USDT": Action.FUT_LONG_FULL})
        self.assertEqual(
            wrapper.last_signal_diagnostics["FIL/USDT"]["reason"],
            "candidate_long",
        )
        self.assertEqual(
            wrapper.last_signal_diagnostics["APT/USDT"]["wrapper_reason"],
            "action_not_allowed",
        )

    def test_futures_replay_signal_fixes_are_opt_in(self):
        class Dummy:
            EMA_S = 120

        instance = Dummy()

        self.assertTrue(
            _apply_futures_replay_signal_fixes("MomentumScalper", instance)
        )
        self.assertTrue(instance.FUTURES_REPLAY_MODE)
        self.assertEqual(instance.EMA_S, 48)
        self.assertEqual(instance.CHECK_INT, 1)

        untouched = Dummy()
        self.assertFalse(_apply_futures_replay_signal_fixes("OtherActor", untouched))
        self.assertEqual(untouched.EMA_S, 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
