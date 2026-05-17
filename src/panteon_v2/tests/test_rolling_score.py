"""Tests for v3 rolling decision scoring."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Metrics
from panteon_v2.selection.rolling_score import (
    RollingDecisionScoreConfig,
    RollingDecisionScoreInput,
    score_rolling_decision,
)


def _m(
    pnl: float,
    closed: int,
    *,
    wins: int,
    losses: int,
    max_dd: float = 0.0,
    signals: int = 10,
    sharpe: float = 0.0,
) -> Metrics:
    return Metrics(
        pnl_pct=pnl,
        closed_trades=closed,
        entries=closed,
        signals=signals,
        wins=wins,
        losses=losses,
        max_dd_pct=max_dd,
        sharpe=sharpe,
    )


class TestRollingDecisionScore(unittest.TestCase):
    def test_no_data_returns_configured_no_data_score(self):
        config = RollingDecisionScoreConfig(no_data_score=-0.42)

        result = score_rolling_decision(RollingDecisionScoreInput(), config=config)

        self.assertAlmostEqual(result.score, -0.42)
        self.assertEqual(result.source, "no_data")

    def test_recent_real_edge_beats_stale_virtual_edge(self):
        config = RollingDecisionScoreConfig()
        stale_virtual = RollingDecisionScoreInput(
            virtual_metrics=_m(25.0, 200, wins=120, losses=80, max_dd=30.0),
        )
        recent_real = RollingDecisionScoreInput(
            recent_real_metrics=_m(3.0, 5, wins=4, losses=1, max_dd=1.0),
            long_real_metrics=_m(1.0, 8, wins=5, losses=3, max_dd=2.0),
            virtual_metrics=_m(2.0, 50, wins=26, losses=24, max_dd=5.0),
        )

        stale_result = score_rolling_decision(stale_virtual, config=config)
        recent_result = score_rolling_decision(recent_real, config=config)

        self.assertGreater(recent_result.score, stale_result.score)
        self.assertEqual(recent_result.source, "real+virtual")

    def test_drawdown_and_negative_expectancy_reduce_score(self):
        config = RollingDecisionScoreConfig()
        stable = RollingDecisionScoreInput(
            recent_real_metrics=_m(2.0, 8, wins=5, losses=3, max_dd=1.0),
        )
        unstable = RollingDecisionScoreInput(
            recent_real_metrics=_m(2.0, 8, wins=3, losses=5, max_dd=20.0),
        )

        stable_result = score_rolling_decision(stable, config=config)
        unstable_result = score_rolling_decision(unstable, config=config)

        self.assertGreater(stable_result.score, unstable_result.score)

    def test_negative_recent_real_overrides_positive_virtual(self):
        config = RollingDecisionScoreConfig()
        result = score_rolling_decision(
            RollingDecisionScoreInput(
                recent_real_metrics=_m(-2.0, 6, wins=1, losses=5, max_dd=4.0),
                virtual_metrics=_m(15.0, 100, wins=60, losses=40, max_dd=10.0),
            ),
            config=config,
        )

        self.assertLess(result.score, 0.0)
        self.assertGreater(result.negative_real_penalty, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
