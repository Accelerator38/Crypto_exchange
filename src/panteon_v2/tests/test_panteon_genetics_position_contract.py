"""Tests for aligning genetics position limits with Panteon runtime limits."""

from __future__ import annotations

import unittest
from pathlib import Path

from panteon_v2.app.startup import _risk_config_from_trade_fraction
from panteon_v2.selection import AgentRegistry
from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter, V1AgentAdapter


def _read_train_max_pos() -> int:
    settings_path = (
        Path(__file__).resolve().parents[3]
        / "Genetics_DL_Agents"
        / "settings_genetic.txt"
    )
    for raw_line in settings_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        if key == "train_max_pos":
            return int(value)
    raise AssertionError("train_max_pos is not configured")


class DummyV1Agent:
    def __init__(self, max_pos: int) -> None:
        self.MAX_POS = max_pos

    def act(self, prices, volumes, **_kwargs):
        return {sym: 0 for sym in prices}


class TestPanteonGeneticsPositionContract(unittest.TestCase):
    def test_training_max_pos_matches_panteon_runtime_limit(self):
        risk_config = _risk_config_from_trade_fraction(0.10)

        self.assertEqual(_read_train_max_pos(), risk_config.max_open_positions)

    def test_genetics_runtime_limit_is_overridden_from_risk_config(self):
        from panteon_v2.app.startup import _apply_genetics_position_limit

        genetics_v1 = DummyV1Agent(max_pos=10)
        regular_v1 = DummyV1Agent(max_pos=10)
        registry = AgentRegistry()
        registry.register(GeneticsV2AgentAdapter("GeneticsBearish", genetics_v1))
        registry.register(V1AgentAdapter("LiveTrendFollow", regular_v1))

        changed = _apply_genetics_position_limit(registry, max_open_positions=4)

        self.assertEqual(changed, ["GeneticsBearish"])
        self.assertEqual(genetics_v1.MAX_POS, 4)
        self.assertEqual(regular_v1.MAX_POS, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
