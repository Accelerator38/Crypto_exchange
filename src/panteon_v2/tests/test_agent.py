"""Тесты Agent Protocol и AgentRegistry."""

from __future__ import annotations

import unittest

from panteon_v2.domain.types import Action
from panteon_v2.selection.agent import Agent, AgentRegistry
from panteon_v2.tests._helpers import FakeAgent, make_market


class TestAgentProtocol(unittest.TestCase):
    def test_fake_agent_satisfies_protocol(self):
        a = FakeAgent("X", {"BTC": Action.HOLD})
        self.assertIsInstance(a, Agent)

    def test_object_without_label_not_agent(self):
        class Bad:
            def act(self, market):
                return {}
        self.assertNotIsInstance(Bad(), Agent)

    def test_object_without_act_not_agent(self):
        class Bad:
            label = "X"
        self.assertNotIsInstance(Bad(), Agent)


class TestAgentRegistry(unittest.TestCase):
    def test_register_and_get(self):
        reg = AgentRegistry()
        a = FakeAgent("X")
        reg.register(a)
        self.assertEqual(reg.get("X"), a)
        self.assertTrue(reg.has("X"))
        self.assertEqual(len(reg), 1)

    def test_register_duplicate_rejected(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("X"))
        with self.assertRaises(ValueError):
            reg.register(FakeAgent("X"))

    def test_register_replace(self):
        reg = AgentRegistry()
        a1 = FakeAgent("X", {"BTC": Action.HOLD})
        a2 = FakeAgent("X", {"ETH": Action.SPOT_BUY_FULL})
        reg.register(a1)
        reg.register(a2, replace=True)
        self.assertEqual(reg.get("X"), a2)

    def test_register_non_agent_raises(self):
        reg = AgentRegistry()
        with self.assertRaises(TypeError):
            reg.register("not an agent")  # type: ignore

    def test_register_empty_label(self):
        reg = AgentRegistry()
        with self.assertRaises(ValueError):
            reg.register(FakeAgent(""))

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("X"))
        self.assertTrue(reg.unregister("X"))
        self.assertFalse(reg.has("X"))
        self.assertFalse(reg.unregister("X"))  # already gone

    def test_all_labels_sorted(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("Z"))
        reg.register(FakeAgent("A"))
        reg.register(FakeAgent("M"))
        self.assertEqual(reg.all_labels(), ["A", "M", "Z"])

    def test_contains(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("X"))
        self.assertIn("X", reg)
        self.assertNotIn("Y", reg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
