"""Tests for safe genetics agent wiring into Panteon v2."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from panteon_v2.app.bootstrap import build_production_pipeline
from panteon_v2.execution import FakeExchange
from panteon_v2.selection import AgentRegistry
from panteon_v2.tests._helpers import FakeAgent


class TestPanteonGeneticsIntegration(unittest.TestCase):
    def test_optional_labels_include_core_and_regime_genetics(self):
        from panteon_v2.app.agent_bootstrap import optional_labels

        labels = set(optional_labels())

        self.assertIn("GeneticsCore", labels)
        self.assertIn("GeneticsBullish", labels)
        self.assertIn("GeneticsBearish", labels)
        self.assertIn("GeneticsNeutral", labels)
        self.assertIn("GeneticsGenomeEnsemble", labels)

    def test_zero_env_does_not_load_optional_genetics(self):
        from panteon_v2.app import agent_bootstrap

        calls = []

        def fake_register(_registry, items, **_kwargs):
            calls.append([label for label, _ in items])
            return [label for label, _ in items]

        with patch.dict(os.environ, {"PANTEON_V2_LOAD_GENETICS": "0"}), \
             patch.object(agent_bootstrap, "_ensure_paths"), \
             patch.object(agent_bootstrap, "_register_from_list", side_effect=fake_register):
            registered = agent_bootstrap.register_all_v1_agents(AgentRegistry())

        self.assertEqual(len(calls), 1)
        self.assertNotIn("GeneticsCore", registered)
        self.assertFalse(any(label.startswith("Genetics") for label in registered))

    def test_optional_registration_can_be_limited_to_genome_ensemble(self):
        from panteon_v2.app import agent_bootstrap

        calls = []

        def fake_register(_registry, items, **_kwargs):
            labels = [label for label, _ in items]
            calls.append(labels)
            return labels

        with patch.object(agent_bootstrap, "_ensure_paths"), \
             patch.object(agent_bootstrap, "_register_from_list", side_effect=fake_register):
            registered = agent_bootstrap.register_all_v1_agents(
                AgentRegistry(),
                include_optional=True,
                optional_agent_labels=("GeneticsGenomeEnsemble",),
            )

        self.assertEqual(calls[-1], ["GeneticsGenomeEnsemble"])
        self.assertIn("GeneticsGenomeEnsemble", registered)
        self.assertNotIn("GeneticsCore", registered)

    def test_shadow_only_genetics_extends_seed_quarantine(self):
        from panteon_v2.app.startup import _seed_quarantine_with_shadow_only_genetics

        merged = _seed_quarantine_with_shadow_only_genetics(
            ("FundingArb",),
            ["LiveTrendFollow", "GeneticsCore", "GeneticsNeutral"],
            include_genetics=True,
            genetics_shadow_only=True,
        )

        self.assertEqual(merged[0], "FundingArb")
        self.assertIn("GeneticsCore", merged)
        self.assertIn("GeneticsNeutral", merged)
        self.assertEqual(len(merged), len(set(merged)))

        real_enabled = _seed_quarantine_with_shadow_only_genetics(
            ("FundingArb",),
            ["GeneticsCore"],
            include_genetics=True,
            genetics_shadow_only=False,
        )
        self.assertEqual(real_enabled, ("FundingArb",))

    def test_startup_loads_genetics_as_shadow_only_when_enabled(self):
        from panteon_v2.app.startup import start_production

        captured = {}

        def register_agents(registry, **kwargs):
            captured["include_optional"] = kwargs.get("include_optional")
            registry.register(FakeAgent("LiveTrendFollow"))
            registry.register(FakeAgent("GeneticsCore"))
            return ["LiveTrendFollow", "GeneticsCore"]

        def build_pipeline_capture(**kwargs):
            captured["seed_quarantine"] = tuple(kwargs["seed_quarantine"])
            return build_production_pipeline(**kwargs)

        with tempfile.TemporaryDirectory() as td, \
             patch("panteon_v2.app.startup.resolve_exchange",
                   return_value=FakeExchange(name="MEXC")), \
             patch("panteon_v2.app.startup.register_all_v1_agents",
                   side_effect=register_agents), \
             patch("panteon_v2.app.startup.build_production_pipeline",
                   side_effect=build_pipeline_capture), \
             patch("panteon_v2.app.startup._resolve_trade_fraction",
                   return_value=0.1):
            rc = start_production(
                exchange="MEXC",
                mode="paper",
                include_genetics=True,
                genetics_shadow_only=True,
                use_v1_bridge=False,
                max_bars=0,
                results_root=td,
                sleep_between_polls_sec=0.0,
            )

        self.assertEqual(rc, 0)
        self.assertIs(captured["include_optional"], True)
        self.assertIn("GeneticsCore", captured["seed_quarantine"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
