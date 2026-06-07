"""Tests for safe genetics agent wiring into Panteon v2."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from panteon_v2.app.bootstrap import build_production_pipeline
from panteon_v2.app.main_loop import main_loop
from panteon_v2.domain.types import Action, Regime
from panteon_v2.execution import FakeExchange
from panteon_v2.selection import (
    AgentRegistry,
    EnsemblePlayer,
    PlayerProfile,
    SwitchDecision,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.domain.types import Signal
from panteon_v2.shadow import make_market_snapshot
from panteon_v2.shadow.feed import ReplayFeed
from panteon_v2.tests._helpers import FakeAgent


class TestPanteonGeneticsIntegration(unittest.TestCase):
    def test_optional_labels_include_core_and_regime_genetics(self):
        from panteon_v2.app.agent_bootstrap import optional_labels

        labels = set(optional_labels())

        self.assertIn("GeneticsCore", labels)
        self.assertNotIn("GeneticsBullish", labels)
        self.assertNotIn("GeneticsBearish", labels)
        self.assertNotIn("GeneticsNeutral", labels)
        self.assertIn("GeneticsGenomeEnsemble", labels)
        self.assertIn("GeneticsRegimeRouter", labels)

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

    def test_genome_ensemble_optional_registration_uses_conservative_regime_gate(self):
        from panteon_v2.app import agent_bootstrap

        registry = AgentRegistry()
        registered = agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=("GeneticsGenomeEnsemble",),
            skip_on_error=False,
        )

        self.assertEqual(registered, ["GeneticsGenomeEnsemble"])
        adapter = registry.get("GeneticsGenomeEnsemble")
        self.assertEqual(
            adapter.allowed_open_regimes,
            frozenset({Regime.BEARISH, Regime.CRASH}),
        )

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

    def test_shadow_only_genetics_seed_quarantine_can_exempt_runtime_candidate(self):
        from panteon_v2.app.startup import _seed_quarantine_with_shadow_only_genetics

        merged = _seed_quarantine_with_shadow_only_genetics(
            ("FundingArb",),
            [
                "GeneticsCore",
                "GeneticsNeutral",
                "GeneticsRegimeAdaptiveBias",
            ],
            include_genetics=True,
            genetics_shadow_only=True,
            quarantine_exempt_labels=("GeneticsRegimeAdaptiveBias",),
        )

        self.assertIn("FundingArb", merged)
        self.assertIn("GeneticsCore", merged)
        self.assertIn("GeneticsNeutral", merged)
        self.assertNotIn("GeneticsRegimeAdaptiveBias", merged)

    def test_startup_resolves_shadow_only_genetics_quarantine_exempt_labels(self):
        from panteon_v2.app import startup

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(
                    startup,
                    "_load_exchange_settings",
                    return_value={
                        "mexc_v2_genetics_shadow_only_quarantine_exempt_labels": (
                            "GeneticsRegimeAdaptiveBias, OtherRuntimeCandidate"
                        ),
                    },
                ):
            self.assertEqual(
                startup._resolve_genetics_shadow_only_quarantine_exempt_labels("MEXC"),
                ("GeneticsRegimeAdaptiveBias", "OtherRuntimeCandidate"),
            )

        with patch.dict(
            os.environ,
            {
                "PANTEON_V2_GENETICS_SHADOW_ONLY_QUARANTINE_EXEMPT_LABELS": (
                    "EnvCandidate"
                ),
            },
        ), patch.object(
            startup,
            "_load_exchange_settings",
            return_value={
                "mexc_v2_genetics_shadow_only_quarantine_exempt_labels": (
                    "SettingsCandidate"
                ),
            },
        ):
            self.assertEqual(
                startup._resolve_genetics_shadow_only_quarantine_exempt_labels("MEXC"),
                ("EnvCandidate",),
            )

    def test_startup_resolves_genetics_from_settings_when_env_absent(self):
        from panteon_v2.app import startup

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup, "_load_exchange_settings",
                             return_value={"agent_genetics": "on"}):
            self.assertTrue(startup._resolve_include_genetics(None, "MEXC"))

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup, "_load_exchange_settings",
                             return_value={"agent_genetics": "on"}):
            self.assertFalse(startup._resolve_include_genetics(False, "MEXC"))

        with patch.dict(os.environ, {"PANTEON_V2_LOAD_GENETICS": "0"}), \
                patch.object(startup, "_load_exchange_settings",
                             return_value={"agent_genetics": "on"}):
            self.assertFalse(startup._resolve_include_genetics(None, "MEXC"))

    def test_startup_resolves_genetics_shadow_only_from_settings(self):
        from panteon_v2.app import startup

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup, "_load_exchange_settings",
                             return_value={"v2_genetics_shadow_only": "off"}):
            self.assertFalse(startup._resolve_genetics_shadow_only(None, "MEXC"))

        with patch.dict(os.environ, {"PANTEON_V2_GENETICS_SHADOW_ONLY": "1"}), \
                patch.object(startup, "_load_exchange_settings",
                             return_value={"v2_genetics_shadow_only": "off"}):
            self.assertTrue(startup._resolve_genetics_shadow_only(None, "MEXC"))

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
             patch(
                 "panteon_v2.app.startup._resolve_genetics_shadow_only_quarantine_exempt_labels",
                 return_value=(),
             ), \
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

    def test_builds_regime_router_from_selection_manifest_under_results_root(self):
        from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

        class FakeGeneticsAgent:
            def __init__(self, genome):
                self.genome = np.asarray(genome, dtype=np.float32)

            def act(self, prices, volumes, **_kwargs):
                return {"BTC": int(self.genome[0])}

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Results" / "neiro_genetics" / "run"
            root.mkdir(parents=True)
            baseline_path = root / "warm_start_genome.npy"
            candidate_path = root / "archive_rank1.npy"
            np.save(baseline_path, np.asarray([3], dtype=np.float32))
            np.save(candidate_path, np.asarray([4], dtype=np.float32))
            manifest_path = root / "selection_router.json"
            manifest_path.write_text(json.dumps({
                "selected_is_baseline": False,
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(candidate_path),
                    "bullish": str(candidate_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                },
                "validation": {"mean_ret": 1.0, "min_ret": 0.5, "positive_period_pct": 100.0},
                "baseline_validation": {"mean_ret": 1.0, "min_ret": 0.5, "positive_period_pct": 100.0},
            }), encoding="utf-8")

            adapter = build_genetics_regime_router_adapter(
                manifest_path,
                FakeGeneticsAgent,
                results_root=Path(td) / "Results" / "neiro_genetics",
            )

        neutral = make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
            regime_confidence=0.9,
        )
        crash = make_market_snapshot(
            bar=2,
            prices={"BTC": 90.0},
            regime="crash",
            regime_confidence=1.0,
        )

        self.assertEqual(adapter.act(neutral)["BTC"], Action.FUT_SHORT_FULL)
        self.assertEqual(adapter.act(crash)["BTC"], Action.FUT_LONG_FULL)
        self.assertFalse(adapter.promotion_eligible)
        self.assertTrue(adapter.shadow_only)

    def test_offline_promoted_regime_router_stays_shadow_only_without_live_gate(self):
        from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

        class FakeGeneticsAgent:
            def __init__(self, genome):
                self.genome = np.asarray(genome, dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Results" / "neiro_genetics" / "run"
            root.mkdir(parents=True)
            baseline_path = root / "warm_start_genome.npy"
            candidate_path = root / "archive_rank1.npy"
            np.save(baseline_path, np.asarray([3], dtype=np.float32))
            np.save(candidate_path, np.asarray([4], dtype=np.float32))
            manifest_path = root / "selection_router.json"
            manifest_path.write_text(json.dumps({
                "selected_is_baseline": False,
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(candidate_path),
                    "bullish": str(baseline_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                },
                "validation": {"mean_ret": 0.5, "min_ret": 0.2, "positive_period_pct": 100.0},
                "baseline_validation": {"mean_ret": 0.1, "min_ret": 0.0, "positive_period_pct": 66.7},
            }), encoding="utf-8")

            adapter = build_genetics_regime_router_adapter(
                manifest_path,
                FakeGeneticsAgent,
                results_root=Path(td) / "Results" / "neiro_genetics",
            )

        self.assertTrue(adapter.promotion_eligible)
        self.assertFalse(adapter.live_trading_eligible)
        self.assertTrue(adapter.shadow_only)

    def test_regime_router_live_gate_requires_paper_gate_confirmation(self):
        from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

        class FakeGeneticsAgent:
            def __init__(self, genome):
                self.genome = np.asarray(genome, dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Results" / "neiro_genetics" / "run"
            root.mkdir(parents=True)
            baseline_path = root / "warm_start_genome.npy"
            candidate_path = root / "archive_rank1.npy"
            np.save(baseline_path, np.asarray([3], dtype=np.float32))
            np.save(candidate_path, np.asarray([4], dtype=np.float32))
            manifest_path = root / "selection_router.json"
            manifest_path.write_text(json.dumps({
                "selected_is_baseline": False,
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(candidate_path),
                    "bullish": str(baseline_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                },
                "validation": {"mean_ret": 0.5, "min_ret": 0.2, "positive_period_pct": 100.0},
                "baseline_validation": {"mean_ret": 0.1, "min_ret": 0.0, "positive_period_pct": 66.7},
                "paper_trading_eligible": True,
                "live_trading_eligible": True,
            }), encoding="utf-8")

            adapter = build_genetics_regime_router_adapter(
                manifest_path,
                FakeGeneticsAgent,
                results_root=Path(td) / "Results" / "neiro_genetics",
            )

        self.assertTrue(adapter.promotion_eligible)
        self.assertTrue(adapter.paper_trading_eligible)
        self.assertFalse(adapter.live_trading_eligible)
        self.assertTrue(adapter.shadow_only)

    def test_shadow_only_genetics_is_not_exposed_as_real_solo_candidate(self):
        from panteon_v2.app.main_loop import _compose_solo_agent_candidates

        registry = AgentRegistry()
        executable = FakeAgent("ExecutableAgent", {"BTC": Action.FUT_LONG_FULL})
        shadow_only = FakeAgent("GeneticsRegimeRouter", {"BTC": Action.FUT_LONG_FULL})
        shadow_only.shadow_only = True
        shadow_only.paper_trading_eligible = True
        shadow_only.live_trading_eligible = False
        registry.register(executable)
        registry.register(shadow_only)
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="TEST"),
            initial_capital=1000.0,
        )
        pipeline.solo_agent_candidate_limit = 3
        pipeline.virtual_perf.restore({
            "trade_fraction": 0.10,
            "state": {
                "ExecutableAgent|neutral": {
                    "closed_trades": 10,
                    "entries": 10,
                    "signals": 20,
                    "wins": 6,
                    "losses": 4,
                    "pnl_pct": 1.0,
                    "returns": [0.10, -0.05, 0.20, 0.10],
                    "max_dd_pct": 0.2,
                },
                "GeneticsRegimeRouter|neutral": {
                    "closed_trades": 20,
                    "entries": 20,
                    "signals": 40,
                    "wins": 14,
                    "losses": 6,
                    "pnl_pct": 5.0,
                    "returns": [0.20, 0.10, -0.05, 0.30],
                    "max_dd_pct": 0.1,
                },
            },
            "open": {},
            "seen_signal_ids": [],
        })

        solo = _compose_solo_agent_candidates(pipeline, Regime.NEUTRAL, [])

        self.assertEqual([player.label for player in solo], ["Solo_ExecutableAgent"])

    def test_shadow_only_genetics_does_not_consume_solo_candidate_limit(self):
        from panteon_v2.app.main_loop import _compose_solo_agent_candidates

        registry = AgentRegistry()
        executable = FakeAgent("ExecutableAgent", {"BTC": Action.FUT_LONG_FULL})
        shadow_only = FakeAgent("GeneticsRegimeRouter", {"BTC": Action.FUT_LONG_FULL})
        shadow_only.shadow_only = True
        shadow_only.paper_trading_eligible = True
        shadow_only.live_trading_eligible = False
        registry.register(executable)
        registry.register(shadow_only)
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="TEST"),
            initial_capital=1000.0,
        )
        pipeline.solo_agent_candidate_limit = 1
        pipeline.virtual_perf.restore({
            "trade_fraction": 0.10,
            "state": {
                "ExecutableAgent|neutral": {
                    "closed_trades": 10,
                    "entries": 10,
                    "signals": 20,
                    "wins": 6,
                    "losses": 4,
                    "pnl_pct": 1.0,
                    "returns": [0.10, -0.05, 0.20, 0.10],
                    "max_dd_pct": 0.2,
                },
                "GeneticsRegimeRouter|neutral": {
                    "closed_trades": 20,
                    "entries": 20,
                    "signals": 40,
                    "wins": 14,
                    "losses": 6,
                    "pnl_pct": 5.0,
                    "returns": [0.20, 0.10, -0.05, 0.30],
                    "max_dd_pct": 0.1,
                },
            },
            "open": {},
            "seen_signal_ids": [],
        })

        solo = _compose_solo_agent_candidates(pipeline, Regime.NEUTRAL, [])

        self.assertEqual([player.label for player in solo], ["Solo_ExecutableAgent"])

    def test_actionable_fallback_does_not_execute_shadow_only_genetics_agent_signal(self):
        selected_agent = FakeAgent("SelectedAgent")
        shadow_only = FakeAgent("GeneticsRegimeRouter", {"BTC": Action.FUT_LONG_FULL})
        shadow_only.shadow_only = True
        shadow_only.paper_trading_eligible = True
        shadow_only.live_trading_eligible = False
        registry = AgentRegistry()
        registry.register(selected_agent)
        registry.register(shadow_only)
        exchange = FakeExchange(name="TEST")
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=exchange,
            initial_capital=1000.0,
            profiles=[
                PlayerProfile(
                    label="SelectedPlayer",
                    voting=WeightedConsensus(),
                    thresholds=ThresholdProfile(),
                    max_agents=1,
                    min_agents=1,
                ),
            ],
        )
        selected_player = EnsemblePlayer(
            label="SelectedPlayer",
            agents=[selected_agent],
            weights={"SelectedAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )

        class FixedComposer:
            def compose_from_profile_with_fallback(self, profile, regime):
                return selected_player

        class FixedStrategist:
            def update_candidates(self, candidates):
                self.candidates = list(candidates)

            def update_current_actionable_labels(self, labels):
                self.actionable_labels = set(labels)

            def update_shadow_actor_updates(self, updates):
                self.shadow_updates = tuple(updates)

            def update_shadow_position_snapshot(self, *, player_positions, real_positions):
                self.player_positions = dict(player_positions)
                self.real_positions = tuple(real_positions)

            def consider_switch(self, regime, current_bar, regime_confidence=1.0):
                return SwitchDecision(
                    new_leader=selected_player,
                    previous=None,
                    score=1.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="test selected",
                    switched=True,
                )

        class ShadowOnlyGeneticsTournament:
            def run_bar(self, market, *, players, balance_usd):
                from panteon_v2.app.shadow_tournament import ShadowStepSummary

                return ShadowStepSummary(agent_signals=1, agent_filled=1, actors=1)

            def last_actor_updates(self):
                return ()

            def last_player_signals(self):
                return {}

            def last_agent_signals(self):
                return {
                    "GeneticsRegimeRouter": (
                        Signal(
                            id=10_000,
                            bar=1,
                            sym="BTC",
                            action=Action.FUT_LONG_FULL,
                            price=100.0,
                            regime=Regime.NEUTRAL,
                            by_player="",
                            by_agent="GeneticsRegimeRouter",
                        ),
                    )
                }

            def last_player_open_positions(self):
                return {}

        pipeline.composer = FixedComposer()
        pipeline.strategist = FixedStrategist()
        pipeline.shadow_tournament = ShadowOnlyGeneticsTournament()
        pipeline.actionable_fallback_enabled = True
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
        ))

        steps = main_loop(pipeline, feed, max_bars=1)

        self.assertEqual(steps[0].selected_leader, "SelectedPlayer")
        self.assertEqual(steps[0].executed_leader, "SelectedPlayer")
        self.assertFalse(steps[0].fallback_used)
        self.assertEqual(steps[0].n_filled, 0)
        self.assertEqual(exchange.orders_log, [])

    def test_fallback_safety_rejects_any_shadow_only_agent_candidate(self):
        from panteon_v2.app.main_loop import _fallback_candidate_safety_issue

        shadow_only = FakeAgent("ShadowOnlyPaperAgent", {"BTC": Action.FUT_LONG_FULL})
        shadow_only.shadow_only = True
        shadow_only.paper_trading_eligible = True
        shadow_only.live_trading_eligible = False
        registry = AgentRegistry()
        registry.register(shadow_only)
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="TEST"),
            initial_capital=1000.0,
        )
        candidate = EnsemblePlayer(
            label="Solo_ShadowOnlyPaperAgent",
            agents=[shadow_only],
            weights={"ShadowOnlyPaperAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        market = make_market_snapshot(
            bar=1,
            prices={"BTC": 100.0},
            regime="neutral",
        )

        self.assertTrue(_fallback_candidate_safety_issue(pipeline, candidate, market))

    def test_regime_router_manifest_rejects_paths_outside_neiro_genetics(self):
        from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

        class FakeGeneticsAgent:
            def __init__(self, genome):
                self.genome = genome

        with tempfile.TemporaryDirectory() as td:
            results_root = Path(td) / "Results" / "neiro_genetics"
            safe_dir = results_root / "run"
            safe_dir.mkdir(parents=True)
            outside_dir = Path(td) / "Results" / "other"
            outside_dir.mkdir(parents=True)
            baseline_path = safe_dir / "warm_start_genome.npy"
            outside_path = outside_dir / "archive_rank1.npy"
            np.save(baseline_path, np.asarray([3], dtype=np.float32))
            np.save(outside_path, np.asarray([4], dtype=np.float32))
            manifest_path = safe_dir / "selection_router.json"
            manifest_path.write_text(json.dumps({
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(outside_path),
                    "bullish": str(baseline_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                },
            }), encoding="utf-8")

            with self.assertRaises(ValueError):
                build_genetics_regime_router_adapter(
                    manifest_path,
                    FakeGeneticsAgent,
                    results_root=results_root,
                )

    def test_regime_router_manifest_rejects_runtime_genome_size_mismatch(self):
        from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

        class FakeGeneticsAgent:
            def __init__(self, genome):
                self.genome = genome

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "Results" / "neiro_genetics" / "run"
            root.mkdir(parents=True)
            baseline_path = root / "warm_start_genome.npy"
            np.save(baseline_path, np.asarray([3.0, 4.0], dtype=np.float32))
            manifest_path = root / "selection_router.json"
            manifest_path.write_text(json.dumps({
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                },
            }), encoding="utf-8")

            with self.assertRaises(ValueError):
                build_genetics_regime_router_adapter(
                    manifest_path,
                    FakeGeneticsAgent,
                    results_root=Path(td) / "Results" / "neiro_genetics",
                    expected_genome_size=1,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
