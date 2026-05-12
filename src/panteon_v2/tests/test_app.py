"""Тесты Phase 9 — production wiring."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from panteon_v2.app import (
    OutputWriter,
    PRODUCTION_PROFILES,
    ProductionPipeline,
    build_dryrun_pipeline,
    build_production_pipeline,
    load_v2_snapshot,
    main_loop,
    migrate_from_v1_memory_file,
    migrate_v1_regime_memory,
    save_v2_snapshot,
)
from panteon_v2.domain.types import Action, Regime
from panteon_v2.execution import FakeExchange
from panteon_v2.memory import PerformanceMemory
from panteon_v2.selection import AgentRegistry
from panteon_v2.shadow.adapters import make_market_snapshot
from panteon_v2.shadow.feed import ReplayFeed
from panteon_v2.tests._helpers import FakeAgent


# ════════════════════════════════════════════════════════════════════
# Bootstrap
# ════════════════════════════════════════════════════════════════════


class TestBootstrap(unittest.TestCase):
    def test_empty_registry_rejected(self):
        with self.assertRaises(ValueError):
            build_dryrun_pipeline(registry=AgentRegistry())

    def test_negative_capital_rejected(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("X"))
        with self.assertRaises(ValueError):
            build_dryrun_pipeline(registry=reg, initial_capital=-1)

    def test_dryrun_pipeline_has_all_components(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("LiveTrendFollow"))
        reg.register(FakeAgent("LiveAfterShock"))
        reg.register(FakeAgent("LiveMeanRev"))
        p = build_dryrun_pipeline(registry=reg, initial_capital=100.0)
        self.assertIsInstance(p, ProductionPipeline)
        for attr in ("registry", "perf", "qm", "selector", "composer",
                     "strategist", "executor", "event_log", "ledger",
                     "renderer"):
            self.assertIsNotNone(getattr(p, attr))
        self.assertEqual(p.exchange_name, "DRY-RUN")
        self.assertEqual(p.initial_capital, 100.0)

    def test_production_pipeline_uses_real_exchange(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("A"))
        ex = FakeExchange(name="MEXC-CUSTOM")
        p = build_production_pipeline(
            registry=reg, exchange=ex, initial_capital=500.0,
        )
        self.assertEqual(p.exchange_name, "MEXC-CUSTOM")

    def test_production_profiles_present(self):
        self.assertGreaterEqual(len(PRODUCTION_PROFILES), 5)


# ════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════


class TestMainLoop(unittest.TestCase):
    def _setup(self):
        reg = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock", "LiveMeanRev"]:
            reg.register(FakeAgent(label, {"BTC": Action.FUT_LONG_FULL}))
        feed = ReplayFeed()
        for bar in range(1, 4):
            feed.append(make_market_snapshot(
                bar=bar, prices={"BTC": 100.0 + bar},
                regime="bullish",
            ))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=1000.0)
        return pipeline, feed

    def test_processes_all_bars(self):
        pipeline, feed = self._setup()
        steps = main_loop(pipeline, feed)
        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].bar, 1)
        self.assertEqual(steps[-1].bar, 3)

    def test_max_bars(self):
        pipeline, feed = self._setup()
        steps = main_loop(pipeline, feed, max_bars=2)
        self.assertEqual(len(steps), 2)

    def test_on_step_callback(self):
        pipeline, feed = self._setup()
        seen = []
        main_loop(pipeline, feed, on_step=lambda s: seen.append(s.bar))
        self.assertEqual(seen, [1, 2, 3])

    def test_on_idle_callback_and_limit(self):
        pipeline, feed = self._setup()
        seen = []
        steps = main_loop(
            pipeline,
            ReplayFeed(),
            sleep_between_polls_sec=0.0,
            max_idle_polls=2,
            on_idle=lambda n: seen.append(n),
        )
        self.assertEqual(steps, [])
        self.assertEqual(seen, [1, 2])

    def test_ledger_populated_after_loop(self):
        pipeline, feed = self._setup()
        main_loop(pipeline, feed)
        # После loop ledger содержит хоть какие-то events
        # (не обязательно closed — зависит от leader/voting)
        self.assertGreaterEqual(len(pipeline.event_log), 3)

    def test_quarantined_blocks_pipeline(self):
        reg = AgentRegistry()
        reg.register(FakeAgent("BadAgent",
                              {"BTC": Action.FUT_LONG_FULL}))
        feed = ReplayFeed()
        feed.append(make_market_snapshot(
            bar=1, prices={"BTC": 100.0}, regime="bullish",
        ))
        pipeline = build_dryrun_pipeline(
            registry=reg, initial_capital=100.0,
            seed_quarantine={"BadAgent"},
        )
        steps = main_loop(pipeline, feed)
        # Карантинный → не выберется в leader
        self.assertEqual(len(steps), 1)
        self.assertIsNone(steps[0].leader)


class TestOutputWriter(unittest.TestCase):
    def test_writer_creates_operator_files_before_first_bar(self):
        reg = AgentRegistry()
        for label in ["LiveTrendFollow", "LiveAfterShock", "LiveMeanRev"]:
            reg.register(FakeAgent(label))
        pipeline = build_dryrun_pipeline(registry=reg, initial_capital=100.0)

        with tempfile.TemporaryDirectory() as td:
            writer = OutputWriter.for_session(pipeline, results_root=td)
            out = writer.output_dir
            for name in (
                "trading.log",
                "status.json",
                "leaderboard_agents.json",
                "leaderboard_players.json",
                "dashboard.txt",
                "dashboard.html",
            ):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
            with open(os.path.join(out, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)
            self.assertEqual(status["version"], "v2")
            self.assertEqual(status["run_state"], "starting")
            self.assertEqual(status["bar_count"], 0)
            writer.close()


# ════════════════════════════════════════════════════════════════════
# Migration
# ════════════════════════════════════════════════════════════════════


class TestMigration(unittest.TestCase):
    def test_migrate_v1_dict(self):
        v1 = {
            "bullish": {
                "FundingArb": {"samples": 10, "wins": 3, "losses": 7,
                              "pnl_pct": -1.5},
                "LiveAfterShock": {"samples": 8, "wins": 5, "losses": 3,
                                   "pnl_pct": 2.0},
            },
            "neutral": {
                "LiveMeanRev": {"samples": 12, "wins": 6, "losses": 6,
                                "pnl_pct": 0.0},
            },
        }
        perf = PerformanceMemory()
        report = migrate_v1_regime_memory(v1, perf)
        self.assertEqual(report.n_labels, 3)
        self.assertEqual(report.n_regime_pairs, 3)
        # Check perf populated
        m = perf.get("FundingArb", regime=Regime.BULLISH)
        self.assertEqual(m.closed_trades, 10)
        self.assertAlmostEqual(m.pnl_pct, -1.5)

    def test_migrate_strips_v_prefix(self):
        v1 = {"bullish": {"V_FundingArb": {"samples": 5, "wins": 2, "losses": 3,
                                            "pnl_pct": -0.5}}}
        perf = PerformanceMemory()
        report = migrate_v1_regime_memory(v1, perf)
        self.assertEqual(report.n_labels, 1)
        self.assertTrue(perf.get("FundingArb").has_data)
        self.assertFalse(perf.get("V_FundingArb").has_data)

    def test_invalid_regime_warning(self):
        # Junk regime string должен попасть в neutral (благодаря Regime.from_string)
        v1 = {"junk_regime": {"X": {"samples": 5, "wins": 2, "losses": 3,
                                     "pnl_pct": 1.0}}}
        perf = PerformanceMemory()
        report = migrate_v1_regime_memory(v1, perf)
        # Junk → NEUTRAL
        self.assertTrue(perf.get("X", regime=Regime.NEUTRAL).has_data)

    def test_migrate_from_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "v1_memory.json")
            with open(path, "w") as f:
                json.dump({"_regime_memory": {
                    "bullish": {"X": {"samples": 5, "wins": 3, "losses": 2,
                                       "pnl_pct": 1.0}},
                }}, f)
            perf = PerformanceMemory()
            report = migrate_from_v1_memory_file(path, perf)
            self.assertEqual(report.n_labels, 1)

    def test_missing_file(self):
        perf = PerformanceMemory()
        report = migrate_from_v1_memory_file("/nonexistent/path.json", perf)
        self.assertEqual(report.n_labels, 0)
        self.assertTrue(any("not found" in w for w in report.warnings))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "snap.json")
            perf1 = PerformanceMemory()
            migrate_v1_regime_memory({"bullish": {
                "Z": {"samples": 7, "wins": 4, "losses": 3, "pnl_pct": 0.5}
            }}, perf1)
            save_v2_snapshot(perf1, path)
            perf2 = PerformanceMemory()
            ok = load_v2_snapshot(perf2, path)
            self.assertTrue(ok)
            self.assertAlmostEqual(
                perf1.get("Z", regime=Regime.BULLISH).pnl_pct,
                perf2.get("Z", regime=Regime.BULLISH).pnl_pct,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
