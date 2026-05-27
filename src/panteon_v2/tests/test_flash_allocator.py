"""Tests for Panteon_Flash per-symbol allocator."""

from __future__ import annotations

import unittest
from dataclasses import replace

import pytest

from panteon_v2.domain.types import Action, Regime, Signal, TechnicalIndicators, Trade
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import AgentRegistry, EnsemblePlayer
from panteon_v2.selection.flash_allocator import (
    FLASH_EXPERIMENTAL_FLAGS,
    FLASH_PRESET_AGGRESSIVE,
    FLASH_PRESET_DEFAULT,
    FLASH_PRESET_SAFE,
    FLASH_PRESET_SAFE_PINNED_VALUES,
    FlashAllocator,
    FlashAllocatorConfig,
)
from panteon_v2.selection.voting import ThresholdProfile, WeightedConsensus
from panteon_v2.tests._helpers import FakeAgent, make_market
from panteon_v2.tests.test_selector import _add_perf


def _record_one_trade(
    perf: PerformanceMemory,
    label: str,
    *,
    entry: float,
    exit: float,
    funding_open: float = 0.0,
    funding_close: float = 0.0,
    start_id: int = 1,
) -> None:
    open_signal = Signal(
        id=start_id,
        bar=start_id,
        sym=f"{label[:3].upper()}USDT",
        action=Action.FUT_LONG_FULL,
        price=entry,
        regime=Regime.BULLISH,
        by_player=label,
        by_agent=label,
    )
    close_signal = replace(
        open_signal,
        id=start_id + 1,
        bar=start_id + 1,
        action=Action.FUT_CLOSE_ALL,
        price=exit,
    )
    perf.update_from_trade(
        Trade(
            signal_id=open_signal.id,
            bar=open_signal.bar,
            sym=open_signal.sym,
            side="long",
            qty=1.0,
            fill_price=entry,
            fee=0.0,
            funding=funding_open,
        ),
        open_signal,
    )
    perf.update_from_trade(
        Trade(
            signal_id=close_signal.id,
            bar=close_signal.bar,
            sym=close_signal.sym,
            side="long",
            qty=1.0,
            fill_price=exit,
            fee=0.0,
            funding=funding_close,
        ),
        close_signal,
    )


class TestFlashAllocator(unittest.TestCase):
    def test_technical_overlay_emits_candidate_audit_fields(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("TechAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "TechAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            technicals_by_symbol={
                "BTC": TechnicalIndicators(
                    rsi_14=55.0,
                    macd_line_pct=0.20,
                    macd_signal_pct=0.10,
                    macd_histogram_pct=0.10,
                    atr_14_pct=2.0,
                )
            },
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(technical_overlay_enabled=True),
        )

        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]
        row = next(item for item in decision.candidates if item.label == "TechAgent")

        self.assertAlmostEqual(row.technical_rsi_14, 55.0)
        self.assertAlmostEqual(row.technical_macd_histogram_pct, 0.10)
        self.assertAlmostEqual(row.technical_atr_14_pct, 2.0)
        self.assertEqual(row.technical_alignment, "long_aligned")

    def test_technical_hard_gate_blocks_long_when_rsi_or_macd_misaligned(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("TechLong", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "TechLong", Regime.BULLISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            technicals_by_symbol={
                "BTC": TechnicalIndicators(
                    rsi_14=82.0,
                    macd_histogram_pct=-0.05,
                    atr_14_pct=2.0,
                )
            },
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                technical_overlay_enabled=True,
                technical_hard_gate_enabled=True,
            ),
        )

        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        row = next(item for item in decision.candidates if item.label == "TechLong")
        self.assertEqual(row.reason, "technical_long_misaligned")

    def test_technical_atr_risk_sizing_reduces_size_when_atr_is_high(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("AtrAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "AtrAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            technicals_by_symbol={
                "BTC": TechnicalIndicators(
                    rsi_14=55.0,
                    macd_histogram_pct=0.10,
                    atr_14_pct=4.0,
                )
            },
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                technical_overlay_enabled=True,
                technical_atr_risk_sizing_enabled=True,
                technical_atr_target_pct=2.0,
                technical_atr_max_mult=1.5,
            ),
        )

        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "AtrAgent")
        self.assertIsNotNone(decision.signal)
        assert decision.signal.risk_mult == pytest.approx(0.5)

    def test_selects_different_best_actor_per_symbol(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        btc_agent = FakeAgent("BtcAgent", {"BTC": Action.FUT_LONG_FULL})
        eth_agent = FakeAgent("EthAgent", {"ETH": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BtcAgent", Regime.BULLISH, 5, 1.0, start_id=1)
        _add_perf(perf, "EthAgent", Regime.BULLISH, 5, 1.2, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decisions = allocator.decide(
            make_market(prices={"BTC": 100.0, "ETH": 50.0}),
            agents=[btc_agent, eth_agent],
            players=[],
            signal_id_start=10,
        )

        by_symbol = {decision.symbol: decision for decision in decisions}
        self.assertEqual(by_symbol["BTC"].selected_actor, "BtcAgent")
        self.assertEqual(by_symbol["BTC"].actor_type, "agent")
        self.assertEqual(by_symbol["BTC"].signal.by_agent, "BtcAgent")
        self.assertEqual(by_symbol["ETH"].selected_actor, "EthAgent")
        self.assertEqual(by_symbol["ETH"].actor_type, "agent")
        self.assertEqual(by_symbol["ETH"].signal.by_agent, "EthAgent")
        self.assertNotEqual(by_symbol["BTC"].selected_actor, by_symbol["ETH"].selected_actor)

    def test_denied_open_regime_blocks_opens_but_allows_closes(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        open_agent = FakeAgent("NeutralOpenAgent", {"BTC": Action.FUT_LONG_FULL})
        close_agent = FakeAgent("NeutralCloseAgent", {"BTC": Action.FUT_CLOSE_ALL})
        _add_perf(perf, "NeutralOpenAgent", Regime.NEUTRAL, 5, 2.0, start_id=1)
        _add_perf(perf, "NeutralCloseAgent", Regime.NEUTRAL, 5, 2.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(denied_open_regimes=(Regime.NEUTRAL,)),
        )
        open_decision = allocator.decide(
            make_market(regime=Regime.NEUTRAL, prices={"BTC": 100.0}),
            agents=[open_agent],
            players=[],
            signal_id_start=10,
        )[0]
        close_decision = allocator.decide(
            make_market(regime=Regime.NEUTRAL, prices={"BTC": 100.0}),
            agents=[close_agent],
            players=[],
            signal_id_start=20,
            open_position_sides_by_symbol={"BTC": "long"},
        )[0]

        open_row = next(
            row for row in open_decision.candidates if row.label == "NeutralOpenAgent"
        )
        self.assertEqual(open_decision.selected_actor, "NoTrade")
        self.assertEqual(open_row.reason, "denied_open_regime")
        self.assertEqual(close_decision.selected_actor, "NeutralCloseAgent")
        self.assertEqual(close_decision.signal.action, Action.FUT_CLOSE_ALL)

    def test_selected_subset_context_score_boost_applies_only_matching_regime(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ContextBoostAgent", {"BTC/USDT": Action.FUT_LONG_FULL})
        _add_perf(perf, "ContextBoostAgent", Regime.BULLISH, 5, 1.0, start_id=1)
        _add_perf(perf, "ContextBoostAgent", Regime.BEARISH, 5, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                selected_subset_context_score_boosts=(
                    "agent:ContextBoostAgent|BTC/USDT|FUT_LONG_FULL|bearish=0.40",
                ),
            ),
        )

        bullish = allocator.decide(
            make_market(regime=Regime.BULLISH, prices={"BTC/USDT": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=10,
        )[0]
        bearish = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC/USDT": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=20,
        )[0]

        bullish_row = next(
            row for row in bullish.candidates if row.label == "ContextBoostAgent"
        )
        bearish_row = next(
            row for row in bearish.candidates if row.label == "ContextBoostAgent"
        )
        self.assertAlmostEqual(bullish_row.selected_subset_score_boost, 0.0)
        self.assertAlmostEqual(bearish_row.selected_subset_score_boost, 0.40)
        self.assertAlmostEqual(
            bearish_row.gate_score - bullish_row.gate_score,
            0.40,
            places=5,
        )

    def test_selected_subset_context_risk_mult_overrides_plain_signal_key(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ContextRiskAgent", {"BTC/USDT": Action.FUT_LONG_FULL})
        _add_perf(perf, "ContextRiskAgent", Regime.BEARISH, 5, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                selected_subset_risk_min_mult=0.5,
                selected_subset_risk_max_mult=1.2,
                selected_subset_risk_mult_overrides=(
                    "agent:ContextRiskAgent|BTC/USDT|FUT_LONG_FULL=1.15",
                ),
                selected_subset_context_risk_mult_overrides=(
                    "agent:ContextRiskAgent|BTC/USDT|FUT_LONG_FULL|bearish=0.80",
                ),
            ),
        )

        decision = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC/USDT": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=10,
        )[0]

        row = next(row for row in decision.candidates if row.label == "ContextRiskAgent")
        self.assertAlmostEqual(row.selected_subset_risk_mult, 0.80)
        self.assertIsNotNone(decision.signal)
        self.assertAlmostEqual(decision.signal.risk_mult, 0.80)

    def test_denied_open_symbol_blocks_opens_but_allows_closes(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        open_agent = FakeAgent("WeakSymbolOpenAgent", {"ATOM/USDT": Action.FUT_LONG_FULL})
        close_agent = FakeAgent("WeakSymbolCloseAgent", {"ATOM/USDT": Action.FUT_CLOSE_ALL})
        _add_perf(perf, "WeakSymbolOpenAgent", Regime.BULLISH, 5, 2.0, start_id=1)
        _add_perf(perf, "WeakSymbolCloseAgent", Regime.BULLISH, 5, 2.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(denied_open_symbols=("atom",)),
        )
        open_decision = allocator.decide(
            make_market(prices={"ATOM/USDT": 10.0}),
            agents=[open_agent],
            players=[],
            signal_id_start=10,
        )[0]
        close_decision = allocator.decide(
            make_market(prices={"ATOM/USDT": 10.0}),
            agents=[close_agent],
            players=[],
            signal_id_start=20,
            open_position_sides_by_symbol={"ATOM/USDT": "long"},
        )[0]

        open_row = next(
            row for row in open_decision.candidates if row.label == "WeakSymbolOpenAgent"
        )
        self.assertEqual(open_decision.selected_actor, "NoTrade")
        self.assertEqual(open_row.reason, "denied_open_symbol")
        self.assertEqual(close_decision.selected_actor, "WeakSymbolCloseAgent")
        self.assertEqual(close_decision.signal.action, Action.FUT_CLOSE_ALL)

    def test_degraded_open_symbol_blocks_opens_but_allows_closes(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        open_agent = FakeAgent("WeakSymbolOpenAgent", {"ETH/USDT": Action.FUT_LONG_FULL})
        close_agent = FakeAgent("WeakSymbolCloseAgent", {"ETH/USDT": Action.FUT_CLOSE_ALL})
        _add_perf(perf, "WeakSymbolOpenAgent", Regime.BULLISH, 5, 2.0, start_id=1)
        _add_perf(perf, "WeakSymbolCloseAgent", Regime.BULLISH, 5, 2.0, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        open_decision = allocator.decide(
            make_market(prices={"ETH/USDT": 100.0}),
            agents=[open_agent],
            players=[],
            signal_id_start=10,
            degraded_open_symbols=("eth",),
        )[0]
        close_decision = allocator.decide(
            make_market(prices={"ETH/USDT": 100.0}),
            agents=[close_agent],
            players=[],
            signal_id_start=20,
            degraded_open_symbols=("eth",),
            open_position_sides_by_symbol={"ETH/USDT": "long"},
        )[0]

        open_row = next(
            row for row in open_decision.candidates if row.label == "WeakSymbolOpenAgent"
        )
        self.assertEqual(open_decision.selected_actor, "NoTrade")
        self.assertEqual(open_row.reason, "flash_symbol_degraded")
        self.assertEqual(close_decision.selected_actor, "WeakSymbolCloseAgent")
        self.assertEqual(close_decision.signal.action, Action.FUT_CLOSE_ALL)

    def test_existing_same_side_position_suppresses_duplicate_open_before_actor_cap(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("LongAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "LongAgent", Regime.BULLISH, 5, 2.0, start_id=1)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=10,
            open_position_sides_by_symbol={"BTC": "long"},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(decision.actor_type, "no_trade")
        self.assertEqual(decision.reason, "duplicate_open_position")
        self.assertIsNone(decision.signal)
        self.assertEqual(decision.original_selected_actor, "LongAgent")
        self.assertEqual(decision.original_actor_type, "agent")

    def test_missing_position_suppresses_stale_close_before_actor_cap(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("CloseAgent", {"BTC": Action.FUT_CLOSE_ALL})
        _add_perf(perf, "CloseAgent", Regime.BULLISH, 5, 2.0, start_id=1)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=10,
            open_position_sides_by_symbol={},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(decision.reason, "stale_close_position")
        self.assertIsNone(decision.signal)
        self.assertEqual(decision.original_selected_actor, "CloseAgent")

    def test_quarantine_rejects_actor_before_scoring(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed={"StarAgent"})
        star = FakeAgent("StarAgent", {"BTC": Action.FUT_LONG_FULL})
        safe = FakeAgent("SafeAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "StarAgent", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "SafeAgent", Regime.BULLISH, 5, 0.5, start_id=200)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[star, safe],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "SafeAgent")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertIn("StarAgent", rejected)
        self.assertEqual(rejected["StarAgent"].reason, "quarantined")

    def test_ensemble_is_marked_as_ensemble_actor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        solo = FakeAgent("SoloAgent", {"BTC": Action.FUT_LONG_HALF})
        left = FakeAgent("Left", {"BTC": Action.FUT_LONG_FULL})
        right = FakeAgent("Right", {"BTC": Action.FUT_LONG_FULL})
        team = EnsemblePlayer(
            label="TeamPlayer",
            agents=[left, right],
            weights={"Left": 0.5, "Right": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "SoloAgent", Regime.BULLISH, 5, 0.2, start_id=1)
        _add_perf(perf, "TeamPlayer", Regime.BULLISH, 5, 2.0, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[solo],
            players=[team],
            signal_id_start=30,
        )[0]

        self.assertEqual(decision.selected_actor, "TeamPlayer")
        self.assertEqual(decision.actor_type, "ensemble")
        self.assertEqual(decision.signal.by_player, "TeamPlayer")
        self.assertIn(decision.signal.by_agent, ("Left", "Right"))

    def test_equal_score_tiebreak_does_not_prefer_agent_actor_type(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        solo = FakeAgent("ZuluAgent", {"BTC": Action.FUT_LONG_FULL})
        left = FakeAgent("Left", {"BTC": Action.FUT_LONG_FULL})
        right = FakeAgent("Right", {"BTC": Action.FUT_LONG_FULL})
        team = EnsemblePlayer(
            label="AlphaTeam",
            agents=[left, right],
            weights={"Left": 0.5, "Right": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "ZuluAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        _add_perf(perf, "AlphaTeam", Regime.BULLISH, 10, 2.0, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[solo],
            players=[team],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "AlphaTeam")
        self.assertEqual(decision.actor_type, "ensemble")

    def test_equal_score_tiebreak_uses_stable_hash_not_label_order(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        solo = FakeAgent("AAgent", {"BTC": Action.FUT_LONG_FULL})
        left = FakeAgent("Left", {"BTC": Action.FUT_LONG_FULL})
        right = FakeAgent("Right", {"BTC": Action.FUT_LONG_FULL})
        team = EnsemblePlayer(
            label="ZTeam",
            agents=[left, right],
            weights={"Left": 0.5, "Right": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "AAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        _add_perf(perf, "ZTeam", Regime.BULLISH, 10, 2.0, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[solo],
            players=[team],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "ZTeam")
        self.assertEqual(decision.actor_type, "ensemble")

    def test_ranking_uses_net_pnl_after_funding_costs(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        costly = FakeAgent("CostlyCarryAgent", {"BTC": Action.FUT_LONG_FULL})
        stable = FakeAgent("StableNetAgent", {"BTC": Action.FUT_LONG_FULL})
        _record_one_trade(
            perf,
            "CostlyCarryAgent",
            entry=100.0,
            exit=110.0,
            funding_close=12.0,
            start_id=1,
        )
        _record_one_trade(
            perf,
            "StableNetAgent",
            entry=100.0,
            exit=102.0,
            start_id=10,
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(min_closed_trades_to_trade=1),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[costly, stable],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "StableNetAgent")
        by_label = {row.label: row for row in decision.candidates}
        self.assertLess(by_label["CostlyCarryAgent"].base_score, 0.0)
        self.assertGreater(by_label["StableNetAgent"].base_score, 0.0)
        costly_row = by_label["CostlyCarryAgent"]
        self.assertAlmostEqual(costly_row.pnl_gross_pct, 10.0, places=5)
        self.assertAlmostEqual(costly_row.funding_pct, 12.0, places=5)
        self.assertAlmostEqual(costly_row.pnl_net_pct, -2.0, places=5)
        self.assertAlmostEqual(costly_row.trading_cost_pct, 12.0, places=5)
        payload = costly_row.as_dict()
        self.assertAlmostEqual(payload["pnl_gross_pct"], 10.0, places=5)
        self.assertAlmostEqual(payload["pnl_net_pct"], -2.0, places=5)

    def test_prefer_solo_player_wrappers_suppresses_matching_raw_agent(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        solo_player = EnsemblePlayer(
            label="Solo_MomentumScalper",
            agents=[agent],
            weights={"MomentumScalper": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 10.0, start_id=1)
        _add_perf(perf, "Solo_MomentumScalper", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_solo_player_wrappers_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[solo_player],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "Solo_MomentumScalper")
        self.assertEqual(decision.actor_type, "ensemble")
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("MomentumScalper", rows)
        self.assertTrue(rows["MomentumScalper"].rejected)
        self.assertEqual(rows["MomentumScalper"].reason, "raw_suppressed_by_solo")

    def test_portfolio_raw_actor_is_not_suppressed_by_solo_wrapper_preference(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        solo_player = EnsemblePlayer(
            label="Solo_MomentumScalper",
            agents=[agent],
            weights={"MomentumScalper": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 10.0, start_id=1)
        _add_perf(perf, "Solo_MomentumScalper", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_solo_player_wrappers_enabled=True,
                portfolio_actor_keys=("agent:MomentumScalper",),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[solo_player],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "MomentumScalper")
        rows = {row.label: row for row in decision.candidates}
        self.assertFalse(rows["MomentumScalper"].rejected)

    def test_prefer_proven_solo_player_wrappers_suppresses_weaker_raw_agent(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        solo_player = EnsemblePlayer(
            label="Solo_MomentumScalper",
            agents=[agent],
            weights={"MomentumScalper": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 0.5, start_id=1)
        _add_perf(perf, "Solo_MomentumScalper", Regime.BULLISH, 10, 5.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_proven_solo_player_wrappers_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[solo_player],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "Solo_MomentumScalper")
        self.assertEqual(decision.actor_type, "ensemble")
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("MomentumScalper", rows)
        self.assertTrue(rows["MomentumScalper"].rejected)
        self.assertEqual(rows["MomentumScalper"].reason, "raw_suppressed_by_solo")

    def test_prefer_proven_solo_player_wrappers_keeps_stronger_raw_agent(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        solo_player = EnsemblePlayer(
            label="Solo_MomentumScalper",
            agents=[agent],
            weights={"MomentumScalper": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "Solo_MomentumScalper", Regime.BULLISH, 10, 0.5, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_proven_solo_player_wrappers_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[solo_player],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "MomentumScalper")
        self.assertIn("MomentumScalper", {row.label for row in decision.candidates})

    def test_prefer_solo_player_wrappers_suppresses_shadow_agent_handoff(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.HOLD})
        solo_player = EnsemblePlayer(
            label="Solo_MomentumScalper",
            agents=[agent],
            weights={"MomentumScalper": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 10.0, start_id=1)
        _add_perf(perf, "Solo_MomentumScalper", Regime.BULLISH, 10, 1.0, start_id=100)
        shadow_signal = Signal(
            id=0,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="MomentumScalper",
            by_agent="MomentumScalper",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_solo_player_wrappers_enabled=True,
                shadow_signal_handoff_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[solo_player],
            signal_id_start=1,
            shadow_agent_signals={"MomentumScalper": [shadow_signal]},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("MomentumScalper", rows)
        self.assertTrue(rows["MomentumScalper"].rejected)
        self.assertEqual(rows["MomentumScalper"].reason, "raw_suppressed_by_solo")

    def test_prefer_solo_player_wrappers_uses_shadow_player_labels(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        solo_signal = Signal(
            id=0,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="Solo_MomentumScalper",
            by_agent="MomentumScalper",
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 10.0, start_id=1)
        _add_perf(perf, "Solo_MomentumScalper", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_solo_player_wrappers_enabled=True,
                shadow_signal_handoff_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=1,
            shadow_player_signals={"Solo_MomentumScalper": [solo_signal]},
        )[0]

        self.assertEqual(decision.selected_actor, "Solo_MomentumScalper")
        self.assertEqual(decision.actor_type, "ensemble")
        self.assertEqual(decision.action, Action.FUT_SHORT_FULL)
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("MomentumScalper", rows)
        self.assertTrue(rows["MomentumScalper"].rejected)
        self.assertEqual(rows["MomentumScalper"].reason, "raw_suppressed_by_solo")

    def test_prefer_solo_player_wrappers_does_not_suppress_on_actionable_label_only(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("MomentumScalper", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 10.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                prefer_solo_player_wrappers_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[agent],
            players=[],
            signal_id_start=1,
            actionable_labels={"Solo_MomentumScalper"},
        )[0]

        self.assertEqual(decision.selected_actor, "MomentumScalper")
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("MomentumScalper", rows)
        self.assertFalse(rows["MomentumScalper"].rejected)

    def test_flash_presets_keep_safe_default_and_aggressive_invariants(self):
        self.assertEqual(FLASH_PRESET_DEFAULT, FlashAllocatorConfig())
        self.assertNotEqual(
            FLASH_PRESET_SAFE,
            FLASH_PRESET_DEFAULT,
            "safe preset must be pinned independently from constructor defaults",
        )
        for flag in FLASH_EXPERIMENTAL_FLAGS:
            self.assertFalse(
                getattr(FLASH_PRESET_SAFE, flag),
                f"safe preset must keep {flag} disabled",
            )
            self.assertTrue(
                getattr(FLASH_PRESET_AGGRESSIVE, flag),
                f"aggressive preset must enable {flag}",
            )
        self.assertEqual(FLASH_PRESET_DEFAULT.degradation_signal_risk_mult, 0.20)
        self.assertEqual(FLASH_PRESET_SAFE.degradation_signal_risk_mult, 0.20)
        self.assertEqual(FLASH_PRESET_AGGRESSIVE.degradation_signal_risk_mult, 0.20)

    def test_flash_safe_preset_pins_high_risk_fields_explicitly(self):
        required_pins = {
            "anchor_actor_keys": (),
            "portfolio_actor_keys": (),
            "denied_signal_keys": (),
            "terminal_denied_signal_keys": (),
            "denied_open_symbols": (),
            "denied_open_regimes": (),
            "degradation_signal_cooldown_bars": 0,
            "degradation_actor_cooldown_bars": 0,
            "degradation_symbol_guard_enabled": False,
            "degradation_symbol_cooldown_bars": 0,
            "degradation_symbol_lookback_bars": 0,
            "degradation_symbol_window_closed_trades": 0,
            "degradation_symbol_min_closed_trades": 0,
            "degradation_symbol_max_recent_pnl_usd": None,
            "shadow_confirmation_min_pnl_per_trade_lcb_usd": None,
            "shadow_confirmation_pnl_per_trade_lcb_penalty_weight": 0.0,
            "shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled": False,
            "shadow_confirmation_pnl_per_trade_lcb_risk_min_mult": 0.25,
            "shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd": 0.0,
            "shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd": 1.0,
            "shadow_symbol_health_enabled": False,
            "shadow_symbol_health_min_closed_trades": 0,
            "shadow_symbol_health_min_pnl_per_trade_lcb_usd": None,
            "shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd": 0.0,
            "shadow_symbol_health_pnl_per_trade_lcb_penalty_weight": 0.0,
            "promoted_actor_cap_overrides": (),
            "actor_risk_sizing_enabled": False,
            "actor_risk_min_mult": 0.25,
            "actor_risk_max_mult": 1.0,
            "actor_risk_edge_scale_pct": 0.50,
            "funding_score_weight": 0.0,
            "funding_risk_mult_weight": 0.0,
            "funding_risk_mult_cap": 0.25,
            "no_trade_fee_saving_score_enabled": False,
            "no_trade_default_fee_bps": 0.0,
            "volatility_risk_sizing_enabled": False,
            "volatility_risk_target_pct": 2.0,
            "volatility_risk_min_volatility_pct": 0.5,
            "volatility_risk_max_mult": 2.0,
        }

        self.assertGreaterEqual(
            set(FLASH_PRESET_SAFE_PINNED_VALUES),
            set(required_pins),
        )
        for field, expected in required_pins.items():
            self.assertEqual(FLASH_PRESET_SAFE_PINNED_VALUES[field], expected)
            self.assertEqual(getattr(FLASH_PRESET_SAFE, field), expected)

    def test_no_trade_fee_saving_score_can_beat_small_positive_edge(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("LowEdge", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "LowEdge", Regime.BULLISH, 10, 0.001, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                no_trade_fee_saving_score_enabled=True,
                no_trade_default_fee_bps=1_000.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        no_trade = {row.actor_key: row for row in decision.candidates}["NoTrade"]
        low_edge = {row.label: row for row in decision.candidates}["LowEdge"]
        self.assertGreater(no_trade.score, low_edge.score)

    def test_funding_score_prefers_funding_aligned_short(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        long_agent = FakeAgent("LongAgent", {"BTC": Action.FUT_LONG_FULL})
        short_agent = FakeAgent("ShortAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "LongAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "ShortAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(funding_score_weight=1.0),
        )
        decision = allocator.decide(
            replace(
                make_market(prices={"BTC": 100.0}),
                funding={"BTC": 0.25},
            ),
            agents=[long_agent, short_agent],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "ShortAgent")
        rows = {row.label: row for row in decision.candidates}
        self.assertGreater(
            rows["ShortAgent"].funding_score_adjustment,
            rows["LongAgent"].funding_score_adjustment,
        )

    def test_actor_edge_risk_sizing_reduces_weak_edge_signal(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        weak = FakeAgent("WeakEdge", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "WeakEdge", Regime.BULLISH, 6, 0.10, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                actor_risk_sizing_enabled=True,
                actor_risk_min_mult=0.25,
                actor_risk_max_mult=1.0,
                actor_risk_edge_scale_pct=1.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[weak],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertIsNotNone(decision.signal)
        self.assertGreater(decision.signal.risk_mult, 0.25)
        self.assertLess(decision.signal.risk_mult, 1.0)

    def test_funding_and_volatility_risk_sizing_adjust_open_risk_mult(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        short_agent = FakeAgent("ShortAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "ShortAgent", Regime.BULLISH, 6, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                funding_risk_mult_weight=1.0,
                funding_risk_mult_cap=0.25,
                volatility_risk_sizing_enabled=True,
                volatility_risk_target_pct=2.0,
                volatility_risk_min_volatility_pct=0.5,
                volatility_risk_max_mult=2.0,
            ),
        )
        decision = allocator.decide(
            replace(
                make_market(prices={"BTC": 100.0}),
                funding={"BTC": 0.25},
                lookback_volatility_pct={"BTC": {12: 4.0}},
            ),
            agents=[short_agent],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertIsNotNone(decision.signal)
        # Funding benefit adds +25%, high volatility halves the position: 1.25 * 0.5.
        self.assertAlmostEqual(decision.signal.risk_mult, 0.625, places=5)

    def test_selected_reasons_preserve_anchor_then_switch_margin_chain(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        greedy = FakeAgent("Greedy", {"BTC": Action.FUT_LONG_FULL})
        anchor = FakeAgent("Anchor", {"BTC": Action.FUT_SHORT_FULL})
        previous = FakeAgent("Previous", {"BTC": Action.SPOT_BUY_FULL})
        _add_perf(perf, "Greedy", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "Anchor", Regime.BULLISH, 10, 4.9, start_id=100)
        _add_perf(perf, "Previous", Regime.BULLISH, 10, 4.8, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                actor_switch_margin=1.0,
                anchor_actor_keys=("agent:Anchor",),
                anchor_min_score_advantage=1.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[greedy, anchor, previous],
            players=[],
            signal_id_start=1,
            previous_actor_by_symbol={"BTC": "agent:Previous"},
        )[0]

        self.assertEqual(decision.reason, "switch_margin_hold")
        self.assertEqual(
            decision.selected_reasons,
            ("selected", "anchor_dominance_hold", "switch_margin_hold"),
        )

    def test_switch_margin_previous_match_uses_actor_key_not_bare_label(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        greedy = FakeAgent("Greedy", {"BTC": Action.FUT_LONG_FULL})
        previous = FakeAgent("Previous", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Greedy", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "Previous", Regime.BULLISH, 10, 4.9, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(actor_switch_margin=1.0),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[greedy, previous],
            players=[],
            signal_id_start=1,
            previous_actor_by_symbol={"BTC": "Previous"},
        )[0]

        self.assertEqual(decision.selected_actor, "Greedy")
        self.assertEqual(decision.reason, "selected")

    def test_duplicate_actor_labels_do_not_cross_wire_selected_signal(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        solo = FakeAgent("Collision", {"BTC": Action.FUT_LONG_FULL})
        left = FakeAgent("Left", {"BTC": Action.FUT_SHORT_FULL})
        right = FakeAgent("Right", {"BTC": Action.FUT_SHORT_FULL})
        team = EnsemblePlayer(
            label="Collision",
            agents=[left, right],
            weights={"Left": 0.5, "Right": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "Collision", Regime.BULLISH, 5, 1.0, start_id=1)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[solo],
            players=[team],
            signal_id_start=30,
        )[0]

        self.assertEqual(decision.selected_actor, "Collision")
        self.assertEqual(decision.signal.by_player, "Collision")
        self.assertEqual(decision.signal.action, decision.action)
        if decision.actor_type == "agent":
            self.assertEqual(decision.action, Action.FUT_LONG_FULL)
            self.assertEqual(decision.signal.by_agent, "Collision")
        else:
            self.assertEqual(decision.actor_type, "ensemble")
            self.assertEqual(decision.action, Action.FUT_SHORT_FULL)
            self.assertIn(decision.signal.by_agent, {"Left", "Right"})

    def test_signal_deny_key_rejects_only_matching_actor_symbol_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        blocked = FakeAgent("BlockedAgent", {"BTC": Action.SPOT_BUY_FULL})
        fallback = FakeAgent("FallbackAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BlockedAgent", Regime.BULLISH, 10, 3.0, start_id=1)
        _add_perf(perf, "FallbackAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                denied_signal_keys=(
                    "agent:BlockedAgent|BTC|SPOT_BUY_FULL",
                ),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[blocked, fallback],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "FallbackAgent")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["BlockedAgent"].reason, "flash_signal_deny_key")

    def test_terminal_signal_deny_key_suppresses_fallback_when_blocked_would_win(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        blocked = FakeAgent("BlockedAgent", {"BTC": Action.SPOT_BUY_FULL})
        fallback = FakeAgent("FallbackAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BlockedAgent", Regime.BULLISH, 10, 3.0, start_id=1)
        _add_perf(perf, "FallbackAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                terminal_denied_signal_keys=(
                    "agent:BlockedAgent|BTC|SPOT_BUY_FULL",
                ),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[blocked, fallback],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(decision.reason, "flash_signal_terminal_deny_key")
        self.assertEqual(decision.original_selected_actor, "BlockedAgent")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["BlockedAgent"].reason,
            "flash_signal_terminal_deny_key",
        )

    def test_terminal_signal_deny_key_allows_stronger_non_denied_candidate(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        blocked = FakeAgent("BlockedAgent", {"BTC": Action.SPOT_BUY_FULL})
        winner = FakeAgent("WinnerAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BlockedAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "WinnerAgent", Regime.BULLISH, 10, 3.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                terminal_denied_signal_keys=(
                    "agent:BlockedAgent|BTC|SPOT_BUY_FULL",
                ),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[blocked, winner],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "WinnerAgent")
        self.assertEqual(decision.reason, "selected")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["BlockedAgent"].reason,
            "flash_signal_terminal_deny_key",
        )

    def test_terminal_context_signal_deny_key_matches_regime_only(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        blocked = FakeAgent("BlockedAgent", {"BTC": Action.SPOT_BUY_FULL})
        fallback = FakeAgent("FallbackAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BlockedAgent", Regime.BULLISH, 10, 3.0, start_id=1)
        _add_perf(perf, "FallbackAgent", Regime.BULLISH, 10, 1.0, start_id=100)
        _add_perf(perf, "BlockedAgent", Regime.BEARISH, 10, 3.0, start_id=200)
        _add_perf(perf, "FallbackAgent", Regime.BEARISH, 10, 1.0, start_id=300)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                terminal_denied_context_signal_keys=(
                    "agent:BlockedAgent|BTC|SPOT_BUY_FULL|bullish",
                ),
            ),
        )

        bullish = allocator.decide(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            agents=[blocked, fallback],
            players=[],
            signal_id_start=1,
        )[0]
        bearish = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0}),
            agents=[blocked, fallback],
            players=[],
            signal_id_start=1000,
        )[0]

        self.assertEqual(bullish.selected_actor, "NoTrade")
        self.assertEqual(bullish.reason, "flash_signal_terminal_deny_key")
        rejected = {row.label: row for row in bullish.candidates if row.rejected}
        self.assertEqual(
            rejected["BlockedAgent"].reason,
            "flash_signal_terminal_deny_key",
        )
        self.assertEqual(bearish.selected_actor, "BlockedAgent")
        self.assertEqual(bearish.reason, "selected")

    def test_runtime_degraded_signal_key_rejects_only_matching_actor_symbol_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        blocked = FakeAgent("BlockedAgent", {"BTC": Action.SPOT_BUY_FULL})
        fallback = FakeAgent("FallbackAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BlockedAgent", Regime.BULLISH, 10, 3.0, start_id=1)
        _add_perf(perf, "FallbackAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[blocked, fallback],
            players=[],
            signal_id_start=1,
            degraded_signal_keys=(
                "agent:BlockedAgent|BTC|SPOT_BUY_FULL",
            ),
        )[0]

        self.assertEqual(decision.selected_actor, "FallbackAgent")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["BlockedAgent"].reason, "flash_signal_degraded")

    def test_degraded_signal_can_risk_size_instead_of_rejecting(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        degraded = FakeAgent("DegradedAgent", {"BTC": Action.SPOT_BUY_FULL})
        fallback = FakeAgent("FallbackAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "DegradedAgent", Regime.BULLISH, 10, 3.0, start_id=1)
        _add_perf(perf, "FallbackAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                degradation_signal_risk_sizing_enabled=True,
                degradation_signal_risk_mult=0.25,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[degraded, fallback],
            players=[],
            signal_id_start=1,
            degraded_signal_keys=(
                "agent:DegradedAgent|BTC|SPOT_BUY_FULL",
            ),
        )[0]

        self.assertEqual(decision.selected_actor, "DegradedAgent")
        self.assertIsNotNone(decision.signal)
        assert decision.signal.risk_mult == pytest.approx(0.25)
        row = next(item for item in decision.candidates if item.label == "DegradedAgent")
        self.assertFalse(row.rejected)
        self.assertEqual(row.reason, "flash_signal_degraded_risk_sized")
        assert row.risk_mult == pytest.approx(0.25)

    def test_runtime_degraded_actor_key_rejects_all_actor_signals(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        weak = FakeAgent(
            "WeakAgent",
            {"BTC": Action.SPOT_BUY_FULL, "ETH": Action.FUT_SHORT_FULL},
        )
        backup = FakeAgent("BackupAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "WeakAgent", Regime.BULLISH, 10, 5.0, start_id=1)
        _add_perf(perf, "BackupAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decisions = allocator.decide(
            make_market(prices={"BTC": 100.0, "ETH": 50.0}),
            agents=[weak, backup],
            players=[],
            signal_id_start=1,
            degraded_actor_keys=("agent:WeakAgent",),
        )

        by_symbol = {decision.symbol: decision for decision in decisions}
        self.assertEqual(by_symbol["BTC"].selected_actor, "BackupAgent")
        self.assertEqual(by_symbol["ETH"].selected_actor, "NoTrade")
        rejected = {row.label: row for row in by_symbol["ETH"].candidates if row.rejected}
        self.assertEqual(rejected["WeakAgent"].reason, "flash_actor_degraded")

    def test_shadow_pnl_per_trade_lcb_gate_rejects_bad_selected_subset(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=3,
                shadow_confirmation_min_pnl_per_trade_lcb_usd=0.0,
                shadow_confirmation_pnl_per_trade_lcb_z=1.0,
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 10.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": -0.1,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("Beta", "BTC", "FUT_SHORT_FULL"): {
                    "score": 5.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": 0.2,
                    "pnl_per_trade_std_usd": 0.0,
                },
            },
        )[0]

        self.assertEqual(decision.selected_actor, "Beta")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["Alpha"].reason,
            "shadow_pnl_per_trade_lcb_below_threshold",
        )
        self.assertEqual(
            rejected["Alpha"].shadow_pnl_per_trade_lcb_usd,
            -0.1,
        )

    def test_shadow_pnl_per_trade_lcb_penalty_demotes_bad_selected_subset(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=3,
                shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd=0.0,
                shadow_confirmation_pnl_per_trade_lcb_penalty_weight=60.0,
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 10.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": -0.1,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("Beta", "BTC", "FUT_SHORT_FULL"): {
                    "score": 5.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": 0.2,
                    "pnl_per_trade_std_usd": 0.0,
                },
            },
        )[0]

        self.assertEqual(decision.selected_actor, "Beta")
        candidates = {row.label: row for row in decision.candidates}
        self.assertFalse(candidates["Alpha"].rejected)
        self.assertEqual(candidates["Alpha"].reason, "eligible")
        self.assertEqual(candidates["Alpha"].shadow_pnl_per_trade_lcb_penalty, 6.0)
        self.assertEqual(candidates["Alpha"].score, 4.0)
        self.assertEqual(candidates["Beta"].score, 5.0)

    def test_shadow_pnl_per_trade_lcb_risk_sizing_reduces_open_size(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=3,
                shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled=True,
                shadow_confirmation_pnl_per_trade_lcb_risk_min_mult=0.4,
                shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd=0.0,
                shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd=1.0,
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 10.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": -0.5,
                    "pnl_per_trade_std_usd": 0.0,
                },
            },
        )[0]

        self.assertEqual(decision.selected_actor, "Alpha")
        self.assertEqual(decision.candidates[0].reason, "eligible")
        self.assertIsNotNone(decision.signal)
        self.assertAlmostEqual(decision.signal.risk_mult, 0.5)
        self.assertAlmostEqual(decision.candidates[0].risk_mult, 0.5)

    def test_selected_subset_score_boost_can_promote_confirmed_cell(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 10, 0.90, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 10, 1.00, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                selected_subset_score_boosts=(
                    "agent:Alpha|BTC|FUT_LONG_FULL=1.0",
                ),
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=10,
        )[0]

        self.assertEqual(decision.selected_actor, "Alpha")
        rows = {row.label: row for row in decision.candidates}
        self.assertAlmostEqual(rows["Alpha"].selected_subset_score_boost, 1.0)
        self.assertGreater(rows["Alpha"].score, rows["Beta"].score)

    def test_selected_subset_do_not_demote_bypasses_lcb_hard_gate(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 10, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
                shadow_confirmation_min_pnl_per_trade_lcb_usd=0.0,
                selected_subset_do_not_demote_signal_keys=(
                    "agent:Alpha|BTC|FUT_LONG_FULL",
                ),
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha],
            players=[],
            signal_id_start=10,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 1.0,
                    "closed_trades": 10,
                    "winning_trades": 3,
                    "win_rate_pct": 30.0,
                    "pnl_per_trade_lcb_usd": -5.0,
                }
            },
        )[0]

        self.assertEqual(decision.selected_actor, "Alpha")
        row = decision.candidates[0]
        self.assertTrue(row.selected_subset_protected)
        self.assertEqual(row.reason, "eligible")
        self.assertAlmostEqual(row.shadow_pnl_per_trade_lcb_penalty, 0.0)

    def test_selected_subset_risk_mult_is_bounded_and_audited(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 10, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                selected_subset_risk_mult_overrides=(
                    "agent:Alpha|BTC|FUT_LONG_FULL=1.5",
                ),
                selected_subset_risk_min_mult=0.75,
                selected_subset_risk_max_mult=1.15,
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha],
            players=[],
            signal_id_start=10,
        )[0]

        self.assertAlmostEqual(decision.signal.risk_mult, 1.15)
        self.assertAlmostEqual(decision.candidates[0].risk_mult, 1.15)
        self.assertAlmostEqual(decision.candidates[0].selected_subset_risk_mult, 1.15)

    def test_shadow_symbol_health_penalty_demotes_bad_symbol_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=3,
                shadow_symbol_health_enabled=True,
                shadow_symbol_health_min_closed_trades=3,
                shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd=0.0,
                shadow_symbol_health_pnl_per_trade_lcb_penalty_weight=60.0,
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 10.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": -0.2,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("Beta", "BTC", "FUT_SHORT_FULL"): {
                    "score": 5.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": 0.2,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("__symbol_health__", "BTC", "FUT_LONG_FULL"): {
                    "score": -1.0,
                    "closed_trades": 8,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": -0.1,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("__symbol_health__", "BTC", "FUT_SHORT_FULL"): {
                    "score": 1.0,
                    "closed_trades": 8,
                    "winning_trades": 5,
                    "pnl_per_trade_mean_usd": 0.1,
                    "pnl_per_trade_std_usd": 0.0,
                },
            },
        )[0]

        self.assertEqual(decision.selected_actor, "Beta")
        candidates = {row.label: row for row in decision.candidates}
        self.assertFalse(candidates["Alpha"].rejected)
        self.assertEqual(candidates["Alpha"].shadow_symbol_health_closed_trades, 8)
        self.assertEqual(candidates["Alpha"].shadow_symbol_health_pnl_per_trade_lcb_usd, -0.1)
        self.assertEqual(candidates["Alpha"].shadow_symbol_health_penalty, 6.0)
        self.assertEqual(candidates["Alpha"].score, 4.0)
        self.assertEqual(candidates["Beta"].score, 5.0)

    def test_shadow_symbol_health_penalty_respects_positive_actor_lcb(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=3,
                shadow_symbol_health_enabled=True,
                shadow_symbol_health_min_closed_trades=3,
                shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd=0.0,
                shadow_symbol_health_pnl_per_trade_lcb_penalty_weight=60.0,
            ),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 10.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": 0.2,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("Beta", "BTC", "FUT_SHORT_FULL"): {
                    "score": 5.0,
                    "closed_trades": 3,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": 0.2,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("__symbol_health__", "BTC", "FUT_LONG_FULL"): {
                    "score": -1.0,
                    "closed_trades": 8,
                    "winning_trades": 2,
                    "pnl_per_trade_mean_usd": -0.1,
                    "pnl_per_trade_std_usd": 0.0,
                },
                ("__symbol_health__", "BTC", "FUT_SHORT_FULL"): {
                    "score": 1.0,
                    "closed_trades": 8,
                    "winning_trades": 5,
                    "pnl_per_trade_mean_usd": 0.1,
                    "pnl_per_trade_std_usd": 0.0,
                },
            },
        )[0]

        self.assertEqual(decision.selected_actor, "Alpha")
        candidates = {row.label: row for row in decision.candidates}
        self.assertEqual(candidates["Alpha"].shadow_symbol_health_penalty, 0.0)
        self.assertEqual(candidates["Alpha"].score, 10.0)

    def test_promotion_manifest_rejects_unpromoted_open_signal_key(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(promotion_manifest_enabled=True),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            promoted_signal_keys=("agent:Beta|BTC|FUT_SHORT_FULL",),
        )[0]

        assert decision.selected_actor == "Beta"
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        assert rejected["Alpha"].reason == "flash_signal_not_promoted"

    def test_promotion_manifest_does_not_block_close_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        closer = FakeAgent("CloserAgent", {"BTC": Action.FUT_CLOSE_ALL})
        _add_perf(perf, "CloserAgent", Regime.BEARISH, 20, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(promotion_manifest_enabled=True),
        )

        decision = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0}),
            agents=[closer],
            players=[],
            signal_id_start=1,
            promoted_signal_keys=(),
        )[0]

        self.assertEqual(decision.selected_actor, "CloserAgent")
        self.assertEqual(decision.action, Action.FUT_CLOSE_ALL)

    def test_promotion_manifest_disabled_keeps_unpromoted_open_signal_eligible(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(promotion_manifest_enabled=False),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            promoted_signal_keys=("agent:Beta|BTC|FUT_SHORT_FULL",),
        )[0]

        self.assertEqual(decision.selected_actor, "Alpha")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertNotIn("Alpha", rejected)

    def test_promotion_manifest_disabled_ignores_invalid_promoted_signal_key(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(promotion_manifest_enabled=False),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
            promoted_signal_keys=("not-a-valid-key",),
        )[0]

        self.assertEqual(decision.selected_actor, "Alpha")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertNotIn("Alpha", rejected)

    def test_promotion_manifest_enabled_rejects_invalid_promoted_signal_key(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(promotion_manifest_enabled=True),
        )

        with self.assertRaises(ValueError):
            allocator.decide(
                make_market(prices={"BTC": 100.0}),
                agents=[alpha],
                players=[],
                signal_id_start=1,
                promoted_signal_keys=("not-a-valid-key",),
            )

    def test_promotion_manifest_accepts_promoted_ensemble_signal_key(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        left = FakeAgent("Left", {"BTC": Action.FUT_LONG_FULL})
        right = FakeAgent("Right", {"BTC": Action.FUT_LONG_FULL})
        team = EnsemblePlayer(
            label="TeamPlayer",
            agents=[left, right],
            weights={"Left": 0.5, "Right": 0.5},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "TeamPlayer", Regime.BULLISH, 20, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(promotion_manifest_enabled=True),
        )

        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[team],
            signal_id_start=1,
            promoted_signal_keys=("ensemble:TeamPlayer|BTC|FUT_LONG_FULL",),
        )[0]

        self.assertEqual(decision.selected_actor, "TeamPlayer")
        self.assertEqual(decision.actor_type, "ensemble")
        self.assertEqual(decision.signal.by_player, "TeamPlayer")

    def test_promoted_signal_key_can_override_actor_signal_cap(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent(
            "Alpha",
            {
                "BTC": Action.FUT_SHORT_FULL,
                "ETH": Action.FUT_SHORT_FULL,
            },
        )
        _add_perf(perf, "Alpha", Regime.BEARISH, 50, 8.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=1,
                promotion_manifest_enabled=True,
                promoted_actor_cap_overrides=("agent:Alpha=2",),
            ),
        )

        decisions = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            agents=[alpha],
            players=[],
            signal_id_start=1,
            promoted_signal_keys=(
                "agent:Alpha|BTC|FUT_SHORT_FULL",
                "agent:Alpha|ETH|FUT_SHORT_FULL",
            ),
        )

        assert [decision.selected_actor for decision in decisions].count("Alpha") == 2

    def test_malformed_promoted_actor_cap_override_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "actor cap override"):
            FlashAllocatorConfig(promoted_actor_cap_overrides=("agent:Alpha",))

    def test_actor_cap_override_applies_when_default_cap_is_zero(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent(
            "Alpha",
            {
                "BTC": Action.FUT_SHORT_FULL,
                "ETH": Action.FUT_SHORT_FULL,
            },
        )
        _add_perf(perf, "Alpha", Regime.BEARISH, 50, 8.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=0,
                promoted_actor_cap_overrides=("agent:Alpha=1",),
            ),
        )

        decisions = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
            agents=[alpha],
            players=[],
            signal_id_start=1,
        )

        selected = [row for row in decisions if row.selected_actor == "Alpha"]
        capped = [row for row in decisions if row.reason == "actor_signal_cap"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(len(capped), 1)

    def test_non_overridden_actor_still_uses_default_actor_signal_cap(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        alpha = FakeAgent(
            "Alpha",
            {
                "BTC": Action.FUT_SHORT_FULL,
                "ETH": Action.FUT_SHORT_FULL,
            },
        )
        beta = FakeAgent(
            "Beta",
            {
                "SOL": Action.FUT_SHORT_FULL,
                "XRP": Action.FUT_SHORT_FULL,
            },
        )
        _add_perf(perf, "Alpha", Regime.BEARISH, 50, 8.0, start_id=1)
        _add_perf(perf, "Beta", Regime.BEARISH, 50, 7.0, start_id=100)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=1,
                promoted_actor_cap_overrides=("agent:Alpha=2",),
            ),
        )

        decisions = allocator.decide(
            make_market(
                regime=Regime.BEARISH,
                prices={"BTC": 100.0, "ETH": 50.0, "SOL": 20.0, "XRP": 0.5},
            ),
            agents=[alpha, beta],
            players=[],
            signal_id_start=1,
        )

        self.assertEqual([row.selected_actor for row in decisions].count("Alpha"), 2)
        self.assertEqual([row.selected_actor for row in decisions].count("Beta"), 1)
        capped = [row for row in decisions if row.reason == "actor_signal_cap"]
        self.assertEqual(len(capped), 1)

    def test_degraded_actor_cap_reservation_works_with_actor_cap_override(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent(
            "MomentumScalper",
            {
                "ATOM": Action.SPOT_BUY_FULL,
                "AVAX": Action.SPOT_BUY_FULL,
                "BTC": Action.SPOT_BUY_FULL,
            },
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=1,
                degradation_reserve_actor_cap=True,
                promoted_actor_cap_overrides=("agent:MomentumScalper=2",),
            ),
        )

        decisions = allocator.decide(
            make_market(prices={"ATOM": 10.0, "AVAX": 20.0, "BTC": 30.0}),
            agents=[agent],
            players=[],
            signal_id_start=1,
            degraded_signal_keys=(
                "agent:MomentumScalper|ATOM|SPOT_BUY_FULL",
            ),
        )
        by_symbol = {decision.symbol: decision for decision in decisions}

        self.assertEqual(by_symbol["ATOM"].selected_actor, "NoTrade")
        self.assertEqual(by_symbol["AVAX"].selected_actor, "MomentumScalper")
        self.assertEqual(by_symbol["BTC"].selected_actor, "NoTrade")
        self.assertEqual(by_symbol["BTC"].reason, "actor_signal_cap")

    def test_degraded_unpromoted_signal_keeps_degraded_reason_for_actor_cap_reservation(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent(
            "MomentumScalper",
            {
                "ATOM": Action.SPOT_BUY_FULL,
                "AVAX": Action.SPOT_BUY_FULL,
            },
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=1,
                degradation_reserve_actor_cap=True,
                promotion_manifest_enabled=True,
            ),
        )

        decisions = allocator.decide(
            make_market(prices={"ATOM": 10.0, "AVAX": 20.0}),
            agents=[agent],
            players=[],
            signal_id_start=1,
            degraded_signal_keys=(
                "agent:MomentumScalper|ATOM|SPOT_BUY_FULL",
            ),
            promoted_signal_keys=(
                "agent:MomentumScalper|AVAX|SPOT_BUY_FULL",
            ),
        )
        by_symbol = {decision.symbol: decision for decision in decisions}

        self.assertEqual(by_symbol["ATOM"].selected_actor, "NoTrade")
        atom_rejected = {
            row.label: row for row in by_symbol["ATOM"].candidates if row.rejected
        }
        self.assertEqual(
            atom_rejected["MomentumScalper"].reason,
            "flash_signal_degraded",
        )
        self.assertEqual(by_symbol["AVAX"].selected_actor, "NoTrade")
        self.assertEqual(by_symbol["AVAX"].reason, "actor_signal_cap")
        self.assertIsNone(by_symbol["AVAX"].signal)

    def test_degraded_signal_can_reserve_actor_cap_to_avoid_reallocation(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent(
            "MomentumScalper",
            {
                "ATOM": Action.SPOT_BUY_FULL,
                "AVAX": Action.SPOT_BUY_FULL,
            },
        )
        _add_perf(perf, "MomentumScalper", Regime.BULLISH, 10, 3.0, start_id=1)
        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=1,
                degradation_reserve_actor_cap=True,
            ),
        )

        decisions = allocator.decide(
            make_market(prices={"ATOM": 10.0, "AVAX": 20.0}),
            agents=[agent],
            players=[],
            signal_id_start=1,
            degraded_signal_keys=(
                "agent:MomentumScalper|ATOM|SPOT_BUY_FULL",
            ),
        )
        by_symbol = {decision.symbol: decision for decision in decisions}

        self.assertEqual(by_symbol["ATOM"].selected_actor, "NoTrade")
        self.assertIsNone(by_symbol["ATOM"].signal)
        self.assertEqual(by_symbol["AVAX"].selected_actor, "NoTrade")
        self.assertEqual(by_symbol["AVAX"].reason, "actor_signal_cap")
        self.assertIsNone(by_symbol["AVAX"].signal)

    def test_component_score_weights_by_closed_trades_and_sums_pnl(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        _add_perf(perf, "DeepHistory", Regime.BULLISH, 200, 1.0, start_id=1)
        _add_perf(perf, "TinyLucky", Regime.BULLISH, 3, 50.0, start_id=1000)
        allocator = FlashAllocator(perf=perf, qm=qm)

        metrics, score = allocator._component_score(
            ("DeepHistory", "TinyLucky"),
            Regime.BULLISH,
        )
        deep_metrics = allocator._metrics_for_label("DeepHistory", Regime.BULLISH)
        tiny_metrics = allocator._metrics_for_label("TinyLucky", Regime.BULLISH)
        deep_score = allocator._score_actor(
            type(
                "Output",
                (),
                {
                    "label": "DeepHistory",
                    "agent_labels": (),
                    "actor_type": "agent",
                },
            )(),
            Regime.BULLISH,
        )[1]
        tiny_score = allocator._score_actor(
            type(
                "Output",
                (),
                {
                    "label": "TinyLucky",
                    "agent_labels": (),
                    "actor_type": "agent",
                },
            )(),
            Regime.BULLISH,
        )[1]
        expected_score = (
            deep_score * deep_metrics.closed_trades
            + tiny_score * tiny_metrics.closed_trades
        ) / metrics.closed_trades

        self.assertEqual(metrics.closed_trades, 203)
        self.assertAlmostEqual(
            metrics.pnl_pct,
            deep_metrics.pnl_pct + tiny_metrics.pnl_pct,
        )
        self.assertAlmostEqual(score, expected_score)

    def test_no_trade_when_all_actors_are_inactive_for_symbol(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        idle = FakeAgent("IdleAgent", {"ETH": Action.FUT_LONG_FULL})

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(min_score_to_trade=-10.0),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[idle],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(decision.actor_type, "no_trade")
        self.assertEqual(decision.action, Action.HOLD)
        self.assertIsNone(decision.signal)
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("NoTrade", rows)
        self.assertFalse(rows["NoTrade"].rejected)
        self.assertEqual(rows["NoTrade"].reason, "eligible_no_trade")

    def test_no_trade_is_first_class_candidate_when_no_actor_eligible(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        weak = FakeAgent("WeakAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "WeakAgent", Regime.BULLISH, 6, -0.4, start_id=1)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[weak],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rows = {row.label: row for row in decision.candidates}
        self.assertIn("NoTrade", rows)
        self.assertFalse(rows["NoTrade"].rejected)
        self.assertEqual(rows["NoTrade"].score, 0.0)
        self.assertEqual(rows["WeakAgent"].reason, "pnl_below_threshold")

    def test_no_trade_when_active_actor_has_no_evidence(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        unproven = FakeAgent("UnprovenAgent", {"BTC": Action.FUT_LONG_FULL})

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[unproven],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["UnprovenAgent"].reason, "no_evidence")

    def test_close_signal_does_not_require_actor_evidence(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        closer = FakeAgent("FreshCloser", {"BTC": Action.FUT_CLOSE_ALL})

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_score_to_trade=4.0,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=5,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[closer],
            players=[],
            signal_id_start=1,
            open_position_sides_by_symbol={"BTC": "long"},
        )[0]

        self.assertEqual(decision.selected_actor, "FreshCloser")
        self.assertEqual(decision.action, Action.FUT_CLOSE_ALL)
        self.assertIsNotNone(decision.signal)
        rows = {row.label: row for row in decision.candidates}
        self.assertFalse(rows["FreshCloser"].rejected)
        self.assertEqual(rows["FreshCloser"].reason, "eligible")

    def test_no_trade_when_active_actor_has_negative_pnl(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        weak = FakeAgent("WeakAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "WeakAgent", Regime.BULLISH, 6, -0.4, start_id=1)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[weak],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["WeakAgent"].reason, "pnl_below_threshold")

    def test_portfolio_actor_uses_aggregate_metrics_over_bad_regime_slice(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        portfolio = FakeAgent("PortfolioAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "PortfolioAgent", Regime.BULLISH, 3, -1.0, start_id=1)
        _add_perf(perf, "PortfolioAgent", Regime.BEARISH, 12, 2.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_closed_trades_to_trade=1,
                min_pnl_pct_to_trade=0.0,
                portfolio_actor_keys=("agent:PortfolioAgent",),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}, regime=Regime.BULLISH),
            agents=[portfolio],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "PortfolioAgent")
        row = {row.label: row for row in decision.candidates}["PortfolioAgent"]
        self.assertGreater(row.pnl_net_pct, 0.0)
        self.assertEqual(row.reason, "eligible")

    def test_portfolio_actor_can_bootstrap_min_closed_from_confirmed_shadow(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("PortfolioAgent", {"BTC": Action.FUT_LONG_FULL})
        player = EnsemblePlayer(
            label="Solo_PortfolioAgent",
            agents=[agent],
            weights={"PortfolioAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "Solo_PortfolioAgent", Regime.BULLISH, 1, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_closed_trades_to_trade=3,
                shadow_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=5,
                portfolio_actor_keys=("ensemble:Solo_PortfolioAgent",),
                portfolio_shadow_bootstrap_min_closed_enabled=True,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[player],
            signal_id_start=10,
            shadow_confirmation={"Solo_PortfolioAgent": (5.0, 10)},
        )[0]

        self.assertEqual(decision.selected_actor, "Solo_PortfolioAgent")
        self.assertEqual(decision.candidates[0].shadow_closed_trades, 10)

    def test_actionable_bonus_does_not_bypass_min_score_gate(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActionableAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActionableAgent", Regime.BULLISH, 6, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_score_to_trade=1.0,
                actionable_bonus=0.5,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            actionable_labels={"ActionableAgent"},
            shadow_confirmation={
                ("ActionableAgent", "BTC", "FUT_LONG_FULL"): (0.75, 3),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        row = {row.label: row for row in decision.candidates}["ActionableAgent"]
        self.assertEqual(row.reason, "score_below_threshold")
        self.assertEqual(row.gate_score, 0.75)
        self.assertEqual(row.effective_score, 1.125)

    def test_actionable_bonus_is_scaled_by_regime_confidence(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActionableAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActionableAgent", Regime.BULLISH, 6, 1.0, start_id=1)
        market = replace(
            make_market(prices={"BTC": 100.0}),
            regime_confidence=0.5,
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_score_to_trade=-100.0,
                actionable_bonus=0.40,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
            actionable_labels={"ActionableAgent"},
            shadow_confirmation={
                ("ActionableAgent", "BTC", "FUT_LONG_FULL"): (1.5, 3),
            },
        )[0]

        row = {row.label: row for row in decision.candidates}["ActionableAgent"]
        self.assertEqual(row.gate_score, 0.75)
        self.assertAlmostEqual(row.effective_score, 0.90, places=5)

    def test_actionable_bonus_is_multiplicative(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActionableAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActionableAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(actionable_bonus=0.25),
        )
        decision = allocator.decide(
            replace(make_market(prices={"BTC": 100.0}), regime_confidence=0.5),
            agents=[active],
            players=[],
            signal_id_start=1,
            actionable_labels={"ActionableAgent"},
        )[0]

        row = {row.label: row for row in decision.candidates}["ActionableAgent"]
        self.assertAlmostEqual(row.effective_score, row.gate_score * 1.125, places=5)

    def test_zero_regime_confidence_blocks_actionable_actor_at_default_gate(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActionableAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActionableAgent", Regime.BULLISH, 6, 1.0, start_id=1)
        market = replace(
            make_market(prices={"BTC": 100.0}),
            regime_confidence=0.0,
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(actionable_bonus=0.50),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
            actionable_labels={"ActionableAgent"},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        row = {row.label: row for row in decision.candidates}["ActionableAgent"]
        self.assertEqual(row.reason, "score_below_threshold")
        self.assertEqual(row.gate_score, 0.0)
        self.assertEqual(row.effective_score, 0.0)

    def test_shadow_confirmation_requires_canonical_symbol_action_key(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "Alpha", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=3,
            ),
        )
        legacy = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC"): {"score": 5.0, "closed_trades": 10},
            },
        )[0]
        canonical = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("Alpha", "BTC", "FUT_LONG_FULL"): {
                    "score": 5.0,
                    "closed_trades": 10,
                },
            },
        )[0]

        self.assertEqual(legacy.selected_actor, "NoTrade")
        self.assertEqual(
            {row.label: row.reason for row in legacy.candidates}["Alpha"],
            "shadow_unconfirmed",
        )
        self.assertEqual(canonical.selected_actor, "Alpha")

    def test_regime_confidence_scales_flash_gate_score(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("TransitionAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "TransitionAgent", Regime.BULLISH, 6, 1.0, start_id=1)
        market = replace(
            make_market(prices={"BTC": 100.0}),
            regime_confidence=0.5,
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_score_to_trade=1.0,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("TransitionAgent", "BTC", "FUT_LONG_FULL"): (1.5, 3),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        row = {row.label: row for row in decision.candidates}["TransitionAgent"]
        self.assertEqual(row.reason, "score_below_threshold")
        self.assertEqual(row.gate_score, 0.75)

    def test_shadow_signal_handoff_can_make_shadow_player_signal_eligible(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ShadowAgent", {"BTC": Action.HOLD})
        player = EnsemblePlayer(
            label="Solo_ShadowAgent",
            agents=[agent],
            weights={"ShadowAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "Solo_ShadowAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        shadow_signal = Signal(
            id=91,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="Solo_ShadowAgent",
            by_agent="ShadowAgent",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(shadow_signal_handoff_enabled=True),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[player],
            signal_id_start=1,
            shadow_player_signals={"Solo_ShadowAgent": (shadow_signal,)},
        )[0]

        self.assertEqual(decision.selected_actor, "Solo_ShadowAgent")
        self.assertEqual(decision.actor_type, "ensemble")
        self.assertEqual(decision.signal.by_player, "Solo_ShadowAgent")
        self.assertEqual(decision.signal.by_agent, "ShadowAgent")

    def test_shadow_signal_handoff_rejects_unknown_shadow_player_by_default(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        _add_perf(perf, "Solo_ShadowOnly", Regime.BULLISH, 10, 2.0, start_id=1)
        shadow_signal = Signal(
            id=91,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="Solo_ShadowOnly",
            by_agent="ShadowOnly",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(shadow_signal_handoff_enabled=True),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[],
            signal_id_start=1,
            shadow_player_signals={"Solo_ShadowOnly": (shadow_signal,)},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(tuple(row.label for row in decision.candidates), ("NoTrade",))
        self.assertFalse(decision.candidates[0].rejected)

    def test_shadow_signal_handoff_resets_shadow_signal_id_in_candidate(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ShadowAgent", {"BTC": Action.HOLD})
        player = EnsemblePlayer(
            label="Solo_ShadowAgent",
            agents=[agent],
            weights={"ShadowAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "Solo_ShadowAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        shadow_signal = Signal(
            id=91,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="Solo_ShadowAgent",
            by_agent="ShadowAgent",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(shadow_signal_handoff_enabled=True),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[player],
            signal_id_start=7,
            shadow_player_signals={"Solo_ShadowAgent": (shadow_signal,)},
        )[0]

        self.assertEqual(decision.selected_actor, "Solo_ShadowAgent")
        self.assertEqual(decision.signal.id, 7)

    def test_shadow_signal_handoff_is_disabled_by_default(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        _add_perf(perf, "Solo_ShadowAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        shadow_signal = Signal(
            id=91,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="Solo_ShadowAgent",
            by_agent="ShadowAgent",
        )

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[],
            signal_id_start=1,
            shadow_player_signals={"Solo_ShadowAgent": (shadow_signal,)},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(tuple(row.label for row in decision.candidates), ("NoTrade",))
        self.assertFalse(decision.candidates[0].rejected)

    def test_genetics_confirmation_overlay_boosts_matching_existing_actor_only(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        genetics_signal = Signal(
            id=0,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsRegimeRouter",),
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long],
            players=[],
            signal_id_start=1,
            shadow_agent_signals={"GeneticsRegimeRouter": [genetics_signal]},
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertNotIn("GeneticsRegimeRouter", rows)
        self.assertGreater(rows["BaselineShort"].score, rows["StrongerLong"].score)
        self.assertGreater(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)
        self.assertEqual(rows["StrongerLong"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_overlay_penalizes_opposing_existing_actor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active_long = FakeAgent("ActiveLong", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveLong", Regime.BULLISH, 5, 1.0, start_id=1)
        genetics_signal = Signal(
            id=0,
            bar=1,
            sym="BTC",
            action=Action.FUT_SHORT_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="GeneticsRegimeRouter",
            by_agent="GeneticsRegimeRouter",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsRegimeRouter",),
                genetics_confirmation_score_bonus=0.0,
                genetics_confirmation_score_penalty=100.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active_long],
            players=[],
            signal_id_start=1,
            shadow_agent_signals={"GeneticsRegimeRouter": [genetics_signal]},
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(rows["ActiveLong"].reason, "score_below_threshold")
        self.assertLess(rows["ActiveLong"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_overlay_uses_raw_genetics_output_without_direct_execution(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        genetics_short = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, 10.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, genetics_short],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertGreater(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_quality_gate_ignores_unproven_raw_source(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_short = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_quality_gate_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_min_closed_trades=3,
                genetics_confirmation_min_pnl_per_trade_pct=0.0,
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_short],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerLong")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertEqual(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_quality_gate_accepts_proven_raw_source(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_short = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 3, 1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_quality_gate_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_min_closed_trades=3,
                genetics_confirmation_min_pnl_per_trade_pct=0.0,
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_short],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertGreater(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_quality_gate_ignores_negative_raw_source(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_short = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 3, -1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_quality_gate_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_min_closed_trades=3,
                genetics_confirmation_min_pnl_per_trade_pct=0.0,
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_short],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerLong")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertEqual(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_signal_allowlist_rejects_unlisted_raw_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_short = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, 1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_allowed_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_short],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerLong")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertEqual(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_signal_allowlist_accepts_exact_raw_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_short = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, 1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_allowed_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_SHORT_FULL",
                ),
                genetics_confirmation_score_bonus=10.0,
                genetics_confirmation_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_short],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertGreater(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_contra_signal_penalizes_matching_raw_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_long = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, -1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_quality_gate_enabled=True,
                genetics_confirmation_min_closed_trades=10,
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_long],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertLess(rows["StrongerLong"].genetics_confirmation_adjustment, 0.0)
        self.assertEqual(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_contra_signal_requires_exact_raw_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_long = FakeAgent("StrongerLong", {"BTC": Action.FUT_LONG_FULL})
        genetics_long = FakeAgent("GeneticsNeutral", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, -1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_SHORT_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_long, genetics_long],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerLong")
        self.assertEqual(rows["StrongerLong"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_contra_side_match_penalizes_canonical_long_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_spot_long = FakeAgent("StrongerSpotLong", {"BTC": Action.SPOT_BUY_FULL})
        genetics_spot_long = FakeAgent("GeneticsNeutral", {"BTC": Action.SPOT_BUY_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerSpotLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, -1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_side_match_enabled=True,
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_spot_long, genetics_spot_long],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertEqual(rows["GeneticsNeutral"].reason, "genetics_confirmation_only")
        self.assertLess(rows["StrongerSpotLong"].genetics_confirmation_adjustment, 0.0)
        self.assertEqual(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_contra_side_match_is_opt_in(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_spot_long = FakeAgent("StrongerSpotLong", {"BTC": Action.SPOT_BUY_FULL})
        genetics_spot_long = FakeAgent("GeneticsNeutral", {"BTC": Action.SPOT_BUY_FULL})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerSpotLong", Regime.BULLISH, 5, 1.0, start_id=100)
        _add_perf(perf, "GeneticsNeutral", Regime.BULLISH, 5, -1.0, start_id=200)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_spot_long, genetics_spot_long],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerSpotLong")
        self.assertEqual(rows["StrongerSpotLong"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_static_contra_penalizes_side_without_raw_open(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_spot_long = FakeAgent("StrongerSpotLong", {"BTC": Action.SPOT_BUY_FULL})
        genetics_hold = FakeAgent("GeneticsNeutral", {"BTC": Action.HOLD})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerSpotLong", Regime.BULLISH, 5, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_static_enabled=True,
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_spot_long, genetics_hold],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "BaselineShort")
        self.assertLess(rows["StrongerSpotLong"].genetics_confirmation_adjustment, 0.0)
        self.assertEqual(rows["BaselineShort"].genetics_confirmation_adjustment, 0.0)

    def test_genetics_confirmation_contra_no_backfill_blocks_local_fallback(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_spot_long = FakeAgent("StrongerSpotLong", {"BTC": Action.SPOT_BUY_FULL})
        genetics_hold = FakeAgent("GeneticsNeutral", {"BTC": Action.HOLD})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerSpotLong", Regime.BULLISH, 5, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_static_enabled=True,
                genetics_confirmation_contra_no_backfill_enabled=True,
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_spot_long, genetics_hold],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(decision.reason, "genetics_contra_no_backfill")
        self.assertEqual(decision.original_selected_actor, "StrongerSpotLong")
        self.assertEqual(decision.signal, None)
        self.assertEqual(
            rows["StrongerSpotLong"].reason,
            "genetics_contra_no_backfill",
        )
        self.assertIn("genetics_contra_no_backfill", decision.selected_reasons)

    def test_genetics_confirmation_static_contra_can_reduce_open_risk_without_blocking(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_spot_long = FakeAgent("StrongerSpotLong", {"BTC": Action.SPOT_BUY_FULL})
        genetics_hold = FakeAgent("GeneticsNeutral", {"BTC": Action.HOLD})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerSpotLong", Regime.BULLISH, 5, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_static_enabled=True,
                genetics_confirmation_contra_risk_sizing_enabled=True,
                genetics_confirmation_contra_risk_mult=0.4,
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=0.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_spot_long, genetics_hold],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerSpotLong")
        self.assertEqual(decision.reason, "selected")
        self.assertIsNotNone(decision.signal)
        self.assertAlmostEqual(decision.signal.risk_mult, 0.4)
        self.assertAlmostEqual(rows["StrongerSpotLong"].risk_mult, 0.4)
        self.assertEqual(rows["StrongerSpotLong"].reason, "eligible")

    def test_genetics_confirmation_static_contra_is_opt_in(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        baseline_short = FakeAgent("BaselineShort", {"BTC": Action.FUT_SHORT_FULL})
        stronger_spot_long = FakeAgent("StrongerSpotLong", {"BTC": Action.SPOT_BUY_FULL})
        genetics_hold = FakeAgent("GeneticsNeutral", {"BTC": Action.HOLD})
        _add_perf(perf, "BaselineShort", Regime.BULLISH, 5, 0.5, start_id=1)
        _add_perf(perf, "StrongerSpotLong", Regime.BULLISH, 5, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                genetics_confirmation_overlay_enabled=True,
                genetics_confirmation_labels=("GeneticsNeutral",),
                genetics_confirmation_contra_signal_keys=(
                    "agent:GeneticsNeutral|BTC|FUT_LONG_FULL",
                ),
                genetics_confirmation_contra_score_penalty=10.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[baseline_short, stronger_spot_long, genetics_hold],
            players=[],
            signal_id_start=1,
        )[0]

        rows = {row.label: row for row in decision.candidates}
        self.assertEqual(decision.selected_actor, "StrongerSpotLong")
        self.assertEqual(rows["StrongerSpotLong"].genetics_confirmation_adjustment, 0.0)

    def test_no_trade_when_selected_signal_has_invalid_market_price(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 6, 1.0, start_id=1)

        allocator = FlashAllocator(perf=perf, qm=qm)
        decision = allocator.decide(
            make_market(prices={"BTC": 0.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        self.assertEqual(decision.reason, "missing_price")
        self.assertEqual(decision.original_selected_actor, "ActiveAgent")
        self.assertIsNone(decision.signal)

    def test_shadow_confirmation_rejects_unconfirmed_active_actor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 6, 0.5, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(shadow_confirmation_enabled=True),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={},
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["ActiveAgent"].reason, "shadow_unconfirmed")

    def test_shadow_confirmation_ranks_confirmed_actor_by_shadow_score(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        base_winner = FakeAgent("BaseWinner", {"BTC": Action.FUT_LONG_FULL})
        shadow_winner = FakeAgent("ShadowWinner", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "BaseWinner", Regime.BULLISH, 6, 1.0, start_id=1)
        _add_perf(perf, "ShadowWinner", Regime.BULLISH, 6, 0.4, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(shadow_confirmation_enabled=True),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[base_winner, shadow_winner],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "BaseWinner": (1.0, 50),
                "ShadowWinner": (5.0, 50),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "ShadowWinner")
        self.assertEqual(decision.score, 5.0)

    def test_min_score_gate_uses_effective_shadow_score_and_audits_scores(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        actor = FakeAgent("ShadowRecovered", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ShadowRecovered", Regime.BULLISH, 10, -2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_score_to_trade=1.0,
                min_pnl_pct_to_trade=-999.0,
                shadow_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[actor],
            players=[],
            signal_id_start=1,
            shadow_confirmation={"ShadowRecovered": (2.0, 5)},
        )[0]

        self.assertEqual(decision.selected_actor, "ShadowRecovered")
        self.assertEqual(decision.score, 2.0)
        row = decision.candidates[0]
        self.assertLess(row.base_score, 1.0)
        self.assertEqual(row.effective_score, 2.0)
        self.assertEqual(row.gate_score, 2.0)
        self.assertEqual(row.as_dict()["base_score"], row.base_score)

    def test_symbol_shadow_confirmation_rejects_unconfirmed_symbol_for_confirmed_actor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent(
            "ActiveAgent",
            {
                "BTC": Action.FUT_LONG_FULL,
                "ETH": Action.FUT_LONG_FULL,
            },
        )
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=5,
            ),
        )
        decisions = allocator.decide(
            make_market(prices={"BTC": 100.0, "ETH": 50.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "ActiveAgent": (10.0, 50),
                ("ActiveAgent", "BTC", "FUT_LONG_FULL"): (2.0, 10),
                ("ActiveAgent", "ETH", "FUT_LONG_FULL"): (-1.0, 10),
            },
        )
        by_symbol = {decision.symbol: decision for decision in decisions}

        self.assertEqual(by_symbol["BTC"].selected_actor, "ActiveAgent")
        self.assertEqual(by_symbol["BTC"].score, 2.0)
        self.assertEqual(by_symbol["ETH"].selected_actor, "NoTrade")
        rejected = {row.label: row for row in by_symbol["ETH"].candidates if row.rejected}
        self.assertEqual(
            rejected["ActiveAgent"].reason,
            "shadow_score_below_threshold",
        )

    def test_symbol_shadow_confirmation_can_fallback_to_actor_confirmation(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "ActiveAgent": (3.0, 80),
                ("ActiveAgent", "BTC", "FUT_LONG_FULL"): (0.5, 3),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "ActiveAgent")
        self.assertEqual(decision.score, 3.0)
        self.assertEqual(decision.candidates[0].shadow_source, "actor_fallback")

    def test_symbol_shadow_confirmation_does_not_fallback_over_confirmed_bad_symbol(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "ActiveAgent": (3.0, 80),
                ("ActiveAgent", "BTC", "FUT_LONG_FULL"): (-1.0, 80),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["ActiveAgent"].reason, "shadow_score_below_threshold")
        self.assertEqual(rejected["ActiveAgent"].shadow_source, "symbol")

    def test_portfolio_actor_uses_actor_shadow_over_bad_symbol_shadow(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        portfolio = FakeAgent("PortfolioAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "PortfolioAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=50,
                portfolio_actor_keys=("agent:PortfolioAgent",),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[portfolio],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "PortfolioAgent": (3.0, 80),
                ("PortfolioAgent", "BTC", "FUT_LONG_FULL"): (-1.0, 80),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "PortfolioAgent")
        self.assertEqual(decision.score, 3.0)
        self.assertEqual(decision.candidates[0].shadow_source, "portfolio_actor")

    def test_portfolio_actor_base_fallback_can_override_bad_actor_shadow(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        portfolio = FakeAgent("PortfolioAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "PortfolioAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_base_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=1.0,
                shadow_base_fallback_actor_keys=("agent:PortfolioAgent",),
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=50,
                portfolio_actor_keys=("agent:PortfolioAgent",),
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[portfolio],
            players=[],
            signal_id_start=1,
            shadow_confirmation={"PortfolioAgent": (-1.0, 80)},
        )[0]

        self.assertEqual(decision.selected_actor, "PortfolioAgent")
        self.assertGreater(decision.score, 1.0)
        self.assertEqual(decision.candidates[0].shadow_source, "portfolio_base_fallback")

    def test_symbol_shadow_confirmation_does_not_fallback_over_bad_underpowered_symbol(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "ActiveAgent": (3.0, 80),
                ("ActiveAgent", "BTC", "FUT_LONG_FULL"): (-0.5, 3),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["ActiveAgent"].reason, "shadow_unconfirmed")
        self.assertEqual(rejected["ActiveAgent"].shadow_source, "symbol")

    def test_actor_fallback_can_use_base_score_floor_for_underpowered_symbol(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=1.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "StrongActor": (4.0, 80),
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "StrongActor")
        self.assertEqual(decision.score, 4.0)
        self.assertEqual(decision.candidates[0].shadow_source, "actor_fallback")

    def test_actor_fallback_does_not_depend_on_empty_symbol_shadow_score(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=0.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "StrongActor": (4.0, 80),
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "StrongActor")
        self.assertEqual(decision.candidates[0].shadow_source, "actor_fallback")

    def test_actor_fallback_base_score_floor_rejects_weak_local_candidate(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("WeakActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "WeakActor", Regime.BULLISH, 10, 0.2, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=100.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "WeakActor": (4.0, 80),
                ("WeakActor", "BTC", "FUT_LONG_FULL"): (0.5, 3),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["WeakActor"].reason, "shadow_unconfirmed")
        self.assertEqual(rejected["WeakActor"].shadow_source, "symbol")

    def test_replay_actor_fallback_floor_rejects_underpowered_replay_signal(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ReplayAgent", {"BTC": Action.HOLD})
        player = EnsemblePlayer(
            label="ReplayActor",
            agents=[agent],
            weights={"ReplayAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "ReplayActor", Regime.BULLISH, 10, 4.0, start_id=1)
        replay_signal = Signal(
            id=91,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="ReplayActor",
            by_agent="ShadowPositionReplay",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_signal_handoff_enabled=True,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=0.0,
                shadow_position_replay_actor_fallback_min_base_score=100.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[player],
            signal_id_start=1,
            shadow_player_signals={"ReplayActor": (replay_signal,)},
            shadow_confirmation={
                "ReplayActor": (4.0, 80),
                ("ReplayActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        replay_row = next(
            row
            for row in decision.candidates
            if row.label == "ReplayActor" and row.action == Action.FUT_LONG_FULL
        )
        self.assertEqual(replay_row.reason, "shadow_unconfirmed")
        self.assertEqual(replay_row.shadow_source, "symbol")

    def test_replay_actor_fallback_floor_does_not_block_regular_actor_signal(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("RegularActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "RegularActor", Regime.BULLISH, 10, 4.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=0.0,
                shadow_position_replay_actor_fallback_min_base_score=100.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                "RegularActor": (4.0, 80),
                ("RegularActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "RegularActor")
        self.assertEqual(decision.candidates[0].shadow_source, "actor_fallback")

    def test_replay_actor_fallback_shadow_floor_rejects_weak_actor_shadow(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ReplayAgent", {"BTC": Action.HOLD})
        player = EnsemblePlayer(
            label="ReplayActor",
            agents=[agent],
            weights={"ReplayAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "ReplayActor", Regime.BULLISH, 10, 8.0, start_id=1)
        replay_signal = Signal(
            id=92,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="ReplayActor",
            by_agent="ShadowPositionReplay",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_signal_handoff_enabled=True,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=0.0,
                shadow_position_replay_actor_fallback_min_shadow_score=5.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[player],
            signal_id_start=1,
            shadow_player_signals={"ReplayActor": (replay_signal,)},
            shadow_confirmation={
                "ReplayActor": (4.0, 80),
                ("ReplayActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        replay_row = next(
            row
            for row in decision.candidates
            if row.label == "ReplayActor" and row.action == Action.FUT_LONG_FULL
        )
        self.assertEqual(replay_row.reason, "shadow_unconfirmed")
        self.assertEqual(replay_row.shadow_source, "symbol")

    def test_replay_actor_fallback_shadow_floor_allows_strong_actor_shadow(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        agent = FakeAgent("ReplayAgent", {"BTC": Action.HOLD})
        player = EnsemblePlayer(
            label="ReplayActor",
            agents=[agent],
            weights={"ReplayAgent": 1.0},
            voting=WeightedConsensus(),
            thresholds=ThresholdProfile(),
        )
        _add_perf(perf, "ReplayActor", Regime.BULLISH, 10, 8.0, start_id=1)
        replay_signal = Signal(
            id=93,
            bar=1,
            sym="BTC",
            action=Action.FUT_LONG_FULL,
            price=100.0,
            regime=Regime.BULLISH,
            by_player="ReplayActor",
            by_agent="ShadowPositionReplay",
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_signal_handoff_enabled=True,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=0.0,
                shadow_position_replay_actor_fallback_min_shadow_score=5.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[],
            players=[player],
            signal_id_start=1,
            shadow_player_signals={"ReplayActor": (replay_signal,)},
            shadow_confirmation={
                "ReplayActor": (6.0, 80),
                ("ReplayActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "ReplayActor")
        self.assertEqual(decision.candidates[0].shadow_source, "actor_fallback")

    def test_base_score_floor_can_confirm_without_actor_shadow_sample(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_base_fallback_confirmation_enabled=True,
                shadow_base_fallback_actor_keys=("agent:StrongActor",),
                shadow_actor_fallback_min_base_score=1.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "StrongActor")
        self.assertGreaterEqual(decision.score, 1.0)
        self.assertEqual(decision.candidates[0].shadow_source, "base_fallback")

    def test_base_score_floor_can_confirm_without_actor_fallback_enabled(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_base_fallback_confirmation_enabled=True,
                shadow_base_fallback_actor_keys=("agent:StrongActor",),
                shadow_actor_fallback_min_base_score=1.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "StrongActor")
        self.assertEqual(decision.candidates[0].shadow_source, "base_fallback")

    def test_base_score_floor_requires_explicit_actor_allowlist(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_base_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=1.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["StrongActor"].reason, "shadow_unconfirmed")
        self.assertEqual(rejected["StrongActor"].shadow_source, "symbol")

    def test_base_score_floor_does_not_confirm_without_base_fallback_flag(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_actor_fallback_min_base_score=1.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(rejected["StrongActor"].reason, "shadow_unconfirmed")
        self.assertEqual(rejected["StrongActor"].shadow_source, "symbol")

    def test_shadow_confirmation_rejects_low_win_rate_symbol_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=2,
                shadow_confirmation_min_win_rate_pct=60.0,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("ActiveAgent", "BTC", "FUT_LONG_FULL"): {
                    "score": 2.0,
                    "closed_trades": 4,
                    "winning_trades": 2,
                },
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["ActiveAgent"].reason,
            "shadow_win_rate_below_threshold",
        )
        self.assertEqual(rejected["ActiveAgent"].shadow_win_rate_pct, 50.0)

    def test_shadow_quality_confirmation_does_not_block_close_action(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        closer = FakeAgent("CloserAgent", {"BTC": Action.FUT_CLOSE_ALL})
        _add_perf(perf, "CloserAgent", Regime.BEARISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_quality_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=1,
                shadow_confirmation_min_win_rate_pct=60.0,
            ),
        )
        decision = allocator.decide(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0}),
            agents=[closer],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("CloserAgent", "BTC", "FUT_CLOSE_ALL"): (2.0, 4),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "CloserAgent")
        self.assertEqual(decision.action, Action.FUT_CLOSE_ALL)

    def test_shadow_reliability_gate_rejects_full_open_with_single_confirmation(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=1,
                shadow_confirmation_min_full_open_closed_trades=2,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("ActiveAgent", "BTC", "FUT_LONG_FULL"): (2.0, 1),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["ActiveAgent"].reason,
            "shadow_full_open_unconfirmed",
        )

    def test_shadow_reliability_gate_accepts_base_confirmed_full_open(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("StrongActor", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "StrongActor", Regime.BULLISH, 10, 8.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_actor_fallback_confirmation_enabled=True,
                shadow_base_fallback_confirmation_enabled=True,
                shadow_base_fallback_actor_keys=("agent:StrongActor",),
                shadow_actor_fallback_min_base_score=1.0,
                shadow_confirmation_min_score=0.0,
                shadow_confirmation_min_closed_trades=50,
                shadow_confirmation_min_full_open_closed_trades=50,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("StrongActor", "BTC", "FUT_LONG_FULL"): (0.0, 0),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "StrongActor")
        self.assertEqual(decision.candidates[0].shadow_source, "base_fallback")

    def test_shadow_reliability_gate_allows_half_open_with_single_confirmation(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_LONG_HALF})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=1,
                shadow_confirmation_min_full_open_closed_trades=2,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[active],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("ActiveAgent", "BTC", "FUT_LONG_HALF"): (2.0, 1),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "ActiveAgent")
        self.assertEqual(decision.action, Action.FUT_LONG_HALF)

    def test_actor_signal_cap_prevents_one_actor_symbol_fanout(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        fanout = FakeAgent(
            "FanoutAgent",
            {
                "BTC": Action.FUT_LONG_FULL,
                "ETH": Action.FUT_LONG_FULL,
            },
        )
        _add_perf(perf, "FanoutAgent", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(max_signals_per_actor=1),
        )
        decisions = allocator.decide(
            make_market(prices={"BTC": 100.0, "ETH": 50.0}),
            agents=[fanout],
            players=[],
            signal_id_start=1,
        )

        selected = [row for row in decisions if row.selected_actor == "FanoutAgent"]
        capped = [row for row in decisions if row.selected_actor == "NoTrade"]

        self.assertEqual(len(selected), 1)
        self.assertEqual(len(capped), 1)
        self.assertEqual(capped[0].reason, "actor_signal_cap")
        self.assertEqual(capped[0].original_selected_actor, "FanoutAgent")

    def test_config_allows_short_degradation_window_when_recovery_disabled(self):
        cfg = FlashAllocatorConfig(
            degradation_window_closed_trades=1,
            degradation_min_closed_trades=1,
            degradation_recovery_enabled=False,
        )

        self.assertEqual(cfg.degradation_window_closed_trades, 1)
        self.assertEqual(cfg.degradation_recovery_min_closed_trades, 3)

    def test_actor_signal_cap_keeps_highest_score_symbol_for_actor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        fanout = FakeAgent(
            "ScoreFanout",
            {
                "AVAX": Action.FUT_LONG_FULL,
                "BTC": Action.FUT_LONG_FULL,
            },
        )
        _add_perf(perf, "ScoreFanout", Regime.BULLISH, 10, 2.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                max_signals_per_actor=1,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decisions = allocator.decide(
            make_market(prices={"AVAX": 20.0, "BTC": 100.0}),
            agents=[fanout],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("ScoreFanout", "AVAX", "FUT_LONG_FULL"): (0.5, 5),
                ("ScoreFanout", "BTC", "FUT_LONG_FULL"): (5.0, 5),
            },
        )

        by_symbol = {decision.symbol: decision for decision in decisions}
        self.assertEqual(by_symbol["BTC"].selected_actor, "ScoreFanout")
        self.assertEqual(by_symbol["AVAX"].selected_actor, "NoTrade")
        self.assertEqual(by_symbol["AVAX"].reason, "actor_signal_cap")
        self.assertEqual(by_symbol["AVAX"].original_selected_actor, "ScoreFanout")

    def test_actor_switch_margin_keeps_previous_actor_until_new_actor_clears_margin(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        previous = FakeAgent("PreviousAgent", {"BTC": Action.FUT_LONG_FULL})
        challenger = FakeAgent("ChallengerAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "PreviousAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "ChallengerAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                actor_switch_margin=0.5,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[previous, challenger],
            players=[],
            signal_id_start=1,
            previous_actor_by_symbol={"BTC": "agent:PreviousAgent"},
            shadow_confirmation={
                ("PreviousAgent", "BTC", "FUT_LONG_FULL"): (2.0, 10),
                ("ChallengerAgent", "BTC", "FUT_SHORT_FULL"): (2.25, 10),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "PreviousAgent")
        self.assertEqual(decision.action, Action.FUT_LONG_FULL)
        self.assertEqual(decision.reason, "switch_margin_hold")
        self.assertEqual(decision.original_selected_actor, "ChallengerAgent")

    def test_actor_switch_margin_allows_new_actor_when_margin_is_large_enough(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        previous = FakeAgent("PreviousAgent", {"BTC": Action.FUT_LONG_FULL})
        challenger = FakeAgent("ChallengerAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "PreviousAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "ChallengerAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                actor_switch_margin=0.5,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[previous, challenger],
            players=[],
            signal_id_start=1,
            previous_actor_by_symbol={"BTC": "agent:PreviousAgent"},
            shadow_confirmation={
                ("PreviousAgent", "BTC", "FUT_LONG_FULL"): (2.0, 10),
                ("ChallengerAgent", "BTC", "FUT_SHORT_FULL"): (2.75, 10),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "ChallengerAgent")
        self.assertEqual(decision.reason, "selected")
        self.assertEqual(decision.original_selected_actor, "ChallengerAgent")

    def test_anchor_dominance_keeps_anchor_until_challenger_clears_margin(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        anchor = FakeAgent("AnchorAgent", {"BTC": Action.FUT_LONG_FULL})
        challenger = FakeAgent("ChallengerAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "AnchorAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "ChallengerAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                anchor_actor_keys=("agent:AnchorAgent",),
                anchor_min_score_advantage=0.5,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[anchor, challenger],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("AnchorAgent", "BTC", "FUT_LONG_FULL"): (2.0, 10),
                ("ChallengerAgent", "BTC", "FUT_SHORT_FULL"): (2.25, 10),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "AnchorAgent")
        self.assertEqual(decision.action, Action.FUT_LONG_FULL)
        self.assertEqual(decision.reason, "anchor_dominance_hold")
        self.assertEqual(decision.original_selected_actor, "ChallengerAgent")

    def test_anchor_dominance_allows_challenger_when_margin_is_large_enough(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        anchor = FakeAgent("AnchorAgent", {"BTC": Action.FUT_LONG_FULL})
        challenger = FakeAgent("ChallengerAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "AnchorAgent", Regime.BULLISH, 10, 1.0, start_id=1)
        _add_perf(perf, "ChallengerAgent", Regime.BULLISH, 10, 1.0, start_id=100)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                anchor_actor_keys=("agent:AnchorAgent",),
                anchor_min_score_advantage=0.5,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[anchor, challenger],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("AnchorAgent", "BTC", "FUT_LONG_FULL"): (2.0, 10),
                ("ChallengerAgent", "BTC", "FUT_SHORT_FULL"): (2.75, 10),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "ChallengerAgent")
        self.assertEqual(decision.reason, "selected")

    def test_anchor_min_score_to_trade_can_use_lower_gate_only_for_anchor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        anchor = FakeAgent("AnchorAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "AnchorAgent", Regime.BULLISH, 10, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                min_score_to_trade=4.0,
                anchor_actor_keys=("agent:AnchorAgent",),
                anchor_min_score_to_trade=2.0,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[anchor],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("AnchorAgent", "BTC", "FUT_LONG_FULL"): (2.5, 10),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "AnchorAgent")
        self.assertEqual(decision.reason, "selected")

    def test_anchor_shadow_min_score_can_use_lower_shadow_floor_only_for_anchor(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        anchor = FakeAgent("AnchorAgent", {"BTC": Action.FUT_LONG_FULL})
        _add_perf(perf, "AnchorAgent", Regime.BULLISH, 10, 1.0, start_id=1)

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                anchor_actor_keys=("agent:AnchorAgent",),
                anchor_shadow_min_score=0.0,
                shadow_confirmation_enabled=True,
                shadow_symbol_confirmation_enabled=True,
                shadow_confirmation_min_score=0.1,
                shadow_confirmation_min_closed_trades=1,
            ),
        )
        decision = allocator.decide(
            make_market(prices={"BTC": 100.0}),
            agents=[anchor],
            players=[],
            signal_id_start=1,
            shadow_confirmation={
                ("AnchorAgent", "BTC", "FUT_LONG_FULL"): (0.05, 10),
            },
        )[0]

        self.assertEqual(decision.selected_actor, "AnchorAgent")
        self.assertEqual(decision.reason, "selected")

    def test_overextension_guard_rejects_short_after_large_downmove(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BEARISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0}),
            lookback_returns_pct={"BTC": {12: -9.5}},
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                open_overextension_guard_enabled=True,
                overextension_lookback_bars=12,
                short_overextension_return_floor_pct=-8.0,
            ),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["ActiveAgent"].reason,
            "short_overextended_downmove",
        )
        self.assertEqual(rejected["ActiveAgent"].lookback_return_pct, -9.5)

    def test_overextension_guard_rejects_long_after_large_upmove(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.SPOT_BUY_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
            lookback_returns_pct={"BTC": {12: 9.25}},
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                open_overextension_guard_enabled=True,
                overextension_lookback_bars=12,
                long_overextension_return_ceiling_pct=8.0,
            ),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["ActiveAgent"].reason,
            "long_overextended_upmove",
        )

    def test_volatility_normalized_overextension_allows_volatile_downmove(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BEARISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0}),
            lookback_returns_pct={"BTC": {12: -9.5}},
            lookback_volatility_pct={"BTC": {12: 5.0}},
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                open_overextension_guard_enabled=True,
                overextension_volatility_normalized_enabled=True,
                overextension_lookback_bars=12,
                short_overextension_return_floor_pct=-8.0,
                short_overextension_z_floor=-2.5,
            ),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "ActiveAgent")
        candidate = decision.candidates[0]
        self.assertFalse(candidate.rejected)
        self.assertEqual(candidate.lookback_return_pct, -9.5)
        self.assertEqual(candidate.lookback_volatility_pct, 5.0)
        self.assertAlmostEqual(candidate.lookback_return_z, -1.9)

    def test_volatility_normalized_overextension_rejects_low_volatility_move(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed=set())
        active = FakeAgent("ActiveAgent", {"BTC": Action.FUT_SHORT_FULL})
        _add_perf(perf, "ActiveAgent", Regime.BEARISH, 10, 2.0, start_id=1)
        market = replace(
            make_market(regime=Regime.BEARISH, prices={"BTC": 100.0}),
            lookback_returns_pct={"BTC": {12: -3.0}},
            lookback_volatility_pct={"BTC": {12: 1.0}},
        )

        allocator = FlashAllocator(
            perf=perf,
            qm=qm,
            config=FlashAllocatorConfig(
                open_overextension_guard_enabled=True,
                overextension_volatility_normalized_enabled=True,
                overextension_lookback_bars=12,
                short_overextension_return_floor_pct=-8.0,
                short_overextension_z_floor=-2.5,
            ),
        )
        decision = allocator.decide(
            market,
            agents=[active],
            players=[],
            signal_id_start=1,
        )[0]

        self.assertEqual(decision.selected_actor, "NoTrade")
        rejected = {row.label: row for row in decision.candidates if row.rejected}
        self.assertEqual(
            rejected["ActiveAgent"].reason,
            "short_overextended_downmove",
        )
        self.assertEqual(rejected["ActiveAgent"].lookback_volatility_pct, 1.0)
        self.assertEqual(rejected["ActiveAgent"].lookback_return_z, -3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
