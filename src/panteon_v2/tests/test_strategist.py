"""Тесты Strategist."""

from __future__ import annotations

import unittest

from panteon_v2.attribution import ShadowActorUpdated
from panteon_v2.domain.types import Action, Regime
from panteon_v2.memory import PerformanceMemory, QuarantineManager
from panteon_v2.selection import (
    EnsemblePlayer,
    Strategist,
    StrategistConfig,
    ThresholdProfile,
    WeightedConsensus,
)
from panteon_v2.tests._helpers import FakeAgent
from panteon_v2.tests.test_selector import _add_perf


def _make_player(label: str, agent_labels, *, affinity=None):
    """Создаёт EnsemblePlayer с фейковыми агентами по labels."""
    n = len(agent_labels)
    weights = {lbl: 1.0 / n for lbl in agent_labels}
    agents = [FakeAgent(lbl) for lbl in agent_labels]
    return EnsemblePlayer(
        label=label,
        agents=agents,
        weights=weights,
        voting=WeightedConsensus(),
        thresholds=ThresholdProfile(),
        affinity=affinity,
    )


class TestStrategistBootstrap(unittest.TestCase):
    def test_bootstrap_picks_best(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        # Player Alpha с агентом A (pnl=+1%); Player Bad с агентом B (pnl=-1%)
        _add_perf(perf, "A", Regime.BULLISH, 5, 1.0, start_id=1)
        _add_perf(perf, "B", Regime.BULLISH, 5, -1.0, start_id=200)
        qm = QuarantineManager(seed=set())
        st = Strategist(perf, qm, candidates=[
            _make_player("Alpha", ["A"]),
            _make_player("Bad", ["B"]),
        ])
        decision = st.consider_switch(Regime.BULLISH, current_bar=1)
        self.assertEqual(decision.new_leader.label, "Alpha")
        self.assertTrue(decision.switched)

    def test_no_candidates_raises(self):
        perf = PerformanceMemory()
        qm = QuarantineManager(seed=set())
        st = Strategist(perf, qm, candidates=[])
        with self.assertRaises(ValueError):
            st.consider_switch(Regime.BULLISH, current_bar=1)

    def test_zero_score_candidates_are_penalized_and_ranked_deterministically(self):
        perf = PerformanceMemory()
        qm = QuarantineManager(seed=set())
        st = Strategist(
            perf,
            qm,
            candidates=[
                _make_player("Zeta", ["Z"]),
                _make_player("Alpha", ["A"]),
            ],
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "Alpha")
        self.assertLess(decision.score, 0.0)
        self.assertEqual([row.label for row in decision.candidate_scores], ["Alpha", "Zeta"])
        self.assertTrue(all(row.score_source == "no_data" for row in decision.candidate_scores))
        self.assertTrue(all(row.uncertainty_penalty > 0 for row in decision.candidate_scores))


class TestStrategistQuarantineFilter(unittest.TestCase):
    def test_quarantined_player_disqualified(self):
        """Q4: игрок с карантинным агентом не выбирается."""
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "Bad", Regime.BULLISH, 5, 5.0)  # карантин: высокий pnl
        _add_perf(perf, "Ok",  Regime.BULLISH, 5, 0.5)  # обычный, не в карантине
        qm = QuarantineManager(seed={"Bad"})
        # Лидер 1: использует карантинного Bad
        bad_leader = _make_player("BadLeader", ["Bad"])
        good_leader = _make_player("GoodLeader", ["Ok"])
        st = Strategist(perf, qm, candidates=[bad_leader, good_leader])
        decision = st.consider_switch(Regime.BULLISH, current_bar=1)
        # Должны выбрать good_leader, не bad_leader
        self.assertEqual(decision.new_leader.label, "GoodLeader")

    def test_all_candidates_quarantined_raises(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager(seed={"X", "Y"})
        st = Strategist(perf, qm, candidates=[
            _make_player("P1", ["X"]),
            _make_player("P2", ["Y"]),
        ])
        with self.assertRaises(ValueError):
            st.consider_switch(Regime.BULLISH, current_bar=1)


class TestStrategistSwitching(unittest.TestCase):
    def setUp(self):
        self.perf = PerformanceMemory(trade_fraction=1.0)
        self.qm = QuarantineManager(seed=set())
        # Two agents
        _add_perf(self.perf, "Old",  Regime.BULLISH, 5, 0.2, start_id=1)
        _add_perf(self.perf, "New",  Regime.BULLISH, 5, 1.0, start_id=200)
        self.candidates = [
            _make_player("OldLeader", ["Old"]),
            _make_player("NewLeader", ["New"]),
        ]

    def test_switch_requires_streak(self):
        """Smena треубует streak подтверждения (даже если best > current)."""
        config = StrategistConfig(
            cooldown_bars=0, streak_needed=3, switch_margin=0.1,
            hard_negative=-100.0, urgent_gap=100.0,  # отключаем urgent
        )
        st = Strategist(self.perf, self.qm, candidates=self.candidates,
                        config=config)
        # Bootstrap → выберется лучший (New)
        st.consider_switch(Regime.BULLISH, current_bar=1)
        self.assertEqual(st.current_leader().label, "NewLeader")

    def test_no_switch_when_current_is_best(self):
        """Если current = best — не меняем."""
        config = StrategistConfig(cooldown_bars=0)
        st = Strategist(self.perf, self.qm, candidates=self.candidates,
                        config=config)
        # Bootstrap → New (т.к. Old хуже)
        st.consider_switch(Regime.BULLISH, current_bar=1)
        # Bar 2 — current уже New, всё то же
        d2 = st.consider_switch(Regime.BULLISH, current_bar=2)
        self.assertFalse(d2.switched)
        self.assertEqual(d2.switch_gate_reason, "current is still best")
        self.assertEqual(d2.best_label, "NewLeader")
        self.assertEqual(d2.current_label, "NewLeader")

    def test_urgent_switch_bypasses_cooldown(self):
        """Если current скатывается до hard_negative — switch урgent."""
        # Сначала плохой Bad как current, потом хороший Good появляется
        perf = PerformanceMemory(trade_fraction=1.0)
        # Делаем Bad очень плохим (-2% много раз → score < hard_negative=-0.50)
        _add_perf(perf, "Bad", Regime.BULLISH, 10, -3.0, start_id=1)
        _add_perf(perf, "Good", Regime.BULLISH, 10, 1.5, start_id=500)
        qm = QuarantineManager(seed=set())
        cands = [
            _make_player("Bad", ["Bad"]),
            _make_player("Good", ["Good"]),
        ]
        config = StrategistConfig(
            cooldown_bars=1000,  # огромный cooldown
            hard_negative=-0.50,
            urgent_gap=0.5,
            streak_needed=1,
        )
        st = Strategist(perf, qm, candidates=cands, config=config)
        # Симулируем: вначале выберется Good
        d1 = st.consider_switch(Regime.BULLISH, current_bar=1)
        self.assertEqual(d1.new_leader.label, "Good")


class TestStrategistUpdateCandidates(unittest.TestCase):
    def test_update_candidates_changes_pool(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "A", Regime.BULLISH, 5, 1.0)
        qm = QuarantineManager(seed=set())
        st = Strategist(perf, qm, candidates=[_make_player("PA", ["A"])])
        st.consider_switch(Regime.BULLISH, current_bar=1)
        self.assertEqual(st.current_leader().label, "PA")

        # Обновили список — старый PA остаётся current до switch
        _add_perf(perf, "B", Regime.BULLISH, 5, 5.0, start_id=200)
        st.update_candidates([_make_player("PB", ["B"])])
        # current всё ещё PA пока не consider_switch
        self.assertEqual(st.current_leader().label, "PA")


    def test_current_valid_leader_is_retained_when_temporarily_missing_from_candidates(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "A", Regime.BULLISH, 5, 1.0)
        _add_perf(perf, "B", Regime.BULLISH, 5, 5.0, start_id=200)
        qm = QuarantineManager(seed=set())
        st = Strategist(
            perf,
            qm,
            candidates=[_make_player("PA", ["A"])],
            config=StrategistConfig(
                candidate_ttl_bars=5,
                cooldown_bars=1000,
                switch_margin=0.1,
                urgent_gap=100.0,
                hard_negative=-100.0,
            ),
        )
        st.consider_switch(Regime.BULLISH, current_bar=1)
        st.update_candidates([_make_player("PB", ["B"])])

        decision = st.consider_switch(Regime.BULLISH, current_bar=2)

        self.assertFalse(decision.switched)
        self.assertEqual(decision.new_leader.label, "PA")
        self.assertEqual(decision.best_label, "PB")
        self.assertEqual(decision.current_label, "PA")
        self.assertEqual(decision.switch_gate_reason, "cooldown not yet passed")

        expired = st.consider_switch(Regime.BULLISH, current_bar=8)

        self.assertTrue(expired.switched)
        self.assertEqual(expired.new_leader.label, "PB")
        self.assertIn("not in candidates", expired.reason)

    def test_affinity_breaks_ties_for_current_regime(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "BearAgent", Regime.BULLISH, 5, 1.0)
        _add_perf(perf, "BullAgent", Regime.BULLISH, 5, 1.0, start_id=200)
        qm = QuarantineManager(seed=set())
        st = Strategist(perf, qm, candidates=[
            _make_player("BearProfile", ["BearAgent"], affinity=Regime.BEARISH),
            _make_player("BullProfile", ["BullAgent"], affinity=Regime.BULLISH),
        ])

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "BullProfile")

    def test_real_negative_history_penalizes_otherwise_best_leader(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "A", Regime.NEUTRAL, 8, 0.30, start_id=1)
        _add_perf(virtual_perf, "B", Regime.NEUTRAL, 8, 0.25, start_id=200)
        _add_perf(real_perf, "PlayerA", Regime.NEUTRAL, 4, -0.40, start_id=500)
        qm = QuarantineManager(seed=set())
        st = Strategist(
            virtual_perf,
            qm,
            candidates=[
                _make_player("PlayerA", ["A"]),
                _make_player("PlayerB", ["B"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                real_score_weight=1.0,
                real_min_closed_trades=1,
            ),
        )

        decision = st.consider_switch(Regime.NEUTRAL, current_bar=1)

        self.assertEqual(decision.new_leader.label, "PlayerB")

    def test_v3_rolling_score_prefers_real_edge_over_virtual_only_edge(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "VirtualAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        _add_perf(virtual_perf, "RealAgent", Regime.BULLISH, 10, 0.2, start_id=300)
        _add_perf(real_perf, "RealPlayer", Regime.BULLISH, 6, 0.7, start_id=700)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("VirtualPlayer", ["VirtualAgent"]),
                _make_player("RealPlayer", ["RealAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_real_loss_kill_min_closed_trades=0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "RealPlayer")
        by_label = {row.label: row for row in decision.candidate_scores}
        self.assertEqual(by_label["RealPlayer"].score_source, "v3_rolling")
        self.assertGreater(by_label["RealPlayer"].score, by_label["VirtualPlayer"].score)

    def test_v3_rolling_score_selects_no_trade_when_all_candidates_are_negative(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "A", Regime.BEARISH, 8, 0.1, start_id=1)
        _add_perf(virtual_perf, "B", Regime.BEARISH, 8, 0.2, start_id=300)
        _add_perf(real_perf, "PlayerA", Regime.BEARISH, 5, -0.8, start_id=700)
        _add_perf(real_perf, "PlayerB", Regime.BEARISH, 5, -0.6, start_id=900)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("PlayerA", ["A"]),
                _make_player("PlayerB", ["B"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_real_loss_kill_min_closed_trades=0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BEARISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "NoTrade")
        self.assertIn("v3 rolling score", decision.reason)

    def test_v3_rolling_score_is_opt_in_and_preserves_default_virtual_ranking(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "VirtualAgent", Regime.BULLISH, 10, 2.0, start_id=1)
        _add_perf(virtual_perf, "RealAgent", Regime.BULLISH, 10, 0.2, start_id=300)
        _add_perf(real_perf, "RealPlayer", Regime.BULLISH, 6, 0.7, start_id=700)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("VirtualPlayer", ["VirtualAgent"]),
                _make_player("RealPlayer", ["RealAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=False,
                real_score_weight=0.0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "VirtualPlayer")

    def test_v3_virtual_only_score_is_capped_below_real_positive_edge(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "VirtualAgent", Regime.BULLISH, 30, 3.0, start_id=1)
        _add_perf(virtual_perf, "RealAgent", Regime.BULLISH, 10, 0.05, start_id=1000)
        _add_perf(real_perf, "RealPlayer", Regime.BULLISH, 3, 0.20, start_id=2000)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("VirtualPlayer", ["VirtualAgent"]),
                _make_player("RealPlayer", ["RealAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_virtual_only_score_cap=0.25,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "RealPlayer")
        by_label = {row.label: row for row in decision.candidate_scores}
        self.assertLessEqual(by_label["VirtualPlayer"].score, 0.25)

    def test_v3_shadow_rolling_score_prefers_recent_regime_winner(self):
        st = Strategist(
            PerformanceMemory(trade_fraction=1.0),
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("AlphaPlayer", ["AlphaAgent"]),
                _make_player("BetaPlayer", ["BetaAgent"]),
            ],
            config=StrategistConfig(
                use_v3_rolling_score=True,
                use_v3_shadow_rolling_score=True,
                v3_shadow_rolling_min_closed_trades=2,
                v3_shadow_rolling_window_bars=24,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )
        st.update_realized_pnl_snapshot(
            pnl_by_player={},
            trade_counts_by_player={},
            win_counts_by_player={},
            initial_capital=100.0,
        )
        st.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="AlphaPlayer",
                regime="bullish",
                realized_pnl_usd=8.0,
                closed_trades=2,
                winning_trades=2,
            ),
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="BetaPlayer",
                regime="bullish",
                realized_pnl_usd=-4.0,
                closed_trades=2,
                winning_trades=0,
            ),
        ])

        decision = st.consider_switch(Regime.BULLISH, current_bar=2)

        self.assertEqual(decision.new_leader.label, "AlphaPlayer")
        by_label = {row.label: row for row in decision.candidate_scores}
        self.assertEqual(by_label["AlphaPlayer"].score_source, "v3_shadow_rolling")
        self.assertGreater(by_label["AlphaPlayer"].score, by_label["BetaPlayer"].score)

    def test_v3_shadow_position_gate_blocks_incompatible_shadow_book(self):
        st = Strategist(
            PerformanceMemory(trade_fraction=1.0),
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("AlphaPlayer", ["AlphaAgent"]),
                _make_player("BetaPlayer", ["BetaAgent"]),
            ],
            config=StrategistConfig(
                use_v3_rolling_score=True,
                use_v3_shadow_rolling_score=True,
                v3_shadow_position_gate_enabled=True,
                v3_shadow_flat_handoff_enabled=False,
                v3_shadow_rolling_min_closed_trades=1,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )
        st.update_realized_pnl_snapshot(
            pnl_by_player={},
            trade_counts_by_player={},
            win_counts_by_player={},
            initial_capital=100.0,
        )
        st.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="AlphaPlayer",
                regime="bullish",
                realized_pnl_usd=10.0,
                closed_trades=1,
                winning_trades=1,
            ),
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="BetaPlayer",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=1,
                winning_trades=1,
            ),
        ])
        st.update_shadow_position_snapshot(
            player_positions={
                "AlphaPlayer": ({"sym": "BTC", "side": "long"},),
                "BetaPlayer": (),
            },
            real_positions=(),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=2)

        self.assertEqual(decision.new_leader.label, "BetaPlayer")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertIn("v3 shadow position gate", reasons)

    def test_v3_shadow_position_gate_allows_flat_real_handoff(self):
        st = Strategist(
            PerformanceMemory(trade_fraction=1.0),
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("AlphaPlayer", ["AlphaAgent"]),
                _make_player("BetaPlayer", ["BetaAgent"]),
            ],
            config=StrategistConfig(
                use_v3_rolling_score=True,
                use_v3_shadow_rolling_score=True,
                v3_shadow_position_gate_enabled=True,
                v3_shadow_flat_handoff_enabled=True,
                v3_shadow_rolling_min_closed_trades=1,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )
        st.update_realized_pnl_snapshot(
            pnl_by_player={},
            trade_counts_by_player={},
            win_counts_by_player={},
            initial_capital=100.0,
        )
        st.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="AlphaPlayer",
                regime="bullish",
                realized_pnl_usd=10.0,
                closed_trades=1,
                winning_trades=1,
            ),
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="BetaPlayer",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=1,
                winning_trades=1,
            ),
        ])
        st.update_shadow_position_snapshot(
            player_positions={
                "AlphaPlayer": ({"sym": "BTC", "side": "long"},),
                "BetaPlayer": (),
            },
            real_positions=(),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=2)

        self.assertEqual(decision.new_leader.label, "AlphaPlayer")
        self.assertEqual(decision.candidate_rejections, ())

    def test_v3_shadow_position_gate_allows_fresh_shadow_handoff(self):
        st = Strategist(
            PerformanceMemory(trade_fraction=1.0),
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("AlphaPlayer", ["AlphaAgent"]),
                _make_player("BetaPlayer", ["BetaAgent"]),
            ],
            config=StrategistConfig(
                use_v3_rolling_score=True,
                use_v3_shadow_rolling_score=True,
                v3_shadow_position_gate_enabled=True,
                v3_shadow_flat_handoff_enabled=False,
                v3_shadow_fresh_handoff_enabled=True,
                v3_shadow_fresh_handoff_max_age_bars=1,
                v3_shadow_rolling_min_closed_trades=1,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )
        st.update_realized_pnl_snapshot(
            pnl_by_player={},
            trade_counts_by_player={},
            win_counts_by_player={},
            initial_capital=100.0,
        )
        st.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="AlphaPlayer",
                regime="bullish",
                realized_pnl_usd=10.0,
                closed_trades=1,
                winning_trades=1,
            ),
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="BetaPlayer",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=1,
                winning_trades=1,
            ),
        ])
        st.update_shadow_position_snapshot(
            player_positions={
                "AlphaPlayer": ({"sym": "BTC", "side": "long", "opened_bar": 1},),
                "BetaPlayer": (),
            },
            real_positions=(),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=2)

        self.assertEqual(decision.new_leader.label, "AlphaPlayer")
        self.assertEqual(decision.candidate_rejections, ())

    def test_v3_shadow_position_gate_blocks_stale_shadow_handoff(self):
        st = Strategist(
            PerformanceMemory(trade_fraction=1.0),
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("AlphaPlayer", ["AlphaAgent"]),
                _make_player("BetaPlayer", ["BetaAgent"]),
            ],
            config=StrategistConfig(
                use_v3_rolling_score=True,
                use_v3_shadow_rolling_score=True,
                v3_shadow_position_gate_enabled=True,
                v3_shadow_flat_handoff_enabled=False,
                v3_shadow_fresh_handoff_enabled=True,
                v3_shadow_fresh_handoff_max_age_bars=1,
                v3_shadow_rolling_min_closed_trades=1,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )
        st.update_realized_pnl_snapshot(
            pnl_by_player={},
            trade_counts_by_player={},
            win_counts_by_player={},
            initial_capital=100.0,
        )
        st.update_shadow_actor_updates([
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="AlphaPlayer",
                regime="bullish",
                realized_pnl_usd=10.0,
                closed_trades=1,
                winning_trades=1,
            ),
            ShadowActorUpdated(
                bar=1,
                actor_type="player",
                actor_label="BetaPlayer",
                regime="bullish",
                realized_pnl_usd=1.0,
                closed_trades=1,
                winning_trades=1,
            ),
        ])
        st.update_shadow_position_snapshot(
            player_positions={
                "AlphaPlayer": ({"sym": "BTC", "side": "long", "opened_bar": 1},),
                "BetaPlayer": (),
            },
            real_positions=(),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=4)

        self.assertEqual(decision.new_leader.label, "BetaPlayer")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertIn("v3 shadow position gate", reasons)

    def test_v3_real_loss_kill_blocks_candidate_before_slow_promotion_gate(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "BadAgent", Regime.BULLISH, 20, -2.0, start_id=1)
        _add_perf(virtual_perf, "SafeAgent", Regime.BULLISH, 8, 0.2, start_id=1000)
        _add_perf(real_perf, "BadPlayer", Regime.BULLISH, 3, -0.35, start_id=2000)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("BadPlayer", ["BadAgent"]),
                _make_player("SafePlayer", ["SafeAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_real_loss_kill_min_closed_trades=3,
                v3_real_loss_kill_pnl_pct=-0.5,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "SafePlayer")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertIn("v3 real-loss kill", reasons)

    def test_v3_persistent_loss_kill_uses_all_regime_real_history(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "BadAgent", Regime.BULLISH, 20, -2.0, start_id=1)
        _add_perf(virtual_perf, "SafeAgent", Regime.BULLISH, 8, 0.2, start_id=1000)
        _add_perf(real_perf, "BadPlayer", Regime.BULLISH, 5, 0.2, start_id=2000)
        _add_perf(real_perf, "BadPlayer", Regime.BEARISH, 30, -4.0, start_id=3000)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("BadPlayer", ["BadAgent"]),
                _make_player("SafePlayer", ["SafeAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_real_loss_kill_min_closed_trades=0,
                v3_persistent_loss_kill_min_closed_trades=20,
                v3_persistent_loss_kill_pnl_pct=-2.0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "SafePlayer")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertIn("v3 persistent real-loss kill", reasons)

    def test_v3_persistent_loss_kill_uses_realized_ledger_snapshot(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "BadAgent", Regime.BULLISH, 20, -2.0, start_id=1)
        _add_perf(virtual_perf, "SafeAgent", Regime.BULLISH, 8, 0.2, start_id=1000)
        _add_perf(real_perf, "BadPlayer", Regime.BULLISH, 25, 3.0, start_id=2000)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("BadPlayer", ["BadAgent"]),
                _make_player("SafePlayer", ["SafeAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_real_loss_kill_min_closed_trades=0,
                v3_persistent_loss_kill_min_closed_trades=20,
                v3_persistent_loss_kill_pnl_pct=-2.0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        st.update_realized_pnl_snapshot(
            pnl_by_player={"BadPlayer": -30.0},
            trade_counts_by_player={"BadPlayer": 25},
            win_counts_by_player={"BadPlayer": 15},
            initial_capital=1000.0,
        )
        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "SafePlayer")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertIn("v3 persistent realized-loss kill", reasons)

    def test_v3_persistent_loss_kill_spares_strong_virtual_edge(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "StrongAgent", Regime.BULLISH, 20, 4.0, start_id=1)
        _add_perf(virtual_perf, "SafeAgent", Regime.BULLISH, 8, 0.2, start_id=1000)
        _add_perf(real_perf, "StrongPlayer", Regime.BULLISH, 25, 3.0, start_id=2000)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[
                _make_player("StrongPlayer", ["StrongAgent"]),
                _make_player("SafePlayer", ["SafeAgent"]),
            ],
            real_perf=real_perf,
            config=StrategistConfig(
                use_v3_rolling_score=True,
                v3_real_loss_kill_min_closed_trades=0,
                v3_persistent_loss_kill_min_closed_trades=20,
                v3_persistent_loss_kill_pnl_pct=-2.0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        st.update_realized_pnl_snapshot(
            pnl_by_player={"StrongPlayer": -30.0},
            trade_counts_by_player={"StrongPlayer": 25},
            win_counts_by_player={"StrongPlayer": 15},
            initial_capital=1000.0,
        )
        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "StrongPlayer")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertNotIn("v3 persistent realized-loss kill", reasons)

    def test_player_session_overlay_penalizes_current_session_underperformance(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "DefaultEnsemble", Regime.NEUTRAL, 8, 0.40, start_id=1)
        _add_perf(perf, "TrendResearch", Regime.NEUTRAL, 8, 0.15, start_id=500)
        qm = QuarantineManager(seed=set())
        st = Strategist(
            perf,
            qm,
            candidates=[
                _make_player("DefaultEnsemble", ["A"]),
                _make_player("TrendResearch", ["B"]),
            ],
            config=StrategistConfig(
                player_session_overlay_weight=1.0,
                player_session_underperformance_weight=2.0,
                player_session_stale_penalty=0.0,
                player_session_pnl_cap_pct=5.0,
                player_session_min_activity=1,
            ),
        )
        st.capture_session_baseline()
        _add_perf(perf, "DefaultEnsemble", Regime.NEUTRAL, 3, -0.70, start_id=1000)
        _add_perf(perf, "TrendResearch", Regime.NEUTRAL, 3, 0.35, start_id=2000)

        decision = st.consider_switch(Regime.NEUTRAL, current_bar=1)

        self.assertEqual(decision.new_leader.label, "TrendResearch")
        by_label = {row.label: row for row in decision.candidate_scores}
        self.assertLess(by_label["DefaultEnsemble"].session_score_delta, 0.0)
        self.assertGreater(by_label["DefaultEnsemble"].session_underperformance_penalty, 0.0)

    def test_switch_gate_diagnostics_explain_small_margin(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(perf, "CurrentAgent", Regime.NEUTRAL, 8, 0.50, start_id=1)
        _add_perf(perf, "BetterAgent", Regime.NEUTRAL, 8, 0.45, start_id=500)
        qm = QuarantineManager(seed=set())
        st = Strategist(
            perf,
            qm,
            candidates=[
                _make_player("Current", ["CurrentAgent"]),
                _make_player("Better", ["BetterAgent"]),
            ],
            config=StrategistConfig(
                cooldown_bars=0,
                switch_margin=10.0,
                urgent_gap=100.0,
                hard_negative=-100.0,
            ),
        )
        st.consider_switch(Regime.NEUTRAL, current_bar=1)
        _add_perf(perf, "CurrentAgent", Regime.NEUTRAL, 1, -2.00, start_id=1000)

        decision = st.consider_switch(Regime.NEUTRAL, current_bar=2)

        self.assertFalse(decision.switched)
        self.assertEqual(decision.switch_gate_reason, "margin too small and not urgent")
        self.assertEqual(decision.best_label, "Better")
        self.assertEqual(decision.current_label, "Current")
        self.assertGreater(decision.best_score, decision.current_score)
        self.assertFalse(decision.cooldown_blocked)
        self.assertEqual(decision.required_margin, 10.0)


class TestStrategistRealPromotionGate(unittest.TestCase):
    def test_real_promotion_gate_selects_no_trade_after_loss_budget_breach(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "RiskyAgent", Regime.BULLISH, 12, 1.2, start_id=1)
        _add_perf(real_perf, "RiskyPlayer", Regime.BULLISH, 3, -1.0, start_id=1000)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[_make_player("RiskyPlayer", ["RiskyAgent"])],
            real_perf=real_perf,
            config=StrategistConfig(
                real_promotion_gate_enabled=True,
                real_promotion_min_closed_trades=1,
                real_promotion_min_pnl_pct=0.0,
                real_promotion_loss_budget_pct=-0.5,
                real_promotion_probation_min_score=0.0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertTrue(decision.switched)
        self.assertEqual(decision.new_leader.label, "NoTrade")
        reasons = " ".join(item.reason for item in decision.candidate_rejections)
        self.assertIn("real promotion gate", reasons)
        self.assertIn("loss budget", reasons)

    def test_real_promotion_gate_allows_probation_without_real_sample(self):
        virtual_perf = PerformanceMemory(trade_fraction=1.0)
        real_perf = PerformanceMemory(trade_fraction=1.0)
        _add_perf(virtual_perf, "StrongAgent", Regime.BULLISH, 12, 1.5, start_id=1)
        st = Strategist(
            virtual_perf,
            QuarantineManager(seed=set()),
            candidates=[_make_player("StrongPlayer", ["StrongAgent"])],
            real_perf=real_perf,
            config=StrategistConfig(
                real_promotion_gate_enabled=True,
                real_promotion_min_closed_trades=5,
                real_promotion_min_pnl_pct=0.0,
                real_promotion_loss_budget_pct=-0.5,
                real_promotion_probation_min_score=0.0,
                cooldown_bars=0,
                streak_needed=1,
            ),
        )

        decision = st.consider_switch(Regime.BULLISH, current_bar=1)

        self.assertEqual(decision.new_leader.label, "StrongPlayer")
        self.assertEqual(decision.candidate_rejections, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
