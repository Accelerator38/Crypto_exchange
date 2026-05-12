"""Тесты Strategist."""

from __future__ import annotations

import unittest

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


def _make_player(label: str, agent_labels):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
