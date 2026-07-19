from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from panteon_v2.domain.types import Regime
from panteon_v2.selection.player_regime import (
    PlayerRegimeConfig,
    PlayerRegimeStrategist,
    rate_player_returns,
)


@dataclass
class _Player:
    label: str

    @property
    def agent_labels(self):
        return []


class _Memory:
    def __init__(self, rows):
        self.rows = dict(rows)
        self.reads = []

    def recent_returns(self, label, regime, *, limit=0):
        self.reads.append((label, regime, limit))
        values = tuple(self.rows.get((label, regime), ()))
        return values[-limit:] if limit > 0 else values

    def get(self, label, regime=None):
        return SimpleNamespace(signals=0, execution_failures=0)


def _config(**overrides):
    values = {
        "recency_decay": 0.5,
        "return_window": 10,
        "min_closed_trades": 3,
        "min_global_closed_trades": 0,
        "global_score_weight": 0.0,
        "downside_penalty": 0.0,
        "switch_margin": 0.0,
        "cooldown_bars": 0,
    }
    values.update(overrides)
    return PlayerRegimeConfig(**values)


def test_newer_results_have_more_influence_on_rating():
    improving = rate_player_returns(
        "P",
        Regime.BULLISH,
        (-4.0, -4.0, 5.0, 5.0),
        config=_config(),
    )
    deteriorating = rate_player_returns(
        "P",
        Regime.BULLISH,
        (5.0, 5.0, -4.0, -4.0),
        config=_config(),
    )

    assert improving.score > deteriorating.score
    assert improving.weighted_pnl_per_trade_pct > 0.0
    assert deteriorating.weighted_pnl_per_trade_pct < 0.0


def test_player_requires_enough_positive_regime_efficiency():
    too_few = rate_player_returns(
        "P", Regime.BULLISH, (5.0, 5.0), config=_config()
    )
    losing = rate_player_returns(
        "P", Regime.BULLISH, (-1.0, -1.0, -1.0), config=_config()
    )

    assert not too_few.eligible
    assert "insufficient" in too_few.reason
    assert not losing.eligible
    assert "non-positive" in losing.reason


def test_multi_selects_different_players_from_separate_regime_memories():
    memory = _Memory({
        ("BullPlayer", Regime.BULLISH): (2.0, 2.0, 3.0),
        ("BearPlayer", Regime.BULLISH): (-2.0, -2.0, -3.0),
        ("BullPlayer", Regime.BEARISH): (-3.0, -2.0, -2.0),
        ("BearPlayer", Regime.BEARISH): (3.0, 2.0, 2.0),
    })
    strategist = PlayerRegimeStrategist(memory, config=_config())
    strategist.update_candidates((_Player("BullPlayer"), _Player("BearPlayer")))

    bullish = strategist.consider_switch(Regime.BULLISH, 1)
    bearish = strategist.consider_switch(Regime.BEARISH, 2)

    assert bullish.new_leader.label == "BullPlayer"
    assert bearish.new_leader.label == "BearPlayer"
    assert all(row.score_source == "player_regime_recency" for row in bullish.candidate_scores)
    assert all(not row.agent_labels for row in bullish.candidate_scores)
    assert {
        key for row in bullish.candidate_scores for key in row.memory_keys_read
    } == {"BullPlayer|bullish", "BearPlayer|bullish"}


def test_multi_uses_no_trade_when_no_player_has_proven_positive_edge():
    memory = _Memory({})
    strategist = PlayerRegimeStrategist(memory, config=_config())
    strategist.update_candidates((_Player("ColdA"), _Player("ColdB")))

    decision = strategist.consider_switch(Regime.NEUTRAL, 1)

    assert decision.new_leader.label == "NoTrade"
    assert "no player has positive efficiency" in decision.reason


def test_multi_requires_positive_global_and_regime_efficiency():
    memory = _Memory({
        ("LocallyLucky", Regime.BULLISH): (2.0, 2.0, 2.0),
        ("LocallyLucky", None): (-2.0, -2.0, -2.0),
        ("Robust", Regime.BULLISH): (1.0, 1.0, 1.0),
        ("Robust", None): (1.0, 1.0, 1.0),
    })
    strategist = PlayerRegimeStrategist(
        memory,
        config=_config(
            min_global_closed_trades=3,
            global_score_weight=0.25,
        ),
    )
    strategist.update_candidates((
        _Player("LocallyLucky"),
        _Player("Robust"),
    ))

    decision = strategist.consider_switch(Regime.BULLISH, 1)
    rejected = {
        row.label: row.reason for row in decision.candidate_rejections
    }

    assert decision.new_leader.label == "Robust"
    assert "global player efficiency gate" in rejected["LocallyLucky"]


def test_multi_defers_no_trade_until_real_portfolio_is_flat():
    memory = _Memory({
        ("Player", Regime.BULLISH): (2.0, 2.0, 2.0),
    })
    strategist = PlayerRegimeStrategist(memory, config=_config())
    strategist.update_candidates((_Player("Player"),))
    selected = strategist.consider_switch(Regime.BULLISH, 1)
    memory.rows[("Player", Regime.BULLISH)] = (-2.0, -2.0, -2.0)

    deferred = strategist.consider_switch(
        Regime.BULLISH,
        2,
        portfolio_flat=False,
    )
    switched = strategist.consider_switch(
        Regime.BULLISH,
        3,
        portfolio_flat=True,
    )

    assert selected.new_leader.label == "Player"
    assert deferred.new_leader.label == "Player"
    assert deferred.manage_only
    assert not deferred.switched
    assert "portfolio is flat" in deferred.reason
    assert switched.new_leader.label == "NoTrade"
    assert switched.switched


def test_multi_defers_regime_player_switch_until_real_portfolio_is_flat():
    memory = _Memory({
        ("Bull", Regime.BULLISH): (2.0, 2.0, 2.0),
        ("Bear", Regime.BULLISH): (-2.0, -2.0, -2.0),
        ("Bull", Regime.BEARISH): (-2.0, -2.0, -2.0),
        ("Bear", Regime.BEARISH): (2.0, 2.0, 2.0),
    })
    strategist = PlayerRegimeStrategist(memory, config=_config())
    strategist.update_candidates((_Player("Bull"), _Player("Bear")))
    strategist.consider_switch(Regime.BULLISH, 1)

    deferred = strategist.consider_switch(
        Regime.BEARISH,
        2,
        portfolio_flat=False,
    )
    switched = strategist.consider_switch(
        Regime.BEARISH,
        3,
        portfolio_flat=True,
    )

    assert deferred.new_leader.label == "Bull"
    assert deferred.best_label == "Bear"
    assert deferred.manage_only
    assert switched.new_leader.label == "Bear"
    assert switched.switched


def test_singleton_pins_configured_player_without_rating_gate():
    memory = _Memory({("Fixed", Regime.CRASH): (-5.0, -4.0, -3.0)})
    strategist = PlayerRegimeStrategist(
        memory,
        config=_config(),
        fixed_player_label="Fixed",
    )
    strategist.update_candidates((_Player("Fixed"), _Player("Other")))

    first = strategist.consider_switch(Regime.CRASH, 1)
    second = strategist.consider_switch(Regime.BULLISH, 2)

    assert first.new_leader.label == "Fixed"
    assert second.new_leader.label == "Fixed"
    assert second.reason == "fixed singleton player"
