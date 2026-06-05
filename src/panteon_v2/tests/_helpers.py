"""Helpers для тестов Phase 3."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

from panteon_v2.domain.types import Action, MarketSnapshot, Regime
from panteon_v2.selection.agent import Agent


class FakeAgent:
    """Простой fake-агент для тестов: отдаёт зафиксированный план действий."""

    def __init__(self, label: str, actions: Dict[str, Action] = None):
        self.label = label
        self._actions: Dict[str, Action] = actions or {}

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        # Возвращаем только symbols которые есть в market.prices
        return {sym: a for sym, a in self._actions.items() if sym in market.prices}


def make_market(
    bar: int = 1,
    regime: Regime = Regime.BULLISH,
    prices: Dict[str, float] = None,
    regimes_by_symbol: Dict[str, Regime] = None,
) -> MarketSnapshot:
    return MarketSnapshot(
        bar=bar,
        timestamp=datetime.now(timezone.utc),
        regime=regime,
        prices=prices or {"BTC": 100.0, "ETH": 50.0},
        volumes={s: 1000.0 for s in (prices or {"BTC": 100.0, "ETH": 50.0})},
        regimes_by_symbol=regimes_by_symbol or {},
    )
