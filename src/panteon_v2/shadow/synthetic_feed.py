"""SyntheticFeed — генератор синтетических MarketSnapshot для smoke / dryrun.

Используется когда live-feed недоступен, но нужно проверить что pipeline
реально работает: принимает решения, эмиттит events, пишет output.

Не для production — только для тестирования инфраструктуры.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..domain.types import MarketSnapshot, Regime
from .adapters import make_market_snapshot


@dataclass
class SyntheticFeed:
    """Генерирует MarketSnapshot с случайным walk цен и регим-циклами.

    Использование:
        feed = SyntheticFeed(
            symbols=["BTC", "ETH", "SOL"],
            base_prices={"BTC": 50000.0, "ETH": 3000.0, "SOL": 100.0},
            volatility=0.005,           # 0.5% волатильность на bar
            regime_cycle_bars=200,      # смена регима каждые 200 баров
            start_bar=1,
        )
        while True:
            snap = feed.next_bar()
            if snap is None: break
            # ...
    """

    symbols:           List[str] = field(default_factory=lambda: ["BTC", "ETH"])
    base_prices:       Dict[str, float] = field(default_factory=dict)
    volatility:        float = 0.005
    regime_cycle_bars: int = 200
    start_bar:         int = 1
    max_bars:          Optional[int] = None
    seed:              int = 42

    # ── State ───────────────────────────────────────────────────────
    _bar:    int = 0
    _prices: Dict[str, float] = field(default_factory=dict)
    _rng:    Optional[random.Random] = None
    _bars_emitted: int = 0

    def __post_init__(self):
        self._bar = self.start_bar - 1
        self._rng = random.Random(self.seed)
        # Default base_prices
        defaults = {"BTC": 50000.0, "ETH": 3000.0, "SOL": 100.0,
                    "BNB": 600.0,   "ADA": 0.5,    "DOGE": 0.1}
        for sym in self.symbols:
            self._prices[sym] = float(
                self.base_prices.get(sym, defaults.get(sym, 100.0))
            )

    def next_bar(self) -> Optional[MarketSnapshot]:
        if self.max_bars is not None and self._bars_emitted >= self.max_bars:
            return None
        self._bar += 1
        self._bars_emitted += 1

        # Random walk
        for sym in self.symbols:
            shock = self._rng.gauss(0, self.volatility)
            self._prices[sym] *= (1.0 + shock)
            if self._prices[sym] < 0.000001:
                self._prices[sym] = 0.000001

        # Регим: циклически переключаем через bar % cycle
        regime = self._regime_for_bar(self._bar)

        return make_market_snapshot(
            bar=self._bar,
            prices=dict(self._prices),
            volumes={sym: 1000.0 + self._rng.random() * 500 for sym in self.symbols},
            regime=regime,
        )

    def _regime_for_bar(self, bar: int) -> str:
        cycle = self.regime_cycle_bars
        phase = (bar // cycle) % 4
        return ["bullish", "neutral", "bearish", "neutral"][phase]
