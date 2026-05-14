"""Shadow adapters — мост между внешним миром (v1, биржа) и v2.

Главная задача: упаковка данных в immutable v2-типы. Никаких импортов
из panteon_runtime — все v1-сущности приходят как duck-typed объекты.

Component:
  V1AgentAdapter  — оборачивает любой объект с `act(prices, volumes, ...)`
                    под v2 Agent Protocol
  V1MarketAdapter — функции сборки MarketSnapshot из raw v1 данных
  V1RegimeAdapter — каноническая конверсия regime-строки в Regime
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from ..domain.types import Action, MarketSnapshot, Regime
from ..selection.agent import Agent


# ────────────────────────────────────────────────────────────────────
# V1AgentAdapter
# ────────────────────────────────────────────────────────────────────


@dataclass
class V1AgentAdapter:
    """Оборачивает v1-агент под v2 Agent Protocol.

    v1-агент должен иметь метод `act(prices, volumes, month=, portfolio_value=, bar_index=)`
    возвращающий dict[sym, int 0..8].

    Использование:
        v1_agent = FundingArb()
        wrapped = V1AgentAdapter(label="FundingArb", v1_agent=v1_agent)
        registry.register(wrapped)
    """

    label:    str
    v1_agent: Any              # любой объект с .act(...)
    portfolio_value_fn: Optional[Callable[[], float]] = None

    def clone_for_shadow(self) -> "V1AgentAdapter":
        clone_fn = getattr(self.v1_agent, "clone_for_shadow", None)
        if callable(clone_fn):
            cloned_v1 = clone_fn()
        else:
            try:
                cloned_v1 = type(self.v1_agent)()
            except Exception:
                cloned_v1 = copy.deepcopy(self.v1_agent)
        return V1AgentAdapter(
            label=self.label,
            v1_agent=cloned_v1,
            portfolio_value_fn=self.portfolio_value_fn,
        )

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        try:
            raw = self.v1_agent.act(
                prices=dict(market.prices),
                volumes=dict(market.volumes),
                month=market.month,
                portfolio_value=(self.portfolio_value_fn() if self.portfolio_value_fn else 0.0),
                bar_index=market.bar,
            )
        except TypeError:
            # Старый сигнатур без kwargs
            try:
                raw = self.v1_agent.act(market.prices, market.volumes)
            except Exception:
                return {}
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Action] = {}
        for sym, value in raw.items():
            sym = str(sym).upper()
            if sym not in market.prices:
                continue
            try:
                out[sym] = Action(int(value))
            except (TypeError, ValueError):
                continue
        return out


# ────────────────────────────────────────────────────────────────────
# V1MarketAdapter — сборка MarketSnapshot из raw данных
# ────────────────────────────────────────────────────────────────────


def make_market_snapshot(
    *,
    bar:       int,
    prices:    Dict[str, float],
    volumes:   Optional[Dict[str, float]] = None,
    funding:   Optional[Dict[str, float]] = None,
    regime:    str = "neutral",
    regime_confidence: float = 1.0,
    month:     Optional[int] = None,
    timestamp: Optional[datetime] = None,
) -> MarketSnapshot:
    """Pure-функция: сборка MarketSnapshot из v1-данных.

    Применяет каноническую регим-конверсию + sanitize:
      • prices: только ненулевые
      • volumes: те же ключи, отсутствующие = 0
      • funding: optional dict
    """
    clean_prices: Dict[str, float] = {}
    for sym, p in (prices or {}).items():
        try:
            v = float(p)
            if v > 0:
                clean_prices[str(sym).upper()] = v
        except (TypeError, ValueError):
            continue

    clean_volumes: Dict[str, float] = {}
    for sym in clean_prices:
        v = (volumes or {}).get(sym, 0.0)
        try:
            clean_volumes[sym] = float(v) if v else 0.0
        except (TypeError, ValueError):
            clean_volumes[sym] = 0.0

    clean_funding: Dict[str, float] = {}
    for sym, f in (funding or {}).items():
        sym = str(sym).upper()
        if sym not in clean_prices:
            continue
        try:
            clean_funding[sym] = float(f)
        except (TypeError, ValueError):
            continue

    return MarketSnapshot(
        bar=int(bar),
        timestamp=timestamp or datetime.now(timezone.utc),
        regime=Regime.from_string(regime),
        prices=clean_prices,
        volumes=clean_volumes,
        regime_confidence=float(regime_confidence),
        funding=clean_funding,
        month=month,
    )


# ────────────────────────────────────────────────────────────────────
# V1RegimeAdapter — обратная связь
# ────────────────────────────────────────────────────────────────────


def regime_to_v1_string(regime: Regime) -> str:
    """v2 Regime → v1 string (для логов / совместимости)."""
    return regime.label
