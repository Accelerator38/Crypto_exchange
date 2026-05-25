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
from typing import Any, Callable, ClassVar, Dict, Iterable, Mapping, Optional, Tuple

from ..domain.types import Action, MarketSnapshot, Regime
from ..selection.agent import Agent


ActionMapper = Callable[[Any], Optional[Action]]


def _coerce_v2_action(value: Any) -> Optional[Action]:
    try:
        return Action(int(value))
    except (TypeError, ValueError):
        return None


_GENETICS_LEGACY_ACTION_MAP: Dict[int, Action] = {
    0: Action.HOLD,
    1: Action.SPOT_BUY_FULL,
    2: Action.SPOT_SELL_ALL,
    3: Action.FUT_LONG_FULL,
    4: Action.FUT_SHORT_FULL,
    5: Action.FUT_CLOSE_ALL,
}


def map_genetics_legacy_action(value: Any) -> Optional[Action]:
    """Map collapsed GeneticsAgent live actions into the v2 Action enum."""
    try:
        return _GENETICS_LEGACY_ACTION_MAP.get(int(value))
    except (TypeError, ValueError):
        return None


def _invoke_v1_agent(
    v1_agent: Any,
    *,
    prices: Dict[str, float],
    volumes: Dict[str, float],
    month: Optional[int],
    portfolio_value: float,
    bar_index: int,
) -> Dict[Any, Any]:
    try:
        raw = v1_agent.act(
            prices,
            volumes,
            month=month,
            portfolio_value=portfolio_value,
            bar_index=bar_index,
        )
    except TypeError:
        try:
            raw = v1_agent.act(
                prices,
                volumes,
                month=month,
                portfolio_value=portfolio_value,
            )
        except TypeError:
            try:
                raw = v1_agent.act(prices, volumes)
            except Exception:
                return {}
        except Exception:
            return {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _map_v1_actions(
    raw: Mapping[Any, Any],
    market: MarketSnapshot,
    action_mapper: ActionMapper,
) -> Dict[str, Action]:
    out: Dict[str, Action] = {}
    for sym, value in raw.items():
        sym = str(sym).upper()
        if sym not in market.prices:
            continue
        action = action_mapper(value)
        if action is None:
            continue
        out[sym] = action
    return out


# ────────────────────────────────────────────────────────────────────
# V1AgentAdapter
# ────────────────────────────────────────────────────────────────────


def _clone_genetics_runtime_agent(agent: Any) -> Any:
    clone_fn = getattr(agent, "clone_for_shadow", None)
    if callable(clone_fn):
        cloned = clone_fn()
        if cloned is not None:
            return cloned

    genome = getattr(agent, "genome", None)
    if genome is not None:
        genome_copy = copy.deepcopy(genome)
        cloned = None
        try:
            cloned = type(agent)(genome=genome_copy)
        except Exception:
            try:
                cloned = type(agent)(genome_copy)
            except Exception:
                cloned = None
        if cloned is not None:
            for attr in ("MAX_POS", "_position_state_features_enabled"):
                if hasattr(agent, attr):
                    try:
                        setattr(cloned, attr, copy.deepcopy(getattr(agent, attr)))
                    except Exception:
                        pass
            return cloned

    try:
        return type(agent)()
    except Exception:
        return copy.deepcopy(agent)


def _state_float(obj: Any, attr: str, sym: str) -> float:
    state = getattr(obj, attr, None)
    if not isinstance(state, dict):
        return 0.0
    try:
        return float(state.get(sym, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_open_regimes(values: Optional[Iterable[Any]]) -> Optional[frozenset[Regime]]:
    if values is None:
        return None
    regimes = set()
    for value in values:
        if isinstance(value, Regime):
            regimes.add(value)
        else:
            regimes.add(Regime.from_string(str(value)))
    return frozenset(regimes)


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
    action_mapper: ActionMapper = _coerce_v2_action

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
            action_mapper=self.action_mapper,
        )

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        prices = dict(market.prices)
        volumes = dict(market.volumes)
        portfolio_value = self.portfolio_value_fn() if self.portfolio_value_fn else 0.0
        try:
            raw = self.v1_agent.act(
                prices,
                volumes,
                month=market.month,
                portfolio_value=portfolio_value,
                bar_index=market.bar,
            )
        except TypeError:
            # Старый сигнатур без kwargs
            try:
                raw = self.v1_agent.act(
                    prices,
                    volumes,
                    month=market.month,
                    portfolio_value=portfolio_value,
                )
            except TypeError:
                try:
                    raw = self.v1_agent.act(prices, volumes)
                except Exception:
                    return {}
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
            action = self.action_mapper(value)
            if action is None:
                continue
            out[sym] = action
        return out

    def sync_from_execution_results(self, results) -> None:
        update = getattr(self.v1_agent, "update_from_exchange", None)
        if not callable(update):
            return
        for result in results or ():
            status = getattr(result, "status", None)
            status_value = str(getattr(status, "value", status)).lower()
            if status_value != "filled":
                continue
            signal = getattr(result, "signal", None)
            trade = getattr(result, "trade", None)
            if signal is None or trade is None:
                continue
            try:
                action = signal.action if isinstance(signal.action, Action) else Action(int(signal.action))
            except (TypeError, ValueError):
                continue
            sym = str(getattr(trade, "sym", None) or getattr(signal, "sym", "")).upper()
            if not sym:
                continue
            spot_qty = _state_float(self.v1_agent, "spot_qty", sym)
            spot_entry = _state_float(self.v1_agent, "spot_entry", sym)
            fut_qty = _state_float(self.v1_agent, "fut_qty", sym)
            fut_entry = _state_float(self.v1_agent, "fut_entry", sym)
            qty = float(getattr(trade, "qty", 0.0) or 0.0)
            entry = float(getattr(trade, "fill_price", 0.0) or 0.0)
            if action in (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL):
                spot_qty = qty
                spot_entry = entry
            elif action == Action.SPOT_SELL_ALL:
                spot_qty = 0.0
                spot_entry = 0.0
            elif action in (Action.FUT_LONG_HALF, Action.FUT_LONG_FULL):
                fut_qty = qty
                fut_entry = entry
            elif action in (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL):
                fut_qty = -qty
                fut_entry = entry
            elif action == Action.FUT_CLOSE_ALL:
                fut_qty = 0.0
                fut_entry = 0.0
            else:
                continue
            try:
                update(sym, spot_qty, spot_entry, fut_qty, fut_entry)
            except Exception:
                continue


class GeneticsV2AgentAdapter(V1AgentAdapter):
    """Adapter for GeneticsAgent collapsed live actions."""

    _act_cache: ClassVar[Dict[Tuple[object, ...], Dict[str, Action]]] = {}
    _act_cache_max_size: ClassVar[int] = 512

    def __init__(
        self,
        label: str,
        v1_agent: Any,
        portfolio_value_fn: Optional[Callable[[], float]] = None,
        allowed_open_regimes: Optional[Iterable[Any]] = None,
    ) -> None:
        super().__init__(
            label=label,
            v1_agent=v1_agent,
            portfolio_value_fn=portfolio_value_fn,
            action_mapper=map_genetics_legacy_action,
        )
        self.allowed_open_regimes = _normalize_open_regimes(allowed_open_regimes)

    def clone_for_shadow(self) -> "GeneticsV2AgentAdapter":
        cloned_v1 = _clone_genetics_runtime_agent(self.v1_agent)
        return GeneticsV2AgentAdapter(
            label=self.label,
            v1_agent=cloned_v1,
            portfolio_value_fn=self.portfolio_value_fn,
            allowed_open_regimes=self.allowed_open_regimes,
        )

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        key = self._cache_key(market)
        cached = self._act_cache.get(key)
        if cached is not None:
            return dict(cached)
        result = super().act(market)
        if (
            self.allowed_open_regimes is not None
            and market.regime not in self.allowed_open_regimes
        ):
            result = {
                sym: (Action.HOLD if action.is_open else action)
                for sym, action in result.items()
            }
        self._act_cache[key] = dict(result)
        if len(self._act_cache) > self._act_cache_max_size:
            self._act_cache.pop(next(iter(self._act_cache)))
        return result

    def _cache_key(self, market: MarketSnapshot) -> Tuple[object, ...]:
        return (
            self.label,
            (
                None
                if self.allowed_open_regimes is None
                else tuple(sorted(regime.label for regime in self.allowed_open_regimes))
            ),
            int(market.bar),
            getattr(market.timestamp, "isoformat", lambda: "")(),
            market.regime.label,
            tuple(sorted((str(sym), round(float(price), 10)) for sym, price in market.prices.items())),
            tuple(sorted((str(sym), round(float(volume), 10)) for sym, volume in market.volumes.items())),
        )


# ────────────────────────────────────────────────────────────────────
# V1MarketAdapter — сборка MarketSnapshot из raw данных
# ────────────────────────────────────────────────────────────────────


class GeneticsRegimeRouterV2AgentAdapter:
    """Route a validated genetics regime map through the v2 Agent interface."""

    def __init__(
        self,
        label: str,
        baseline_agent: Any,
        regime_agents: Mapping[Any, Any],
        portfolio_value_fn: Optional[Callable[[], float]] = None,
        *,
        min_regime_confidence: float = 0.70,
        allowed_open_regimes: Optional[Iterable[Any]] = None,
    ) -> None:
        if not 0.0 <= float(min_regime_confidence) <= 1.0:
            raise ValueError("min_regime_confidence must be in [0, 1]")
        self.label = str(label)
        self.baseline_agent = baseline_agent
        self.regime_agents: Dict[Regime, Any] = {
            (regime if isinstance(regime, Regime) else Regime.from_string(str(regime))): agent
            for regime, agent in dict(regime_agents or {}).items()
        }
        self.portfolio_value_fn = portfolio_value_fn
        self.min_regime_confidence = float(min_regime_confidence)
        self.allowed_open_regimes = _normalize_open_regimes(allowed_open_regimes)

    def clone_for_shadow(self) -> "GeneticsRegimeRouterV2AgentAdapter":
        clones: Dict[int, Any] = {}

        def clone_agent(agent: Any) -> Any:
            key = id(agent)
            if key in clones:
                return clones[key]
            cloned = _clone_genetics_runtime_agent(agent)
            clones[key] = cloned
            return cloned

        return GeneticsRegimeRouterV2AgentAdapter(
            label=self.label,
            baseline_agent=clone_agent(self.baseline_agent),
            regime_agents={
                regime: clone_agent(agent)
                for regime, agent in self.regime_agents.items()
            },
            portfolio_value_fn=self.portfolio_value_fn,
            min_regime_confidence=self.min_regime_confidence,
            allowed_open_regimes=self.allowed_open_regimes,
        )

    def _selected_agent(self, market: MarketSnapshot) -> Any:
        if float(market.regime_confidence) < self.min_regime_confidence:
            return self.baseline_agent
        return self.regime_agents.get(market.regime, self.baseline_agent)

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        prices = dict(market.prices)
        volumes = dict(market.volumes)
        portfolio_value = self.portfolio_value_fn() if self.portfolio_value_fn else 0.0
        raw = _invoke_v1_agent(
            self._selected_agent(market),
            prices=prices,
            volumes=volumes,
            month=market.month,
            portfolio_value=portfolio_value,
            bar_index=market.bar,
        )
        result = _map_v1_actions(raw, market, map_genetics_legacy_action)
        if (
            self.allowed_open_regimes is not None
            and market.regime not in self.allowed_open_regimes
        ):
            result = {
                sym: (Action.HOLD if action.is_open else action)
                for sym, action in result.items()
            }
        return result

    def sync_from_execution_results(self, results) -> None:
        seen: set[int] = set()
        for agent in [self.baseline_agent, *self.regime_agents.values()]:
            key = id(agent)
            if key in seen:
                continue
            seen.add(key)
            V1AgentAdapter(self.label, agent).sync_from_execution_results(results)


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
