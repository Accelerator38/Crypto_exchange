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
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar, Dict, Iterable, Mapping, Optional, Tuple

from ..domain.types import Action, MarketSnapshot, Regime, TechnicalIndicators
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


def _compact_symbol_key(symbol: Any) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "").replace("_", "")


def _map_v1_diagnostics(raw: Any, market: MarketSnapshot) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return {}
    market_symbols = {
        _compact_symbol_key(symbol): str(symbol).upper()
        for symbol in (getattr(market, "prices", {}) or {})
    }
    out: Dict[str, Dict[str, Any]] = {}
    for raw_symbol, payload in raw.items():
        symbol = market_symbols.get(_compact_symbol_key(raw_symbol))
        if not symbol:
            continue
        if isinstance(payload, Mapping):
            out[symbol] = dict(payload)
        else:
            out[symbol] = {"value": payload}
    return out


_GENETICS_PRICE_HISTORY_TAIL_ITEMS = 18_000
_GENETICS_VOLUME_HISTORY_TAIL_ITEMS = 18_000


def _copy_genome_for_shadow(genome: Any) -> Any:
    copy_fn = getattr(genome, "copy", None)
    if callable(copy_fn):
        try:
            return copy_fn()
        except Exception:
            pass
    return copy.deepcopy(genome)


def _copy_runtime_history_series(value: Any, *, max_items: int) -> Any:
    if max_items <= 0:
        return copy.deepcopy(value)
    source_maxlen = getattr(value, "maxlen", None)
    target_maxlen = source_maxlen if isinstance(source_maxlen, int) and source_maxlen > 0 else None
    limit = min(max_items, target_maxlen) if target_maxlen is not None else max_items
    if isinstance(value, deque):
        return deque(deque(value, maxlen=limit), maxlen=target_maxlen)
    if isinstance(value, list):
        return list(value[-limit:])
    if isinstance(value, tuple):
        return tuple(value[-limit:])
    try:
        tail = deque(value, maxlen=limit)
    except TypeError:
        return copy.deepcopy(value)
    return deque(tail, maxlen=target_maxlen)


def _copy_runtime_history_mapping(value: Any, *, max_items: int) -> Any:
    if not isinstance(value, Mapping):
        return _copy_runtime_history_series(value, max_items=max_items)
    return {
        key: _copy_runtime_history_series(series, max_items=max_items)
        for key, series in value.items()
    }


def _copy_genetics_runtime_attr(attr: str, value: Any) -> Any:
    if attr == "ph":
        return _copy_runtime_history_mapping(
            value,
            max_items=_GENETICS_PRICE_HISTORY_TAIL_ITEMS,
        )
    if attr == "vh":
        return _copy_runtime_history_mapping(
            value,
            max_items=_GENETICS_VOLUME_HISTORY_TAIL_ITEMS,
        )
    if attr in {"spot_qty", "spot_entry", "fut_qty", "fut_entry", "pos"} and isinstance(value, dict):
        return dict(value)
    if attr == "_regime_adaptive_output_bias_map" and isinstance(value, Mapping):
        return dict(value)
    if attr == "last_regime_adaptive_output_bias" and isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return copy.deepcopy(value)


def _clone_genetics_runtime_agent(agent: Any) -> Any:
    clone_fn = getattr(agent, "clone_for_shadow", None)
    if callable(clone_fn):
        cloned = clone_fn()
        if cloned is not None:
            return cloned

    genome = getattr(agent, "genome", None)
    if genome is not None:
        genome_copy = _copy_genome_for_shadow(genome)
        cloned = None
        try:
            cloned = type(agent)(genome=genome_copy)
        except Exception:
            try:
                cloned = type(agent)(genome_copy)
            except Exception:
                cloned = None
        if cloned is not None:
            for attr in (
                "MAX_POS",
                "_position_state_features_enabled",
                "_regime_adaptive_output_bias_enabled",
                "_regime_adaptive_output_bias_map",
                "last_regime_adaptive_output_bias",
                "ph",
                "vh",
                "spot_qty",
                "spot_entry",
                "fut_qty",
                "fut_entry",
                "pos",
                "t",
                "_regime",
                "_breadth",
                "_regime_conf",
                "_ema_breadth",
                "source_genome_path",
                "selection_manifest_path",
                "genetics_signal_source",
            ):
                if hasattr(agent, attr):
                    try:
                        setattr(
                            cloned,
                            attr,
                            _copy_genetics_runtime_attr(attr, getattr(agent, attr)),
                        )
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
    last_signal_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    prefers_full_market_snapshot: ClassVar[bool] = True

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
        self.last_signal_diagnostics = {}
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
        self.last_signal_diagnostics = _map_v1_diagnostics(
            getattr(self.v1_agent, "last_signal_diagnostics", {}),
            market,
        )
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
    _act_trace_cache: ClassVar[Dict[Tuple[object, ...], Dict[str, object]]] = {}
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
        self.last_regime_adaptive_output_bias: Dict[str, object] = {}

    def clone_for_shadow(self) -> "GeneticsV2AgentAdapter":
        cloned_v1 = _clone_genetics_runtime_agent(self.v1_agent)
        cloned = GeneticsV2AgentAdapter(
            label=self.label,
            v1_agent=cloned_v1,
            portfolio_value_fn=self.portfolio_value_fn,
            allowed_open_regimes=self.allowed_open_regimes,
        )
        for attr in (
            "source_genome_path",
            "source_meta_path",
            "selection_manifest_path",
            "genetics_signal_source",
        ):
            if hasattr(self, attr):
                try:
                    setattr(cloned, attr, copy.deepcopy(getattr(self, attr)))
                except Exception:
                    setattr(cloned, attr, getattr(self, attr))
        return cloned

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        key = self._cache_key(market)
        cached = self._act_cache.get(key)
        if cached is not None:
            self.last_regime_adaptive_output_bias = dict(self._act_trace_cache.get(key, {}))
            return dict(cached)
        if self._regime_adaptive_output_bias_enabled():
            result = self._act_with_symbol_local_regime_bias(market)
        else:
            result = super().act(market)
            trace = getattr(self.v1_agent, "last_regime_adaptive_output_bias", None)
            self.last_regime_adaptive_output_bias = dict(trace) if isinstance(trace, dict) else {}
        if self.allowed_open_regimes is not None:
            result = {
                sym: (
                    Action.HOLD
                    if action.is_open
                    and market.regime_for_symbol(sym) not in self.allowed_open_regimes
                    else action
                )
                for sym, action in result.items()
            }
        self._act_cache[key] = dict(result)
        self._act_trace_cache[key] = dict(self.last_regime_adaptive_output_bias)
        if len(self._act_cache) > self._act_cache_max_size:
            expired_key = next(iter(self._act_cache))
            self._act_cache.pop(expired_key)
            self._act_trace_cache.pop(expired_key, None)
        return result

    def _regime_adaptive_output_bias_enabled(self) -> bool:
        return bool(getattr(self.v1_agent, "_regime_adaptive_output_bias_enabled", False))

    def _act_with_symbol_local_regime_bias(
        self,
        market: MarketSnapshot,
    ) -> Dict[str, Action]:
        regimes_by_symbol = {
            str(sym).upper(): market.regime_for_symbol(str(sym))
            for sym in market.prices
        }
        unique_regimes = tuple(dict.fromkeys(regimes_by_symbol.values()))
        clones = {
            regime: _clone_genetics_runtime_agent(self.v1_agent)
            for regime in unique_regimes
            if regime != market.regime
        }
        result, trace = self._act_v1_with_regime_override(
            self.v1_agent,
            market,
            market.regime,
        )
        by_symbol_trace: Dict[str, object] = {}
        for sym, regime in regimes_by_symbol.items():
            if regime == market.regime and isinstance(trace, dict):
                by_symbol_trace[sym] = dict(trace)

        for regime, clone in clones.items():
            local_market = replace(market, regime=regime)
            local_result, local_trace = self._act_v1_with_regime_override(
                clone,
                local_market,
                regime,
            )
            for sym, sym_regime in regimes_by_symbol.items():
                if sym_regime != regime:
                    continue
                if sym in local_result:
                    result[sym] = local_result[sym]
                if isinstance(local_trace, dict):
                    by_symbol_trace[sym] = dict(local_trace)

        self.last_regime_adaptive_output_bias = dict(trace) if isinstance(trace, dict) else {}
        if by_symbol_trace:
            self.last_regime_adaptive_output_bias["by_symbol"] = by_symbol_trace
        return result

    def _act_v1_with_regime_override(
        self,
        legacy_agent: Any,
        market: MarketSnapshot,
        regime: Regime,
    ) -> Tuple[Dict[str, Action], Dict[str, object]]:
        old_present = hasattr(
            legacy_agent,
            "_regime_adaptive_output_bias_regime_override",
        )
        old_value = getattr(
            legacy_agent,
            "_regime_adaptive_output_bias_regime_override",
            None,
        )
        try:
            setattr(
                legacy_agent,
                "_regime_adaptive_output_bias_regime_override",
                regime.label,
            )
            prices = dict(market.prices)
            volumes = dict(market.volumes)
            portfolio_value = self.portfolio_value_fn() if self.portfolio_value_fn else 0.0
            raw = _invoke_v1_agent(
                legacy_agent,
                prices=prices,
                volumes=volumes,
                month=market.month,
                portfolio_value=portfolio_value,
                bar_index=market.bar,
            )
            result = _map_v1_actions(raw, market, self.action_mapper)
            trace = getattr(legacy_agent, "last_regime_adaptive_output_bias", None)
            return result, dict(trace) if isinstance(trace, dict) else {}
        finally:
            if old_present:
                setattr(
                    legacy_agent,
                    "_regime_adaptive_output_bias_regime_override",
                    old_value,
                )
            else:
                try:
                    delattr(
                        legacy_agent,
                        "_regime_adaptive_output_bias_regime_override",
                    )
                except AttributeError:
                    pass

    def _cache_key(self, market: MarketSnapshot) -> Tuple[object, ...]:
        bias_map = getattr(self.v1_agent, "_regime_adaptive_output_bias_map", {}) or {}
        if isinstance(bias_map, Mapping):
            bias_items = tuple(sorted((str(key), float(value)) for key, value in bias_map.items()))
        else:
            bias_items = ()
        return (
            self.label,
            self._runtime_source_key(),
            bool(getattr(self.v1_agent, "_regime_adaptive_output_bias_enabled", False)),
            bias_items,
            (
                None
                if self.allowed_open_regimes is None
                else tuple(sorted(regime.label for regime in self.allowed_open_regimes))
            ),
            int(market.bar),
            getattr(market.timestamp, "isoformat", lambda: "")(),
            market.regime.label,
            tuple(sorted(
                (str(sym), regime.label)
                for sym, regime in market.regimes_by_symbol.items()
            )),
            tuple(sorted((str(sym), round(float(price), 10)) for sym, price in market.prices.items())),
            tuple(sorted((str(sym), round(float(volume), 10)) for sym, volume in market.volumes.items())),
        )

    def _runtime_source_key(self) -> Tuple[object, ...]:
        for owner in (self, self.v1_agent):
            for attr in (
                "source_genome_path",
                "selection_manifest_path",
                "genetics_signal_source",
            ):
                value = getattr(owner, attr, None)
                if value:
                    return (attr, str(value))
        genome = getattr(self.v1_agent, "genome", None)
        if genome is not None:
            shape = getattr(genome, "shape", None)
            dtype = getattr(genome, "dtype", None)
            return ("genome_object", id(genome), str(shape), str(dtype))
        agent_type = type(self.v1_agent)
        return (
            "agent_type",
            getattr(agent_type, "__module__", ""),
            getattr(agent_type, "__qualname__", getattr(agent_type, "__name__", "")),
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
        result: Dict[str, Action] = {}
        for sym in prices:
            symbol_market = market.with_regime_for_symbol(sym)
            raw = _invoke_v1_agent(
                self._selected_agent(symbol_market),
                prices=prices,
                volumes=volumes,
                month=market.month,
                portfolio_value=portfolio_value,
                bar_index=market.bar,
            )
            mapped = _map_v1_actions(raw, symbol_market, map_genetics_legacy_action)
            action = mapped.get(sym)
            if action is None:
                continue
            if (
                self.allowed_open_regimes is not None
                and action.is_open
                and symbol_market.regime not in self.allowed_open_regimes
            ):
                action = Action.HOLD
            result[sym] = action
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
    technicals_by_symbol: Optional[Dict[str, TechnicalIndicators]] = None,
    regimes_by_symbol: Optional[Dict[str, object]] = None,
    regime_features_by_symbol: Optional[Dict[str, dict]] = None,
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

    clean_technicals: Dict[str, TechnicalIndicators] = {}
    for sym, indicators in (technicals_by_symbol or {}).items():
        sym = str(sym).upper()
        if sym in clean_prices and indicators is not None:
            clean_technicals[sym] = indicators

    clean_regimes: Dict[str, Regime] = {}
    for sym, raw_regime in (regimes_by_symbol or {}).items():
        sym = str(sym).upper()
        if sym in clean_prices:
            clean_regimes[sym] = (
                raw_regime
                if isinstance(raw_regime, Regime)
                else Regime.from_string(str(raw_regime or ""))
            )

    clean_regime_features: Dict[str, dict] = {}
    for sym, raw_features in (regime_features_by_symbol or {}).items():
        sym = str(sym).upper()
        if sym in clean_prices and isinstance(raw_features, dict):
            clean_regime_features[sym] = dict(raw_features)

    return MarketSnapshot(
        bar=int(bar),
        timestamp=timestamp or datetime.now(timezone.utc),
        regime=Regime.from_string(regime),
        prices=clean_prices,
        volumes=clean_volumes,
        regime_confidence=float(regime_confidence),
        funding=clean_funding,
        month=month,
        technicals_by_symbol=clean_technicals,
        regimes_by_symbol=clean_regimes,
        regime_features_by_symbol=clean_regime_features,
    )


# ────────────────────────────────────────────────────────────────────
# V1RegimeAdapter — обратная связь
# ────────────────────────────────────────────────────────────────────


def regime_to_v1_string(regime: Regime) -> str:
    """v2 Regime → v1 string (для логов / совместимости)."""
    return regime.label
