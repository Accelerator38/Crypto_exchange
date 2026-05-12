"""V1BridgeRunner — интеграция с v1-bridge как infrastructure-слой.

Главное препятствие production-запуска v2 — нужен источник live-данных.
v1 уже имеет готовый bridge для BITGET/MEXC. Мы НЕ переписываем его, а
**переиспользуем** через данный модуль:

  • импортируем v1 exchange_runtime через exchange_registry
  • создаём bridge и используем только warmup/_fetch_market как market-data feed
  • НЕ запускаем v1 _run_cycle()/run_once(), чтобы не исполнять v1-ордера
  • на каждом тике v1-bridge вытаскиваем prices/volumes/regime
  • строим MarketSnapshot и передаём в v2 pipeline
  • v2 принимает решения → v2 TradeExecutor отправляет через v2 exchange adapter

Это даёт production-готовый запуск по данным из существующих connector-ов.
Когда появится чистый v2 market-feed для BITGET/MEXC — этот файл можно будет
удалить.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..domain.types import MarketSnapshot, Regime
from ..shadow.adapters import make_market_snapshot
from ..shadow.feed import MarketFeed
from .bootstrap import ProductionPipeline
from .main_loop import StepResult, main_loop
from .output_writer import OutputWriter


log = logging.getLogger(__name__)


def _is_usable_bridge(candidate: Any) -> bool:
    return (
        candidate is not None
        and callable(getattr(candidate, "warmup", None))
        and callable(getattr(candidate, "_fetch_market", None))
    )


def _ensure_v1_paths():
    project_root = Path(__file__).resolve().parents[3]
    for p in (
        project_root / "src" / "panteon_runtime",
        project_root / "Genetics_DL_Agents",
        project_root / "src",
    ):
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)


# ────────────────────────────────────────────────────────────────────
# V1BridgeFeed — обёртка v1-bridge под MarketFeed Protocol
# ────────────────────────────────────────────────────────────────────


class V1BridgeFeed:
    """Тонкая обёртка над v1-bridge. Эмиттит MarketSnapshot на каждом тике.

    Контракт:
      • При первом next_bar() — bridge.warmup() (если не сделан)
      • Каждый последующий next_bar() = один live-тик v1 bridge

    bridge должен иметь:
      bridge._fetch_market() → возвращает prices/volumes без исполнения v1-ордеров
      bridge._price_hist  → последние цены (list of dict OR dict)
      bridge._bar         → номер текущего бара
      bridge.funding.get_global() → ставки

    Это duck-typed подход — не зависим от конкретной версии bridge.
    """

    def __init__(self, bridge: Any, exchange_name: str):
        self._bridge = bridge
        self._exchange = exchange_name
        self._warmed_up = False

    def warmup(self, warmup_bars: Optional[int] = None) -> None:
        if self._warmed_up:
            return
        try:
            bars = warmup_bars or getattr(self._bridge, "warmup_bars", None) or 5760
            self._bridge.warmup(n_bars=int(bars))
            self._warmed_up = True
            log.info("[%s] v1-bridge warmup completed (%d bars)",
                     self._exchange, bars)
        except Exception:
            log.exception("[%s] bridge warmup failed", self._exchange)

    def next_bar(self) -> Optional[MarketSnapshot]:
        if not self._warmed_up:
            self.warmup()

        try:
            current_bar = int(getattr(self._bridge, "_bar", 0) or 0) + 1
            setattr(self._bridge, "_bar", current_bar)
            fetched = self._bridge._fetch_market()
        except Exception:
            log.exception("[%s] bridge._fetch_market() error", self._exchange)
            return None

        bar = int(getattr(self._bridge, "_bar", 0) or 0)
        prices: Dict[str, float] = {}
        volumes: Dict[str, float] = {}
        if isinstance(fetched, tuple) and len(fetched) >= 2:
            raw_prices, raw_volumes = fetched[0], fetched[1]
            if isinstance(raw_prices, dict):
                prices = {str(k).upper(): float(v) for k, v in raw_prices.items() if v}
            if isinstance(raw_volumes, dict):
                volumes = {
                    str(k).upper(): float(v)
                    for k, v in raw_volumes.items()
                    if str(k).upper() in prices
                }

        # Fallback: цены — пытаемся вытащить из _price_hist
        try:
            if not prices:
                ph = getattr(self._bridge, "_price_hist", None)
                if isinstance(ph, list) and ph:
                    last = ph[-1]
                    if isinstance(last, dict):
                        prices = {str(k).upper(): float(v) for k, v in last.items() if v}
                elif isinstance(ph, dict):
                    # dict[sym → list[float]]; берём последнее значение
                    for sym, series in ph.items():
                        if series and series[-1]:
                            prices[str(sym).upper()] = float(series[-1])
        except Exception:
            log.exception("[%s] failed to extract prices", self._exchange)

        if not prices:
            return None

        try:
            ph = getattr(self._bridge, "_price_hist", None)
            if isinstance(ph, list):
                ph.append(dict(prices))
            vh = getattr(self._bridge, "_volume_hist", None)
            if isinstance(vh, list):
                vh.append(dict(volumes))
        except Exception:
            log.debug("[%s] failed to append bridge market history", self._exchange)

        # Регим — пытаемся из bridge agent
        regime_str = "neutral"
        try:
            agents = getattr(self._bridge, "agents", {}) or {}
            for ag in agents.values():
                inner = getattr(ag, "_inner", ag)
                if hasattr(inner, "_inner"):
                    inner = inner._inner
                r = getattr(inner, "_r", None) or getattr(inner, "_last_regime", None)
                if r:
                    regime_str = str(r)
                    break
        except Exception:
            pass

        return make_market_snapshot(
            bar=bar,
            prices=prices,
            volumes=volumes,
            regime=regime_str,
        )


# ────────────────────────────────────────────────────────────────────
# Bridge factory
# ────────────────────────────────────────────────────────────────────


def _history_rows(history: Any) -> List[Dict[str, float]]:
    """Normalize v1 bridge history into ordered market rows."""
    rows: List[Dict[str, float]] = []

    if isinstance(history, list):
        for raw_row in history:
            if not isinstance(raw_row, dict):
                continue
            row: Dict[str, float] = {}
            for sym, value in raw_row.items():
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    continue
                if v:
                    row[str(sym).upper()] = v
            if row:
                rows.append(row)
        return rows

    if isinstance(history, dict):
        max_len = 0
        series_by_sym: Dict[str, Any] = {}
        for sym, series in history.items():
            if isinstance(series, (list, tuple)):
                series_by_sym[str(sym).upper()] = series
                max_len = max(max_len, len(series))
        for idx in range(max_len):
            row = {}
            for sym, series in series_by_sym.items():
                if idx >= len(series):
                    continue
                try:
                    v = float(series[idx])
                except (TypeError, ValueError):
                    continue
                if v:
                    row[sym] = v
            if row:
                rows.append(row)

    return rows


def _warmup_v2_agents_from_bridge(
    bridge: Any,
    registry: Any,
    *,
    exchange_name: str,
    max_bars: Optional[int] = None,
) -> int:
    """Replay v1 bridge market history through v2 agents without executing trades."""
    if getattr(bridge, "_panteon_v2_agent_warmup_done", False):
        return 0

    agents = list(registry.all_agents()) if callable(getattr(registry, "all_agents", None)) else []
    if not agents:
        return 0

    price_rows = _history_rows(getattr(bridge, "_price_hist", None))
    volume_rows = _history_rows(getattr(bridge, "_volume_hist", None))
    if max_bars is not None and int(max_bars) > 0:
        limit = int(max_bars)
        price_rows = price_rows[-limit:]
        volume_rows = volume_rows[-limit:]
    if not price_rows:
        log.warning("[%s] v2-agent warmup skipped: bridge price history is empty",
                    exchange_name)
        return 0

    warmed = 0
    for idx, prices in enumerate(price_rows, start=1):
        volumes = volume_rows[idx - 1] if idx - 1 < len(volume_rows) else {}
        market = make_market_snapshot(
            bar=idx,
            prices=prices,
            volumes=volumes,
            regime="neutral",
        )
        for agent in agents:
            try:
                agent.act(market)
            except Exception:
                log.debug("[%s] v2-agent warmup failed for %s",
                          exchange_name, getattr(agent, "label", type(agent).__name__),
                          exc_info=True)
        warmed += 1

    try:
        setattr(bridge, "_panteon_v2_agent_warmup_done", True)
    except Exception:
        pass
    log.info("[%s] v2-agent warmup completed: bars=%d agents=%d",
             exchange_name, warmed, len(agents))
    return warmed


def _create_bridge(
    exchange_name: str,
    mode: str = "live_futures",
    *,
    output_dir: Optional[str] = None,
    initial_capital: Optional[float] = None,
) -> Any:
    """Создаёт v1-bridge через exchange_registry.load_exchange_runtime().

    Bridge нужен v2 только как источник market-data: warmup + _fetch_market.
    Если bridge нельзя создать — возвращает None и логирует предупреждение.
    """
    _ensure_v1_paths()
    try:
        import os as _os
        _os.environ["CRYPTO_EXCHANGE"] = exchange_name.upper()
        _os.environ[f"{exchange_name.upper()}_TRADING_MODE"] = mode
        try:
            from exchange_registry import load_exchange_runtime  # type: ignore

            runtime = load_exchange_runtime(exchange_name)
            cfg = runtime.parse_settings(runtime.load_settings())
            if initial_capital is not None and float(initial_capital) > 0:
                cfg["initial_capital"] = float(initial_capital)
                cfg["paper_capital"] = float(initial_capital)
            bridge_cls = runtime.resolve_bridge_class()
            api_key = _os.getenv(runtime.api_key_env, "")
            api_secret = _os.getenv(runtime.api_secret_env, "")
            api_passphrase = _os.getenv(runtime.api_passphrase_env, "")
            direct_client = None
            if api_key and api_secret:
                try:
                    direct_client = runtime.create_direct_client(
                        api_key,
                        api_secret,
                        api_passphrase,
                    )
                except Exception as exc:
                    log.warning("[%s] direct client unavailable: %s", exchange_name, exc)

            bridge_kwargs = {
                "agents": {},
                "cfg": cfg,
                "mode": mode,
                "api_key": api_key,
                "api_secret": api_secret,
                "output_dir": output_dir,
                "direct_client": direct_client,
            }
            if api_passphrase:
                bridge_kwargs["api_passphrase"] = api_passphrase

            try:
                bridge = bridge_cls(**bridge_kwargs)
            except TypeError:
                bridge_kwargs.pop("api_passphrase", None)
                bridge = bridge_cls(**bridge_kwargs)

            try:
                setattr(bridge, "warmup_bars", int(runtime.warmup_bars or 0))
            except Exception:
                pass

            if _is_usable_bridge(bridge):
                log.info("[%s] v1 bridge created through exchange_runtime", exchange_name)
                return bridge
            log.warning("[%s] exchange_runtime returned unusable bridge", exchange_name)
        except (ImportError, AttributeError):
            pass
        log.warning("[%s] no v1 bridge factory available", exchange_name)
        return None
    except Exception:
        log.exception("[%s] failed to create v1-bridge — running without live feed",
                      exchange_name)
        return None


# ────────────────────────────────────────────────────────────────────
# Main entry: run_with_v1_bridge
# ────────────────────────────────────────────────────────────────────


def run_with_v1_bridge(
    pipeline:                  ProductionPipeline,
    *,
    exchange_name:             str,
    mode:                      str = "live_futures",
    bridge:                    Optional[Any] = None,
    output_writer:             Optional[OutputWriter] = None,
    max_bars:                  Optional[int] = None,
    max_idle_polls:            Optional[int] = None,
    sleep_between_polls_sec:   float = 5.0,
    warmup_bars:               Optional[int] = None,
) -> int:
    """Запускает pipeline через v1-bridge."""
    if bridge is None:
        bridge = _create_bridge(
            exchange_name,
            mode=mode,
            output_dir=(output_writer.output_dir if output_writer else None),
            initial_capital=pipeline.initial_capital,
        )
    if bridge is None:
        log.error("[%s] cannot start: bridge unavailable", exchange_name)
        return 2

    log.info("[%s] v1 bridge feed active: market-data only, v1 order cycle disabled",
             exchange_name)
    feed = V1BridgeFeed(bridge, exchange_name=exchange_name)
    feed.warmup(warmup_bars=warmup_bars)
    _warmup_v2_agents_from_bridge(
        bridge,
        pipeline.registry,
        exchange_name=exchange_name,
        max_bars=warmup_bars,
    )

    n_step = 0
    def _on_step(step: StepResult) -> None:
        nonlocal n_step
        n_step += 1
        if output_writer:
            try:
                output_writer.write(step)
            except Exception:
                log.exception("output_writer.write failed")
        if n_step % 10 == 0:
            log.info("  bar=%d leader=%s signals=%d filled=%d rejected=%d blocked=%d",
                     step.bar, step.leader or "-",
                     step.n_signals, step.n_filled,
                     step.n_rejected, step.n_blocked)

    def _on_error(exc: Exception) -> None:
        log.error("main_loop error: %s", exc, exc_info=True)

    def _on_idle(idle_polls: int) -> None:
        if idle_polls == 1 or idle_polls % 12 == 0:
            log.info("[%s] v1-bridge waiting for market bar: idle_poll=%d",
                     exchange_name, idle_polls)
        if output_writer:
            try:
                output_writer.write_heartbeat(
                    run_state="idle",
                    feed_status="v1_bridge_waiting",
                    message=f"idle_poll={idle_polls}",
                )
            except Exception:
                log.exception("output_writer heartbeat failed")

    try:
        steps = main_loop(
            pipeline, feed,
            max_bars=max_bars,
            sleep_between_polls_sec=sleep_between_polls_sec,
            on_step=_on_step,
            on_error=_on_error,
            on_idle=_on_idle,
            max_idle_polls=max_idle_polls,
        )
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — graceful shutdown")
        steps = []

    if output_writer:
        output_writer.close()

    log.info("Run finished, processed %d bars", len(steps))
    return 0
