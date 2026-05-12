"""V1BridgeRunner — интеграция с v1-bridge как infrastructure-слой.

Главное препятствие production-запуска v2 — нужен источник live-данных.
v1 уже имеет готовый bridge для BITGET/MEXC. Мы НЕ переписываем его, а
**переиспользуем** через данный модуль:

  • импортируем v1 exchange_api_runtime
  • создаём bridge через standard API v1 (warmup, потом live)
  • на каждом тике v1-bridge вытаскиваем prices/volumes/regime
  • строим MarketSnapshot и передаём в v2 pipeline
  • v2 принимает решения → v2 TradeExecutor отправляет через v1
    exchange-клиент (тот же bridge.exchange API)

Это даёт production-готовый запуск БЕЗ необходимости переписывать
connector-ы. Когда оператор сделает чистые `bitget_adapter.py` /
`mexc_adapter.py` — этот файл можно будет удалить.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
        and callable(getattr(candidate, "run_cycle", None))
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
      bridge.run_cycle()  → запускает 1 итерацию
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
            self._bridge.run_cycle()
        except Exception:
            log.exception("[%s] bridge.run_cycle() error", self._exchange)
            return None

        bar = int(getattr(self._bridge, "_bar", 0) or 0)
        # Цены — пытаемся вытащить из _price_hist
        prices: Dict[str, float] = {}
        try:
            ph = getattr(self._bridge, "_price_hist", None)
            if isinstance(ph, list) and ph:
                last = ph[-1]
                if isinstance(last, dict):
                    prices = {k: float(v) for k, v in last.items() if v}
            elif isinstance(ph, dict):
                # dict[sym → list[float]]; берём последнее значение
                for sym, series in ph.items():
                    if series and series[-1]:
                        prices[sym] = float(series[-1])
        except Exception:
            log.exception("[%s] failed to extract prices", self._exchange)

        if not prices:
            return None

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
            regime=regime_str,
        )


# ────────────────────────────────────────────────────────────────────
# Bridge factory
# ────────────────────────────────────────────────────────────────────


def _create_bridge(exchange_name: str, mode: str = "live_futures") -> Any:
    """Создаёт v1-bridge через стандартную v1-фабрику.

    Использует exchange_api_runtime.bridge_factory или аналогичный
    API. Если bridge нельзя создать — возвращает None и логирует
    предупреждение.
    """
    _ensure_v1_paths()
    try:
        # v1 предоставляет фабрику через _exchange_adapter в Panteon_Trade
        # или через exchange_registry. Пробуем оба пути.
        import os as _os
        _os.environ.setdefault("CRYPTO_EXCHANGE", exchange_name.upper())
        try:
            from exchange_registry import build_bridge_for  # type: ignore
            bridge = build_bridge_for(exchange_name.upper())
            if _is_usable_bridge(bridge):
                return bridge
            log.warning("[%s] build_bridge_for returned unusable bridge", exchange_name)
        except (ImportError, AttributeError):
            pass
        # Fallback: создаём через MexcDirectClient / BitgetDirectClient
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
        bridge = _create_bridge(exchange_name, mode=mode)
    if bridge is None:
        log.error("[%s] cannot start: bridge unavailable", exchange_name)
        return 2

    feed = V1BridgeFeed(bridge, exchange_name=exchange_name)
    feed.warmup(warmup_bars=warmup_bars)

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
