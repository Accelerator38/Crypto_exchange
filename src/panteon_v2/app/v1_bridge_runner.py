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

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..analysis.technical_indicators import TechnicalIndicatorState
from ..domain.types import MarketSnapshot, Regime, TechnicalIndicators
from ..shadow.adapters import make_market_snapshot
from ..shadow.feed import MarketFeed
from .bootstrap import ProductionPipeline
from .live_state import (
    prepare_v2_agents_for_live_after_warmup as _shared_prepare_live_agents,
)
from .main_loop import StepResult, main_loop
from .output_writer import OutputWriter
from .regime_detector import PriceRegimeDetector


log = logging.getLogger(__name__)
_BRIDGE_CACHE_SCHEMA_VERSION = 1


def _account_log_fields(pipeline: ProductionPipeline) -> str:
    balance = _float_or_zero(getattr(pipeline, "current_balance", 0.0))
    return f"balance=${balance:.2f} positions={_position_count(pipeline)}"


def _position_count(pipeline: ProductionPipeline) -> int:
    tracked_count = 0
    try:
        tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
        if tracker is not None and hasattr(tracker, "all_open"):
            tracked_count = len(tracker.all_open() or {})
    except Exception:
        tracked_count = 0

    exchange_count = 0
    try:
        exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
        getter = getattr(exchange, "get_all_positions", None)
        if callable(getter):
            exchange_count = len(getter() or {})
    except Exception:
        exchange_count = 0
    return max(tracked_count, exchange_count)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _is_usable_bridge(candidate: Any) -> bool:
    return (
        candidate is not None
        and callable(getattr(candidate, "warmup", None))
        and callable(getattr(candidate, "_fetch_market", None))
    )


def _bitget_contract_reject_reason(futures_client: Any, sym: str) -> str:
    bad = getattr(futures_client, "_bad_symbols", None)
    if isinstance(bad, set) and sym in bad:
        return "runtime_bad_symbol"
    getter = getattr(futures_client, "_get_contract_meta", None)
    if not callable(getter):
        return ""
    try:
        meta = getter(sym)
    except Exception as exc:
        return f"contract metadata unavailable: {type(exc).__name__}: {exc}"
    if not isinstance(meta, dict) or not meta:
        return "empty contract metadata"
    if bool(meta.get("metadataFallback")):
        return "fallback contract metadata"
    source = str(meta.get("metadataSource") or "").strip().lower()
    if source in {"fallback", "default", "synthetic"}:
        return "fallback contract metadata"
    if meta.get("apiAllowed") is False:
        return "contract API disabled"
    state = meta.get("state")
    if state not in (None, "", 0, "0", "online", "enabled", "normal", "live", "trading"):
        return f"contract state {state!r}"
    try:
        step = float(
            meta.get("contractSize")
            or meta.get("amountStep")
            or meta.get("sizeIncrement")
            or 0.0
        )
    except (TypeError, ValueError):
        step = 0.0
    if step <= 0:
        return "invalid contract size"
    return ""


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

    def __init__(
        self,
        bridge: Any,
        exchange_name: str,
        *,
        regime_detector: Optional[PriceRegimeDetector] = None,
        technical_indicator_state: Optional[TechnicalIndicatorState] = None,
    ):
        self._bridge = bridge
        self._exchange = exchange_name
        self._warmed_up = False
        self._regime_detector = regime_detector or PriceRegimeDetector()
        self._regime_seeded = False
        self._technical_indicator_state = (
            technical_indicator_state or TechnicalIndicatorState()
        )
        self._technicals_seeded = False

    def warmup(self, warmup_bars: Optional[int] = None) -> None:
        if self._warmed_up:
            return
        try:
            bars = warmup_bars or getattr(self._bridge, "warmup_bars", None) or 5760
            self._bridge.warmup(n_bars=int(bars))
            self._warmed_up = True
            self._seed_regime_detector_from_bridge_history()
            self._seed_technicals_from_bridge_history()
            log.info("[%s] v1-bridge warmup completed (%d bars)",
                     self._exchange, bars)
        except Exception:
            log.exception("[%s] bridge warmup failed", self._exchange)

    def mark_warmup_restored_from_cache(self) -> None:
        self._warmed_up = True
        self._regime_detector = PriceRegimeDetector()
        self._regime_seeded = False
        self._technical_indicator_state = TechnicalIndicatorState()
        self._technicals_seeded = False
        self._seed_regime_detector_from_bridge_history()
        self._seed_technicals_from_bridge_history()

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

        prices, volumes = self._filter_contract_symbols(prices, volumes)
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

        # Regime is owned by the v2 price detector, not by agent internals.
        regime = self._regime_detector.update(prices)
        technicals_by_symbol = self._update_technicals_from_prices(prices)

        return make_market_snapshot(
            bar=bar,
            prices=prices,
            volumes=volumes,
            regime=regime.label,
            regime_confidence=self._regime_detector.confidence,
            technicals_by_symbol=technicals_by_symbol,
        )

    def _filter_contract_symbols(
        self,
        prices: Dict[str, float],
        volumes: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        if self._exchange.upper() != "BITGET":
            return prices, volumes
        bridge_mode = str(getattr(self._bridge, "mode", "") or "").lower()
        if bridge_mode and bridge_mode != "live_futures":
            return prices, volumes
        futures_client = getattr(self._bridge, "futures_client", None)
        if futures_client is None:
            return prices, volumes

        kept_prices: Dict[str, float] = {}
        kept_volumes: Dict[str, float] = {}
        for sym, price in prices.items():
            reason = _bitget_contract_reject_reason(futures_client, sym)
            if reason:
                bad = getattr(futures_client, "_bad_symbols", None)
                if isinstance(bad, set):
                    bad.add(sym)
                logged = getattr(self, "_contract_filter_logged", set())
                key = (sym, reason)
                if key not in logged:
                    log.warning(
                        "[BITGET] symbol %s excluded before selector/risk: %s",
                        sym,
                        reason,
                    )
                    logged.add(key)
                    setattr(self, "_contract_filter_logged", logged)
                continue
            kept_prices[sym] = price
            if sym in volumes:
                kept_volumes[sym] = volumes[sym]
        return kept_prices, kept_volumes


# ────────────────────────────────────────────────────────────────────
# Bridge factory
# ────────────────────────────────────────────────────────────────────

    def _seed_regime_detector_from_bridge_history(self) -> None:
        if self._regime_seeded:
            return
        try:
            rows = _history_rows(getattr(self._bridge, "_price_hist", None))
            self._regime_detector.seed(rows)
        except Exception:
            log.debug("[%s] failed to seed regime detector", self._exchange, exc_info=True)
        self._regime_seeded = True

    def _seed_technicals_from_bridge_history(self) -> None:
        if self._technicals_seeded:
            return
        try:
            for row in _history_rows(getattr(self._bridge, "_price_hist", None)):
                self._update_technicals_from_prices(row)
        except Exception:
            log.debug("[%s] failed to seed technical indicators", self._exchange, exc_info=True)
        self._technicals_seeded = True

    def _update_technicals_from_prices(
        self,
        prices: Dict[str, float],
    ) -> Dict[str, TechnicalIndicators]:
        technicals: Dict[str, TechnicalIndicators] = {}
        for symbol, price in dict(prices or {}).items():
            try:
                close = float(price)
                technicals[str(symbol).upper()] = (
                    self._technical_indicator_state.update_symbol(
                        str(symbol).upper(),
                        high=close,
                        low=close,
                        close=close,
                    )
                )
            except (TypeError, ValueError):
                log.debug(
                    "[%s] skipped technical indicator update for %s",
                    self._exchange,
                    symbol,
                    exc_info=True,
                )
        return technicals


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


def _bridge_symbol_set(bridge: Any) -> set[str]:
    raw_symbols = getattr(bridge, "symbols", None)
    if not raw_symbols:
        kwargs = getattr(bridge, "kwargs", None)
        if isinstance(kwargs, dict):
            cfg = kwargs.get("cfg")
            if isinstance(cfg, dict):
                raw_symbols = cfg.get("symbols")
    if not raw_symbols:
        return set()
    out = set()
    for symbol in tuple(raw_symbols or ()):
        value = str(symbol or "").upper()
        if value:
            out.add(value)
    return out


def _history_symbol_set(rows: List[Dict[str, float]]) -> set[str]:
    out = set()
    for row in rows:
        out.update(str(sym).upper() for sym in row)
    return out


def _parse_cache_timestamp(value: object) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _bridge_timeframe_seconds(bridge: Any) -> int:
    raw = str(
        getattr(bridge, "_tf_kline", None)
        or getattr(bridge, "timeframe", None)
        or "1m"
    ).strip().lower()
    if not raw:
        return 60
    unit = raw[-1]
    try:
        value = int(raw[:-1] or "1")
    except ValueError:
        return 60
    if unit == "s":
        return max(1, value)
    if unit == "m":
        return max(1, value * 60)
    if unit == "h":
        return max(1, value * 3600)
    if unit == "d":
        return max(1, value * 86400)
    return 60


def restore_bridge_state_cache(
    bridge: Any,
    cache_path: Optional[str],
    *,
    exchange_name: str,
    max_age_sec: float = 6 * 3600,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    if not cache_path:
        return {"restored": False, "reason": "disabled"}
    path = Path(cache_path)
    if not path.exists():
        return {"restored": False, "reason": "missing"}

    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return {"restored": False, "reason": "cache_bad", "error": str(exc)}
    if not isinstance(payload, dict):
        return {"restored": False, "reason": "cache_bad"}
    if int(payload.get("schema_version") or 0) != _BRIDGE_CACHE_SCHEMA_VERSION:
        return {"restored": False, "reason": "schema_changed"}
    if str(payload.get("exchange") or "").upper() != str(exchange_name or "").upper():
        return {"restored": False, "reason": "exchange_changed"}

    generated_at = _parse_cache_timestamp(payload.get("generated_at"))
    if generated_at is None:
        return {"restored": False, "reason": "cache_bad"}
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    age_sec = max(0.0, (now_dt - generated_at).total_seconds())
    if max_age_sec is not None and age_sec > float(max_age_sec):
        return {"restored": False, "reason": "stale"}

    price_rows = _history_rows(payload.get("price_hist"))
    volume_rows = _history_rows(payload.get("volume_hist"))
    if not price_rows:
        return {"restored": False, "reason": "cache_bad"}

    expected_symbols = _bridge_symbol_set(bridge)
    cached_symbols = {
        str(symbol or "").upper()
        for symbol in tuple(payload.get("symbols") or ())
        if str(symbol or "").strip()
    }
    if not cached_symbols:
        cached_symbols = _history_symbol_set(price_rows)
    if expected_symbols and cached_symbols and expected_symbols != cached_symbols:
        return {"restored": False, "reason": "symbols_changed"}

    try:
        bar = int(payload.get("bar") or len(price_rows))
    except (TypeError, ValueError):
        bar = len(price_rows)
    setattr(bridge, "_price_hist", price_rows)
    setattr(bridge, "_volume_hist", volume_rows)
    setattr(bridge, "_bar", bar)
    return {
        "restored": True,
        "reason": "ok",
        "rows": len(price_rows),
        "bar": bar,
        "symbols": sorted(cached_symbols),
        "age_sec": age_sec,
    }


def save_bridge_state_cache(
    bridge: Any,
    cache_path: Optional[str],
    *,
    exchange_name: str,
    max_rows: int = 6000,
) -> Dict[str, Any]:
    if not cache_path:
        return {"saved": False, "reason": "disabled"}
    price_rows = _history_rows(getattr(bridge, "_price_hist", None))
    volume_rows = _history_rows(getattr(bridge, "_volume_hist", None))
    if max_rows and max_rows > 0:
        price_rows = price_rows[-int(max_rows):]
        volume_rows = volume_rows[-int(max_rows):]
    if not price_rows:
        return {"saved": False, "reason": "empty"}
    symbols = _bridge_symbol_set(bridge) or _history_symbol_set(price_rows)
    try:
        bar = int(getattr(bridge, "_bar", 0) or len(price_rows))
    except (TypeError, ValueError):
        bar = len(price_rows)
    payload = {
        "schema_version": _BRIDGE_CACHE_SCHEMA_VERSION,
        "exchange": str(exchange_name or "").upper(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bar": bar,
        "symbols": sorted(symbols),
        "price_hist": price_rows,
        "volume_hist": volume_rows,
    }
    path = Path(cache_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        return {"saved": False, "reason": "write_failed", "error": str(exc)}
    return {"saved": True, "rows": len(price_rows), "bar": bar}


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

    detector = PriceRegimeDetector()
    warmed = 0
    for idx, prices in enumerate(price_rows, start=1):
        volumes = volume_rows[idx - 1] if idx - 1 < len(volume_rows) else {}
        regime = detector.update(prices)
        market = make_market_snapshot(
            bar=idx,
            prices=prices,
            volumes=volumes,
            regime=regime.label,
            regime_confidence=detector.confidence,
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


def _prepare_v2_agents_for_live_after_warmup(
    pipeline: ProductionPipeline,
    *,
    exchange_name: str,
    bar_index: int = 0,
) -> Dict[str, int]:
    """Clear warmup-only positions and seed agents with real exchange positions."""
    return _shared_prepare_live_agents(
        pipeline,
        exchange_name=exchange_name,
        bar_index=bar_index,
    )


def _bridge_warmup_bars(bridge: Any, warmup_bars: Optional[int]) -> int:
    if warmup_bars is not None:
        return int(warmup_bars)
    return int(getattr(bridge, "warmup_bars", None) or 5760)


def _write_bridge_heartbeat(
    output_writer: Optional[OutputWriter],
    *,
    run_state: str,
    feed_status: str,
    message: str,
) -> None:
    if not output_writer:
        return
    try:
        output_writer.write_heartbeat(
            run_state=run_state,
            feed_status=feed_status,
            message=message,
        )
    except Exception:
        log.exception("output_writer heartbeat failed")


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
    bridge_cache_path:         Optional[str] = None,
    bridge_cache_max_age_sec:  float = 6 * 3600,
    snapshot_callback:         Optional[Callable[[], None]] = None,
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
    bars_to_warmup = _bridge_warmup_bars(bridge, warmup_bars)
    cache_status = restore_bridge_state_cache(
        bridge,
        bridge_cache_path,
        exchange_name=exchange_name,
        max_age_sec=bridge_cache_max_age_sec,
    )
    if cache_status.get("restored"):
        feed.mark_warmup_restored_from_cache()
        gap_bars = int(
            float(cache_status.get("age_sec") or 0.0)
            // float(_bridge_timeframe_seconds(bridge))
        )
        if gap_bars > 0:
            try:
                bridge.warmup(n_bars=gap_bars)
                feed.mark_warmup_restored_from_cache()
                save_bridge_state_cache(
                    bridge,
                    bridge_cache_path,
                    exchange_name=exchange_name,
                )
                log.info(
                    "[%s] bridge warmup cache topped up: gap_bars=%d",
                    exchange_name,
                    gap_bars,
                )
            except Exception:
                log.debug("[%s] bridge cache top-up failed", exchange_name, exc_info=True)
        _write_bridge_heartbeat(
            output_writer,
            run_state="warming_up",
            feed_status="warmup_cached",
            message=(
                "using cached warmup "
                f"rows={cache_status.get('rows', 0)} bar={cache_status.get('bar', 0)} "
                f"gap={gap_bars}"
            ),
        )
        log.info(
            "[%s] bridge warmup restored from cache: rows=%s bar=%s",
            exchange_name,
            cache_status.get("rows", 0),
            cache_status.get("bar", 0),
        )
    else:
        reason = str(cache_status.get("reason") or "missing")
        log.info("[%s] bridge warmup cache unavailable: %s", exchange_name, reason)
        _write_bridge_heartbeat(
            output_writer,
            run_state="warming_up",
            feed_status="warmup_loading",
            message=f"loading {bars_to_warmup} historical bars through v1 bridge",
        )
        feed.warmup(warmup_bars=warmup_bars)
        if bridge_cache_path:
            save_bridge_state_cache(
                bridge,
                bridge_cache_path,
                exchange_name=exchange_name,
            )
    _write_bridge_heartbeat(
        output_writer,
        run_state="warming_up",
        feed_status="agent_warmup",
        message="replaying warmup history into v2 agents",
    )
    warmed_bars = _warmup_v2_agents_from_bridge(
        bridge,
        pipeline.registry,
        exchange_name=exchange_name,
        max_bars=warmup_bars,
    )
    live_bar_index = max(
        int(getattr(bridge, "_bar", 0) or 0),
        int(warmed_bars or 0),
    )
    _prepare_v2_agents_for_live_after_warmup(
        pipeline,
        exchange_name=exchange_name,
        bar_index=live_bar_index,
    )
    _write_bridge_heartbeat(
        output_writer,
        run_state="running",
        feed_status="active",
        message=f"warmup completed; waiting for live market bar at index {live_bar_index}",
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
        if bridge_cache_path:
            cache_save_status = save_bridge_state_cache(
                bridge,
                bridge_cache_path,
                exchange_name=exchange_name,
            )
            if not cache_save_status.get("saved"):
                log.debug(
                    "[%s] bridge cache save skipped: %s",
                    exchange_name,
                    cache_save_status.get("reason"),
                )
        if snapshot_callback is not None:
            try:
                snapshot_callback()
            except Exception:
                log.debug("[%s] runtime snapshot callback failed", exchange_name, exc_info=True)
        if n_step % 10 == 0:
            filtered = getattr(step, "n_filtered_real_signals", 0)
            filtered_msg = f" filtered={filtered}" if filtered else ""
            log.info("  bar=%d leader=%s %s signals=%d filled=%d rejected=%d blocked=%d%s",
                     step.bar, step.leader or "-",
                     _account_log_fields(pipeline),
                     step.n_signals, step.n_filled,
                     step.n_rejected, step.n_blocked, filtered_msg)

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
        steps = None

    if output_writer:
        output_writer.close()
    if bridge_cache_path:
        save_bridge_state_cache(bridge, bridge_cache_path, exchange_name=exchange_name)

    processed = len(steps) if steps is not None else n_step
    log.info("Run finished, processed %d bars", processed)
    return 0
