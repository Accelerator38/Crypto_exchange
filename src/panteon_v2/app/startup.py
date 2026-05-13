"""Production startup — единая точка запуска v2 для конкретной биржи.

Вызывается тонкими Start_*_v2.py скриптами в корне репозитория.

Что делает:
  1. Регистрирует v1-агенты через agent_bootstrap.
  2. Собирает Exchange-адаптер (FakeExchange для paper, real adapter
     иначе — пока нет, использует bridge как infrastructure).
  3. (опционально) загружает persisted snapshot v2 или мигрирует v1.
  4. Собирает ProductionPipeline (DI всех компонентов).
  5. Создаёт OutputWriter (status.json + leaderboard + dashboard).
  6. Запускает либо v1-bridge mode (live данные), либо empty-feed (smoke).
  7. main_loop с on_step → OutputWriter.write().

После Ctrl+C — сохраняет state и финальный snapshot.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..execution import Exchange, FakeExchange, RiskLimitsConfig
from ..execution.position_tracker import TrackedPosition
from ..selection import AgentRegistry, PlayerProfile
from ..shadow.feed import MarketFeed, PollingFeed, ReplayFeed
from ..shadow.synthetic_feed import SyntheticFeed
from .agent_bootstrap import register_all_v1_agents
from .bootstrap import PRODUCTION_PROFILES, build_production_pipeline
from .main_loop import main_loop
from .migration import (
    load_v2_snapshot,
    migrate_from_v1_memory_files,
    save_v2_snapshot,
)
from .output_writer import OutputWriter, OutputWriterConfig
from .shadow_tournament import ProductionShadowTournament


log = logging.getLogger(__name__)


def _resolve_initial_capital(
    *,
    exchange_adapter: Exchange,
    mode: str,
    requested_initial_capital: Optional[float],
    exchange_name: str,
) -> float:
    """Resolve v2 capital base.

    For live modes, absent explicit *_INITIAL_CAPITAL means "use real exchange
    equity". Paper keeps a deterministic configured/default balance.
    """
    if requested_initial_capital is not None:
        capital = float(requested_initial_capital)
        if mode != "paper":
            live_equity = _read_exchange_equity(exchange_adapter)
            if live_equity is not None:
                log.info(
                    "[%s] live account equity=$%.2f; explicit initial_capital=$%.2f kept",
                    exchange_name,
                    live_equity,
                    capital,
                )
        return capital

    if mode != "paper":
        live_equity = _read_exchange_equity(exchange_adapter)
        if live_equity is not None:
            log.info(
                "[%s] live account equity detected: $%.2f — using as v2 capital",
                exchange_name,
                live_equity,
            )
            return live_equity
        log.warning(
            "[%s] live account equity unavailable — falling back to $100.00",
            exchange_name,
        )

    return 100.0


def _read_exchange_equity(exchange_adapter: Exchange) -> Optional[float]:
    getter = getattr(exchange_adapter, "get_account_equity", None)
    if not callable(getter):
        return None
    try:
        equity = float(getter() or 0.0)
    except Exception:
        log.debug("exchange equity read failed", exc_info=True)
        return None
    return equity if equity > 0 else None


def _risk_config_from_trade_fraction(trade_fraction: float) -> RiskLimitsConfig:
    fraction = float(trade_fraction)
    if not 0 < fraction <= 1.0:
        raise ValueError(f"trade_fraction must be in (0, 1], got {trade_fraction}")
    return RiskLimitsConfig(capital_fraction=fraction)


def _resolve_trade_fraction(exchange_name: str) -> float:
    try:
        from .v1_futures_adapter import load_runtime_trade_fraction

        return load_runtime_trade_fraction(default=0.10)
    except Exception:
        log.debug("[%s] runtime trade_fraction unavailable", exchange_name, exc_info=True)
        return 0.10


def _perf_has_state(perf) -> bool:
    try:
        return bool((perf.snapshot().get("state") or {}))
    except Exception:
        return False


def _default_v1_memory_paths(exchange_name: str) -> List[str]:
    name = exchange_name.upper()
    repo_root = Path(__file__).resolve().parents[3]
    memory_dir = repo_root / "state" / "memory"
    candidates = [
        memory_dir / f"Real_Player_Memory_{name}.txt",
        memory_dir / f"Real_Player_Aggregator_Memory_{name}.txt",
    ]
    return [str(path) for path in candidates if path.exists()]


def _normalize_migration_paths(
    migrate_from_v1: Optional[Union[str, Sequence[str]]],
    *,
    exchange_name: str,
) -> List[str]:
    if migrate_from_v1 is None:
        return _default_v1_memory_paths(exchange_name)
    if isinstance(migrate_from_v1, (str, os.PathLike)):
        return [str(migrate_from_v1)]
    return [str(path) for path in migrate_from_v1]


def _load_or_migrate_state(
    *,
    perf,
    snapshot_path: Optional[str],
    migrate_from_v1: Optional[Union[str, Sequence[str]]],
    exchange_name: str,
) -> None:
    loaded = False
    if snapshot_path and os.path.exists(snapshot_path):
        loaded = load_v2_snapshot(perf, snapshot_path)
        if loaded:
            if _perf_has_state(perf):
                log.info("Loaded v2-snapshot from %s", snapshot_path)
                return
            log.warning(
                "Loaded v2-snapshot from %s, but it has no ratings; "
                "trying v1 memory migration",
                snapshot_path,
            )

    paths = _normalize_migration_paths(
        migrate_from_v1,
        exchange_name=exchange_name,
    )
    if not paths:
        if not loaded:
            log.info("[%s] no v1 memory files found for migration", exchange_name)
        return

    report = migrate_from_v1_memory_files(paths, perf)
    if report.n_regime_pairs:
        log.info(
            "Migrated from v1: %d labels, %d pairs from %d file(s)",
            report.n_labels,
            report.n_regime_pairs,
            len(paths),
        )
        if snapshot_path:
            save_v2_snapshot(perf, snapshot_path)
            log.info("Saved migrated v2-snapshot -> %s", snapshot_path)
    else:
        log.warning(
            "[%s] v1 memory migration found no ratings in: %s",
            exchange_name,
            ", ".join(paths),
        )
    for warning in report.warnings[:5]:
        log.warning("[%s] v1 migration: %s", exchange_name, warning)


# ────────────────────────────────────────────────────────────────────
# Exchange adapter resolution
# ────────────────────────────────────────────────────────────────────


def resolve_exchange(exchange_name: str, *, mode: str) -> Exchange:
    """Реальный Exchange-адаптер или FakeExchange.

    В live_futures используются bitget_adapter.py / mexc_adapter.py.
    FakeExchange остаётся только для paper или неподдержанной биржи.
    """
    name = exchange_name.upper()
    if mode == "paper":
        log.info("[%s] paper mode → FakeExchange", name)
        return FakeExchange(name=f"{name}-PAPER")

    if name == "BITGET":
        try:
            from .bitget_adapter import BitgetExchangeAdapter  # noqa
            log.info("[BITGET] real adapter detected")
            return BitgetExchangeAdapter()
        except ImportError:
            log.warning(
                "[BITGET] no bitget_adapter.py — using FakeExchange (DRYRUN). "
                "Implement bitget_adapter.py based on exchange_adapter_template.py"
            )
            return FakeExchange(name="BITGET-DRYRUN")

    if name == "MEXC":
        try:
            from .mexc_adapter import MexcExchangeAdapter  # noqa
            log.info("[MEXC] real adapter detected")
            return MexcExchangeAdapter()
        except ImportError:
            log.warning(
                "[MEXC] no mexc_adapter.py — using FakeExchange (DRYRUN). "
                "Implement mexc_adapter.py based on exchange_adapter_template.py"
            )
            return FakeExchange(name="MEXC-DRYRUN")

    log.warning("Unknown exchange %s; using FakeExchange", name)
    return FakeExchange(name=f"{name}-DRYRUN")


def _recover_exchange_positions(pipeline) -> int:
    """Seed the v2 tracker from live exchange positions after restart."""
    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    getter = getattr(exchange, "get_all_positions", None)
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if not callable(getter) or tracker is None:
        return 0

    try:
        exchange_positions = getter() or {}
    except Exception:
        log.warning("[%s] exchange open-position recovery failed",
                    pipeline.exchange_name, exc_info=True)
        return 0

    memory_by_symbol = _open_memory_by_symbol(pipeline)
    recovered = 0
    external = 0
    for raw_sym, exchange_pos in exchange_positions.items():
        sym = str(getattr(exchange_pos, "sym", raw_sym) or raw_sym).upper()
        if not sym or tracker.get(sym) is not None:
            continue
        memory = memory_by_symbol.get(sym, {})
        qty = _float_or_zero(getattr(exchange_pos, "qty", None), memory.get("qty"))
        entry = _float_or_zero(getattr(exchange_pos, "entry", None), memory.get("entry_price"))
        if qty <= 0 or entry <= 0:
            log.warning(
                "[%s] skipping recovered position %s: qty=%.8f entry=%.8f",
                pipeline.exchange_name,
                sym,
                qty,
                entry,
            )
            continue
        side = str(getattr(exchange_pos, "side", memory.get("side", "long")) or "long").lower()
        if side not in ("long", "short"):
            side = "long"
        by_player = str(memory.get("by_player") or "RecoveredExchangePosition")
        by_agent = str(memory.get("by_agent") or "")
        if not memory.get("by_player") and not memory.get("by_agent"):
            external += 1
        tracker.force_set(TrackedPosition(
            open_signal_id=int(memory.get("signal_id") or 0),
            sym=sym,
            side=side,
            entry_price=entry,
            qty=qty,
            fee_open=float(memory.get("fee_open") or 0.0),
            by_player=by_player,
            by_agent=by_agent,
            opened_at=datetime.now(timezone.utc),
        ))
        recovered += 1

    if recovered:
        log.info(
            "[%s] recovered %d open exchange position(s) into v2 tracker "
            "(external=%d)",
            pipeline.exchange_name,
            recovered,
            external,
        )
    return recovered


def _open_memory_by_symbol(pipeline) -> Dict[str, dict]:
    try:
        open_memory = pipeline.perf.snapshot().get("open") or {}
    except Exception:
        return {}
    try:
        agent_labels = set(pipeline.registry.all_labels())
    except Exception:
        agent_labels = set()
    player_labels = {getattr(profile, "label", "") for profile in getattr(pipeline, "profiles", [])}

    by_symbol: Dict[str, dict] = {}
    for key, payload in open_memory.items():
        if not isinstance(payload, dict):
            continue
        parts = str(key).split("|")
        if len(parts) == 3 and parts[0] not in ("", "real"):
            continue
        sym = str(parts[-1] if parts else "").upper()
        if not sym:
            continue
        label = str(payload.get("label") or (parts[-2] if len(parts) >= 2 else ""))
        if not label:
            continue
        rec = by_symbol.setdefault(sym, {})
        for field in ("signal_id", "side", "entry_price", "qty", "fee_open"):
            if field not in rec and field in payload:
                rec[field] = payload[field]
        if label in player_labels:
            rec["by_player"] = label
        elif label in agent_labels:
            rec["by_agent"] = label
        elif not rec.get("by_player"):
            rec["by_player"] = label
    return by_symbol


def _float_or_zero(*values) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


# ────────────────────────────────────────────────────────────────────
# Main entry: start_production
# ────────────────────────────────────────────────────────────────────


def start_production(
    *,
    exchange:           str,
    mode:               str = "live_futures",
    initial_capital:    Optional[float] = None,
    seed_quarantine:    Sequence[str] = ("FundingArb", "RichardDennis", "MomentumScalper"),
    profiles:           Optional[Sequence[PlayerProfile]] = None,
    polling_session_dir: Optional[str] = None,
    snapshot_path:      Optional[str] = None,
    migrate_from_v1:    Optional[Union[str, Sequence[str]]] = None,
    jsonl_event_log:    Optional[str] = None,
    max_bars:           Optional[int] = None,
    max_idle_polls:     Optional[int] = None,
    sleep_between_polls_sec: float = 5.0,
    use_v1_bridge:      bool = True,
    results_root:       str = "Results",
    warmup_bars:        Optional[int] = None,
) -> int:
    """Запустить production main_loop для указанной биржи.

    Параметры:
      use_v1_bridge — если True, попытается подключиться к v1-bridge
                      для получения live-данных. Если False или bridge
                      недоступен — используется PollingFeed (если задан
                      polling_session_dir) или ReplayFeed (smoke).
      results_root  — корневая папка для OutputWriter (по умолчанию Results/)
    """
    log.info("=" * 70)
    log.info("Panteon v2 production startup: %s mode=%s", exchange, mode)
    log.info("=" * 70)
    os.environ["CRYPTO_EXCHANGE"] = exchange.upper()
    os.environ[f"{exchange.upper()}_TRADING_MODE"] = mode

    # 1. Exchange adapter + capital base
    exchange_adapter = resolve_exchange(exchange, mode=mode)
    resolved_initial_capital = _resolve_initial_capital(
        exchange_adapter=exchange_adapter,
        mode=mode,
        requested_initial_capital=initial_capital,
        exchange_name=exchange,
    )
    trade_fraction = _resolve_trade_fraction(exchange)
    risk_config = _risk_config_from_trade_fraction(trade_fraction)
    log.info("[%s] v2 risk capital_fraction=%.2f%%",
             exchange, risk_config.capital_fraction * 100.0)

    # 2. Регистрируем v1-агенты. Они получают актуальный v2 balance.
    registry = AgentRegistry()
    pipeline_holder = {}
    def _portfolio_value() -> float:
        pipeline = pipeline_holder.get("pipeline")
        if pipeline is not None:
            return float(getattr(pipeline, "current_balance", resolved_initial_capital) or resolved_initial_capital)
        return resolved_initial_capital

    registered = register_all_v1_agents(
        registry,
        portfolio_value_fn=_portfolio_value,
        skip_on_error=True,
    )
    log.info("Registered %d v1-agents: %s", len(registered),
             ", ".join(registered[:5]) + ("…" if len(registered) > 5 else ""))
    if not registered:
        log.error(
            "No v1-agents registered. Check sys.path / panteon_runtime presence."
        )
        return 2

    # 3. ProductionPipeline
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange_adapter,
        initial_capital=resolved_initial_capital,
        seed_quarantine=seed_quarantine,
        profiles=profiles or PRODUCTION_PROFILES,
        risk_config=risk_config,
        perf_trade_fraction=trade_fraction,
        jsonl_event_log=jsonl_event_log,
    )
    pipeline_holder["pipeline"] = pipeline
    log.info("Pipeline built: %d profiles, capital=$%.2f",
             len(pipeline.profiles), pipeline.initial_capital)

    # 4. Загрузка persisted state
    _load_or_migrate_state(
        perf=pipeline.perf,
        snapshot_path=snapshot_path,
        migrate_from_v1=migrate_from_v1,
        exchange_name=exchange,
    )
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=pipeline.registry,
        perf=pipeline.perf,
        risk_config=risk_config,
    )
    log.info(
        "[%s] production shadow tournament enabled: agents=%d profiles=%d",
        exchange,
        len(pipeline.registry),
        len(pipeline.profiles),
    )

    # 5. OutputWriter — обязательно для видимости работы
    if mode != "paper":
        _recover_exchange_positions(pipeline)

    writer = OutputWriter.for_session(pipeline, results_root=results_root)
    log.info("Output directory: %s", writer.output_dir)

    # 6. Feed selection: v1-bridge > PollingFeed > SyntheticFeed (paper) > empty
    # paper-mode сразу идёт в SyntheticFeed, минуя v1-bridge.
    if use_v1_bridge and mode != "paper":
        try:
            from .v1_bridge_runner import run_with_v1_bridge
            log.info("[%s] starting via v1-bridge integration", exchange)
            rc = run_with_v1_bridge(
                pipeline,
                exchange_name=exchange,
                mode=mode,
                output_writer=writer,
                max_bars=max_bars,
                max_idle_polls=max_idle_polls,
                sleep_between_polls_sec=sleep_between_polls_sec,
                warmup_bars=warmup_bars,
            )
            # Сохраняем snapshot
            if snapshot_path:
                save_v2_snapshot(pipeline.perf, snapshot_path)
                log.info("Saved snapshot → %s", snapshot_path)
            if rc == 0:
                return rc
            log.warning("v1-bridge returned rc=%s; falling back to local feed", rc)
        except Exception as exc:
            log.warning("v1-bridge runner failed (%s) — falling back to empty feed",
                        type(exc).__name__)

    # Fallback path: PollingFeed или empty
    if polling_session_dir:
        feed: MarketFeed = PollingFeed(session_dir=polling_session_dir)
        log.info("Using PollingFeed: %s", polling_session_dir)
    elif mode == "paper":
        # paper-mode → smoke через SyntheticFeed (random-walk цены)
        feed = SyntheticFeed(
            symbols=["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE"],
            max_bars=max_bars,
        )
        log.warning(
            "[%s] paper mode → SyntheticFeed (synthetic prices for "
            "infrastructure verification, NOT production)", exchange)
        # SyntheticFeed follows sleep_between_polls_sec from the caller.
    else:
        feed = ReplayFeed()
        log.warning(
            "No feed configured and live-bridge unavailable — main_loop "
            "will idle. Pass polling_session_dir or set mode='paper' for "
            "synthetic feed.")

    n_step = [0]
    def _on_step(step):
        n_step[0] += 1
        try:
            writer.write(step)
        except Exception:
            log.exception("output_writer.write failed")
        if n_step[0] % 10 == 0:
            log.info("  bar=%d leader=%s signals=%d filled=%d",
                     step.bar, step.leader or "-",
                     step.n_signals, step.n_filled)

    def _on_error(exc):
        log.error("main_loop error: %s", exc, exc_info=True)

    def _on_idle(idle_polls: int):
        if idle_polls == 1 or idle_polls % 12 == 0:
            log.info(
                "waiting for market feed: idle_poll=%d output=%s",
                idle_polls,
                writer.output_dir,
            )
        try:
            writer.write_heartbeat(
                run_state="idle",
                feed_status="waiting_for_market_bar",
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
        log.info("KeyboardInterrupt — saving state and exit")
        steps = []

    writer.close()
    if snapshot_path:
        save_v2_snapshot(pipeline.perf, snapshot_path)
        log.info("Saved snapshot → %s", snapshot_path)
    log.info("Shutdown. Processed %d bars.", len(steps))
    return 0
