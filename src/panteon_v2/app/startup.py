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
from pathlib import Path
from typing import List, Optional, Sequence

from ..execution import Exchange, FakeExchange
from ..selection import AgentRegistry, PlayerProfile
from ..shadow.feed import MarketFeed, PollingFeed, ReplayFeed
from ..shadow.synthetic_feed import SyntheticFeed
from .agent_bootstrap import register_all_v1_agents
from .bootstrap import PRODUCTION_PROFILES, build_production_pipeline
from .main_loop import main_loop
from .migration import load_v2_snapshot, migrate_from_v1_memory_file, save_v2_snapshot
from .output_writer import OutputWriter, OutputWriterConfig


log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Exchange adapter resolution
# ────────────────────────────────────────────────────────────────────


def resolve_exchange(exchange_name: str, *, mode: str) -> Exchange:
    """Реальный Exchange-адаптер или FakeExchange.

    Когда оператор реализует bitget_adapter.py / mexc_adapter.py —
    они подхватятся автоматически.
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


# ────────────────────────────────────────────────────────────────────
# Main entry: start_production
# ────────────────────────────────────────────────────────────────────


def start_production(
    *,
    exchange:           str,
    mode:               str = "live_futures",
    initial_capital:    float = 100.0,
    seed_quarantine:    Sequence[str] = ("FundingArb", "RichardDennis", "MomentumScalper"),
    profiles:           Optional[Sequence[PlayerProfile]] = None,
    polling_session_dir: Optional[str] = None,
    snapshot_path:      Optional[str] = None,
    migrate_from_v1:    Optional[str] = None,
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

    # 1. Регистрируем v1-агенты
    registry = AgentRegistry()
    registered = register_all_v1_agents(registry, skip_on_error=True)
    log.info("Registered %d v1-agents: %s", len(registered),
             ", ".join(registered[:5]) + ("…" if len(registered) > 5 else ""))
    if not registered:
        log.error(
            "No v1-agents registered. Check sys.path / panteon_runtime presence."
        )
        return 2

    # 2. Exchange adapter
    exchange_adapter = resolve_exchange(exchange, mode=mode)

    # 3. ProductionPipeline
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange_adapter,
        initial_capital=initial_capital,
        seed_quarantine=seed_quarantine,
        profiles=profiles or PRODUCTION_PROFILES,
        jsonl_event_log=jsonl_event_log,
    )
    log.info("Pipeline built: %d profiles, capital=$%.2f",
             len(pipeline.profiles), pipeline.initial_capital)

    # 4. Загрузка persisted state
    if snapshot_path and os.path.exists(snapshot_path):
        if load_v2_snapshot(pipeline.perf, snapshot_path):
            log.info("Loaded v2-snapshot from %s", snapshot_path)
    elif migrate_from_v1 and os.path.exists(migrate_from_v1):
        report = migrate_from_v1_memory_file(migrate_from_v1, pipeline.perf)
        log.info("Migrated from v1: %d labels, %d pairs",
                 report.n_labels, report.n_regime_pairs)

    # 5. OutputWriter — обязательно для видимости работы
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
