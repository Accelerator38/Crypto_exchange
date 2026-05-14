"""Production bootstrap — единственное место сборки всех компонентов.

Это **composition root** в терминах DI. Все зависимости создаются здесь
и пробрасываются явно. Никакие компоненты не лезут друг к другу за
state — только через переданные ссылки.

Использование:
    components = build_production_pipeline(
        registry=my_registry,
        exchange=my_bitget_adapter,
        seed_quarantine={"FundingArb", ...},
        initial_capital=120.0,
    )
    # запустить main loop:
    runner_main_loop(components, feed=my_feed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..attribution import AttributionLedger, EventLog
from ..dashboards import DashboardRenderer
from ..execution import (
    Exchange,
    FakeExchange,
    OrderLedger,
    PositionTracker,
    RiskLimits,
    RiskLimitsConfig,
    SymbolHealthConfig,
    SymbolHealthMonitor,
    TradeExecutor,
)
from ..memory import PerformanceMemory, QuarantineManager
from ..scoring import DEFAULT_SCORING, ScoringConfig
from ..selection import (
    AgentRegistry,
    AgentSelector,
    PROFILE_BOMBERMAN_STRONG,
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_TREND_RESEARCH,
    PlayerComposer,
    PlayerProfile,
    Strategist,
    StrategistConfig,
)


@dataclass(frozen=True)
class LiveExecutionConfig:
    """Guardrails for real exchange execution."""

    max_new_opens_per_bar: int = 1
    max_daily_loss_pct: float = 0.0
    max_consecutive_failed_orders: int = 5
    max_exchange_desync_events: int = 0
    max_stale_feed_polls: int = 0
    max_slippage_pct: float = 0.0
    max_api_error_streak: int = 5

    def __post_init__(self) -> None:
        if self.max_new_opens_per_bar < 0:
            raise ValueError("max_new_opens_per_bar must be >= 0")
        for name in (
            "max_daily_loss_pct",
            "max_slippage_pct",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in (
            "max_consecutive_failed_orders",
            "max_exchange_desync_events",
            "max_stale_feed_polls",
            "max_api_error_streak",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class KillSwitchState:
    disabled_reason: str = ""
    consecutive_failed_orders: int = 0
    api_error_streak: int = 0
    exchange_desync_events: int = 0
    stale_feed_polls: int = 0


# ────────────────────────────────────────────────────────────────────
# Готовый набор profile-ов для production
# ────────────────────────────────────────────────────────────────────

PRODUCTION_PROFILES: List[PlayerProfile] = [
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_TREND_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_BOMBERMAN_STRONG,
]


# ────────────────────────────────────────────────────────────────────
# Production composite
# ────────────────────────────────────────────────────────────────────


@dataclass
class ProductionPipeline:
    """Контейнер всех собранных компонентов.

    После build_production_pipeline() — pipeline готов к main_loop.
    Все компоненты — single-instance, переиспользуются bar-to-bar.
    """

    registry:    AgentRegistry
    perf:        PerformanceMemory
    virtual_perf: PerformanceMemory
    real_perf:   PerformanceMemory
    qm:          QuarantineManager
    selector:    AgentSelector
    composer:    PlayerComposer
    strategist:  Strategist
    executor:    TradeExecutor
    event_log:   EventLog
    ledger:      AttributionLedger
    order_ledger: OrderLedger
    renderer:    DashboardRenderer

    # Метаданные
    profiles:        List[PlayerProfile]
    initial_capital: float
    exchange_name:   str

    # Состояние loop
    current_balance: float = 0.0
    account_snapshot: Optional[Dict[str, float]] = None
    shadow_tournament: Optional[object] = None
    shadow_last_summary: Optional[dict] = None
    live_execution: LiveExecutionConfig = field(default_factory=LiveExecutionConfig)
    kill_switch: KillSwitchState = field(default_factory=KillSwitchState)


# ────────────────────────────────────────────────────────────────────
# Builder
# ────────────────────────────────────────────────────────────────────


def build_production_pipeline(
    *,
    registry:           AgentRegistry,
    exchange:           Exchange,
    initial_capital:    float,
    seed_quarantine:    Sequence[str] = (),
    profiles:           Optional[Sequence[PlayerProfile]] = None,
    scoring_config:     ScoringConfig = DEFAULT_SCORING,
    strategist_config:  Optional[StrategistConfig] = None,
    risk_config:        Optional[RiskLimitsConfig] = None,
    health_config:      Optional[SymbolHealthConfig] = None,
    live_execution_config: Optional[LiveExecutionConfig] = None,
    perf_trade_fraction: float = 0.10,
    jsonl_event_log:    Optional[str] = None,
) -> ProductionPipeline:
    """Собрать полный production pipeline.

    Параметры:
      registry         — AgentRegistry с уже зарегистрированными агентами
                         (см. agent_bootstrap.py для примера)
      exchange         — реальный Exchange-адаптер (см. exchange_adapter_template.py)
      initial_capital  — стартовый капитал ($)
      seed_quarantine  — начальный набор карантинных лейблов
      profiles         — список PlayerProfile-ов для Strategist (по умолчанию
                         PRODUCTION_PROFILES)
      *_config         — опциональные кастомные конфиги
      jsonl_event_log  — путь, куда писать events JSONL для post-mortem

    Возвращает ProductionPipeline со всеми wired компонентами.
    """
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
    if len(registry) == 0:
        raise ValueError(
            "registry is empty — register agents through agent_bootstrap.py "
            "before build_production_pipeline()"
        )

    profiles = list(profiles or PRODUCTION_PROFILES)
    strategist_config = strategist_config or StrategistConfig()
    risk_config = risk_config or RiskLimitsConfig()
    health_config = health_config or SymbolHealthConfig()
    live_execution_config = live_execution_config or LiveExecutionConfig()

    event_log = EventLog(jsonl_path=jsonl_event_log)
    virtual_perf = PerformanceMemory(trade_fraction=perf_trade_fraction)
    real_perf = PerformanceMemory(trade_fraction=perf_trade_fraction)
    qm = QuarantineManager(seed=set(seed_quarantine), config=scoring_config)
    selector = AgentSelector(registry, virtual_perf, qm, config=scoring_config)
    composer = PlayerComposer(selector)
    health = SymbolHealthMonitor(config=health_config)
    risk_limits = RiskLimits(config=risk_config)
    position_tracker = PositionTracker()
    ledger = AttributionLedger()
    order_ledger = OrderLedger()
    strategist = Strategist(
        virtual_perf, qm, candidates=[],
        config=strategist_config,
        scoring_config=scoring_config,
    )
    executor = TradeExecutor(
        exchange=exchange,
        health=health,
        risk_limits=risk_limits,
        position_tracker=position_tracker,
        perf=real_perf,
        event_log=event_log,
        order_ledger=order_ledger,
    )
    renderer = DashboardRenderer(
        ledger=ledger,
        perf=virtual_perf,
        qm=qm,
        event_log=event_log,
        health=health,
    )

    return ProductionPipeline(
        registry=registry,
        perf=virtual_perf,
        virtual_perf=virtual_perf,
        real_perf=real_perf,
        qm=qm,
        selector=selector,
        composer=composer,
        strategist=strategist,
        executor=executor,
        event_log=event_log,
        ledger=ledger,
        order_ledger=order_ledger,
        renderer=renderer,
        profiles=profiles,
        initial_capital=initial_capital,
        exchange_name=getattr(exchange, "name", "UNKNOWN"),
        current_balance=initial_capital,
        live_execution=live_execution_config,
    )


def build_dryrun_pipeline(
    *,
    registry:        AgentRegistry,
    initial_capital: float = 1000.0,
    seed_quarantine: Sequence[str] = (),
    profiles:        Optional[Sequence[PlayerProfile]] = None,
) -> ProductionPipeline:
    """Аналогично build_production_pipeline, но с FakeExchange.

    Используется для debugging / shadow-run без реальной биржи.
    """
    return build_production_pipeline(
        registry=registry,
        exchange=FakeExchange(name="DRY-RUN"),
        initial_capital=initial_capital,
        seed_quarantine=seed_quarantine,
        profiles=profiles,
    )
