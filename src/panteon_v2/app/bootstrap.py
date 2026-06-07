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
from ..memory import (
    DegradationGate,
    DegradationGateConfig,
    PerformanceMemory,
    QuarantineManager,
)
from ..scoring import DEFAULT_SCORING, ScoringConfig
from ..selection import (
    AgentRegistry,
    AgentSelector,
    FlashAllocator,
    FlashAllocatorConfig,
    PROFILE_BOMBERMAN_STRONG,
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_GENETICS_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_NEUTRAL_EDGE_RESEARCH,
    PROFILE_TREND_RESEARCH,
    PlayerComposer,
    PlayerProfile,
    SessionOverlayConfig,
    Strategist,
    StrategistConfig,
)


def _parse_label_value_map(raw: object, *, numeric_type: type) -> Dict[str, object]:
    if isinstance(raw, dict):
        iterable = raw.items()
    elif isinstance(raw, str):
        iterable = (
            part.strip()
            for part in raw.replace(";", ",").split(",")
            if part.strip()
        )
    else:
        try:
            iterable = tuple(raw or ())  # type: ignore[arg-type]
        except TypeError:
            iterable = ()
    parsed: Dict[str, object] = {}
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2:
            label, value = item
        elif isinstance(item, str):
            sep = "=" if "=" in item else ":" if ":" in item else ""
            if not sep:
                continue
            label, value = item.split(sep, 1)
        else:
            continue
        clean_label = str(label or "").strip()
        if not clean_label:
            continue
        try:
            parsed[clean_label] = numeric_type(value)
        except (TypeError, ValueError):
            continue
    return parsed


@dataclass(frozen=True)
class LiveExecutionConfig:
    """Guardrails for real exchange execution."""

    max_new_opens_per_bar: int = 1
    max_daily_loss_pct: float = 0.0
    max_equity_peak_drawdown_pct: float = 0.0
    max_consecutive_failed_orders: int = 5
    max_exchange_desync_events: int = 0
    max_stale_feed_polls: int = 0
    max_slippage_pct: float = 0.0
    max_api_error_streak: int = 5
    pending_order_timeout_sec: float = 180.0
    # Kill-switch auto-recovery (Phase 0 / A7). По умолчанию ВЫКЛЮЧЕНО —
    # включение явно через конфиг, т.к. это safety-critical поведение.
    # При включении защёлка kill-switch может сняться после восстановления
    # equity и истечения cooldown (вместо вечного manage-only).
    kill_switch_auto_recovery_enabled: bool = False
    kill_switch_recovery_cooldown_bars: int = 0
    # Максимальная остаточная просадка от пика (%), при которой допускается
    # снятие equity-защёлки. 0 = требовать полного восстановления до пика/initial.
    kill_switch_recovery_max_drawdown_pct: float = 0.0
    adopt_existing_positions_enabled: bool = False
    adopt_existing_position_symbols: tuple[str, ...] = ()
    adopt_existing_position_player: str = "PanteonFlashAdopted"
    adopt_existing_position_agent: str = "AdoptedExchangePosition"
    genetics_probation_execution_enabled: bool = False
    genetics_probation_labels: tuple[str, ...] = ("GeneticsResearch",)
    genetics_probation_allowed_regimes: tuple[str, ...] = ("bearish", "crash")
    genetics_probation_allowed_signal_keys: tuple[str, ...] = ()
    genetics_probation_risk_mult: float = 0.25
    genetics_probation_min_regime_confidence: float = 0.0
    genetics_probation_max_real_trades: int = 20
    genetics_probation_max_daily_trades: int = 0
    genetics_probation_require_shadow_confirmation: bool = True
    genetics_probation_max_consecutive_failed_orders: int = 2
    genetics_probation_max_realized_loss_pct: float = 0.25
    genetics_probation_risk_mult_by_label: Dict[str, float] = field(default_factory=dict)
    genetics_probation_max_real_trades_by_label: Dict[str, int] = field(default_factory=dict)
    genetics_probation_max_daily_trades_by_label: Dict[str, int] = field(default_factory=dict)
    genetics_probation_max_consecutive_failed_orders_by_label: Dict[str, int] = field(default_factory=dict)
    genetics_probation_max_realized_loss_pct_by_label: Dict[str, float] = field(default_factory=dict)
    max_real_symbol_min_executable_notional_usd: float = 40.0

    def __post_init__(self) -> None:
        if self.max_new_opens_per_bar < 0:
            raise ValueError("max_new_opens_per_bar must be >= 0")
        for name in (
            "max_daily_loss_pct",
            "max_equity_peak_drawdown_pct",
            "max_slippage_pct",
            "pending_order_timeout_sec",
            "max_real_symbol_min_executable_notional_usd",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 < self.genetics_probation_risk_mult <= 1.0:
            raise ValueError("genetics_probation_risk_mult must be in (0, 1]")
        if not 0.0 <= self.genetics_probation_min_regime_confidence <= 1.0:
            raise ValueError("genetics_probation_min_regime_confidence must be in [0, 1]")
        if self.genetics_probation_max_real_trades < 0:
            raise ValueError("genetics_probation_max_real_trades must be >= 0")
        if self.genetics_probation_max_daily_trades < 0:
            raise ValueError("genetics_probation_max_daily_trades must be >= 0")
        if self.genetics_probation_max_consecutive_failed_orders < 0:
            raise ValueError("genetics_probation_max_consecutive_failed_orders must be >= 0")
        if self.genetics_probation_max_realized_loss_pct < 0:
            raise ValueError("genetics_probation_max_realized_loss_pct must be >= 0")
        for name, numeric_type in (
            ("genetics_probation_risk_mult_by_label", float),
            ("genetics_probation_max_realized_loss_pct_by_label", float),
            ("genetics_probation_max_real_trades_by_label", int),
            ("genetics_probation_max_daily_trades_by_label", int),
            ("genetics_probation_max_consecutive_failed_orders_by_label", int),
        ):
            object.__setattr__(
                self,
                name,
                _parse_label_value_map(getattr(self, name), numeric_type=numeric_type),
            )
        for label, risk_mult in self.genetics_probation_risk_mult_by_label.items():
            if not 0.0 < float(risk_mult) <= 1.0:
                raise ValueError(
                    f"genetics_probation_risk_mult_by_label[{label}] must be in (0, 1]"
                )
        for label, loss_pct in self.genetics_probation_max_realized_loss_pct_by_label.items():
            if float(loss_pct) < 0:
                raise ValueError(
                    f"genetics_probation_max_realized_loss_pct_by_label[{label}] must be >= 0"
                )
        for name in (
            "genetics_probation_max_real_trades_by_label",
            "genetics_probation_max_daily_trades_by_label",
            "genetics_probation_max_consecutive_failed_orders_by_label",
        ):
            for label, value in getattr(self, name).items():
                if int(value) < 0:
                    raise ValueError(f"{name}[{label}] must be >= 0")
        for name in (
            "genetics_probation_labels",
            "genetics_probation_allowed_regimes",
            "genetics_probation_allowed_signal_keys",
            "adopt_existing_position_symbols",
        ):
            value = getattr(self, name)
            if isinstance(value, str):
                normalized = tuple(
                    item.strip()
                    for item in value.replace(";", ",").split(",")
                    if item.strip()
                )
            else:
                normalized = tuple(
                    str(item).strip()
                    for item in (value or ())
                    if str(item).strip()
                )
            object.__setattr__(self, name, normalized)
        for name, default in (
            ("adopt_existing_position_player", "PanteonFlashAdopted"),
            ("adopt_existing_position_agent", "AdoptedExchangePosition"),
        ):
            value = str(getattr(self, name, "") or "").strip() or default
            object.__setattr__(self, name, value)
        for name in (
            "max_consecutive_failed_orders",
            "max_exchange_desync_events",
            "max_stale_feed_polls",
            "max_api_error_streak",
            "genetics_probation_max_consecutive_failed_orders",
            "kill_switch_recovery_cooldown_bars",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.kill_switch_recovery_max_drawdown_pct < 0:
            raise ValueError("kill_switch_recovery_max_drawdown_pct must be >= 0")


@dataclass
class KillSwitchState:
    disabled_reason: str = ""
    peak_equity_usd: float = 0.0
    consecutive_failed_orders: int = 0
    api_error_streak: int = 0
    exchange_desync_events: int = 0
    stale_feed_polls: int = 0
    # Метаданные защёлки для авто-recovery (Phase 0 / A7)
    disabled_bar: int = -1
    disabled_kind: str = ""
    last_seen_bar: int = 0
    genetics_probation_disabled_reason: str = ""
    genetics_probation_consecutive_failed_orders: int = 0
    genetics_probation_disabled_reasons_by_label: Dict[str, str] = field(default_factory=dict)
    genetics_probation_consecutive_failed_orders_by_label: Dict[str, int] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────
# Готовый набор profile-ов для production
# ────────────────────────────────────────────────────────────────────

PRODUCTION_PROFILES: List[PlayerProfile] = [
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_NEUTRAL_EDGE_RESEARCH,
    PROFILE_TREND_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_GENETICS_RESEARCH,
    PROFILE_BOMBERMAN_STRONG,
]

DEFAULT_REGIME_SWITCH_PLAYER_SETS: tuple[tuple[str, Dict[str, object]], ...] = (
    (
        "Antonius_conservative",
        {
            "bullish": "VolBreakoutHunter",
            "bearish": "FundingArb",
            "neutral": "GeneticsCore",
            "crash": "CrashPanicShortAgent",
        },
    ),
    (
        "Perfect_OIBreakout",
        {
            "bullish": "LiveOIBreakout",
            "neutral": "LiveOIBreakout",
            "bearish": "LiveOIBreakout",
            "crash": "LiveOIBreakout",
        },
    ),
    (
        "Perfect_CrashSwitch",
        {
            "bearish": "LiveCrashHunter",
            "crash": "LiveCrashHunter",
        },
    ),
    (
        "Perfect_NeutralValidator",
        {
            "neutral": "ResearchValidatorAgent",
        },
    ),
    (
        "Perfect_GeneticsBearCrash",
        {
            "bearish": "GeneticsCore",
            "crash": "GeneticsCore",
        },
    ),
    (
        "Perfect_MeanRev",
        {
            "neutral": "LiveMeanRev",
        },
    ),
)


# ────────────────────────────────────────────────────────────────────
# Production composite
# ────────────────────────────────────────────────────────────────────


DEFAULT_ROTATING_AGENT_PLAYER_SETS: tuple[
    tuple[str, Dict[str, tuple[str, ...]], tuple[str, ...]],
    ...
] = (
    (
        "Optimal_StaticRotator",
        {
            "bullish": (
                "LiveAfterShock",
                "VolBreakoutHunter",
                "LiveVolCompress",
                "BullRotationAgent",
                "LiveOIBreakout",
                "MomentumScalper",
            ),
            "bearish": (
                "MomentumScalper",
                "GeneticsCore",
                "LiveCrashHunter",
                "ResearchValidatorAgent",
                "FundingArb",
                "LiveRegimePullback",
            ),
            "neutral": (
                "ResearchValidatorAgent",
                "GeneticsCore",
                "NeutralLiquiditySweep",
                "NeutralRangeScalper",
                "LiveAfterShock",
            ),
            "crash": (
                "LiveCrashHunter",
                "MomentumScalper",
                "LiveRegimePullback",
                "FundingArb",
                "CrashPanicShortAgent",
                "LiveOIBreakout",
            ),
        },
        (
            "ResearchValidatorAgent",
            "LiveCrashHunter",
            "MomentumScalper",
            "LiveOIBreakout",
            "GeneticsCore",
        ),
    ),
)


def _protected_composite_labels() -> tuple[str, ...]:
    labels = [
        label
        for label, mapping in DEFAULT_REGIME_SWITCH_PLAYER_SETS
        if len({
            str(regime).strip().lower()
            for regime in mapping
            if str(regime).strip()
        }) > 1
    ]
    labels.extend(label for label, _mapping, _fallback in DEFAULT_ROTATING_AGENT_PLAYER_SETS)
    return tuple(dict.fromkeys(labels))


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
    degradation_gate: DegradationGate
    selector:    AgentSelector
    composer:    PlayerComposer
    strategist:  Strategist
    flash_allocator: Optional[FlashAllocator]
    executor:    TradeExecutor
    event_log:   EventLog
    ledger:      AttributionLedger
    order_ledger: OrderLedger
    renderer:    DashboardRenderer
    risk_config: RiskLimitsConfig

    # Метаданные
    profiles:        List[PlayerProfile]
    initial_capital: float
    exchange_name:   str
    mode:            str = ""
    timeframe:       str = ""
    run_id:          str = ""
    session_id:      str = ""
    regime_switch_player_sets: tuple[tuple[str, Dict[str, object]], ...] = field(
        default_factory=lambda: DEFAULT_REGIME_SWITCH_PLAYER_SETS
    )
    rotating_agent_player_sets: tuple[
        tuple[str, Dict[str, tuple[str, ...]], tuple[str, ...]],
        ...
    ] = field(default_factory=lambda: DEFAULT_ROTATING_AGENT_PLAYER_SETS)

    # Состояние loop
    current_balance: float = 0.0
    account_snapshot: Optional[Dict[str, float]] = None
    shadow_tournament: Optional[object] = None
    shadow_last_summary: Optional[dict] = None
    live_execution: LiveExecutionConfig = field(default_factory=LiveExecutionConfig)
    kill_switch: KillSwitchState = field(default_factory=KillSwitchState)
    flash_enabled: bool = False
    shadow_agent_labels: tuple[str, ...] = ()


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
    flash_enabled:     bool = False,
    flash_allocator_config: Optional[FlashAllocatorConfig] = None,
    degradation_config: Optional[DegradationGateConfig] = None,
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
    flash_allocator_config = flash_allocator_config or FlashAllocatorConfig()
    degradation_config = degradation_config or DegradationGateConfig()

    event_log = EventLog(
        jsonl_path=jsonl_event_log,
        raise_on_persist_error=False,
        persist_retry_attempts=3,
        persist_retry_delay_sec=0.05,
    )
    exchange_scope = getattr(exchange, "name", "UNKNOWN")
    virtual_perf = PerformanceMemory(
        trade_fraction=perf_trade_fraction,
        exchange_scope=exchange_scope,
    )
    real_perf = PerformanceMemory(
        trade_fraction=perf_trade_fraction,
        exchange_scope=exchange_scope,
    )
    qm = QuarantineManager(
        seed=set(seed_quarantine),
        config=scoring_config,
        protected_labels=_protected_composite_labels(),
    )
    degradation_gate = DegradationGate(degradation_config)
    selector = AgentSelector(
        registry,
        virtual_perf,
        qm,
        config=scoring_config,
        session_overlay=SessionOverlayConfig(enabled=True),
    )
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
        real_perf=real_perf,
    )
    flash_allocator = FlashAllocator(
        perf=virtual_perf,
        qm=qm,
        config=flash_allocator_config,
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
        degradation_gate=degradation_gate,
        selector=selector,
        composer=composer,
        strategist=strategist,
        flash_allocator=flash_allocator,
        executor=executor,
        event_log=event_log,
        ledger=ledger,
        order_ledger=order_ledger,
        renderer=renderer,
        risk_config=risk_config,
        profiles=profiles,
        initial_capital=initial_capital,
        exchange_name=getattr(exchange, "name", "UNKNOWN"),
        current_balance=initial_capital,
        live_execution=live_execution_config,
        flash_enabled=bool(flash_enabled),
    )


def build_dryrun_pipeline(
    *,
    registry:        AgentRegistry,
    initial_capital: float = 1000.0,
    seed_quarantine: Sequence[str] = (),
    profiles:        Optional[Sequence[PlayerProfile]] = None,
    flash_enabled:   bool = False,
    flash_allocator_config: Optional[FlashAllocatorConfig] = None,
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
        flash_enabled=flash_enabled,
        flash_allocator_config=flash_allocator_config,
    )
