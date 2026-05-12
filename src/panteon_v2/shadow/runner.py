"""ShadowRunner — главный orchestrator dry-run pipeline.

Каждый bar:
  1. feed.next_bar() → MarketSnapshot
  2. (раз в N баров) qm.recompute(perf) → notify subscribers
  3. composer.compose_from_profile(...) → ensemble (+ optional профильные игроки)
  4. strategist.consider_switch(regime, bar) → leader
  5. leader.vote(market) → signals
  6. для каждого signal: executor.execute(s) — на FakeExchange (dry-run)
  7. ledger.replay_from_event_log(event_log) → актуальная атрибуция

Всё под одной крышей, один лог, никаких параллельных state. После
пробега — DashboardRenderer выдаёт MainDashboardData для сравнения с v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from ..attribution import (
    AttributionLedger,
    BarStarted,
    EventLog,
    LeaderSelected,
    QuarantineRecomputed,
    RegimeDetected,
)
from ..domain.types import MarketSnapshot, Regime, Signal
from ..execution import (
    ExecutionResult,
    FakeExchange,
    PositionTracker,
    RiskLimits,
    SymbolHealthMonitor,
    TradeExecutor,
)
from ..memory import PerformanceMemory, QuarantineManager
from ..selection import (
    AgentRegistry,
    AgentSelector,
    EnsemblePlayer,
    PlayerComposer,
    PlayerProfile,
    PROFILE_DEFAULT_ENSEMBLE,
    Strategist,
    StrategistConfig,
    SwitchDecision,
)
from .feed import MarketFeed


# ────────────────────────────────────────────────────────────────────
# Step result
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StepResult:
    """Что произошло за один bar в shadow-режиме."""

    bar:        int
    regime:     Regime
    leader:     Optional[str]                 # label выбранного лидера
    leader_changed: bool
    signals:    List[Signal] = field(default_factory=list)
    executions: List[ExecutionResult] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# ShadowRunner
# ────────────────────────────────────────────────────────────────────


class ShadowRunner:
    """Один runner — один dry-run pipeline.

    Пример:
        feed = ReplayFeed([snap1, snap2, ...])
        runner = ShadowRunner.with_defaults(
            registry=registry,
            feed=feed,
            seed_quarantine={"FundingArb"},
            profiles=[PROFILE_DEFAULT_ENSEMBLE, PROFILE_TREND_RESEARCH],
            initial_capital=1000.0,
        )
        while True:
            step = runner.step()
            if step is None: break
            print(step.bar, step.leader, len(step.signals))
        # После завершения — runner.ledger даёт REAL attribution
    """

    def __init__(
        self,
        *,
        registry:             AgentRegistry,
        feed:                 MarketFeed,
        composer:             PlayerComposer,
        strategist:           Strategist,
        executor:             TradeExecutor,
        perf:                 PerformanceMemory,
        qm:                   QuarantineManager,
        event_log:            EventLog,
        ledger:               AttributionLedger,
        balance_usd:          float = 1000.0,
        profiles:             Sequence[PlayerProfile] = (PROFILE_DEFAULT_ENSEMBLE,),
        recompute_quarantine_every: int = 10,
        signal_id_start:      int = 1,
    ):
        self._registry = registry
        self._feed = feed
        self._composer = composer
        self._strategist = strategist
        self._executor = executor
        self._perf = perf
        self._qm = qm
        self._event_log = event_log
        self._ledger = ledger
        self._balance = float(balance_usd)
        self._profiles = list(profiles)
        self._recompute_every = max(1, int(recompute_quarantine_every))
        self._signal_id = int(signal_id_start)
        self._last_qm_bar = -10**9
        self._last_regime: Optional[Regime] = None

    # ── Factory: shadow runner с дефолтными компонентами ────────────

    @classmethod
    def with_defaults(
        cls,
        *,
        registry:           AgentRegistry,
        feed:               MarketFeed,
        seed_quarantine:    Sequence[str] = (),
        profiles:           Sequence[PlayerProfile] = (PROFILE_DEFAULT_ENSEMBLE,),
        balance_usd:        float = 1000.0,
        strategist_config:  Optional[StrategistConfig] = None,
    ) -> "ShadowRunner":
        """Создаёт ShadowRunner с разумными defaults для всех компонентов."""
        event_log = EventLog()
        perf = PerformanceMemory(trade_fraction=0.10)
        qm = QuarantineManager(seed=set(seed_quarantine))
        selector = AgentSelector(registry, perf, qm)
        composer = PlayerComposer(selector)
        ledger = AttributionLedger()
        # candidates пока пустой — runner сам пересоберёт на каждом баре.
        strategist = Strategist(
            perf, qm, candidates=[],
            config=strategist_config or StrategistConfig(),
        )
        executor = TradeExecutor(
            exchange=FakeExchange(),
            health=SymbolHealthMonitor(),
            risk_limits=RiskLimits(),
            position_tracker=PositionTracker(),
            perf=perf,
            event_log=event_log,
        )
        return cls(
            registry=registry,
            feed=feed,
            composer=composer,
            strategist=strategist,
            executor=executor,
            perf=perf,
            qm=qm,
            event_log=event_log,
            ledger=ledger,
            balance_usd=balance_usd,
            profiles=profiles,
        )

    # ── Public API ──────────────────────────────────────────────────

    def step(self) -> Optional[StepResult]:
        """Один bar pipeline. Возвращает None если feed исчерпан."""
        market = self._feed.next_bar()
        if market is None:
            return None

        trace = f"shadow-{market.bar}"
        self._event_log.emit(BarStarted(bar=market.bar, trace_id=trace))

        # Регим detected
        regime_changed = self._last_regime != market.regime
        self._event_log.emit(RegimeDetected(
            bar=market.bar, trace_id=trace,
            regime=market.regime,
            from_regime=self._last_regime or market.regime,
            is_change=regime_changed,
        ))
        self._last_regime = market.regime

        # Quarantine recompute (раз в N баров)
        if market.bar - self._last_qm_bar >= self._recompute_every:
            previous = self._qm.all_quarantined()
            result = self._qm.recompute(self._perf)
            self._last_qm_bar = market.bar
            if not result.is_no_op:
                self._event_log.emit(QuarantineRecomputed(
                    bar=market.bar, trace_id=trace,
                    added=result.added,
                    removed=result.removed,
                    current=result.current,
                ))

        # Compose candidate-players по profiles
        candidates: List[EnsemblePlayer] = []
        for profile in self._profiles:
            p = self._composer.compose_from_profile_with_fallback(
                profile, market.regime,
            )
            if p is not None:
                candidates.append(p)
        # Если ни один профиль не дал композицию — нечего делать на этом баре
        if not candidates:
            return StepResult(
                bar=market.bar,
                regime=market.regime,
                leader=None,
                leader_changed=False,
            )

        # Strategist: выбирает лидера
        self._strategist.update_candidates(candidates)
        try:
            decision: SwitchDecision = self._strategist.consider_switch(
                market.regime, current_bar=market.bar,
            )
        except ValueError:
            # Все кандидаты дисквалифицированы (например, все используют
            # карантинных) — пропускаем bar.
            return StepResult(
                bar=market.bar,
                regime=market.regime,
                leader=None,
                leader_changed=False,
            )

        if decision.switched:
            self._event_log.emit(LeaderSelected(
                bar=market.bar, trace_id=trace,
                player_label=decision.new_leader.label,
                previous_label=(decision.previous.label if decision.previous else ""),
                score=decision.score,
                margin=decision.margin,
                is_urgent=decision.is_urgent,
                reason=decision.reason,
            ))

        # Лидер голосует
        leader = decision.new_leader
        signals = leader.vote(market, signal_id_start=self._signal_id)
        # Сдвигаем счётчик
        if signals:
            self._signal_id = max(s.id for s in signals) + 1

        # Каждый signal → executor (dry-run на FakeExchange)
        executions: List[ExecutionResult] = []
        for sig in signals:
            res = self._executor.execute(sig, balance_usd=self._balance)
            executions.append(res)

        return StepResult(
            bar=market.bar,
            regime=market.regime,
            leader=leader.label,
            leader_changed=decision.switched,
            signals=signals,
            executions=executions,
        )

    def run_until_exhausted(self, *, max_steps: int = 100_000) -> List[StepResult]:
        """Прогон до конца feed-а или max_steps."""
        out: List[StepResult] = []
        for _ in range(max_steps):
            step = self.step()
            if step is None:
                break
            out.append(step)
        # После прогона — replay ledger
        self._ledger.replay_from_event_log(self._event_log)
        return out

    # ── Read-only access for downstream tools ──────────────────────

    @property
    def event_log(self) -> EventLog:
        return self._event_log

    @property
    def perf(self) -> PerformanceMemory:
        return self._perf

    @property
    def qm(self) -> QuarantineManager:
        return self._qm

    @property
    def ledger(self) -> AttributionLedger:
        return self._ledger
