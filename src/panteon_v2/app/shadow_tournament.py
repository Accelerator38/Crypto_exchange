"""Production shadow tournament for Panteon v2.

All registered agents and composed player candidates trade virtually on every
market bar. Virtual order events stay local to each actor runtime; only
PerformanceMemory is shared, so ratings/quarantine learn from shadow trading
without polluting real exchange ledgers or open-position views.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Sequence

from ..attribution import EventLog
from ..domain.types import Action, MarketSnapshot, Signal
from ..execution import (
    ExecutionResult,
    ExecutionStatus,
    FakeExchange,
    PositionTracker,
    RiskLimits,
    RiskLimitsConfig,
    SymbolHealthMonitor,
    TradeExecutor,
)
from ..memory import PerformanceMemory
from ..selection import Agent, AgentRegistry, Player


log = logging.getLogger(__name__)

_SHADOW_SIGNAL_ID_START = 1_000_000_000


@dataclass(frozen=True)
class ShadowStepSummary:
    agent_signals: int = 0
    agent_filled: int = 0
    agent_rejected: int = 0
    agent_blocked: int = 0
    player_signals: int = 0
    player_filled: int = 0
    player_rejected: int = 0
    player_blocked: int = 0
    actors: int = 0

    @property
    def total_signals(self) -> int:
        return self.agent_signals + self.player_signals

    @property
    def total_filled(self) -> int:
        return self.agent_filled + self.player_filled

    @property
    def total_rejected(self) -> int:
        return self.agent_rejected + self.player_rejected

    @property
    def total_blocked(self) -> int:
        return self.agent_blocked + self.player_blocked

    def as_dict(self) -> dict:
        data = asdict(self)
        data.update({
            "total_signals": self.total_signals,
            "total_filled": self.total_filled,
            "total_rejected": self.total_rejected,
            "total_blocked": self.total_blocked,
        })
        return data


class _VirtualActorRuntime:
    def __init__(
        self,
        *,
        actor_label: str,
        perf: PerformanceMemory,
        risk_config: RiskLimitsConfig,
    ) -> None:
        self.actor_label = actor_label
        self._executor = TradeExecutor(
            exchange=FakeExchange(name=f"SHADOW-{actor_label}"),
            health=SymbolHealthMonitor(),
            risk_limits=RiskLimits(config=risk_config),
            position_tracker=PositionTracker(),
            perf=perf,
            event_log=EventLog(),
        )

    def execute_many(
        self,
        signals: Sequence[Signal],
        *,
        balance_usd: float,
    ) -> List[ExecutionResult]:
        out: List[ExecutionResult] = []
        for signal in signals:
            try:
                out.append(self._executor.execute(signal, balance_usd=balance_usd))
            except Exception:
                log.debug("shadow execute failed for %s", self.actor_label, exc_info=True)
        return out


class ProductionShadowTournament:
    """Stateful virtual tournament used by the live production loop."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        perf: PerformanceMemory,
        risk_config: RiskLimitsConfig,
        virtual_balance_floor: float = 1000.0,
    ) -> None:
        self._registry = registry
        self._perf = perf
        self._risk_config = risk_config
        self._balance_floor = float(virtual_balance_floor)
        self._runtimes: Dict[str, _VirtualActorRuntime] = {}
        self._signal_id = _SHADOW_SIGNAL_ID_START

    def run_bar(
        self,
        market: MarketSnapshot,
        *,
        players: Sequence[Player],
        balance_usd: float,
    ) -> ShadowStepSummary:
        virtual_balance = max(float(balance_usd or 0.0), self._balance_floor)
        agent_summary = self._run_agents(market, balance_usd=virtual_balance)
        player_summary = self._run_players(market, players, balance_usd=virtual_balance)
        return ShadowStepSummary(
            agent_signals=agent_summary["signals"],
            agent_filled=agent_summary["filled"],
            agent_rejected=agent_summary["rejected"],
            agent_blocked=agent_summary["blocked"],
            player_signals=player_summary["signals"],
            player_filled=player_summary["filled"],
            player_rejected=player_summary["rejected"],
            player_blocked=player_summary["blocked"],
            actors=len(self._registry.all_labels()) + len(players),
        )

    def _run_agents(self, market: MarketSnapshot, *, balance_usd: float) -> Dict[str, int]:
        counts = _empty_counts()
        for agent in self._registry.all_agents():
            signals = self._signals_from_agent(agent, market)
            counts["signals"] += len(signals)
            _add_execution_counts(
                counts,
                self._runtime(f"agent:{agent.label}").execute_many(
                    signals,
                    balance_usd=balance_usd,
                ),
            )
        return counts

    def _run_players(
        self,
        market: MarketSnapshot,
        players: Sequence[Player],
        *,
        balance_usd: float,
    ) -> Dict[str, int]:
        counts = _empty_counts()
        for player in players:
            try:
                raw_signals = player.vote(market, signal_id_start=self._signal_id)
            except Exception:
                log.debug("shadow player vote failed for %s", player.label, exc_info=True)
                raw_signals = []
            if raw_signals:
                self._signal_id = max(signal.id for signal in raw_signals) + 1
            signals = [
                replace(
                    signal,
                    by_agent="",
                    position_scope=f"shadow:player:{player.label}",
                )
                for signal in raw_signals
            ]
            counts["signals"] += len(signals)
            _add_execution_counts(
                counts,
                self._runtime(f"player:{player.label}").execute_many(
                    signals,
                    balance_usd=balance_usd,
                ),
            )
        return counts

    def _signals_from_agent(self, agent: Agent, market: MarketSnapshot) -> List[Signal]:
        try:
            actions = agent.act(market)
        except Exception:
            log.debug("shadow agent act failed for %s", agent.label, exc_info=True)
            return []

        signals: List[Signal] = []
        for sym, action in (actions or {}).items():
            if sym not in market.prices:
                continue
            if not isinstance(action, Action):
                try:
                    action = Action(int(action))
                except (TypeError, ValueError):
                    continue
            if action.is_hold:
                continue
            signals.append(Signal(
                id=self._next_signal_id(),
                bar=market.bar,
                sym=sym,
                action=action,
                price=float(market.prices.get(sym, 0.0)),
                regime=market.regime,
                by_player=agent.label,
                by_agent=agent.label,
                position_scope=f"shadow:agent:{agent.label}",
                timestamp=market.timestamp,
            ))
        return signals

    def _next_signal_id(self) -> int:
        signal_id = self._signal_id
        self._signal_id += 1
        return signal_id

    def _runtime(self, actor_key: str) -> _VirtualActorRuntime:
        runtime = self._runtimes.get(actor_key)
        if runtime is None:
            runtime = _VirtualActorRuntime(
                actor_label=actor_key,
                perf=self._perf,
                risk_config=self._risk_config,
            )
            self._runtimes[actor_key] = runtime
        return runtime


def _empty_counts() -> Dict[str, int]:
    return {"signals": 0, "filled": 0, "rejected": 0, "blocked": 0}


def _add_execution_counts(counts: Dict[str, int], results: Iterable[ExecutionResult]) -> None:
    for result in results:
        if result.status == ExecutionStatus.FILLED:
            counts["filled"] += 1
        elif result.status == ExecutionStatus.REJECTED:
            counts["rejected"] += 1
        elif result.status == ExecutionStatus.BLOCKED:
            counts["blocked"] += 1
