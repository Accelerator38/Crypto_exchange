"""Production shadow tournament for Panteon v2.

All registered agents and composed player candidates trade virtually on every
market bar. Virtual order events stay local to each actor runtime; only
PerformanceMemory is shared, so ratings/quarantine learn from shadow trading
without polluting real exchange ledgers or open-position views.
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..attribution import AgentVoteFailed, EventLog, PlayerVoteFailed, ShadowActorUpdated
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
from ..selection import Agent, AgentRegistry, EnsemblePlayer, Player
from ..selection.player import normalize_vote_result


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

    def open_positions(self) -> Dict[str, object]:
        return self._executor._tracker.all_open()


class ProductionShadowTournament:
    """Stateful virtual tournament used by the live production loop."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        perf: PerformanceMemory,
        risk_config: RiskLimitsConfig,
        virtual_balance_floor: float = 1000.0,
        event_log: Optional[EventLog] = None,
    ) -> None:
        self._source_registry = registry
        self._registry = AgentRegistry()
        self._perf = perf
        self._risk_config = risk_config
        self._balance_floor = float(virtual_balance_floor)
        self._event_log = event_log
        self._runtimes: Dict[str, _VirtualActorRuntime] = {}
        self._source_agents: Dict[str, Agent] = {}
        self._player_agent_clones: Dict[Tuple[str, str], Agent] = {}
        self._player_clones: Dict[str, Player] = {}
        self._signal_id = _SHADOW_SIGNAL_ID_START
        self._last_updates: List[ShadowActorUpdated] = []
        self._last_agent_signals: Dict[str, Tuple[Signal, ...]] = {}
        self._last_player_signals: Dict[str, Tuple[Signal, ...]] = {}
        self._last_market: Optional[MarketSnapshot] = None
        self._refresh_shadow_agents()

    def last_actor_updates(self) -> Tuple[ShadowActorUpdated, ...]:
        return tuple(self._last_updates)

    def last_player_signals(self) -> Dict[str, Tuple[Signal, ...]]:
        return {
            label: tuple(signals)
            for label, signals in self._last_player_signals.items()
        }

    def last_agent_signals(self) -> Dict[str, Tuple[Signal, ...]]:
        return {
            label: tuple(signals)
            for label, signals in self._last_agent_signals.items()
        }

    def last_player_open_positions(self) -> Dict[str, Tuple[Dict[str, object], ...]]:
        out: Dict[str, Tuple[Dict[str, object], ...]] = {}
        prefix = "player:"
        for actor_key, runtime in self._runtimes.items():
            if not actor_key.startswith(prefix):
                continue
            label = actor_key[len(prefix):]
            out[label] = _position_payloads(
                runtime.open_positions(),
                market=self._last_market,
            )
        return out

    def run_bar(
        self,
        market: MarketSnapshot,
        *,
        players: Sequence[Player],
        balance_usd: float,
    ) -> ShadowStepSummary:
        virtual_balance = max(float(balance_usd or 0.0), self._balance_floor)
        self._last_market = market
        self._last_updates = []
        self._last_agent_signals = {}
        self._last_player_signals = {}
        self._refresh_shadow_agents()
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
            runtime = self._runtime(f"agent:{agent.label}")
            signals = _filter_position_aware_signals(
                signals,
                runtime.open_positions(),
                self._risk_config.max_open_positions,
            )
            counts["signals"] += len(signals)
            results = runtime.execute_many(
                signals,
                balance_usd=balance_usd,
            )
            filled_signals = tuple(
                result.signal for result in results if result.is_success
            )
            if filled_signals:
                self._last_agent_signals[agent.label] = filled_signals
            _sync_actor_from_results(agent, results)
            _add_execution_counts(
                counts,
                results,
            )
            self._emit_shadow_actor_updated(
                market,
                actor_type="agent",
                actor_label=agent.label,
                signals=len(signals),
                results=results,
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
            shadow_player = self._shadow_player_for(player)
            if shadow_player is None:
                continue
            try:
                raw_signals, vote_errors = normalize_vote_result(
                    shadow_player.vote(market, signal_id_start=self._signal_id)
                )
            except Exception as exc:
                log.debug("shadow player vote failed for %s", player.label, exc_info=True)
                self._record_player_failure(player.label, market, exc)
                raw_signals = []
                vote_errors = []
            self._record_player_agent_failures(
                player.label,
                shadow_player,
                market,
                errors=vote_errors,
            )
            if raw_signals:
                self._signal_id = max(signal.id for signal in raw_signals) + 1
            signals = [
                replace(
                    signal,
                    position_scope=f"shadow:player:{player.label}",
                )
                for signal in raw_signals
            ]
            runtime = self._runtime(f"player:{player.label}")
            signals = _filter_position_aware_signals(
                signals,
                runtime.open_positions(),
                self._risk_config.max_open_positions,
            )
            counts["signals"] += len(signals)
            results = runtime.execute_many(
                signals,
                balance_usd=balance_usd,
            )
            filled_signals = tuple(
                result.signal for result in results if result.is_success
            )
            if filled_signals:
                self._last_player_signals[player.label] = filled_signals
            _sync_actor_from_results(shadow_player, results)
            _add_execution_counts(
                counts,
                results,
            )
            self._emit_shadow_actor_updated(
                market,
                actor_type="player",
                actor_label=player.label,
                signals=len(signals),
                results=results,
            )
        return counts

    def _signals_from_agent(self, agent: Agent, market: MarketSnapshot) -> List[Signal]:
        try:
            actions = agent.act(market)
        except Exception as exc:
            log.debug("shadow agent act failed for %s", agent.label, exc_info=True)
            self._record_agent_failure(
                player_label="shadow:agent",
                agent_label=agent.label,
                market=market,
                exc=exc,
            )
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

    def _refresh_shadow_agents(self) -> None:
        for source in self._source_registry.all_agents():
            self._source_agents[source.label] = source
            if self._registry.has(source.label):
                continue
            clone = self._clone_agent(source, owner=f"agent:{source.label}")
            if clone is None:
                continue
            try:
                self._registry.register(clone, replace=True)
            except Exception:
                log.warning(
                    "Skipping shadow agent clone for %s: invalid clone",
                    source.label,
                    exc_info=True,
                )

    def _shadow_player_for(self, player: Player) -> Optional[Player]:
        if isinstance(player, EnsemblePlayer):
            agents: List[Agent] = []
            for source in player.agents:
                key = (player.label, source.label)
                clone = self._player_agent_clones.get(key)
                if clone is None:
                    clone = self._clone_agent(
                        source,
                        owner=f"player:{player.label}:{source.label}",
                    )
                    if clone is None:
                        continue
                    self._player_agent_clones[key] = clone
                agents.append(clone)
            if not agents:
                return None
            weights = {
                agent.label: float(player.weights.get(agent.label, 0.0))
                for agent in agents
            }
            total = sum(weights.values())
            if total <= 0:
                equal = 1.0 / len(agents)
                weights = {agent.label: equal for agent in agents}
            elif abs(total - 1.0) > 1e-6:
                weights = {label: value / total for label, value in weights.items()}
            return EnsemblePlayer(
                label=player.label,
                agents=agents,
                weights=weights,
                voting=player.voting,
                thresholds=player.thresholds,
                affinity=player.affinity,
            )

        clone = self._player_clones.get(player.label)
        if clone is None:
            clone = self._clone_actor(player, owner=f"player:{player.label}")
            if clone is None:
                return None
            self._player_clones[player.label] = clone
        return clone

    def _clone_agent(self, agent: Agent, *, owner: str) -> Optional[Agent]:
        clone = self._clone_actor(agent, owner=owner)
        if clone is None:
            return None
        if getattr(clone, "label", None) != agent.label:
            log.warning(
                "Skipping shadow clone for %s: label changed to %s",
                agent.label,
                getattr(clone, "label", None),
            )
            return None
        return clone

    @staticmethod
    def _clone_actor(actor, *, owner: str):
        clone_fn = getattr(actor, "clone_for_shadow", None)
        try:
            if callable(clone_fn):
                return clone_fn()
            return copy.deepcopy(actor)
        except Exception:
            log.warning("Skipping shadow clone for %s", owner, exc_info=True)
            return None

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

    def _record_player_agent_failures(
        self,
        player_label: str,
        player: Player,
        market: MarketSnapshot,
        *,
        errors: Optional[Sequence[object]] = None,
    ) -> None:
        for err in (errors if errors is not None else getattr(player, "last_vote_errors", []) or []):
            label = str(getattr(err, "agent_label", "") or "")
            reason = str(getattr(err, "reason", "") or "")
            if not label:
                continue
            self._emit_agent_failure(
                player_label=player_label,
                agent_label=label,
                reason=reason,
                market=market,
            )
            self._perf.record_actor_failure(label, market.regime)

    def _record_agent_failure(
        self,
        *,
        player_label: str,
        agent_label: str,
        market: MarketSnapshot,
        exc: Exception,
    ) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        self._emit_agent_failure(
            player_label=player_label,
            agent_label=agent_label,
            reason=reason,
            market=market,
        )
        self._perf.record_actor_failure(agent_label, market.regime)

    def _record_player_failure(
        self,
        player_label: str,
        market: MarketSnapshot,
        exc: Exception,
    ) -> None:
        reason = f"{type(exc).__name__}: {exc}"
        if self._event_log is not None:
            self._event_log.emit(PlayerVoteFailed(
                bar=market.bar,
                trace_id=f"shadow-{market.bar}",
                player_label=player_label,
                reason=reason,
            ))
        self._perf.record_actor_failure(player_label, market.regime)

    def _emit_agent_failure(
        self,
        *,
        player_label: str,
        agent_label: str,
        reason: str,
        market: MarketSnapshot,
    ) -> None:
        if self._event_log is None:
            return
        self._event_log.emit(AgentVoteFailed(
            bar=market.bar,
            trace_id=f"shadow-{market.bar}",
            player_label=player_label,
            agent_label=agent_label,
            reason=reason,
        ))

    def _emit_shadow_actor_updated(
        self,
        market: MarketSnapshot,
        *,
        actor_type: str,
        actor_label: str,
        signals: int,
        results: Sequence[ExecutionResult],
    ) -> None:
        counts = _empty_counts()
        _add_execution_counts(counts, results)
        realized_pnl, closed_trades, winning_trades = _realized_counts(results)
        event = ShadowActorUpdated(
            bar=market.bar,
            timestamp=market.timestamp,
            trace_id=f"shadow-{market.bar}",
            actor_type=actor_type,
            actor_label=actor_label,
            regime=market.regime.label,
            signals=signals,
            filled=counts["filled"],
            rejected=counts["rejected"],
            blocked=counts["blocked"],
            rejected_reasons=_reason_counts(results, ExecutionStatus.REJECTED),
            blocked_reasons=_reason_counts(results, ExecutionStatus.BLOCKED),
            agent_outcomes=_agent_outcome_counts(results),
            agent_rejected_reasons=_agent_reason_counts(results, ExecutionStatus.REJECTED),
            agent_blocked_reasons=_agent_reason_counts(results, ExecutionStatus.BLOCKED),
            realized_pnl_usd=realized_pnl,
            closed_trades=closed_trades,
            winning_trades=winning_trades,
            symbol_outcomes=_symbol_outcome_counts(results),
            symbol_action_outcomes=_symbol_action_outcome_counts(results),
        )
        self._last_updates.append(event)
        if self._event_log is not None:
            self._event_log.emit(event)


def _empty_counts() -> Dict[str, int]:
    return {"signals": 0, "filled": 0, "rejected": 0, "blocked": 0}


def _sync_actor_from_results(actor, results: Sequence[ExecutionResult]) -> None:
    sync = getattr(actor, "sync_from_execution_results", None)
    if callable(sync):
        try:
            sync(results)
        except Exception:
            log.debug("shadow actor sync failed for %s", getattr(actor, "label", actor), exc_info=True)
    agents = getattr(actor, "agents", None)
    if not agents:
        return
    by_agent: Dict[str, List[ExecutionResult]] = {}
    for result in results or ():
        label = _result_agent_label(result)
        if label:
            by_agent.setdefault(label, []).append(result)
    for agent in agents:
        label = str(getattr(agent, "label", "") or "")
        agent_results = by_agent.get(label, [])
        if not agent_results:
            continue
        sync = getattr(agent, "sync_from_execution_results", None)
        if not callable(sync):
            continue
        try:
            sync(agent_results)
        except Exception:
            log.debug("shadow agent sync failed for %s", label, exc_info=True)


def _position_payloads(
    open_positions: Dict[str, object],
    *,
    market: Optional[MarketSnapshot] = None,
) -> Tuple[Dict[str, object], ...]:
    payloads: List[Dict[str, object]] = []
    for key, pos in (open_positions or {}).items():
        sym = str(getattr(pos, "sym", key) or key).upper()
        side = str(getattr(pos, "side", "") or "").lower()
        if sym and side in ("long", "short"):
            opened_bar = int(getattr(pos, "opened_bar", 0) or 0)
            current_bar = int(getattr(market, "bar", opened_bar) or opened_bar)
            age_bars = max(0, current_bar - opened_bar) if opened_bar > 0 else 0
            entry_price = float(getattr(pos, "entry_price", 0.0) or 0.0)
            qty = float(getattr(pos, "qty", 0.0) or 0.0)
            current_price = 0.0
            if market is not None:
                current_price = float((market.prices or {}).get(sym, 0.0) or 0.0)
            if current_price <= 0:
                current_price = entry_price
            unrealized = 0.0
            if entry_price > 0 and current_price > 0 and qty > 0:
                if side == "long":
                    unrealized = (current_price - entry_price) * qty
                else:
                    unrealized = (entry_price - current_price) * qty
            payloads.append({
                "sym": sym,
                "side": side,
                "opened_bar": opened_bar,
                "age_bars": age_bars,
                "entry_price": entry_price,
                "current_price": current_price,
                "qty": qty,
                "unrealized_pnl_usd": unrealized,
                "stop_price": getattr(pos, "stop_price", None),
                "take_profit_price": getattr(pos, "take_profit_price", None),
                "fresh": age_bars <= 1,
            })
    payloads.sort(key=lambda item: (item["sym"], item["side"]))
    return tuple(payloads)


def _filter_position_aware_signals(
    signals: Sequence[Signal],
    open_positions: Dict[str, object],
    max_open_positions: int,
) -> List[Signal]:
    max_positions = max(0, int(max_open_positions))
    planned_symbols = {str(sym).upper() for sym in open_positions.keys()}
    planned_count = len(planned_symbols)
    kept: List[Signal] = []
    for signal in signals:
        if not signal.action.is_open:
            kept.append(signal)
            continue
        sym = str(signal.sym).upper()
        if sym in planned_symbols:
            continue
        if planned_count >= max_positions:
            continue
        kept.append(signal)
        planned_symbols.add(sym)
        planned_count += 1
    return kept


def _add_execution_counts(counts: Dict[str, int], results: Iterable[ExecutionResult]) -> None:
    for result in results:
        if result.status == ExecutionStatus.FILLED:
            counts["filled"] += 1
        elif result.status == ExecutionStatus.REJECTED:
            counts["rejected"] += 1
        elif result.status == ExecutionStatus.BLOCKED:
            counts["blocked"] += 1


def _reason_counts(
    results: Iterable[ExecutionResult],
    status: ExecutionStatus,
) -> Tuple[Tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.status != status:
            continue
        reason = str(result.reason or "unspecified")
        counts[reason] += 1
    return tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _agent_outcome_counts(
    results: Iterable[ExecutionResult],
) -> Tuple[Tuple[str, int, int, int, int], ...]:
    counts: Dict[str, Dict[str, int]] = {}
    for result in results:
        label = _result_agent_label(result)
        if not label:
            continue
        bucket = counts.setdefault(
            label,
            {"signals": 0, "filled": 0, "rejected": 0, "blocked": 0},
        )
        bucket["signals"] += 1
        if result.status == ExecutionStatus.FILLED:
            bucket["filled"] += 1
        elif result.status == ExecutionStatus.REJECTED:
            bucket["rejected"] += 1
        elif result.status == ExecutionStatus.BLOCKED:
            bucket["blocked"] += 1
    return tuple(
        (
            label,
            bucket["signals"],
            bucket["filled"],
            bucket["rejected"],
            bucket["blocked"],
        )
        for label, bucket in sorted(counts.items())
    )


def _agent_reason_counts(
    results: Iterable[ExecutionResult],
    status: ExecutionStatus,
) -> Tuple[Tuple[str, str, int], ...]:
    counts: Counter[Tuple[str, str]] = Counter()
    for result in results:
        if result.status != status:
            continue
        label = _result_agent_label(result)
        if not label:
            continue
        reason = str(result.reason or "unspecified")
        counts[(label, reason)] += 1
    return tuple(
        (label, reason, count)
        for (label, reason), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    )


def _result_agent_label(result: ExecutionResult) -> str:
    signal = result.signal
    return str(getattr(signal, "by_agent", "") or getattr(signal, "by_player", "") or "")


def _realized_counts(results: Iterable[ExecutionResult]) -> Tuple[float, int, int]:
    realized_pnl = 0.0
    closed_trades = 0
    winning_trades = 0
    for result in results:
        for _, pnl in getattr(result, "realized_pnl_by_player", ()) or ():
            value = float(pnl or 0.0)
            realized_pnl += value
            if value > 0:
                winning_trades += 1
        for _, count in getattr(result, "closed_trade_counts_by_player", ()) or ():
            closed_trades += int(count or 0)
    return realized_pnl, closed_trades, winning_trades


def _symbol_outcome_counts(
    results: Iterable[ExecutionResult],
) -> Tuple[Tuple[str, float, int, int], ...]:
    counts: Dict[str, Dict[str, float]] = {}
    for result in results:
        signal = result.signal
        symbol = str(getattr(signal, "sym", "") or "").upper()
        if not symbol:
            continue
        pnl = sum(
            float(value or 0.0)
            for _, value in getattr(result, "realized_pnl_by_player", ()) or ()
        )
        closed = sum(
            int(value or 0)
            for _, value in getattr(result, "closed_trade_counts_by_player", ()) or ()
        )
        wins = sum(
            int(value or 0)
            for _, value in getattr(result, "win_counts_by_player", ()) or ()
        )
        if closed <= 0 and abs(pnl) <= 1e-12 and wins <= 0:
            continue
        bucket = counts.setdefault(
            symbol,
            {"pnl": 0.0, "closed": 0.0, "wins": 0.0},
        )
        bucket["pnl"] += float(pnl)
        bucket["closed"] += float(closed)
        bucket["wins"] += float(wins)
    return tuple(
        (
            symbol,
            float(bucket["pnl"]),
            int(bucket["closed"]),
            int(bucket["wins"]),
        )
        for symbol, bucket in sorted(counts.items())
    )


def _symbol_action_outcome_counts(
    results: Iterable[ExecutionResult],
) -> Tuple[Tuple[str, str, float, int, int], ...]:
    counts: Dict[Tuple[str, str], Dict[str, float]] = {}
    for result in results:
        for symbol_raw, action_raw, pnl_raw in (
            getattr(result, "closed_position_outcomes", ()) or ()
        ):
            symbol = str(symbol_raw or "").upper()
            action = _normalize_shadow_outcome_action(action_raw)
            if not symbol or not action:
                continue
            pnl = float(pnl_raw or 0.0)
            bucket = counts.setdefault(
                (symbol, action),
                {"pnl": 0.0, "closed": 0.0, "wins": 0.0},
            )
            bucket["pnl"] += pnl
            bucket["closed"] += 1.0
            if pnl > 0:
                bucket["wins"] += 1.0
    return tuple(
        (
            symbol,
            action,
            float(bucket["pnl"]),
            int(bucket["closed"]),
            int(bucket["wins"]),
        )
        for (symbol, action), bucket in sorted(counts.items())
    )


def _normalize_shadow_outcome_action(value: object) -> str:
    action = str(value or "").strip().upper()
    if action == "LONG":
        return "FUT_LONG_FULL"
    if action == "SHORT":
        return "FUT_SHORT_FULL"
    if action in Action.__members__ and Action[action].is_open:
        return action
    return ""
