"""Player Protocol и EnsemblePlayer.

Player — тот, кто решает, какие сигналы подать на исполнение. В v2
есть один основной тип: EnsemblePlayer, который берёт top-k агентов
от AgentSelector и агрегирует их голоса через VotingPolicy.

EnsemblePlayer заменяет в v1 всю иерархию:
  Panteon → PanteonResearch → _PanteonShadowVariant → 6 подклассов.

Вместо классов — конфигурационные профили (PlayerProfile в composer.py).

Гарантия:
  • EnsemblePlayer.agents — НИКОГДА не содержит карантинных агентов.
    Это обеспечивается AgentSelector (вызывается через PlayerComposer
    при каждой инстанциации).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, cast

from ..domain.types import Action, MarketSnapshot, Regime, Signal
from .agent import Agent
from .voting import AgentVotes, ThresholdProfile, VotingPolicy


# ────────────────────────────────────────────────────────────────────
# Player Protocol
# ────────────────────────────────────────────────────────────────────


class Player(Protocol):
    """Структурный интерфейс игрока — принимает рынок, возвращает сигналы.

    Player — это тот, кого Strategist выбирает текущим лидером.
    Его vote() — финальные решения, которые идут в TradeExecutor.
    """

    label: str
    affinity: Optional[Regime]  # None = универсальный

    def vote(self, market: MarketSnapshot, *, signal_id_start: int) -> "VoteResult":
        ...

    @property
    def agent_labels(self) -> List[str]:
        """Список labels агентов, которыми пользуется этот игрок.

        ОБЯЗАТЕЛЬНОЕ свойство — Strategist использует его для проверки
        Q4 (никакой карантинный не появляется в Player.agents).
        """
        ...


@dataclass(frozen=True)
class VoteError:
    agent_label: str
    reason: str


VoteResult = Tuple[List[Signal], List[VoteError]]
LegacyVoteResult = Union[VoteResult, List[Signal], Tuple[Signal, ...]]


def normalize_vote_result(result: LegacyVoteResult | object) -> VoteResult:
    """Return ``(signals, errors)`` for both new and legacy Player.vote results."""
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], list)
        and isinstance(result[1], list)
    ):
        return cast(VoteResult, result)
    if result is None:
        return [], []
    return list(cast(Sequence[Signal], result)), []


# ────────────────────────────────────────────────────────────────────
# EnsemblePlayer
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NoTradePlayer:
    """Cash leader used when no real-promoted candidate is allowed to trade."""

    label: str = "NoTrade"
    affinity: Optional[Regime] = None

    @property
    def agent_labels(self) -> List[str]:
        return []

    @property
    def actor_type(self) -> str:
        return "no_trade"

    def vote(
        self,
        market: MarketSnapshot,
        *,
        signal_id_start: int,
    ) -> VoteResult:
        return [], []


@dataclass
class StrategyPlayer:
    """One strategy exposed as one selectable player.

    The wrapped strategy is an implementation detail.  Selection, memory and
    attribution use only ``label``; no agent score can leak into the player
    rating.  This is the production building block for the player-only runtime.
    """

    label: str
    strategy: Agent
    affinity: Optional[Regime] = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("StrategyPlayer.label must be non-empty")

    @property
    def agent_labels(self) -> List[str]:
        """Return no selectable sub-components: the strategy is the player."""
        return []

    @property
    def actor_type(self) -> str:
        return "player"

    def vote(
        self,
        market: MarketSnapshot,
        *,
        signal_id_start: int,
    ) -> VoteResult:
        signals: List[Signal] = []
        errors: List[VoteError] = []
        signal_id = int(signal_id_start)
        full_market = bool(
            getattr(self.strategy, "prefers_full_market_snapshot", False)
        )
        action_rows: List[Tuple[str, object]] = []
        if full_market:
            try:
                raw_actions = self.strategy.act(market) or {}
            except Exception as exc:
                errors.append(VoteError(
                    agent_label=self.label,
                    reason=f"{type(exc).__name__}: {exc}",
                ))
                raw_actions = {}
            action_rows.extend(
                (sym, raw_actions.get(sym, Action.HOLD))
                for sym in market.prices
            )
        else:
            for sym in market.prices:
                symbol_market = market.with_regime_for_symbol(sym)
                try:
                    raw_actions = self.strategy.act(symbol_market) or {}
                except Exception as exc:
                    errors.append(VoteError(
                        agent_label=self.label,
                        reason=f"{type(exc).__name__}: {exc}",
                    ))
                    continue
                action_rows.append((sym, raw_actions.get(sym, Action.HOLD)))

        for sym, action in action_rows:
            if not isinstance(action, Action):
                try:
                    action = Action(int(action))
                except (TypeError, ValueError):
                    continue
            if action.is_hold:
                continue
            signals.append(Signal(
                id=signal_id,
                bar=market.bar,
                sym=sym,
                action=action,
                price=float(market.prices.get(sym, 0.0)),
                regime=market.regime_for_symbol(sym),
                by_player=self.label,
                by_agent="",
                risk_mult=1.0,
                timestamp=market.timestamp,
                metadata={"strategy_impl": str(getattr(self.strategy, "label", self.label))},
            ))
            signal_id += 1
        return signals, errors


@dataclass
class EnsemblePlayer:
    """Игрок-ансамбль из набора агентов и voting policy.

    Состояние:
      • label, affinity — статические
      • agents          — список Agent (frozen после создания через composer)
      • weights         — weight per agent (нормализованные на 1.0)
      • voting          — VotingPolicy
      • thresholds      — ThresholdProfile

    EnsemblePlayer создаётся через PlayerComposer.compose_*, который
    гарантирует что:
      • agents отфильтрованы Selector-ом (нет карантинных)
      • weights нормализованы (sum = 1.0)
      • thresholds валидны (open_floor ≤ open_single и т. д.)
    """

    label:        str
    agents:       List[Agent]
    weights:      Dict[str, float]
    voting:       VotingPolicy
    thresholds:   ThresholdProfile
    affinity:     Optional[Regime] = None
    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("EnsemblePlayer.label must be non-empty")
        if not self.agents:
            # В v2 это допустимо (компонент возвращает пустой список signals)
            # но логируется как warning callsite-ом — не raise здесь.
            pass
        # Sanity-check: каждый agent имеет вес
        agent_labels = {a.label for a in self.agents}
        weight_labels = set(self.weights.keys())
        if agent_labels != weight_labels:
            raise ValueError(
                f"weights keys ({sorted(weight_labels)}) does not match "
                f"agents labels ({sorted(agent_labels)})"
            )
        # Нормализация: проверяем что веса нормированы (с допуском)
        total = sum(self.weights.values())
        if self.agents and abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"weights must sum to 1.0, got {total:.6f}"
            )

    @property
    def agent_labels(self) -> List[str]:
        return sorted(self.weights.keys())

    @property
    def actor_type(self) -> str:
        return "ensemble"

    def vote(
        self,
        market: MarketSnapshot,
        *,
        signal_id_start: int,
    ) -> VoteResult:
        """Сначала собираем голоса всех агентов, потом агрегируем
        через VotingPolicy, затем превращаем в Signal-ы."""
        if not self.agents:
            return [], []

        errors: List[VoteError] = []
        votes: Dict[str, Dict[str, Action]] = {}
        for agent in self.agents:
            cleaned: Dict[str, Action] = {}
            for sym in market.prices:
                symbol_market = market.with_regime_for_symbol(sym)
                try:
                    agent_actions = agent.act(symbol_market)
                except Exception as exc:
                # Агент не должен падать в production. Если упал —
                # его как будто нет в этом баре.
                    errors.append(VoteError(
                        agent_label=agent.label,
                        reason=f"{type(exc).__name__}: {exc}",
                    ))
                    continue
            # Sanitize — только sym из market
                if sym not in (agent_actions or {}):
                    continue
                action = (agent_actions or {}).get(sym, Action.HOLD)
                if not isinstance(action, Action):
                    try:
                        action = Action(int(action))
                    except (ValueError, TypeError):
                        continue
                cleaned[sym] = action
            votes[agent.label] = cleaned

        # Агрегируем
        final = self.voting.aggregate(votes, self.weights, self.thresholds, market)

        # Превращаем в Signal
        signals: List[Signal] = []
        sid = int(signal_id_start)
        for sym, action in final.items():
            if action == Action.HOLD:
                continue
            # Кто из агентов проголосовал — берём ОДНОГО основного
            # (первого с тем же направлением или с весомым голосом).
            by_agent = self._main_contributor(sym, action, votes)
            vote_actions = {
                label: votes.get(label, {}).get(sym, Action.HOLD)
                for label in self.agent_labels
            }
            vote_weights = {
                label: float(self.weights.get(label, 0.0))
                for label in self.agent_labels
            }
            signals.append(Signal(
                id=sid,
                bar=market.bar,
                sym=sym,
                action=action,
                price=float(market.prices.get(sym, 0.0)),
                regime=market.regime_for_symbol(sym),
                by_player=self.label,
                by_agent=by_agent,
                vote_weights=vote_weights,
                vote_actions=vote_actions,
                risk_mult=1.0,
                timestamp=market.timestamp,
            ))
            sid += 1
        return signals, errors

    def _main_contributor(
        self,
        sym: str,
        final_action: Action,
        votes: Dict[str, Dict[str, Action]],
    ) -> str:
        """Основной агент-инициатор для атрибуции."""
        # Кандидаты — кто голосовал за то же направление
        same_dir: List[tuple] = []  # (weight, label)
        for label, agent_votes in votes.items():
            a = agent_votes.get(sym, Action.HOLD)
            same_direction = (
                (final_action.is_long_open and a.is_long_open)
                or (final_action.is_short_open and a.is_short_open)
                or (final_action.is_close and a.is_close)
            )
            if same_direction:
                same_dir.append((self.weights.get(label, 0.0), label))
        if not same_dir:
            return ""
        same_dir.sort(reverse=True)  # max weight first
        return same_dir[0][1]


@dataclass
class RotatingAgentPlayer:
    """Fixed agent-set player that executes the first current actionable agent.

    Unlike ``EnsemblePlayer``, this player does not dilute a specialist signal
    through consensus. It keeps a stable static agent set for attribution and
    shadow accounting, but chooses the first agent with a non-HOLD signal from a
    regime-specific priority list on each bar.
    """

    label: str
    agents: List[Agent]
    regime_agent_order: Mapping[str, Sequence[str]]
    fallback_agent_order: Sequence[str] = ()
    affinity: Optional[Regime] = None
    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("RotatingAgentPlayer.label must be non-empty")
        labels = [agent.label for agent in self.agents]
        if len(labels) != len(set(labels)):
            raise ValueError("RotatingAgentPlayer agents must not contain duplicates")
        self._agents_by_label = {agent.label: agent for agent in self.agents}

    @property
    def agent_labels(self) -> List[str]:
        return sorted(self._agents_by_label)

    @property
    def actor_type(self) -> str:
        return "ensemble"

    def vote(
        self,
        market: MarketSnapshot,
        *,
        signal_id_start: int,
    ) -> VoteResult:
        errors: List[VoteError] = []
        if not self.agents:
            return [], []
        signals: List[Signal] = []
        sid = int(signal_id_start)
        for sym in market.prices:
            symbol_market = market.with_regime_for_symbol(sym)
            for agent_label in self._ordered_labels(symbol_market.regime):
                agent = self._agents_by_label.get(agent_label)
                if agent is None:
                    continue
                cleaned, agent_errors = self._agent_actions(agent, symbol_market)
                errors.extend(agent_errors)
                action = cleaned.get(sym, Action.HOLD)
                if action.is_hold:
                    continue
                created = self._signals_from_actions(
                    {sym: action},
                    symbol_market,
                    by_agent=agent.label,
                    signal_id_start=sid,
                )
                signals.extend(created)
                sid += len(created)
                break
        return signals, errors

    def _ordered_labels(self, regime: Regime) -> Tuple[str, ...]:
        ordered: List[str] = []
        for raw_label in self.regime_agent_order.get(regime.label, ()):
            label = str(raw_label or "").strip()
            if label and label not in ordered:
                ordered.append(label)
        for raw_label in self.fallback_agent_order:
            label = str(raw_label or "").strip()
            if label and label not in ordered:
                ordered.append(label)
        return tuple(ordered)

    def _agent_actions(
        self,
        agent: Agent,
        market: MarketSnapshot,
    ) -> Tuple[Dict[str, Action], List[VoteError]]:
        errors: List[VoteError] = []
        try:
            raw_actions = agent.act(market)
        except Exception as exc:
            errors.append(VoteError(
                agent_label=agent.label,
                reason=f"{type(exc).__name__}: {exc}",
            ))
            raw_actions = {}
        cleaned: Dict[str, Action] = {}
        for sym, action in (raw_actions or {}).items():
            if sym not in market.prices:
                continue
            if not isinstance(action, Action):
                try:
                    action = Action(int(action))
                except (ValueError, TypeError):
                    continue
            cleaned[sym] = action
        return cleaned, errors

    def _signals_from_actions(
        self,
        actions: Mapping[str, Action],
        market: MarketSnapshot,
        *,
        by_agent: str,
        signal_id_start: int,
    ) -> List[Signal]:
        signals: List[Signal] = []
        sid = int(signal_id_start)
        equal_weight = 1.0 / len(self._agents_by_label) if self._agents_by_label else 0.0
        vote_weights = {
            label: equal_weight
            for label in self.agent_labels
        }
        vote_actions = {
            label: {}
            for label in self.agent_labels
        }
        vote_actions[by_agent] = dict(actions)
        for sym, action in actions.items():
            if action == Action.HOLD:
                continue
            signals.append(Signal(
                id=sid,
                bar=market.bar,
                sym=sym,
                action=action,
                price=float(market.prices.get(sym, 0.0)),
                regime=market.regime_for_symbol(sym),
                by_player=self.label,
                by_agent=by_agent,
                vote_weights=vote_weights,
                vote_actions={
                    label: vote_actions.get(label, {}).get(sym, Action.HOLD)
                    for label in self.agent_labels
                },
                risk_mult=1.0,
                timestamp=market.timestamp,
            ))
            sid += 1
        return signals
