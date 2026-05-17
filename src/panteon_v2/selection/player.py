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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

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

    def vote(self, market: MarketSnapshot, *, signal_id_start: int) -> List[Signal]:
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

    def vote(
        self,
        market: MarketSnapshot,
        *,
        signal_id_start: int,
    ) -> List[Signal]:
        return []


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
    last_vote_errors: List[VoteError] = field(default_factory=list, init=False, repr=False)

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

    def vote(
        self,
        market: MarketSnapshot,
        *,
        signal_id_start: int,
    ) -> List[Signal]:
        """Сначала собираем голоса всех агентов, потом агрегируем
        через VotingPolicy, затем превращаем в Signal-ы."""
        if not self.agents:
            self.last_vote_errors = []
            return []

        self.last_vote_errors = []
        votes: Dict[str, Dict[str, Action]] = {}
        for agent in self.agents:
            try:
                agent_actions = agent.act(market)
            except Exception as exc:
                # Агент не должен падать в production. Если упал —
                # его как будто нет в этом баре.
                self.last_vote_errors.append(VoteError(
                    agent_label=agent.label,
                    reason=f"{type(exc).__name__}: {exc}",
                ))
                agent_actions = {}
            # Sanitize — только sym из market
            cleaned: Dict[str, Action] = {}
            for sym, action in (agent_actions or {}).items():
                if sym not in market.prices:
                    continue
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
                regime=market.regime,
                by_player=self.label,
                by_agent=by_agent,
                vote_weights=vote_weights,
                vote_actions=vote_actions,
                risk_mult=1.0,
                timestamp=market.timestamp,
            ))
            sid += 1
        return signals

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
