"""Agent Protocol и AgentRegistry.

Agent в Panteon v2 — это структурный интерфейс (Protocol), а не базовый
класс. Любой объект с правильной формой (`label` + `act(market)`) — это
агент. Это позволяет тривиально оборачивать v1-агенты адаптерами без
наследования.

AgentRegistry — единый каталог. Регистрация explicit: никаких автомагических
сканирований, всё вручную в сервисе bootstrap.
"""

from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional, Protocol, runtime_checkable

from ..domain.types import Action, MarketSnapshot


# ────────────────────────────────────────────────────────────────────
# Protocol
# ────────────────────────────────────────────────────────────────────


@runtime_checkable
class Agent(Protocol):
    """Структурный интерфейс агента.

    Гарантии вызывающих:
      • не мутируют market
      • получают Dict[sym → Action]; sym всегда из market.prices

    Гарантии агента:
      • detminism: один и тот же market → один и тот же ответ (для тестируемости)
      • no I/O, no time.time(): использует только market.timestamp
      • не выбрасывает исключений (вместо этого возвращает HOLD)
    """

    label: str  # уникальное короткое имя, например "LiveAfterShock"

    def act(self, market: MarketSnapshot) -> Dict[str, Action]:
        ...


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────


class AgentRegistry:
    """Единый каталог зарегистрированных агентов.

    Использование:
        registry = AgentRegistry()
        registry.register(LiveAfterShockAdapter())
        registry.register(FundingArbAdapter())
        ...
        agent = registry.get("LiveAfterShock")
        for label in registry.all_labels(): ...

    Никаких автомагических сканов. Регистрация — explicit, проверяемая.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Agent] = {}
        self._lock = threading.RLock()

    def register(self, agent: Agent, *, replace: bool = False) -> None:
        """Зарегистрировать агента. Дубликаты по label запрещены если
        replace=False."""
        with self._lock:
            if not isinstance(agent, Agent):
                # runtime_checkable Protocol — проверяет наличие label и act
                raise TypeError(
                    f"register expects an Agent (with label + act), "
                    f"got {type(agent).__name__}"
                )
            if not getattr(agent, "label", ""):
                raise ValueError("Agent.label must be a non-empty string")
            if agent.label in self._agents and not replace:
                raise ValueError(
                    f"Agent label {agent.label!r} already registered. "
                    f"Pass replace=True to override."
                )
            self._agents[agent.label] = agent

    def unregister(self, label: str) -> bool:
        with self._lock:
            return self._agents.pop(label, None) is not None

    def get(self, label: str) -> Optional[Agent]:
        with self._lock:
            return self._agents.get(label)

    def has(self, label: str) -> bool:
        with self._lock:
            return label in self._agents

    def all_labels(self) -> List[str]:
        with self._lock:
            return sorted(self._agents.keys())

    def all_agents(self) -> List[Agent]:
        with self._lock:
            return [self._agents[k] for k in sorted(self._agents.keys())]

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __contains__(self, label: str) -> bool:
        with self._lock:
            return label in self._agents
