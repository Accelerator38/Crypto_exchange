from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


OPEN_LONG_ACTIONS = frozenset((1, 2, 4, 5))
OPEN_SHORT_ACTIONS = frozenset((6, 7))
CLOSE_ACTIONS = frozenset((3, 8))


@dataclass
class MarketSnapshot:
    prices: Dict[str, float]
    volumes: Dict[str, float]
    month: Optional[int] = None
    portfolio_value: Optional[float] = None
    bar_index: Optional[int] = None
    regime: str = "neutral"
    context: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentIntent:
    agent: str
    symbol: str
    action: int
    score: float
    direction: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    actions: Dict[str, int] = field(default_factory=dict)
    contributors: Dict[str, Dict[str, float]] = field(default_factory=dict)
    long_scores: Dict[str, float] = field(default_factory=dict)
    short_scores: Dict[str, float] = field(default_factory=dict)
    close_scores: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)


def action_direction(action: int) -> Optional[str]:
    action = int(action)
    if action in OPEN_LONG_ACTIONS:
        return "long"
    if action in OPEN_SHORT_ACTIONS:
        return "short"
    if action in CLOSE_ACTIONS:
        return "close"
    return None


def action_is_open(action: int) -> bool:
    return int(action) in OPEN_LONG_ACTIONS or int(action) in OPEN_SHORT_ACTIONS


def action_is_close(action: int) -> bool:
    return int(action) in CLOSE_ACTIONS


def intent_from_action(
    agent: str,
    symbol: str,
    action: int,
    score: float,
    metadata: Optional[Dict[str, object]] = None,
) -> Optional[AgentIntent]:
    direction = action_direction(action)
    if direction is None:
        return None
    return AgentIntent(
        agent=str(agent),
        symbol=str(symbol),
        action=int(action),
        score=float(score),
        direction=direction,
        metadata=dict(metadata or {}),
    )
