"""Domain — immutable core types.

Все типы здесь — frozen dataclasses или Enum. Никакого мутабельного
состояния. Это позволяет использовать их в pure-функциях без риска
side-effects.
"""

from .types import (
    Action,
    Regime,
    Signal,
    Trade,
    Metrics,
    MarketSnapshot,
)

__all__ = [
    "Action",
    "Regime",
    "Signal",
    "Trade",
    "Metrics",
    "MarketSnapshot",
]
