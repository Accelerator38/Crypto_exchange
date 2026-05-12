"""Panteon v2 — clean re-architecture.

См. README.md и docs/PANTEON_V2_ARCHITECTURE.md.

Этот пакет ИЗОЛИРОВАН от panteon_runtime (v1). Не импортирует ничего
из v1 и не должен импортироваться оттуда.
"""

__version__ = "0.1.0-skeleton"

# Public API будет добавляться по мере имплементации фаз.
# Сейчас экспортируем только core types из domain/.
from .domain.types import (
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
