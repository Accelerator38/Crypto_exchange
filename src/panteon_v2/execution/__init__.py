"""Execution layer (Phase 4).

Все компоненты для перехода Signal → Trade:
  exchange       — Exchange Protocol + FakeExchange (для тестов)
  symbol_health  — SymbolHealthMonitor с adaptive blocklist
  risk_limits    — Pre-flight risk-проверки
  position_tracker — Учёт открытых позиций для парного close-расчёта
  executor       — TradeExecutor: единственный путь signal → биржа
"""

from .exchange import (
    Exchange,
    ExchangePosition,
    FakeExchange,
    OrderResult,
    OrderStatus,
)
from .executor import (
    ExecutionResult,
    ExecutionStatus,
    TradeExecutor,
)
from .position_tracker import PositionTracker, TrackedPosition
from .risk_limits import (
    DEFAULT_RISK,
    RiskCheckResult,
    RiskLimits,
    RiskLimitsConfig,
)
from .symbol_health import (
    DEFAULT_HEALTH_CONFIG,
    SymbolHealthConfig,
    SymbolHealthMonitor,
    SymbolStatus,
)

__all__ = [
    # exchange
    "Exchange",
    "ExchangePosition",
    "FakeExchange",
    "OrderResult",
    "OrderStatus",
    # executor
    "ExecutionResult",
    "ExecutionStatus",
    "TradeExecutor",
    # position_tracker
    "PositionTracker",
    "TrackedPosition",
    # risk_limits
    "DEFAULT_RISK",
    "RiskCheckResult",
    "RiskLimits",
    "RiskLimitsConfig",
    # symbol_health
    "DEFAULT_HEALTH_CONFIG",
    "SymbolHealthConfig",
    "SymbolHealthMonitor",
    "SymbolStatus",
]
