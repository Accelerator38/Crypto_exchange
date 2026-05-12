"""Pure scoring functions.

Все функции здесь — без состояния (pure). Принимают Metrics и Regime,
возвращают число. Не делают I/O, не зависят от времени.

Это позволяет:
- тривиальную testability;
- использование в backtests без рантайма;
- параллельное вычисление скоров без race-conditions.
"""

from .scoring import (
    ScoringConfig,
    DEFAULT_SCORING,
    regime_score,
    confidence_from_sample,
    is_locally_proven,
    is_hopeless_in_all_regimes,
)

__all__ = [
    "ScoringConfig",
    "DEFAULT_SCORING",
    "regime_score",
    "confidence_from_sample",
    "is_locally_proven",
    "is_hopeless_in_all_regimes",
]
