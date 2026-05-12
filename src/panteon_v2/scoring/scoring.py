"""Pure scoring functions для Panteon v2.

Заменяют 5 разных скоринговых функций v1:
- `_live_score` (agent_meta.py)
- `_combined_score` (agent_meta.py)
- `_risk_adjusted_positive_score` (agent_meta.py)
- `_score_shadow_candidate` (panteon_agents.py — две версии)

Одна функция `regime_score(metrics, regime, confidence)` для всех мест,
где нужно оценить агента/игрока. Никаких per-class реализаций.

Инварианты:
- pure (no state, no I/O, no time);
- deterministic (один и тот же вход → один и тот же выход);
- monotonic в pnl_pct: при неизменности других полей score(pnl_a) ≥
  score(pnl_b) ⇔ pnl_a ≥ pnl_b.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..domain.types import Metrics, Regime


# ────────────────────────────────────────────────────────────────────
# Конфигурация скоринга — все коэффициенты в одном месте
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoringConfig:
    """Параметры скоринга. Frozen — никаких мутаций после создания.

    Все коэффициенты в одном месте — централизованный тюнинг через
    создание нового экземпляра, не точечную правку магических чисел.
    """

    # Веса компонент в regime_score
    pnl_weight:        float = 0.62
    sharpe_weight:     float = 0.34
    max_dd_weight:     float = 0.18
    activity_weight:   float = 0.008
    win_bonus_divisor: float = 18.0  # (win_rate - 50) / divisor

    # Cap-ы для активности (чтобы не давать мега-бонус за shotgun-стрельбу)
    activity_signals_cap: int = 30

    # Минимальная выборка для уверенности
    min_closed_for_full_confidence: int = 6

    # Карантин: пороги (см. is_locally_proven, is_hopeless_in_all_regimes)
    quarantine_recovery_pnl_pct: float = 0.10  # один режим с +0.10% PnL → reнабором
    quarantine_recovery_closed:  int   = 3
    quarantine_hard_neg_pnl_pct: float = -0.30  # все режимы хуже −0.30% и ≥ 5 closed → карантин
    quarantine_hard_min_closed:  int   = 5

    # Минимальный score для попадания в Selector top-k
    min_eligible_score: float = 0.0

    # Penalty за полное отсутствие активности
    inactivity_penalty: float = 0.65

    def __post_init__(self) -> None:
        if self.activity_signals_cap <= 0:
            raise ValueError("activity_signals_cap must be positive")
        if self.min_closed_for_full_confidence <= 0:
            raise ValueError("min_closed_for_full_confidence must be positive")


DEFAULT_SCORING: "ScoringConfig" = ScoringConfig()


# ────────────────────────────────────────────────────────────────────
# Pure-функции скоринга
# ────────────────────────────────────────────────────────────────────


def confidence_from_sample(
    closed_trades: int,
    *,
    config: ScoringConfig = DEFAULT_SCORING,
) -> float:
    """Confidence-коэффициент в зависимости от размера выборки.

    Возвращает [0, 1]. При 0 закрытых — 0; при ≥ min_closed_for_full_confidence
    — 1; линейная интерполяция между.

    Это заменяет в v1 россыпь магических `min(closed/8, 1.0)`,
    `min(closed/10, 1.0)` и т. п. — теперь один параметр.
    """
    if closed_trades <= 0:
        return 0.0
    capped = min(int(closed_trades), config.min_closed_for_full_confidence)
    return capped / config.min_closed_for_full_confidence


def regime_score(
    metrics: Metrics,
    regime: Regime,
    *,
    config: ScoringConfig = DEFAULT_SCORING,
    confidence: float = -1.0,  # -1 = compute from metrics.closed_trades
) -> float:
    """ЕДИНАЯ скоринговая функция для всех мест в Panteon v2.

    Аргументы:
      metrics: per-regime Metrics (от PerformanceMemory.get(label, regime))
      regime: рассматриваемый режим (для будущих regime-specific тюнингов)
      config: параметры скоринга (default = DEFAULT_SCORING)
      confidence: [0..1] переопределение confidence-коэффициента
                  (default: вычисляется из metrics.closed_trades)

    Семантика:
      - score > 0  → агент полезен для этого режима
      - score = 0  → нейтрален / нет данных
      - score < 0  → агент вредит в этом режиме

    Никаких per-class перегрузок. Если нужен другой скоринг для
    специального случая — передавайте другой `ScoringConfig`.
    """
    # Защита: пустые данные → 0
    if not metrics.has_data:
        return 0.0

    if confidence < 0:
        confidence = confidence_from_sample(metrics.closed_trades, config=config)

    # Нет закрытых сделок → даём ослабленный сигнал
    if metrics.closed_trades < 2:
        # Только небольшой штраф за необкатанность, не отрицание полностью
        return metrics.pnl_pct * 0.40 * confidence

    pnl_component     = metrics.pnl_pct * config.pnl_weight
    sharpe_component  = metrics.sharpe * config.sharpe_weight
    dd_component      = -abs(metrics.max_dd_pct) * config.max_dd_weight
    activity_component = (
        min(metrics.signals, config.activity_signals_cap)
        * config.activity_weight
    )
    win_bonus = (
        (metrics.win_rate - 50.0) / config.win_bonus_divisor
        if metrics.closed_trades >= 3
        else 0.0
    )

    score = (
        pnl_component
        + sharpe_component
        + dd_component
        + activity_component
        + win_bonus
    ) * confidence

    # Inactivity penalty: если мало сигналов — небольшое штрафование,
    # чтобы Selector предпочитал активных при равном score.
    if metrics.signals == 0 and metrics.entries == 0:
        score -= config.inactivity_penalty

    return float(score)


# ────────────────────────────────────────────────────────────────────
# Карантинная логика — pure-функции
# ────────────────────────────────────────────────────────────────────


def is_locally_proven(
    per_regime_metrics: Mapping[Regime, Metrics],
    *,
    config: ScoringConfig = DEFAULT_SCORING,
) -> bool:
    """True если в каком-либо режиме есть положительный накопленный опыт
    (pnl_pct ≥ recovery_pnl_pct при closed_trades ≥ recovery_closed).

    Используется QuarantineManager для досрочного снятия карантина.
    """
    threshold_pnl = config.quarantine_recovery_pnl_pct
    threshold_closed = config.quarantine_recovery_closed
    for metrics in per_regime_metrics.values():
        if (
            metrics.pnl_pct >= threshold_pnl
            and metrics.closed_trades >= threshold_closed
        ):
            return True
    return False


def is_hopeless_in_all_regimes(
    per_regime_metrics: Mapping[Regime, Metrics],
    *,
    config: ScoringConfig = DEFAULT_SCORING,
) -> bool:
    """True если ВО ВСЕХ режимах с данными pnl_pct < 0 и общая выборка
    достаточна для уверенного отрицательного вердикта.

    Условия:
      - есть хотя бы один режим с метриками
      - все режимы имеют pnl_pct ≤ 0
      - суммарный closed_trades ≥ hard_min_closed
      - худший pnl_pct ≤ hard_neg_pnl_pct
    """
    any_data = False
    all_neg = True
    total_closed = 0
    worst_pnl = 0.0
    for metrics in per_regime_metrics.values():
        if not metrics.has_data:
            continue
        any_data = True
        total_closed += metrics.closed_trades
        if metrics.pnl_pct > 0.0:
            all_neg = False
        if metrics.pnl_pct < worst_pnl:
            worst_pnl = metrics.pnl_pct
    return (
        any_data
        and all_neg
        and total_closed >= config.quarantine_hard_min_closed
        and worst_pnl <= config.quarantine_hard_neg_pnl_pct
    )
