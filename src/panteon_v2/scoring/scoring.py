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

    # Per-trade PnL компонента (live no-trade fix). При True pnl_component берётся
    # из СРЕДНЕГО PnL на сделку, а не кумулятивного pnl_pct. Это убирает перекос,
    # где безубыточный актор с сотнями сделок (cum≈−6.75% = −0.01%/сделку) получает
    # огромный отрицательный score и навсегда блокируется гейтом. Default False —
    # сохраняет историческую калибровку (min_score_to_trade и т.п.). Включать
    # вместе с рекалибровкой порогов и ретро-валидацией.
    use_per_trade_pnl: bool = False
    # Множитель шкалы для per-trade pnl (per-trade ~ в 100x меньше кумулятива).
    # Приводит per-trade компоненту к сопоставимому со sharpe/win_bonus масштабу.
    per_trade_pnl_scale: float = 1.0

    # Cap-ы для активности (чтобы не давать мега-бонус за shotgun-стрельбу)
    activity_signals_cap: int = 30

    # Минимальная выборка для уверенности
    min_closed_for_full_confidence: int = 6

    # Карантин: пороги (см. is_locally_proven, is_hopeless_in_all_regimes)
    quarantine_recovery_pnl_pct: float = 0.10  # один режим с +0.10% PnL → reнабором
    quarantine_recovery_closed:  int   = 3
    quarantine_hard_neg_pnl_pct: float = -0.30  # все режимы хуже −0.30% и ≥ 5 closed → карантин
    quarantine_hard_min_closed:  int   = 5
    quarantine_dominant_loss_closed_share: float = 0.80

    # Минимальный score для попадания в Selector top-k
    min_eligible_score: float = 0.0

    # Penalty за полное отсутствие активности
    inactivity_penalty: float = 0.65

    def __post_init__(self) -> None:
        if self.activity_signals_cap <= 0:
            raise ValueError("activity_signals_cap must be positive")
        if self.min_closed_for_full_confidence <= 0:
            raise ValueError("min_closed_for_full_confidence must be positive")
        if not (0.0 < self.quarantine_dominant_loss_closed_share <= 1.0):
            raise ValueError("quarantine_dominant_loss_closed_share must be in (0, 1]")


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
    per_regime_configs: Mapping[Regime, ScoringConfig] | None = None,
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

    if per_regime_configs:
        config = per_regime_configs.get(regime, config)

    if confidence < 0:
        confidence = confidence_from_sample(metrics.closed_trades, config=config)

    # Единая формула: sharpe и win_bonus гасятся на малых выборках через свои
    # собственные ворота (closed >= 2 / closed >= 3), но pnl и просадка
    # учитываются всегда. Это снимает немонотонность в pnl_pct при closed < 2.

    if config.use_per_trade_pnl:
        pnl_component = (
            metrics.pnl_per_trade * config.per_trade_pnl_scale * config.pnl_weight
        )
    else:
        pnl_component = metrics.pnl_pct * config.pnl_weight
    sharpe_component  = (
        metrics.sharpe * config.sharpe_weight
        if metrics.closed_trades >= 2
        else 0.0
    )
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
    if metrics.closed_trades == 0 and metrics.signals == 0 and metrics.entries == 0:
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
    worst_metrics: Metrics | None = None
    data_metrics: list[Metrics] = []
    for metrics in per_regime_metrics.values():
        if not metrics.has_data:
            continue
        any_data = True
        data_metrics.append(metrics)
        total_closed += metrics.closed_trades
        if metrics.pnl_pct > 0.0:
            all_neg = False
        if metrics.pnl_pct < worst_pnl:
            worst_pnl = metrics.pnl_pct
            worst_metrics = metrics

    # Phase 3 / C3: явный защитный guard — агента с НЕОТРИЦАТЕЛЬНЫМ агрегатным
    # PnL по всем режимам с данными никогда не считаем безнадёжным. Это
    # кодифицирует намерение (не выбивать чистоприбыльных) и не даёт одному
    # шумному убыточному режиму вычистить полезного агента из пула.
    if any_data and sum(m.pnl_pct for m in data_metrics) >= 0.0:
        return False
    # Keep meaningful positive regimes alive, but do not let a tiny positive
    # side sample mask a catastrophic loss in the dominant traded regime.
    dominant_loss = False
    if (
        worst_metrics is not None
        and total_closed >= config.quarantine_hard_min_closed
        and worst_pnl <= config.quarantine_hard_neg_pnl_pct
    ):
        aggregate_pnl = sum(metrics.pnl_pct for metrics in data_metrics)
        closed_share = (
            worst_metrics.closed_trades / total_closed
            if total_closed > 0
            else 0.0
        )
        dominant_loss = (
            aggregate_pnl <= config.quarantine_hard_neg_pnl_pct
            and worst_metrics.closed_trades >= config.quarantine_hard_min_closed
            and closed_share >= config.quarantine_dominant_loss_closed_share
        )
    return (
        any_data
        and (all_neg or dominant_loss)
        and total_closed >= config.quarantine_hard_min_closed
        and worst_pnl <= config.quarantine_hard_neg_pnl_pct
    )
