"""V3 rolling/decayed decision score for allocator selection."""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.types import Metrics


@dataclass(frozen=True)
class RollingDecisionScoreConfig:
    no_data_score: float = -0.25
    recent_real_weight: float = 1.35
    long_real_weight: float = 0.55
    virtual_weight: float = 0.12
    pnl_weight: float = 0.85
    expectancy_weight: float = 5.5
    win_rate_weight: float = 0.65
    sharpe_weight: float = 0.15
    drawdown_weight: float = 0.20
    recent_full_confidence_trades: int = 8
    long_full_confidence_trades: int = 20
    virtual_full_confidence_trades: int = 40
    negative_real_penalty_weight: float = 1.25
    real_loss_override_trades: int = 3


@dataclass(frozen=True)
class RollingDecisionScoreInput:
    recent_real_metrics: Metrics = Metrics.empty()
    long_real_metrics: Metrics = Metrics.empty()
    virtual_metrics: Metrics = Metrics.empty()


@dataclass(frozen=True)
class RollingDecisionScore:
    score: float
    source: str
    recent_real_component: float = 0.0
    long_real_component: float = 0.0
    virtual_component: float = 0.0
    negative_real_penalty: float = 0.0


def score_rolling_decision(
    payload: RollingDecisionScoreInput,
    *,
    config: RollingDecisionScoreConfig = RollingDecisionScoreConfig(),
) -> RollingDecisionScore:
    """Score a player for allocator decisions using real-first rolling evidence."""

    recent_has_data = payload.recent_real_metrics.has_data
    long_has_data = payload.long_real_metrics.has_data
    virtual_has_data = payload.virtual_metrics.has_data
    if not (recent_has_data or long_has_data or virtual_has_data):
        return RollingDecisionScore(score=float(config.no_data_score), source="no_data")

    recent_component = (
        _metric_component(
            payload.recent_real_metrics,
            full_confidence_trades=config.recent_full_confidence_trades,
            config=config,
        )
        if recent_has_data
        else 0.0
    )
    long_component = (
        _metric_component(
            payload.long_real_metrics,
            full_confidence_trades=config.long_full_confidence_trades,
            config=config,
        )
        if long_has_data
        else 0.0
    )
    virtual_component = (
        _metric_component(
            payload.virtual_metrics,
            full_confidence_trades=config.virtual_full_confidence_trades,
            config=config,
        )
        if virtual_has_data
        else 0.0
    )
    negative_penalty = _negative_real_penalty(payload, config=config)
    score = (
        config.recent_real_weight * recent_component
        + config.long_real_weight * long_component
        + config.virtual_weight * virtual_component
        - negative_penalty
    )

    if recent_has_data or long_has_data:
        source = "real+virtual" if virtual_has_data else "real"
    else:
        source = "virtual"
    return RollingDecisionScore(
        score=float(score),
        source=source,
        recent_real_component=recent_component,
        long_real_component=long_component,
        virtual_component=virtual_component,
        negative_real_penalty=negative_penalty,
    )


def _metric_component(
    metrics: Metrics,
    *,
    full_confidence_trades: int,
    config: RollingDecisionScoreConfig,
) -> float:
    if not metrics.has_data:
        return 0.0
    if metrics.closed_trades <= 0:
        return -0.05 if metrics.signals > 0 else 0.0

    confidence = min(1.0, metrics.closed_trades / max(1, full_confidence_trades))
    expectancy = metrics.pnl_per_trade
    win_edge = (metrics.win_rate - 50.0) / 50.0
    raw = (
        metrics.pnl_pct * config.pnl_weight
        + expectancy * config.expectancy_weight
        + win_edge * config.win_rate_weight
        + metrics.sharpe * config.sharpe_weight
        - abs(metrics.max_dd_pct) * config.drawdown_weight
    )
    return float(raw * confidence)


def _negative_real_penalty(
    payload: RollingDecisionScoreInput,
    *,
    config: RollingDecisionScoreConfig,
) -> float:
    penalty = 0.0
    for metrics, weight in (
        (payload.recent_real_metrics, config.recent_real_weight),
        (payload.long_real_metrics, config.long_real_weight),
    ):
        if (
            metrics.closed_trades >= config.real_loss_override_trades
            and metrics.pnl_pct < 0.0
        ):
            penalty += abs(metrics.pnl_pct) * config.negative_real_penalty_weight * weight
    return float(penalty)
