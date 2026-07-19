"""Causal player-only selection from one player×regime memory source."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple

from ..domain.types import Regime
from ..memory.performance import PerformanceMemory
from .player import NoTradePlayer, Player
from .strategist import CandidateRejection, CandidateScore, SwitchDecision


@dataclass(frozen=True)
class PlayerRegimeConfig:
    """Small, explicit contract for the production player rating."""

    recency_decay: float = 0.94
    return_window: int = 100
    min_closed_trades: int = 5
    min_global_closed_trades: int = 20
    global_score_weight: float = 0.25
    downside_penalty: float = 0.25
    min_positive_score: float = 0.0
    switch_margin: float = 0.02
    cooldown_bars: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < float(self.recency_decay) <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        if int(self.return_window) <= 0:
            raise ValueError("return_window must be positive")
        if int(self.min_closed_trades) <= 0:
            raise ValueError("min_closed_trades must be positive")
        if int(self.min_global_closed_trades) < 0:
            raise ValueError("min_global_closed_trades must be non-negative")
        if not 0.0 <= float(self.global_score_weight) <= 1.0:
            raise ValueError("global_score_weight must be in [0, 1]")
        if float(self.downside_penalty) < 0.0:
            raise ValueError("downside_penalty must be non-negative")
        if float(self.switch_margin) < 0.0:
            raise ValueError("switch_margin must be non-negative")
        if int(self.cooldown_bars) < 0:
            raise ValueError("cooldown_bars must be non-negative")


@dataclass(frozen=True)
class PlayerRegimeRating:
    label: str
    regime: Regime
    score: float
    weighted_pnl_per_trade_pct: float
    weighted_win_rate: float
    downside_rms_pct: float
    effective_trades: float
    closed_trades: int
    eligible: bool
    reason: str = ""
    global_score: float = 0.0
    global_closed_trades: int = 0
    global_eligible: bool = True


def rate_player_returns(
    label: str,
    regime: Regime,
    returns: Sequence[float],
    *,
    config: PlayerRegimeConfig,
) -> PlayerRegimeRating:
    """Rate retained trade returns with the newest observation weighted most."""
    values = tuple(float(value) for value in returns)[-config.return_window:]
    if not values:
        return PlayerRegimeRating(
            label=label,
            regime=regime,
            score=0.0,
            weighted_pnl_per_trade_pct=0.0,
            weighted_win_rate=0.0,
            downside_rms_pct=0.0,
            effective_trades=0.0,
            closed_trades=0,
            eligible=False,
            reason="no player/regime trades",
        )

    newest_first = tuple(reversed(values))
    weights = tuple(config.recency_decay ** age for age in range(len(newest_first)))
    weight_sum = sum(weights)
    weight_square_sum = sum(weight * weight for weight in weights)
    weighted_mean = sum(
        weight * value for weight, value in zip(weights, newest_first)
    ) / weight_sum
    weighted_win_rate = sum(
        weight for weight, value in zip(weights, newest_first) if value > 0.0
    ) / weight_sum
    downside_rms = math.sqrt(
        sum(
            weight * min(0.0, value) ** 2
            for weight, value in zip(weights, newest_first)
        )
        / weight_sum
    )
    effective_trades = (
        (weight_sum * weight_sum) / weight_square_sum
        if weight_square_sum > 0.0
        else 0.0
    )
    confidence = min(1.0, effective_trades / float(config.min_closed_trades))
    positive_efficiency = weighted_mean * (0.5 + weighted_win_rate)
    score = positive_efficiency * confidence - config.downside_penalty * downside_rms

    if len(values) < config.min_closed_trades:
        eligible = False
        reason = (
            f"insufficient player/regime trades: {len(values)}/"
            f"{config.min_closed_trades}"
        )
    elif score <= config.min_positive_score:
        eligible = False
        reason = "non-positive recency-weighted efficiency"
    else:
        eligible = True
        reason = "positive recency-weighted efficiency"

    return PlayerRegimeRating(
        label=label,
        regime=regime,
        score=float(score),
        weighted_pnl_per_trade_pct=float(weighted_mean),
        weighted_win_rate=float(weighted_win_rate),
        downside_rms_pct=float(downside_rms),
        effective_trades=float(effective_trades),
        closed_trades=len(values),
        eligible=eligible,
        reason=reason,
    )


class PlayerRegimeStrategist:
    """Select exactly one player, or NoTrade, from prior shadow memory."""

    def __init__(
        self,
        perf: PerformanceMemory,
        *,
        config: PlayerRegimeConfig | None = None,
        fixed_player_label: str = "",
    ) -> None:
        self._perf = perf
        self._config = config or PlayerRegimeConfig()
        self._fixed_player_label = str(fixed_player_label or "").strip()
        self._candidates: dict[str, Player] = {}
        self._current: Optional[Player] = None
        self._no_trade = NoTradePlayer()
        self._last_switch_bar = -10**12
        self._last_regime: Optional[Regime] = None

    def update_candidates(self, candidates: Sequence[Player]) -> None:
        unique: dict[str, Player] = {}
        for player in candidates:
            label = str(getattr(player, "label", "") or "").strip()
            if label and label != self._no_trade.label and label not in unique:
                unique[label] = player
        self._candidates = unique
        if self._current is not None and self._current.label in unique:
            self._current = unique[self._current.label]

    def current_leader(self) -> Optional[Player]:
        return self._current

    def capture_session_baseline(self, regime: Optional[Regime] = None) -> None:
        del regime

    def update_shadow_actor_updates(self, updates: Sequence[object]) -> None:
        del updates

    def update_shadow_position_snapshot(
        self,
        *,
        player_positions: object,
        real_positions: object,
    ) -> None:
        del player_positions, real_positions

    def consider_switch(
        self,
        regime: Regime,
        current_bar: int,
        *,
        regime_confidence: float = 1.0,
        market_tags: Iterable[str] = (),
        portfolio_flat: bool = True,
    ) -> SwitchDecision:
        del regime_confidence, market_tags
        rows, ratings = self._rank(regime)
        rejections = tuple(
            CandidateRejection(label=rating.label, reason=rating.reason)
            for rating in ratings
            if not rating.eligible
        )
        regime_changed = self._last_regime is not None and self._last_regime != regime
        self._last_regime = regime

        if self._fixed_player_label:
            player = self._candidates.get(self._fixed_player_label)
            if player is None:
                return self._select_no_trade(
                    current_bar=current_bar,
                    rows=rows,
                    rejections=rejections,
                    reason=f"fixed singleton player unavailable: {self._fixed_player_label}",
                )
            rating = next(
                (item for item in ratings if item.label == player.label),
                rate_player_returns(player.label, regime, (), config=self._config),
            )
            return self._switch_to(
                player,
                rating=rating,
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="fixed singleton player",
                urgent=True,
            )

        eligible = [rating for rating in ratings if rating.eligible]
        if not eligible:
            if (
                self._current is not None
                and self._current.label != self._no_trade.label
                and not portfolio_flat
            ):
                current_rating = next(
                    (
                        rating
                        for rating in ratings
                        if rating.label == self._current.label
                    ),
                    rate_player_returns(
                        self._current.label,
                        regime,
                        (),
                        config=self._config,
                    ),
                )
                return self._keep_current(
                    current_rating,
                    current_bar=current_bar,
                    rows=rows,
                    rejections=rejections,
                    reason=(
                        "NoTrade deferred until the real portfolio is flat; "
                        "current player is manage-only"
                    ),
                    manage_only=True,
                )
            return self._select_no_trade(
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="no player has positive efficiency for current regime",
            )

        best = eligible[0]
        best_player = self._candidates[best.label]
        if self._current is None or self._current.label == self._no_trade.label:
            return self._switch_to(
                best_player,
                rating=best,
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="best positive player for regime",
                urgent=True,
            )

        current_rating = next(
            (rating for rating in ratings if rating.label == self._current.label),
            None,
        )
        if current_rating is None or not current_rating.eligible:
            if not portfolio_flat:
                current_rating = current_rating or rate_player_returns(
                    self._current.label,
                    regime,
                    (),
                    config=self._config,
                )
                return self._keep_current(
                    current_rating,
                    current_bar=current_bar,
                    rows=rows,
                    rejections=rejections,
                    reason=(
                        "player switch deferred until the real portfolio is flat; "
                        "current player is manage-only"
                    ),
                    best=best,
                    manage_only=True,
                )
            return self._switch_to(
                best_player,
                rating=best,
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="current player lost positive efficiency",
                urgent=True,
            )
        if best.label == self._current.label:
            return self._keep_current(
                current_rating,
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="current player remains best for regime",
            )

        margin = best.score - current_rating.score
        cooldown_passed = (
            int(current_bar) - int(self._last_switch_bar) >= self._config.cooldown_bars
        )
        if regime_changed:
            if not portfolio_flat:
                return self._keep_current(
                    current_rating,
                    current_bar=current_bar,
                    rows=rows,
                    rejections=rejections,
                    reason=(
                        "regime switch deferred until the real portfolio is flat; "
                        "current player is manage-only"
                    ),
                    best=best,
                    manage_only=True,
                )
            return self._switch_to(
                best_player,
                rating=best,
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="market regime changed",
                urgent=True,
                current_rating=current_rating,
            )
        if margin >= self._config.switch_margin and cooldown_passed:
            if not portfolio_flat:
                return self._keep_current(
                    current_rating,
                    current_bar=current_bar,
                    rows=rows,
                    rejections=rejections,
                    reason=(
                        "better player deferred until the real portfolio is flat; "
                        "current player is manage-only"
                    ),
                    best=best,
                    cooldown_passed=cooldown_passed,
                    manage_only=True,
                )
            return self._switch_to(
                best_player,
                rating=best,
                current_bar=current_bar,
                rows=rows,
                rejections=rejections,
                reason="better positive player confirmed",
                urgent=False,
                current_rating=current_rating,
            )
        reason = "switch margin too small" if margin < self._config.switch_margin else "switch cooldown active"
        return self._keep_current(
            current_rating,
            current_bar=current_bar,
            rows=rows,
            rejections=rejections,
            reason=reason,
            best=best,
            cooldown_passed=cooldown_passed,
        )

    def _rank(
        self,
        regime: Regime,
    ) -> Tuple[Tuple[CandidateScore, ...], Tuple[PlayerRegimeRating, ...]]:
        rated = []
        for label in self._candidates:
            returns = self._perf.recent_returns(
                label,
                regime,
                limit=self._config.return_window,
            )
            regime_rating = rate_player_returns(
                label,
                regime,
                returns,
                config=self._config,
            )
            if self._config.min_global_closed_trades <= 0:
                rated.append(regime_rating)
                continue
            global_returns = self._perf.recent_returns(
                label,
                None,
                limit=self._config.return_window,
            )
            global_config = replace(
                self._config,
                min_closed_trades=self._config.min_global_closed_trades,
                min_global_closed_trades=0,
                global_score_weight=0.0,
            )
            global_rating = rate_player_returns(
                label,
                regime,
                global_returns,
                config=global_config,
            )
            global_weight = float(self._config.global_score_weight)
            combined_score = (
                (1.0 - global_weight) * regime_rating.score
                + global_weight * global_rating.score
            )
            eligible = regime_rating.eligible and global_rating.eligible
            reason = regime_rating.reason
            if not global_rating.eligible:
                reason = f"global player efficiency gate: {global_rating.reason}"
            elif regime_rating.eligible:
                reason = "positive regime and global recency-weighted efficiency"
            rated.append(replace(
                regime_rating,
                score=combined_score,
                eligible=eligible,
                reason=reason,
                global_score=global_rating.score,
                global_closed_trades=global_rating.closed_trades,
                global_eligible=global_rating.eligible,
            ))
        rated.sort(key=lambda row: (-row.score, -row.closed_trades, row.label))
        rows = []
        for rank, rating in enumerate(rated, start=1):
            metrics = self._perf.get(rating.label, regime=regime)
            rows.append(CandidateScore(
                label=rating.label,
                score=rating.score,
                rank=rank,
                has_data=rating.closed_trades > 0,
                closed_trades=rating.closed_trades,
                signals=metrics.signals,
                execution_failures=metrics.execution_failures,
                score_source=(
                    "player_regime_global_recency"
                    if self._config.min_global_closed_trades > 0
                    else "player_regime_recency"
                ),
                uncertainty_penalty=rating.downside_rms_pct * self._config.downside_penalty,
                agent_labels=(),
                memory_keys_read=(
                    (f"{rating.label}|{regime.label}", f"{rating.label}|all")
                    if self._config.min_global_closed_trades > 0
                    else (f"{rating.label}|{regime.label}",)
                ),
                session_pnl_pct=rating.weighted_pnl_per_trade_pct,
                actionable_share=rating.weighted_win_rate,
                recent_filled=rating.closed_trades,
            ))
        return tuple(rows), tuple(rated)

    def _switch_to(
        self,
        player: Player,
        *,
        rating: PlayerRegimeRating,
        current_bar: int,
        rows: Tuple[CandidateScore, ...],
        rejections: Tuple[CandidateRejection, ...],
        reason: str,
        urgent: bool,
        current_rating: Optional[PlayerRegimeRating] = None,
    ) -> SwitchDecision:
        previous = self._current
        switched = previous is None or previous.label != player.label
        if switched:
            self._current = player
            self._last_switch_bar = int(current_bar)
        else:
            self._current = player
        current_score = current_rating.score if current_rating is not None else (
            rating.score if previous is not None and previous.label == player.label else 0.0
        )
        return SwitchDecision(
            new_leader=player,
            previous=previous,
            score=rating.score,
            margin=rating.score - current_score,
            is_urgent=urgent,
            reason=reason,
            switched=switched,
            candidate_scores=rows,
            candidate_rejections=rejections,
            switch_gate_reason=reason,
            best_label=player.label,
            current_label=previous.label if previous is not None else "",
            best_score=rating.score,
            current_score=current_score,
            required_margin=self._config.switch_margin,
            cooldown_passed=True,
            cooldown_blocked=False,
            streak_count=1,
            streak_needed=1,
        )

    def _keep_current(
        self,
        current: PlayerRegimeRating,
        *,
        current_bar: int,
        rows: Tuple[CandidateScore, ...],
        rejections: Tuple[CandidateRejection, ...],
        reason: str,
        best: Optional[PlayerRegimeRating] = None,
        cooldown_passed: bool = True,
        manage_only: bool = False,
    ) -> SwitchDecision:
        del current_bar
        assert self._current is not None
        best = best or current
        return SwitchDecision(
            new_leader=self._current,
            previous=self._current,
            score=current.score,
            margin=best.score - current.score,
            is_urgent=False,
            reason=reason,
            switched=False,
            candidate_scores=rows,
            candidate_rejections=rejections,
            switch_gate_reason=reason,
            best_label=best.label,
            current_label=self._current.label,
            best_score=best.score,
            current_score=current.score,
            required_margin=self._config.switch_margin,
            cooldown_passed=cooldown_passed,
            cooldown_blocked=not cooldown_passed,
            streak_count=1,
            streak_needed=1,
            manage_only=manage_only,
        )

    def _select_no_trade(
        self,
        *,
        current_bar: int,
        rows: Tuple[CandidateScore, ...],
        rejections: Tuple[CandidateRejection, ...],
        reason: str,
    ) -> SwitchDecision:
        previous = self._current
        switched = previous is None or previous.label != self._no_trade.label
        self._current = self._no_trade
        if switched:
            self._last_switch_bar = int(current_bar)
        return SwitchDecision(
            new_leader=self._no_trade,
            previous=previous,
            score=0.0,
            margin=0.0,
            is_urgent=True,
            reason=reason,
            switched=switched,
            candidate_scores=rows,
            candidate_rejections=rejections,
            switch_gate_reason=reason,
            best_label=self._no_trade.label,
            current_label=previous.label if previous is not None else "",
            best_score=0.0,
            current_score=0.0,
            required_margin=self._config.switch_margin,
            cooldown_passed=True,
            cooldown_blocked=False,
            streak_count=0,
            streak_needed=1,
        )
