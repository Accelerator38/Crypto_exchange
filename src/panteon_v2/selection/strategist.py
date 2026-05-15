"""Strategist — выбор текущего лидера-игрока.

Strategist решает: какой Player станет current_leader на этом баре.
Никаких параллельных механизмов (`_active_weights` + `_selected_shadow_player`
из v1) — один лидер.

ГАРАНТИИ (Q3, Q4 из docs/PANTEON_V2_ARCHITECTURE.md):
  • Q3: Strategist всегда использует тот же regime, что был передан.
  • Q4: ни один лидер не имеет в Player.agent_labels карантинного агента.
        Это проверяется здесь как sanity check — даже если в кандидаты
        попал нечестный игрок, он отбрасывается.

Логика выбора:
  1. Кандидаты — список Player'ов (обычно: один EnsemblePlayer + несколько
     профильных EnsemblePlayer'ов с разными affinity).
  2. Каждого скорим через PerformanceMemory + regime_score.
  3. Best vs current — переключение по правилам:
       • cooldown баров с последней смены
       • margin: best.score >= current.score + margin
       • urgent: current.score <= hard_negative ИЛИ gap >= urgent_gap
       • streak: confirm `streak_needed` баров подряд (anti-flapping)

Анти-flapping: smena leader-а — серьёзная операция. Без правил можно
скакать каждый бар. Параметры см. StrategistConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..domain.types import Metrics, Regime
from ..memory import PerformanceMemory, QuarantineManager
from ..scoring import DEFAULT_SCORING, ScoringConfig, regime_score
from .player import Player


# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategistConfig:
    """Параметры выбора лидера.

    Заменяет в v1 magic numbers:
      PLAYER_SWITCH_COOLDOWN_BARS, PLAYER_SWITCH_MARGIN,
      PLAYER_HARD_NEGATIVE_SCORE, PLAYER_SWITCH_CONFIRMATIONS, etc.
    """

    cooldown_bars:    int   = 30      # минимум баров между сменами (anti-flapping)
    switch_margin:    float = 0.30    # best.score >= current.score + margin
    min_score_to_switch: float = -0.05  # лидер не должен иметь ужасный score
    streak_needed:    int   = 2       # baрs подряд best должен быть лучшим
    hard_negative:    float = -0.50   # current.score ≤ hard_negative → urgent
    urgent_gap:       float = 0.80    # best - current ≥ gap → urgent
    affinity_bonus:   float = 0.05
    affinity_mismatch_penalty: float = 0.02
    min_live_closed_trades: int = 0
    min_live_score: float = -1_000_000.0
    max_live_drawdown_pct: float = 100.0
    min_regime_confidence: float = 0.0
    real_score_weight: float = 0.35
    real_min_closed_trades: int = 3
    zero_score_uncertainty_penalty: float = 0.05
    execution_failure_score_penalty: float = 0.02

    def __post_init__(self) -> None:
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be ≥ 0")
        if self.switch_margin < 0:
            raise ValueError("switch_margin must be ≥ 0")
        if self.streak_needed < 1:
            raise ValueError("streak_needed must be ≥ 1")
        if self.affinity_bonus < 0 or self.affinity_mismatch_penalty < 0:
            raise ValueError("affinity score adjustments must be >= 0")
        if self.min_live_closed_trades < 0:
            raise ValueError("min_live_closed_trades must be >= 0")
        if self.max_live_drawdown_pct < 0:
            raise ValueError("max_live_drawdown_pct must be >= 0")
        if not 0.0 <= self.min_regime_confidence <= 1.0:
            raise ValueError("min_regime_confidence must be in [0, 1]")
        if self.real_score_weight < 0:
            raise ValueError("real_score_weight must be >= 0")
        if self.real_min_closed_trades < 0:
            raise ValueError("real_min_closed_trades must be >= 0")
        if self.zero_score_uncertainty_penalty < 0:
            raise ValueError("zero_score_uncertainty_penalty must be >= 0")
        if self.execution_failure_score_penalty < 0:
            raise ValueError("execution_failure_score_penalty must be >= 0")


DEFAULT_STRATEGIST = StrategistConfig()


# ────────────────────────────────────────────────────────────────────
# Decision result
# ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CandidateScore:
    label: str
    score: float
    rank: int
    has_data: bool
    closed_trades: int
    signals: int
    execution_failures: int
    score_source: str
    uncertainty_penalty: float
    agent_labels: Tuple[str, ...]
    memory_keys_read: Tuple[str, ...]


@dataclass(frozen=True)
class CandidateRejection:
    label: str
    reason: str


@dataclass(frozen=True)
class SwitchDecision:
    """Результат consider_switch."""

    new_leader:  Player
    previous:    Optional[Player]
    score:       float
    margin:      float
    is_urgent:   bool
    reason:      str
    switched:    bool   # True если сменили; False если оставили текущего
    candidate_scores: Tuple[CandidateScore, ...] = ()
    candidate_rejections: Tuple[CandidateRejection, ...] = ()

    @property
    def label(self) -> str:
        return self.new_leader.label


# Sentinel — внутренний результат когда не выбран никто
@dataclass(frozen=True)
class _DisqualificationReason:
    label: str
    reason: str


# ────────────────────────────────────────────────────────────────────
# Strategist
# ────────────────────────────────────────────────────────────────────


class Strategist:
    """Один Strategist — один current_leader.

    Жизненный цикл:
        st = Strategist(perf, qm, candidates=[player1, player2, ...])
        # ... в bar loop:
        decision = st.consider_switch(regime=Regime.BULLISH, current_bar=N)
        leader = decision.new_leader  # тот же или новый
        signals = leader.vote(market, signal_id_start=...)

    candidates обновляется через update_candidates() — обычно после
    re-compose в PlayerComposer (когда AgentSelector выдал новый набор).
    """

    def __init__(
        self,
        perf: PerformanceMemory,
        qm:   QuarantineManager,
        candidates: Sequence[Player] = (),
        *,
        config: StrategistConfig = DEFAULT_STRATEGIST,
        scoring_config: ScoringConfig = DEFAULT_SCORING,
        real_perf: Optional[PerformanceMemory] = None,
    ):
        self._perf = perf
        self._real_perf = real_perf
        self._qm = qm
        self._candidates: List[Player] = list(candidates)
        self._config = config
        self._scoring = scoring_config
        self._current: Optional[Player] = None
        self._last_switch_bar: int = -10**9
        # streak tracking: какой кандидат был best на прошлых барах
        self._streak_label: str = ""
        self._streak_count: int = 0

    # ── Public API ──────────────────────────────────────────────────

    def update_candidates(self, candidates: Sequence[Player]) -> None:
        """Обновить список кандидатов. Если current не в новом списке —
        он всё равно остаётся current до следующего switch."""
        self._candidates = list(candidates)

    def current_leader(self) -> Optional[Player]:
        return self._current

    def consider_switch(
        self,
        regime: Regime,
        current_bar: int,
        regime_confidence: float = 1.0,
    ) -> SwitchDecision:
        """Главная функция. Решает: сменить лидера или оставить.

        Возвращает SwitchDecision со ссылкой на нового лидера
        (или того же самого, если switched=False).
        """
        # 1. Фильтруем кандидатов: убираем тех, кто использует карантинных
        eligible: List[Player] = []
        disqualified: List[_DisqualificationReason] = []
        for cand in self._candidates:
            issue = self._validate(cand)
            if issue is None:
                issue = self._validate_promotion(
                    cand,
                    regime,
                    regime_confidence=regime_confidence,
                )
            if issue is None:
                eligible.append(cand)
            else:
                disqualified.append(issue)
        candidate_rejections = tuple(
            CandidateRejection(label=d.label, reason=d.reason)
            for d in disqualified
        )

        if not eligible:
            # Нет валидных кандидатов. Если есть current, оставляем.
            # Но если current тоже не валиден — возвращаем "no leader".
            if self._current is not None and self._validate(self._current) is None:
                return SwitchDecision(
                    new_leader=self._current,
                    previous=self._current,
                    score=0.0,
                    margin=0.0,
                    is_urgent=False,
                    reason="no eligible candidates; keeping current",
                    switched=False,
                    candidate_rejections=candidate_rejections,
                )
            raise ValueError(
                "No eligible candidates and current leader is invalid; "
                f"disqualified: {[(d.label, d.reason) for d in disqualified]}"
            )

        # 2. Скорим каждого
        scored_rows = self._rank_candidates(eligible, regime)
        by_label = {cand.label: cand for cand in eligible}
        best_row = scored_rows[0]
        best_score = best_row.score
        best_player = by_label[best_row.label]

        # 3. Если current нет — выбираем best
        if self._current is None:
            self._current = best_player
            self._last_switch_bar = int(current_bar)
            self._streak_label = best_player.label
            self._streak_count = 1
            return SwitchDecision(
                new_leader=best_player,
                previous=None,
                score=best_score,
                margin=0.0,
                is_urgent=False,
                reason="bootstrap (no current)",
                switched=True,
                candidate_scores=scored_rows,
                candidate_rejections=candidate_rejections,
            )

        # 4. Current есть — оцениваем нужно ли менять
        current_in_candidates = any(
            cand.label == self._current.label for cand in eligible
        )
        if not current_in_candidates:
            previous = self._current
            self._current = best_player
            self._last_switch_bar = int(current_bar)
            self._streak_label = ""
            self._streak_count = 0
            return SwitchDecision(
                new_leader=best_player,
                previous=previous,
                score=best_score,
                margin=0.0,
                is_urgent=True,
                reason="current not in candidates",
                switched=True,
                candidate_scores=scored_rows,
                candidate_rejections=candidate_rejections,
            )

        current_score = self._score_player(self._current, regime)
        margin = best_score - current_score
        urgent = (
            current_score <= self._config.hard_negative
            or margin >= self._config.urgent_gap
        )
        cooldown_passed = (
            current_bar - self._last_switch_bar >= self._config.cooldown_bars
        )

        # Проверка best vs current
        if best_player.label == self._current.label:
            # Best — это сам current. Сбрасываем streak.
            self._streak_label = ""
            self._streak_count = 0
            return SwitchDecision(
                new_leader=self._current,
                previous=self._current,
                score=current_score,
                margin=0.0,
                is_urgent=False,
                reason="current is still best",
                switched=False,
                candidate_scores=scored_rows,
                candidate_rejections=candidate_rejections,
            )

        # 5. Best ≠ current. Проверяем margin и cooldown.
        ready_by_margin = (
            margin >= self._config.switch_margin
            and best_score >= self._config.min_score_to_switch
        )

        if not ready_by_margin and not urgent:
            # Не готовы менять. Но трекаем streak.
            self._streak_label = ""
            self._streak_count = 0
            return SwitchDecision(
                new_leader=self._current,
                previous=self._current,
                score=current_score,
                margin=margin,
                is_urgent=urgent,
                reason="margin too small and not urgent",
                switched=False,
                candidate_scores=scored_rows,
                candidate_rejections=candidate_rejections,
            )

        # 6. Урgent или прошёл cooldown? Иначе streak-ловим.
        if urgent or cooldown_passed:
            # Streak-confirm: если уже подтверждали — switch
            if self._streak_label == best_player.label:
                self._streak_count += 1
            else:
                self._streak_label = best_player.label
                self._streak_count = 1
            if urgent or self._streak_count >= self._config.streak_needed:
                # SWITCH!
                previous = self._current
                self._current = best_player
                self._last_switch_bar = int(current_bar)
                self._streak_label = ""
                self._streak_count = 0
                return SwitchDecision(
                    new_leader=best_player,
                    previous=previous,
                    score=best_score,
                    margin=margin,
                    is_urgent=urgent,
                    reason="urgent" if urgent else "streak confirmed",
                    switched=True,
                    candidate_scores=scored_rows,
                    candidate_rejections=candidate_rejections,
                )
            # Streak ещё не подтверждён, ждём.
            return SwitchDecision(
                new_leader=self._current,
                previous=self._current,
                score=current_score,
                margin=margin,
                is_urgent=False,
                reason=f"awaiting streak confirmation ({self._streak_count}/{self._config.streak_needed})",
                switched=False,
                candidate_scores=scored_rows,
                candidate_rejections=candidate_rejections,
            )

        # 7. Не урgent и cooldown не прошёл — ждём.
        return SwitchDecision(
            new_leader=self._current,
            previous=self._current,
            score=current_score,
            margin=margin,
            is_urgent=False,
            reason="cooldown not yet passed",
            switched=False,
            candidate_scores=scored_rows,
            candidate_rejections=candidate_rejections,
        )

    # ── Internal ────────────────────────────────────────────────────

    def _validate(self, player: Player) -> Optional[_DisqualificationReason]:
        """Q4 sanity check: лидер не должен использовать карантинных."""
        for label in player.agent_labels:
            if self._qm.is_quarantined(label):
                return _DisqualificationReason(
                    label=player.label,
                    reason=f"uses quarantined agent {label!r}",
                )
        return None

    def _validate_promotion(
        self,
        player: Player,
        regime: Regime,
        *,
        regime_confidence: float,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        if regime_confidence < cfg.min_regime_confidence:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"regime confidence {regime_confidence:.2f} below "
                    f"{cfg.min_regime_confidence:.2f}"
                ),
            )
        if (
            cfg.min_live_closed_trades <= 0
            and cfg.min_live_score <= -999_999.0
            and cfg.max_live_drawdown_pct >= 100.0
        ):
            return None
        metrics = self._promotion_metrics(player, regime)
        score = self._score_player(player, regime)
        if metrics.closed_trades < cfg.min_live_closed_trades:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"promotion gate: closed_trades "
                    f"{metrics.closed_trades} < {cfg.min_live_closed_trades}"
                ),
            )
        if score < cfg.min_live_score:
            return _DisqualificationReason(
                label=player.label,
                reason=f"promotion gate: score {score:.4f} < {cfg.min_live_score:.4f}",
            )
        if metrics.max_dd_pct > cfg.max_live_drawdown_pct:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"promotion gate: max_dd_pct {metrics.max_dd_pct:.2f} > "
                    f"{cfg.max_live_drawdown_pct:.2f}"
                ),
            )
        return None

    def _promotion_metrics(self, player: Player, regime: Regime) -> Metrics:
        own = self._perf.get(player.label, regime=regime)
        if own.has_data:
            return own
        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        pnl = 0.0
        max_dd = 0.0
        sharpe_values: List[float] = []
        for label in player.agent_labels:
            metrics = self._perf.get(label, regime=regime)
            if not metrics.has_data:
                continue
            closed += metrics.closed_trades
            entries += metrics.entries
            signals += metrics.signals
            wins += metrics.wins
            losses += metrics.losses
            blocked += metrics.blocked_signals
            rejected += metrics.rejected_signals
            pending += metrics.pending_signals
            execution_failures += metrics.execution_failures
            pnl += metrics.pnl_pct
            max_dd = max(max_dd, metrics.max_dd_pct)
            sharpe_values.append(metrics.sharpe)
        if closed == entries == signals == 0:
            return Metrics.empty()
        sharpe = sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0
        return Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=sharpe,
            max_dd_pct=max_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
        )

    def _rank_candidates(
        self,
        candidates: Sequence[Player],
        regime: Regime,
    ) -> Tuple[CandidateScore, ...]:
        rows = [self._score_player_detail(player, regime) for player in candidates]
        rows.sort(
            key=lambda row: (
                -row.score,
                self._cold_start_tie_priority(row.label),
                -row.closed_trades,
                -row.signals,
                row.label,
            )
        )
        return tuple(replace(row, rank=i + 1) for i, row in enumerate(rows))

    def _score_player(self, player: Player, regime: Regime) -> float:
        return self._score_player_detail(player, regime).score

    def _score_player_detail(self, player: Player, regime: Regime) -> CandidateScore:
        """Score для игрока plus forensic details."""
        agent_labels = tuple(player.agent_labels)
        memory_keys = (self._memory_key(player.label, regime),) + tuple(
            self._memory_key(label, regime) for label in agent_labels
        )
        own_metrics = self._perf.get(player.label, regime=regime)
        if own_metrics.has_data:
            base_score = regime_score(own_metrics, regime, config=self._scoring)
            score_source = "player"
            score = base_score
            if own_metrics.closed_trades > 0:
                score = self._apply_affinity(player, regime, score)
            score = self._apply_real_overlay(player, regime, score)
            score, penalty = self._apply_uncertainty_penalty(score, own_metrics)
            score = self._apply_execution_penalty(score, own_metrics)
            return CandidateScore(
                label=player.label,
                score=score,
                rank=0,
                has_data=True,
                closed_trades=own_metrics.closed_trades,
                signals=own_metrics.signals,
                execution_failures=own_metrics.execution_failures,
                score_source=score_source,
                uncertainty_penalty=penalty,
                agent_labels=agent_labels,
                memory_keys_read=memory_keys,
            )

        agent_scores = []
        has_closed_experience = False
        agg_closed = agg_signals = agg_exec_failures = 0
        for label in agent_labels:
            metrics = self._perf.get(label, regime=regime)
            if metrics.has_data:
                has_closed_experience = has_closed_experience or metrics.closed_trades > 0
                agg_closed += metrics.closed_trades
                agg_signals += metrics.signals
                agg_exec_failures += metrics.execution_failures
                agent_scores.append(
                    regime_score(metrics, regime, config=self._scoring)
                    - metrics.execution_failures * self._config.execution_failure_score_penalty
                )
        if not agent_scores:
            penalty = self._config.zero_score_uncertainty_penalty
            return CandidateScore(
                label=player.label,
                score=-penalty,
                rank=0,
                has_data=False,
                closed_trades=0,
                signals=0,
                execution_failures=0,
                score_source="no_data",
                uncertainty_penalty=penalty,
                agent_labels=agent_labels,
                memory_keys_read=memory_keys,
            )
        base_score = sum(agent_scores) / len(agent_scores)
        if has_closed_experience:
            base_score = self._apply_affinity(player, regime, base_score)
        score = self._apply_real_overlay(player, regime, base_score)
        metrics_view = Metrics(
            closed_trades=agg_closed,
            signals=agg_signals,
            execution_failures=agg_exec_failures,
        )
        score, penalty = self._apply_uncertainty_penalty(score, metrics_view)
        return CandidateScore(
            label=player.label,
            score=score,
            rank=0,
            has_data=True,
            closed_trades=agg_closed,
            signals=agg_signals,
            execution_failures=agg_exec_failures,
            score_source="agents",
            uncertainty_penalty=penalty,
            agent_labels=agent_labels,
            memory_keys_read=memory_keys,
        )

    @staticmethod
    def _memory_key(label: str, regime: Regime) -> str:
        return f"{label}|{regime.label}"

    @staticmethod
    def _cold_start_tie_priority(label: str) -> int:
        return 0 if label == "DefaultEnsemble" else 1

    def _apply_uncertainty_penalty(self, score: float, metrics: Metrics) -> Tuple[float, float]:
        if metrics.closed_trades > 0:
            return score, 0.0
        penalty = self._config.zero_score_uncertainty_penalty
        if penalty <= 0:
            return score, 0.0
        return score - penalty, penalty

    def _apply_execution_penalty(self, score: float, metrics: Metrics) -> float:
        if metrics.execution_failures <= 0:
            return score
        return score - metrics.execution_failures * self._config.execution_failure_score_penalty

    def _apply_affinity(self, player: Player, regime: Regime, score: float) -> float:
        affinity = getattr(player, "affinity", None)
        if affinity is None:
            return score
        if affinity == regime:
            return score + self._config.affinity_bonus
        return score - self._config.affinity_mismatch_penalty

    def _apply_real_overlay(self, player: Player, regime: Regime, score: float) -> float:
        real_perf = getattr(self, "_real_perf", None)
        cfg = self._config
        if real_perf is None or cfg.real_score_weight <= 0:
            return score

        metrics = real_perf.get(player.label, regime=regime)
        if metrics.closed_trades < cfg.real_min_closed_trades:
            metrics = real_perf.get(player.label)
        if metrics.closed_trades < cfg.real_min_closed_trades:
            return score

        real_score = regime_score(metrics, regime, config=self._scoring)
        return score + cfg.real_score_weight * real_score
