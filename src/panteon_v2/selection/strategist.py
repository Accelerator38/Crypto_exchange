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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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


DEFAULT_STRATEGIST = StrategistConfig()


# ────────────────────────────────────────────────────────────────────
# Decision result
# ────────────────────────────────────────────────────────────────────


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
    ):
        self._perf = perf
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
                )
            raise ValueError(
                "No eligible candidates and current leader is invalid; "
                f"disqualified: {[(d.label, d.reason) for d in disqualified]}"
            )

        # 2. Скорим каждого
        scored: List[tuple] = []  # (score, player)
        for cand in eligible:
            s = self._score_player(cand, regime)
            scored.append((s, cand))
        scored.sort(key=lambda t: -t[0])
        best_score, best_player = scored[0]

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
        )

    def _score_player(self, player: Player, regime: Regime) -> float:
        """Score для игрока.

        Используем:
          • Метрики самого player (если он торговал ранее)
          • Если у player'а нет истории — score = средний score его агентов
            (это reasonable bootstrap для свежих ансамблей).
        """
        own_metrics = self._perf.get(player.label, regime=regime)
        if own_metrics.has_data:
            base_score = regime_score(own_metrics, regime, config=self._scoring)
            if own_metrics.closed_trades > 0:
                return self._apply_affinity(player, regime, base_score)
            return base_score
        # Fallback: средний score агентов
        agent_scores = []
        has_closed_experience = False
        for label in player.agent_labels:
            metrics = self._perf.get(label, regime=regime)
            if metrics.has_data:
                has_closed_experience = has_closed_experience or metrics.closed_trades > 0
                agent_scores.append(
                    regime_score(metrics, regime, config=self._scoring)
                )
        if not agent_scores:
            return 0.0
        base_score = sum(agent_scores) / len(agent_scores)
        if has_closed_experience:
            return self._apply_affinity(player, regime, base_score)
        return base_score

    def _apply_affinity(self, player: Player, regime: Regime, score: float) -> float:
        affinity = getattr(player, "affinity", None)
        if affinity is None:
            return score
        if affinity == regime:
            return score + self._config.affinity_bonus
        return score - self._config.affinity_mismatch_penalty
