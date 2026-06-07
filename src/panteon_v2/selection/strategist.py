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

from bisect import bisect_right
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..domain.types import Action, Metrics, Regime
from ..memory import PerformanceMemory, QuarantineManager
from ..scoring import DEFAULT_SCORING, ScoringConfig, regime_score
from .player import NoTradePlayer, Player
from .rolling_score import (
    RollingDecisionScoreConfig,
    RollingDecisionScoreInput,
    score_rolling_decision,
)
from .entry_causal_score import EntryCausalScoreState
from .soft_shadow_score import SoftShadowScoreState


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
    # Phase 1 / B1: при бутстрапе скора игрока из скоров его агентов используем
    # max-смещённую агрегацию вместо чистого среднего, чтобы ансамбль не
    # регрессировал к среднему слабых компонентов и не оказывался ниже своего
    # лучшего агента. base = bias*max + (1-bias)*mean. 1.0 = чистый max,
    # 0.0 = старое поведение (mean).
    ensemble_bootstrap_max_bias: float = 0.70
    player_session_overlay_weight: float = 1.00
    player_session_underperformance_weight: float = 0.75
    player_session_stale_penalty: float = 0.15
    player_session_pnl_cap_pct: float = 5.0
    player_session_min_activity: int = 1
    candidate_ttl_bars: int = 30
    hard_policy_enabled: bool = True
    hard_policy_deny_labels: Tuple[str, ...] = (
        "Solo_PlayerFunding",
        "Solo_LiveAfterShock",
        "Solo_CarryFlowAgentV2",
    )
    hard_policy_genetics_min_closed_trades: int = 20
    hard_policy_neutral_penalty_labels: Tuple[str, ...] = (
        "TrendResearch",
        "DefaultEnsemble",
        "DefensiveResearch",
        "MeanRevResearch",
    )
    hard_policy_neutral_penalty: float = 0.35
    hard_policy_low_actionability_share: float = 0.05
    hard_policy_min_actionability_bars: int = 12
    hard_policy_manual_regime_override_labels: Tuple[str, ...] = ("Antonius_strategy",)
    hard_policy_experimental_real_labels: Tuple[str, ...] = (
        "Antonius_strategy",
        "Antonius_conservative",
        "Optimal_StaticRotator",
    )
    hard_policy_experimental_min_bar: int = 5000
    hard_policy_experimental_min_closed_trades: int = 30
    hard_policy_experimental_min_pnl_pct: float = 0.0
    real_promotion_gate_enabled: bool = False
    no_trade_when_all_rejected: bool = True
    real_promotion_min_closed_trades: int = 20
    real_promotion_min_pnl_pct: float = 0.0
    real_promotion_max_drawdown_pct: float = 25.0
    real_promotion_loss_budget_pct: float = -1.0
    real_promotion_probation_min_score: float = 0.0
    use_v3_rolling_score: bool = False
    use_v3_entry_causal_score: bool = False
    use_v3_soft_shadow_score: bool = False
    v3_current_actionable_gate_enabled: bool = False
    v3_solo_current_actionable_gate_enabled: bool = False
    v3_candidate_allow_labels: Tuple[str, ...] = ()
    use_v3_shadow_rolling_score: bool = False
    v3_shadow_position_gate_enabled: bool = True
    v3_shadow_flat_handoff_enabled: bool = False
    v3_shadow_fresh_handoff_enabled: bool = False
    v3_shadow_fresh_handoff_max_age_bars: int = 1
    v3_shadow_fresh_handoff_require_positive_unrealized: bool = True
    v3_shadow_rolling_window_bars: int = 24
    v3_shadow_rolling_min_closed_trades: int = 50
    v3_entry_causal_min_filled: int = 3
    v3_entry_causal_actionability_weight: float = 1.0
    v3_min_score_to_trade: float = 0.0
    v3_virtual_only_score_cap: float = 0.35
    v3_real_loss_kill_min_closed_trades: int = 3
    v3_real_loss_kill_pnl_pct: float = -1.0
    v3_real_loss_rescue_enabled: bool = False
    v3_real_loss_rescue_min_virtual_pnl_pct: float = 10.0
    v3_real_loss_rescue_max_virtual_dd_pct: float = 50.0
    v3_real_loss_rescue_min_actionable_share: float = 0.05
    v3_real_loss_rescue_min_recent_filled: int = 1
    v3_real_loss_rescue_max_real_loss_pct: float = -3.0
    v3_real_loss_rescue_allow_genetics: bool = False
    v3_probation_shadow_rescue_enabled: bool = False
    v3_probation_shadow_rescue_min_virtual_pnl_pct: float = 10.0
    v3_probation_shadow_rescue_max_virtual_dd_pct: float = 50.0
    v3_probation_shadow_rescue_min_actionable_share: float = 0.05
    v3_probation_shadow_rescue_min_recent_filled: int = 1
    v3_probation_shadow_rescue_min_recent_pnl_usd: float = 0.0
    v3_probation_shadow_rescue_allow_genetics: bool = False
    v3_persistent_loss_kill_min_closed_trades: int = 0
    v3_persistent_loss_kill_pnl_pct: float = -2.0
    v3_persistent_loss_kill_win_rate_pct: float = 0.0
    v3_persistent_loss_requires_virtual_weakness: bool = True
    v3_persistent_loss_virtual_max_pnl_pct: float = 0.0
    v3_persistent_loss_virtual_min_dd_pct: float = 25.0
    v3_probation_loss_kill_min_closed_trades: int = 0
    v3_probation_loss_kill_pnl_pct: float = -0.15
    v3_probation_loss_kill_win_rate_pct: float = 50.0
    v3_probation_loss_kill_label_prefixes: Tuple[str, ...] = (
        "Solo_",
        "Fixed_",
        "Antonius_",
        "Optimal_",
    )
    v3_realized_profit_lock_min_closed_trades: int = 0
    v3_realized_profit_lock_min_peak_pnl_pct: float = 0.75
    v3_realized_profit_lock_max_giveback_pct: float = 0.55
    v3_realized_profit_lock_floor_pnl_pct: float = 0.25
    v3_panteon_equity_guard_enabled: bool = False
    v3_panteon_equity_guard_min_peak_pnl_pct: float = 2.0
    v3_panteon_equity_guard_max_giveback_pct: float = 1.0
    v3_panteon_equity_guard_floor_pnl_pct: float = 2.0
    v3_panteon_equity_guard_cooldown_bars: int = 720
    v3_score_config: RollingDecisionScoreConfig = field(
        default_factory=RollingDecisionScoreConfig
    )

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
        if not 0.0 <= self.ensemble_bootstrap_max_bias <= 1.0:
            raise ValueError("ensemble_bootstrap_max_bias must be in [0, 1]")
        if self.player_session_overlay_weight < 0:
            raise ValueError("player_session_overlay_weight must be >= 0")
        if self.player_session_underperformance_weight < 0:
            raise ValueError("player_session_underperformance_weight must be >= 0")
        if self.player_session_stale_penalty < 0:
            raise ValueError("player_session_stale_penalty must be >= 0")
        if self.player_session_pnl_cap_pct < 0:
            raise ValueError("player_session_pnl_cap_pct must be >= 0")
        if self.player_session_min_activity < 0:
            raise ValueError("player_session_min_activity must be >= 0")
        if self.candidate_ttl_bars < 0:
            raise ValueError("candidate_ttl_bars must be >= 0")
        if self.hard_policy_genetics_min_closed_trades < 0:
            raise ValueError("hard_policy_genetics_min_closed_trades must be >= 0")
        if self.hard_policy_neutral_penalty < 0:
            raise ValueError("hard_policy_neutral_penalty must be >= 0")
        if not 0.0 <= self.hard_policy_low_actionability_share <= 1.0:
            raise ValueError("hard_policy_low_actionability_share must be in [0, 1]")
        if self.hard_policy_min_actionability_bars < 0:
            raise ValueError("hard_policy_min_actionability_bars must be >= 0")
        if self.hard_policy_experimental_min_bar < 0:
            raise ValueError("hard_policy_experimental_min_bar must be >= 0")
        if self.hard_policy_experimental_min_closed_trades < 0:
            raise ValueError("hard_policy_experimental_min_closed_trades must be >= 0")
        if self.real_promotion_min_closed_trades < 0:
            raise ValueError("real_promotion_min_closed_trades must be >= 0")
        if self.real_promotion_max_drawdown_pct < 0:
            raise ValueError("real_promotion_max_drawdown_pct must be >= 0")
        if self.v3_virtual_only_score_cap < self.v3_min_score_to_trade:
            raise ValueError("v3_virtual_only_score_cap must be >= v3_min_score_to_trade")
        if self.v3_shadow_rolling_window_bars < 1:
            raise ValueError("v3_shadow_rolling_window_bars must be >= 1")
        if self.v3_shadow_rolling_min_closed_trades < 0:
            raise ValueError("v3_shadow_rolling_min_closed_trades must be >= 0")
        if self.v3_entry_causal_min_filled < 0:
            raise ValueError("v3_entry_causal_min_filled must be >= 0")
        if self.v3_entry_causal_actionability_weight < 0:
            raise ValueError("v3_entry_causal_actionability_weight must be >= 0")
        if self.v3_shadow_fresh_handoff_max_age_bars < 0:
            raise ValueError("v3_shadow_fresh_handoff_max_age_bars must be >= 0")
        if self.v3_real_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_real_loss_kill_min_closed_trades must be >= 0")
        if not 0.0 <= self.v3_real_loss_rescue_min_actionable_share <= 1.0:
            raise ValueError("v3_real_loss_rescue_min_actionable_share must be in [0, 1]")
        if self.v3_real_loss_rescue_min_recent_filled < 0:
            raise ValueError("v3_real_loss_rescue_min_recent_filled must be >= 0")
        if self.v3_real_loss_rescue_max_virtual_dd_pct < 0:
            raise ValueError("v3_real_loss_rescue_max_virtual_dd_pct must be >= 0")
        if not 0.0 <= self.v3_probation_shadow_rescue_min_actionable_share <= 1.0:
            raise ValueError("v3_probation_shadow_rescue_min_actionable_share must be in [0, 1]")
        if self.v3_probation_shadow_rescue_min_recent_filled < 0:
            raise ValueError("v3_probation_shadow_rescue_min_recent_filled must be >= 0")
        if self.v3_probation_shadow_rescue_max_virtual_dd_pct < 0:
            raise ValueError("v3_probation_shadow_rescue_max_virtual_dd_pct must be >= 0")
        if self.v3_persistent_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_persistent_loss_kill_min_closed_trades must be >= 0")
        if not 0.0 <= self.v3_persistent_loss_kill_win_rate_pct <= 100.0:
            raise ValueError("v3_persistent_loss_kill_win_rate_pct must be in [0, 100]")
        if self.v3_persistent_loss_virtual_min_dd_pct < 0:
            raise ValueError("v3_persistent_loss_virtual_min_dd_pct must be >= 0")
        if self.v3_probation_loss_kill_min_closed_trades < 0:
            raise ValueError("v3_probation_loss_kill_min_closed_trades must be >= 0")
        if not 0.0 <= self.v3_probation_loss_kill_win_rate_pct <= 100.0:
            raise ValueError("v3_probation_loss_kill_win_rate_pct must be in [0, 100]")
        if self.v3_realized_profit_lock_min_closed_trades < 0:
            raise ValueError("v3_realized_profit_lock_min_closed_trades must be >= 0")
        if self.v3_realized_profit_lock_min_peak_pnl_pct < 0:
            raise ValueError("v3_realized_profit_lock_min_peak_pnl_pct must be >= 0")
        if self.v3_realized_profit_lock_max_giveback_pct < 0:
            raise ValueError("v3_realized_profit_lock_max_giveback_pct must be >= 0")
        if self.v3_panteon_equity_guard_min_peak_pnl_pct < 0:
            raise ValueError("v3_panteon_equity_guard_min_peak_pnl_pct must be >= 0")
        if self.v3_panteon_equity_guard_max_giveback_pct < 0:
            raise ValueError("v3_panteon_equity_guard_max_giveback_pct must be >= 0")
        if self.v3_panteon_equity_guard_cooldown_bars < 0:
            raise ValueError("v3_panteon_equity_guard_cooldown_bars must be >= 0")
        if not isinstance(self.v3_score_config, RollingDecisionScoreConfig):
            raise ValueError("v3_score_config must be RollingDecisionScoreConfig")


DEFAULT_STRATEGIST = StrategistConfig()


def _max_biased_aggregate(scores: Sequence[float], bias: float) -> float:
    """max-смещённая агрегация скоров компонентов (Phase 1 / B1).

    Возвращает ``bias*max + (1-bias)*mean``. При bias=0 — чистое среднее
    (старое поведение), при bias=1 — чистый максимум. Это не даёт скору
    ансамбля упасть ниже его лучшего компонента из-за слабых участников.
    Корректно работает и для отрицательных скоров (max = наименее плохой).
    """
    values = [float(s) for s in scores]
    if not values:
        return 0.0
    b = max(0.0, min(1.0, float(bias)))
    mean = sum(values) / len(values)
    return b * max(values) + (1.0 - b) * mean


def _shadow_action_key(action: object) -> str:
    if isinstance(action, Action):
        return action.side or action.name.lower()
    text = str(action or "").strip().lower()
    if not text:
        return ""
    if "long" in text or "buy" in text:
        return "long"
    if "short" in text:
        return "short"
    return text


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
    session_score_delta: float = 0.0
    session_pnl_pct: float = 0.0
    session_underperformance_penalty: float = 0.0
    session_stale_penalty: float = 0.0
    recent_bars: int = 0
    recent_actionable_bars: int = 0
    actionable_share: float = 0.0
    recent_filled: int = 0
    recent_pnl_usd: float = 0.0


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
    switch_gate_reason: str = ""
    best_label: str = ""
    current_label: str = ""
    best_score: float = 0.0
    current_score: float = 0.0
    required_margin: float = 0.0
    cooldown_passed: bool = True
    cooldown_blocked: bool = False
    streak_count: int = 0
    streak_needed: int = 1

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


@dataclass
class _ShadowRollingActorState:
    total_trades: int = 0
    total_wins: int = 0
    recent_bars: List[int] = field(default_factory=list)
    recent_pnl_prefix: List[float] = field(default_factory=lambda: [0.0])

    def append(self, *, bar: int, pnl_usd: float, closed_trades: int, wins: int) -> None:
        self.total_trades += max(0, int(closed_trades))
        self.total_wins += max(0, int(wins))
        self.recent_bars.append(int(bar))
        self.recent_pnl_prefix.append(self.recent_pnl_prefix[-1] + float(pnl_usd))

    def rolling_pnl_usd(self, *, current_bar: int, window_bars: int) -> float:
        cutoff = int(current_bar) - max(1, int(window_bars))
        first_index = bisect_right(self.recent_bars, cutoff)
        return float(self.recent_pnl_prefix[-1] - self.recent_pnl_prefix[first_index])


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
        self._session_baseline: Dict[Tuple[str, Regime], Metrics] = {}
        self._real_session_baseline: Dict[Tuple[str, Regime], Metrics] = {}
        self._candidate_cache: Dict[str, Tuple[Player, int]] = {}
        self._no_trade: Player = NoTradePlayer()
        self._realized_pnl_by_player: Dict[str, float] = {}
        self._realized_trade_counts_by_player: Dict[str, int] = {}
        self._realized_win_counts_by_player: Dict[str, int] = {}
        self._realized_peak_pnl_by_player: Dict[str, float] = {}
        self._panteon_realized_peak_pnl_usd: float = 0.0
        self._panteon_equity_guard_until_bar: int = -1
        self._realized_initial_capital: float = 0.0
        self._shadow_rolling_by_player_regime: Dict[
            Tuple[str, str],
            _ShadowRollingActorState,
        ] = {}
        self._soft_shadow_score = SoftShadowScoreState(
            window_bars=self._config.v3_shadow_rolling_window_bars,
            min_closed_trades=self._config.v3_shadow_rolling_min_closed_trades,
        )
        self._entry_causal_score = EntryCausalScoreState(
            window_bars=self._config.v3_shadow_rolling_window_bars,
            min_closed_trades=self._config.v3_shadow_rolling_min_closed_trades,
            min_filled=self._config.v3_entry_causal_min_filled,
            actionability_weight=self._config.v3_entry_causal_actionability_weight,
        )
        self._shadow_positions_by_player: Dict[str, Tuple[Tuple[str, str], ...]] = {}
        self._shadow_position_details_by_player: Dict[
            str,
            Tuple[Tuple[str, str, int], ...],
        ] = {}
        self._real_position_signature: Tuple[Tuple[str, str], ...] = ()
        self._shadow_position_snapshot_seen: bool = False
        self._current_actionable_labels: Optional[set[str]] = None

    # ── Public API ──────────────────────────────────────────────────

    def update_candidates(self, candidates: Sequence[Player]) -> None:
        """Обновить список кандидатов. Если current не в новом списке —
        он всё равно остаётся current до следующего switch."""
        self._candidates = list(candidates)

    def update_current_actionable_labels(self, labels: Optional[Iterable[str]]) -> None:
        if labels is None:
            self._current_actionable_labels = None
            return
        self._current_actionable_labels = {
            str(label).strip()
            for label in labels
            if str(label).strip()
        }

    def shadow_soft_score_for_label(
        self,
        label: str,
        regime: Regime,
        current_bar: int,
    ) -> tuple[float, int]:
        return self._soft_shadow_score.score_with_trade_count(
            label=str(label),
            regime=regime.label,
            current_bar=int(current_bar),
        )

    def shadow_soft_stats_for_label(
        self,
        label: str,
        regime: Regime,
        current_bar: int,
    ) -> dict:
        return self._soft_shadow_score.stats(
            label=str(label),
            regime=regime.label,
            current_bar=int(current_bar),
        ).as_confirmation_payload()

    def shadow_soft_score_for_label_all_regimes(
        self,
        label: str,
        current_bar: int,
    ) -> tuple[float, int]:
        return self._soft_shadow_score.score_with_trade_count(
            label=str(label),
            regime="all",
            current_bar=int(current_bar),
        )

    def shadow_soft_stats_for_label_all_regimes(
        self,
        label: str,
        current_bar: int,
    ) -> dict:
        return self._soft_shadow_score.stats(
            label=str(label),
            regime="all",
            current_bar=int(current_bar),
        ).as_confirmation_payload()

    def shadow_soft_score_for_signal(
        self,
        label: str,
        regime: Regime,
        symbol: str,
        current_bar: int,
        action: object = "",
    ) -> tuple[float, int]:
        return self._soft_shadow_score.score_with_trade_count(
            label=str(label),
            regime=regime.label,
            symbol=str(symbol or "").upper(),
            action=_shadow_action_key(action),
            current_bar=int(current_bar),
        )

    def shadow_soft_stats_for_signal(
        self,
        label: str,
        regime: Regime,
        symbol: str,
        current_bar: int,
        action: object = "",
    ) -> dict:
        return self._soft_shadow_score.stats(
            label=str(label),
            regime=regime.label,
            symbol=str(symbol or "").upper(),
            action=_shadow_action_key(action),
            current_bar=int(current_bar),
        ).as_confirmation_payload()

    def shadow_soft_score_for_signal_all_regimes(
        self,
        label: str,
        symbol: str,
        current_bar: int,
        action: object = "",
    ) -> tuple[float, int]:
        return self._soft_shadow_score.score_with_trade_count(
            label=str(label),
            regime="all",
            symbol=str(symbol or "").upper(),
            action=_shadow_action_key(action),
            current_bar=int(current_bar),
        )

    def shadow_soft_stats_for_signal_all_regimes(
        self,
        label: str,
        symbol: str,
        current_bar: int,
        action: object = "",
    ) -> dict:
        return self._soft_shadow_score.stats(
            label=str(label),
            regime="all",
            symbol=str(symbol or "").upper(),
            action=_shadow_action_key(action),
            current_bar=int(current_bar),
        ).as_confirmation_payload()

    def capture_session_baseline(self, regime: Optional[Regime] = None) -> None:
        regimes = [regime] if regime is not None else list(Regime)
        labels = set()
        for player in self._candidates:
            labels.add(player.label)
            labels.update(player.agent_labels)
        for reg in regimes:
            for label in labels:
                self._session_baseline[(label, reg)] = self._perf.get(label, regime=reg)
                if self._real_perf is not None:
                    self._real_session_baseline[(label, reg)] = self._real_perf.get(
                        label,
                        regime=reg,
                    )

    def update_realized_pnl_snapshot(
        self,
        *,
        pnl_by_player: Dict[str, float],
        trade_counts_by_player: Dict[str, int],
        win_counts_by_player: Dict[str, int],
        initial_capital: float,
    ) -> None:
        self._realized_pnl_by_player = {
            str(label): float(value or 0.0)
            for label, value in dict(pnl_by_player or {}).items()
        }
        self._realized_trade_counts_by_player = {
            str(label): max(0, int(value or 0))
            for label, value in dict(trade_counts_by_player or {}).items()
        }
        self._realized_win_counts_by_player = {
            str(label): max(0, int(value or 0))
            for label, value in dict(win_counts_by_player or {}).items()
        }
        for label, pnl in self._realized_pnl_by_player.items():
            self._realized_peak_pnl_by_player[label] = max(
                float(self._realized_peak_pnl_by_player.get(label, pnl) or 0.0),
                float(pnl),
            )
        total_pnl = sum(float(value or 0.0) for value in self._realized_pnl_by_player.values())
        self._panteon_realized_peak_pnl_usd = max(
            float(self._panteon_realized_peak_pnl_usd or 0.0),
            float(total_pnl),
        )
        self._realized_initial_capital = max(0.0, float(initial_capital or 0.0))

    def update_shadow_actor_updates(self, updates: Sequence[object]) -> None:
        for event in updates or ():
            actor_type = str(self._object_value(event, "actor_type") or "")
            if actor_type not in {"agent", "player"}:
                continue
            actor_label = str(self._object_value(event, "actor_label") or "").strip()
            if not actor_label:
                continue
            regime = str(self._object_value(event, "regime") or "").strip().lower() or "neutral"
            labels = (
                (actor_label,)
                if actor_type == "player"
                else (f"Solo_{actor_label}",)
            )
            bar = self._safe_int(self._object_value(event, "bar"))
            pnl_usd = self._safe_float(self._object_value(event, "realized_pnl_usd"))
            closed_trades = self._safe_int(self._object_value(event, "closed_trades"))
            wins = self._safe_int(self._object_value(event, "winning_trades"))
            signals = self._safe_int(self._object_value(event, "signals"))
            filled = self._safe_int(self._object_value(event, "filled"))
            for label in labels:
                self._update_shadow_actor_score_state(
                    label=label,
                    regime=regime,
                    bar=bar,
                    pnl_usd=pnl_usd,
                    closed_trades=closed_trades,
                    wins=wins,
                    signals=signals,
                    filled=filled,
                )
                for symbol, symbol_pnl, symbol_closed, _symbol_wins in (
                    self._object_value(event, "symbol_outcomes") or ()
                ):
                    self._soft_shadow_score.update(
                        bar=bar,
                        label=label,
                        regime=regime,
                        symbol=str(symbol or "").upper(),
                        pnl_usd=self._safe_float(symbol_pnl),
                        closed_trades=self._safe_int(symbol_closed),
                        winning_trades=self._safe_int(_symbol_wins),
                    )
                    self._soft_shadow_score.update(
                        bar=bar,
                        label=label,
                        regime="all",
                        symbol=str(symbol or "").upper(),
                        pnl_usd=self._safe_float(symbol_pnl),
                        closed_trades=self._safe_int(symbol_closed),
                        winning_trades=self._safe_int(_symbol_wins),
                    )
                for symbol, action, action_pnl, action_closed, _action_wins in (
                    self._object_value(event, "symbol_action_outcomes") or ()
                ):
                    self._soft_shadow_score.update(
                        bar=bar,
                        label=label,
                        regime=regime,
                        symbol=str(symbol or "").upper(),
                        action=_shadow_action_key(action),
                        pnl_usd=self._safe_float(action_pnl),
                        closed_trades=self._safe_int(action_closed),
                        winning_trades=self._safe_int(_action_wins),
                    )
                    self._soft_shadow_score.update(
                        bar=bar,
                        label=label,
                        regime="all",
                        symbol=str(symbol or "").upper(),
                        action=_shadow_action_key(action),
                        pnl_usd=self._safe_float(action_pnl),
                        closed_trades=self._safe_int(action_closed),
                        winning_trades=self._safe_int(_action_wins),
                    )

    def _update_shadow_actor_score_state(
        self,
        *,
        label: str,
        regime: str,
        bar: int,
        pnl_usd: float,
        closed_trades: int,
        wins: int,
        signals: int,
        filled: int,
    ) -> None:
        state = self._shadow_rolling_by_player_regime.setdefault(
            (label, regime),
            _ShadowRollingActorState(),
        )
        state.append(
            bar=bar,
            pnl_usd=pnl_usd,
            closed_trades=closed_trades,
            wins=wins,
        )
        self._soft_shadow_score.update(
            bar=bar,
            label=label,
            regime=regime,
            pnl_usd=pnl_usd,
            closed_trades=closed_trades,
            winning_trades=wins,
        )
        self._soft_shadow_score.update(
            bar=bar,
            label=label,
            regime="all",
            pnl_usd=pnl_usd,
            closed_trades=closed_trades,
            winning_trades=wins,
        )
        self._entry_causal_score.update(
            bar=bar,
            label=label,
            regime=regime,
            pnl_usd=pnl_usd,
            closed_trades=closed_trades,
            signals=signals,
            filled=filled,
        )

    def update_shadow_position_snapshot(
        self,
        *,
        player_positions: object,
        real_positions: object,
    ) -> None:
        if not isinstance(player_positions, dict):
            player_positions = {}
        details_by_player = {
            str(label): self._position_details(positions)
            for label, positions in dict(player_positions).items()
        }
        self._shadow_position_details_by_player = details_by_player
        self._shadow_positions_by_player = {
            label: tuple((sym, side) for sym, side, _ in details)
            for label, details in details_by_player.items()
        }
        self._real_position_signature = self._position_signature(real_positions)
        self._shadow_position_snapshot_seen = True

    def current_leader(self) -> Optional[Player]:
        return self._current

    def consider_switch(
        self,
        regime: Regime,
        current_bar: int,
        regime_confidence: float = 1.0,
        market_tags: Sequence[str] = (),
    ) -> SwitchDecision:
        """Главная функция. Решает: сменить лидера или оставить.

        Возвращает SwitchDecision со ссылкой на нового лидера
        (или того же самого, если switched=False).
        """
        # 1. Фильтруем кандидатов: убираем тех, кто использует карантинных
        market_tags_tuple = self._normalize_market_tags(market_tags)
        candidates = self._candidate_pool_for_bar(current_bar)
        eligible: List[Player] = []
        disqualified: List[_DisqualificationReason] = []
        for cand in candidates:
            issue = self._validate(cand)
            if issue is None:
                issue = self._validate_promotion(
                    cand,
                    regime,
                    current_bar=current_bar,
                    regime_confidence=regime_confidence,
                    market_tags=market_tags_tuple,
                )
            if issue is None:
                eligible.append(cand)
            else:
                disqualified.append(issue)
        candidate_rejections = tuple(
            CandidateRejection(label=d.label, reason=d.reason)
            for d in disqualified
        )
        equity_guard_reason = self._panteon_equity_guard_reason(
            current_bar=current_bar
        )
        if equity_guard_reason:
            return self._select_no_trade(
                current_bar=current_bar,
                candidate_rejections=candidate_rejections,
                reason=equity_guard_reason,
            )

        if not eligible:
            hard_policy_rejected = any(
                str(d.reason).startswith("hard policy ") for d in disqualified
            )
            if (
                (
                    self._config.real_promotion_gate_enabled
                    or self._config.use_v3_rolling_score
                    or self._config.v3_current_actionable_gate_enabled
                    or hard_policy_rejected
                )
                and self._config.no_trade_when_all_rejected
            ):
                return self._select_no_trade(
                    current_bar=current_bar,
                    candidate_rejections=candidate_rejections,
                    reason="no real-promoted candidates; selecting NoTrade",
                )
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
                    switch_gate_reason="no eligible candidates; keeping current",
                    current_label=self._current.label,
                    current_score=0.0,
                    required_margin=self._config.switch_margin,
                )
            raise ValueError(
                "No eligible candidates and current leader is invalid; "
                f"disqualified: {[(d.label, d.reason) for d in disqualified]}"
            )

        # 2. Скорим каждого
        self._ensure_session_baseline(eligible, regime)
        scored_rows = self._rank_candidates(
            eligible,
            regime,
            current_bar=current_bar,
            market_tags=market_tags_tuple,
        )
        by_label = {cand.label: cand for cand in eligible}
        best_row = scored_rows[0]
        best_score = best_row.score
        best_player = by_label[best_row.label]
        if (
            self._config.use_v3_rolling_score
            and best_score < self._config.v3_min_score_to_trade
        ):
            return self._select_no_trade(
                current_bar=current_bar,
                candidate_rejections=candidate_rejections,
                candidate_scores=scored_rows,
                reason=(
                    f"v3 rolling score {best_score:.4f} below trade threshold "
                    f"{self._config.v3_min_score_to_trade:.4f}"
                ),
            )

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
                switch_gate_reason="bootstrap (no current)",
                best_label=best_player.label,
                current_label="",
                best_score=best_score,
                current_score=0.0,
                required_margin=self._config.switch_margin,
                cooldown_passed=True,
                cooldown_blocked=False,
                streak_count=self._streak_count,
                streak_needed=self._config.streak_needed,
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
                switch_gate_reason="current not in candidates",
                best_label=best_player.label,
                current_label=previous.label,
                best_score=best_score,
                current_score=0.0,
                required_margin=self._config.switch_margin,
                cooldown_passed=True,
                cooldown_blocked=False,
            )

        current_score = self._score_player(
            self._current,
            regime,
            current_bar=current_bar,
            market_tags=market_tags_tuple,
        )
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
                switch_gate_reason="current is still best",
                best_label=best_player.label,
                current_label=self._current.label,
                best_score=best_score,
                current_score=current_score,
                required_margin=self._config.switch_margin,
                cooldown_passed=cooldown_passed,
                cooldown_blocked=False,
                streak_count=self._streak_count,
                streak_needed=self._config.streak_needed,
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
                switch_gate_reason="margin too small and not urgent",
                best_label=best_player.label,
                current_label=self._current.label,
                best_score=best_score,
                current_score=current_score,
                required_margin=self._config.switch_margin,
                cooldown_passed=cooldown_passed,
                cooldown_blocked=False,
                streak_count=self._streak_count,
                streak_needed=self._config.streak_needed,
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
                    switch_gate_reason="urgent" if urgent else "streak confirmed",
                    best_label=best_player.label,
                    current_label=previous.label,
                    best_score=best_score,
                    current_score=current_score,
                    required_margin=self._config.switch_margin,
                    cooldown_passed=cooldown_passed,
                    cooldown_blocked=False,
                    streak_count=self._streak_count,
                    streak_needed=self._config.streak_needed,
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
                switch_gate_reason=(
                    f"awaiting streak confirmation ({self._streak_count}/{self._config.streak_needed})"
                ),
                best_label=best_player.label,
                current_label=self._current.label,
                best_score=best_score,
                current_score=current_score,
                required_margin=self._config.switch_margin,
                cooldown_passed=cooldown_passed,
                cooldown_blocked=False,
                streak_count=self._streak_count,
                streak_needed=self._config.streak_needed,
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
            switch_gate_reason="cooldown not yet passed",
            best_label=best_player.label,
            current_label=self._current.label,
            best_score=best_score,
            current_score=current_score,
            required_margin=self._config.switch_margin,
            cooldown_passed=cooldown_passed,
            cooldown_blocked=True,
            streak_count=self._streak_count,
            streak_needed=self._config.streak_needed,
        )

    # ── Internal ────────────────────────────────────────────────────

    def _select_no_trade(
        self,
        *,
        current_bar: int,
        candidate_rejections: Tuple[CandidateRejection, ...],
        candidate_scores: Tuple[CandidateScore, ...] = (),
        reason: str,
    ) -> SwitchDecision:
        previous = self._current
        switched = previous is None or previous.label != self._no_trade.label
        self._current = self._no_trade
        self._last_switch_bar = int(current_bar)
        self._streak_label = ""
        self._streak_count = 0
        return SwitchDecision(
            new_leader=self._no_trade,
            previous=previous,
            score=0.0,
            margin=0.0,
            is_urgent=True,
            reason=reason,
            switched=switched,
            candidate_scores=candidate_scores,
            candidate_rejections=candidate_rejections,
            switch_gate_reason=reason,
            best_label=self._no_trade.label,
            current_label=previous.label if previous is not None else "",
            best_score=0.0,
            current_score=0.0,
            required_margin=self._config.switch_margin,
            cooldown_passed=True,
            cooldown_blocked=False,
            streak_count=0,
            streak_needed=self._config.streak_needed,
        )

    def _candidate_pool_for_bar(self, current_bar: int) -> List[Player]:
        bar = int(current_bar)
        ttl = int(self._config.candidate_ttl_bars)
        out: List[Player] = []
        seen: set[str] = set()

        for cand in self._candidates:
            label = str(getattr(cand, "label", "") or "")
            if not label:
                continue
            self._candidate_cache[label] = (cand, bar)
            if label not in seen:
                out.append(cand)
                seen.add(label)

        if ttl > 0:
            min_seen_bar = bar - ttl
            for label, (cand, last_seen_bar) in sorted(self._candidate_cache.items()):
                if label in seen:
                    continue
                if int(last_seen_bar) < min_seen_bar:
                    continue
                out.append(cand)
                seen.add(label)

        min_seen_bar = bar - ttl if ttl > 0 else bar
        self._candidate_cache = {
            label: (cand, last_seen_bar)
            for label, (cand, last_seen_bar) in self._candidate_cache.items()
            if label in seen or (ttl > 0 and int(last_seen_bar) >= min_seen_bar)
        }
        return out

    def _validate(self, player: Player) -> Optional[_DisqualificationReason]:
        """Q4 sanity check: лидер не должен использовать карантинных."""
        for label in player.agent_labels:
            if self._qm.is_quarantined(label):
                if self._is_genetics_candidate(player) and self._is_genetics_label(label):
                    continue
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
        current_bar: int,
        regime_confidence: float,
        market_tags: Sequence[str] = (),
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        if player.label == self._no_trade.label:
            return None
        issue = self._validate_hard_policy(
            player,
            regime,
            current_bar=current_bar,
            market_tags=market_tags,
        )
        if issue is not None:
            return issue
        issue = self._validate_current_actionability(player)
        if issue is not None:
            return issue
        issue = self._validate_candidate_allowlist(player)
        if issue is not None:
            return issue
        if cfg.use_v3_rolling_score:
            issue = self._validate_v3_real_loss(
                player,
                regime,
                current_bar=current_bar,
            )
            if issue is not None:
                return issue
            issue = self._validate_v3_shadow_position_gate(
                player,
                current_bar=current_bar,
            )
            if issue is not None:
                return issue
        if regime_confidence < cfg.min_regime_confidence:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"regime confidence {regime_confidence:.2f} below "
                    f"{cfg.min_regime_confidence:.2f}"
                ),
            )
        score: Optional[float] = None
        virtual_gate_enabled = not (
            cfg.min_live_closed_trades <= 0
            and cfg.min_live_score <= -999_999.0
            and cfg.max_live_drawdown_pct >= 100.0
        )
        if virtual_gate_enabled:
            metrics = self._promotion_metrics(player, regime)
            score = self._score_player(
                player,
                regime,
                current_bar=current_bar,
                market_tags=market_tags,
            )
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
        if cfg.real_promotion_gate_enabled:
            if score is None:
                score = self._score_player(
                    player,
                    regime,
                    current_bar=current_bar,
                    market_tags=market_tags,
                )
            return self._validate_real_promotion(
                player,
                regime,
                score,
                current_bar=current_bar,
            )
        return None

    def _validate_candidate_allowlist(
        self,
        player: Player,
    ) -> Optional[_DisqualificationReason]:
        allowed = {
            str(label).strip()
            for label in self._config.v3_candidate_allow_labels
            if str(label).strip()
        }
        if not allowed or player.label == self._no_trade.label:
            return None
        if player.label in allowed:
            return None
        return _DisqualificationReason(
            label=player.label,
            reason=f"candidate allowlist: {player.label} is not enabled for real execution",
        )

    def _validate_current_actionability(
        self,
        player: Player,
    ) -> Optional[_DisqualificationReason]:
        if player.label == self._no_trade.label:
            return None
        labels = self._current_actionable_labels
        if self._config.v3_current_actionable_gate_enabled:
            if labels is not None and player.label in labels:
                return None
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    "current actionability gate: no current raw/shadow signal "
                    "or fresh shadow fill"
                ),
            )
        if (
            self._config.v3_solo_current_actionable_gate_enabled
            and player.label.startswith("Solo_")
        ):
            if labels is None or player.label in labels:
                return None
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    "solo current actionability gate: no current raw/shadow signal "
                    "or fresh shadow fill"
                ),
            )
        return None

    def _validate_hard_policy(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int = 0,
        market_tags: Sequence[str] = (),
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        if not cfg.hard_policy_enabled or player.label == self._no_trade.label:
            return None
        deny_labels = {str(label) for label in cfg.hard_policy_deny_labels}
        if player.label in deny_labels:
            return _DisqualificationReason(
                label=player.label,
                reason=f"hard policy denylist: {player.label}",
            )
        if self._is_genetics_candidate(player):
            metrics = self._promotion_metrics(player, regime)
            aggregate = self._virtual_persistent_metrics(player)
            if aggregate.closed_trades > metrics.closed_trades:
                metrics = aggregate
            if metrics.closed_trades < cfg.hard_policy_genetics_min_closed_trades:
                return _DisqualificationReason(
                    label=player.label,
                    reason=(
                        "hard policy untrained genetics: closed_trades "
                        f"{metrics.closed_trades} < "
                        f"{cfg.hard_policy_genetics_min_closed_trades}"
                    ),
                )
        issue = self._validate_experimental_real_maturity(
            player,
            regime,
            current_bar=current_bar,
        )
        if issue is not None:
            return issue
        if player.label in {str(label) for label in cfg.hard_policy_manual_regime_override_labels}:
            return None
        specialist = self._solo_specialist_label(player)
        if not specialist:
            return None
        allowed_regimes, allowed_tags = self._specialist_allow_policy(specialist)
        if not allowed_regimes and not allowed_tags:
            return None
        regime_key = regime.label
        tag_set = set(self._normalize_market_tags(market_tags))
        if regime_key in allowed_regimes or tag_set.intersection(allowed_tags):
            return None
        return _DisqualificationReason(
            label=player.label,
            reason=(
                "hard policy regime allowlist: "
                f"{specialist} allowed in regimes "
                f"{'/'.join(sorted(allowed_regimes)) or '-'}"
                f" or tags {'/'.join(sorted(allowed_tags)) or '-'}"
            ),
        )

    def _validate_experimental_real_maturity(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        experimental_labels = {
            str(label)
            for label in cfg.hard_policy_experimental_real_labels
            if str(label)
        }
        if player.label not in experimental_labels:
            return None
        min_bar = int(cfg.hard_policy_experimental_min_bar)
        if min_bar > 0 and int(current_bar) < min_bar:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    "hard policy experimental maturity: bar "
                    f"{int(current_bar)} < {min_bar}"
                ),
            )
        metrics = self._promotion_metrics(player, regime)
        min_trades = int(cfg.hard_policy_experimental_min_closed_trades)
        if metrics.closed_trades < min_trades:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    "hard policy experimental maturity: shadow closed_trades "
                    f"{metrics.closed_trades} < {min_trades}"
                ),
            )
        if metrics.pnl_pct < cfg.hard_policy_experimental_min_pnl_pct:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    "hard policy experimental maturity: shadow pnl_pct "
                    f"{metrics.pnl_pct:.2f} < "
                    f"{cfg.hard_policy_experimental_min_pnl_pct:.2f}"
                ),
            )
        return None

    def _validate_v3_shadow_position_gate(
        self,
        player: Player,
        *,
        current_bar: int,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        if not (
            cfg.use_v3_shadow_rolling_score
            and cfg.v3_shadow_position_gate_enabled
            and self._shadow_position_snapshot_seen
        ):
            return None
        if self._current is not None and player.label == self._current.label:
            return None
        shadow_signature = self._shadow_positions_by_player.get(player.label, ())
        real_signature = self._real_position_signature
        if shadow_signature == real_signature:
            return None
        if cfg.v3_shadow_flat_handoff_enabled and not real_signature:
            return None
        if (
            cfg.v3_shadow_fresh_handoff_enabled
            and not real_signature
            and self._is_fresh_shadow_handoff(player.label, current_bar=current_bar)
        ):
            return None
        return _DisqualificationReason(
            label=player.label,
            reason=(
                "v3 shadow position gate: shadow positions "
                f"{self._format_position_signature(shadow_signature)} != real positions "
                f"{self._format_position_signature(real_signature)}"
            ),
        )

    def _is_fresh_shadow_handoff(self, label: str, *, current_bar: int) -> bool:
        details = self._shadow_position_details_by_player.get(label, ())
        if not details:
            return False
        max_age = max(0, int(self._config.v3_shadow_fresh_handoff_max_age_bars))
        for _, _, opened_bar in details:
            if opened_bar <= 0:
                return False
            age = int(current_bar) - int(opened_bar)
            if age < 0 or age > max_age:
                return False
        return True

    def _validate_v3_real_loss(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        if cfg.v3_real_loss_kill_min_closed_trades > 0:
            metrics = self._real_promotion_metrics(player, regime)
            if (
                metrics.closed_trades >= cfg.v3_real_loss_kill_min_closed_trades
                and metrics.pnl_pct <= cfg.v3_real_loss_kill_pnl_pct
            ):
                if self._has_v3_real_loss_rescue(
                    player,
                    regime,
                    current_bar=current_bar,
                    real_metrics=metrics,
                ):
                    return None
                return _DisqualificationReason(
                    label=player.label,
                    reason=(
                        f"v3 real-loss kill: pnl_pct {metrics.pnl_pct:.2f} <= "
                        f"{cfg.v3_real_loss_kill_pnl_pct:.2f} with "
                        f"{metrics.closed_trades} real closed trades"
                    ),
                )
        issue = self._validate_v3_realized_profit_lock(player)
        if issue is not None:
            return issue
        issue = self._validate_v3_probation_realized_loss(
            player,
            regime,
            current_bar=current_bar,
        )
        if issue is not None:
            return issue
        return self._validate_v3_persistent_loss(player)

    def _has_v3_real_loss_rescue(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int,
        real_metrics: Metrics,
    ) -> bool:
        cfg = self._config
        if not cfg.v3_real_loss_rescue_enabled:
            return False
        if (
            self._is_genetics_candidate(player)
            and not cfg.v3_real_loss_rescue_allow_genetics
        ):
            return False
        if real_metrics.pnl_pct < cfg.v3_real_loss_rescue_max_real_loss_pct:
            return False
        virtual = self._virtual_persistent_metrics(player)
        if not virtual.has_data:
            return False
        if virtual.pnl_pct < cfg.v3_real_loss_rescue_min_virtual_pnl_pct:
            return False
        if virtual.max_dd_pct > cfg.v3_real_loss_rescue_max_virtual_dd_pct:
            return False
        stats = self._entry_causal_score.score_with_stats(
            label=player.label,
            regime=regime.label,
            current_bar=current_bar,
        )
        if stats.actionable_share < cfg.v3_real_loss_rescue_min_actionable_share:
            return False
        if stats.recent_filled < cfg.v3_real_loss_rescue_min_recent_filled:
            return False
        return True

    def _validate_v3_persistent_loss(
        self,
        player: Player,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        if cfg.v3_persistent_loss_kill_min_closed_trades <= 0:
            return None
        realized_issue = self._validate_v3_persistent_realized_loss(player)
        if realized_issue is not None:
            return realized_issue
        metrics = self._real_persistent_metrics(player)
        if metrics.closed_trades < cfg.v3_persistent_loss_kill_min_closed_trades:
            return None
        win_rate = (
            metrics.wins / metrics.closed_trades * 100.0
            if metrics.closed_trades > 0 else 0.0
        )
        pnl_breach = metrics.pnl_pct <= cfg.v3_persistent_loss_kill_pnl_pct
        weak_negative_edge = (
            cfg.v3_persistent_loss_kill_win_rate_pct > 0.0
            and
            metrics.pnl_pct < 0.0
            and win_rate <= cfg.v3_persistent_loss_kill_win_rate_pct
        )
        if not (pnl_breach or weak_negative_edge):
            return None
        if not self._has_v3_persistent_virtual_weakness(player):
            return None
        return _DisqualificationReason(
            label=player.label,
            reason=(
                f"v3 persistent real-loss kill: pnl_pct {metrics.pnl_pct:.2f}, "
                f"win_rate {win_rate:.2f}% with {metrics.closed_trades} "
                "all-regime real closed trades"
            ),
        )

    def _validate_v3_probation_realized_loss(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        min_trades = int(cfg.v3_probation_loss_kill_min_closed_trades)
        if min_trades <= 0:
            return None
        label = player.label
        prefixes = tuple(
            str(prefix)
            for prefix in cfg.v3_probation_loss_kill_label_prefixes
            if str(prefix)
        )
        if prefixes and not label.startswith(prefixes):
            return None
        closed_trades = int(self._realized_trade_counts_by_player.get(label, 0) or 0)
        if closed_trades < min_trades:
            return None
        initial_capital = float(self._realized_initial_capital or 0.0)
        if initial_capital <= 0:
            return None
        pnl_usd = float(self._realized_pnl_by_player.get(label, 0.0) or 0.0)
        pnl_pct = pnl_usd / initial_capital * 100.0
        wins = int(self._realized_win_counts_by_player.get(label, 0) or 0)
        win_rate = wins / closed_trades * 100.0 if closed_trades > 0 else 0.0
        pnl_breach = pnl_pct <= cfg.v3_probation_loss_kill_pnl_pct
        weak_negative_edge = (
            cfg.v3_probation_loss_kill_win_rate_pct > 0.0
            and pnl_pct < 0.0
            and win_rate <= cfg.v3_probation_loss_kill_win_rate_pct
        )
        if not (pnl_breach or weak_negative_edge):
            return None
        if self._has_v3_probation_shadow_rescue(
            player,
            regime,
            current_bar=current_bar,
        ):
            return None
        return _DisqualificationReason(
            label=label,
            reason=(
                f"v3 probation realized-loss kill: pnl_usd {pnl_usd:.2f}, "
                f"pnl_pct {pnl_pct:.2f}, win_rate {win_rate:.2f}% with "
                f"{closed_trades} realized closed trades"
            ),
        )

    def _validate_v3_realized_profit_lock(
        self,
        player: Player,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        min_trades = int(cfg.v3_realized_profit_lock_min_closed_trades)
        if min_trades <= 0:
            return None
        label = player.label
        closed_trades = int(self._realized_trade_counts_by_player.get(label, 0) or 0)
        if closed_trades < min_trades:
            return None
        initial_capital = float(self._realized_initial_capital or 0.0)
        if initial_capital <= 0:
            return None
        pnl_usd = float(self._realized_pnl_by_player.get(label, 0.0) or 0.0)
        peak_usd = float(self._realized_peak_pnl_by_player.get(label, pnl_usd) or 0.0)
        pnl_pct = pnl_usd / initial_capital * 100.0
        peak_pct = peak_usd / initial_capital * 100.0
        giveback_pct = max(0.0, peak_pct - pnl_pct)
        if peak_pct < cfg.v3_realized_profit_lock_min_peak_pnl_pct:
            return None
        if pnl_pct > cfg.v3_realized_profit_lock_floor_pnl_pct:
            return None
        if giveback_pct < cfg.v3_realized_profit_lock_max_giveback_pct:
            return None
        return _DisqualificationReason(
            label=label,
            reason=(
                f"v3 realized profit-lock: pnl_pct {pnl_pct:.2f}, "
                f"peak_pnl_pct {peak_pct:.2f}, giveback_pct {giveback_pct:.2f} "
                f"with {closed_trades} realized closed trades"
            ),
        )

    def _panteon_equity_guard_reason(self, *, current_bar: int) -> Optional[str]:
        cfg = self._config
        if not cfg.v3_panteon_equity_guard_enabled:
            return None
        if int(current_bar) <= int(self._panteon_equity_guard_until_bar):
            return (
                "v3 panteon equity guard: cooldown active until bar "
                f"{self._panteon_equity_guard_until_bar}"
            )
        initial_capital = float(self._realized_initial_capital or 0.0)
        if initial_capital <= 0:
            return None
        pnl_usd = sum(
            float(value or 0.0) for value in self._realized_pnl_by_player.values()
        )
        peak_usd = max(float(self._panteon_realized_peak_pnl_usd or 0.0), pnl_usd)
        peak_pct = peak_usd / initial_capital * 100.0
        pnl_pct = pnl_usd / initial_capital * 100.0
        giveback_pct = max(0.0, peak_pct - pnl_pct)
        if peak_pct < cfg.v3_panteon_equity_guard_min_peak_pnl_pct:
            return None
        if giveback_pct < cfg.v3_panteon_equity_guard_max_giveback_pct:
            return None
        if pnl_pct > cfg.v3_panteon_equity_guard_floor_pnl_pct:
            return None
        cooldown = int(cfg.v3_panteon_equity_guard_cooldown_bars)
        self._panteon_equity_guard_until_bar = int(current_bar) + max(0, cooldown)
        return (
            "v3 panteon equity guard: realized pnl_pct "
            f"{pnl_pct:.2f}, peak_pnl_pct {peak_pct:.2f}, "
            f"giveback_pct {giveback_pct:.2f}"
        )

    def _validate_v3_persistent_realized_loss(
        self,
        player: Player,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        label = player.label
        closed_trades = int(self._realized_trade_counts_by_player.get(label, 0) or 0)
        if closed_trades < cfg.v3_persistent_loss_kill_min_closed_trades:
            return None
        initial_capital = float(self._realized_initial_capital or 0.0)
        if initial_capital <= 0:
            return None
        pnl_usd = float(self._realized_pnl_by_player.get(label, 0.0) or 0.0)
        pnl_pct = pnl_usd / initial_capital * 100.0
        wins = int(self._realized_win_counts_by_player.get(label, 0) or 0)
        win_rate = wins / closed_trades * 100.0 if closed_trades > 0 else 0.0
        pnl_breach = pnl_pct <= cfg.v3_persistent_loss_kill_pnl_pct
        weak_negative_edge = (
            cfg.v3_persistent_loss_kill_win_rate_pct > 0.0
            and pnl_pct < 0.0
            and win_rate <= cfg.v3_persistent_loss_kill_win_rate_pct
        )
        if not (pnl_breach or weak_negative_edge):
            return None
        if not self._has_v3_persistent_virtual_weakness(player):
            return None
        return _DisqualificationReason(
            label=label,
            reason=(
                f"v3 persistent realized-loss kill: pnl_usd {pnl_usd:.2f}, "
                f"pnl_pct {pnl_pct:.2f}, win_rate {win_rate:.2f}% with "
                f"{closed_trades} realized closed trades"
            ),
        )

    def _has_v3_persistent_virtual_weakness(self, player: Player) -> bool:
        cfg = self._config
        if not cfg.v3_persistent_loss_requires_virtual_weakness:
            return True
        metrics = self._virtual_persistent_metrics(player)
        if not metrics.has_data:
            return True
        return (
            metrics.pnl_pct <= cfg.v3_persistent_loss_virtual_max_pnl_pct
            or metrics.max_dd_pct >= cfg.v3_persistent_loss_virtual_min_dd_pct
        )

    def _validate_real_promotion(
        self,
        player: Player,
        regime: Regime,
        score: float,
        *,
        current_bar: int,
    ) -> Optional[_DisqualificationReason]:
        cfg = self._config
        metrics = self._real_promotion_metrics(player, regime)
        if metrics.pnl_pct < cfg.real_promotion_loss_budget_pct:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"real promotion gate: pnl_pct {metrics.pnl_pct:.2f} "
                    f"below loss budget {cfg.real_promotion_loss_budget_pct:.2f}"
                ),
            )
        if metrics.closed_trades < cfg.real_promotion_min_closed_trades:
            if self._current is not None and player.label == self._current.label:
                return None
            if (
                self._is_genetics_candidate(player)
                and not self._allows_genetics_probation_shadow_rescue(player, regime)
            ):
                if cfg.v3_probation_shadow_rescue_allow_genetics:
                    reason = (
                        "real promotion gate: genetics probation ineligible for "
                        "scoped rescue with "
                    )
                else:
                    reason = "real promotion gate: genetics probation disabled with "
                return _DisqualificationReason(
                    label=player.label,
                    reason=f"{reason}{metrics.closed_trades} real closed trades",
                )
            if self._has_v3_probation_shadow_rescue(
                player,
                regime,
                current_bar=current_bar,
                real_metrics=metrics,
            ):
                return None
            if score >= cfg.real_promotion_probation_min_score:
                return None
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"real promotion gate: probation score {score:.4f} < "
                    f"{cfg.real_promotion_probation_min_score:.4f} with "
                    f"{metrics.closed_trades} real closed trades"
                ),
            )
        if metrics.pnl_pct < cfg.real_promotion_min_pnl_pct:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"real promotion gate: pnl_pct {metrics.pnl_pct:.2f} < "
                    f"{cfg.real_promotion_min_pnl_pct:.2f}"
                ),
            )
        if metrics.max_dd_pct > cfg.real_promotion_max_drawdown_pct:
            return _DisqualificationReason(
                label=player.label,
                reason=(
                    f"real promotion gate: max_dd_pct {metrics.max_dd_pct:.2f} > "
                    f"{cfg.real_promotion_max_drawdown_pct:.2f}"
                ),
            )
        return None

    def _has_v3_probation_shadow_rescue(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int,
        real_metrics: Optional[Metrics] = None,
    ) -> bool:
        cfg = self._config
        if not (
            cfg.real_promotion_gate_enabled
            and cfg.v3_probation_shadow_rescue_enabled
        ):
            return False
        if (
            self._is_genetics_candidate(player)
            and not self._allows_genetics_probation_shadow_rescue(player, regime)
        ):
            return False
        metrics = real_metrics or self._real_promotion_metrics(player, regime)
        if metrics.closed_trades >= cfg.real_promotion_min_closed_trades:
            return False
        if metrics.pnl_pct < cfg.real_promotion_loss_budget_pct:
            return False
        virtual = self._virtual_persistent_metrics(player)
        if not virtual.has_data:
            return False
        if virtual.pnl_pct < cfg.v3_probation_shadow_rescue_min_virtual_pnl_pct:
            return False
        if virtual.max_dd_pct > cfg.v3_probation_shadow_rescue_max_virtual_dd_pct:
            return False
        stats = self._entry_causal_score.score_with_stats(
            label=player.label,
            regime=regime.label,
            current_bar=current_bar,
        )
        if stats.actionable_share < cfg.v3_probation_shadow_rescue_min_actionable_share:
            return False
        if stats.recent_filled < cfg.v3_probation_shadow_rescue_min_recent_filled:
            return False
        if stats.recent_pnl_usd < cfg.v3_probation_shadow_rescue_min_recent_pnl_usd:
            return False
        return True

    def _real_promotion_metrics(self, player: Player, regime: Regime) -> Metrics:
        real_perf = getattr(self, "_real_perf", None)
        if real_perf is None:
            return Metrics.empty()
        own = real_perf.get(player.label, regime=regime)
        if own.has_data:
            return own
        own_all = real_perf.get(player.label)
        if own_all.has_data:
            return own_all

        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        pnl = 0.0
        max_dd = 0.0
        sharpe_values: List[float] = []
        for label in player.agent_labels:
            metrics = real_perf.get(label, regime=regime)
            if not metrics.has_data:
                metrics = real_perf.get(label)
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
        if closed == entries == signals == execution_failures == 0:
            return Metrics.empty()
        return Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=(sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0),
            max_dd_pct=max_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
        )

    def _real_persistent_metrics(self, player: Player) -> Metrics:
        real_perf = getattr(self, "_real_perf", None)
        if real_perf is None:
            return Metrics.empty()
        own_all = real_perf.get(player.label)
        if own_all.has_data:
            return own_all

        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        pnl = 0.0
        max_dd = 0.0
        sharpe_values: List[float] = []
        for label in player.agent_labels:
            metrics = real_perf.get(label)
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
        if closed == entries == signals == execution_failures == 0:
            return Metrics.empty()
        return Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=(sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0),
            max_dd_pct=max_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
        )

    def _virtual_persistent_metrics(self, player: Player) -> Metrics:
        own_all = self._perf.get(player.label)
        if own_all.has_data:
            return own_all

        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        pnl = 0.0
        max_dd = 0.0
        sharpe_values: List[float] = []
        for label in player.agent_labels:
            metrics = self._perf.get(label)
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
        if closed == entries == signals == execution_failures == 0:
            return Metrics.empty()
        return Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=(sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0),
            max_dd_pct=max_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
        )

    def _real_session_metrics_for_player(self, player: Player, regime: Regime) -> Metrics:
        real_perf = getattr(self, "_real_perf", None)
        if real_perf is None:
            return Metrics.empty()
        if not self._real_session_baseline:
            return self._real_promotion_metrics(player, regime)

        own = self._metrics_delta_from_baseline(
            real_perf.get(player.label, regime=regime),
            self._real_session_baseline.get((player.label, regime), Metrics.empty()),
        )
        if own.has_data:
            return own

        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        pnl = 0.0
        max_dd = 0.0
        sharpe_values: List[float] = []
        for label in player.agent_labels:
            current = real_perf.get(label, regime=regime)
            baseline = self._real_session_baseline.get((label, regime), Metrics.empty())
            metrics = self._metrics_delta_from_baseline(current, baseline)
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
        if closed == entries == signals == execution_failures == 0:
            return Metrics.empty()
        return Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=(sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0),
            max_dd_pct=max_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
        )

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

    @staticmethod
    def _v3_metrics_view(
        recent_real: Metrics,
        long_real: Metrics,
        virtual: Metrics,
    ) -> Metrics:
        for metrics in (recent_real, long_real, virtual):
            if metrics.has_data:
                return Metrics(
                    closed_trades=metrics.closed_trades,
                    entries=metrics.entries,
                    signals=metrics.signals,
                    execution_failures=metrics.execution_failures,
                )
        return Metrics.empty()

    @staticmethod
    def _metrics_delta_from_baseline(current: Metrics, baseline: Metrics) -> Metrics:
        return Metrics(
            pnl_pct=current.pnl_pct - baseline.pnl_pct,
            closed_trades=max(0, current.closed_trades - baseline.closed_trades),
            entries=max(0, current.entries - baseline.entries),
            signals=max(0, current.signals - baseline.signals),
            wins=max(0, current.wins - baseline.wins),
            losses=max(0, current.losses - baseline.losses),
            max_dd_pct=max(0.0, current.max_dd_pct - baseline.max_dd_pct),
            blocked_signals=max(0, current.blocked_signals - baseline.blocked_signals),
            rejected_signals=max(0, current.rejected_signals - baseline.rejected_signals),
            pending_signals=max(0, current.pending_signals - baseline.pending_signals),
            execution_failures=max(
                0,
                current.execution_failures - baseline.execution_failures,
            ),
        )

    def _ensure_session_baseline(
        self,
        players: Sequence[Player],
        regime: Regime,
    ) -> None:
        for player in players:
            for label in (player.label, *tuple(player.agent_labels)):
                key = (label, regime)
                if key not in self._session_baseline:
                    self._session_baseline[key] = self._perf.get(label, regime=regime)

    def _session_metrics_for_player(self, player: Player, regime: Regime) -> Metrics:
        own = self._session_delta(player.label, regime)
        if self._metrics_activity(own) > 0:
            return own
        closed = entries = signals = wins = losses = 0
        blocked = rejected = pending = execution_failures = 0
        pnl = 0.0
        max_dd = 0.0
        sharpe_values: List[float] = []
        for label in player.agent_labels:
            metrics = self._session_delta(label, regime)
            if self._metrics_activity(metrics) <= 0:
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
        if closed == entries == signals == execution_failures == 0:
            return Metrics.empty()
        return Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            sharpe=(sum(sharpe_values) / len(sharpe_values) if sharpe_values else 0.0),
            max_dd_pct=max_dd,
            blocked_signals=blocked,
            rejected_signals=rejected,
            pending_signals=pending,
            execution_failures=execution_failures,
        )

    def _session_delta(self, label: str, regime: Regime) -> Metrics:
        current = self._perf.get(label, regime=regime)
        baseline = self._session_baseline.get((label, regime), Metrics.empty())
        return Metrics(
            pnl_pct=current.pnl_pct - baseline.pnl_pct,
            closed_trades=max(0, current.closed_trades - baseline.closed_trades),
            entries=max(0, current.entries - baseline.entries),
            signals=max(0, current.signals - baseline.signals),
            wins=max(0, current.wins - baseline.wins),
            losses=max(0, current.losses - baseline.losses),
            max_dd_pct=max(0.0, current.max_dd_pct - baseline.max_dd_pct),
            blocked_signals=max(0, current.blocked_signals - baseline.blocked_signals),
            rejected_signals=max(0, current.rejected_signals - baseline.rejected_signals),
            pending_signals=max(0, current.pending_signals - baseline.pending_signals),
            execution_failures=max(0, current.execution_failures - baseline.execution_failures),
        )

    @staticmethod
    def _metrics_activity(metrics: Metrics) -> int:
        return (
            int(metrics.closed_trades)
            + int(metrics.entries)
            + int(metrics.signals)
            + int(metrics.execution_failures)
        )

    @staticmethod
    def _cap_abs(value: float, cap: float) -> float:
        if cap <= 0:
            return float(value)
        return max(-cap, min(cap, float(value)))

    def _apply_player_session_overlay(
        self,
        player: Player,
        regime: Regime,
        score: float,
    ) -> Tuple[float, float, float, float, float]:
        cfg = self._config
        metrics = self._session_metrics_for_player(player, regime)
        activity = self._metrics_activity(metrics)
        session_pnl = self._cap_abs(metrics.pnl_pct, cfg.player_session_pnl_cap_pct)
        stale_penalty = 0.0
        underperformance_penalty = 0.0
        if activity >= cfg.player_session_min_activity or abs(session_pnl) > 1e-12:
            delta = cfg.player_session_overlay_weight * session_pnl
            underperformance_penalty = max(0.0, -session_pnl) * cfg.player_session_underperformance_weight
        else:
            delta = 0.0
            stale_penalty = cfg.player_session_stale_penalty
        adjusted = float(score) + delta - underperformance_penalty - stale_penalty
        return adjusted, adjusted - float(score), session_pnl, underperformance_penalty, stale_penalty

    def _rank_candidates(
        self,
        candidates: Sequence[Player],
        regime: Regime,
        *,
        current_bar: int = 0,
        market_tags: Sequence[str] = (),
    ) -> Tuple[CandidateScore, ...]:
        rows = [
            self._score_player_detail(
                player,
                regime,
                current_bar=current_bar,
                market_tags=market_tags,
            )
            for player in candidates
        ]
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

    def _score_player(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int = 0,
        market_tags: Sequence[str] = (),
    ) -> float:
        return self._score_player_detail(
            player,
            regime,
            current_bar=current_bar,
            market_tags=market_tags,
        ).score

    def _score_player_detail(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int = 0,
        market_tags: Sequence[str] = (),
    ) -> CandidateScore:
        """Score для игрока plus forensic details."""
        if self._config.use_v3_rolling_score:
            row = self._score_player_detail_v3(
                player,
                regime,
                current_bar=current_bar,
                market_tags=market_tags,
            )
            return self._apply_hard_policy_score_penalty(
                row,
                player,
                regime,
                current_bar=current_bar,
                market_tags=market_tags,
            )

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
            (
                score,
                session_delta,
                session_pnl,
                session_underperformance,
                session_stale,
            ) = self._apply_player_session_overlay(player, regime, score)
            score, penalty = self._apply_uncertainty_penalty(score, own_metrics)
            score = self._apply_execution_penalty(score, own_metrics)
            return self._apply_hard_policy_score_penalty(
                CandidateScore(
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
                    session_score_delta=session_delta,
                    session_pnl_pct=session_pnl,
                    session_underperformance_penalty=session_underperformance,
                    session_stale_penalty=session_stale,
                ),
                player,
                regime,
                current_bar=current_bar,
                market_tags=market_tags,
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
            return self._apply_hard_policy_score_penalty(
                CandidateScore(
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
                ),
                player,
                regime,
                current_bar=current_bar,
                market_tags=market_tags,
            )
        base_score = _max_biased_aggregate(
            agent_scores, self._config.ensemble_bootstrap_max_bias
        )
        if has_closed_experience:
            base_score = self._apply_affinity(player, regime, base_score)
        score = self._apply_real_overlay(player, regime, base_score)
        (
            score,
            session_delta,
            session_pnl,
            session_underperformance,
            session_stale,
        ) = self._apply_player_session_overlay(player, regime, score)
        metrics_view = Metrics(
            closed_trades=agg_closed,
            signals=agg_signals,
            execution_failures=agg_exec_failures,
        )
        score, penalty = self._apply_uncertainty_penalty(score, metrics_view)
        return self._apply_hard_policy_score_penalty(
            CandidateScore(
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
                session_score_delta=session_delta,
                session_pnl_pct=session_pnl,
                session_underperformance_penalty=session_underperformance,
                session_stale_penalty=session_stale,
            ),
            player,
            regime,
            current_bar=current_bar,
            market_tags=market_tags,
        )

    def _score_player_detail_v3(
        self,
        player: Player,
        regime: Regime,
        *,
        current_bar: int = 0,
        market_tags: Sequence[str] = (),
    ) -> CandidateScore:
        agent_labels = tuple(player.agent_labels)
        memory_keys = (self._memory_key(player.label, regime),) + tuple(
            self._memory_key(label, regime) for label in agent_labels
        )
        if self._config.use_v3_entry_causal_score:
            return self._score_player_detail_v3_entry_causal(
                player,
                regime,
                agent_labels=agent_labels,
                memory_keys=memory_keys,
                current_bar=current_bar,
            )
        if self._config.use_v3_soft_shadow_score:
            return self._score_player_detail_v3_soft_shadow(
                player,
                regime,
                agent_labels=agent_labels,
                memory_keys=memory_keys,
                current_bar=current_bar,
            )
        if self._config.use_v3_shadow_rolling_score:
            return self._score_player_detail_v3_shadow(
                player,
                regime,
                agent_labels=agent_labels,
                memory_keys=memory_keys,
                current_bar=current_bar,
            )
        long_real = self._real_promotion_metrics(player, regime)
        recent_real = self._real_session_metrics_for_player(player, regime)
        virtual = self._promotion_metrics(player, regime)
        result = score_rolling_decision(
            RollingDecisionScoreInput(
                recent_real_metrics=recent_real,
                long_real_metrics=long_real,
                virtual_metrics=virtual,
            ),
            config=self._config.v3_score_config,
        )
        metrics_view = self._v3_metrics_view(recent_real, long_real, virtual)
        score = result.score
        if result.source == "virtual":
            score = min(score, self._config.v3_virtual_only_score_cap)
        if metrics_view.closed_trades > 0:
            score = self._apply_affinity(player, regime, score)
        score = self._apply_execution_penalty(score, metrics_view)
        return CandidateScore(
            label=player.label,
            score=score,
            rank=0,
            has_data=metrics_view.has_data,
            closed_trades=metrics_view.closed_trades,
            signals=metrics_view.signals,
            execution_failures=metrics_view.execution_failures,
            score_source="v3_rolling",
            uncertainty_penalty=0.0,
            agent_labels=agent_labels,
            memory_keys_read=memory_keys,
            session_score_delta=result.recent_real_component,
            session_pnl_pct=recent_real.pnl_pct,
            session_underperformance_penalty=result.negative_real_penalty,
            session_stale_penalty=0.0,
        )

    def _score_player_detail_v3_soft_shadow(
        self,
        player: Player,
        regime: Regime,
        *,
        agent_labels: Tuple[str, ...],
        memory_keys: Tuple[str, ...],
        current_bar: int,
    ) -> CandidateScore:
        score, closed_trades = self._soft_shadow_score.score_with_trade_count(
            label=player.label,
            regime=regime.label,
            current_bar=current_bar,
        )
        has_data = closed_trades >= self._config.v3_shadow_rolling_min_closed_trades
        if not has_data:
            score = min(score, float(self._config.v3_score_config.no_data_score))
        capital = max(1e-9, float(self._realized_initial_capital or 0.0))
        session_pnl_pct = float(score) / capital * 100.0
        return CandidateScore(
            label=player.label,
            score=float(score),
            rank=0,
            has_data=has_data,
            closed_trades=int(closed_trades),
            signals=0,
            execution_failures=0,
            score_source="v3_soft_shadow",
            uncertainty_penalty=0.0,
            agent_labels=agent_labels,
            memory_keys_read=memory_keys,
            session_score_delta=float(score),
            session_pnl_pct=session_pnl_pct,
            session_underperformance_penalty=max(0.0, -session_pnl_pct),
            session_stale_penalty=0.0,
        )

    def _score_player_detail_v3_entry_causal(
        self,
        player: Player,
        regime: Regime,
        *,
        agent_labels: Tuple[str, ...],
        memory_keys: Tuple[str, ...],
        current_bar: int,
    ) -> CandidateScore:
        stats = self._entry_causal_score.score_with_stats(
            label=player.label,
            regime=regime.label,
            current_bar=current_bar,
        )
        score = float(stats.score)
        has_data = stats.has_data
        score_source = "v3_entry_causal"
        if not stats.has_data:
            if self._has_v3_probation_shadow_rescue(
                player,
                regime,
                current_bar=current_bar,
            ):
                fill_confidence = (
                    1.0
                    if self._config.v3_entry_causal_min_filled <= 0
                    else min(
                        1.0,
                        stats.recent_filled
                        / max(1, self._config.v3_entry_causal_min_filled),
                    )
                )
                actionability_factor = (
                    stats.actionable_share
                    ** self._config.v3_entry_causal_actionability_weight
                    if self._config.v3_entry_causal_actionability_weight > 0.0
                    else 1.0
                )
                score = max(
                    float(self._config.v3_score_config.no_data_score),
                    float(stats.recent_pnl_usd) * fill_confidence * actionability_factor,
                )
                has_data = True
                score_source = "v3_entry_causal_probation_rescue"
            else:
                score = min(score, float(self._config.v3_score_config.no_data_score))
        capital = max(1e-9, float(self._realized_initial_capital or 0.0))
        session_pnl_pct = float(stats.recent_pnl_usd) / capital * 100.0
        return CandidateScore(
            label=player.label,
            score=score,
            rank=0,
            has_data=has_data,
            closed_trades=int(stats.closed_trades),
            signals=int(stats.recent_signals),
            execution_failures=0,
            score_source=score_source,
            uncertainty_penalty=0.0,
            agent_labels=agent_labels,
            memory_keys_read=memory_keys,
            session_score_delta=score,
            session_pnl_pct=session_pnl_pct,
            session_underperformance_penalty=max(0.0, -session_pnl_pct),
            session_stale_penalty=max(0.0, 1.0 - stats.actionable_share),
            recent_bars=int(stats.recent_bars),
            recent_actionable_bars=int(stats.recent_actionable_bars),
            actionable_share=float(stats.actionable_share),
            recent_filled=int(stats.recent_filled),
            recent_pnl_usd=float(stats.recent_pnl_usd),
        )

    def _score_player_detail_v3_shadow(
        self,
        player: Player,
        regime: Regime,
        *,
        agent_labels: Tuple[str, ...],
        memory_keys: Tuple[str, ...],
        current_bar: int,
    ) -> CandidateScore:
        state = self._shadow_rolling_by_player_regime.get((player.label, regime.label))
        if state is None:
            return CandidateScore(
                label=player.label,
                score=float(self._config.v3_score_config.no_data_score),
                rank=0,
                has_data=False,
                closed_trades=0,
                signals=0,
                execution_failures=0,
                score_source="v3_shadow_rolling",
                uncertainty_penalty=0.0,
                agent_labels=agent_labels,
                memory_keys_read=memory_keys,
            )
        rolling_pnl_usd = state.rolling_pnl_usd(
            current_bar=current_bar,
            window_bars=self._config.v3_shadow_rolling_window_bars,
        )
        capital = max(1e-9, float(self._realized_initial_capital or 0.0))
        score = rolling_pnl_usd / capital * 100.0
        has_data = state.total_trades >= self._config.v3_shadow_rolling_min_closed_trades
        if not has_data:
            score = min(score, float(self._config.v3_score_config.no_data_score))
        return CandidateScore(
            label=player.label,
            score=score,
            rank=0,
            has_data=has_data,
            closed_trades=state.total_trades,
            signals=0,
            execution_failures=0,
            score_source="v3_shadow_rolling",
            uncertainty_penalty=0.0,
            agent_labels=agent_labels,
            memory_keys_read=memory_keys,
            session_score_delta=score,
            session_pnl_pct=score,
            session_underperformance_penalty=max(0.0, -score),
            session_stale_penalty=0.0,
        )

    def _apply_hard_policy_score_penalty(
        self,
        row: CandidateScore,
        player: Player,
        regime: Regime,
        *,
        current_bar: int,
        market_tags: Sequence[str] = (),
    ) -> CandidateScore:
        cfg = self._config
        if (
            not cfg.hard_policy_enabled
            or cfg.hard_policy_neutral_penalty <= 0.0
            or regime is not Regime.NEUTRAL
            or player.label == self._no_trade.label
            or player.label not in set(cfg.hard_policy_neutral_penalty_labels)
        ):
            return row
        tag_set = set(self._normalize_market_tags(market_tags))
        penalize = bool({"noisy", "low_actionability"}.intersection(tag_set))
        if not penalize:
            stats = self._entry_causal_score.score_with_stats(
                label=player.label,
                regime=regime.label,
                current_bar=current_bar,
            )
            penalize = (
                stats.recent_bars >= cfg.hard_policy_min_actionability_bars
                and stats.actionable_share <= cfg.hard_policy_low_actionability_share
            )
        if not penalize:
            return row
        penalty = float(cfg.hard_policy_neutral_penalty)
        return replace(
            row,
            score=float(row.score) - penalty,
            session_stale_penalty=float(row.session_stale_penalty) + penalty,
        )

    @staticmethod
    def _normalize_market_tags(tags: Sequence[str]) -> Tuple[str, ...]:
        out: List[str] = []
        seen = set()
        for raw in tags or ():
            tag = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
            while "__" in tag:
                tag = tag.replace("__", "_")
            if not tag:
                continue
            aliases = {
                "highrange": "high_range",
                "rangehigh": "high_range",
                "range_high": "high_range",
                "high_vol": "high_range",
                "high_volatility": "high_range",
                "noise": "noisy",
                "high_noise": "noisy",
                "low_action": "low_actionability",
                "low_actionable": "low_actionability",
            }
            tag = aliases.get(tag, tag)
            if tag in seen:
                continue
            seen.add(tag)
            out.append(tag)
        return tuple(out)

    @staticmethod
    def _is_genetics_candidate(player: Player) -> bool:
        labels = (player.label, *tuple(player.agent_labels))
        return any("genetic" in str(label).lower() for label in labels)

    def _allows_genetics_probation_shadow_rescue(
        self,
        player: Player,
        regime: Regime,
    ) -> bool:
        if not self._config.v3_probation_shadow_rescue_allow_genetics:
            return False
        if str(player.label) != "GeneticsResearch":
            return False
        return regime in (Regime.BEARISH, Regime.CRASH)

    @staticmethod
    def _is_genetics_label(label: str) -> bool:
        return "genetic" in str(label).lower()

    @staticmethod
    def _solo_specialist_label(player: Player) -> str:
        agent_labels = tuple(str(label) for label in player.agent_labels)
        if len(agent_labels) == 1:
            return agent_labels[0]
        label = str(player.label or "")
        if label.startswith("Solo_"):
            return label[len("Solo_"):]
        return ""

    @staticmethod
    def _specialist_allow_policy(label: str) -> Tuple[frozenset[str], frozenset[str]]:
        if label == "FundingArb":
            return frozenset({"bearish"}), frozenset()
        if label == "VolBreakoutHunter" or "VolBreakout" in label:
            return frozenset({"bullish"}), frozenset({"high_range"})
        if label == "LiveCrashHunter" or "CrashHunter" in label:
            return frozenset({"bearish", "crash"}), frozenset()
        if label == "ResearchValidatorAgent":
            return frozenset({"neutral"}), frozenset({"noisy"})
        return frozenset(), frozenset()

    @staticmethod
    def _memory_key(label: str, regime: Regime) -> str:
        return f"{label}|{regime.label}"

    @staticmethod
    def _object_value(event: object, key: str) -> object:
        if isinstance(event, dict):
            return event.get(key)
        return getattr(event, key, None)

    @staticmethod
    def _safe_int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_float(value: object) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _position_signature(cls, positions: object) -> Tuple[Tuple[str, str], ...]:
        return tuple((sym, side) for sym, side, _ in cls._position_details(positions))

    @classmethod
    def _position_details(cls, positions: object) -> Tuple[Tuple[str, str, int], ...]:
        out: List[Tuple[str, str, int]] = []
        if isinstance(positions, dict):
            iterable = positions.values()
        else:
            iterable = positions or ()
        for pos in iterable:
            sym = str(cls._position_value(pos, "sym") or "").upper()
            side = str(cls._position_value(pos, "side") or "").lower()
            if sym and side in ("long", "short"):
                opened_bar = cls._safe_int(cls._position_value(pos, "opened_bar"))
                out.append((sym, side, opened_bar))
        return tuple(sorted(set(out)))

    @staticmethod
    def _position_value(pos: object, key: str) -> object:
        if isinstance(pos, dict):
            return pos.get(key)
        return getattr(pos, key, None)

    @staticmethod
    def _format_position_signature(signature: Tuple[Tuple[str, str], ...]) -> str:
        if not signature:
            return "flat"
        return ",".join(f"{sym}:{side}" for sym, side in signature)

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
