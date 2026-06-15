"""Panteon_Flash per-symbol decision allocator."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..domain.types import Action, MarketSnapshot, Metrics, Regime, Signal
from ..memory import PerformanceMemory, QuarantineManager
from ..scoring import DEFAULT_SCORING, ScoringConfig, regime_score
from .agent import Agent
from .player import Player, normalize_vote_result


_SHADOW_POSITION_REPLAY_AGENT = "ShadowPositionReplay"
_SYSTEM_CLOSE_AGENTS = frozenset((
    "CashFlat",
    "GeneticsProbationRegimeExit",
    "PartialProfitLock",
    "ShadowPositionReplay",
    "StalePositionGuard",
))
_QUOTE_ASSET_SUFFIXES = ("USDT", "USDC", "USD")
_SYMBOL_SEPARATORS = ("/", "-", "_")


def _actor_cap_override_map(overrides: Sequence[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in overrides:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.count("=") != 1:
            raise ValueError(
                "promoted actor cap override must use 'actor_key=cap' format"
            )
        actor_key, raw_cap = (part.strip() for part in text.split("=", 1))
        if not actor_key or not raw_cap:
            raise ValueError(
                "promoted actor cap override must use 'actor_key=cap' format"
            )
        try:
            cap = int(raw_cap)
        except ValueError as exc:
            raise ValueError(
                "promoted actor cap override cap must be a positive integer"
            ) from exc
        if cap <= 0 or str(cap) != raw_cap:
            raise ValueError(
                "promoted actor cap override cap must be a positive integer"
            )
        parsed[actor_key] = cap
    return parsed


def _signal_key_float_map(overrides: Sequence[str], *, field_name: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in overrides:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.count("=") != 1:
            raise ValueError(f"{field_name} must use 'actor_key|symbol|action=value' format")
        raw_key, raw_value = (part.strip() for part in text.split("=", 1))
        if not raw_key or not raw_value:
            raise ValueError(f"{field_name} must use 'actor_key|symbol|action=value' format")
        key = _normalize_signal_deny_key(raw_key)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field_name} value must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field_name} value must be a finite number")
        parsed[key] = value
    return parsed


def _signal_context_key_float_map(
    overrides: Sequence[str],
    *,
    field_name: str,
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for raw in overrides:
        text = str(raw or "").strip()
        if not text:
            continue
        if text.count("=") != 1:
            raise ValueError(
                f"{field_name} must use 'actor_key|symbol|action|regime=value' format"
            )
        raw_key, raw_value = (part.strip() for part in text.split("=", 1))
        if not raw_key or not raw_value:
            raise ValueError(
                f"{field_name} must use 'actor_key|symbol|action|regime=value' format"
            )
        key = _normalize_signal_context_key(raw_key)
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field_name} value must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{field_name} value must be a finite number")
        parsed[key] = value
    return parsed


def _normalize_string_tuple(raw: object) -> tuple[str, ...]:
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        items: Iterable[object] = (item.strip() for item in raw.split(","))
    else:
        items = raw if isinstance(raw, Iterable) else (raw,)
    return tuple(str(item).strip() for item in items if str(item or "").strip())


def _normalize_signal_key_tuple(raw: object, *, field_name: str) -> tuple[str, ...]:
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        items: Iterable[object] = (item.strip() for item in raw.split(","))
    else:
        items = raw if isinstance(raw, Iterable) else (raw,)
    keys: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        try:
            keys.append(_normalize_signal_deny_key(text))
        except ValueError as exc:
            raise ValueError(f"{field_name} must use 'actor_key|symbol|action' format") from exc
    return tuple(dict.fromkeys(keys))


def _normalize_regime_tuple(raw: object) -> tuple[Regime, ...]:
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        items: Iterable[object] = (item.strip() for item in raw.split(","))
    else:
        items = raw if isinstance(raw, Iterable) else (raw,)

    regimes: list[Regime] = []
    for item in items:
        if item in (None, ""):
            continue
        regimes.append(_normalize_regime_value(item))
    return tuple(dict.fromkeys(regimes))


def _normalize_regime_value(raw: object) -> Regime:
    if isinstance(raw, Regime):
        return raw
    if isinstance(raw, int):
        try:
            return Regime(raw)
        except ValueError as exc:
            raise ValueError("Flash denied open regime is invalid") from exc
    value = str(raw or "").strip().lower()
    aliases = {
        "bull": Regime.BULLISH,
        "up": Regime.BULLISH,
        "uptrend": Regime.BULLISH,
        "bear": Regime.BEARISH,
        "down": Regime.BEARISH,
        "downtrend": Regime.BEARISH,
        "flat": Regime.NEUTRAL,
        "sideways": Regime.NEUTRAL,
        "range": Regime.NEUTRAL,
        "crash": Regime.CRASH,
        "panic": Regime.CRASH,
        "flash_crash": Regime.CRASH,
        "low-vol range": Regime.RANGE_LOW_VOL,
        "low_vol_range": Regime.RANGE_LOW_VOL,
        "choppy down": Regime.CHOPPY_DOWN,
        "choppy up": Regime.CHOPPY_UP,
        "mixed/rotational": Regime.MIXED_ROTATIONAL,
        "rotational": Regime.MIXED_ROTATIONAL,
    }
    labels = {regime.label: regime for regime in Regime}
    if value in labels:
        return labels[value]
    if value in aliases:
        return aliases[value]
    raise ValueError(
        "Flash denied open regime must be one of the canonical Regime labels"
    )


def _normalize_symbol_tuple(raw: object) -> tuple[str, ...]:
    if raw in (None, ""):
        return ()
    if isinstance(raw, str):
        items: Iterable[object] = (item.strip() for item in raw.split(","))
    else:
        items = raw if isinstance(raw, Iterable) else (raw,)
    symbols = [
        str(item or "").strip().upper()
        for item in items
        if str(item or "").strip()
    ]
    return tuple(dict.fromkeys(symbols))


def _normalize_actor_key_tuple(raw: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_normalize_string_tuple(raw)))


@dataclass(frozen=True)
class FlashAllocatorConfig:
    """Conservative scoring knobs for the first Flash kernel."""

    min_score_to_trade: float = 0.0
    actionable_bonus: float = 0.25
    no_data_score: float = 0.0
    min_closed_trades_to_trade: int = 3
    min_pnl_pct_to_trade: float = 0.0
    # Phase 2 / A3: гейтить по СРЕДНЕМУ PnL на сделку, а не по кумулятивному
    # pnl_pct. Кумулятив отвергает любой исторически-минусовой актор навсегда
    # (одна ранняя серия убытков хоронит актора). pnl_per_trade сравнивает
    # средний эдж за сделку с порогом min_pnl_pct_to_trade (что и подразумевалось
    # в пресетах вида 0.10). Default False — поведение сохраняется; включать
    # после ретро-валидации (Фаза 4).
    gate_pnl_per_trade_enabled: bool = False
    # Phase 1 / B1+B2: агрегация синтетических метрик ансамбля из компонентов.
    # Счётчики (closed/signals/...) суммируются (наследование допуска, B4),
    # а pnl/комиссии/просадка — трейд-взвешенные средние (не суммы, чтобы не
    # умножать издержки ~N×). Скор — max-смещённый: bias*max + (1-bias)*wmean.
    component_score_max_bias: float = 0.70
    # Selection fix: вето по ГЛОБАЛЬНОМУ здоровью актора. Per-(symbol,regime)
    # скоринг прячет катастрофичный агрегат: глобально-убыточный высокочастотный
    # актор (напр. PlayerFunding −29% за 2528 сделок) выигрывает узкие срезы и
    # переселектируется. Гейт отклоняет открытие, если агрегатный (по всем
    # режимам) кумулятивный PnL ниже порога ИЛИ просадка выше порога — при
    # достаточной выборке. Default off (поведение сохраняется).
    global_health_gate_enabled: bool = False
    global_health_min_cum_pnl_pct: float = -5.0
    global_health_max_drawdown_pct: float = 100.0
    global_health_min_closed_trades: int = 50
    # Agent-quality fix: regime-edge gate. Открывать актором только в режиме, где
    # у него доказанный положительный per-trade эдж (с достаточной выборкой).
    # Эдж регим-специфичен: LiveMeanRev прибылен в bearish (+7.97%), но бьётся в
    # neutral (−1.35%). Гейт не даёт торговать агентом в режиме, где он не зарабатывает.
    # Default off.
    regime_edge_gate_enabled: bool = False
    regime_edge_min_pnl_per_trade_pct: float = 0.0
    regime_edge_min_closed_trades: int = 10
    # Stable-pool ранжирование: выбирать актора по СТАБИЛЬНОМУ per-regime эджу
    # (pnl_per_trade по всем символам режима), а не по волатильному
    # per-(symbol,regime) score. Это устраняет анти-предиктивный performance-
    # chasing (score пиково растёт после выигрыша → вход на реверсию). Default off.
    stable_pool_ranking_enabled: bool = False
    # #3 (предиктивный вход, non-ML): не открывать, когда уверенность режима ниже
    # порога — неуверенные (mixed/переходные) рынки = зона кровотечения. Мягко
    # предиктивно: уверенные режимы устойчивее. Default 0.0 = off.
    min_regime_confidence_to_open: float = 0.0
    # ── Live-audit fixes (real-aware execution) ───────────────────────────
    # Real-PnL gate: вето открытия, если REAL realized PnL актора ниже порога
    # при достаточной выборке real-сделок. Чинит корень: shadow≠real.
    real_loss_gate_enabled: bool = False
    real_loss_gate_min_pnl_pct: float = -0.5
    real_loss_gate_min_closed_trades: int = 5
    # Fee-aware admission: не открывать, если ожидаемый per-trade edge меньше
    # round-trip издержек (fee+spread+slippage) × safety_mult.
    fee_aware_admission_enabled: bool = False
    fee_aware_round_trip_cost_pct: float = 0.06  # 2×fee + спред/slippage, %
    fee_aware_safety_mult: float = 1.5
    # Churn limiter: cooldown между открытиями по символу, дневной лимит, anti-flip.
    churn_cooldown_bars: int = 0
    churn_max_opens_per_day: int = 0
    churn_anti_flip_enabled: bool = False
    # Single-agent whitelist (#1): если задан, ТОЛЬКО эти акторы могут открывать
    # реальные позиции; всё остальное (включая ансамбли) → shadow-only. Убирает
    # shadow-mirage селекции и ensemble-churn. Имена матчатся по чистому лейблу
    # и actor_key (agent:X / ensemble:X / Solo_X).
    live_real_actor_whitelist: Tuple[str, ...] = ()
    shadow_confirmation_enabled: bool = False
    shadow_symbol_confirmation_enabled: bool = False
    shadow_actor_fallback_confirmation_enabled: bool = False
    shadow_base_fallback_confirmation_enabled: bool = False
    shadow_signal_handoff_enabled: bool = False
    shadow_actor_fallback_min_base_score: float = 0.0
    shadow_position_replay_actor_fallback_min_base_score: float = 0.0
    shadow_position_replay_actor_fallback_min_shadow_score: float = 0.0
    shadow_base_fallback_actor_keys: Tuple[str, ...] = ()
    actor_switch_margin: float = 0.0
    anchor_actor_keys: Tuple[str, ...] = ()
    portfolio_actor_keys: Tuple[str, ...] = ()
    range_low_vol_real_actor_allowlist: Tuple[str, ...] = ()
    flash_genetics_core_primary_enabled: bool = False
    flash_genetics_core_primary_labels: Tuple[str, ...] = ("GeneticsCore",)
    portfolio_shadow_bootstrap_min_closed_enabled: bool = False
    anchor_min_score_to_trade: Optional[float] = None
    anchor_shadow_min_score: Optional[float] = None
    anchor_min_score_advantage: float = 0.0
    prefer_solo_player_wrappers_enabled: bool = False
    prefer_proven_solo_player_wrappers_enabled: bool = False
    proven_solo_min_score_advantage: float = 0.0
    shadow_confirmation_min_score: float = 0.0
    shadow_confirmation_min_closed_trades: int = 50
    shadow_confirmation_min_full_open_closed_trades: int = 0
    shadow_quality_confirmation_enabled: bool = False
    shadow_confirmation_min_win_rate_pct: float = 0.0
    shadow_confirmation_max_recent_downside_usd: float = 0.0
    shadow_confirmation_min_pnl_per_trade_lcb_usd: Optional[float] = None
    shadow_confirmation_pnl_per_trade_lcb_z: float = 1.0
    shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd: float = 0.0
    shadow_confirmation_pnl_per_trade_lcb_penalty_weight: float = 0.0
    shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled: bool = False
    shadow_confirmation_pnl_per_trade_lcb_risk_min_mult: float = 0.25
    shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd: float = 0.0
    shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd: float = 1.0
    shadow_recovery_admission_enabled: bool = False
    shadow_recovery_actor_keys: Tuple[str, ...] = ()
    shadow_recovery_min_score: Optional[float] = None
    shadow_recovery_min_closed_trades: int = 0
    shadow_symbol_health_enabled: bool = False
    shadow_symbol_health_min_closed_trades: int = 0
    shadow_symbol_health_min_pnl_per_trade_lcb_usd: Optional[float] = None
    shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd: float = 0.0
    shadow_symbol_health_pnl_per_trade_lcb_penalty_weight: float = 0.0
    genetics_confirmation_overlay_enabled: bool = False
    genetics_confirmation_labels: Tuple[str, ...] = ()
    genetics_confirmation_allowed_signal_keys: Tuple[str, ...] = ()
    genetics_confirmation_contra_signal_keys: Tuple[str, ...] = ()
    genetics_confirmation_contra_side_match_enabled: bool = False
    genetics_confirmation_contra_static_enabled: bool = False
    genetics_confirmation_contra_no_backfill_enabled: bool = False
    genetics_confirmation_contra_risk_sizing_enabled: bool = False
    genetics_confirmation_contra_risk_mult: float = 1.0
    genetics_confirmation_quality_gate_enabled: bool = False
    genetics_confirmation_min_closed_trades: int = 0
    genetics_confirmation_min_pnl_per_trade_pct: float = 0.0
    genetics_confirmation_score_bonus: float = 0.0
    genetics_confirmation_score_penalty: float = 0.0
    genetics_confirmation_contra_score_penalty: float = 0.0
    max_signals_per_actor: int = 0
    open_overextension_guard_enabled: bool = False
    overextension_lookback_bars: int = 12
    short_overextension_return_floor_pct: float = -8.0
    long_overextension_return_ceiling_pct: float = 8.0
    overextension_volatility_normalized_enabled: bool = False
    short_overextension_z_floor: float = -2.5
    long_overextension_z_ceiling: float = 2.5
    overextension_min_volatility_pct: float = 0.1
    denied_signal_keys: Tuple[str, ...] = ()
    terminal_denied_signal_keys: Tuple[str, ...] = ()
    terminal_denied_context_signal_keys: Tuple[str, ...] = ()
    denied_open_symbols: Tuple[str, ...] = ()
    denied_open_regimes: Tuple[Regime, ...] = ()
    degradation_guard_enabled: bool = False
    degradation_actor_guard_enabled: bool = False
    degradation_actor_scope: str = "actor"
    degradation_signal_cooldown_bars: int = 0
    degradation_actor_cooldown_bars: int = 0
    degradation_symbol_guard_enabled: bool = False
    degradation_symbol_cooldown_bars: int = 0
    degradation_symbol_lookback_bars: int = 0
    degradation_symbol_window_closed_trades: int = 0
    degradation_symbol_min_closed_trades: int = 0
    degradation_symbol_max_recent_pnl_usd: Optional[float] = None
    degradation_window_closed_trades: int = 3
    degradation_min_closed_trades: int = 3
    degradation_max_recent_pnl_usd: float = -25.0
    degradation_signal_min_pnl_per_trade_lcb_usd: Optional[float] = None
    degradation_pnl_per_trade_lcb_z: float = 1.0
    degradation_signal_risk_sizing_enabled: bool = False
    degradation_signal_risk_mult: float = 0.20
    degradation_reserve_actor_cap: bool = False
    degradation_recovery_enabled: bool = False
    degradation_recovery_min_closed_trades: int = 3
    degradation_recovery_min_recent_pnl_usd: float = 0.0
    promotion_manifest_enabled: bool = False
    promoted_actor_cap_overrides: Tuple[str, ...] = ()
    actor_risk_sizing_enabled: bool = False
    actor_risk_min_mult: float = 0.25
    actor_risk_max_mult: float = 1.0
    actor_risk_edge_scale_pct: float = 0.50
    funding_score_weight: float = 0.0
    funding_risk_mult_weight: float = 0.0
    funding_risk_mult_cap: float = 0.25
    no_trade_fee_saving_score_enabled: bool = False
    no_trade_default_fee_bps: float = 0.0
    volatility_risk_sizing_enabled: bool = False
    volatility_risk_target_pct: float = 2.0
    volatility_risk_min_volatility_pct: float = 0.5
    volatility_risk_max_mult: float = 2.0
    regime_feature_risk_sizing_enabled: bool = True
    regime_feature_risk_min_mult: float = 0.40
    regime_feature_risk_max_pressure: float = 2.50
    mixed_rotational_exposure_cap_enabled: bool = True
    mixed_rotational_max_directional_opens: int = 1
    mixed_rotational_min_symbol_confidence: float = 0.55
    mixed_rotational_symbol_local_shadow_gate_enabled: bool = False
    mixed_rotational_symbol_local_shadow_min_score: float = 4.0
    mixed_rotational_symbol_local_shadow_min_closed_trades: int = 5
    trend_alignment_gate_enabled: bool = True
    counter_trend_min_closed_trades: int = 3
    counter_trend_min_pnl_pct: float = 1.0
    genetics_probation_bypass_min_closed_enabled: bool = False
    genetics_probation_bypass_trend_gate_enabled: bool = False
    genetics_probation_bypass_terminal_deny_enabled: bool = False
    genetics_probation_bypass_regime_edge_enabled: bool = False
    genetics_probation_bypass_pnl_enabled: bool = False
    technical_overlay_enabled: bool = False
    technical_hard_gate_enabled: bool = False
    technical_score_bonus: float = 0.10
    technical_score_penalty: float = 0.25
    technical_rsi_long_min: float = 45.0
    technical_rsi_long_max: float = 72.0
    technical_rsi_short_min: float = 28.0
    technical_rsi_short_max: float = 55.0
    technical_macd_histogram_min_abs_pct: float = 0.0
    technical_atr_risk_sizing_enabled: bool = False
    technical_atr_target_pct: float = 2.0
    technical_atr_min_pct: float = 0.25
    technical_atr_max_mult: float = 1.5
    denied_actor_keys: Tuple[str, ...] = ()
    selected_subset_score_boosts: Tuple[str, ...] = ()
    selected_subset_context_score_boosts: Tuple[str, ...] = ()
    selected_subset_do_not_demote_signal_keys: Tuple[str, ...] = ()
    selected_subset_risk_mult_overrides: Tuple[str, ...] = ()
    selected_subset_context_risk_mult_overrides: Tuple[str, ...] = ()
    selected_subset_risk_min_mult: float = 0.75
    selected_subset_risk_max_mult: float = 1.15

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "denied_signal_keys",
            (
                _normalize_signal_deny_key_set(self.denied_signal_keys)
                if self.denied_signal_keys
                else ()
            ),
        )
        object.__setattr__(
            self,
            "terminal_denied_signal_keys",
            (
                _normalize_signal_deny_key_set(self.terminal_denied_signal_keys)
                if self.terminal_denied_signal_keys
                else ()
            ),
        )
        object.__setattr__(
            self,
            "terminal_denied_context_signal_keys",
            (
                _normalize_signal_context_key_set(
                    self.terminal_denied_context_signal_keys
                )
                if self.terminal_denied_context_signal_keys
                else ()
            ),
        )
        object.__setattr__(
            self,
            "denied_open_regimes",
            _normalize_regime_tuple(self.denied_open_regimes),
        )
        object.__setattr__(
            self,
            "denied_open_symbols",
            _normalize_symbol_tuple(self.denied_open_symbols),
        )
        object.__setattr__(
            self,
            "denied_actor_keys",
            _normalize_actor_key_tuple(self.denied_actor_keys),
        )
        promoted_actor_cap_overrides = tuple(
            str(item).strip()
            for item in self.promoted_actor_cap_overrides
            if str(item or "").strip()
        )
        _actor_cap_override_map(promoted_actor_cap_overrides)
        object.__setattr__(
            self,
            "promoted_actor_cap_overrides",
            promoted_actor_cap_overrides,
        )
        raw_base_fallback_keys = self.shadow_base_fallback_actor_keys
        if isinstance(raw_base_fallback_keys, str):
            base_fallback_keys = tuple(
                item.strip()
                for item in raw_base_fallback_keys.split(",")
                if item.strip()
            )
        else:
            base_fallback_keys = tuple(
                str(item).strip()
                for item in raw_base_fallback_keys
                if str(item or "").strip()
            )
        object.__setattr__(
            self,
            "shadow_base_fallback_actor_keys",
            base_fallback_keys,
        )
        raw_recovery_actor_keys = self.shadow_recovery_actor_keys
        if isinstance(raw_recovery_actor_keys, str):
            recovery_actor_keys = tuple(
                item.strip()
                for item in raw_recovery_actor_keys.split(",")
                if item.strip()
            )
        else:
            recovery_actor_keys = tuple(
                str(item).strip()
                for item in raw_recovery_actor_keys
                if str(item or "").strip()
            )
        object.__setattr__(
            self,
            "shadow_recovery_actor_keys",
            tuple(dict.fromkeys(recovery_actor_keys)),
        )
        raw_genetics_labels = self.genetics_confirmation_labels
        if isinstance(raw_genetics_labels, str):
            genetics_labels = tuple(
                item.strip()
                for item in raw_genetics_labels.split(",")
                if item.strip()
            )
        else:
            genetics_labels = tuple(
                str(item).strip()
                for item in raw_genetics_labels
                if str(item or "").strip()
            )
        object.__setattr__(
            self,
            "genetics_confirmation_labels",
            tuple(dict.fromkeys(genetics_labels)),
        )
        raw_genetics_signal_keys = self.genetics_confirmation_allowed_signal_keys
        if isinstance(raw_genetics_signal_keys, str):
            genetics_signal_keys = tuple(
                item.strip()
                for item in raw_genetics_signal_keys.split(",")
                if item.strip()
            )
        else:
            genetics_signal_keys = tuple(
                str(item).strip()
                for item in raw_genetics_signal_keys
                if str(item or "").strip()
            )
        object.__setattr__(
            self,
            "genetics_confirmation_allowed_signal_keys",
            tuple(
                dict.fromkeys(
                    _normalize_signal_deny_key(key) for key in genetics_signal_keys
                )
            ),
        )
        raw_contra_signal_keys = self.genetics_confirmation_contra_signal_keys
        if isinstance(raw_contra_signal_keys, str):
            contra_signal_keys = tuple(
                item.strip()
                for item in raw_contra_signal_keys.split(",")
                if item.strip()
            )
        else:
            contra_signal_keys = tuple(
                str(item).strip()
                for item in raw_contra_signal_keys
                if str(item or "").strip()
            )
        object.__setattr__(
            self,
            "genetics_confirmation_contra_signal_keys",
            tuple(
                dict.fromkeys(
                    _normalize_signal_deny_key(key) for key in contra_signal_keys
                )
            ),
        )
        raw_anchor_keys = self.anchor_actor_keys
        if isinstance(raw_anchor_keys, str):
            anchor_keys = tuple(
                item.strip()
                for item in raw_anchor_keys.split(",")
                if item.strip()
            )
        else:
            anchor_keys = tuple(
                str(item).strip()
                for item in raw_anchor_keys
                if str(item or "").strip()
            )
        object.__setattr__(self, "anchor_actor_keys", anchor_keys)
        raw_portfolio_keys = self.portfolio_actor_keys
        if isinstance(raw_portfolio_keys, str):
            portfolio_keys = tuple(
                item.strip()
                for item in raw_portfolio_keys.split(",")
                if item.strip()
            )
        else:
            portfolio_keys = tuple(
                str(item).strip()
                for item in raw_portfolio_keys
                if str(item or "").strip()
            )
        object.__setattr__(self, "portfolio_actor_keys", portfolio_keys)
        object.__setattr__(
            self,
            "range_low_vol_real_actor_allowlist",
            _normalize_actor_key_tuple(self.range_low_vol_real_actor_allowlist),
        )
        object.__setattr__(
            self,
            "flash_genetics_core_primary_labels",
            _normalize_actor_key_tuple(self.flash_genetics_core_primary_labels),
        )
        selected_boosts = _normalize_string_tuple(self.selected_subset_score_boosts)
        selected_context_boosts = _normalize_string_tuple(
            self.selected_subset_context_score_boosts
        )
        selected_risk = _normalize_string_tuple(self.selected_subset_risk_mult_overrides)
        selected_context_risk = _normalize_string_tuple(
            self.selected_subset_context_risk_mult_overrides
        )
        selected_boost_map = _signal_key_float_map(
            selected_boosts,
            field_name="selected_subset_score_boosts",
        )
        selected_context_boost_map = _signal_context_key_float_map(
            selected_context_boosts,
            field_name="selected_subset_context_score_boosts",
        )
        selected_risk_map = _signal_key_float_map(
            selected_risk,
            field_name="selected_subset_risk_mult_overrides",
        )
        selected_context_risk_map = _signal_context_key_float_map(
            selected_context_risk,
            field_name="selected_subset_context_risk_mult_overrides",
        )
        if any(value < 0.0 for value in selected_boost_map.values()):
            raise ValueError("selected_subset_score_boosts values must be >= 0")
        if any(value < 0.0 for value in selected_context_boost_map.values()):
            raise ValueError(
                "selected_subset_context_score_boosts values must be >= 0"
            )
        if any(value <= 0.0 for value in selected_risk_map.values()):
            raise ValueError("selected_subset_risk_mult_overrides values must be > 0")
        if any(value <= 0.0 for value in selected_context_risk_map.values()):
            raise ValueError(
                "selected_subset_context_risk_mult_overrides values must be > 0"
            )
        object.__setattr__(self, "selected_subset_score_boosts", selected_boosts)
        object.__setattr__(
            self,
            "selected_subset_context_score_boosts",
            selected_context_boosts,
        )
        object.__setattr__(
            self,
            "selected_subset_do_not_demote_signal_keys",
            _normalize_signal_key_tuple(
                self.selected_subset_do_not_demote_signal_keys,
                field_name="selected_subset_do_not_demote_signal_keys",
            ),
        )
        object.__setattr__(self, "selected_subset_risk_mult_overrides", selected_risk)
        object.__setattr__(
            self,
            "selected_subset_context_risk_mult_overrides",
            selected_context_risk,
        )
        if self.actionable_bonus < 0:
            raise ValueError("actionable_bonus must be >= 0")
        if self.min_closed_trades_to_trade < 0:
            raise ValueError("min_closed_trades_to_trade must be >= 0")
        if not 0.0 <= self.component_score_max_bias <= 1.0:
            raise ValueError("component_score_max_bias must be in [0, 1]")
        if self.global_health_min_closed_trades < 0:
            raise ValueError("global_health_min_closed_trades must be >= 0")
        if self.regime_edge_min_closed_trades < 0:
            raise ValueError("regime_edge_min_closed_trades must be >= 0")
        if self.real_loss_gate_min_closed_trades < 0:
            raise ValueError("real_loss_gate_min_closed_trades must be >= 0")
        if self.churn_cooldown_bars < 0:
            raise ValueError("churn_cooldown_bars must be >= 0")
        if self.churn_max_opens_per_day < 0:
            raise ValueError("churn_max_opens_per_day must be >= 0")
        if self.fee_aware_round_trip_cost_pct < 0:
            raise ValueError("fee_aware_round_trip_cost_pct must be >= 0")
        if self.shadow_confirmation_min_closed_trades < 0:
            raise ValueError("shadow_confirmation_min_closed_trades must be >= 0")
        if self.shadow_actor_fallback_min_base_score < 0:
            raise ValueError("shadow_actor_fallback_min_base_score must be >= 0")
        if self.shadow_position_replay_actor_fallback_min_base_score < 0:
            raise ValueError(
                "shadow_position_replay_actor_fallback_min_base_score must be >= 0"
            )
        if self.shadow_position_replay_actor_fallback_min_shadow_score < 0:
            raise ValueError(
                "shadow_position_replay_actor_fallback_min_shadow_score must be >= 0"
            )
        if self.actor_switch_margin < 0:
            raise ValueError("actor_switch_margin must be >= 0")
        if self.anchor_min_score_advantage < 0:
            raise ValueError("anchor_min_score_advantage must be >= 0")
        if self.proven_solo_min_score_advantage < 0:
            raise ValueError("proven_solo_min_score_advantage must be >= 0")
        if self.shadow_confirmation_min_full_open_closed_trades < 0:
            raise ValueError("shadow_confirmation_min_full_open_closed_trades must be >= 0")
        if self.shadow_confirmation_min_win_rate_pct < 0:
            raise ValueError("shadow_confirmation_min_win_rate_pct must be >= 0")
        if self.shadow_confirmation_max_recent_downside_usd < 0:
            raise ValueError("shadow_confirmation_max_recent_downside_usd must be >= 0")
        if self.shadow_confirmation_pnl_per_trade_lcb_z < 0:
            raise ValueError("shadow_confirmation_pnl_per_trade_lcb_z must be >= 0")
        if self.shadow_confirmation_pnl_per_trade_lcb_penalty_weight < 0:
            raise ValueError(
                "shadow_confirmation_pnl_per_trade_lcb_penalty_weight must be >= 0"
            )
        if self.shadow_confirmation_pnl_per_trade_lcb_risk_min_mult < 0:
            raise ValueError(
                "shadow_confirmation_pnl_per_trade_lcb_risk_min_mult must be >= 0"
            )
        if self.shadow_confirmation_pnl_per_trade_lcb_risk_min_mult > 1.0:
            raise ValueError(
                "shadow_confirmation_pnl_per_trade_lcb_risk_min_mult must be <= 1"
            )
        if self.shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd <= 0:
            raise ValueError(
                "shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd must be > 0"
            )
        if self.shadow_recovery_min_score is not None and self.shadow_recovery_min_score < 0:
            raise ValueError("shadow_recovery_min_score must be >= 0")
        if self.shadow_recovery_min_closed_trades < 0:
            raise ValueError("shadow_recovery_min_closed_trades must be >= 0")
        if self.shadow_symbol_health_min_closed_trades < 0:
            raise ValueError("shadow_symbol_health_min_closed_trades must be >= 0")
        if self.shadow_symbol_health_pnl_per_trade_lcb_penalty_weight < 0:
            raise ValueError(
                "shadow_symbol_health_pnl_per_trade_lcb_penalty_weight must be >= 0"
            )
        if self.genetics_confirmation_score_bonus < 0:
            raise ValueError("genetics_confirmation_score_bonus must be >= 0")
        if self.genetics_confirmation_score_penalty < 0:
            raise ValueError("genetics_confirmation_score_penalty must be >= 0")
        if self.genetics_confirmation_contra_score_penalty < 0:
            raise ValueError(
                "genetics_confirmation_contra_score_penalty must be >= 0"
            )
        if self.genetics_confirmation_contra_risk_mult < 0:
            raise ValueError("genetics_confirmation_contra_risk_mult must be >= 0")
        if self.genetics_confirmation_min_closed_trades < 0:
            raise ValueError("genetics_confirmation_min_closed_trades must be >= 0")
        if self.max_signals_per_actor < 0:
            raise ValueError("max_signals_per_actor must be >= 0")
        if self.overextension_lookback_bars <= 0:
            raise ValueError("overextension_lookback_bars must be > 0")
        if self.overextension_min_volatility_pct <= 0:
            raise ValueError("overextension_min_volatility_pct must be > 0")
        if self.short_overextension_z_floor >= 0:
            raise ValueError("short_overextension_z_floor must be < 0")
        if self.long_overextension_z_ceiling <= 0:
            raise ValueError("long_overextension_z_ceiling must be > 0")
        if self.degradation_window_closed_trades <= 0:
            raise ValueError("degradation_window_closed_trades must be > 0")
        if self.degradation_min_closed_trades <= 0:
            raise ValueError("degradation_min_closed_trades must be > 0")
        if self.degradation_actor_scope not in {"actor", "actor_regime"}:
            raise ValueError("degradation_actor_scope must be 'actor' or 'actor_regime'")
        if self.degradation_signal_cooldown_bars < 0:
            raise ValueError("degradation_signal_cooldown_bars must be >= 0")
        if self.degradation_actor_cooldown_bars < 0:
            raise ValueError("degradation_actor_cooldown_bars must be >= 0")
        if self.degradation_symbol_lookback_bars < 0:
            raise ValueError("degradation_symbol_lookback_bars must be >= 0")
        if self.degradation_min_closed_trades > self.degradation_window_closed_trades:
            raise ValueError(
                "degradation_min_closed_trades must be <= degradation_window_closed_trades"
            )
        if self.degradation_pnl_per_trade_lcb_z < 0:
            raise ValueError("degradation_pnl_per_trade_lcb_z must be >= 0")
        if self.degradation_signal_risk_mult < 0:
            raise ValueError("degradation_signal_risk_mult must be >= 0")
        if self.degradation_signal_risk_mult > 1.0:
            raise ValueError("degradation_signal_risk_mult must be <= 1")
        if self.degradation_recovery_enabled:
            if self.degradation_recovery_min_closed_trades <= 0:
                raise ValueError("degradation_recovery_min_closed_trades must be > 0")
            if self.degradation_recovery_min_closed_trades > self.degradation_window_closed_trades:
                raise ValueError(
                    "degradation_recovery_min_closed_trades must be <= degradation_window_closed_trades"
                )
        if self.actor_risk_min_mult < 0:
            raise ValueError("actor_risk_min_mult must be >= 0")
        if self.actor_risk_max_mult <= 0:
            raise ValueError("actor_risk_max_mult must be > 0")
        if self.actor_risk_min_mult > self.actor_risk_max_mult:
            raise ValueError("actor_risk_min_mult must be <= actor_risk_max_mult")
        if self.actor_risk_edge_scale_pct <= 0:
            raise ValueError("actor_risk_edge_scale_pct must be > 0")
        if self.funding_score_weight < 0:
            raise ValueError("funding_score_weight must be >= 0")
        if self.funding_risk_mult_weight < 0:
            raise ValueError("funding_risk_mult_weight must be >= 0")
        if self.funding_risk_mult_cap < 0:
            raise ValueError("funding_risk_mult_cap must be >= 0")
        if self.no_trade_default_fee_bps < 0:
            raise ValueError("no_trade_default_fee_bps must be >= 0")
        if self.volatility_risk_target_pct <= 0:
            raise ValueError("volatility_risk_target_pct must be > 0")
        if self.volatility_risk_min_volatility_pct <= 0:
            raise ValueError("volatility_risk_min_volatility_pct must be > 0")
        if self.volatility_risk_max_mult <= 0:
            raise ValueError("volatility_risk_max_mult must be > 0")
        if self.regime_feature_risk_min_mult < 0:
            raise ValueError("regime_feature_risk_min_mult must be >= 0")
        if self.regime_feature_risk_max_pressure <= 0:
            raise ValueError("regime_feature_risk_max_pressure must be > 0")
        if self.mixed_rotational_max_directional_opens < 0:
            raise ValueError("mixed_rotational_max_directional_opens must be >= 0")
        if not 0.0 <= self.mixed_rotational_min_symbol_confidence <= 1.0:
            raise ValueError("mixed_rotational_min_symbol_confidence must be in [0, 1]")
        if self.mixed_rotational_symbol_local_shadow_min_score < 0.0:
            raise ValueError(
                "mixed_rotational_symbol_local_shadow_min_score must be >= 0"
            )
        if self.mixed_rotational_symbol_local_shadow_min_closed_trades < 0:
            raise ValueError(
                "mixed_rotational_symbol_local_shadow_min_closed_trades must be >= 0"
            )
        if self.counter_trend_min_closed_trades < 0:
            raise ValueError("counter_trend_min_closed_trades must be >= 0")
        if self.counter_trend_min_pnl_pct < 0:
            raise ValueError("counter_trend_min_pnl_pct must be >= 0")
        if self.technical_score_bonus < 0.0:
            raise ValueError("technical_score_bonus must be >= 0")
        if self.technical_score_penalty < 0.0:
            raise ValueError("technical_score_penalty must be >= 0")
        if not 0.0 <= self.technical_rsi_long_min <= self.technical_rsi_long_max <= 100.0:
            raise ValueError("technical RSI long bounds must be ordered within [0, 100]")
        if not 0.0 <= self.technical_rsi_short_min <= self.technical_rsi_short_max <= 100.0:
            raise ValueError("technical RSI short bounds must be ordered within [0, 100]")
        if self.technical_macd_histogram_min_abs_pct < 0.0:
            raise ValueError("technical_macd_histogram_min_abs_pct must be >= 0")
        if self.technical_atr_target_pct <= 0.0:
            raise ValueError("technical_atr_target_pct must be > 0")
        if self.technical_atr_min_pct <= 0.0:
            raise ValueError("technical_atr_min_pct must be > 0")
        if self.technical_atr_max_mult <= 0.0:
            raise ValueError("technical_atr_max_mult must be > 0")
        if self.selected_subset_risk_min_mult < 0:
            raise ValueError("selected_subset_risk_min_mult must be >= 0")
        if self.selected_subset_risk_max_mult <= 0:
            raise ValueError("selected_subset_risk_max_mult must be > 0")
        if self.selected_subset_risk_min_mult > self.selected_subset_risk_max_mult:
            raise ValueError(
                "selected_subset_risk_min_mult must be <= selected_subset_risk_max_mult"
            )


FLASH_EXPERIMENTAL_FLAGS: Tuple[str, ...] = (
    "shadow_confirmation_enabled",
    "shadow_symbol_confirmation_enabled",
    "shadow_actor_fallback_confirmation_enabled",
    "shadow_base_fallback_confirmation_enabled",
    "shadow_signal_handoff_enabled",
    "shadow_quality_confirmation_enabled",
    "genetics_confirmation_overlay_enabled",
    "genetics_confirmation_quality_gate_enabled",
    "genetics_confirmation_contra_no_backfill_enabled",
    "shadow_symbol_health_enabled",
    "shadow_recovery_admission_enabled",
    "prefer_solo_player_wrappers_enabled",
    "prefer_proven_solo_player_wrappers_enabled",
    "portfolio_shadow_bootstrap_min_closed_enabled",
    "open_overextension_guard_enabled",
    "overextension_volatility_normalized_enabled",
    "degradation_guard_enabled",
    "degradation_actor_guard_enabled",
    "degradation_symbol_guard_enabled",
    "degradation_signal_risk_sizing_enabled",
    "degradation_reserve_actor_cap",
    "degradation_recovery_enabled",
    "promotion_manifest_enabled",
    "actor_risk_sizing_enabled",
    "shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled",
    "no_trade_fee_saving_score_enabled",
    "volatility_risk_sizing_enabled",
    "technical_overlay_enabled",
    "technical_hard_gate_enabled",
    "technical_atr_risk_sizing_enabled",
    "genetics_probation_bypass_min_closed_enabled",
    "genetics_probation_bypass_trend_gate_enabled",
    "genetics_probation_bypass_terminal_deny_enabled",
    "genetics_probation_bypass_regime_edge_enabled",
    "genetics_probation_bypass_pnl_enabled",
    "flash_genetics_core_primary_enabled",
)

FLASH_PRESET_SAFE_PINNED_VALUES: dict[str, object] = {
    "min_score_to_trade": 0.10,
    "min_closed_trades_to_trade": 10,
    "min_pnl_pct_to_trade": 0.10,
    "actor_switch_margin": 0.25,
    "anchor_actor_keys": (),
    "portfolio_actor_keys": (),
    "range_low_vol_real_actor_allowlist": (),
    "flash_genetics_core_primary_enabled": False,
    "flash_genetics_core_primary_labels": ("GeneticsCore",),
    "shadow_confirmation_enabled": False,
    "shadow_symbol_confirmation_enabled": False,
    "shadow_actor_fallback_confirmation_enabled": False,
    "shadow_base_fallback_confirmation_enabled": False,
    "shadow_signal_handoff_enabled": False,
    "shadow_position_replay_actor_fallback_min_base_score": 0.0,
    "shadow_position_replay_actor_fallback_min_shadow_score": 0.0,
    "shadow_quality_confirmation_enabled": False,
    "shadow_confirmation_min_pnl_per_trade_lcb_usd": None,
    "shadow_confirmation_pnl_per_trade_lcb_penalty_weight": 0.0,
    "shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled": False,
    "shadow_confirmation_pnl_per_trade_lcb_risk_min_mult": 0.25,
    "shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd": 0.0,
    "shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd": 1.0,
    "shadow_recovery_admission_enabled": False,
    "shadow_recovery_actor_keys": (),
    "shadow_recovery_min_score": None,
    "shadow_recovery_min_closed_trades": 0,
    "shadow_symbol_health_enabled": False,
    "shadow_symbol_health_min_closed_trades": 0,
    "shadow_symbol_health_min_pnl_per_trade_lcb_usd": None,
    "shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd": 0.0,
    "shadow_symbol_health_pnl_per_trade_lcb_penalty_weight": 0.0,
    "genetics_confirmation_overlay_enabled": False,
    "genetics_confirmation_labels": (),
    "genetics_confirmation_allowed_signal_keys": (),
    "genetics_confirmation_contra_signal_keys": (),
    "genetics_confirmation_contra_side_match_enabled": False,
    "genetics_confirmation_contra_static_enabled": False,
    "genetics_confirmation_contra_no_backfill_enabled": False,
    "genetics_confirmation_contra_risk_sizing_enabled": False,
    "genetics_confirmation_contra_risk_mult": 1.0,
    "genetics_confirmation_quality_gate_enabled": False,
    "genetics_confirmation_min_closed_trades": 0,
    "genetics_confirmation_min_pnl_per_trade_pct": 0.0,
    "genetics_confirmation_score_bonus": 0.0,
    "genetics_confirmation_score_penalty": 0.0,
    "genetics_confirmation_contra_score_penalty": 0.0,
    "prefer_solo_player_wrappers_enabled": False,
    "prefer_proven_solo_player_wrappers_enabled": False,
    "portfolio_shadow_bootstrap_min_closed_enabled": False,
    "max_signals_per_actor": 1,
    "open_overextension_guard_enabled": False,
    "overextension_lookback_bars": 12,
    "overextension_volatility_normalized_enabled": False,
    "denied_signal_keys": (),
    "terminal_denied_signal_keys": (),
    "denied_open_symbols": (),
    "denied_open_regimes": (),
    "degradation_guard_enabled": False,
    "degradation_actor_guard_enabled": False,
    "degradation_signal_cooldown_bars": 0,
    "degradation_actor_cooldown_bars": 0,
    "degradation_symbol_guard_enabled": False,
    "degradation_symbol_cooldown_bars": 0,
    "degradation_symbol_lookback_bars": 0,
    "degradation_symbol_window_closed_trades": 0,
    "degradation_symbol_min_closed_trades": 0,
    "degradation_symbol_max_recent_pnl_usd": None,
    "degradation_window_closed_trades": 3,
    "degradation_min_closed_trades": 3,
    "degradation_max_recent_pnl_usd": -25.0,
    "degradation_signal_min_pnl_per_trade_lcb_usd": None,
    "degradation_pnl_per_trade_lcb_z": 1.0,
    "degradation_signal_risk_sizing_enabled": False,
    "degradation_signal_risk_mult": 0.20,
    "degradation_reserve_actor_cap": False,
    "degradation_recovery_enabled": False,
    "promotion_manifest_enabled": False,
    "promoted_actor_cap_overrides": (),
    "actor_risk_sizing_enabled": False,
    "actor_risk_min_mult": 0.25,
    "actor_risk_max_mult": 1.0,
    "actor_risk_edge_scale_pct": 0.50,
    "funding_score_weight": 0.0,
    "funding_risk_mult_weight": 0.0,
    "funding_risk_mult_cap": 0.25,
    "no_trade_fee_saving_score_enabled": False,
    "no_trade_default_fee_bps": 0.0,
    "volatility_risk_sizing_enabled": False,
    "volatility_risk_target_pct": 2.0,
    "volatility_risk_min_volatility_pct": 0.5,
    "volatility_risk_max_mult": 2.0,
    "regime_feature_risk_sizing_enabled": True,
    "regime_feature_risk_min_mult": 0.40,
    "regime_feature_risk_max_pressure": 2.50,
    "mixed_rotational_exposure_cap_enabled": True,
    "mixed_rotational_max_directional_opens": 1,
    "mixed_rotational_min_symbol_confidence": 0.55,
    "mixed_rotational_symbol_local_shadow_gate_enabled": False,
    "mixed_rotational_symbol_local_shadow_min_score": 4.0,
    "mixed_rotational_symbol_local_shadow_min_closed_trades": 5,
    "trend_alignment_gate_enabled": True,
    "counter_trend_min_closed_trades": 3,
    "counter_trend_min_pnl_pct": 1.0,
    "genetics_probation_bypass_min_closed_enabled": False,
    "genetics_probation_bypass_trend_gate_enabled": False,
    "genetics_probation_bypass_terminal_deny_enabled": False,
    "genetics_probation_bypass_regime_edge_enabled": False,
    "genetics_probation_bypass_pnl_enabled": False,
    "technical_overlay_enabled": False,
    "technical_hard_gate_enabled": False,
    "technical_score_bonus": 0.10,
    "technical_score_penalty": 0.25,
    "technical_rsi_long_min": 45.0,
    "technical_rsi_long_max": 72.0,
    "technical_rsi_short_min": 28.0,
    "technical_rsi_short_max": 55.0,
    "technical_macd_histogram_min_abs_pct": 0.0,
    "technical_atr_risk_sizing_enabled": False,
    "technical_atr_target_pct": 2.0,
    "technical_atr_min_pct": 0.25,
    "technical_atr_max_mult": 1.5,
    "selected_subset_score_boosts": (),
    "selected_subset_context_score_boosts": (),
    "selected_subset_do_not_demote_signal_keys": (),
    "selected_subset_risk_mult_overrides": (),
    "selected_subset_context_risk_mult_overrides": (),
    "selected_subset_risk_min_mult": 0.75,
    "selected_subset_risk_max_mult": 1.15,
}

FLASH_PRESET_SAFE = FlashAllocatorConfig(**FLASH_PRESET_SAFE_PINNED_VALUES)
FLASH_PRESET_DEFAULT = FlashAllocatorConfig()
FLASH_PRESET_AGGRESSIVE = FlashAllocatorConfig(
    shadow_confirmation_enabled=True,
    shadow_symbol_confirmation_enabled=True,
    shadow_actor_fallback_confirmation_enabled=True,
    shadow_base_fallback_confirmation_enabled=True,
    shadow_signal_handoff_enabled=True,
    shadow_quality_confirmation_enabled=True,
    prefer_solo_player_wrappers_enabled=True,
    prefer_proven_solo_player_wrappers_enabled=True,
    portfolio_shadow_bootstrap_min_closed_enabled=True,
    shadow_symbol_health_enabled=True,
    open_overextension_guard_enabled=True,
    overextension_volatility_normalized_enabled=True,
    degradation_guard_enabled=True,
    degradation_actor_guard_enabled=True,
    degradation_symbol_guard_enabled=True,
    degradation_signal_risk_sizing_enabled=True,
    degradation_reserve_actor_cap=True,
    degradation_recovery_enabled=True,
    promotion_manifest_enabled=True,
    actor_risk_sizing_enabled=True,
    shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled=True,
    shadow_recovery_admission_enabled=True,
    genetics_confirmation_overlay_enabled=True,
    genetics_confirmation_quality_gate_enabled=True,
    genetics_confirmation_contra_no_backfill_enabled=True,
    no_trade_fee_saving_score_enabled=True,
    volatility_risk_sizing_enabled=True,
    technical_overlay_enabled=True,
    technical_hard_gate_enabled=True,
    technical_atr_risk_sizing_enabled=True,
    genetics_probation_bypass_min_closed_enabled=True,
    genetics_probation_bypass_trend_gate_enabled=True,
    genetics_probation_bypass_terminal_deny_enabled=True,
    genetics_probation_bypass_regime_edge_enabled=True,
    genetics_probation_bypass_pnl_enabled=True,
    flash_genetics_core_primary_enabled=True,
    funding_score_weight=1.0,
    funding_risk_mult_weight=1.0,
    no_trade_default_fee_bps=5.0,
)


@dataclass(frozen=True)
class FlashCandidateAudit:
    symbol: str
    label: str
    actor_type: str
    score: float
    action: Action
    base_score: float = 0.0
    effective_score: float = 0.0
    gate_score: float = 0.0
    # Стабильный per-regime эдж (pnl_per_trade по всем символам режима) — для
    # stable-pool ранжирования вместо волатильного per-(symbol,regime) score.
    stable_edge: float = 0.0
    rank: int = 0
    rejected: bool = False
    reason: str = ""
    has_data: bool = False
    closed_trades: int = 0
    agent_labels: Tuple[str, ...] = ()
    pnl_net_pct: float = 0.0
    pnl_gross_pct: float = 0.0
    fee_pct: float = 0.0
    funding_pct: float = 0.0
    trading_cost_pct: float = 0.0
    shadow_score: float = 0.0
    shadow_closed_trades: int = 0
    shadow_winning_trades: int = 0
    shadow_win_rate_pct: float = 0.0
    shadow_recent_downside_usd: float = 0.0
    shadow_pnl_per_trade_mean_usd: Optional[float] = None
    shadow_pnl_per_trade_std_usd: Optional[float] = None
    shadow_pnl_per_trade_lcb_usd: Optional[float] = None
    shadow_pnl_per_trade_lcb_penalty: float = 0.0
    shadow_source: str = ""
    shadow_symbol_health_score: float = 0.0
    shadow_symbol_health_closed_trades: int = 0
    shadow_symbol_health_pnl_per_trade_lcb_usd: Optional[float] = None
    shadow_symbol_health_penalty: float = 0.0
    genetics_confirmation_adjustment: float = 0.0
    funding_score_adjustment: float = 0.0
    selected_subset_score_boost: float = 0.0
    selected_subset_protected: bool = False
    selected_subset_risk_mult: float = 1.0
    risk_mult: float = 1.0
    lookback_return_pct: Optional[float] = None
    lookback_volatility_pct: Optional[float] = None
    lookback_return_z: Optional[float] = None
    technical_rsi_14: Optional[float] = None
    technical_macd_histogram_pct: Optional[float] = None
    technical_atr_14_pct: Optional[float] = None
    technical_alignment: str = ""
    technical_score_adjustment: float = 0.0
    technical_gate_reason: str = ""
    trend_alignment: str = ""
    trend_gate_reason: str = ""
    regime_feature_risk_mult: float = 1.0
    actor_key: str = ""

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "label": self.label,
            "actor_type": self.actor_type,
            "actor_key": self.actor_key or _actor_key(self.actor_type, self.label),
            "score": float(self.score),
            "base_score": float(self.base_score),
            "effective_score": float(self.effective_score),
            "gate_score": float(self.gate_score),
            "action": self.action.name,
            "rank": int(self.rank),
            "rejected": bool(self.rejected),
            "reason": self.reason,
            "has_data": bool(self.has_data),
            "closed_trades": int(self.closed_trades),
            "agent_labels": list(self.agent_labels),
            "pnl_net_pct": float(self.pnl_net_pct),
            "pnl_gross_pct": float(self.pnl_gross_pct),
            "fee_pct": float(self.fee_pct),
            "funding_pct": float(self.funding_pct),
            "trading_cost_pct": float(self.trading_cost_pct),
            "shadow_score": float(self.shadow_score),
            "shadow_closed_trades": int(self.shadow_closed_trades),
            "shadow_winning_trades": int(self.shadow_winning_trades),
            "shadow_win_rate_pct": float(self.shadow_win_rate_pct),
            "shadow_recent_downside_usd": float(self.shadow_recent_downside_usd),
            "shadow_pnl_per_trade_mean_usd": self.shadow_pnl_per_trade_mean_usd,
            "shadow_pnl_per_trade_std_usd": self.shadow_pnl_per_trade_std_usd,
            "shadow_pnl_per_trade_lcb_usd": self.shadow_pnl_per_trade_lcb_usd,
            "shadow_pnl_per_trade_lcb_penalty": float(
                self.shadow_pnl_per_trade_lcb_penalty
            ),
            "shadow_source": self.shadow_source,
            "shadow_symbol_health_score": float(self.shadow_symbol_health_score),
            "shadow_symbol_health_closed_trades": int(
                self.shadow_symbol_health_closed_trades
            ),
            "shadow_symbol_health_pnl_per_trade_lcb_usd": (
                self.shadow_symbol_health_pnl_per_trade_lcb_usd
            ),
            "shadow_symbol_health_penalty": float(
                self.shadow_symbol_health_penalty
            ),
            "genetics_confirmation_adjustment": float(
                self.genetics_confirmation_adjustment
            ),
            "funding_score_adjustment": float(self.funding_score_adjustment),
            "selected_subset_score_boost": float(self.selected_subset_score_boost),
            "selected_subset_protected": bool(self.selected_subset_protected),
            "selected_subset_risk_mult": float(self.selected_subset_risk_mult),
            "risk_mult": float(self.risk_mult),
            "lookback_return_pct": (
                None if self.lookback_return_pct is None else float(self.lookback_return_pct)
            ),
            "lookback_volatility_pct": (
                None
                if self.lookback_volatility_pct is None
                else float(self.lookback_volatility_pct)
            ),
            "lookback_return_z": (
                None if self.lookback_return_z is None else float(self.lookback_return_z)
            ),
            "technical_rsi_14": (
                None if self.technical_rsi_14 is None else float(self.technical_rsi_14)
            ),
            "technical_macd_histogram_pct": (
                None
                if self.technical_macd_histogram_pct is None
                else float(self.technical_macd_histogram_pct)
            ),
            "technical_atr_14_pct": (
                None
                if self.technical_atr_14_pct is None
                else float(self.technical_atr_14_pct)
            ),
            "technical_alignment": self.technical_alignment,
            "technical_score_adjustment": float(self.technical_score_adjustment),
            "technical_gate_reason": self.technical_gate_reason,
            "trend_alignment": self.trend_alignment,
            "trend_gate_reason": self.trend_gate_reason,
            "regime_feature_risk_mult": float(self.regime_feature_risk_mult),
        }


@dataclass(frozen=True)
class FlashDecision:
    symbol: str
    selected_actor: str
    actor_type: str
    score: float
    action: Action
    reason: str
    signal: Optional[Signal]
    candidates: Tuple[FlashCandidateAudit, ...]
    original_selected_actor: str = ""
    original_actor_type: str = ""
    selected_reasons: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "selected_actor": self.selected_actor,
            "actor_type": self.actor_type,
            "original_selected_actor": self.original_selected_actor,
            "original_actor_type": self.original_actor_type,
            "score": float(self.score),
            "action": self.action.name,
            "reason": self.reason,
            "selected_reasons": list(self.selected_reasons),
            "signal": _signal_payload(self.signal),
            "candidates": [row.as_dict() for row in self.candidates],
        }


@dataclass(frozen=True)
class _ActorSignal:
    symbol: str
    label: str
    actor_type: str
    actor_key: str
    action: Action
    signal: Optional[Signal]
    agent_labels: Tuple[str, ...]


@dataclass(frozen=True)
class _ShadowConfirmation:
    score: float = 0.0
    closed_trades: int = 0
    winning_trades: int = 0
    win_rate_pct: float = 0.0
    recent_downside_usd: float = 0.0
    pnl_per_trade_mean_usd: Optional[float] = None
    pnl_per_trade_std_usd: Optional[float] = None
    pnl_per_trade_lcb_usd: Optional[float] = None

    def pnl_per_trade_lcb(self, *, z: float = 1.0) -> Optional[float]:
        if self.pnl_per_trade_mean_usd is not None:
            closed = max(1, int(self.closed_trades or 0))
            std = max(0.0, float(self.pnl_per_trade_std_usd or 0.0))
            return float(self.pnl_per_trade_mean_usd) - (
                max(0.0, float(z)) * std / math.sqrt(float(closed))
            )
        return self.pnl_per_trade_lcb_usd


class FlashAllocator:
    """Select one best actor independently for each market symbol."""

    def __init__(
        self,
        *,
        perf: PerformanceMemory,
        qm: QuarantineManager,
        config: FlashAllocatorConfig = FlashAllocatorConfig(),
        scoring_config: ScoringConfig = DEFAULT_SCORING,
        scoring_per_regime: Optional[Mapping[Regime, ScoringConfig]] = None,
        real_perf: Optional[PerformanceMemory] = None,
    ) -> None:
        self._perf = perf
        # Live-audit fix: реальная память для real-aware гейтинга (shadow лжёт).
        self._real_perf = real_perf
        # Churn-limiter state: последний бар открытия и сторона по символу.
        self._last_open_bar_by_symbol: Dict[str, int] = {}
        self._last_open_side_by_symbol: Dict[str, str] = {}
        self._opens_today_count: int = 0
        self._opens_today_day: int = -1
        self._qm = qm
        self._config = config
        self._scoring = scoring_config
        self._scoring_per_regime = dict(scoring_per_regime or {})
        self._selected_subset_score_boosts = _signal_key_float_map(
            self._config.selected_subset_score_boosts,
            field_name="selected_subset_score_boosts",
        )
        self._selected_subset_context_score_boosts = _signal_context_key_float_map(
            self._config.selected_subset_context_score_boosts,
            field_name="selected_subset_context_score_boosts",
        )
        self._selected_subset_risk_mult_overrides = _signal_key_float_map(
            self._config.selected_subset_risk_mult_overrides,
            field_name="selected_subset_risk_mult_overrides",
        )
        self._selected_subset_context_risk_mult_overrides = (
            _signal_context_key_float_map(
                self._config.selected_subset_context_risk_mult_overrides,
                field_name="selected_subset_context_risk_mult_overrides",
            )
        )
        self._selected_subset_do_not_demote_signal_keys = set(
            self._config.selected_subset_do_not_demote_signal_keys
        )

    @property
    def config(self) -> FlashAllocatorConfig:
        return self._config

    def decide(
        self,
        market: MarketSnapshot,
        *,
        agents: Sequence[Agent],
        players: Sequence[Player],
        signal_id_start: int,
        actionable_labels: Optional[Iterable[str]] = None,
        shadow_confirmation: Optional[Mapping[object, object]] = None,
        shadow_player_signals: Optional[Mapping[str, Sequence[Signal]]] = None,
        shadow_agent_signals: Optional[Mapping[str, Sequence[Signal]]] = None,
        degraded_signal_keys: Optional[Iterable[str]] = None,
        degraded_actor_keys: Optional[Iterable[str]] = None,
        degraded_open_symbols: Optional[Iterable[str]] = None,
        promoted_signal_keys: Optional[Iterable[str]] = None,
        probation_signal_keys: Optional[Iterable[str]] = None,
        previous_actor_by_symbol: Optional[Mapping[str, str]] = None,
        open_position_sides_by_symbol: Optional[Mapping[str, str]] = None,
        open_position_actor_keys_by_symbol: Optional[Mapping[str, Iterable[str] | str]] = None,
    ) -> Tuple[FlashDecision, ...]:
        actionable = {
            str(label).strip()
            for label in (actionable_labels or ())
            if str(label).strip()
        }
        degraded = {
            _normalize_signal_deny_key(key)
            for key in (degraded_signal_keys or ())
            if str(key or "").strip()
        }
        degraded_actors = {
            str(key).strip()
            for key in (degraded_actor_keys or ())
            if str(key or "").strip()
        }
        degraded_symbols = set(_normalize_symbol_tuple(degraded_open_symbols or ()))
        position_state_available = open_position_sides_by_symbol is not None
        open_position_sides = {
            str(symbol).upper(): str(side).strip().lower()
            for symbol, side in (open_position_sides_by_symbol or {}).items()
            if str(symbol or "").strip() and str(side or "").strip()
        }
        open_position_actor_keys = {
            str(symbol).upper(): frozenset(_normalize_string_tuple(actor_keys))
            for symbol, actor_keys in (open_position_actor_keys_by_symbol or {}).items()
            if str(symbol or "").strip()
        }
        promoted = (
            {
                _normalize_signal_deny_key(key)
                for key in (promoted_signal_keys or ())
                if str(key or "").strip()
            }
            if self._config.promotion_manifest_enabled
            else set()
        )
        probation = {
            _normalize_signal_deny_key(key)
            for key in (probation_signal_keys or ())
            if str(key or "").strip()
        }
        agent_rows = self._agent_outputs(market, agents)
        player_rows = self._player_outputs(market, players, signal_id_start=signal_id_start)
        known_rows = tuple(agent_rows + player_rows)
        shadow_player_signal_map = shadow_player_signals or {}
        shadow_agent_signal_map = shadow_agent_signals or {}
        suppressed_agent_labels = self._suppressed_agent_labels_for_solo_players(
            player_rows,
            regime=market.regime,
            shadow_player_labels=shadow_player_signal_map.keys(),
            actionable_labels=actionable,
        )
        active_solo_wrapper_keys = self._active_solo_wrapper_keys_for_suppression(
            player_rows,
            market=market,
            shadow_player_signals=shadow_player_signal_map,
        )
        base_rows = agent_rows + player_rows
        no_trade_rows = (
            []
            if any(row.actor_key == "NoTrade" for row in base_rows)
            else self._no_trade_outputs(market)
        )
        outputs = tuple(
            base_rows
            + no_trade_rows
            + self._shadow_signal_outputs(
                market,
                base_rows + no_trade_rows,
                known_outputs=known_rows,
                shadow_player_signals=shadow_player_signal_map,
                shadow_agent_signals=shadow_agent_signal_map,
                suppressed_agent_labels=suppressed_agent_labels,
                active_solo_wrapper_keys=active_solo_wrapper_keys,
            )
        )
        genetics_confirmation_by_symbol = self._genetics_confirmation_by_symbol(
            market,
            shadow_player_signal_map,
            shadow_agent_signal_map,
            outputs=outputs,
        )

        decisions: List[FlashDecision] = []
        next_signal_id = int(signal_id_start)
        for symbol in sorted(str(sym).upper() for sym in market.prices):
            decision = self._decide_symbol(
                market,
                symbol,
                outputs,
                actionable=actionable,
                signal_id=next_signal_id,
                shadow_confirmation=shadow_confirmation or {},
                degraded_signal_keys=degraded,
                degraded_actor_keys=degraded_actors,
                degraded_open_symbols=degraded_symbols,
                promoted_signal_keys=promoted,
                probation_signal_keys=probation,
                previous_actor_by_symbol=previous_actor_by_symbol or {},
                suppressed_agent_labels=suppressed_agent_labels,
                active_solo_wrapper_keys=active_solo_wrapper_keys,
                open_position_sides_by_symbol=open_position_sides,
                open_position_actor_keys_by_symbol=open_position_actor_keys,
                position_state_available=position_state_available,
                genetics_confirmation_by_symbol=genetics_confirmation_by_symbol,
            )
            if decision.signal is not None:
                next_signal_id += 1
                # Churn-limiter: учитываем фактическое открытие по символу.
                sig_action = getattr(decision.signal, "action", None)
                if sig_action is not None and getattr(sig_action, "is_open", False):
                    side = "long" if getattr(sig_action, "is_long_open", False) else "short"
                    self._churn_record_open(symbol, side, market.bar)
            decisions.append(decision)
        capped = self._apply_mixed_rotational_exposure_cap(market, tuple(decisions))
        return self._apply_actor_signal_cap(capped)

    def _decide_symbol(
        self,
        market: MarketSnapshot,
        symbol: str,
        outputs: Sequence[_ActorSignal],
        *,
        actionable: set[str],
        signal_id: int,
        shadow_confirmation: Mapping[object, object],
        degraded_signal_keys: set[str],
        degraded_actor_keys: set[str],
        degraded_open_symbols: set[str],
        promoted_signal_keys: set[str],
        probation_signal_keys: set[str],
        previous_actor_by_symbol: Mapping[str, str],
        suppressed_agent_labels: set[str],
        active_solo_wrapper_keys: set[tuple[str, str]],
        open_position_sides_by_symbol: Mapping[str, str],
        open_position_actor_keys_by_symbol: Mapping[str, frozenset[str]],
        position_state_available: bool,
        genetics_confirmation_by_symbol: Mapping[str, Mapping[str, int]],
    ) -> FlashDecision:
        local_regime = market.regime_for_symbol(symbol)
        rows: List[FlashCandidateAudit] = []
        signal_by_actor_key: dict[str, Signal] = {}
        metrics_by_actor_key: dict[str, Metrics] = {}
        admissible_solo_wrapper_keys = (
            self._admissible_solo_wrapper_keys_for_suppression(
                outputs,
                market=market,
                symbol=symbol,
                regime=local_regime,
                degraded_actor_keys=degraded_actor_keys,
                degraded_open_symbols=degraded_open_symbols,
                promoted_signal_keys=promoted_signal_keys,
            )
        )
        for output in outputs:
            if output.symbol != symbol:
                continue
            metrics, base_score = self._score_actor(output, market)
            metrics_by_actor_key[output.actor_key] = metrics
            effective_score = base_score
            signal_key = _signal_deny_key(output.actor_key, symbol, output.action)
            context_key = _signal_context_key(
                output.actor_key,
                symbol,
                output.action,
                local_regime,
            )
            probation_admission = bool(
                output.action.is_open and signal_key in probation_signal_keys
            )
            selected_subset_protected = (
                signal_key in self._selected_subset_do_not_demote_signal_keys
            )
            selected_subset_score_boost = self._selected_subset_score_boosts.get(
                signal_key,
                0.0,
            ) + self._selected_subset_context_score_boosts.get(
                context_key,
                0.0,
            )
            selected_subset_risk_mult = self._selected_subset_risk_mult(
                signal_key,
                context_key,
                output.action,
            )
            portfolio_actor = self._is_portfolio_actor(output.actor_key, output.label)
            symbol_shadow_required = (
                bool(self._config.shadow_symbol_confirmation_enabled)
                and not portfolio_actor
            )
            shadow = self._shadow_confirmation_for(
                output.label,
                shadow_confirmation,
                symbol=symbol,
                action=output.action,
                require_symbol=symbol_shadow_required,
            )
            shadow_source = (
                "symbol"
                if symbol_shadow_required
                else "portfolio_actor" if portfolio_actor else "actor"
            )
            shadow_confirmed_by_base = False
            fallback_min_base_score = self._actor_fallback_min_base_score_for(output)
            shadow_min_score = self._shadow_min_score_for(output)
            session_recovery_override = self._session_recovery_override_active(output)
            shadow_recovery_candidate = self._shadow_recovery_candidate_active(
                output,
                actionable,
            )
            if (
                session_recovery_override
                or shadow_recovery_candidate
                or fallback_min_base_score <= 0.0
            ):
                actor_fallback_base_allowed = True
            else:
                actor_fallback_base_allowed = base_score >= fallback_min_base_score
            symbol_shadow_not_bad = (
                shadow.closed_trades <= 0
                or shadow.score >= shadow_min_score
            )
            fallback_allowed = (
                self._config.shadow_confirmation_enabled
                and symbol_shadow_required
                and shadow.closed_trades < self._config.shadow_confirmation_min_closed_trades
                and actor_fallback_base_allowed
                and symbol_shadow_not_bad
            )
            if fallback_allowed:
                actor_shadow = self._shadow_confirmation_for(
                    output.label,
                    shadow_confirmation,
                    require_symbol=False,
                )
                if (
                    self._config.shadow_actor_fallback_confirmation_enabled
                    and actor_shadow.closed_trades
                    >= self._config.shadow_confirmation_min_closed_trades
                    and actor_shadow.score > shadow_min_score
                    and self._actor_fallback_shadow_score_allowed(
                        output,
                        actor_shadow,
                    )
                ):
                    shadow = actor_shadow
                    shadow_source = "actor_fallback"
                if (
                    shadow_source == "symbol"
                    and fallback_min_base_score > 0.0
                    and self._config.shadow_base_fallback_confirmation_enabled
                    and self._is_base_fallback_allowed(output)
                    and base_score >= fallback_min_base_score
                ):
                    shadow = _ShadowConfirmation(
                        score=base_score,
                        closed_trades=metrics.closed_trades,
                        winning_trades=metrics.wins,
                        win_rate_pct=metrics.win_rate,
                    )
                    shadow_source = "base_fallback"
                    shadow_confirmed_by_base = True
            if (
                self._config.shadow_confirmation_enabled
                and portfolio_actor
                and self._config.shadow_base_fallback_confirmation_enabled
                and self._is_base_fallback_allowed(output)
                and base_score >= fallback_min_base_score
                and (
                    shadow.closed_trades
                    < self._config.shadow_confirmation_min_closed_trades
                    or shadow.score <= shadow_min_score
                )
            ):
                shadow = _ShadowConfirmation(
                    score=base_score,
                    closed_trades=metrics.closed_trades,
                    winning_trades=metrics.wins,
                    win_rate_pct=metrics.win_rate,
                )
                shadow_source = "portfolio_base_fallback"
                shadow_confirmed_by_base = True
            shadow_recovery_admission = self._shadow_recovery_admission_active(
                output,
                actionable,
                shadow,
                shadow_min_score=shadow_min_score,
            )
            shadow_bootstrap_evidence = (
                (
                    portfolio_actor
                    and self._config.portfolio_shadow_bootstrap_min_closed_enabled
                    and self._config.shadow_confirmation_enabled
                    and shadow.closed_trades
                    >= self._config.shadow_confirmation_min_closed_trades
                    and shadow.score > shadow_min_score
                )
                or shadow_recovery_admission
            )
            session_recovery_shadow_confirmed = (
                session_recovery_override
                and output.action.is_open
                and self._shadow_confirmation_meets_minimum(
                    shadow,
                    shadow_min_score,
                )
            )

            shadow_pnl_per_trade_lcb_usd = shadow.pnl_per_trade_lcb(
                z=self._config.shadow_confirmation_pnl_per_trade_lcb_z,
            )
            shadow_pnl_per_trade_lcb_penalty = (
                self._shadow_pnl_per_trade_lcb_penalty(
                    output.action,
                    shadow_pnl_per_trade_lcb_usd,
                )
            )
            symbol_health = self._shadow_symbol_health_for(
                shadow_confirmation,
                symbol=symbol,
                action=output.action,
            )
            symbol_health_pnl_per_trade_lcb_usd = symbol_health.pnl_per_trade_lcb(
                z=self._config.shadow_confirmation_pnl_per_trade_lcb_z,
            )
            symbol_health_penalty = self._shadow_symbol_health_penalty(
                output.action,
                symbol_health,
                symbol_health_pnl_per_trade_lcb_usd,
                actor_pnl_per_trade_lcb_usd=shadow_pnl_per_trade_lcb_usd,
            )
            if selected_subset_protected and output.action.is_open:
                shadow_pnl_per_trade_lcb_penalty = 0.0
                symbol_health_penalty = 0.0
            regime_confidence_scale = _regime_confidence_scale(market, symbol=symbol)
            gate_score = shadow.score if self._config.shadow_confirmation_enabled else base_score
            gate_score *= regime_confidence_scale
            gate_score -= shadow_pnl_per_trade_lcb_penalty
            gate_score -= symbol_health_penalty
            funding_score_adjustment = self._funding_score_adjustment(
                market,
                symbol,
                output.action,
            )
            gate_score += funding_score_adjustment
            genetics_counts = genetics_confirmation_by_symbol.get(symbol, {})
            genetics_confirmation_adjustment = (
                self._genetics_confirmation_adjustment(
                    output,
                    genetics_counts,
                )
            )
            gate_score += genetics_confirmation_adjustment
            gate_score += selected_subset_score_boost
            technical_alignment, technical_score_adjustment, technical_gate_reason = (
                self._technical_alignment(market, symbol, output.action)
            )
            gate_score += technical_score_adjustment
            trend_alignment, trend_gate_reason = self._trend_alignment_gate(
                market,
                symbol,
                output.action,
                metrics,
            )
            if probation_admission:
                probation_gate_score = base_score * regime_confidence_scale
                probation_gate_score += funding_score_adjustment
                probation_gate_score += genetics_confirmation_adjustment
                probation_gate_score += selected_subset_score_boost
                probation_gate_score += technical_score_adjustment
                gate_score = max(gate_score, probation_gate_score)
                shadow_source = "genetics_probation"
                shadow_confirmed_by_base = True
            tech = _technical_indicators_for_symbol(market, symbol)
            signal_degraded = self._is_signal_degraded(
                output,
                symbol,
                degraded_signal_keys,
            )
            effective_score = gate_score
            if output.label in actionable:
                effective_score *= (
                    1.0 + self._config.actionable_bonus * regime_confidence_scale
                )

            lookback_return_pct = _lookback_return_pct(
                market,
                symbol,
                self._config.overextension_lookback_bars,
            )
            lookback_volatility_pct = _lookback_volatility_pct(
                market,
                symbol,
                self._config.overextension_lookback_bars,
            )
            lookback_return_z = _lookback_return_z(
                lookback_return_pct,
                lookback_volatility_pct,
                self._config.overextension_min_volatility_pct,
            )
            overextension_reason = self._overextension_reason(
                output.action,
                lookback_return_pct,
                lookback_return_z,
            )
            position_owner_block_reason = self._position_owner_block_reason(
                output,
                symbol=symbol,
                open_position_sides_by_symbol=open_position_sides_by_symbol,
                open_position_actor_keys_by_symbol=open_position_actor_keys_by_symbol,
                position_state_available=position_state_available,
            )

            rejected = False
            reason = "eligible"
            if self._qm.is_quarantined(output.label):
                rejected = True
                reason = "quarantined"
            elif _is_no_trade_output(output):
                reason = "eligible_no_trade"
            elif self._is_genetics_confirmation_source(output):
                rejected = True
                reason = "genetics_confirmation_only"
            elif (
                output.actor_type == "agent"
                and output.label in suppressed_agent_labels
                and not probation_admission
                and self._raw_agent_has_active_solo_wrapper(
                    output.label,
                    symbol,
                    active_solo_wrapper_keys,
                )
                and self._raw_agent_has_active_solo_wrapper(
                    output.label,
                    symbol,
                    admissible_solo_wrapper_keys,
                )
            ):
                rejected = True
                reason = "raw_suppressed_by_solo"
            elif output.action.is_hold:
                rejected = True
                reason = "inactive"
            elif self._is_open_regime_denied(output, local_regime):
                rejected = True
                reason = "denied_open_regime"
            elif self._is_open_symbol_denied(output, symbol):
                rejected = True
                reason = "denied_open_symbol"
            elif self._is_range_low_vol_actor_denied(
                output,
                local_regime,
                probation_admission=probation_admission,
            ):
                rejected = True
                reason = "range_low_vol_actor_not_allowed"
            elif self._is_open_symbol_degraded(output, symbol, degraded_open_symbols):
                rejected = True
                reason = "flash_symbol_degraded"
            elif output.action.is_open and not metrics.has_data:
                rejected = True
                reason = "no_evidence"
            elif output.action.is_open and self._global_health_rejected(output.label):
                rejected = True
                reason = "global_health_veto"
            elif (
                output.action.is_open
                and self._regime_edge_rejected(output.label, local_regime)
                and not (
                    probation_admission
                    and self._config.genetics_probation_bypass_regime_edge_enabled
                )
            ):
                rejected = True
                reason = "regime_edge_veto"
            elif (
                output.action.is_open
                and self._config.min_regime_confidence_to_open > 0.0
                and float(getattr(market, "regime_confidence", 1.0))
                < self._config.min_regime_confidence_to_open
            ):
                rejected = True
                reason = "low_regime_confidence"
            elif output.action.is_open and self._real_loss_rejected(output.label):
                rejected = True
                reason = "real_loss_veto"
            elif output.action.is_open and self._fee_aware_rejected(metrics):
                rejected = True
                reason = "expected_edge_below_cost"
            elif output.action.is_open and self._not_in_live_whitelist(output):
                rejected = True
                reason = "not_in_live_whitelist"
            elif output.action.is_open and self._churn_rejected(output, symbol, market.bar):
                rejected = True
                reason = "churn_limited"
            elif (
                output.action.is_open
                and
                metrics.closed_trades < self._config.min_closed_trades_to_trade
                and not shadow_bootstrap_evidence
                and not (
                    probation_admission
                    and self._config.genetics_probation_bypass_min_closed_enabled
                )
            ):
                rejected = True
                reason = "insufficient_closed_trades"
            elif (
                output.action.is_open
                and (
                    (
                        metrics.pnl_per_trade
                        if self._config.gate_pnl_per_trade_enabled
                        else metrics.pnl_pct
                    )
                    < self._config.min_pnl_pct_to_trade
                )
                and not session_recovery_shadow_confirmed
                and not shadow_recovery_admission
                and not (
                    probation_admission
                    and self._config.genetics_probation_bypass_pnl_enabled
                )
            ):
                rejected = True
                reason = "pnl_below_threshold"
            elif (
                self._config.shadow_confirmation_enabled
                and output.action.is_open
                and not probation_admission
                and not shadow_confirmed_by_base
                and shadow.closed_trades < self._config.shadow_confirmation_min_closed_trades
            ):
                rejected = True
                reason = "shadow_unconfirmed"
            elif (
                self._config.shadow_confirmation_enabled
                and not shadow_confirmed_by_base
                and output.action.is_open
                and not probation_admission
                and output.action.is_fraction_full
                and self._config.shadow_confirmation_min_full_open_closed_trades > 0
                and shadow.closed_trades
                < self._config.shadow_confirmation_min_full_open_closed_trades
            ):
                rejected = True
                reason = "shadow_full_open_unconfirmed"
            elif (
                self._config.shadow_confirmation_enabled
                and not shadow_confirmed_by_base
                and output.action.is_open
                and not probation_admission
                and shadow.score <= shadow_min_score
            ):
                rejected = True
                reason = "shadow_score_below_threshold"
            elif (
                self._config.shadow_confirmation_enabled
                and self._config.shadow_quality_confirmation_enabled
                and output.action.is_open
                and not probation_admission
                and self._config.shadow_confirmation_min_win_rate_pct > 0.0
                and shadow.win_rate_pct < self._config.shadow_confirmation_min_win_rate_pct
            ):
                rejected = True
                reason = "shadow_win_rate_below_threshold"
            elif (
                self._config.shadow_confirmation_enabled
                and self._config.shadow_quality_confirmation_enabled
                and output.action.is_open
                and not probation_admission
                and self._config.shadow_confirmation_max_recent_downside_usd > 0.0
                and shadow.recent_downside_usd > self._config.shadow_confirmation_max_recent_downside_usd
            ):
                rejected = True
                reason = "shadow_downside_above_threshold"
            elif (
                self._config.shadow_confirmation_enabled
                and self._config.shadow_quality_confirmation_enabled
                and output.action.is_open
                and not probation_admission
                and not shadow_confirmed_by_base
                and not selected_subset_protected
                and self._config.shadow_confirmation_min_pnl_per_trade_lcb_usd is not None
                and (
                    shadow_pnl_per_trade_lcb_usd is None
                    or shadow_pnl_per_trade_lcb_usd
                    < self._config.shadow_confirmation_min_pnl_per_trade_lcb_usd
                )
            ):
                rejected = True
                reason = "shadow_pnl_per_trade_lcb_below_threshold"
            elif (
                self._config.shadow_confirmation_enabled
                and self._config.shadow_symbol_health_enabled
                and output.action.is_open
                and not probation_admission
                and symbol_health.closed_trades
                >= self._config.shadow_symbol_health_min_closed_trades
                and not selected_subset_protected
                and self._config.shadow_symbol_health_min_pnl_per_trade_lcb_usd is not None
                and (
                    symbol_health_pnl_per_trade_lcb_usd is None
                    or symbol_health_pnl_per_trade_lcb_usd
                    < self._config.shadow_symbol_health_min_pnl_per_trade_lcb_usd
                )
            ):
                rejected = True
                reason = "shadow_symbol_health_lcb_below_threshold"
            elif (
                self._config.technical_overlay_enabled
                and self._config.technical_hard_gate_enabled
                and output.action.is_open
                and technical_gate_reason
            ):
                rejected = True
                reason = technical_gate_reason
            elif trend_gate_reason and not (
                probation_admission
                and self._config.genetics_probation_bypass_trend_gate_enabled
            ):
                rejected = True
                reason = trend_gate_reason
            elif overextension_reason:
                rejected = True
                reason = overextension_reason
            elif position_owner_block_reason:
                rejected = True
                reason = position_owner_block_reason
            elif output.action.is_open and self._is_actor_degraded(output, degraded_actor_keys):
                rejected = True
                reason = "flash_actor_degraded"
            elif output.action.is_open and self._is_actor_denied(output):
                rejected = True
                reason = "flash_actor_deny_key"
            elif (
                output.action.is_open
                and
                signal_degraded
                and not self._config.degradation_signal_risk_sizing_enabled
            ):
                rejected = True
                reason = "flash_signal_degraded"
            elif (
                self._is_signal_terminal_denied(output, symbol, local_regime)
                and not (
                    probation_admission
                    and self._config.genetics_probation_bypass_terminal_deny_enabled
                )
            ):
                rejected = True
                reason = "flash_signal_terminal_deny_key"
            elif self._is_signal_denied(output, symbol):
                rejected = True
                reason = "flash_signal_deny_key"
            elif (
                self._config.promotion_manifest_enabled
                and output.action.is_open
                and not self._is_signal_promoted(output, symbol, promoted_signal_keys)
            ):
                rejected = True
                reason = "flash_signal_not_promoted"
            elif self._is_genetics_contra_no_backfill_candidate(
                output,
                genetics_counts,
            ):
                rejected = True
                reason = "genetics_contra_no_backfill"
            elif output.action.is_open and gate_score <= self._min_score_to_trade_for(output):
                rejected = True
                reason = "score_below_threshold"
            if (
                not rejected
                and reason == "eligible"
                and probation_admission
            ):
                reason = "genetics_probation_admission"
            if (
                not rejected
                and reason == "eligible"
                and signal_degraded
                and output.action.is_open
                and self._config.degradation_signal_risk_sizing_enabled
            ):
                reason = "flash_signal_degraded_risk_sized"

            signal = output.signal
            if signal is not None and signal.sym.upper() == symbol:
                signal_by_actor_key[output.actor_key] = signal

            rows.append(FlashCandidateAudit(
                symbol=symbol,
                label=output.label,
                actor_type=output.actor_type,
                actor_key=output.actor_key,
                score=effective_score,
                base_score=base_score,
                effective_score=effective_score,
                gate_score=gate_score,
                stable_edge=self._stable_regime_edge(output.label, local_regime),
                action=output.action,
                rejected=rejected,
                reason=reason,
                has_data=metrics.has_data,
                closed_trades=metrics.closed_trades,
                agent_labels=output.agent_labels,
                pnl_net_pct=float(getattr(metrics, "pnl_net_pct", metrics.pnl_pct)),
                pnl_gross_pct=float(getattr(metrics, "pnl_gross_pct", metrics.pnl_pct)),
                fee_pct=float(getattr(metrics, "fee_pct", 0.0)),
                funding_pct=float(getattr(metrics, "funding_pct", 0.0)),
                trading_cost_pct=float(getattr(metrics, "trading_cost_pct", 0.0)),
                shadow_score=shadow.score,
                shadow_closed_trades=shadow.closed_trades,
                shadow_winning_trades=shadow.winning_trades,
                shadow_win_rate_pct=shadow.win_rate_pct,
                shadow_recent_downside_usd=shadow.recent_downside_usd,
                shadow_pnl_per_trade_mean_usd=shadow.pnl_per_trade_mean_usd,
                shadow_pnl_per_trade_std_usd=shadow.pnl_per_trade_std_usd,
                shadow_pnl_per_trade_lcb_usd=shadow_pnl_per_trade_lcb_usd,
                shadow_pnl_per_trade_lcb_penalty=(
                    shadow_pnl_per_trade_lcb_penalty
                ),
                shadow_source=shadow_source,
                shadow_symbol_health_score=symbol_health.score,
                shadow_symbol_health_closed_trades=symbol_health.closed_trades,
                shadow_symbol_health_pnl_per_trade_lcb_usd=(
                    symbol_health_pnl_per_trade_lcb_usd
                ),
                shadow_symbol_health_penalty=symbol_health_penalty,
                genetics_confirmation_adjustment=genetics_confirmation_adjustment,
                funding_score_adjustment=funding_score_adjustment,
                selected_subset_score_boost=selected_subset_score_boost,
                selected_subset_protected=selected_subset_protected,
                selected_subset_risk_mult=selected_subset_risk_mult,
                risk_mult=self._risk_mult_for_candidate(
                    metrics,
                    market,
                    symbol,
                    output.action,
                    shadow_pnl_per_trade_lcb_usd=shadow_pnl_per_trade_lcb_usd,
                    selected_subset_risk_mult=selected_subset_risk_mult,
                    genetics_confirmation_counts=genetics_counts,
                    degraded_signal_risk_sized=signal_degraded,
                ),
                lookback_return_pct=lookback_return_pct,
                lookback_volatility_pct=lookback_volatility_pct,
                lookback_return_z=lookback_return_z,
                technical_rsi_14=None if tech is None else tech.rsi_14,
                technical_macd_histogram_pct=(
                    None if tech is None else tech.macd_histogram_pct
                ),
                technical_atr_14_pct=None if tech is None else tech.atr_14_pct,
                technical_alignment=technical_alignment,
                technical_score_adjustment=technical_score_adjustment,
                technical_gate_reason=technical_gate_reason,
                trend_alignment=trend_alignment,
                trend_gate_reason=trend_gate_reason,
                regime_feature_risk_mult=self._regime_feature_risk_mult(
                    market,
                    symbol,
                    output.action,
                ),
            ))

        if self._config.stable_pool_ranking_enabled:
            # Ранжируем по стабильному per-regime эджу (не по волатильному score):
            # закреплённый актор торгует свой полный поток, а не реверсивную подвыборку.
            rows.sort(
                key=lambda row: (
                    row.rejected,
                    not row.action.is_close,
                    -row.stable_edge,
                    -row.closed_trades,
                    -row.score,
                    _stable_candidate_tiebreak(row, market.bar),
                )
            )
        else:
            rows.sort(
                key=lambda row: (
                    row.rejected,
                    not row.action.is_close,
                    -row.score,
                    -row.closed_trades,
                    _stable_candidate_tiebreak(row, market.bar),
                )
            )
        ranked = tuple(replace(row, rank=i + 1) for i, row in enumerate(rows))
        terminal_denied = self._terminal_denied_top_candidate(ranked, market.bar)
        if terminal_denied is not None:
            return FlashDecision(
                symbol=symbol,
                selected_actor="NoTrade",
                actor_type="no_trade",
                score=0.0,
                action=Action.HOLD,
                reason="flash_signal_terminal_deny_key",
                signal=None,
                candidates=ranked,
                original_selected_actor=terminal_denied.label,
                original_actor_type=terminal_denied.actor_type,
                selected_reasons=("flash_signal_terminal_deny_key",),
            )
        no_backfill = self._genetics_contra_no_backfill_top_candidate(
            ranked,
            market.bar,
        )
        if no_backfill is not None:
            return FlashDecision(
                symbol=symbol,
                selected_actor="NoTrade",
                actor_type="no_trade",
                score=0.0,
                action=Action.HOLD,
                reason="genetics_contra_no_backfill",
                signal=None,
                candidates=ranked,
                original_selected_actor=no_backfill.label,
                original_actor_type=no_backfill.actor_type,
                selected_reasons=("genetics_contra_no_backfill",),
            )
        selected = next((row for row in ranked if not row.rejected), None)
        if selected is None:
            return FlashDecision(
                symbol=symbol,
                selected_actor="NoTrade",
                actor_type="no_trade",
                score=0.0,
                action=Action.HOLD,
                reason="no_eligible_actor",
                signal=None,
                candidates=ranked,
            )
        no_trade_admission_block = (
            self._top_rejected_positive_shadow_candidate(ranked)
            if _is_no_trade_candidate(selected)
            else None
        )
        if no_trade_admission_block is not None:
            return FlashDecision(
                symbol=symbol,
                selected_actor="NoTrade",
                actor_type="no_trade",
                score=0.0,
                action=Action.HOLD,
                reason=f"no_real_admission:{no_trade_admission_block.reason}",
                signal=None,
                candidates=ranked,
                original_selected_actor=no_trade_admission_block.label,
                original_actor_type=no_trade_admission_block.actor_type,
                selected_reasons=(
                    "selected",
                    "no_real_admission",
                    no_trade_admission_block.reason,
                ),
            )

        original_selected = selected
        selected_reasons: List[str] = ["selected"]
        anchor = self._anchor_dominance_candidate(ranked, selected)
        if anchor is not None:
            selected = anchor
            selected_reasons.append("anchor_dominance_hold")
        previous_actor_key = _previous_actor_key_for_symbol(
            previous_actor_by_symbol,
            symbol,
        )
        if self._config.actor_switch_margin > 0.0 and previous_actor_key:
            previous = next(
                (
                    row
                    for row in ranked
                    if not row.rejected
                    and row.actor_key == previous_actor_key
                ),
                None,
            )
            if (
                previous is not None
                and previous.actor_key != selected.actor_key
                and selected.score - previous.score < self._config.actor_switch_margin
            ):
                selected = previous
                selected_reasons.append("switch_margin_hold")

        signal = signal_by_actor_key.get(selected.actor_key)
        position_block_reason = self._position_state_block_reason(
            selected,
            signal,
            symbol=symbol,
            open_position_sides_by_symbol=open_position_sides_by_symbol,
            open_position_actor_keys_by_symbol=open_position_actor_keys_by_symbol,
            position_state_available=position_state_available,
        )
        if position_block_reason:
            replacement = next(
                (
                    row
                    for row in ranked
                    if not row.rejected
                    and row.actor_key != selected.actor_key
                    and (row.action.is_open or row.action.is_close)
                    and not self._position_state_block_reason(
                        row,
                        signal_by_actor_key.get(row.actor_key),
                        symbol=symbol,
                        open_position_sides_by_symbol=open_position_sides_by_symbol,
                        open_position_actor_keys_by_symbol=open_position_actor_keys_by_symbol,
                        position_state_available=position_state_available,
                    )
                ),
                None,
            )
            if replacement is not None:
                selected = replacement
                signal = signal_by_actor_key.get(selected.actor_key)
                selected_reasons.append("position_state_fallback")
            else:
                return FlashDecision(
                    symbol=symbol,
                    selected_actor="NoTrade",
                    actor_type="no_trade",
                    score=0.0,
                    action=Action.HOLD,
                    reason=position_block_reason,
                    signal=None,
                    candidates=ranked,
                    original_selected_actor=selected.label,
                    original_actor_type=selected.actor_type,
                    selected_reasons=tuple(selected_reasons),
                )
        if signal is not None:
            if (
                position_state_available
                and signal.action.is_close
                and symbol not in open_position_sides_by_symbol
            ):
                return FlashDecision(
                    symbol=symbol,
                    selected_actor="NoTrade",
                    actor_type="no_trade",
                    score=0.0,
                    action=Action.HOLD,
                    reason="stale_close_position",
                    signal=None,
                    candidates=ranked,
                    original_selected_actor=selected.label,
                    original_actor_type=selected.actor_type,
                    selected_reasons=tuple(selected_reasons),
                )
            existing_side = open_position_sides_by_symbol.get(symbol, "")
            if (
                signal.action.is_open
                and existing_side
                and signal.action.side == existing_side
            ):
                return FlashDecision(
                    symbol=symbol,
                    selected_actor="NoTrade",
                    actor_type="no_trade",
                    score=0.0,
                    action=Action.HOLD,
                    reason="duplicate_open_position",
                    signal=None,
                    candidates=ranked,
                    original_selected_actor=selected.label,
                    original_actor_type=selected.actor_type,
                    selected_reasons=tuple(selected_reasons),
                )
            price = _market_price_for_symbol(market, symbol)
            if price is None or price <= 0.0:
                return FlashDecision(
                    symbol=symbol,
                    selected_actor="NoTrade",
                    actor_type="no_trade",
                    score=0.0,
                    action=Action.HOLD,
                    reason="missing_price",
                    signal=None,
                    candidates=ranked,
                    original_selected_actor=selected.label,
                    original_actor_type=selected.actor_type,
                    selected_reasons=tuple(selected_reasons),
                )
            signal = replace(
                signal,
                id=signal_id,
                bar=market.bar,
                sym=symbol,
                price=price,
                regime=local_regime,
                risk_mult=self._risk_mult_for_signal(
                    signal,
                    metrics_by_actor_key.get(selected.actor_key),
                    market,
                    symbol,
                    selected.action,
                    shadow_pnl_per_trade_lcb_usd=selected.shadow_pnl_per_trade_lcb_usd,
                    selected_subset_risk_mult=selected.selected_subset_risk_mult,
                    genetics_confirmation_counts=genetics_confirmation_by_symbol.get(
                        symbol,
                        {},
                    ),
                    degraded_signal_risk_sized=(
                        _signal_deny_key(selected.actor_key, symbol, selected.action)
                        in degraded_signal_keys
                    ),
                ),
                timestamp=market.timestamp,
            )
        return FlashDecision(
            symbol=symbol,
            selected_actor=selected.label,
            actor_type=selected.actor_type,
            score=selected.score,
            action=selected.action,
            reason=selected_reasons[-1],
            signal=signal,
            candidates=ranked,
            original_selected_actor=original_selected.label,
            original_actor_type=original_selected.actor_type,
            selected_reasons=tuple(selected_reasons),
        )

    def _position_state_block_reason(
        self,
        row: FlashCandidateAudit,
        signal: Optional[Signal],
        *,
        symbol: str,
        open_position_sides_by_symbol: Mapping[str, str],
        open_position_actor_keys_by_symbol: Mapping[str, frozenset[str]],
        position_state_available: bool,
    ) -> str:
        if signal is None or not position_state_available:
            return ""
        if signal.action.is_close and symbol not in open_position_sides_by_symbol:
            return "stale_close_position"
        existing_side = open_position_sides_by_symbol.get(symbol, "")
        owner_keys = open_position_actor_keys_by_symbol.get(symbol, frozenset())
        if (
            owner_keys
            and (
                signal.action.is_close
                or (
                    signal.action.is_open
                    and existing_side
                    and signal.action.side != existing_side
                )
            )
            and not self._signal_matches_position_owner(signal, row, owner_keys)
        ):
            return "foreign_position_owner"
        if signal.action.is_open and existing_side and signal.action.side == existing_side:
            return "duplicate_open_position"
        return ""

    def _position_owner_block_reason(
        self,
        output: _ActorSignal,
        *,
        symbol: str,
        open_position_sides_by_symbol: Mapping[str, str],
        open_position_actor_keys_by_symbol: Mapping[str, frozenset[str]],
        position_state_available: bool,
    ) -> str:
        if not position_state_available or symbol not in open_position_sides_by_symbol:
            return ""
        owner_keys = open_position_actor_keys_by_symbol.get(symbol, frozenset())
        if not owner_keys:
            return ""
        existing_side = open_position_sides_by_symbol.get(symbol, "")
        would_close = output.action.is_close or (
            output.action.is_open
            and existing_side
            and output.action.side != existing_side
        )
        if not would_close:
            return ""
        if self._signal_matches_position_owner(output.signal, output, owner_keys):
            return ""
        return "foreign_position_owner"

    def _signal_matches_position_owner(
        self,
        signal: Optional[Signal],
        actor: _ActorSignal | FlashCandidateAudit,
        owner_keys: frozenset[str],
    ) -> bool:
        actor_key = str(getattr(actor, "actor_key", "") or "").strip()
        if actor_key and actor_key in owner_keys:
            return True
        if signal is None:
            return False
        closer_agent = str(getattr(signal, "by_agent", "") or "").strip()
        if closer_agent in _SYSTEM_CLOSE_AGENTS:
            return True
        closer_player = str(getattr(signal, "by_player", "") or "").strip()
        signal_keys = {
            key
            for key in (
                f"agent:{closer_agent}" if closer_agent else "",
                f"ensemble:{closer_player}" if closer_player else "",
                f"agent:{closer_player}" if closer_player else "",
            )
            if key
        }
        return bool(signal_keys & owner_keys)

    def _score_actor(
        self,
        output: _ActorSignal,
        market: MarketSnapshot | Regime,
    ) -> tuple[Metrics, float]:
        symbol = str(getattr(output, "symbol", "") or "").upper()
        if isinstance(market, MarketSnapshot) and symbol:
            regime = market.regime_for_symbol(symbol)
        else:
            raw_regime = getattr(market, "regime", market)
            regime = raw_regime if isinstance(raw_regime, Regime) else Regime.from_string(str(raw_regime or ""))
        if _is_no_trade_output(output):
            if not isinstance(market, MarketSnapshot):
                return Metrics.empty(), 0.0
            return Metrics.empty(), self._no_trade_synthetic_score(
                market,
                getattr(output, "symbol", ""),
            )
        actor_key = str(
            getattr(output, "actor_key", "")
            or _actor_key(output.actor_type, output.label)
        )
        if self._is_portfolio_actor(actor_key, output.label):
            metrics = (
                self._metrics_for_label(output.label, regime, symbol=symbol)
                if symbol
                else self._perf.get(output.label)
            )
            if metrics.has_data:
                return metrics, self._score_metrics(metrics, regime)
            if output.actor_type == "ensemble":
                return self._component_score(
                    output.agent_labels,
                    regime,
                    use_aggregate=not bool(symbol),
                    symbol=symbol,
                )
            return metrics, float(self._config.no_data_score)
        metrics = self._metrics_for_label(output.label, regime, symbol=symbol)
        if not metrics.has_data and output.actor_type == "ensemble":
            return self._component_score(output.agent_labels, regime, symbol=symbol)
        if not metrics.has_data:
            return metrics, float(self._config.no_data_score)
        return metrics, self._score_metrics(metrics, regime)

    def _score_metrics(self, metrics: Metrics, regime: Regime) -> float:
        return float(
            regime_score(
                metrics,
                regime,
                config=self._scoring,
                per_regime_configs=self._scoring_per_regime,
            )
        )

    def _no_trade_synthetic_score(
        self,
        market: MarketSnapshot,
        symbol: str,
    ) -> float:
        if not self._config.no_trade_fee_saving_score_enabled:
            return 0.0
        fee_bps = _fee_bps_for_symbol(
            market,
            symbol,
            default_bps=self._config.no_trade_default_fee_bps,
        )
        # Metrics.pnl_pct is expressed in percent, while fee inputs are bps.
        return (fee_bps / 100.0) * float(self._scoring.pnl_weight)

    def _funding_score_adjustment(
        self,
        market: MarketSnapshot,
        symbol: str,
        action: Action,
    ) -> float:
        if self._config.funding_score_weight <= 0.0 or not action.is_open:
            return 0.0
        funding = _market_funding_for_symbol(market, symbol)
        if funding == 0.0:
            return 0.0
        if action.is_long_open:
            return -funding * self._config.funding_score_weight
        if action.is_short_open:
            return funding * self._config.funding_score_weight
        return 0.0

    def _trend_alignment_gate(
        self,
        market: MarketSnapshot,
        symbol: str,
        action: Action,
        metrics: Metrics,
    ) -> tuple[str, str]:
        if not self._config.trend_alignment_gate_enabled or not action.is_open:
            return "", ""
        features = _regime_features_for_symbol(market, symbol)
        if not features:
            return "", ""
        micro = str(features.get("micro_direction") or "flat").lower()
        macro = str(features.get("macro_direction") or "flat").lower()
        alignment = str(features.get("micro_macro_alignment") or "").lower()
        if not alignment:
            alignment = _micro_macro_alignment_from_features(micro, macro)
        action_direction = _open_action_direction(action)
        expected = "up" if action_direction == "long" else "down" if action_direction == "short" else ""
        if not expected:
            return alignment, ""
        if self._counter_trend_edge_allowed(metrics):
            return alignment, ""
        if alignment == "conflict" and micro in {"up", "down"} and macro in {"up", "down"}:
            return alignment, "micro_macro_conflict"
        if macro in {"up", "down"} and macro != expected:
            return alignment or f"macro_{macro}", "trend_contra_unsupported"
        if macro == "flat" and micro in {"up", "down"} and micro != expected:
            return alignment or f"micro_{micro}", "trend_contra_unsupported"
        return alignment, ""

    def _counter_trend_edge_allowed(self, metrics: Metrics) -> bool:
        if metrics is None or not getattr(metrics, "has_data", False):
            return False
        if int(getattr(metrics, "closed_trades", 0) or 0) < int(
            self._config.counter_trend_min_closed_trades
        ):
            return False
        closed = max(1, int(getattr(metrics, "closed_trades", 0) or 0))
        pnl_per_trade_pct = (
            float(getattr(metrics, "pnl_net_pct", metrics.pnl_pct) or 0.0)
            / closed
        )
        return pnl_per_trade_pct >= float(self._config.counter_trend_min_pnl_pct)

    def _technical_alignment(
        self,
        market: MarketSnapshot,
        symbol: str,
        action: Action,
    ) -> tuple[str, float, str]:
        if not self._config.technical_overlay_enabled or not action.is_open:
            return "", 0.0, ""
        tech = _technical_indicators_for_symbol(market, symbol)
        if tech is None:
            return "missing", 0.0, "technical_missing"
        rsi = getattr(tech, "rsi_14", None)
        hist = getattr(tech, "macd_histogram_pct", None)
        if rsi is None or hist is None:
            return "missing", 0.0, "technical_missing"
        min_hist = float(self._config.technical_macd_histogram_min_abs_pct)
        if action.is_long_open:
            rsi_ok = (
                self._config.technical_rsi_long_min
                <= float(rsi)
                <= self._config.technical_rsi_long_max
            )
            macd_ok = float(hist) > min_hist
            if rsi_ok and macd_ok:
                return (
                    "long_aligned",
                    float(self._config.technical_score_bonus),
                    "",
                )
            return (
                "long_misaligned",
                -float(self._config.technical_score_penalty),
                "technical_long_misaligned",
            )
        if action.is_short_open:
            rsi_ok = (
                self._config.technical_rsi_short_min
                <= float(rsi)
                <= self._config.technical_rsi_short_max
            )
            macd_ok = float(hist) < -min_hist
            if rsi_ok and macd_ok:
                return (
                    "short_aligned",
                    float(self._config.technical_score_bonus),
                    "",
                )
            return (
                "short_misaligned",
                -float(self._config.technical_score_penalty),
                "technical_short_misaligned",
            )
        return "", 0.0, ""

    def _genetics_confirmation_by_symbol(
        self,
        market: MarketSnapshot,
        shadow_player_signals: Mapping[str, Sequence[Signal]],
        shadow_agent_signals: Mapping[str, Sequence[Signal]],
        *,
        outputs: Sequence[_ActorSignal] = (),
    ) -> dict[str, dict[str, int]]:
        if not self._config.genetics_confirmation_overlay_enabled:
            return {}
        labels = set(self._config.genetics_confirmation_labels)
        if not labels:
            return {}
        counts: dict[str, dict[str, int]] = {}
        seen: set[tuple[str, str, str]] = set()

        def add_confirmation(
            label: str,
            symbol: str,
            direction: str,
            *,
            contra: bool = False,
        ) -> None:
            count_key = f"contra_{direction}" if contra else direction
            key = (label, symbol, count_key)
            if key in seen:
                return
            seen.add(key)
            symbol_counts = counts.setdefault(
                symbol,
                {"long": 0, "short": 0, "contra_long": 0, "contra_short": 0},
            )
            symbol_counts[count_key] += 1

        if self._config.genetics_confirmation_contra_static_enabled:
            for raw_key in self._config.genetics_confirmation_contra_signal_keys:
                static = _static_contra_signal_from_key(raw_key, labels)
                if static is None:
                    continue
                label, symbol, direction = static
                add_confirmation(label, symbol, direction, contra=True)

        for output in outputs:
            label = str(output.label or "").strip()
            if label not in labels:
                continue
            direction = _open_action_direction(output.action)
            if direction not in {"long", "short"}:
                continue
            if self._genetics_confirmation_contra_signal_allowed(
                output.actor_key,
                output.symbol,
                output.action,
            ):
                add_confirmation(label, output.symbol, direction, contra=True)
                continue
            if not self._genetics_confirmation_source_allowed(label, market):
                continue
            if not self._genetics_confirmation_signal_allowed(
                output.actor_key,
                output.symbol,
                output.action,
            ):
                continue
            add_confirmation(label, output.symbol, direction)
        for actor_type, signal_map in (
            ("player", shadow_player_signals),
            ("agent", shadow_agent_signals),
        ):
            for label, signals in tuple(signal_map.items()):
                clean_label = str(label or "").strip()
                if clean_label not in labels:
                    continue
                if not self._genetics_confirmation_source_allowed(clean_label, market):
                    continue
                actor_key = _actor_key(actor_type, clean_label)
                for signal in signals or ():
                    symbol = str(getattr(signal, "sym", "") or "").upper()
                    action = _coerce_action(getattr(signal, "action", Action.HOLD))
                    direction = _open_action_direction(action)
                    if not symbol or direction not in {"long", "short"}:
                        continue
                    if self._genetics_confirmation_contra_signal_allowed(
                        actor_key,
                        symbol,
                        action,
                    ):
                        add_confirmation(clean_label, symbol, direction, contra=True)
                        continue
                    if not self._genetics_confirmation_signal_allowed(
                        actor_key,
                        symbol,
                        action,
                    ):
                        continue
                    add_confirmation(clean_label, symbol, direction)
        return counts

    def _genetics_confirmation_signal_allowed(
        self,
        actor_key: str,
        symbol: str,
        action: Action,
    ) -> bool:
        allowed = set(self._config.genetics_confirmation_allowed_signal_keys)
        if not allowed:
            return True
        return _signal_deny_key(actor_key, symbol, action) in allowed

    def _genetics_confirmation_contra_signal_allowed(
        self,
        actor_key: str,
        symbol: str,
        action: Action,
    ) -> bool:
        contra = set(self._config.genetics_confirmation_contra_signal_keys)
        if not contra:
            return False
        if _signal_deny_key(actor_key, symbol, action) in contra:
            return True
        if not self._config.genetics_confirmation_contra_side_match_enabled:
            return False
        side_key = _signal_side_key(actor_key, symbol, action)
        if not side_key:
            return False
        return side_key in {
            key
            for key in (
                _signal_side_key_from_normalized_key(raw_key)
                for raw_key in contra
            )
            if key
        }

    def _genetics_confirmation_source_allowed(
        self,
        label: str,
        market: MarketSnapshot,
    ) -> bool:
        if not self._config.genetics_confirmation_quality_gate_enabled:
            return True
        metrics = self._metrics_for_label(label, market.regime)
        if metrics.closed_trades <= 0:
            metrics = self._perf.get(label)
        if metrics.closed_trades < self._config.genetics_confirmation_min_closed_trades:
            return False
        return (
            metrics.pnl_per_trade
            >= self._config.genetics_confirmation_min_pnl_per_trade_pct
        )

    def _genetics_confirmation_adjustment(
        self,
        output: _ActorSignal,
        confirmation_counts: Mapping[str, int],
    ) -> float:
        if (
            not self._config.genetics_confirmation_overlay_enabled
            or not output.action.is_open
            or self._is_genetics_confirmation_source(output)
        ):
            return 0.0
        side = _open_action_direction(output.action)
        if side not in {"long", "short"}:
            return 0.0
        opposite = "short" if side == "long" else "long"
        matching = int(confirmation_counts.get(side, 0) or 0)
        opposing = int(confirmation_counts.get(opposite, 0) or 0)
        contra_matching = self._genetics_confirmation_contra_count(
            output.action,
            confirmation_counts,
        )
        return (
            matching * float(self._config.genetics_confirmation_score_bonus)
            - opposing * float(self._config.genetics_confirmation_score_penalty)
            - contra_matching
            * float(self._config.genetics_confirmation_contra_score_penalty)
        )

    @staticmethod
    def _genetics_confirmation_contra_count(
        action: Action,
        confirmation_counts: Mapping[str, int],
    ) -> int:
        side = _open_action_direction(action)
        if side not in {"long", "short"}:
            return 0
        return int(confirmation_counts.get(f"contra_{side}", 0) or 0)

    def _is_genetics_confirmation_source(self, output: _ActorSignal) -> bool:
        return (
            self._config.genetics_confirmation_overlay_enabled
            and str(output.label or "").strip()
            in set(self._config.genetics_confirmation_labels)
        )

    def _risk_mult_for_signal(
        self,
        signal: Signal,
        metrics: Optional[Metrics],
        market: MarketSnapshot,
        symbol: str,
        action: Action,
        *,
        shadow_pnl_per_trade_lcb_usd: Optional[float] = None,
        selected_subset_risk_mult: float = 1.0,
        genetics_confirmation_counts: Mapping[str, int] | None = None,
        degraded_signal_risk_sized: bool = False,
    ) -> float:
        base = _finite_positive_or_default(getattr(signal, "risk_mult", 1.0), 1.0)
        return base * self._risk_mult_for_candidate(
            metrics,
            market,
            symbol,
            action,
            shadow_pnl_per_trade_lcb_usd=shadow_pnl_per_trade_lcb_usd,
            selected_subset_risk_mult=selected_subset_risk_mult,
            genetics_confirmation_counts=genetics_confirmation_counts,
            degraded_signal_risk_sized=degraded_signal_risk_sized,
        )

    def _risk_mult_for_candidate(
        self,
        metrics: Optional[Metrics],
        market: MarketSnapshot,
        symbol: str,
        action: Action,
        *,
        shadow_pnl_per_trade_lcb_usd: Optional[float] = None,
        selected_subset_risk_mult: float = 1.0,
        genetics_confirmation_counts: Mapping[str, int] | None = None,
        degraded_signal_risk_sized: bool = False,
    ) -> float:
        if not action.is_open:
            return 1.0
        risk_mult = 1.0
        risk_mult *= self._degraded_signal_risk_mult(
            action,
            degraded_signal_risk_sized,
        )
        risk_mult *= self._shadow_pnl_per_trade_lcb_risk_mult(
            action,
            shadow_pnl_per_trade_lcb_usd,
        )
        if self._config.actor_risk_sizing_enabled:
            risk_mult *= self._actor_edge_risk_mult(metrics)
        if self._config.funding_risk_mult_weight > 0.0:
            risk_mult *= self._funding_risk_mult(market, symbol, action)
        if self._config.volatility_risk_sizing_enabled:
            risk_mult *= self._volatility_risk_mult(market, symbol)
        if self._config.regime_feature_risk_sizing_enabled:
            risk_mult *= self._regime_feature_risk_mult(market, symbol, action)
        if (
            self._config.technical_overlay_enabled
            and self._config.technical_atr_risk_sizing_enabled
        ):
            risk_mult *= self._technical_atr_risk_mult(market, symbol)
        risk_mult *= self._genetics_contra_risk_mult(
            action,
            genetics_confirmation_counts or {},
        )
        risk_mult *= _finite_positive_or_default(selected_subset_risk_mult, 1.0)
        return max(0.0, float(risk_mult))

    def _degraded_signal_risk_mult(
        self,
        action: Action,
        signal_degraded: bool,
    ) -> float:
        if not (
            signal_degraded
            and action.is_open
            and self._config.degradation_signal_risk_sizing_enabled
        ):
            return 1.0
        return _clamp(
            float(self._config.degradation_signal_risk_mult),
            0.0,
            1.0,
        )

    def _genetics_contra_risk_mult(
        self,
        action: Action,
        confirmation_counts: Mapping[str, int],
    ) -> float:
        if (
            not self._config.genetics_confirmation_contra_risk_sizing_enabled
            or not action.is_open
            or self._genetics_confirmation_contra_count(action, confirmation_counts) <= 0
        ):
            return 1.0
        return max(0.0, float(self._config.genetics_confirmation_contra_risk_mult))

    def _selected_subset_risk_mult(
        self,
        signal_key: str,
        context_key: str,
        action: Action,
    ) -> float:
        if not action.is_open:
            return 1.0
        raw = self._selected_subset_context_risk_mult_overrides.get(context_key)
        if raw is None:
            raw = self._selected_subset_risk_mult_overrides.get(signal_key)
        if raw is None:
            return 1.0
        return _clamp(
            float(raw),
            float(self._config.selected_subset_risk_min_mult),
            float(self._config.selected_subset_risk_max_mult),
        )

    def _shadow_pnl_per_trade_lcb_risk_mult(
        self,
        action: Action,
        shadow_pnl_per_trade_lcb_usd: Optional[float],
    ) -> float:
        if not (
            self._config.shadow_confirmation_enabled
            and self._config.shadow_quality_confirmation_enabled
            and self._config.shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled
            and action.is_open
        ):
            return 1.0
        if shadow_pnl_per_trade_lcb_usd is None:
            return 1.0
        floor = float(
            self._config.shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd
        )
        gap = floor - float(shadow_pnl_per_trade_lcb_usd)
        if gap <= 0.0:
            return 1.0
        scale = max(
            1e-9,
            float(self._config.shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd),
        )
        min_mult = float(
            self._config.shadow_confirmation_pnl_per_trade_lcb_risk_min_mult
        )
        return _clamp(1.0 - gap / scale, min_mult, 1.0)

    def _actor_edge_risk_mult(self, metrics: Optional[Metrics]) -> float:
        if metrics is None or not metrics.has_data or metrics.closed_trades <= 0:
            return float(self._config.actor_risk_min_mult)
        pnl_per_trade = float(metrics.pnl_net_pct) / max(1, int(metrics.closed_trades))
        if pnl_per_trade <= 0.0:
            return float(self._config.actor_risk_min_mult)
        scale = max(1e-9, float(self._config.actor_risk_edge_scale_pct))
        edge_share = min(1.0, pnl_per_trade / scale)
        confidence = min(
            1.0,
            float(metrics.closed_trades)
            / float(self._scoring.min_closed_for_full_confidence),
        )
        min_mult = float(self._config.actor_risk_min_mult)
        max_mult = float(self._config.actor_risk_max_mult)
        return min_mult + (max_mult - min_mult) * edge_share * confidence

    def _funding_risk_mult(
        self,
        market: MarketSnapshot,
        symbol: str,
        action: Action,
    ) -> float:
        funding = _market_funding_for_symbol(market, symbol)
        benefit = 0.0
        if action.is_long_open:
            benefit = -funding
        elif action.is_short_open:
            benefit = funding
        delta = benefit * float(self._config.funding_risk_mult_weight)
        cap = float(self._config.funding_risk_mult_cap)
        return max(0.0, 1.0 + _clamp(delta, -cap, cap))

    def _technical_atr_risk_mult(self, market: MarketSnapshot, symbol: str) -> float:
        tech = _technical_indicators_for_symbol(market, symbol)
        if tech is None or getattr(tech, "atr_14_pct", None) is None:
            return 1.0
        atr = max(
            float(self._config.technical_atr_min_pct),
            float(getattr(tech, "atr_14_pct")),
        )
        raw = float(self._config.technical_atr_target_pct) / atr
        return _clamp(raw, 0.0, float(self._config.technical_atr_max_mult))

    def _volatility_risk_mult(self, market: MarketSnapshot, symbol: str) -> float:
        volatility = _lookback_volatility_pct(
            market,
            symbol,
            self._config.overextension_lookback_bars,
        )
        if volatility is None:
            return 1.0
        min_vol = float(self._config.volatility_risk_min_volatility_pct)
        adjusted_vol = max(min_vol, float(volatility))
        raw = float(self._config.volatility_risk_target_pct) / adjusted_vol
        return _clamp(raw, 0.0, float(self._config.volatility_risk_max_mult))

    def _regime_feature_risk_mult(
        self,
        market: MarketSnapshot,
        symbol: str,
        action: Action,
    ) -> float:
        if not action.is_open:
            return 1.0
        features = _regime_features_for_symbol(market, symbol)
        if not features:
            return 1.0
        pressure = _finite_float_or_none(features.get("risk_pressure"))
        if pressure is None or pressure <= 1.0:
            return 1.0
        capped = min(float(pressure), float(self._config.regime_feature_risk_max_pressure))
        raw = 1.0 / capped
        return _clamp(raw, float(self._config.regime_feature_risk_min_mult), 1.0)

    def _suppressed_agent_labels_for_solo_players(
        self,
        player_rows: Sequence[_ActorSignal],
        *,
        regime: Regime,
        shadow_player_labels: Iterable[str],
        actionable_labels: Iterable[str],
    ) -> set[str]:
        if not (
            self._config.prefer_solo_player_wrappers_enabled
            or self._config.prefer_proven_solo_player_wrappers_enabled
        ):
            return set()
        solo_by_raw: dict[str, str] = {}
        solo_sources_by_raw: dict[str, set[str]] = {}

        def add_solo_source(raw_label: str, solo_label: str, source: str) -> None:
            if not raw_label:
                return
            solo_by_raw.setdefault(raw_label, solo_label)
            solo_sources_by_raw.setdefault(raw_label, set()).add(source)

        for row in player_rows:
            if row.actor_type != "ensemble":
                continue
            raw_label = _raw_agent_label_from_solo(row.label)
            if raw_label:
                add_solo_source(raw_label, row.label, "player")
        for label in shadow_player_labels:
            solo_label = str(label or "")
            raw_label = _raw_agent_label_from_solo(solo_label)
            if raw_label:
                add_solo_source(raw_label, solo_label, "shadow_player")
        for label in actionable_labels:
            solo_label = str(label or "")
            raw_label = _raw_agent_label_from_solo(solo_label)
            if raw_label:
                add_solo_source(raw_label, solo_label, "actionable")
        if self._config.prefer_solo_player_wrappers_enabled:
            return {
                raw_label
                for raw_label, solo_label in solo_by_raw.items()
                if not self._is_portfolio_actor(
                    _actor_key("agent", raw_label),
                    raw_label,
                )
                and (
                    "player" in solo_sources_by_raw.get(raw_label, set())
                    or "shadow_player" in solo_sources_by_raw.get(raw_label, set())
                    or self._solo_wrapper_has_trade_evidence(solo_label, regime)
                )
            }
        labels: set[str] = set()
        for raw_label, solo_label in solo_by_raw.items():
            if self._is_portfolio_actor(_actor_key("agent", raw_label), raw_label):
                continue
            if self._solo_wrapper_outscores_raw(
                raw_label,
                solo_label,
                regime=regime,
            ):
                labels.add(raw_label)
        return labels

    def _active_solo_wrapper_keys_for_suppression(
        self,
        player_rows: Sequence[_ActorSignal],
        *,
        market: MarketSnapshot,
        shadow_player_signals: Mapping[str, Sequence[Signal]],
    ) -> set[tuple[str, str]]:
        """Return raw-agent/symbol pairs where an admissible active Solo wrapper competes."""
        keys: set[tuple[str, str]] = set()

        def add_if_active(label: str, symbol: str, action: Action) -> None:
            raw_label = _raw_agent_label_from_solo(label)
            clean_symbol = str(symbol or "").strip().upper()
            if not raw_label or not clean_symbol or action.is_hold:
                return
            local_regime = market.regime_for_symbol(clean_symbol)
            if not self._solo_wrapper_can_suppress_raw(
                raw_label,
                str(label or "").strip(),
                regime=local_regime,
            ):
                return
            keys.add((raw_label, clean_symbol))

        for row in player_rows:
            if row.actor_type != "ensemble":
                continue
            add_if_active(row.label, row.symbol, row.action)

        for label, signals in shadow_player_signals.items():
            clean_label = str(label or "").strip()
            if not clean_label:
                continue
            for signal in signals or ():
                try:
                    action = _coerce_action(getattr(signal, "action", Action.HOLD))
                except (TypeError, ValueError):
                    continue
                add_if_active(clean_label, getattr(signal, "sym", ""), action)
        return keys

    def _admissible_solo_wrapper_keys_for_suppression(
        self,
        outputs: Sequence[_ActorSignal],
        *,
        market: MarketSnapshot,
        symbol: str,
        regime: Regime,
        degraded_actor_keys: set[str],
        degraded_open_symbols: set[str],
        promoted_signal_keys: set[str],
    ) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        clean_symbol = str(symbol or "").strip().upper()
        if not clean_symbol:
            return keys
        for output in outputs:
            if output.symbol != clean_symbol or output.actor_type != "ensemble":
                continue
            raw_label = _raw_agent_label_from_solo(output.label)
            if not raw_label or output.action.is_hold:
                continue
            if self._qm.is_quarantined(output.label):
                continue
            metrics, _base_score = self._score_actor(output, market)
            if output.action.is_open and not metrics.has_data:
                continue
            if self._is_open_regime_denied(output, regime):
                continue
            if self._is_open_symbol_denied(output, clean_symbol):
                continue
            if self._is_range_low_vol_actor_denied(
                output,
                regime,
                probation_admission=False,
            ):
                continue
            if self._is_open_symbol_degraded(output, clean_symbol, degraded_open_symbols):
                continue
            if output.action.is_open and self._global_health_rejected(output.label):
                continue
            if output.action.is_open and self._regime_edge_rejected(output.label, regime):
                continue
            if (
                output.action.is_open
                and metrics.closed_trades < self._config.min_closed_trades_to_trade
            ):
                continue
            pnl_gate_value = (
                metrics.pnl_per_trade
                if self._config.gate_pnl_per_trade_enabled
                else metrics.pnl_pct
            )
            if output.action.is_open and pnl_gate_value < self._config.min_pnl_pct_to_trade:
                continue
            if output.action.is_open and self._is_actor_degraded(output, degraded_actor_keys):
                continue
            if output.action.is_open and self._is_actor_denied(output):
                continue
            if self._is_signal_terminal_denied(output, clean_symbol, regime):
                continue
            if self._is_signal_denied(output, clean_symbol):
                continue
            if (
                self._config.promotion_manifest_enabled
                and output.action.is_open
                and not self._is_signal_promoted(output, clean_symbol, promoted_signal_keys)
            ):
                continue
            keys.add((raw_label, clean_symbol))
        return keys

    @staticmethod
    def _raw_agent_has_active_solo_wrapper(
        raw_label: str,
        symbol: str,
        active_solo_wrapper_keys: set[tuple[str, str]],
    ) -> bool:
        clean_label = str(raw_label or "").strip()
        clean_symbol = str(symbol or "").strip().upper()
        return bool(clean_label and clean_symbol) and (
            clean_label,
            clean_symbol,
        ) in active_solo_wrapper_keys

    def _solo_wrapper_has_trade_evidence(
        self,
        solo_label: str,
        regime: Regime,
    ) -> bool:
        metrics = self._direct_metrics_for_label(solo_label, regime)
        return bool(metrics.has_data)

    def _solo_wrapper_can_suppress_raw(
        self,
        raw_label: str,
        solo_label: str,
        *,
        regime: Regime,
    ) -> bool:
        if self._is_portfolio_actor(_actor_key("agent", raw_label), raw_label):
            return False
        if (
            self._config.prefer_proven_solo_player_wrappers_enabled
            and not self._config.prefer_solo_player_wrappers_enabled
        ):
            return self._solo_wrapper_outscores_raw(
                raw_label,
                solo_label,
                regime=regime,
            )
        metrics = self._direct_metrics_for_label(solo_label, regime)
        if not metrics.has_data:
            return False
        return metrics.closed_trades >= self._config.min_closed_trades_to_trade

    def _solo_wrapper_outscores_raw(
        self,
        raw_label: str,
        solo_label: str,
        *,
        regime: Regime,
    ) -> bool:
        solo_metrics = self._direct_metrics_for_label(solo_label, regime)
        if not solo_metrics.has_data:
            return False
        if solo_metrics.closed_trades < self._config.min_closed_trades_to_trade:
            return False
        raw_metrics = self._metrics_for_label(raw_label, regime)
        solo_score = self._score_metrics(solo_metrics, regime)
        raw_score = (
            self._score_metrics(raw_metrics, regime)
            if raw_metrics.has_data
            else float(self._config.no_data_score)
        )
        return (
            float(solo_score)
            >= float(raw_score) + float(self._config.proven_solo_min_score_advantage)
        )

    def _is_signal_denied(self, output: _ActorSignal, symbol: str) -> bool:
        if not self._config.denied_signal_keys:
            return False
        key = _signal_deny_key(output.actor_key, symbol, output.action)
        return key in self._config.denied_signal_keys

    def _is_actor_denied(self, output: _ActorSignal) -> bool:
        if not self._config.denied_actor_keys:
            return False
        denied = set(self._config.denied_actor_keys)
        return output.actor_key in denied or output.label in denied

    def _is_signal_terminal_denied(
        self,
        output: _ActorSignal,
        symbol: str,
        regime: Regime,
    ) -> bool:
        if not (
            self._config.terminal_denied_signal_keys
            or self._config.terminal_denied_context_signal_keys
        ):
            return False
        key = _signal_deny_key(output.actor_key, symbol, output.action)
        if key in self._config.terminal_denied_signal_keys:
            return True
        context_key = _signal_context_key(output.actor_key, symbol, output.action, regime)
        return context_key in self._config.terminal_denied_context_signal_keys

    def _terminal_denied_top_candidate(
        self,
        rows: Sequence[FlashCandidateAudit],
        bar: int,
    ) -> Optional[FlashCandidateAudit]:
        if not (
            self._config.terminal_denied_signal_keys
            or self._config.terminal_denied_context_signal_keys
        ) or not rows:
            return None
        top = min(
            rows,
            key=lambda row: (
                -row.score,
                -row.closed_trades,
                _stable_candidate_tiebreak(row, bar),
            ),
        )
        if top.reason != "flash_signal_terminal_deny_key":
            return None
        if not top.action.is_open:
            return None
        return top

    def _is_genetics_contra_no_backfill_candidate(
        self,
        output: _ActorSignal,
        confirmation_counts: Mapping[str, int],
    ) -> bool:
        return (
            self._config.genetics_confirmation_contra_no_backfill_enabled
            and self._genetics_confirmation_contra_count(
                output.action,
                confirmation_counts,
            )
            > 0
            and not self._is_genetics_confirmation_source(output)
        )

    def _genetics_contra_no_backfill_top_candidate(
        self,
        rows: Sequence[FlashCandidateAudit],
        bar: int,
    ) -> Optional[FlashCandidateAudit]:
        if (
            not self._config.genetics_confirmation_contra_no_backfill_enabled
            or not rows
        ):
            return None
        top = min(
            rows,
            key=lambda row: (
                -(float(row.score) - float(row.genetics_confirmation_adjustment)),
                -row.closed_trades,
                _stable_candidate_tiebreak(row, bar),
            ),
        )
        if top.reason != "genetics_contra_no_backfill":
            return None
        if not top.action.is_open:
            return None
        return top

    def _is_open_regime_denied(self, output: _ActorSignal, regime: Regime) -> bool:
        if not self._config.denied_open_regimes:
            return False
        return output.action.is_open and regime in self._config.denied_open_regimes

    def _is_open_symbol_denied(self, output: _ActorSignal, symbol: str) -> bool:
        if not self._config.denied_open_symbols:
            return False
        normalized = symbol.upper()
        base = normalized.split("/", 1)[0]
        denied = set(self._config.denied_open_symbols)
        return output.action.is_open and (normalized in denied or base in denied)

    def _is_range_low_vol_actor_denied(
        self,
        output: _ActorSignal,
        regime: Regime,
        *,
        probation_admission: bool,
    ) -> bool:
        allowed = set(self._config.range_low_vol_real_actor_allowlist)
        if not allowed or not output.action.is_open:
            return False
        if regime != Regime.RANGE_LOW_VOL:
            return False
        if self._is_flash_genetics_core_primary_actor(output):
            return False
        if output.actor_key in allowed or output.label in allowed:
            return False
        return not probation_admission

    def _is_flash_genetics_core_primary_actor(self, output: _ActorSignal) -> bool:
        if not bool(self._config.flash_genetics_core_primary_enabled):
            return False
        primary_labels = set(self._config.flash_genetics_core_primary_labels)
        if not primary_labels:
            return False
        variants: set[str] = set()
        for raw in (output.label, output.actor_key):
            clean = str(raw or "").strip()
            if not clean:
                continue
            variants.add(clean)
            suffix = clean.split(":", 1)[1] if ":" in clean else clean
            variants.add(suffix)
            if suffix.startswith("Solo_"):
                variants.add(suffix.removeprefix("Solo_"))
        return bool(variants & primary_labels)

    @staticmethod
    def _is_open_symbol_degraded(
        output: _ActorSignal,
        symbol: str,
        degraded_open_symbols: set[str],
    ) -> bool:
        if not degraded_open_symbols:
            return False
        normalized = symbol.upper()
        base = normalized.split("/", 1)[0]
        return output.action.is_open and (
            normalized in degraded_open_symbols or base in degraded_open_symbols
        )

    def _is_base_fallback_allowed(self, output: _ActorSignal) -> bool:
        allowed = set(self._config.shadow_base_fallback_actor_keys)
        if not allowed:
            return False
        return output.actor_key in allowed or output.label in allowed

    def _min_score_to_trade_for(self, output: _ActorSignal) -> float:
        if (
            self._config.anchor_min_score_to_trade is not None
            and self._is_anchor_actor(output.actor_key, output.label)
        ):
            return float(self._config.anchor_min_score_to_trade)
        return float(self._config.min_score_to_trade)

    def _shadow_min_score_for(self, output: _ActorSignal) -> float:
        if (
            self._config.anchor_shadow_min_score is not None
            and self._is_anchor_actor(output.actor_key, output.label)
        ):
            return float(self._config.anchor_shadow_min_score)
        return float(self._config.shadow_confirmation_min_score)

    def _actor_fallback_min_base_score_for(self, output: _ActorSignal) -> float:
        base_floor = float(self._config.shadow_actor_fallback_min_base_score)
        replay_floor = float(
            self._config.shadow_position_replay_actor_fallback_min_base_score
        )
        if replay_floor > 0.0 and _is_shadow_position_replay_output(output):
            return max(base_floor, replay_floor)
        return base_floor

    def _actor_fallback_shadow_score_allowed(
        self,
        output: _ActorSignal,
        actor_shadow: _ShadowConfirmation,
    ) -> bool:
        floor = float(
            self._config.shadow_position_replay_actor_fallback_min_shadow_score
        )
        if floor <= 0.0 or not _is_shadow_position_replay_output(output):
            return True
        return actor_shadow.score >= floor

    def _shadow_recovery_candidate_active(
        self,
        output: _ActorSignal,
        actionable: set[str],
    ) -> bool:
        if not (
            self._config.shadow_recovery_admission_enabled
            and output.action.is_open
            and self._config.shadow_confirmation_enabled
        ):
            return False
        if not self._shadow_recovery_actor_allowed(output):
            return False
        actionable_labels = {
            str(label or "").strip()
            for label in actionable or set()
            if str(label or "").strip()
        }
        if not actionable_labels:
            return False
        actor_labels = set(_solo_label_lookup_sequence(output.label))
        actor_labels.add(output.label)
        actor_labels.add(output.actor_key)
        raw_label = _raw_agent_label_from_solo(output.label)
        if raw_label:
            actor_labels.add(raw_label)
        else:
            solo_label = f"Solo_{str(output.label or '').strip()}"
            if solo_label.strip() != "Solo_":
                actor_labels.add(solo_label)
        return any(label in actionable_labels for label in actor_labels)

    def _shadow_recovery_actor_allowed(self, output: _ActorSignal) -> bool:
        allowed = {
            str(item or "").strip()
            for item in self._config.shadow_recovery_actor_keys
            if str(item or "").strip()
        }
        if not allowed:
            return True
        aliases = set(_solo_label_lookup_sequence(output.label))
        aliases.add(output.label)
        aliases.add(output.actor_key)
        if not _raw_agent_label_from_solo(output.label):
            solo_label = f"Solo_{str(output.label or '').strip()}"
            if solo_label.strip() != "Solo_":
                aliases.add(solo_label)
                aliases.add(_actor_key("ensemble", solo_label))
        for label in tuple(aliases):
            aliases.add(_actor_key(output.actor_type, label))
            raw_label = _raw_agent_label_from_solo(label)
            if raw_label:
                aliases.add(raw_label)
                aliases.add(_actor_key("agent", raw_label))
        return bool(aliases & allowed)

    def _shadow_recovery_admission_active(
        self,
        output: _ActorSignal,
        actionable: set[str],
        shadow: _ShadowConfirmation,
        *,
        shadow_min_score: float,
    ) -> bool:
        if not self._shadow_recovery_candidate_active(output, actionable):
            return False
        min_closed = int(self._config.shadow_recovery_min_closed_trades or 0)
        if min_closed <= 0:
            min_closed = int(self._config.shadow_confirmation_min_closed_trades)
        min_score = self._config.shadow_recovery_min_score
        if min_score is None:
            min_score = float(shadow_min_score)
        return (
            int(shadow.closed_trades or 0) >= max(0, min_closed)
            and float(shadow.score or 0.0) > float(min_score)
        )

    def _session_recovery_override_active(self, output: _ActorSignal) -> bool:
        record_for = getattr(self._qm, "record_for", None)
        if not callable(record_for):
            return False
        labels: List[str] = []
        for raw_label in (
            output.label,
            *tuple(getattr(output, "agent_labels", ()) or ()),
        ):
            for label in _solo_label_lookup_sequence(str(raw_label or "")):
                if label and label not in labels:
                    labels.append(label)
        for label in labels:
            record = record_for(label)
            if (
                record is not None
                and str(getattr(record, "state", "") or "") == "probation"
                and str(getattr(record, "reason", "") or "").startswith(
                    "session_recovery:"
                )
            ):
                return True
        return False

    def _shadow_confirmation_meets_minimum(
        self,
        shadow: _ShadowConfirmation,
        shadow_min_score: float,
    ) -> bool:
        return (
            self._config.shadow_confirmation_enabled
            and shadow.closed_trades
            >= self._config.shadow_confirmation_min_closed_trades
            and shadow.score > shadow_min_score
        )

    def _shadow_pnl_per_trade_lcb_penalty(
        self,
        action: Action,
        shadow_pnl_per_trade_lcb_usd: Optional[float],
    ) -> float:
        if not (
            self._config.shadow_confirmation_enabled
            and self._config.shadow_quality_confirmation_enabled
            and action.is_open
        ):
            return 0.0
        weight = float(
            self._config.shadow_confirmation_pnl_per_trade_lcb_penalty_weight
            or 0.0
        )
        if weight <= 0.0 or shadow_pnl_per_trade_lcb_usd is None:
            return 0.0
        floor = float(
            self._config.shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd
            or 0.0
        )
        gap = floor - float(shadow_pnl_per_trade_lcb_usd)
        if gap <= 0.0:
            return 0.0
        return float(gap * weight)

    def _shadow_symbol_health_for(
        self,
        shadow_confirmation: Mapping[object, object],
        *,
        symbol: str,
        action: Action,
    ) -> _ShadowConfirmation:
        if not (
            self._config.shadow_confirmation_enabled
            and self._config.shadow_symbol_health_enabled
        ):
            return _ShadowConfirmation()
        raw = shadow_confirmation.get(_shadow_symbol_health_key(symbol, action))
        return _shadow_confirmation_from_raw(raw)

    def _shadow_symbol_health_penalty(
        self,
        action: Action,
        symbol_health: _ShadowConfirmation,
        symbol_health_pnl_per_trade_lcb_usd: Optional[float],
        *,
        actor_pnl_per_trade_lcb_usd: Optional[float],
    ) -> float:
        if not (
            self._config.shadow_confirmation_enabled
            and self._config.shadow_symbol_health_enabled
            and action.is_open
        ):
            return 0.0
        if (
            symbol_health.closed_trades
            < self._config.shadow_symbol_health_min_closed_trades
        ):
            return 0.0
        weight = float(
            self._config.shadow_symbol_health_pnl_per_trade_lcb_penalty_weight
            or 0.0
        )
        if weight <= 0.0 or symbol_health_pnl_per_trade_lcb_usd is None:
            return 0.0
        floor = float(
            self._config.shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd
            or 0.0
        )
        if (
            actor_pnl_per_trade_lcb_usd is not None
            and float(actor_pnl_per_trade_lcb_usd) >= floor
        ):
            return 0.0
        gap = floor - float(symbol_health_pnl_per_trade_lcb_usd)
        if gap <= 0.0:
            return 0.0
        return float(gap * weight)

    def _anchor_dominance_candidate(
        self,
        rows: Sequence[FlashCandidateAudit],
        selected: FlashCandidateAudit,
    ) -> Optional[FlashCandidateAudit]:
        if not self._config.anchor_actor_keys:
            return None
        if self._is_anchor_actor(selected.actor_key, selected.label):
            return None
        anchor = next(
            (
                row
                for row in rows
                if not row.rejected and self._is_anchor_actor(row.actor_key, row.label)
            ),
            None,
        )
        if anchor is None:
            return None
        if selected.score - anchor.score < self._config.anchor_min_score_advantage:
            return anchor
        return None

    @staticmethod
    def _top_rejected_positive_shadow_candidate(
        rows: Sequence[FlashCandidateAudit],
    ) -> Optional[FlashCandidateAudit]:
        for row in rows:
            if not bool(getattr(row, "rejected", False)):
                continue
            if _is_no_trade_candidate(row):
                continue
            action = getattr(row, "action", Action.HOLD)
            if not bool(getattr(action, "is_open", False)):
                continue
            if (
                float(getattr(row, "shadow_score", 0.0) or 0.0) > 0.0
                or float(getattr(row, "score", 0.0) or 0.0) > 0.0
                or float(getattr(row, "pnl_net_pct", 0.0) or 0.0) > 0.0
            ):
                return row
        return None

    def _is_anchor_actor(self, actor_key: str, label: str) -> bool:
        anchors = set(self._config.anchor_actor_keys)
        if not anchors:
            return False
        return actor_key in anchors or label in anchors

    def _is_portfolio_actor(self, actor_key: str, label: str) -> bool:
        actors = set(self._config.portfolio_actor_keys)
        if not actors:
            return False
        return actor_key in actors or label in actors

    @staticmethod
    def _is_signal_degraded(
        output: _ActorSignal,
        symbol: str,
        degraded_signal_keys: set[str],
    ) -> bool:
        if not degraded_signal_keys:
            return False
        key = _signal_deny_key(output.actor_key, symbol, output.action)
        return key in degraded_signal_keys

    @staticmethod
    def _is_actor_degraded(
        output: _ActorSignal,
        degraded_actor_keys: set[str],
    ) -> bool:
        if not degraded_actor_keys:
            return False
        return output.actor_key in degraded_actor_keys

    @staticmethod
    def _is_signal_promoted(
        output: _ActorSignal,
        symbol: str,
        promoted_signal_keys: set[str],
    ) -> bool:
        if not promoted_signal_keys:
            return False
        key = _signal_deny_key(output.actor_key, symbol, output.action)
        return key in promoted_signal_keys

    def _overextension_reason(
        self,
        action: Action,
        lookback_return_pct: Optional[float],
        lookback_return_z: Optional[float],
    ) -> str:
        if not self._config.open_overextension_guard_enabled:
            return ""
        if lookback_return_pct is None or not action.is_open:
            return ""
        if (
            self._config.overextension_volatility_normalized_enabled
            and lookback_return_z is not None
        ):
            if (
                action.is_short_open
                and lookback_return_z <= self._config.short_overextension_z_floor
            ):
                return "short_overextended_downmove"
            if (
                action.is_long_open
                and lookback_return_z >= self._config.long_overextension_z_ceiling
            ):
                return "long_overextended_upmove"
            return ""
        if (
            action.is_short_open
            and lookback_return_pct <= self._config.short_overextension_return_floor_pct
        ):
            return "short_overextended_downmove"
        if (
            action.is_long_open
            and lookback_return_pct >= self._config.long_overextension_return_ceiling_pct
        ):
            return "long_overextended_upmove"
        return ""

    def _apply_mixed_rotational_exposure_cap(
        self,
        market: MarketSnapshot,
        decisions: Tuple[FlashDecision, ...],
    ) -> Tuple[FlashDecision, ...]:
        if (
            not self._config.mixed_rotational_exposure_cap_enabled
            or market.regime != Regime.MIXED_ROTATIONAL
        ):
            return decisions
        cap = int(self._config.mixed_rotational_max_directional_opens)
        min_confidence = float(self._config.mixed_rotational_min_symbol_confidence)
        out: List[FlashDecision] = list(decisions)
        directional: List[tuple[int, FlashDecision]] = []
        for index, decision in enumerate(decisions):
            signal = decision.signal
            if signal is None or not signal.action.is_open:
                continue
            local_regime = market.regime_for_symbol(decision.symbol)
            local_shadow_allowed = (
                self._mixed_rotational_symbol_local_shadow_gate_allows(
                    decision,
                    local_regime,
                )
            )
            if (
                local_regime not in {Regime.BULLISH, Regime.BEARISH}
                and not local_shadow_allowed
            ):
                out[index] = _blocked_flash_decision(
                    decision,
                    reason="mixed_rotational_symbol_not_directional",
                )
                continue
            features = _regime_features_for_symbol(market, decision.symbol)
            confidence = _finite_float_or_none(features.get("regime_confidence"))
            if confidence is None:
                confidence = 1.0
            if confidence < min_confidence:
                out[index] = _blocked_flash_decision(
                    decision,
                    reason="mixed_rotational_symbol_low_confidence",
                )
                continue
            directional.append((index, decision))
        if cap < 0 or len(directional) <= cap:
            return tuple(out)
        keep = {
            index
            for index, _ in sorted(
                directional,
                key=lambda item: (-float(item[1].score), item[1].symbol),
            )[:cap]
        }
        for index, decision in directional:
            if index in keep:
                continue
            out[index] = _blocked_flash_decision(
                decision,
                reason="mixed_rotational_exposure_cap",
            )
        return tuple(out)

    def _mixed_rotational_symbol_local_shadow_gate_allows(
        self,
        decision: FlashDecision,
        local_regime: Regime,
    ) -> bool:
        if not self._config.mixed_rotational_symbol_local_shadow_gate_enabled:
            return False
        if local_regime not in {Regime.RANGE_LOW_VOL, Regime.BEARISH}:
            return False
        row = _selected_candidate_audit(decision)
        if row is None:
            return False
        shadow_score = float(getattr(row, "shadow_score", 0.0) or 0.0)
        shadow_closed = int(getattr(row, "shadow_closed_trades", 0) or 0)
        return (
            shadow_score
            >= float(self._config.mixed_rotational_symbol_local_shadow_min_score)
            and shadow_closed
            >= int(
                self._config.mixed_rotational_symbol_local_shadow_min_closed_trades
            )
        )

    def _apply_actor_signal_cap(
        self,
        decisions: Tuple[FlashDecision, ...],
    ) -> Tuple[FlashDecision, ...]:
        cap = int(self._config.max_signals_per_actor or 0)
        override_caps = _actor_cap_override_map(
            self._config.promoted_actor_cap_overrides
        )
        if cap <= 0 and not override_caps:
            return decisions
        counts: dict[str, int] = {}
        out: List[FlashDecision] = list(decisions)
        for decision in decisions:
            actor = _actor_key(decision.actor_type, str(decision.selected_actor or ""))
            if actor == "NoTrade" or decision.signal is None:
                reserved = self._degraded_actor_cap_reservation(decision)
                if reserved:
                    counts[reserved] = counts.get(reserved, 0) + 1
                continue

        selectable: List[tuple[int, str, int, FlashDecision]] = []
        for index, decision in enumerate(decisions):
            actor = _actor_key(decision.actor_type, str(decision.selected_actor or ""))
            if actor == "NoTrade" or decision.signal is None:
                continue
            actor_cap = override_caps.get(actor, cap)
            if actor_cap <= 0:
                continue
            selectable.append((index, actor, actor_cap, decision))

        for index, actor, actor_cap, decision in sorted(
            selectable,
            key=lambda item: (item[1], -float(item[3].score), item[3].symbol),
        ):
            count = counts.get(actor, 0)
            if count >= actor_cap:
                out[index] = replace(
                    decision,
                    selected_actor="NoTrade",
                    actor_type="no_trade",
                    score=0.0,
                    action=Action.HOLD,
                    reason="actor_signal_cap",
                    signal=None,
                    original_selected_actor=(
                        decision.original_selected_actor or decision.selected_actor
                    ),
                    original_actor_type=(
                        decision.original_actor_type or decision.actor_type
                    ),
                )
                continue
            counts[actor] = count + 1
        return tuple(out)

    def _degraded_actor_cap_reservation(self, decision: FlashDecision) -> str:
        if not self._config.degradation_reserve_actor_cap:
            return ""
        for row in getattr(decision, "candidates", ()) or ():
            if str(getattr(row, "reason", "") or "") != "flash_signal_degraded":
                continue
            return str(getattr(row, "actor_key", "") or _actor_key(row.actor_type, row.label))
        return ""

    def _component_score(
        self,
        labels: Sequence[str],
        regime: Regime,
        *,
        use_aggregate: bool = False,
        symbol: str = "",
    ) -> tuple[Metrics, float]:
        scored: List[tuple[Metrics, float]] = []
        for label in labels:
            if symbol:
                metrics = self._metrics_for_label(label, regime, symbol=symbol)
            elif use_aggregate:
                metrics = self._perf.get(label)
            else:
                metrics = self._metrics_for_label(label, regime)
            if metrics.has_data:
                scored.append((metrics, self._score_metrics(metrics, regime)))
        if not scored:
            return Metrics.empty(), float(self._config.no_data_score)
        # Счётчики — суммы (наследование допуска от компонентов, B4; корректный
        # win_rate из суммарных wins/losses).
        closed = sum(row[0].closed_trades for row in scored)
        signals = sum(row[0].signals for row in scored)
        entries = sum(row[0].entries for row in scored)
        wins = sum(row[0].wins for row in scored)
        losses = sum(row[0].losses for row in scored)

        # Трейд-взвешенное среднее: вес = closed_trades компонента (fallback —
        # равные веса). Это делает синтетический ансамбль сопоставимым с одним
        # актором и не умножает издержки ~N× (B2).
        weights = [max(row[0].closed_trades, 0) for row in scored]
        wsum = sum(weights)

        def _wavg(values: List[float]) -> float:
            if wsum > 0:
                return sum(v * w for v, w in zip(values, weights)) / wsum
            return sum(values) / len(values)

        component_scores = [row[1] for row in scored]
        score_mean = _wavg(component_scores)
        bias = max(0.0, min(1.0, float(self._config.component_score_max_bias)))
        # max-смещённый скор: ансамбль не ниже своего лучшего компонента (B1).
        score = bias * max(component_scores) + (1.0 - bias) * score_mean

        pnl = _wavg([row[0].pnl_pct for row in scored])
        pnl_gross = _wavg(
            [float(getattr(row[0], "pnl_gross_pct", row[0].pnl_pct)) for row in scored]
        )
        fee_pct = _wavg([float(getattr(row[0], "fee_pct", 0.0)) for row in scored])
        funding_pct = _wavg(
            [float(getattr(row[0], "funding_pct", 0.0)) for row in scored]
        )
        # Просадка — взвешенное среднее (репрезентативно), а не худшая по компонентам.
        max_dd = _wavg([float(row[0].max_dd_pct) for row in scored])
        metrics = Metrics(
            pnl_pct=pnl,
            closed_trades=closed,
            entries=entries,
            signals=signals,
            wins=wins,
            losses=losses,
            max_dd_pct=max_dd,
            pnl_gross_pct=pnl_gross,
            fee_pct=fee_pct,
            funding_pct=funding_pct,
        )
        return metrics, score

    @staticmethod
    def _shadow_confirmation_for(
        label: str,
        shadow_confirmation: Mapping[object, object],
        *,
        symbol: str = "",
        action: Action = Action.HOLD,
        require_symbol: bool = False,
    ) -> _ShadowConfirmation:
        for lookup_label in _solo_label_lookup_sequence(label):
            raw = _shadow_confirmation_raw(
                lookup_label,
                shadow_confirmation,
                symbol=symbol,
                action=action,
                require_symbol=require_symbol,
            )
            if raw is not None:
                return _shadow_confirmation_from_raw(raw)
        return _ShadowConfirmation()

    def _global_health_rejected(self, label: str) -> bool:
        """Вето селекции по агрегатному (по всем режимам) здоровью актора.

        Возвращает True, если глобально актор катастрофичен (кумулятивный PnL
        ниже порога или просадка выше порога) при достаточной выборке. Это не
        даёт глобально-убыточному актору выигрывать узкие (symbol,regime)-срезы.
        """
        cfg = self._config
        if not cfg.global_health_gate_enabled:
            return False
        agg: Optional[Metrics] = None
        for lookup_label in _solo_label_lookup_sequence(label):
            candidate = self._perf.get(lookup_label)
            if candidate.has_data:
                agg = candidate
                break
        if agg is None or agg.closed_trades < cfg.global_health_min_closed_trades:
            return False
        if agg.pnl_pct <= cfg.global_health_min_cum_pnl_pct:
            return True
        if abs(agg.max_dd_pct) >= cfg.global_health_max_drawdown_pct:
            return True
        return False

    def _not_in_live_whitelist(self, output: object) -> bool:
        """Single-agent whitelist (#1): не открывать реально, если актор не в списке.

        Матчит по чистому лейблу, actor_key и solo-lookup-последовательности.
        """
        wl = self._config.live_real_actor_whitelist
        if not wl:
            return False
        allowed = {str(w or "").strip() for w in wl if str(w or "").strip()}
        if not allowed:
            return False
        label = str(getattr(output, "label", "") or "")
        actor_key = str(getattr(output, "actor_key", "") or "")
        candidates = {label, actor_key}
        for lk in _solo_label_lookup_sequence(label):
            candidates.add(str(lk))
        # Также сравниваем «голые» имена без префиксов agent:/ensemble:.
        bare = {c.split(":", 1)[-1] for c in candidates if c}
        return not (candidates & allowed or bare & allowed or {b for b in bare} & {a.split(":",1)[-1] for a in allowed})

    def _real_loss_rejected(self, label: str) -> bool:
        """Вето открытия по РЕАЛЬНОЙ realized убыточности актора (root fix: shadow≠real)."""
        cfg = self._config
        if not cfg.real_loss_gate_enabled or self._real_perf is None:
            return False
        agg: Optional[Metrics] = None
        for lookup_label in _solo_label_lookup_sequence(label):
            candidate = self._real_perf.get(lookup_label)
            if candidate.has_data:
                agg = candidate
                break
        if agg is None or agg.closed_trades < cfg.real_loss_gate_min_closed_trades:
            return False
        return agg.pnl_pct <= cfg.real_loss_gate_min_pnl_pct

    def _fee_aware_rejected(self, metrics: Metrics) -> bool:
        """Fee-aware admission: ожидаемый per-trade edge < round-trip издержек × safety."""
        cfg = self._config
        if not cfg.fee_aware_admission_enabled:
            return False
        if not metrics.has_data or metrics.closed_trades <= 0:
            return False
        expected_edge = float(metrics.pnl_per_trade)
        min_edge = cfg.fee_aware_round_trip_cost_pct * cfg.fee_aware_safety_mult
        return expected_edge < min_edge

    def _churn_rejected(self, output: object, symbol: str, bar: int) -> bool:
        """Churn limiter: cooldown по символу, дневной лимит открытий, anti-flip."""
        cfg = self._config
        if cfg.churn_cooldown_bars <= 0 and cfg.churn_max_opens_per_day <= 0 and not cfg.churn_anti_flip_enabled:
            return False
        sym = str(symbol or "").upper()
        action = getattr(output, "action", None)
        side = "long" if getattr(action, "is_long_open", False) else "short"
        bar = int(bar)
        last_bar = self._last_open_bar_by_symbol.get(sym)
        # Cooldown между открытиями по символу.
        if cfg.churn_cooldown_bars > 0 and last_bar is not None and (bar - last_bar) < cfg.churn_cooldown_bars:
            return True
        # Anti-flip: запрет немедленной смены стороны по символу.
        if cfg.churn_anti_flip_enabled and last_bar is not None and (bar - last_bar) < max(1, cfg.churn_cooldown_bars):
            if self._last_open_side_by_symbol.get(sym) not in (None, side):
                return True
        # Дневной лимит открытий (день = bar // 24 при часовом страйде).
        if cfg.churn_max_opens_per_day > 0:
            day = bar // 24
            if day != self._opens_today_day:
                self._opens_today_day = day
                self._opens_today_count = 0
            if self._opens_today_count >= cfg.churn_max_opens_per_day:
                return True
        return False

    def _churn_record_open(self, symbol: str, side: str, bar: int) -> None:
        sym = str(symbol or "").upper()
        self._last_open_bar_by_symbol[sym] = int(bar)
        self._last_open_side_by_symbol[sym] = side
        day = int(bar) // 24
        if day != self._opens_today_day:
            self._opens_today_day = day
            self._opens_today_count = 0
        self._opens_today_count += 1

    def _stable_regime_edge(self, label: str, regime: Regime) -> float:
        """Стабильный per-regime эдж актора = pnl_per_trade по всем символам
        режима (медленно меняется, не шумит как per-symbol score).
        """
        for lookup_label in _solo_label_lookup_sequence(label):
            metrics = self._perf.get(lookup_label, regime=regime)
            if metrics.has_data:
                return float(metrics.pnl_per_trade)
        return 0.0

    def _regime_edge_rejected(self, label: str, regime: Regime) -> bool:
        """Вето открытия, если у актора нет положительного per-trade эджа в ТЕКУЩЕМ
        режиме (при достаточной выборке). Эдж регим-специфичен — не торгуем
        агентом там, где он исторически не зарабатывает.
        """
        cfg = self._config
        if not cfg.regime_edge_gate_enabled:
            return False
        metrics: Optional[Metrics] = None
        for lookup_label in _solo_label_lookup_sequence(label):
            candidate = self._perf.get(lookup_label, regime=regime)
            if candidate.has_data:
                metrics = candidate
                break
        if metrics is None or metrics.closed_trades < cfg.regime_edge_min_closed_trades:
            return False
        return metrics.pnl_per_trade <= cfg.regime_edge_min_pnl_per_trade_pct

    def _metrics_for_label(
        self,
        label: str,
        regime: Regime,
        *,
        symbol: str = "",
    ) -> Metrics:
        lookup_labels = _solo_label_lookup_sequence(label)
        clean_symbol = str(symbol or "").strip().upper()
        if clean_symbol:
            for lookup_label in lookup_labels:
                metrics = self._perf.get(
                    lookup_label,
                    regime=regime,
                    symbol=clean_symbol,
                )
                if metrics.has_data:
                    return metrics
            return self._perf.get(
                str(label or "").strip(),
                regime=regime,
                symbol=clean_symbol,
            )
        for lookup_label in lookup_labels:
            metrics = self._perf.get(lookup_label, regime=regime)
            if metrics.has_data:
                return metrics
        for lookup_label in lookup_labels:
            metrics = self._perf.get(lookup_label)
            if metrics.has_data:
                return metrics
        return self._perf.get(str(label or "").strip())

    def _direct_metrics_for_label(self, label: str, regime: Regime) -> Metrics:
        clean_label = str(label or "").strip()
        metrics = self._perf.get(clean_label, regime=regime)
        if metrics.has_data:
            return metrics
        return self._perf.get(clean_label)

    @staticmethod
    def _agent_outputs(market: MarketSnapshot, agents: Sequence[Agent]) -> List[_ActorSignal]:
        rows: List[_ActorSignal] = []
        for agent in agents:
            label = str(getattr(agent, "label", "") or "")
            if not label:
                continue
            for symbol in market.prices:
                symbol_market = market.with_regime_for_symbol(symbol)
                try:
                    actions = agent.act(symbol_market) or {}
                except Exception:
                    actions = {}
                action = _coerce_action(actions.get(symbol, Action.HOLD))
                signal = None
                if not action.is_hold:
                    signal = Signal(
                        id=0,
                        bar=market.bar,
                        sym=str(symbol).upper(),
                        action=action,
                        price=float(market.prices.get(symbol, 0.0)),
                        regime=symbol_market.regime,
                        by_player=label,
                        by_agent="",
                        timestamp=market.timestamp,
                    )
                rows.append(_ActorSignal(
                    symbol=str(symbol).upper(),
                    label=label,
                    actor_type="agent",
                    actor_key=_actor_key("agent", label),
                    action=action,
                    signal=signal,
                    agent_labels=(label,),
                ))
        return rows

    @staticmethod
    def _no_trade_outputs(market: MarketSnapshot) -> List[_ActorSignal]:
        return [
            _ActorSignal(
                symbol=str(symbol).upper(),
                label="NoTrade",
                actor_type="no_trade",
                actor_key="NoTrade",
                action=Action.HOLD,
                signal=None,
                agent_labels=(),
            )
            for symbol in market.prices
        ]

    @staticmethod
    def _player_outputs(
        market: MarketSnapshot,
        players: Sequence[Player],
        *,
        signal_id_start: int,
    ) -> List[_ActorSignal]:
        rows: List[_ActorSignal] = []
        for player in players:
            label = str(getattr(player, "label", "") or "")
            if not label:
                continue
            try:
                signals, _errors = normalize_vote_result(
                    player.vote(market, signal_id_start=signal_id_start)
                )
                signals = tuple(signals)
            except Exception:
                signals = ()
            by_symbol = {str(sig.sym).upper(): sig for sig in signals}
            actor_type = str(getattr(player, "actor_type", "") or "ensemble")
            agent_labels = tuple(str(item) for item in getattr(player, "agent_labels", ()) or ())
            for symbol in market.prices:
                clean = str(symbol).upper()
                signal = by_symbol.get(clean)
                action = signal.action if signal is not None else Action.HOLD
                rows.append(_ActorSignal(
                    symbol=clean,
                    label=label,
                    actor_type=actor_type,
                    actor_key=_actor_key(actor_type, label),
                    action=action,
                    signal=signal,
                    agent_labels=agent_labels,
                ))
        return rows

    def _shadow_signal_outputs(
        self,
        market: MarketSnapshot,
        existing_outputs: Sequence[_ActorSignal],
        *,
        known_outputs: Sequence[_ActorSignal] = (),
        shadow_player_signals: Mapping[str, Sequence[Signal]],
        shadow_agent_signals: Mapping[str, Sequence[Signal]],
        suppressed_agent_labels: set[str],
        active_solo_wrapper_keys: set[tuple[str, str]],
    ) -> List[_ActorSignal]:
        if not self._config.shadow_signal_handoff_enabled:
            return []

        existing_active = {
            (row.actor_key, row.symbol)
            for row in existing_outputs
            if not row.action.is_hold
        }
        known_actor_keys = {
            row.actor_key
            for row in tuple(existing_outputs) + tuple(known_outputs)
            if str(row.actor_key or "").strip()
        }
        used = set(existing_active)
        rows: List[_ActorSignal] = []

        for label, signals in shadow_player_signals.items():
            clean_label = str(label or "").strip()
            if not clean_label:
                continue
            actor_key = _actor_key("ensemble", clean_label)
            if actor_key not in known_actor_keys and not self._known_solo_wrapper_label(
                clean_label,
                known_actor_keys,
            ):
                continue
            for signal in signals or ():
                row = _shadow_signal_output(
                    market,
                    clean_label,
                    "ensemble",
                    actor_key,
                    signal,
                    agent_labels=(
                        (str(signal.by_agent),)
                        if str(getattr(signal, "by_agent", "") or "").strip()
                        else ()
                    ),
                    by_player=clean_label,
                    by_agent=str(getattr(signal, "by_agent", "") or "").strip(),
                )
                if row is None or (row.actor_key, row.symbol) in used:
                    continue
                used.add((row.actor_key, row.symbol))
                rows.append(row)

        for label, signals in shadow_agent_signals.items():
            clean_label = str(label or "").strip()
            if not clean_label:
                continue
            actor_key = _actor_key("agent", clean_label)
            if actor_key not in known_actor_keys:
                continue
            for signal in signals or ():
                if (
                    clean_label in suppressed_agent_labels
                    and self._raw_agent_has_active_solo_wrapper(
                        clean_label,
                        _market_symbol_for_signal(market, getattr(signal, "sym", ""))
                        or getattr(signal, "sym", ""),
                        active_solo_wrapper_keys,
                    )
                ):
                    continue
                row = _shadow_signal_output(
                    market,
                    clean_label,
                    "agent",
                    actor_key,
                    signal,
                    agent_labels=(),
                    by_player=clean_label,
                    by_agent="",
                )
                if row is None or (row.actor_key, row.symbol) in used:
                    continue
                used.add((row.actor_key, row.symbol))
                rows.append(row)
        return rows

    def _known_solo_wrapper_label(
        self,
        label: str,
        known_actor_keys: set[str],
    ) -> bool:
        if not (
            self._config.prefer_solo_player_wrappers_enabled
            or self._config.prefer_proven_solo_player_wrappers_enabled
        ):
            return False
        raw_label = _raw_agent_label_from_solo(label)
        if not raw_label:
            return False
        return _actor_key("agent", raw_label) in known_actor_keys


def _coerce_action(value: object) -> Action:
    if isinstance(value, Action):
        return value
    try:
        return Action(int(value))
    except (TypeError, ValueError):
        return Action.HOLD


def _open_action_direction(action: Action) -> str:
    if action.is_long_open:
        return "long"
    if action.is_short_open:
        return "short"
    return ""


def _shadow_signal_output(
    market: MarketSnapshot,
    label: str,
    actor_type: str,
    actor_key: str,
    signal: Signal,
    *,
    agent_labels: Tuple[str, ...],
    by_player: str,
    by_agent: str,
) -> Optional[_ActorSignal]:
    symbol = _market_symbol_for_signal(market, getattr(signal, "sym", ""))
    if not symbol:
        return None
    price = _market_price_for_symbol(market, symbol)
    if price is None:
        return None
    action = _coerce_action(getattr(signal, "action", Action.HOLD))
    if action.is_hold:
        return None
    clean_signal = replace(
        signal,
        id=0,
        sym=symbol,
        action=action,
        price=price,
        by_player=by_player,
        by_agent=by_agent,
    )
    return _ActorSignal(
        symbol=symbol,
        label=label,
        actor_type=actor_type,
        actor_key=actor_key,
        action=action,
        signal=clean_signal,
        agent_labels=agent_labels,
    )


def _stable_candidate_tiebreak(row: FlashCandidateAudit, bar: int) -> int:
    payload = f"{row.label}|{row.actor_type}|{int(bar)}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _is_no_trade_output(output: _ActorSignal) -> bool:
    return str(getattr(output, "actor_key", "") or "") == "NoTrade" or (
        str(getattr(output, "actor_type", "") or "") == "no_trade"
        and str(getattr(output, "label", "") or "") == "NoTrade"
    )


def _is_no_trade_candidate(row: FlashCandidateAudit) -> bool:
    return str(getattr(row, "actor_key", "") or "") == "NoTrade" or (
        str(getattr(row, "actor_type", "") or "") == "no_trade"
        and str(getattr(row, "label", "") or "") == "NoTrade"
    )


def _is_shadow_position_replay_output(output: _ActorSignal) -> bool:
    signal = getattr(output, "signal", None)
    return (
        str(getattr(signal, "by_agent", "") or "").strip()
        == _SHADOW_POSITION_REPLAY_AGENT
    )


def _actor_key(actor_type: str, label: str) -> str:
    if label == "NoTrade":
        return "NoTrade"
    clean_type = str(actor_type or "unknown").strip() or "unknown"
    clean_label = str(label or "").strip()
    return f"{clean_type}:{clean_label}"


def _selected_candidate_audit(decision: FlashDecision) -> Optional[FlashCandidateAudit]:
    selected_key = _actor_key(
        str(getattr(decision, "actor_type", "") or ""),
        str(getattr(decision, "selected_actor", "") or ""),
    )
    for row in getattr(decision, "candidates", ()) or ():
        row_key = _actor_key(
            str(getattr(row, "actor_type", "") or ""),
            str(getattr(row, "label", "") or ""),
        )
        if row_key != selected_key:
            continue
        if getattr(row, "action", None) != getattr(decision, "action", None):
            continue
        return row
    return None


def _raw_agent_label_from_solo(label: str) -> str:
    clean = str(label or "").strip()
    if not clean.startswith("Solo_"):
        return ""
    return clean[len("Solo_") :].strip()


def _solo_label_lookup_sequence(label: str) -> Tuple[str, ...]:
    clean = str(label or "").strip()
    if not clean:
        return tuple()
    raw_label = _raw_agent_label_from_solo(clean)
    if raw_label:
        return (clean, raw_label)
    return (clean,)


def _previous_actor_key_for_symbol(mapping: Mapping[str, str], symbol: str) -> str:
    clean_symbol = str(symbol or "").upper()
    for raw_symbol, raw_actor in (mapping or {}).items():
        if str(raw_symbol or "").upper() != clean_symbol:
            continue
        return str(raw_actor or "").strip()
    return ""


def _regime_confidence_scale(
    market: MarketSnapshot,
    *,
    symbol: str = "",
) -> float:
    raw: object = None
    if symbol:
        features = _regime_features_for_symbol(market, symbol)
        raw = features.get("regime_confidence")
    if raw is None:
        raw = getattr(market, "regime_confidence", 1.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _signal_deny_key(actor_key: str, symbol: str, action: Action) -> str:
    return _normalize_signal_deny_key(
        f"{actor_key}|{str(symbol or '').upper()}|{action.name}"
    )


def _signal_context_key(
    actor_key: str,
    symbol: str,
    action: Action,
    regime: Regime,
) -> str:
    return _normalize_signal_context_key(
        f"{actor_key}|{str(symbol or '').upper()}|{action.name}|"
        f"{_normalize_regime_value(regime).label}"
    )


def _signal_side_key(actor_key: str, symbol: str, action: Action) -> str:
    side = _open_action_direction(action)
    if side not in {"long", "short"}:
        return ""
    return f"{str(actor_key or '').strip()}|{str(symbol or '').upper()}|{side}"


def _signal_side_key_from_normalized_key(raw: object) -> str:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 3 or not all(parts):
        return ""
    side = _action_name_direction(parts[2])
    if side not in {"long", "short"}:
        return ""
    return f"{parts[0]}|{parts[1].upper()}|{side}"


def _action_name_direction(name: object) -> str:
    try:
        action = Action[str(name or "").strip().upper()]
    except KeyError:
        return ""
    return _open_action_direction(action)


def _static_contra_signal_from_key(
    raw: object,
    labels: set[str],
) -> Optional[tuple[str, str, str]]:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 3 or not all(parts):
        return None
    actor_key, symbol, action_name = parts
    label = _label_from_actor_key(actor_key)
    if labels and label not in labels:
        return None
    direction = _action_name_direction(action_name)
    if direction not in {"long", "short"}:
        return None
    return label, symbol.upper(), direction


def _label_from_actor_key(actor_key: str) -> str:
    clean = str(actor_key or "").strip()
    if ":" in clean:
        return clean.split(":", 1)[1]
    return clean


def _normalize_signal_deny_key(raw: object) -> str:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Flash signal deny key must use 'actor_key|symbol|action' format"
        )
    return f"{parts[0]}|{parts[1].upper()}|{parts[2].upper()}"


def _signal_symbol_aliases(symbol: object) -> Tuple[str, ...]:
    clean = str(symbol or "").strip().upper()
    if not clean:
        return ()
    aliases: list[str] = [clean]
    if "/" in clean:
        base, quote = clean.split("/", 1)
        if base:
            aliases.append(base)
        if base and quote:
            aliases.append(f"{base}{quote}")
    elif clean.endswith("USDT") and len(clean) > 4:
        base = clean[:-4]
        aliases.append(base)
        aliases.append(f"{base}/USDT")
    else:
        aliases.append(f"{clean}/USDT")
    return tuple(dict.fromkeys(alias for alias in aliases if alias))


def _normalize_signal_deny_key_variants(raw: object) -> Tuple[str, ...]:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Flash signal deny key must use 'actor_key|symbol|action' format"
        )
    actor_key, symbol, action = parts
    return tuple(
        _normalize_signal_deny_key(f"{actor_key}|{alias}|{action}")
        for alias in _signal_symbol_aliases(symbol)
    )


def _normalize_signal_deny_key_set(raw_keys: Iterable[object]) -> Tuple[str, ...]:
    keys: list[str] = []
    for raw in raw_keys:
        if not str(raw or "").strip():
            continue
        keys.extend(_normalize_signal_deny_key_variants(raw))
    return tuple(dict.fromkeys(keys))


def _normalize_signal_context_key(raw: object) -> str:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            "Flash signal context key must use 'actor_key|symbol|action|regime' format"
        )
    signal_key = _normalize_signal_deny_key("|".join(parts[:3]))
    regime = _normalize_regime_value(parts[3]).label
    return f"{signal_key}|{regime}"


def _normalize_signal_context_key_variants(raw: object) -> Tuple[str, ...]:
    parts = [part.strip() for part in str(raw or "").split("|")]
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            "Flash signal context key must use 'actor_key|symbol|action|regime' format"
        )
    regime = _normalize_regime_value(parts[3]).label
    return tuple(
        f"{signal_key}|{regime}"
        for signal_key in _normalize_signal_deny_key_variants("|".join(parts[:3]))
    )


def _normalize_signal_context_key_set(raw_keys: Iterable[object]) -> Tuple[str, ...]:
    keys: list[str] = []
    for raw in raw_keys:
        if not str(raw or "").strip():
            continue
        keys.extend(_normalize_signal_context_key_variants(raw))
    return tuple(dict.fromkeys(keys))


def _blocked_flash_decision(decision: FlashDecision, *, reason: str) -> FlashDecision:
    return replace(
        decision,
        selected_actor="NoTrade",
        actor_type="no_trade",
        score=0.0,
        action=Action.HOLD,
        reason=reason,
        signal=None,
        original_selected_actor=(
            decision.original_selected_actor or decision.selected_actor
        ),
        original_actor_type=(
            decision.original_actor_type or decision.actor_type
        ),
        selected_reasons=tuple((*decision.selected_reasons, reason)),
    )


def _regime_features_for_symbol(
    market: MarketSnapshot,
    symbol: str,
) -> Mapping[str, Any]:
    getter = getattr(market, "regime_features_for_symbol", None)
    if callable(getter):
        try:
            features = getter(symbol)
            if isinstance(features, Mapping):
                return features
        except Exception:
            return {}
    clean_symbol = str(symbol or "").strip().upper()
    raw = (getattr(market, "regime_features_by_symbol", {}) or {}).get(clean_symbol)
    return raw if isinstance(raw, Mapping) else {}


def _micro_macro_alignment_from_features(micro: str, macro: str) -> str:
    micro = str(micro or "flat").lower()
    macro = str(macro or "flat").lower()
    if micro in {"up", "down"} and macro in {"up", "down"}:
        return "aligned" if micro == macro else "conflict"
    if micro in {"up", "down"}:
        return "micro_only"
    if macro in {"up", "down"}:
        return "macro_only"
    return "flat"


def _lookback_return_pct(
    market: MarketSnapshot,
    symbol: str,
    lookback_bars: int,
) -> Optional[float]:
    returns = getattr(market, "lookback_returns_pct", {}) or {}
    symbol_returns = returns.get(str(symbol).upper())
    if not isinstance(symbol_returns, Mapping):
        return None
    raw = symbol_returns.get(int(lookback_bars))
    if raw is None:
        raw = symbol_returns.get(str(int(lookback_bars)))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _technical_indicators_for_symbol(market: MarketSnapshot, symbol: str):
    clean_symbol = str(symbol or "").upper()
    for raw_symbol, indicators in (
        getattr(market, "technicals_by_symbol", {}) or {}
    ).items():
        if str(raw_symbol).upper() == clean_symbol:
            return indicators
    return None


def _lookback_volatility_pct(
    market: MarketSnapshot,
    symbol: str,
    lookback_bars: int,
) -> Optional[float]:
    volatilities = getattr(market, "lookback_volatility_pct", {}) or {}
    symbol_volatilities = volatilities.get(str(symbol).upper())
    if not isinstance(symbol_volatilities, Mapping):
        return None
    raw = symbol_volatilities.get(int(lookback_bars))
    if raw is None:
        raw = symbol_volatilities.get(str(int(lookback_bars)))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return value


def _lookback_return_z(
    lookback_return_pct: Optional[float],
    lookback_volatility_pct: Optional[float],
    min_volatility_pct: float,
) -> Optional[float]:
    if lookback_return_pct is None or lookback_volatility_pct is None:
        return None
    if lookback_volatility_pct < min_volatility_pct:
        return None
    return float(lookback_return_pct) / float(lookback_volatility_pct)


def _market_price_for_symbol(market: MarketSnapshot, symbol: str) -> Optional[float]:
    clean_symbol = str(symbol or "").upper()
    for raw_symbol, raw_price in (getattr(market, "prices", {}) or {}).items():
        if str(raw_symbol).upper() != clean_symbol:
            continue
        try:
            return float(raw_price)
        except (TypeError, ValueError):
                return None
    return None


def _market_symbol_for_signal(market: MarketSnapshot, symbol: object) -> str:
    clean_symbol = str(symbol or "").strip().upper()
    if not clean_symbol:
        return ""
    prices = getattr(market, "prices", {}) or {}
    for raw_symbol in prices:
        market_symbol = str(raw_symbol or "").strip().upper()
        if market_symbol == clean_symbol:
            return market_symbol
    for raw_symbol in prices:
        market_symbol = str(raw_symbol or "").strip().upper()
        if _symbols_equivalent_for_market(clean_symbol, market_symbol):
            return market_symbol
    return ""


def _symbols_equivalent_for_market(left: str, right: str) -> bool:
    left_base, left_quote = _symbol_base_quote(left)
    right_base, right_quote = _symbol_base_quote(right)
    if not left_base or not right_base or left_base != right_base:
        return False
    return left_quote == right_quote or not left_quote or not right_quote


def _symbol_base_quote(symbol: object) -> tuple[str, str]:
    clean = str(symbol or "").strip().upper()
    if not clean:
        return "", ""
    for separator in _SYMBOL_SEPARATORS:
        if separator not in clean:
            continue
        base, quote = clean.rsplit(separator, 1)
        base = base.strip()
        quote = quote.strip()
        if base and quote in _QUOTE_ASSET_SUFFIXES:
            return base, quote
    compact = clean
    for separator in _SYMBOL_SEPARATORS:
        compact = compact.replace(separator, "")
    for quote in _QUOTE_ASSET_SUFFIXES:
        if compact.endswith(quote) and len(compact) > len(quote):
            return compact[: -len(quote)], quote
    return compact, ""


def _market_funding_for_symbol(market: MarketSnapshot, symbol: str) -> float:
    clean_symbol = str(symbol or "").upper()
    for raw_symbol, raw_funding in (getattr(market, "funding", {}) or {}).items():
        if str(raw_symbol).upper() != clean_symbol:
            continue
        try:
            parsed = float(raw_funding)
        except (TypeError, ValueError):
            return 0.0
        return parsed if math.isfinite(parsed) else 0.0
    return 0.0


def _fee_bps_for_symbol(
    market: MarketSnapshot,
    symbol: str,
    *,
    default_bps: float,
) -> float:
    clean_symbol = str(symbol or "").upper()
    for raw_symbol, raw_fee_bps in (
        getattr(market, "fees_bps_by_symbol", {}) or {}
    ).items():
        if str(raw_symbol).upper() != clean_symbol:
            continue
        try:
            parsed = float(raw_fee_bps)
        except (TypeError, ValueError):
            return max(0.0, float(default_bps))
        return max(0.0, parsed) if math.isfinite(parsed) else 0.0
    return max(0.0, float(default_bps))


def _finite_positive_or_default(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed) or parsed < 0.0:
        return float(default)
    return parsed


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _finite_float_or_none(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _win_rate_pct(winning_trades: int, closed_trades: int) -> float:
    if closed_trades <= 0:
        return 0.0
    return float(max(0, winning_trades)) / float(closed_trades) * 100.0


def _shadow_confirmation_raw(
    label: str,
    shadow_confirmation: Mapping[object, object],
    *,
    symbol: str,
    action: Action,
    require_symbol: bool,
) -> object:
    """Lookup shadow confirmation.

    Symbol-scoped confirmation accepts only ``(label, symbol, action_name)``.
    Actor-level fallback continues to use ``label``.
    """
    clean_label = str(label)
    clean_symbol = str(symbol or "").upper()
    action_name = action.name if isinstance(action, Action) else str(action)
    if not require_symbol:
        return shadow_confirmation.get(clean_label)

    return shadow_confirmation.get((clean_label, clean_symbol, action_name))


def _shadow_symbol_health_key(symbol: str, action: Action) -> tuple[str, str, str]:
    action_name = action.name if isinstance(action, Action) else str(action)
    return ("__symbol_health__", str(symbol or "").upper(), action_name)


def _shadow_confirmation_from_raw(raw: object) -> _ShadowConfirmation:
    if raw is None:
        return _ShadowConfirmation()
    if isinstance(raw, Mapping):
        score = float(raw.get("score", 0.0) or 0.0)
        closed_trades = int(raw.get("closed_trades", 0) or 0)
        winning_trades = int(raw.get("winning_trades", 0) or 0)
        raw_win_rate = raw.get("win_rate_pct")
        win_rate_pct = (
            float(raw_win_rate)
            if raw_win_rate is not None
            else _win_rate_pct(winning_trades, closed_trades)
        )
        return _ShadowConfirmation(
            score=score,
            closed_trades=closed_trades,
            winning_trades=winning_trades,
            win_rate_pct=win_rate_pct,
            recent_downside_usd=float(raw.get("recent_downside_usd", 0.0) or 0.0),
            pnl_per_trade_mean_usd=_finite_float_or_none(
                raw.get("pnl_per_trade_mean_usd")
            ),
            pnl_per_trade_std_usd=_finite_float_or_none(
                raw.get("pnl_per_trade_std_usd")
            ),
            pnl_per_trade_lcb_usd=_finite_float_or_none(
                raw.get("pnl_per_trade_lcb_usd")
            ),
        )
    try:
        score, closed_trades = raw  # type: ignore[misc]
    except (TypeError, ValueError):
        return _ShadowConfirmation()
    return _ShadowConfirmation(
        score=float(score or 0.0),
        closed_trades=int(closed_trades or 0),
    )


def _signal_payload(signal: Optional[Signal]) -> Optional[dict]:
    if signal is None:
        return None
    return {
        "id": signal.id,
        "bar": signal.bar,
        "sym": signal.sym,
        "action": signal.action.name,
        "price": float(signal.price),
        "regime": signal.regime.label,
        "by_player": signal.by_player,
        "by_agent": signal.by_agent,
        "risk_mult": float(signal.risk_mult),
    }
