"""Production startup — единая точка запуска v2 для конкретной биржи.

Вызывается тонкими Start_*_v2.py скриптами в корне репозитория.

Что делает:
  1. Регистрирует v1-агенты через agent_bootstrap.
  2. Собирает Exchange-адаптер (FakeExchange для paper, real adapter
     иначе — пока нет, использует bridge как infrastructure).
  3. (опционально) загружает persisted snapshot v2 или мигрирует v1.
  4. Собирает ProductionPipeline (DI всех компонентов).
  5. Создаёт OutputWriter (status.json + leaderboard + dashboard).
  6. Запускает либо v1-bridge mode (live данные), либо empty-feed (smoke).
  7. main_loop с on_step → OutputWriter.write().

После Ctrl+C — сохраняет state и финальный snapshot.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from ..domain.types import Action, Regime, Signal
from ..execution import Exchange, ExecutionStatus, FakeExchange, RiskLimitsConfig
from ..execution.position_tracker import TrackedPosition
from ..scoring import DEFAULT_SCORING, ScoringConfig
from ..selection import AgentRegistry, FlashAllocatorConfig, PlayerProfile, StrategistConfig
from ..selection.component_memory import ComponentMemory
from ..shadow.feed import MarketFeed, PollingFeed, ReplayFeed
from ..shadow.synthetic_feed import SyntheticFeed
from .agent_bootstrap import (
    legacy_flash_real_agent_labels,
    optional_labels,
    promote_legacy_flash_real_agents,
    register_all_v1_agents,
)
from .bootstrap import LiveExecutionConfig, PRODUCTION_PROFILES, build_production_pipeline
from .main_loop import main_loop
from .live_state import (
    adopted_position_labels,
    seed_perf_open_position_for_adoption,
    should_adopt_existing_position,
)
from .migration import (
    load_v2_snapshot,
    migrate_from_v1_memory_files,
    save_v2_snapshot,
)
from ..memory.retro_prior import load_retro_prior_file_into_memory
from .output_writer import OutputWriter, OutputWriterConfig
from .shadow_tournament import ProductionShadowTournament


log = logging.getLogger(__name__)


VIRTUAL_EXCHANGE_MODES = frozenset({"paper", "paper_live_feed", "shadow_live_feed"})
SYNTHETIC_FEED_MODES = frozenset({"paper"})


def _is_virtual_exchange_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in VIRTUAL_EXCHANGE_MODES


def _resolve_initial_capital(
    *,
    exchange_adapter: Exchange,
    mode: str,
    requested_initial_capital: Optional[float],
    exchange_name: str,
) -> float:
    """Resolve v2 capital base.

    For live modes, absent explicit *_INITIAL_CAPITAL means "use real exchange
    equity". Paper keeps a deterministic configured/default balance.
    """
    mode_key = str(mode or "").strip().lower()
    if requested_initial_capital is not None:
        capital = float(requested_initial_capital)
        if not _is_virtual_exchange_mode(mode_key):
            live_equity = _read_exchange_equity(exchange_adapter)
            if live_equity is not None:
                log.info(
                    "[%s] live account equity=$%.2f; explicit initial_capital=$%.2f kept",
                    exchange_name,
                    live_equity,
                    capital,
                )
        return capital

    if not _is_virtual_exchange_mode(mode_key):
        live_equity = _read_exchange_equity(exchange_adapter)
        if live_equity is not None:
            log.info(
                "[%s] live account equity detected: $%.2f — using as v2 capital",
                exchange_name,
                live_equity,
            )
            return live_equity
        log.warning(
            "[%s] live account equity unavailable — falling back to $100.00",
            exchange_name,
        )

    return 100.0


def _read_exchange_equity(exchange_adapter: Exchange) -> Optional[float]:
    getter = getattr(exchange_adapter, "get_account_equity", None)
    if not callable(getter):
        return None
    try:
        equity = float(getter() or 0.0)
    except Exception:
        log.debug("exchange equity read failed", exc_info=True)
        return None
    return equity if equity > 0 else None


def _risk_config_from_trade_fraction(
    trade_fraction: float,
    *,
    max_open_positions: int = 4,
) -> RiskLimitsConfig:
    fraction = float(trade_fraction)
    if not 0 < fraction <= 1.0:
        raise ValueError(f"trade_fraction must be in (0, 1], got {trade_fraction}")
    return RiskLimitsConfig(
        max_open_positions=max(0, int(max_open_positions)),
        capital_fraction=fraction,
        floor_to_exchange_min_notional=True,
    )


def _risk_config_from_settings(
    settings: dict,
    *,
    trade_fraction: float,
    exchange_name: str = "",
) -> RiskLimitsConfig:
    def scoped_names(*base_names: str) -> tuple[str, ...]:
        out: list[str] = []
        for base in base_names:
            out.extend(_exchange_scoped_setting_names(exchange_name, base))
        return tuple(dict.fromkeys(out))

    def int_any(names: Sequence[str], default: int) -> int:
        for name in names:
            if name in settings:
                try:
                    return int(float(settings.get(name)))
                except (TypeError, ValueError):
                    return int(default)
        return int(default)

    return _risk_config_from_trade_fraction(
        trade_fraction,
        max_open_positions=int_any(
            scoped_names(
                "v2_risk_max_open_positions",
                "v2_max_open_positions",
                "risk_max_open_positions",
                "max_open_positions",
            ),
            4,
        ),
    )


def _resolve_risk_config(exchange_name: str, trade_fraction: float) -> RiskLimitsConfig:
    return _risk_config_from_settings(
        _load_exchange_settings(exchange_name),
        trade_fraction=trade_fraction,
        exchange_name=exchange_name,
    )


def _resolve_trade_fraction(exchange_name: str) -> float:
    try:
        from .v1_futures_adapter import load_runtime_trade_fraction

        return load_runtime_trade_fraction(exchange_name, default=0.10)
    except Exception:
        log.debug("[%s] runtime trade_fraction unavailable", exchange_name, exc_info=True)
        return 0.10


def _resolve_timeframe(exchange_name: str) -> str:
    return (
        os.getenv(f"{exchange_name.upper()}_TIMEFRAME")
        or os.getenv("PANTEON_TIMEFRAME")
        or "bridge_poll"
    )


def _resolve_strategist_config(exchange_name: str) -> StrategistConfig:
    return _strategist_config_from_settings(_load_exchange_settings(exchange_name))


def _strategist_config_from_settings(settings: dict) -> StrategistConfig:
    def num(name: str, default: float) -> float:
        try:
            return float(settings.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    def num_any(names: Sequence[str], default: float) -> float:
        for name in names:
            if name in settings:
                return num(name, default)
        return float(default)

    def bool_any(names: Sequence[str], default: bool) -> bool:
        for name in names:
            if name not in settings:
                continue
            value = settings.get(name)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
            return bool(default)
        return bool(default)

    executable_soft_top1 = bool_any(
        ("v2_executable_soft_top1_enabled", "executable_soft_top1_enabled"),
        False,
    )
    shadow_window = int(num_any(
        ("v2_shadow_rolling_window_bars", "v3_shadow_rolling_window_bars"),
        24,
    ))
    shadow_min_closed = int(num_any(
        ("v2_shadow_rolling_min_closed_trades", "v3_shadow_rolling_min_closed_trades"),
        20 if executable_soft_top1 else 50,
    ))
    probation_loss_kill_all = bool_any(
        (
            "v2_probation_loss_kill_all_labels",
            "v3_probation_loss_kill_all_labels",
        ),
        False,
    )

    return StrategistConfig(
        cooldown_bars=int(num("player_switch_cooldown_bars", 30)),
        switch_margin=num("player_switch_margin", 0.30),
        min_score_to_switch=num("player_min_score_to_switch", -0.05),
        streak_needed=int(num("player_switch_confirmations", 2)),
        hard_negative=num_any(("player_hard_negative_score", "player_hard_negative_pnl"), -0.50),
        urgent_gap=num("player_urgent_gap", 0.80),
        min_live_closed_trades=int(num("player_min_live_closed_trades", 0)),
        min_live_score=num("player_min_live_score", 0.0),
        max_live_drawdown_pct=num("player_max_live_drawdown_pct", 100.0),
        min_regime_confidence=num("regime_min_confidence", 0.0),
        player_session_overlay_weight=num("player_session_overlay_weight", 1.00),
        player_session_underperformance_weight=num(
            "player_session_underperformance_weight",
            0.75,
        ),
        player_session_stale_penalty=num("player_session_stale_penalty", 0.15),
        player_session_pnl_cap_pct=num("player_session_pnl_cap_pct", 5.0),
        player_session_min_activity=int(num("player_session_min_activity", 1)),
        use_v3_rolling_score=bool_any(
            ("v2_use_v3_rolling_score", "use_v3_rolling_score"),
            False,
        ) or executable_soft_top1,
        use_v3_soft_shadow_score=bool_any(
            ("v2_use_v3_soft_shadow_score", "use_v3_soft_shadow_score"),
            False,
        ) or executable_soft_top1,
        v3_current_actionable_gate_enabled=bool_any(
            (
                "v2_current_actionable_candidate_layer_enabled",
                "current_actionable_candidate_layer_enabled",
            ),
            False,
        ) or executable_soft_top1,
        v3_shadow_rolling_window_bars=shadow_window,
        v3_shadow_rolling_min_closed_trades=shadow_min_closed,
        v3_probation_loss_kill_min_closed_trades=int(num_any(
            (
                "v2_probation_loss_kill_min_closed_trades",
                "v3_probation_loss_kill_min_closed_trades",
            ),
            0,
        )),
        v3_probation_loss_kill_pnl_pct=num_any(
            ("v2_probation_loss_kill_pnl_pct", "v3_probation_loss_kill_pnl_pct"),
            -0.15,
        ),
        v3_probation_loss_kill_win_rate_pct=num_any(
            (
                "v2_probation_loss_kill_win_rate_pct",
                "v3_probation_loss_kill_win_rate_pct",
            ),
            50.0,
        ),
        v3_probation_loss_kill_label_prefixes=(
            ()
            if probation_loss_kill_all
            else ("Solo_", "Fixed_", "Antonius_", "Optimal_")
        ),
        v3_shadow_fresh_handoff_enabled=bool_any(
            (
                "v2_shadow_fresh_handoff_enabled",
                "v3_shadow_fresh_handoff_enabled",
            ),
            False,
        ),
        v3_shadow_fresh_handoff_max_age_bars=int(num_any(
            (
                "v2_shadow_fresh_handoff_max_age_bars",
                "v3_shadow_fresh_handoff_max_age_bars",
            ),
            1,
        )),
        v3_shadow_fresh_handoff_require_positive_unrealized=bool_any(
            (
                "v2_shadow_fresh_handoff_require_positive_unrealized",
                "v3_shadow_fresh_handoff_require_positive_unrealized",
            ),
            True,
        ),
    )


def _live_execution_config_from_settings(
    settings: dict,
    exchange_name: str = "",
) -> LiveExecutionConfig:
    def scoped_names(*base_names: str) -> tuple[str, ...]:
        out: list[str] = []
        for base in base_names:
            out.extend(_exchange_scoped_setting_names(exchange_name, base))
        return tuple(dict.fromkeys(out))

    def num_any(names: Sequence[str], default: float) -> float:
        for name in names:
            if name in settings:
                try:
                    return float(settings.get(name))
                except (TypeError, ValueError):
                    return float(default)
        return float(default)

    def int_any(names: Sequence[str], default: int) -> int:
        return int(num_any(names, float(default)))

    def bool_any(names: Sequence[str], default: bool) -> bool:
        for name in names:
            if name not in settings:
                continue
            value = settings.get(name)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "on"}:
                    return True
                if normalized in {"0", "false", "no", "off"}:
                    return False
            return bool(default)
        return bool(default)

    def tuple_any(names: Sequence[str], default: Sequence[str] = ()) -> tuple[str, ...]:
        for name in names:
            if name not in settings:
                continue
            value = settings.get(name)
            if isinstance(value, str):
                return tuple(
                    item.strip()
                    for item in value.replace(";", ",").split(",")
                    if item.strip()
                )
            try:
                return tuple(
                    str(item).strip()
                    for item in (value or ())
                    if str(item).strip()
                )
            except TypeError:
                item = str(value or "").strip()
                return (item,) if item else tuple(default)
        return tuple(default)

    def str_any(names: Sequence[str], default: str) -> str:
        for name in names:
            if name in settings:
                value = str(settings.get(name) or "").strip()
                return value or default
        return default

    return LiveExecutionConfig(
        max_new_opens_per_bar=int_any(
            scoped_names("v2_max_new_opens_per_bar", "max_new_opens_per_bar"),
            1,
        ),
        max_daily_loss_pct=num_any(
            scoped_names("v2_max_daily_loss_pct", "max_daily_loss_pct"),
            5.0,
        ),
        max_equity_peak_drawdown_pct=num_any(
            scoped_names(
                "v2_max_equity_peak_drawdown_pct",
                "max_equity_peak_drawdown_pct",
                "v2_max_trailing_equity_stop_pct",
                "max_trailing_equity_stop_pct",
            ),
            5.0,
        ),
        kill_switch_auto_recovery_enabled=bool_any(
            scoped_names(
                "v2_kill_switch_auto_recovery_enabled",
                "kill_switch_auto_recovery_enabled",
            ),
            False,
        ),
        kill_switch_recovery_cooldown_bars=int_any(
            scoped_names(
                "v2_kill_switch_recovery_cooldown_bars",
                "kill_switch_recovery_cooldown_bars",
            ),
            0,
        ),
        kill_switch_recovery_max_drawdown_pct=num_any(
            scoped_names(
                "v2_kill_switch_recovery_max_drawdown_pct",
                "kill_switch_recovery_max_drawdown_pct",
            ),
            0.0,
        ),
        max_consecutive_failed_orders=int_any(
            scoped_names(
                "v2_max_consecutive_failed_orders",
                "max_consecutive_failed_orders",
            ),
            3,
        ),
        max_exchange_desync_events=int_any(
            scoped_names(
                "v2_max_exchange_desync_events",
                "max_exchange_desync_events",
            ),
            5,
        ),
        max_stale_feed_polls=int_any(
            scoped_names("v2_max_stale_feed_polls", "max_stale_feed_polls"),
            12,
        ),
        max_slippage_pct=num_any(
            scoped_names("v2_max_slippage_pct", "max_slippage_pct"),
            0.75,
        ),
        max_api_error_streak=int_any(
            scoped_names("v2_max_api_error_streak", "max_api_error_streak"),
            3,
        ),
        pending_order_timeout_sec=num_any(
            scoped_names("v2_pending_order_timeout_sec", "pending_order_timeout_sec"),
            180.0,
        ),
        adopt_existing_positions_enabled=bool_any(
            scoped_names(
                "v2_adopt_existing_positions_enabled",
                "adopt_existing_positions_enabled",
                "v2_adopt_existing_positions",
                "adopt_existing_positions",
            ),
            False,
        ),
        adopt_existing_position_symbols=tuple_any(
            scoped_names(
                "v2_adopt_existing_position_symbols",
                "adopt_existing_position_symbols",
                "v2_adopt_existing_positions_symbols",
                "adopt_existing_positions_symbols",
            ),
            (),
        ),
        adopt_existing_position_player=str_any(
            scoped_names(
                "v2_adopt_existing_position_player",
                "adopt_existing_position_player",
            ),
            "PanteonFlashAdopted",
        ),
        adopt_existing_position_agent=str_any(
            scoped_names(
                "v2_adopt_existing_position_agent",
                "adopt_existing_position_agent",
            ),
            "AdoptedExchangePosition",
        ),
        genetics_probation_execution_enabled=bool_any(
            scoped_names(
                "v2_genetics_probation_execution_enabled",
                "genetics_probation_execution_enabled",
            ),
            False,
        ),
        genetics_probation_labels=tuple_any(
            scoped_names("v2_genetics_probation_labels", "genetics_probation_labels"),
            ("GeneticsResearch",),
        ),
        genetics_probation_allowed_regimes=tuple_any(
            scoped_names(
                "v2_genetics_probation_allowed_regimes",
                "genetics_probation_allowed_regimes",
            ),
            ("bearish", "crash"),
        ),
        genetics_probation_allowed_signal_keys=tuple_any(
            scoped_names(
                "v2_genetics_probation_allowed_signal_keys",
                "genetics_probation_allowed_signal_keys",
            ),
            (),
        ),
        genetics_probation_risk_mult=num_any(
            scoped_names(
                "v2_genetics_probation_risk_mult",
                "genetics_probation_risk_mult",
            ),
            0.20,
        ),
        genetics_probation_min_regime_confidence=num_any(
            scoped_names(
                "v2_genetics_probation_min_regime_confidence",
                "genetics_probation_min_regime_confidence",
            ),
            0.0,
        ),
        genetics_probation_max_real_trades=int_any(
            scoped_names(
                "v2_genetics_probation_max_real_trades",
                "genetics_probation_max_real_trades",
            ),
            20,
        ),
        genetics_probation_max_daily_trades=int_any(
            scoped_names(
                "v2_genetics_probation_max_daily_trades",
                "genetics_probation_max_daily_trades",
            ),
            0,
        ),
        genetics_probation_require_shadow_confirmation=bool_any(
            scoped_names(
                "v2_genetics_probation_require_shadow_confirmation",
                "genetics_probation_require_shadow_confirmation",
            ),
            True,
        ),
        flash_genetics_core_primary_enabled=bool_any(
            scoped_names(
                "panteon_flash_genetics_core_primary_enabled",
                "v2_flash_genetics_core_primary_enabled",
                "flash_genetics_core_primary_enabled",
            ),
            False,
        ),
        flash_genetics_core_primary_labels=tuple_any(
            scoped_names(
                "panteon_flash_genetics_core_primary_labels",
                "v2_flash_genetics_core_primary_labels",
                "flash_genetics_core_primary_labels",
            ),
            ("GeneticsCore",),
        ),
        flash_genetics_core_primary_bypass_shadow_confirmation_enabled=bool_any(
            scoped_names(
                "panteon_flash_genetics_core_primary_bypass_shadow_confirmation_enabled",
                "v2_flash_genetics_core_primary_bypass_shadow_confirmation_enabled",
                "flash_genetics_core_primary_bypass_shadow_confirmation_enabled",
            ),
            False,
        ),
        genetics_probation_max_consecutive_failed_orders=int_any(
            scoped_names(
                "v2_genetics_probation_max_consecutive_failed_orders",
                "genetics_probation_max_consecutive_failed_orders",
            ),
            2,
        ),
        genetics_probation_max_realized_loss_pct=num_any(
            scoped_names(
                "v2_genetics_probation_max_realized_loss_pct",
                "genetics_probation_max_realized_loss_pct",
            ),
            0.25,
        ),
        genetics_probation_risk_mult_by_label=tuple_any(
            scoped_names(
                "v2_genetics_probation_risk_mult_by_label",
                "genetics_probation_risk_mult_by_label",
            ),
            (),
        ),
        genetics_probation_max_real_trades_by_label=tuple_any(
            scoped_names(
                "v2_genetics_probation_max_real_trades_by_label",
                "genetics_probation_max_real_trades_by_label",
            ),
            (),
        ),
        genetics_probation_max_daily_trades_by_label=tuple_any(
            scoped_names(
                "v2_genetics_probation_max_daily_trades_by_label",
                "genetics_probation_max_daily_trades_by_label",
            ),
            (),
        ),
        genetics_probation_max_consecutive_failed_orders_by_label=tuple_any(
            scoped_names(
                "v2_genetics_probation_max_consecutive_failed_orders_by_label",
                "genetics_probation_max_consecutive_failed_orders_by_label",
            ),
            (),
        ),
        genetics_probation_max_realized_loss_pct_by_label=tuple_any(
            scoped_names(
                "v2_genetics_probation_max_realized_loss_pct_by_label",
                "genetics_probation_max_realized_loss_pct_by_label",
            ),
            (),
        ),
        max_real_symbol_min_executable_notional_usd=num_any(
            scoped_names(
                "v2_max_real_symbol_min_executable_notional_usd",
                "max_real_symbol_min_executable_notional_usd",
                "v2_max_real_contract_min_notional_usd",
                "max_real_contract_min_notional_usd",
            ),
            40.0,
        ),
    )


def _resolve_live_execution_config(exchange_name: str) -> LiveExecutionConfig:
    return _live_execution_config_from_settings(
        _load_exchange_settings(exchange_name),
        exchange_name=exchange_name,
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _settings_bool(settings: dict, names: Sequence[str], default: bool = False) -> bool:
    for name in names:
        if name not in settings:
            continue
        value = settings.get(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(default)
    return bool(default)


def _resolve_paper_canary_shutdown_flatten_enabled(exchange_name: str) -> bool:
    """Resolve virtual canary shutdown flattening.

    This flag is intentionally opt-in and the execution helper below also
    refuses every non-virtual mode. It must never become an implicit live
    position closer.
    """
    exchange_key = str(exchange_name or "").strip().upper()
    exchange_prefix = str(exchange_name or "").strip().lower()
    if exchange_key and _env_flag(
        f"{exchange_key}_PANTEON_V2_PAPER_CANARY_CLOSE_ON_SHUTDOWN",
        False,
    ):
        return True
    if _env_flag("PANTEON_V2_PAPER_CANARY_CLOSE_ON_SHUTDOWN", False):
        return True
    settings = _load_exchange_settings(exchange_name)
    names = (
        f"{exchange_prefix}_v2_paper_canary_close_on_shutdown",
        f"{exchange_prefix}_paper_canary_close_on_shutdown",
        f"{exchange_prefix}_v2_paper_canary_shutdown_flatten_enabled",
        f"{exchange_prefix}_paper_canary_shutdown_flatten_enabled",
        "v2_paper_canary_close_on_shutdown",
        "paper_canary_close_on_shutdown",
        "v2_paper_canary_shutdown_flatten_enabled",
        "paper_canary_shutdown_flatten_enabled",
    )
    return _settings_bool(settings, names, False)


def _paper_canary_shutdown_summary(
    *,
    enabled: bool = False,
    skipped_reason: str = "",
) -> Dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "skipped_reason": str(skipped_reason or ""),
        "attempted": 0,
        "closed": 0,
        "failed": 0,
        "skipped_external": 0,
        "closed_symbols": [],
        "failed_symbols": [],
        "remaining_open_position_count": 0,
    }


def _is_external_or_recovered_position(position: TrackedPosition) -> bool:
    by_player = str(getattr(position, "by_player", "") or "").strip()
    by_agent = str(getattr(position, "by_agent", "") or "").strip()
    external_labels = {
        "",
        "RecoveredExchangePosition",
        "ExternalExchangePosition",
        "AdoptedExchangePosition",
    }
    return by_player in external_labels or (
        bool(by_agent) and by_agent in external_labels
    )


def _lookup_shutdown_close_price(
    symbol: str,
    position: TrackedPosition,
    latest_prices: Mapping[str, float] | None,
) -> float:
    candidates = [str(symbol or ""), str(getattr(position, "sym", "") or "")]
    expanded: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        expanded.extend(
            [
                candidate,
                candidate.upper(),
                candidate.replace("/", "").upper(),
                candidate.replace("USDT", "/USDT").upper(),
            ]
        )
    prices = latest_prices if isinstance(latest_prices, Mapping) else {}
    for key in expanded:
        if key in prices:
            try:
                price = float(prices[key])
            except (TypeError, ValueError):
                continue
            if price > 0:
                return price
    try:
        return float(getattr(position, "entry_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _shutdown_close_action_for(position: TrackedPosition) -> Action:
    open_action = str(getattr(position, "open_action", "") or "").strip().upper()
    if open_action.startswith("SPOT_"):
        return Action.SPOT_SELL_ALL
    return Action.FUT_CLOSE_ALL


def _next_shutdown_signal_id(open_positions: Mapping[str, TrackedPosition]) -> int:
    max_open_id = 0
    for position in open_positions.values():
        try:
            max_open_id = max(max_open_id, int(position.open_signal_id))
        except (TypeError, ValueError):
            continue
    return max_open_id + 1


def _close_paper_canary_positions_on_shutdown(
    pipeline: object,
    *,
    mode: str,
    enabled: bool,
    latest_prices: Mapping[str, float] | None = None,
    bar: int | None = None,
) -> Dict[str, Any]:
    """Close owned virtual canary positions before process shutdown.

    The helper deliberately runs through TradeExecutor so FakeExchange,
    PositionTracker, EventLog and AttributionLedger observe the same lifecycle
    as normal paper fills. It skips recovered/external positions and refuses
    every non-virtual mode even when enabled=True.
    """
    if not enabled:
        return _paper_canary_shutdown_summary(skipped_reason="disabled")
    if not _is_virtual_exchange_mode(mode):
        return _paper_canary_shutdown_summary(skipped_reason="non_virtual_mode")

    executor = getattr(pipeline, "executor", None)
    tracker = getattr(executor, "_tracker", None)
    if executor is None or tracker is None or not hasattr(tracker, "all_open"):
        return _paper_canary_shutdown_summary(skipped_reason="missing_executor")

    try:
        open_positions = dict(tracker.all_open() or {})
    except Exception:
        log.exception("paper canary shutdown flatten failed to inspect positions")
        return _paper_canary_shutdown_summary(skipped_reason="position_read_failed")

    summary = _paper_canary_shutdown_summary(enabled=True)
    if not open_positions:
        return summary

    try:
        close_bar = int(bar) if bar is not None else max(
            int(getattr(pos, "opened_bar", 0) or 0) for pos in open_positions.values()
        ) + 1
    except (TypeError, ValueError):
        close_bar = 1
    signal_id = _next_shutdown_signal_id(open_positions)
    balance = float(
        getattr(pipeline, "current_balance", None)
        or getattr(pipeline, "initial_capital", None)
        or 1.0
    )

    for symbol, position in sorted(open_positions.items()):
        sym = str(symbol or getattr(position, "sym", "") or "").upper()
        if not sym:
            continue
        if _is_external_or_recovered_position(position):
            summary["skipped_external"] += 1
            continue

        price = _lookup_shutdown_close_price(sym, position, latest_prices)
        if price <= 0:
            summary["failed"] += 1
            summary["failed_symbols"].append(
                {"symbol": sym, "reason": "missing_close_price"}
            )
            continue

        signal = Signal(
            id=signal_id,
            bar=max(close_bar, int(getattr(position, "opened_bar", 0) or 0) + 1),
            sym=sym,
            action=_shutdown_close_action_for(position),
            price=price,
            regime=Regime.from_string(str(getattr(position, "open_regime", "") or "")),
            by_player=str(getattr(position, "by_player", "") or "PaperCanaryShutdown"),
            by_agent="PaperCanaryShutdownClose",
            metadata={
                "reason": "paper_canary_shutdown_flatten",
                "source": "startup_shutdown",
            },
        )
        signal_id += 1
        summary["attempted"] += 1
        try:
            result = executor.execute(signal, balance_usd=balance)
        except Exception as exc:
            log.exception("paper canary shutdown close failed for %s", sym)
            summary["failed"] += 1
            summary["failed_symbols"].append(
                {"symbol": sym, "reason": type(exc).__name__}
            )
            continue
        if result.status == ExecutionStatus.FILLED:
            summary["closed"] += 1
            summary["closed_symbols"].append(sym)
        else:
            summary["failed"] += 1
            summary["failed_symbols"].append(
                {
                    "symbol": sym,
                    "status": getattr(result.status, "name", str(result.status)),
                    "reason": str(getattr(result, "reason", "") or ""),
                }
            )

    event_log = getattr(pipeline, "event_log", None)
    ledger = getattr(pipeline, "ledger", None)
    replay = getattr(ledger, "replay_from_event_log", None)
    if event_log is not None and callable(replay):
        try:
            replay(event_log)
        except Exception:
            log.exception("paper canary shutdown ledger replay failed")

    try:
        summary["remaining_open_position_count"] = int(tracker.open_count)
    except Exception:
        summary["remaining_open_position_count"] = None
    return summary


def _latest_prices_from_writer(writer: object) -> Dict[str, float]:
    status = getattr(writer, "_last_status_data", {}) or {}
    if not isinstance(status, dict):
        return {}
    current = status.get("current_prices")
    if isinstance(current, dict) and current:
        return {
            str(symbol): float(price)
            for symbol, price in current.items()
            if _safe_positive_float(price) is not None
        }
    history = status.get("price_history")
    if isinstance(history, list):
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            prices = row.get("prices")
            if not isinstance(prices, dict) or not prices:
                continue
            return {
                str(symbol): float(price)
                for symbol, price in prices.items()
                if _safe_positive_float(price) is not None
            }
    return {}


def _safe_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _apply_paper_canary_shutdown_flatten(
    pipeline: object,
    writer: object,
    *,
    exchange: str,
    mode: str,
    bar: int | None = None,
) -> Dict[str, Any]:
    flatten_summary = _close_paper_canary_positions_on_shutdown(
        pipeline,
        mode=mode,
        enabled=_resolve_paper_canary_shutdown_flatten_enabled(exchange),
        latest_prices=_latest_prices_from_writer(writer),
        bar=bar,
    )
    if flatten_summary.get("enabled") or flatten_summary.get("skipped_reason") not in {
        "",
        "disabled",
    }:
        pipeline.paper_canary_shutdown_flatten = flatten_summary
        if flatten_summary.get("enabled"):
            failed = int(flatten_summary.get("failed", 0) or 0)
            remaining = int(flatten_summary.get("remaining_open_position_count", 0) or 0)
            feed_status = (
                "shutdown_flattened" if failed == 0 and remaining == 0
                else "shutdown_flatten_failed"
            )
            try:
                writer.write_heartbeat(
                    run_state="stopping",
                    feed_status=feed_status,
                    message=(
                        "paper_canary_shutdown_flatten "
                        f"closed={flatten_summary.get('closed', 0)} "
                        f"failed={failed} remaining={remaining}"
                    ),
                )
            except Exception:
                log.exception("paper canary shutdown heartbeat failed")
    return flatten_summary


def _settings_float(settings: dict, names: Sequence[str], default: float) -> float:
    for name in names:
        if name not in settings:
            continue
        try:
            return float(settings.get(name))
        except (TypeError, ValueError):
            return float(default)
    return float(default)


def _settings_optional_float(
    settings: dict,
    names: Sequence[str],
    default: float | None = None,
) -> float | None:
    for name in names:
        if name not in settings:
            continue
        value = settings.get(name)
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _settings_str(settings: dict, names: Sequence[str], default: str) -> str:
    for name in names:
        if name not in settings:
            continue
        value = settings.get(name)
        if value in (None, ""):
            return str(default)
        return str(value).strip()
    return str(default)


def _settings_csv_tuple(settings: dict, names: Sequence[str]) -> tuple[str, ...]:
    for name in names:
        if name not in settings:
            continue
        raw = settings.get(name)
        if raw is None:
            return ()
        if isinstance(raw, str):
            return tuple(item.strip() for item in raw.split(",") if item.strip())
        if isinstance(raw, (list, tuple, set)):
            return tuple(str(item).strip() for item in raw if str(item).strip())
        text = str(raw).strip()
        return (text,) if text else ()
    return ()


def _exchange_scoped_setting_names(exchange_name: str, base: str) -> tuple[str, ...]:
    exchange_key = str(exchange_name or "").strip().lower()
    base_key = str(base or "").strip().lower()
    if not exchange_key:
        return (base_key,)
    return (
        f"{exchange_key}_{base_key}",
        f"{base_key}_{exchange_key}",
        base_key,
    )


def _run_scoped_event_log_path(
    exchange_name: str,
    jsonl_event_log: Optional[str],
    *,
    now: Optional[datetime] = None,
    pid: Optional[int] = None,
) -> Optional[str]:
    if not jsonl_event_log:
        return jsonl_event_log
    exchange_key = str(exchange_name or "").strip().lower().replace("-", "_")
    if not exchange_key:
        return jsonl_event_log
    path = Path(jsonl_event_log)
    if path.name.lower() != f"v2_{exchange_key}_events.jsonl":
        return str(path)

    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    process_id = os.getpid() if pid is None else int(pid)
    scoped_name = (
        f"v2_{exchange_key}_{ts.strftime('%Y%m%d_%H%M%S')}_{process_id}_events.jsonl"
    )
    return str(path.with_name(scoped_name))


def _quarantine_override_labels_from_settings(
    settings: dict,
    exchange_name: str,
) -> tuple[str, ...]:
    return _settings_csv_tuple(
        settings,
        _exchange_scoped_setting_names(
            exchange_name,
            "v2_quarantine_override_labels",
        ) + _exchange_scoped_setting_names(
            exchange_name,
            "quarantine_override_labels",
        ),
    )


DEFAULT_MANUAL_QUARANTINE_LABELS: tuple[str, ...] = (
    "Optimal_StaticRotator",
    "RichardDennis",
    "LiveCrashHunter",
    "CandlePatternAgent",
    "LiveVolCompress",
    "AnchorFlowMomentum",
    "PlayerFunding",
    "FundingArb",
    "NeutralRangeScalper",
    "LiveTrendFollow",
    "GeneticsGenomeEnsemble",
)


DEFAULT_COMPACT_CAUSAL_ENTRY_INCLUDE_LABELS: tuple[str, ...] = (
    "CarryFlowAgentV2",
    "agent:CarryFlowAgentV2",
    "Solo_CarryFlowAgentV2",
    "ensemble:Solo_CarryFlowAgentV2",
    "MomentumScalper",
    "agent:MomentumScalper",
    "Solo_MomentumScalper",
    "ensemble:Solo_MomentumScalper",
    "LiveVolCompress",
    "agent:LiveVolCompress",
    "Solo_LiveVolCompress",
    "ensemble:Solo_LiveVolCompress",
    "LiveCrashHunter",
    "agent:LiveCrashHunter",
    "Solo_LiveCrashHunter",
    "ensemble:Solo_LiveCrashHunter",
)


DEFAULT_BITGET_REPORT_DERIVED_TERMINAL_DENIED_CONTEXT_SIGNAL_KEYS: tuple[str, ...] = (
    "agent:LiveOIBreakout|*|*|range_low_vol",
    "agent:CarryFlowAgentV2|*|*|bearish",
    "agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*",
    "agent:CarryFlowAgentV2|*|FUT_LONG_HALF|*",
    "agent:CarryFlowAgentV2|DOGE/USDT|*|*",
    "agent:CarryFlowAgentV2|XRP/USDT|*|*",
    "agent:LiveVolCompress|*|*|*",
    "agent:NeutralLiquiditySweep|*|*|*",
    "agent:LiveRegimePullback|*|*|*",
)


def _resolve_compact_causal_entry_include_labels(exchange_name: str) -> tuple[str, ...]:
    settings = _load_exchange_settings(exchange_name)
    configured = _settings_csv_tuple(
        settings,
        _exchange_scoped_setting_names(
            exchange_name,
            "v2_compact_causal_entry_include_labels",
        ) + _exchange_scoped_setting_names(
            exchange_name,
            "compact_causal_entry_include_labels",
        ),
    )
    return tuple(dict.fromkeys(DEFAULT_COMPACT_CAUSAL_ENTRY_INCLUDE_LABELS + configured))


def _manual_quarantine_labels_from_settings(
    settings: dict,
    exchange_name: str,
) -> tuple[str, ...]:
    names = _exchange_scoped_setting_names(
        exchange_name,
        "v2_manual_quarantine_labels",
    ) + _exchange_scoped_setting_names(
        exchange_name,
        "manual_quarantine_labels",
    )
    if any(name in settings for name in names):
        return _settings_csv_tuple(settings, names)
    return DEFAULT_MANUAL_QUARANTINE_LABELS


def _quarantine_recovery_config_from_settings(
    settings: dict,
    exchange_name: str,
) -> dict[str, object]:
    return {
        "enabled": _settings_bool(
            settings,
            _exchange_scoped_setting_names(
                exchange_name,
                "v2_quarantine_recovery_enabled",
            ) + _exchange_scoped_setting_names(
                exchange_name,
                "quarantine_recovery_enabled",
            ),
            True,
        ),
        "min_pnl_pct": _settings_float(
            settings,
            _exchange_scoped_setting_names(
                exchange_name,
                "v2_quarantine_recovery_min_pnl_pct",
            ) + _exchange_scoped_setting_names(
                exchange_name,
                "quarantine_recovery_min_pnl_pct",
            ),
            0.10,
        ),
        "min_closed_trades": max(
            1,
            int(_settings_float(
                settings,
                _exchange_scoped_setting_names(
                    exchange_name,
                    "v2_quarantine_recovery_min_closed_trades",
                ) + _exchange_scoped_setting_names(
                    exchange_name,
                    "quarantine_recovery_min_closed_trades",
                ),
                3,
            )),
        ),
    }


def _load_exchange_settings(exchange_name: str) -> dict:
    try:
        from .exchange_profile import load_exchange_profile

        profile = load_exchange_profile(exchange_name)
        settings: dict = {}
        for source in (
            getattr(profile, "raw_settings", {}) or {},
            getattr(profile, "parsed_settings", {}) or {},
        ):
            for key, value in dict(source).items():
                settings[str(key).strip().lower()] = value
        return settings
    except Exception:
        return {}


def _flash_enabled_from_settings(settings: dict, default: bool = False) -> bool:
    return _settings_bool(
        settings,
        (
            "panteon_flash_enabled",
            "v2_flash_enabled",
            "flash_enabled",
        ),
        default,
    )


def _flash_legacy_real_agents_enabled_from_settings(
    settings: dict,
    exchange_name: str,
    default: bool = False,
) -> bool:
    return _settings_bool(
        settings,
        _exchange_scoped_setting_names(
            exchange_name,
            "v2_flash_legacy_real_agents_enabled",
        ) + _exchange_scoped_setting_names(
            exchange_name,
            "panteon_flash_legacy_real_agents_enabled",
        ),
        default,
    )


def _flash_legacy_real_agent_labels_from_settings(
    settings: dict,
    exchange_name: str,
) -> tuple[str, ...]:
    labels = _settings_csv_tuple(
        settings,
        _exchange_scoped_setting_names(
            exchange_name,
            "v2_flash_legacy_real_agent_labels",
        ) + _exchange_scoped_setting_names(
            exchange_name,
            "panteon_flash_legacy_real_agent_labels",
        ),
    )
    return labels or tuple(legacy_flash_real_agent_labels())


def _flash_allocator_config_from_settings(
    settings: dict,
    exchange_name: str = "",
) -> FlashAllocatorConfig:
    def scoped_names(*base_names: str) -> tuple[str, ...]:
        out: list[str] = []
        for base in base_names:
            out.extend(_exchange_scoped_setting_names(exchange_name, base))
        return tuple(dict.fromkeys(out))

    def csv_merged(*name_groups: Sequence[str]) -> tuple[str, ...]:
        items: list[str] = []
        for names in name_groups:
            items.extend(_settings_csv_tuple(settings, names))
        return tuple(dict.fromkeys(items))

    report_context_denies = (
        DEFAULT_BITGET_REPORT_DERIVED_TERMINAL_DENIED_CONTEXT_SIGNAL_KEYS
        if str(exchange_name or "").strip().upper() == "BITGET"
        else ()
    )

    return FlashAllocatorConfig(
        min_score_to_trade=_settings_float(
            settings,
            (
                "panteon_flash_min_score_to_trade",
                "v2_flash_min_score_to_trade",
                "flash_min_score_to_trade",
            ),
            0.0,
        ),
        actionable_bonus=_settings_float(
            settings,
            (
                "panteon_flash_actionable_bonus",
                "v2_flash_actionable_bonus",
                "flash_actionable_bonus",
            ),
            0.25,
        ),
        no_data_score=_settings_float(
            settings,
            (
                "panteon_flash_no_data_score",
                "v2_flash_no_data_score",
                "flash_no_data_score",
            ),
            0.0,
        ),
        min_closed_trades_to_trade=int(_settings_float(
            settings,
            (
                "panteon_flash_min_closed_trades_to_trade",
                "v2_flash_min_closed_trades_to_trade",
                "flash_min_closed_trades_to_trade",
            ),
            3.0,
        )),
        min_pnl_pct_to_trade=_settings_float(
            settings,
            (
                "panteon_flash_min_pnl_pct_to_trade",
                "v2_flash_min_pnl_pct_to_trade",
                "flash_min_pnl_pct_to_trade",
            ),
            0.0,
        ),
        gate_pnl_per_trade_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_gate_pnl_per_trade_enabled",
                "v2_flash_gate_pnl_per_trade_enabled",
                "flash_gate_pnl_per_trade_enabled",
            ),
            False,
        ),
        global_health_gate_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_global_health_gate_enabled",
                "v2_flash_global_health_gate_enabled",
                "flash_global_health_gate_enabled",
            ),
            False,
        ),
        global_health_min_cum_pnl_pct=_settings_float(
            settings,
            (
                "panteon_flash_global_health_min_cum_pnl_pct",
                "v2_flash_global_health_min_cum_pnl_pct",
                "flash_global_health_min_cum_pnl_pct",
            ),
            -5.0,
        ),
        global_health_min_closed_trades=int(_settings_float(
            settings,
            (
                "panteon_flash_global_health_min_closed_trades",
                "v2_flash_global_health_min_closed_trades",
                "flash_global_health_min_closed_trades",
            ),
            50.0,
        )),
        real_loss_gate_enabled=_settings_bool(
            settings, ("v2_flash_real_loss_gate_enabled", "flash_real_loss_gate_enabled"), False,
        ),
        real_loss_gate_min_pnl_pct=_settings_float(
            settings, ("v2_flash_real_loss_gate_min_pnl_pct", "flash_real_loss_gate_min_pnl_pct"), -0.5,
        ),
        real_loss_gate_min_closed_trades=int(_settings_float(
            settings, ("v2_flash_real_loss_gate_min_closed_trades", "flash_real_loss_gate_min_closed_trades"), 5.0,
        )),
        fee_aware_admission_enabled=_settings_bool(
            settings, ("v2_flash_fee_aware_admission_enabled", "flash_fee_aware_admission_enabled"), False,
        ),
        fee_aware_round_trip_cost_pct=_settings_float(
            settings, ("v2_flash_fee_aware_round_trip_cost_pct", "flash_fee_aware_round_trip_cost_pct"), 0.06,
        ),
        controlled_exploration_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_enabled",
                "flash_controlled_exploration_enabled",
            ),
            False,
        ),
        controlled_exploration_allowed_reasons=_settings_csv_tuple(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_allowed_reasons",
                "flash_controlled_exploration_allowed_reasons",
            ),
        ),
        controlled_exploration_risk_mult=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_risk_mult",
                "flash_controlled_exploration_risk_mult",
            ),
            0.05,
        ),
        controlled_exploration_min_shadow_score=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_min_shadow_score",
                "flash_controlled_exploration_min_shadow_score",
            ),
            2.0,
        ),
        controlled_exploration_min_shadow_closed=int(_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_min_shadow_closed",
                "flash_controlled_exploration_min_shadow_closed",
            ),
            10.0,
        )),
        controlled_exploration_max_daily_trades=int(_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_max_daily_trades",
                "flash_controlled_exploration_max_daily_trades",
            ),
            1.0,
        )),
        controlled_exploration_max_open_positions=int(_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_max_open_positions",
                "flash_controlled_exploration_max_open_positions",
            ),
            1.0,
        )),
        controlled_exploration_min_rolling_expectancy=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_min_rolling_expectancy",
                "flash_controlled_exploration_min_rolling_expectancy",
            ),
            0.0,
        ),
        controlled_exploration_min_notional_sizing_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_min_notional_sizing_enabled",
                "flash_controlled_exploration_min_notional_sizing_enabled",
            ),
            False,
        ),
        controlled_exploration_account_equity_usd=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_account_equity_usd",
                "flash_controlled_exploration_account_equity_usd",
            ),
            0.0,
        ),
        controlled_exploration_capital_fraction=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_capital_fraction",
                "flash_controlled_exploration_capital_fraction",
            ),
            0.10,
        ),
        controlled_exploration_min_notional_max_risk_mult=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_min_notional_max_risk_mult",
                "flash_controlled_exploration_min_notional_max_risk_mult",
            ),
            0.10,
        ),
        controlled_exploration_default_min_notional_usd=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_default_min_notional_usd",
                "flash_controlled_exploration_default_min_notional_usd",
            ),
            5.0,
        ),
        controlled_exploration_apply_leverage_to_notional=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_apply_leverage_to_notional",
                "flash_controlled_exploration_apply_leverage_to_notional",
            ),
            False,
        ),
        controlled_exploration_max_leverage=_settings_float(
            settings,
            scoped_names(
                "v2_flash_controlled_exploration_max_leverage",
                "flash_controlled_exploration_max_leverage",
            ),
            1.0,
        ),
        causal_actor_router_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_causal_actor_router_enabled",
                "flash_causal_actor_router_enabled",
            ),
            False,
        ),
        causal_actor_router_min_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "v2_flash_causal_actor_router_min_closed_trades",
                "flash_causal_actor_router_min_closed_trades",
            ),
            5.0,
        )),
        causal_actor_router_min_expectancy=_settings_float(
            settings,
            scoped_names(
                "v2_flash_causal_actor_router_min_expectancy",
                "flash_causal_actor_router_min_expectancy",
            ),
            0.0,
        ),
        causal_actor_router_min_pnl_lcb=_settings_optional_float(
            settings,
            scoped_names(
                "v2_flash_causal_actor_router_min_pnl_lcb",
                "flash_causal_actor_router_min_pnl_lcb",
            ),
            None,
        ),
        causal_actor_router_exploration_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_causal_actor_router_exploration_enabled",
                "flash_causal_actor_router_exploration_enabled",
            ),
            False,
        ),
        causal_actor_router_exploration_risk_mult=_settings_float(
            settings,
            scoped_names(
                "v2_flash_causal_actor_router_exploration_risk_mult",
                "flash_causal_actor_router_exploration_risk_mult",
            ),
            0.05,
        ),
        promotion_derived_router_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_router_enabled",
                "flash_promotion_derived_router_enabled",
            ),
            False,
        ),
        promotion_derived_actor_labels=_settings_csv_tuple(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_actor_labels",
                "flash_promotion_derived_actor_labels",
            ),
        ),
        promotion_derived_dynamic_best_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_dynamic_best_enabled",
                "flash_promotion_derived_dynamic_best_enabled",
            ),
            False,
        ),
        promotion_derived_min_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_min_closed_trades",
                "flash_promotion_derived_min_closed_trades",
            ),
            5.0,
        )),
        promotion_derived_min_expectancy=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_min_expectancy",
                "flash_promotion_derived_min_expectancy",
            ),
            0.0,
        ),
        promotion_derived_risk_mult=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_risk_mult",
                "flash_promotion_derived_risk_mult",
            ),
            0.03,
        ),
        promotion_derived_min_notional_sizing_enabled=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_min_notional_sizing_enabled",
                "flash_promotion_derived_min_notional_sizing_enabled",
            ),
            False,
        ),
        promotion_derived_account_equity_usd=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_account_equity_usd",
                "flash_promotion_derived_account_equity_usd",
            ),
            0.0,
        ),
        promotion_derived_capital_fraction=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_capital_fraction",
                "flash_promotion_derived_capital_fraction",
            ),
            0.10,
        ),
        promotion_derived_min_notional_max_risk_mult=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_min_notional_max_risk_mult",
                "flash_promotion_derived_min_notional_max_risk_mult",
            ),
            0.10,
        ),
        promotion_derived_default_min_notional_usd=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_default_min_notional_usd",
                "flash_promotion_derived_default_min_notional_usd",
            ),
            5.0,
        ),
        promotion_derived_apply_leverage_to_notional=_settings_bool(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_apply_leverage_to_notional",
                "flash_promotion_derived_apply_leverage_to_notional",
            ),
            False,
        ),
        promotion_derived_max_leverage=_settings_float(
            settings,
            scoped_names(
                "v2_flash_promotion_derived_max_leverage",
                "flash_promotion_derived_max_leverage",
            ),
            1.0,
        ),
        churn_cooldown_bars=int(_settings_float(
            settings, ("v2_flash_churn_cooldown_bars", "flash_churn_cooldown_bars"), 0.0,
        )),
        churn_max_opens_per_day=int(_settings_float(
            settings, ("v2_flash_churn_max_opens_per_day", "flash_churn_max_opens_per_day"), 0.0,
        )),
        churn_anti_flip_enabled=_settings_bool(
            settings, ("v2_flash_churn_anti_flip_enabled", "flash_churn_anti_flip_enabled"), False,
        ),
        live_real_actor_whitelist=_settings_csv_tuple(
            settings,
            scoped_names(
                "panteon_flash_live_real_actor_whitelist",
                "v2_flash_live_real_actor_whitelist",
                "flash_live_real_actor_whitelist",
            ),
        ),
        regime_edge_gate_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_regime_edge_gate_enabled",
                "v2_flash_regime_edge_gate_enabled",
                "flash_regime_edge_gate_enabled",
            ),
            False,
        ),
        regime_edge_min_pnl_per_trade_pct=_settings_float(
            settings,
            (
                "panteon_flash_regime_edge_min_pnl_per_trade_pct",
                "v2_flash_regime_edge_min_pnl_per_trade_pct",
                "flash_regime_edge_min_pnl_per_trade_pct",
            ),
            0.0,
        ),
        regime_edge_min_closed_trades=int(_settings_float(
            settings,
            (
                "panteon_flash_regime_edge_min_closed_trades",
                "v2_flash_regime_edge_min_closed_trades",
                "flash_regime_edge_min_closed_trades",
            ),
            10.0,
        )),
        shadow_confirmation_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_confirmation_enabled",
                "v2_flash_shadow_confirmation_enabled",
                "flash_shadow_confirmation_enabled",
            ),
            False,
        ),
        shadow_symbol_confirmation_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_symbol_shadow_confirmation_enabled",
                "v2_flash_symbol_shadow_confirmation_enabled",
                "flash_symbol_shadow_confirmation_enabled",
            ),
            False,
        ),
        shadow_actor_fallback_confirmation_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_actor_fallback_confirmation_enabled",
                "v2_flash_shadow_actor_fallback_confirmation_enabled",
                "flash_shadow_actor_fallback_confirmation_enabled",
            ),
            False,
        ),
        shadow_base_fallback_confirmation_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_base_fallback_confirmation_enabled",
                "v2_flash_shadow_base_fallback_confirmation_enabled",
                "flash_shadow_base_fallback_confirmation_enabled",
            ),
            False,
        ),
        shadow_signal_handoff_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_signal_handoff_enabled",
                "v2_flash_shadow_signal_handoff_enabled",
                "flash_shadow_signal_handoff_enabled",
            ),
            False,
        ),
        shadow_actor_fallback_min_base_score=_settings_float(
            settings,
            (
                "panteon_flash_shadow_actor_fallback_min_base_score",
                "v2_flash_shadow_actor_fallback_min_base_score",
                "flash_shadow_actor_fallback_min_base_score",
            ),
            0.0,
        ),
        shadow_base_fallback_actor_keys=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_shadow_base_fallback_actor_keys",
                "v2_flash_shadow_base_fallback_actor_keys",
                "flash_shadow_base_fallback_actor_keys",
            ),
        ),
        actor_switch_margin=_settings_float(
            settings,
            (
                "panteon_flash_actor_switch_margin",
                "v2_flash_actor_switch_margin",
                "flash_actor_switch_margin",
            ),
            0.0,
        ),
        anchor_actor_keys=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_anchor_actor_keys",
                "v2_flash_anchor_actor_keys",
                "flash_anchor_actor_keys",
            ),
        ),
        portfolio_actor_keys=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_portfolio_actor_keys",
                "v2_flash_portfolio_actor_keys",
                "flash_portfolio_actor_keys",
            ),
        ),
        range_low_vol_real_actor_allowlist=_settings_csv_tuple(
            settings,
            scoped_names(
                "panteon_flash_range_low_vol_real_actor_allowlist",
                "v2_flash_range_low_vol_real_actor_allowlist",
                "flash_range_low_vol_real_actor_allowlist",
            ),
        ),
        flash_genetics_core_primary_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_genetics_core_primary_enabled",
                "v2_flash_genetics_core_primary_enabled",
                "flash_genetics_core_primary_enabled",
            ),
            False,
        ),
        flash_genetics_core_primary_labels=_settings_csv_tuple(
            settings,
            scoped_names(
                "panteon_flash_genetics_core_primary_labels",
                "v2_flash_genetics_core_primary_labels",
                "flash_genetics_core_primary_labels",
            ),
        ) or ("GeneticsCore",),
        portfolio_shadow_bootstrap_min_closed_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_portfolio_shadow_bootstrap_min_closed_enabled",
                "v2_flash_portfolio_shadow_bootstrap_min_closed_enabled",
                "flash_portfolio_shadow_bootstrap_min_closed_enabled",
            ),
            False,
        ),
        anchor_min_score_to_trade=_settings_optional_float(
            settings,
            (
                "panteon_flash_anchor_min_score_to_trade",
                "v2_flash_anchor_min_score_to_trade",
                "flash_anchor_min_score_to_trade",
            ),
            None,
        ),
        anchor_shadow_min_score=_settings_optional_float(
            settings,
            (
                "panteon_flash_anchor_shadow_min_score",
                "v2_flash_anchor_shadow_min_score",
                "flash_anchor_shadow_min_score",
            ),
            None,
        ),
        anchor_min_score_advantage=_settings_float(
            settings,
            (
                "panteon_flash_anchor_min_score_advantage",
                "v2_flash_anchor_min_score_advantage",
                "flash_anchor_min_score_advantage",
            ),
            0.0,
        ),
        prefer_solo_player_wrappers_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_prefer_solo_player_wrappers_enabled",
                "v2_flash_prefer_solo_player_wrappers_enabled",
                "flash_prefer_solo_player_wrappers_enabled",
            ),
            False,
        ),
        prefer_proven_solo_player_wrappers_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_prefer_proven_solo_player_wrappers_enabled",
                "v2_flash_prefer_proven_solo_player_wrappers_enabled",
                "flash_prefer_proven_solo_player_wrappers_enabled",
            ),
            False,
        ),
        proven_solo_min_score_advantage=_settings_float(
            settings,
            (
                "panteon_flash_proven_solo_min_score_advantage",
                "v2_flash_proven_solo_min_score_advantage",
                "flash_proven_solo_min_score_advantage",
            ),
            0.0,
        ),
        shadow_confirmation_min_score=_settings_float(
            settings,
            (
                "panteon_flash_shadow_min_score",
                "v2_flash_shadow_min_score",
                "flash_shadow_min_score",
            ),
            0.0,
        ),
        shadow_confirmation_min_closed_trades=int(_settings_float(
            settings,
            (
                "panteon_flash_shadow_min_closed_trades",
                "v2_flash_shadow_min_closed_trades",
                "flash_shadow_min_closed_trades",
            ),
            50.0,
        )),
        shadow_confirmation_min_full_open_closed_trades=int(_settings_float(
            settings,
            (
                "panteon_flash_shadow_min_full_open_closed_trades",
                "v2_flash_shadow_min_full_open_closed_trades",
                "flash_shadow_min_full_open_closed_trades",
            ),
            0.0,
        )),
        shadow_quality_confirmation_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_quality_confirmation_enabled",
                "v2_flash_shadow_quality_confirmation_enabled",
                "flash_shadow_quality_confirmation_enabled",
            ),
            False,
        ),
        shadow_confirmation_min_win_rate_pct=_settings_float(
            settings,
            (
                "panteon_flash_shadow_min_win_rate_pct",
                "v2_flash_shadow_min_win_rate_pct",
                "flash_shadow_min_win_rate_pct",
            ),
            0.0,
        ),
        shadow_confirmation_max_recent_downside_usd=_settings_float(
            settings,
            (
                "panteon_flash_shadow_max_recent_downside_usd",
                "v2_flash_shadow_max_recent_downside_usd",
                "flash_shadow_max_recent_downside_usd",
            ),
            0.0,
        ),
        shadow_confirmation_min_pnl_per_trade_lcb_usd=_settings_optional_float(
            settings,
            (
                "panteon_flash_shadow_min_pnl_per_trade_lcb_usd",
                "v2_flash_shadow_min_pnl_per_trade_lcb_usd",
                "flash_shadow_min_pnl_per_trade_lcb_usd",
            ),
            None,
        ),
        shadow_confirmation_pnl_per_trade_lcb_z=_settings_float(
            settings,
            (
                "panteon_flash_shadow_pnl_per_trade_lcb_z",
                "v2_flash_shadow_pnl_per_trade_lcb_z",
                "flash_shadow_pnl_per_trade_lcb_z",
            ),
            1.0,
        ),
        shadow_confirmation_pnl_per_trade_lcb_penalty_floor_usd=_settings_float(
            settings,
            (
                "panteon_flash_shadow_pnl_per_trade_lcb_penalty_floor_usd",
                "v2_flash_shadow_pnl_per_trade_lcb_penalty_floor_usd",
                "flash_shadow_pnl_per_trade_lcb_penalty_floor_usd",
            ),
            0.0,
        ),
        shadow_confirmation_pnl_per_trade_lcb_penalty_weight=_settings_float(
            settings,
            (
                "panteon_flash_shadow_pnl_per_trade_lcb_penalty_weight",
                "v2_flash_shadow_pnl_per_trade_lcb_penalty_weight",
                "flash_shadow_pnl_per_trade_lcb_penalty_weight",
            ),
            0.0,
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_sizing_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_pnl_lcb_risk_sizing_enabled",
                "v2_flash_shadow_pnl_lcb_risk_sizing_enabled",
                "flash_shadow_pnl_lcb_risk_sizing_enabled",
            ),
            False,
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_min_mult=_settings_float(
            settings,
            (
                "panteon_flash_shadow_pnl_lcb_risk_min_mult",
                "v2_flash_shadow_pnl_lcb_risk_min_mult",
                "flash_shadow_pnl_lcb_risk_min_mult",
            ),
            0.25,
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_floor_usd=_settings_float(
            settings,
            (
                "panteon_flash_shadow_pnl_lcb_risk_floor_usd",
                "v2_flash_shadow_pnl_lcb_risk_floor_usd",
                "flash_shadow_pnl_lcb_risk_floor_usd",
            ),
            0.0,
        ),
        shadow_confirmation_pnl_per_trade_lcb_risk_scale_usd=_settings_float(
            settings,
            (
                "panteon_flash_shadow_pnl_lcb_risk_scale_usd",
                "v2_flash_shadow_pnl_lcb_risk_scale_usd",
                "flash_shadow_pnl_lcb_risk_scale_usd",
            ),
            1.0,
        ),
        shadow_recovery_admission_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_shadow_recovery_admission_enabled",
                "v2_flash_shadow_recovery_admission_enabled",
                "flash_shadow_recovery_admission_enabled",
            ),
            False,
        ),
        shadow_recovery_actor_keys=_settings_csv_tuple(
            settings,
            scoped_names(
                "panteon_flash_shadow_recovery_actor_keys",
                "v2_flash_shadow_recovery_actor_keys",
                "flash_shadow_recovery_actor_keys",
            ),
        ),
        shadow_recovery_min_score=_settings_optional_float(
            settings,
            scoped_names(
                "panteon_flash_shadow_recovery_min_score",
                "v2_flash_shadow_recovery_min_score",
                "flash_shadow_recovery_min_score",
            ),
            None,
        ),
        shadow_recovery_min_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_shadow_recovery_min_closed_trades",
                "v2_flash_shadow_recovery_min_closed_trades",
                "flash_shadow_recovery_min_closed_trades",
            ),
            0.0,
        )),
        shadow_symbol_health_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_shadow_symbol_health_enabled",
                "v2_flash_shadow_symbol_health_enabled",
                "flash_shadow_symbol_health_enabled",
            ),
            False,
        ),
        shadow_symbol_health_min_closed_trades=int(_settings_float(
            settings,
            (
                "panteon_flash_shadow_symbol_health_min_closed_trades",
                "v2_flash_shadow_symbol_health_min_closed_trades",
                "flash_shadow_symbol_health_min_closed_trades",
            ),
            0.0,
        )),
        shadow_symbol_health_min_pnl_per_trade_lcb_usd=_settings_optional_float(
            settings,
            (
                "panteon_flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd",
                "v2_flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd",
                "flash_shadow_symbol_health_min_pnl_per_trade_lcb_usd",
            ),
            None,
        ),
        shadow_symbol_health_pnl_per_trade_lcb_penalty_floor_usd=_settings_float(
            settings,
            (
                "panteon_flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd",
                "v2_flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd",
                "flash_shadow_symbol_health_pnl_lcb_penalty_floor_usd",
            ),
            0.0,
        ),
        shadow_symbol_health_pnl_per_trade_lcb_penalty_weight=_settings_float(
            settings,
            (
                "panteon_flash_shadow_symbol_health_pnl_lcb_penalty_weight",
                "v2_flash_shadow_symbol_health_pnl_lcb_penalty_weight",
                "flash_shadow_symbol_health_pnl_lcb_penalty_weight",
            ),
            0.0,
        ),
        actor_risk_sizing_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_actor_risk_sizing_enabled",
                "v2_flash_actor_risk_sizing_enabled",
                "flash_actor_risk_sizing_enabled",
            ),
            False,
        ),
        actor_risk_min_mult=_settings_float(
            settings,
            (
                "panteon_flash_actor_risk_min_mult",
                "v2_flash_actor_risk_min_mult",
                "flash_actor_risk_min_mult",
            ),
            0.25,
        ),
        actor_risk_max_mult=_settings_float(
            settings,
            (
                "panteon_flash_actor_risk_max_mult",
                "v2_flash_actor_risk_max_mult",
                "flash_actor_risk_max_mult",
            ),
            1.0,
        ),
        actor_risk_edge_scale_pct=_settings_float(
            settings,
            (
                "panteon_flash_actor_risk_edge_scale_pct",
                "v2_flash_actor_risk_edge_scale_pct",
                "flash_actor_risk_edge_scale_pct",
            ),
            0.50,
        ),
        funding_score_weight=_settings_float(
            settings,
            (
                "panteon_flash_funding_score_weight",
                "v2_flash_funding_score_weight",
                "flash_funding_score_weight",
            ),
            0.0,
        ),
        funding_risk_mult_weight=_settings_float(
            settings,
            (
                "panteon_flash_funding_risk_mult_weight",
                "v2_flash_funding_risk_mult_weight",
                "flash_funding_risk_mult_weight",
            ),
            0.0,
        ),
        funding_risk_mult_cap=_settings_float(
            settings,
            (
                "panteon_flash_funding_risk_mult_cap",
                "v2_flash_funding_risk_mult_cap",
                "flash_funding_risk_mult_cap",
            ),
            0.25,
        ),
        no_trade_fee_saving_score_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_no_trade_fee_saving_score_enabled",
                "v2_flash_no_trade_fee_saving_score_enabled",
                "flash_no_trade_fee_saving_score_enabled",
            ),
            False,
        ),
        no_trade_default_fee_bps=_settings_float(
            settings,
            (
                "panteon_flash_no_trade_default_fee_bps",
                "v2_flash_no_trade_default_fee_bps",
                "flash_no_trade_default_fee_bps",
            ),
            0.0,
        ),
        volatility_risk_sizing_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_volatility_risk_sizing_enabled",
                "v2_flash_volatility_risk_sizing_enabled",
                "flash_volatility_risk_sizing_enabled",
            ),
            False,
        ),
        volatility_risk_target_pct=_settings_float(
            settings,
            (
                "panteon_flash_volatility_risk_target_pct",
                "v2_flash_volatility_risk_target_pct",
                "flash_volatility_risk_target_pct",
            ),
            2.0,
        ),
        volatility_risk_min_volatility_pct=_settings_float(
            settings,
            (
                "panteon_flash_volatility_risk_min_volatility_pct",
                "v2_flash_volatility_risk_min_volatility_pct",
                "flash_volatility_risk_min_volatility_pct",
            ),
            0.5,
        ),
        volatility_risk_max_mult=_settings_float(
            settings,
            (
                "panteon_flash_volatility_risk_max_mult",
                "v2_flash_volatility_risk_max_mult",
                "flash_volatility_risk_max_mult",
            ),
            2.0,
        ),
        genetics_probation_bypass_min_closed_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_genetics_probation_bypass_min_closed_enabled",
                "v2_flash_genetics_probation_bypass_min_closed_enabled",
                "flash_genetics_probation_bypass_min_closed_enabled",
            ),
            False,
        ),
        genetics_probation_bypass_trend_gate_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_genetics_probation_bypass_trend_gate_enabled",
                "v2_flash_genetics_probation_bypass_trend_gate_enabled",
                "flash_genetics_probation_bypass_trend_gate_enabled",
            ),
            False,
        ),
        genetics_probation_bypass_terminal_deny_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_genetics_probation_bypass_terminal_deny_enabled",
                "v2_flash_genetics_probation_bypass_terminal_deny_enabled",
                "flash_genetics_probation_bypass_terminal_deny_enabled",
            ),
            False,
        ),
        genetics_probation_bypass_regime_edge_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_genetics_probation_bypass_regime_edge_enabled",
                "v2_flash_genetics_probation_bypass_regime_edge_enabled",
                "flash_genetics_probation_bypass_regime_edge_enabled",
            ),
            False,
        ),
        genetics_probation_bypass_pnl_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_genetics_probation_bypass_pnl_enabled",
                "v2_flash_genetics_probation_bypass_pnl_enabled",
                "flash_genetics_probation_bypass_pnl_enabled",
            ),
            False,
        ),
        technical_overlay_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_technical_overlay_enabled",
                "v2_flash_technical_overlay_enabled",
                "flash_technical_overlay_enabled",
            ),
            False,
        ),
        technical_hard_gate_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_technical_hard_gate_enabled",
                "v2_flash_technical_hard_gate_enabled",
                "flash_technical_hard_gate_enabled",
            ),
            False,
        ),
        technical_score_bonus=_settings_float(
            settings,
            (
                "panteon_flash_technical_score_bonus",
                "v2_flash_technical_score_bonus",
                "flash_technical_score_bonus",
            ),
            0.10,
        ),
        technical_score_penalty=_settings_float(
            settings,
            (
                "panteon_flash_technical_score_penalty",
                "v2_flash_technical_score_penalty",
                "flash_technical_score_penalty",
            ),
            0.25,
        ),
        technical_rsi_long_min=_settings_float(
            settings,
            (
                "panteon_flash_technical_rsi_long_min",
                "v2_flash_technical_rsi_long_min",
                "flash_technical_rsi_long_min",
            ),
            45.0,
        ),
        technical_rsi_long_max=_settings_float(
            settings,
            (
                "panteon_flash_technical_rsi_long_max",
                "v2_flash_technical_rsi_long_max",
                "flash_technical_rsi_long_max",
            ),
            72.0,
        ),
        technical_rsi_short_min=_settings_float(
            settings,
            (
                "panteon_flash_technical_rsi_short_min",
                "v2_flash_technical_rsi_short_min",
                "flash_technical_rsi_short_min",
            ),
            28.0,
        ),
        technical_rsi_short_max=_settings_float(
            settings,
            (
                "panteon_flash_technical_rsi_short_max",
                "v2_flash_technical_rsi_short_max",
                "flash_technical_rsi_short_max",
            ),
            55.0,
        ),
        technical_macd_histogram_min_abs_pct=_settings_float(
            settings,
            (
                "panteon_flash_technical_macd_histogram_min_abs_pct",
                "v2_flash_technical_macd_histogram_min_abs_pct",
                "flash_technical_macd_histogram_min_abs_pct",
            ),
            0.0,
        ),
        technical_atr_risk_sizing_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_technical_atr_risk_sizing_enabled",
                "v2_flash_technical_atr_risk_sizing_enabled",
                "flash_technical_atr_risk_sizing_enabled",
            ),
            False,
        ),
        technical_atr_target_pct=_settings_float(
            settings,
            (
                "panteon_flash_technical_atr_target_pct",
                "v2_flash_technical_atr_target_pct",
                "flash_technical_atr_target_pct",
            ),
            2.0,
        ),
        technical_atr_min_pct=_settings_float(
            settings,
            (
                "panteon_flash_technical_atr_min_pct",
                "v2_flash_technical_atr_min_pct",
                "flash_technical_atr_min_pct",
            ),
            0.25,
        ),
        technical_atr_max_mult=_settings_float(
            settings,
            (
                "panteon_flash_technical_atr_max_mult",
                "v2_flash_technical_atr_max_mult",
                "flash_technical_atr_max_mult",
            ),
            1.5,
        ),
        selected_subset_score_boosts=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_selected_subset_score_boosts",
                "v2_flash_selected_subset_score_boosts",
                "flash_selected_subset_score_boosts",
            ),
        ),
        selected_subset_context_score_boosts=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_selected_subset_context_score_boosts",
                "v2_flash_selected_subset_context_score_boosts",
                "flash_selected_subset_context_score_boosts",
            ),
        ),
        selected_subset_do_not_demote_signal_keys=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_selected_subset_do_not_demote_signal_keys",
                "v2_flash_selected_subset_do_not_demote_signal_keys",
                "flash_selected_subset_do_not_demote_signal_keys",
            ),
        ),
        selected_subset_risk_mult_overrides=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_selected_subset_risk_mult_overrides",
                "v2_flash_selected_subset_risk_mult_overrides",
                "flash_selected_subset_risk_mult_overrides",
            ),
        ),
        selected_subset_context_risk_mult_overrides=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_selected_subset_context_risk_mult_overrides",
                "v2_flash_selected_subset_context_risk_mult_overrides",
                "flash_selected_subset_context_risk_mult_overrides",
            ),
        ),
        selected_subset_risk_min_mult=_settings_float(
            settings,
            (
                "panteon_flash_selected_subset_risk_min_mult",
                "v2_flash_selected_subset_risk_min_mult",
                "flash_selected_subset_risk_min_mult",
            ),
            0.75,
        ),
        selected_subset_risk_max_mult=_settings_float(
            settings,
            (
                "panteon_flash_selected_subset_risk_max_mult",
                "v2_flash_selected_subset_risk_max_mult",
                "flash_selected_subset_risk_max_mult",
            ),
            1.15,
        ),
        max_signals_per_actor=int(_settings_float(
            settings,
            (
                "panteon_flash_max_signals_per_actor",
                "v2_flash_max_signals_per_actor",
                "flash_max_signals_per_actor",
            ),
            0.0,
        )),
        open_overextension_guard_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_overextension_guard_enabled",
                "v2_flash_overextension_guard_enabled",
                "flash_overextension_guard_enabled",
            ),
            False,
        ),
        overextension_lookback_bars=int(_settings_float(
            settings,
            (
                "panteon_flash_overextension_lookback_bars",
                "v2_flash_overextension_lookback_bars",
                "flash_overextension_lookback_bars",
            ),
            12.0,
        )),
        short_overextension_return_floor_pct=_settings_float(
            settings,
            (
                "panteon_flash_short_overextension_return_floor_pct",
                "v2_flash_short_overextension_return_floor_pct",
                "flash_short_overextension_return_floor_pct",
            ),
            -8.0,
        ),
        long_overextension_return_ceiling_pct=_settings_float(
            settings,
            (
                "panteon_flash_long_overextension_return_ceiling_pct",
                "v2_flash_long_overextension_return_ceiling_pct",
                "flash_long_overextension_return_ceiling_pct",
            ),
            8.0,
        ),
        overextension_volatility_normalized_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_overextension_volatility_normalized_enabled",
                "v2_flash_overextension_volatility_normalized_enabled",
                "flash_overextension_volatility_normalized_enabled",
            ),
            False,
        ),
        short_overextension_z_floor=_settings_float(
            settings,
            (
                "panteon_flash_short_overextension_z_floor",
                "v2_flash_short_overextension_z_floor",
                "flash_short_overextension_z_floor",
            ),
            -2.5,
        ),
        long_overextension_z_ceiling=_settings_float(
            settings,
            (
                "panteon_flash_long_overextension_z_ceiling",
                "v2_flash_long_overextension_z_ceiling",
                "flash_long_overextension_z_ceiling",
            ),
            2.5,
        ),
        overextension_min_volatility_pct=_settings_float(
            settings,
            (
                "panteon_flash_overextension_min_volatility_pct",
                "v2_flash_overextension_min_volatility_pct",
                "flash_overextension_min_volatility_pct",
            ),
            0.1,
        ),
        denied_actor_keys=_settings_csv_tuple(
            settings,
            scoped_names(
                "panteon_flash_denied_actor_keys",
                "v2_flash_denied_actor_keys",
                "flash_denied_actor_keys",
            ),
        ),
        denied_signal_keys=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_denied_signal_keys",
                "v2_flash_denied_signal_keys",
                "flash_denied_signal_keys",
            ),
        ),
        terminal_denied_signal_keys=csv_merged(
            scoped_names(
                "panteon_flash_terminal_denied_signal_keys",
                "v2_flash_terminal_denied_signal_keys",
                "flash_terminal_denied_signal_keys",
            ),
            scoped_names(
                "panteon_flash_terminal_denied_signal_keys_extra",
                "v2_flash_terminal_denied_signal_keys_extra",
                "flash_terminal_denied_signal_keys_extra",
            ),
        ),
        terminal_denied_context_signal_keys=tuple(
            dict.fromkeys(
                (
                    *report_context_denies,
                    *csv_merged(
                        scoped_names(
                            "panteon_flash_terminal_denied_context_signal_keys",
                            "v2_flash_terminal_denied_context_signal_keys",
                            "flash_terminal_denied_context_signal_keys",
                        ),
                        scoped_names(
                            "panteon_flash_terminal_denied_context_signal_keys_extra",
                            "v2_flash_terminal_denied_context_signal_keys_extra",
                            "flash_terminal_denied_context_signal_keys_extra",
                        ),
                    ),
                )
            )
        ),
        denied_open_symbols=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_denied_open_symbols",
                "v2_flash_denied_open_symbols",
                "flash_denied_open_symbols",
            ),
        ),
        denied_open_regimes=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_denied_open_regimes",
                "v2_flash_denied_open_regimes",
                "flash_denied_open_regimes",
            ),
        ),
        degradation_guard_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_degradation_guard_enabled",
                "v2_flash_degradation_guard_enabled",
                "flash_degradation_guard_enabled",
            ),
            False,
        ),
        degradation_actor_guard_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_degradation_actor_guard_enabled",
                "v2_flash_degradation_actor_guard_enabled",
                "flash_degradation_actor_guard_enabled",
            ),
            False,
        ),
        degradation_actor_scope=_settings_str(
            settings,
            scoped_names(
                "panteon_flash_degradation_actor_scope",
                "v2_flash_degradation_actor_scope",
                "flash_degradation_actor_scope",
            ),
            "actor",
        ),
        degradation_signal_cooldown_bars=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_signal_cooldown_bars",
                "v2_flash_degradation_signal_cooldown_bars",
                "flash_degradation_signal_cooldown_bars",
            ),
            0.0,
        )),
        degradation_actor_cooldown_bars=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_actor_cooldown_bars",
                "v2_flash_degradation_actor_cooldown_bars",
                "flash_degradation_actor_cooldown_bars",
            ),
            0.0,
        )),
        degradation_symbol_guard_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_degradation_symbol_guard_enabled",
                "v2_flash_degradation_symbol_guard_enabled",
                "flash_degradation_symbol_guard_enabled",
            ),
            False,
        ),
        degradation_symbol_cooldown_bars=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_symbol_cooldown_bars",
                "v2_flash_degradation_symbol_cooldown_bars",
                "flash_degradation_symbol_cooldown_bars",
            ),
            0.0,
        )),
        degradation_symbol_lookback_bars=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_symbol_lookback_bars",
                "v2_flash_degradation_symbol_lookback_bars",
                "flash_degradation_symbol_lookback_bars",
            ),
            0.0,
        )),
        degradation_symbol_window_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_symbol_window_closed_trades",
                "v2_flash_degradation_symbol_window_closed_trades",
                "flash_degradation_symbol_window_closed_trades",
            ),
            0.0,
        )),
        degradation_symbol_min_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_symbol_min_closed_trades",
                "v2_flash_degradation_symbol_min_closed_trades",
                "flash_degradation_symbol_min_closed_trades",
            ),
            0.0,
        )),
        degradation_symbol_max_recent_pnl_usd=_settings_optional_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_symbol_max_recent_pnl_usd",
                "v2_flash_degradation_symbol_max_recent_pnl_usd",
                "flash_degradation_symbol_max_recent_pnl_usd",
            ),
            None,
        ),
        degradation_window_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_window_closed_trades",
                "v2_flash_degradation_window_closed_trades",
                "flash_degradation_window_closed_trades",
            ),
            3.0,
        )),
        degradation_min_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_min_closed_trades",
                "v2_flash_degradation_min_closed_trades",
                "flash_degradation_min_closed_trades",
            ),
            3.0,
        )),
        degradation_max_recent_pnl_usd=_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_max_recent_pnl_usd",
                "v2_flash_degradation_max_recent_pnl_usd",
                "flash_degradation_max_recent_pnl_usd",
            ),
            -25.0,
        ),
        degradation_signal_min_pnl_per_trade_lcb_usd=_settings_optional_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_signal_min_pnl_per_trade_lcb_usd",
                "v2_flash_degradation_signal_min_pnl_per_trade_lcb_usd",
                "flash_degradation_signal_min_pnl_per_trade_lcb_usd",
            ),
            None,
        ),
        degradation_pnl_per_trade_lcb_z=_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_pnl_per_trade_lcb_z",
                "v2_flash_degradation_pnl_per_trade_lcb_z",
                "flash_degradation_pnl_per_trade_lcb_z",
            ),
            1.0,
        ),
        degradation_signal_risk_sizing_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_degradation_signal_risk_sizing_enabled",
                "v2_flash_degradation_signal_risk_sizing_enabled",
                "flash_degradation_signal_risk_sizing_enabled",
            ),
            False,
        ),
        degradation_signal_risk_mult=_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_signal_risk_mult",
                "v2_flash_degradation_signal_risk_mult",
                "flash_degradation_signal_risk_mult",
            ),
            0.25,
        ),
        degradation_reserve_actor_cap=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_degradation_reserve_actor_cap",
                "v2_flash_degradation_reserve_actor_cap",
                "flash_degradation_reserve_actor_cap",
            ),
            False,
        ),
        degradation_recovery_enabled=_settings_bool(
            settings,
            scoped_names(
                "panteon_flash_degradation_recovery_enabled",
                "v2_flash_degradation_recovery_enabled",
                "flash_degradation_recovery_enabled",
            ),
            False,
        ),
        degradation_recovery_min_closed_trades=int(_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_recovery_min_closed_trades",
                "v2_flash_degradation_recovery_min_closed_trades",
                "flash_degradation_recovery_min_closed_trades",
            ),
            3.0,
        )),
        degradation_recovery_min_recent_pnl_usd=_settings_float(
            settings,
            scoped_names(
                "panteon_flash_degradation_recovery_min_recent_pnl_usd",
                "v2_flash_degradation_recovery_min_recent_pnl_usd",
                "flash_degradation_recovery_min_recent_pnl_usd",
            ),
            0.0,
        ),
        promotion_manifest_enabled=_settings_bool(
            settings,
            (
                "panteon_flash_promotion_manifest_enabled",
                "v2_flash_promotion_manifest_enabled",
                "flash_promotion_manifest_enabled",
            ),
            False,
        ),
        promoted_actor_cap_overrides=_settings_csv_tuple(
            settings,
            (
                "panteon_flash_promoted_actor_cap_overrides",
                "v2_flash_promoted_actor_cap_overrides",
                "flash_promoted_actor_cap_overrides",
            ),
        ),
    )


def _resolve_flash_enabled(exchange_name: str) -> bool:
    env_default = (
        _env_flag(f"{exchange_name.upper()}_PANTEON_FLASH_ENABLED", False)
        or _env_flag(f"{exchange_name.upper()}_PANTEON_FLASH", False)
        or _env_flag("PANTEON_FLASH_ENABLED", False)
        or _env_flag("PANTEON_FLASH", False)
    )
    return _flash_enabled_from_settings(
        _load_exchange_settings(exchange_name),
        default=env_default,
    )


def _resolve_futures_signal_fixes_enabled(exchange_name: str) -> bool:
    exchange_key = str(exchange_name or "").strip().upper()
    env_default = (
        _env_flag(f"{exchange_key}_PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED", False)
        or _env_flag(f"{exchange_key}_PANTEON_FUTURES_SIGNAL_FIXES_ENABLED", False)
        or _env_flag("PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED", False)
        or _env_flag("PANTEON_FUTURES_SIGNAL_FIXES_ENABLED", False)
    )
    settings = _load_exchange_settings(exchange_name)
    return _settings_bool(
        settings,
        _exchange_scoped_setting_names(
            exchange_name,
            "v2_futures_signal_fixes_enabled",
        )
        + _exchange_scoped_setting_names(
            exchange_name,
            "panteon_futures_signal_fixes_enabled",
        )
        + _exchange_scoped_setting_names(
            exchange_name,
            "v2_futures_replay_signal_fixes_enabled",
        ),
        env_default,
    )


def _resolve_flash_component_memory(exchange_name: str) -> Optional[ComponentMemory]:
    exchange_key = str(exchange_name or "").strip().upper()
    raw = (
        os.getenv(f"{exchange_key}_PANTEON_FLASH_COMPONENT_MEMORY_JSONL")
        or os.getenv("PANTEON_FLASH_COMPONENT_MEMORY_JSONL")
        or _settings_str(
            _load_exchange_settings(exchange_name),
            _exchange_scoped_setting_names(
                exchange_name,
                "v2_flash_component_memory_jsonl",
            )
            + _exchange_scoped_setting_names(
                exchange_name,
                "flash_component_memory_jsonl",
            ),
            "",
        )
    )
    path_text = str(raw or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Flash component memory JSONL not found: {path}")
    memory = ComponentMemory.from_jsonl(path)
    log.info("[%s] Flash component memory loaded: %s", exchange_key, path)
    return memory


def _resolve_scoring_config(exchange_name: str) -> ScoringConfig:
    """Scoring overrides из settings: per-trade PnL + consistency-tilt веса."""
    settings = _load_exchange_settings(exchange_name)
    use_per_trade = _settings_bool(
        settings,
        (
            "panteon_use_per_trade_pnl_score",
            "v2_use_per_trade_pnl_score",
            "use_per_trade_pnl_score",
        ),
        False,
    )
    pnl_w = _settings_float(
        settings, ("v2_scoring_pnl_weight", "scoring_pnl_weight"),
        DEFAULT_SCORING.pnl_weight,
    )
    sharpe_w = _settings_float(
        settings, ("v2_scoring_sharpe_weight", "scoring_sharpe_weight"),
        DEFAULT_SCORING.sharpe_weight,
    )
    win_div = _settings_float(
        settings, ("v2_scoring_win_bonus_divisor", "scoring_win_bonus_divisor"),
        DEFAULT_SCORING.win_bonus_divisor,
    )
    min_eligible = _settings_float(
        settings, ("v2_min_eligible_score", "min_eligible_score"),
        DEFAULT_SCORING.min_eligible_score,
    )
    if (
        not use_per_trade
        and pnl_w == DEFAULT_SCORING.pnl_weight
        and sharpe_w == DEFAULT_SCORING.sharpe_weight
        and win_div == DEFAULT_SCORING.win_bonus_divisor
        and min_eligible == DEFAULT_SCORING.min_eligible_score
    ):
        return DEFAULT_SCORING
    scale = _settings_float(
        settings,
        ("panteon_per_trade_pnl_scale", "v2_per_trade_pnl_scale", "per_trade_pnl_scale"),
        1.0,
    )
    return ScoringConfig(
        pnl_weight=pnl_w,
        sharpe_weight=sharpe_w,
        win_bonus_divisor=win_div,
        use_per_trade_pnl=use_per_trade,
        per_trade_pnl_scale=scale,
        min_eligible_score=min_eligible,
    )


def _resolve_voting_directional(exchange_name: str) -> bool:
    """Phase 2/A1: directional WeightedConsensus для ансамблей (opt-in)."""
    return _settings_bool(
        _load_exchange_settings(exchange_name),
        (
            "panteon_voting_directional",
            "v2_voting_directional",
            "voting_directional",
        ),
        False,
    )


def _resolve_perf_max_returns_history(exchange_name: str) -> int:
    """Phase 3/C6: ограничение длины returns (0 = безлимит)."""
    return int(_settings_float(
        _load_exchange_settings(exchange_name),
        (
            "panteon_perf_max_returns_history",
            "v2_perf_max_returns_history",
            "perf_max_returns_history",
        ),
        0.0,
    ))


def _resolve_flash_legacy_real_agents(exchange_name: str) -> tuple[bool, tuple[str, ...]]:
    settings = _load_exchange_settings(exchange_name)
    env_default = (
        _env_flag(f"{exchange_name.upper()}_PANTEON_FLASH_LEGACY_REAL_AGENTS", False)
        or _env_flag("PANTEON_FLASH_LEGACY_REAL_AGENTS", False)
    )
    enabled = _flash_legacy_real_agents_enabled_from_settings(
        settings,
        exchange_name,
        default=env_default,
    )
    labels = _flash_legacy_real_agent_labels_from_settings(settings, exchange_name)
    return enabled, labels


def _resolve_flash_allocator_config(exchange_name: str) -> FlashAllocatorConfig:
    return _flash_allocator_config_from_settings(
        _load_exchange_settings(exchange_name),
        exchange_name=exchange_name,
    )


def _flash_stale_position_exit_config_from_settings(settings: dict) -> dict[str, object]:
    return {
        "enabled": _settings_bool(
            settings,
            (
                "panteon_flash_stale_position_exit_enabled",
                "v2_flash_stale_position_exit_enabled",
                "flash_stale_position_exit_enabled",
            ),
            False,
        ),
        "max_age_bars": max(0, int(_settings_float(
            settings,
            (
                "panteon_flash_stale_position_exit_max_age_bars",
                "v2_flash_stale_position_exit_max_age_bars",
                "flash_stale_position_exit_max_age_bars",
            ),
            168.0,
        ))),
        "require_nonpositive_unrealized": _settings_bool(
            settings,
            (
                "panteon_flash_stale_position_exit_require_nonpositive_unrealized",
                "v2_flash_stale_position_exit_require_nonpositive_unrealized",
                "flash_stale_position_exit_require_nonpositive_unrealized",
            ),
            True,
        ),
    }


def _resolve_flash_stale_position_exit_config(exchange_name: str) -> dict[str, object]:
    return _flash_stale_position_exit_config_from_settings(
        _load_exchange_settings(exchange_name)
    )


def _flash_partial_profit_lock_config_from_settings(settings: dict) -> dict[str, object]:
    close_fraction = _settings_float(
        settings,
        (
            "panteon_flash_partial_profit_lock_close_fraction",
            "v2_flash_partial_profit_lock_close_fraction",
            "flash_partial_profit_lock_close_fraction",
        ),
        0.5,
    )
    return {
        "enabled": _settings_bool(
            settings,
            (
                "panteon_flash_partial_profit_lock_enabled",
                "v2_flash_partial_profit_lock_enabled",
                "flash_partial_profit_lock_enabled",
            ),
            False,
        ),
        "trigger_pnl_pct": max(0.0, _settings_float(
            settings,
            (
                "panteon_flash_partial_profit_lock_trigger_pnl_pct",
                "v2_flash_partial_profit_lock_trigger_pnl_pct",
                "flash_partial_profit_lock_trigger_pnl_pct",
            ),
            1.5,
        )),
        "close_fraction": min(1.0, max(1e-9, close_fraction)),
        "min_age_bars": max(0, int(_settings_float(
            settings,
            (
                "panteon_flash_partial_profit_lock_min_age_bars",
                "v2_flash_partial_profit_lock_min_age_bars",
                "flash_partial_profit_lock_min_age_bars",
            ),
            2.0,
        ))),
        "skip_protected_signal_keys": _settings_bool(
            settings,
            (
                "panteon_flash_partial_profit_lock_skip_protected_signal_keys",
                "v2_flash_partial_profit_lock_skip_protected_signal_keys",
                "flash_partial_profit_lock_skip_protected_signal_keys",
            ),
            True,
        ),
        "skip_signal_keys": _settings_csv_tuple(
            settings,
            (
                "panteon_flash_partial_profit_lock_skip_signal_keys",
                "v2_flash_partial_profit_lock_skip_signal_keys",
                "flash_partial_profit_lock_skip_signal_keys",
            ),
        ),
    }


def _resolve_flash_partial_profit_lock_config(exchange_name: str) -> dict[str, object]:
    return _flash_partial_profit_lock_config_from_settings(
        _load_exchange_settings(exchange_name)
    )


def _resolve_include_genetics(
    value: Optional[bool],
    exchange_name: str = "",
) -> bool:
    if value is not None:
        return bool(value)
    raw_env = os.getenv("PANTEON_V2_LOAD_GENETICS")
    if raw_env is not None:
        return _env_flag("PANTEON_V2_LOAD_GENETICS", False)
    settings = _load_exchange_settings(exchange_name) if exchange_name else {}
    return _settings_bool(
        settings,
        _exchange_scoped_setting_names(exchange_name, "v2_load_genetics")
        + _exchange_scoped_setting_names(exchange_name, "load_genetics")
        + _exchange_scoped_setting_names(exchange_name, "agent_genetics"),
        False,
    )


def _resolve_genetics_shadow_only(
    value: Optional[bool],
    exchange_name: str = "",
) -> bool:
    if value is not None:
        return bool(value)
    raw_env = os.getenv("PANTEON_V2_GENETICS_SHADOW_ONLY")
    if raw_env is not None:
        return _env_flag("PANTEON_V2_GENETICS_SHADOW_ONLY", True)
    settings = _load_exchange_settings(exchange_name) if exchange_name else {}
    return _settings_bool(
        settings,
        _exchange_scoped_setting_names(exchange_name, "v2_genetics_shadow_only")
        + _exchange_scoped_setting_names(exchange_name, "genetics_shadow_only"),
        True,
    )


def _resolve_genetics_shadow_only_quarantine_exempt_labels(
    exchange_name: str = "",
) -> tuple[str, ...]:
    raw_env = os.getenv("PANTEON_V2_GENETICS_SHADOW_ONLY_QUARANTINE_EXEMPT_LABELS")
    if raw_env is not None:
        return _settings_csv_tuple(
            {"env": raw_env},
            ("env",),
        )
    settings = _load_exchange_settings(exchange_name) if exchange_name else {}
    return _settings_csv_tuple(
        settings,
        _exchange_scoped_setting_names(
            exchange_name,
            "v2_genetics_shadow_only_quarantine_exempt_labels",
        )
        + _exchange_scoped_setting_names(
            exchange_name,
            "genetics_shadow_only_quarantine_exempt_labels",
        ),
    )


def _seed_quarantine_with_shadow_only_genetics(
    seed_quarantine: Sequence[str],
    registered: Sequence[str],
    *,
    include_genetics: bool,
    genetics_shadow_only: bool,
    quarantine_exempt_labels: Sequence[str] = (),
) -> tuple:
    if not include_genetics or not genetics_shadow_only:
        return tuple(seed_quarantine)

    optional = set(optional_labels())
    exempt = {
        str(label).strip()
        for label in quarantine_exempt_labels
        if str(label).strip()
    }
    merged = []
    seen = set()
    for label in list(seed_quarantine) + [
        label for label in registered
        if label in optional or str(label).startswith("Genetics")
    ]:
        if label in exempt:
            continue
        if label in seen:
            continue
        merged.append(label)
        seen.add(label)
    return tuple(merged)


def _apply_genetics_position_limit(
    registry: AgentRegistry,
    *,
    max_open_positions: int,
) -> List[str]:
    max_pos = int(max_open_positions)
    changed: List[str] = []
    for agent in registry.all_agents():
        label = str(getattr(agent, "label", "") or "")
        if not label.startswith("Genetics"):
            continue
        target = getattr(agent, "v1_agent", agent)
        try:
            setattr(target, "MAX_POS", max_pos)
        except Exception:
            log.warning("Failed to apply genetics MAX_POS=%s to %s", max_pos, label, exc_info=True)
            continue
        changed.append(label)
    return changed


def _should_fail_closed_after_bridge_error(
    *,
    mode: str,
    polling_session_dir: Optional[str],
    allow_live_feed_fallback: bool,
) -> bool:
    return (
        mode != "paper"
        and not polling_session_dir
        and not allow_live_feed_fallback
    )


def _perf_has_state(perf) -> bool:
    try:
        return bool((perf.snapshot().get("state") or {}))
    except Exception:
        return False


def _default_v1_memory_paths(exchange_name: str) -> List[str]:
    name = exchange_name.upper()
    repo_root = Path(__file__).resolve().parents[3]
    memory_dir = repo_root / "state" / "memory"
    candidates = [
        memory_dir / f"Real_Player_Memory_{name}.txt",
        memory_dir / f"Real_Player_Aggregator_Memory_{name}.txt",
    ]
    return [str(path) for path in candidates if path.exists()]


def _normalize_migration_paths(
    migrate_from_v1: Optional[Union[str, Sequence[str]]],
    *,
    exchange_name: str,
) -> List[str]:
    if migrate_from_v1 is None:
        return _default_v1_memory_paths(exchange_name)
    if isinstance(migrate_from_v1, (str, os.PathLike)):
        return [str(migrate_from_v1)]
    return [str(path) for path in migrate_from_v1]


def _retro_prior_disabled() -> bool:
    return str(os.getenv("PANTEON_V2_DISABLE_RETRO_PRIOR_MEMORY") or "").strip().upper() in {
        "1",
        "ON",
        "TRUE",
        "YES",
        "Y",
    }


def _default_retro_prior_memory_path(
    *,
    snapshot_path: Optional[str],
    exchange_name: str,
) -> Optional[str]:
    if _retro_prior_disabled():
        return None
    name = str(exchange_name or "").strip().upper()
    env_path = (
        os.getenv(f"PANTEON_V2_{name}_RETRO_PRIOR_MEMORY_PATH")
        or os.getenv("PANTEON_V2_RETRO_PRIOR_MEMORY_PATH")
    )
    if env_path:
        return str(env_path)
    if name != "MEXC" or not snapshot_path:
        return None
    candidate = Path(snapshot_path).with_name("mexc_retro_prior_memory.json")
    return str(candidate) if candidate.exists() else None


def _apply_retro_prior_memory(
    *,
    perf,
    snapshot_path: Optional[str],
    exchange_name: str,
    real_perf=None,
    order_ledger=None,
    position_tracker=None,
    shadow_positions_target: Optional[dict] = None,
) -> None:
    prior_path = _default_retro_prior_memory_path(
        snapshot_path=snapshot_path,
        exchange_name=exchange_name,
    )
    if not prior_path:
        return
    report = load_retro_prior_file_into_memory(
        perf,
        prior_path,
        expected_exchange=exchange_name,
        fill_missing_only=True,
    )
    for warning in report.warnings[:5]:
        log.warning("[%s] retro prior memory: %s", exchange_name, warning)
    if report.applied_pairs <= 0:
        if report.total_pairs:
            log.info(
                "[%s] retro prior memory loaded from %s, no missing pairs applied "
                "(existing=%d invalid=%d)",
                exchange_name,
                prior_path,
                report.skipped_existing_pairs,
                report.skipped_invalid_pairs,
            )
        return
    log.info(
        "[%s] retro prior memory applied from %s: applied=%d existing=%d invalid=%d",
        exchange_name,
        prior_path,
        report.applied_pairs,
        report.skipped_existing_pairs,
        report.skipped_invalid_pairs,
    )
    if snapshot_path:
        save_v2_snapshot(
            perf,
            snapshot_path,
            real_perf=real_perf,
            order_ledger=order_ledger,
            position_tracker=position_tracker,
            shadow_positions=shadow_positions_target,
        )
        log.info("[%s] Saved v2-snapshot after retro prior merge -> %s", exchange_name, snapshot_path)


def _load_or_migrate_state(
    *,
    perf,
    real_perf=None,
    qm=None,
    order_ledger=None,
    position_tracker=None,
    shadow_positions_target: Optional[dict] = None,
    snapshot_path: Optional[str],
    migrate_from_v1: Optional[Union[str, Sequence[str]]],
    exchange_name: str,
) -> None:
    loaded = False
    if snapshot_path and os.path.exists(snapshot_path):
        loaded = load_v2_snapshot(
            perf,
            snapshot_path,
            real_perf=real_perf,
            order_ledger=order_ledger,
            position_tracker=position_tracker,
            shadow_positions_target=shadow_positions_target,
            qm=qm,
        )
        if loaded:
            if _perf_has_state(perf):
                log.info("Loaded v2-snapshot from %s", snapshot_path)
                _apply_retro_prior_memory(
                    perf=perf,
                    snapshot_path=snapshot_path,
                    exchange_name=exchange_name,
                    real_perf=real_perf,
                    order_ledger=order_ledger,
                    position_tracker=position_tracker,
                    shadow_positions_target=shadow_positions_target,
                )
                return
            log.warning(
                "Loaded v2-snapshot from %s, but it has no ratings; "
                "trying v1 memory migration",
                snapshot_path,
            )

    paths = _normalize_migration_paths(
        migrate_from_v1,
        exchange_name=exchange_name,
    )
    if not paths:
        if not loaded:
            log.info("[%s] no v1 memory files found for migration", exchange_name)
        _apply_retro_prior_memory(
            perf=perf,
            snapshot_path=snapshot_path,
            exchange_name=exchange_name,
            real_perf=real_perf,
            order_ledger=order_ledger,
            position_tracker=position_tracker,
            shadow_positions_target=shadow_positions_target,
        )
        return

    report = migrate_from_v1_memory_files(paths, perf)
    if report.n_regime_pairs:
        log.info(
            "Migrated from v1: %d labels, %d pairs from %d file(s)",
            report.n_labels,
            report.n_regime_pairs,
            len(paths),
        )
        if snapshot_path:
            save_v2_snapshot(
                perf,
                snapshot_path,
                real_perf=real_perf,
                order_ledger=order_ledger,
                position_tracker=position_tracker,
                shadow_positions=shadow_positions_target,
                qm=qm,
            )
            log.info("Saved migrated v2-snapshot -> %s", snapshot_path)
        _apply_retro_prior_memory(
            perf=perf,
            snapshot_path=snapshot_path,
            exchange_name=exchange_name,
            real_perf=real_perf,
            order_ledger=order_ledger,
            position_tracker=position_tracker,
            shadow_positions_target=shadow_positions_target,
        )
    else:
        log.warning(
            "[%s] v1 memory migration found no ratings in: %s",
            exchange_name,
            ", ".join(paths),
        )
        _apply_retro_prior_memory(
            perf=perf,
            snapshot_path=snapshot_path,
            exchange_name=exchange_name,
            real_perf=real_perf,
            order_ledger=order_ledger,
            position_tracker=position_tracker,
            shadow_positions_target=shadow_positions_target,
        )
    for warning in report.warnings[:5]:
        log.warning("[%s] v1 migration: %s", exchange_name, warning)


# ────────────────────────────────────────────────────────────────────
# Exchange adapter resolution
# ────────────────────────────────────────────────────────────────────


def resolve_exchange(exchange_name: str, *, mode: str) -> Exchange:
    """Реальный Exchange-адаптер или FakeExchange.

    В live_futures используются bitget_adapter.py / mexc_adapter.py.
    FakeExchange остаётся только для paper или неподдержанной биржи.
    """
    name = exchange_name.upper()
    if _is_virtual_exchange_mode(mode):
        log.info("[%s] %s mode → FakeExchange", name, mode)
        return FakeExchange(name=f"{name}-PAPER")

    if name == "BITGET":
        try:
            from .bitget_adapter import BitgetExchangeAdapter  # noqa
            log.info("[BITGET] real adapter detected")
            return BitgetExchangeAdapter()
        except ImportError:
            log.warning(
                "[BITGET] no bitget_adapter.py — using FakeExchange (DRYRUN). "
                "Implement bitget_adapter.py based on exchange_adapter_template.py"
            )
            return FakeExchange(name="BITGET-DRYRUN")

    if name == "MEXC":
        try:
            from .mexc_adapter import MexcExchangeAdapter  # noqa
            log.info("[MEXC] real adapter detected")
            return MexcExchangeAdapter()
        except ImportError:
            log.warning(
                "[MEXC] no mexc_adapter.py — using FakeExchange (DRYRUN). "
                "Implement mexc_adapter.py based on exchange_adapter_template.py"
            )
            return FakeExchange(name="MEXC-DRYRUN")

    log.warning("Unknown exchange %s; using FakeExchange", name)
    return FakeExchange(name=f"{name}-DRYRUN")


def _recover_exchange_positions(pipeline) -> int:
    """Seed the v2 tracker from live exchange positions after restart."""
    exchange = getattr(getattr(pipeline, "executor", None), "_exchange", None)
    getter = getattr(exchange, "get_all_positions", None)
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    if not callable(getter) or tracker is None:
        return 0

    try:
        exchange_positions = getter() or {}
    except Exception:
        log.warning("[%s] exchange open-position recovery failed",
                    pipeline.exchange_name, exc_info=True)
        return 0

    memory_by_symbol = _open_memory_by_symbol(pipeline)
    recovered = 0
    external = 0
    adopted = 0
    preserved = 0
    for raw_sym, exchange_pos in exchange_positions.items():
        sym = str(getattr(exchange_pos, "sym", raw_sym) or raw_sym).upper()
        if not sym:
            continue
        memory = memory_by_symbol.get(sym, {})
        qty = _float_or_zero(getattr(exchange_pos, "qty", None), memory.get("qty"))
        entry = _float_or_zero(getattr(exchange_pos, "entry", None), memory.get("entry_price"))
        if qty <= 0 or entry <= 0:
            log.warning(
                "[%s] skipping recovered position %s: qty=%.8f entry=%.8f",
                pipeline.exchange_name,
                sym,
                qty,
                entry,
            )
            continue
        side = str(getattr(exchange_pos, "side", memory.get("side", "long")) or "long").lower()
        if side not in ("long", "short"):
            side = "long"
        existing = tracker.get(sym)
        if existing is not None and str(existing.side).lower() == side:
            tracker.force_set(TrackedPosition(
                open_signal_id=existing.open_signal_id,
                sym=sym,
                side=side,
                entry_price=entry,
                qty=qty,
                fee_open=existing.fee_open,
                by_player=existing.by_player,
                by_agent=existing.by_agent,
                opened_at=existing.opened_at,
                opened_bar=existing.opened_bar,
                open_action=existing.open_action,
                open_regime=existing.open_regime,
                funding_open=existing.funding_open,
                partial_profit_locked=existing.partial_profit_locked,
            ))
            preserved += 1
            recovered += 1
            continue
        if existing is not None:
            log.warning(
                "[%s] exchange position %s side changed from tracker=%s to exchange=%s; "
                "not preserving tracker provenance",
                pipeline.exchange_name,
                sym,
                existing.side,
                side,
            )
            memory = {}
        should_adopt = (
            not memory.get("by_player")
            and not memory.get("by_agent")
            and should_adopt_existing_position(pipeline, sym)
        )
        if should_adopt:
            by_player, by_agent = adopted_position_labels(pipeline)
        else:
            by_player = str(memory.get("by_player") or "RecoveredExchangePosition")
            by_agent = str(memory.get("by_agent") or "")
        if not memory.get("by_player") and not memory.get("by_agent"):
            if should_adopt:
                adopted += 1
            else:
                external += 1
        open_action = "FUT_SHORT_FULL" if side == "short" else "FUT_LONG_FULL"
        open_regime = str(memory.get("regime") or ("neutral" if should_adopt else ""))
        tracked = TrackedPosition(
            open_signal_id=int(memory.get("signal_id") or 0),
            sym=sym,
            side=side,
            entry_price=entry,
            qty=qty,
            fee_open=float(memory.get("fee_open") or 0.0),
            by_player=by_player,
            by_agent=by_agent,
            opened_at=datetime.now(timezone.utc),
            opened_bar=0,
            open_action=open_action,
            open_regime=open_regime,
            funding_open=float(memory.get("funding_open") or 0.0),
        )
        tracker.force_set(tracked)
        if should_adopt:
            seed_perf_open_position_for_adoption(pipeline, tracked)
        recovered += 1

    if recovered:
        log.info(
            "[%s] recovered %d open exchange position(s) into v2 tracker "
            "(preserved=%d external=%d adopted=%d)",
            pipeline.exchange_name,
            recovered,
            preserved,
            external,
            adopted,
        )
    return recovered


def _recover_pending_orders(pipeline) -> Dict[str, int]:
    """Poll restored accepted orders once after startup snapshot load."""
    ledger = getattr(pipeline, "order_ledger", None)
    executor = getattr(pipeline, "executor", None)
    pending_records = getattr(ledger, "pending_records", None)
    recover = getattr(executor, "recover_pending_order", None)
    summary = {"checked": 0, "filled": 0, "rejected": 0, "pending": 0, "skipped": 0}
    if not callable(pending_records) or not callable(recover):
        return summary

    for record in pending_records():
        order_id = str(getattr(record, "exchange_order_id", "") or "")
        signal = getattr(record, "signal", None)
        if not order_id or signal is None:
            summary["skipped"] += 1
            continue
        summary["checked"] += 1
        try:
            result = recover(signal, order_id)
        except Exception:
            log.warning(
                "[%s] pending order recovery failed: %s",
                getattr(pipeline, "exchange_name", "UNKNOWN"),
                order_id,
                exc_info=True,
            )
            summary["pending"] += 1
            continue
        if result.status == ExecutionStatus.FILLED:
            summary["filled"] += 1
        elif result.status == ExecutionStatus.REJECTED:
            summary["rejected"] += 1
        else:
            summary["pending"] += 1

    if summary["checked"] or summary["skipped"]:
        log.info(
            "[%s] pending order recovery: checked=%d filled=%d rejected=%d pending=%d skipped=%d",
            getattr(pipeline, "exchange_name", "UNKNOWN"),
            summary["checked"],
            summary["filled"],
            summary["rejected"],
            summary["pending"],
            summary["skipped"],
        )
    return summary


def _open_memory_by_symbol(pipeline) -> Dict[str, dict]:
    try:
        open_memory = pipeline.perf.snapshot().get("open") or {}
    except Exception:
        return {}
    try:
        agent_labels = set(pipeline.registry.all_labels())
    except Exception:
        agent_labels = set()
    player_labels = {getattr(profile, "label", "") for profile in getattr(pipeline, "profiles", [])}

    by_symbol: Dict[str, dict] = {}
    for key, payload in open_memory.items():
        if not isinstance(payload, dict):
            continue
        parts = str(key).split("|")
        if len(parts) == 3 and parts[0] not in ("", "real"):
            continue
        sym = str(parts[-1] if parts else "").upper()
        if not sym:
            continue
        label = str(payload.get("label") or (parts[-2] if len(parts) >= 2 else ""))
        if not label:
            continue
        rec = by_symbol.setdefault(sym, {})
        for field in ("signal_id", "side", "entry_price", "qty", "fee_open"):
            if field not in rec and field in payload:
                rec[field] = payload[field]
        if label in player_labels:
            rec["by_player"] = label
        elif label in agent_labels:
            rec["by_agent"] = label
        elif not rec.get("by_player"):
            rec["by_player"] = label
    return by_symbol


def _float_or_zero(*values) -> float:
    for value in values:
        try:
            if value is not None and value != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _snapshot_shadow_positions(pipeline) -> Dict[str, tuple]:
    tournament = getattr(pipeline, "shadow_tournament", None)
    getter = getattr(tournament, "last_player_open_positions", None)
    raw = None
    if callable(getter):
        try:
            raw = getter()
        except Exception:
            log.debug("failed to snapshot shadow positions", exc_info=True)
            raw = None
    if raw is None:
        raw = getattr(pipeline, "_pending_shadow_player_positions", {}) or {}
    out: Dict[str, tuple] = {}
    if not isinstance(raw, dict):
        return out
    for label, payloads in raw.items():
        rows = []
        for payload in tuple(payloads or ()):
            if isinstance(payload, dict):
                rows.append(dict(payload))
        out[str(label)] = tuple(rows)
    return out


def _save_runtime_snapshot(pipeline, snapshot_path: Optional[str]) -> None:
    if not snapshot_path:
        return
    tracker = getattr(getattr(pipeline, "executor", None), "_tracker", None)
    save_v2_snapshot(
        pipeline.perf,
        snapshot_path,
        real_perf=pipeline.real_perf,
        order_ledger=pipeline.order_ledger,
        position_tracker=tracker,
        shadow_positions=_snapshot_shadow_positions(pipeline),
    )


def _default_bridge_cache_path(snapshot_path: Optional[str], exchange: str) -> Optional[str]:
    if not snapshot_path:
        return None
    return str(Path(snapshot_path).with_name(f"{exchange.lower()}_bridge_cache.json"))


# ────────────────────────────────────────────────────────────────────
# Main entry: start_production
# ────────────────────────────────────────────────────────────────────


def start_production(
    *,
    exchange:           str,
    mode:               str = "live_futures",
    initial_capital:    Optional[float] = None,
    seed_quarantine:    Sequence[str] = (),
    profiles:           Optional[Sequence[PlayerProfile]] = None,
    polling_session_dir: Optional[str] = None,
    snapshot_path:      Optional[str] = None,
    migrate_from_v1:    Optional[Union[str, Sequence[str]]] = None,
    jsonl_event_log:    Optional[str] = None,
    max_bars:           Optional[int] = None,
    max_idle_polls:     Optional[int] = None,
    sleep_between_polls_sec: float = 5.0,
    use_v1_bridge:      bool = True,
    results_root:       str = "Results",
    warmup_bars:        Optional[int] = None,
    bridge_cache_path:  Optional[str] = None,
    bridge_cache_max_age_sec: float = 6 * 3600,
    allow_live_feed_fallback: Optional[bool] = None,
    include_genetics:   Optional[bool] = None,
    genetics_shadow_only: Optional[bool] = None,
    risk_config_override: Optional[RiskLimitsConfig] = None,
    strategist_config_override: Optional[StrategistConfig] = None,
    live_execution_config_override: Optional[LiveExecutionConfig] = None,
    flash_enabled_override: Optional[bool] = None,
    flash_allocator_config_override: Optional[FlashAllocatorConfig] = None,
    configure_pipeline: Optional[Callable[[object], None]] = None,
) -> int:
    """Запустить production main_loop для указанной биржи.

    Параметры:
      use_v1_bridge — если True, попытается подключиться к v1-bridge
                      для получения live-данных. Если False или bridge
                      недоступен — используется PollingFeed (если задан
                      polling_session_dir) или ReplayFeed (smoke).
      results_root  — корневая папка для OutputWriter (по умолчанию Results/)
    """
    log.info("=" * 70)
    log.info("Panteon v2 production startup: %s mode=%s", exchange, mode)
    log.info("=" * 70)
    os.environ["CRYPTO_EXCHANGE"] = exchange.upper()
    os.environ[f"{exchange.upper()}_TRADING_MODE"] = mode
    if allow_live_feed_fallback is None:
        allow_live_feed_fallback = _env_flag("PANTEON_ALLOW_LIVE_FEED_FALLBACK", False)
    include_genetics = _resolve_include_genetics(include_genetics, exchange)
    genetics_shadow_only = _resolve_genetics_shadow_only(genetics_shadow_only, exchange)
    jsonl_event_log = _run_scoped_event_log_path(exchange, jsonl_event_log)
    if jsonl_event_log:
        log.info("[%s] EventLog JSONL: %s", exchange, jsonl_event_log)

    # 1. Exchange adapter + capital base
    exchange_adapter = resolve_exchange(exchange, mode=mode)
    resolved_initial_capital = _resolve_initial_capital(
        exchange_adapter=exchange_adapter,
        mode=mode,
        requested_initial_capital=initial_capital,
        exchange_name=exchange,
    )
    trade_fraction = _resolve_trade_fraction(exchange)
    risk_config = _resolve_risk_config(exchange, trade_fraction)
    strategist_config = _resolve_strategist_config(exchange)
    live_execution_config = _resolve_live_execution_config(exchange)
    flash_enabled = _resolve_flash_enabled(exchange)
    flash_allocator_config = _resolve_flash_allocator_config(exchange)
    if risk_config_override is not None:
        risk_config = risk_config_override
        trade_fraction = float(risk_config.capital_fraction)
    if strategist_config_override is not None:
        strategist_config = strategist_config_override
    if live_execution_config_override is not None:
        live_execution_config = live_execution_config_override
    if flash_enabled_override is not None:
        flash_enabled = bool(flash_enabled_override)
    if flash_allocator_config_override is not None:
        flash_allocator_config = flash_allocator_config_override
    flash_stale_exit_config = _resolve_flash_stale_position_exit_config(exchange)
    flash_partial_profit_lock_config = _resolve_flash_partial_profit_lock_config(exchange)
    log.info(
        "[%s] v2 risk capital_fraction=%.2f%% max_open_positions=%d",
        exchange,
        risk_config.capital_fraction * 100.0,
        risk_config.max_open_positions,
    )
    log.info(
        "[%s] v2 live guardrails: daily_loss=%.2f%% equity_peak_dd=%.2f%% slippage=%.2f%% "
        "api_errors=%d stale_polls=%d pending_timeout=%.0fs",
        exchange,
        live_execution_config.max_daily_loss_pct,
        live_execution_config.max_equity_peak_drawdown_pct,
        live_execution_config.max_slippage_pct,
        live_execution_config.max_api_error_streak,
        live_execution_config.max_stale_feed_polls,
        live_execution_config.pending_order_timeout_sec,
    )
    if flash_enabled:
        log.info(
            "[%s] Panteon_Flash enabled: min_score=%.4f actionable_bonus=%.4f no_data_score=%.4f",
            exchange,
            flash_allocator_config.min_score_to_trade,
            flash_allocator_config.actionable_bonus,
            flash_allocator_config.no_data_score,
        )
        if flash_stale_exit_config["enabled"]:
            log.info(
                "[%s] Panteon_Flash stale-position exit enabled: max_age_bars=%d require_nonpositive_unrealized=%s",
                exchange,
                int(flash_stale_exit_config["max_age_bars"]),
                bool(flash_stale_exit_config["require_nonpositive_unrealized"]),
            )
        if flash_partial_profit_lock_config["enabled"]:
            log.info(
                "[%s] Panteon_Flash partial profit-lock enabled: trigger=%.2f%% close_fraction=%.2f min_age_bars=%d",
                exchange,
                float(flash_partial_profit_lock_config["trigger_pnl_pct"]),
                float(flash_partial_profit_lock_config["close_fraction"]),
                int(flash_partial_profit_lock_config["min_age_bars"]),
            )

    # 2. Регистрируем v1-агенты. Они получают актуальный v2 balance.
    registry = AgentRegistry()
    pipeline_holder = {}
    def _portfolio_value() -> float:
        pipeline = pipeline_holder.get("pipeline")
        if pipeline is not None:
            return float(getattr(pipeline, "current_balance", resolved_initial_capital) or resolved_initial_capital)
        return resolved_initial_capital

    futures_signal_fixes_enabled = _resolve_futures_signal_fixes_enabled(exchange)
    registered = register_all_v1_agents(
        registry,
        portfolio_value_fn=_portfolio_value,
        skip_on_error=True,
        include_optional=include_genetics,
        futures_replay_signal_fixes_enabled=futures_signal_fixes_enabled,
    )
    log.info("Registered %d v1-agents: %s", len(registered),
             ", ".join(registered[:5]) + ("…" if len(registered) > 5 else ""))
    if futures_signal_fixes_enabled:
        log.info("[%s] futures signal fixes enabled for v1 agents", exchange)
    if not registered:
        log.error(
            "No v1-agents registered. Check sys.path / panteon_runtime presence."
        )
        return 2
    legacy_real_enabled, legacy_real_labels = _resolve_flash_legacy_real_agents(exchange)
    if flash_enabled and legacy_real_enabled:
        promoted_legacy = promote_legacy_flash_real_agents(
            registry,
            labels=legacy_real_labels,
            skip_missing=True,
        )
        if promoted_legacy:
            log.info(
                "[%s] Panteon_Flash legacy real agents enabled: %s",
                exchange,
                ", ".join(promoted_legacy),
            )
        else:
            log.warning(
                "[%s] Panteon_Flash legacy real agents requested but none promoted",
                exchange,
            )
    genetics_limit_labels = _apply_genetics_position_limit(
        registry,
        max_open_positions=risk_config.max_open_positions,
    )
    if genetics_limit_labels:
        log.info(
            "[%s] genetics MAX_POS aligned to runtime max_open_positions=%d: %s",
            exchange,
            risk_config.max_open_positions,
            ", ".join(genetics_limit_labels),
        )
    effective_seed_quarantine = _seed_quarantine_with_shadow_only_genetics(
        seed_quarantine,
        registered,
        include_genetics=include_genetics,
        genetics_shadow_only=genetics_shadow_only,
        quarantine_exempt_labels=_resolve_genetics_shadow_only_quarantine_exempt_labels(
            exchange,
        ),
    )
    if include_genetics:
        genetics_registered = [
            label for label in registered
            if label in set(optional_labels()) or label.startswith("Genetics")
        ]
        log.info(
            "[%s] genetics agents loaded: %s%s",
            exchange,
            ", ".join(genetics_registered) if genetics_registered else "none",
            " (shadow-only quarantine)" if genetics_shadow_only else "",
        )

    # 3. ProductionPipeline
    pipeline = build_production_pipeline(
        registry=registry,
        exchange=exchange_adapter,
        initial_capital=resolved_initial_capital,
        seed_quarantine=effective_seed_quarantine,
        profiles=profiles or PRODUCTION_PROFILES,
        strategist_config=strategist_config,
        risk_config=risk_config,
        live_execution_config=live_execution_config,
        flash_enabled=flash_enabled,
        flash_allocator_config=flash_allocator_config,
        flash_component_memory=_resolve_flash_component_memory(exchange),
        scoring_config=_resolve_scoring_config(exchange),
        perf_trade_fraction=trade_fraction,
        perf_max_returns_history=_resolve_perf_max_returns_history(exchange),
        voting_directional=_resolve_voting_directional(exchange),
        jsonl_event_log=jsonl_event_log,
    )
    if configure_pipeline is not None:
        configure_pipeline(pipeline)
    pipeline.mode = mode
    pipeline.timeframe = _resolve_timeframe(exchange)
    pipeline.flash_stale_position_exit_enabled = bool(
        flash_stale_exit_config["enabled"]
    )
    pipeline.flash_stale_position_exit_max_age_bars = int(
        flash_stale_exit_config["max_age_bars"]
    )
    pipeline.flash_stale_position_exit_require_nonpositive_unrealized = bool(
        flash_stale_exit_config["require_nonpositive_unrealized"]
    )
    pipeline.flash_partial_profit_lock_enabled = bool(
        flash_partial_profit_lock_config["enabled"]
    )
    pipeline.flash_partial_profit_lock_trigger_pnl_pct = float(
        flash_partial_profit_lock_config["trigger_pnl_pct"]
    )
    pipeline.flash_partial_profit_lock_close_fraction = float(
        flash_partial_profit_lock_config["close_fraction"]
    )
    pipeline.flash_partial_profit_lock_min_age_bars = int(
        flash_partial_profit_lock_config["min_age_bars"]
    )
    pipeline.flash_partial_profit_lock_skip_protected_signal_keys = bool(
        flash_partial_profit_lock_config["skip_protected_signal_keys"]
    )
    pipeline.flash_partial_profit_lock_skip_signal_keys = tuple(
        flash_partial_profit_lock_config["skip_signal_keys"]
    )
    pipeline_holder["pipeline"] = pipeline
    log.info("Pipeline built: %d profiles, capital=$%.2f",
             len(pipeline.profiles), pipeline.initial_capital)
    exchange_settings = _load_exchange_settings(exchange)
    quarantine_override_labels = _quarantine_override_labels_from_settings(
        exchange_settings,
        exchange,
    )
    manual_quarantine_labels = _manual_quarantine_labels_from_settings(
        exchange_settings,
        exchange,
    )
    quarantine_recovery_config = _quarantine_recovery_config_from_settings(
        exchange_settings,
        exchange,
    )
    pipeline.quarantine_override_labels = quarantine_override_labels
    pipeline.manual_quarantine_labels = manual_quarantine_labels
    pipeline.quarantine_recovery_enabled = bool(
        quarantine_recovery_config["enabled"]
    )
    pipeline.quarantine_recovery_min_pnl_pct = float(
        quarantine_recovery_config["min_pnl_pct"]
    )
    pipeline.quarantine_recovery_min_closed_trades = int(
        quarantine_recovery_config["min_closed_trades"]
    )
    for label in quarantine_override_labels:
        pipeline.qm.force_release_override(
            label,
            reason="startup_release_override",
            bar=0,
        )
    manual_allowlist = {str(label).strip() for label in quarantine_override_labels}
    forced_manual = []
    for label in manual_quarantine_labels:
        clean = str(label or "").strip()
        if not clean or clean in manual_allowlist:
            continue
        pipeline.qm.force_quarantine(
            clean,
            reason="manual_allowlist_required",
            bar=0,
        )
        forced_manual.append(clean)
    if quarantine_override_labels:
        log.warning(
            "[%s] quarantine release override active: %s",
            exchange,
            ", ".join(quarantine_override_labels),
        )
    if forced_manual:
        log.warning(
            "[%s] manual quarantine active: %s",
            exchange,
            ", ".join(forced_manual),
        )

    # 4. Загрузка persisted state
    restored_shadow_positions: Dict[str, tuple] = {}
    _load_or_migrate_state(
        qm=pipeline.qm,
        perf=pipeline.perf,
        real_perf=pipeline.real_perf,
        order_ledger=pipeline.order_ledger,
        position_tracker=getattr(pipeline.executor, "_tracker", None),
        shadow_positions_target=restored_shadow_positions,
        snapshot_path=snapshot_path,
        migrate_from_v1=migrate_from_v1,
        exchange_name=exchange,
    )
    pipeline.selector.capture_session_baseline()
    pipeline.strategist.capture_session_baseline()
    pipeline.degradation_gate.capture_baseline(
        pipeline.perf,
        labels=pipeline.registry.all_labels(),
    )
    pipeline.shadow_tournament = ProductionShadowTournament(
        registry=pipeline.registry,
        perf=pipeline.perf,
        risk_config=risk_config,
        event_log=pipeline.event_log,
    )
    if restored_shadow_positions:
        pipeline._pending_shadow_player_positions = dict(restored_shadow_positions)
    log.info(
        "[%s] production shadow tournament enabled: agents=%d profiles=%d",
        exchange,
        len(pipeline.registry),
        len(pipeline.profiles),
    )

    # 5. OutputWriter — обязательно для видимости работы
    if mode != "paper":
        _recover_exchange_positions(pipeline)
        _recover_pending_orders(pipeline)
        expire = getattr(pipeline.executor, "expire_stale_pending_orders", None)
        if callable(expire):
            timed_out = expire(max_age_sec=live_execution_config.pending_order_timeout_sec)
            if timed_out:
                log.warning("[%s] timed out %d restored pending order(s)", exchange, timed_out)

    writer = OutputWriter.for_session(
        pipeline,
        results_root=results_root,
        compact_causal_entry_include_labels=(
            _resolve_compact_causal_entry_include_labels(exchange)
        ),
    )
    log.info("Output directory: %s", writer.output_dir)

    # 6. Feed selection: v1-bridge > PollingFeed > SyntheticFeed (paper) > empty
    # paper-mode сразу идёт в SyntheticFeed, минуя v1-bridge.
    if use_v1_bridge and mode != "paper":
        try:
            from .v1_bridge_runner import run_with_v1_bridge
            log.info("[%s] starting via v1-bridge integration", exchange)
            rc = run_with_v1_bridge(
                pipeline,
                exchange_name=exchange,
                mode=mode,
                output_writer=writer,
                max_bars=max_bars,
                max_idle_polls=max_idle_polls,
                sleep_between_polls_sec=sleep_between_polls_sec,
                warmup_bars=warmup_bars,
                bridge_cache_path=(
                    bridge_cache_path
                    or _default_bridge_cache_path(snapshot_path, exchange)
                ),
                bridge_cache_max_age_sec=bridge_cache_max_age_sec,
                snapshot_callback=lambda: _save_runtime_snapshot(pipeline, snapshot_path),
                shutdown_callback=lambda steps: _apply_paper_canary_shutdown_flatten(
                    pipeline,
                    writer,
                    exchange=exchange,
                    mode=mode,
                    bar=(steps[-1].bar + 1) if steps else None,
                ),
            )
            # Сохраняем snapshot
            if snapshot_path:
                _save_runtime_snapshot(pipeline, snapshot_path)
                log.info("Saved snapshot → %s", snapshot_path)
            if rc == 0:
                return rc
            if _should_fail_closed_after_bridge_error(
                mode=mode,
                polling_session_dir=polling_session_dir,
                allow_live_feed_fallback=bool(allow_live_feed_fallback),
            ):
                log.error(
                    "[%s] v1-bridge returned rc=%s; failing closed",
                    exchange,
                    rc,
                )
                writer.close()
                return int(rc or 3)
            log.warning("v1-bridge returned rc=%s; falling back to local feed", rc)
        except Exception as exc:
            if _should_fail_closed_after_bridge_error(
                mode=mode,
                polling_session_dir=polling_session_dir,
                allow_live_feed_fallback=bool(allow_live_feed_fallback),
            ):
                log.error(
                    "v1-bridge runner failed (%s); failing closed",
                    type(exc).__name__,
                    exc_info=True,
                )
                writer.close()
                return 3
            log.warning("v1-bridge runner failed (%s) — falling back to empty feed",
                        type(exc).__name__)

    # Fallback path: PollingFeed или empty
    if polling_session_dir:
        feed: MarketFeed = PollingFeed(session_dir=polling_session_dir)
        log.info("Using PollingFeed: %s", polling_session_dir)
    elif str(mode or "").strip().lower() in SYNTHETIC_FEED_MODES:
        # paper-mode → smoke через SyntheticFeed (random-walk цены)
        feed = SyntheticFeed(
            symbols=["BTC", "ETH", "SOL", "BNB", "ADA", "DOGE"],
            max_bars=max_bars,
        )
        log.warning(
            "[%s] paper mode → SyntheticFeed (synthetic prices for "
            "infrastructure verification, NOT production)", exchange)
        # SyntheticFeed follows sleep_between_polls_sec from the caller.
    else:
        if _should_fail_closed_after_bridge_error(
            mode=mode,
            polling_session_dir=polling_session_dir,
            allow_live_feed_fallback=bool(allow_live_feed_fallback),
        ):
            log.error("No live feed configured; failing closed")
            writer.close()
            return 3
        feed = ReplayFeed()
        log.warning(
            "No feed configured and live-bridge unavailable — main_loop "
            "will idle. Pass polling_session_dir or set mode='paper' for "
            "synthetic feed.")

    n_step = [0]
    def _on_step(step):
        n_step[0] += 1
        try:
            writer.write(step)
        except Exception:
            log.exception("output_writer.write failed")
        if n_step[0] % 10 == 0:
            log.info("  bar=%d leader=%s signals=%d filled=%d",
                     step.bar, step.leader or "-",
                     step.n_signals, step.n_filled)

    def _on_error(exc):
        log.error("main_loop error: %s", exc, exc_info=True)

    def _on_idle(idle_polls: int):
        if idle_polls == 1 or idle_polls % 12 == 0:
            log.info(
                "waiting for market feed: idle_poll=%d output=%s",
                idle_polls,
                writer.output_dir,
            )
        try:
            writer.write_heartbeat(
                run_state="idle",
                feed_status="waiting_for_market_bar",
                message=f"idle_poll={idle_polls}",
            )
        except Exception:
            log.exception("output_writer heartbeat failed")

    try:
        steps = main_loop(
            pipeline, feed,
            max_bars=max_bars,
            sleep_between_polls_sec=sleep_between_polls_sec,
            on_step=_on_step,
            on_error=_on_error,
            on_idle=_on_idle,
            max_idle_polls=max_idle_polls,
        )
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — saving state and exit")
        steps = None

    _apply_paper_canary_shutdown_flatten(
        pipeline,
        writer,
        exchange=exchange,
        mode=mode,
        bar=(steps[-1].bar + 1) if steps else None,
    )

    writer.close()
    if snapshot_path:
        _save_runtime_snapshot(pipeline, snapshot_path)
        log.info("Saved snapshot → %s", snapshot_path)
    processed = len(steps) if steps is not None else n_step[0]
    log.info("Shutdown. Processed %d bars.", processed)
    return 0
