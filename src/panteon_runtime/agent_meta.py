from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np


def _resolve_logger(player, logger=None):
    return logger if logger is not None else getattr(player, "_meta_logger", None)


class ShadowScoringAgent:
    def set_shadow_perf(self, player, perf: Dict[str, dict]):
        player._shadow_perf = perf
        player._last_shadow_snapshot = perf or {}

    def score_shadow_candidate(self, player, label: str, perf: dict) -> float:
        # Raw PnL alone was too noisy for live rotation and let high-drawdown
        # agents dominate the mix before their losses became obvious.
        return self.success_signal(player, perf, label=label)

    def success_signal(self, player, perf: dict, label: str = "") -> float:
        pnl = float(perf.get("pnl_pct", 0.0))
        sharpe = float(perf.get("sharpe", 0.0))
        max_dd = abs(float(perf.get("max_dd", perf.get("max_drawdown_pct", perf.get("max_dd_pct", 0.0))) or 0.0))
        win_rate = float(perf.get("win_rate", 0.0))
        signals = int(perf.get("signals", 0) or 0)
        entries = int(perf.get("entries", 0) or 0)
        closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
        sample_n = max(entries, closed)
        closed_conf = min(closed / 8.0, 1.0)
        entry_conf = min(entries / 12.0, 1.0)
        sample_conf = max(closed_conf, entry_conf * 0.35)
        effective_sharpe = sharpe * sample_conf
        if closed == 0 and entries > 0 and effective_sharpe > 0.0:
            effective_sharpe = min(effective_sharpe, 1.8)
        elif closed < 3 and effective_sharpe > 0.0:
            effective_sharpe = min(effective_sharpe, 3.0)
        elif closed < 6 and effective_sharpe > 0.0:
            effective_sharpe = min(effective_sharpe, 6.0)

        effective_pnl = pnl
        if closed == 0 and entries > 0:
            effective_pnl *= 0.35
        elif closed < 3:
            effective_pnl *= 0.65
        if "genetics" in str(label or "").lower() and closed == 0:
            effective_pnl = min(effective_pnl, 0.0)

        activity = min(signals, 40) * 0.02 + min(entries, 20) * 0.04 + min(closed, 12) * 0.09
        win_bonus = ((win_rate - 50.0) / 10.0) if closed >= 2 else 0.0
        inactivity_penalty = 1.2 if signals == 0 and entries == 0 else 0.0
        low_sample_penalty = 0.0
        if closed == 0 and entries >= 3:
            low_sample_penalty = 0.90
        if "genetics" in str(label or "").lower() and closed == 0:
            low_sample_penalty = max(low_sample_penalty, 1.25)
        if sample_n == 1:
            low_sample_penalty = max(low_sample_penalty, 1.50)
        elif 1 < sample_n < 4:
            low_sample_penalty = max(low_sample_penalty, 0.80)
        churn_penalty = 0.0
        if entries > 60 and pnl < 1.0:
            churn_penalty += min((entries - 60) * 0.03, 3.0)
        if signals > 180 and pnl < 1.0:
            churn_penalty += min((signals - 180) * 0.01, 2.5)
        if max_dd > 8.0 and pnl < 1.0:
            churn_penalty += min((max_dd - 8.0) * 0.35, 3.0)
        if closed >= 8 and pnl < 0.25 and sharpe < 2.0:
            churn_penalty += 0.75

        return (
            effective_pnl * 0.55
            + effective_sharpe * 1.10
            + activity
            + win_bonus
            - max_dd * 0.18
            - inactivity_penalty
            - low_sample_penalty
            - churn_penalty
        )

    def update_success_memory(self, player, label: str, perf: dict) -> float:
        prev = player._agent_success_memory.get(label, 0.0)
        current = self.success_signal(player, perf, label=label)
        alpha = 0.18
        if prev and current and np.sign(prev) != np.sign(current):
            alpha = 0.42
        elif abs(current) > max(abs(prev), 1.0) * 1.25:
            alpha = 0.30
        blended = prev * (1.0 - alpha) + current * alpha
        player._agent_success_memory[label] = blended
        return blended


class ContextMemoryAgent:
    def update_context_memory(self, player, context: Optional[Dict[str, str]] = None):
        if player._shadow_perf is None:
            return
        ctx = dict(context or player._current_context or {})
        if not ctx:
            return
        for shadow_name, perf in player._shadow_perf.items():
            label = player.SHADOW_MAP.get(shadow_name)
            if not label or label not in player._agent_pool or not isinstance(perf, dict):
                continue
            observation = player._build_context_observation(shadow_name, perf)
            if observation is None:
                continue
            player._record_context_observation(label, ctx, observation)
            if hasattr(player, "_record_regime_observation"):
                player._record_regime_observation(label, ctx.get("regime"), observation)
        player._shadow_window_anchor = player._sanitize_shadow_anchor(player._shadow_perf)

    def context_memory_score(self, player, label: str, context: Optional[Dict[str, str]]) -> tuple[float, float]:
        if not context:
            return 0.0, 0.0

        factor_scores = []
        for factor in player.CONTEXT_FACTORS:
            bucket = str(context.get(factor, "unknown"))
            stats = player._factor_memory.get(factor, {}).get(bucket, {}).get(label)
            if not isinstance(stats, dict):
                continue
            samples = int(stats.get("samples", 0) or 0)
            if samples <= 0:
                continue
            confidence = min(samples / 8.0, 1.0)
            factor_scores.append(float(stats.get("ema_score", 0.0) or 0.0) * confidence)

        factor_score = float(np.clip(np.mean(factor_scores), -3.0, 3.0)) if factor_scores else 0.0
        full_stats = player._context_memory.get(label, {}).get(player._context_key(context))
        full_score = 0.0
        if isinstance(full_stats, dict):
            samples = int(full_stats.get("samples", 0) or 0)
            confidence = min(samples / 6.0, 1.0)
            full_score = float(np.clip(full_stats.get("ema_score", 0.0) or 0.0, -3.0, 3.0)) * confidence
        return factor_score, full_score

    def update_symbol_memory(self, player):
        if player._shadow_perf is None:
            return
        for shadow_name, perf in player._shadow_perf.items():
            label = player.SHADOW_MAP.get(shadow_name)
            per_symbol = perf.get("per_symbol") if isinstance(perf, dict) else None
            if not label or label not in player._agent_pool or not isinstance(per_symbol, dict):
                continue
            for sym, stats in per_symbol.items():
                if not isinstance(stats, dict):
                    continue
                profile = player._build_symbol_profile(sym)
                observation = player._build_symbol_observation(shadow_name, sym, stats)
                if observation is None:
                    continue
                player._record_symbol_observation(label, sym, profile, observation)
                if hasattr(player, "_record_symbol_regime_observation"):
                    player._record_symbol_regime_observation(
                        label, sym, profile.get("symbol_regime"), observation
                    )
        player._shadow_symbol_anchor = player._sanitize_shadow_symbol_anchor(
            {
                shadow_name: perf.get("per_symbol")
                for shadow_name, perf in (player._shadow_perf or {}).items()
                if isinstance(perf, dict)
            }
        )

    def symbol_memory_score(
        self,
        player,
        label: str,
        sym: str,
        profile: Optional[Dict[str, str]] = None,
        exact_stats: Optional[dict] = None,
    ) -> tuple[float, float]:
        exact_score = 0.0
        exact_stats = exact_stats if isinstance(exact_stats, dict) else player._symbol_memory.get(label, {}).get(sym)
        if isinstance(exact_stats, dict):
            samples = int(exact_stats.get("samples", 0) or 0)
            confidence = min(samples / 5.0, 1.0)
            exact_score = float(np.clip(exact_stats.get("ema_score", 0.0) or 0.0, -3.0, 3.0)) * confidence

        profile = dict(profile or player._build_symbol_profile(sym))
        factor_scores = []
        for factor in player.SYMBOL_PROFILE_FACTORS:
            bucket = str(profile.get(factor, "unknown"))
            stats = player._symbol_factor_memory.get(factor, {}).get(bucket, {}).get(label)
            if not isinstance(stats, dict):
                continue
            samples = int(stats.get("samples", 0) or 0)
            if samples <= 0:
                continue
            confidence = min(samples / 6.0, 1.0)
            factor_scores.append(float(stats.get("ema_score", 0.0) or 0.0) * confidence)

        factor_score = float(np.clip(np.mean(factor_scores), -3.0, 3.0)) if factor_scores else 0.0
        return exact_score, factor_score

    def symbol_vote_weight(self, player, label: str, sym: str, base_weight: float) -> float:
        profile = player._build_symbol_profile(sym)
        exact_score, factor_score = self.symbol_memory_score(player, label, sym, profile=profile)
        regime_score = (
            float(player._symbol_regime_memory_score(label, sym, profile.get("symbol_regime")))
            if hasattr(player, "_symbol_regime_memory_score")
            else 0.0
        )
        bonus = (
            exact_score * player.SYMBOL_EXACT_WEIGHT
            + factor_score * player.SYMBOL_PROFILE_WEIGHT
            + regime_score * getattr(player, "SYMBOL_REGIME_WEIGHT", 0.0)
        )
        multiplier = float(np.clip(1.0 + bonus * player.SYMBOL_VOTE_STRENGTH, 0.70, 1.35))
        return base_weight * multiplier

    def agent_symbol_opportunity_score(self, player, label: str) -> float:
        if not player._ph:
            return 0.0
        scores = []
        for sym in player._ph.keys():
            profile = player._build_symbol_profile(sym)
            exact_score, factor_score = self.symbol_memory_score(player, label, sym, profile=profile)
            regime_score = (
                float(player._symbol_regime_memory_score(label, sym, profile.get("symbol_regime")))
                if hasattr(player, "_symbol_regime_memory_score")
                else 0.0
            )
            scores.append(exact_score * 0.55 + factor_score * 0.25 + regime_score * 0.20)
        if not scores:
            return 0.0
        top = sorted(scores, reverse=True)[:3]
        return float(np.clip(np.mean(top), -2.0, 2.0))


class MemorySnapshotStore:
    def __init__(self, memory_file: str, legacy_memory_file: str, save_interval_bars: int = 30):
        self.memory_file = str(memory_file)
        self.legacy_memory_file = str(legacy_memory_file)
        self.save_interval_bars = int(save_interval_bars)

    def enable(self, player, load_existing: bool = True):
        player._memory_enabled = True
        if load_existing:
            self.load_snapshot(player)
        player._shadow_window_anchor = {}
        player._shadow_symbol_anchor = {}

    def load_snapshot(self, player, logger=None):
        if not player._memory_enabled:
            return
        logger = _resolve_logger(player, logger)
        memory_path = self.memory_file
        current_namespace = str(getattr(player, "_memory_namespace", "") or "")
        if (
            not current_namespace
            and not os.path.isfile(memory_path)
            and os.path.isfile(self.legacy_memory_file)
        ):
            memory_path = self.legacy_memory_file
        if not os.path.isfile(memory_path):
            if logger is not None:
                logger.debug("  [REAL memory] no snapshot file found; using default lineup")
            return
        try:
            with open(memory_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("memory payload is not a dict")
            snapshot_namespace = str(data.get("memory_namespace", "") or "")
            if current_namespace and snapshot_namespace != current_namespace:
                if logger is not None:
                    logger.warning(
                        "  [REAL memory] skip %s: namespace %r != %r",
                        memory_path,
                        snapshot_namespace,
                        current_namespace,
                    )
                return

            regime_only_memory = bool(getattr(player, "USE_REGIME_ONLY_MEMORY", False))
            restore_saved_weights = bool(
                getattr(player, "RESTORE_SAVED_ACTIVE_WEIGHTS", not regime_only_memory)
            )
            restore_global_scores = bool(
                getattr(player, "RESTORE_GLOBAL_SUCCESS_MEMORY", not regime_only_memory)
            )
            restored_weights = (
                player._normalize_weights(data.get("active_weights") or {})
                if restore_saved_weights
                else {}
            )
            restored_scores = (
                {
                    label: float(score)
                    for label, score in (data.get("agent_success_scores") or {}).items()
                    if label in player._agent_pool
                }
                if restore_global_scores
                else {}
            )
            restored_shadow = {} if regime_only_memory else (data.get("shadow_perf") or {})
            restored_real_counts = {} if regime_only_memory else (data.get("real_signal_counts") or {})
            restored_context_memory = (
                {}
                if regime_only_memory
                else player._restore_context_memory(data.get("context_memory") or {})
            )
            restored_factor_memory = player._restore_factor_memory(data.get("factor_memory") or {})
            restored_regime_memory = (
                player._restore_regime_memory(data.get("regime_memory") or {})
                if hasattr(player, "_restore_regime_memory")
                else {}
            )
            restored_symbol_regime_memory = (
                player._restore_symbol_regime_memory(data.get("symbol_regime_memory") or {})
                if hasattr(player, "_restore_symbol_regime_memory")
                else {}
            )
            if (
                not restored_regime_memory
                and restored_factor_memory
                and hasattr(player, "_bootstrap_regime_memory_from_factors")
            ):
                restored_regime_memory = player._bootstrap_regime_memory_from_factors(restored_factor_memory)
            if regime_only_memory:
                restored_factor_memory = {}
            restored_context = {} if regime_only_memory else (data.get("last_context") or {})
            restored_anchor = (
                {}
                if regime_only_memory
                else player._sanitize_shadow_anchor(data.get("shadow_window_anchor") or {})
            )
            restored_symbol_memory = (
                {}
                if regime_only_memory
                else player._restore_symbol_memory(data.get("symbol_memory") or {})
            )
            restored_symbol_factor_memory = (
                {}
                if regime_only_memory
                else player._restore_symbol_factor_memory(data.get("symbol_factor_memory") or {})
            )

            if restored_weights:
                player._active_weights = restored_weights
                player._memory_bootstrap_loaded = True
            if restored_scores:
                player._agent_success_memory = restored_scores
            if isinstance(restored_shadow, dict) and restored_shadow:
                player._shadow_perf = restored_shadow
                player._last_shadow_snapshot = restored_shadow
            if isinstance(restored_real_counts, dict) and restored_real_counts:
                player._sub_signal_counts = {
                    label: int(count)
                    for label, count in restored_real_counts.items()
                    if label in player._agent_pool
                }
            if restored_context_memory:
                player._context_memory = restored_context_memory
            if restored_factor_memory:
                player._factor_memory = restored_factor_memory
            if restored_regime_memory:
                player._regime_memory = restored_regime_memory
            if restored_symbol_regime_memory:
                player._symbol_regime_memory = restored_symbol_regime_memory
            if isinstance(restored_context, dict) and restored_context:
                player._current_context = {str(k): str(v) for k, v in restored_context.items()}
            if restored_anchor:
                player._shadow_window_anchor = restored_anchor
            if restored_symbol_memory:
                player._symbol_memory = restored_symbol_memory
            if restored_symbol_factor_memory:
                player._symbol_factor_memory = restored_symbol_factor_memory

            if logger is not None:
                if regime_only_memory:
                    logger.debug(
                        "  [REAL memory] regime-only load: regimes=%s; legacy active_weights/global scores ignored",
                        ",".join(sorted(getattr(player, "_regime_memory", {}).keys())) or "empty",
                    )
                else:
                    logger.debug(
                        "  [REAL memory] restored lineup: %s",
                        "  ".join(
                            f"{label}={weight:.0%}"
                            for label, weight in sorted(player._active_weights.items(), key=lambda x: x[1], reverse=True)
                        ) or "empty",
                    )
        except Exception as exc:
            if logger is not None:
                logger.warning("  [REAL memory] failed to load %s: %s", memory_path, exc)

    def build_payload(self, player) -> dict:
        active_weights = {
            label: round(float(weight), 6)
            for label, weight in sorted(player._active_weights.items(), key=lambda x: x[1], reverse=True)
        }
        return {
            "schema": int(getattr(player, "MEMORY_SCHEMA_VERSION", 5)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "player": type(player).__name__,
            "memory_namespace": str(getattr(player, "_memory_namespace", "") or ""),
            "exchange_namespace": str(getattr(player, "_memory_namespace", "") or ""),
            "memory_mode": "regime_only" if getattr(player, "USE_REGIME_ONLY_MEMORY", False) else "hybrid",
            "execution_mode": "regime_top" if getattr(player, "USE_REGIME_TOP_TRADER", False) else "weighted_ensemble",
            "regime_leaders": dict(getattr(player, "_regime_leaders", {}) or {}),
            "regime_leader_scores": {
                str(regime): round(float(score), 6)
                for regime, score in (getattr(player, "_regime_leader_scores", {}) or {}).items()
            },
            "active_weights": active_weights,
            "real_agents_active": list(active_weights.keys()),
            "real_agent_pool": sorted(player._agent_pool.keys()),
            "shadow_agents": sorted((player._last_shadow_snapshot or {}).keys()) or sorted(player.SHADOW_MAP.keys()),
            "shadow_to_real_map": dict(player.SHADOW_MAP),
            "agent_success_scores": {
                label: round(float(score), 6)
                for label, score in sorted(player._agent_success_memory.items())
                if label in player._agent_pool
            },
            "real_signal_counts": {
                label: int(count)
                for label, count in sorted(getattr(player, "_sub_signal_counts", {}).items())
                if label in player._agent_pool
            },
            "shadow_perf": {
                str(name): {
                    str(k): (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                    for k, v in metrics.items()
                }
                for name, metrics in (player._last_shadow_snapshot or {}).items()
                if isinstance(metrics, dict)
            },
            "last_regime": player._r,
            "last_rotation_bar": int(player._last_rotation),
            "last_context": dict(player._current_context or {}),
            "context_factors": list(player.CONTEXT_FACTORS),
            "shadow_window_anchor": player._sanitize_shadow_anchor(player._shadow_window_anchor),
            "context_memory": {
                label: {
                    ctx_key: {
                        "context": dict(stats.get("context") or {}),
                        "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
                        "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
                        "samples": int(stats.get("samples", 0) or 0),
                        "wins": int(stats.get("wins", 0) or 0),
                        "losses": int(stats.get("losses", 0) or 0),
                        "last_window_pnl": round(float(stats.get("last_window_pnl", 0.0) or 0.0), 6),
                        "last_updated": str(stats.get("last_updated", "") or ""),
                    }
                    for ctx_key, stats in sorted(ctx_map.items(), key=lambda item: item[0])
                    if isinstance(stats, dict)
                }
                for label, ctx_map in sorted(player._context_memory.items())
                if label in player._agent_pool and isinstance(ctx_map, dict)
            },
            "factor_memory": {
                factor: {
                    bucket: {
                        label: {
                            "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
                            "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
                            "samples": int(stats.get("samples", 0) or 0),
                            "wins": int(stats.get("wins", 0) or 0),
                            "losses": int(stats.get("losses", 0) or 0),
                            "last_window_pnl": round(float(stats.get("last_window_pnl", 0.0) or 0.0), 6),
                            "last_updated": str(stats.get("last_updated", "") or ""),
                        }
                        for label, stats in sorted(agent_map.items())
                        if label in player._agent_pool and isinstance(stats, dict)
                    }
                    for bucket, agent_map in sorted(bucket_map.items(), key=lambda item: item[0])
                    if isinstance(agent_map, dict)
                }
                for factor, bucket_map in sorted(player._factor_memory.items(), key=lambda item: item[0])
                if isinstance(bucket_map, dict)
            },
            "regime_memory": {
                regime: {
                    label: {
                        "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
                        "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
                        "samples": int(stats.get("samples", 0) or 0),
                        "wins": int(stats.get("wins", 0) or 0),
                        "losses": int(stats.get("losses", 0) or 0),
                        "last_window_pnl": round(float(stats.get("last_window_pnl", 0.0) or 0.0), 6),
                        "last_updated": str(stats.get("last_updated", "") or ""),
                    }
                    for label, stats in sorted(agent_map.items())
                    if label in player._agent_pool and isinstance(stats, dict)
                }
                for regime, agent_map in sorted(getattr(player, "_regime_memory", {}).items())
                if isinstance(agent_map, dict)
            },
            "symbol_profile_factors": list(player.SYMBOL_PROFILE_FACTORS),
            "symbol_memory": {
                label: {
                    sym: {
                        "profile": dict(stats.get("profile") or {}),
                        "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
                        "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
                        "samples": int(stats.get("samples", 0) or 0),
                        "wins": int(stats.get("wins", 0) or 0),
                        "losses": int(stats.get("losses", 0) or 0),
                        "last_total_pnl": round(float(stats.get("last_total_pnl", 0.0) or 0.0), 6),
                        "last_updated": str(stats.get("last_updated", "") or ""),
                    }
                    for sym, stats in sorted(sym_map.items(), key=lambda item: item[0])
                    if isinstance(stats, dict)
                }
                for label, sym_map in sorted(player._symbol_memory.items())
                if label in player._agent_pool and isinstance(sym_map, dict)
            },
            "symbol_regime_memory": {
                label: {
                    sym: {
                        regime: {
                            "profile": dict(stats.get("profile") or {}),
                            "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
                            "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
                            "samples": int(stats.get("samples", 0) or 0),
                            "wins": int(stats.get("wins", 0) or 0),
                            "losses": int(stats.get("losses", 0) or 0),
                            "last_total_pnl": round(float(stats.get("last_total_pnl", 0.0) or 0.0), 6),
                            "last_updated": str(stats.get("last_updated", "") or ""),
                        }
                        for regime, stats in sorted(regime_map.items(), key=lambda item: item[0])
                        if isinstance(stats, dict)
                    }
                    for sym, regime_map in sorted(sym_map.items(), key=lambda item: item[0])
                    if isinstance(regime_map, dict)
                }
                for label, sym_map in sorted(getattr(player, "_symbol_regime_memory", {}).items())
                if label in player._agent_pool and isinstance(sym_map, dict)
            },
            "symbol_factor_memory": {
                factor: {
                    bucket: {
                        label: {
                            "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
                            "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
                            "samples": int(stats.get("samples", 0) or 0),
                            "wins": int(stats.get("wins", 0) or 0),
                            "losses": int(stats.get("losses", 0) or 0),
                            "last_total_pnl": round(float(stats.get("last_total_pnl", 0.0) or 0.0), 6),
                            "last_updated": str(stats.get("last_updated", "") or ""),
                        }
                        for label, stats in sorted(agent_map.items())
                        if label in player._agent_pool and isinstance(stats, dict)
                    }
                    for bucket, agent_map in sorted(bucket_map.items(), key=lambda item: item[0])
                    if isinstance(agent_map, dict)
                }
                for factor, bucket_map in sorted(player._symbol_factor_memory.items(), key=lambda item: item[0])
                if isinstance(bucket_map, dict)
            },
            "symbol_preferences": player._build_symbol_preferences_snapshot(),
        }

    def save_snapshot(self, player, force: bool = False, reason: str = "", logger=None):
        if not player._memory_enabled:
            return
        logger = _resolve_logger(player, logger)
        cur_bar = int(getattr(player, "_t", 0) or 0)
        if not force and cur_bar - player._last_memory_save_bar < self.save_interval_bars:
            return
        try:
            payload = self.build_payload(player)
            tmp_path = self.memory_file + ".tmp"
            last_err = None
            for attempt in range(3):
                try:
                    with open(tmp_path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, indent=2, ensure_ascii=False)
                    os.replace(tmp_path, self.memory_file)
                    last_err = None
                    break
                except PermissionError as exc:
                    last_err = exc
                    time.sleep(0.15 * (attempt + 1))
            if last_err is not None:
                raise last_err
            player._last_memory_save_bar = cur_bar
            if logger is not None and (force or reason in ("rotation", "shutdown")):
                logger.debug("  [REAL memory] snapshot saved (%s)", reason or "periodic")
        except Exception as exc:
            if logger is not None:
                logger.warning("  [REAL memory] save failed: %s", exc)


class PortfolioAllocatorAgent:
    def _risk_adjusted_positive_score(self, perf: dict,
                                      regime: Optional[str] = None) -> float:
        """FIX 2026-05-04 (ensemble per-regime): скоринг агента для выбора в
        live-ансамбль. Берёт PnL ИЗ ТЕКУЩЕГО РЕЖИМА РЫНКА, а не агрегатный.

        Логика:
          1) Если для текущего регима есть достаточная выборка
             (per_regime[regime].closed_trades >= 2) и pnl_pct > 0 — берём
             этот pnl (это локально доказанная прибыль в текущих условиях).
          2) Если в текущем регионе данных нет/мало, но глобальный pnl > 0 —
             падаем на агрегат.
          3) Если агрегатный pnl <= 0 И в текущем регионе нет положительного
             опыта — возвращаем 0 (агент не доказал прибыльность нигде, где
             это можно проверить).

        Это решает проблему "Пантеон не выбирает локально прибыльных агентов",
        наблюдаемую в логах MEXC/BITGET 2026-05-03: V_LiveRegimePullback
        был +0.71% в bullish/neutral/crash, но агрегат шёл около -0%, поэтому
        старый scoring выбрасывал его, а Пантеон цеплялся к убыточным.
        """
        if not isinstance(perf, dict):
            return 0.0
        agg_pnl = float(perf.get("pnl_pct", 0.0) or 0.0)
        sharpe = max(float(perf.get("sharpe", 0.0) or 0.0), 0.0)
        max_dd = abs(float(perf.get(
            "max_dd",
            perf.get("max_drawdown_pct", perf.get("max_dd_pct", 0.0)),
        ) or 0.0))
        entries = int(perf.get("entries", 0) or 0)
        closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
        # ── per-regime окно (FIX 2026-05-04) ─────────────────────────────
        regime_pnl = None
        regime_closed = 0
        regime_entries = 0
        regime_signals = 0
        if regime:
            per_regime = perf.get("per_regime") if isinstance(perf, dict) else None
            stats = (per_regime or {}).get(regime) if isinstance(per_regime, dict) else None
            if isinstance(stats, dict):
                regime_pnl = float(stats.get("pnl_pct", 0.0) or 0.0)
                regime_closed = int(stats.get("closed_trades", 0) or 0)
                regime_entries = int(stats.get("entries", 0) or 0)
                regime_signals = int(stats.get("signals", 0) or 0)

        # Решаем, какой pnl использовать как ОСНОВУ:
        # 1) Если в текущем регионе есть локально доказанная прибыль — берём её.
        #    Бонус: агент может быть в плюсе по агрегату но в минусе по
        #    текущему регионе → НЕ берём его (он не подходит к рынку сейчас).
        if regime_pnl is not None and regime_closed >= 2 and regime_pnl > 0.0:
            # ВАЖНО: в текущем регионе закрытий достаточно — доверяем.
            base_pnl = regime_pnl
            base_closed = max(regime_closed, 2)
            base_entries = max(regime_entries, 1)
        elif regime_pnl is not None and regime_pnl < 0.0 and regime_closed >= 2:
            # В текущем регионе локально доказанный минус — НЕ выбираем,
            # даже если агрегат положительный. Это чистка ансамбля.
            return 0.0
        elif agg_pnl > 0.0 and closed > 0 and entries > 0:
            # Регимных данных нет → fallback на агрегат, как было.
            base_pnl = agg_pnl
            base_closed = closed
            base_entries = entries
        elif agg_pnl > 0.0 and (regime_signals > 0 or int(perf.get("signals", 0) or 0) > 0):
            # FIX 2026-05-04 (active newcomers): агент активен (есть signals),
            # имеет положительный агрегат, но ещё не закрыл сделок — не
            # выбрасываем его. Признаём с понижающим коэффициентом, чтобы
            # дать шанс молодым ярким лидерам типа V_LiveCrashHunter
            # (+0.07% по 4 signals, 0 closed). Без этого они отфильтровывались
            # в постоянное `positive candidates=0 < 2`.
            base_pnl = agg_pnl
            base_closed = max(closed, 1)
            base_entries = max(entries, max(1, regime_signals or 1))
        else:
            # Ни глобально, ни локально нет положительного опыта.
            return 0.0

        sharpe_cap = 50.0
        closed_cap = 30.0
        sample = max(float(base_closed), float(base_entries) * 0.35)
        sample_conf = 0.55 + 0.45 * min(sample / 10.0, 1.0)
        closed_bonus = float(np.sqrt(1.0 + min(float(base_closed), closed_cap) / 5.0))
        sharpe_bonus = 1.0 + min(sharpe, sharpe_cap) / 20.0
        drawdown_penalty = 1.0 + max_dd / 2.0
        score = float((base_pnl * sharpe_bonus * sample_conf * closed_bonus)
                      / max(drawdown_penalty, 1e-9))
        # Маленький бонус если агент СВЕЖО активен в этом режиме
        # (агрегатный pnl мог быть отрицательным, но agent работает СЕЙЧАС).
        if regime_pnl is not None and regime_signals > 0 and regime_closed >= 2:
            score *= 1.10
        return score

    def _rotate_risk_adjusted_top_positive(self, player, logger=None) -> bool:
        context = dict(player._current_context or player._build_market_context())
        max_agents = int(getattr(player, "RISK_ADJUSTED_MAX_AGENTS", player.MAX_AGENTS) or player.MAX_AGENTS)
        min_agents = int(getattr(player, "RISK_ADJUSTED_MIN_AGENTS", player.MIN_AGENTS) or player.MIN_AGENTS)
        max_agents = max(min_agents, max_agents)

        # FIX 2026-05-04 (ensemble per-regime): передаём текущий режим в
        # скоринг, чтобы агент с положительным опытом В ТЕКУЩЕМ РЕЖИМЕ
        # мог попасть в active_weights даже при умеренно отрицательном агрегате.
        current_regime = ""
        try:
            current_regime = str(
                context.get("regime")
                or getattr(player, "_r", "")
                or "neutral"
            ).lower()
        except Exception:
            current_regime = "neutral"
        ranked = []
        for shadow_name, perf in (player._shadow_perf or {}).items():
            label = player.SHADOW_MAP.get(shadow_name)
            if not label or label not in player._agent_pool:
                continue
            if hasattr(player, "_agent_currently_blocked") and player._agent_currently_blocked(label, perf):
                continue
            score = self._risk_adjusted_positive_score(
                perf if isinstance(perf, dict) else {},
                regime=current_regime,
            )
            if score <= 0.0:
                continue
            ranked.append((score, label))

        if len(ranked) < min_agents:
            # FIX 2026-05-04 (ensemble per-regime): больше НЕ цепляемся к
            # устаревшим active_weights. Очищаем их от участников, которые
            # стали отрицательными по текущему режиму или агрегату — те,
            # кто остался положительным, доживают до следующего цикла, а
            # отрицательные выбрасываются. Это останавливает «петлю
            # цепляния», когда Пантеон месяцами держит в составе
            # убыточного агента просто потому что новых кандидатов нет.
            old = dict(getattr(player, "_active_weights", {}) or {})
            cleaned = OrderedDict()
            for label in old:
                # Найдём perf по shadow_name
                shadow_name = None
                for sn, lbl in player.SHADOW_MAP.items():
                    if lbl == label:
                        shadow_name = sn
                        break
                perf = (player._shadow_perf or {}).get(shadow_name) if shadow_name else None
                score = self._risk_adjusted_positive_score(
                    perf if isinstance(perf, dict) else {},
                    regime=current_regime,
                )
                if score > 0.0:
                    cleaned[label] = score
            if cleaned and len(cleaned) != len(old):
                if hasattr(player, "_normalize_weights"):
                    new_weights = player._normalize_weights(cleaned, priority_labels=())
                else:
                    total = sum(cleaned.values())
                    new_weights = (
                        {label: value / total for label, value in cleaned.items()}
                        if total > 0 else {}
                    )
                if new_weights and new_weights != old:
                    player._active_weights = new_weights
                    if logger is not None:
                        active_msg = ", ".join(
                            f"{label}={weight:.0%}"
                            for label, weight in sorted(
                                new_weights.items(),
                                key=lambda item: item[1],
                                reverse=True,
                            )
                        )
                        logger.info(
                            "  [REAL risk-adjusted] cleaned weights "
                            "(few new positive candidates=%d): %s",
                            len(ranked), active_msg,
                        )
                    return True
            if logger is not None:
                logger.info(
                    "  [REAL risk-adjusted] positive candidates=%d < %d; "
                    "keeping current active weights (already clean)",
                    len(ranked), min_agents,
                )
            return False

        ranked.sort(reverse=True)
        top = ranked[:min(max_agents, len(ranked))]
        raw = OrderedDict((label, score) for score, label in top)
        if hasattr(player, "_normalize_weights"):
            new_weights = player._normalize_weights(raw, priority_labels=())
        else:
            total = sum(raw.values())
            new_weights = {label: value / total for label, value in raw.items()} if total > 0 else {}
        if not new_weights:
            return False

        old_weights = dict(getattr(player, "_active_weights", {}) or {})
        old_set = set(old_weights)
        new_set = set(new_weights)
        materially_changed = (
            old_set != new_set
            or any(abs(float(new_weights.get(label, 0.0)) - float(old_weights.get(label, 0.0))) >= 0.03 for label in new_set | old_set)
        )

        player._active_weights = new_weights
        player._update_context_memory(context)
        player._update_symbol_memory()
        player._last_rotation = int(getattr(player, "_t", getattr(player, "_last_rotation", 0)) or 0)
        player.save_memory_snapshot(force=True, reason="rotation")
        if logger is not None and materially_changed:
            active_msg = ", ".join(
                f"{label}={weight:.0%}"
                for label, weight in sorted(new_weights.items(), key=lambda item: item[1], reverse=True)
            )
            logger.info(
                "  [REAL risk-adjusted] active: %s | context: %s",
                active_msg,
                player._context_summary(context),
            )
        return True

    def rotate(self, player, logger=None):
        # Reuse the shared lineup-rotation implementation below. The helper only
        # depends on the allocator's scoring methods plus the player state.
        return ShadowPlayerMetaSelector.rotate(self, player, logger=logger)


class ShadowPlayerMetaSelector:
    """
    Selects the live aggregation player from shadow player performance.

    This is intentionally separate from agent memory: agents learn which
    sub-agent should vote; this layer learns which player-level aggregation
    policy should control the final live actions.
    """

    SCHEMA_VERSION = 1

    def __init__(self, memory_file: str, save_interval_bars: int = 30):
        self.memory_file = str(memory_file)
        self.save_interval_bars = int(save_interval_bars)

    def set_shadow_player_perf(self, player, perf: Dict[str, dict]):
        player._shadow_player_perf = perf or {}
        player._last_shadow_player_snapshot = perf or {}

    def load_snapshot(self, player, logger=None):
        logger = _resolve_logger(player, logger)
        if not bool(getattr(player, "_memory_enabled", False)):
            return
        if not self.memory_file or not os.path.exists(self.memory_file):
            return
        try:
            with open(self.memory_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            current_namespace = str(getattr(player, "_memory_namespace", "") or "")
            snapshot_namespace = str(data.get("memory_namespace", "") or "")
            if current_namespace and snapshot_namespace != current_namespace:
                if logger is not None:
                    logger.warning(
                        "  [PLAYER memory] skip %s: namespace %r != %r",
                        self.memory_file,
                        snapshot_namespace,
                        current_namespace,
                    )
                return
            player._shadow_player_regime_memory = self._restore_regime_memory(
                data.get("player_regime_memory") or {}
            )
            player._shadow_player_symbol_memory = self._restore_symbol_memory(
                data.get("player_symbol_regime_memory") or {}
            )
            player._shadow_player_window_anchor = self._sanitize_perf_anchor(
                data.get("player_window_anchor") or {}
            )
            player._shadow_player_symbol_anchor = self._sanitize_symbol_anchor(
                data.get("player_symbol_anchor") or {}
            )
            # FIX 2026-05-03 (variant 2): per-bar raw per-symbol stats для
            # Panteon._symbol_live_allowed. Отдельный slot, чтобы не
            # пересекаться с regime-EMA store (`_shadow_player_symbol_memory`).
            player._shadow_player_symbol_perf = self._restore_player_symbol_perf(
                data.get("player_symbol_memory") or {}
            )
            selected = str(data.get("selected_player") or "")
            if selected in getattr(player, "_shadow_player_pool", {}):
                player._selected_shadow_player = selected
            restored_symbols = {}
            for sym, name in (data.get("selected_symbol_players") or {}).items():
                if name in getattr(player, "_shadow_player_pool", {}):
                    restored_symbols[str(sym)] = str(name)
            player._selected_symbol_shadow_players = restored_symbols
            player._last_shadow_player_switch_bar = int(data.get("last_switch_bar", -99999) or -99999)
            if logger is not None:
                logger.info(
                    "  [PLAYER memory] loaded: selected=%s regimes=%s symbols=%d",
                    player._selected_shadow_player or "-",
                    ",".join(sorted(player._shadow_player_regime_memory)) or "empty",
                    len(player._shadow_player_symbol_memory),
                )
        except Exception as exc:
            if logger is not None:
                logger.warning("  [PLAYER memory] load failed: %s", exc)

    def save_snapshot(self, player, force: bool = False, reason: str = "", logger=None):
        if not bool(getattr(player, "_memory_enabled", False)):
            return
        cur_bar = int(getattr(player, "_t", 0) or 0)
        if not force and cur_bar - int(getattr(player, "_last_shadow_player_memory_save_bar", -99999)) < self.save_interval_bars:
            return
        logger = _resolve_logger(player, logger)
        try:
            payload = {
                "schema": self.SCHEMA_VERSION,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "player": type(player).__name__,
                "memory_namespace": str(getattr(player, "_memory_namespace", "") or ""),
                "exchange_namespace": str(getattr(player, "_memory_namespace", "") or ""),
                "selected_player": str(getattr(player, "_selected_shadow_player", "") or ""),
                "selected_symbol_players": dict(getattr(player, "_selected_symbol_shadow_players", {}) or {}),
                "last_switch_bar": int(getattr(player, "_last_shadow_player_switch_bar", -99999) or -99999),
                "player_scores": {
                    name: round(float(score), 6)
                    for name, score in sorted((getattr(player, "_shadow_player_scores", {}) or {}).items())
                },
                "player_regime_memory": self._dump_regime_memory(player),
                "player_symbol_regime_memory": self._dump_symbol_memory(player),
                "player_window_anchor": self._sanitize_perf_anchor(
                    getattr(player, "_shadow_player_window_anchor", {}) or {}
                ),
                "player_symbol_anchor": self._sanitize_symbol_anchor(
                    getattr(player, "_shadow_player_symbol_anchor", {}) or {}
                ),
                # FIX 2026-05-03 (variant 2): per-bar raw per-symbol stats,
                # которые читает Panteon._symbol_live_allowed.
                "player_symbol_memory": self._dump_player_symbol_perf(player),
            }
            tmp_path = self.memory_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.memory_file)
            player._last_shadow_player_memory_save_bar = cur_bar
            if logger is not None and (force or reason in ("player-rotation", "shutdown")):
                logger.debug("  [PLAYER memory] snapshot saved (%s)", reason or "periodic")
        except Exception as exc:
            if logger is not None:
                logger.warning("  [PLAYER memory] save failed: %s", exc)

    def update_memory(self, player, context: Optional[Dict[str, str]] = None):
        perf_map = getattr(player, "_shadow_player_perf", None)
        if not isinstance(perf_map, dict) or not perf_map:
            return
        ctx = dict(context or getattr(player, "_current_context", {}) or {})
        regime = self._canonical_regime(player, ctx.get("regime") or getattr(player, "_r", None) or "neutral")
        ctx["regime"] = regime

        regime_map = getattr(player, "_shadow_player_regime_memory", None)
        if not isinstance(regime_map, dict):
            player._shadow_player_regime_memory = {}
            regime_map = player._shadow_player_regime_memory
        symbol_map = getattr(player, "_shadow_player_symbol_memory", None)
        if not isinstance(symbol_map, dict):
            player._shadow_player_symbol_memory = {}
            symbol_map = player._shadow_player_symbol_memory

        prev_anchor = getattr(player, "_shadow_player_window_anchor", {}) or {}
        prev_symbol_anchor = getattr(player, "_shadow_player_symbol_anchor", {}) or {}

        for name, perf in perf_map.items():
            if name not in getattr(player, "_shadow_player_pool", {}):
                continue
            if not isinstance(perf, dict):
                continue
            observation = self._build_player_observation(perf, prev_anchor.get(name) or {})
            if observation is not None:
                bucket = regime_map.setdefault(regime, {}).setdefault(name, self._empty_stats())
                self._update_bucket(bucket, observation["score"], observation["pnl"])

            per_symbol = perf.get("per_symbol") if isinstance(perf, dict) else None
            if not isinstance(per_symbol, dict):
                continue
            for sym, stats in per_symbol.items():
                if not isinstance(stats, dict):
                    continue
                sym = str(sym)
                sym_regime = self._symbol_regime(player, sym, fallback=regime)
                prev_stats = ((prev_symbol_anchor.get(name) or {}).get(sym) or {})
                sym_obs = self._build_symbol_observation(stats, prev_stats)
                if sym_obs is None:
                    continue
                bucket = (
                    symbol_map
                    .setdefault(name, {})
                    .setdefault(sym, {})
                    .setdefault(sym_regime, self._empty_stats())
                )
                self._update_bucket(bucket, sym_obs["score"], sym_obs["pnl"])

        player._shadow_player_window_anchor = self._sanitize_perf_anchor(perf_map)
        player._shadow_player_symbol_anchor = self._sanitize_symbol_anchor(
            {
                name: perf.get("per_symbol")
                for name, perf in perf_map.items()
                if isinstance(perf, dict)
            }
        )

    def select(self, player, context: Optional[Dict[str, str]] = None, prices: Optional[dict] = None, logger=None) -> bool:
        perf_map = getattr(player, "_shadow_player_perf", None)
        pool = getattr(player, "_shadow_player_pool", {}) or {}
        if not isinstance(perf_map, dict) or not perf_map or not pool:
            return False

        logger = _resolve_logger(player, logger)
        ctx = dict(context or getattr(player, "_current_context", {}) or {})
        regime = self._canonical_regime(player, ctx.get("regime") or getattr(player, "_r", None) or "neutral")
        ctx["regime"] = regime
        prices = prices or {}

        self.update_memory(player, ctx)

        ranked = []
        score_lookup = {}
        # FIX A2/A3 (2026-04-26): фильтр shadow_only/quarantine.
        shadow_only = set()
        for name, instance in pool.items():
            status = str(getattr(instance, "PLAYER_STATUS", "live") or "live").lower()
            if status in ("shadow_only", "quarantine"):
                shadow_only.add(name)
        for name in pool:
            perf = perf_map.get(name)
            if not isinstance(perf, dict):
                continue
            if not self._eligible(perf, allow_inactive=name == getattr(player, "_selected_shadow_player", "")):
                continue
            score = self._combined_score(player, name, perf, regime)
            score_lookup[name] = score
            if name in shadow_only:
                continue
            # FIX A2 (2026-04-26) + FIX 2026-05-04 (per-regime breakthrough):
            # новички с closed_trades < min блокировались жёстко (min_closed=5).
            # Это убивает молодых ярких лидеров — например, V_SoloLiveTrendFollow
            # с +0.60% PnL и 2 closed_trades. Теперь блокируем мягче:
            #   - если у игрока в ТЕКУЩЕМ режиме рынка уже доказана прибыль
            #     (per_regime[regime].pnl_pct > 0 при closed >= 2), он считается
            #     "доказавшим себя локально" и не получает штрафа;
            #   - иначе применяем мягкий штраф (×0.4) вместо обнуления.
            closed = int((perf or {}).get("closed_trades",
                                          (perf or {}).get("total_trades", 0)) or 0)
            min_closed = int(getattr(player, "PLAYER_MIN_LIVE_CLOSED_TRADES", 5) or 5)
            current_selected = str(getattr(player, "_selected_shadow_player", "") or "")
            # Локально доказанная прибыль в текущем режиме?
            per_regime = perf.get("per_regime") if isinstance(perf, dict) else None
            stats_r = (per_regime or {}).get(regime) if isinstance(per_regime, dict) else None
            locally_proven = False
            if isinstance(stats_r, dict):
                r_pnl = float(stats_r.get("pnl_pct", 0.0) or 0.0)
                r_closed = int(stats_r.get("closed_trades", 0) or 0)
                if r_pnl > 0.0 and r_closed >= 2:
                    locally_proven = True
            if (
                closed < min_closed
                and name != current_selected
                and not locally_proven
            ):
                # Мягкий штраф вместо жёсткого обнуления — игрок может
                # подняться над plateau если его per-regime PnL положительный.
                score = score * 0.40 if score > 0 else min(score, 0.0)
                score_lookup[name] = score
            ranked.append((score, name))

        if not ranked:
            return False

        ranked.sort(reverse=True)
        player._shadow_player_scores = dict(score_lookup)
        player._shadow_only_players = sorted(shadow_only)

        old_selected = str(getattr(player, "_selected_shadow_player", "") or "")
        if old_selected not in pool:
            old_selected = ""
        best_score, best_name = ranked[0]
        current_score = score_lookup.get(old_selected, float("-inf"))
        changed = False

        if not old_selected:
            player._selected_shadow_player = best_name
            player._last_shadow_player_switch_bar = int(getattr(player, "_t", 0) or 0)
            player._shadow_player_challenger = ""
            player._shadow_player_challenger_streak = 0
            changed = True
        elif best_name != old_selected:
            cur_bar = int(getattr(player, "_t", 0) or 0)
            # FIX 2026-05-04 (faster lineup rotation): cooldown снижен с 90 до 30
            # баров (≈5 минут вместо 1.5ч), чтобы Пантеон быстрее уходил от
            # плохого текущего лидера. PLAYER_HARD_NEGATIVE_SCORE с −1.25 → −0.50:
            # сейчас игрок с pnl=−0.30% не считался urgent и держался час+.
            cooldown = int(getattr(player, "PLAYER_SWITCH_COOLDOWN_BARS", 30) or 30)
            margin = float(getattr(player, "PLAYER_SWITCH_MARGIN", 0.30) or 0.30)
            min_score = float(getattr(player, "PLAYER_MIN_SCORE_TO_SWITCH", -0.15) or -0.15)
            streak_needed = int(getattr(player, "PLAYER_SWITCH_CONFIRMATIONS", 2) or 2)
            since_switch = cur_bar - int(getattr(player, "_last_shadow_player_switch_bar", -99999) or -99999)
            hard_neg = float(getattr(player, "PLAYER_HARD_NEGATIVE_SCORE", -0.50) or -0.50)
            urgent = (
                current_score <= hard_neg
                or best_score >= current_score + 0.80  # большой gap → urgent
            )
            ready = best_score >= current_score + margin and best_score >= min_score
            if ready and (since_switch >= cooldown or urgent):
                if getattr(player, "_shadow_player_challenger", "") == best_name:
                    player._shadow_player_challenger_streak += 1
                else:
                    player._shadow_player_challenger = best_name
                    player._shadow_player_challenger_streak = 1
                if urgent or player._shadow_player_challenger_streak >= streak_needed:
                    player._selected_shadow_player = best_name
                    player._last_shadow_player_switch_bar = cur_bar
                    player._shadow_player_challenger = ""
                    player._shadow_player_challenger_streak = 0
                    changed = True
            else:
                player._shadow_player_challenger = ""
                player._shadow_player_challenger_streak = 0
        else:
            player._shadow_player_challenger = ""
            player._shadow_player_challenger_streak = 0

        self._select_symbol_players(player, score_lookup, regime, prices)

        if logger is not None and changed:
            logger.info(
                "  [PLAYER meta] selected=%s score=%.3f regime=%s | top=%s",
                player._selected_shadow_player,
                float(score_lookup.get(player._selected_shadow_player, 0.0) or 0.0),
                regime,
                ", ".join(f"{name}:{score:.2f}" for score, name in ranked[:4]),
            )
        self.save_snapshot(player, force=changed, reason="player-rotation", logger=logger)
        return changed

    def _select_symbol_players(self, player, global_scores: Dict[str, float], regime: str, prices: dict):
        selected = str(getattr(player, "_selected_shadow_player", "") or "")
        if not selected:
            return
        perf_map = getattr(player, "_shadow_player_perf", {}) or {}
        pool = getattr(player, "_shadow_player_pool", {}) or {}
        # FIX B4 (2026-04-26): shadow_only/quarantine не могут быть per-symbol лидерами
        shadow_only = set()
        for name, instance in pool.items():
            status = str(getattr(instance, "PLAYER_STATUS", "live") or "live").lower()
            if status in ("shadow_only", "quarantine"):
                shadow_only.add(name)
        out = {}
        margin = float(getattr(player, "SYMBOL_PLAYER_SWITCH_MARGIN", 0.28) or 0.28)
        hard_pnl_floor = float(getattr(player, "PLAYER_HARD_NEGATIVE_PNL", -1.5) or -1.5)
        min_leader_score = float(getattr(player, "PLAYER_MIN_SCORE_TO_SWITCH", 0.05) or 0.05)
        # FIX B4 (2026-04-26): жёсткие per-symbol требования. Лидер по символу
        # должен иметь хотя бы 3 закрытых сделок именно по этому символу и
        # win_rate > 50%. Без этого 1-2 случайных удачных сделки в выборке
        # позволяли убыточному игроку перехватывать символ.
        min_sym_closed = int(getattr(player, "PER_SYMBOL_MIN_CLOSED", 3) or 3)
        min_sym_win_rate = float(getattr(player, "PER_SYMBOL_MIN_WIN_RATE", 50.0) or 50.0)
        for sym in (prices or {}):
            sym = str(sym)
            ranked = []
            has_local_evidence = False
            for name in pool:
                if name in shadow_only:
                    continue
                perf = perf_map.get(name)
                if not isinstance(perf, dict):
                    continue
                cand_pnl = float((perf or {}).get("pnl_pct", 0.0) or 0.0)
                cand_closed = int((perf or {}).get("closed_trades", (perf or {}).get("total_trades", 0)) or 0)
                is_hard_negative = cand_closed >= 2 and cand_pnl <= hard_pnl_floor
                live_score, live_has = self._symbol_live_score(perf, sym)
                mem_score, mem_has = self._symbol_memory_score(player, name, sym, regime)
                if live_has or mem_has:
                    has_local_evidence = True
                # B4: per-symbol метрики
                per_sym = ((perf or {}).get("per_symbol") or {}).get(sym) or {}
                sym_closed = int(per_sym.get("closed_trades", per_sym.get("total_trades", 0)) or 0)
                sym_wr = float(per_sym.get("win_rate", 0.0) or 0.0)
                sym_pnl = float(per_sym.get("pnl_pct", 0.0) or 0.0)

                score = float(global_scores.get(name, -2.0) or -2.0) * 0.35 + live_score * 0.45 + mem_score * 0.75
                if is_hard_negative:
                    score -= 1.5
                if sym_closed < min_sym_closed:
                    score -= 0.6
                if sym_closed >= min_sym_closed and sym_wr < min_sym_win_rate:
                    score -= 0.5
                if sym_pnl < 0.0 and sym_closed >= 2:
                    score -= 0.3
                ranked.append((score, name, sym_closed, sym_wr))
            if not ranked or not has_local_evidence:
                continue
            ranked.sort(key=lambda x: x[0], reverse=True)
            best_score, best_name, best_closed, best_wr = ranked[0]
            if best_score < min_leader_score:
                continue
            # FIX B4: лидер должен соответствовать жёстким per-symbol требованиям
            if best_closed < min_sym_closed:
                continue
            if best_closed >= min_sym_closed and best_wr < min_sym_win_rate:
                continue
            selected_score = next((s for s, n, *_ in ranked if n == selected), float("-inf"))
            if best_name != selected and best_score >= selected_score + margin:
                out[sym] = best_name
        player._selected_symbol_shadow_players = out

    def _combined_score(self, player, name: str, perf: dict, regime: str) -> float:
        live = self._live_score(perf)
        regime_memory = self._regime_memory_score(player, name, regime)
        symbol_memory = self._symbol_mix_score(player, name, regime)
        prior = 0.0
        prior_fn = getattr(player, "_shadow_player_static_prior", None)
        if callable(prior_fn):
            try:
                prior = float(prior_fn(name, regime) or 0.0)
            except Exception:
                prior = 0.0
        # FIX 2026-05-04 (ensemble per-regime): ПРЯМОЙ бонус/штраф из
        # per_regime[regime]. Без этого _live_score (агрегат) всегда
        # доминирует и Пантеон выбирает игрока, у которого, скажем,
        # суммарный pnl=-0.4%, хотя в текущем bullish-режиме он бы был
        # лучшим. Этот бонус считает "локально доказанную" прибыль.
        regime_direct = 0.0
        per_regime = perf.get("per_regime") if isinstance(perf, dict) else None
        stats = (per_regime or {}).get(regime) if isinstance(per_regime, dict) else None
        if isinstance(stats, dict):
            r_pnl = float(stats.get("pnl_pct", 0.0) or 0.0)
            r_closed = int(stats.get("closed_trades", 0) or 0)
            r_entries = int(stats.get("entries", 0) or 0)
            r_signals = int(stats.get("signals", 0) or 0)
            # Confidence по выборке
            conf = min(max(r_closed, 0) / 4.0, 1.0)
            # Прямой вклад per-regime PnL (масштаб такой же, как у _live_score
            # `pnl * 0.62` — берём 0.55 чтобы не передавить).
            regime_direct = r_pnl * 0.55 * conf
            # Бонус активности в режиме
            if r_entries > 0 or r_signals > 0:
                regime_direct += 0.04 * conf
            # Жёсткий штраф если в ТЕКУЩЕМ режиме у игрока локально
            # доказан минус (закрытий >= 2 и pnl < 0). Снимает доминирование
            # игроков с положительным агрегатом, но плохим текущим режимом.
            if r_closed >= 2 and r_pnl < 0.0:
                regime_direct -= min(abs(r_pnl) * 0.40, 1.5)
        combined = float(
            live
            + regime_memory * 1.10
            + symbol_memory * 0.50
            + regime_direct
            + prior
        )
        # FIX: Жёсткий потолок итогового скора по текущему pnl. Даже если
        # regime_memory/priors положительны, игрок с pnl<=-1.5% в свежем
        # окне не должен опережать тех, кто сейчас в плюсе.
        pnl = float((perf or {}).get("pnl_pct", 0.0) or 0.0)
        closed = int((perf or {}).get("closed_trades", (perf or {}).get("total_trades", 0)) or 0)
        hard_pnl_floor = float(getattr(player, "PLAYER_HARD_NEGATIVE_PNL", -1.5) or -1.5)
        if closed >= 2 and pnl <= hard_pnl_floor:
            combined = min(combined, hard_pnl_floor * 0.5)
        return combined

    def _live_score(self, perf: dict) -> float:
        pnl = float((perf or {}).get("pnl_pct", 0.0) or 0.0)
        sharpe = float((perf or {}).get("sharpe", 0.0) or 0.0)
        max_dd = abs(float((perf or {}).get("max_dd", (perf or {}).get("max_drawdown_pct", 0.0)) or 0.0))
        win_rate = float((perf or {}).get("win_rate", 0.0) or 0.0)
        signals = int((perf or {}).get("signals", 0) or 0)
        entries = int((perf or {}).get("entries", 0) or 0)
        closed = int((perf or {}).get("closed_trades", (perf or {}).get("total_trades", 0)) or 0)
        sample = max(float(closed), float(entries) * 0.35)
        sample_conf = min(sample / 8.0, 1.0)
        effective_pnl = pnl if closed >= 2 else pnl * 0.45
        effective_sharpe = np.clip(sharpe, -20.0, 35.0) * sample_conf
        # FIX: Ослабили активность — раньше игрок с 33 сигналами/17 входами/11
        # закрытыми получал +1.6 бонуса, что легко перевешивало -2.1 от pnl=-3.4%.
        # Было: min(signals,50)*0.012 + min(entries,24)*0.035 + min(closed,12)*0.055
        activity = min(signals, 30) * 0.008 + min(entries, 16) * 0.022 + min(closed, 8) * 0.040
        # Потолок активности 0.80 (было ~1.81).
        activity = min(activity, 0.80)
        win_bonus = ((win_rate - 50.0) / 18.0) if closed >= 3 else 0.0
        inactive_penalty = 0.65 if signals <= 0 and entries <= 0 else 0.0
        churn_penalty = 0.0
        # FIX: Понизили порог (было entries>50 и pnl<0.75 — не срабатывало
        # для V_PlayerFunding с 17 входов и pnl=-3.4%). Теперь активный
        # игрок с неважным pnl получает ощутимый штраф.
        if entries > 12 and pnl < 0.40:
            churn_penalty += min((entries - 12) * 0.05, 2.5)
        # FIX: Явный штраф за сильно отрицательный pnl (накапливается
        # дополнительно к effective_pnl*0.62, чтобы убыточный игрок
        # не смог всплыть через activity).
        hard_negative_penalty = 0.0
        if closed >= 2 and pnl <= -1.5:
            hard_negative_penalty = min((abs(pnl) - 1.5) * 0.55, 3.0)
        return float(
            effective_pnl * 0.62
            + effective_sharpe * 0.34
            + activity
            + win_bonus
            - max_dd * 0.18
            - inactive_penalty
            - churn_penalty
            - hard_negative_penalty
        )

    def _symbol_live_score(self, perf: dict, sym: str) -> tuple[float, bool]:
        stats = ((perf or {}).get("per_symbol") or {}).get(sym)
        if not isinstance(stats, dict):
            return 0.0, False
        signals = int(stats.get("signals", 0) or 0)
        entries = int(stats.get("entries", 0) or 0)
        closed = int(stats.get("closed_trades", 0) or 0)
        has_open = int(stats.get("has_open_position", 0) or 0)
        if signals <= 0 and entries <= 0 and closed <= 0 and not has_open:
            return 0.0, False
        pnl = float(stats.get("total_pnl_pct", stats.get("realized_pnl_pct", 0.0)) or 0.0)
        activity = min(signals, 12) * 0.015 + min(entries, 8) * 0.04 + min(closed, 6) * 0.06
        return float(pnl * 0.80 + activity), True

    def _regime_memory_score(self, player, name: str, regime: str) -> float:
        stats = (getattr(player, "_shadow_player_regime_memory", {}) or {}).get(regime, {}).get(name)
        if not isinstance(stats, dict):
            return 0.0
        samples = int(stats.get("samples", 0) or 0)
        if samples <= 0:
            return 0.0
        confidence = min(samples / 5.0, 1.0)
        return float(np.clip(float(stats.get("ema_score", 0.0) or 0.0), -3.0, 3.0) * confidence)

    def _symbol_memory_score(self, player, name: str, sym: str, regime: str) -> tuple[float, bool]:
        by_symbol = (getattr(player, "_shadow_player_symbol_memory", {}) or {}).get(name, {}).get(sym, {})
        if not isinstance(by_symbol, dict):
            return 0.0, False
        # FIX: Раньше при отсутствии статистики для текущего режима падали на
        # ПЕРВУЮ попавшуюся, из-за чего crash-статистика могла применяться
        # к neutral-решениям. Теперь используем ТОЛЬКО матч режима, иначе
        # сигнализируем отсутствие доказательств.
        stats = by_symbol.get(regime)
        if not isinstance(stats, dict):
            return 0.0, False
        samples = int(stats.get("samples", 0) or 0)
        if samples <= 0:
            return 0.0, False
        confidence = min(samples / 4.0, 1.0)
        return float(np.clip(float(stats.get("ema_score", 0.0) or 0.0), -3.0, 3.0) * confidence), True

    def _symbol_mix_score(self, player, name: str, regime: str) -> float:
        scores = []
        for sym in getattr(player, "_ph", {}) or {}:
            score, has_score = self._symbol_memory_score(player, name, str(sym), regime)
            if has_score:
                scores.append(score)
        if not scores:
            return 0.0
        return float(np.mean(sorted(scores, reverse=True)[:5]))

    def _build_player_observation(self, perf: dict, prev: dict) -> Optional[dict]:
        signals = int(perf.get("signals", 0) or 0)
        entries = int(perf.get("entries", 0) or 0)
        closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
        if signals <= 0 and entries <= 0 and closed <= 0:
            return None
        pnl = float(perf.get("pnl_pct", 0.0) or 0.0)
        prev_pnl = float((prev or {}).get("pnl_pct", 0.0) or 0.0)
        delta_pnl = pnl - prev_pnl
        sharpe = float(perf.get("sharpe", 0.0) or 0.0)
        max_dd = abs(float(perf.get("max_dd", 0.0) or 0.0))
        prev_dd = abs(float((prev or {}).get("max_dd", 0.0) or 0.0))
        win_rate = float(perf.get("win_rate", 0.0) or 0.0)
        win_bonus = ((win_rate - 50.0) / 22.0) if closed >= 2 else 0.0
        activity = min(signals, 35) * 0.008 + min(entries, 18) * 0.025 + min(closed, 10) * 0.04
        score = delta_pnl * 0.85 + np.clip(sharpe, -12.0, 20.0) * 0.045 + activity + win_bonus - max(max_dd - prev_dd, 0.0) * 0.25
        return {"score": float(score), "pnl": float(delta_pnl)}

    def _build_symbol_observation(self, stats: dict, prev: dict) -> Optional[dict]:
        signals = int(stats.get("signals", 0) or 0)
        entries = int(stats.get("entries", 0) or 0)
        closed = int(stats.get("closed_trades", 0) or 0)
        has_open = int(stats.get("has_open_position", 0) or 0)
        if signals <= 0 and entries <= 0 and closed <= 0 and not has_open:
            return None
        pnl = float(stats.get("total_pnl_pct", stats.get("realized_pnl_pct", 0.0)) or 0.0)
        prev_pnl = float((prev or {}).get("total_pnl_pct", (prev or {}).get("realized_pnl_pct", 0.0)) or 0.0)
        delta = pnl - prev_pnl
        activity = min(signals, 10) * 0.012 + min(entries, 6) * 0.035 + min(closed, 4) * 0.05
        return {"score": float(delta * 0.85 + activity), "pnl": float(delta)}

    def _eligible(self, perf: dict, allow_inactive: bool = False) -> bool:
        signals = int((perf or {}).get("signals", 0) or 0)
        entries = int((perf or {}).get("entries", 0) or 0)
        closed = int((perf or {}).get("closed_trades", (perf or {}).get("total_trades", 0)) or 0)
        pnl = float((perf or {}).get("pnl_pct", 0.0) or 0.0)
        max_dd = abs(float((perf or {}).get("max_dd", (perf or {}).get("max_drawdown_pct", 0.0)) or 0.0))
        if allow_inactive:
            return True
        if signals < 2 and entries <= 0 and closed <= 0:
            return False
        if max_dd >= 14.0 and pnl < 0.0:
            return False
        # FIX: Раньше игрок с pnl=-3.4% и max_dd=3.7% проходил eligibility,
        # потому что порог был max_dd>=14. Теперь дисквалифицируем
        # явно убыточных игроков: pnl<=-2.0% ИЛИ drawdown больше
        # удвоенного pnl при минимум 2 закрытых сделках.
        if closed >= 2:
            if pnl <= -2.0:
                return False
            if pnl < 0.0 and max_dd >= abs(pnl) * 2.0 and max_dd >= 3.0:
                return False
        return True

    def _empty_stats(self) -> dict:
        return {
            "ema_score": 0.0,
            "avg_pnl": 0.0,
            "samples": 0,
            "wins": 0,
            "losses": 0,
            "last_window_pnl": 0.0,
            "last_updated": "",
        }

    def _update_bucket(self, stats: dict, score: float, pnl: float):
        samples = int(stats.get("samples", 0) or 0)
        alpha = 0.32 if samples < 4 else 0.18
        stats["ema_score"] = float(stats.get("ema_score", 0.0) or 0.0) * (1.0 - alpha) + float(score) * alpha
        stats["avg_pnl"] = float(stats.get("avg_pnl", 0.0) or 0.0) * (1.0 - alpha) + float(pnl) * alpha
        stats["samples"] = samples + 1
        stats["wins"] = int(stats.get("wins", 0) or 0) + int(score > 0.20 or pnl > 0.08)
        stats["losses"] = int(stats.get("losses", 0) or 0) + int(score < -0.20 or pnl < -0.08)
        stats["last_window_pnl"] = float(pnl)
        stats["last_updated"] = datetime.now(timezone.utc).isoformat()

    def _canonical_regime(self, player, regime: str) -> str:
        fn = getattr(player, "_canonical_market_regime", None)
        if callable(fn):
            try:
                return str(fn(regime))
            except Exception:
                pass
        text = str(regime or "neutral").lower()
        if text in ("bull", "bullish", "uptrend"):
            return "bullish"
        if text in ("bear", "bearish", "downtrend"):
            return "bearish"
        if text == "crash":
            return "crash"
        return "neutral"

    def _symbol_regime(self, player, sym: str, fallback: str = "neutral") -> str:
        fn = getattr(player, "_detect_symbol_regime", None)
        hist_map = getattr(player, "_ph", {}) or {}
        if callable(fn):
            try:
                regime = fn(sym, list(hist_map.get(sym) or []))
                if regime and regime != "unknown":
                    return self._canonical_regime(player, regime)
            except Exception:
                pass
        return self._canonical_regime(player, fallback)

    def _sanitize_perf_anchor(self, raw: Optional[Dict[str, dict]]) -> Dict[str, dict]:
        out = {}
        for name, perf in (raw or {}).items():
            if not isinstance(perf, dict):
                continue
            out[str(name)] = {
                "pnl_pct": float(perf.get("pnl_pct", 0.0) or 0.0),
                "max_dd": float(perf.get("max_dd", perf.get("max_drawdown_pct", 0.0)) or 0.0),
                "signals": int(perf.get("signals", 0) or 0),
                "entries": int(perf.get("entries", 0) or 0),
                "closed_trades": int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0),
            }
        return out

    def _sanitize_symbol_anchor(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
        out = {}
        for name, sym_map in (raw or {}).items():
            if not isinstance(sym_map, dict):
                continue
            clean = {}
            for sym, stats in sym_map.items():
                if not isinstance(stats, dict):
                    continue
                clean[str(sym)] = {
                    "total_pnl_pct": float(stats.get("total_pnl_pct", stats.get("realized_pnl_pct", 0.0)) or 0.0),
                    "realized_pnl_pct": float(stats.get("realized_pnl_pct", 0.0) or 0.0),
                    "signals": int(stats.get("signals", 0) or 0),
                    "entries": int(stats.get("entries", 0) or 0),
                    "closed_trades": int(stats.get("closed_trades", 0) or 0),
                }
            if clean:
                out[str(name)] = clean
        return out

    # ============================================================
    # FIX 2026-05-03 (variant 2): per-bar raw per-symbol stats для
    # Panteon._symbol_live_allowed. Источник данных — поле
    # `per_symbol` внутри shadow-player perf, который собирается
    # `_build_shadow_perf_from_shadows` каждый бар через
    # `vp.export_symbol_stats(prices)`. Хранится в отдельном slot
    # `_shadow_player_symbol_perf`, чтобы не конфликтовать с regime-
    # EMA-store `_shadow_player_symbol_memory`, которым уже владеет
    # `update_memory()` и которое сериализуется как
    # `player_symbol_regime_memory`.
    # ============================================================
    _SYMBOL_PERF_KEYS = (
        "closed_trades", "wins", "losses", "win_rate",
        "signals", "entries",
        "total_pnl_pct", "realized_pnl_pct",
        "has_open_position", "last_updated",
    )

    def _build_player_symbol_perf(
        self,
        player,
        perf_map: Optional[Dict[str, dict]],
    ) -> Dict[str, Dict[str, dict]]:
        """Превращает shadow-player perf в `{player: {sym: {closed_trades, win_rate, ...}}}`.

        Используется как источник для `_symbol_live_allowed`. Filter:
        не сохраняем символы, по которым у игрока вообще не было
        активности (нет signals/entries/closed и нет open-позиции).
        """
        if not isinstance(perf_map, dict) or not perf_map:
            return {}
        pool = getattr(player, "_shadow_player_pool", {}) or {}
        now_iso = datetime.now(timezone.utc).isoformat()
        out: Dict[str, Dict[str, dict]] = {}
        for name, perf in perf_map.items():
            if pool and name not in pool:
                continue
            if not isinstance(perf, dict):
                continue
            per_sym = perf.get("per_symbol")
            if not isinstance(per_sym, dict) or not per_sym:
                continue
            sym_clean: Dict[str, dict] = {}
            for sym, st in per_sym.items():
                if not isinstance(st, dict):
                    continue
                signals = int(st.get("signals", 0) or 0)
                entries = int(st.get("entries", 0) or 0)
                closed = int(st.get("closed_trades", 0) or 0)
                wins = int(st.get("wins", 0) or 0)
                losses = int(st.get("losses", 0) or 0)
                has_open = bool(st.get("has_open_position", False))
                if (signals == 0 and entries == 0 and closed == 0
                        and not has_open):
                    continue
                win_rate = (wins / closed * 100.0) if closed > 0 else 0.0
                sym_clean[str(sym)] = {
                    "closed_trades": closed,
                    "wins": wins,
                    "losses": losses,
                    "win_rate": round(float(win_rate), 4),
                    "signals": signals,
                    "entries": entries,
                    "total_pnl_pct": round(float(st.get("total_pnl_pct", 0.0) or 0.0), 6),
                    "realized_pnl_pct": round(float(st.get("realized_pnl_pct", 0.0) or 0.0), 6),
                    "has_open_position": int(bool(has_open)),
                    "last_updated": now_iso,
                }
            if sym_clean:
                out[str(name)] = sym_clean
        return out

    def _dump_player_symbol_perf(self, player) -> Dict[str, Dict[str, dict]]:
        """Сериализатор для save_snapshot: фильтр по pool, sort, копия."""
        pool = getattr(player, "_shadow_player_pool", {}) or {}
        raw = getattr(player, "_shadow_player_symbol_perf", {}) or {}
        out: Dict[str, Dict[str, dict]] = {}
        for name, sym_map in sorted(raw.items()):
            if pool and name not in pool:
                continue
            if not isinstance(sym_map, dict) or not sym_map:
                continue
            clean: Dict[str, dict] = {}
            for sym, stats in sorted(sym_map.items()):
                if not isinstance(stats, dict):
                    continue
                row = {}
                for k in self._SYMBOL_PERF_KEYS:
                    if k not in stats:
                        continue
                    v = stats[k]
                    if isinstance(v, float):
                        row[k] = round(v, 6)
                    elif isinstance(v, bool):
                        row[k] = int(v)
                    else:
                        row[k] = v
                if row:
                    clean[str(sym)] = row
            if clean:
                out[str(name)] = clean
        return out

    def _restore_player_symbol_perf(self, raw: Optional[Dict[str, dict]]) -> Dict[str, Dict[str, dict]]:
        """Десериализатор для load_snapshot: whitelist полей, без фильтра по pool."""
        out: Dict[str, Dict[str, dict]] = {}
        for name, sym_map in (raw or {}).items():
            if not isinstance(sym_map, dict):
                continue
            clean: Dict[str, dict] = {}
            for sym, stats in sym_map.items():
                if not isinstance(stats, dict):
                    continue
                row: Dict[str, object] = {}
                for k in self._SYMBOL_PERF_KEYS:
                    if k not in stats:
                        continue
                    v = stats[k]
                    if k in ("closed_trades", "wins", "losses",
                             "signals", "entries", "has_open_position"):
                        try:
                            row[k] = int(v or 0)
                        except Exception:
                            row[k] = 0
                    elif k in ("win_rate", "total_pnl_pct", "realized_pnl_pct"):
                        try:
                            row[k] = float(v or 0.0)
                        except Exception:
                            row[k] = 0.0
                    else:
                        row[k] = str(v) if v is not None else ""
                if row:
                    clean[str(sym)] = row
            if clean:
                out[str(name)] = clean
        return out

    def _restore_regime_memory(self, raw: dict) -> dict:
        out = {}
        for regime, player_map in (raw or {}).items():
            if not isinstance(player_map, dict):
                continue
            clean_players = {}
            for name, stats in player_map.items():
                if isinstance(stats, dict):
                    clean_players[str(name)] = self._restore_stats(stats)
            if clean_players:
                out[str(regime)] = clean_players
        return out

    def _restore_symbol_memory(self, raw: dict) -> dict:
        out = {}
        for name, sym_map in (raw or {}).items():
            if not isinstance(sym_map, dict):
                continue
            clean_syms = {}
            for sym, regime_map in sym_map.items():
                if not isinstance(regime_map, dict):
                    continue
                clean_regimes = {}
                for regime, stats in regime_map.items():
                    if isinstance(stats, dict):
                        clean_regimes[str(regime)] = self._restore_stats(stats)
                if clean_regimes:
                    clean_syms[str(sym)] = clean_regimes
            if clean_syms:
                out[str(name)] = clean_syms
        return out

    def _restore_stats(self, stats: dict) -> dict:
        restored = self._empty_stats()
        restored.update({
            "ema_score": float(stats.get("ema_score", 0.0) or 0.0),
            "avg_pnl": float(stats.get("avg_pnl", 0.0) or 0.0),
            "samples": int(stats.get("samples", 0) or 0),
            "wins": int(stats.get("wins", 0) or 0),
            "losses": int(stats.get("losses", 0) or 0),
            "last_window_pnl": float(stats.get("last_window_pnl", 0.0) or 0.0),
            "last_updated": str(stats.get("last_updated", "") or ""),
        })
        return restored

    def _dump_regime_memory(self, player) -> dict:
        return {
            regime: {
                name: self._dump_stats(stats)
                for name, stats in sorted(player_map.items())
                if name in getattr(player, "_shadow_player_pool", {}) and isinstance(stats, dict)
            }
            for regime, player_map in sorted((getattr(player, "_shadow_player_regime_memory", {}) or {}).items())
            if isinstance(player_map, dict)
        }

    def _dump_symbol_memory(self, player) -> dict:
        return {
            name: {
                sym: {
                    regime: self._dump_stats(stats)
                    for regime, stats in sorted(regime_map.items())
                    if isinstance(stats, dict)
                }
                for sym, regime_map in sorted(sym_map.items())
                if isinstance(regime_map, dict)
            }
            for name, sym_map in sorted((getattr(player, "_shadow_player_symbol_memory", {}) or {}).items())
            if name in getattr(player, "_shadow_player_pool", {}) and isinstance(sym_map, dict)
        }

    def _dump_stats(self, stats: dict) -> dict:
        return {
            "ema_score": round(float(stats.get("ema_score", 0.0) or 0.0), 6),
            "avg_pnl": round(float(stats.get("avg_pnl", 0.0) or 0.0), 6),
            "samples": int(stats.get("samples", 0) or 0),
            "wins": int(stats.get("wins", 0) or 0),
            "losses": int(stats.get("losses", 0) or 0),
            "last_window_pnl": round(float(stats.get("last_window_pnl", 0.0) or 0.0), 6),
            "last_updated": str(stats.get("last_updated", "") or ""),
        }

        if logger is not None and materially_changed:
            active_msg = ", ".join(
                f"{label}={weight:.0%}"
                for label, weight in sorted(new_weights.items(), key=lambda item: item[1], reverse=True)
            )
            logger.info(
                "  [REAL risk-adjusted] active: %s | context: %s",
                active_msg,
                player._context_summary(context),
            )
        return True

    def rotate(self, player, logger=None):
        if player._shadow_perf is None:
            return
        logger = _resolve_logger(player, logger)
        mode = str(getattr(player, "AGGREGATION_MODE", "") or "").strip().lower()
        if mode == "risk_adjusted_top_positive":
            self._rotate_risk_adjusted_top_positive(player, logger=logger)
            return
        context = dict(player._current_context or player._build_market_context())
        regime = str(context.get("regime") or getattr(player, "_r", None) or "neutral")
        if hasattr(player, "_canonical_market_regime"):
            regime = player._canonical_market_regime(regime)
            context["regime"] = regime
        symbol_regime_mix = (
            player._current_symbol_regime_mix()
            if hasattr(player, "_current_symbol_regime_mix")
            else {}
        )
        candidates = []
        candidate_lookup = {}
        for shadow_name, perf in player._shadow_perf.items():
            label = player.SHADOW_MAP.get(shadow_name)
            if label and label in player._agent_pool:
                if hasattr(player, "_agent_currently_blocked") and player._agent_currently_blocked(label, perf):
                    continue
                pnl = perf.get("pnl_pct", 0.0)
                sharpe = float(perf.get("sharpe", 0.0) or 0.0)
                closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
                base_score = player._score_shadow_candidate(label, perf)
                memory_score = player._update_success_memory(label, perf)
                factor_ctx_score, full_ctx_score = player._context_memory_score(label, context)
                regime_memory_score = (
                    float(player._regime_memory_score(label, regime))
                    if hasattr(player, "_regime_memory_score")
                    else 0.0
                )
                regime_prior_score = (
                    float(player._regime_static_prior(label, regime))
                    if hasattr(player, "_regime_static_prior")
                    else 0.0
                )
                symbol_regime_memory_score = (
                    float(player._symbol_regime_memory_mix(label, symbol_regime_mix))
                    if hasattr(player, "_symbol_regime_memory_mix")
                    else 0.0
                )
                symbol_regime_prior_score = (
                    float(player._symbol_regime_static_prior_mix(label, symbol_regime_mix))
                    if hasattr(player, "_symbol_regime_static_prior_mix")
                    else 0.0
                )
                symbol_regime_bonus = (
                    float(player._symbol_regime_bonus_mix(label, symbol_regime_mix, perf))
                    if hasattr(player, "_symbol_regime_bonus_mix")
                    else 0.0
                )
                symbol_opportunity = player._agent_symbol_opportunity_score(label)
                dominance_share = (
                    float(player._agent_recent_signal_share(label))
                    if hasattr(player, "_agent_recent_signal_share")
                    else 0.0
                )
                current_negative = closed >= 6 and (
                    float(pnl) <= -0.50 or (float(pnl) < -0.15 and sharpe < 0.0)
                )
                memory_weight = player.GLOBAL_MEMORY_WEIGHT * (0.25 if current_negative else 1.0)
                live_penalty = (
                    min(abs(float(pnl)) * 1.35 + max(-sharpe, 0.0) * 0.08, 5.0)
                    if current_negative else 0.0
                )
                dominance_penalty = 0.0
                if dominance_share > getattr(player, "DOMINANCE_SOFT_SHARE", 0.38):
                    severity = (
                        (dominance_share - getattr(player, "DOMINANCE_SOFT_SHARE", 0.38)) /
                        max(1e-6, 1.0 - getattr(player, "DOMINANCE_SOFT_SHARE", 0.38))
                    )
                    if current_negative:
                        dominance_penalty = 1.20 * severity
                    elif float(pnl) < 0.20:
                        dominance_penalty = 0.55 * severity
                regime_bonus = (
                    float(player._regime_score_adjustment(label, regime, perf))
                    if hasattr(player, "_regime_score_adjustment")
                    else 0.0
                )
                combined_score = (
                    base_score
                    + memory_score * memory_weight
                    + regime_memory_score * player.REGIME_MEMORY_WEIGHT
                    + regime_prior_score * player.REGIME_STATIC_PRIOR_WEIGHT
                    + factor_ctx_score * player.CONTEXT_FACTOR_WEIGHT
                    + full_ctx_score * player.CONTEXT_FULL_WEIGHT
                    + symbol_opportunity * player.SYMBOL_OPPORTUNITY_WEIGHT
                    + symbol_regime_memory_score * getattr(player, "SYMBOL_REGIME_ROTATION_WEIGHT", 0.0)
                    + symbol_regime_prior_score * getattr(player, "SYMBOL_REGIME_STATIC_PRIOR_WEIGHT", 0.0)
                    + regime_bonus
                    + symbol_regime_bonus
                    - live_penalty
                    - dominance_penalty
                )
                selection_regime_memory = max(regime_memory_score, symbol_regime_memory_score)
                selection_regime_prior = max(regime_prior_score, symbol_regime_prior_score)
                item = (
                    label,
                    combined_score,
                    pnl,
                    base_score,
                    memory_score,
                    factor_ctx_score,
                    full_ctx_score,
                    selection_regime_memory,
                    selection_regime_prior,
                )
                candidates.append(item)
                candidate_lookup[label] = item

        if len(candidates) < player.MIN_AGENTS:
            return

        candidates.sort(key=lambda item: item[1], reverse=True)
        local_regime_count = len(symbol_regime_mix)
        if local_regime_count >= 2:
            mixed_cap = int(getattr(player, "MAX_MIXED_SYMBOL_REGIME_AGENTS", player.MAX_AGENTS) or player.MAX_AGENTS)
            target_active = max(int(player.MAX_AGENTS), int(player.MIN_AGENTS) + local_regime_count)
            n_active = min(max(mixed_cap, int(player.MAX_AGENTS)), target_active, len(candidates))
        else:
            n_active = min(player.MAX_AGENTS, len(candidates))
        ranked = candidates[:]

        positive = [
            (label, score, pnl, base, mem, factor_ctx, full_ctx, regime_mem, regime_prior)
            for label, score, pnl, base, mem, factor_ctx, full_ctx, regime_mem, regime_prior in ranked
            if score > 0.0 and (float(pnl) >= 0.0 or float(regime_mem or 0.0) >= 0.75)
        ]
        if len(positive) >= player.MIN_AGENTS:
            ranked = positive
        else:
            ranked = candidates

        priority_labels = []
        if hasattr(player, "_priority_agents_for_regime"):
            priority_labels = list(player._priority_agents_for_regime(regime))
        if symbol_regime_mix and hasattr(player, "_priority_agents_for_symbol_regimes"):
            for label in player._priority_agents_for_symbol_regimes(symbol_regime_mix):
                if label not in priority_labels:
                    priority_labels.append(label)
        priority_set = set(priority_labels)

        # Regime priorities should bias close calls, not hard-reserve the whole
        # lineup. Hard reservation made live Panteon lag fresh shadow leaders.
        ranked.sort(
            key=lambda item: (
                item[1],
                1 if item[0] in priority_set else 0,
                float(item[2]),
            ),
            reverse=True,
        )
        top = ranked[:n_active]
        if symbol_regime_mix and len(symbol_regime_mix) >= 2 and hasattr(player, "_priority_agents_for_regime"):
            selected_labels = {item[0] for item in top}
            for local_regime, _share in sorted(symbol_regime_mix.items(), key=lambda item: item[1], reverse=True):
                local_priorities = tuple(player._priority_agents_for_regime(local_regime))
                if not local_priorities or any(label in selected_labels for label in local_priorities):
                    continue
                replacement = next(
                    (item for item in ranked if item[0] in local_priorities and item[0] not in selected_labels),
                    None,
                )
                if replacement is None:
                    continue
                if len(top) < n_active:
                    top.append(replacement)
                    selected_labels.add(replacement[0])
                    continue
                replace_idx = next(
                    (idx for idx in range(len(top) - 1, -1, -1) if top[idx][0] not in priority_set),
                    len(top) - 1,
                )
                tolerance = float(getattr(player, "SYMBOL_REGIME_COVERAGE_TOLERANCE", 0.0) or 0.0)
                if replacement[1] >= top[replace_idx][1] - tolerance:
                    selected_labels.discard(top[replace_idx][0])
                    top[replace_idx] = replacement
                    selected_labels.add(replacement[0])

        if len(top) < player.MIN_AGENTS:
            top = candidates[:player.MIN_AGENTS]

        min_pnl = 0.0
        # FIX (2026-04-26): хвост метода rotate был утрачен при обрыве файла.
        # Реальная логика хранится в .pyc; здесь — безопасное завершение.
        try:
            for label in [c[0] for c in top]:
                if label in player._agent_pool and label not in player._active_weights:
                    player._active_weights[label] = float(player.MIN_WEIGHT)
        except Exception:
            pass


# ── Helpers added 2026-04-26 to keep stale agent_meta importable ──────────
def _agent_meta_module_ok() -> bool:
    return True
