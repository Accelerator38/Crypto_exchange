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
        return self.success_signal(player, perf)

    def success_signal(self, player, perf: dict) -> float:
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

        activity = min(signals, 40) * 0.02 + min(entries, 20) * 0.04 + min(closed, 12) * 0.09
        win_bonus = ((win_rate - 50.0) / 10.0) if closed >= 2 else 0.0
        inactivity_penalty = 1.2 if signals == 0 and entries == 0 else 0.0
        low_sample_penalty = 0.0
        if closed == 0 and entries >= 3:
            low_sample_penalty = 0.90
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
        current = self.success_signal(player, perf)
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
        if not os.path.isfile(memory_path) and os.path.isfile(self.legacy_memory_file):
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
            "memory_mode": "regime_only" if getattr(player, "USE_REGIME_ONLY_MEMORY", False) else "hybrid",
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
    def rotate(self, player, logger=None):
        if player._shadow_perf is None:
            return
        logger = _resolve_logger(player, logger)
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

        min_pnl = min(score for _, score, _, _, _, _, _, _, _ in top)
        shift = abs(min_pnl) + 1.0
        raw = OrderedDict(
            (
                label,
                max(score + shift, 0.5)
                * (
                    1.0
                    + max(float(regime_prior or 0.0), 0.0) * 0.75
                    + max(float(regime_mem or 0.0), 0.0) * 0.08
                )
            )
            for label, score, _, _, _, _, _, regime_mem, regime_prior in top
        )
        if hasattr(player, "_normalize_weights"):
            new_weights = player._normalize_weights(raw, priority_labels=priority_labels)
        else:
            total = sum(raw.values())
            new_weights = {label: max(value / total, player.MIN_WEIGHT) for label, value in raw.items()}
            wsum = sum(new_weights.values())
            new_weights = {label: weight / wsum for label, weight in new_weights.items()}

        old_set = set(player._active_weights.keys())
        new_set = set(new_weights.keys())
        added = new_set - old_set
        removed = old_set - new_set
        if logger is not None and (added or removed):
            added_msg = ", ".join(
                f"{label}={new_weights[label]:.0%}"
                for label in sorted(added, key=lambda item: new_weights.get(item, 0), reverse=True)
            )
            removed_msg = ", ".join(
                f"{label}={player._active_weights.get(label, 0):.0%}"
                for label in sorted(removed, key=lambda item: player._active_weights.get(item, 0), reverse=True)
            )
            active_msg = ", ".join(
                f"{label}={weight:.0%}"
                for label, weight in sorted(new_weights.items(), key=lambda item: item[1], reverse=True)
            )
            parts = []
            if added_msg:
                parts.append(f"added: {added_msg}")
            if removed_msg:
                parts.append(f"removed: {removed_msg}")
            logger.info(
                "  [REAL rotation] %s | active: %s | context: %s",
                " | ".join(parts),
                active_msg,
                player._context_summary(context),
            )

        player._active_weights = new_weights
        player._update_context_memory(context)
        player._update_symbol_memory()
        player.save_memory_snapshot(force=True, reason="rotation")
