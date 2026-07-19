"""OutputWriter — периодическая запись output на диск.

Заменяет v1 функциональность: status.json + leaderboard_*.json +
dashboard ASCII + trading.log. Файлы пишутся в той же папке как v1:
  Results/{EXCHANGE}/{TIMESTAMP}/

Каждые N баров (или N секунд) делается snapshot всего state.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
import html
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..dashboards import (
    DashboardRenderer,
    TextRenderer,
    build_attribution_panel,
    build_leaderboard,
    build_quarantine_panel,
    build_regime_heatmap,
    write_operator_pngs,
)
from ..domain.types import Regime
from .bootstrap import ProductionPipeline
from .live_state import adopted_position_labels
from .main_loop import StepResult, sync_pipeline_balance


log = logging.getLogger(__name__)


def _first_positive(*values) -> float:
    for value in values:
        try:
            out = float(value)
            if out > 0:
                return out
        except (TypeError, ValueError):
            continue
    return 0.0


def _is_external_player(label: object) -> bool:
    return str(label or "") in {"", "RecoveredExchangePosition"}


def _is_adopted_player(label: object, pipeline: object = None) -> bool:
    adopted_player, adopted_agent = adopted_position_labels(pipeline)
    return str(label or "") in {adopted_player, adopted_agent}


def _configured_dynamic_player_labels(pipeline: object) -> set[str]:
    labels: set[str] = set()
    for attr in ("regime_switch_player_sets", "rotating_agent_player_sets"):
        for raw_spec in getattr(pipeline, attr, ()) or ():
            if isinstance(raw_spec, dict):
                label = raw_spec.get("label")
            else:
                try:
                    parts = tuple(raw_spec)  # type: ignore[arg-type]
                except TypeError:
                    parts = ()
                label = parts[0] if parts else ""
            clean = str(label or "").strip()
            if clean:
                labels.add(clean)
    for label in getattr(pipeline, "_current_actionable_player_labels", set()) or set():
        clean = str(label or "").strip()
        if clean:
            labels.add(clean)
    return labels


def _registered_strategy_labels(pipeline: object) -> set[str]:
    labels: set[str] = set()
    registry = getattr(pipeline, "registry", None)
    all_labels = getattr(registry, "all_labels", None)
    if callable(all_labels):
        try:
            labels.update(
                str(label).strip()
                for label in all_labels()
                if str(label or "").strip()
            )
        except Exception:
            pass
    labels.update(
        str(label).strip()
        for label in (getattr(pipeline, "shadow_agent_labels", ()) or ())
        if str(label or "").strip()
    )
    return labels


def _flash_decision_has_selected_signal(decision: Any) -> bool:
    if not isinstance(decision, dict):
        return False
    if str(decision.get("selected_actor") or "") == "NoTrade":
        return False
    return isinstance(decision.get("signal"), dict)


def _format_float(value: object) -> str:
    try:
        return f"{float(value or 0.0):.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _safe_float_or_none(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _object_float_attr(obj: object, *names: str) -> Optional[float]:
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        parsed = _safe_float_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _position_unrealized_pnl(
    tracked_pos: object,
    exchange_pos: object = None,
    *,
    current_prices: Optional[Dict[str, float]] = None,
) -> float:
    exchange_unrealized = _object_float_attr(exchange_pos, "unrealized_pnl")
    symbol = str(
        getattr(exchange_pos, "sym", "")
        or getattr(tracked_pos, "sym", "")
        or ""
    ).upper()
    side = str(
        getattr(exchange_pos, "side", "")
        or getattr(tracked_pos, "side", "")
        or ""
    ).lower()
    entry = _object_float_attr(exchange_pos, "entry", "entry_price")
    if entry is None or entry <= 0.0:
        entry = _object_float_attr(tracked_pos, "entry_price", "entry")
    qty = _object_float_attr(exchange_pos, "qty")
    if qty is None or qty <= 0.0:
        qty = _object_float_attr(tracked_pos, "qty")

    computed: Optional[float] = None
    if (
        current_prices
        and symbol
        and side in {"long", "short"}
        and entry is not None
        and entry > 0.0
        and qty is not None
        and qty > 0.0
    ):
        price = _safe_float_or_none(current_prices.get(symbol))
        if price is not None and price > 0.0:
            if side == "long":
                computed = (price - entry) * qty
            else:
                computed = (entry - price) * qty

    if computed is not None and (
        exchange_unrealized is None or abs(exchange_unrealized) <= 1e-12
    ):
        return computed
    return float(exchange_unrealized or 0.0)


def _clean_regimes_by_symbol(value: object) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for symbol, regime in value.items():
        clean_symbol = str(symbol or "").strip().upper()
        if not clean_symbol:
            continue
        label = getattr(regime, "label", str(regime or "")).strip().lower()
        if label:
            out[clean_symbol] = label
    return out


def _clean_regime_features_by_symbol(value: object) -> Dict[str, dict]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, dict] = {}
    for symbol, payload in value.items():
        clean_symbol = str(symbol or "").strip().upper()
        if not clean_symbol or not isinstance(payload, dict):
            continue
        clean_payload: Dict[str, object] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            if isinstance(raw_value, (str, bool)):
                clean_payload[key] = raw_value
                continue
            parsed = _safe_float_or_none(raw_value)
            if parsed is not None:
                clean_payload[key] = parsed
        if clean_payload:
            out[clean_symbol] = clean_payload
    return out


def _coerce_timestamp_text(value: object) -> str:
    if isinstance(value, dict):
        for key in ("utc", "timestamp_utc", "timestamp", "time"):
            text = _coerce_timestamp_text(value.get(key))
            if text:
                return text
        return ""
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return str(value or "")
    return str(value or "").strip()


def _utc_msk_timestamps() -> Dict[str, str]:
    now_utc = datetime.now(timezone.utc)
    now_msk = now_utc.astimezone(timezone(timedelta(hours=3)))
    return {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_msk": now_msk.isoformat(),
    }


@dataclass
class OutputWriterConfig:
    """Конфигурация писателя."""

    output_dir:           str
    write_every_bars:     int = 1       # каждый bar
    full_snapshot_every:  int = 30      # leaderboard каждые N баров
    trading_log_filename: str = "trading.log"
    status_filename:      str = "status.json"
    leaderboard_agents:   str = "leaderboard_agents.json"
    leaderboard_players:  str = "leaderboard_players.json"
    leaderboard_real_executable_agents: str = "leaderboard_real_executable_agents.json"
    leaderboard_shadow_only_agents: str = "leaderboard_shadow_only_agents.json"
    leaderboard_real_executable_players: str = "leaderboard_real_executable_players.json"
    leaderboard_shadow_only_players: str = "leaderboard_shadow_only_players.json"
    dashboard_filename:   str = "dashboard.txt"
    dashboard_html:       str = "dashboard.html"
    events_jsonl:         str = "events.jsonl"
    causal_entry_decisions_jsonl: str = "causal_entry_decisions.jsonl"
    missed_opportunities_jsonl: str = "missed_opportunities.jsonl"
    compact_causal_entry_decisions: bool = False
    compact_causal_entry_selected_only: bool = False
    compact_causal_entry_top_rejected_candidates: int = 5
    compact_causal_entry_include_labels: tuple[str, ...] = ()
    latest_dir:           Optional[str] = None


class OutputWriter:
    """Periodic snapshot writer.

    Использование:
        writer = OutputWriter(pipeline, OutputWriterConfig(output_dir=...))
        def on_step(step):
            writer.write(step)
        main_loop(pipeline, feed, on_step=on_step)
    """

    def __init__(self, pipeline: ProductionPipeline, config: OutputWriterConfig):
        self._pipeline = pipeline
        self._config = config
        self._start_time = time.time()
        self._initial_capital = pipeline.initial_capital
        self._current_balance = pipeline.initial_capital
        self._last_step: Optional[StepResult] = None
        self._last_status_data: Dict[str, object] = {}
        self._last_agents_payload: Dict[str, object] = {"metadata": {}, "agents": {}}
        self._last_players_payload: Dict[str, object] = {"metadata": {}, "players": {}}
        self._wrote_first_live_dashboard = False
        self._assets_history: List[float] = []
        self._panteon_equity_history: List[float] = []
        self._panteon_realized_equity_history: List[float] = []
        self._clean_panteon_equity_history: List[float] = []
        self._clean_panteon_realized_equity_history: List[float] = []
        self._equity_history_timestamps: List[str] = []
        self._price_history: List[Dict[str, object]] = []
        self._shadow_session_counts = {
            "signals": 0,
            "filled": 0,
            "rejected": 0,
            "blocked": 0,
        }
        self._fallback_session_counts = {
            "used": 0,
            "skipped": 0,
        }
        self._latest_dashboard_publish_disabled = False
        self._session_perf_baseline = self._capture_session_perf_baseline()
        # Создаём output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        # Открываем trading.log (append)
        self._log_path = os.path.join(config.output_dir, config.trading_log_filename)
        self._causal_entry_path = os.path.join(
            config.output_dir,
            config.causal_entry_decisions_jsonl,
        )
        self._missed_opportunities_path = os.path.join(
            config.output_dir,
            config.missed_opportunities_jsonl,
        )
        Path(self._causal_entry_path).touch(exist_ok=True)
        Path(self._missed_opportunities_path).touch(exist_ok=True)
        start_ts = _utc_msk_timestamps()
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(
                "Panteon v2 started: "
                f"UTC={start_ts['timestamp_utc']} "
                f"MSK={start_ts['timestamp_msk']} "
                f"pid={os.getpid()}\n"
            )
            f.write(f"  exchange: {pipeline.exchange_name}\n")
            f.write(f"  run_id: {getattr(pipeline, 'run_id', '')}\n")
            f.write(f"  session_id: {getattr(pipeline, 'session_id', '')}\n")
            f.write(f"  mode: {getattr(pipeline, 'mode', '')}\n")
            f.write(f"  timeframe: {getattr(pipeline, 'timeframe', '')}\n")
            f.write(f"  initial_capital: ${pipeline.initial_capital:.4f}\n")
            f.write(f"  profiles: {len(pipeline.profiles)}\n")
            f.write(f"  registered_agents: {len(pipeline.registry)}\n")
            f.write(f"{'='*70}\n\n")
        self._write_status(
            None,
            run_state="starting",
            feed_status="initializing",
            message="waiting for first market bar",
        )
        self._write_leaderboards()
        self._write_dashboard()

    @classmethod
    def for_session(
        cls,
        pipeline: ProductionPipeline,
        results_root: str = "Results",
        **kwargs,
    ) -> "OutputWriter":
        """Создаёт writer с автоматическим путём Results/{EXCHANGE}/{TS}/.

        Тот же путь как у v1.
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(
            results_root, pipeline.exchange_name.upper().replace("-DRYRUN","").replace("-PAPER",""),
            ts + "_v2",
        )
        session_id = os.path.basename(output_dir)
        if not getattr(pipeline, "session_id", ""):
            pipeline.session_id = session_id
        if not getattr(pipeline, "run_id", ""):
            pipeline.run_id = pipeline.session_id
        kwargs.setdefault("latest_dir", results_root)
        kwargs.setdefault("compact_causal_entry_decisions", True)
        cfg = OutputWriterConfig(output_dir=output_dir, **kwargs)
        return cls(pipeline, cfg)

    def write(self, step: StepResult) -> None:
        """Вызывается после каждого bar (через on_step callback)."""
        self._ensure_output_dir()
        self._last_step = step
        self._accumulate_shadow_counts(step)
        self._accumulate_fallback_counts(step)
        self._log_step(step)
        self._write_causal_entry_decision(step)
        self._write_missed_opportunities(step)

        # каждый bar — status.json
        if step.bar % self._config.write_every_bars == 0:
            self._write_status(step, run_state="running", feed_status="active")

        # реже — leaderboards + dashboard
        snapshot_due = (
            self._config.full_snapshot_every > 0
            and step.bar % self._config.full_snapshot_every == 0
        )
        if not self._wrote_first_live_dashboard or snapshot_due:
            self._write_leaderboards()
            self._write_dashboard()
            self._wrote_first_live_dashboard = True

    def write_heartbeat(
        self,
        *,
        run_state: str = "idle",
        feed_status: str = "waiting",
        message: str = "",
    ) -> None:
        """Update operator-visible files when no market bar has arrived yet."""
        self._ensure_output_dir()
        self._write_status(
            self._last_step,
            run_state=run_state,
            feed_status=feed_status,
            message=message,
        )

    def _log_step(self, step: StepResult) -> None:
        """Append одну строку в trading.log."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._refresh_account_view()
        position_counts = self._position_counts()
        selected_leader = step.selected_leader or step.leader or "-"
        executed_leader = step.executed_leader or step.leader or "-"
        market_view = self._market_view_payload(step, fallback_regime=step.regime.label)
        parts = [
            ts,
            f"bar={step.bar}",
            f"market={market_view['market']}",
            f"regime={step.regime.label}",
            f"leader={step.leader or '-'}",
            f"selected_leader={selected_leader}",
            f"executed_leader={executed_leader}",
            f"balance=${self._current_balance:.2f}",
            f"positions={position_counts['n_positions']}",
            f"raw_signals={step.n_raw_signals}",
            f"signals={step.n_signals}",
            f"filled={step.n_filled}",
            f"rejected={step.n_rejected}",
            f"blocked={step.n_blocked}",
            f"shadow_signals={step.n_shadow_signals}",
            f"shadow_filled={step.n_shadow_filled}",
            f"shadow_session_signals={self._shadow_session_counts['signals']}",
            f"shadow_session_filled={self._shadow_session_counts['filled']}",
        ]
        if market_view.get("regime_confidence") is not None:
            parts.append(f"market_confidence={float(market_view['regime_confidence']):.2f}")
        if market_view.get("funding_symbols"):
            avg_funding = market_view.get("avg_funding")
            if avg_funding is not None:
                parts.append(f"avg_funding={float(avg_funding) * 100:.4f}%")
            parts.append(f"funding_symbols={market_view['funding_symbols']}")
        if step.leader_vote_errors:
            parts.append(f"leader_vote_errors={step.leader_vote_errors}")
        if step.n_filtered_real_signals:
            parts.append(f"filtered={step.n_filtered_real_signals}")
            parts.append(f"stale_closes={step.n_stale_close_signals}")
            parts.append(f"duplicate_opens={step.n_duplicate_open_signals}")
            parts.append(f"rate_limited_opens={step.n_rate_limited_open_signals}")
            parts.append(f"max_position_saturated_opens={step.n_max_position_saturated_open_signals}")
            parts.append(f"external_position_signals={step.n_external_position_signals}")
        if step.blocked_reasons:
            parts.append(f"blocked_reasons={self._format_reason_counts(step.blocked_reasons)}")
        flash_payload = self._flash_status_payload(step)
        flash_actors = flash_payload.get("selected_actors_by_symbol", {})
        if flash_actors:
            parts.append(f"flash_actors={self._format_symbol_actor_map(flash_actors)}")
        if step.fallback_used or step.fallback_skipped or step.fallback_candidate or step.fallback_reason:
            parts.append(f"fallback_used={step.fallback_used}")
            parts.append(f"fallback_skipped={step.fallback_skipped}")
            if step.fallback_candidate:
                parts.append(f"fallback_candidate={step.fallback_candidate}")
            if step.fallback_reason:
                parts.append(f"fallback_reason={step.fallback_reason}")
        if step.leader_changed:
            parts.append("LEADER_CHANGED")
        if step.error:
            parts.append(f"ERROR={step.error}")
        line = "  ".join(parts)
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            log.exception("trading.log write failed")

    def _write_causal_entry_decision(self, step: StepResult) -> None:
        payload: Any = getattr(step, "causal_decision", None)
        if not payload:
            return
        row = dict(payload)
        if self._config.compact_causal_entry_decisions:
            row = self._compact_causal_entry_decision(row)
            policy_event = self._is_persistable_policy_decision(row)
            if (
                self._config.compact_causal_entry_selected_only
                and not policy_event
                and not row.get("flash_decisions")
                and not row.get("shadow_position_diagnostics")
                and not self._has_soft_allocator_execution(row)
            ):
                return
        row.setdefault("bar", step.bar)
        row.setdefault("regime", step.regime.label)
        row.setdefault("selected_leader", step.selected_leader or step.leader or "")
        row.setdefault("executed_leader", step.executed_leader or step.leader or "")
        row.setdefault("n_signals", step.n_signals)
        row.setdefault("n_filled", step.n_filled)
        row.setdefault("n_rejected", step.n_rejected)
        row.setdefault("n_blocked", step.n_blocked)
        try:
            with open(self._causal_entry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            log.exception("causal_entry_decisions.jsonl write failed")

    def _write_missed_opportunities(self, step: StepResult) -> None:
        row = self._missed_opportunity_row(step)
        if not row:
            return
        try:
            with open(self._missed_opportunities_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except Exception:
            log.exception("missed_opportunities.jsonl write failed")

    def _missed_opportunity_row(self, step: StepResult) -> Dict[str, Any]:
        causal = dict(getattr(step, "causal_decision", {}) or {})
        decisions = causal.get("flash_decisions")
        if not isinstance(decisions, list):
            return {}

        best: Dict[str, Any] = {}
        best_selected: Dict[str, Any] = {}
        best_decision: Dict[str, Any] = {}
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            candidates = decision.get("candidates")
            if not isinstance(candidates, list):
                continue
            selected_actor = str(decision.get("selected_actor") or "")
            selected_type = str(decision.get("actor_type") or "")
            selected_candidate = self._selected_candidate_from_flash_decision(
                candidates,
                selected_actor=selected_actor,
                selected_type=selected_type,
            )
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if not bool(candidate.get("rejected")):
                    continue
                if str(candidate.get("actor_type") or "") == "no_trade":
                    continue
                if not self._candidate_action_is_open(candidate):
                    continue
                shadow_score = _safe_float_or_none(candidate.get("shadow_score")) or 0.0
                pnl_net = _safe_float_or_none(candidate.get("pnl_net_pct")) or 0.0
                if shadow_score <= 0.0 and pnl_net <= 0.0:
                    continue
                if not best or (
                    shadow_score,
                    pnl_net,
                    -int(candidate.get("rank") or 999999),
                ) > (
                    _safe_float_or_none(best.get("shadow_score")) or 0.0,
                    _safe_float_or_none(best.get("pnl_net_pct")) or 0.0,
                    -int(best.get("rank") or 999999),
                ):
                    best = dict(candidate)
                    best_selected = dict(selected_candidate or {})
                    best_decision = dict(decision)
        if not best:
            return {}

        selected_pnl = _safe_float_or_none(best_selected.get("pnl_net_pct")) or 0.0
        candidate_pnl = _safe_float_or_none(best.get("pnl_net_pct")) or 0.0
        timestamp = causal.get("timestamp")
        if isinstance(timestamp, dict):
            timestamp_utc = str(timestamp.get("utc") or timestamp.get("timestamp_utc") or "")
        else:
            timestamp_utc = str(timestamp or "")
        return {
            "bar": int(causal.get("bar") or step.bar or 0),
            "timestamp_utc": timestamp_utc,
            "regime": str(causal.get("regime") or step.regime.label),
            "symbol": str(best_decision.get("symbol") or best.get("symbol") or ""),
            "top_shadow_actor": str(best.get("label") or ""),
            "actor_type": str(best.get("actor_type") or ""),
            "action": str(best.get("action") or ""),
            "rejection_reason": str(best.get("reason") or ""),
            "shadow_score": _safe_float_or_none(best.get("shadow_score")) or 0.0,
            "shadow_closed_trades": int(best.get("shadow_closed_trades") or 0),
            "selected_actor": str(best_decision.get("selected_actor") or ""),
            "selected_actor_pnl_pct": selected_pnl,
            "top_shadow_pnl_pct": candidate_pnl,
            "estimated_lost_pnl_pct": max(0.0, candidate_pnl - selected_pnl),
        }

    @staticmethod
    def _selected_candidate_from_flash_decision(
        candidates: List[Any],
        *,
        selected_actor: str,
        selected_type: str,
    ) -> Dict[str, Any]:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            if str(candidate.get("label") or "") != selected_actor:
                continue
            if selected_type and str(candidate.get("actor_type") or "") != selected_type:
                continue
            return dict(candidate)
        return {}

    @staticmethod
    def _candidate_action_is_open(candidate: Dict[str, Any]) -> bool:
        action = str(candidate.get("action") or "").upper()
        return action.startswith("FUT_LONG") or action.startswith("FUT_SHORT")

    def _compact_causal_entry_decision(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        decisions = out.get("flash_decisions")
        if isinstance(decisions, list):
            if self._config.compact_causal_entry_selected_only:
                decisions = [
                    decision
                    for decision in decisions
                    if _flash_decision_has_selected_signal(decision)
                ]
            out["flash_decisions"] = [
                self._compact_flash_decision(
                    decision,
                    top_rejected_candidates=max(
                        0,
                        int(self._config.compact_causal_entry_top_rejected_candidates),
                    ),
                    include_labels=self._config.compact_causal_entry_include_labels,
                )
                for decision in decisions
            ]
        return out

    @staticmethod
    def _has_soft_allocator_execution(row: Dict[str, Any]) -> bool:
        soft = row.get("soft_allocator")
        if not isinstance(soft, dict) or not bool(soft.get("enabled")):
            return False
        for key in ("executable_signals", "guarded_signals", "raw_signals"):
            value = row.get(key)
            if isinstance(value, list) and value:
                return True
        for key in ("executable_signal_count", "raw_signal_count"):
            try:
                if int(row.get(key) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _compact_flash_decision(
        decision: Any,
        *,
        top_rejected_candidates: int = 5,
        include_labels: tuple[str, ...] = (),
    ) -> Any:
        if not isinstance(decision, dict):
            return decision
        out = dict(decision)
        candidates = decision.get("candidates")
        if not isinstance(candidates, list):
            return out

        rejected_candidates = [
            dict(candidate)
            for candidate in candidates
            if isinstance(candidate, dict) and bool(candidate.get("rejected"))
        ]
        rejection_counts = Counter(
            str(candidate.get("reason") or "")
            for candidate in rejected_candidates
        )
        out["candidate_count"] = len(candidates)
        out["rejected_candidate_count"] = len(rejected_candidates)
        out["eligible_candidate_count"] = len(candidates) - len(rejected_candidates)
        out["candidate_rejection_counts"] = dict(sorted(rejection_counts.items()))
        included_rejected_candidates = OutputWriter._compact_rejected_candidates(
            rejected_candidates,
            top_rejected_candidates=top_rejected_candidates,
            include_labels=include_labels,
        )
        out["top_rejected_candidates"] = [
            OutputWriter._compact_candidate_audit(candidate)
            for candidate in included_rejected_candidates
        ]

        selected_actor = str(decision.get("selected_actor") or "")
        actor_type = str(decision.get("actor_type") or "")
        if selected_actor == "NoTrade" and not decision.get("signal"):
            out["candidates"] = []
            return out
        original_selected_actor = str(decision.get("original_selected_actor") or "")
        original_actor_type = str(decision.get("original_actor_type") or "")
        compact_candidates: List[Dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            label = str(candidate.get("label") or "")
            candidate_type = str(candidate.get("actor_type") or "")
            is_selected = (
                label == selected_actor
                and (not actor_type or candidate_type == actor_type)
            )
            is_original_selected = bool(original_selected_actor) and (
                label == original_selected_actor
                and (not original_actor_type or candidate_type == original_actor_type)
            )
            is_diagnostic_include = OutputWriter._candidate_matches_include_labels(
                candidate,
                include_labels,
            )
            if not (is_selected or is_original_selected or is_diagnostic_include):
                continue
            key = (label, candidate_type)
            if key in seen:
                continue
            seen.add(key)
            compact_candidates.append(
                OutputWriter._compact_candidate_audit(dict(candidate))
            )

        out["candidates"] = compact_candidates
        return out

    @staticmethod
    def _compact_rejected_candidates(
        rejected_candidates: List[Dict[str, Any]],
        *,
        top_rejected_candidates: int,
        include_labels: tuple[str, ...],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            rejected_candidates,
            key=lambda item: int(item.get("rank") or 0),
        )
        selected = list(ranked[: max(0, int(top_rejected_candidates))])
        seen = {
            (
                str(candidate.get("label") or ""),
                str(candidate.get("actor_type") or ""),
                str(candidate.get("actor_key") or ""),
            )
            for candidate in selected
        }
        for candidate in ranked:
            if not OutputWriter._candidate_matches_include_labels(
                candidate,
                include_labels,
            ):
                continue
            key = (
                str(candidate.get("label") or ""),
                str(candidate.get("actor_type") or ""),
                str(candidate.get("actor_key") or ""),
            )
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
        return selected

    @staticmethod
    def _candidate_matches_include_labels(
        candidate: Dict[str, Any],
        include_labels: tuple[str, ...],
    ) -> bool:
        labels = {
            str(candidate.get("label") or "").strip().lower(),
            str(candidate.get("actor_key") or "").strip().lower(),
        }
        labels.discard("")
        for raw_label in include_labels or ():
            label = str(raw_label or "").strip().lower()
            if label and label in labels:
                return True
        return False

    @staticmethod
    def _is_persistable_policy_decision(row: Dict[str, Any]) -> bool:
        if str(row.get("decision_path") or "") != "policy_v1":
            return False
        if row.get("policy_bar") is not None:
            return True
        return str(row.get("status") or "") in {"safety_exit", "error"}

    @staticmethod
    def _compact_candidate_audit(candidate: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "label",
            "actor_type",
            "actor_key",
            "action",
            "rank",
            "score",
            "base_score",
            "effective_score",
            "gate_score",
            "reason",
            "shadow_score",
            "shadow_closed_trades",
            "shadow_win_rate_pct",
            "shadow_recent_downside_usd",
            "shadow_pnl_per_trade_lcb_usd",
            "shadow_pnl_per_trade_lcb_penalty",
            "shadow_symbol_health_score",
            "shadow_symbol_health_closed_trades",
            "shadow_symbol_health_pnl_per_trade_lcb_usd",
            "shadow_symbol_health_penalty",
            "selected_subset_score_boost",
            "selected_subset_protected",
            "selected_subset_risk_mult",
            "risk_mult",
            "lookback_return_z",
            "agent_diagnostics",
        )
        return {key: candidate[key] for key in keys if key in candidate}

    def _write_status(
        self,
        step: Optional[StepResult],
        *,
        run_state: str,
        feed_status: str,
        message: str = "",
    ) -> None:
        """status.json — аналог v1."""
        # Реплеим ledger чтобы получить актуальный PnL
        try:
            self._pipeline.ledger.replay_from_event_log(self._pipeline.event_log)
        except Exception:
            pass

        realized_pnl = self._pipeline.ledger.total_realized_pnl
        account_snapshot = self._refresh_account_view(realized_pnl=realized_pnl)
        total_assets = float(account_snapshot.get("total_assets", self._current_balance) or self._current_balance)
        timestamps = _utc_msk_timestamps()

        pnl_pct = (
            (total_assets - self._initial_capital) / self._initial_capital * 100.0
            if self._initial_capital > 0 else 0.0
        )
        self._assets_history = self._append_curve_value(self._assets_history, total_assets)
        self._equity_history_timestamps = self._append_curve_timestamp(
            self._equity_history_timestamps,
            timestamps["timestamp_utc"],
        )

        # Открытые позиции
        uptime_sec = time.time() - self._start_time
        h = int(uptime_sec // 3600)
        m = int((uptime_sec % 3600) // 60)
        s = int(uptime_sec % 60)
        bar = step.bar if step is not None else 0
        regime = step.regime.label if step is not None else "unknown"
        market_view = self._market_view_payload(step, fallback_regime=regime)
        current_prices = self._append_price_history(
            step,
            regime=regime,
            market_view=market_view,
        )
        open_positions = self._tracked_open_positions(current_prices=current_prices)
        position_counts = self._position_counts(tracked_count=len(open_positions))
        n_signals = step.n_signals if step is not None else 0
        n_filled = step.n_filled if step is not None else 0
        n_rejected = step.n_rejected if step is not None else 0
        n_blocked = step.n_blocked if step is not None else 0
        n_filtered = step.n_filtered_real_signals if step is not None else 0
        n_stale_closes = step.n_stale_close_signals if step is not None else 0
        n_duplicate_opens = step.n_duplicate_open_signals if step is not None else 0
        n_rate_limited_opens = step.n_rate_limited_open_signals if step is not None else 0
        n_max_position_saturated_opens = step.n_max_position_saturated_open_signals if step is not None else 0
        n_external_position_signals = step.n_external_position_signals if step is not None else 0
        blocked_reasons = dict(step.blocked_reasons) if step is not None else {}
        raw_leader = step.leader if step is not None else None
        selected_leader = (
            step.selected_leader if step is not None and step.selected_leader else raw_leader
        )
        executed_leader = (
            step.executed_leader if step is not None and step.executed_leader else raw_leader
        )
        leader = selected_leader
        leader_session = self._session_metrics_for(leader) if leader else {
            "pnl_pct": 0.0,
            "closed_trades": 0,
            "win_rate": 0.0,
        }
        position_pnl = self._position_pnl_summary(
            leader=leader,
            current_prices=current_prices,
        )
        player_pnl = self._pipeline.ledger.total_pnl_by_player()
        external_realized_pnl = sum(
            pnl for player, pnl in player_pnl.items()
            if _is_external_player(player)
        )
        adopted_realized_pnl = sum(
            pnl for player, pnl in player_pnl.items()
            if _is_adopted_player(player, self._pipeline)
        )
        clean_panteon_realized_pnl = sum(
            pnl for player, pnl in player_pnl.items()
            if (
                not _is_external_player(player)
                and not _is_adopted_player(player, self._pipeline)
            )
        )
        panteon_owned_realized_pnl = sum(
            pnl for player, pnl in player_pnl.items()
            if not _is_external_player(player)
        )
        leader_realized_pnl = player_pnl.get(leader, 0.0) if leader else 0.0
        executed_leader_realized_pnl = (
            player_pnl.get(executed_leader, 0.0) if executed_leader else 0.0
        )
        panteon_owned_total_pnl = (
            panteon_owned_realized_pnl
            + position_pnl["panteon_owned_unrealized_pnl_usd"]
        )
        clean_panteon_total_pnl = (
            clean_panteon_realized_pnl
            + position_pnl["clean_panteon_unrealized_pnl_usd"]
        )
        panteon_realized_equity = self._initial_capital + panteon_owned_realized_pnl
        panteon_equity = self._initial_capital + panteon_owned_total_pnl
        clean_panteon_realized_equity = (
            self._initial_capital + clean_panteon_realized_pnl
        )
        clean_panteon_equity = self._initial_capital + clean_panteon_total_pnl
        self._panteon_realized_equity_history = self._append_curve_value(
            self._panteon_realized_equity_history,
            panteon_realized_equity,
        )
        self._panteon_equity_history = self._append_curve_value(
            self._panteon_equity_history,
            panteon_equity,
        )
        self._clean_panteon_realized_equity_history = self._append_curve_value(
            self._clean_panteon_realized_equity_history,
            clean_panteon_realized_equity,
        )
        self._clean_panteon_equity_history = self._append_curve_value(
            self._clean_panteon_equity_history,
            clean_panteon_equity,
        )
        panteon_realized_max_dd_pct = self._curve_max_drawdown_pct(
            self._panteon_realized_equity_history
        )
        panteon_max_dd_pct = self._curve_max_drawdown_pct(self._panteon_equity_history)
        clean_panteon_realized_max_dd_pct = self._curve_max_drawdown_pct(
            self._clean_panteon_realized_equity_history
        )
        clean_panteon_max_dd_pct = self._curve_max_drawdown_pct(
            self._clean_panteon_equity_history
        )
        panteon_owned_pnl_pct = (
            panteon_owned_total_pnl / self._initial_capital * 100.0
            if self._initial_capital > 0 else 0.0
        )
        clean_panteon_pnl_pct = (
            clean_panteon_total_pnl / self._initial_capital * 100.0
            if self._initial_capital > 0 else 0.0
        )
        real_trades = self._real_trade_summary(position_pnl=position_pnl)
        real_trading_actors = self._real_actor_payload(
            open_positions=open_positions,
            position_counts=position_counts,
        )
        live_session = {
            "account_equity_pnl_usd": total_assets - self._initial_capital,
            "account_realized_pnl_usd": realized_pnl,
            "account_pnl_pct": pnl_pct,
            "panteon_equity_pnl_usd": panteon_owned_total_pnl,
            "panteon_realized_pnl_usd": panteon_owned_realized_pnl,
            "panteon_pnl_pct": panteon_owned_pnl_pct,
            "panteon_equity_usd": panteon_equity,
            "panteon_realized_equity_usd": panteon_realized_equity,
            "panteon_max_drawdown_pct": panteon_max_dd_pct,
            "panteon_realized_max_drawdown_pct": panteon_realized_max_dd_pct,
            "clean_panteon_equity_pnl_usd": clean_panteon_total_pnl,
            "clean_panteon_realized_pnl_usd": clean_panteon_realized_pnl,
            "clean_panteon_unrealized_pnl_usd": position_pnl["clean_panteon_unrealized_pnl_usd"],
            "clean_panteon_total_pnl_usd": clean_panteon_total_pnl,
            "clean_panteon_pnl_pct": clean_panteon_pnl_pct,
            "clean_panteon_equity_usd": clean_panteon_equity,
            "clean_panteon_realized_equity_usd": clean_panteon_realized_equity,
            "clean_panteon_max_drawdown_pct": clean_panteon_max_dd_pct,
            "clean_panteon_realized_max_drawdown_pct": clean_panteon_realized_max_dd_pct,
            "clean_panteon_positions_count": position_pnl["clean_panteon_positions_count"],
            "panteon_owned_realized_pnl_usd": panteon_owned_realized_pnl,
            "panteon_owned_unrealized_pnl_usd": position_pnl["panteon_owned_unrealized_pnl_usd"],
            "panteon_owned_total_pnl_usd": panteon_owned_total_pnl,
            "panteon_owned_pnl_pct": panteon_owned_pnl_pct,
            "panteon_owned_positions_count": position_pnl["panteon_owned_positions_count"],
            "external_realized_pnl_usd": external_realized_pnl,
            "external_unrealized_pnl_usd": position_pnl["external_unrealized_pnl_usd"],
            "external_positions_count": position_pnl["external_positions_count"],
            "adopted_realized_pnl_usd": adopted_realized_pnl,
            "adopted_unrealized_pnl_usd": position_pnl["adopted_unrealized_pnl_usd"],
            "adopted_positions_count": position_pnl["adopted_positions_count"],
            "leader_realized_pnl_usd": leader_realized_pnl,
            "leader_unrealized_pnl_usd": position_pnl["leader_unrealized_pnl_usd"],
            "leader_positions_count": position_pnl["leader_positions_count"],
            "handoff_positions_count": position_pnl["handoff_positions_count"],
            "comparison_scope": "panteon_owned",
            "profit_accounting_scope": "clean_panteon_owned",
            "leader": leader,
            "selected_leader": selected_leader,
            "executed_leader": executed_leader,
            "leader_virtual_session_pnl_pct": leader_session["pnl_pct"],
            "leader_virtual_session_closed_trades": leader_session["closed_trades"],
            "leader_virtual_session_win_rate": leader_session["win_rate"],
            "executed_leader_realized_pnl_usd": executed_leader_realized_pnl,
            "real_total_trades": real_trades["total"],
            "real_closed_trades": real_trades["closed"],
            "real_successful_trades": real_trades["successful"],
            "real_unsuccessful_trades": real_trades["unsuccessful"],
            "real_unresolved_trades": real_trades["unresolved"],
            "clean_real_total_trades": real_trades["clean_total"],
            "clean_real_closed_trades": real_trades["clean_closed"],
            "clean_real_successful_trades": real_trades["clean_successful"],
            "clean_real_unsuccessful_trades": real_trades["clean_unsuccessful"],
            "clean_real_unresolved_trades": real_trades["clean_unresolved"],
            "external_closed_trades": real_trades["external_closed"],
            "external_unresolved_trades": real_trades["external_unresolved"],
            "adopted_closed_trades": real_trades["adopted_closed"],
            "adopted_unresolved_trades": real_trades["adopted_unresolved"],
        }
        shadow_bar = {
            "actors": step.n_shadow_actors if step is not None else 0,
            "signals": step.n_shadow_signals if step is not None else 0,
            "filled": step.n_shadow_filled if step is not None else 0,
            "rejected": step.n_shadow_rejected if step is not None else 0,
            "blocked": step.n_shadow_blocked if step is not None else 0,
        }
        shadow_session = dict(self._shadow_session_counts)
        shadow = {
            "actors": shadow_bar["actors"],
            "signals": shadow_session["signals"],
            "filled": shadow_session["filled"],
            "rejected": shadow_session["rejected"],
            "blocked": shadow_session["blocked"],
            "signals_bar": shadow_bar["signals"],
            "filled_bar": shadow_bar["filled"],
            "rejected_bar": shadow_bar["rejected"],
            "blocked_bar": shadow_bar["blocked"],
            "session": shadow_session,
            "last_bar": shadow_bar,
            "last_summary": getattr(self._pipeline, "shadow_last_summary", None),
        }
        flash = self._flash_status_payload(step)
        policy = self._policy_status_payload(step)
        decision_debug = self._decision_debug(
            step=step,
            shadow_bar=shadow_bar,
            shadow_session=shadow_session,
            blocked_reasons=blocked_reasons,
        )
        data = {
            "version":          "v2",
            "timestamp":        timestamps["timestamp_utc"],
            "timestamp_utc":    timestamps["timestamp_utc"],
            "timestamp_msk":    timestamps["timestamp_msk"],
            "active_pid":       os.getpid(),
            "exchange":         self._pipeline.exchange_name,
            "run_id":           getattr(self._pipeline, "run_id", ""),
            "session_id":       getattr(self._pipeline, "session_id", ""),
            "mode":             getattr(self._pipeline, "mode", ""),
            "trade_mode":       getattr(self._pipeline, "trade_mode", "multi"),
            "fixed_player_label": getattr(self._pipeline, "fixed_player_label", ""),
            "player_only_runtime": bool(
                getattr(self._pipeline, "player_only_runtime", False)
            ),
            "timeframe":        getattr(self._pipeline, "timeframe", ""),
            "run_state":        run_state,
            "feed_status":      feed_status,
            "message":          message,
            "uptime":           f"{h:02d}:{m:02d}:{s:02d}",
            "bar_count":        bar,
            "market":           market_view["market"],
            "regime":           regime,
            "regime_confidence": market_view.get("regime_confidence"),
            "regimes_by_symbol": market_view.get("regimes_by_symbol", {}),
            "regime_features_by_symbol": market_view.get(
                "regime_features_by_symbol",
                {},
            ),
            "market_view":      market_view,
            "current_prices":    current_prices,
            "price_history":     list(self._price_history),
            "initial_capital":  self._initial_capital,
            "current_balance":  self._current_balance,
            "futures_equity_usd": account_snapshot.get("futures_equity", self._current_balance),
            "available_balance_usd": account_snapshot.get("available_balance", self._current_balance),
            "spot_assets_usd": account_snapshot.get("spot_assets", 0.0),
            "total_assets_usd": total_assets,
            "assets_curve":     list(self._assets_history),
            "assets_curve_timestamps": list(self._equity_history_timestamps),
            "equity_curve":     list(self._panteon_equity_history),
            "equity_curve_timestamps": list(self._equity_history_timestamps),
            "panteon_equity_usd": panteon_equity,
            "panteon_realized_equity_usd": panteon_realized_equity,
            "panteon_equity_curve": list(self._panteon_equity_history),
            "panteon_equity_curve_timestamps": list(self._equity_history_timestamps),
            "panteon_realized_equity_curve": list(self._panteon_realized_equity_history),
            "panteon_realized_equity_curve_timestamps": list(
                self._equity_history_timestamps
            ),
            "panteon_max_drawdown_pct": panteon_max_dd_pct,
            "panteon_realized_max_drawdown_pct": panteon_realized_max_dd_pct,
            "clean_panteon_equity_usd": clean_panteon_equity,
            "clean_panteon_realized_equity_usd": clean_panteon_realized_equity,
            "clean_panteon_equity_curve": list(self._clean_panteon_equity_history),
            "clean_panteon_equity_curve_timestamps": list(
                self._equity_history_timestamps
            ),
            "clean_panteon_realized_equity_curve": list(
                self._clean_panteon_realized_equity_history
            ),
            "clean_panteon_realized_equity_curve_timestamps": list(
                self._equity_history_timestamps
            ),
            "clean_panteon_max_drawdown_pct": clean_panteon_max_dd_pct,
            "clean_panteon_realized_max_drawdown_pct": clean_panteon_realized_max_dd_pct,
            "unrealized_pnl_usd": account_snapshot.get("unrealized_pnl", 0.0),
            "pnl_usd":          realized_pnl,
            "pnl_pct":          pnl_pct,
            "n_signals_bar":    n_signals,
            "n_filled_bar":     n_filled,
            "n_rejected_bar":   n_rejected,
            "n_blocked_bar":    n_blocked,
            "n_filtered_real_signals_bar": n_filtered,
            "n_stale_close_signals_bar": n_stale_closes,
            "n_duplicate_open_signals_bar": n_duplicate_opens,
            "n_rate_limited_open_signals_bar": n_rate_limited_opens,
            "n_max_position_saturated_open_signals_bar": n_max_position_saturated_opens,
            "n_external_position_signals_bar": n_external_position_signals,
            "blocked_reasons_bar": blocked_reasons,
            "n_positions":      position_counts["n_positions"],
            "tracked_positions_count": position_counts["tracked_positions_count"],
            "exchange_positions_count": position_counts["exchange_positions_count"],
            "current_leader":   leader,
            "selected_leader":  selected_leader,
            "executed_leader":  executed_leader,
            "live_session":     live_session,
            "real_trades":      real_trades,
            "real_trading_actors": real_trading_actors,
            "configured_actor_pool": self._configured_actor_pool_payload(),
            "live_state_sync":  getattr(self._pipeline, "live_state_sync", None),
            "paper_canary_shutdown_flatten": getattr(
                self._pipeline,
                "paper_canary_shutdown_flatten",
                {},
            ),
            "shadow":           shadow,
            "flash":            flash,
            "policy":           policy,
            "decision_debug":   decision_debug,
            "quarantined":      sorted(self._pipeline.qm.all_quarantined()),
            "quarantine_records": self._quarantine_records(),
            "degradation_gate":  self._degradation_gate_snapshot(),
            "open_positions":   open_positions,
            "ledger_closed_count": self._pipeline.ledger.closed_count,
        }
        self._last_status_data = data
        path = os.path.join(self._config.output_dir, self._config.status_filename)
        self._write_json_atomic(path, data)

    def _accumulate_shadow_counts(self, step: StepResult) -> None:
        self._shadow_session_counts["signals"] += int(step.n_shadow_signals or 0)
        self._shadow_session_counts["filled"] += int(step.n_shadow_filled or 0)
        self._shadow_session_counts["rejected"] += int(step.n_shadow_rejected or 0)
        self._shadow_session_counts["blocked"] += int(step.n_shadow_blocked or 0)

    def _accumulate_fallback_counts(self, step: StepResult) -> None:
        if step.fallback_used:
            self._fallback_session_counts["used"] += 1
        if step.fallback_skipped:
            self._fallback_session_counts["skipped"] += 1

    def _decision_debug(
        self,
        *,
        step: Optional[StepResult],
        shadow_bar: Dict[str, int],
        shadow_session: Dict[str, int],
        blocked_reasons: Dict[str, int],
    ) -> Dict[str, object]:
        kill_switch = getattr(self._pipeline, "kill_switch", None)
        debug: Dict[str, object] = {
            "kill_switch_reason": str(getattr(kill_switch, "disabled_reason", "") or ""),
            "shadow_last_bar": dict(shadow_bar),
            "shadow_session": dict(shadow_session),
        }
        if step is None:
            policy_payload = self._policy_status_payload(None)
            debug.update({
                "leader": "",
                "selected_leader": "",
                "executed_leader": "",
                "leader_execution_mismatch": False,
                "leader_agents": [],
                "leader_raw_signals_bar": 0,
                "leader_vote_errors_bar": 0,
                "real_signals_after_guard_bar": 0,
                "filtered_real_signals_bar": 0,
                "fallback_used": False,
                "fallback_skipped": False,
                "fallback_candidate": "",
                "fallback_reason": "",
                "fallback_session": dict(self._fallback_session_counts),
                "signal_filter_details": [],
                "blocked_reasons_bar": {},
                "flash_enabled": bool(getattr(self._pipeline, "flash_enabled", False)),
                "flash_selected_actors_by_symbol": {},
                "flash_actor_types_by_symbol": {},
                "flash_scores_by_symbol": {},
                "flash_decisions_count": 0,
                "policy_enabled": bool(policy_payload.get("enabled", False)),
                "policy_id": str(policy_payload.get("policy_id") or ""),
                "policy_manifest_sha256": str(
                    policy_payload.get("manifest_sha256") or ""
                ),
                "policy_status": str(policy_payload.get("status") or ""),
                "policy_reason": str(policy_payload.get("reason") or ""),
                "policy_candidate_count": 0,
                "policy_decision_count": 0,
                "step_error": "",
            })
            return debug
        selected_leader = step.selected_leader or step.leader or ""
        executed_leader = step.executed_leader or step.leader or ""
        flash_payload = self._flash_status_payload(step)
        policy_payload = self._policy_status_payload(step)
        debug.update({
            "leader": step.leader or "",
            "selected_leader": selected_leader,
            "executed_leader": executed_leader,
            "leader_execution_mismatch": (
                bool(selected_leader)
                and bool(executed_leader)
                and selected_leader != executed_leader
            ),
            "leader_agents": list(step.leader_agent_labels),
            "leader_raw_signals_bar": int(step.n_raw_signals or 0),
            "leader_vote_errors_bar": int(step.leader_vote_errors or 0),
            "real_signals_after_guard_bar": int(step.n_signals or 0),
            "filled_real_signals_bar": int(step.n_filled or 0),
            "rejected_real_signals_bar": int(step.n_rejected or 0),
            "blocked_real_signals_bar": int(step.n_blocked or 0),
            "filtered_real_signals_bar": int(step.n_filtered_real_signals or 0),
            "stale_close_signals_bar": int(step.n_stale_close_signals or 0),
            "duplicate_open_signals_bar": int(step.n_duplicate_open_signals or 0),
            "rate_limited_open_signals_bar": int(step.n_rate_limited_open_signals or 0),
            "max_position_saturated_open_signals_bar": int(
                step.n_max_position_saturated_open_signals or 0
            ),
            "external_position_signals_bar": int(step.n_external_position_signals or 0),
            "fallback_used": bool(step.fallback_used),
            "fallback_skipped": bool(step.fallback_skipped),
            "fallback_candidate": step.fallback_candidate or "",
            "fallback_reason": step.fallback_reason or "",
            "fallback_session": dict(self._fallback_session_counts),
            "signal_filter_details": list(step.signal_filter_details[:20]),
            "blocked_reasons_bar": dict(blocked_reasons),
            "flash_enabled": bool(flash_payload.get("enabled", False)),
            "flash_selected_actors_by_symbol": dict(
                flash_payload.get("selected_actors_by_symbol", {}) or {}
            ),
            "flash_actor_types_by_symbol": dict(
                flash_payload.get("actor_types_by_symbol", {}) or {}
            ),
            "flash_scores_by_symbol": dict(
                flash_payload.get("scores_by_symbol", {}) or {}
            ),
            "flash_decisions_count": len(flash_payload.get("decisions", []) or []),
            "policy_enabled": bool(policy_payload.get("enabled", False)),
            "policy_id": str(policy_payload.get("policy_id") or ""),
            "policy_manifest_sha256": str(
                policy_payload.get("manifest_sha256") or ""
            ),
            "policy_status": str(policy_payload.get("status") or ""),
            "policy_reason": str(policy_payload.get("reason") or ""),
            "policy_candidate_count": int(
                policy_payload.get("candidate_count") or 0
            ),
            "policy_decision_count": int(
                policy_payload.get("decision_count") or 0
            ),
            "step_error": step.error or "",
        })
        return debug

    def _flash_status_payload(self, step: Optional[StepResult]) -> Dict[str, object]:
        causal = dict(getattr(step, "causal_decision", {}) or {}) if step is not None else {}
        if str(causal.get("decision_path") or "") == "policy_v1":
            return {
                "enabled": False,
                "selected_actors_by_symbol": {},
                "actor_types_by_symbol": {},
                "scores_by_symbol": {},
                "decisions": [],
            }
        selected = dict(getattr(step, "selected_actors_by_symbol", {}) or {}) if step is not None else {}
        if not selected:
            selected = dict(causal.get("flash_selected_actors_by_symbol") or {})
        decisions = list(causal.get("flash_decisions") or [])
        return {
            "enabled": bool(
                getattr(self._pipeline, "flash_enabled", False)
                or causal.get("flash_enabled")
                or selected
            ),
            "selected_actors_by_symbol": selected,
            "actor_types_by_symbol": dict(causal.get("flash_actor_types_by_symbol") or {}),
            "scores_by_symbol": dict(causal.get("flash_scores_by_symbol") or {}),
            "decisions": decisions,
        }

    def _policy_status_payload(self, step: Optional[StepResult]) -> Dict[str, object]:
        runtime = getattr(self._pipeline, "policy_runtime_v1", None)
        causal = (
            dict(getattr(step, "causal_decision", {}) or {})
            if step is not None
            else {}
        )
        manifest = getattr(runtime, "manifest", None)
        loaded = getattr(runtime, "loaded_manifest", None)
        enabled = runtime is not None
        target = getattr(getattr(manifest, "target", None), "value", "")
        payload: Dict[str, object] = {
            "enabled": enabled,
            "decision_path": "policy_v1" if enabled else "",
            "policy_id": str(getattr(manifest, "policy_id", "") or ""),
            "manifest_sha256": str(
                getattr(manifest, "manifest_sha256", "") or ""
            ),
            "manifest_path": str(getattr(loaded, "path", "") or ""),
            "target": str(target or ""),
            "actor": str(getattr(manifest, "actor", "") or ""),
            "status": "configured" if enabled else "disabled",
            "reason": "",
            "policy_bar": None,
            "cadence_timestamp": None,
            "candidate_count": 0,
            "denied_count": 0,
            "decision_count": 0,
            "market_quality": [],
            "warmup": dict(getattr(self._pipeline, "policy_warmup", {}) or {}),
            "risk_state": dict(getattr(runtime, "risk_status", {}) or {}),
        }
        if str(causal.get("decision_path") or "") != "policy_v1":
            return payload
        payload.update({
            "policy_id": str(causal.get("policy_id") or payload["policy_id"]),
            "manifest_sha256": str(
                causal.get("manifest_sha256") or payload["manifest_sha256"]
            ),
            "manifest_path": str(
                causal.get("manifest_path") or payload["manifest_path"]
            ),
            "target": str(causal.get("target") or payload["target"]),
            "actor": str(causal.get("actor") or payload["actor"]),
            "status": str(causal.get("status") or ""),
            "reason": str(causal.get("reason") or ""),
            "policy_bar": causal.get("policy_bar"),
            "cadence_timestamp": causal.get("cadence_timestamp"),
            "candidate_count": int(causal.get("candidate_count") or 0),
            "denied_count": int(causal.get("policy_denied_count") or 0),
            "decision_count": len(causal.get("policy_decisions") or []),
            "market_quality": list(causal.get("market_quality") or []),
        })
        return payload

    def _market_view_payload(
        self,
        step: Optional[StepResult],
        *,
        fallback_regime: str,
    ) -> Dict[str, object]:
        causal = (
            dict(getattr(step, "causal_decision", {}) or {})
            if step is not None
            else {}
        )
        market = str(
            causal.get("market")
            or causal.get("regime")
            or fallback_regime
            or "unknown"
        ).strip().lower()
        confidence = _safe_float_or_none(causal.get("regime_confidence"))
        funding = causal.get("funding")
        funding_values: List[float] = []
        if isinstance(funding, dict):
            for raw_value in funding.values():
                parsed = _safe_float_or_none(raw_value)
                if parsed is not None:
                    funding_values.append(parsed)
        avg_funding = (
            sum(funding_values) / len(funding_values)
            if funding_values
            else None
        )
        if avg_funding is None:
            funding_bias = "unknown"
        elif avg_funding > 0:
            funding_bias = "positive"
        elif avg_funding < 0:
            funding_bias = "negative"
        else:
            funding_bias = "neutral"
        return {
            "market": market,
            "regime": str(fallback_regime or "unknown").lower(),
            "regime_confidence": confidence,
            "regimes_by_symbol": _clean_regimes_by_symbol(
                causal.get("regimes_by_symbol")
            ),
            "regime_features_by_symbol": _clean_regime_features_by_symbol(
                causal.get("regime_features_by_symbol")
            ),
            "funding_symbols": len(funding_values),
            "avg_funding": avg_funding,
            "funding_bias": funding_bias,
        }

    def _refresh_account_view(self, *, realized_pnl: float = 0.0) -> Dict[str, float]:
        try:
            sync_pipeline_balance(self._pipeline)
        except Exception:
            log.debug("account sync failed in OutputWriter", exc_info=True)

        snapshot = dict(getattr(self._pipeline, "account_snapshot", None) or {})
        fallback_balance = float(getattr(self._pipeline, "current_balance", 0.0) or 0.0)
        if fallback_balance <= 0:
            fallback_balance = self._initial_capital + float(realized_pnl or 0.0)
        current_balance = _first_positive(
            snapshot.get("current_balance"),
            snapshot.get("futures_equity"),
            fallback_balance,
        )
        total_assets = _first_positive(
            snapshot.get("total_assets"),
            current_balance + float(snapshot.get("spot_assets", 0.0) or 0.0),
            current_balance,
        )
        snapshot.setdefault("current_balance", current_balance)
        snapshot.setdefault("futures_equity", current_balance)
        snapshot.setdefault("available_balance", current_balance)
        snapshot.setdefault("spot_assets", 0.0)
        snapshot.setdefault("total_assets", total_assets)
        snapshot.setdefault("unrealized_pnl", 0.0)
        self._current_balance = current_balance
        self._pipeline.current_balance = current_balance
        self._pipeline.account_snapshot = snapshot
        return snapshot

    def _tracked_open_positions(
        self,
        *,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, dict]:
        open_positions: Dict[str, dict] = {}
        tracker = getattr(getattr(self._pipeline, "executor", None), "_tracker", None)
        if tracker is None or not hasattr(tracker, "all_open"):
            return open_positions
        try:
            tracked = tracker.all_open()
        except Exception:
            return open_positions
        exchange_positions = self._exchange_positions()
        for sym, pos in tracked.items():
            exchange_pos = exchange_positions.get(str(sym).upper())
            unrealized = _position_unrealized_pnl(
                pos,
                exchange_pos,
                current_prices=current_prices,
            )
            open_positions[sym] = {
                "side":  pos.side,
                "qty":   pos.qty,
                "entry": pos.entry_price,
                "by_player": pos.by_player,
                "by_agent":  pos.by_agent,
                "unrealized_pnl_usd": unrealized,
                "external": _is_external_player(pos.by_player),
            }
        return open_positions

    @staticmethod
    def _append_curve_value(history: List[float], value: float, *, limit: int = 720) -> List[float]:
        out = list(history)
        out.append(float(value))
        if len(out) > limit:
            out = out[-limit:]
        return out

    @staticmethod
    def _append_curve_timestamp(history: List[str], value: str, *, limit: int = 720) -> List[str]:
        out = list(history)
        timestamp = str(value or "").strip()
        if timestamp:
            out.append(timestamp)
        if len(out) > limit:
            out = out[-limit:]
        return out

    def _append_price_history(
        self,
        step: Optional[StepResult],
        *,
        regime: str,
        market_view: Dict[str, object],
        limit: int = 720,
    ) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        if step is not None:
            causal = getattr(step, "causal_decision", {}) or {}
            if isinstance(causal, dict):
                prices = self._clean_price_mapping(causal.get("prices"))
        if prices:
            timestamp = ""
            if step is not None:
                causal = getattr(step, "causal_decision", {}) or {}
                if isinstance(causal, dict):
                    timestamp = _coerce_timestamp_text(
                        causal.get("timestamp")
                        or causal.get("timestamp_utc")
                        or causal.get("time")
                    )
            if not timestamp:
                timestamp = datetime.now(timezone.utc).isoformat()
            self._price_history.append({
                "bar": int(step.bar if step is not None else 0),
                "timestamp": timestamp,
                "regime": str(regime or ""),
                "market": str(market_view.get("market", regime) or regime or ""),
                "regime_confidence": market_view.get("regime_confidence"),
                "regimes_by_symbol": dict(
                    market_view.get("regimes_by_symbol", {}) or {}
                ),
                "regime_features_by_symbol": dict(
                    market_view.get("regime_features_by_symbol", {}) or {}
                ),
                "prices": prices,
            })
            if len(self._price_history) > limit:
                self._price_history = self._price_history[-limit:]
        if self._price_history:
            last = self._price_history[-1].get("prices")
            if isinstance(last, dict):
                return {
                    str(symbol): float(price)
                    for symbol, price in last.items()
                    if _safe_float_or_none(price) is not None
                }
        return {}

    @staticmethod
    def _clean_price_mapping(raw: object) -> Dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, float] = {}
        for raw_symbol, raw_price in raw.items():
            symbol = str(raw_symbol or "").strip().upper()
            price = _safe_float_or_none(raw_price)
            if symbol and price is not None and price > 0.0:
                out[symbol] = float(price)
        return out

    @staticmethod
    def _curve_max_drawdown_pct(curve: List[float]) -> float:
        peak = 0.0
        max_dd = 0.0
        for raw in curve:
            value = float(raw or 0.0)
            if value <= 0:
                continue
            if value > peak:
                peak = value
            if peak <= 0:
                continue
            dd = (peak - value) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _position_pnl_summary(
        self,
        *,
        leader: Optional[str],
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        summary = {
            "panteon_owned_unrealized_pnl_usd": 0.0,
            "clean_panteon_unrealized_pnl_usd": 0.0,
            "external_unrealized_pnl_usd": 0.0,
            "adopted_unrealized_pnl_usd": 0.0,
            "leader_unrealized_pnl_usd": 0.0,
            "panteon_owned_positions_count": 0,
            "clean_panteon_positions_count": 0,
            "external_positions_count": 0,
            "adopted_positions_count": 0,
            "leader_positions_count": 0,
            "handoff_positions_count": 0,
        }
        tracker = getattr(getattr(self._pipeline, "executor", None), "_tracker", None)
        if tracker is None or not hasattr(tracker, "all_open"):
            return summary
        try:
            tracked = tracker.all_open()
        except Exception:
            return summary
        exchange_positions = self._exchange_positions()
        for sym, pos in tracked.items():
            owner = getattr(pos, "by_player", "")
            exchange_pos = exchange_positions.get(str(sym).upper())
            unrealized = _position_unrealized_pnl(
                pos,
                exchange_pos,
                current_prices=current_prices,
            )
            if _is_external_player(owner):
                summary["external_unrealized_pnl_usd"] += unrealized
                summary["external_positions_count"] += 1
                continue
            summary["panteon_owned_unrealized_pnl_usd"] += unrealized
            summary["panteon_owned_positions_count"] += 1
            if _is_adopted_player(owner, self._pipeline):
                summary["adopted_unrealized_pnl_usd"] += unrealized
                summary["adopted_positions_count"] += 1
            else:
                summary["clean_panteon_unrealized_pnl_usd"] += unrealized
                summary["clean_panteon_positions_count"] += 1
            if leader and owner == leader:
                summary["leader_unrealized_pnl_usd"] += unrealized
                summary["leader_positions_count"] += 1
            elif leader:
                summary["handoff_positions_count"] += 1
        return summary

    def _real_trade_summary(self, *, position_pnl: Dict[str, float]) -> Dict[str, int]:
        panteon_closed = 0
        successful = 0
        unsuccessful = 0
        clean_closed = 0
        clean_successful = 0
        clean_unsuccessful = 0
        adopted_closed = 0
        external_closed = 0
        try:
            closed = self._pipeline.ledger.realized_attributions()
        except Exception:
            closed = []
        for attr in closed:
            by_player = getattr(attr, "by_player", "")
            realized = float(getattr(attr, "realized_pnl", 0.0) or 0.0)
            if _is_external_player(by_player):
                external_closed += 1
                continue
            panteon_closed += 1
            if _is_adopted_player(by_player, self._pipeline):
                adopted_closed += 1
            else:
                clean_closed += 1
                if realized > 0:
                    clean_successful += 1
                else:
                    clean_unsuccessful += 1
            if realized > 0:
                successful += 1
            else:
                unsuccessful += 1
        unresolved = int(position_pnl.get("panteon_owned_positions_count", 0) or 0)
        clean_unresolved = int(position_pnl.get("clean_panteon_positions_count", 0) or 0)
        adopted_unresolved = int(position_pnl.get("adopted_positions_count", 0) or 0)
        external_unresolved = int(position_pnl.get("external_positions_count", 0) or 0)
        return {
            "total": panteon_closed + unresolved,
            "closed": panteon_closed,
            "successful": successful,
            "unsuccessful": unsuccessful,
            "unresolved": unresolved,
            "clean_total": clean_closed + clean_unresolved,
            "clean_closed": clean_closed,
            "clean_successful": clean_successful,
            "clean_unsuccessful": clean_unsuccessful,
            "clean_unresolved": clean_unresolved,
            "adopted_closed": adopted_closed,
            "adopted_unresolved": adopted_unresolved,
            "external_closed": external_closed,
            "external_unresolved": external_unresolved,
        }

    def _real_actor_payload(
        self,
        *,
        open_positions: Dict[str, dict],
        position_counts: Dict[str, int],
    ) -> Dict[str, object]:
        try:
            player_pnl = self._pipeline.ledger.total_pnl_by_player()
            player_trades = self._pipeline.ledger.trade_counts_by_player()
            player_wins = self._pipeline.ledger.win_counts_by_player()
        except Exception:
            player_pnl = {}
            player_trades = {}
            player_wins = {}
        try:
            agent_pnl = self._pipeline.ledger.total_pnl_by_agent()
        except Exception:
            agent_pnl = {}
        player_label_set = set(player_pnl.keys())
        player_label_set.update(_configured_dynamic_player_labels(self._pipeline))
        player_label_set.update(_registered_strategy_labels(self._pipeline))

        closed_players = []
        for label, pnl in sorted(
            player_pnl.items(),
            key=lambda item: -float(item[1] or 0.0),
        ):
            trades = int(player_trades.get(label, 0) or 0)
            wins = int(player_wins.get(label, 0) or 0)
            if _is_external_player(label):
                scope = "external"
            elif _is_adopted_player(label, self._pipeline):
                scope = "adopted"
            else:
                scope = "panteon"
            closed_players.append({
                "player": str(label),
                "scope": scope,
                "realized_pnl_usd": float(pnl or 0.0),
                "closed_trades": trades,
                "wins": wins,
                "win_rate": (wins / trades * 100.0) if trades else 0.0,
            })

        closed_agents = [
            {
                "agent": str(label),
                "realized_pnl_usd": float(pnl or 0.0),
            }
            for label, pnl in sorted(
                agent_pnl.items(),
                key=lambda item: -float(item[1] or 0.0),
            )
            if str(label) not in player_label_set
        ]

        open_rows = []
        for symbol, pos in sorted((open_positions or {}).items()):
            player = str(pos.get("by_player", "") or "")
            if _is_external_player(player):
                scope = "external"
            elif _is_adopted_player(player, self._pipeline):
                scope = "adopted"
            else:
                scope = "panteon"
            open_rows.append({
                "symbol": str(symbol),
                "side": str(pos.get("side", "") or ""),
                "qty": float(pos.get("qty", 0.0) or 0.0),
                "entry": float(pos.get("entry", 0.0) or 0.0),
                "player": player,
                "agent": str(pos.get("by_agent", "") or ""),
                "scope": scope,
                "unrealized_pnl_usd": float(pos.get("unrealized_pnl_usd", 0.0) or 0.0),
                "external": bool(pos.get("external", False)),
            })

        return {
            "source": "real ledger closed trades + tracked live open positions",
            "shadow_leaderboard_is_virtual": True,
            "closed_trade_count": int(getattr(self._pipeline.ledger, "closed_count", 0) or 0),
            "open_position_count": len(open_rows),
            "tracked_positions_count": int(position_counts.get("tracked_positions_count", 0) or 0),
            "exchange_positions_count": int(position_counts.get("exchange_positions_count", 0) or 0),
            "closed_players": closed_players,
            "closed_agents": closed_agents,
            "open_positions": open_rows,
        }

    def _position_counts(self, *, tracked_count: Optional[int] = None) -> Dict[str, int]:
        if tracked_count is None:
            tracked_count = len(self._tracked_open_positions())
        exchange_count = self._exchange_positions_count()
        return {
            "n_positions": max(int(tracked_count), int(exchange_count)),
            "tracked_positions_count": int(tracked_count),
            "exchange_positions_count": int(exchange_count),
        }

    def _quarantine_records(self) -> Dict[str, dict]:
        records: Dict[str, dict] = {}
        for label in sorted(self._pipeline.qm.all_quarantined()):
            record = self._pipeline.qm.record_for(label)
            if record is None:
                records[label] = {"state": "quarantined", "reason": ""}
                continue
            records[label] = {
                "state": record.state,
                "reason": record.reason,
                "entered_bar": record.entered_bar,
                "updated_bar": record.updated_bar,
            }
        return records

    def _degradation_gate_snapshot(self) -> Dict[str, object]:
        gate = getattr(self._pipeline, "degradation_gate", None)
        if gate is None:
            return {"enabled": False}
        cfg = getattr(gate, "config", None)
        config = {}
        if cfg is not None:
            config = {
                "enabled": bool(cfg.enabled),
                "label_prefixes": list(cfg.label_prefixes),
                "min_session_signals": int(cfg.min_session_signals),
                "min_session_closed_trades": int(cfg.min_session_closed_trades),
                "max_session_loss_pct": float(cfg.max_session_loss_pct),
                "max_session_drawdown_pct": float(cfg.max_session_drawdown_pct),
                "max_execution_failure_rate": float(cfg.max_execution_failure_rate),
                "max_blocked_signal_rate": float(cfg.max_blocked_signal_rate),
            }
        decisions = []
        for decision in getattr(gate, "last_decisions", ()) or ():
            metrics = decision.session_metrics
            decisions.append({
                "label": decision.label,
                "should_disable": bool(decision.should_disable),
                "reasons": list(decision.reasons),
                "session_pnl_pct": metrics.pnl_pct,
                "session_closed_trades": metrics.closed_trades,
                "session_signals": metrics.signals,
                "session_execution_failures": metrics.rejected_signals,
                "session_unexecuted_signals": metrics.execution_failures,
                "session_blocked_signals": metrics.blocked_signals,
                "session_rejected_signals": metrics.rejected_signals,
                "session_max_drawdown_pct": metrics.max_dd_pct,
                "execution_failure_rate": decision.execution_failure_rate,
                "blocked_signal_rate": decision.blocked_signal_rate,
            })
        return {
            "enabled": bool(config.get("enabled", True)),
            "baseline_captured": bool(getattr(gate, "baseline_captured", False)),
            "config": config,
            "last_decisions": decisions,
        }

    def _capture_session_perf_baseline(self) -> Dict[str, dict]:
        labels = set()
        try:
            labels.update(self._pipeline.perf.all_labels())
        except Exception:
            pass
        try:
            labels.update(self._pipeline.registry.all_labels())
        except Exception:
            pass
        try:
            labels.update(profile.label for profile in self._pipeline.profiles)
        except Exception:
            pass
        return {
            label: self._metrics_snapshot(label)
            for label in labels
            if label
        }

    def _metrics_snapshot(self, label: str) -> dict:
        metrics = self._pipeline.perf.get(label)
        return {
            "pnl_pct": float(metrics.pnl_pct),
            "closed_trades": int(metrics.closed_trades),
            "wins": int(metrics.wins),
            "losses": int(metrics.losses),
            "entries": int(metrics.entries),
            "signals": int(metrics.signals),
        }

    def _session_metrics_for(self, label: Optional[str], metrics=None) -> dict:
        if not label:
            return {
                "pnl_pct": 0.0,
                "closed_trades": 0,
                "wins": 0,
                "losses": 0,
                "entries": 0,
                "signals": 0,
                "win_rate": 0.0,
            }
        metrics = metrics or self._pipeline.perf.get(label)
        baseline = self._session_perf_baseline.get(label, {})
        closed = max(0, int(metrics.closed_trades) - int(baseline.get("closed_trades", 0)))
        wins = max(0, int(metrics.wins) - int(baseline.get("wins", 0)))
        losses = max(0, int(metrics.losses) - int(baseline.get("losses", 0)))
        entries = max(0, int(metrics.entries) - int(baseline.get("entries", 0)))
        signals = max(0, int(metrics.signals) - int(baseline.get("signals", 0)))
        return {
            "pnl_pct": float(metrics.pnl_pct) - float(baseline.get("pnl_pct", 0.0)),
            "closed_trades": closed,
            "wins": wins,
            "losses": losses,
            "entries": entries,
            "signals": signals,
            "win_rate": (wins / closed * 100.0) if closed else 0.0,
        }

    def _actor_equity_curve(self, label: str) -> List[float]:
        getter = getattr(getattr(self._pipeline, "perf", None), "equity_curve", None)
        if not callable(getter):
            return []
        try:
            curve = [float(item) for item in getter(label) if float(item) > 0.0]
        except Exception:
            return []
        return curve if len(curve) > 1 else []

    def _actor_equity_curve_timestamps(self, label: str) -> List[str]:
        getter = getattr(
            getattr(self._pipeline, "perf", None),
            "equity_curve_timestamps",
            None,
        )
        if not callable(getter):
            return []
        try:
            return [str(item) for item in getter(label) if str(item or "").strip()]
        except Exception:
            return []

    def _actor_session_equity_curve(self, label: str) -> List[float]:
        curve = self._actor_equity_curve(label)
        if len(curve) < 2:
            return []
        metrics = self._pipeline.perf.get(label)
        baseline = self._session_perf_baseline.get(label, {})
        session_closed = max(
            0,
            int(metrics.closed_trades) - int(baseline.get("closed_trades", 0)),
        )
        if session_closed <= 0:
            return []
        current_closed = max(0, int(metrics.closed_trades))
        baseline_closed = max(0, int(baseline.get("closed_trades", 0)))
        if len(curve) >= current_closed + 1:
            start = min(baseline_closed, len(curve) - 2)
        else:
            start = max(0, len(curve) - (session_closed + 1))
        session_curve = curve[start:]
        if len(session_curve) < 2:
            return []
        first = float(session_curve[0] or 0.0)
        if first <= 0.0:
            return []
        return [round(float(item) / first * 100.0, 10) for item in session_curve]

    def _actor_session_equity_curve_timestamps(self, label: str) -> List[str]:
        curve = self._actor_session_equity_curve(label)
        if len(curve) < 2:
            return []
        timestamps = self._actor_equity_curve_timestamps(label)
        if not timestamps:
            return []
        needed = len(curve)
        return timestamps[-needed:] if len(timestamps) >= needed else timestamps

    def _exchange_positions_count(self) -> int:
        return len(self._exchange_positions())

    def _exchange_positions(self) -> Dict[str, object]:
        exchange = getattr(getattr(self._pipeline, "executor", None), "_exchange", None)
        getter = getattr(exchange, "get_all_positions", None)
        if not callable(getter):
            return {}
        try:
            positions = getter()
        except Exception:
            log.debug("exchange positions sync failed in OutputWriter", exc_info=True)
            return {}
        if not isinstance(positions, dict):
            return {}
        return {str(sym).upper(): pos for sym, pos in positions.items()}

    def _write_leaderboards(self) -> None:
        """leaderboard_agents.json / leaderboard_players.json как у v1."""
        try:
            self._pipeline.ledger.replay_from_event_log(self._pipeline.event_log)
        except Exception:
            return

        player_pnl = self._pipeline.ledger.total_pnl_by_player()
        trade_counts = self._pipeline.ledger.trade_counts_by_player()
        win_counts = self._pipeline.ledger.win_counts_by_player()
        actor_pool_players = {
            str(row.get("label") or ""): row
            for row in self._actor_pool_player_rows()
            if str(row.get("label") or "")
        }
        player_labels = {
            str(profile.label)
            for profile in self._pipeline.profiles
            if str(getattr(profile, "label", "") or "")
        }
        player_labels.update(actor_pool_players.keys())
        player_labels.update(_configured_dynamic_player_labels(self._pipeline))
        player_labels.update(player_pnl.keys())

        # AGENTS: component labels referenced by players, excluding standalone strategies.
        agents: Dict[str, dict] = {}
        agent_pnl = self._pipeline.ledger.total_pnl_by_agent()
        actor_pool_agents = {
            str(row.get("label") or ""): row
            for row in self._actor_pool_agent_rows()
            if str(row.get("label") or "") and str(row.get("label") or "") not in player_labels
        }
        agent_labels = set(agent_pnl.keys())
        agent_labels.update(actor_pool_agents.keys())
        agent_labels.difference_update(player_labels)
        for label in sorted(agent_labels):
            metrics_agg = self._pipeline.perf.get(label)
            configured_pool = label in actor_pool_agents
            has_agent_pnl = label in agent_pnl
            if not metrics_agg.has_data and not configured_pool and not has_agent_pnl:
                continue
            session = self._session_metrics_for(label, metrics_agg)
            per_regime = {}
            for r in Regime:
                rm = self._pipeline.perf.get(label, regime=r)
                if rm.has_data:
                    per_regime[r.label] = {
                        "pnl_pct":        rm.pnl_pct,
                        "closed_trades":  rm.closed_trades,
                        "entries":        rm.entries,
                        "signals":        rm.signals,
                        "wins":           rm.wins,
                        "losses":         rm.losses,
                        "win_rate":       rm.win_rate,
                        "sharpe":         rm.sharpe,
                        "max_drawdown_pct": rm.max_dd_pct,
                    }
            quarantine_record = self._pipeline.qm.record_for(label)
            configured_pool_only = configured_pool and not metrics_agg.has_data
            agents["V_" + label] = {
                "pnl_pct":        metrics_agg.pnl_pct,
                "session_pnl_pct": session["pnl_pct"],
                "equity_curve":   self._actor_equity_curve(label),
                "equity_curve_timestamps": self._actor_equity_curve_timestamps(label),
                "session_equity_curve": self._actor_session_equity_curve(label),
                "session_equity_curve_timestamps": (
                    self._actor_session_equity_curve_timestamps(label)
                ),
                "closed_trades":  metrics_agg.closed_trades,
                "session_closed_trades": session["closed_trades"],
                "entries":        metrics_agg.entries,
                "session_entries": session["entries"],
                "signals":        metrics_agg.signals,
                "session_signals": session["signals"],
                "wins":           metrics_agg.wins,
                "session_wins":    session["wins"],
                "losses":         metrics_agg.losses,
                "session_losses":  session["losses"],
                "win_rate":       metrics_agg.win_rate,
                "session_win_rate": session["win_rate"],
                "sharpe":         metrics_agg.sharpe,
                "max_drawdown_pct": metrics_agg.max_dd_pct,
                "is_quarantined": self._pipeline.qm.is_quarantined(label),
                "quarantine_reason": quarantine_record.reason if quarantine_record else "",
                "configured_pool": configured_pool,
                "configured_pool_only": configured_pool_only,
                "actor_pool_flags": str(actor_pool_agents.get(label, {}).get("flags", "")),
                "status_reason": (
                    "configured actor pool; no leaderboard activity yet"
                    if configured_pool_only else ""
                ),
                "per_regime":     per_regime,
            }

        # PLAYERS: profile labels from PerformanceMemory plus real ledger overlay.
        players: Dict[str, dict] = {}
        for label in sorted(player_labels):
            metrics_agg = self._pipeline.perf.get(label)
            configured_pool = label in actor_pool_players
            if not metrics_agg.has_data and label not in player_pnl and not configured_pool:
                continue
            session = self._session_metrics_for(label, metrics_agg)
            per_regime = {}
            for r in Regime:
                rm = self._pipeline.perf.get(label, regime=r)
                if rm.has_data:
                    per_regime[r.label] = {
                        "pnl_pct":        rm.pnl_pct,
                        "closed_trades":  rm.closed_trades,
                        "entries":        rm.entries,
                        "signals":        rm.signals,
                        "wins":           rm.wins,
                        "losses":         rm.losses,
                        "win_rate":       rm.win_rate,
                        "sharpe":         rm.sharpe,
                        "max_drawdown_pct": rm.max_dd_pct,
                    }
            real_trades = trade_counts.get(label, 0)
            real_wins = win_counts.get(label, 0)
            configured_pool_only = (
                configured_pool
                and not metrics_agg.has_data
                and label not in player_pnl
            )
            players["V_" + label] = {
                "pnl_pct":          metrics_agg.pnl_pct,
                "session_pnl_pct":  session["pnl_pct"],
                "equity_curve":     self._actor_equity_curve(label),
                "equity_curve_timestamps": self._actor_equity_curve_timestamps(label),
                "session_equity_curve": self._actor_session_equity_curve(label),
                "session_equity_curve_timestamps": (
                    self._actor_session_equity_curve_timestamps(label)
                ),
                "closed_trades":    metrics_agg.closed_trades,
                "session_closed_trades": session["closed_trades"],
                "entries":          metrics_agg.entries,
                "session_entries":  session["entries"],
                "signals":          metrics_agg.signals,
                "session_signals":  session["signals"],
                "wins":             metrics_agg.wins,
                "session_wins":     session["wins"],
                "losses":           metrics_agg.losses,
                "session_losses":   session["losses"],
                "win_rate":         metrics_agg.win_rate,
                "session_win_rate": session["win_rate"],
                "sharpe":           metrics_agg.sharpe,
                "max_drawdown_pct": metrics_agg.max_dd_pct,
                "realized_pnl_usd": player_pnl.get(label, 0.0),
                "real_trades":      real_trades,
                "real_wins":        real_wins,
                "real_win_rate":    (real_wins / real_trades * 100.0) if real_trades else 0.0,
                "is_quarantined":    self._pipeline.qm.is_quarantined(label),
                "configured_pool":   configured_pool,
                "configured_pool_only": configured_pool_only,
                "actor_pool_kind":   str(actor_pool_players.get(label, {}).get("kind", "")),
                "status_reason": (
                    "configured actor pool; no leaderboard activity yet"
                    if configured_pool_only else ""
                ),
                "per_regime":       per_regime,
            }

        agents_meta = {
            "schema":    "v2",
            "exchange":  self._pipeline.exchange_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime":    "—",  # фиксируется по последнему step, ниже
        }
        split_meta = dict(agents_meta)
        split_meta["real_executable_criteria"] = (
            "registered live actor, or explicit controlled-probation Genetics label; "
            "per-symbol Flash gates, min-notional, slippage, fees, and daily limits still apply"
        )
        real_agents, shadow_agents = self._split_agent_leaderboard(agents)
        real_players, shadow_players = self._split_player_leaderboard(players)
        self._last_agents_payload = {"metadata": agents_meta, "agents": agents}
        self._last_players_payload = {"metadata": agents_meta, "players": players}
        self._write_json_atomic(
            os.path.join(self._config.output_dir, self._config.leaderboard_agents),
            self._last_agents_payload,
        )
        self._write_json_atomic(
            os.path.join(self._config.output_dir, self._config.leaderboard_players),
            self._last_players_payload,
        )
        self._write_json_atomic(
            os.path.join(
                self._config.output_dir,
                self._config.leaderboard_real_executable_agents,
            ),
            {"metadata": dict(split_meta, scope="real_executable"), "agents": real_agents},
        )
        self._write_json_atomic(
            os.path.join(
                self._config.output_dir,
                self._config.leaderboard_shadow_only_agents,
            ),
            {"metadata": dict(split_meta, scope="shadow_only"), "agents": shadow_agents},
        )
        self._write_json_atomic(
            os.path.join(
                self._config.output_dir,
                self._config.leaderboard_real_executable_players,
            ),
            {"metadata": dict(split_meta, scope="real_executable"), "players": real_players},
        )
        self._write_json_atomic(
            os.path.join(
                self._config.output_dir,
                self._config.leaderboard_shadow_only_players,
            ),
            {"metadata": dict(split_meta, scope="shadow_only"), "players": shadow_players},
        )

    def _split_agent_leaderboard(
        self,
        agents: Dict[str, dict],
    ) -> tuple[Dict[str, dict], Dict[str, dict]]:
        real: Dict[str, dict] = {}
        shadow: Dict[str, dict] = {}
        for key, row in sorted(agents.items()):
            label = key[2:] if key.startswith("V_") else key
            if self._agent_is_real_executable_for_leaderboard(label):
                real[key] = row
            else:
                shadow[key] = row
        return real, shadow

    def _split_player_leaderboard(
        self,
        players: Dict[str, dict],
    ) -> tuple[Dict[str, dict], Dict[str, dict]]:
        real: Dict[str, dict] = {}
        shadow: Dict[str, dict] = {}
        for key, row in sorted(players.items()):
            label = key[2:] if key.startswith("V_") else key
            if self._player_is_real_executable_for_leaderboard(label, row):
                real[key] = row
            else:
                shadow[key] = row
        return real, shadow

    def _agent_is_real_executable_for_leaderboard(self, label: str) -> bool:
        registry = getattr(self._pipeline, "registry", None)
        agent = registry.get(label) if registry is not None and hasattr(registry, "get") else None
        if agent is None:
            return False
        if self._agent_is_controlled_probation_for_leaderboard(label):
            return not self._pipeline.qm.is_quarantined(label)
        if bool(getattr(agent, "shadow_only", False)):
            return False
        if getattr(agent, "live_trading_eligible", True) is False:
            return False
        if self._pipeline.qm.is_quarantined(label):
            return False
        return True

    def _agent_is_controlled_probation_for_leaderboard(self, label: str) -> bool:
        cfg = getattr(self._pipeline, "live_execution", None)
        if not bool(getattr(cfg, "genetics_probation_execution_enabled", False)):
            return False
        clean = str(label or "").strip()
        if not clean.startswith("Genetics"):
            return False
        labels = {
            str(item or "").strip()
            for item in (getattr(cfg, "genetics_probation_labels", ()) or ())
            if str(item or "").strip()
        }
        return clean in labels

    def _player_is_real_executable_for_leaderboard(
        self,
        label: str,
        row: Dict[str, object],
    ) -> bool:
        if self._pipeline.qm.is_quarantined(label):
            return False
        if int(row.get("real_trades") or 0) > 0:
            return True
        if bool(row.get("configured_pool_only")):
            return False
        profile_labels = {
            str(getattr(profile, "label", "") or "")
            for profile in getattr(self._pipeline, "profiles", ()) or ()
        }
        if label in profile_labels or label in _configured_dynamic_player_labels(self._pipeline):
            return True
        if label in _registered_strategy_labels(self._pipeline):
            return self._agent_is_real_executable_for_leaderboard(label)
        return False

    def _write_dashboard(self) -> None:
        """ASCII dashboard через TextRenderer."""
        try:
            main = self._pipeline.renderer.build_main(timeline_max=20)
            text = TextRenderer().render_main(main)
            text = self._append_actor_pool_summary(text)
            text = self._append_real_actor_summary(text)
            path = os.path.join(self._config.output_dir, self._config.dashboard_filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._write_visual_dashboards()
            self._write_dashboard_html(text)
        except Exception:
            log.exception("dashboard render failed")

    def _append_actor_pool_summary(self, text: str) -> str:
        title = "Configured actor pool"
        line = "=" * max(len(title) + 4, 60)
        player_rows = self._actor_pool_player_rows()
        agent_rows = self._actor_pool_agent_rows()
        actionable = sorted(
            str(label)
            for label in (getattr(self._pipeline, "_current_actionable_player_labels", set()) or set())
            if str(label)
        )

        out = ["", line, f"  {title}", line]
        out.append(
            "  Source: configured profiles/dynamic players + standalone strategies; "
            "rows are shown even before leaderboard activity."
        )
        out.append(
            "  players={players} component_agents={agents} actionable={actionable}".format(
                players=len(player_rows),
                agents=len(agent_rows),
                actionable=len(actionable),
            )
        )
        if actionable:
            out.append(f"  Current actionable players: {', '.join(actionable)}")

        out.append("  Players:")
        if player_rows:
            out.append(
                f"    {'Label':<32s} {'Kind':<14s} {'Affinity/regimes':<24s} Agents"
            )
            out.append(
                f"    {'-'*32} {'-'*14} {'-'*24} {'-'*24}"
            )
            for row in player_rows:
                out.append(
                    "    {label:<32s} {kind:<14s} {scope:<24s} {agents}".format(
                        label=row["label"],
                        kind=row["kind"],
                        scope=row["scope"],
                        agents=row["agents"],
                    )
                )
        else:
            out.append("    none")

        out.append("  Agents:")
        if agent_rows:
            out.append(
                f"    {'Label':<32s} {'Flags':<22s} {'Pnl%':>8s} {'Closed':>6s} {'Signals':>7s}"
            )
            out.append(
                f"    {'-'*32} {'-'*22} {'-'*8} {'-'*6} {'-'*7}"
            )
            for row in agent_rows:
                out.append(
                    "    {label:<32s} {flags:<22s} {pnl:>+8.4f} {closed:>6d} {signals:>7d}".format(
                        label=row["label"],
                        flags=row["flags"],
                        pnl=float(row["session_pnl_pct"]),
                        closed=int(row["session_closed_trades"]),
                        signals=int(row["session_signals"]),
                    )
                )
        else:
            out.append("    none")
        return text + "\n" + "\n".join(out)

    def _configured_actor_pool_payload(self) -> Dict[str, object]:
        actionable = sorted(
            str(label)
            for label in (getattr(self._pipeline, "_current_actionable_player_labels", set()) or set())
            if str(label)
        )
        return {
            "source": (
                "registered standalone strategies exposed as players; "
                "no agent candidates"
                if bool(getattr(self._pipeline, "player_only_runtime", False))
                else
                "configured profiles/dynamic players + standalone strategies; "
                "rows are shown even before leaderboard activity"
            ),
            "players": self._actor_pool_player_rows(),
            "agents": self._actor_pool_agent_rows(),
            "current_actionable_players": actionable,
        }

    def _actor_pool_player_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for profile in getattr(self._pipeline, "profiles", ()) or ():
            label = str(getattr(profile, "label", "") or "").strip()
            if not label:
                continue
            affinity = getattr(getattr(profile, "affinity", None), "label", None) or "any"
            agent_labels = tuple(
                str(item).strip()
                for item in tuple(getattr(profile, "allowed_labels", ()) or ())
                if str(item).strip()
            )
            if not agent_labels:
                raw_bias = getattr(profile, "bias", {}) or {}
                bias_keys = raw_bias.keys() if isinstance(raw_bias, dict) else ()
                agent_labels = tuple(
                    str(item).strip()
                    for item in bias_keys
                    if str(item).strip()
                )
            rows.append({
                "label": label,
                "kind": "profile",
                "scope": str(affinity),
                "agents": self._format_pool_items(agent_labels or ("selector",)),
            })

        for label, mapping in self._regime_switch_pool_specs():
            rows.append({
                "label": label,
                "kind": "regime_switch",
                "scope": self._format_pool_items(sorted(mapping.keys())),
                "agents": self._format_regime_mapping(mapping),
            })

        for label, mapping, fallback in self._rotating_pool_specs():
            agents = self._format_regime_mapping(mapping)
            if fallback:
                agents = f"{agents}; fallback={self._format_pool_items(fallback)}"
            rows.append({
                "label": label,
                "kind": "rotating",
                "scope": self._format_pool_items(sorted(mapping.keys())),
                "agents": agents,
            })

        existing_player_labels = {
            str(row.get("label") or "").strip()
            for row in rows
            if str(row.get("label") or "").strip()
        }
        for label in sorted(_registered_strategy_labels(self._pipeline)):
            if label in existing_player_labels:
                continue
            rows.append({
                "label": label,
                "kind": "strategy",
                "scope": "standalone",
                "agents": "-",
            })

        deduped: Dict[tuple[str, str], Dict[str, str]] = {}
        for row in rows:
            deduped[(row["kind"], row["label"])] = row
        return list(deduped.values())

    def _component_agent_labels(self) -> set[str]:
        labels: set[str] = set()
        for profile in getattr(self._pipeline, "profiles", ()) or ():
            labels.update(
                str(label).strip()
                for label in tuple(getattr(profile, "allowed_labels", ()) or ())
                if str(label or "").strip()
            )
            raw_bias = getattr(profile, "bias", {}) or {}
            if isinstance(raw_bias, dict):
                labels.update(
                    str(label).strip()
                    for label in raw_bias.keys()
                    if str(label or "").strip()
                )
        for _label, mapping in self._regime_switch_pool_specs():
            for agent_labels in mapping.values():
                labels.update(agent_labels)
        for _label, mapping, fallback in self._rotating_pool_specs():
            for agent_labels in mapping.values():
                labels.update(agent_labels)
            labels.update(fallback)
        try:
            labels.update(self._pipeline.ledger.total_pnl_by_agent().keys())
        except Exception:
            pass
        for signals in (
            getattr(self._pipeline, "_current_shadow_player_signals", {}) or {}
        ).values():
            for signal in signals or ():
                by_agent = str(getattr(signal, "by_agent", "") or "").strip()
                if by_agent:
                    labels.add(by_agent)
        return {label for label in labels if str(label or "").strip()}

    def _actor_pool_agent_rows(self) -> List[Dict[str, object]]:
        registry = getattr(self._pipeline, "registry", None)
        labels = self._component_agent_labels()
        labels.difference_update(_registered_strategy_labels(self._pipeline))
        rows: List[Dict[str, object]] = []
        for label in sorted(labels):
            agent = registry.get(label) if registry is not None and hasattr(registry, "get") else None
            metrics = self._pipeline.perf.get(label)
            session = self._session_metrics_for(label, metrics)
            rows.append({
                "label": label,
                "flags": self._actor_pool_agent_flags(label, agent),
                "session_pnl_pct": session["pnl_pct"],
                "session_closed_trades": session["closed_trades"],
                "session_signals": session["signals"],
            })
        return rows

    def _actor_pool_agent_flags(self, label: str, agent: object) -> str:
        flags: List[str] = []
        if agent is None:
            flags.append("missing")
        else:
            if bool(getattr(agent, "shadow_only", False)):
                flags.append("shadow")
            elif getattr(agent, "live_trading_eligible", True) is False:
                flags.append("not_live")
            else:
                flags.append("live")
        if self._pipeline.qm.is_quarantined(label):
            flags.append("Q")
        return ",".join(flags) or "-"

    def _regime_switch_pool_specs(self) -> List[tuple[str, Dict[str, tuple[str, ...]]]]:
        rows: List[tuple[str, Dict[str, tuple[str, ...]]]] = []
        for raw_spec in getattr(self._pipeline, "regime_switch_player_sets", ()) or ():
            if isinstance(raw_spec, dict):
                label = str(raw_spec.get("label") or "").strip()
                raw_mapping = raw_spec.get("regime_agents") or raw_spec.get("agents") or {}
            else:
                try:
                    parts = tuple(raw_spec)  # type: ignore[arg-type]
                except TypeError:
                    parts = ()
                label = str(parts[0] or "").strip() if parts else ""
                raw_mapping = parts[1] if len(parts) > 1 else {}
            mapping = self._normalize_pool_regime_mapping(raw_mapping)
            if label and mapping:
                rows.append((label, mapping))
        return rows

    def _rotating_pool_specs(self) -> List[tuple[str, Dict[str, tuple[str, ...]], tuple[str, ...]]]:
        rows: List[tuple[str, Dict[str, tuple[str, ...]], tuple[str, ...]]] = []
        for raw_spec in getattr(self._pipeline, "rotating_agent_player_sets", ()) or ():
            if isinstance(raw_spec, dict):
                label = str(raw_spec.get("label") or "").strip()
                raw_mapping = raw_spec.get("regime_agent_order") or raw_spec.get("agents") or {}
                raw_fallback = raw_spec.get("fallback_agent_order") or raw_spec.get("fallback") or ()
            else:
                try:
                    parts = tuple(raw_spec)  # type: ignore[arg-type]
                except TypeError:
                    parts = ()
                label = str(parts[0] or "").strip() if parts else ""
                raw_mapping = parts[1] if len(parts) > 1 else {}
                raw_fallback = parts[2] if len(parts) > 2 else ()
            mapping = self._normalize_pool_regime_mapping(raw_mapping)
            fallback = self._normalize_pool_label_tuple(raw_fallback)
            if label and (mapping or fallback):
                rows.append((label, mapping, fallback))
        return rows

    @staticmethod
    def _normalize_pool_regime_mapping(raw_mapping: object) -> Dict[str, tuple[str, ...]]:
        if not isinstance(raw_mapping, dict):
            return {}
        aliases = {
            "bull": "bullish",
            "bear": "bearish",
            "flat": "neutral",
            "sideways": "neutral",
            "panic": "crash",
        }
        out: Dict[str, tuple[str, ...]] = {}
        for raw_regime, raw_agents in raw_mapping.items():
            raw_key = str(raw_regime or "").strip().lower()
            regime = aliases.get(raw_key, raw_key)
            labels = OutputWriter._normalize_pool_label_tuple(raw_agents)
            if regime and labels:
                out[regime] = labels
        return out

    @staticmethod
    def _normalize_pool_label_tuple(raw_value: object) -> tuple[str, ...]:
        if isinstance(raw_value, str):
            values = raw_value.replace(";", ",").split(",")
        else:
            try:
                values = tuple(raw_value)  # type: ignore[arg-type]
            except TypeError:
                values = (raw_value,)
        labels: List[str] = []
        for value in values:
            label = str(value or "").strip()
            if label and label not in labels:
                labels.append(label)
        return tuple(labels)

    @staticmethod
    def _format_regime_mapping(mapping: Dict[str, tuple[str, ...]]) -> str:
        return "; ".join(
            f"{regime}:{OutputWriter._format_pool_items(labels)}"
            for regime, labels in sorted(mapping.items())
        ) or "-"

    @staticmethod
    def _format_pool_items(items: object, *, limit: int = 12) -> str:
        values = tuple(str(item).strip() for item in (items or ()) if str(item).strip())
        if not values:
            return "-"
        head = ", ".join(values[:limit])
        if len(values) > limit:
            return f"{head}, +{len(values) - limit} more"
        return head

    def _append_real_actor_summary(self, text: str) -> str:
        status = self._last_status_data if isinstance(self._last_status_data, dict) else {}
        payload = status.get("real_trading_actors")
        if not isinstance(payload, dict):
            return text

        title = "Real trading actors (live ledger/tracker)"
        line = "=" * max(len(title) + 4, 60)
        out = ["", line, f"  {title}", line]
        out.append(
            "  Source: real closed ledger + tracked open exchange positions; "
            "Shadow Leaderboard below is virtual."
        )
        out.append(
            "  closed_trades={closed} open_positions={open_} "
            "tracked={tracked} exchange_positions={exchange}".format(
                closed=int(payload.get("closed_trade_count", 0) or 0),
                open_=int(payload.get("open_position_count", 0) or 0),
                tracked=int(payload.get("tracked_positions_count", 0) or 0),
                exchange=int(payload.get("exchange_positions_count", 0) or 0),
            )
        )

        open_rows = payload.get("open_positions")
        if isinstance(open_rows, list) and open_rows:
            out.append("  Open real positions:")
            for row in open_rows[:10]:
                if not isinstance(row, dict):
                    continue
                out.append(
                    "    {symbol:<12s} {side:<5s} qty={qty:.8g} entry={entry:.8g} "
                    "player={player} agent={agent} scope={scope}".format(
                        symbol=str(row.get("symbol", "") or ""),
                        side=str(row.get("side", "") or ""),
                        qty=float(row.get("qty", 0.0) or 0.0),
                        entry=float(row.get("entry", 0.0) or 0.0),
                        player=str(row.get("player", "") or "-"),
                        agent=str(row.get("agent", "") or "-"),
                        scope=str(row.get("scope", "") or "-"),
                    )
                )
        else:
            out.append("  Open real positions: none")

        closed_players = payload.get("closed_players")
        if isinstance(closed_players, list) and closed_players:
            out.append("  Closed real players:")
            for row in closed_players[:10]:
                if not isinstance(row, dict):
                    continue
                out.append(
                    "    {player:<28s} pnl={pnl:+.4f} trades={trades} "
                    "win={win:.1f}% scope={scope}".format(
                        player=str(row.get("player", "") or ""),
                        pnl=float(row.get("realized_pnl_usd", 0.0) or 0.0),
                        trades=int(row.get("closed_trades", 0) or 0),
                        win=float(row.get("win_rate", 0.0) or 0.0),
                        scope=str(row.get("scope", "") or "-"),
                    )
                )
        else:
            out.append("  Closed real players: none")
        return text.rstrip() + "\n" + "\n".join(out) + "\n"

    def _write_visual_dashboards(self) -> None:
        try:
            paths = write_operator_pngs(
                self._config.output_dir,
                status=self._last_status_data,
                agents_payload=self._last_agents_payload,
                players_payload=self._last_players_payload,
            )
            self._publish_latest_visual_dashboards(paths)
        except Exception:
            log.exception("visual dashboard render failed")

    def _publish_latest_visual_dashboards(self, paths: List[str]) -> None:
        latest_dir = self._config.latest_dir
        if not latest_dir or getattr(self, "_latest_dashboard_publish_disabled", False):
            return
        exchange = self._latest_exchange_suffix()
        selected = {
            "dashboard_latest.png": f"dashboard_latest_{exchange}.png",
            "shadow_dashboard.png": f"shadow_dashboard_{exchange}.png",
            "regime_dashboard.png": f"regime_dashboard_{exchange}.png",
            "memory_dashboard.png": f"memory_dashboard_{exchange}.png",
        }
        Path(latest_dir).mkdir(parents=True, exist_ok=True)
        for raw_path in paths:
            src = Path(raw_path)
            dest_name = selected.get(src.name)
            if not dest_name or not src.exists():
                continue
            dest = Path(latest_dir) / dest_name
            if not self._publish_latest_dashboard_file(src, dest):
                continue

    def _publish_latest_dashboard_file(
        self,
        src: Path,
        dest: Path,
        *,
        attempts: int = 5,
    ) -> bool:
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
        last_error: Exception | None = None
        for attempt in range(max(1, int(attempts))):
            try:
                shutil.copyfile(src, tmp)
                os.replace(tmp, dest)
                return True
            except PermissionError as exc:
                last_error = exc
                try:
                    tmp.unlink()
                except OSError:
                    pass
                time.sleep(min(0.2 * (attempt + 1), 1.0))
            except Exception:
                log.exception("latest dashboard publish failed: %s -> %s", src, dest)
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return False

        try:
            shutil.copyfile(src, dest)
            return True
        except PermissionError as exc:
            last_error = exc
        except Exception:
            log.exception("latest dashboard publish failed: %s -> %s", src, dest)
            return False
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

        log.warning("latest dashboard publish skipped; destination is locked: %s -> %s (%s)",
                    src, dest, last_error)
        return False

    def _latest_exchange_suffix(self) -> str:
        raw = str(self._pipeline.exchange_name or "UNKNOWN").upper()
        for suffix in ("-DRYRUN", "-PAPER"):
            raw = raw.replace(suffix, "")
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
        return cleaned or "UNKNOWN"

    def _write_dashboard_html(self, text: str) -> None:
        status_path = self._config.status_filename
        escaped = html.escape(text)
        now = datetime.now(timezone.utc).isoformat()
        image_names = (
            "dashboard_latest.png",
            "shadow_dashboard.png",
            "regime_dashboard.png",
            "memory_dashboard.png",
        )
        image_links = "\n".join(
            f'<figure><a href="{html.escape(name)}"><img src="{html.escape(name)}" alt="{html.escape(name)}"></a>'
            f'<figcaption>{html.escape(name)}</figcaption></figure>'
            for name in image_names
        )
        page = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%23161b22'/%3E%3Cpath d='M14 42h36L32 12z' fill='%2358a6ff'/%3E%3Ccircle cx='32' cy='38' r='6' fill='%23d6deeb'/%3E%3C/svg%3E">
<title>Panteon v2 dashboard</title>
<style>
body {{ margin: 0; background: #0d1117; color: #d6deeb; font-family: Consolas, monospace; }}
header {{ padding: 12px 16px; background: #161b22; border-bottom: 1px solid #30363d; }}
main {{ padding: 16px; }}
a {{ color: #58a6ff; }}
pre {{ white-space: pre-wrap; line-height: 1.35; font-size: 13px; }}
.gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-bottom: 18px; }}
figure {{ margin: 0; background: #161b22; border: 1px solid #30363d; padding: 8px; }}
img {{ width: 100%; height: auto; display: block; }}
figcaption {{ color: #8b949e; font-size: 12px; padding-top: 6px; }}
.muted {{ color: #8b949e; }}
.sr-only {{ position: absolute; left: -10000px; top: auto; width: 1px; height: 1px; overflow: hidden; }}
</style>
</head>
<body>
<header>
<strong>Panteon v2</strong>
<span class="muted">updated {html.escape(now)}</span>
<span class="muted">status: <a href="{html.escape(status_path)}">{html.escape(status_path)}</a></span>
</header>
<main>
<div class="sr-only">Панель №1</div>
<section class="gallery">
{image_links}
</section>
<pre>{escaped}</pre>
</main>
</body>
</html>
"""
        path = os.path.join(self._config.output_dir, self._config.dashboard_html)
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _write_json_atomic(path: str, data: dict) -> None:
        """Атомарная запись: пишем в tmp + rename."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = path + ".tmp"
        last_exc: Optional[Exception] = None
        for attempt in range(5):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str, ensure_ascii=False)
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_exc = exc
                time.sleep(0.05 * (attempt + 1))
            except Exception:
                log.exception("atomic write failed: %s", path)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            try:
                os.remove(tmp)
            except OSError:
                pass
        except Exception:
            log.exception("atomic write failed: %s", path)
            if last_exc is not None:
                log.debug("last atomic replace error: %r", last_exc)
            try:
                os.remove(tmp)
            except OSError:
                pass

    @staticmethod
    def _format_reason_counts(reasons: Dict[str, int]) -> str:
        return "; ".join(
            f"{reason} x{count}"
            for reason, count in sorted(reasons.items())
        )

    @staticmethod
    def _format_symbol_actor_map(values: Dict[str, object]) -> str:
        return ",".join(
            f"{symbol}:{actor}"
            for symbol, actor in sorted((str(k), str(v)) for k, v in dict(values or {}).items())
        )

    @property
    def output_dir(self) -> str:
        return self._config.output_dir

    def close(self) -> None:
        """Финальная фиксация state перед выходом."""
        try:
            self._ensure_output_dir()
            self._write_status(
                self._last_step,
                run_state="stopped",
                feed_status="closed",
                message="writer closed",
            )
            self._write_leaderboards()
            self._write_dashboard()
            self._write_final_session_reports()
            stop_ts = _utc_msk_timestamps()
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\n")
                f.write(
                    "Panteon v2 stopped: "
                    f"UTC={stop_ts['timestamp_utc']} "
                    f"MSK={stop_ts['timestamp_msk']} "
                    f"pid={os.getpid()}\n"
                )
                f.write(f"{'='*70}\n")
        except Exception:
            log.exception("close failed")

    def _write_final_session_reports(self) -> None:
        status = dict(getattr(self, "_last_status_data", {}) or {})
        status.setdefault("exchange", getattr(self._pipeline, "exchange_name", ""))
        status.setdefault("run_state", "stopped")
        status.setdefault("session_id", getattr(self._pipeline, "session_id", ""))
        status.setdefault("run_id", getattr(self._pipeline, "run_id", ""))
        report_ts = _utc_msk_timestamps()
        status["final_report_generated_at"] = report_ts["timestamp_utc"]
        status["final_report_generated_at_msk"] = report_ts["timestamp_msk"]
        status.setdefault("active_pid", os.getpid())
        kill_switch = getattr(self._pipeline, "kill_switch", None)
        status["kill_switch"] = {
            "disabled_reason": str(getattr(kill_switch, "disabled_reason", "") or ""),
            "peak_equity_usd": float(getattr(kill_switch, "peak_equity_usd", 0.0) or 0.0),
        }

        json_path = os.path.join(self._config.output_dir, "final_session_report.json")
        md_path = os.path.join(self._config.output_dir, "final_session_report.md")
        self._write_json_atomic(json_path, status)
        Path(md_path).write_text(self._final_session_markdown(status), encoding="utf-8")

    def _final_session_markdown(self, status: Dict[str, Any]) -> str:
        live = status.get("live_session") if isinstance(status.get("live_session"), dict) else {}
        real_trades = status.get("real_trades") if isinstance(status.get("real_trades"), dict) else {}
        kill = status.get("kill_switch") if isinstance(status.get("kill_switch"), dict) else {}
        lines = [
            "# Panteon v2 final session report",
            "",
            f"- Exchange: `{status.get('exchange', '')}`",
            f"- Session: `{status.get('session_id', '')}`",
            f"- Run state: `{status.get('run_state', '')}`",
            f"- Generated: `{status.get('final_report_generated_at', '')}`",
            f"- Mode: `{status.get('mode', '')}`",
            f"- Timeframe: `{status.get('timeframe', '')}`",
            "",
            "## PnL",
            "",
            f"- Account PnL: `${_format_float(live.get('account_equity_pnl_usd'))}` / `{_format_float(live.get('account_pnl_pct'))}%`",
            f"- Clean Panteon PnL: `${_format_float(live.get('clean_panteon_total_pnl_usd'))}` / `{_format_float(live.get('clean_panteon_pnl_pct'))}%`",
            f"- Panteon realized PnL: `${_format_float(live.get('panteon_realized_pnl_usd'))}`",
            f"- Panteon total PnL: `${_format_float(live.get('panteon_owned_total_pnl_usd'))}` / `{_format_float(live.get('panteon_owned_pnl_pct'))}%`",
            f"- Panteon max drawdown: `{_format_float(live.get('panteon_max_drawdown_pct'))}%`",
            "",
            "## Trading",
            "",
            f"- Real trades total: `{real_trades.get('total', live.get('real_total_trades', 0))}`",
            f"- Clean real trades total: `{real_trades.get('clean_total', live.get('clean_real_total_trades', 0))}`",
            f"- Closed: `{real_trades.get('closed', live.get('real_closed_trades', 0))}`",
            f"- Successful / unsuccessful: `{real_trades.get('successful', 0)}` / `{real_trades.get('unsuccessful', 0)}`",
            f"- Open Panteon positions: `{live.get('panteon_owned_positions_count', 0)}`",
            "",
            "## Safety",
            "",
            f"- Kill-switch reason: `{kill.get('disabled_reason', '')}`",
            f"- Peak equity tracked: `${_format_float(kill.get('peak_equity_usd'))}`",
            f"- Last message: `{status.get('message', '')}`",
        ]
        return "\n".join(lines) + "\n"

    def _ensure_output_dir(self) -> None:
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
