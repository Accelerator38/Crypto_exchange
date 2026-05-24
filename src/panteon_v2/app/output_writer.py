"""OutputWriter — периодическая запись output на диск.

Заменяет v1 функциональность: status.json + leaderboard_*.json +
dashboard ASCII + trading.log. Файлы пишутся в той же папке как v1:
  Results/{EXCHANGE}/{TIMESTAMP}/

Каждые N баров (или N секунд) делается snapshot всего state.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import html
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
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
    dashboard_filename:   str = "dashboard.txt"
    dashboard_html:       str = "dashboard.html"
    events_jsonl:         str = "events.jsonl"
    causal_entry_decisions_jsonl: str = "causal_entry_decisions.jsonl"
    compact_causal_entry_decisions: bool = False
    compact_causal_entry_top_rejected_candidates: int = 5
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
        self._session_perf_baseline = self._capture_session_perf_baseline()
        # Создаём output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        # Открываем trading.log (append)
        self._log_path = os.path.join(config.output_dir, config.trading_log_filename)
        self._causal_entry_path = os.path.join(
            config.output_dir,
            config.causal_entry_decisions_jsonl,
        )
        Path(self._causal_entry_path).touch(exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Panteon v2 started: {datetime.now(timezone.utc).isoformat()}\n")
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
        parts = [
            ts,
            f"bar={step.bar}",
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

    def _compact_causal_entry_decision(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        decisions = out.get("flash_decisions")
        if isinstance(decisions, list):
            out["flash_decisions"] = [
                self._compact_flash_decision(
                    decision,
                    top_rejected_candidates=max(
                        0,
                        int(self._config.compact_causal_entry_top_rejected_candidates),
                    ),
                )
                for decision in decisions
            ]
        return out

    @staticmethod
    def _compact_flash_decision(
        decision: Any,
        *,
        top_rejected_candidates: int = 5,
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
        out["top_rejected_candidates"] = [
            OutputWriter._compact_candidate_audit(candidate)
            for candidate in sorted(
                rejected_candidates,
                key=lambda item: int(item.get("rank") or 0),
            )[:top_rejected_candidates]
        ]

        selected_actor = str(decision.get("selected_actor") or "")
        actor_type = str(decision.get("actor_type") or "")
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
            if not (is_selected or is_original_selected):
                continue
            key = (label, candidate_type)
            if key in seen:
                continue
            seen.add(key)
            compact_candidates.append(dict(candidate))

        out["candidates"] = compact_candidates
        return out

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
            "lookback_return_z",
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

        pnl_pct = (
            (total_assets - self._initial_capital) / self._initial_capital * 100.0
            if self._initial_capital > 0 else 0.0
        )
        self._assets_history = self._append_curve_value(self._assets_history, total_assets)

        # Открытые позиции
        open_positions = self._tracked_open_positions()
        position_counts = self._position_counts(tracked_count=len(open_positions))

        uptime_sec = time.time() - self._start_time
        h = int(uptime_sec // 3600)
        m = int((uptime_sec % 3600) // 60)
        s = int(uptime_sec % 60)
        bar = step.bar if step is not None else 0
        regime = step.regime.label if step is not None else "unknown"
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
        position_pnl = self._position_pnl_summary(leader=leader)
        player_pnl = self._pipeline.ledger.total_pnl_by_player()
        external_realized_pnl = sum(
            pnl for player, pnl in player_pnl.items()
            if _is_external_player(player)
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
        panteon_realized_equity = self._initial_capital + panteon_owned_realized_pnl
        panteon_equity = self._initial_capital + panteon_owned_total_pnl
        self._panteon_realized_equity_history = self._append_curve_value(
            self._panteon_realized_equity_history,
            panteon_realized_equity,
        )
        self._panteon_equity_history = self._append_curve_value(
            self._panteon_equity_history,
            panteon_equity,
        )
        panteon_realized_max_dd_pct = self._curve_max_drawdown_pct(
            self._panteon_realized_equity_history
        )
        panteon_max_dd_pct = self._curve_max_drawdown_pct(self._panteon_equity_history)
        panteon_owned_pnl_pct = (
            panteon_owned_total_pnl / self._initial_capital * 100.0
            if self._initial_capital > 0 else 0.0
        )
        real_trades = self._real_trade_summary(position_pnl=position_pnl)
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
            "panteon_owned_realized_pnl_usd": panteon_owned_realized_pnl,
            "panteon_owned_unrealized_pnl_usd": position_pnl["panteon_owned_unrealized_pnl_usd"],
            "panteon_owned_total_pnl_usd": panteon_owned_total_pnl,
            "panteon_owned_pnl_pct": panteon_owned_pnl_pct,
            "panteon_owned_positions_count": position_pnl["panteon_owned_positions_count"],
            "external_realized_pnl_usd": external_realized_pnl,
            "external_unrealized_pnl_usd": position_pnl["external_unrealized_pnl_usd"],
            "external_positions_count": position_pnl["external_positions_count"],
            "leader_realized_pnl_usd": leader_realized_pnl,
            "leader_unrealized_pnl_usd": position_pnl["leader_unrealized_pnl_usd"],
            "leader_positions_count": position_pnl["leader_positions_count"],
            "handoff_positions_count": position_pnl["handoff_positions_count"],
            "comparison_scope": "panteon_owned",
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
            "external_closed_trades": real_trades["external_closed"],
            "external_unresolved_trades": real_trades["external_unresolved"],
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
        decision_debug = self._decision_debug(
            step=step,
            shadow_bar=shadow_bar,
            shadow_session=shadow_session,
            blocked_reasons=blocked_reasons,
        )

        data = {
            "version":          "v2",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "exchange":         self._pipeline.exchange_name,
            "run_id":           getattr(self._pipeline, "run_id", ""),
            "session_id":       getattr(self._pipeline, "session_id", ""),
            "mode":             getattr(self._pipeline, "mode", ""),
            "timeframe":        getattr(self._pipeline, "timeframe", ""),
            "run_state":        run_state,
            "feed_status":      feed_status,
            "message":          message,
            "uptime":           f"{h:02d}:{m:02d}:{s:02d}",
            "bar_count":        bar,
            "regime":           regime,
            "initial_capital":  self._initial_capital,
            "current_balance":  self._current_balance,
            "futures_equity_usd": account_snapshot.get("futures_equity", self._current_balance),
            "available_balance_usd": account_snapshot.get("available_balance", self._current_balance),
            "spot_assets_usd": account_snapshot.get("spot_assets", 0.0),
            "total_assets_usd": total_assets,
            "assets_curve":     list(self._assets_history),
            "equity_curve":     list(self._panteon_equity_history),
            "panteon_equity_usd": panteon_equity,
            "panteon_realized_equity_usd": panteon_realized_equity,
            "panteon_equity_curve": list(self._panteon_equity_history),
            "panteon_realized_equity_curve": list(self._panteon_realized_equity_history),
            "panteon_max_drawdown_pct": panteon_max_dd_pct,
            "panteon_realized_max_drawdown_pct": panteon_realized_max_dd_pct,
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
            "live_state_sync":  getattr(self._pipeline, "live_state_sync", None),
            "shadow":           shadow,
            "flash":            flash,
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
                "step_error": "",
            })
            return debug
        selected_leader = step.selected_leader or step.leader or ""
        executed_leader = step.executed_leader or step.leader or ""
        flash_payload = self._flash_status_payload(step)
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
            "step_error": step.error or "",
        })
        return debug

    def _flash_status_payload(self, step: Optional[StepResult]) -> Dict[str, object]:
        causal = dict(getattr(step, "causal_decision", {}) or {}) if step is not None else {}
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

    def _tracked_open_positions(self) -> Dict[str, dict]:
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
            unrealized = float(getattr(exchange_pos, "unrealized_pnl", 0.0) or 0.0)
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

    def _position_pnl_summary(self, *, leader: Optional[str]) -> Dict[str, float]:
        summary = {
            "panteon_owned_unrealized_pnl_usd": 0.0,
            "external_unrealized_pnl_usd": 0.0,
            "leader_unrealized_pnl_usd": 0.0,
            "panteon_owned_positions_count": 0,
            "external_positions_count": 0,
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
            unrealized = float(getattr(exchange_pos, "unrealized_pnl", 0.0) or 0.0)
            if _is_external_player(owner):
                summary["external_unrealized_pnl_usd"] += unrealized
                summary["external_positions_count"] += 1
                continue
            summary["panteon_owned_unrealized_pnl_usd"] += unrealized
            summary["panteon_owned_positions_count"] += 1
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
        external_closed = 0
        try:
            closed = self._pipeline.ledger.realized_attributions()
        except Exception:
            closed = []
        for attr in closed:
            if _is_external_player(getattr(attr, "by_player", "")):
                external_closed += 1
                continue
            panteon_closed += 1
            if float(getattr(attr, "realized_pnl", 0.0) or 0.0) > 0:
                successful += 1
            else:
                unsuccessful += 1
        unresolved = int(position_pnl.get("panteon_owned_positions_count", 0) or 0)
        external_unresolved = int(position_pnl.get("external_positions_count", 0) or 0)
        return {
            "total": panteon_closed + unresolved,
            "closed": panteon_closed,
            "successful": successful,
            "unsuccessful": unsuccessful,
            "unresolved": unresolved,
            "external_closed": external_closed,
            "external_unresolved": external_unresolved,
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

        # AGENTS: registered agent labels from PerformanceMemory
        agents: Dict[str, dict] = {}
        agent_labels = set(self._pipeline.registry.all_labels())
        agent_labels.update(
            str(label)
            for label in (getattr(self._pipeline, "shadow_agent_labels", ()) or ())
            if str(label)
        )
        for label in sorted(agent_labels):
            metrics_agg = self._pipeline.perf.get(label)
            if not metrics_agg.has_data:
                continue
            session = self._session_metrics_for(label, metrics_agg)
            per_regime = {}
            for r in (Regime.BULLISH, Regime.BEARISH, Regime.NEUTRAL, Regime.CRASH):
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
            agents["V_" + label] = {
                "pnl_pct":        metrics_agg.pnl_pct,
                "session_pnl_pct": session["pnl_pct"],
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
                "per_regime":     per_regime,
            }

        # PLAYERS: profile labels from PerformanceMemory plus real ledger overlay.
        players: Dict[str, dict] = {}
        player_pnl = self._pipeline.ledger.total_pnl_by_player()
        trade_counts = self._pipeline.ledger.trade_counts_by_player()
        win_counts = self._pipeline.ledger.win_counts_by_player()
        player_labels = {profile.label for profile in self._pipeline.profiles}
        player_labels.update(player_pnl.keys())
        for label in sorted(player_labels):
            metrics_agg = self._pipeline.perf.get(label)
            if not metrics_agg.has_data and label not in player_pnl:
                continue
            session = self._session_metrics_for(label, metrics_agg)
            per_regime = {}
            for r in (Regime.BULLISH, Regime.BEARISH, Regime.NEUTRAL, Regime.CRASH):
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
            players["V_" + label] = {
                "pnl_pct":          metrics_agg.pnl_pct,
                "session_pnl_pct":  session["pnl_pct"],
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
                "per_regime":       per_regime,
            }

        agents_meta = {
            "schema":    "v2",
            "exchange":  self._pipeline.exchange_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime":    "—",  # фиксируется по последнему step, ниже
        }
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

    def _write_dashboard(self) -> None:
        """ASCII dashboard через TextRenderer."""
        try:
            main = self._pipeline.renderer.build_main(timeline_max=20)
            text = TextRenderer().render_main(main)
            path = os.path.join(self._config.output_dir, self._config.dashboard_filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self._write_visual_dashboards()
            self._write_dashboard_html(text)
        except Exception:
            log.exception("dashboard render failed")

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
        if not latest_dir:
            return
        exchange = self._latest_exchange_suffix()
        selected = {
            "dashboard_latest.png": f"dashboard_latest_{exchange}.png",
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
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            try:
                shutil.copyfile(src, tmp)
                os.replace(tmp, dest)
            except Exception:
                log.exception("latest dashboard publish failed: %s -> %s", src, dest)
                try:
                    tmp.unlink()
                except OSError:
                    pass

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
</style>
</head>
<body>
<header>
<strong>Panteon v2</strong>
<span class="muted">updated {html.escape(now)}</span>
<span class="muted">status: <a href="{html.escape(status_path)}">{html.escape(status_path)}</a></span>
</header>
<main>
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
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Panteon v2 stopped: {datetime.now(timezone.utc).isoformat()}\n")
                f.write(f"{'='*70}\n")
        except Exception:
            log.exception("close failed")

    def _ensure_output_dir(self) -> None:
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
