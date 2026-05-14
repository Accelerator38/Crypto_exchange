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
import time
import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

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
        self._assets_history: List[float] = []
        self._session_perf_baseline = self._capture_session_perf_baseline()
        # Создаём output dir
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        # Открываем trading.log (append)
        self._log_path = os.path.join(config.output_dir, config.trading_log_filename)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Panteon v2 started: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"  exchange: {pipeline.exchange_name}\n")
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
        cfg = OutputWriterConfig(output_dir=output_dir, **kwargs)
        return cls(pipeline, cfg)

    def write(self, step: StepResult) -> None:
        """Вызывается после каждого bar (через on_step callback)."""
        self._last_step = step
        self._log_step(step)

        # каждый bar — status.json
        if step.bar % self._config.write_every_bars == 0:
            self._write_status(step, run_state="running", feed_status="active")

        # реже — leaderboards + dashboard
        if step.bar % self._config.full_snapshot_every == 0:
            self._write_leaderboards()
            self._write_dashboard()

    def write_heartbeat(
        self,
        *,
        run_state: str = "idle",
        feed_status: str = "waiting",
        message: str = "",
    ) -> None:
        """Update operator-visible files when no market bar has arrived yet."""
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
        parts = [
            ts,
            f"bar={step.bar}",
            f"regime={step.regime.label}",
            f"leader={step.leader or '-'}",
            f"balance=${self._current_balance:.2f}",
            f"positions={position_counts['n_positions']}",
            f"signals={step.n_signals}",
            f"filled={step.n_filled}",
            f"rejected={step.n_rejected}",
            f"blocked={step.n_blocked}",
            f"shadow_signals={step.n_shadow_signals}",
            f"shadow_filled={step.n_shadow_filled}",
        ]
        if step.n_filtered_real_signals:
            parts.append(f"filtered={step.n_filtered_real_signals}")
            parts.append(f"stale_closes={step.n_stale_close_signals}")
            parts.append(f"duplicate_opens={step.n_duplicate_open_signals}")
            parts.append(f"rate_limited_opens={step.n_rate_limited_open_signals}")
        if step.blocked_reasons:
            parts.append(f"blocked_reasons={self._format_reason_counts(step.blocked_reasons)}")
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
        self._assets_history.append(total_assets)
        if len(self._assets_history) > 720:
            self._assets_history = self._assets_history[-720:]

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
        blocked_reasons = dict(step.blocked_reasons) if step is not None else {}
        leader = step.leader if step is not None else None
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
        panteon_owned_total_pnl = (
            panteon_owned_realized_pnl
            + position_pnl["panteon_owned_unrealized_pnl_usd"]
        )
        panteon_owned_pnl_pct = (
            panteon_owned_total_pnl / self._initial_capital * 100.0
            if self._initial_capital > 0 else 0.0
        )
        live_session = {
            "account_equity_pnl_usd": total_assets - self._initial_capital,
            "account_realized_pnl_usd": realized_pnl,
            "account_pnl_pct": pnl_pct,
            "panteon_equity_pnl_usd": panteon_owned_total_pnl,
            "panteon_realized_pnl_usd": panteon_owned_realized_pnl,
            "panteon_pnl_pct": panteon_owned_pnl_pct,
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
            "leader_virtual_session_pnl_pct": leader_session["pnl_pct"],
            "leader_virtual_session_closed_trades": leader_session["closed_trades"],
            "leader_virtual_session_win_rate": leader_session["win_rate"],
        }
        shadow = {
            "actors": step.n_shadow_actors if step is not None else 0,
            "signals": step.n_shadow_signals if step is not None else 0,
            "filled": step.n_shadow_filled if step is not None else 0,
            "rejected": step.n_shadow_rejected if step is not None else 0,
            "blocked": step.n_shadow_blocked if step is not None else 0,
            "last_summary": getattr(self._pipeline, "shadow_last_summary", None),
        }

        data = {
            "version":          "v2",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "exchange":         self._pipeline.exchange_name,
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
            "blocked_reasons_bar": blocked_reasons,
            "n_positions":      position_counts["n_positions"],
            "tracked_positions_count": position_counts["tracked_positions_count"],
            "exchange_positions_count": position_counts["exchange_positions_count"],
            "current_leader":   leader,
            "live_session":     live_session,
            "live_state_sync":  getattr(self._pipeline, "live_state_sync", None),
            "shadow":           shadow,
            "quarantined":      sorted(self._pipeline.qm.all_quarantined()),
            "open_positions":   open_positions,
            "ledger_closed_count": self._pipeline.ledger.closed_count,
        }
        self._last_status_data = data
        path = os.path.join(self._config.output_dir, self._config.status_filename)
        self._write_json_atomic(path, data)

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

    def _position_counts(self, *, tracked_count: Optional[int] = None) -> Dict[str, int]:
        if tracked_count is None:
            tracked_count = len(self._tracked_open_positions())
        exchange_count = self._exchange_positions_count()
        return {
            "n_positions": max(int(tracked_count), int(exchange_count)),
            "tracked_positions_count": int(tracked_count),
            "exchange_positions_count": int(exchange_count),
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
                        "win_rate":       rm.win_rate,
                    }
            agents["V_" + label] = {
                "pnl_pct":        metrics_agg.pnl_pct,
                "session_pnl_pct": session["pnl_pct"],
                "closed_trades":  metrics_agg.closed_trades,
                "session_closed_trades": session["closed_trades"],
                "win_rate":       metrics_agg.win_rate,
                "session_win_rate": session["win_rate"],
                "sharpe":         metrics_agg.sharpe,
                "max_drawdown_pct": metrics_agg.max_dd_pct,
                "is_quarantined": self._pipeline.qm.is_quarantined(label),
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
                        "win_rate":       rm.win_rate,
                    }
            real_trades = trade_counts.get(label, 0)
            real_wins = win_counts.get(label, 0)
            players["V_" + label] = {
                "pnl_pct":          metrics_agg.pnl_pct,
                "session_pnl_pct":  session["pnl_pct"],
                "closed_trades":    metrics_agg.closed_trades,
                "session_closed_trades": session["closed_trades"],
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
            write_operator_pngs(
                self._config.output_dir,
                status=self._last_status_data,
                agents_payload=self._last_agents_payload,
                players_payload=self._last_players_payload,
            )
        except Exception:
            log.exception("visual dashboard render failed")

    def _write_dashboard_html(self, text: str) -> None:
        status_path = self._config.status_filename
        escaped = html.escape(text)
        now = datetime.now(timezone.utc).isoformat()
        image_names = (
            "dashboard_latest.png",
            "shadow_dashboard.png",
            "regime_dashboard.png",
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

    @property
    def output_dir(self) -> str:
        return self._config.output_dir

    def close(self) -> None:
        """Финальная фиксация state перед выходом."""
        try:
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
