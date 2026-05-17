"""Offline Retrodate what-if analysis for Pantheon selector decisions."""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from ..domain.types import Metrics, Regime
from ..scoring import DEFAULT_SCORING, regime_score
from ..selection.composer import (
    PROFILE_BOMBERMAN_STRONG,
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_NEUTRAL_EDGE_RESEARCH,
    PROFILE_TREND_RESEARCH,
    PlayerProfile,
)


@dataclass(frozen=True)
class RetroWhatIfConfig:
    """Configuration for log-only session-aware selector what-if replay."""

    exchange: str
    event_log_path: Path | str
    results_dir: Path | str
    output_root: Path | str = Path("Results") / "Retrodate"
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None
    overlay_weight: float = 0.50
    stale_penalty: float = 0.10
    player_underperformance_weight: float = 0.35
    agent_underperformance_weight: float = 0.25
    player_overlay_weight: float = 0.25
    session_pnl_cap_pct: float = 3.0
    min_session_activity_for_overlay: int = 1
    max_decisions: Optional[int] = None
    include_png: bool = True
    apply_switch_gate: bool = True
    switch_margin: float = 0.30
    cooldown_bars: int = 30
    streak_needed: int = 2
    hard_negative: float = -0.50
    urgent_gap: float = 0.80
    include_recomposed: bool = True
    recomposed_min_agent_score: float = 0.0


DEFAULT_RECOMPOSE_PROFILES: tuple[PlayerProfile, ...] = (
    PROFILE_DEFAULT_ENSEMBLE,
    PROFILE_NEUTRAL_EDGE_RESEARCH,
    PROFILE_TREND_RESEARCH,
    PROFILE_MEAN_REV_RESEARCH,
    PROFILE_DEFENSIVE_RESEARCH,
    PROFILE_BOMBERMAN_STRONG,
)


def select_whatif_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    agent_metrics: Mapping[str, Mapping[str, Any]],
    player_metrics: Mapping[str, Mapping[str, Any]],
    config: RetroWhatIfConfig,
) -> tuple[dict, list[dict]]:
    """Return the session-aware selected row and full scored candidate list."""
    scored = [
        score_candidate_with_session_overlay(
            candidate,
            agent_metrics=agent_metrics,
            player_metrics=player_metrics,
            config=config,
        )
        for candidate in candidates
    ]
    scored.sort(
        key=lambda row: (
            -float(row["whatif_score"]),
            -float(row["base_score"]),
            int(row.get("rank") or 999_999),
            str(row["player_label"]),
        )
    )
    return (scored[0] if scored else {}, scored)


def score_candidate_with_session_overlay(
    candidate: Mapping[str, Any],
    *,
    agent_metrics: Mapping[str, Mapping[str, Any]],
    player_metrics: Mapping[str, Mapping[str, Any]],
    config: RetroWhatIfConfig,
) -> dict:
    """Score one logged candidate with current-session overlay and penalties."""
    label = _strip_v(candidate.get("player_label") or candidate.get("label") or "")
    agent_labels = [_strip_v(v) for v in _list(candidate.get("agent_labels"))]
    base_score = _finite_float(candidate.get("score"))
    player = _metric_lookup(player_metrics, label)
    agents = [_metric_lookup(agent_metrics, agent) for agent in agent_labels]
    active_agents = [item for item in agents if _session_activity(item) > 0]

    agent_session_pnls = [
        _capped_session_pnl(item, config)
        for item in active_agents
        if _has_overlay_activity(item, config)
    ]
    agent_session_pnl = _mean(agent_session_pnls)
    player_session_pnl = (
        _capped_session_pnl(player, config)
        if _has_overlay_activity(player, config)
        else 0.0
    )
    agent_session_overlay = config.overlay_weight * agent_session_pnl
    player_session_overlay = config.player_overlay_weight * player_session_pnl

    player_active = _session_activity(player) > 0
    agent_active = bool(active_agents)
    stale = bool(not player_active and not agent_active)
    stale_score_penalty = float(config.stale_penalty if stale else 0.0)
    underperformance_penalty = (
        max(0.0, -player_session_pnl) * float(config.player_underperformance_weight)
        + max(0.0, -agent_session_pnl) * float(config.agent_underperformance_weight)
    )
    whatif_score = (
        base_score
        + agent_session_overlay
        + player_session_overlay
        - stale_score_penalty
        - underperformance_penalty
    )
    return {
        "player_label": label,
        "rank": _int(candidate.get("rank"), 0),
        "base_score": base_score,
        "whatif_score": whatif_score,
        "score_delta": whatif_score - base_score,
        "selected_by_pantheon": bool(candidate.get("selected_by_pantheon")),
        "score_source": str(candidate.get("score_source") or ""),
        "has_data": bool(candidate.get("has_data", True)),
        "closed_trades": _int(candidate.get("closed_trades"), 0),
        "signals": _int(candidate.get("signals"), 0),
        "agent_labels": agent_labels,
        "agent_session_pnl_pct": agent_session_pnl,
        "player_session_pnl_pct": player_session_pnl,
        "agent_session_overlay": agent_session_overlay,
        "player_session_overlay": player_session_overlay,
        "stale_penalty": stale_score_penalty,
        "underperformance_penalty": underperformance_penalty,
        "session_closed_trades": _session_closed_trades(player),
        "session_signals": _session_signals(player),
        "agent_session_closed_trades": sum(_session_closed_trades(v) for v in agents),
        "agent_session_signals": sum(_session_signals(v) for v in agents),
        "memory_keys_read": _list(candidate.get("memory_keys_read")),
    }


def build_recomposed_candidates(
    *,
    agent_metrics: Mapping[str, Mapping[str, Any]],
    config: RetroWhatIfConfig,
    regime: str | Regime = Regime.NEUTRAL,
    profiles: Sequence[PlayerProfile] = DEFAULT_RECOMPOSE_PROFILES,
) -> list[dict]:
    """Build what-if player candidates from the full agent leaderboard."""
    regime_obj = _normalize_regime(regime)
    agent_rows = _session_aware_agent_rows(agent_metrics, config=config, regime=regime_obj)
    if not agent_rows:
        return []
    candidates: list[dict] = []
    for profile in profiles:
        eligible = [
            row for row in agent_rows
            if float(row["selector_score"]) > float(config.recomposed_min_agent_score)
        ]
        fallback_used = False
        if len(eligible) < profile.min_agents:
            eligible = list(agent_rows)
            fallback_used = True
        selected = eligible[: profile.max_agents]
        if len(selected) < profile.min_agents:
            continue
        profile_score, bias_score, affinity_adjustment = _recomposed_profile_score(
            profile,
            selected,
            regime_obj,
        )
        candidates.append(
            {
                "player_label": profile.label,
                "rank": 0,
                "base_score": profile_score,
                "whatif_score": profile_score,
                "score_delta": 0.0,
                "selected_by_pantheon": False,
                "score_source": "recomposed_agents",
                "candidate_origin": "recomposed",
                "selection_scope": "full_agent_leaderboard",
                "agent_labels": [str(row["label"]) for row in selected],
                "selector_agent_scores": selected,
                "profile_bias_score": bias_score,
                "affinity_adjustment": affinity_adjustment,
                "recomposed_fallback_used": fallback_used,
                "agent_session_pnl_pct": _mean([
                    _finite_float(row.get("session_pnl_pct")) for row in selected
                ]),
                "agent_session_overlay": _mean([
                    _finite_float(row.get("session_overlay")) for row in selected
                ]),
                "agent_session_closed_trades": sum(
                    _int(row.get("session_closed_trades"), 0) for row in selected
                ),
                "agent_session_signals": sum(
                    _int(row.get("session_signals"), 0) for row in selected
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["whatif_score"]),
            int(row.get("rank") or 999_999),
            str(row["player_label"]),
        )
    )
    for idx, row in enumerate(candidates, start=1):
        row["rank"] = idx
    return candidates


def run_retrodate_whatif(config: RetroWhatIfConfig) -> dict:
    """Run log-only what-if analysis and write Retrodate artifacts."""
    exchange = config.exchange.upper()
    event_log_path = Path(config.event_log_path)
    results_dir = Path(config.results_dir)
    output_root = Path(config.output_root)
    status = _load_json(results_dir / "status.json")
    agents_payload = _load_json(results_dir / "leaderboard_agents.json")
    players_payload = _load_json(results_dir / "leaderboard_players.json")
    session_start = _ensure_utc(config.session_start) or _infer_session_start(
        results_dir,
        status,
    )
    session_end = _ensure_utc(config.session_end)
    events = _load_decision_events(
        event_log_path,
        exchange=exchange,
        session_start=session_start,
        session_end=session_end,
    )
    decisions = _build_decision_records(events["events"])
    if config.max_decisions and config.max_decisions > 0:
        decisions = decisions[-int(config.max_decisions):]

    agent_metrics = _entries(agents_payload, "agents")
    player_metrics = _entries(players_payload, "players")
    top_session_agents = _top_session_agents(agent_metrics)
    default_regime = str(status.get("regime") or (agents_payload.get("metadata") or {}).get("regime") or "")
    analyzed = _analyze_decisions(
        decisions,
        agent_metrics=agent_metrics,
        player_metrics=player_metrics,
        top_session_agents=top_session_agents,
        config=config,
    )
    recomposed = (
        _analyze_recomposed_decisions(
            decisions,
            agent_metrics=agent_metrics,
            default_regime=default_regime,
            config=config,
        )
        if config.include_recomposed
        else []
    )
    changed = [row for row in analyzed if row.get("changed")]
    recomposed_changed = [row for row in recomposed if row.get("changed")]
    output_dir = output_root / exchange / f"{results_dir.name}_whatif"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_summary(
        exchange=exchange,
        event_log_path=event_log_path,
        results_dir=results_dir,
        output_dir=output_dir,
        session_start=session_start,
        session_end=session_end,
        events=events,
        decisions=analyzed,
        changed=changed,
        recomposed_decisions=recomposed,
        recomposed_changed=recomposed_changed,
        top_session_agents=top_session_agents,
        config=config,
    )
    _write_artifacts(
        output_dir=output_dir,
        status=status,
        agents_payload=agents_payload,
        players_payload=players_payload,
        summary=summary,
        decisions=analyzed,
        recomposed_decisions=recomposed,
        include_png=config.include_png,
    )
    return summary


def run_retrodate_whatif_for_exchange(
    exchange: str,
    *,
    project_root: Optional[Path] = None,
    include_png: bool = True,
    max_decisions: Optional[int] = None,
) -> int:
    """CLI helper used by Retrostart_MEXC.py and Retrostart_BITGET.py."""
    project_root = project_root or Path(__file__).resolve().parents[3]
    exchange = exchange.upper()
    results_dir = _latest_results_dir(project_root / "Results" / exchange)
    if results_dir is None:
        print(f"[retrodate] no Results/{exchange} session directory found")
        return 2
    event_log_path = project_root / "logs" / f"v2_{exchange.lower()}_events.jsonl"
    if not event_log_path.exists():
        print(f"[retrodate] event log not found: {event_log_path}")
        return 2
    try:
        summary = run_retrodate_whatif(
            RetroWhatIfConfig(
                exchange=exchange,
                event_log_path=event_log_path,
                results_dir=results_dir,
                output_root=project_root / "Results" / "Retrodate",
                include_png=include_png,
                max_decisions=max_decisions,
            )
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[retrodate] failed: {exc}")
        return 1
    decisions = summary["decisions"]
    recomposed = summary.get("recomposed_decisions") or {}
    print(
        "[retrodate] "
        f"{exchange}: decisions={decisions['count']} "
        f"changed={decisions['changed_count']} "
        f"changed_pct={decisions['changed_pct']:.2f}% "
        f"recomposed_changed={recomposed.get('changed_count', 0)}"
    )
    print(f"[retrodate] artifacts: {summary['output_dir']}")
    return 0


def _analyze_decision(
    decision: Mapping[str, Any],
    *,
    agent_metrics: Mapping[str, Mapping[str, Any]],
    player_metrics: Mapping[str, Mapping[str, Any]],
    top_session_agents: Sequence[Mapping[str, Any]],
    config: RetroWhatIfConfig,
) -> dict:
    actual = _actual_selected(decision)
    whatif_best, scored = select_whatif_candidate(
        decision.get("candidates") or [],
        agent_metrics=agent_metrics,
        player_metrics=player_metrics,
        config=config,
    )
    whatif_best_label = str(whatif_best.get("player_label") or "")
    by_label = {str(row["player_label"]): row for row in scored}
    actual_row = by_label.get(actual, {})
    return {
        "decision_id": str(decision.get("decision_id") or ""),
        "trace_id": str(decision.get("trace_id") or ""),
        "timestamp": str(decision.get("timestamp") or ""),
        "bar": _int(decision.get("bar"), 0),
        "exchange": str(decision.get("exchange") or config.exchange.upper()),
        "regime": str(decision.get("regime") or ""),
        "actual_selected": actual,
        "whatif_selected": whatif_best_label,
        "whatif_best": whatif_best_label,
        "changed": bool(actual and whatif_best_label and actual != whatif_best_label),
        "classification": "choice_session_overlay"
        if actual and whatif_best_label and actual != whatif_best_label
        else "unchanged",
        "actual_base_score": _finite_float(actual_row.get("base_score")),
        "actual_whatif_score": _finite_float(actual_row.get("whatif_score")),
        "whatif_base_score": _finite_float(whatif_best.get("base_score")),
        "whatif_score": _finite_float(whatif_best.get("whatif_score")),
        "whatif_score_margin": (
            _finite_float(whatif_best.get("whatif_score"))
            - _finite_float(actual_row.get("whatif_score"))
        ),
        "candidate_count": len(scored),
        "candidates": scored,
        "top_session_agents": list(top_session_agents[:8]),
    }


def _analyze_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    agent_metrics: Mapping[str, Mapping[str, Any]],
    player_metrics: Mapping[str, Mapping[str, Any]],
    top_session_agents: Sequence[Mapping[str, Any]],
    config: RetroWhatIfConfig,
) -> list[dict]:
    state = _SwitchState()
    analyzed: list[dict] = []
    for decision in decisions:
        if not decision.get("candidates"):
            continue
        row = _analyze_decision(
            decision,
            agent_metrics=agent_metrics,
            player_metrics=player_metrics,
            top_session_agents=top_session_agents,
            config=config,
        )
        if config.apply_switch_gate:
            _apply_switch_gate(row, state, config)
        analyzed.append(row)
    return analyzed


def _analyze_recomposed_decisions(
    decisions: Sequence[Mapping[str, Any]],
    *,
    agent_metrics: Mapping[str, Mapping[str, Any]],
    default_regime: str,
    config: RetroWhatIfConfig,
) -> list[dict]:
    state = _SwitchState()
    analyzed: list[dict] = []
    for decision in decisions:
        if not decision.get("candidates"):
            continue
        regime = decision.get("regime") or default_regime or "neutral"
        candidates = build_recomposed_candidates(
            agent_metrics=agent_metrics,
            config=config,
            regime=regime,
        )
        if not candidates:
            continue
        actual = _actual_selected(decision)
        best = candidates[0]
        by_label = {str(row["player_label"]): row for row in candidates}
        actual_row = by_label.get(actual, {})
        logged_agents = _actual_logged_agent_labels(decision, actual)
        actual_recomposed_agents = list(actual_row.get("agent_labels") or [])
        added_agents = sorted(set(actual_recomposed_agents) - set(logged_agents))
        removed_agents = sorted(set(logged_agents) - set(actual_recomposed_agents))
        row = {
            "decision_id": str(decision.get("decision_id") or ""),
            "trace_id": str(decision.get("trace_id") or ""),
            "timestamp": str(decision.get("timestamp") or ""),
            "bar": _int(decision.get("bar"), 0),
            "exchange": str(decision.get("exchange") or config.exchange.upper()),
            "regime": _normalize_regime(regime).label,
            "actual_selected": actual,
            "whatif_best": str(best.get("player_label") or ""),
            "whatif_selected": str(best.get("player_label") or ""),
            "actual_whatif_score": _finite_float(actual_row.get("whatif_score")),
            "whatif_score": _finite_float(best.get("whatif_score")),
            "whatif_score_margin": (
                _finite_float(best.get("whatif_score"))
                - _finite_float(actual_row.get("whatif_score"))
            ),
            "candidate_count": len(candidates),
            "candidates": candidates,
            "recomposed_candidates": candidates,
            "actual_logged_agent_labels": logged_agents,
            "actual_recomposed_agent_labels": actual_recomposed_agents,
            "actual_profile_agent_set_changed": bool(added_agents or removed_agents),
            "actual_profile_added_agents": added_agents,
            "actual_profile_removed_agents": removed_agents,
        }
        row["changed"] = bool(
            row["actual_selected"]
            and row["whatif_selected"]
            and row["actual_selected"] != row["whatif_selected"]
        )
        row["classification"] = (
            "recomposed_choice_session_overlay" if row["changed"] else "unchanged"
        )
        if config.apply_switch_gate:
            _apply_switch_gate(row, state, config)
        _copy_recomposed_aliases(row)
        analyzed.append(row)
    return analyzed


def _copy_recomposed_aliases(row: dict) -> None:
    row["recomposed_best"] = row.get("whatif_best", "")
    row["recomposed_selected"] = row.get("whatif_selected", "")
    row["recomposed_switch_reason"] = row.get("whatif_switch_reason", "")
    row["recomposed_switched"] = row.get("whatif_switched", False)
    row["recomposed_score"] = row.get("whatif_score", 0.0)
    row["recomposed_score_margin"] = row.get("whatif_score_margin", 0.0)
    if row.get("changed"):
        row["classification"] = "recomposed_choice_session_overlay"
    elif row.get("recomposed_best") != row.get("recomposed_selected"):
        row["classification"] = "recomposed_unchanged_switch_gate"


def _session_aware_agent_rows(
    agent_metrics: Mapping[str, Mapping[str, Any]],
    *,
    config: RetroWhatIfConfig,
    regime: Regime,
) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for raw_label, metrics in agent_metrics.items():
        label = _strip_v(raw_label)
        if not label or label in seen:
            continue
        seen.add(label)
        if _is_quarantined(metrics):
            continue
        base_score = _leaderboard_agent_base_score(metrics, regime)
        session_pnl = (
            _capped_session_pnl(metrics, config)
            if _has_overlay_activity(metrics, config)
            else 0.0
        )
        session_overlay = float(config.overlay_weight) * session_pnl
        stale_penalty = (
            float(config.stale_penalty)
            if _session_activity(metrics) <= 0
            else 0.0
        )
        underperformance_penalty = (
            max(0.0, -session_pnl) * float(config.agent_underperformance_weight)
        )
        selector_score = (
            base_score
            + session_overlay
            - stale_penalty
            - underperformance_penalty
        )
        rows.append(
            {
                "label": label,
                "base_score": base_score,
                "selector_score": selector_score,
                "session_pnl_pct": session_pnl,
                "session_overlay": session_overlay,
                "stale_penalty": stale_penalty,
                "underperformance_penalty": underperformance_penalty,
                "closed_trades": _int(metrics.get("closed_trades"), 0),
                "signals": _int(metrics.get("signals"), 0),
                "session_closed_trades": _session_closed_trades(metrics),
                "session_signals": _session_signals(metrics),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["selector_score"]),
            -float(row["session_pnl_pct"]),
            -int(row["session_closed_trades"]),
            str(row["label"]),
        )
    )
    return rows


def _recomposed_profile_score(
    profile: PlayerProfile,
    selected: Sequence[Mapping[str, Any]],
    regime: Regime,
) -> tuple[float, float, float]:
    adjusted_scores = []
    bias_total = 0.0
    for row in selected:
        label = str(row.get("label") or "")
        bias = float(profile.bias.get(label, 0.0))
        bias_total += bias
        adjusted_scores.append(_finite_float(row.get("selector_score")) + bias)
    base = _mean(adjusted_scores)
    bias_score = bias_total / len(selected) if selected else 0.0
    affinity_adjustment = 0.0
    if profile.affinity is not None:
        affinity_adjustment = 0.05 if profile.affinity == regime else -0.02
    return base + affinity_adjustment, bias_score, affinity_adjustment


def _leaderboard_agent_base_score(metrics: Mapping[str, Any], regime: Regime) -> float:
    return regime_score(
        _metrics_from_leaderboard(metrics),
        regime,
        config=DEFAULT_SCORING,
    )


def _metrics_from_leaderboard(metrics: Mapping[str, Any]) -> Metrics:
    closed = _int(metrics.get("closed_trades"), 0)
    wins = min(_int(metrics.get("wins"), 0), closed)
    losses = min(_int(metrics.get("losses"), 0), max(0, closed - wins))
    return Metrics(
        pnl_pct=_finite_float(metrics.get("pnl_pct")),
        closed_trades=closed,
        entries=_int(metrics.get("entries"), 0),
        signals=_int(metrics.get("signals"), 0),
        wins=wins,
        losses=losses,
        sharpe=_finite_float(metrics.get("sharpe")),
        max_dd_pct=_finite_float(metrics.get("max_drawdown_pct")),
        blocked_signals=_int(metrics.get("blocked_signals"), 0),
        rejected_signals=_int(metrics.get("rejected_signals"), 0),
        pending_signals=_int(metrics.get("pending_signals"), 0),
        execution_failures=_int(metrics.get("execution_failures"), 0),
    )


class _SwitchState:
    def __init__(self) -> None:
        self.current = ""
        self.last_switch_bar = -10**9
        self.streak_label = ""
        self.streak_count = 0


def _apply_switch_gate(
    row: dict,
    state: _SwitchState,
    config: RetroWhatIfConfig,
) -> None:
    candidates = row.get("candidates") or []
    by_label = {str(candidate.get("player_label")): candidate for candidate in candidates}
    best_label = str(row.get("whatif_best") or "")
    actual = str(row.get("actual_selected") or "")
    bar = _int(row.get("bar"), 0)
    if not best_label:
        row["whatif_selected"] = state.current
        row["whatif_switch_reason"] = "no candidates"
        row["whatif_switched"] = False
        row["changed"] = bool(actual and state.current and actual != state.current)
        row["classification"] = "choice_session_overlay" if row["changed"] else "unchanged"
        return

    if not state.current:
        state.current = best_label
        state.last_switch_bar = bar
        state.streak_label = best_label
        state.streak_count = 1
        _set_switch_result(row, state.current, actual, True, "bootstrap")
        return

    if state.current not in by_label:
        previous = state.current
        state.current = best_label
        state.last_switch_bar = bar
        state.streak_label = ""
        state.streak_count = 0
        _set_switch_result(row, state.current, actual, previous != state.current, "current not in candidates")
        return

    current_score = _finite_float(by_label[state.current].get("whatif_score"))
    best_score = _finite_float(by_label.get(best_label, {}).get("whatif_score"))
    margin = best_score - current_score
    row["whatif_current_before"] = state.current
    row["whatif_current_score"] = current_score
    row["whatif_best_margin_vs_current"] = margin

    if best_label == state.current:
        state.streak_label = ""
        state.streak_count = 0
        _set_switch_result(row, state.current, actual, False, "current is still best")
        return

    urgent = current_score <= config.hard_negative or margin >= config.urgent_gap
    ready_by_margin = margin >= config.switch_margin
    if not ready_by_margin and not urgent:
        state.streak_label = ""
        state.streak_count = 0
        _set_switch_result(row, state.current, actual, False, "switch gate: margin too small")
        return

    cooldown_passed = bar - state.last_switch_bar >= config.cooldown_bars
    if not urgent and not cooldown_passed:
        _set_switch_result(row, state.current, actual, False, "switch gate: cooldown not passed")
        return

    if urgent:
        previous = state.current
        state.current = best_label
        state.last_switch_bar = bar
        state.streak_label = ""
        state.streak_count = 0
        _set_switch_result(row, state.current, actual, previous != state.current, "switch gate: urgent")
        return

    if state.streak_label == best_label:
        state.streak_count += 1
    else:
        state.streak_label = best_label
        state.streak_count = 1
    if state.streak_count < config.streak_needed:
        _set_switch_result(
            row,
            state.current,
            actual,
            False,
            f"switch gate: awaiting streak {state.streak_count}/{config.streak_needed}",
        )
        return

    previous = state.current
    state.current = best_label
    state.last_switch_bar = bar
    state.streak_label = ""
    state.streak_count = 0
    _set_switch_result(row, state.current, actual, previous != state.current, "switch gate: streak confirmed")


def _set_switch_result(
    row: dict,
    selected: str,
    actual: str,
    switched: bool,
    reason: str,
) -> None:
    row["whatif_selected"] = selected
    row["whatif_switched"] = switched
    row["whatif_switch_reason"] = reason
    row["changed"] = bool(actual and selected and actual != selected)
    if row["changed"]:
        row["classification"] = "choice_session_overlay"
    elif row.get("whatif_best") and row.get("whatif_best") != selected:
        row["classification"] = "unchanged_switch_gate"
    else:
        row["classification"] = "unchanged"


def _actual_selected(decision: Mapping[str, Any]) -> str:
    for candidate in decision.get("candidates") or []:
        if candidate.get("selected_by_pantheon"):
            return _strip_v(candidate.get("player_label") or candidate.get("label") or "")
    leader = decision.get("leader_selected")
    if isinstance(leader, Mapping):
        label = leader.get("player_label")
        if label:
            return _strip_v(label)
    candidates = list(decision.get("candidates") or [])
    if not candidates:
        return ""
    candidates.sort(key=lambda row: (_int(row.get("rank"), 999_999), -_finite_float(row.get("score"))))
    return _strip_v(candidates[0].get("player_label") or candidates[0].get("label") or "")


def _actual_logged_agent_labels(
    decision: Mapping[str, Any],
    actual_label: str,
) -> list[str]:
    for candidate in decision.get("candidates") or []:
        label = _strip_v(candidate.get("player_label") or candidate.get("label") or "")
        if label == actual_label:
            return [_strip_v(v) for v in _list(candidate.get("agent_labels"))]
    return []


def _build_decision_records(events: Sequence[Mapping[str, Any]]) -> list[dict]:
    decisions: "OrderedDict[str, dict]" = OrderedDict()
    for idx, event in enumerate(events):
        event_type = _event_type(event)
        decision_id = str(event.get("decision_id") or event.get("trace_id") or "")
        if not decision_id:
            decision_id = f"bar-{event.get('bar', 0)}-line-{idx}"
        if event_type == "DecisionStarted":
            item = decisions.setdefault(decision_id, {"candidates": []})
            item.update(
                {
                    "decision_id": decision_id,
                    "trace_id": str(event.get("trace_id") or ""),
                    "timestamp": str(event.get("timestamp") or ""),
                    "bar": _int(event.get("bar"), 0),
                    "exchange": str(event.get("exchange") or ""),
                    "symbol": str(event.get("symbol") or ""),
                    "timeframe": str(event.get("timeframe") or ""),
                    "regime": str(event.get("regime") or ""),
                    "candidate_labels": _list(event.get("candidate_labels")),
                }
            )
            continue
        if event_type == "CandidateScored":
            item = decisions.setdefault(
                decision_id,
                {
                    "decision_id": decision_id,
                    "trace_id": str(event.get("trace_id") or ""),
                    "timestamp": str(event.get("timestamp") or ""),
                    "bar": _int(event.get("bar"), 0),
                    "candidates": [],
                },
            )
            item["candidates"].append(dict(event))
            continue
        if event_type == "LeaderSelected":
            item = decisions.setdefault(
                decision_id,
                {
                    "decision_id": decision_id,
                    "trace_id": str(event.get("trace_id") or ""),
                    "timestamp": str(event.get("timestamp") or ""),
                    "bar": _int(event.get("bar"), 0),
                    "candidates": [],
                },
            )
            item["leader_selected"] = dict(event)
    return list(decisions.values())


def _load_decision_events(
    path: Path,
    *,
    exchange: str,
    session_start: Optional[datetime],
    session_end: Optional[datetime],
) -> dict:
    out: list[dict] = []
    malformed = 0
    total = 0
    if not path.exists():
        return {"events": [], "total_lines": 0, "malformed_lines": 0}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            total += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(event, dict):
                continue
            event_type = _event_type(event)
            if event_type not in {"DecisionStarted", "CandidateScored", "LeaderSelected"}:
                continue
            ts = _parse_dt(event.get("timestamp"))
            if session_start and ts and ts < session_start:
                continue
            if session_end and ts and ts > session_end:
                continue
            event_exchange = str(event.get("exchange") or "").upper()
            if event_type == "DecisionStarted" and event_exchange and event_exchange != exchange:
                continue
            out.append(event)
    return {"events": out, "total_lines": total, "malformed_lines": malformed}


def _build_summary(
    *,
    exchange: str,
    event_log_path: Path,
    results_dir: Path,
    output_dir: Path,
    session_start: Optional[datetime],
    session_end: Optional[datetime],
    events: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    changed: Sequence[Mapping[str, Any]],
    recomposed_decisions: Sequence[Mapping[str, Any]],
    recomposed_changed: Sequence[Mapping[str, Any]],
    top_session_agents: Sequence[Mapping[str, Any]],
    config: RetroWhatIfConfig,
) -> dict:
    count = len(decisions)
    changed_count = len(changed)
    recomposed_count = len(recomposed_decisions)
    recomposed_changed_count = len(recomposed_changed)
    best_diff = [
        row for row in decisions
        if row.get("actual_selected")
        and row.get("whatif_best")
        and row.get("actual_selected") != row.get("whatif_best")
    ]
    recomposed_best_diff = [
        row for row in recomposed_decisions
        if row.get("actual_selected")
        and row.get("recomposed_best")
        and row.get("actual_selected") != row.get("recomposed_best")
    ]
    actual_profile_changed = [
        row for row in recomposed_decisions
        if row.get("actual_profile_agent_set_changed")
    ]
    added_counter: dict[str, int] = {}
    removed_counter: dict[str, int] = {}
    for row in actual_profile_changed:
        for label in row.get("actual_profile_added_agents") or []:
            added_counter[str(label)] = added_counter.get(str(label), 0) + 1
        for label in row.get("actual_profile_removed_agents") or []:
            removed_counter[str(label)] = removed_counter.get(str(label), 0) + 1
    return {
        "schema": "retrodate_whatif_v1",
        "exchange": exchange,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_log_path": str(event_log_path),
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "session_start": session_start.isoformat() if session_start else "",
        "session_end": session_end.isoformat() if session_end else "",
        "config": {
            "overlay_weight": config.overlay_weight,
            "stale_penalty": config.stale_penalty,
            "player_underperformance_weight": config.player_underperformance_weight,
            "agent_underperformance_weight": config.agent_underperformance_weight,
            "player_overlay_weight": config.player_overlay_weight,
            "session_pnl_cap_pct": config.session_pnl_cap_pct,
            "min_session_activity_for_overlay": config.min_session_activity_for_overlay,
            "apply_switch_gate": config.apply_switch_gate,
            "switch_margin": config.switch_margin,
            "cooldown_bars": config.cooldown_bars,
            "streak_needed": config.streak_needed,
            "hard_negative": config.hard_negative,
            "urgent_gap": config.urgent_gap,
            "include_recomposed": config.include_recomposed,
            "recomposed_min_agent_score": config.recomposed_min_agent_score,
        },
        "events": {
            "total_lines": int(events.get("total_lines", 0) or 0),
            "loaded_decision_events": int(len(events.get("events") or [])),
            "malformed_lines": int(events.get("malformed_lines", 0) or 0),
        },
        "decisions": {
            "count": count,
            "changed_count": changed_count,
            "unchanged_count": count - changed_count,
            "changed_pct": (changed_count / count * 100.0) if count else 0.0,
        },
        "recomposed_decisions": {
            "count": recomposed_count,
            "changed_count": recomposed_changed_count,
            "unchanged_count": recomposed_count - recomposed_changed_count,
            "changed_pct": (
                recomposed_changed_count / recomposed_count * 100.0
                if recomposed_count
                else 0.0
            ),
        },
        "actual_profile_recomposition": {
            "agent_set_changed_count": len(actual_profile_changed),
            "agent_set_changed_pct": (
                len(actual_profile_changed) / recomposed_count * 100.0
                if recomposed_count
                else 0.0
            ),
            "top_added_agents": _counter_rows(added_counter),
            "top_removed_agents": _counter_rows(removed_counter),
        },
        "top_session_agents": list(top_session_agents[:12]),
        "changed_episodes_top20": _changed_episode_rows(changed[:20]),
        "best_diff_episodes_top20": _changed_episode_rows(best_diff[:20]),
        "recomposed_changed_episodes_top20": _recomposed_episode_rows(recomposed_changed[:20]),
        "recomposed_best_diff_episodes_top20": _recomposed_episode_rows(recomposed_best_diff[:20]),
    }


def _write_artifacts(
    *,
    output_dir: Path,
    status: Mapping[str, Any],
    agents_payload: Mapping[str, Any],
    players_payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    recomposed_decisions: Sequence[Mapping[str, Any]],
    include_png: bool,
) -> None:
    _write_json(output_dir / "whatif_summary.json", summary)
    _write_json(output_dir / "whatif_changed_episodes_top20.json", summary["changed_episodes_top20"])
    _write_json(output_dir / "whatif_best_diff_episodes_top20.json", summary["best_diff_episodes_top20"])
    _write_json(output_dir / "whatif_recomposed_changed_episodes_top20.json", summary["recomposed_changed_episodes_top20"])
    _write_json(output_dir / "whatif_recomposed_best_diff_episodes_top20.json", summary["recomposed_best_diff_episodes_top20"])
    _write_jsonl(output_dir / "whatif_decisions.jsonl", decisions)
    _write_jsonl(output_dir / "whatif_recomposed_decisions.jsonl", recomposed_decisions)
    _write_candidates_csv(output_dir / "whatif_candidates.csv", decisions)
    _write_recomposed_candidates_csv(
        output_dir / "whatif_recomposed_candidates.csv",
        recomposed_decisions,
    )
    _write_json(output_dir / "status.json", _status_with_summary(status, summary))
    _write_json(output_dir / "leaderboard_agents.json", agents_payload)
    _write_json(output_dir / "leaderboard_players.json", players_payload)
    (output_dir / "dashboard.txt").write_text(
        _render_dashboard_text(summary, decisions),
        encoding="utf-8",
    )
    (output_dir / "trading.log").write_text(
        _render_trading_log(summary, decisions),
        encoding="utf-8",
    )
    if include_png:
        _write_pngs(output_dir, status, agents_payload, players_payload)


def _write_candidates_csv(path: Path, decisions: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "decision_id",
        "timestamp",
        "bar",
        "actual_selected",
        "whatif_selected",
        "whatif_best",
        "whatif_switch_reason",
        "changed",
        "player_label",
        "rank",
        "base_score",
        "whatif_score",
        "score_delta",
        "agent_session_pnl_pct",
        "player_session_pnl_pct",
        "agent_session_overlay",
        "player_session_overlay",
        "stale_penalty",
        "underperformance_penalty",
        "session_closed_trades",
        "agent_session_closed_trades",
        "agent_labels",
        "selected_by_pantheon",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            for candidate in decision.get("candidates") or []:
                row = {field: "" for field in fields}
                row.update(
                    {
                        "decision_id": decision.get("decision_id", ""),
                        "timestamp": decision.get("timestamp", ""),
                        "bar": decision.get("bar", 0),
                        "actual_selected": decision.get("actual_selected", ""),
                        "whatif_selected": decision.get("whatif_selected", ""),
                        "whatif_best": decision.get("whatif_best", ""),
                        "whatif_switch_reason": decision.get("whatif_switch_reason", ""),
                        "changed": decision.get("changed", False),
                        "agent_labels": "+".join(candidate.get("agent_labels") or []),
                    }
                )
                for field in fields:
                    if field in candidate and field not in {"agent_labels"}:
                        row[field] = candidate[field]
                writer.writerow(row)


def _write_recomposed_candidates_csv(
    path: Path,
    decisions: Sequence[Mapping[str, Any]],
) -> None:
    fields = [
        "decision_id",
        "timestamp",
        "bar",
        "actual_selected",
        "recomposed_selected",
        "recomposed_best",
        "recomposed_switch_reason",
        "changed",
        "player_label",
        "rank",
        "whatif_score",
        "agent_session_pnl_pct",
        "agent_session_overlay",
        "agent_session_closed_trades",
        "agent_labels",
        "profile_bias_score",
        "affinity_adjustment",
        "recomposed_fallback_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            for candidate in decision.get("recomposed_candidates") or []:
                row = {field: "" for field in fields}
                row.update(
                    {
                        "decision_id": decision.get("decision_id", ""),
                        "timestamp": decision.get("timestamp", ""),
                        "bar": decision.get("bar", 0),
                        "actual_selected": decision.get("actual_selected", ""),
                        "recomposed_selected": decision.get("recomposed_selected", ""),
                        "recomposed_best": decision.get("recomposed_best", ""),
                        "recomposed_switch_reason": decision.get("recomposed_switch_reason", ""),
                        "changed": decision.get("changed", False),
                        "agent_labels": "+".join(candidate.get("agent_labels") or []),
                    }
                )
                for field in fields:
                    if field in candidate and field != "agent_labels":
                        row[field] = candidate[field]
                writer.writerow(row)


def _status_with_summary(status: Mapping[str, Any], summary: Mapping[str, Any]) -> dict:
    out = dict(status or {})
    out["run_state"] = "retrodate_whatif"
    out["feed_status"] = "offline"
    out["retrodate_whatif"] = {
        "generated_at": summary.get("generated_at"),
        "session_start": summary.get("session_start"),
        "decisions": summary.get("decisions"),
        "output_dir": summary.get("output_dir"),
    }
    return out


def _render_dashboard_text(
    summary: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    decision_stats = summary.get("decisions") or {}
    recomposed_stats = summary.get("recomposed_decisions") or {}
    profile_recomp = summary.get("actual_profile_recomposition") or {}
    lines = [
        "RETRODATE WHAT-IF SELECTOR DASHBOARD",
        f"exchange: {summary.get('exchange')}",
        f"session_start: {summary.get('session_start')}",
        f"source_results: {summary.get('results_dir')}",
        (
            "decisions: "
            f"{decision_stats.get('count', 0)}; "
            f"changed: {decision_stats.get('changed_count', 0)} "
            f"({float(decision_stats.get('changed_pct', 0.0)):.2f}%)"
        ),
        (
            "recomposed decisions: "
            f"{recomposed_stats.get('count', 0)}; "
            f"changed: {recomposed_stats.get('changed_count', 0)} "
            f"({float(recomposed_stats.get('changed_pct', 0.0)):.2f}%)"
        ),
        (
            "actual-profile agent set changed: "
            f"{profile_recomp.get('agent_set_changed_count', 0)} "
            f"({float(profile_recomp.get('agent_set_changed_pct', 0.0)):.2f}%)"
        ),
        "",
        "Top current-session agents:",
    ]
    for row in summary.get("top_session_agents") or []:
        lines.append(
            "  "
            f"{row.get('label')}: session_pnl={float(row.get('session_pnl_pct', 0.0)):+.4f}% "
            f"closed={row.get('session_closed_trades', 0)} "
            f"signals={row.get('session_signals', 0)}"
        )
    lines.extend(["", "Changed episodes top 20:"])
    for row in summary.get("changed_episodes_top20") or []:
        lines.append(
            "  "
            f"bar={row.get('bar')} {row.get('actual_selected')} -> "
            f"{row.get('whatif_selected')} "
            f"margin={float(row.get('whatif_score_margin', 0.0)):+.4f}"
        )
    if not summary.get("changed_episodes_top20"):
        lines.append("  none")
    lines.extend(["", "Raw best differs before switch gate top 20:"])
    for row in summary.get("best_diff_episodes_top20") or []:
        lines.append(
            "  "
            f"bar={row.get('bar')} actual={row.get('actual_selected')} "
            f"best={row.get('whatif_selected')} "
            f"margin={float(row.get('whatif_score_margin', 0.0)):+.4f}"
        )
    if not summary.get("best_diff_episodes_top20"):
        lines.append("  none")
    lines.extend(["", "Recomposed changed episodes top 20:"])
    for row in summary.get("recomposed_changed_episodes_top20") or []:
        lines.append(
            "  "
            f"bar={row.get('bar')} {row.get('actual_selected')} -> "
            f"{row.get('recomposed_selected')} "
            f"best={row.get('recomposed_best')} "
            f"reason={row.get('recomposed_switch_reason')}"
        )
    if not summary.get("recomposed_changed_episodes_top20"):
        lines.append("  none")
    lines.extend(["", "Recomposed raw best differs top 20:"])
    for row in summary.get("recomposed_best_diff_episodes_top20") or []:
        lines.append(
            "  "
            f"bar={row.get('bar')} actual={row.get('actual_selected')} "
            f"best={row.get('recomposed_best')} "
            f"selected={row.get('recomposed_selected')}"
        )
    if not summary.get("recomposed_best_diff_episodes_top20"):
        lines.append("  none")
    lines.extend(["", "Actual-profile recomposition top added agents:"])
    for row in profile_recomp.get("top_added_agents") or []:
        lines.append(f"  {row.get('label')}: {row.get('count')}")
    if not profile_recomp.get("top_added_agents"):
        lines.append("  none")
    lines.extend(["", "Actual-profile recomposition top removed agents:"])
    for row in profile_recomp.get("top_removed_agents") or []:
        lines.append(f"  {row.get('label')}: {row.get('count')}")
    if not profile_recomp.get("top_removed_agents"):
        lines.append("  none")
    lines.extend(["", "Latest decisions:"])
    for row in list(decisions)[-10:]:
        lines.append(
            "  "
            f"bar={row.get('bar')} actual={row.get('actual_selected')} "
            f"whatif={row.get('whatif_selected')} changed={row.get('changed')}"
        )
    return "\n".join(lines) + "\n"


def _render_trading_log(
    summary: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    now = str(summary.get("generated_at") or "")
    lines = [
        f"{now} RETRODATE what-if started exchange={summary.get('exchange')}",
        (
            f"{now} source event_log={summary.get('event_log_path')} "
            f"results_dir={summary.get('results_dir')}"
        ),
        f"{now} jsonl malformed_lines={summary.get('events', {}).get('malformed_lines', 0)}",
    ]
    for row in decisions:
        if row.get("changed"):
            lines.append(
                f"{row.get('timestamp')} CHOICE_OVERLAY bar={row.get('bar')} "
                f"decision_id={row.get('decision_id')} "
                f"actual={row.get('actual_selected')} "
                f"whatif={row.get('whatif_selected')} "
                f"margin={float(row.get('whatif_score_margin', 0.0)):+.6f}"
            )
    recomposed = summary.get("recomposed_decisions") or {}
    lines.append(
        f"{now} RECOMPOSED summary decisions={recomposed.get('count', 0)} "
        f"changed={recomposed.get('changed_count', 0)}"
    )
    lines.append(f"{now} RETRODATE what-if finished output_dir={summary.get('output_dir')}")
    return "\n".join(lines) + "\n"


def _write_pngs(
    output_dir: Path,
    status: Mapping[str, Any],
    agents_payload: Mapping[str, Any],
    players_payload: Mapping[str, Any],
) -> None:
    try:
        from ..dashboards.png_renderer import write_operator_pngs

        write_operator_pngs(
            str(output_dir),
            status=status,
            agents_payload=agents_payload,
            players_payload=players_payload,
        )
    except Exception as exc:
        (output_dir / "png_render_error.txt").write_text(str(exc), encoding="utf-8")


def _changed_episode_rows(changed: Sequence[Mapping[str, Any]]) -> list[dict]:
    rows = []
    for item in changed:
        target = item.get("whatif_selected", "")
        if item.get("classification") == "unchanged_switch_gate" and item.get("whatif_best"):
            target = item.get("whatif_best")
        rows.append(
            {
                "decision_id": item.get("decision_id", ""),
                "timestamp": item.get("timestamp", ""),
                "bar": item.get("bar", 0),
                "actual_selected": item.get("actual_selected", ""),
                "whatif_selected": target,
                "actual_base_score": item.get("actual_base_score", 0.0),
                "actual_whatif_score": item.get("actual_whatif_score", 0.0),
                "whatif_base_score": item.get("whatif_base_score", 0.0),
                "whatif_score": item.get("whatif_score", 0.0),
                "whatif_score_margin": item.get("whatif_score_margin", 0.0),
                "classification": item.get("classification", ""),
            }
        )
    rows.sort(key=lambda row: -abs(_finite_float(row.get("whatif_score_margin"))))
    return rows


def _recomposed_episode_rows(changed: Sequence[Mapping[str, Any]]) -> list[dict]:
    rows = []
    for item in changed:
        rows.append(
            {
                "decision_id": item.get("decision_id", ""),
                "timestamp": item.get("timestamp", ""),
                "bar": item.get("bar", 0),
                "actual_selected": item.get("actual_selected", ""),
                "recomposed_best": item.get("recomposed_best", ""),
                "recomposed_selected": item.get("recomposed_selected", ""),
                "recomposed_switch_reason": item.get("recomposed_switch_reason", ""),
                "recomposed_score": item.get("recomposed_score", 0.0),
                "recomposed_score_margin": item.get("recomposed_score_margin", 0.0),
                "classification": item.get("classification", ""),
            }
        )
    rows.sort(key=lambda row: -abs(_finite_float(row.get("recomposed_score_margin"))))
    return rows


def _counter_rows(counter: Mapping[str, int], *, limit: int = 12) -> list[dict]:
    return [
        {"label": label, "count": count}
        for label, count in sorted(
            counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:limit]
    ]


def _top_session_agents(
    agent_metrics: Mapping[str, Mapping[str, Any]],
    *,
    limit: int = 20,
) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for label, metrics in agent_metrics.items():
        plain = _strip_v(label)
        if plain in seen:
            continue
        seen.add(plain)
        session_pnl = _finite_float(metrics.get("session_pnl_pct"))
        session_closed = _session_closed_trades(metrics)
        session_signals = _session_signals(metrics)
        if session_closed <= 0 and session_signals <= 0 and abs(session_pnl) <= 1e-12:
            continue
        rows.append(
            {
                "label": plain,
                "session_pnl_pct": session_pnl,
                "session_closed_trades": session_closed,
                "session_signals": session_signals,
                "is_quarantined": bool(
                    metrics.get("is_quarantined")
                    or str(metrics.get("status") or "").lower() == "quarantine"
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            bool(row.get("is_quarantined")),
            -float(row.get("session_pnl_pct", 0.0)),
            -int(row.get("session_closed_trades", 0)),
            str(row.get("label")),
        )
    )
    return rows[:limit]


def _entries(payload: Mapping[str, Any], key: str) -> dict[str, Mapping[str, Any]]:
    raw = (payload or {}).get(key)
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for label, values in raw.items():
        if isinstance(values, Mapping):
            out[_strip_v(label)] = values
            out[f"V_{_strip_v(label)}"] = values
    return out


def _metric_lookup(
    metrics: Mapping[str, Mapping[str, Any]],
    label: str,
) -> Mapping[str, Any]:
    plain = _strip_v(label)
    return metrics.get(plain) or metrics.get(f"V_{plain}") or {}


def _has_overlay_activity(
    metrics: Mapping[str, Any],
    config: RetroWhatIfConfig,
) -> bool:
    if not metrics:
        return False
    activity = _session_activity(metrics)
    return (
        activity >= max(0, int(config.min_session_activity_for_overlay))
        or abs(_finite_float(metrics.get("session_pnl_pct"))) > 1e-12
    )


def _session_activity(metrics: Mapping[str, Any]) -> int:
    return (
        _session_closed_trades(metrics)
        + _int(metrics.get("session_entries"), 0)
        + _session_signals(metrics)
    )


def _session_closed_trades(metrics: Mapping[str, Any]) -> int:
    return _int(metrics.get("session_closed_trades"), 0)


def _session_signals(metrics: Mapping[str, Any]) -> int:
    return _int(metrics.get("session_signals"), 0)


def _capped_session_pnl(
    metrics: Mapping[str, Any],
    config: RetroWhatIfConfig,
) -> float:
    value = _finite_float(metrics.get("session_pnl_pct"))
    cap = abs(float(config.session_pnl_cap_pct))
    if cap <= 0:
        return value
    return max(-cap, min(cap, value))


def _is_quarantined(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("is_quarantined")
        or str(metrics.get("status") or "").strip().lower() == "quarantine"
    )


def _normalize_regime(value: str | Regime | Any) -> Regime:
    if isinstance(value, Regime):
        return value
    try:
        return Regime(int(value))
    except (TypeError, ValueError):
        return Regime.from_string(str(value or ""))


def _latest_results_dir(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    sessions = [p for p in results_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not sessions:
        return None
    sessions.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
    return sessions[0]


def _infer_session_start(
    results_dir: Path,
    status: Mapping[str, Any],
) -> Optional[datetime]:
    status_ts = _parse_dt((status or {}).get("timestamp"))
    uptime = _parse_uptime((status or {}).get("uptime"))
    if status_ts and uptime:
        return status_ts - uptime
    parsed = _parse_results_dir_timestamp(results_dir.name)
    if parsed:
        return parsed
    return None


def _parse_results_dir_timestamp(name: str) -> Optional[datetime]:
    base = name.replace("_v2", "")
    try:
        return datetime.strptime(base, "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_uptime(value: Any) -> Optional[timedelta]:
    text = str(value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
        if len(parts) == 2:
            return timedelta(hours=int(parts[0]), minutes=int(parts[1]))
    except ValueError:
        return None
    return None


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("_type") or event.get("event_type") or event.get("type") or "")


def _parse_dt(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _ensure_utc(dt)


def _ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _strip_v(value: Any) -> str:
    text = str(value or "").strip()
    return text[2:] if text.startswith("V_") else text


def _list(value: Any) -> list:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value or 0.0)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
