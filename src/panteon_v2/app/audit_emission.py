"""Audit event emission helpers for production decision paths."""

from __future__ import annotations

from typing import Dict, Sequence

from ..attribution import CandidateRejected, CandidateScored
from ..domain.types import Action, MarketSnapshot
from ..selection import FlashDecision, SwitchDecision


def emit_candidate_audit_events(
    pipeline: object,
    market: MarketSnapshot,
    decision: SwitchDecision,
    *,
    decision_id: str,
    trace_id: str,
    context: Dict[str, str],
) -> None:
    selected_label = getattr(decision.new_leader, "label", "")
    for row in getattr(decision, "candidate_scores", ()) or ():
        pipeline.event_log.emit(CandidateScored(
            bar=market.bar,
            trace_id=trace_id,
            decision_id=decision_id,
            **context,
            player_label=row.label,
            rank=row.rank,
            score=row.score,
            score_source=row.score_source,
            selected_by_pantheon=(row.label == selected_label),
            has_data=row.has_data,
            closed_trades=row.closed_trades,
            signals=row.signals,
            execution_failures=row.execution_failures,
            uncertainty_penalty=row.uncertainty_penalty,
            memory_keys_read=row.memory_keys_read,
            agent_labels=row.agent_labels,
            session_score_delta=getattr(row, "session_score_delta", 0.0),
            session_pnl_pct=getattr(row, "session_pnl_pct", 0.0),
            session_underperformance_penalty=getattr(
                row,
                "session_underperformance_penalty",
                0.0,
            ),
            session_stale_penalty=getattr(row, "session_stale_penalty", 0.0),
            recent_bars=getattr(row, "recent_bars", 0),
            recent_actionable_bars=getattr(row, "recent_actionable_bars", 0),
            actionable_share=getattr(row, "actionable_share", 0.0),
            recent_filled=getattr(row, "recent_filled", 0),
            recent_pnl_usd=getattr(row, "recent_pnl_usd", 0.0),
        ))
    for row in getattr(decision, "candidate_rejections", ()) or ():
        pipeline.event_log.emit(CandidateRejected(
            bar=market.bar,
            trace_id=trace_id,
            decision_id=decision_id,
            **context,
            player_label=row.label,
            reason=row.reason,
        ))


def emit_flash_audit_events(
    pipeline: object,
    market: MarketSnapshot,
    decisions: Sequence[FlashDecision],
    *,
    decision_id: str,
    trace_id: str,
    context: Dict[str, str],
) -> None:
    for decision in decisions or ():
        symbol = str(getattr(decision, "symbol", "") or "")
        regime = market.regime_for_symbol(symbol) if symbol else market.regime
        regime_label = getattr(regime, "label", str(regime))
        symbol_context = dict(context)
        symbol_context["symbol"] = symbol
        selected_label = str(getattr(decision, "selected_actor", "") or "")
        for row in getattr(decision, "candidates", ()) or ():
            row_label = str(getattr(row, "label", "") or "")
            agent_labels = tuple(
                str(label)
                for label in (getattr(row, "agent_labels", ()) or ())
                if str(label)
            )
            memory_keys = tuple(dict.fromkeys(
                [f"{row_label}|{regime_label}"]
                + [f"{label}|{regime_label}" for label in agent_labels]
            ))
            pipeline.event_log.emit(CandidateScored(
                bar=market.bar,
                trace_id=trace_id,
                decision_id=decision_id,
                **symbol_context,
                player_label=row_label,
                rank=int(getattr(row, "rank", 0) or 0),
                score=float(getattr(row, "score", 0.0) or 0.0),
                score_source="flash_per_symbol",
                selected_by_pantheon=(row_label == selected_label),
                has_data=bool(getattr(row, "has_data", False)),
                closed_trades=int(getattr(row, "closed_trades", 0) or 0),
                signals=0 if getattr(row, "action", Action.HOLD).is_hold else 1,
                execution_failures=0,
                uncertainty_penalty=0.0,
                memory_keys_read=memory_keys,
                agent_labels=agent_labels,
            ))
            if bool(getattr(row, "rejected", False)):
                pipeline.event_log.emit(CandidateRejected(
                    bar=market.bar,
                    trace_id=trace_id,
                    decision_id=decision_id,
                    **symbol_context,
                    player_label=row_label,
                    reason=f"flash:{str(getattr(row, 'reason', '') or '')}",
                ))
