"""Post-run monitoring helpers for genetics degradation gates."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional


def build_genetics_degradation_report(
    status: Mapping[str, Any],
    leaderboard_agents: Mapping[str, Any],
    *,
    label_prefixes: Iterable[str] = ("Genetics",),
    shadow_events: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    prefixes = tuple(str(prefix) for prefix in label_prefixes if str(prefix))
    shadow_diagnostics = _shadow_event_diagnostics(shadow_events or (), prefixes)
    decisions = {
        str(item.get("label", "")): dict(item)
        for item in (status.get("degradation_gate", {}) or {}).get("last_decisions", []) or []
        if _is_monitored(str(item.get("label", "")), prefixes)
    }
    records = {
        str(label): dict(payload or {})
        for label, payload in (status.get("quarantine_records") or {}).items()
        if _is_monitored(str(label), prefixes)
    }
    leaderboard_rows = _leaderboard_rows(leaderboard_agents, prefixes)
    labels = sorted(set(decisions) | set(records) | set(leaderboard_rows))

    rows: List[Dict[str, Any]] = []
    for label in labels:
        decision = decisions.get(label, {})
        leaderboard = leaderboard_rows.get(label, {})
        record = records.get(label, {})
        shadow = shadow_diagnostics.get(label, {})
        quarantine_reason = str(
            record.get("reason")
            or leaderboard.get("quarantine_reason")
            or ""
        )
        is_quarantined = bool(record) or bool(leaderboard.get("is_quarantined", False))
        gate_should_disable = bool(decision.get("should_disable", False))
        preexisting = is_quarantined and quarantine_reason and not quarantine_reason.startswith(
            "degradation_gate"
        )

        rows.append({
            "label": label,
            "is_quarantined": is_quarantined,
            "quarantine_reason": quarantine_reason,
            "preexisting_quarantine": preexisting,
            "gate_should_disable": gate_should_disable,
            "gate_reasons": list(decision.get("reasons") or []),
            "session_pnl_pct": _float(
                decision.get("session_pnl_pct", leaderboard.get("session_pnl_pct", 0.0))
            ),
            "session_closed_trades": _int(
                decision.get(
                    "session_closed_trades",
                    leaderboard.get("session_closed_trades", 0),
                )
            ),
            "session_signals": _int(
                decision.get("session_signals", leaderboard.get("session_signals", 0))
            ),
            "session_execution_failures": _int(
                decision.get(
                    "session_execution_failures",
                    decision.get("session_rejected_signals", 0),
                )
            ),
            "session_unexecuted_signals": _int(
                decision.get(
                    "session_unexecuted_signals",
                    decision.get("session_execution_failures", 0),
                )
            ),
            "session_blocked_signals": _int(decision.get("session_blocked_signals", 0)),
            "session_rejected_signals": _int(decision.get("session_rejected_signals", 0)),
            "session_max_drawdown_pct": _float(
                decision.get("session_max_drawdown_pct", leaderboard.get("max_drawdown_pct", 0.0))
            ),
            "execution_failure_rate": _float(decision.get("execution_failure_rate", 0.0)),
            "blocked_signal_rate": _float(decision.get("blocked_signal_rate", 0.0)),
            "shadow_event_bars": _int(shadow.get("bars", 0)),
            "shadow_event_signals": _int(shadow.get("signals", 0)),
            "shadow_event_filled": _int(shadow.get("filled", 0)),
            "shadow_event_rejected": _int(shadow.get("rejected", 0)),
            "shadow_event_blocked": _int(shadow.get("blocked", 0)),
            "shadow_rejected_reasons": _reason_rows(
                shadow.get("rejected_reasons", Counter())
            ),
            "shadow_blocked_reasons": _reason_rows(
                shadow.get("blocked_reasons", Counter())
            ),
        })

    summary = {
        "monitored_labels": len(rows),
        "quarantined_count": sum(1 for row in rows if row["is_quarantined"]),
        "gate_disable_count": sum(1 for row in rows if row["gate_should_disable"]),
        "preexisting_quarantine_count": sum(1 for row in rows if row["preexisting_quarantine"]),
        "max_execution_failure_rate": max(
            (row["execution_failure_rate"] for row in rows),
            default=0.0,
        ),
        "max_blocked_signal_rate": max(
            (row["blocked_signal_rate"] for row in rows),
            default=0.0,
        ),
        "min_session_pnl_pct": min(
            (row["session_pnl_pct"] for row in rows),
            default=0.0,
        ),
        "top_shadow_blocked_reasons": _top_shadow_blocked_reasons(rows),
    }
    warnings = _warnings(summary)
    return {
        "schema": "genetics_degradation_monitoring_v1",
        "summary": summary,
        "warnings": warnings,
        "labels": rows,
        "gate_config": dict((status.get("degradation_gate", {}) or {}).get("config") or {}),
    }


def _leaderboard_rows(
    leaderboard_agents: Mapping[str, Any],
    prefixes: tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_label, payload in (leaderboard_agents.get("agents") or {}).items():
        label = str(raw_label)
        if label.startswith("V_"):
            label = label[2:]
        if not _is_monitored(label, prefixes):
            continue
        out[label] = dict(payload or {})
    return out


def _is_monitored(label: str, prefixes: tuple[str, ...]) -> bool:
    if not label:
        return False
    if not prefixes:
        return True
    return any(label.startswith(prefix) for prefix in prefixes)


def _shadow_event_diagnostics(
    shadow_events: Iterable[Any],
    prefixes: tuple[str, ...],
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for event in shadow_events:
        if _event_type(event) != "ShadowActorUpdated":
            continue
        agent_outcomes = list(_iter_agent_outcomes(_event_get(event, "agent_outcomes", ())))
        if agent_outcomes:
            bar = _event_get(event, "bar", None)
            for label, signals, filled, rejected, blocked in agent_outcomes:
                if not _is_monitored(label, prefixes):
                    continue
                bucket = _shadow_bucket(buckets, label)
                if bar is not None:
                    bucket["bars"].add(bar)
                bucket["signals"] += signals
                bucket["filled"] += filled
                bucket["rejected"] += rejected
                bucket["blocked"] += blocked
                _add_labeled_reason_counts(
                    bucket["rejected_reasons"],
                    _event_get(event, "agent_rejected_reasons", ()),
                    label=label,
                    fallback_count=rejected,
                )
                _add_labeled_reason_counts(
                    bucket["blocked_reasons"],
                    _event_get(event, "agent_blocked_reasons", ()),
                    label=label,
                    fallback_count=blocked,
                )
            continue

        if str(_event_get(event, "actor_type", "")) != "agent":
            continue
        label = str(_event_get(event, "actor_label", ""))
        if not _is_monitored(label, prefixes):
            continue
        bucket = _shadow_bucket(buckets, label)
        bar = _event_get(event, "bar", None)
        if bar is not None:
            bucket["bars"].add(bar)
        bucket["signals"] += _int(_event_get(event, "signals", 0))
        bucket["filled"] += _int(_event_get(event, "filled", 0))
        rejected = _int(_event_get(event, "rejected", 0))
        blocked = _int(_event_get(event, "blocked", 0))
        bucket["rejected"] += rejected
        bucket["blocked"] += blocked
        _add_reason_counts(
            bucket["rejected_reasons"],
            _event_get(event, "rejected_reasons", ()),
            fallback_count=rejected,
        )
        _add_reason_counts(
            bucket["blocked_reasons"],
            _event_get(event, "blocked_reasons", ()),
            fallback_count=blocked,
        )

    out: Dict[str, Dict[str, Any]] = {}
    for label, bucket in buckets.items():
        out[label] = {
            "bars": len(bucket["bars"]),
            "signals": bucket["signals"],
            "filled": bucket["filled"],
            "rejected": bucket["rejected"],
            "blocked": bucket["blocked"],
            "rejected_reasons": bucket["rejected_reasons"],
            "blocked_reasons": bucket["blocked_reasons"],
        }
    return out


def _shadow_bucket(buckets: Dict[str, Dict[str, Any]], label: str) -> Dict[str, Any]:
    return buckets.setdefault(label, {
        "bars": set(),
        "signals": 0,
        "filled": 0,
        "rejected": 0,
        "blocked": 0,
        "rejected_reasons": Counter(),
        "blocked_reasons": Counter(),
    })


def _event_type(event: Any) -> str:
    if isinstance(event, Mapping):
        return str(event.get("_type") or event.get("event_type") or "")
    return type(event).__name__


def _event_get(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(key, default)
    return getattr(event, key, default)


def _add_reason_counts(
    counter: Counter,
    raw_reasons: Any,
    *,
    fallback_count: int = 0,
) -> None:
    added = False
    for reason, count in _iter_reason_counts(raw_reasons):
        if count <= 0:
            continue
        counter[reason] += count
        added = True
    if not added and fallback_count > 0:
        counter["unavailable"] += fallback_count


def _add_labeled_reason_counts(
    counter: Counter,
    raw_reasons: Any,
    *,
    label: str,
    fallback_count: int = 0,
) -> None:
    added = False
    for item_label, reason, count in _iter_labeled_reason_counts(raw_reasons):
        if item_label != label or count <= 0:
            continue
        counter[reason] += count
        added = True
    if not added and fallback_count > 0:
        counter["unavailable"] += fallback_count


def _iter_reason_counts(raw_reasons: Any) -> Iterable[tuple[str, int]]:
    for item in raw_reasons or ():
        reason: Any = None
        count: Any = 0
        if isinstance(item, Mapping):
            reason = item.get("reason")
            count = item.get("count")
        else:
            try:
                reason = item[0]
                count = item[1]
            except (TypeError, IndexError):
                continue
        reason_text = str(reason or "").strip()
        if not reason_text:
            continue
        yield reason_text, _int(count)


def _iter_labeled_reason_counts(raw_reasons: Any) -> Iterable[tuple[str, str, int]]:
    for item in raw_reasons or ():
        label: Any = None
        reason: Any = None
        count: Any = 0
        if isinstance(item, Mapping):
            label = item.get("label") or item.get("agent_label")
            reason = item.get("reason")
            count = item.get("count")
        else:
            try:
                label = item[0]
                reason = item[1]
                count = item[2]
            except (TypeError, IndexError):
                continue
        label_text = str(label or "").strip()
        reason_text = str(reason or "").strip()
        if not label_text or not reason_text:
            continue
        yield label_text, reason_text, _int(count)


def _iter_agent_outcomes(raw_outcomes: Any) -> Iterable[tuple[str, int, int, int, int]]:
    for item in raw_outcomes or ():
        label: Any = None
        signals = filled = rejected = blocked = 0
        if isinstance(item, Mapping):
            label = item.get("label") or item.get("agent_label")
            signals = item.get("signals", 0)
            filled = item.get("filled", 0)
            rejected = item.get("rejected", 0)
            blocked = item.get("blocked", 0)
        else:
            try:
                label = item[0]
                signals = item[1]
                filled = item[2]
                rejected = item[3]
                blocked = item[4]
            except (TypeError, IndexError):
                continue
        label_text = str(label or "").strip()
        if not label_text:
            continue
        yield (
            label_text,
            _int(signals),
            _int(filled),
            _int(rejected),
            _int(blocked),
        )


def _reason_rows(counter: Counter) -> List[Dict[str, Any]]:
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _top_shadow_blocked_reasons(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        label = str(row.get("label", ""))
        for reason_row in row.get("shadow_blocked_reasons") or []:
            out.append({
                "label": label,
                "reason": str(reason_row.get("reason", "")),
                "count": _int(reason_row.get("count", 0)),
            })
    return sorted(out, key=lambda item: (-item["count"], item["label"], item["reason"]))[:10]


def _warnings(summary: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    if int(summary.get("preexisting_quarantine_count", 0)) > 0:
        warnings.append(
            "old-memory quarantine overlaps monitored genetics labels; reset or "
            "version memory by genome before treating the run as fresh promotion evidence"
        )
    if (
        int(summary.get("gate_disable_count", 0)) > 0
        and float(summary.get("max_execution_failure_rate", 0.0)) >= 0.5
    ):
        warnings.append(
            "degradation gate observed high execution/actor failure rate; inspect "
            "AgentVoteFailed events or adapter contracts before enabling live allocation"
        )
    if (
        int(summary.get("gate_disable_count", 0)) > 0
        and float(summary.get("max_blocked_signal_rate", 0.0)) >= 0.9
    ):
        warnings.append(
            "degradation gate observed high blocked-signal rate; inspect risk/position "
            "constraints and agent action cadence before enabling live allocation"
        )
    return warnings


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
