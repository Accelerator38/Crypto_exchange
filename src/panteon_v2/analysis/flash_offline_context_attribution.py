from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from panteon_v2.attribution import EventLog, ExecutionAttributed, PositionClosed
from panteon_v2.domain.types import Action, Regime, Signal
from panteon_v2.execution import (
    ExecutionStatus,
    FakeExchange,
    PositionTracker,
    RiskLimits,
    RiskLimitsConfig,
    SymbolHealthMonitor,
    TradeExecutor,
)
from panteon_v2.memory.performance import PerformanceMemory

from .retrodate_market_runner import write_flash_attribution_summary


@dataclass(frozen=True)
class OfflineExecutionReplay:
    rows_read: int
    replayed_signals: int
    status_counts: dict[str, int]
    execution_events: tuple[object, ...]
    position_closed_events: tuple[object, ...]

    @property
    def realized_pnl_usd(self) -> float:
        return float(
            sum(
                float(getattr(event, "realized_pnl", 0.0) or 0.0)
                for event in self.position_closed_events
            )
        )


@dataclass(frozen=True)
class OfflineContextAttributionSummary:
    path: Path
    rows_read: int
    replayed_signals: int
    status_counts: dict[str, int]
    execution_event_count: int
    position_closed_event_count: int
    realized_pnl_usd: float


def replay_offline_causal_executions(
    output_dir: str | Path,
    *,
    initial_capital: Optional[float] = None,
    risk_capital_fraction: Optional[float] = None,
    risk_max_open_positions: Optional[int] = None,
) -> OfflineExecutionReplay:
    output = Path(output_dir)
    run_config = _load_run_summary_config(output)
    capital = _float_config(
        initial_capital,
        run_config.get("initial_capital"),
        default=1000.0,
    )
    capital_fraction = _float_config(
        risk_capital_fraction,
        run_config.get("risk_capital_fraction"),
        default=0.10,
    )
    max_open_positions = _int_config(
        risk_max_open_positions,
        run_config.get("risk_max_open_positions"),
        default=8,
    )

    event_log = EventLog()
    executor = TradeExecutor(
        exchange=FakeExchange(name="RETRODATE_MARKET"),
        health=SymbolHealthMonitor(),
        risk_limits=RiskLimits(
            config=RiskLimitsConfig(
                max_open_positions=max_open_positions,
                capital_fraction=capital_fraction,
            )
        ),
        position_tracker=PositionTracker(),
        perf=PerformanceMemory(),
        event_log=event_log,
    )
    status_counts = {status.value: 0 for status in ExecutionStatus}
    rows_read = 0
    replayed_signals = 0

    for row in _iter_jsonl(output / "causal_entry_decisions.jsonl"):
        rows_read += 1
        fallback_regime = row.get("regime")
        executable_signals = row.get("executable_signals")
        if not isinstance(executable_signals, list):
            continue
        for payload in executable_signals:
            if not isinstance(payload, dict):
                continue
            signal = _signal_from_payload(payload, fallback_regime=fallback_regime)
            result = executor.execute(signal, balance_usd=capital)
            status_counts[result.status.value] = (
                int(status_counts.get(result.status.value, 0) or 0) + 1
            )
            replayed_signals += 1

    execution_events = tuple(event_log.query(event_types=[ExecutionAttributed]))
    position_closed_events = tuple(event_log.query(event_types=[PositionClosed]))
    return OfflineExecutionReplay(
        rows_read=rows_read,
        replayed_signals=replayed_signals,
        status_counts=status_counts,
        execution_events=execution_events,
        position_closed_events=position_closed_events,
    )


def write_offline_context_attribution_summary(
    output_dir: str | Path,
    *,
    filename: str = "flash_attribution_summary_offline_context.json",
    initial_capital: Optional[float] = None,
    risk_capital_fraction: Optional[float] = None,
    risk_max_open_positions: Optional[int] = None,
) -> OfflineContextAttributionSummary:
    replay = replay_offline_causal_executions(
        output_dir,
        initial_capital=initial_capital,
        risk_capital_fraction=risk_capital_fraction,
        risk_max_open_positions=risk_max_open_positions,
    )
    path = write_flash_attribution_summary(
        output_dir,
        execution_events=replay.execution_events,
        position_closed_events=replay.position_closed_events,
        filename=filename,
    )
    return OfflineContextAttributionSummary(
        path=path,
        rows_read=replay.rows_read,
        replayed_signals=replay.replayed_signals,
        status_counts=dict(replay.status_counts),
        execution_event_count=len(replay.execution_events),
        position_closed_event_count=len(replay.position_closed_events),
        realized_pnl_usd=replay.realized_pnl_usd,
    )


def _signal_from_payload(payload: Mapping[str, Any], *, fallback_regime: object) -> Signal:
    return Signal(
        id=_required_int(payload.get("id"), field_name="id"),
        bar=_int_config(payload.get("bar"), None, default=0),
        sym=str(payload.get("sym") or payload.get("symbol") or "").upper(),
        action=_parse_action(payload.get("action")),
        price=_float_config(payload.get("price"), None, default=0.0),
        regime=_parse_regime(payload.get("regime") or fallback_regime),
        by_player=str(payload.get("by_player") or "Panteon_Flash"),
        by_agent=str(payload.get("by_agent") or ""),
        position_scope=str(payload.get("position_scope") or ""),
        risk_mult=_float_config(payload.get("risk_mult"), None, default=1.0),
        close_fraction=_float_config(payload.get("close_fraction"), None, default=1.0),
        timestamp=_parse_timestamp(payload.get("timestamp")),
    )


def _parse_action(raw: object) -> Action:
    if isinstance(raw, Action):
        return raw
    if isinstance(raw, int):
        return Action(raw)
    value = str(raw or "").strip()
    if value.startswith("Action."):
        value = value.split(".", 1)[1]
    if value.isdigit():
        return Action(int(value))
    return Action[value]


def _parse_regime(raw: object) -> Regime:
    if isinstance(raw, Regime):
        return raw
    return Regime.from_string(str(raw or ""))


def _parse_timestamp(raw: object) -> datetime:
    value = str(raw or "").strip()
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload


def _load_run_summary_config(output: Path) -> dict[str, Any]:
    path = output / "run_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _required_int(raw: object, *, field_name: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"signal payload missing valid {field_name}") from exc


def _float_config(primary: object, secondary: object, *, default: float) -> float:
    for raw in (primary, secondary):
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return float(default)


def _int_config(primary: object, secondary: object, *, default: int) -> int:
    for raw in (primary, secondary):
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return int(default)
