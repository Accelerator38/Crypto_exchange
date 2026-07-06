from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EXCHANGES = ("MEXC", "BITGET")
FEE_KEYS = (
    "fees_usd",
    "fee_usd",
    "total_fees_usd",
    "total_fees",
    "fees",
    "fee",
    "commission_usd",
    "commission_usdt",
    "commission",
    "cumulative_fee",
)
FUNDING_KEYS = ("funding_usd", "total_funding_usd", "total_funding", "funding")
SLIPPAGE_KEYS = ("slippage_usd", "total_slippage_usd", "total_slippage", "slippage")
COST_NESTED_KEYS = ("trade", "fill", "order", "execution", "info")


def build_canary_summary(
    *,
    results_root: str | Path,
    reports_dir: str | Path,
    exchanges: Sequence[str] = DEFAULT_EXCHANGES,
    now: datetime | None = None,
    lookback_minutes: float = 360.0,
    require_positive_expectancy: bool = True,
    calibration_only: bool = False,
    actor_overrides: Mapping[str, Any] | None = None,
    terminal_denied_context_signal_keys: Sequence[str] | None = None,
    candidate_policy: Mapping[str, Any] | None = None,
    execution_smoke: bool = False,
) -> dict[str, Any]:
    current = _aware_utc(now)
    root = Path(results_root)
    report_root = Path(reports_dir)
    exchange_payloads: dict[str, dict[str, Any]] = {}
    for exchange in exchanges:
        exchange_key = str(exchange or "").strip().upper()
        if not exchange_key:
            continue
        exchange_payloads[exchange_key] = _build_exchange_canary(
            root,
            exchange_key,
            now=current,
            lookback_minutes=lookback_minutes,
            require_positive_expectancy=bool(require_positive_expectancy),
        )

    summary = {
        "generated_at": current.isoformat(),
        "expectancy_gate_required": bool(require_positive_expectancy),
        "calibration_only": bool(calibration_only),
        "execution_smoke": bool(execution_smoke),
        "actor_overrides": _clean_metadata_mapping(actor_overrides),
        "terminal_denied_context_signal_keys": _clean_string_list(
            terminal_denied_context_signal_keys
        ),
        "candidate_policy": _clean_json_mapping(candidate_policy),
        "passed": bool(exchange_payloads) and all(item.get("passed") for item in exchange_payloads.values()),
        "exchanges": exchange_payloads,
        "cross_exchange_diagnostics": _cross_exchange_diagnostics(exchange_payloads),
    }
    if bool(calibration_only) or actor_overrides:
        for item in exchange_payloads.values():
            item["calibration_only"] = bool(calibration_only)
            item["actor_overrides"] = _clean_metadata_mapping(actor_overrides)
    if execution_smoke:
        for item in exchange_payloads.values():
            item["execution_smoke"] = True
    if terminal_denied_context_signal_keys:
        for item in exchange_payloads.values():
            item["terminal_denied_context_signal_keys"] = _clean_string_list(
                terminal_denied_context_signal_keys
            )
    _write_summary_files(report_root, current, summary)
    return summary


def _build_exchange_canary(
    results_root: Path,
    exchange: str,
    *,
    now: datetime,
    lookback_minutes: float,
    require_positive_expectancy: bool,
) -> dict[str, Any]:
    session_dir = _latest_session_dir(results_root, exchange)
    if session_dir is None:
        return _exchange_failure(
            exchange,
            "session_missing",
            require_positive_expectancy=bool(require_positive_expectancy),
        )

    status_path = session_dir / "status.json"
    status = _load_json(status_path)
    if not isinstance(status, Mapping):
        return _exchange_failure(
            exchange,
            "status_invalid",
            session_dir=session_dir,
            require_positive_expectancy=bool(require_positive_expectancy),
        )

    rows = _recent_rows(
        _iter_jsonl(session_dir / "causal_entry_decisions.jsonl"),
        now=now,
        lookback_minutes=lookback_minutes,
    )
    event_rows = _recent_rows(
        _iter_session_event_jsonl(session_dir),
        now=now,
        lookback_minutes=lookback_minutes,
    )
    context_event_rows = _recent_rows(
        _iter_context_event_jsonl(results_root, exchange=exchange, session_dir=session_dir),
        now=now,
        lookback_minutes=lookback_minutes,
    )

    signals = sum(_row_signal_count(row) for row in rows)
    orders = sum(_row_order_count(row) for row in rows) + sum(_event_order_count(row) for row in event_rows)
    fills = sum(_row_fill_count(row) for row in rows) + sum(_event_fill_count(row) for row in event_rows)
    open_positions = _open_positions_payload(status)

    pnl_payload = _pnl_payload(
        rows,
        event_rows,
        status,
        supplemental_rows=context_event_rows,
    )
    expectancy = _expectancy_after_costs(pnl_payload, fills)
    reconcile_ok, reconcile_warnings = _reconcile_status(status)
    health_warnings = _health_warnings(status)
    cost_attribution = _cost_attribution_payload(
        rows,
        event_rows,
        status,
        fills=fills,
        supplemental_rows=context_event_rows,
    )
    if not bool(cost_attribution["complete"]):
        health_warnings = sorted(set([*health_warnings, "cost_attribution_missing"]))
    execution_block_diagnostics = _execution_block_diagnostics(context_event_rows)
    negative_contexts = _negative_closed_trade_contexts(context_event_rows)
    negative_context_keys = sorted({
        key
        for context in negative_contexts
        for key in _negative_context_signal_keys(context)
    })
    fail_reasons = _fail_reasons(
        signals=signals,
        orders=orders,
        fills=fills,
        expectancy_after_costs=expectancy,
        require_positive_expectancy=bool(require_positive_expectancy),
        reconcile_ok=reconcile_ok,
        reconcile_warnings=reconcile_warnings,
        health_warnings=health_warnings,
        cost_attribution_missing=not bool(cost_attribution["complete"]),
        min_notional_blocked_count=int(execution_block_diagnostics["min_notional_blocked_count"]),
        owned_open_position_count=int(open_positions["owned_open_position_count"]),
        status=status,
    )
    zero_signal_diagnostics = _zero_signal_diagnostics(
        exchange=exchange,
        decision_rows=rows,
        context_event_rows=context_event_rows,
        signals=signals,
    )
    context_symbols = _context_symbols(context_event_rows)

    return {
        "exchange": exchange,
        "passed": not fail_reasons,
        "session_dir": str(session_dir),
        "status_path": str(status_path),
        "timestamp_utc": str(status.get("timestamp_utc") or status.get("timestamp") or ""),
        "mode": str(status.get("mode") or ""),
        "run_state": str(status.get("run_state") or ""),
        "feed_status": str(status.get("feed_status") or ""),
        "signals": signals,
        "orders": orders,
        "fills": fills,
        "expectancy_after_costs": round(expectancy, 10),
        "expectancy_gate_required": bool(require_positive_expectancy),
        "gross_pnl_usd": round(float(pnl_payload["gross_pnl_usd"]), 10),
        "fees_usd": round(float(pnl_payload["fees_usd"]), 10),
        "funding_usd": round(float(pnl_payload["funding_usd"]), 10),
        "slippage_usd": round(float(pnl_payload["slippage_usd"]), 10),
        "cost_attribution": cost_attribution,
        "execution_block_diagnostics": execution_block_diagnostics,
        "reconcile_ok": reconcile_ok,
        "reconcile_warnings": reconcile_warnings,
        "health_warnings": health_warnings,
        "open_position_count": int(open_positions["open_position_count"]),
        "owned_open_position_count": int(open_positions["owned_open_position_count"]),
        "external_open_position_count": int(open_positions["external_open_position_count"]),
        "open_position_symbols": open_positions["open_position_symbols"],
        "owned_open_position_symbols": open_positions["owned_open_position_symbols"],
        "external_open_position_symbols": open_positions["external_open_position_symbols"],
        "negative_closed_trade_contexts": negative_contexts,
        "negative_context_signal_keys": negative_context_keys,
        "context_symbols": context_symbols,
        "rows_evaluated": len(rows),
        "event_rows_evaluated": len(event_rows),
        "context_event_rows_evaluated": len(context_event_rows),
        "zero_signal_diagnostics": zero_signal_diagnostics,
        "fail_reasons": fail_reasons,
    }


def _clean_metadata_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()
        if not clean_key:
            continue
        if isinstance(raw, (str, int, float, bool)) or raw is None:
            out[clean_key] = raw
        else:
            out[clean_key] = str(raw)
    return out


def _clean_json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    try:
        encoded = json.dumps(value, sort_keys=True, default=str)
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return _clean_metadata_mapping(value)
    return decoded if isinstance(decoded, dict) else {}


def _clean_string_list(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    return list(dict.fromkeys(str(item or "").strip() for item in values if str(item or "").strip()))


def _exchange_failure(
    exchange: str,
    reason: str,
    *,
    session_dir: Path | None = None,
    require_positive_expectancy: bool = True,
) -> dict[str, Any]:
    return {
        "exchange": exchange,
        "passed": False,
        "session_dir": str(session_dir or ""),
        "signals": 0,
        "orders": 0,
        "fills": 0,
        "expectancy_after_costs": 0.0,
        "expectancy_gate_required": bool(require_positive_expectancy),
        "reconcile_ok": False,
        "reconcile_warnings": [],
        "health_warnings": [],
        "open_position_count": 0,
        "owned_open_position_count": 0,
        "external_open_position_count": 0,
        "open_position_symbols": [],
        "owned_open_position_symbols": [],
        "external_open_position_symbols": [],
        "negative_closed_trade_contexts": [],
        "negative_context_signal_keys": [],
        "execution_block_diagnostics": _empty_execution_block_diagnostics(),
        "context_event_rows_evaluated": 0,
        "fail_reasons": [reason],
    }


def _latest_session_dir(results_root: Path, exchange: str) -> Path | None:
    exchange_root = results_root / exchange
    if not exchange_root.exists():
        return None
    statuses = [path for path in exchange_root.glob("**/status.json") if path.is_file()]
    if not statuses:
        return None
    return max(statuses, key=lambda path: path.stat().st_mtime).parent


def _recent_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    lookback_minutes: float,
) -> list[Mapping[str, Any]]:
    if lookback_minutes <= 0:
        return list(rows)
    cutoff_seconds = float(lookback_minutes) * 60.0
    recent: list[Mapping[str, Any]] = []
    for row in rows:
        timestamp = _row_timestamp(row)
        if timestamp is None or (now - timestamp).total_seconds() <= cutoff_seconds:
            recent.append(row)
    return recent


def _row_signal_count(row: Mapping[str, Any]) -> int:
    explicit = _max_int(row, "signal_count", "raw_signal_count", "executable_signal_count", "n_signals")
    raw = _sequence_len(row.get("raw_signals"))
    executable = _sequence_len(row.get("executable_signals"))
    flash_signals = 0
    decisions = row.get("flash_decisions")
    if isinstance(decisions, Sequence) and not isinstance(decisions, (str, bytes)):
        for decision in decisions:
            if isinstance(decision, Mapping) and decision.get("signal"):
                flash_signals += 1
    return max(explicit, raw, executable, flash_signals)


def _row_order_count(row: Mapping[str, Any]) -> int:
    explicit = _max_int(row, "orders", "order_count", "sent_orders", "submitted_orders", "n_orders", "n_sent_orders")
    fills = _row_fill_count(row)
    rejected = _max_int(row, "rejected", "n_rejected", "rejection_count", "rejected_orders")
    return max(explicit, fills, rejected)


def _row_fill_count(row: Mapping[str, Any]) -> int:
    return _max_int(row, "fills", "filled", "filled_signals", "n_filled", "fill_count")


def _event_order_count(row: Mapping[str, Any]) -> int:
    if _max_int(row, "orders", "order_count", "sent_orders", "submitted_orders") > 0:
        return _max_int(row, "orders", "order_count", "sent_orders", "submitted_orders")
    name = _event_name(row)
    if "order" in name and any(token in name for token in ("sent", "submit", "place", "accepted")):
        return 1
    return 0


def _event_fill_count(row: Mapping[str, Any]) -> int:
    if _max_int(row, "fills", "filled", "fill_count") > 0:
        return _max_int(row, "fills", "filled", "fill_count")
    name = _event_name(row)
    if "fill" in name or "filled" in name:
        return 1
    return 0


def _event_name(row: Mapping[str, Any]) -> str:
    parts = [
        str(row.get("_type") or ""),
        str(row.get("event") or ""),
        str(row.get("type") or ""),
        str(row.get("action") or ""),
        str(row.get("name") or ""),
    ]
    return " ".join(parts).lower()


def _pnl_payload(
    rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any],
    supplemental_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, float]:
    combined = list(rows) + list(event_rows)
    gross = sum(_first_float(row, "realized_pnl_usd", "pnl_usd", "realized_pnl", "pnl") for row in combined)
    fees = _sum_payload_costs(combined, FEE_KEYS)
    funding = _sum_payload_costs(combined, FUNDING_KEYS)
    slippage = _sum_payload_costs(combined, SLIPPAGE_KEYS)

    if gross == 0.0:
        gross = _first_float(status, "realized_pnl_usd", "pnl_usd", "realized_pnl", "pnl")
    if fees == 0.0:
        fees = _first_float(status, *FEE_KEYS)
    if fees == 0.0:
        fees = _sum_payload_costs(supplemental_rows, FEE_KEYS)
    if funding == 0.0:
        funding = _first_float(status, *FUNDING_KEYS)
    if funding == 0.0:
        funding = _sum_payload_costs(supplemental_rows, FUNDING_KEYS)
    if slippage == 0.0:
        slippage = _first_float(status, *SLIPPAGE_KEYS)
    if slippage == 0.0:
        slippage = _sum_payload_costs(supplemental_rows, SLIPPAGE_KEYS)

    return {
        "gross_pnl_usd": gross,
        "fees_usd": fees,
        "funding_usd": funding,
        "slippage_usd": slippage,
        "direct_expectancy": _first_float(status, "expectancy_after_costs", "expectancy_usd", "expectancy"),
    }


def _cost_attribution_payload(
    rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    status: Mapping[str, Any],
    *,
    fills: int,
    supplemental_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    observed: list[str] = []
    fee_value = 0.0
    for scope, payload in [("status", status)]:
        for key in FEE_KEYS:
            if key in payload:
                observed.append(f"{scope}.{key}")
                fee_value += abs(_first_float(payload, key))
    for scope, source_rows in (
        ("row", [*rows, *event_rows]),
        ("context", list(supplemental_rows)),
    ):
        for idx, row in enumerate(source_rows):
            for key in FEE_KEYS:
                if key in row:
                    observed.append(f"{scope}[{idx}].{key}")
                    fee_value += abs(_first_float(row, key))
            for nested_key in COST_NESTED_KEYS:
                nested = row.get(nested_key)
                if not isinstance(nested, Mapping):
                    continue
                for key in FEE_KEYS:
                    if key in nested:
                        observed.append(f"{scope}[{idx}].{nested_key}.{key}")
                        fee_value += abs(_first_float(nested, key))

    complete = fills <= 0 or fee_value > 0.0
    if complete:
        missing = []
    elif observed:
        missing = ["fee_nonzero"]
    else:
        missing = ["fee"]
    return {
        "complete": bool(complete),
        "observed_keys": sorted(set(observed)),
        "missing": missing,
    }


def _sum_payload_costs(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> float:
    fill_total = 0.0
    fill_seen: set[str] = set()
    fallback_total = 0.0
    for idx, row in enumerate(rows):
        cost = _row_payload_cost(row, keys)
        if cost == 0.0:
            continue
        fallback_total += cost
        if _event_fill_count(row) <= 0:
            continue
        identity = _payload_cost_identity(row) or f"row:{idx}"
        if identity in fill_seen:
            continue
        fill_seen.add(identity)
        fill_total += cost
    return fill_total if fill_seen else fallback_total


def _row_payload_cost(row: Mapping[str, Any], keys: Sequence[str]) -> float:
    direct = _first_float(row, *keys)
    if direct != 0.0:
        return direct
    nested_total = 0.0
    for nested_key in COST_NESTED_KEYS:
        nested = row.get(nested_key)
        if isinstance(nested, Mapping):
            nested_total += _first_float(nested, *keys)
    return nested_total


def _payload_cost_identity(row: Mapping[str, Any]) -> str:
    nested_values: list[Any] = []
    for nested_key in COST_NESTED_KEYS:
        nested = row.get(nested_key)
        if isinstance(nested, Mapping):
            nested_values.extend([
                nested.get("exchange_order_id"),
                nested.get("order_id"),
                nested.get("signal_id"),
            ])
    values = [
        row.get("order_id"),
        row.get("exchange_order_id"),
        *nested_values,
        row.get("trace_id"),
        row.get("signal_id"),
        row.get("timestamp"),
        row.get("timestamp_utc"),
    ]
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _expectancy_after_costs(payload: Mapping[str, float], fills: int) -> float:
    if fills <= 0:
        return 0.0
    direct = float(payload.get("direct_expectancy") or 0.0)
    if direct != 0.0:
        return direct
    net = (
        float(payload.get("gross_pnl_usd") or 0.0)
        - abs(float(payload.get("fees_usd") or 0.0))
        + float(payload.get("funding_usd") or 0.0)
        - abs(float(payload.get("slippage_usd") or 0.0))
    )
    return net / float(fills)


def _negative_closed_trade_contexts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for row in rows:
        if "positionclosed" not in _event_name(row).replace(" ", ""):
            continue
        realized_pnl = _first_float(row, "realized_pnl_usd", "realized_pnl", "pnl_usd", "pnl")
        if realized_pnl >= 0.0:
            continue
        actor_label = str(row.get("by_player") or row.get("by_agent") or "").strip()
        symbol = str(row.get("symbol") or row.get("sym") or "").strip().upper()
        action = str(row.get("open_action") or row.get("action") or "").strip().upper()
        regime = str(row.get("open_regime") or row.get("regime") or "").strip()
        if not actor_label or not symbol or not action or not regime:
            continue
        contexts.append({
            "actor_label": actor_label,
            "symbol": symbol,
            "action": action,
            "regime": regime,
            "realized_pnl": realized_pnl,
            "timestamp": str(row.get("timestamp") or row.get("timestamp_utc") or ""),
        })
    return contexts


def _negative_context_signal_keys(context: Mapping[str, Any]) -> list[str]:
    actor_label = str(context.get("actor_label") or "").strip()
    symbol = str(context.get("symbol") or "").strip().upper()
    action = str(context.get("action") or "").strip().upper()
    regime = str(context.get("regime") or "").strip()
    if not actor_label or not symbol or not action or not regime:
        return []
    return [
        f"{actor_key}|{symbol}|{action}|{regime}"
        for actor_key in _actor_key_aliases(actor_label)
    ]


def _actor_key_aliases(actor_label: str) -> tuple[str, ...]:
    clean = str(actor_label or "").strip()
    if not clean:
        return ()
    aliases = [clean]
    if ":" not in clean:
        aliases.append(f"agent:{clean}")
        if not clean.startswith("Solo_"):
            aliases.append(f"Solo_{clean}")
            aliases.append(f"ensemble:Solo_{clean}")
    return tuple(dict.fromkeys(aliases))


def _zero_signal_diagnostics(
    *,
    exchange: str,
    decision_rows: Sequence[Mapping[str, Any]],
    context_event_rows: Sequence[Mapping[str, Any]],
    signals: int,
) -> dict[str, Any]:
    if signals > 0 or not context_event_rows:
        return {"enabled": False}
    symbols = _context_symbols(context_event_rows)
    reason_counts: dict[str, int] = {}
    for row in context_event_rows:
        reason = str(
            row.get("first_rejection_reason")
            or row.get("rejection_reason")
            or row.get("deny_reason")
            or row.get("reason")
            or ""
        ).strip()
        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "enabled": True,
        "exchange": exchange,
        "decision_rows": len(decision_rows),
        "context_event_rows": len(context_event_rows),
        "symbols_with_context": symbols,
        "first_rejection_reason_counts": dict(sorted(reason_counts.items())),
    }


def _empty_execution_block_diagnostics() -> dict[str, Any]:
    return {
        "enabled": False,
        "blocked_count": 0,
        "first_blocked_reason_counts": {},
        "min_notional_blocked_count": 0,
        "blocked_symbols": [],
        "blocked_actions": [],
        "first_blocked_examples": [],
    }


def _execution_block_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    symbols: set[str] = set()
    actions: set[str] = set()
    examples: list[dict[str, str]] = []
    min_notional_blocked = 0

    for row in rows:
        if "executionattributed" not in _event_name(row).replace(" ", ""):
            continue
        status = str(row.get("status") or row.get("attribution_bucket") or "").strip().lower()
        bucket = str(row.get("attribution_bucket") or "").strip().lower()
        if status != "blocked" and bucket != "blocked":
            continue

        reason = str(row.get("reason") or row.get("blocked_reason") or "").strip()
        symbol = str(row.get("symbol") or row.get("sym") or "").strip().upper()
        action = str(row.get("action") or "").strip().upper()
        timestamp = str(row.get("timestamp") or row.get("timestamp_utc") or "").strip()

        if reason:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if _is_min_notional_block_reason(reason):
                min_notional_blocked += 1
        if symbol:
            symbols.add(symbol)
        if action:
            actions.add(action)
        if len(examples) < 5:
            examples.append({
                "symbol": symbol,
                "action": action,
                "reason": reason,
                "timestamp": timestamp,
            })

    if not reason_counts and not examples:
        return _empty_execution_block_diagnostics()

    return {
        "enabled": True,
        "blocked_count": sum(reason_counts.values()) if reason_counts else len(examples),
        "first_blocked_reason_counts": dict(sorted(reason_counts.items())),
        "min_notional_blocked_count": min_notional_blocked,
        "blocked_symbols": sorted(symbols),
        "blocked_actions": sorted(actions),
        "first_blocked_examples": examples,
    }


def _is_min_notional_block_reason(reason: str) -> bool:
    text = str(reason or "").lower()
    return "min_notional" in text or ("notional" in text and ("< min" in text or "min $" in text))


def _context_symbols(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({
        str(row.get("symbol") or row.get("sym") or row.get("asset") or "").strip().upper()
        for row in rows
        if str(row.get("symbol") or row.get("sym") or row.get("asset") or "").strip()
    })


def _cross_exchange_diagnostics(
    exchange_payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for zero_exchange, zero_payload in exchange_payloads.items():
        if _safe_int_from_payload(zero_payload, "signals") > 0:
            continue
        if _safe_int_from_payload(zero_payload, "context_event_rows_evaluated") <= 0:
            continue
        for active_exchange, active_payload in exchange_payloads.items():
            if active_exchange == zero_exchange:
                continue
            if _safe_int_from_payload(active_payload, "signals") <= 0:
                continue
            diagnostics.append({
                "zero_signal_exchange": str(zero_exchange),
                "active_exchange": str(active_exchange),
                "signals_delta": (
                    _safe_int_from_payload(zero_payload, "signals")
                    - _safe_int_from_payload(active_payload, "signals")
                ),
                "context_event_rows_delta": (
                    _safe_int_from_payload(zero_payload, "context_event_rows_evaluated")
                    - _safe_int_from_payload(active_payload, "context_event_rows_evaluated")
                ),
                "suspect_layers": ["symbol_mapping", "contract_filters", "min_notional"],
                "zero_signal_symbols": list(zero_payload.get("context_symbols") or []),
                "active_symbols": list(active_payload.get("context_symbols") or []),
            })
    return diagnostics


def _safe_int_from_payload(payload: Mapping[str, Any], key: str) -> int:
    try:
        return int(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _reconcile_status(status: Mapping[str, Any]) -> tuple[bool, list[str]]:
    payloads = []
    for key in ("live_state_sync", "exchange_reconcile", "reconcile", "reconcile_summary"):
        value = status.get(key)
        if isinstance(value, Mapping):
            payloads.append(value)
    warnings: list[str] = []
    ok = True
    for payload in payloads:
        if payload.get("reconcile_ok") is False or payload.get("ok") is False:
            ok = False
        warnings.extend(_strings(payload.get("warnings")))
        warnings.extend(_strings(payload.get("reconcile_warnings")))
        warnings.extend(_strings(payload.get("desync_warnings")))
    warnings.extend(_strings(status.get("reconcile_warnings")))
    return ok and not warnings, sorted(set(warnings))


def _health_warnings(status: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    warnings.extend(_strings(status.get("health_warnings")))
    health = status.get("data_health")
    if isinstance(health, Mapping):
        warnings.extend(_strings(health.get("warnings")))
        if _first_float(health, "failed_orders") > 0:
            warnings.append("failed_orders")
    return sorted(set(warnings))


def _open_positions_payload(status: Mapping[str, Any]) -> dict[str, Any]:
    positions = status.get("open_positions")
    rows: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(positions, Mapping):
        for symbol, payload in positions.items():
            if isinstance(payload, Mapping):
                rows.append((str(symbol).upper(), payload))
            else:
                rows.append((str(symbol).upper(), {}))
    elif isinstance(positions, Sequence) and not isinstance(positions, (str, bytes)):
        for index, payload in enumerate(positions):
            if isinstance(payload, Mapping):
                symbol = str(payload.get("symbol") or payload.get("sym") or payload.get("asset") or index).upper()
                rows.append((symbol, payload))

    open_symbols = sorted({symbol for symbol, _ in rows if symbol})
    owned_symbols = sorted({symbol for symbol, payload in rows if symbol and not _is_external_position(payload)})
    external_symbols = sorted({symbol for symbol, payload in rows if symbol and _is_external_position(payload)})
    return {
        "open_position_count": len(open_symbols),
        "owned_open_position_count": len(owned_symbols),
        "external_open_position_count": len(external_symbols),
        "open_position_symbols": open_symbols,
        "owned_open_position_symbols": owned_symbols,
        "external_open_position_symbols": external_symbols,
    }


def _is_external_position(position: Mapping[str, Any]) -> bool:
    if bool(position.get("external")) or bool(position.get("inherited_from_exchange")):
        return True
    external_labels = {"", "AdoptedExchangePosition", "RecoveredExchangePosition"}
    owner = str(position.get("by_player") or position.get("by_agent") or position.get("owner") or "").strip()
    return owner in external_labels


def _fail_reasons(
    *,
    signals: int,
    orders: int,
    fills: int,
    expectancy_after_costs: float,
    require_positive_expectancy: bool,
    reconcile_ok: bool,
    reconcile_warnings: Sequence[str],
    health_warnings: Sequence[str],
    cost_attribution_missing: bool,
    min_notional_blocked_count: int,
    owned_open_position_count: int,
    status: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if signals <= 0:
        reasons.append("zero_signals")
    if orders <= 0:
        reasons.append("zero_orders")
    if fills <= 0:
        reasons.append("zero_fills")
    if require_positive_expectancy and expectancy_after_costs <= 0.0:
        reasons.append("nonpositive_expectancy")
    if not reconcile_ok:
        reasons.append("reconcile_failed")
    if reconcile_warnings:
        reasons.append("reconcile_warnings")
    if health_warnings:
        reasons.append("health_warnings")
    if cost_attribution_missing:
        reasons.append("cost_attribution_missing")
    if min_notional_blocked_count > 0:
        reasons.append("min_notional_blocked")
    if owned_open_position_count > 0:
        reasons.append("open_positions_not_flat")
    if _feed_not_active_is_failure(status, owned_open_position_count=owned_open_position_count):
        reasons.append("feed_not_active")
    return reasons


def _feed_not_active_is_failure(
    status: Mapping[str, Any],
    *,
    owned_open_position_count: int,
) -> bool:
    feed_status = str(status.get("feed_status") or "").strip().lower()
    if feed_status in ("", "active", "ok", "healthy"):
        return False

    # A bounded paper/shadow canary is expected to close its feed after the run.
    # Keep this strict for real live workers, where an inactive feed is unsafe.
    mode = str(status.get("mode") or "").strip().lower()
    run_state = str(status.get("run_state") or "").strip().lower()
    if (
        mode in ("paper", "paper_live_feed", "shadow_live_feed")
        and run_state in ("stopped", "finished", "completed")
        and owned_open_position_count <= 0
    ):
        return False
    return True


def _write_summary_files(report_root: Path, now: datetime, summary: Mapping[str, Any]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    run_dir = report_root / now.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "canary_summary.json"
    latest_path = report_root / "latest_canary_summary.json"
    text = json.dumps(summary, indent=2, sort_keys=True)
    run_path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    shutil.copyfile(run_path, report_root / "panteon3_live_canary_summary.json")


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    if not path.exists():
        return ()

    def _generator() -> Iterable[Mapping[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, Mapping):
                    yield row

    return _generator()


def _iter_session_event_jsonl(session_dir: Path) -> Iterable[Mapping[str, Any]]:
    candidates: list[Path] = [
        session_dir / "events.jsonl",
    ]
    candidates.extend((session_dir / "logs").glob("*events*.jsonl"))
    yield from _iter_unique_jsonl(candidates)


def _iter_context_event_jsonl(
    results_root: Path,
    *,
    exchange: str,
    session_dir: Path,
) -> Iterable[Mapping[str, Any]]:
    candidates: list[Path] = [session_dir / "events.jsonl"]
    candidates.extend((session_dir / "logs").glob("*events*.jsonl"))
    logs_dir = results_root / "logs"
    exchange_prefix = str(exchange or "").strip().lower()
    if logs_dir.exists():
        candidates.extend(logs_dir.glob(f"{exchange_prefix}*events*.jsonl"))
    yield from _iter_unique_jsonl(candidates)


def _iter_unique_jsonl(paths: Iterable[Path]) -> Iterable[Mapping[str, Any]]:
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        yield from _iter_jsonl(path)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row_timestamp(row: Mapping[str, Any]) -> datetime | None:
    for key in ("timestamp", "timestamp_utc", "created_at"):
        parsed = _parse_time(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _aware_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _max_int(payload: Mapping[str, Any], *keys: str) -> int:
    values = []
    for key in keys:
        try:
            values.append(int(payload.get(key)))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0


def _first_float(payload: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        value = _float_value(payload.get(key))
        if value is not None:
            return value
    return 0.0


def _float_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("cost", "amount", "value", "total", "usdt", "usd"):
            nested = _float_value(value.get(key))
            if nested is not None:
                return nested
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_trade_float(payload: Mapping[str, Any], *keys: str) -> float:
    for nested_key in COST_NESTED_KEYS:
        nested = payload.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        value = _first_float(nested, *keys)
        if value != 0.0:
            return value
    return 0.0


def _sequence_len(value: Any) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item or "").strip()]
    return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Panteon v3 live canary summary from runtime artifacts.")
    parser.add_argument("--results-root", default="Results")
    parser.add_argument("--reports-dir", default=str(Path("Reports") / "Panteon3Canary"))
    parser.add_argument("--exchange", action="append", dest="exchanges")
    parser.add_argument("--lookback-minutes", type=float, default=360.0)
    args = parser.parse_args(argv)

    summary = build_canary_summary(
        results_root=args.results_root,
        reports_dir=args.reports_dir,
        exchanges=tuple(args.exchanges or DEFAULT_EXCHANGES),
        lookback_minutes=args.lookback_minutes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
