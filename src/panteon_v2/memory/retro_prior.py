"""Conservative retrotest-derived priors for PerformanceMemory.

Retrotest data is useful as a cold-start hint, but it must not be treated as
live exchange evidence. This module builds and merges small virtual-memory
priors while preserving real-memory accounting.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..domain.types import Regime
from .performance import PerformanceMemory


DEFAULT_EXCLUDED_LABELS = frozenset({"CashFlat", "NoTrade"})


@dataclass(frozen=True)
class RetroPriorReport:
    source_path: str = ""
    total_pairs: int = 0
    applied_pairs: int = 0
    skipped_existing_pairs: int = 0
    skipped_invalid_pairs: int = 0
    warnings: list[str] = field(default_factory=list)


def build_retro_prior_snapshot_from_breakdown_files(
    paths: Sequence[str | Path],
    *,
    exchange: str,
    source_run_id: str = "",
    prior_weight: float = 0.25,
    max_prior_closed_trades: int = 24,
    max_abs_prior_pnl_pct: float = 8.0,
    min_closed_trades: int = 5,
    excluded_labels: Iterable[str] = DEFAULT_EXCLUDED_LABELS,
) -> dict:
    """Build a small PerformanceMemory-compatible snapshot from breakdown CSVs."""
    weight = _positive_float(prior_weight, "prior_weight")
    max_closed = _positive_int(max_prior_closed_trades, "max_prior_closed_trades")
    max_abs_pnl = _positive_float(max_abs_prior_pnl_pct, "max_abs_prior_pnl_pct")
    min_closed = max(1, int(min_closed_trades))
    excluded = {str(label) for label in excluded_labels}
    state: dict[str, dict] = {}
    rows_read = 0
    rows_used = 0
    warnings: list[str] = []

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            warnings.append(f"missing breakdown: {path}")
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_read += 1
                item = _prior_state_from_breakdown_row(
                    row,
                    prior_weight=weight,
                    max_prior_closed_trades=max_closed,
                    max_abs_prior_pnl_pct=max_abs_pnl,
                    min_closed_trades=min_closed,
                    excluded_labels=excluded,
                )
                if item is None:
                    continue
                key, payload = item
                rows_used += 1
                if key in state:
                    state[key] = _combine_state_payloads(
                        state[key],
                        payload,
                        max_prior_closed_trades=max_closed,
                        max_abs_prior_pnl_pct=max_abs_pnl,
                    )
                else:
                    state[key] = payload

    exchange_scope = str(exchange or "").strip().upper() or "MEXC"
    return {
        "trade_fraction": 1.0,
        "exchange_scope": exchange_scope,
        "state": state,
        "context_state": {},
        "open": {},
        "seen_signal_ids": [],
        "_retro_prior": {
            "source": "retrotest_regime_breakdown",
            "source_run_id": str(source_run_id or ""),
            "exchange": exchange_scope,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prior_weight": weight,
            "max_prior_closed_trades": max_closed,
            "max_abs_prior_pnl_pct": max_abs_pnl,
            "min_closed_trades": min_closed,
            "rows_read": rows_read,
            "rows_used": rows_used,
            "warnings": warnings,
        },
    }


def merge_retro_prior_into_memory(
    perf: PerformanceMemory,
    prior_snapshot: Mapping[str, object],
    *,
    expected_exchange: str | None = None,
    fill_missing_only: bool = True,
    source_path: str = "",
) -> RetroPriorReport:
    """Merge retro prior state into virtual PerformanceMemory.

    Existing live/shadow memory wins by default. That makes the prior a
    cold-start seed, not an overwrite of current MEXC evidence.
    """
    warnings: list[str] = []
    if not isinstance(prior_snapshot, Mapping):
        return RetroPriorReport(source_path=source_path, warnings=["prior snapshot is not a mapping"])

    expected = str(expected_exchange or "").strip().upper()
    prior_exchange = str(prior_snapshot.get("exchange_scope") or "").strip().upper()
    if expected and prior_exchange and prior_exchange != expected:
        return RetroPriorReport(
            source_path=source_path,
            warnings=[f"prior exchange_scope={prior_exchange} does not match {expected}"],
        )

    incoming = prior_snapshot.get("state") or {}
    if not isinstance(incoming, Mapping):
        return RetroPriorReport(source_path=source_path, warnings=["prior state is not a mapping"])

    snapshot = perf.snapshot()
    state = dict(snapshot.get("state") or {})

    # Phase 3 / C2: приор может быть построен при ином trade_fraction (исторически
    # 1.0), чем живая память (например 0.10). PnL-метрики приора нужно привести к
    # масштабу живого учёта, иначе seed выглядит в (live/prior) раз крупнее и
    # искажает scoring/quarantine. При равных fraction коэффициент = 1.0.
    live_fraction = _float(snapshot.get("trade_fraction"), default=1.0)
    prior_fraction = _float(prior_snapshot.get("trade_fraction"), default=1.0)
    pnl_scale = 1.0
    if prior_fraction > 0.0 and live_fraction > 0.0 and prior_fraction != live_fraction:
        pnl_scale = live_fraction / prior_fraction
        warnings.append(
            f"rescaled prior pnl by {pnl_scale:.4g} "
            f"(prior trade_fraction={prior_fraction:.4g} -> live={live_fraction:.4g})"
        )

    applied = skipped_existing = skipped_invalid = 0
    for raw_key, raw_payload in incoming.items():
        key = str(raw_key)
        if not _valid_state_key(key) or not isinstance(raw_payload, Mapping):
            skipped_invalid += 1
            continue
        payload = _sanitize_state_payload(raw_payload)
        if payload is None:
            skipped_invalid += 1
            continue
        if pnl_scale != 1.0:
            payload = _rescale_payload_pnl(payload, pnl_scale)
        if fill_missing_only and _state_payload_has_data(state.get(key)):
            skipped_existing += 1
            continue
        state[key] = payload
        applied += 1

    snapshot["state"] = state
    if applied:
        perf.restore(snapshot)
    return RetroPriorReport(
        source_path=source_path,
        total_pairs=len(incoming),
        applied_pairs=applied,
        skipped_existing_pairs=skipped_existing,
        skipped_invalid_pairs=skipped_invalid,
        warnings=warnings,
    )


def load_retro_prior_file_into_memory(
    perf: PerformanceMemory,
    path: str | Path,
    *,
    expected_exchange: str | None = None,
    fill_missing_only: bool = True,
) -> RetroPriorReport:
    prior_path = Path(path)
    if not prior_path.exists():
        return RetroPriorReport(source_path=str(prior_path), warnings=[f"file not found: {prior_path}"])
    try:
        with prior_path.open("r", encoding="utf-8") as f:
            snapshot = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return RetroPriorReport(source_path=str(prior_path), warnings=[f"failed to read prior: {exc}"])
    return merge_retro_prior_into_memory(
        perf,
        snapshot,
        expected_exchange=expected_exchange,
        fill_missing_only=fill_missing_only,
        source_path=str(prior_path),
    )


def write_retro_prior_snapshot(snapshot: Mapping[str, object], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(dict(snapshot), f, indent=2, sort_keys=True)


def _prior_state_from_breakdown_row(
    row: Mapping[str, object],
    *,
    prior_weight: float,
    max_prior_closed_trades: int,
    max_abs_prior_pnl_pct: float,
    min_closed_trades: int,
    excluded_labels: set[str],
) -> tuple[str, dict] | None:
    label = str(row.get("label") or "").strip()
    if not label or label in excluded_labels:
        return None
    try:
        regime = Regime.from_string(str(row.get("regime") or ""))
    except Exception:
        return None
    closed_raw = _non_negative_int(row.get("closed_trades"))
    if closed_raw < min_closed_trades:
        return None
    wins_raw = _non_negative_int(row.get("wins"))
    losses_raw = _non_negative_int(row.get("losses"))
    if wins_raw + losses_raw <= 0:
        wins_raw = max(0, int(round(closed_raw * (_float(row.get("win_rate_pct")) / 100.0))))
        losses_raw = max(0, closed_raw - wins_raw)

    closed = min(max_prior_closed_trades, max(1, int(round(closed_raw * prior_weight))))
    win_rate = min(1.0, max(0.0, wins_raw / max(1, closed_raw)))
    wins = min(closed, int(round(closed * win_rate)))
    losses = max(0, closed - wins)
    pnl_pct = _clamp(
        _float(row.get("pnl_pct")) * prior_weight,
        -max_abs_prior_pnl_pct,
        max_abs_prior_pnl_pct,
    )
    entries = _scaled_activity_count(
        row.get("entries"),
        prior_weight=prior_weight,
        floor=closed,
        cap=max_prior_closed_trades * 3,
    )
    signals = _scaled_activity_count(
        row.get("signals"),
        prior_weight=prior_weight,
        floor=max(closed, entries),
        cap=max_prior_closed_trades * 4,
    )
    equity = max(0.01, 1.0 + pnl_pct / 100.0)
    max_dd_pct = max(0.0, _float(row.get("max_dd_pct")), -min(0.0, pnl_pct))
    return (
        f"{label}|{regime.label}",
        {
            "closed_trades": closed,
            "entries": entries,
            "signals": signals,
            "wins": wins,
            "losses": losses,
            "pnl_pct": pnl_pct,
            "pnl_gross_pct": pnl_pct,
            "fee_pct": 0.0,
            "funding_pct": 0.0,
            "returns": [],
            "equity": equity,
            "equity_curve": [100.0, equity * 100.0],
            "equity_curve_timestamps": [],
            "peak": max(1.0, equity),
            "max_dd_pct": max_dd_pct,
            "blocked_signals": 0,
            "rejected_signals": 0,
            "pending_signals": 0,
            "execution_failures": 0,
        },
    )


def _combine_state_payloads(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    max_prior_closed_trades: int,
    max_abs_prior_pnl_pct: float,
) -> dict:
    closed = min(
        max_prior_closed_trades,
        _non_negative_int(left.get("closed_trades")) + _non_negative_int(right.get("closed_trades")),
    )
    wins = min(closed, _non_negative_int(left.get("wins")) + _non_negative_int(right.get("wins")))
    losses = min(closed - wins, _non_negative_int(left.get("losses")) + _non_negative_int(right.get("losses")))
    pnl_pct = _clamp(
        _float(left.get("pnl_pct")) + _float(right.get("pnl_pct")),
        -max_abs_prior_pnl_pct,
        max_abs_prior_pnl_pct,
    )
    entries = max(closed, min(max_prior_closed_trades * 3, _non_negative_int(left.get("entries")) + _non_negative_int(right.get("entries"))))
    signals = max(entries, min(max_prior_closed_trades * 4, _non_negative_int(left.get("signals")) + _non_negative_int(right.get("signals"))))
    equity = max(0.01, 1.0 + pnl_pct / 100.0)
    combined = dict(left)
    combined.update(
        {
            "closed_trades": closed,
            "entries": entries,
            "signals": signals,
            "wins": wins,
            "losses": losses,
            "pnl_pct": pnl_pct,
            "pnl_gross_pct": pnl_pct,
            "equity": equity,
            "equity_curve": [100.0, equity * 100.0],
            "peak": max(1.0, equity),
            "max_dd_pct": max(_float(left.get("max_dd_pct")), _float(right.get("max_dd_pct")), -min(0.0, pnl_pct)),
        }
    )
    return combined


def _sanitize_state_payload(payload: Mapping[str, object]) -> dict | None:
    closed = _non_negative_int(payload.get("closed_trades"))
    entries = _non_negative_int(payload.get("entries"))
    signals = _non_negative_int(payload.get("signals"))
    wins = _non_negative_int(payload.get("wins"))
    losses = _non_negative_int(payload.get("losses"))
    if closed <= 0 and entries <= 0 and signals <= 0:
        return None
    wins = min(wins, closed)
    losses = min(losses, max(0, closed - wins))
    pnl_pct = _float(payload.get("pnl_pct"))
    equity = max(0.01, _float(payload.get("equity"), default=1.0 + pnl_pct / 100.0))
    curve = payload.get("equity_curve")
    if not isinstance(curve, list) or not curve:
        curve = [100.0, equity * 100.0]
    return {
        "closed_trades": closed,
        "entries": max(entries, closed),
        "signals": max(signals, entries, closed),
        "wins": wins,
        "losses": losses,
        "pnl_pct": pnl_pct,
        "pnl_gross_pct": _float(payload.get("pnl_gross_pct"), default=pnl_pct),
        "fee_pct": _float(payload.get("fee_pct")),
        "funding_pct": _float(payload.get("funding_pct")),
        "returns": [float(x) for x in payload.get("returns", []) if _is_finite_number(x)],
        "equity": equity,
        "equity_curve": [float(x) for x in curve if _is_finite_number(x)],
        "equity_curve_timestamps": [
            str(x)
            for x in payload.get("equity_curve_timestamps", [])
            if str(x or "").strip()
        ],
        "peak": max(1.0, _float(payload.get("peak"), default=max(1.0, equity))),
        "max_dd_pct": max(0.0, _float(payload.get("max_dd_pct"))),
        "blocked_signals": _non_negative_int(payload.get("blocked_signals")),
        "rejected_signals": _non_negative_int(payload.get("rejected_signals")),
        "pending_signals": _non_negative_int(payload.get("pending_signals")),
        "execution_failures": _non_negative_int(payload.get("execution_failures")),
    }


def _rescale_payload_pnl(payload: dict, scale: float) -> dict:
    """Привести pnl-пропорциональные поля приора к масштабу живого trade_fraction.

    Масштабируются: pnl_pct, pnl_gross_pct, fee_pct, funding_pct, returns,
    max_dd_pct. equity/peak/equity_curve пересобираются из нового pnl_pct, чтобы
    остаться согласованными. Счётчики (closed/wins/...) не трогаем.
    """
    out = dict(payload)
    try:
        factor = float(scale)
    except (TypeError, ValueError):
        return out
    if not _is_finite_number(factor) or factor <= 0.0 or factor == 1.0:
        return out
    for field in ("pnl_pct", "pnl_gross_pct", "fee_pct", "funding_pct", "max_dd_pct"):
        if field in out:
            out[field] = float(out[field]) * factor
    if isinstance(out.get("returns"), list):
        out["returns"] = [float(x) * factor for x in out["returns"] if _is_finite_number(x)]
    # Пересобираем equity-инварианты из нового pnl_pct.
    pnl_pct = float(out.get("pnl_pct", 0.0))
    equity = max(0.01, 1.0 + pnl_pct / 100.0)
    out["equity"] = equity
    out["equity_curve"] = [100.0, equity * 100.0]
    out["peak"] = max(1.0, equity)
    out["max_dd_pct"] = max(0.0, float(out.get("max_dd_pct", 0.0)))
    return out


def _state_payload_has_data(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return any(
        _non_negative_int(payload.get(key)) > 0
        for key in ("signals", "entries", "closed_trades", "execution_failures")
    )


def _valid_state_key(key: str) -> bool:
    if "|" not in key:
        return False
    label, regime = key.rsplit("|", 1)
    if not label:
        return False
    try:
        Regime.from_string(regime)
    except Exception:
        return False
    return True


def _scaled_activity_count(
    value: object,
    *,
    prior_weight: float,
    floor: int,
    cap: int,
) -> int:
    raw = _non_negative_int(value)
    if raw <= 0:
        return int(floor)
    return max(int(floor), min(int(cap), int(round(raw * prior_weight))))


def _positive_float(value: float, name: str) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return out


def _positive_int(value: int, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return out


def _float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _non_negative_int(value: object) -> int:
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, out)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

