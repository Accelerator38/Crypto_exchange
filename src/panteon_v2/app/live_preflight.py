"""Read-only live restart pre-flight checks for Panteon v3."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


VIRTUAL_TRADING_MODES = frozenset({"paper", "paper_live_feed", "shadow_live_feed"})


@dataclass(frozen=True)
class LivePreflightResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    matrix_summary_path: str = ""
    canary_summary_path: str = ""


@dataclass(frozen=True)
class LivePreflightConfig:
    project_root: str | Path
    matrix_summary_path: str | Path | None = None
    canary_summary_path: str | Path | None = None
    max_age_hours: float = 24.0
    now: datetime | None = None


def run_live_preflight(
    exchange: str,
    mode: str,
    *,
    config: LivePreflightConfig,
) -> LivePreflightResult:
    clean_mode = str(mode or "").strip().lower()
    if clean_mode in VIRTUAL_TRADING_MODES:
        return LivePreflightResult(True)

    reasons: list[str] = []
    root = Path(config.project_root)
    now = _aware_utc(config.now)
    matrix_path = _configured_or_latest(
        config.matrix_summary_path,
        env_name="PANTEON_LIVE_PREFLIGHT_MATRIX_SUMMARY",
        root=root,
        patterns=(
            "Reports/Panteon3PreLiveMatrix/**/panteon3_pre_live_matrix_summary.json",
        ),
    )
    canary_path = _configured_or_latest(
        config.canary_summary_path,
        env_name="PANTEON_LIVE_PREFLIGHT_CANARY_SUMMARY",
        root=root,
        patterns=(
            "Reports/Panteon3Canary/**/canary_summary.json",
            "Reports/Panteon3Canary/**/panteon3_live_canary_summary.json",
            "Reports/Panteon3Canary/latest_canary_summary.json",
        ),
    )

    matrix_payload = _load_json(matrix_path, reasons, missing_reason="matrix_missing")
    canary_payload = _load_json(canary_path, reasons, missing_reason="canary_missing")

    if matrix_payload is not None:
        _append_stale_reason(
            matrix_payload,
            matrix_path,
            now=now,
            max_age_hours=float(config.max_age_hours),
            reason="matrix_stale",
            reasons=reasons,
        )
        _validate_matrix(matrix_payload, reasons)
    if canary_payload is not None:
        _append_stale_reason(
            canary_payload,
            canary_path,
            now=now,
            max_age_hours=float(config.max_age_hours),
            reason="canary_stale",
            reasons=reasons,
        )
        _validate_canary(canary_payload, str(exchange or "").strip().upper(), reasons)

    return LivePreflightResult(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        matrix_summary_path=str(matrix_path or ""),
        canary_summary_path=str(canary_path or ""),
    )


def config_from_env(project_root: str | Path) -> LivePreflightConfig:
    raw_max_age = os.getenv("PANTEON_LIVE_PREFLIGHT_MAX_AGE_HOURS", "24")
    try:
        max_age_hours = float(raw_max_age)
    except (TypeError, ValueError):
        max_age_hours = 24.0
    return LivePreflightConfig(project_root=project_root, max_age_hours=max_age_hours)


def _configured_or_latest(
    configured: str | Path | None,
    *,
    env_name: str,
    root: Path,
    patterns: Sequence[str],
) -> Path | None:
    raw = str(configured or os.getenv(env_name, "") or "").strip()
    if raw:
        return Path(raw)
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _load_json(
    path: Path | None,
    reasons: list[str],
    *,
    missing_reason: str,
) -> Mapping[str, Any] | None:
    if path is None or not path.exists():
        reasons.append(missing_reason)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        reasons.append(f"{missing_reason.replace('missing', 'invalid')}:{exc}")
        return None
    if not isinstance(payload, Mapping):
        reasons.append(f"{missing_reason.replace('missing', 'invalid')}:not_object")
        return None
    return payload


def _validate_matrix(payload: Mapping[str, Any], reasons: list[str]) -> None:
    verdict = payload.get("promotion_verdict")
    if not isinstance(verdict, Mapping):
        reasons.append("matrix_verdict_missing")
        return
    if bool(verdict.get("passed")):
        _validate_matrix_best_component(payload, reasons)
        return
    fail_reasons = verdict.get("fail_reasons")
    if isinstance(fail_reasons, Sequence) and not isinstance(fail_reasons, (str, bytes)):
        for reason in fail_reasons:
            clean = str(reason or "").strip()
            if clean:
                reasons.append(f"matrix_failed:{clean}")
        if fail_reasons:
            return
    reasons.append("matrix_failed")


def _validate_matrix_best_component(
    payload: Mapping[str, Any],
    reasons: list[str],
) -> None:
    single_component = payload.get("single_component_candidate")
    if (
        isinstance(single_component, Mapping)
        and single_component.get("requires_separate_matrix_artifact") is False
    ):
        return
    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping):
        return
    beats_flag = candidate.get("panteon_beats_best_component")
    candidate_pnl = _float_metric(candidate, "realized_pnl_usd", "pnl_usd", "net_pnl")
    best_component_pnl = _float_metric(candidate, "best_component_pnl_usd")
    best_component_label = str(candidate.get("best_component_label") or "").strip()
    explicit_loss = beats_flag is False
    numeric_loss = bool(best_component_label) and best_component_pnl > candidate_pnl
    if explicit_loss or numeric_loss:
        reasons.append("matrix_not_beating_best_component")


def _validate_canary(
    payload: Mapping[str, Any],
    exchange: str,
    reasons: list[str],
) -> None:
    exchanges = payload.get("exchanges")
    if isinstance(exchanges, Mapping) and exchange:
        raw_exchange_payload = exchanges.get(exchange)
        if not isinstance(raw_exchange_payload, Mapping):
            reasons.append("canary_exchange_missing")
            return
        exchange_payload = raw_exchange_payload
    else:
        exchange_payload = payload

    if bool(payload.get("calibration_only")) or bool(exchange_payload.get("calibration_only")):
        reasons.append("canary_calibration_only")
    if _has_actor_overrides(payload) or _has_actor_overrides(exchange_payload):
        reasons.append("canary_actor_overrides")

    raw_fail_reasons = exchange_payload.get("fail_reasons")
    if isinstance(raw_fail_reasons, Sequence) and not isinstance(raw_fail_reasons, (str, bytes)):
        for reason in raw_fail_reasons:
            clean = str(reason or "").strip()
            if clean:
                reasons.append(f"canary_{clean}")

    if not bool(exchange_payload.get("passed", payload.get("passed", False))):
        reasons.append("canary_failed")
    if _int_metric(exchange_payload, "signals", "signal_count") <= 0:
        reasons.append("canary_zero_signals")
    if _int_metric(exchange_payload, "orders", "order_count", "sent_orders") <= 0:
        reasons.append("canary_zero_orders")
    if _int_metric(exchange_payload, "fills", "filled", "filled_signals") <= 0:
        reasons.append("canary_zero_fills")
    if (
        exchange_payload.get(
            "expectancy_gate_required",
            payload.get("expectancy_gate_required", True),
        )
        is False
    ):
        reasons.append("canary_expectancy_gate_disabled")
    expectancy = _float_metric(
        exchange_payload,
        "expectancy_after_costs",
        "expectancy_usd",
        "expectancy",
    )
    if expectancy <= 0.0:
        reasons.append("canary_nonpositive_expectancy")
    if not bool(exchange_payload.get("reconcile_ok", True)):
        reasons.append("canary_reconcile_failed")
    owned_open_count = _int_metric(exchange_payload, "owned_open_position_count")
    open_count = _int_metric(exchange_payload, "open_position_count")
    if owned_open_count > 0 or ("owned_open_position_count" not in exchange_payload and open_count > 0):
        reasons.append("canary_open_positions")
    for key in ("reconcile_warnings", "health_warnings"):
        value = exchange_payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
            reasons.append(f"canary_{key}")


def _has_actor_overrides(payload: Mapping[str, Any]) -> bool:
    value = payload.get("actor_overrides")
    return isinstance(value, Mapping) and bool(value)


def _append_stale_reason(
    payload: Mapping[str, Any],
    path: Path | None,
    *,
    now: datetime,
    max_age_hours: float,
    reason: str,
    reasons: list[str],
) -> None:
    if max_age_hours <= 0:
        return
    generated_at = _timestamp_from_payload(payload)
    if generated_at is None and path is not None and path.exists():
        generated_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    if generated_at is None:
        reasons.append(reason)
        return
    age_hours = (now - generated_at).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        reasons.append(reason)


def _timestamp_from_payload(payload: Mapping[str, Any]) -> datetime | None:
    for key in ("generated_at", "timestamp", "timestamp_utc", "created_at"):
        parsed = _parse_datetime(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: Any) -> datetime | None:
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


def _int_metric(payload: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        try:
            return int(payload.get(key))
        except (TypeError, ValueError):
            continue
    return 0


def _float_metric(payload: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            return float(payload.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0
