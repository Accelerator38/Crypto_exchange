"""Read-only live restart pre-flight checks for Panteon v3."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..policy.manifest import ManifestError, PolicyTarget, load_policy_manifest


VIRTUAL_TRADING_MODES = frozenset({"paper", "paper_live_feed", "shadow_live_feed"})


@dataclass(frozen=True)
class LivePreflightResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    matrix_summary_path: str = ""
    canary_summary_path: str = ""
    policy_manifest_path: str = ""


@dataclass(frozen=True)
class LivePreflightConfig:
    project_root: str | Path
    matrix_summary_path: str | Path | None = None
    canary_summary_path: str | Path | None = None
    policy_manifest_path: str | Path | None = None
    expected_policy_sha256: str = ""
    require_bitget_policy_v1: bool = False
    require_player_only_v1: bool = False
    require_clean_git: bool = False
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
            "Results/PlayerEfficiency*/**/player_efficiency_summary.json",
            "Results/PlayerEfficiency/**/player_efficiency_summary.json",
            "Reports/PlayerEfficiency/**/player_efficiency_summary.json",
            "Reports/Panteon3PreLiveMatrix/**/panteon3_pre_live_matrix_summary.json",
        ),
    )
    canary_path = _configured_or_latest(
        config.canary_summary_path,
        env_name="PANTEON_LIVE_PREFLIGHT_CANARY_SUMMARY",
        root=root,
        patterns=(
            "Reports/PlayerCanary/**/player_canary_summary.json",
            "Reports/Panteon3Canary/**/canary_summary.json",
            "Reports/Panteon3Canary/**/panteon3_live_canary_summary.json",
            "Reports/Panteon3Canary/latest_canary_summary.json",
        ),
    )
    policy_manifest_path = _configured_policy_manifest_path(config, root=root)

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

    if bool(config.require_player_only_v1):
        if matrix_payload is not None:
            _validate_player_only_contract(matrix_payload, reasons, source="matrix")
        if canary_payload is not None:
            canary_contract = _exchange_payload(
                canary_payload,
                str(exchange or "").strip().upper(),
            )
            if canary_contract is not None:
                _validate_player_only_contract(
                    canary_contract,
                    reasons,
                    source="canary",
                )

    if (
        bool(config.require_bitget_policy_v1)
        and str(exchange or "").strip().upper() == "BITGET"
    ):
        _validate_bitget_policy_manifest(
            policy_manifest_path,
            expected_sha256=str(config.expected_policy_sha256 or ""),
            root=root,
            now=now,
            require_clean_git=bool(config.require_clean_git),
            reasons=reasons,
        )

    return LivePreflightResult(
        passed=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        matrix_summary_path=str(matrix_path or ""),
        canary_summary_path=str(canary_path or ""),
        policy_manifest_path=str(policy_manifest_path or ""),
    )


def config_from_env(project_root: str | Path) -> LivePreflightConfig:
    raw_max_age = os.getenv("PANTEON_LIVE_PREFLIGHT_MAX_AGE_HOURS", "24")
    try:
        max_age_hours = float(raw_max_age)
    except (TypeError, ValueError):
        max_age_hours = 24.0
    # The production launcher now has only player-only multi/singleton modes.
    # Matrix/canary freshness remains mandatory; the retired policy manifest is
    # no longer live authority for those modes.
    return LivePreflightConfig(
        project_root=project_root,
        policy_manifest_path=(
            os.getenv("BITGET_POLICY_MANIFEST_V1")
            or "Runtime/BITGET/active_policy_manifest_v1.json"
        ),
        expected_policy_sha256=os.getenv("BITGET_POLICY_MANIFEST_SHA256", ""),
        require_bitget_policy_v1=False,
        require_player_only_v1=True,
        require_clean_git=True,
        max_age_hours=max_age_hours,
    )


def _configured_policy_manifest_path(
    config: LivePreflightConfig,
    *,
    root: Path,
) -> Path | None:
    raw = str(config.policy_manifest_path or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else root / path


def _validate_bitget_policy_manifest(
    path: Path | None,
    *,
    expected_sha256: str,
    root: Path,
    now: datetime,
    require_clean_git: bool,
    reasons: list[str],
) -> None:
    pin = str(expected_sha256 or "").strip().lower()
    if not pin:
        reasons.append("policy_manifest_sha256_pin_missing")
        return
    if path is None:
        reasons.append("policy_manifest_missing")
        return
    try:
        loaded = load_policy_manifest(
            path,
            project_root=root,
            expected_sha256=pin,
            now=now,
            verify_artifacts=True,
            required_target=PolicyTarget.MICRO_LIVE,
            required_exchange="BITGET",
        )
    except ManifestError as exc:
        reasons.extend(f"policy_manifest_invalid:{reason}" for reason in exc.reasons)
        return

    revision, dirty, git_error = _git_state(root)
    if git_error:
        reasons.append(f"policy_runtime_git_error:{git_error}")
        return
    source_revision = loaded.manifest.source_revision
    if not revision.startswith(source_revision):
        reasons.append("policy_manifest_source_revision_mismatch")
    if require_clean_git and dirty:
        reasons.append("policy_runtime_git_dirty")


def _git_state(root: Path) -> tuple[str, bool, str]:
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if revision_result.returncode != 0:
            return "", False, "revision_unavailable"
        revision = revision_result.stdout.strip().lower()
        status_result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status_result.returncode != 0:
            return revision, False, "status_unavailable"
        return revision, bool(status_result.stdout.strip()), ""
    except (OSError, subprocess.SubprocessError):
        return "", False, "command_failed"


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
    exchange_payload = _exchange_payload(payload, exchange)
    if exchange_payload is None:
        reasons.append("canary_exchange_missing")
        return

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


def _exchange_payload(
    payload: Mapping[str, Any],
    exchange: str,
) -> Mapping[str, Any] | None:
    exchanges = payload.get("exchanges")
    if isinstance(exchanges, Mapping) and exchange:
        raw_exchange_payload = exchanges.get(exchange)
        return raw_exchange_payload if isinstance(raw_exchange_payload, Mapping) else None
    return payload


def _validate_player_only_contract(
    payload: Mapping[str, Any],
    reasons: list[str],
    *,
    source: str,
) -> None:
    """Reject legacy evidence that did not exercise the production runtime."""
    checks = {
        "runtime_contract": payload.get("runtime_contract") == "pantheon_players_v1",
        "player_only_runtime": payload.get("player_only_runtime") is True,
        "causal_selection": payload.get("causal_selection") is True,
        "player_regime_memory": payload.get("memory_scope") == "player_regime",
        "recency_weighted": payload.get("recency_weighted") is True,
        "single_real_player": _int_metric(payload, "max_real_players") == 1,
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"{source}_player_contract_{name}_missing")
    if source == "canary" and _int_metric(payload, "shadow_player_count") <= 0:
        reasons.append("canary_player_contract_shadow_players_missing")


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
