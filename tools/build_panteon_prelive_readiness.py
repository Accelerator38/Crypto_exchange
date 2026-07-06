from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.app.live_preflight import (  # noqa: E402
    LivePreflightConfig,
    run_live_preflight,
)


DEFAULT_EXCHANGES = ("MEXC", "BITGET")
DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "TRX")
GENETICS_CORE_LABEL = "geneticscore"
LIVE_ADMISSION_KEY_PARTS = (
    "live_real_actor_whitelist",
    "range_low_vol_real_actor_allowlist",
    "promotion_derived_actor_labels",
)
BYPASS_KEY_PARTS = ("genetics", "bypass")
TRUTHY = {"1", "true", "yes", "y", "on", "enabled", "enable"}
FALSEY = {"0", "false", "no", "n", "off", "disabled", "disable"}


def _aware_utc(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_symbols(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_SYMBOLS
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    else:
        parts: list[str] = []
        for item in raw:
            parts.extend(str(item or "").replace(";", ",").split(","))
    cleaned = tuple(dict.fromkeys(part.strip().upper() for part in parts if part.strip()))
    return cleaned or DEFAULT_SYMBOLS


def _parse_settings(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    if not path.exists():
        return settings
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip().lower()
        if clean_key:
            settings[clean_key] = value.strip()
    return settings


def _is_truthy(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in FALSEY or not text:
        return False
    return text in TRUTHY


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").replace(";", ",").split(",") if part.strip())


def _base_actor_label(value: str) -> str:
    text = str(value or "").strip()
    if "|" in text:
        text = text.split("|", 1)[0]
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    changed = True
    while changed:
        changed = False
        for prefix in ("Solo_", "V_", "ensemble:Solo_", "agent:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
    return re.sub(r"[^A-Za-z0-9_]", "", text).lower()


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_settings_safety_snapshot(
    *,
    project_root: str | Path,
    exchanges: Sequence[str] = DEFAULT_EXCHANGES,
    now: datetime | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    settings_path = root / "settings.txt"
    settings = _parse_settings(settings_path)
    current = _aware_utc(now)
    tracked: dict[str, Any] = {}
    genetics_core_live_admission: list[dict[str, Any]] = []
    bypass_enabled: list[str] = []
    probation_enabled: list[str] = []
    primary_enabled: list[str] = []

    for key, value in sorted(settings.items()):
        include_key = (
            any(part in key for part in LIVE_ADMISSION_KEY_PARTS)
            or all(part in key for part in BYPASS_KEY_PARTS)
            or key.endswith("genetics_probation_execution_enabled")
            or key.endswith("flash_genetics_core_primary_enabled")
            or key.endswith("v2_genetics_shadow_only")
        )
        if not include_key:
            continue
        if any(part in key for part in LIVE_ADMISSION_KEY_PARTS):
            values = _split_csv(value)
            normalized = [_base_actor_label(item) for item in values]
            tracked[key] = {
                "value_count": len(values),
                "contains_genetics_core": GENETICS_CORE_LABEL in normalized,
                "normalized_values": normalized,
            }
            matches = [
                item
                for item in values
                if _base_actor_label(item) == GENETICS_CORE_LABEL
            ]
            if matches:
                genetics_core_live_admission.append({"key": key, "values": matches})
        else:
            truthy = _is_truthy(value)
            tracked[key] = {"truthy": truthy}
            if all(part in key for part in BYPASS_KEY_PARTS) and truthy:
                bypass_enabled.append(key)
            if key.endswith("genetics_probation_execution_enabled") and truthy:
                probation_enabled.append(key)
            if key.endswith("flash_genetics_core_primary_enabled") and truthy:
                primary_enabled.append(key)

    missing_explicit_flags: list[str] = []
    for exchange in exchanges:
        prefix = str(exchange or "").strip().lower()
        flag = f"{prefix}_v2_genetics_probation_execution_enabled"
        if prefix and flag not in settings:
            missing_explicit_flags.append(flag)

    safety_blockers = []
    if genetics_core_live_admission:
        safety_blockers.append("genetics_core_live_admission")
    if bypass_enabled:
        safety_blockers.append("genetics_bypass_enabled")
    if probation_enabled:
        safety_blockers.append("genetics_probation_execution_enabled")
    if primary_enabled:
        safety_blockers.append("genetics_core_primary_enabled")

    return {
        "schema": "panteon_settings_safety_snapshot.v1",
        "generated_at": current.isoformat(),
        "source_path": str(settings_path),
        "source_sha256": _sha256(settings_path),
        "source_git_ignored": _git_check_ignore(Path("settings.txt"), cwd=root),
        "exchanges": [str(item or "").strip().upper() for item in exchanges],
        "tracked_settings": tracked,
        "safety_verdict": {
            "passed": not safety_blockers,
            "blockers": safety_blockers,
            "genetics_core_live_admission": genetics_core_live_admission,
            "genetics_bypass_enabled_keys": bypass_enabled,
            "genetics_probation_execution_enabled_keys": probation_enabled,
            "genetics_core_primary_enabled_keys": primary_enabled,
            "missing_explicit_flags": missing_explicit_flags,
        },
    }


def _run_git(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _git_check_ignore(path: Path, *, cwd: Path) -> bool:
    try:
        result = _run_git(["check-ignore", str(path)], cwd=cwd)
    except OSError:
        return False
    return result.returncode == 0


def _add_blocker(blockers: list[dict[str, Any]], item_id: str, message: str, **extra: Any) -> None:
    payload = {"id": item_id, "severity": "blocker", "message": message}
    payload.update(extra)
    blockers.append(payload)


def _add_warning(warnings: list[dict[str, Any]], item_id: str, message: str, **extra: Any) -> None:
    payload = {"id": item_id, "severity": "warning", "message": message}
    payload.update(extra)
    warnings.append(payload)


def check_git_section(project_root: Path, blockers: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        result = _run_git(["status", "--short"], cwd=project_root)
    except OSError as exc:
        _add_blocker(blockers, "git.unavailable", f"git is unavailable: {exc}")
        return {"passed": False, "status_short": "", "error": str(exc)}
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git status failed").strip()
        _add_blocker(blockers, "git.status_failed", message)
        return {"passed": False, "status_short": result.stdout, "error": message}
    status_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if status_lines:
        _add_blocker(
            blockers,
            "git.dirty_worktree",
            "worktree has unstaged, staged, or untracked changes",
            files=status_lines,
        )
    branch = _run_git(["branch", "--show-current"], cwd=project_root)
    return {
        "passed": not status_lines,
        "branch": branch.stdout.strip() if branch.returncode == 0 else "",
        "status_short": status_lines,
    }


def check_settings_section(
    project_root: Path,
    exchanges: Sequence[str],
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    settings_snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    settings_path = project_root / "settings.txt"
    settings = _parse_settings(settings_path)
    section: dict[str, Any] = {
        "path": str(settings_path),
        "exists": settings_path.exists(),
        "sha256": _sha256(settings_path),
        "git_ignored": _git_check_ignore(Path("settings.txt"), cwd=project_root),
        "settings_snapshot_path": str(settings_snapshot_path or ""),
        "genetics_core_live_admission_keys": [],
        "genetics_bypass_enabled_keys": [],
        "genetics_probation_execution_enabled_keys": [],
    }
    if not settings_path.exists():
        _add_blocker(blockers, "settings.missing", "settings.txt is missing")
        section["passed"] = False
        return section

    if section["git_ignored"] and not settings_snapshot_path:
        _add_blocker(
            blockers,
            "settings.snapshot_missing",
            "settings.txt is git-ignored; create or pass a reviewed settings snapshot before live",
        )
    elif settings_snapshot_path:
        snapshot_path = Path(settings_snapshot_path)
        if not snapshot_path.is_absolute():
            snapshot_path = project_root / snapshot_path
        section["settings_snapshot_path"] = str(snapshot_path)
        if not snapshot_path.exists():
            _add_blocker(
                blockers,
                "settings.snapshot_missing",
                f"settings snapshot does not exist: {settings_snapshot_path}",
            )
        else:
            try:
                snapshot = _load_json(snapshot_path)
            except Exception as exc:
                snapshot = None
                _add_blocker(
                    blockers,
                    "settings.snapshot_invalid",
                    f"settings snapshot is invalid: {exc}",
                )
            if isinstance(snapshot, Mapping):
                section["settings_snapshot_sha256"] = str(
                    snapshot.get("source_sha256") or snapshot.get("settings_sha256") or ""
                )
                if section["settings_snapshot_sha256"] != section["sha256"]:
                    _add_blocker(
                        blockers,
                        "settings.snapshot_stale",
                        "settings snapshot SHA-256 does not match current settings.txt",
                        snapshot_path=str(snapshot_path),
                    )
                verdict = snapshot.get("safety_verdict")
                if isinstance(verdict, Mapping) and not bool(verdict.get("passed", False)):
                    _add_blocker(
                        blockers,
                        "settings.snapshot_failed",
                        "settings snapshot safety verdict is failed",
                        snapshot_path=str(snapshot_path),
                        snapshot_blockers=list(verdict.get("blockers") or []),
                    )
                elif not isinstance(verdict, Mapping):
                    _add_blocker(
                        blockers,
                        "settings.snapshot_verdict_missing",
                        "settings snapshot has no safety_verdict object",
                        snapshot_path=str(snapshot_path),
                    )

    for key, value in settings.items():
        if any(part in key for part in LIVE_ADMISSION_KEY_PARTS):
            matches = [
                item
                for item in _split_csv(value)
                if _base_actor_label(item) == GENETICS_CORE_LABEL
            ]
            if matches:
                section["genetics_core_live_admission_keys"].append(
                    {"key": key, "values": matches}
                )
        if all(part in key for part in BYPASS_KEY_PARTS) and _is_truthy(value):
            section["genetics_bypass_enabled_keys"].append(key)
        if key.endswith("genetics_probation_execution_enabled") and _is_truthy(value):
            section["genetics_probation_execution_enabled_keys"].append(key)
        if key.endswith("flash_genetics_core_primary_enabled") and _is_truthy(value):
            section.setdefault("genetics_core_primary_enabled_keys", []).append(key)

    if section["genetics_core_live_admission_keys"]:
        _add_blocker(
            blockers,
            "settings.genetics_core_live_admission",
            "GeneticsCore is present in live admission/allowlist settings",
            keys=section["genetics_core_live_admission_keys"],
        )
    if section["genetics_bypass_enabled_keys"]:
        _add_blocker(
            blockers,
            "settings.genetics_bypass_enabled",
            "genetics bypass flag is enabled",
            keys=section["genetics_bypass_enabled_keys"],
        )
    if section["genetics_probation_execution_enabled_keys"]:
        _add_blocker(
            blockers,
            "settings.genetics_probation_execution_enabled",
            "genetics probation execution is enabled before promotion",
            keys=section["genetics_probation_execution_enabled_keys"],
        )
    if section.get("genetics_core_primary_enabled_keys"):
        _add_blocker(
            blockers,
            "settings.genetics_core_primary_enabled",
            "GeneticsCore primary live path is enabled before promotion",
            keys=section["genetics_core_primary_enabled_keys"],
        )
    for exchange in exchanges:
        prefix = str(exchange or "").strip().lower()
        if prefix and f"{prefix}_v2_genetics_probation_execution_enabled" not in settings:
            _add_warning(
                warnings,
                f"settings.{prefix}.probation_flag_missing",
                f"{prefix} genetics probation execution flag is not explicit",
            )

    section["passed"] = not any(
        item["id"].startswith("settings.") for item in blockers
    )
    return section


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _status_timestamp(status: Mapping[str, Any]) -> datetime | None:
    for key in ("timestamp_utc", "timestamp", "timestamp_msk", "generated_at"):
        parsed = _parse_datetime(status.get(key))
        if parsed is not None:
            return parsed
    return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_status_path(project_root: Path, exchange: str) -> Path | None:
    exchange_root = project_root / "Results" / exchange.upper()
    if not exchange_root.exists():
        return None
    statuses = [path for path in exchange_root.glob("**/status.json") if path.is_file()]
    if not statuses:
        return None
    return max(statuses, key=lambda path: path.stat().st_mtime)


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _stale_status_cleanup_hints(project_root: Path, status_path: Path) -> list[str]:
    return [
        f"Review or archive stale status: {_display_path(project_root, status_path)}",
        "Run a fresh isolated paper canary before re-checking readiness",
    ]


def _orphan_status_cleanup_hints(
    project_root: Path,
    status_path: Path,
    *,
    exchange: str,
    pid: int,
) -> list[str]:
    return [
        f"Confirm no live {exchange} worker owns pid {pid}",
        f"Review or archive stale status: {_display_path(project_root, status_path)}",
    ]


def _pid_exists(pid: int, active_pids: set[int] | None) -> bool:
    if pid <= 0:
        return False
    if active_pids is not None:
        return pid in active_pids
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _activity_summary(log_path: Path) -> dict[str, int]:
    summary = {"raw_signals": 0, "signals": 0, "filled": 0, "rejected": 0, "blocked": 0}
    if not log_path.exists():
        return summary
    pattern = re.compile(
        r"raw_signals=(?P<raw_signals>\d+).*?"
        r"signals=(?P<signals>\d+).*?"
        r"filled=(?P<filled>\d+).*?"
        r"rejected=(?P<rejected>\d+).*?"
        r"blocked=(?P<blocked>\d+)"
    )
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return summary
    for line in lines[-2000:]:
        match = pattern.search(line)
        if not match:
            continue
        for key, raw in match.groupdict().items():
            summary[key] = max(summary[key], int(raw))
    return summary


def check_live_status_section(
    project_root: Path,
    exchanges: Sequence[str],
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    now: datetime,
    active_pids: set[int] | None,
    max_status_age_hours: float,
) -> dict[str, Any]:
    section: dict[str, Any] = {"passed": True, "exchanges": {}}
    for exchange in exchanges:
        key = str(exchange or "").strip().upper()
        status_path = _latest_status_path(project_root, key)
        if status_path is None:
            _add_blocker(blockers, f"live_status.{key}.missing", f"{key}: status.json is missing")
            section["exchanges"][key] = {"passed": False, "status_path": ""}
            section["passed"] = False
            continue
        try:
            status = _load_json(status_path)
        except Exception as exc:
            _add_blocker(
                blockers,
                f"live_status.{key}.invalid",
                f"{key}: status.json is invalid: {exc}",
            )
            section["exchanges"][key] = {"passed": False, "status_path": str(status_path)}
            section["passed"] = False
            continue
        if not isinstance(status, Mapping):
            _add_blocker(blockers, f"live_status.{key}.invalid", f"{key}: status.json is not an object")
            section["exchanges"][key] = {"passed": False, "status_path": str(status_path)}
            section["passed"] = False
            continue
        timestamp = _status_timestamp(status)
        age_hours = None
        passed = True
        pid = int(status.get("pid") or status.get("active_pid") or 0)
        looks_running = str(status.get("run_state") or "").lower() == "running" or str(
            status.get("feed_status") or ""
        ).lower() == "active"
        pid_running = _pid_exists(pid, active_pids)
        if timestamp is None:
            passed = False
            _add_blocker(blockers, f"live_status.{key}.timestamp_missing", f"{key}: status timestamp is missing")
        else:
            age_hours = (now - timestamp).total_seconds() / 3600.0
            if age_hours > float(max_status_age_hours):
                if looks_running and pid_running:
                    passed = False
                    _add_blocker(
                        blockers,
                        f"live_status.{key}.stale_status",
                        f"{key}: active live status is stale ({age_hours:.1f}h old)",
                        status_path=str(status_path),
                        cleanup_hints=_stale_status_cleanup_hints(project_root, status_path),
                    )
                else:
                    _add_warning(
                        warnings,
                        f"live_status.{key}.stale_inactive_status",
                        f"{key}: latest status is stale ({age_hours:.1f}h old), but no matching live process is running",
                        status_path=str(status_path),
                        cleanup_hints=_stale_status_cleanup_hints(project_root, status_path),
                    )
        if looks_running and not pid_running:
            _add_warning(
                warnings,
                f"live_status.{key}.orphan_running_status",
                f"{key}: status says running/active but pid {pid} is not running",
                status_path=str(status_path),
                cleanup_hints=_orphan_status_cleanup_hints(
                    project_root,
                    status_path,
                    exchange=key,
                    pid=pid,
                ),
            )
        activity = _activity_summary(status_path.parent / "trading.log")
        if activity["filled"] > 0:
            _add_warning(
                warnings,
                f"live_status.{key}.real_fills_detected",
                f"{key}: latest trading.log contains filled={activity['filled']}",
                status_path=str(status_path),
            )
        section["exchanges"][key] = {
            "passed": passed,
            "status_path": str(status_path),
            "timestamp_utc": timestamp.isoformat() if timestamp else "",
            "age_hours": age_hours,
            "pid": pid,
            "pid_running": pid_running,
            "run_state": str(status.get("run_state") or ""),
            "feed_status": str(status.get("feed_status") or ""),
            "mode": str(status.get("mode") or ""),
            "activity": activity,
        }
        section["passed"] = section["passed"] and passed
    return section


def check_live_preflight_section(
    project_root: Path,
    exchanges: Sequence[str],
    blockers: list[dict[str, Any]],
    *,
    now: datetime,
    max_age_hours: float,
) -> dict[str, Any]:
    section: dict[str, Any] = {"passed": True, "exchanges": {}}
    for exchange in exchanges:
        key = str(exchange or "").strip().upper()
        result = run_live_preflight(
            key,
            "live_futures",
            config=LivePreflightConfig(
                project_root=project_root,
                max_age_hours=float(max_age_hours),
                now=now,
            ),
        )
        payload = {
            "passed": result.passed,
            "reasons": list(result.reasons),
            "matrix_summary_path": result.matrix_summary_path,
            "canary_summary_path": result.canary_summary_path,
        }
        section["exchanges"][key] = payload
        section["passed"] = section["passed"] and result.passed
        for reason in result.reasons:
            _add_blocker(
                blockers,
                f"live_preflight.{key}.{reason}",
                f"{key}: live preflight failed: {reason}",
            )
    return section


def _latest_file(root: Path, patterns: Sequence[str]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    files = [path for path in candidates if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _resolve_artifact_path(
    project_root: Path,
    explicit_path: str | Path | None,
    patterns: Sequence[str],
) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_absolute() else project_root / path
    return _latest_file(project_root, patterns)


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _has_overrides(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(list(value))
    return bool(value)


def check_matrix_artifact_section(
    project_root: Path,
    blockers: list[dict[str, Any]],
    *,
    matrix_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    path = _resolve_artifact_path(
        project_root,
        matrix_summary_path,
        ("Reports/Panteon3PreLiveMatrix/**/panteon3_pre_live_matrix_summary.json",),
    )
    section: dict[str, Any] = {"passed": True, "path": str(path or "")}
    if path is None or not path.exists():
        _add_blocker(blockers, "matrix.missing", "pre-live matrix summary is missing")
        section["passed"] = False
        return section
    try:
        payload = _load_json(path)
    except Exception as exc:
        _add_blocker(blockers, "matrix.invalid", f"pre-live matrix summary is invalid: {exc}")
        section["passed"] = False
        return section
    if not isinstance(payload, Mapping):
        _add_blocker(blockers, "matrix.invalid", "pre-live matrix summary is not an object")
        section["passed"] = False
        return section
    verdict = payload.get("promotion_verdict")
    if not isinstance(verdict, Mapping):
        _add_blocker(blockers, "matrix.verdict_missing", "pre-live matrix has no promotion_verdict object")
        section["passed"] = False
        return section
    fail_reasons = list(verdict.get("fail_reasons") or [])
    candidate = payload.get("candidate") if isinstance(payload.get("candidate"), Mapping) else {}
    section.update(
        {
            "passed": bool(verdict.get("passed", False)),
            "fail_reasons": fail_reasons,
            "candidate_variant": str(candidate.get("variant") or candidate.get("base_variant") or ""),
            "filled_signals": _as_int(candidate.get("filled_signals")),
            "closed_trades": _as_int(candidate.get("closed_trades")),
            "expectancy_usd": _as_float(candidate.get("expectancy_usd")),
            "cost_attribution_present": bool(candidate.get("cost_attribution_present", False)),
        }
    )
    if not section["passed"]:
        _add_blocker(
            blockers,
            "matrix.failed",
            "pre-live matrix promotion verdict failed",
            matrix_summary_path=str(path),
            fail_reasons=fail_reasons,
        )
    return section


def check_canary_artifact_section(
    project_root: Path,
    exchanges: Sequence[str],
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    canary_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    path = _resolve_artifact_path(
        project_root,
        canary_summary_path,
        ("Reports/Panteon3Canary/**/panteon3_live_canary_summary.json",),
    )
    section: dict[str, Any] = {"passed": True, "path": str(path or ""), "exchanges": {}}
    if path is None or not path.exists():
        _add_blocker(blockers, "canary.missing", "paper/live canary summary is missing")
        section["passed"] = False
        return section
    try:
        payload = _load_json(path)
    except Exception as exc:
        _add_blocker(blockers, "canary.invalid", f"paper/live canary summary is invalid: {exc}")
        section["passed"] = False
        return section
    if not isinstance(payload, Mapping):
        _add_blocker(blockers, "canary.invalid", "paper/live canary summary is not an object")
        section["passed"] = False
        return section
    exchanges_payload = payload.get("exchanges")
    if not isinstance(exchanges_payload, Mapping):
        _add_blocker(blockers, "canary.exchanges_missing", "paper/live canary has no exchanges object")
        section["passed"] = False
        return section

    actor_overrides = payload.get("actor_overrides")
    calibration_only = bool(payload.get("calibration_only", False))
    execution_smoke = bool(payload.get("execution_smoke", False))
    expectancy_gate_required = bool(payload.get("expectancy_gate_required", True))
    section.update(
        {
            "passed": bool(payload.get("passed", False)),
            "generated_at": str(payload.get("generated_at") or ""),
            "calibration_only": calibration_only,
            "execution_smoke": execution_smoke,
            "actor_overrides": actor_overrides or {},
            "expectancy_gate_required": expectancy_gate_required,
        }
    )
    if not section["passed"]:
        _add_blocker(
            blockers,
            "canary.failed",
            "paper/live canary summary failed",
            canary_summary_path=str(path),
        )
    if execution_smoke:
        section["passed"] = False
        _add_blocker(
            blockers,
            "canary.execution_smoke",
            "latest canary is execution-smoke and cannot promote to live",
            canary_summary_path=str(path),
        )
    if calibration_only:
        _add_blocker(
            blockers,
            "canary.calibration_only",
            "latest canary is calibration-only and cannot promote to live",
            canary_summary_path=str(path),
        )
    if _has_overrides(actor_overrides):
        _add_blocker(
            blockers,
            "canary.actor_overrides",
            "latest canary used actor overrides and cannot promote to live",
            canary_summary_path=str(path),
            actor_overrides=actor_overrides,
        )

    for exchange in exchanges:
        key = str(exchange or "").strip().upper()
        item = exchanges_payload.get(key)
        if not isinstance(item, Mapping):
            _add_blocker(
                blockers,
                f"canary.{key}.missing_exchange",
                f"{key}: canary exchange summary is missing",
                canary_summary_path=str(path),
            )
            section["exchanges"][key] = {"passed": False}
            section["passed"] = False
            continue
        signals = _as_int(item.get("signals"))
        orders = _as_int(item.get("orders"))
        fills = _as_int(item.get("fills"))
        expectancy = _as_float(item.get("expectancy_after_costs"))
        fail_reasons = list(item.get("fail_reasons") or [])
        exchange_passed = bool(item.get("passed", False))
        exchange_section = {
            "passed": exchange_passed,
            "fail_reasons": fail_reasons,
            "signals": signals,
            "orders": orders,
            "fills": fills,
            "expectancy_after_costs": expectancy,
            "mode": str(item.get("mode") or ""),
            "status_path": str(item.get("status_path") or ""),
            "negative_context_signal_keys": list(item.get("negative_context_signal_keys") or []),
        }
        section["exchanges"][key] = exchange_section
        if not exchange_passed:
            _add_blocker(
                blockers,
                f"canary.{key}.failed",
                f"{key}: canary exchange summary failed",
                canary_summary_path=str(path),
                fail_reasons=fail_reasons,
            )
        if signals <= 0:
            _add_blocker(blockers, f"canary.{key}.zero_signals", f"{key}: canary produced zero signals")
        if orders <= 0:
            _add_blocker(blockers, f"canary.{key}.zero_orders", f"{key}: canary produced zero orders")
        if fills <= 0:
            _add_blocker(blockers, f"canary.{key}.zero_fills", f"{key}: canary produced zero fills")
        if expectancy_gate_required and expectancy <= 0:
            _add_blocker(
                blockers,
                f"canary.{key}.nonpositive_expectancy",
                f"{key}: canary expectancy after costs is non-positive",
                expectancy_after_costs=expectancy,
            )
        if exchange_section["negative_context_signal_keys"]:
            _add_warning(
                warnings,
                f"canary.{key}.negative_contexts",
                f"{key}: canary has negative closed-trade contexts",
                negative_context_signal_keys=exchange_section["negative_context_signal_keys"],
            )
        section["passed"] = section["passed"] and exchange_passed and signals > 0 and orders > 0 and fills > 0
        if expectancy_gate_required:
            section["passed"] = section["passed"] and expectancy > 0
    return section


def check_activation_artifact_section(
    project_root: Path,
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    activation_report_path: str | Path | None = None,
) -> dict[str, Any]:
    path = _resolve_artifact_path(
        project_root,
        activation_report_path,
        ("Reports/Panteon3Canary/**/live_oi_breakout_activation_report.json",),
    )
    section: dict[str, Any] = {"passed": True, "path": str(path or "")}
    if path is None or not path.exists():
        _add_blocker(blockers, "activation.missing", "LiveOIBreakout activation report is missing")
        section["passed"] = False
        return section
    try:
        payload = _load_json(path)
    except Exception as exc:
        _add_blocker(blockers, "activation.invalid", f"activation report is invalid: {exc}")
        section["passed"] = False
        return section
    if not isinstance(payload, Mapping):
        _add_blocker(blockers, "activation.invalid", "activation report is not an object")
        section["passed"] = False
        return section
    hard_blocked = bool(payload.get("hard_blocked", False))
    activation_blockers = list(payload.get("activation_blockers") or [])
    diagnostic_warnings = list(payload.get("diagnostic_warnings") or [])
    candidate_signal_count = _as_int(payload.get("candidate_signal_count"))
    section.update(
        {
            "passed": not hard_blocked and candidate_signal_count > 0,
            "hard_blocked": hard_blocked,
            "activation_blockers": activation_blockers,
            "diagnostic_warnings": diagnostic_warnings,
            "candidate_signal_count": candidate_signal_count,
            "real_check_count": _as_int(payload.get("real_check_count")),
            "wait_check_count": _as_int(payload.get("wait_check_count")),
            "sufficient_real_checks": bool(payload.get("sufficient_real_checks", False)),
            "sufficient_real_checks_per_file": bool(
                payload.get("sufficient_real_checks_per_file", False)
            ),
        }
    )
    if hard_blocked:
        _add_blocker(
            blockers,
            "activation.hard_blocked",
            "LiveOIBreakout activation report is hard-blocked",
            activation_report_path=str(path),
            activation_blockers=activation_blockers,
        )
    if candidate_signal_count <= 0:
        _add_blocker(
            blockers,
            "activation.zero_candidate_signals",
            "LiveOIBreakout activation report has zero candidate signals",
            activation_report_path=str(path),
        )
    for warning in diagnostic_warnings:
        warning_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(warning or "warning"))
        _add_warning(
            warnings,
            f"activation.{warning_id}",
            f"LiveOIBreakout activation diagnostic warning: {warning}",
            activation_report_path=str(path),
        )
    return section


def check_exchange_rules_section(
    project_root: Path,
    exchanges: Sequence[str],
    blockers: list[dict[str, Any]],
    *,
    exchange_rules_path: str | Path | None,
) -> dict[str, Any]:
    path = Path(exchange_rules_path) if exchange_rules_path else None
    if path is not None and not path.is_absolute():
        path = project_root / path
    if path is None:
        path = _latest_file(
            project_root,
            (
                "Reports/PreLive/**/exchange_futures_rules*.json",
                "Reports/PreLive/exchange_futures_rules*.json",
            ),
        )
    section: dict[str, Any] = {"passed": True, "path": str(path or ""), "exchanges": {}}
    if path is None or not path.exists():
        _add_blocker(
            blockers,
            "exchange_rules.missing",
            "exchange futures rules snapshot is missing",
        )
        section["passed"] = False
        return section
    try:
        payload = _load_json(path)
    except Exception as exc:
        _add_blocker(blockers, "exchange_rules.invalid", f"exchange rules snapshot is invalid: {exc}")
        section["passed"] = False
        return section
    exchange_payloads = payload.get("exchanges") if isinstance(payload, Mapping) else None
    if not isinstance(exchange_payloads, Mapping):
        _add_blocker(blockers, "exchange_rules.invalid", "exchange rules snapshot has no exchanges object")
        section["passed"] = False
        return section
    for exchange in exchanges:
        key = str(exchange or "").strip().upper()
        item = exchange_payloads.get(key)
        if not isinstance(item, Mapping):
            _add_blocker(blockers, f"exchange_rules.{key}.missing", f"{key}: exchange rules are missing")
            section["exchanges"][key] = {"passed": False, "failures": ["missing"]}
            section["passed"] = False
            continue
        failures = list(item.get("failures") or [])
        passed = bool(item.get("passed", False))
        section["exchanges"][key] = {
            "passed": passed,
            "rules_count": len(item.get("rules") or []),
            "failures": failures,
        }
        if not passed:
            _add_blocker(
                blockers,
                f"exchange_rules.{key}.failed",
                f"{key}: exchange futures rules failed",
                failures=failures,
            )
            section["passed"] = False
    return section


def check_promotion_section(
    project_root: Path,
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    *,
    promotion_selection_path: str | Path | None = None,
    required: bool = False,
) -> dict[str, Any]:
    selection_path = Path(promotion_selection_path) if promotion_selection_path else None
    if selection_path is not None and not selection_path.is_absolute():
        selection_path = project_root / selection_path
    if selection_path is None:
        selection_path = _latest_file(
            project_root,
            (
                "Reports/PreLive/**/selection_*fitness_v4*.json",
                "Results/neiro_genetics/**/selection_*fitness_v4*.json",
                "Results/neiro_genetics/**/selection_*.json",
            ),
        )
    section: dict[str, Any] = {
        "passed": True,
        "required": bool(required),
        "selection_path": str(selection_path or ""),
    }
    if selection_path is None:
        if required:
            _add_blocker(blockers, "promotion.selection_missing", "genetics promotion selection is missing")
            section["passed"] = False
        else:
            _add_warning(
                warnings,
                "promotion.selection_missing_rnd_only",
                "genetics promotion selection is missing, but GeneticsCore is not enabled in live path",
            )
        return section
    try:
        selection = _load_json(selection_path)
    except Exception as exc:
        _add_blocker(blockers, "promotion.selection_invalid", f"selection JSON is invalid: {exc}")
        section["passed"] = False
        return section
    if not isinstance(selection, Mapping):
        _add_blocker(blockers, "promotion.selection_invalid", "selection JSON is not an object")
        section["passed"] = False
        return section
    promotion_eligible = bool(selection.get("promotion_eligible", False))
    paper_eligible = bool(selection.get("paper_trading_eligible", False))
    live_eligible = bool(selection.get("live_trading_eligible", False))
    failures = list(selection.get("promotion_failures") or [])
    section.update(
        {
            "passed": promotion_eligible or not required,
            "promotion_eligible": promotion_eligible,
            "paper_trading_eligible": paper_eligible,
            "live_trading_eligible": live_eligible,
            "baseline_kind": str(selection.get("baseline_kind") or ""),
            "exchange": str(selection.get("exchange") or ""),
            "promotion_failures": failures,
            "strict_promotion_contracts": selection.get("strict_promotion_contracts", {}),
            "final_holdout_gate": selection.get("final_holdout_gate", {}),
        }
    )
    if not promotion_eligible:
        if required:
            _add_blocker(
                blockers,
                "promotion.not_eligible",
                "latest genetics selection is not promotion_eligible",
                selection_path=str(selection_path),
                failures=failures,
            )
        else:
            _add_warning(
                warnings,
                "promotion.not_eligible_rnd_only",
                "latest genetics selection is not promotion_eligible, but GeneticsCore is not enabled in live path",
                selection_path=str(selection_path),
                failures=failures,
            )
    return section


def _settings_requires_genetics_promotion(settings_section: Mapping[str, Any]) -> bool:
    return bool(
        settings_section.get("genetics_core_live_admission_keys")
        or settings_section.get("genetics_bypass_enabled_keys")
        or settings_section.get("genetics_probation_execution_enabled_keys")
        or settings_section.get("genetics_core_primary_enabled_keys")
    )


def runbook_section(
    *,
    exchanges: Sequence[str],
    symbols: Sequence[str],
    exchange_rules_path: str,
) -> dict[str, Any]:
    symbols_arg = ",".join(symbols)
    return {
        "commands": {
            "exchange_rules": (
                "python tools/check_exchange_futures_rules.py "
                f"--exchange MEXC --exchange BITGET --symbols {symbols_arg} "
                "--out Reports/PreLive/exchange_futures_rules_latest.json"
            ),
            "pre_live_matrix": (
                "python tools/run_panteon3_pre_live_matrix.py "
                "--years 2024,2025,2026 --max-bars 240 "
                "--window-skip-bars 0 --window-skip-bars 240 --window-skip-bars 480 "
                "--stop-on-failure"
            ),
            "single_exchange_paper_canary": (
                "python tools/run_panteon3_single_component_canary.py "
                "--exchange MEXC --exchange BITGET --actor LiveOIBreakout "
                "--results-root Results/Panteon3SingleComponentCanary_isolated/prelive_liveoibreakout_65bar "
                "--reports-dir Reports/Panteon3Canary/prelive_liveoibreakout_65bar "
                "--max-bars 65 --max-idle-polls 120 --sleep-between-polls-sec 1"
            ),
            "canary_summary": (
                "python tools/run_panteon3_live_canary_check.py "
                "--results-root Results/Panteon3SingleComponentCanary_isolated/prelive_liveoibreakout_65bar "
                "--reports-dir Reports/Panteon3Canary --lookback-minutes 4320 "
                "--exchange MEXC --exchange BITGET"
            ),
            "readiness": (
                "python tools/build_panteon_prelive_readiness.py "
                f"--exchange MEXC --exchange BITGET --exchange-rules {exchange_rules_path}"
            ),
        }
    }


def build_readiness_report(
    *,
    project_root: str | Path = ROOT,
    exchanges: Sequence[str] = DEFAULT_EXCHANGES,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    now: datetime | None = None,
    active_pids: set[int] | None = None,
    include_git: bool = True,
    include_live_status: bool = True,
    include_live_preflight: bool = True,
    include_exchange_rules: bool = True,
    include_promotion: bool = True,
    include_artifact_doctor: bool = False,
    exchange_rules_path: str | Path | None = None,
    promotion_selection_path: str | Path | None = None,
    matrix_summary_path: str | Path | None = None,
    canary_summary_path: str | Path | None = None,
    activation_report_path: str | Path | None = None,
    settings_snapshot_path: str | Path | None = None,
    max_status_age_hours: float = 24.0,
    live_preflight_max_age_hours: float = 24.0,
) -> dict[str, Any]:
    root = Path(project_root)
    current = _aware_utc(now)
    exchange_keys = tuple(dict.fromkeys(str(item or "").strip().upper() for item in exchanges if str(item or "").strip()))
    symbol_keys = _parse_symbols(symbols)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    sections: dict[str, Any] = {}

    if include_git:
        sections["git"] = check_git_section(root, blockers)
    sections["settings"] = check_settings_section(
        root,
        exchange_keys,
        blockers,
        warnings,
        settings_snapshot_path=settings_snapshot_path,
    )
    if include_live_status:
        sections["live_status"] = check_live_status_section(
            root,
            exchange_keys,
            blockers,
            warnings,
            now=current,
            active_pids=active_pids,
            max_status_age_hours=float(max_status_age_hours),
        )
    if include_live_preflight:
        sections["live_preflight"] = check_live_preflight_section(
            root,
            exchange_keys,
            blockers,
            now=current,
            max_age_hours=float(live_preflight_max_age_hours),
        )
    if include_artifact_doctor:
        sections["matrix"] = check_matrix_artifact_section(
            root,
            blockers,
            matrix_summary_path=matrix_summary_path,
        )
        sections["canary"] = check_canary_artifact_section(
            root,
            exchange_keys,
            blockers,
            warnings,
            canary_summary_path=canary_summary_path,
        )
        sections["activation"] = check_activation_artifact_section(
            root,
            blockers,
            warnings,
            activation_report_path=activation_report_path,
        )
    if include_exchange_rules:
        sections["exchange_rules"] = check_exchange_rules_section(
            root,
            exchange_keys,
            blockers,
            exchange_rules_path=exchange_rules_path,
        )
    if include_promotion:
        sections["promotion"] = check_promotion_section(
            root,
            blockers,
            warnings,
            promotion_selection_path=promotion_selection_path,
            required=_settings_requires_genetics_promotion(sections["settings"]),
        )
    sections["runbook"] = runbook_section(
        exchanges=exchange_keys,
        symbols=symbol_keys,
        exchange_rules_path=str(exchange_rules_path or "Reports/PreLive/exchange_futures_rules_latest.json"),
    )

    blockers = list({item["id"]: item for item in blockers}.values())
    warnings = list({item["id"]: item for item in warnings}.values())
    return {
        "generated_at": current.isoformat(),
        "project_root": str(root),
        "exchanges": list(exchange_keys),
        "symbols": list(symbol_keys),
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "sections": sections,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Panteon pre-live readiness")
    lines.append("")
    lines.append(f"- generated_at: {report.get('generated_at', '')}")
    lines.append(f"- verdict: {'PASS' if report.get('passed') else 'BLOCKED'}")
    lines.append("")
    blockers = report.get("blockers") if isinstance(report.get("blockers"), Sequence) else []
    lines.append("## Blockers")
    if blockers:
        for item in blockers:
            if not isinstance(item, Mapping):
                continue
            lines.append(f"- {item.get('id', '')}: {item.get('message', '')}")
    else:
        lines.append("- none")
    lines.append("")
    warnings = report.get("warnings") if isinstance(report.get("warnings"), Sequence) else []
    lines.append("## Warnings")
    if warnings:
        for item in warnings:
            if not isinstance(item, Mapping):
                continue
            lines.append(f"- {item.get('id', '')}: {item.get('message', '')}")
    else:
        lines.append("- none")
    lines.append("")
    sections = report.get("sections") if isinstance(report.get("sections"), Mapping) else {}
    lines.append("## Sections")
    for name, payload in sections.items():
        if not isinstance(payload, Mapping) or name == "runbook":
            continue
        verdict = "PASS" if payload.get("passed") else "BLOCKED"
        lines.append(f"- {name}: {verdict}")
        source_path = payload.get("path") or payload.get("selection_path")
        if source_path:
            lines.append(f"  - source: {source_path}")
        if payload.get("execution_smoke"):
            lines.append("  - execution_smoke: true")
        fail_reasons = payload.get("fail_reasons")
        if isinstance(fail_reasons, Sequence) and not isinstance(fail_reasons, (str, bytes)) and fail_reasons:
            lines.append(f"  - fail_reasons: {', '.join(str(item) for item in fail_reasons)}")
        promotion_failures = payload.get("promotion_failures")
        if (
            isinstance(promotion_failures, Sequence)
            and not isinstance(promotion_failures, (str, bytes))
            and promotion_failures
        ):
            lines.append(
                f"  - promotion_failures: {', '.join(str(item) for item in promotion_failures)}"
            )
        activation_blockers = payload.get("activation_blockers")
        if (
            isinstance(activation_blockers, Sequence)
            and not isinstance(activation_blockers, (str, bytes))
            and activation_blockers
        ):
            lines.append(
                f"  - activation_blockers: {', '.join(str(item) for item in activation_blockers)}"
            )
        terminal_denied_context_signal_keys = payload.get(
            "terminal_denied_context_signal_keys"
        )
        if (
            isinstance(terminal_denied_context_signal_keys, Sequence)
            and not isinstance(terminal_denied_context_signal_keys, (str, bytes))
            and terminal_denied_context_signal_keys
        ):
            joined_keys = ", ".join(
                str(item) for item in terminal_denied_context_signal_keys
            )
            lines.append(f"  - terminal_denied_context_signal_keys: {joined_keys}")
        exchange_payloads = payload.get("exchanges")
        if isinstance(exchange_payloads, Mapping):
            exchange_summaries = []
            for exchange, exchange_payload in exchange_payloads.items():
                if not isinstance(exchange_payload, Mapping):
                    continue
                exchange_verdict = "PASS" if exchange_payload.get("passed") else "BLOCKED"
                metrics = []
                for key in ("signals", "orders", "fills", "expectancy_after_costs"):
                    if key in exchange_payload:
                        metrics.append(f"{key}={exchange_payload.get(key)}")
                if metrics:
                    exchange_summaries.append(f"{exchange}:{exchange_verdict} ({', '.join(metrics)})")
                else:
                    exchange_summaries.append(f"{exchange}:{exchange_verdict}")
            if exchange_summaries:
                lines.append(f"  - exchanges: {'; '.join(exchange_summaries)}")
    runbook = sections.get("runbook") if isinstance(sections, Mapping) else None
    commands = runbook.get("commands") if isinstance(runbook, Mapping) else None
    if isinstance(commands, Mapping) and commands:
        lines.append("")
        lines.append("## Runbook Commands")
        for name, command in commands.items():
            lines.append(f"### {name}")
            lines.append("```powershell")
            lines.append(str(command))
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _timestamp_slug(now: datetime) -> str:
    return now.strftime("%Y-%m-%d_%H-%M-%S_utc")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an aggregated pre-live readiness report for Panteon."
    )
    parser.add_argument("--exchange", action="append", dest="exchanges", default=None)
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--exchange-rules", default="")
    parser.add_argument("--promotion-selection", default="")
    parser.add_argument("--matrix-summary", default="")
    parser.add_argument("--canary-summary", default="")
    parser.add_argument("--activation-report", default="")
    parser.add_argument("--settings-snapshot", default="")
    parser.add_argument(
        "--write-settings-snapshot",
        action="store_true",
        help="Write a sanitized settings safety snapshot into the report directory and use it.",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "Reports" / "PreLive"))
    parser.add_argument("--max-status-age-hours", type=float, default=24.0)
    parser.add_argument("--live-preflight-max-age-hours", type=float, default=24.0)
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--skip-live-status", action="store_true")
    parser.add_argument("--skip-live-preflight", action="store_true")
    parser.add_argument("--skip-exchange-rules", action="store_true")
    parser.add_argument("--skip-promotion", action="store_true")
    parser.add_argument("--skip-artifact-doctor", action="store_true")
    args = parser.parse_args(argv)

    now = _aware_utc()
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_dir = out_root / f"prelive_readiness_{_timestamp_slug(now)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    exchanges = tuple(args.exchanges or DEFAULT_EXCHANGES)
    settings_snapshot = args.settings_snapshot or None
    if args.write_settings_snapshot:
        snapshot_path = out_dir / "settings_safety_snapshot.json"
        snapshot = build_settings_safety_snapshot(
            project_root=ROOT,
            exchanges=exchanges,
            now=now,
        )
        snapshot_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        settings_snapshot = str(snapshot_path)

    report = build_readiness_report(
        project_root=ROOT,
        exchanges=exchanges,
        symbols=_parse_symbols(args.symbols),
        now=now,
        include_git=not args.skip_git,
        include_live_status=not args.skip_live_status,
        include_live_preflight=not args.skip_live_preflight,
        include_exchange_rules=not args.skip_exchange_rules,
        include_promotion=not args.skip_promotion,
        include_artifact_doctor=not args.skip_artifact_doctor,
        exchange_rules_path=args.exchange_rules or None,
        promotion_selection_path=args.promotion_selection or None,
        matrix_summary_path=args.matrix_summary or None,
        canary_summary_path=args.canary_summary or None,
        activation_report_path=args.activation_report or None,
        settings_snapshot_path=settings_snapshot,
        max_status_age_hours=float(args.max_status_age_hours),
        live_preflight_max_age_hours=float(args.live_preflight_max_age_hours),
    )
    json_path = out_dir / "prelive_readiness_report.json"
    md_path = out_dir / "prelive_readiness_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
