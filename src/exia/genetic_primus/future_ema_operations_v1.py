"""Read-only operations checks; stdlib only, no market imports or recovery writes."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import math
import platform
import struct
import sys
from pathlib import Path
from datetime import datetime, timezone

PACKAGES = ("numpy", "pandas", "pyarrow", "python-dateutil", "six", "tzdata",
            "bottleneck", "numexpr")
ARTIFACTS = ("data_manifest.json", "analysis_reservation.json", "result.json", "analysis_failure.json")
MAX_ARTIFACT_BYTES = 1024 * 1024


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def environment_snapshot() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
            continue
        files = {}
        for filename in ("METADATA", "RECORD"):
            value = dist.read_text(filename)
            if value is None:
                raise ValueError(f"missing installed distribution metadata: {name}/{filename}")
            files[filename.lower() + "_sha256"] = sha_bytes(value.encode("utf-8"))
        packages[name] = {"version": dist.version, **files}
    executable = Path(sys.executable).resolve()
    venv_config = executable.parent.parent / "pyvenv.cfg"
    return {
        "python": {"version": sys.version, "implementation": platform.python_implementation(),
                   "cache_tag": sys.implementation.cache_tag, "machine": platform.machine(),
                   "pointer_bits": struct.calcsize("P") * 8,
                   "executable": executable.as_posix(),
                   "executable_sha256": sha_bytes(executable.read_bytes()),
                   "venv_config_sha256": sha_bytes(venv_config.read_bytes())},
        "packages": packages,
    }


def environment_differences(expected: dict, observed: dict) -> list[str]:
    differences = []
    for section in ("python", "packages"):
        keys = set(expected.get(section, {})) | set(observed.get(section, {}))
        for key in sorted(keys):
            if expected.get(section, {}).get(key) != observed.get(section, {}).get(key):
                differences.append(f"{section}.{key}")
    return differences


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _reject_constant(_):
    raise ValueError("nonfinite JSON constant")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def read_json_artifact(path: Path) -> tuple[dict, str]:
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("oversized artifact")
    with path.open("rb") as stream:
        raw = stream.read(MAX_ARTIFACT_BYTES + 1)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("oversized artifact")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                       parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("artifact is not a JSON object")
    return value, sha_bytes(raw)


def inspect_artifacts(directory: Path, *, objective_id: str, config_sha256: str,
                      implementation_lock_sha256: str, safety: dict) -> dict:
    """Inspect provenance/state only. Never reads segment paths or returns metrics.

    JSON validity and mutually consistent hashes are not proof of committed HEAD,
    authentic statistical results, fsync durability or completeness of market data.
    """
    present, hashes, issues = {}, {}, []
    for name in ARTIFACTS:
        path = directory / name
        if not path.exists() and not path.is_symlink():
            continue
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("artifact link outside inspection scope")
            present[name], hashes[name] = read_json_artifact(path)
        except (OSError, ValueError, UnicodeError) as exc:
            issues.append(f"{name}:{type(exc).__name__}")
    m = present.get("data_manifest.json")
    reservation = present.get("analysis_reservation.json")
    result = present.get("result.json")
    failure = present.get("analysis_failure.json")
    for name, value in present.items():
        if any(value.get(k) != v for k, v in {
            "objective_id": objective_id, "config_sha256": config_sha256,
            "implementation_lock_sha256": implementation_lock_sha256,
        }.items()):
            issues.append(f"{name}:lineage_mismatch")
    if m is not None and (m.get("schema_version") != "exia.genetic_primus.future_ema_segment/1"
                          or m.get("immutable") is not True or m.get("safety") != safety
                          or m.get("outcome_metrics_computed") is not False
                          or m.get("network_used") is not False):
        issues.append("data_manifest.json:invalid_commitment")
    for name in ("analysis_reservation.json", "result.json", "analysis_failure.json"):
        if name in present and (m is None or present[name].get("data_manifest_sha256")
                                != hashes.get("data_manifest.json")):
            issues.append(f"{name}:manifest_sha_mismatch")
        if name in present:
            try:
                stamp = datetime.fromisoformat(present[name]["analysis_started_at_utc"])
                if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
                    raise ValueError("timestamp must be UTC")
            except (KeyError, TypeError, ValueError):
                issues.append(f"{name}:invalid_analysis_timestamp")
    if (result is not None or failure is not None) and reservation is None:
        issues.append("missing_reservation")
    if result is not None and failure is not None:
        issues.append("result_and_failure_coexist")
    if failure is not None and not isinstance(failure.get("exception_type"), str):
        issues.append("analysis_failure.json:missing_exception_type")
    for name, value in (("result.json", result), ("analysis_failure.json", failure)):
        if value is not None:
            if value.get("promotion_authority") is not False:
                issues.append(f"{name}:unsafe_authority")
            if reservation is not None and value.get("analysis_started_at_utc") != reservation.get("analysis_started_at_utc"):
                issues.append(f"{name}:reservation_time_mismatch")
    if result is not None and (result.get("schema_version") != "exia.genetic_primus.future_ema_result/1"
                               or result.get("safety") != safety or result.get("final_policy") != "NoTrade"
                               or result.get("automatic_promotion") is not False
                               or result.get("market_readiness") is not False
                               or result.get("network_used") is not False):
        issues.append("result.json:invalid_safety")
    if issues:
        status = "MANUAL_REVIEW_BLOCKED"
    elif result is not None:
        status = "FINAL_RESULT_PRESENT_NO_REANALYSIS"
    elif failure is not None:
        status = "ANALYSIS_FAILED_BLOCKED"
    elif reservation is not None:
        status = "ANALYSIS_STARTED_OR_INTERRUPTED_NO_RETRY"
    elif m is not None:
        status = "INPUT_MANIFEST_PRESENT_COMMIT_NOT_VERIFIED"
    else:
        status = "AWAITING_INPUT_COMMIT"
    return {"status": status, "artifact_sha256": hashes, "issues": issues,
            "read_only": True, "rerun_authorized": False, "metrics_recomputed": False,
            "git_commit_verified": False, "power_loss_durability_verified": False,
            "promotion_authority": False}
