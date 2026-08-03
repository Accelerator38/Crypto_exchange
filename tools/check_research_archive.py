from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_MANIFEST = (
    PROJECT_ROOT / "configs" / "research_archive" / "archive_manifest_v1.json"
)
HYPOTHESES = (
    PROJECT_ROOT / "configs" / "research_archive" / "hypotheses_v1.json"
)
FREEZE_CONFIG = PROJECT_ROOT / "configs" / "panteon_legacy_freeze_v1.json"
FREEZE_SOURCE = PROJECT_ROOT / "src" / "panteon_v2" / "app" / "live_freeze.py"
MODEL_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".joblib",
    ".keras",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
}
TERMINAL_STATUSES = {
    "ARCHIVED_ARCHITECTURE",
    "TERMINAL_REJECTED",
    "DATA_BLOCKED",
    "CALIBRATION_ONLY",
    "ENGINEERING_ONLY",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(
    checks: list[dict[str, object]], name: str, passed: bool, detail: object
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _model_artifacts() -> list[str]:
    found: list[str] = []
    for root, directories, files in os.walk(PROJECT_ROOT):
        directories[:] = [
            name
            for name in directories
            if name != ".git" and not name.startswith(".venv")
        ]
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if path.suffix.lower() in MODEL_SUFFIXES:
                found.append(path.relative_to(PROJECT_ROOT).as_posix())
    return sorted(found)


def _tracked_backup_or_model_artifacts() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    rejected: list[str] = []
    for raw in result.stdout.splitlines():
        path = Path(raw)
        lower = raw.lower()
        if path.suffix.lower() in MODEL_SUFFIXES or lower.endswith(
            (".truncated_backup", ".bak", ".old")
        ):
            rejected.append(raw)
    return sorted(rejected)


def check_archive(*, require_local_data: bool) -> dict[str, object]:
    manifest = _json(ARCHIVE_MANIFEST)
    hypotheses = _json(HYPOTHESES)
    freeze = _json(FREEZE_CONFIG)
    checks: list[dict[str, object]] = []

    safety = manifest["safety"]
    _check(
        checks,
        "archive_safety_contract",
        safety.get("legacy_panteon_bitget_live_frozen") is True
        and safety.get("orders_enabled") is False
        and safety.get("promotion_authority") is False
        and safety.get("approved_model_checkpoints") == 0
        and safety.get("approved_live_strategies") == 0,
        safety,
    )
    _check(
        checks,
        "freeze_config",
        freeze.get("frozen") is True
        and freeze.get("orders_enabled") is False
        and freeze.get("promotion_authority") is False
        and freeze.get("environment_bypass_supported") is False,
        FREEZE_CONFIG.relative_to(PROJECT_ROOT).as_posix(),
    )
    freeze_source = FREEZE_SOURCE.read_text(encoding="utf-8")
    _check(
        checks,
        "freeze_source_constant",
        "LEGACY_PANTEON_BITGET_LIVE_FROZEN = True" in freeze_source,
        FREEZE_SOURCE.relative_to(PROJECT_ROOT).as_posix(),
    )

    snapshot = PROJECT_ROOT / manifest["source_freeze"]["snapshot"]
    _check(
        checks,
        "source_snapshot_sha256",
        snapshot.is_file()
        and _sha256(snapshot) == manifest["source_freeze"]["snapshot_sha256"],
        snapshot.relative_to(PROJECT_ROOT).as_posix(),
    )

    for dataset in manifest["retained_local_datasets"]:
        dataset_path = PROJECT_ROOT / dataset["path"]
        present = dataset_path.exists()
        _check(
            checks,
            f"dataset_present:{dataset['id']}",
            present or not require_local_data,
            {
                "path": dataset["path"],
                "present": present,
                "required_by_command": require_local_data,
            },
        )
        if not present:
            continue
        if "integrity_manifest" in dataset:
            integrity_path = PROJECT_ROOT / dataset["integrity_manifest"]
            integrity = _json(integrity_path) if integrity_path.is_file() else {}
            _check(
                checks,
                f"dataset_manifest:{dataset['id']}",
                integrity_path.is_file()
                and _sha256(integrity_path)
                == dataset["integrity_manifest_sha256"]
                and integrity.get("dataset_sha256") == dataset["dataset_sha256"],
                dataset["integrity_manifest"],
            )
        for file_record in dataset.get("files", []):
            path = PROJECT_ROOT / file_record["path"]
            _check(
                checks,
                f"dataset_file:{dataset['id']}:{path.name}",
                path.is_file() and _sha256(path) == file_record["sha256"],
                file_record["path"],
            )

    statuses = {
        str(item.get("status")) for item in hypotheses.get("hypotheses", [])
    }
    _check(
        checks,
        "hypothesis_registry_terminal",
        hypotheses.get("active_candidates") == 0
        and hypotheses.get("authoritative_for_trading") is False
        and hypotheses.get("orders_enabled") is False
        and hypotheses.get("promotion_authority") is False
        and bool(statuses)
        and statuses <= TERMINAL_STATUSES,
        {"statuses": sorted(statuses), "count": len(hypotheses["hypotheses"])},
    )

    local_models = _model_artifacts()
    tracked_junk = _tracked_backup_or_model_artifacts()
    _check(checks, "no_local_model_checkpoints", not local_models, local_models)
    _check(checks, "no_tracked_backups_or_models", not tracked_junk, tracked_junk)

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema_version": "panteon.research_archive_check.v1",
        "passed": not failed,
        "require_local_data": require_local_data,
        "checks": checks,
        "failed": failed,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen research archive.")
    parser.add_argument(
        "--require-local-data",
        action="store_true",
        help="Fail when canonical local datasets are absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the JSON report as UTF-8.",
    )
    args = parser.parse_args()
    report = check_archive(require_local_data=args.require_local_data)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
