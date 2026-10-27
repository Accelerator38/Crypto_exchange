from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "Reports/Exia/Genetic_Primus/development_v3_100g_20260821"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_terminal_manifest(run_dir: Path) -> dict[str, Any]:
    freeze_path = run_dir / "freeze_manifest.json"
    validation_path = run_dir / "validation_manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "DEVELOPMENT_V3_PASSED_STRICT_GATES":
        raise ValueError("unexpected V3 development freeze status")
    if validation.get("status") != "VALIDATION_V3_REJECTED":
        raise ValueError("V3 can be terminally rejected only after rejected validation")
    if validation.get("oos_allowed") is not False:
        raise ValueError("rejected V3 validation unexpectedly allows OOS")
    for flag in ("oos_opened", "sanity_opened", "paper_allowed", "live_allowed", "orders_enabled", "promotion_authority"):
        if validation.get(flag) is not False:
            raise ValueError(f"unsafe V3 validation flag: {flag}")
    return {
        "schema_version": "exia.genetic_primus_terminal_rejection.v3",
        "agent_id": "Genetic_Primus",
        "version": "V3",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "TERMINAL_REJECTED_VALIDATION_GENERALIZATION",
        "reason": "positive development evidence did not generalize to one-time validation 2024",
        "development_freeze_sha256": _sha256(freeze_path),
        "validation_manifest_sha256": _sha256(validation_path),
        "champion_genome_sha256": validation["champion_genome_sha256"],
        "dataset_sha256": validation["dataset_sha256"],
        "validation_metrics": validation["metrics"],
        "retraining_allowed": False,
        "validation_reuse_allowed": False,
        "oos_allowed": False,
        "oos_opened": False,
        "sanity_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "terminal_rejection_manifest.json"
    manifest = build_terminal_manifest(run_dir)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in manifest.items() if key != "generated_at"}
        existing_comparable = {key: value for key, value in existing.items() if key != "generated_at"}
        if existing_comparable != comparable:
            raise FileExistsError("existing V3 terminal rejection manifest does not match")
        return existing
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Genetic_Primus V3 rejection")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
