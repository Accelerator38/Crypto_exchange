from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "configs"
    / "research_snapshots"
    / "panteon_freeze_2026-08-03.json"
)
DEFAULT_PATHS = (
    "Retrodate/bitget_futures_history_v3_2022_20260714/integrity_manifest.json",
    "Retrodate/bitget_1m_discovery_v1_20260615_20260728/integrity_manifest.json",
    "Retrodate/bitget_data_v3/research/precursor_20260729T105449Z-68d7ac53/microstructure_research_v3.manifest.json",
    "Retrodate/bitget_data_v3/research/precursor_20260729T105449Z-68d7ac53/microstructure_research_v3.sqlite",
    "Retrodate/bitget_data_v3/research/precursor_20260801T213112Z-b434b442/microstructure_research_v3.manifest.json",
    "Retrodate/bitget_data_v3/research/precursor_20260801T213112Z-b434b442/microstructure_research_v3.sqlite",
    "Retrodate/bitget_data_v3/research/precursor_20260802T111040Z-f2ed3d03/microstructure_research_v3.manifest.json",
    "Retrodate/bitget_data_v3/research/precursor_20260802T111040Z-f2ed3d03/microstructure_research_v3.sqlite",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    ).strip()


def build_snapshot(*, output: Path, paths: tuple[str, ...]) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for relative in paths:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        files.append(
            {
                "path": relative.replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    snapshot_commit = _git("rev-list", "-n", "1", "panteon-research-freeze-2026-08-03")
    payload: dict[str, object] = {
        "schema_version": "panteon.research_reset_snapshot.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "branch": _git("branch", "--show-current"),
        "snapshot_tag": "panteon-research-freeze-2026-08-03",
        "snapshot_commit": snapshot_commit,
        "head_at_manifest_build": _git("rev-parse", "HEAD"),
        "files": files,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal the Pantheon reset snapshot.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_snapshot(output=args.output, paths=DEFAULT_PATHS)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
