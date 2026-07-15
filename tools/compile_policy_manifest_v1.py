from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.policy.manifest import (  # noqa: E402
    ManifestError,
    format_datetime,
    parse_datetime,
    parse_manifest_payload,
    prepare_manifest_payload,
    seal_manifest_payload,
    validate_policy_manifest,
)


def _load_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ManifestError("draft.not_object")
    return dict(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compile_manifest(
    *,
    draft_path: str | Path,
    output_path: str | Path,
    readiness_path: str | Path,
    project_root: str | Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    out = Path(output_path)
    readiness_out = Path(readiness_path)
    manifest_payload: dict[str, Any] | None = None
    reasons: tuple[str, ...] = ()
    try:
        draft = _load_object(draft_path)
        prepared = prepare_manifest_payload(draft, project_root=project_root)
        manifest_payload = seal_manifest_payload(prepared)
        manifest = parse_manifest_payload(manifest_payload)
        validation = validate_policy_manifest(
            manifest,
            project_root=project_root,
            now=current,
            verify_artifacts=True,
        )
        reasons = validation.reasons
    except ManifestError as exc:
        reasons = exc.reasons
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reasons = (f"draft.invalid:{type(exc).__name__}",)

    passed = manifest_payload is not None and not reasons
    if passed:
        _atomic_write_json(out, manifest_payload)

    target = str((manifest_payload or {}).get("target") or "")
    readiness = {
        "schema_version": "panteon.policy_readiness.v1",
        "generated_at": format_datetime(current),
        "status": (
            "RESEARCH_ONLY"
            if passed and target == "replay"
            else "PASS"
            if passed
            else "BLOCKED"
        ),
        "passed": passed,
        "draft_path": str(Path(draft_path).resolve()),
        "manifest_path": str(out.resolve()),
        "manifest_written": passed,
        "existing_manifest_unchanged": bool(not passed and out.exists()),
        "policy_id": str((manifest_payload or {}).get("policy_id") or ""),
        "target": target,
        "manifest_sha256": str(
            (manifest_payload or {}).get("manifest_sha256") or ""
        ),
        "blockers": list(reasons),
    }
    _atomic_write_json(readiness_out, readiness)
    return readiness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a strict Pantheon vNext policy manifest. A blocked draft "
            "produces readiness only and never overwrites the manifest."
        )
    )
    parser.add_argument("--draft", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--readiness-out")
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--now", help="UTC/offset ISO timestamp for reproducible runs")
    args = parser.parse_args(argv)

    output = Path(args.out)
    readiness_path = Path(args.readiness_out) if args.readiness_out else output.with_suffix(
        output.suffix + ".readiness.json"
    )
    current = parse_datetime(args.now) if args.now else datetime.now(timezone.utc)
    readiness = compile_manifest(
        draft_path=args.draft,
        output_path=output,
        readiness_path=readiness_path,
        project_root=args.project_root,
        now=current,
    )
    print(json.dumps(readiness, ensure_ascii=False, sort_keys=True))
    return 0 if readiness["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
