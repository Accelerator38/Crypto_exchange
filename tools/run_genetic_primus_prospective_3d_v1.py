"""Local-only manifest and origin runner for the prospective 3-day contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.prospective_contract_v1 import file_sha256, load_and_validate
from exia.genetic_primus.prospective_evaluator_v1 import evaluate_panel_origin, origin_preflight
from exia.genetic_primus.prospective_manifest_v1 import (
    build_manifest,
    load_verified_snapshot,
    save_manifest_exclusive,
)


CONFIG = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v5.json"
OUTPUT_ROOT = (ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609").resolve()


def _load_contract() -> dict:
    expected = (ROOT / ".venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError("only the project .venv Python is authorized")
    contract, _ = load_and_validate(CONFIG)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    registered = [
        item for item in registry["objectives"] if item["objective_id"] == contract["objective_id"]
    ]
    if len(registered) != 1 or registered[0]["config_sha256"] != file_sha256(CONFIG):
        raise ValueError("contract is not pinned by objective registry v5")
    previous = ROOT / registry["previous_registry_path"]
    if file_sha256(previous) != registry["previous_registry_sha256"]:
        raise ValueError("objective registry ancestry changed")
    if any(contract["safety"].values()):
        raise ValueError("offline-only contract violated")
    return contract


def _inside_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(OUTPUT_ROOT):
        raise ValueError("artifact must stay inside the registered result root")
    return resolved


def _save_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def command_manifest(args: argparse.Namespace) -> None:
    contract = _load_contract()
    output = _inside_output(args.output)
    manifest = build_manifest(
        args.snapshot, root=ROOT, contract_path=CONFIG, contract=contract,
        created_at=datetime.now(timezone.utc),
    )
    save_manifest_exclusive(output, manifest)
    print(json.dumps({
        "status": "MANIFEST_COMMITTED_NO_EVALUATION",
        "path": output.relative_to(ROOT).as_posix(),
        "manifest_sha256": file_sha256(output),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "coverage_end_exclusive": manifest["metadata"]["coverage_end_exclusive"],
    }, separators=(",", ":")))


def _verified(args: argparse.Namespace):
    contract = _load_contract()
    manifest_path = _inside_output(args.manifest)
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=CONFIG, contract=contract,
    )
    start_ms, end_ms = origin_preflight(contract, manifest, args.origin, now=datetime.now(timezone.utc))
    return contract, manifest_path, manifest, panel, start_ms, end_ms


def command_preflight(args: argparse.Namespace) -> None:
    _, manifest_path, manifest, panel, start_ms, end_ms = _verified(args)
    print(json.dumps({
        "status": "PREFLIGHT_PASS_NO_EVALUATION",
        "origin_index": args.origin,
        "origin_start": start_ms,
        "origin_end_exclusive": end_ms,
        "rows": len(panel),
        "manifest_sha256": file_sha256(manifest_path),
        "snapshot_sha256": manifest["snapshot_sha256"],
    }, separators=(",", ":")))


def command_evaluate(args: argparse.Namespace) -> None:
    contract, manifest_path, manifest, panel, start_ms, end_ms = _verified(args)
    target = OUTPUT_ROOT / f"origin_{args.origin + 1:02d}.json"
    started = OUTPUT_ROOT / f"origin_{args.origin + 1:02d}_started.json"
    if target.exists() or started.exists():
        raise FileExistsError("origin was already attempted; no silent repeat")
    _save_new(started, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_index": args.origin,
        "origin_start": start_ms,
        "origin_end_exclusive": end_ms,
        "contract_sha256": file_sha256(CONFIG),
        "manifest_sha256": file_sha256(manifest_path),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "safety": contract["safety"],
    })
    result = evaluate_panel_origin(
        panel, contract, manifest, args.origin, now=datetime.now(timezone.utc),
    )
    result["contract_sha256"] = file_sha256(CONFIG)
    result["manifest_sha256"] = file_sha256(manifest_path)
    _save_new(target, result)
    print(json.dumps({
        "status": "ORIGIN_EVALUATED_OFFLINE_NO_PROMOTION",
        "origin_index": args.origin,
        "result": target.relative_to(ROOT).as_posix(),
        "result_sha256": file_sha256(target),
        "search_evaluations": result["search_evaluations"],
    }, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--snapshot", action="append", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=command_manifest)
    for name, handler in (("preflight", command_preflight), ("evaluate", command_evaluate)):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--origin", type=int, required=True)
        command.set_defaults(handler=handler)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
