"""Local-only V2 commitment and continuous origin-prefix evaluation.

This runner accepts only a separately registered future contract. It never
reinterprets historical V1 manifests as V2 evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_continuous_v2 import (
    evaluate_committed_chain, validate_continuous_contract,
)
from exia.genetic_primus.prospective_manifest_v2 import (
    commit_origin_manifest_exclusive, load_verified_origin,
    reserve_origin_evaluation,
)

REPORTS_ROOT = (ROOT / "Reports/Exia/Genetic_Primus").resolve()


def _under(path: Path, root: Path) -> Path:
    result = path.resolve()
    if not result.is_relative_to(root):
        raise ValueError("path is outside its allowed root")
    return result


def _contract(args: argparse.Namespace) -> tuple[Path, dict]:
    expected = (ROOT / ".venv/Scripts/python.exe").resolve()
    if Path(sys.executable).resolve() != expected:
        raise ValueError("only the project .venv Python is authorized")
    config_path = _under(args.contract, ROOT)
    registry_path = _under(args.registry, ROOT)
    contract = json.loads(config_path.read_text(encoding="utf-8"))
    validate_continuous_contract(contract)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [item for item in registry["objectives"]
               if item["objective_id"] == contract["objective_id"]]
    if len(entries) != 1 or entries[0]["config_path"] != config_path.relative_to(ROOT).as_posix():
        raise ValueError("contract is not uniquely registered")
    if entries[0]["config_sha256"] != file_sha256(config_path):
        raise ValueError("registered contract bytes changed")
    previous = _under(ROOT / registry["previous_registry_path"], ROOT)
    if file_sha256(previous) != registry["previous_registry_sha256"]:
        raise ValueError("objective registry ancestry changed")
    if contract["data_commitment"].get("manifest_version") != 2:
        raise ValueError("future V2 per-origin contract required")
    if any(contract["safety"].values()):
        raise ValueError("offline-only contract violated")
    selection = contract["model_selection"]
    selection_report = _under(ROOT / selection["report_path"], ROOT)
    if file_sha256(selection_report) != selection["report_sha256"]:
        raise ValueError("registered model selection report changed")
    report = json.loads(selection_report.read_text(encoding="utf-8"))
    if (report.get("selected_genome") != contract["candidates"][2]["genome"]
            or report.get("window") != selection["window"]
            or report.get("scaler") != selection["scaler"]):
        raise ValueError("selection report does not support registered model")
    cutoff = datetime.fromisoformat(report["selection_cutoff_utc"].replace("Z", "+00:00"))
    outer_start = datetime.fromisoformat(contract["outer"]["start_utc"].replace("Z", "+00:00"))
    if cutoff.tzinfo is None or cutoff >= outer_start:
        raise ValueError("selection report cutoff must precede outer start")
    return config_path, contract


def _paths(args: argparse.Namespace, index: int) -> tuple[Path, Path, Path]:
    output = _under(args.output_root, REPORTS_ROOT)
    stem = f"origin_{index + 1:02d}"
    return (output / f"{stem}_manifest.json",
            output / f"{stem}_started.json",
            output / f"{stem}_continuous.json")


def _verified_prefix(args: argparse.Namespace, config_path: Path,
                     contract: dict) -> list[tuple[dict, object]]:
    if args.origin < 0 or args.origin >= contract["outer"]["origin_count"]:
        raise ValueError("origin index outside registered interval")
    manifests = []
    latest_panel = None
    for index in range(args.origin + 1):
        manifest_path, started_path, result_path = _paths(args, index)
        previous = _paths(args, index - 1)[0] if index else None
        manifest, panel = load_verified_origin(
            manifest_path, root=ROOT, contract_path=config_path,
            contract=contract, origin_index=index,
            previous_manifest_path=previous)
        if index < args.origin and (not started_path.is_file() or not result_path.is_file()):
            raise ValueError("previous origin lacks a completed evaluation")
        manifests.append(manifest)
        latest_panel = panel
    return [(manifest, latest_panel) for manifest in manifests]


def command_commit(args: argparse.Namespace) -> None:
    config_path, contract = _contract(args)
    manifest_path, _, _ = _paths(args, args.origin)
    previous = _paths(args, args.origin - 1)[0] if args.origin else None
    manifest = commit_origin_manifest_exclusive(
        manifest_path, args.snapshot, root=ROOT, contract_path=config_path,
        contract=contract, origin_index=args.origin,
        previous_manifest_path=previous)
    print(json.dumps({"status": "V2_ORIGIN_COMMITTED_NO_EVALUATION",
                      "origin_index": args.origin,
                      "manifest_sha256": file_sha256(manifest_path),
                      "snapshot_sha256": manifest["snapshot_sha256"]}, separators=(",", ":")))


def command_preflight(args: argparse.Namespace) -> None:
    config_path, contract = _contract(args)
    verified = _verified_prefix(args, config_path, contract)
    print(json.dumps({"status": "V2_PREFIX_VERIFIED_NO_EVALUATION",
                      "origin_count": len(verified),
                      "snapshot_sha256": verified[-1][0]["snapshot_sha256"]},
                     separators=(",", ":")))


def command_evaluate(args: argparse.Namespace) -> None:
    config_path, contract = _contract(args)
    verified = _verified_prefix(args, config_path, contract)
    manifest_path, started_path, result_path = _paths(args, args.origin)
    previous = _paths(args, args.origin - 1)[0] if args.origin else None
    committed, _ = reserve_origin_evaluation(
        started_path, result_path, manifest_path, root=ROOT,
        contract_path=config_path, contract=contract,
        origin_index=args.origin, previous_manifest_path=previous)
    if committed != verified[-1][0]:
        raise ValueError("manifest changed between preflight and reservation")
    result = evaluate_committed_chain(contract, verified)
    result["contract_sha256"] = file_sha256(config_path)
    result["manifest_sha256_by_origin"] = [
        file_sha256(_paths(args, index)[0]) for index in range(args.origin + 1)]
    result["previous_result_sha256"] = (
        file_sha256(_paths(args, args.origin - 1)[2]) if args.origin else None)
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "V2_CONTINUOUS_PREFIX_EVALUATED_OFFLINE",
                      "origin_count": result["origin_count"],
                      "result_sha256": file_sha256(result_path),
                      "search_evaluations": 0}, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--contract", type=Path, required=True)
    common.add_argument("--registry", type=Path, required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--origin", type=int, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("commit", command_commit),
                          ("preflight", command_preflight),
                          ("evaluate", command_evaluate)):
        command = commands.add_parser(name, parents=[common])
        if name == "commit":
            command.add_argument("--snapshot", type=Path, action="append", required=True)
        command.set_defaults(handler=handler)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
