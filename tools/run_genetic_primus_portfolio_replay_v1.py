"""Bounded FOREGROUND offline portfolio replay; default only verifies contracts."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/genetic_primus_portfolio_replay_v1.json"
PARENT = "configs/genetic_primus_release_candidate_v1.json"
SOURCES = (CONFIG, "src/exia/genetic_primus/portfolio_replay_v1.py",
           "tools/run_genetic_primus_portfolio_replay_v1.py")
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/portfolio_replay_v1_20260910"


def read_inputs():
    """Check frozen source bytes BEFORE importing any project module."""
    contract = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    required = {PARENT, "src/exia/genetic_primus/release_contract_v1.py",
                "src/exia/genetic_primus/execution_replay_v1.py"}
    pins = contract.get("source_pins", {})
    if set(pins) != required:
        raise ValueError("frozen dependency pin set mismatch")
    for name, expected in pins.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("frozen dependency changed: " + name)
    sys.path.insert(0, str(ROOT / "src/exia/genetic_primus"))
    from release_contract_v1 import object_sha, validate_contract
    from portfolio_replay_v1 import validate_portfolio_contract
    release = json.loads((ROOT / PARENT).read_text(encoding="utf-8"))
    check = validate_contract(release, root=ROOT)
    if not check["structurally_valid"]:
        raise ValueError("parent research source integrity failure")
    validate_portfolio_contract(contract, release)
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    sources.update(pins)
    return contract, release, {"source_hashes": sources, "sha256": object_sha(sources)}


def save_new(name, payload):
    destination = (ROOT / name).resolve()
    if destination.parent != OUTPUT.resolve() or destination.suffix != ".json":
        raise ValueError("output must be a NEW JSON in the scoped portfolio report directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=True, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="explicit local JSON bundle, at most 2 MiB")
    parser.add_argument("--output", help="NEW JSON comparison in scoped report directory")
    args = parser.parse_args()
    contract, release, identity = read_inputs()
    if not args.input and not args.output:
        print(json.dumps({"contract_valid": True, "source_hash_checks": len(identity["source_hashes"]),
                          "replay_run": False, "trading_authorized": False}))
        return 0
    if not args.input or not args.output:
        parser.error("--input and --output must be supplied together")
    path = (ROOT / args.input).resolve()
    destination = (ROOT / args.output).resolve()
    if not path.is_relative_to(ROOT) or path.suffix != ".json" or not path.is_file():
        raise ValueError("workspace-local JSON input required")
    if destination.parent != OUTPUT.resolve() or destination.suffix != ".json" or destination.exists():
        raise ValueError("new scoped output required; no overwrites")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("input exceeds 2 MiB safety limit")
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("input grew beyond safety limit")
    from portfolio_replay_v1 import run_comparison
    result = run_comparison(json.loads(raw.decode("utf-8")), contract, release, identity["sha256"])
    receipt = {"schema_version": "exia.genetic_primus.portfolio_run/1", "implementation": identity,
               "input_file_sha256": hashlib.sha256(raw).hexdigest(), "comparison": result,
               "network_used": False, "training_run": False, "trading_authorized": False}
    save_new(args.output, receipt)
    print(json.dumps({"status": "OFFLINE_ENGINEERING_ONLY", "input_events": result["unique_input_events"],
                      "decision_boundaries": result["decision_boundaries"],
                      "fills": {k: v["fills"] for k, v in result["candidates"].items()},
                      "evidence_kind": result["evidence_kind"], "trading_authorized": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": type(exc).__name__ + ": " + str(exc)[:240]}))
        raise SystemExit(1)
