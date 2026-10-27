"""Short, foreground-only contract checks and explicitly selected tiny fixtures.

Default never runs an evaluator. No pytest/collector imports, training or APIs.
"""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "exia" / "genetic_primus"))
from release_contract_v1 import object_sha, validate_contract

FILES = [
    "configs/genetic_primus_release_candidate_v1.json",
    "src/exia/genetic_primus/release_contract_v1.py",
    "src/exia/genetic_primus/execution_replay_v1.py",
    "tests/test_genetic_primus_release_candidate_v1.py",
    "tools/check_genetic_primus_release_candidate_v1.py",
    "tools/replay_genetic_primus_release_candidate_v1.py",
]


def save_new(name, value):
    destination = (ROOT / name).resolve()
    allowed = (ROOT / "Reports/Exia/Genetic_Primus/release_execution_v1_20260908").resolve()
    if destination.parent != allowed or destination.suffix != ".json":
        raise ValueError("output must be a new JSON in the scoped report directory")
    allowed.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["contract", "execution", "accounting", "safety", "bundle"])
    parser.add_argument("--receipt", help="optional NEW JSON receipt under release_execution_v1_20260908")
    parser.add_argument("--demo", help="with --group bundle only: write a NEW synthetic input bundle")
    args = parser.parse_args()
    if args.demo and args.group != "bundle":
        parser.error("--demo requires --group bundle")
    config = json.loads((ROOT / FILES[0]).read_text(encoding="utf-8"))
    check = validate_contract(config, root=ROOT)
    if not check["structurally_valid"]:
        print(json.dumps({"status": "INVALID", "errors": check["errors"]})[:850])
        return 1
    passed = []
    if args.group:
        namespace = runpy.run_path(str(ROOT / FILES[3]))
        for test in namespace["GROUPS"][args.group]:
            try:
                test()
            except Exception as exc:
                print(json.dumps({"status": "FAIL", "test": test.__name__,
                                  "error": type(exc).__name__ + ": " + str(exc)[:250]}))
                return 1
            passed.append(test.__name__)
    pins = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    receipt = {"schema_version": "exia.genetic_primus.offline_execution_verification/1",
               "source_hashes": pins, "check": check, "group": args.group,
               "passed_tests": passed, "synthetic_only": True, "pytest_used": False,
               "training_run": False, "network_used": False, "trading_authorized": False}
    receipt["receipt_body_sha256"] = object_sha(receipt)
    if args.receipt:
        save_new(args.receipt, receipt)
    if args.demo:
        save_new(args.demo, namespace["demo_bundle"]())
    print(json.dumps({"status": check["status"], "group": args.group, "passed": len(passed),
                      "blockers": len(check["qualification_blockers"]),
                      "contract_sha256": check["contract_sha256"], "trading_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
