"""Explicit short direct test groups; no pytest, collector or training."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = "tests/test_genetic_primus_portfolio_replay_v1.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["policy", "comparison", "execution", "accounting", "safety"])
    parser.add_argument("--receipt", help="optional NEW scoped verification receipt")
    parser.add_argument("--demo", help="with --group accounting: NEW synthetic full8 input")
    args = parser.parse_args()
    if args.demo and args.group != "accounting":
        parser.error("--demo requires --group accounting")
    api = runpy.run_path(str(ROOT / "tools/run_genetic_primus_portfolio_replay_v1.py"))
    contract, release, identity = api["read_inputs"]()
    passed = []
    if args.group:
        tests = runpy.run_path(str(ROOT / TESTS))
        for test in tests["GROUPS"][args.group]:
            try:
                test()
            except Exception as exc:
                print(json.dumps({"status": "FAIL", "test": test.__name__,
                                  "error": type(exc).__name__ + ": " + str(exc)[:240]}))
                return 1
            passed.append(test.__name__)
    from release_contract_v1 import object_sha
    pins = dict(identity["source_hashes"])
    for name in (TESTS, "tools/check_genetic_primus_portfolio_replay_v1.py"):
        pins[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    receipt = {"schema_version": "exia.genetic_primus.portfolio_verification/1",
               "portfolio_contract_sha256": object_sha(contract), "implementation_sha256": identity["sha256"],
               "source_hashes": pins, "group": args.group, "passed_tests": passed,
               "synthetic_only": True, "pytest_used": False, "training_run": False,
               "network_used": False, "trading_authorized": False}
    receipt["receipt_body_sha256"] = object_sha(receipt)
    if args.receipt:
        api["save_new"](args.receipt, receipt)
    if args.demo:
        api["save_new"](args.demo, tests["demo_bundle"]())
    print(json.dumps({"status": "PASS", "group": args.group, "passed": len(passed),
                      "trading_authorized": False, "synthetic_only": True}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": type(exc).__name__ + ": " + str(exc)[:240]}))
        raise SystemExit(1)
