"""Explicit short admission regressions and NEW synthetic package fixture."""
import argparse
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["coverage", "provenance"], required=True)
    parser.add_argument("--receipt", help="NEW scoped JSON receipt")
    parser.add_argument("--demo", action="store_true", help="write NEW package_manifest/bundle/provenance JSONs")
    args = parser.parse_args()
    api = runpy.run_path(str(ROOT / "tools/import_genetic_primus_recorded_v1.py"))
    _, _, identity = api["context"]()
    test_path = "tests/test_genetic_primus_recorded_admission_v1.py"
    tests = runpy.run_path(str(ROOT / test_path))
    passed = []
    for test in tests["GROUPS"][args.group]:
        try:
            test()
        except Exception as exc:
            print(json.dumps({"status": "FAIL", "test": test.__name__, "error": type(exc).__name__ + ": " + str(exc)[:240]}))
            return 1
        passed.append(test.__name__)
    from release_contract_v1 import object_sha
    pins = api["importer_sources"]()
    for name in (test_path, "tools/check_genetic_primus_recorded_admission_v1.py"):
        pins[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    receipt = {"schema_version": "exia.genetic_primus.admission_verification/1", "source_hashes": pins,
               "frozen_portfolio_implementation_sha256": identity["sha256"], "group": args.group,
               "passed_tests": passed, "synthetic_only": True, "pytest_used": False,
               "training_run": False, "network_used": False, "trading_authorized": False}
    receipt["receipt_body_sha256"] = object_sha(receipt)
    if args.receipt:
        api["save_new"](args.receipt, receipt)
    if args.demo:
        m, b, p = tests["fixture"]()
        directory = "Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911/"
        names = {"bundle": "package_bundle.json", "provenance": "package_provenance.json"}
        for role, value in (("bundle", b), ("provenance", p)):
            api["save_new"](directory + names[role], value)
        m["artifacts"] = {}
        for role, name in names.items():
            raw = (ROOT / directory / name).read_bytes()
            m["artifacts"][role] = {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        api["save_new"](directory + "package_manifest.json", m)
    print(json.dumps({"status": "PASS", "group": args.group, "passed": len(passed),
                      "synthetic_only": True, "trading_authorized": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": type(exc).__name__ + ": " + str(exc)[:240]}))
        raise SystemExit(1)
