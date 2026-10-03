"""Read-only environment/state inspection; does not launch the frozen runner."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.future_ema_operations_v1 import (
    environment_differences, environment_snapshot, inspect_artifacts, read_json_artifact, sha_bytes,
)

CONFIG = ROOT / "configs/genetic_primus_future_ema_viability_v1.json"
CONFIG_SHA = "11b83d2e3ecbdda572307dd902b73c9c2d8ae108499be85e22b988f193b227b6"
LOCK = ROOT / "configs/genetic_primus_future_ema_implementation_v1.json"
LOCK_SHA = "654477fad14c36bef92ef3ea584bb759f3243d98be9fe1872d13fbbba5b3832a"
ENVIRONMENT = ROOT / "configs/genetic_primus_future_ema_environment_v1.json"
ENVIRONMENT_SHA = "8f82114b946f09afcd4aa8b6e5dcd721fb7c2f303a09f1c2dac3a0fc51137360"
DESTINATION = ROOT / "Reports/Exia/Genetic_Primus/future_ema_viability_v1_20261005"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("environment", "inspect"))
    args = parser.parse_args()
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("use the registered workspace Python")
    config, digest = read_json_artifact(CONFIG)
    if digest != CONFIG_SHA:
        raise ValueError("registered contract changed")
    lock, lock_sha = read_json_artifact(LOCK)
    if lock_sha != LOCK_SHA:
        raise ValueError("registered implementation lock changed")
    for relative, expected_sha in {**config["frozen_code_sha256"], **lock["files_sha256"]}.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha_bytes(path.read_bytes()) != expected_sha:
            raise ValueError("frozen implementation changed")
    if args.mode == "environment":
        expected, environment_sha = read_json_artifact(ENVIRONMENT)
        if (environment_sha != ENVIRONMENT_SHA
                or expected.get("schema_version") != "exia.genetic_primus.future_ema_environment/1"
                or expected.get("config_sha256") != digest
                or expected.get("implementation_lock_sha256") != lock_sha):
            raise ValueError("environment registration lineage differs")
        differences = environment_differences(expected["snapshot"], environment_snapshot())
        result = {"status": "ENVIRONMENT_MATCH" if not differences else "ENVIRONMENT_DRIFT_BLOCKED",
                  "differences": differences, "market_files_opened": False,
                  "runner_launched": False, "promotion_authority": False}
        print(json.dumps(result, separators=(",", ":")))
        return bool(differences)
    result = inspect_artifacts(DESTINATION, objective_id=config["objective_id"],
                              config_sha256=digest, implementation_lock_sha256=lock_sha,
                              safety=config["safety"])
    print(json.dumps(result, separators=(",", ":")))
    return result["status"] == "MANUAL_REVIEW_BLOCKED"


if __name__ == "__main__":
    raise SystemExit(main())
