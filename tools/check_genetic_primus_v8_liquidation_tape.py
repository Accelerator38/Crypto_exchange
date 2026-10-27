from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.prospective_v8 import load_v8_contract, validate_tape  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/genetic_primus_v8_prospective_link_contract.json"


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def check(run_dir: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_v8_contract(contract_path, root=ROOT)
    run_dir = run_dir.resolve()
    status_path = run_dir / "collector_status.json"
    tape_path = run_dir / "liquidation_tape.jsonl"
    blockers: list[str] = []
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "FAILURE", "blockers": [f"invalid_status:{type(exc).__name__}"], "orders_enabled": False}
    if status.get("orders_enabled") is not False or status.get("promotion_authority") is not False:
        blockers.append("status_not_fail_closed")
    if status.get("contract_sha256") != contract.sha256:
        blockers.append("status_contract_mismatch")
    try:
        tape = validate_tape(tape_path, contract_sha256=contract.sha256)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        tape = None
        blockers.append(f"invalid_tape:{type(exc).__name__}:{exc}")
    updated = datetime.fromisoformat(str(status.get("updated_at")))
    if updated.tzinfo is None:
        blockers.append("status_timestamp_naive")
        age_seconds = None
    else:
        age_seconds = (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds()
    run_state = str(status.get("run_state") or "")
    process_running = _pid_exists(int(status.get("pid") or 0))
    if run_state in {"starting", "collecting", "reconnecting"}:
        if not process_running:
            blockers.append("collector_process_not_running")
        if age_seconds is None or age_seconds > float(contract.stream["status_stale_seconds"]):
            blockers.append("collector_status_stale")
    elif run_state != "completed":
        blockers.append(f"collector_state_{run_state or 'missing'}")
    if blockers:
        verdict = "FAILURE"
    elif run_state == "completed":
        verdict = "COMPLETED_BOUNDED_ROOT"
    else:
        verdict = "IN_PROGRESS"
    uptime = float(status.get("uptime_seconds") or 0.0)
    connected = float(status.get("connected_seconds") or 0.0)
    return {
        "status": verdict,
        "run_id": status.get("run_id"),
        "run_state": run_state,
        "process_running": process_running,
        "status_age_seconds": age_seconds,
        "connections": int(status.get("connections") or 0),
        "connected": bool(status.get("connected")),
        "connected_seconds": connected,
        "uptime_seconds": uptime,
        "source_coverage": connected / uptime if uptime > 0 else 0.0,
        "liquidation_events": int(status.get("liquidation_events") or 0),
        "link_events": int(status.get("link_events") or 0),
        "quality_events": int(status.get("quality_events") or 0),
        "tape": tape,
        "blockers": blockers,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Genetic_Primus V8 liquidation evidence root")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    result = check(args.run_dir, args.contract.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "FAILURE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
