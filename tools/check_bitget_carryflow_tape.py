"""One-command health check for the latest Bitget CarryFlow tape collector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_v2.policy import (  # noqa: E402
    CarryFlowEvidenceTape,
    CarryFlowWarmupSeed,
)


DEFAULT_ROOT = ROOT / "Retrodate" / "bitget_carryflow_tape"


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            return True
        error = ctypes.windll.kernel32.GetLastError()  # type: ignore[attr-defined]
        return error == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def find_latest_run(root: str | Path = DEFAULT_ROOT) -> Path:
    candidates = list(Path(root).glob("*/collector_status.json"))
    if not candidates:
        raise FileNotFoundError("no CarryFlow collector status found")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest.parent


def summarize_run(
    run_dir: str | Path,
    *,
    now: datetime | None = None,
    pid_exists_fn: Callable[[int], bool] = _pid_exists,
) -> dict[str, Any]:
    directory = Path(run_dir).resolve()
    status_path = directory / "collector_status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"collector status not found: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    updated_at = datetime.fromisoformat(str(status["updated_at"]).replace("Z", "+00:00"))
    if updated_at.tzinfo is None:
        raise ValueError("collector status timestamp must be timezone-aware")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age_sec = max(0.0, (current - updated_at.astimezone(timezone.utc)).total_seconds())
    pid = int(status.get("pid", 0) or 0)
    process_alive = pid_exists_fn(pid)
    run_state = str(status.get("run_state") or "unknown")
    status_fresh = age_sec <= 2 * 60 * 60
    tape_path = directory / "carryflow_evidence_tape.jsonl"
    tape_valid = True
    tape_error = ""
    tape_summary = None
    if tape_path.exists() and tape_path.stat().st_size > 0:
        try:
            tape_summary = CarryFlowEvidenceTape.from_jsonl(tape_path).describe()
        except Exception as exc:
            tape_valid = False
            tape_error = f"{type(exc).__name__}: {exc}"
    active_state = run_state in {
        "waiting_for_bar_close",
        "collecting",
        "running",
    }
    scheduled_mode = status.get("collection_mode") == "scheduled_one_shot"
    scheduled_idle = scheduled_mode and run_state == "scheduled_idle"
    warmup_seed_path = directory / "carryflow_warmup_seed.json"
    warmup_seed_valid = not scheduled_mode
    warmup_seed_error = ""
    warmup_seed_summary = None
    if warmup_seed_path.is_file() and tape_summary is not None:
        try:
            tape = CarryFlowEvidenceTape.from_jsonl(tape_path)
            warmup_seed = CarryFlowWarmupSeed.from_json(
                warmup_seed_path,
                expected_symbols=tape.symbols,
            )
            warmup_seed.validate_for_tape(tape)
            prospective = warmup_seed.is_prospective_for_tape(tape)
            code_matches = (
                warmup_seed.payload["collector_code_sha256"]
                == tape.collector_code_sha256
            )
            warmup_seed_valid = prospective and code_matches
            warmup_seed_summary = {
                **warmup_seed.describe(),
                "prospective_for_tape": prospective,
                "collector_code_matches_tape": code_matches,
            }
            if not warmup_seed_valid:
                warmup_seed_error = "warm-up seed is not prospective or code-pinned"
        except Exception as exc:
            warmup_seed_valid = False
            warmup_seed_error = f"{type(exc).__name__}: {exc}"
    elif scheduled_mode:
        warmup_seed_error = "scheduled root has no warm-up seed"
    healthy = bool(
        status.get("orders_enabled") is False
        and (process_alive or scheduled_idle)
        and status_fresh
        and tape_valid
        and (active_state or scheduled_idle)
        and warmup_seed_valid
    )
    blockers: list[str] = []
    if status.get("orders_enabled") is not False:
        blockers.append("orders_enabled_not_false")
    if not process_alive and not scheduled_idle:
        blockers.append("collector_process_not_running")
    if not status_fresh:
        blockers.append("collector_status_stale")
    if not tape_valid:
        blockers.append("evidence_tape_invalid")
    if not active_state and not scheduled_idle:
        blockers.append(f"collector_state_{run_state}")
    if not warmup_seed_valid:
        blockers.append("warmup_seed_invalid")
    return {
        "healthy": healthy,
        "blockers": blockers,
        "run_dir": str(directory),
        "run_state": run_state,
        "orders_enabled": status.get("orders_enabled"),
        "pid": pid,
        "process_alive": process_alive,
        "collection_mode": status.get("collection_mode", "continuous_process"),
        "scheduled_idle": scheduled_idle,
        "status_age_sec": age_sec,
        "next_collection_at": status.get("next_collection_at", ""),
        "sample_index": int(status.get("sample_index", 0) or 0),
        "samples_target": int(status.get("samples_target", 0) or 0),
        "last_sample_complete": status.get("last_sample_complete"),
        "tape": tape_summary,
        "tape_error": tape_error,
        "warmup_seed": warmup_seed_summary,
        "warmup_seed_error": warmup_seed_error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the latest public Bitget CarryFlow tape collector."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run(args.root)
    summary = summarize_run(run_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
