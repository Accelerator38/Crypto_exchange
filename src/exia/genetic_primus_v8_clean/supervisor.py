from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from .collector import collect_public_liquidations
from .core import load_contract
from .status import check_run


CollectFn = Callable[..., Awaitable[dict[str, Any]]]
CheckFn = Callable[..., dict[str, Any]]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _root_directories(base_dir: Path) -> list[Path]:
    return sorted(
        (path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("root_")),
        key=lambda path: path.name,
    )


class SupervisorLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "SupervisorLock":
        self.descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(self.descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(self.descriptor)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


async def run_bounded_supervisor(
    *,
    root: Path,
    contract_path: Path,
    base_dir: Path,
    cycles: int,
    root_duration_seconds: float,
    pause_seconds: float,
    collect_fn: CollectFn = collect_public_liquidations,
    check_fn: CheckFn = check_run,
) -> dict[str, Any]:
    if not 1 <= cycles <= 24:
        raise ValueError("cycles must be between 1 and 24")
    if not 0 < root_duration_seconds <= 300:
        raise ValueError("root_duration_seconds must be between 0 and 300")
    if not 0 <= pause_seconds <= 60:
        raise ValueError("pause_seconds must be between 0 and 60")

    root = Path(root).resolve()
    contract_path = Path(contract_path).resolve()
    base_dir = Path(base_dir).resolve()
    load_contract(contract_path, root=root)
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    base_dir.mkdir(parents=True, exist_ok=True)
    status_path = base_dir / "supervisor_status.json"
    completed_roots: list[str] = []
    started = time.monotonic()

    def update(state: str, *, cycle: int, current_root: str | None, blocker: str | None = None) -> None:
        _atomic_json(
            status_path,
            {
                "schema": "genetic_primus.v8.clean.supervisor_status/1",
                "state": state,
                "pid": os.getpid(),
                "updated_at": datetime.now(UTC).isoformat(),
                "contract_sha256": contract_sha256,
                "cycle": cycle,
                "cycles_requested": cycles,
                "completed_roots": completed_roots,
                "current_root": current_root,
                "uptime_seconds": time.monotonic() - started,
                "blocker": blocker,
                "credentials_allowed": False,
                "orders_enabled": False,
                "promotion_authority": False,
            },
        )

    with SupervisorLock(base_dir / "supervisor.lock"):
        update("starting", cycle=0, current_root=None)
        try:
            for cycle in range(1, cycles + 1):
                roots = _root_directories(base_dir)
                if roots:
                    previous = roots[-1]
                    previous_check = check_fn(
                        run_dir=previous,
                        contract_path=contract_path,
                        root=root,
                    )
                    if previous_check.get("result") != "SUCCESS":
                        blocker = f"previous_root_failed:{previous.name}"
                        update("blocked", cycle=cycle, current_root=None, blocker=blocker)
                        return {
                            "result": "FAILURE",
                            "cycles_completed": len(completed_roots),
                            "roots": completed_roots,
                            "blocker": blocker,
                            "previous_check": previous_check,
                            "orders_enabled": False,
                            "promotion_authority": False,
                        }

                root_id = (
                    f"root_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_"
                    f"{contract_sha256[:8]}_{uuid.uuid4().hex[:8]}"
                )
                run_dir = base_dir / root_id
                update("collecting", cycle=cycle, current_root=root_id)
                await collect_fn(
                    contract_path=contract_path,
                    run_dir=run_dir,
                    duration_seconds=root_duration_seconds,
                )
                current_check = check_fn(
                    run_dir=run_dir,
                    contract_path=contract_path,
                    root=root,
                )
                if current_check.get("result") != "SUCCESS":
                    blocker = f"current_root_failed:{root_id}"
                    update("blocked", cycle=cycle, current_root=root_id, blocker=blocker)
                    return {
                        "result": "FAILURE",
                        "cycles_completed": len(completed_roots),
                        "roots": completed_roots,
                        "blocker": blocker,
                        "current_check": current_check,
                        "orders_enabled": False,
                        "promotion_authority": False,
                    }
                completed_roots.append(root_id)
                update("cycle_completed", cycle=cycle, current_root=root_id)
                if cycle < cycles and pause_seconds:
                    await asyncio.sleep(pause_seconds)
            update("completed", cycle=cycles, current_root=completed_roots[-1])
        except BaseException as exc:
            update(
                "failed",
                cycle=len(completed_roots) + 1,
                current_root=None,
                blocker=f"{type(exc).__name__}: {exc}"[:500],
            )
            raise

    return {
        "result": "SUCCESS",
        "cycles_completed": len(completed_roots),
        "roots": completed_roots,
        "contract_sha256": contract_sha256,
        "orders_enabled": False,
        "promotion_authority": False,
    }
