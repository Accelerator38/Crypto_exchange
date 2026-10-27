from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from exia.genetic_primus_v8_clean.supervisor import run_bounded_supervisor


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/genetic_primus_v8_clean.json"


def test_supervisor_runs_one_checked_cycle(tmp_path: Path) -> None:
    collected: list[Path] = []

    async def fake_collect(**kwargs: Any) -> dict[str, Any]:
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(exist_ok=False)
        collected.append(run_dir)
        return {"root_id": run_dir.name, "orders_enabled": False}

    def fake_check(**kwargs: Any) -> dict[str, Any]:
        return {"result": "SUCCESS", "root_id": Path(kwargs["run_dir"]).name}

    result = asyncio.run(
        run_bounded_supervisor(
            root=ROOT,
            contract_path=CONTRACT,
            base_dir=tmp_path,
            cycles=1,
            root_duration_seconds=1,
            pause_seconds=0,
            collect_fn=fake_collect,
            check_fn=fake_check,
        )
    )
    assert result["result"] == "SUCCESS"
    assert result["cycles_completed"] == 1
    assert len(collected) == 1
    assert not (tmp_path / "supervisor.lock").exists()


def test_supervisor_blocks_before_collection_when_previous_root_fails(tmp_path: Path) -> None:
    previous = tmp_path / "root_20260824T000000Z_deadbeef_previous"
    previous.mkdir()
    collected = False

    async def fake_collect(**_kwargs: Any) -> dict[str, Any]:
        nonlocal collected
        collected = True
        return {}

    def fake_check(**_kwargs: Any) -> dict[str, Any]:
        return {"result": "FAILURE", "blockers": ["tampered_tape"]}

    result = asyncio.run(
        run_bounded_supervisor(
            root=ROOT,
            contract_path=CONTRACT,
            base_dir=tmp_path,
            cycles=1,
            root_duration_seconds=1,
            pause_seconds=0,
            collect_fn=fake_collect,
            check_fn=fake_check,
        )
    )
    assert result["result"] == "FAILURE"
    assert result["blocker"].startswith("previous_root_failed:")
    assert collected is False
    assert not (tmp_path / "supervisor.lock").exists()
