from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "check_bitget_carryflow_tape.py"
NOW = datetime(2026, 7, 12, 14, 0, tzinfo=timezone.utc)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "check_bitget_carryflow_tape",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_status(run_dir: Path, *, updated_at: datetime, state: str = "waiting_for_bar_close"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "collector_status.json").write_text(
        json.dumps(
            {
                "updated_at": updated_at.isoformat(),
                "pid": 123,
                "orders_enabled": False,
                "run_state": state,
                "sample_index": 0,
                "samples_target": 720,
                "next_collection_at": (updated_at + timedelta(hours=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def test_status_tool_reports_healthy_waiting_collector_without_first_sample(tmp_path):
    tool = _load_tool()
    run_dir = tmp_path / "run"
    _write_status(run_dir, updated_at=NOW - timedelta(minutes=30))

    summary = tool.summarize_run(
        run_dir,
        now=NOW,
        pid_exists_fn=lambda pid: pid == 123,
    )

    assert summary["healthy"] is True
    assert summary["orders_enabled"] is False
    assert summary["sample_index"] == 0
    assert summary["tape"] is None


def test_status_tool_hard_blocks_dead_or_stale_collector(tmp_path):
    tool = _load_tool()
    run_dir = tmp_path / "run"
    _write_status(run_dir, updated_at=NOW - timedelta(hours=3))

    summary = tool.summarize_run(
        run_dir,
        now=NOW,
        pid_exists_fn=lambda _pid: False,
    )

    assert summary["healthy"] is False
    assert "collector_process_not_running" in summary["blockers"]
    assert "collector_status_stale" in summary["blockers"]
