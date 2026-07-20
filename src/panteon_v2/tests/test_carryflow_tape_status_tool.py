from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "check_bitget_carryflow_tape.py"
COLLECTOR_PATH = ROOT / "tools" / "collect_bitget_carryflow_tape.py"
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


def _load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_bitget_carryflow_tape_for_status_test",
        COLLECTOR_PATH,
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


def test_status_tool_accepts_healthy_scheduled_idle_collector(tmp_path):
    from panteon_v2.policy import append_tape_sample, write_warmup_seed

    tool = _load_tool()
    collector = _load_collector()
    run_dir = tmp_path / "scheduled"
    run_dir.mkdir()
    close_ms = int(NOW.timestamp() * 1000)
    observed = NOW + timedelta(seconds=30)
    tape_path = run_dir / "carryflow_evidence_tape.jsonl"
    append_tape_sample(
        tape_path,
        collector.build_tape_payload(
            collector_run_id="scheduled-run",
            source_revision="d863585",
            collector_code_sha256="1" * 64,
            funding_fetcher_code_sha256="2" * 64,
            observed_at=observed,
            bar_interval_seconds=3600,
            bar_close_timestamp_ms=close_ms,
            max_bar_close_lag_seconds=120,
            max_derivatives_age_seconds=1200,
            symbols=("BTC",),
            candles={
                "BTC": {
                    "candle_timestamp_ms": close_ms - 3_600_000,
                    "open": 100.0,
                    "high": 102.0,
                    "low": 99.0,
                    "close": 101.0,
                    "volume": 500.0,
                }
            },
            derivatives={
                "BTC": {
                    "funding_rate": 0.0002,
                    "open_interest_usdt": 1_000_000.0,
                    "long_ratio": 0.65,
                    "short_ratio": 0.35,
                    "mark_price": 100.0,
                    "index_price": 100.0,
                    "last_price": 101.0,
                    "next_funding_ts": 0,
                    "long_short_ratio_ts": int(observed.timestamp() * 1000),
                    "long_short_ratio_age_sec": 0.0,
                    "long_short_source": "account_long_short_v2",
                    "updated_ts": observed.timestamp() - 5.0,
                    "context_complete": True,
                }
            },
        ),
    )
    bars = []
    for index in range(3):
        start = close_ms - (4 - index) * 3_600_000
        bars.append(
            {
                "bar_close_timestamp_ms": start + 3_600_000,
                "symbols": {
                    "BTC": {
                        "candle_timestamp_ms": start,
                        "open": 100.0,
                        "high": 102.0,
                        "low": 99.0,
                        "close": 101.0,
                        "volume": 500.0,
                    }
                },
            }
        )
    write_warmup_seed(
        run_dir / "carryflow_warmup_seed.json",
        collector.build_warmup_seed_payload(
            collector_run_id="scheduled-run",
            source_revision="d863585",
            collector_code_sha256="1" * 64,
            created_at=NOW - timedelta(minutes=1),
            bar_interval_seconds=3600,
            intended_first_evidence_bar_close_timestamp_ms=close_ms,
            symbols=("BTC",),
            bars=bars,
        ),
    )
    (run_dir / "collector_status.json").write_text(
        json.dumps(
            {
                "updated_at": observed.isoformat(),
                "pid": 123,
                "orders_enabled": False,
                "run_state": "scheduled_idle",
                "collection_mode": "scheduled_one_shot",
                "sample_index": 1,
                "samples_target": 0,
            }
        ),
        encoding="utf-8",
    )

    summary = tool.summarize_run(
        run_dir,
        now=NOW + timedelta(minutes=30),
        pid_exists_fn=lambda _pid: False,
    )

    assert summary["healthy"] is True
    assert summary["scheduled_idle"] is True
    assert summary["process_alive"] is False
    assert summary["warmup_seed"]["prospective_for_tape"] is True
