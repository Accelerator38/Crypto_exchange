from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "run_panteon3_live_canary_check.py"
    spec = importlib.util.spec_from_file_location("run_panteon3_live_canary_check", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_canary_summary_fails_when_runtime_has_no_orders_or_fills(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "MEXC" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "MEXC",
            "mode": "live_futures",
            "run_state": "running",
            "feed_status": "active",
            "live_state_sync": {"reconcile_ok": True},
        },
    )
    _write_jsonl(
        session / "causal_entry_decisions.jsonl",
        [
            {"timestamp": "2026-06-27T10:00:00+00:00", "raw_signal_count": 2, "executable_signal_count": 0, "n_filled": 0},
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
    )

    exchange = summary["exchanges"]["MEXC"]
    assert summary["passed"] is False
    assert exchange["passed"] is False
    assert exchange["signals"] == 2
    assert exchange["orders"] == 0
    assert exchange["fills"] == 0
    assert "zero_orders" in exchange["fail_reasons"]
    assert "zero_fills" in exchange["fail_reasons"]
    assert (tmp_path / "Reports" / "Panteon3Canary" / "latest_canary_summary.json").exists()


def test_canary_summary_passes_with_positive_fills_and_clean_reconcile(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "BITGET" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "BITGET",
            "mode": "paper_live_feed",
            "run_state": "running",
            "feed_status": "active",
            "live_state_sync": {"reconcile_ok": True, "warnings": []},
        },
    )
    _write_jsonl(
        session / "causal_entry_decisions.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:00+00:00",
                "raw_signal_count": 1,
                "executable_signal_count": 1,
                "n_orders": 1,
                "n_filled": 1,
                "realized_pnl_usd": 0.42,
                "fees_usd": 0.05,
                "funding_usd": -0.01,
                "slippage_usd": 0.02,
            },
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("BITGET",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
    )

    exchange = summary["exchanges"]["BITGET"]
    assert summary["passed"] is True
    assert exchange["passed"] is True
    assert exchange["signals"] == 1
    assert exchange["orders"] == 1
    assert exchange["fills"] == 1
    assert exchange["expectancy_after_costs"] == 0.34
    assert exchange["reconcile_ok"] is True
    assert exchange["fail_reasons"] == []


def test_canary_summary_counts_order_fill_events_and_nested_trade_costs(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "MEXC" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "MEXC",
            "mode": "paper_live_feed",
            "run_state": "running",
            "feed_status": "active",
            "live_state_sync": {"reconcile_ok": True, "warnings": []},
        },
    )
    _write_jsonl(
        session / "causal_entry_decisions.jsonl",
        [
            {"timestamp": "2026-06-27T10:01:00+00:00", "raw_signal_count": 1},
        ],
    )
    _write_jsonl(
        session / "events.jsonl",
        [
            {"timestamp": "2026-06-27T10:01:01+00:00", "_type": "OrderSent"},
            {
                "timestamp": "2026-06-27T10:01:02+00:00",
                "_type": "OrderFilled",
                "trade": {"fee": 0.04, "funding": -0.01},
            },
            {
                "timestamp": "2026-06-27T10:02:00+00:00",
                "_type": "PositionClosed",
                "realized_pnl": 0.20,
                "slippage_usd": 0.01,
            },
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
    )

    exchange = summary["exchanges"]["MEXC"]
    assert exchange["passed"] is True
    assert exchange["signals"] == 1
    assert exchange["orders"] == 1
    assert exchange["fills"] == 1
    assert exchange["fees_usd"] == 0.04
    assert exchange["funding_usd"] == -0.01
    assert exchange["slippage_usd"] == 0.01
    assert exchange["expectancy_after_costs"] == 0.14


def test_canary_summary_counts_filled_decision_as_order_when_order_counter_missing(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "MEXC" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "MEXC",
            "mode": "paper_live_feed",
            "run_state": "running",
            "feed_status": "active",
            "live_state_sync": {"reconcile_ok": True, "warnings": []},
        },
    )
    _write_jsonl(
        session / "causal_entry_decisions.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:00+00:00",
                "raw_signal_count": 1,
                "executable_signal_count": 1,
                "n_filled": 1,
                "realized_pnl_usd": 0.30,
                "fees_usd": 0.02,
            },
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
    )

    exchange = summary["exchanges"]["MEXC"]
    assert exchange["signals"] == 1
    assert exchange["orders"] == 1
    assert exchange["fills"] == 1
    assert "zero_orders" not in exchange["fail_reasons"]
    assert exchange["passed"] is True


def test_canary_summary_blocks_owned_open_positions_at_session_end(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "MEXC" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "MEXC",
            "mode": "paper_live_feed",
            "run_state": "stopped",
            "feed_status": "closed",
            "live_state_sync": {"reconcile_ok": True, "warnings": []},
            "open_positions": {
                "ETH": {
                    "side": "short",
                    "qty": 0.003,
                    "entry": 1564.33,
                    "by_player": "LiveVolCompress",
                    "external": False,
                }
            },
        },
    )
    _write_jsonl(
        session / "causal_entry_decisions.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:00+00:00",
                "raw_signal_count": 1,
                "executable_signal_count": 1,
                "n_orders": 1,
                "n_filled": 1,
                "realized_pnl_usd": 0.30,
                "fees_usd": 0.02,
            },
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
    )

    exchange = summary["exchanges"]["MEXC"]
    assert exchange["passed"] is False
    assert exchange["open_position_count"] == 1
    assert exchange["owned_open_position_count"] == 1
    assert exchange["open_position_symbols"] == ["ETH"]
    assert "open_positions_not_flat" in exchange["fail_reasons"]
