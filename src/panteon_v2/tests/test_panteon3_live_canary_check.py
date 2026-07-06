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


def test_canary_summary_persists_calibration_metadata(tmp_path):
    module = _load_tool()
    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
        calibration_only=True,
        actor_overrides={"CHECK_INT": 1, "MOM_MIN": 0.0015},
    )

    latest = json.loads(
        (tmp_path / "Reports" / "Panteon3Canary" / "latest_canary_summary.json").read_text(
            encoding="utf-8",
        )
    )

    assert summary["calibration_only"] is True
    assert summary["actor_overrides"] == {"CHECK_INT": 1, "MOM_MIN": 0.0015}
    assert latest["calibration_only"] is True
    assert latest["actor_overrides"] == {"CHECK_INT": 1, "MOM_MIN": 0.0015}


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


def test_canary_summary_passes_completed_flat_paper_canary_with_closed_feed(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "BITGET" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "BITGET",
            "mode": "paper_live_feed",
            "run_state": "stopped",
            "feed_status": "closed",
            "live_state_sync": {"reconcile_ok": True, "warnings": []},
            "open_positions": {},
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
    assert exchange["feed_status"] == "closed"
    assert exchange["run_state"] == "stopped"
    assert exchange["fail_reasons"] == []


def test_canary_summary_can_mark_execution_smoke_without_positive_expectancy_gate(tmp_path):
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
            "open_positions": {},
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
                "realized_pnl_usd": -0.01,
                "fees_usd": 0.01,
            },
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC",),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
        require_positive_expectancy=False,
        execution_smoke=True,
    )

    exchange = summary["exchanges"]["MEXC"]
    assert summary["passed"] is True
    assert summary["expectancy_gate_required"] is False
    assert summary["execution_smoke"] is True
    assert exchange["passed"] is True
    assert exchange["expectancy_gate_required"] is False
    assert exchange["execution_smoke"] is True
    assert exchange["expectancy_after_costs"] == -0.02
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


def test_canary_summary_blocks_filled_trade_without_cost_attribution(tmp_path):
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
                "realized_pnl_usd": 0.20,
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
    assert exchange["passed"] is False
    assert "cost_attribution_missing" in exchange["fail_reasons"]
    assert "cost_attribution_missing" in exchange["health_warnings"]
    assert exchange["cost_attribution"]["complete"] is False


def test_canary_summary_uses_root_event_log_for_fill_cost_attribution(tmp_path):
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
                "realized_pnl_usd": 0.20,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "bitget_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "OrderFilled",
                "symbol": "ADA",
                "fees": 0.03,
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
    assert exchange["passed"] is True
    assert exchange["fees_usd"] == 0.03
    assert exchange["expectancy_after_costs"] == 0.17
    assert "cost_attribution_missing" not in exchange["fail_reasons"]
    assert exchange["cost_attribution"]["complete"] is True
    assert exchange["cost_attribution"]["missing"] == []
    assert "context[0].fees" in exchange["cost_attribution"]["observed_keys"]


def test_canary_summary_deduplicates_repeated_root_fill_fee_events(tmp_path):
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
                "realized_pnl_usd": 0.20,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "bitget_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "OrderFilled",
                "symbol": "ADA",
                "fees": 0.03,
                "trade": {"fee": 0.03, "exchange_order_id": "FAKE-1"},
                "order_id": "FAKE-1",
            },
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "PositionOpened",
                "symbol": "ADA",
                "fees": 0.03,
                "order_id": "FAKE-1",
            },
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "ExecutionAttributed",
                "symbol": "ADA",
                "fees": 0.03,
                "order_id": "FAKE-1",
            },
            {
                "timestamp": "2026-06-27T10:02:01+00:00",
                "_type": "OrderFilled",
                "symbol": "ADA",
                "fees": 0.04,
                "trade": {"fee": 0.04, "exchange_order_id": "FAKE-2"},
                "order_id": "FAKE-2",
            },
            {
                "timestamp": "2026-06-27T10:02:01+00:00",
                "_type": "PositionClosed",
                "symbol": "ADA",
                "fees": 0.04,
                "order_id": "FAKE-2",
            },
            {
                "timestamp": "2026-06-27T10:02:01+00:00",
                "_type": "ExecutionAttributed",
                "symbol": "ADA",
                "fees": 0.04,
                "order_id": "FAKE-2",
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
    assert exchange["passed"] is True
    assert exchange["fees_usd"] == 0.07
    assert exchange["expectancy_after_costs"] == 0.13


def test_canary_summary_exports_zero_signal_diagnostics_with_context_rows(tmp_path):
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
            {"timestamp": "2026-06-27T10:01:00+00:00", "raw_signal_count": 0},
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "mexc_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "ContextEvaluated",
                "symbol": "BTC",
                "first_rejection_reason": "min_notional",
            },
            {
                "timestamp": "2026-06-27T10:01:02+00:00",
                "_type": "ContextEvaluated",
                "symbol": "ETH",
                "first_rejection_reason": "contract_filter",
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
    assert exchange["signals"] == 0
    assert exchange["context_event_rows_evaluated"] == 2
    assert exchange["zero_signal_diagnostics"] == {
        "enabled": True,
        "exchange": "MEXC",
        "decision_rows": 1,
        "context_event_rows": 2,
        "symbols_with_context": ["BTC", "ETH"],
        "first_rejection_reason_counts": {
            "contract_filter": 1,
            "min_notional": 1,
        },
    }


def test_canary_summary_exports_signal_execution_block_diagnostics(tmp_path):
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
                "n_orders": 0,
                "n_filled": 0,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "mexc_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "SignalEmitted",
                "symbol": "BNB",
                "signal": {
                    "sym": "BNB",
                    "action": 2,
                    "risk_mult": 0.03,
                    "by_player": "LiveOIBreakout",
                },
            },
            {
                "timestamp": "2026-06-27T10:01:02+00:00",
                "_type": "ExecutionAttributed",
                "symbol": "BNB",
                "sym": "BNB",
                "action": "SPOT_BUY_FULL",
                "status": "blocked",
                "reason": "risk_limits: notional $0.30 < min $5.00",
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
    assert "min_notional_blocked" in exchange["fail_reasons"]
    assert exchange["execution_block_diagnostics"] == {
        "enabled": True,
        "blocked_count": 1,
        "first_blocked_reason_counts": {
            "risk_limits: notional $0.30 < min $5.00": 1,
        },
        "min_notional_blocked_count": 1,
        "blocked_symbols": ["BNB"],
        "blocked_actions": ["SPOT_BUY_FULL"],
        "first_blocked_examples": [
            {
                "symbol": "BNB",
                "action": "SPOT_BUY_FULL",
                "reason": "risk_limits: notional $0.30 < min $5.00",
                "timestamp": "2026-06-27T10:01:02+00:00",
            }
        ],
    }


def test_canary_summary_exports_cross_exchange_zero_signal_diff(tmp_path):
    module = _load_tool()
    mexc = tmp_path / "Results" / "MEXC" / "run"
    bitget = tmp_path / "Results" / "BITGET" / "run"
    for exchange, session in (("MEXC", mexc), ("BITGET", bitget)):
        _write_json(
            session / "status.json",
            {
                "timestamp_utc": "2026-06-27T10:00:00+00:00",
                "exchange": exchange,
                "mode": "paper_live_feed",
                "run_state": "running",
                "feed_status": "active",
                "live_state_sync": {"reconcile_ok": True, "warnings": []},
            },
        )
    _write_jsonl(
        mexc / "causal_entry_decisions.jsonl",
        [{"timestamp": "2026-06-27T10:01:00+00:00", "raw_signal_count": 0}],
    )
    _write_jsonl(
        bitget / "causal_entry_decisions.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:00+00:00",
                "raw_signal_count": 1,
                "n_orders": 1,
                "n_filled": 1,
                "realized_pnl_usd": 0.30,
                "fees_usd": 0.02,
            }
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "mexc_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "ContextEvaluated",
                "symbol": "BTC",
                "first_rejection_reason": "min_notional",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "bitget_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:01:01+00:00",
                "_type": "ContextEvaluated",
                "symbol": "BTC",
            },
            {
                "timestamp": "2026-06-27T10:01:02+00:00",
                "_type": "ContextEvaluated",
                "symbol": "ETH",
            },
        ],
    )

    summary = module.build_canary_summary(
        results_root=tmp_path / "Results",
        reports_dir=tmp_path / "Reports" / "Panteon3Canary",
        exchanges=("MEXC", "BITGET"),
        now=module._parse_time("2026-06-27T10:05:00+00:00"),
    )

    assert summary["cross_exchange_diagnostics"] == [
        {
            "zero_signal_exchange": "MEXC",
            "active_exchange": "BITGET",
            "signals_delta": -1,
            "context_event_rows_delta": -1,
            "suspect_layers": ["symbol_mapping", "contract_filters", "min_notional"],
            "zero_signal_symbols": ["BTC"],
            "active_symbols": ["BTC", "ETH"],
        }
    ]


def test_canary_summary_exports_negative_context_deny_keys(tmp_path):
    module = _load_tool()
    session = tmp_path / "Results" / "BITGET" / "run"
    _write_json(
        session / "status.json",
        {
            "timestamp_utc": "2026-06-27T10:00:00+00:00",
            "exchange": "BITGET",
            "mode": "paper_live_feed",
            "run_state": "stopped",
            "feed_status": "closed",
            "live_state_sync": {"reconcile_ok": True, "warnings": []},
            "open_positions": {},
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
                "realized_pnl_usd": -0.20,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "Results" / "logs" / "bitget_single_component_events.jsonl",
        [
            {
                "timestamp": "2026-06-27T10:02:00+00:00",
                "_type": "PositionClosed",
                "symbol": "BNB",
                "open_action": "FUT_SHORT_HALF",
                "open_regime": "range_low_vol",
                "by_player": "LiveOIBreakout",
                "realized_pnl": -0.20,
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
    assert exchange["event_rows_evaluated"] == 0
    assert exchange["context_event_rows_evaluated"] == 1
    assert exchange["negative_closed_trade_contexts"] == [
        {
            "actor_label": "LiveOIBreakout",
            "symbol": "BNB",
            "action": "FUT_SHORT_HALF",
            "regime": "range_low_vol",
            "realized_pnl": -0.20,
            "timestamp": "2026-06-27T10:02:00+00:00",
        }
    ]
    assert "agent:LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol" in exchange[
        "negative_context_signal_keys"
    ]
    assert "ensemble:Solo_LiveOIBreakout|BNB|FUT_SHORT_HALF|range_low_vol" in exchange[
        "negative_context_signal_keys"
    ]


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
