from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "diagnose_benchmark_flash_parity.py"
    spec = importlib.util.spec_from_file_location("benchmark_flash_parity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_benchmark_flash_parity_classifies_actor_flash_status_by_bar(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "regime": "range_low_vol",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "action": "HOLD",
                        "reason": "selected",
                        "candidate_count": 2,
                        "candidates": [
                            {"label": "NoTrade", "actor_key": "NoTrade"},
                        ],
                    }
                ],
            },
            {
                "bar": 2,
                "timestamp": "2026-01-01T00:01:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "action": "HOLD",
                        "reason": "selected",
                        "candidates": [
                            {
                                "label": "FundingArb",
                                "actor_key": "agent:FundingArb",
                                "actor_type": "agent",
                                "action": "HOLD",
                                "reason": "inactive",
                                "rejected": True,
                                "rank": 4,
                                "score": 0.0,
                            },
                        ],
                    }
                ],
            },
            {
                "bar": 3,
                "timestamp": "2026-01-01T00:02:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "action": "HOLD",
                        "reason": "selected",
                        "candidates": [
                            {
                                "label": "LiveTrendFollow",
                                "actor_key": "agent:LiveTrendFollow",
                                "actor_type": "agent",
                                "action": "SPOT_BUY_FULL",
                                "reason": "insufficient_closed_trades",
                                "rejected": True,
                                "closed_trades": 1,
                                "pnl_net_pct": 0.3,
                                "rank": 2,
                                "score": 0.8,
                            },
                        ],
                    }
                ],
            },
            {
                "bar": 4,
                "timestamp": "2026-01-01T00:03:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "LiveAfterShock",
                        "action": "SPOT_BUY_HALF",
                        "reason": "selected",
                        "candidates": [
                            {
                                "label": "LiveTrendFollow",
                                "actor_key": "agent:LiveTrendFollow",
                                "actor_type": "agent",
                                "action": "SPOT_BUY_FULL",
                                "reason": "eligible",
                                "rejected": False,
                                "closed_trades": 6,
                                "pnl_net_pct": 1.2,
                                "rank": 2,
                                "score": 0.7,
                            },
                        ],
                    }
                ],
            },
            {
                "bar": 5,
                "timestamp": "2026-01-01T00:04:00+00:00",
                "regime": "bearish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "LiveTrendFollow",
                        "actor_type": "agent",
                        "action": "SPOT_BUY_FULL",
                        "reason": "selected",
                        "candidates": [
                            {
                                "label": "LiveTrendFollow",
                                "actor_key": "agent:LiveTrendFollow",
                                "actor_type": "agent",
                                "action": "SPOT_BUY_FULL",
                                "reason": "selected",
                                "rejected": False,
                                "closed_trades": 6,
                                "pnl_net_pct": 1.2,
                                "rank": 1,
                                "score": 0.9,
                            },
                        ],
                    }
                ],
            },
        ],
    )

    report = module.build_benchmark_flash_parity_report(
        [path],
        actor_labels=("FundingArb", "LiveTrendFollow"),
    )

    assert report["summary"]["bars_read"] == 5
    assert report["summary"]["detail_rows"] == 10
    assert report["actor_status_counts"]["FundingArb"] == {
        "hold_in_flash:inactive": 1,
        "missing_candidate": 4,
    }
    assert report["actor_status_counts"]["LiveTrendFollow"] == {
        "eligible_not_selected": 1,
        "missing_candidate": 2,
        "rejected:insufficient_closed_trades": 1,
        "selected": 1,
    }
    live_rows = [
        row for row in report["details"] if row["actor"] == "LiveTrendFollow"
    ]
    assert [row["flash_status"] for row in live_rows] == [
        "missing_candidate",
        "missing_candidate",
        "rejected:insufficient_closed_trades",
        "eligible_not_selected",
        "selected",
    ]


def test_benchmark_flash_parity_writes_json_details_and_markdown(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 1,
                "timestamp": "2026-01-01T00:00:00+00:00",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "candidates": [],
                    }
                ],
            }
        ],
    )
    report = module.build_benchmark_flash_parity_report(
        [path],
        actor_labels=("FundingArb",),
    )

    json_path, details_path, md_path = module.write_reports(
        report,
        tmp_path / "benchmark_flash_parity.json",
    )

    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"][
        "actor_labels"
    ] == ["FundingArb"]
    assert details_path.read_text(encoding="utf-8").count("\n") == 1
    markdown = md_path.read_text(encoding="utf-8")
    assert "# Benchmark-to-Flash Parity Diagnostics" in markdown
    assert "| missing_candidate | 1 |" in markdown


def test_benchmark_flash_parity_links_closed_standalone_events_to_flash_rows(tmp_path):
    module = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    _write_jsonl(
        path,
        [
            {
                "bar": 10,
                "timestamp": "2026-01-01T00:10:00+00:00",
                "regime": "bullish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "LiveAfterShock",
                        "action": "SPOT_BUY_HALF",
                        "reason": "selected",
                        "candidates": [
                            {
                                "label": "LiveTrendFollow",
                                "actor_key": "agent:LiveTrendFollow",
                                "action": "SPOT_BUY_FULL",
                                "reason": "insufficient_closed_trades",
                                "rejected": True,
                                "closed_trades": 1,
                                "pnl_net_pct": 0.2,
                                "score": 0.8,
                            }
                        ],
                    }
                ],
            },
            {
                "bar": 11,
                "timestamp": "2026-01-01T00:11:00+00:00",
                "regime": "bullish",
                "flash_decisions": [
                    {
                        "symbol": "BTC/USDT",
                        "selected_actor": "NoTrade",
                        "action": "HOLD",
                        "reason": "selected",
                        "candidates": [],
                    }
                ],
            },
        ],
    )
    _write_jsonl(
        tmp_path / "shadow_agent_pnl_events.jsonl",
        [
            {
                "bar": 10,
                "label": "LiveTrendFollow",
                "timestamp": "2026-01-01T00:10:00+00:00",
                "regime": "bullish",
                "pnl_usd": 3.5,
                "closed_trades": 1,
                "wins": 1,
            },
            {
                "bar": 11,
                "label": "FundingArb",
                "timestamp": "2026-01-01T00:11:00+00:00",
                "regime": "bullish",
                "pnl_usd": 1.2,
                "closed_trades": 1,
                "wins": 1,
            },
        ],
    )

    report = module.build_benchmark_flash_parity_report(
        [path],
        actor_labels=("LiveTrendFollow", "FundingArb"),
    )

    assert report["summary"]["standalone_trade_rows_available"] is True
    assert report["summary"]["standalone_trade_rows"] == 2
    assert report["standalone_trade_status_counts"] == {
        "missing_candidate": 1,
        "rejected:insufficient_closed_trades": 1,
    }
    rows = report["standalone_trade_parity"]
    assert rows[0]["actor"] == "LiveTrendFollow"
    assert rows[0]["standalone_pnl_usd"] == 3.5
    assert rows[0]["flash_status"] == "rejected:insufficient_closed_trades"
    assert rows[0]["flash_action"] == "SPOT_BUY_FULL"
    assert rows[1]["actor"] == "FundingArb"
    assert rows[1]["flash_status"] == "missing_candidate"
