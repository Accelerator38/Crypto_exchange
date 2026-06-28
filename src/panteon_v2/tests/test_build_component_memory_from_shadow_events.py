from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from panteon_v2.selection.component_memory import ComponentMemory


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "build_component_memory_from_shadow_events.py"
    spec = importlib.util.spec_from_file_location("component_memory_builder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_component_memory_rows_from_shadow_events(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 8,
                "label": "LiveVolCompress",
                "regime": "range_low_vol",
                "pnl_usd": 0.30,
                "closed_trades": 3,
            },
            {
                "bar": 9,
                "label": "CarryFlowAgentV2",
                "regime": "bullish",
                "pnl_usd": -0.10,
                "closed_trades": 2,
            },
            {
                "bar": 10,
                "label": "OtherActor",
                "regime": "bullish",
                "pnl_usd": 10.0,
                "closed_trades": 10,
            },
            {
                "bar": 11,
                "label": "LiveVolCompress",
                "regime": "range_low_vol",
                "pnl_usd": 0.0,
                "closed_trades": 0,
            },
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveVolCompress", "CarryFlowAgentV2"],
        min_closed_trades=1,
    )

    assert [row["actor_label"] for row in rows] == [
        "CarryFlowAgentV2",
        "LiveVolCompress",
    ]
    assert rows[0]["expectancy"] == pytest.approx(-0.05)
    assert rows[1]["expectancy"] == pytest.approx(0.10)
    assert rows[1]["symbol"] == "*"
    assert rows[1]["action"] == "*"


def test_written_component_memory_is_usable_as_prior_bar_wildcard(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    out_path = tmp_path / "component_memory.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 8,
                "label": "LiveVolCompress",
                "regime": "range_low_vol",
                "pnl_usd": 0.30,
                "closed_trades": 3,
            }
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveVolCompress"],
        min_closed_trades=1,
    )
    module.write_component_memory_jsonl(out_path, rows)

    memory = ComponentMemory.from_jsonl(out_path)
    stat = memory.best_prior(
        "LiveVolCompress",
        symbol="BTC/USDT",
        regime="range_low_vol",
        action="FUT_SHORT_FULL",
        bar=9,
    )

    assert stat is not None
    assert stat.bar == 8
    assert stat.expectancy == pytest.approx(0.10)


def test_component_memory_builder_can_seed_prior_bar_for_new_live_sessions(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    out_path = tmp_path / "component_memory.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 85,
                "label": "LiveOIBreakout",
                "regime": "range_low_vol",
                "pnl_usd": 0.60,
                "closed_trades": 3,
                "symbol_action_outcomes": [
                    ["BTC/USDT", "FUT_SHORT_HALF", 0.60, 3, 3],
                ],
            }
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveOIBreakout"],
        min_closed_trades=1,
        include_cumulative_rollups=True,
        seed_prior_bar=0,
    )
    module.write_component_memory_jsonl(out_path, rows)

    assert rows
    assert {row["bar"] for row in rows} == {0}
    memory = ComponentMemory.from_jsonl(out_path)
    stat = memory.best_prior(
        "LiveOIBreakout",
        symbol="BTC/USDT",
        regime="range_low_vol",
        action="FUT_SHORT_HALF",
        bar=1,
        min_closed_trades=3,
        min_expectancy=0.0,
    )

    assert stat is not None
    assert stat.bar == 0
    assert stat.expectancy == pytest.approx(0.20)


def test_seed_prior_bar_preserves_cumulative_growth_when_bars_collapse(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 85,
                "label": "LiveOIBreakout",
                "regime": "bearish",
                "pnl_usd": 0.30,
                "closed_trades": 1,
            },
            {
                "bar": 120,
                "label": "LiveOIBreakout",
                "regime": "bearish",
                "pnl_usd": 0.70,
                "closed_trades": 2,
            },
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveOIBreakout"],
        min_closed_trades=1,
        include_cumulative_rollups=True,
        seed_prior_bar=0,
    )

    global_rollups = [
        row
        for row in rows
        if row.get("memory_scope") == "cumulative_global"
        and row.get("actor_label") == "LiveOIBreakout"
    ]
    assert {row["bar"] for row in global_rollups} == {0}
    assert max(row["closed_trades"] for row in global_rollups) == 3
    assert max(row["expectancy"] for row in global_rollups) == pytest.approx(1.0 / 3.0)


def test_build_component_memory_can_emit_cumulative_global_rollups(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 8,
                "label": "LiveVolCompress",
                "regime": "neutral",
                "pnl_usd": -0.30,
                "closed_trades": 3,
            },
            {
                "bar": 10,
                "label": "LiveVolCompress",
                "regime": "bearish",
                "pnl_usd": 0.90,
                "closed_trades": 2,
            },
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveVolCompress"],
        min_closed_trades=1,
        include_cumulative_rollups=True,
    )

    global_rows = [
        row
        for row in rows
        if row.get("memory_scope") == "cumulative_global"
        and row.get("regime") == "*"
        and row.get("bar") == 10
    ]
    assert len(global_rows) == 1
    assert global_rows[0]["closed_trades"] == 5
    assert global_rows[0]["expectancy"] == pytest.approx(0.12)


def test_build_component_memory_rows_include_symbol_action_outcomes(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 8,
                "label": "LiveVolCompress",
                "regime": "bearish",
                "pnl_usd": 0.20,
                "closed_trades": 2,
                "symbol_action_outcomes": [
                    ["BNB/USDT", "FUT_LONG_FULL", 0.40, 1, 1],
                    ["BNB/USDT", "FUT_SHORT_HALF", -0.20, 1, 0],
                ],
            },
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveVolCompress"],
        min_closed_trades=1,
    )

    by_key = {
        (row["symbol"], row["action"]): row
        for row in rows
        if row.get("memory_scope") == "symbol_action"
    }
    assert by_key[("BNB/USDT", "FUT_LONG_FULL")]["expectancy"] == pytest.approx(0.40)
    assert by_key[("BNB/USDT", "FUT_SHORT_HALF")]["expectancy"] == pytest.approx(-0.20)


def test_build_component_memory_can_emit_cumulative_symbol_action_rollups(tmp_path):
    module = _load_tool()
    events_path = tmp_path / "shadow_agent_pnl_events.jsonl"
    _write_jsonl(
        events_path,
        [
            {
                "bar": 8,
                "label": "LiveVolCompress",
                "regime": "neutral",
                "pnl_usd": -0.20,
                "closed_trades": 1,
                "symbol_action_outcomes": [
                    ["BNB/USDT", "FUT_SHORT_HALF", -0.20, 1, 0],
                ],
            },
            {
                "bar": 10,
                "label": "LiveVolCompress",
                "regime": "bearish",
                "pnl_usd": 0.10,
                "closed_trades": 1,
                "symbol_action_outcomes": [
                    ["BNB/USDT", "FUT_SHORT_HALF", 0.10, 1, 1],
                ],
            },
        ],
    )

    rows = module.build_component_memory_rows(
        [events_path],
        actor_labels=["LiveVolCompress"],
        min_closed_trades=1,
        include_cumulative_rollups=True,
    )

    rollups = [
        row
        for row in rows
        if row.get("memory_scope") == "cumulative_symbol_action"
        and row.get("symbol") == "BNB/USDT"
        and row.get("action") == "FUT_SHORT_HALF"
        and row.get("bar") == 10
    ]
    assert len(rollups) == 1
    assert rollups[0]["closed_trades"] == 2
    assert rollups[0]["expectancy"] == pytest.approx(-0.05)
