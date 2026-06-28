from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from panteon_v2.app.live_preflight import (
    LivePreflightConfig,
    run_live_preflight,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_preflight_allows_virtual_modes_without_matrix_or_canary(tmp_path):
    result = run_live_preflight(
        "MEXC",
        "shadow_live_feed",
        config=LivePreflightConfig(project_root=tmp_path),
    )

    assert result.passed is True
    assert result.reasons == ()


def test_preflight_blocks_live_when_matrix_verdict_failed(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"promotion_verdict": {"passed": False, "fail_reasons": ["filled < min"]}},
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {"passed": True, "exchanges": {"MEXC": {"passed": True}}},
    )

    result = run_live_preflight(
        "MEXC",
        "live_futures",
        config=LivePreflightConfig(
            project_root=tmp_path,
            matrix_summary_path=matrix,
            canary_summary_path=canary,
        ),
    )

    assert result.passed is False
    assert "matrix_failed:filled < min" in result.reasons


def test_preflight_blocks_live_when_canary_has_zero_activity(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"promotion_verdict": {"passed": True, "fail_reasons": []}},
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "passed": True,
            "exchanges": {
                "BITGET": {
                    "passed": False,
                    "signals": 0,
                    "orders": 0,
                    "fills": 0,
                    "expectancy_after_costs": 0.01,
                    "reconcile_ok": True,
                }
            },
        },
    )

    result = run_live_preflight(
        "BITGET",
        "live_futures",
        config=LivePreflightConfig(
            project_root=tmp_path,
            matrix_summary_path=matrix,
            canary_summary_path=canary,
        ),
    )

    assert result.passed is False
    assert "canary_zero_signals" in result.reasons
    assert "canary_zero_orders" in result.reasons
    assert "canary_zero_fills" in result.reasons


def test_preflight_blocks_live_when_candidate_loses_to_best_component(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "candidate": {
                "realized_pnl_usd": 1.0,
                "best_component_label": "LiveVolCompress",
                "best_component_pnl_usd": 3.0,
                "panteon_beats_best_component": False,
            },
        },
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "passed": True,
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "signals": 2,
                    "orders": 2,
                    "fills": 2,
                    "expectancy_after_costs": 0.02,
                    "reconcile_ok": True,
                }
            },
        },
    )

    result = run_live_preflight(
        "MEXC",
        "live_futures",
        config=LivePreflightConfig(
            project_root=tmp_path,
            matrix_summary_path=matrix,
            canary_summary_path=canary,
        ),
    )

    assert result.passed is False
    assert "matrix_not_beating_best_component" in result.reasons


def test_preflight_blocks_live_when_matrix_or_canary_is_stale(tmp_path):
    now = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "generated_at": (now - timedelta(hours=30)).isoformat(),
            "promotion_verdict": {"passed": True, "fail_reasons": []},
        },
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "generated_at": now.isoformat(),
            "passed": True,
            "exchanges": {"MEXC": {"passed": True, "signals": 1, "orders": 1, "fills": 1, "expectancy_after_costs": 0.01, "reconcile_ok": True}},
        },
    )

    result = run_live_preflight(
        "MEXC",
        "live_futures",
        config=LivePreflightConfig(
            project_root=tmp_path,
            matrix_summary_path=matrix,
            canary_summary_path=canary,
            max_age_hours=24,
            now=now,
        ),
    )

    assert result.passed is False
    assert "matrix_stale" in result.reasons


def test_preflight_blocks_live_when_canary_reports_open_positions(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"promotion_verdict": {"passed": True, "fail_reasons": []}},
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "passed": True,
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "signals": 2,
                    "orders": 2,
                    "fills": 2,
                    "expectancy_after_costs": 0.02,
                    "reconcile_ok": True,
                    "open_position_count": 1,
                    "owned_open_position_count": 1,
                    "open_position_symbols": ["ETH"],
                }
            },
        },
    )

    result = run_live_preflight(
        "MEXC",
        "live_futures",
        config=LivePreflightConfig(
            project_root=tmp_path,
            matrix_summary_path=matrix,
            canary_summary_path=canary,
        ),
    )

    assert result.passed is False
    assert "canary_open_positions" in result.reasons
