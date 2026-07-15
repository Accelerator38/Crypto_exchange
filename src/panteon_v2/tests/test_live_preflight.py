from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from panteon_v2.app.live_preflight import (
    LivePreflightConfig,
    config_from_env,
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


def test_preflight_propagates_exchange_canary_fail_reasons(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"promotion_verdict": {"passed": True, "fail_reasons": []}},
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "passed": False,
            "exchanges": {
                "MEXC": {
                    "passed": False,
                    "signals": 1,
                    "orders": 0,
                    "fills": 0,
                    "expectancy_after_costs": 0.0,
                    "reconcile_ok": True,
                    "fail_reasons": ["min_notional_blocked"],
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
    assert "canary_min_notional_blocked" in result.reasons


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


def test_preflight_accepts_selected_single_component_matrix_artifact(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "promotion_verdict": {"passed": True, "fail_reasons": []},
            "single_component_candidate": {
                "recommended": False,
                "label": "LiveOIBreakout",
                "requires_separate_matrix_artifact": False,
            },
            "candidate": {
                "variant": "single_component__LiveOIBreakout",
                "realized_pnl_usd": 2.5,
                "best_component_label": "LiveOIBreakout",
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

    assert "matrix_not_beating_best_component" not in result.reasons


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


def test_preflight_blocks_live_when_canary_expectancy_gate_disabled(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"promotion_verdict": {"passed": True, "fail_reasons": []}},
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "passed": True,
            "expectancy_gate_required": False,
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "expectancy_gate_required": False,
                    "signals": 1,
                    "orders": 1,
                    "fills": 1,
                    "expectancy_after_costs": -0.02,
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
    assert "canary_expectancy_gate_disabled" in result.reasons


def test_preflight_blocks_live_when_canary_is_calibration_only(tmp_path):
    matrix = _write_json(
        tmp_path / "matrix.json",
        {"promotion_verdict": {"passed": True, "fail_reasons": []}},
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "passed": True,
            "calibration_only": True,
            "actor_overrides": {"CHECK_INT": 1, "MOM_MIN": 0.0015},
            "exchanges": {
                "MEXC": {
                    "passed": True,
                    "signals": 3,
                    "orders": 3,
                    "fills": 3,
                    "expectancy_after_costs": 0.04,
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
    assert "canary_calibration_only" in result.reasons
    assert "canary_actor_overrides" in result.reasons


def test_real_launcher_config_hard_requires_bitget_policy_v1(tmp_path, monkeypatch):
    monkeypatch.delenv("BITGET_POLICY_MANIFEST_V1", raising=False)
    monkeypatch.delenv("BITGET_POLICY_MANIFEST_SHA256", raising=False)

    config = config_from_env(tmp_path)

    assert config.require_bitget_policy_v1 is True
    assert config.require_clean_git is True
    assert config.policy_manifest_path == "Runtime/BITGET/active_policy_manifest_v1.json"
    assert config.expected_policy_sha256 == ""


def test_bitget_live_preflight_blocks_unpinned_policy_even_when_old_gates_pass(tmp_path):
    now = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "generated_at": now.isoformat(),
            "promotion_verdict": {"passed": True, "fail_reasons": []},
        },
    )
    canary = _write_json(
        tmp_path / "canary.json",
        {
            "generated_at": now.isoformat(),
            "passed": True,
            "exchanges": {
                "BITGET": {
                    "passed": True,
                    "signals": 3,
                    "orders": 3,
                    "fills": 3,
                    "expectancy_after_costs": 0.04,
                    "reconcile_ok": True,
                    "owned_open_position_count": 0,
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
            policy_manifest_path=tmp_path / "policy.json",
            expected_policy_sha256="",
            require_bitget_policy_v1=True,
            now=now,
        ),
    )

    assert result.passed is False
    assert result.reasons == ("policy_manifest_sha256_pin_missing",)
