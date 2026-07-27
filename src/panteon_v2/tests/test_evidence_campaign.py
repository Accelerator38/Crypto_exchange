from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

from panteon_v2.policy.evidence_campaign import (
    BITGET_FULL8,
    CarryFlowEvidenceCampaign,
    EvidenceCampaignError,
    build_campaign_payload,
    build_negative_root_verdict,
    build_root_payload,
    decide_campaign_collection,
    validate_campaign_payload,
    validate_root_payload,
    validate_root_verdict_payload,
    write_atomic_json,
)


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_carryflow_evidence_campaign.py"


class _FakeTape:
    symbols = BITGET_FULL8
    bar_interval_seconds = 3600
    max_bar_close_lag_seconds = 120
    max_derivatives_age_seconds = 1200


def _report() -> dict:
    candidate = {
        "profile_id": "divergence_short_systemic_guard_v1",
        "filled_orders": 22,
        "closed_trades": 11,
        "mean_net_pnl_usd": 0.024,
        "expectancy_lcb_95_usd": -0.018,
        "max_drawdown_usd": 0.18,
        "max_drawdown_limit_usd": 0.20,
        "active_roots": 3,
        "root_expectancy_collapses": [],
        "failures": ["nonpositive_lcb"],
        "screening_passed": False,
        "evidence_extension_eligible": True,
        "segment_results": [
            {"tape": "root-1.jsonl"},
            {"tape": "root-2.jsonl"},
            {"tape": "root-3.jsonl"},
        ],
    }
    return {
        "schema_version": "panteon.carryflow_exact_profile_evaluation.v1",
        "generated_at": "2026-07-20T12:00:00+00:00",
        "research_only": True,
        "profile_selection_only": True,
        "root_robustness_required": True,
        "fresh_prospective_validation_required": True,
        "orders_enabled": False,
        "promotion_authority": False,
        "segments_concatenated": False,
        "evidence_extension_orders_enabled": False,
        "evidence_extension_is_promotion": False,
        "selected_for_prospective_validation": None,
        "selected_for_evidence_extension": candidate,
    }


def _campaign_payload() -> dict:
    return build_campaign_payload(
        report=_report(),
        report_file="screening_report.json",
        report_sha256="a" * 64,
        source_revision="b" * 40,
        runtime_fingerprint_sha256="c" * 64,
        created_at=datetime(2026, 7, 20, 12, tzinfo=timezone.utc),
        tape_loader=lambda _path: _FakeTape(),
    )


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "run_carryflow_evidence_campaign_test",
        TOOL_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_is_derived_from_one_profile_and_sealed():
    payload = _campaign_payload()

    assert payload["strategy_config"] == {
        "PROFILE_ID": "divergence_short_systemic_guard_v1"
    }
    assert payload["symbol_set"] == list(BITGET_FULL8)
    assert payload["orders_enabled"] is False
    assert payload["promotion_authority"] is False
    assert payload["segments_concatenated"] is False
    assert payload["gap_handling"] == "start_new_independent_root"
    assert validate_campaign_payload(payload) == payload


def test_campaign_rejects_any_failure_beyond_lcb():
    report = _report()
    report["selected_for_evidence_extension"]["failures"] = [
        "nonpositive_lcb",
        "drawdown_above_limit",
    ]

    with pytest.raises(EvidenceCampaignError, match="only for nonpositive LCB"):
        build_campaign_payload(
            report=report,
            report_file="screening_report.json",
            report_sha256="a" * 64,
            source_revision="b" * 40,
            runtime_fingerprint_sha256="c" * 64,
            created_at=datetime.now(timezone.utc),
            tape_loader=lambda _path: _FakeTape(),
        )


def test_campaign_rejects_order_capability_or_profile_knobs():
    payload = _campaign_payload()
    payload["orders_enabled"] = True
    with pytest.raises(EvidenceCampaignError, match="SHA-256 mismatch|enable orders"):
        validate_campaign_payload(payload)

    payload = _campaign_payload()
    payload["strategy_config"]["OI_SPIKE"] = 0.01
    with pytest.raises(EvidenceCampaignError, match="independent knobs"):
        validate_campaign_payload(payload)


@pytest.mark.parametrize(
    ("minute", "latest_hour", "action", "reason"),
    [
        (0, None, "wait", "bar_close_settlement_delay"),
        (3, None, "wait", "current_collection_window_missed"),
        (1, None, "new_root", "campaign_first_root"),
        (1, 11, "resume_root", "exact_next_bar"),
        (1, 12, "idle", "current_bar_already_collected"),
        (1, 10, "new_root", "real_bar_gap"),
    ],
)
def test_collection_decision_is_windowed_and_gap_tolerant(
    minute: int,
    latest_hour: int | None,
    action: str,
    reason: str,
):
    now = datetime(2026, 7, 20, 12, minute, tzinfo=timezone.utc)
    latest = (
        int(datetime(2026, 7, 20, latest_hour, tzinfo=timezone.utc).timestamp() * 1000)
        if latest_hour is not None
        else None
    )

    decision = decide_campaign_collection(
        now=now,
        bar_interval_seconds=3600,
        alignment_delay_seconds=30,
        max_bar_close_lag_seconds=120,
        latest_bar_close_timestamp_ms=latest,
    )

    assert decision.action == action
    assert decision.reason == reason


def test_root_lock_pins_campaign_and_cannot_be_rewritten(tmp_path):
    campaign_path = tmp_path / "campaign_lock.json"
    write_atomic_json(campaign_path, _campaign_payload())
    campaign = CarryFlowEvidenceCampaign.from_json(campaign_path)
    root = build_root_payload(
        campaign=campaign,
        root_index=1,
        created_at=datetime(2026, 7, 20, 12, 1, tzinfo=timezone.utc),
        intended_first_bar_close_timestamp_ms=int(
            datetime(2026, 7, 20, 12, tzinfo=timezone.utc).timestamp() * 1000
        ),
        start_reason="campaign_first_root",
    )

    assert validate_root_payload(root, campaign=campaign) == root
    root["orders_enabled"] = True
    with pytest.raises(EvidenceCampaignError, match="SHA-256 mismatch|enable orders"):
        validate_root_payload(root, campaign=campaign)


def test_collector_command_has_no_strategy_or_order_knobs(tmp_path):
    tool = _load_tool()
    campaign_path = tmp_path / "campaign_lock.json"
    write_atomic_json(campaign_path, _campaign_payload())
    campaign = CarryFlowEvidenceCampaign.from_json(campaign_path)
    root_payload = build_root_payload(
        campaign=campaign,
        root_index=1,
        created_at=datetime(2026, 7, 20, 12, 1, tzinfo=timezone.utc),
        intended_first_bar_close_timestamp_ms=int(
            datetime(2026, 7, 20, 12, tzinfo=timezone.utc).timestamp() * 1000
        ),
        start_reason="campaign_first_root",
    )
    command = tool.collector_command(
        campaign,
        {"payload": root_payload, "root_dir": tmp_path / "root"},
    )

    assert "--profile" not in command
    assert "--orders-enabled" not in command
    assert "--resume" in command
    assert command[command.index("--symbols") + 1] == ",".join(BITGET_FULL8)
    assert command[command.index("--warmup-bars") + 1] == "53"


def test_negative_root_verdict_is_terminal_and_hash_sealed(tmp_path):
    campaign_path = tmp_path / "campaign_lock.json"
    write_atomic_json(campaign_path, _campaign_payload())
    campaign = CarryFlowEvidenceCampaign.from_json(campaign_path)
    root = build_root_payload(
        campaign=campaign,
        root_index=1,
        created_at=datetime(2026, 7, 22, 8, tzinfo=timezone.utc),
        intended_first_bar_close_timestamp_ms=int(
            datetime(2026, 7, 22, 8, tzinfo=timezone.utc).timestamp() * 1000
        ),
        start_reason="campaign_first_root",
    )
    description = {
        "samples": 74,
        "complete_samples": 72,
        "context_coverage_pct": 97.2972972972973,
        "head_sha256": "d" * 64,
    }
    summary = {
        "evidence_tape": {
            "collector_run_id": root["root_id"],
            "file_sha256": "e" * 64,
            "samples": 74,
            "complete_samples": 72,
            "context_coverage_pct": 97.2972972972973,
        },
        "evidence_eligible": False,
        "research_only": True,
        "promotion_authority": False,
        "evidence_failures": [
            "fills_below_20",
            "closed_trades_below_10",
            "nonpositive_expectancy",
            "nonpositive_lcb",
        ],
        "filled_orders": 10,
        "closed_trades": 5,
        "gross_pnl_usd": 0.038,
        "total_costs_usd": 0.060,
        "net_pnl_usd": -0.022,
        "expectancy_after_costs_usd": -0.0044,
        "expectancy_lcb_usd": -0.053,
        "max_drawdown_usd": 0.077,
        "remaining_open_positions": 0,
        "manifest_sha256": "f" * 64,
    }

    verdict = build_negative_root_verdict(
        campaign=campaign,
        root_payload=root,
        tape_description=description,
        evidence_tape_sha256="e" * 64,
        replay_summary=summary,
        replay_summary_sha256="a" * 64,
        recorded_at=datetime(2026, 7, 27, 8, tzinfo=timezone.utc),
    )

    assert verdict["verdict"] == "rejected"
    assert verdict["terminal"] is True
    assert verdict["continuation_allowed"] is False
    assert verdict["orders_enabled"] is False
    assert verdict["promotion_authority"] is False
    assert validate_root_verdict_payload(
        verdict,
        campaign=campaign,
        root_payload=root,
    ) == verdict

    verdict["continuation_allowed"] = True
    with pytest.raises(EvidenceCampaignError, match="safety flags|SHA-256"):
        validate_root_verdict_payload(
            verdict,
            campaign=campaign,
            root_payload=root,
        )


def test_terminal_root_makes_campaign_runner_a_noop(tmp_path, monkeypatch):
    tool = _load_tool()
    campaign_path = tmp_path / "campaign_lock.json"
    write_atomic_json(campaign_path, _campaign_payload())
    campaign = CarryFlowEvidenceCampaign.from_json(campaign_path)
    root = build_root_payload(
        campaign=campaign,
        root_index=1,
        created_at=datetime(2026, 7, 22, 8, tzinfo=timezone.utc),
        intended_first_bar_close_timestamp_ms=int(
            datetime(2026, 7, 22, 8, tzinfo=timezone.utc).timestamp() * 1000
        ),
        start_reason="campaign_first_root",
    )

    class _TerminalTape:
        samples = [{"bar_close_timestamp_ms": 123}]

        @staticmethod
        def describe():
            return {
                "samples": 74,
                "complete_samples": 72,
                "context_coverage_pct": 97.2972972972973,
                "head_sha256": "d" * 64,
                "last_bar_close_timestamp_ms": 123,
                "symbol_observations": 592,
                "complete_symbol_observations": 576,
            }

    roots = [
        {
            "payload": root,
            "root_dir": tmp_path / "roots" / "root_001",
            "tape": _TerminalTape(),
            "seed": object(),
            "verdict": {
                "verdict": "rejected",
                "reason": "negative_costed_expectancy_and_lcb",
                "continuation_allowed": False,
            },
        }
    ]
    monkeypatch.setattr(tool, "prepare_campaign", lambda *args, **kwargs: campaign)
    monkeypatch.setattr(tool, "load_campaign_roots", lambda _campaign: roots)

    def _unexpected_environment_check(_campaign):
        raise AssertionError("terminal campaign must not verify or collect")

    monkeypatch.setattr(
        tool,
        "verify_campaign_environment",
        _unexpected_environment_check,
    )

    status, return_code = tool.run_campaign_once(
        tmp_path,
        now=datetime(2026, 7, 27, 8, tzinfo=timezone.utc),
    )

    assert return_code == 0
    assert status["run_state"] == "completed_negative"
    assert status["decision"]["action"] == "terminal"
    assert status["roots"][0]["terminal"] is True
    assert status["roots"][0]["continuation_allowed"] is False
