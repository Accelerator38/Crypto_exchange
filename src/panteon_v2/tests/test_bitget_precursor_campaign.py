from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from panteon_v2.data.bitget import precursor_campaign


ROOT = Path(__file__).resolve().parents[3]


def _session(session_id: str, *, start_ms: int, end_ms: int, rows=None):
    return {
        "dataset_dir": f"fixture/{session_id}",
        "source_session_id": session_id,
        "manifest_sha256": session_id.ljust(64, "a")[:64],
        "database_sha256": session_id.ljust(64, "b")[:64],
        "start_timestamp_ms": start_ms,
        "end_timestamp_ms": end_ms,
        "duration_hours": (end_ms - start_ms) / 3_600_000.0,
        "eligible_bar_coverage_pct": 99.0,
        "frame_coverage_pct": 99.0,
        "rows": list(rows or []),
    }


def _row(direction: str = "LONG"):
    return {
        "bar_timestamp_ms": 1_000,
        "symbol": "BTCUSDT",
        "continuity_id": 1,
        "spread_mean_bps": 1.0,
        "microprice_edge_mean_bps": 0.20 if direction == "LONG" else -0.20,
        "book_imbalance_mean": 0.30 if direction == "LONG" else -0.30,
        "bid_depth_close_usd": 2_000.0 if direction == "LONG" else 1_000.0,
        "ask_depth_close_usd": 1_000.0 if direction == "LONG" else 2_000.0,
        "flow_imbalance": 0.50 if direction == "LONG" else -0.50,
        "return_5m_bps": 0.0,
        "oi_change_5m_bps": 25.0,
        "direction": direction,
        "gross_mid_bps": 30.0,
        "net_execution_bps": 15.0,
        "total_cost_bps": 15.0,
    }


def test_rule_score_uses_entry_context_not_future_label():
    rule = precursor_campaign.CANDIDATE_RULES[0]
    first = _row()
    second = {**first, "net_execution_bps": -100.0, "gross_mid_bps": -90.0}

    assert precursor_campaign._rule_score(first, rule) > 0.0
    assert precursor_campaign._rule_score(first, rule) == precursor_campaign._rule_score(second, rule)


def test_campaign_lock_keeps_old_session_development_and_assigns_future_roles(tmp_path):
    frozen = datetime(2026, 1, 2, tzinfo=timezone.utc)
    frozen_ms = int(frozen.timestamp() * 1000)
    old = _session("old", start_ms=frozen_ms - 86_400_000, end_ms=frozen_ms - 1)
    lock, created = precursor_campaign._ensure_lock(
        tmp_path,
        sessions=[old],
        frozen_at=frozen,
    )
    future = [
        _session(
            f"new-{index}",
            start_ms=frozen_ms + index * 86_400_000,
            end_ms=frozen_ms + index * 86_400_000 + 43_200_000,
        )
        for index in range(1, 6)
    ]

    classified, failures = precursor_campaign._classify_sessions([old, *future], lock)

    assert failures == []
    assert [row["role"] for row in classified] == [
        "development",
        "validation",
        "validation",
        "oos",
        "oos",
        "oos",
    ]
    assert created is True


def test_campaign_lock_rejects_tampering(tmp_path):
    frozen = datetime(2026, 1, 2, tzinfo=timezone.utc)
    session = _session("old", start_ms=1, end_ms=2)
    precursor_campaign._ensure_lock(tmp_path, sessions=[session], frozen_at=frozen)
    path = tmp_path / "campaign_lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["orders_enabled"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="lock hash mismatch"):
        precursor_campaign._ensure_lock(
            tmp_path,
            sessions=[session],
            frozen_at=frozen,
        )


def test_bad_quality_session_is_quarantined_without_consuming_validation(tmp_path):
    frozen = datetime(2026, 1, 2, tzinfo=timezone.utc)
    frozen_ms = int(frozen.timestamp() * 1000)
    old = _session("old", start_ms=frozen_ms - 86_400_000, end_ms=frozen_ms - 1)
    lock, _ = precursor_campaign._ensure_lock(
        tmp_path,
        sessions=[old],
        frozen_at=frozen,
    )
    bad = _session(
        "bad",
        start_ms=frozen_ms + 1,
        end_ms=frozen_ms + 43_200_001,
    )
    bad["eligible_bar_coverage_pct"] = 92.0
    good = _session(
        "good",
        start_ms=frozen_ms + 86_400_000,
        end_ms=frozen_ms + 86_400_000 + 43_200_000,
    )

    classified, failures = precursor_campaign._classify_sessions(
        [old, bad, good],
        lock,
    )

    assert failures == []
    assert [row["role"] for row in classified] == [
        "development",
        "quarantined",
        "validation",
    ]


def test_incomplete_campaign_is_research_only(monkeypatch, tmp_path):
    frozen = datetime(2026, 1, 2, tzinfo=timezone.utc)
    frozen_ms = int(frozen.timestamp() * 1000)
    session = _session(
        "development",
        start_ms=frozen_ms - 86_400_000,
        end_ms=frozen_ms - 1,
        rows=[_row("LONG"), {**_row("SHORT"), "bar_timestamp_ms": 2_000}],
    )
    monkeypatch.setattr(precursor_campaign, "_discover_sessions", lambda _: [session])

    report = precursor_campaign.run_precursor_campaign(
        research_root=tmp_path / "research",
        campaign_dir=tmp_path / "campaign",
        frozen_at=frozen,
    )

    assert report["verdict"] == "collecting_prospective_evidence"
    assert report["evidence"]["role_counts"] == {"development": 1}
    assert report["runtime_profile_created"] is False
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False
    assert all(
        row["failures"] == ["prospective_session_evidence_incomplete"]
        for row in report["candidates"]
    )


def test_v2_campaign_freezes_required_data_contract(monkeypatch, tmp_path):
    frozen = datetime(2026, 1, 2, tzinfo=timezone.utc)
    frozen_ms = int(frozen.timestamp() * 1000)
    session = _session(
        "development-v2",
        start_ms=frozen_ms - 86_400_000,
        end_ms=frozen_ms - 1,
        rows=[_row("LONG")],
    )
    monkeypatch.setattr(
        precursor_campaign,
        "_discover_sessions",
        lambda *args, **kwargs: [session],
    )

    report = precursor_campaign.run_precursor_campaign_v2(
        research_root=tmp_path / "research-v2",
        campaign_dir=tmp_path / "campaign-v2",
        frozen_at=frozen,
    )

    lock = json.loads(
        (tmp_path / "campaign-v2" / "campaign_lock.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema_version"].endswith(".v2")
    assert lock["schema_version"].endswith(".v2")
    assert report["fixed_contract"]["required_data_contract"] == (
        "event_snapshot_continuity_bounded_v2"
    )
    assert report["orders_enabled"] is False
    assert report["promotion_authority"] is False


def test_h120_rule_follows_return_and_uses_absolute_oi_change():
    rule = precursor_campaign.CANDIDATE_RULES_V3[0]
    long_row = {
        **_row("LONG"),
        "return_5m_bps": 12.0,
        "oi_change_5m_bps": -11.0,
    }
    short_label = {**long_row, "direction": "SHORT"}

    assert precursor_campaign._rule_score(long_row, rule) > 0.0
    assert precursor_campaign._rule_score(short_label, rule) == 0.0
    assert precursor_campaign._rule_score(
        {**long_row, "return_5m_bps": 9.9}, rule
    ) == 0.0


def test_h120_first_validation_early_stop_is_explicit():
    reports = [
        {
            "role": "validation",
            "metrics": {"closed_trades": 3, "mean_stress_net_bps": -0.1},
        }
    ]

    assert precursor_campaign._prospective_early_stop_reason(reports) == (
        "first_validation_nonpositive_stress_early_stop"
    )


def test_precursor_campaign_cli_bootstraps_outside_repository(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_bitget_precursor_campaign_v1.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "multi-session Bitget precursor campaign" in result.stdout
