from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from panteon_v2.data.bitget import evidence_session


ROOT = Path(__file__).resolve().parents[3]


def _collection_summary(duration: float | None = None):
    return {
        "session_id": "prospective-session",
        "duration_seconds": (
            evidence_session.COLLECTION_DURATION_SECONDS
            if duration is None
            else duration
        ),
        "last_manifest_sha256": "a" * 64,
        "orders_enabled": False,
        "promotion_authority": False,
    }


def test_fixed_evidence_pipeline_completes_without_order_authority(
    tmp_path,
    monkeypatch,
):
    session_dir = (
        tmp_path
        / "Retrodate"
        / "bitget_data_v1"
        / "sessions"
        / "prospective-session"
    )
    session_dir.mkdir(parents=True)
    (session_dir / "status.json").write_text(
        json.dumps(
            {
                "run_state": "stopped",
                "stop_reason": "duration_complete",
                "orders_enabled": False,
                "promotion_authority": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evidence_session,
        "_collect",
        lambda **_: (_collection_summary(), []),
    )
    monkeypatch.setattr(
        evidence_session,
        "validate_data_session",
        lambda _: {"segments_valid": True},
    )
    monkeypatch.setattr(
        evidence_session,
        "materialize_frame_1s",
        lambda **_: SimpleNamespace(manifest_sha256="b" * 64),
    )
    monkeypatch.setattr(
        evidence_session,
        "validate_frame_dataset",
        lambda _: {"dataset_valid": True},
    )
    monkeypatch.setattr(
        evidence_session,
        "materialize_research_dataset",
        lambda **_: SimpleNamespace(manifest_sha256="c" * 64),
    )
    monkeypatch.setattr(
        evidence_session,
        "validate_research_dataset",
        lambda _: {"dataset_valid": True},
    )
    monkeypatch.setattr(
        evidence_session,
        "run_precursor_campaign",
        lambda **_: {
            "campaign_lock": {"lock_sha256": "d" * 64},
            "verdict": "collecting_prospective_evidence",
            "evidence": {"role_counts": {"development": 1, "validation": 1}},
            "passed_candidates": [],
        },
    )

    result = evidence_session.run_precursor_evidence_session(tmp_path)

    assert result["stage"] == "complete"
    assert result["orders_enabled"] is False
    assert result["promotion_authority"] is False
    assert result["campaign_role_counts"]["validation"] == 1
    states = list(
        (
            tmp_path
            / "Reports"
            / "BitgetData"
            / "precursor_campaign_v1"
            / "session_runs"
        ).glob("*/state.json")
    )
    assert len(states) == 1
    assert json.loads(states[0].read_text(encoding="utf-8"))["stage"] == "complete"


def test_short_collection_fails_before_materialization(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evidence_session,
        "_collect",
        lambda **_: (_collection_summary(duration=60.0), []),
    )
    called = False

    def materialize(**_):
        nonlocal called
        called = True

    monkeypatch.setattr(evidence_session, "materialize_frame_1s", materialize)

    with pytest.raises(RuntimeError, match="below fixed 12h"):
        evidence_session.run_precursor_evidence_session(tmp_path)

    assert called is False
    state_path = next(
        (
            tmp_path
            / "Reports"
            / "BitgetData"
            / "precursor_campaign_v1"
            / "session_runs"
        ).glob("*/state.json")
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["stage"] == "failed"
    assert state["orders_enabled"] is False


def test_collection_summary_rejects_order_authority():
    summary = _collection_summary()
    summary["orders_enabled"] = True

    with pytest.raises(RuntimeError, match="orders_enabled"):
        evidence_session._validate_collection_summary(summary)


def test_evidence_session_cli_has_no_trading_or_duration_settings(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "run_bitget_precursor_evidence_session_v1.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed 12-hour read-only" in result.stdout
    assert "duration" not in result.stdout.lower().replace("fixed 12-hour", "")
    assert "orders" not in result.stdout.lower()


def test_evidence_session_v2_cli_bootstraps_outside_repository(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "tools"
                / "run_bitget_precursor_evidence_session_v2.py"
            ),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "event-aware frame v2 contract" in " ".join(result.stdout.split())


def test_evidence_session_v3_cli_has_no_candidate_settings(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "tools"
                / "run_bitget_precursor_evidence_session_v3.py"
            ),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    help_text = " ".join(result.stdout.split())
    assert "sealed single h120 candidate" in help_text
    assert "threshold" not in help_text
    assert "symbol" not in help_text
