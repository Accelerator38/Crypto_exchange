"""One-command prospective evidence for the fixed h120 candidate."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from .evidence_session import (
    COLLECTION_DURATION_SECONDS,
    PROFILE_ID,
    _collect,
    _read_json,
    _validate_collection_summary,
    _validate_sealed_session,
    _write_state,
)
from .materializer import materialize_frame_1s_v2, validate_frame_dataset_v2
from .precursor_campaign import run_precursor_campaign_v3
from .research_dataset_v2 import (
    materialize_research_dataset_v3,
    validate_research_dataset_v3,
)
from .segment_store import validate_data_session


EVIDENCE_SESSION_SCHEMA_VERSION_V3 = (
    "panteon.bitget_precursor_evidence_session.v3"
)


def run_precursor_evidence_session_v3(
    workspace_root: str | Path,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    raw_root = root / "Retrodate" / "bitget_data_v1"
    frame_root = root / "Retrodate" / "bitget_data_v2" / "datasets"
    research_root = root / "Retrodate" / "bitget_data_v3" / "research"
    campaign_dir = root / "Reports" / "BitgetData" / "precursor_campaign_v3"
    profile_path = (
        root / "configs" / "bitget_data_profiles" / f"{PROFILE_ID}.json"
    )
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    run_dir = campaign_dir / "session_runs" / run_id
    state_path = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    base_state = {
        "schema_version": EVIDENCE_SESSION_SCHEMA_VERSION_V3,
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "fixed_collection_duration_seconds": COLLECTION_DURATION_SECONDS,
        "data_contract": "event_snapshot_continuity_bounded_v2_h120",
        "candidate_id": "oi_change_return_momentum_h120_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "orders_enabled": False,
        "promotion_authority": False,
    }
    _write_state(state_path, {**base_state, "stage": "collecting"})
    try:
        collection, recovered = _collect(
            root=root,
            data_root=raw_root,
            profile_path=profile_path,
        )
        _validate_collection_summary(collection)
        session_id = str(collection["session_id"])
        session_dir = raw_root / "sessions" / session_id
        session_validation = validate_data_session(session_dir)
        _validate_sealed_session(
            _read_json(session_dir / "status.json"),
            session_validation,
        )
        frame_dir = frame_root / f"precursor_{session_id}"
        research_dir = research_root / f"precursor_{session_id}"
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "materializing_frame_v2",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "collection": collection,
                "recovered_sessions": recovered,
            },
        )
        frame = materialize_frame_1s_v2(
            session_dir=session_dir,
            output_dir=frame_dir,
        )
        if validate_frame_dataset_v2(frame_dir)["dataset_valid"] is not True:
            raise RuntimeError("frame v2 dataset validation failed")
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "materializing_research_v3",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "frame_dir": str(frame_dir),
                "frame_manifest_sha256": frame.manifest_sha256,
            },
        )
        research = materialize_research_dataset_v3(
            frame_dataset_dir=frame_dir,
            output_dir=research_dir,
        )
        if validate_research_dataset_v3(research_dir)["dataset_valid"] is not True:
            raise RuntimeError("research v3 dataset validation failed")
        campaign = run_precursor_campaign_v3(
            research_root=research_root,
            campaign_dir=campaign_dir,
        )
        result = {
            **base_state,
            "stage": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "session_dir": str(session_dir),
            "frame_dir": str(frame_dir),
            "research_dir": str(research_dir),
            "frame_manifest_sha256": frame.manifest_sha256,
            "research_manifest_sha256": research.manifest_sha256,
            "campaign_lock_sha256": campaign["campaign_lock"]["lock_sha256"],
            "campaign_verdict": campaign["verdict"],
            "campaign_role_counts": campaign["evidence"]["role_counts"],
            "passed_candidates": campaign["passed_candidates"],
            "early_stop_reason": campaign["candidates"][0][
                "early_stop_reason"
            ],
        }
        _write_state(state_path, result)
        return result
    except BaseException as exc:
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc)[:2_000],
            },
        )
        raise
