"""One-command bounded Bitget evidence session for the v2 data contract."""

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
from .precursor_campaign import run_precursor_campaign_v2
from .research_dataset_v2 import (
    materialize_research_dataset_v2,
    validate_research_dataset_v2,
)
from .segment_store import validate_data_session


EVIDENCE_SESSION_SCHEMA_VERSION_V2 = (
    "panteon.bitget_precursor_evidence_session.v2"
)


def run_precursor_evidence_session_v2(
    workspace_root: str | Path,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    raw_root = root / "Retrodate" / "bitget_data_v1"
    derived_root = root / "Retrodate" / "bitget_data_v2"
    campaign_dir = root / "Reports" / "BitgetData" / "precursor_campaign_v2"
    profile_path = (
        root / "configs" / "bitget_data_profiles" / f"{PROFILE_ID}.json"
    )
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    run_dir = campaign_dir / "session_runs" / run_id
    state_path = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    base_state = {
        "schema_version": EVIDENCE_SESSION_SCHEMA_VERSION_V2,
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "fixed_collection_duration_seconds": COLLECTION_DURATION_SECONDS,
        "data_contract": "event_snapshot_continuity_bounded_v2",
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
        frame_dir = derived_root / "datasets" / f"precursor_{session_id}"
        research_dir = derived_root / "research" / f"precursor_{session_id}"
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
                "stage": "materializing_research_v2",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "frame_dir": str(frame_dir),
                "frame_manifest_sha256": frame.manifest_sha256,
            },
        )
        research = materialize_research_dataset_v2(
            frame_dataset_dir=frame_dir,
            output_dir=research_dir,
        )
        if validate_research_dataset_v2(research_dir)["dataset_valid"] is not True:
            raise RuntimeError("research v2 dataset validation failed")
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "updating_campaign_v2",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "frame_dir": str(frame_dir),
                "research_dir": str(research_dir),
                "research_manifest_sha256": research.manifest_sha256,
            },
        )
        campaign = run_precursor_campaign_v2(
            research_root=derived_root / "research",
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
