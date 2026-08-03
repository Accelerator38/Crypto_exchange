"""One-command bounded collection and precursor campaign update."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .materializer import materialize_frame_1s, validate_frame_dataset
from .precursor_campaign import run_precursor_campaign
from .profile import load_bitget_data_profile
from .research_dataset import (
    materialize_research_dataset,
    validate_research_dataset,
)
from .segment_store import (
    BitgetDataCollectorLock,
    BitgetSegmentStore,
    compute_collector_fingerprint,
    current_git_revision,
    recover_unsealed_sessions,
    validate_data_session,
)
from .websocket_collector import run_public_collection


EVIDENCE_SESSION_SCHEMA_VERSION = "panteon.bitget_precursor_evidence_session.v1"
PROFILE_ID = "bitget_full8_microstructure_v1"
COLLECTION_DURATION_SECONDS = 12 * 60 * 60
MIN_ACCEPTED_DURATION_SECONDS = COLLECTION_DURATION_SECONDS * 0.995


def run_precursor_evidence_session(workspace_root: str | Path) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    data_root = root / "Retrodate" / "bitget_data_v1"
    campaign_dir = root / "Reports" / "BitgetData" / "precursor_campaign_v1"
    profile_path = (
        root / "configs" / "bitget_data_profiles" / f"{PROFILE_ID}.json"
    )
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    run_dir = campaign_dir / "session_runs" / run_id
    state_path = run_dir / "state.json"
    run_dir.mkdir(parents=True, exist_ok=False)
    base_state = {
        "schema_version": EVIDENCE_SESSION_SCHEMA_VERSION,
        "run_id": run_id,
        "profile_id": PROFILE_ID,
        "fixed_collection_duration_seconds": COLLECTION_DURATION_SECONDS,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "orders_enabled": False,
        "promotion_authority": False,
    }
    _write_state(state_path, {**base_state, "stage": "collecting"})
    try:
        collection, recovered = _collect(
            root=root,
            data_root=data_root,
            profile_path=profile_path,
        )
        _validate_collection_summary(collection)
        session_id = str(collection["session_id"])
        session_dir = data_root / "sessions" / session_id
        session_validation = validate_data_session(session_dir)
        status = _read_json(session_dir / "status.json")
        _validate_sealed_session(status, session_validation)
        frame_dir = data_root / "datasets" / f"precursor_{session_id}"
        research_dir = data_root / "research" / f"precursor_{session_id}"
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "materializing_frame",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "collection": collection,
                "recovered_sessions": recovered,
            },
        )
        frame = materialize_frame_1s(
            session_dir=session_dir,
            output_dir=frame_dir,
        )
        frame_validation = validate_frame_dataset(frame_dir)
        if frame_validation["dataset_valid"] is not True:
            raise RuntimeError("frame dataset validation failed")
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "materializing_research",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "frame_dir": str(frame_dir),
                "frame_manifest_sha256": frame.manifest_sha256,
            },
        )
        research = materialize_research_dataset(
            frame_dataset_dir=frame_dir,
            output_dir=research_dir,
        )
        research_validation = validate_research_dataset(research_dir)
        if research_validation["dataset_valid"] is not True:
            raise RuntimeError("research dataset validation failed")
        _write_state(
            state_path,
            {
                **base_state,
                "stage": "updating_campaign",
                "session_id": session_id,
                "session_dir": str(session_dir),
                "frame_dir": str(frame_dir),
                "research_dir": str(research_dir),
                "research_manifest_sha256": research.manifest_sha256,
            },
        )
        campaign = run_precursor_campaign(
            research_root=data_root / "research",
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


def _collect(
    *,
    root: Path,
    data_root: Path,
    profile_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collector_tool = root / "tools" / "run_bitget_data_layer.py"
    source = root / "src" / "panteon_v2" / "data" / "bitget"
    collector_files = (
        collector_tool,
        Path(__file__).resolve(),
        source / "profile.py",
        source / "rest_reconciler.py",
        source / "segment_store.py",
        source / "websocket_collector.py",
    )
    with BitgetDataCollectorLock(data_root):
        profile = load_bitget_data_profile(profile_path)
        recovered_rows = recover_unsealed_sessions(data_root)
        store = BitgetSegmentStore(
            data_dir=data_root,
            profile=profile,
            source_revision=current_git_revision(root),
            collector_fingerprint_sha256=compute_collector_fingerprint(
                collector_files
            ),
        )
        summary = asyncio.run(
            run_public_collection(
                profile=profile,
                store=store,
                duration_seconds=COLLECTION_DURATION_SECONDS,
            )
        )
    recovered = [
        {
            "session_id": row.session_id,
            "recovered_segments": row.recovered_segments,
            "manifest_sha256": row.manifest_sha256,
        }
        for row in recovered_rows
    ]
    return summary, recovered


def _validate_collection_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("orders_enabled") is not False:
        raise RuntimeError("collector orders_enabled is not false")
    if summary.get("promotion_authority") is not False:
        raise RuntimeError("collector promotion_authority is not false")
    if float(summary.get("duration_seconds") or 0.0) < MIN_ACCEPTED_DURATION_SECONDS:
        raise RuntimeError("collector duration is below fixed 12h evidence window")
    if not str(summary.get("last_manifest_sha256") or ""):
        raise RuntimeError("collector produced no sealed manifest")


def _validate_sealed_session(
    status: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    if validation.get("segments_valid") is not True:
        raise RuntimeError("collector session hash-chain validation failed")
    if status.get("run_state") != "stopped":
        raise RuntimeError("collector session did not stop cleanly")
    if status.get("stop_reason") != "duration_complete":
        raise RuntimeError("collector session did not complete fixed duration")
    if status.get("orders_enabled") is not False:
        raise RuntimeError("session orders_enabled is not false")
    if status.get("promotion_authority") is not False:
        raise RuntimeError("session promotion_authority is not false")


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
